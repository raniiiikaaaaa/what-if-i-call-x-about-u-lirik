"""Lyric Float.

Menampilkan lirik sebagai kotak melayang yang naik dari bawah layar sambil
memutar lagu. Setelah semua lirik selesai, sebuah video diputar fullscreen
(lagu tetap berjalan). Program menutup diri setelah video dan lagu selesai.

Dependensi:
    pip install pygame opencv-python pillow
"""

from __future__ import annotations

import logging
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import cv2
import pygame
from PIL import Image, ImageTk

logger = logging.getLogger("lyric_float")

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Konfigurasi & data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    """Seluruh pengaturan program dikumpulkan di satu tempat."""

    audio_file: Path = BASE_DIR / "audio.mp3"
    video_file: Path = BASE_DIR / "animasi.mp4"

    # Tampilan kotak lirik
    box_width: int = 350
    box_height: int = 300
    font: tuple = ("Helvetica", 25, "bold")
    bg_color: str = "#fdfdf5"
    fg_color: str = "#111111"

    # Posisi & pergerakan
    rise_speed: float = 150.0          # piksel per detik
    gap: int = 40                      # jarak antar kotak (piksel)
    bottom_spawn_offset: int = 0
    column_margin: int = 60            # jarak kolom dari tengah layar

    # Timing (milidetik)
    typewriter_delay_ms: int = 50
    flip_interval_ms: int = 150
    frame_interval_ms: int = 16        # ~60 FPS
    music_poll_ms: int = 200

    offscreen_margin: int = 50


class LyricLine(NamedTuple):
    """Satu baris lirik: kapan muncul (detik) dan teksnya."""

    time: float
    text: str


LYRICS: list[LyricLine] = [
    LyricLine(0, "and i get the chance to say"),
    LyricLine(3, "i wish i didn't"),
    LyricLine(4, "but i miss you"),
    LyricLine(10, "do you think i have forgotten?"),
    LyricLine(14, "do you think i have forgotten?"),
    LyricLine(18, "do you think i have forgotten?"),
]


# --------------------------------------------------------------------------- #
# Kotak lirik
# --------------------------------------------------------------------------- #
class LyricCard:
    """Jendela tanpa border berisi satu baris lirik dengan efek mesin ketik."""

    def __init__(self, parent: tk.Misc, text: str, x: float, y: float,
                 cfg: Config) -> None:
        self._cfg = cfg
        self._full_text = text
        self._char_index = 0
        self._inverted = False
        self.x = float(x)
        self.y = float(y)

        self.win = tk.Toplevel(parent)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.resizable(False, False)
        self.win.configure(bg=cfg.bg_color)
        self._apply_geometry()

        self.label = tk.Label(
            self.win,
            text="",
            font=cfg.font,
            bg=cfg.bg_color,
            fg=cfg.fg_color,
            wraplength=cfg.box_width - 25,
            justify="center",
        )
        self.label.pack(expand=True, fill="both", padx=15, pady=15)

        self._type_next_char()

    # -- Tampilan ----------------------------------------------------------- #
    def _apply_geometry(self) -> None:
        c = self._cfg
        self.win.geometry(f"{c.box_width}x{c.box_height}+{int(self.x)}+{int(self.y)}")

    def _type_next_char(self) -> None:
        if self._char_index > len(self._full_text):
            return
        self.label.config(text=self._full_text[: self._char_index])
        self._char_index += 1
        self.win.after(self._cfg.typewriter_delay_ms, self._type_next_char)

    def toggle_invert(self) -> None:
        """Balik warna hitam/putih (efek berkedip)."""
        self._inverted = not self._inverted
        bg, fg = ("black", "white") if self._inverted else ("white", "black")
        self.win.config(bg=bg)
        self.label.config(bg=bg, fg=fg)

    # -- Pergerakan --------------------------------------------------------- #
    def move_up(self, dy: float) -> None:
        self.y -= dy
        self._apply_geometry()

    def is_offscreen(self) -> bool:
        return self.y + self._cfg.box_height < -self._cfg.offscreen_margin

    def destroy(self) -> None:
        try:
            self.win.destroy()
        except tk.TclError:
            pass


# --------------------------------------------------------------------------- #
# Pemutar video fullscreen
# --------------------------------------------------------------------------- #
class FullscreenVideo:
    """Memutar video (tanpa suara) fullscreen, tanpa border/title bar.

    Tekan Esc untuk menutup lebih awal. `on_finish` dipanggil tepat satu kali
    ketika video selesai, gagal dibuka, atau ditutup manual.
    """

    def __init__(self, root: tk.Tk, path: Path, on_finish: Callable[[], None],
                 cfg: Config) -> None:
        self._root = root
        self._on_finish = on_finish
        self._screen_w = root.winfo_screenwidth()
        self._screen_h = root.winfo_screenheight()
        self._closed = False
        self._after_id: Optional[str] = None
        self._photo: Optional[ImageTk.PhotoImage] = None

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            logger.error("Gagal membuka video: %s", path)
            self._closed = True
            self._on_finish()
            return

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_delay = 1.0 / fps if fps and fps > 0 else 1 / 30

        self._win = tk.Toplevel(root)
        self._win.overrideredirect(True)
        self._win.attributes("-topmost", True)
        self._win.configure(bg="black", cursor="none")
        self._win.geometry(f"{self._screen_w}x{self._screen_h}+0+0")
        self._win.bind("<Escape>", lambda _event: self.close())
        self._win.focus_force()

        self._label = tk.Label(self._win, bg="black", bd=0)
        self._label.pack(expand=True, fill="both")

        self._next_time = time.monotonic()
        self._update_frame()

    def _fit_cover(self, frame):
        """Skalakan frame agar memenuhi layar (crop tengah, tanpa distorsi)."""
        h, w = frame.shape[:2]
        scale = max(self._screen_w / w, self._screen_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
        x0 = (new_w - self._screen_w) // 2
        y0 = (new_h - self._screen_h) // 2
        return frame[y0:y0 + self._screen_h, x0:x0 + self._screen_w]

    def _update_frame(self) -> None:
        if self._closed:
            return

        ok, frame = self._cap.read()
        if not ok:
            self.close()
            return

        rgb = cv2.cvtColor(self._fit_cover(frame), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._label.config(image=self._photo)

        # Pertahankan kecepatan sesuai FPS asli video.
        self._next_time += self._frame_delay
        delay_ms = max(1, int((self._next_time - time.monotonic()) * 1000))
        self._after_id = self._win.after(delay_ms, self._update_frame)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._after_id is not None:
            try:
                self._win.after_cancel(self._after_id)
            except tk.TclError:
                pass
        self._cap.release()
        try:
            self._win.destroy()
        except tk.TclError:
            pass
        self._on_finish()


# --------------------------------------------------------------------------- #
# Aplikasi utama
# --------------------------------------------------------------------------- #
class LyricFloatApp:
    """Mengatur alur: lagu + lirik melayang -> video fullscreen -> selesai."""

    def __init__(self, root: tk.Tk, lyrics: list[LyricLine], cfg: Config) -> None:
        self._root = root
        self._lyrics = lyrics
        self._cfg = cfg

        self._screen_w = root.winfo_screenwidth()
        self._screen_h = root.winfo_screenheight()

        self._next_index = 0
        self._cards: list[LyricCard] = []
        self._use_left_column = True
        self._video_started = False

        self._start_time = 0.0
        self._last_frame_time = 0.0

    # -- Alur --------------------------------------------------------------- #
    def start(self) -> None:
        self._root.iconify()
        self._start_time = time.monotonic()
        self._last_frame_time = self._start_time

        pygame.mixer.music.play()
        self._tick()
        self._flip_all()

    def _tick(self) -> None:
        now = time.monotonic()
        elapsed = now - self._start_time
        dt = now - self._last_frame_time
        self._last_frame_time = now

        self._spawn_due_cards(elapsed)
        self._move_cards(dt)

        if self._next_index < len(self._lyrics) or self._cards:
            self._root.after(self._cfg.frame_interval_ms, self._tick)
        else:
            self._play_video()

    # -- Kartu lirik -------------------------------------------------------- #
    def _next_column_x(self) -> int:
        """Tentukan posisi X, bergantian antara kolom kiri dan kanan."""
        c = self._cfg
        center = self._screen_w // 2
        left_x = center - (c.box_width + c.column_margin)
        right_x = center + c.column_margin
        x = left_x if self._use_left_column else right_x
        self._use_left_column = not self._use_left_column
        return x

    def _spawn_y(self) -> float:
        c = self._cfg
        if self._cards:
            return self._cards[-1].y + c.box_height + c.gap
        return self._screen_h - c.box_height - c.bottom_spawn_offset

    def _spawn_due_cards(self, elapsed: float) -> None:
        while (self._next_index < len(self._lyrics)
               and self._lyrics[self._next_index].time <= elapsed):
            line = self._lyrics[self._next_index]
            card = LyricCard(self._root, line.text, self._next_column_x(),
                             self._spawn_y(), self._cfg)
            self._cards.append(card)
            self._next_index += 1

    def _move_cards(self, dt: float) -> None:
        dy = self._cfg.rise_speed * dt
        visible: list[LyricCard] = []
        for card in self._cards:
            card.move_up(dy)
            if card.is_offscreen():
                card.destroy()
            else:
                visible.append(card)
        self._cards = visible

    def _flip_all(self) -> None:
        if self._video_started:
            return
        for card in self._cards:
            card.toggle_invert()
        self._root.after(self._cfg.flip_interval_ms, self._flip_all)

    # -- Video & penutupan -------------------------------------------------- #
    def _play_video(self) -> None:
        self._video_started = True
        FullscreenVideo(self._root, self._cfg.video_file,
                        on_finish=self._wait_for_music_end, cfg=self._cfg)

    def _wait_for_music_end(self) -> None:
        """Tutup program setelah lagu selesai agar tidak terpotong."""
        if pygame.mixer.music.get_busy():
            self._root.after(self._cfg.music_poll_ms, self._wait_for_music_end)
        else:
            self._root.destroy()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def validate_files(cfg: Config) -> bool:
    """Pastikan file audio & video tersedia sebelum program dimulai."""
    ok = True
    for path in (cfg.audio_file, cfg.video_file):
        if not path.is_file():
            logger.error("File tidak ditemukan: %s", path)
            ok = False
    return ok


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = Config()

    if not LYRICS:
        logger.error("Daftar LYRICS masih kosong.")
        return 1
    if not validate_files(cfg):
        return 1

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(str(cfg.audio_file))
    except pygame.error as exc:
        logger.error("Gagal memuat audio: %s", exc)
        return 1

    root = tk.Tk()
    LyricFloatApp(root, LYRICS, cfg).start()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
