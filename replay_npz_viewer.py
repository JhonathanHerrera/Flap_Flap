# file: replay_npz_viewer.py
"""
Replay a recorded Flappy Bird trajectory saved as .npz using the SAME assets
and window sizes as your flappy_bird.py.

Usage:
    python replay_npz_viewer.py exports/replay_best_multih.npz --fps 30 --loop

Controls:
    Space: play/pause
    ← / →: step frame backward/forward
    +/-  : decrease/increase playback FPS
    R    : restart
    L    : toggle loop
    Esc/Q: quit
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pygame
import os

# --- Match your flappy_bird.py constants ---
WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
ANIMATION_TIME = 5
MAX_ROTATION = 25
ROT_VEL = 20

# --------------- Asset loading ---------------
def load_assets(img_dir="imgs"):
    # identical paths / transforms as your game
    pipe_img = pygame.transform.scale2x(
        pygame.image.load(os.path.join(img_dir, "pipe.png")).convert_alpha()
    )
    pipe_top_img = pygame.transform.flip(pipe_img, False, True)

    bg_img = pygame.transform.scale(
        pygame.image.load(os.path.join(img_dir, "bg.png")).convert_alpha(), (600, 900)
    )
    bird_images = [
        pygame.transform.scale2x(pygame.image.load(os.path.join(img_dir, f"bird{x}.png"))).convert_alpha()
        for x in range(1, 4)
    ]
    base_img = pygame.transform.scale2x(
        pygame.image.load(os.path.join(img_dir, "base.png")).convert_alpha()
    )

    return {
        "pipe_bottom": pipe_img,
        "pipe_top": pipe_top_img,
        "bg": bg_img,
        "birds": bird_images,
        "base": base_img,
    }

# --------------- NPZ loading ---------------
def load_replay(npz_path: Path):
    data = np.load(str(npz_path))
    meta = json.loads(data["meta"].tobytes().decode("utf-8"))
    # cur_top / nxt_top were recorded as pipe.height (not top y).
    # cur_bot / nxt_bot were recorded as pipe.bottom.
    return {
        "t": data["t"],
        "bx": data["bx"],
        "by": data["by"],
        "bv": data.get("bv"),
        "score": data["score"],
        "cur_px": data["cur_px"],
        "cur_height": data["cur_top"],
        "cur_bottom": data["cur_bot"],
        "nxt_px": data["nxt_px"],
        "nxt_height": data["nxt_top"],
        "nxt_bottom": data["nxt_bot"],
        "meta": meta,
    }

# --------------- Helpers ---------------
def infer_pipe_vel(px_series):
    """Infer horizontal pipe velocity from consecutive frames."""
    if len(px_series) < 2:
        return 5.0
    # velocity is roughly px[i-1] - px[i] (positive)
    diffs = np.diff(px_series[: min(10, len(px_series))])
    v = np.median(-diffs)  # px decreases as pipes move left; make it positive
    if not np.isfinite(v) or v <= 0:
        v = 5.0
    return float(v)

class BaseScroller:
    """Scroll base using inferred velocity to match parallax."""
    def __init__(self, base_img, init_vel=5.0):
        self.IMG = base_img
        self.WIDTH = base_img.get_width()
        self.y = FLOOR
        self.x1 = 0
        self.x2 = self.WIDTH
        self.vel = init_vel

    def set_vel(self, vel):
        self.vel = max(1.0, float(vel))

    def move(self):
        self.x1 -= self.vel
        self.x2 -= self.vel
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

class BirdSprite:
    """Animate & rotate bird using recorded velocity (approximation of game logic)."""
    def __init__(self, bird_imgs):
        self.IMGS = bird_imgs
        self.img = self.IMGS[0]
        self.img_count = 0
        self.tilt = 0  # degrees

    def update(self, vel_y):
        # animation
        self.img_count += 1
        if self.img_count <= ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count <= ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count <= ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        else:
            self.img = self.IMGS[0]
            self.img_count = 0

        # rotation heuristic from velocity:
        # upward (negative vel) -> tilt up; otherwise tilt down toward -90
        if vel_y is None:
            vel_y = 0.0
        if vel_y < -1.0:
            self.tilt = min(MAX_ROTATION, self.tilt + ROT_VEL)
        else:
            self.tilt = max(-90, self.tilt - ROT_VEL)

    @staticmethod
    def blit_rotate_center(surf, image, topleft, angle):
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)
        surf.blit(rotated_image, new_rect.topleft)

    def draw(self, win, x, y):
        # when diving steeply force middle wing like the game
        img = self.img
        if self.tilt <= -80:
            img = self.IMGS[1]
        self.blit_rotate_center(win, img, (x, y), self.tilt)

# --------------- Draw pipes using sprites ---------------
def draw_pipe_pair(win, assets, px, height_value, bottom_value):
    """Given pipe x, height and bottom (from replay), compute top y and blit sprites."""
    if not np.isfinite(px):
        return
    pipe_top = assets["pipe_top"]
    pipe_bottom = assets["pipe_bottom"]

    # In your game: top_y = height - PIPE_TOP.get_height(), bottom_y = bottom
    top_y = float(height_value) - pipe_top.get_height()
    bottom_y = float(bottom_value)

    win.blit(pipe_top, (float(px), top_y))
    win.blit(pipe_bottom, (float(px), bottom_y))

# --------------- Main viewer ---------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=str, help="Path to replay .npz")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--imgdir", type=str, default="imgs", help="Assets directory (same as game)")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    pygame.init()
    pygame.display.set_caption("Flappy Bird Replay (using game assets)")
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    # fonts to match your game
    stat_font = pygame.font.SysFont("comicsans", 24)

    # assets
    assets = load_assets(args.imgdir)
    bg = assets["bg"]
    base = BaseScroller(assets["base"], init_vel=5.0)
    bird_sprite = BirdSprite(assets["birds"])

    # replay data
    R = load_replay(npz_path)
    t = R["t"]; bx = R["bx"]; by = R["by"]; bv = R["bv"]; score = R["score"]
    cur_px = R["cur_px"]; cur_h = R["cur_height"]; cur_b = R["cur_bottom"]
    nxt_px = R["nxt_px"]; nxt_h = R["nxt_height"]; nxt_b = R["nxt_bottom"]
    meta = R["meta"]

    # infer scrolling speed from first few pipe deltas
    base.set_vel(infer_pipe_vel(cur_px))

    i = 0
    playing = True
    fps = max(1, args.fps)
    loop = bool(args.loop)

    def draw_frame(idx: int):
        win.blit(bg, (0, 0))  # same as game

        # pipes (current + next)
        draw_pipe_pair(win, assets, cur_px[idx], cur_h[idx], cur_b[idx])
        draw_pipe_pair(win, assets, nxt_px[idx], nxt_h[idx], nxt_b[idx])

        # bird
        vy = float(bv[idx]) if (bv is not None and idx < len(bv)) else 0.0
        bird_sprite.update(vy)
        bird_sprite.draw(win, float(bx[idx]), float(by[idx]))

        # base scroll
        base.move()
        base.draw(win)

        # HUD
        hud_lines = [
            f"t={int(t[idx])}",
            f"score={int(score[idx])}",
            f"fps={fps}",
            f"seed={meta.get('seed','?')} best={meta.get('best_score','?')}",
            f"double_q={meta.get('use_double_q', False)} traces={meta.get('use_traces', False)}",
            f"{'LOOP' if loop else ''} {'PAUSED' if not playing else ''}",
        ]
        y = 8
        for line in hud_lines:
            surf = stat_font.render(line, True, (255, 255, 255))
            win.blit(surf, (10, y))
            y += 22

        pygame.display.update()

    while True:
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); return
                if event.key == pygame.K_SPACE:
                    playing = not playing
                if event.key == pygame.K_LEFT:
                    i = max(0, i - 1)
                if event.key == pygame.K_RIGHT:
                    i = min(len(t) - 1, i + 1)
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    fps = min(240, fps + 5)
                if event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    fps = max(1, fps - 5)
                if event.key == pygame.K_r:
                    i = 0; playing = True
                if event.key == pygame.K_l:
                    loop = not loop

        if playing:
            i += 1
            if i >= len(t):
                if loop:
                    i = 0
                else:
                    playing = False
                    i = len(t) - 1

        i = max(0, min(i, len(t) - 1))
        draw_frame(i)

if __name__ == "__main__":
    main()
