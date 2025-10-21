# file: replay_npz_viewer.py
"""
Replay a recorded Flappy Bird .npz using the same assets as flappy_bird.py,
with optional gap-midline overlay and reconstruction of all on-screen pipes.

Usage:
  python replay_npz_viewer.py exports/replay_best_multih.npz --fps 30 --loop --overlay --reconstruct-all --loss-weight 0.02

Keys:
  Space pause/resume | ←/→ step | +/- speed | R restart | L loop | Esc/Q quit
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pygame

# Match your game constants
WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
ANIMATION_TIME = 5
MAX_ROTATION = 25
ROT_VEL = 20
PIPE_VISUAL_WIDTH = 52  # sprite width, visual only

# ---------- Assets ----------
def load_assets(img_dir="imgs"):
    pipe_img = pygame.transform.scale2x(
        pygame.image.load(os.path.join(img_dir, "pipe.png")).convert_alpha()
    )
    pipe_top_img = pygame.transform.flip(pipe_img, False, True)
    bg_img = pygame.transform.scale(
        pygame.image.load(os.path.join(img_dir, "bg.png")).convert_alpha(), (600, 900)
    )
    bird_images = [
        pygame.transform.scale2x(pygame.image.load(os.path.join(img_dir, f"bird{x}.png"))).convert_alpha()
        for x in range(1, 3 + 1)
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

# ---------- NPZ ----------
def load_replay(npz_path: Path):
    data = np.load(str(npz_path))
    meta = json.loads(data["meta"].tobytes().decode("utf-8"))
    return {
        "t": data["t"],
        "bx": data["bx"],
        "by": data["by"],
        "bv": data.get("bv"),
        "score": data["score"],
        "cur_px": data["cur_px"],
        "cur_height": data["cur_top"],   # recorded as pipe.height
        "cur_bottom": data["cur_bot"],   # recorded as pipe.bottom
        "nxt_px": data["nxt_px"],
        "nxt_height": data["nxt_top"],
        "nxt_bottom": data["nxt_bot"],
        "meta": meta,
    }

# ---------- Helpers ----------
def infer_pipe_vel(px_series):
    if len(px_series) < 2:
        return 5.0
    diffs = np.diff(px_series[: min(16, len(px_series))])
    v = np.median(-diffs)
    return float(v if np.isfinite(v) and v > 0 else 5.0)

def gap_center(height_val, bottom_val):
    return 0.5 * (float(height_val) + float(bottom_val))

class BaseScroller:
    def __init__(self, base_img, init_vel=5.0):
        self.IMG = base_img
        self.WIDTH = base_img.get_width()
        self.y = FLOOR
        self.x1 = 0
        self.x2 = self.WIDTH
        self.vel = init_vel

    def set_vel(self, vel): self.vel = max(1.0, float(vel))

    def move(self):
        self.x1 -= self.vel
        self.x2 -= self.vel
        if self.x1 + self.WIDTH < 0: self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0: self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

class BirdSprite:
    def __init__(self, bird_imgs):
        self.IMGS = bird_imgs
        self.img = self.IMGS[0]
        self.img_count = 0
        self.tilt = 0

    def update(self, vel_y):
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
            self.img = self.IMGS[0]; self.img_count = 0

        vy = 0.0 if vel_y is None else float(vel_y)
        if vy < -1.0: self.tilt = min(MAX_ROTATION, self.tilt + ROT_VEL)
        else:         self.tilt = max(-90, self.tilt - ROT_VEL)

    @staticmethod
    def blit_rotate_center(surf, image, topleft, angle):
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)
        surf.blit(rotated_image, new_rect.topleft)

    def draw(self, win, x, y):
        img = self.IMGS[1] if self.tilt <= -80 else self.img
        self.blit_rotate_center(win, img, (float(x), float(y)), self.tilt)

def draw_pipe_pair(win, assets, px, height_value, bottom_value):
    if not np.isfinite(px): return
    pipe_top = assets["pipe_top"]
    pipe_bottom = assets["pipe_bottom"]
    top_y = float(height_value) - pipe_top.get_height()
    bottom_y = float(bottom_value)
    win.blit(pipe_top, (float(px), top_y))
    win.blit(pipe_bottom, (float(px), bottom_y))

def reconstruct_all_pipes(cur_px, cur_h, cur_b, nxt_px, nxt_h, nxt_b):
    """Linear extrapolation: spacing & heights from (cur, next)."""
    pipes = []
    # Start with the two we have
    pipes.append((float(cur_px), float(cur_h), float(cur_b)))
    pipes.append((float(nxt_px), float(nxt_h), float(nxt_b)))

    spacing = float(nxt_px - cur_px)
    dh = float(nxt_h - cur_h)
    db = float(nxt_b - cur_b)

    # Guard
    if not np.isfinite(spacing) or spacing <= 5:
        spacing = 200.0  # fallback
    if not np.isfinite(dh): dh = 0.0
    if not np.isfinite(db): db = 0.0

    # Extrapolate to the right (on-screen only)
    px, h, b = float(nxt_px), float(nxt_h), float(nxt_b)
    for _ in range(6):  # a few more possible pipes
        px += spacing
        h += dh
        b += db
        if px - PIPE_VISUAL_WIDTH > WIN_WIDTH + 10:
            break
        pipes.append((px, h, b))

    # Optionally backfill to the left (only if still visible)
    px, h, b = float(cur_px), float(cur_h), float(cur_b)
    for _ in range(2):
        px -= spacing
        h -= dh
        b -= db
        if px + PIPE_VISUAL_WIDTH < -10:
            break
        pipes.append((px, h, b))

    # Keep visible pipes, sorted by x
    pipes = [p for p in pipes if -PIPE_VISUAL_WIDTH <= p[0] <= WIN_WIDTH + PIPE_VISUAL_WIDTH]
    pipes.sort(key=lambda tup: tup[0])
    return pipes

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=str, help="Path to replay .npz")
    ap.add_argument("--fps", type=int, default=30, help="Playback FPS")
    ap.add_argument("--loop", action="store_true", help="Loop playback")
    ap.add_argument("--imgdir", type=str, default="imgs", help="Assets directory")
    ap.add_argument("--overlay", action="store_true", help="Show gap midline + distance/loss")
    ap.add_argument("--loss-weight", type=float, default=0.0, help="Optional loss weight to show weighted loss")
    ap.add_argument("--reconstruct-all", action="store_true", help="Draw all on-screen pipes (extrapolated)")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    pygame.init()
    pygame.display.set_caption("Flappy Bird Replay (game assets)")
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("comicsans", 22)

    assets = load_assets(args.imgdir)
    bg = assets["bg"]
    base = BaseScroller(assets["base"], init_vel=5.0)
    bird_sprite = BirdSprite(assets["birds"])

    R = load_replay(npz_path)
    t = R["t"]; bx = R["bx"]; by = R["by"]; bv = R["bv"]; score = R["score"]
    cur_px = R["cur_px"]; cur_h = R["cur_height"]; cur_b = R["cur_bottom"]
    nxt_px = R["nxt_px"]; nxt_h = R["nxt_height"]; nxt_b = R["nxt_bottom"]
    meta = R["meta"]

    base.set_vel(infer_pipe_vel(cur_px))

    i = 0
    playing = True
    fps = max(1, int(args.fps))
    loop = bool(args.loop)

    def draw_overlay_for_current(idx):
        # Midline from current pipe
        gc = gap_center(cur_h[idx], cur_b[idx])
        loss_px = abs(float(by[idx]) - gc)
        # Horizontal midline
        pygame.draw.line(win, (255, 200, 0), (0, int(gc)), (WIN_WIDTH, int(gc)), 2)
        # Bird→midline connector
        pygame.draw.line(win, (255, 100, 100),
                         (int(bx[idx]), int(by[idx])),
                         (int(bx[idx]), int(gc)), 2)
        # Label
        txt = f"dist={loss_px:.1f}px"
        if args.loss_weight > 0:
            txt += f"  loss={args.loss_weight*loss_px:.3f}"
        surf = font.render(txt, True, (255, 255, 255))
        win.blit(surf, (10, WIN_HEIGHT - 30))

    def draw_frame(idx: int):
        win.blit(bg, (0, 0))

        if args.reconstruct_all:
            pipe_list = reconstruct_all_pipes(cur_px[idx], cur_h[idx], cur_b[idx],
                                              nxt_px[idx], nxt_h[idx], nxt_b[idx])
            for (px, h, b) in pipe_list:
                draw_pipe_pair(win, assets, px, h, b)
        else:
            draw_pipe_pair(win, assets, cur_px[idx], cur_h[idx], cur_b[idx])
            draw_pipe_pair(win, assets, nxt_px[idx], nxt_h[idx], nxt_b[idx])

        vy = float(bv[idx]) if (bv is not None and idx < len(bv)) else 0.0
        bird_sprite.update(vy)
        bird_sprite.draw(win, float(bx[idx]), float(by[idx]))

        base.move()
        base.draw(win)

        # HUD
        hud = [
            f"t={int(t[idx])}",
            f"score={int(score[idx])}",
            f"fps={fps}",
            f"seed={meta.get('seed','?')}",
            f"{'LOOP' if loop else ''} {'PAUSED' if not playing else ''}"
        ]
        y = 8
        for line in hud:
            win.blit(font.render(line, True, (255, 255, 255)), (10, y))
            y += 22

        if args.overlay:
            draw_overlay_for_current(idx)

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
                if loop: i = 0
                else: playing = False; i = len(t) - 1

        i = max(0, min(i, len(t) - 1))
        draw_frame(i)

if __name__ == "__main__":
    main()
