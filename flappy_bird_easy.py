# file: flappy_bird_easy.py
"""
EASY MODE Flappy Bird - Configurable difficulty for AI training

Key changes you can make:
1. Larger pipe gaps
2. Slower pipe movement
3. Reduced gravity
4. Gentler jump
5. Wider pipes (easier to avoid)
6. More forgiving collision detection
"""

import pygame
import random
import os

pygame.font.init()

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730

# ==================== DIFFICULTY SETTINGS ====================

# PIPE DIFFICULTY
PIPE_GAP = 250  # Default: 200 | Easier: 250-300 | Harder: 150-180
PIPE_VELOCITY = 4  # Default: 5 | Easier: 3-4 | Harder: 6-8

# BIRD PHYSICS
GRAVITY_ACCEL = 1.2  # Default: 1.5 | Easier: 0.8-1.2 | Harder: 2.0+
JUMP_STRENGTH = -9.0  # Default: -10.5 | Easier: -8 to -9 | Harder: -12+
TERMINAL_VELOCITY = 14  # Default: 16 | Easier: 12-14 | Harder: 18+

# COLLISION TOLERANCE
COLLISION_SHRINK = 0.85  # Default: 1.0 | Easier: 0.7-0.9 | Range: 0.5-1.0
# Lower = more forgiving (bird hitbox is smaller)

# PIPE SPACING
MIN_PIPE_HEIGHT = 80  # Default: 50 | Easier: 100-150 | Range: 50-200
MAX_PIPE_HEIGHT = 400  # Default: 450 | Easier: 350-400 | Range: 300-450

# =============================================================

# Check if running headless
import os as _os
_HEADLESS = _os.environ.get("SDL_VIDEODRIVER") == "dummy" or _os.environ.get("FB_NO_DISPLAY") == "1"

if not _HEADLESS:
    WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - Easy Mode")
    
    pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
    bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
    bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
    base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())
else:
    WIN = None
    pipe_img = pygame.Surface((52, 320))
    pipe_img.fill((0, 255, 0))
    bg_img = pygame.Surface((600, 900))
    bg_img.fill((135, 206, 250))
    bird_images = [pygame.Surface((34, 24)) for _ in range(3)]
    for img in bird_images:
        img.fill((255, 255, 0))
    base_img = pygame.Surface((336, 112))
    base_img.fill((222, 216, 149))


class Bird:
    """Bird with configurable physics"""
    MAX_ROTATION = 25
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        """Jump with configurable strength"""
        self.vel = JUMP_STRENGTH
        self.tick_count = 0
        self.height = self.y

    def move(self):
        """Move with configurable gravity"""
        self.tick_count += 1
        
        # Displacement with configurable gravity
        displacement = self.vel * 1 + GRAVITY_ACCEL * (self.tick_count)
        
        # Terminal velocity
        if displacement >= TERMINAL_VELOCITY:
            displacement = TERMINAL_VELOCITY
        
        if displacement < 0:
            displacement -= 2
        
        self.y = self.y + displacement
        
        # Tilt
        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        if win is None:
            return
            
        self.img_count += 1

        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2

        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    def get_mask(self):
        """Get collision mask with optional shrinking for forgiveness"""
        if COLLISION_SHRINK >= 1.0:
            # No shrinking needed
            return pygame.mask.from_surface(self.img)
        
        # Shrink the bird image before creating mask
        original_size = self.img.get_size()
        new_width = int(original_size[0] * COLLISION_SHRINK)
        new_height = int(original_size[1] * COLLISION_SHRINK)
        
        # Create a smaller version
        shrunk_img = pygame.transform.scale(self.img, (new_width, new_height))
        
        # Create a transparent surface of original size
        padded_surface = pygame.Surface(original_size, pygame.SRCALPHA)
        padded_surface.fill((0, 0, 0, 0))
        
        # Center the shrunk image
        offset_x = (original_size[0] - new_width) // 2
        offset_y = (original_size[1] - new_height) // 2
        padded_surface.blit(shrunk_img, (offset_x, offset_y))
        
        return pygame.mask.from_surface(padded_surface)


class Pipe:
    """Pipe with configurable difficulty"""
    GAP = PIPE_GAP
    VEL = PIPE_VELOCITY

    def __init__(self, x):
        self.x = x
        self.height = 0
        
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img
        
        self.passed = False
        self.set_height()

    def set_height(self):
        """Random pipe height with configurable range"""
        self.height = random.randrange(MIN_PIPE_HEIGHT, MAX_PIPE_HEIGHT)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        """Move pipe with configurable velocity"""
        self.x -= self.VEL

    def draw(self, win):
        if win is None:
            return
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird, win):
        """Check collision with configurable forgiveness"""
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if b_point or t_point:
            return True
        return False


class Base:
    """Moving floor - velocity matches pipes"""
    VEL = PIPE_VELOCITY
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        if win is None:
            return
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blitRotateCenter(surf, image, topleft, angle):
    if surf is None:
        return
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)
    surf.blit(rotated_image, new_rect)


def print_difficulty_settings():
    """Print current difficulty settings"""
    print("\n" + "=" * 60)
    print("FLAPPY BIRD - EASY MODE SETTINGS")
    print("=" * 60)
    print(f"Pipe Gap:          {PIPE_GAP} pixels (default: 200)")
    print(f"Pipe Velocity:     {PIPE_VELOCITY} px/frame (default: 5)")
    print(f"Gravity:           {GRAVITY_ACCEL} px/frame² (default: 1.5)")
    print(f"Jump Strength:     {JUMP_STRENGTH} px/frame (default: -10.5)")
    print(f"Terminal Velocity: {TERMINAL_VELOCITY} px/frame (default: 16)")
    print(f"Collision Shrink:  {COLLISION_SHRINK}x (default: 1.0)")
    print(f"Pipe Height Range: {MIN_PIPE_HEIGHT}-{MAX_PIPE_HEIGHT} (default: 50-450)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_difficulty_settings()