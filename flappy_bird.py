"""
FIXED Flappy Bird - Compatible with Q-learning
Key fixes:
- Proper physics (tick_count reset)
- Standard difficulty (no progressive changes during training)
- Simpler Pipe class for training
"""

import pygame
import random
import os

pygame.font.init()

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730

# Check if running headless
import os as _os
_HEADLESS = _os.environ.get("SDL_VIDEODRIVER") == "dummy" or _os.environ.get("FB_NO_DISPLAY") == "1"

if not _HEADLESS:
    WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird")
    
    pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
    bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
    bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
    base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())
else:
    # Headless mode - create dummy surfaces
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
    """FIXED Bird class with proper physics"""
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
        """Standard jump"""
        self.vel = -10.5  # Negative = upward
        self.tick_count = 0  # Reset tick count on jump
        self.height = self.y

    def move(self):
        """FIXED: Proper physics simulation"""
        self.tick_count += 1
        
        # FIXED: Simple velocity-based movement (not accumulating tick_count)
        # displacement = initial_velocity + acceleration
        displacement = self.vel * 1 + 1.5 * (self.tick_count)  # gravity acceleration
        
        # Terminal velocity (max fall speed)
        if displacement >= 16:
            displacement = 16
        
        # If moving up, apply small boost
        if displacement < 0:
            displacement -= 2
        
        self.y = self.y + displacement
        
        # Tilt bird based on movement
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

        # Animation cycle
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
        return pygame.mask.from_surface(self.img)


class Pipe:
    """FIXED: Simple pipe with constant difficulty"""
    GAP = 200  # Fixed gap size
    VEL = 5    # Fixed velocity

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
        """Random pipe height"""
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        """Move pipe left"""
        self.x -= self.VEL

    def draw(self, win):
        if win is None:
            return
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird, win):
        """Check collision with bird"""
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
    """Moving floor"""
    VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        """Move base left"""
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
    """Rotate image around center"""
    if surf is None:
        return
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)
    surf.blit(rotated_image, new_rect.topleft)