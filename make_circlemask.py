import numpy as np
from PIL import Image, ImageFilter
import random

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

# mask = create_circular_mask(32, 32, center=None)
# im = Image.fromarray(mask)
# im.save("filename.png")

for i in range(10000):
    h = random.uniform(10, 118)
    w = random.uniform(10, 118)
    r = random.uniform(10, 15)
    mask = create_circular_mask(128, 128, center=(h, w), radius=r)*255

    mask = Image.fromarray(mask.astype(np.uint8))
    mask.convert("L")
    
    mask = mask.filter(ImageFilter.BLUR)
    mask = mask.resize((256, 256))
    number = str(i).zfill(6)
    mask.save("/home/circle_mask/" + number + "_smooth.png")
    print("/home/circle_mask/" + number + "_smooth.png")