import numpy as np
from PIL import Image, ImageFilter
import random

def create_rectangular_mask(H, W, center):

    mask_w = np.ones([H, W])
    mask_b = np.zeros([H, W])
    mask_w_ = Image.fromarray(mask_w)
    mask_b_ = Image.fromarray(mask_b)

    rotation=[-90, 90]

    width=[int(W/4), int(W)-1]
    length=[int(H/16), int(H/4) - 10]

    cut_dy = random.randint(*width)
    cut_dx = random.randint(*length)

    from_location_y = random.randint(0, H - cut_dy - 1)
    from_location_x = random.randint(0, W - cut_dx - 1)

    box = [from_location_x, from_location_y, from_location_x + cut_dx, from_location_y + cut_dy]
    patch = mask_w_.crop(box)

    rot_deg = random.uniform(*rotation)
    patch = patch.rotate(rot_deg, expand=True)

    p_x, p_y = np.array(patch).shape
    if p_x >= 128:
        p_x = 127
    if p_y >= 128:
        p_x = 127
    patch = patch.resize((p_y, p_x))

    mask_width, mask_height = patch.size
    to_location_y = random.randint(0, H - mask_height - 1)
    to_location_x = random.randint(0, W - mask_width - 1)
    mask = np.array(patch)

    mask_b[to_location_y : to_location_y + mask_height, to_location_x : to_location_x + mask_width] = mask*255

    return mask_b

for i in range(10000):
    h = random.uniform(50, 78)
    w = random.uniform(50, 78)
    mask = create_rectangular_mask(H=128, W=128, center=(h, w))

    mask = Image.fromarray(mask.astype(np.uint8))
    mask.convert("L")
    mask = mask.filter(ImageFilter.BLUR)
    mask = mask.resize((256, 256))

    number = str(i).zfill(6)
    mask.save("/home/rectangular_mask/" + number + "_smooth.png")
    print("/home/rectangular_mask/" + number + "_smooth.png")

