import skimage.transform as imgtf
import skimage.filters as imgfilt
import skimage.io as imgio
import numpy as np
import random
import os
from tensorflow.python.platform import gfile

def rotate_right(image):
    angle = random.randint(5, 9)
    image = imgtf.rotate(image, angle, mode='wrap', preserve_range=True)
    return np.clip(image, 0, 255).astype(np.uint8)

# Rotate left by random angle
def rotate_left(image):
    angle = -random.randint(5, 9)
    image = imgtf.rotate(image, angle, mode='wrap', preserve_range=True)
    return np.clip(image, 0, 255).astype(np.uint8)

# Scale up by random factor
def scale_up(image):
    scale_x = random.uniform(0.90, 0.95)
    scale_y = scale_x
    c = (image.shape[0] / 2 * (1 - scale_x), image.shape[1] / 2 * (1 - scale_y))
    transform = imgtf.AffineTransform(scale=(scale_x, scale_y), translation=c)
    image = imgtf.warp(image, transform, mode='wrap', preserve_range=True)
    return np.clip(image, 0, 255).astype(np.uint8)

# Scale down by random factor
def scale_down(image):
    scale_x = random.uniform(1.1, 1.15)
    scale_y = scale_x
    c = (image.shape[0] / 2 * (1 - scale_x), image.shape[1] / 2 * (1 - scale_y))
    transform = imgtf.AffineTransform(scale=(scale_x, scale_y), translation=c)
    image = imgtf.warp(image, transform, mode='wrap', preserve_range=True)
    return np.clip(image, 0, 255).astype(np.uint8)

# Squash horizonally by random factor
def squash(image):
    scale_x = random.uniform(1.1, 1.2)
    c = (image.shape[0] / 2 * (1 - scale_x), 0)
    transform = imgtf.AffineTransform(scale=(scale_x, 1.0), translation=c)
    image = imgtf.warp(image, transform, mode='wrap', preserve_range=True)
    return np.clip(image, 0, 255).astype(np.uint8)

# Offset right with random y movement
def offset_right(image):
    offset_x = 2
    offset_y = random.randint(-2, 2)
    transform = imgtf.AffineTransform(translation=(offset_x, offset_y))
    image = imgtf.warp(image, transform, mode='wrap', preserve_range=True)
    return np.clip(image, 0, 255).astype(np.uint8)

# Offset left with random y movement
def offset_left(image):
    offset_x = -2
    offset_y = random.randint(-2, 2)
    transform = imgtf.AffineTransform(translation=(offset_x, offset_y))
    image = imgtf.warp(image, transform, mode='wrap', preserve_range=True)
    return np.clip(image, 0, 255).astype(np.uint8)

# Shear right by random angle
def shear_right(image):
    angle = random.uniform(0.05, 0.10)
    c = (4 * angle, 2)
    transform = imgtf.AffineTransform(shear=angle, translation=c)
    image = imgtf.warp(image, transform, mode='wrap', preserve_range=True)
    return np.clip(image, 0, 255).astype(np.uint8)

# Shear left by random angle
def shear_left(image):
    angle = -random.uniform(0.05, 0.10)
    c = (4 * angle, 2)
    transform = imgtf.AffineTransform(shear=angle, translation=c)
    image = imgtf.warp(image, transform, mode='wrap', preserve_range=True)
    return np.clip(image, 0, 255).astype(np.uint8)

# Add horizontal motion blur
def blur(image):
    sigma = random.uniform(0.75, 1.0)
    image = imgfilt.gaussian(image, sigma=sigma, multichannel=True) * 255
    return np.clip(image, 0, 255).astype(np.uint8)

# Add color noise
def color_noise(image):
    sigma = random.uniform(10.0, 15.0)
    noise = np.random.normal(0.0, sigma, image.shape[0] * image.shape[1] * image.shape[2])
    noise = noise.reshape(image.shape[0], image.shape[1], image.shape[2])
    image = image.astype(np.float32)
    image += noise
    return np.clip(image, 0, 255).astype(np.uint8)

# Dictionary of supported augmentation operations
operations = {
    0: rotate_right,
    1: rotate_left,
    2: scale_up,
    3: scale_down,
    4: squash,
    5: offset_right,
    6: offset_left,
    7: shear_right,
    8: shear_left,
    9: blur,
    10: color_noise}


# Augment the data
def augment(base_dir, aug_count):
    if os.path.isdir(base_dir):
        # Shortcut: load from file
        print("Loading..")
        sub_dirs = [x[0] for x in gfile.Walk(base_dir)]

        # The root directory comes first, so skip it.
        is_root_dir = True
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue

            extensions = ['png']
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if dir_name == base_dir:
                continue

            print("Looking for images in '" + dir_name + "'")
            for extension in extensions:
                file_glob = os.path.join(base_dir, dir_name, '*.' + extension)
                file_list.extend(gfile.Glob(file_glob))
            if not file_list:
                print('No files found')
                continue

            print("Augmenting.." + str(len(file_list)) + " images")
            for file_name in file_list:
                # Perform augmentation by random operation
                image = imgio.imread(file_name)
                selection = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], aug_count, replace=False)
                print(len(selection))
                for operation in selection:
                    augmented = operations[operation](image)
                    imgio.imsave(file_name.replace('.png', str(operation)+'.jpg'), augmented)


if __name__ == '__main__':
    augment("/home/thomas/Projects/Python/flower_photos", 11)
