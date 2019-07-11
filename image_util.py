from PIL import Image
import numpy as np
import torch
import os
from pathlib import Path
from datetime import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))

# creates a pytorch tensor out of an image path
# depending on the parameter flatten, the resulting vector either is of size
# 3 * image height * image width or 1 * (image height * image width)
# actually it's 0 * 3 * .. because we need a batch size for further processing
def png2torchtensor(image_path, scale=True, flatten=False):
    im = Image.open(image_path)
    pixels = im.load()
    pixels_vector = []
    r_vector = []
    g_vector = []
    b_vector = []

    for i in range(im.size[0]):
        for j in range(im.size[1]):
            for k in range(len(pixels[i, j])): # 0, 1, 2

                if flatten:
                    if scale:
                        pixels_vector.append((k / 255) * 0.99 + 0.01)
                    else:
                        pixels_vector.append(k)
                else:
                    if k == 0:
                        if scale:
                            r_vector.append((k / 255) * 0.99 + 0.01)
                        else:
                            r_vector.append(k)
                    elif k == 1:
                        if scale:
                            g_vector.append((k / 255) * 0.99 + 0.01)
                        else:
                            g_vector.append(k)
                    else:
                        if scale:
                            b_vector.append((k / 255) * 0.99 + 0.01)
                        else:
                            b_vector.append(k)

    if flatten:
        pixels_vector = torch.tensor(pixels_vector).float()
    else:
        r_vector = torch.reshape(torch.tensor(r_vector).float(), (im.size[0], im.size[1]))
        g_vector = torch.reshape(torch.tensor(g_vector).float(), (im.size[0], im.size[1]))
        b_vector = torch.reshape(torch.tensor(b_vector).float(), (im.size[0], im.size[1]))

        # stack along which axis; 0 right now
        pixel_tensor = torch.stack((r_vector, g_vector, b_vector), dim=0)
        pixels_vector = pixel_tensor.unsqueeze_(0)

    return pixels_vector

# creates a fancy grayscale image out of an image path
def to_fancy_grayscale(image_path):
    im = Image.open(image_path)
    pixels = im.load()

    for i in range(im.size[0]):
        for j in range(im.size[1]):
            grayscale_value = int(np.sum(np.multiply(list(pixels[i, j]), [0.299, 0.587, 0.114])))
            pixels[i, j] = (grayscale_value, grayscale_value, grayscale_value)

    return im

# creates a grayscale image out of an image path
def to_simple_grayscale(image_path):
    im = Image.open(image_path)
    pixels = im.load()

    for i in range(im.size[0]):
        for j in range(im.size[1]):
            grayscale_value = int(np.sum(list(pixels[i, j])) / 3)
            pixels[i, j] = (grayscale_value, grayscale_value, grayscale_value)

    return im

# returns a list of the first pixel values (grayscale values since they're all
# the same) of an PIL image
def grayscale_image_to_list(image):
    pixel_list = []
    im = image
    pixels = im.load()

    for i in range(im.size[0]):
        for j in range(im.size[1]):
            pixel_list.append(pixels[i, j][0]) # 0 since the tupel is 3 times the same number

    return pixel_list


# TODO: don't hardcode image size!
def show_result_image(result_data, image_path, save=False):
    rgb_vector = result_data
    im_new = Image.new(mode='RGB', size=(128, 128))
    im_original = Image.open(image_path)
    pixels_new = []
    pixels_original = im_original.load()

    for i in range(im_original.size[0]):
        for j in range(im_original.size[1]):
            # have to do [j, i] to reverse the rotation from putdata()
            grayscale_value = int(np.sum(np.multiply(list(pixels_original[j, i]), rgb_vector)))
            pixels_new.append((grayscale_value, grayscale_value, grayscale_value))

    im_new.putdata(pixels_new)
    im_new.show()

    if save:
        new_image_path = Path(dir_path + f"\\result_images\\image_{str(datetime.now())[:-7].replace(' ', '_').replace(':', '-')}.png")
        im_new.save(new_image_path)

def main():

    im_fancy = to_fancy_grayscale('D:/workFolder/NeuroEvolution/thumbnails/00975.png')
    im_fancy.show()

    print(grayscale_image_to_list(im_fancy))

    im_simple = to_simple_grayscale('D:/workFolder/NeuroEvolution/thumbnails/00975.png')
    im_simple.show()

if __name__ == '__main__':
    main()
