from PIL import Image
import numpy as np
import torch
import os
from pathlib import Path
from datetime import datetime
import torchvision.transforms as transforms

dir_path = os.path.dirname(os.path.realpath(__file__))

# creates a pytorch tensor out of an image path
# depending on the parameter flatten, the resulting vector either is of size
# 3 * image height * image width or 1 * (image height * image width)
# actually it's 0 * 3 * .. because we need a batch size for further processing
def png2torchtensor(image_path, scale=True, flatten=False):
    im = Image.open(image_path)
    im = im.resize((128, 128))
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
    print(pixels_vector.size())
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

# creates a greyscale image out of the given image with the given values
# this function always saves the result image
def to_selfmade_grayscale(image_path, r_value, g_value, b_value):
    im = Image.open(image_path)
    pixels = im.load()
    print(im.size[0], im.size[1])
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            grayscale_value = int(np.sum(np.multiply(list(pixels[i, j]), [r_value, g_value, b_value])))
            pixels[i, j] = (grayscale_value, grayscale_value, grayscale_value)
    print("Or got here")
    new_image_path = Path(dir_path + f"\\result_images\\a0096_{int(r_value * 1000)}_{int(g_value * 1000)}_{int(b_value * 1000)}.jpg")
    im.save(new_image_path)

def to_lightness_grayscale(image_path):
    im = Image.open(image_path)
    pixels = im.load()
    print(im.size[0], im.size[1])
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            grayscale_value = int((max(pixels[i, j]) + min(pixels[i, j])) / 2)
            pixels[i, j] = (grayscale_value, grayscale_value, grayscale_value)
    print("Or got here")
    new_image_path = Path(dir_path + f"\\result_images\\a0096_lightness.jpg")
    im.save(new_image_path)

# transforms a jpg image into an pytorch tensor, ready for use in the model
# 224 is size of evo net and nima net
def to_nima_vector(image_path, is_im=False):
    if is_im:
        im = image_path
    else:
        im = Image.open(image_path)
    resize_maker = transforms.Resize((224, 224))
    im = resize_maker.__call__(img = im)
    tensor_maker = transforms.ToTensor()
    im_tensor = tensor_maker.__call__(pic = im)
    im_tensor = im_tensor.unsqueeze_(0)
    return im_tensor

# 224 is the needed size for the NIMACNN (hardcoded in to_nima_vector)
def to_network_vector(image_path):
    return to_nima_vector(image_path)

# makes a network vector to be grayscale for nima to evaluate
def network_and_rgb_to_nima_vector(network_vector, rgb_vector):
    # results in 3 x 224 x 224 tensor
    base_vector = torch.unbind(network_vector, 0)[0]
    red = rgb_vector[0]
    green = rgb_vector[1]
    blue = rgb_vector[2]
    # generate tensors with image size and fill them with the r g b percentage
    red_vector = torch.ones([224, 224]) * red
    green_vector = torch.ones([224, 224]) * green
    blue_vector = torch.ones([224, 224]) * blue
    # unsqueeze so we can add them up to one tensor
    red_vector = torch.unsqueeze(red_vector, 0)
    green_vector = torch.unsqueeze(green_vector, 0)
    blue_vector = torch.unsqueeze(blue_vector, 0)
    # rgb_mul_vector is of size 3 x 224 x 224
    rgb_mul_vector = torch.cat((red_vector, green_vector, blue_vector), 0)
    # only works if pytorch has the RGB order
    # multiply elementwise
    result_vector = base_vector * rgb_mul_vector
    # add the weighted r g b parts back into one size * size tensor, triple that
    # tensor and unsqueeze to get the right tensor size
    grayscale_vector = torch.unbind(result_vector, 0)[0] + torch.unbind(result_vector, 0)[1] + torch.unbind(result_vector, 0)[2]
    grayscale_vector = torch.unsqueeze(grayscale_vector, 0)
    grayscale_vector = torch.cat((grayscale_vector, grayscale_vector, grayscale_vector), 0)
    return torch.unsqueeze(grayscale_vector, 0)

def rgb_vector_to_PIL_image(rgb_vector, image_path):
    im_original = Image.open(image_path)
    im_new = Image.new(mode='RGB', size=im_original.size)
    pixels_new = []
    pixels_original = im_original.load()

    for i in range(im_original.size[1]):
        for j in range(im_original.size[0]):
            grayscale_value = int(np.sum(np.multiply(list(pixels_original[j, i]), rgb_vector)))
            pixels_new.append((grayscale_value, grayscale_value, grayscale_value))

    im_new.putdata(pixels_new)
    im_new.show()
    return im_new

def main():

    # fill in image path
    print(network_and_rgb_to_nima_vector(to_network_vector('image_path'), [0.33, 0.34, 0.33]).size())

if __name__ == '__main__':
    main()
