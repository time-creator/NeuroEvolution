from PIL import Image
import numpy as np
import torch
import os
from pathlib import Path
from datetime import datetime
import torchvision.transforms as transforms

dir_path = os.path.dirname(os.path.realpath(__file__))

def to_fancy_grayscale(image_path):
    """
    Creates a grayscale image out of a given path to an image.
    The grayscaling uses the luma / luminance method.

    Args:
        image_path: path to the image.

    Returns:
        Grayscale version of the given image as PIL Image.
    """
    im = Image.open(image_path)
    pixels = im.load()

    for i in range(im.size[0]):
        for j in range(im.size[1]):
            grayscale_value = int(np.sum(np.multiply(list(pixels[i, j]), [0.299, 0.587, 0.114])))
            pixels[i, j] = (grayscale_value, grayscale_value, grayscale_value)

    return im

def to_simple_grayscale(image_path):
    """
    Creates a grayscale image out of a given path to an image.
    The grayscaling uses the averaging method.

    Args:
        image_path: path to the image.

    Returns:
        Grayscale version of the given image as PIL Image.
    """
    im = Image.open(image_path)
    pixels = im.load()

    for i in range(im.size[0]):
        for j in range(im.size[1]):
            grayscale_value = int(np.sum(list(pixels[i, j])) / 3)
            pixels[i, j] = (grayscale_value, grayscale_value, grayscale_value)

    return im

def to_selfmade_grayscale(image_path, r_value, g_value, b_value):
    """
    Creates a grayscale image out of a given path to an image.
    The grayscaling uses the given weights to calculate the grayscale value.
    The resulting image gets saved.

    Args:
        image_path: path to the image.
        r_value: weight for red.
        g_value: weight for green.
        b_value: weight for blue.
    """
    im = Image.open(image_path)
    pixels = im.load()
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            grayscale_value = int(np.sum(np.multiply(list(pixels[i, j]), [r_value, g_value, b_value])))
            pixels[i, j] = (grayscale_value, grayscale_value, grayscale_value)
    new_image_path = Path(dir_path + f"\\result_images\\a0096_{int(r_value * 1000)}_{int(g_value * 1000)}_{int(b_value * 1000)}.jpg")
    im.save(new_image_path)

def to_lightness_grayscale(image_path):
    """
    Creates a grayscale image out of a given path to an image.
    The grayscaling uses the lightness method.
    The result gets saved.

    Args:
        image_path: path to the image.
    """
    im = Image.open(image_path)
    pixels = im.load()
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            grayscale_value = int((max(pixels[i, j]) + min(pixels[i, j])) / 2)
            pixels[i, j] = (grayscale_value, grayscale_value, grayscale_value)
    new_image_path = Path(dir_path + f"\\result_images\\a0096_lightness.jpg")
    im.save(new_image_path)

# 224 is size of evo net and nima net (evo = squeezenet now)
def to_nima_vector(image_path, is_im=False):
    """
    Transforms a given .jpeg/.jpg image into a pytorch tensor ready for use in
    the differenct components of this project.

    Args:
        image_path: Path to the image that gets converted.
        is_im: Set to true if image_path is not a path but a PIL Image

    Returns:
        Returns a pytorch tensor.
    """
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

def network_and_rgb_to_nima_vector(network_vector, rgb_vector):
    """
    This function takes in a pytorch tensor, fit to be used in this project, and
    a RGB vector to calculate the resulting grayscale tensor. The resulting
    vector can be used as input for the NIMA model.

    Args:
        network_vector: A pytorch tensor of size 1 x 3 x 224 x 224.
        rgb_vector: A 3-element list where the first element represents the
            red_vector part, the second green and the third blue.

    Returns:
        The return value is a pytorch tensor of size 1 x 3 x 224 x 224 that is
        equal to the image represented by the network_vector converted to
        grayscale (using the rgb_vector elements as weights) as tensor.
    """
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
    """
    Turns an image into a grayscale PIL image using the given weights.

    Args:
        rgb_vector: List with the red, green and blue percentages.
        image_path: Path to the image.

    Returns:
        Returns a PIL Image of the grayscale original image.
    """
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

def to_squeezenet_vector(image_path):
    """
    Turns an image into a pytorch tensor ready for use in the pytorch
    squeezenet1_1 implementation.
    From sample execution https://pytorch.org/hub/pytorch_vision_squeezenet/

    Args:
        image_path: Path to the image that will get converted.

    Returns:
        Returns a pytorch tensor that can be used as input for squeezenet1_1
    """
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def main():
    print("Nothing to run in main().")

if __name__ == '__main__':
    main()
