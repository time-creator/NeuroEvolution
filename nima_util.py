# this will be used to get NIMA to work inside my project
import os

import torch

import torchvision.transforms as transforms
import torchvision.models as models

from model import *

from PIL import Image
import concurrent.futures

# TODO: Does any of this belong into the function evaluate_images? Find out!

base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)

device = torch.device("cpu")

# path to where the epoch-57.pkl file is located (Pretrained model download)
model.load_state_dict(torch.load(os.path.join("PATH/weights", "epoch-57.pkl"), map_location=device))

model.eval()

def evaluate_single_mean(vector):
    output = model(vector)
    output = output.view(10, 1)
    predicted_mean, predicted_std = 0.0, 0.0
    for i, elem in enumerate(output, 1):
        predicted_mean += i * elem
    return predicted_mean.item()

def evaluate_images(vector_list):
    """
    This function evaluates a given list of images using NIMA.

    Args:
        vector_list: List of images in pytorch tensor format. Must be of size to
            be used as NIMA model inputs.

    Return:
        Returns the mean value and standard diviation provided by the NIMA
        model.
    """
    mean_preds = []
    std_preds = []
    for vector in vector_list:
        output = model(vector)
        output = output.view(10, 1)
        predicted_mean, predicted_std = 0.0, 0.0
        for i, elem in enumerate(output, 1):
            predicted_mean += i * elem
        for j, elem in enumerate(output, 1):
            predicted_std += elem * (j - predicted_mean) ** 2
        mean_preds.append(predicted_mean.item())
        std_preds.append(predicted_std.item())
    return mean_preds, std_preds


def main():
    # testing on a single image
    # path to image that will get evaluated
    im = Image.open("PATH")
    # only works with resized to 224 images
    resize_maker = transforms.Resize((224, 224))
    im = resize_maker.__call__(img = im)
    im.show()
    tensor_maker = transforms.ToTensor()
    im_tensor = tensor_maker.__call__(pic = im)
    # refer to image_util and unsqueeze_ documentation and why 4 dim is needed
    im_tensor = im_tensor.unsqueeze_(0)
    print(im_tensor.size())

    # single image:
    mean_preds = []
    std_preds = []
    output = model(im_tensor)
    output = output.view(10, 1)
    predicted_mean, predicted_std = 0.0, 0.0
    for i, elem in enumerate(output, 1):
        predicted_mean += i * elem
    for j, elem in enumerate(output, 1):
        predicted_std += elem * (j - predicted_mean) ** 2
    mean_preds.append(predicted_mean.item())
    std_preds.append(predicted_std.item())
    print(mean_preds, std_preds)

if __name__ == '__main__':
    main()
