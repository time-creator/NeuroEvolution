import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

squeezynet = models.squeezenet1_1(pretrained=True)

# this vs with torch.no_grad()
squeezynet.eval()

# change layers https://pytorch.org/docs/master/notes/autograd.html
for param in squeezynet.parameters():
    param.requires_grad = False

def main():
    # Replace the fully-connected last part https://pytorch.org/docs/stable/_modules/torchvision/models/squeezenet.html
    final_conv = nn.Conv2d(512, 3, kernel_size=1)
    squeezynet.classifier = nn.Sequential(
                #nn.Dropout(p=0.5),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

    # sample execution https://pytorch.org/hub/pytorch_vision_squeezenet/
    input_image = Image.open(dir_path + '\\dataset\\dataset(1).jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = squeezynet(input_batch)

    print(output[0])
    print(torch.nn.functional.softmax(output[0], dim=0))


def run_and_get_values(final_conv, images):
    """
    Args:
        final_conv: A pytorch Conv2D layer to be used as final_conv layer in the
            squeezenet1_1 model.
        images: List of images in pytorch tensor format. Used as inputs for the
            squeezenet1_1 model.

    Return:
        Returns a list of lists. Each sub-list containing predicted RBG values
        to the respective input image.
    """
    output = []

    # TODO: is this following line possible or do I have to init a squeeze net
    squeezynet.classifier = nn.Sequential(
                #nn.Dropout(p=0.5),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

    with torch.no_grad():
        for image_vector in images:
            output_value = squeezynet(image_vector)
            output.append(torch.nn.functional.softmax(output_value[0], dim=0))

    return output

if __name__ == '__main__':
    main()
