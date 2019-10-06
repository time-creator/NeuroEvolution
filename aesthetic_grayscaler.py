import squeezenet_util as squeezeu
import image_util as imageu
import nima_util as nimau
from pathlib import Path
from PIL import Image
import os
import torch
import torch.nn as nn

dir_path = os.path.dirname(os.path.realpath(__file__))

def _load_layer(gen_number, performance_rank):
    layer = nn.Conv2d(512, 3, kernel_size=1)
    layer.weight.data = torch.load(Path(dir_path + f'\\generations\\gen{gen_number}\\weights{performance_rank}.pt'))
    return layer

def _load_image(image_path):
    return imageu.to_squeezenet_vector(image_path)

def _show_results(gen_number, performance_rank, image_path):
    image_vector = _load_image(image_path)
    layer = _load_layer(gen_number, performance_rank)
    rgb_vector = squeezeu.run_and_get_values(layer, [image_vector])[0]
    nima_vector = imageu.network_and_rgb_to_nima_vector(image_vector, rgb_vector)

    result_image = imageu.rgb_vector_to_PIL_image(rgb_vector.numpy(), image_path)

    return nimau.evaluate_images([nima_vector]), rgb_vector, result_image

def main():
    gen_number = 10
    performance_rank = 0
    image_path = 'PATH'

    results = _show_results(gen_number, performance_rank, image_path)

    print(results[0]) # NIMA results
    print(results[1]) # RGB vector
    results[2].show() # result image in grayscale

    # Other grayscale method images:
    simple_grayscale_image = imageu.to_simple_grayscale(image_path)
    simple_grayscale_image.show()


if __name__ == '__main__':
    main()
