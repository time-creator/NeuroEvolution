from PIL import Image
from pathlib import Path
import torch
import concurrent.futures
import numpy as np
import os
import image_util as imageu
import nima_util as nimau
import torchvision.transforms as transforms

dir_path = os.path.dirname(os.path.realpath(__file__))

def tuple2str(weights):
    start = str(weights)
    start = start.replace('(', '').replace(')', '')
    start = start.replace('.', '').replace(',', '')
    start = start.replace(' ', '')
    return start

def bruteforce_images(image_number):
    """
    """
    possibilities = set()

    for j in range(0, 101, 10):
        if j == 100:
            possibilities.add((100 * 0.01, 0, 0))
        else:
            for k in range(0, 101 - j, 10):
                possibilities.add((round(j * 0.01, 1), round(k * 0.01, 1), round((100 - j - k) * 0.01, 1)))

    for weights in sorted(possibilities):

        im = Image.open(dir_path + f'\\dataset\\dataset({image_number}).jpg')
        pixels = im.load()

        print(f'At {weights} of image {image_number}')
        for i in range(im.size[0]):
            for j in range(im.size[1]):
                greyscale_value = int(np.sum(np.multiply(list(pixels[i, j]), list(weights))))
                pixels[i, j] = (greyscale_value, greyscale_value, greyscale_value)

        if not os.path.exists(Path(dir_path + f'\\output\\bruteforce\\{image_number}')):
            os.mkdir(Path(dir_path + f'\\output\\bruteforce\\{image_number}'))
        new_image_path = Path(dir_path + f'\\output\\bruteforce\\{image_number}\\{tuple2str(weights)}.jpg')
        im.save(new_image_path)

def bruteforce_eval(image_number):
    """
    """
    image_vectors = []
    for filename in os.listdir(dir_path + f'\\output\\bruteforce\\{image_number}'):
        image_vectors.append((imageu.to_nima_vector(dir_path + f'\\output\\bruteforce\\{image_number}\\{filename}'), filename))

    best_score = (-1, "None")
    all_scores = []
    for image in image_vectors:
        print(f'At {image[1]}')
        score = nimau.evaluate_single_mean(image[0])
        all_scores.append((score, image[1]))
        if score > best_score[0]:
            best_score = (score, image[1])

    return all_scores, best_score

def main():
    # 1, 21, 28, 59, 72, 83, 104, 118, 145, 168, 175,
    # images = [208, 241, 255, 265,
    #           280, 296, 307, 323, 336, 377, 391, 420, 433, 454, 474, 487, 510, 531, 545,
    #           559, 582, 613, 626, 635, 643, 677, 709, 720, 754, 792, 805, 811, 848, 994]
    # with concurrent.futures.ProcessPoolExecutor() as executer:
    #     results = executer.map(bruteforce_images, images)

    results = bruteforce_eval(1)
    print(f'best performing = {results[1]}\n Others: {results[0]}')

if __name__ == '__main__':
    main()
