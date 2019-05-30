from PIL import Image
import numpy as np

""" PIL """

def to_fancy_grayscale(image_path):
    im = Image.open(image_path)
    pixels = im.load()

    for i in range(im.size[0]):
        for j in range(im.size[1]):
            grayscale_value = int(np.sum(np.multiply(list(pixels[i, j]), [0.299, 0.587, 0.114])))
            pixels[i, j] = (grayscale_value, grayscale_value, grayscale_value)

    return im

def to_simple_grayscale(image_path):
    im = Image.open(image_path)
    pixels = im.load()

    for i in range(im.size[0]):
        for j in range(im.size[1]):
            grayscale_value = int(np.sum(list(pixels[i, j])) / 3)
            pixels[i, j] = (grayscale_value, grayscale_value, grayscale_value)

    return im

def main():
    im_fancy = to_fancy_grayscale('D:/workFolder/NeuroEvolution/Images/small_wiki_flower_color.jpg')
    im_fancy.show()

    im_simple = to_simple_grayscale('D:/workFolder/NeuroEvolution/Images/small_wiki_flower_color.jpg')
    im_simple.show()

if __name__ == '__main__':
    main()
