import torch
import torch.utils.data
import cv2
import os
import copy
import skimage

from skimage import img_as_float, img_as_ubyte


class ImageSet(torch.utils.data.Dataset):
    def __init__(self, img):
        """
        :param img: The file path of datasets.
        """
        super(ImageSet, self).__init__()
        self.imgs = [os.path.join(img, file) for file in os.listdir(img) if ImageSet.is_img(file)]
    
    @staticmethod
    def open2sk(img):
        """
        :param img: Any openCV image.
        :return: skimage image.
        """
        return img_as_float(img)

    @staticmethod
    def sk2open(img):
        """
        :param img: Any skimage image.
        :return: openCV image.
        """
        return img_as_ubyte(img)

    @staticmethod
    def bgr2ycbcr(img):
        """
        :param img: Imread image.
        :return: YCBcr color.
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    @staticmethod
    def is_img(img):
        """
        :param img: Image file path.
        :return: Wheter is is an image, type boolean.
        """
        return any([img.endswidth(extension) for extension in [".jpg", ".png", ".jpeg"]])

    def __getitem__(self, index):
        """
        :return: The noise image and label
        """
        image = cv2.imread(self.imgs[index])
        skimg = ImageSet.open2sk(image)
        noise_skimg = skimage.util.random_noise(skimg, mode='gaussian')
        noise_opimg = ImageSet.sk2open(noise_skimg)
        noise_opimg = ImageSet.bgr2ycbcr(noise_opimg)
        label = copy.deepcopy(image)
        return noise_opimg, label

    def __len__(self):
        """
        :return: The length of datasets
        """
        return len(self.imgs)