import torch
import torch.utils.data
import cv2
import os
import copy
import skimage
import numpy as np

from skimage import img_as_float, img_as_ubyte


class ImageSet(torch.utils.data.Dataset):
    def __init__(self, img, start=0.0, end=1.0):
        """
        :param img: The file path of datasets.
        """
        super(ImageSet, self).__init__()
        self.imgs = [os.path.join(img, file).replace('\\', '/') for file in os.listdir(img) if ImageSet.is_img(file)]
        length = len(self.imgs)
        if length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(img))
        self.imgs = self.imgs[int(length * start):int(length * end)]
    
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
        return any([img.endswith(extension) for extension in [".jpg", ".png", ".jpeg"]])

    def __getitem__(self, index):
        """
        :return: The noise image and label.
        """
        origin_image = cv2.imread(self.imgs[index])
        skimg = ImageSet.open2sk(origin_image)
        noise_skimg = skimage.util.random_noise(skimg, mode='gaussian')
        noise_opimg = ImageSet.sk2open(noise_skimg)
        data = torch.from_numpy(ImageSet.bgr2ycbcr(noise_opimg)[:, :, 0].astype(np.float32)).unsqueeze(0) / 255
        label = torch.from_numpy(ImageSet.bgr2ycbcr(origin_image)[:, :, 0].astype(np.float32)).unsqueeze(0) / 255
        return data, label

    def __len__(self):
        """
        :return: The length of datasets.
        """
        return len(self.imgs)
