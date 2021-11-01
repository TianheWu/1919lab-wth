import torch
import torch.utils.data
import cv2
import os
import skimage
import numpy as np
import skimage.util

from skimage import img_as_float, img_as_ubyte


class ImageSet(torch.utils.data.Dataset):
    def __init__(self, img, start=0.0, end=1.0, args=None):
        """
        :param img: The file path of datasets.
        """
        super(ImageSet, self).__init__()
        self.args = args
        self.imgs = [os.path.join(img, file).replace('\\', '/') for file in os.listdir(img) if ImageSet.is_img_file(file)]
        length = len(self.imgs)
        if length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(img))
        self.imgs = self.imgs[int(length * start):int(length * end)]
    
    @staticmethod
    def open2sk(img):
        """
        :param img: Any open-cv image.
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
        :param img: Any open-cv image.
        :return: Ycbcr color image.
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    
    @staticmethod
    def ycbcr2bgr(img):
        """
        :param img: Any open-cv image.
        :return: BGR color image.
        """
        return cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def is_img_file(img_path):
        """
        :param img: Image file path.
        :return: Wheter is is an image, type boolean.
        """
        return any([img_path.endswith(extension) for extension in [".jpg", ".png", ".jpeg"]])
    
    @staticmethod
    def resize_img(img, size=(500, 500)):
        """
        :param img: Any open-cv image.
        :param size: The size of image you want to resize.
        :return: After resizing image.
        """
        return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    
    @staticmethod
    def add_noise(img, method='gaussian'):
        """
        :param img: Any opencv-type image.
        :return: Image with gaussian_noise.
        """
        img = ImageSet.open2sk(img)
        return ImageSet.sk2open(skimage.util.random_noise(img, mode=method, var=0.01))

    @staticmethod
    def crop(img, y1, y2, x1, x2):
        """
        :param img: Any open-cv image.
        :param size: The croping size.
        :return: Croped image.
        """
        return img[y1:y2, x1:x2]
    
    @staticmethod
    def center(img):
        """
        :param img: Any open-cv image.
        :return: The image center location (y, x).
        """
        return img.shape[0] // 2, img.shape[1] // 2

    @staticmethod
    def extract_channel(img, channel=1):
        """
        :param img: Any open-cv image.
        :return: The image extracted channel.
        """
        if channel == 1:
            ret = torch.from_numpy(ImageSet.bgr2ycbcr(img)[:, :, 0].astype(np.float32)).unsqueeze(0) / 255
        elif channel == 3:
            data_y = torch.from_numpy(ImageSet.bgr2ycbcr(img)[:, :, 0].astype(np.float32)).unsqueeze(0) / 255
            data_cr = torch.from_numpy(ImageSet.bgr2ycbcr(img)[:, :, 1].astype(np.float32)).unsqueeze(0) / 255
            data_cb = torch.from_numpy(ImageSet.bgr2ycbcr(img)[:, :, 2].astype(np.float32)).unsqueeze(0) / 255
            ret = torch.cat((data_y, data_cr, data_cb), dim=0)
        else:
            raise ValueError("Please input right channel between y and ycrcb")
        return ret
    
    @staticmethod
    def instead_channel(origin_img, image, channel='y'):
        """
        :param origin_img: The image you want to restore.
        :param image: The goal image.
        :return: The image restored.
        """
        origin_image = ImageSet.bgr2ycbcr(origin_img)
        if channel == 'y':
            for i in range(origin_image.shape[0]):
                for j in range(origin_image.shape[1]):
                    origin_image[i][j][0] = image[i][j]

        elif channel == 'ycrcb':
            for i in range(origin_image.shape[0]):
                for j in range(origin_image.shape[1]):
                    origin_image[i][j][0] = image[0][i][j]
                    origin_image[i][j][1] = image[1][i][j]
                    origin_image[i][j][2] = image[2][i][j]
        else:
            raise ValueError("Please input right channel between y and ycrcb")
        ret_img = ImageSet.ycbcr2bgr(origin_image)
        return ret_img

    def __getitem__(self, index):
        """
        :return: The noise image and label.
        """
        origin_image = cv2.imread(self.imgs[index])
        noise_image = ImageSet.add_noise(origin_image)
        center_y, center_x = ImageSet.center(origin_image)
        y1 = center_y - 10; y2 = center_y + 9
        x1 = center_x - 10; x2 = center_x + 9
        central_image = ImageSet.crop(origin_image, y1, y2, x1, x2)
        data = ImageSet.extract_channel(noise_image, channel=self.args.num_channels)
        label = ImageSet.extract_channel(central_image, channel=self.args.num_channels)
        return data, label

    def __len__(self):
        """
        :return: The length of datasets.
        """
        return len(self.imgs)
