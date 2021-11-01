import torch
import torch.utils.data
import cv2
import os
import copy
import skimage
import numpy as np

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
        :return: Ycbcr color.
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    
    @staticmethod
    def ycbcr2bgr(img):
        """
        :param img: Imread image.
        :return: BGR color.
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
        :param img: Image opencv-imread image.
        :return: After resizing image.
        """
        return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    
    @staticmethod
    def extract_channel(img, channel='y'):
        """
        :param img: The image you want to extract.
        :return: The image extracted channel.
        """
        if channel == 'y':
            ret = torch.from_numpy(ImageSet.bgr2ycbcr(img)[:, :, 0].astype(np.float32)).unsqueeze(0) / 255
        elif channel == 'ycrcb':
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
        :param img: The image you want to restore.
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
        origin_image = ImageSet.resize_img(cv2.imread(self.imgs[index]), (400, 400))
        zoom_image = cv2.pyrDown(origin_image, (100, 100))
        zoom_image = ImageSet.resize_img(zoom_image, (400, 400))
        input_image = zoom_image
        if self.args.num_channels == 3:
            data_y = torch.from_numpy(ImageSet.bgr2ycbcr(input_image)[:, :, 0].astype(np.float32)).unsqueeze(0) / 255
            data_cr = torch.from_numpy(ImageSet.bgr2ycbcr(input_image)[:, :, 1].astype(np.float32)).unsqueeze(0) / 255
            data_cb = torch.from_numpy(ImageSet.bgr2ycbcr(input_image)[:, :, 2].astype(np.float32)).unsqueeze(0) / 255
            label_y = torch.from_numpy(ImageSet.bgr2ycbcr(origin_image)[:, :, 0].astype(np.float32)).unsqueeze(0) / 255
            label_cr = torch.from_numpy(ImageSet.bgr2ycbcr(origin_image)[:, :, 1].astype(np.float32)).unsqueeze(0) / 255
            label_cb = torch.from_numpy(ImageSet.bgr2ycbcr(origin_image)[:, :, 2].astype(np.float32)).unsqueeze(0) / 255
            data = torch.cat((data_y, data_cr, data_cb), dim=0)
            label = torch.cat((label_y, label_cr, label_cb), dim=0)
        elif self.args.num_channels == 1:
            data = torch.from_numpy(ImageSet.bgr2ycbcr(input_image)[:, :, 0].astype(np.float32)).unsqueeze(0) / 255
            label = torch.from_numpy(ImageSet.bgr2ycbcr(origin_image)[:, :, 0].astype(np.float32)).unsqueeze(0) / 255
        else:
            raise ValueError("Please input right channels between 3 and 1")

        return data, label

    def __len__(self):
        """
        :return: The length of datasets.
        """
        return len(self.imgs)
