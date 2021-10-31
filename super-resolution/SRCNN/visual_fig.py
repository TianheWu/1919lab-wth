import torch
import cv2
from utils.data.Datasets import ImageSet
from models.model import SRCNN
import numpy



def visual(img):
    """
    :param img: The figure path you want to super-resolustion.
    :return: After SRCNN model processing.
    """
    origin_image = cv2.imread(img)
    origin_image = ImageSet.resize_img(origin_image, (500, 500))
    cv2.imwrite('bicubic.jpg', origin_image)
    image_channel = ImageSet.extract_channel(origin_image)
    net = SRCNN(3)
    model_path = "output/epoch149_loss_0.0465_statedict.pt"
    net.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    net.eval()
    with torch.no_grad():
        pred_fig_channel = net(image_channel.unsqueeze(0))
    pred_fig_channel = pred_fig_channel.squeeze(0).detach().numpy() * 255
    ret_fig = ImageSet.instead_channel(origin_image, pred_fig_channel, channel='ycrcb')
    cv2.imwrite('srcnn.jpg', ret_fig)


img = 'origin.jpg'
# img = cv2.imread(img)
# img = ImageSet.resize_img(img, (200, 200))
# cv2.imwrite("origin.jpg", img)
# print("img shape: ", img.shape)
# img = ImageSet.bgr2ycbcr(img)
# print("img shape: ", img.shape)
# cv2.imshow('img: ', img)
# cv2.waitKey(0)
visual(img)