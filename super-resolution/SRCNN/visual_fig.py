import torch
import cv2
from utils.data.Datasets import ImageSet
from models.model import SRCNN


def visual(img):
    """
    :param img: The figure path you want to super-resolustion.
    :return: After SRCNN model processing.
    """
    origin_image = cv2.imread(img)
    height = origin_image.shape[0]
    width = origin_image.shape[1]
    origin_image = ImageSet.resize_img(origin_image, (1500, 1500))
    cv2.imwrite('bicubic.jpg', origin_image)
    image_channel = ImageSet.extract_channel(origin_image, channel=1)
    net = SRCNN(1)
    model_path = "output/epoch199_loss_0.0003_statedict.pt"
    net.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    net.eval()
    with torch.no_grad():
        pred_fig_channel = net(image_channel.unsqueeze(0))
    pred_fig_channel = pred_fig_channel.squeeze(0).squeeze(0).detach().numpy() * 255
    ret_fig = ImageSet.instead_channel(origin_image, pred_fig_channel, channel=1)
    cv2.imwrite('srcnn.jpg', ret_fig)


img = '1.jpg'
image = cv2.imread(img)
print(image.shape)
print(type(image))
# visual(img)
