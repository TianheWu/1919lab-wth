from Datasets import ImageSet
import cv2
import os


def augment(img_path, stride):
    """
    This function is to make data augment.
    :param img_path: The origin images path.
    """
    outdir = "aug_dataset"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for file in os.listdir(img_path):
        origin_image_file = os.path.join(img_path, file).replace('\\', '/')
        origin_image = cv2.imread(origin_image_file)
        height = origin_image.shape[0]; width = origin_image.shape[1]
        height_idx = 0; width_idx = 0
        idx = 0
        while height_idx < height:
            width_idx = 0
            while width_idx < width:
                cropped = ImageSet.crop(origin_image, height_idx, height_idx + 32, width_idx, width_idx + 32)
                cv2.imwrite(os.path.join(outdir, file[:-4] + str(idx) + ".png"), cropped)
                if width_idx + 32 + stride > width:
                    break
                width_idx += stride
                idx += 1
            if height_idx + 32 + stride > height:
                break
            height_idx += stride

if __name__ == '__main__':
    augment("dataset", 14)
        

