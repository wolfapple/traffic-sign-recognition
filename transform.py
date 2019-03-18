import PIL
import cv2
from torchvision import transforms


class CLAHE_GRAY:
    def __init__(self, clipLimit=2.5, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, im):
        img_y = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit,
                                tileGridSize=self.tileGridSize)
        img_y = clahe.apply(img_y)
        img_output = img_y.reshape(img_y.shape + (1,))
        return img_output


def get_train_transforms():
    return transforms.Compose([
        CLAHE_GRAY(),
        transforms.ToPILImage(),
        transforms.RandomApply([
            transforms.RandomRotation(20, resample=PIL.Image.BICUBIC),
            transforms.RandomAffine(0, translate=(
                0.2, 0.2), resample=PIL.Image.BICUBIC),
            transforms.RandomAffine(0, shear=20, resample=PIL.Image.BICUBIC),
            transforms.RandomAffine(0, scale=(0.8, 1.2),
                                    resample=PIL.Image.BICUBIC)
        ]),
        transforms.ToTensor()
    ])


def get_test_transforms():
    return transforms.Compose([
        CLAHE_GRAY(),
        transforms.ToTensor()
    ])
