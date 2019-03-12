import PIL
import cv2
from torchvision import transforms


class CLAHE:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, im):
        img_yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit,
                                tileGridSize=self.tileGridSize)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return img_output


class CLAHE_GRAY:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, im):
        img_y = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit,
                                tileGridSize=self.tileGridSize)
        img_y = clahe.apply(img_y)
        img_output = img_y.reshape(img_y.shape + (1,))
        return img_output


def get_train_transforms(gray=True, augmentation=True):
    if gray:
        preprocess = CLAHE_GRAY()
        mean = (0.4715,)
        std = (0.2415,)
    else:
        preprocess = CLAHE()
        mean = (0.4898, 0.4619, 0.4708)
        std = (0.2476, 0.2441, 0.2514)
    trans = []
    trans.append(preprocess)
    if augmentation:
        trans.append(transforms.ToPILImage())
        trans.append(
            transforms.RandomApply([
                transforms.RandomRotation(20, resample=PIL.Image.BICUBIC),
                transforms.RandomAffine(0, translate=(
                    0.2, 0.2), resample=PIL.Image.BICUBIC),
                transforms.RandomAffine(
                    0, shear=20, resample=PIL.Image.BICUBIC),
                transforms.RandomAffine(
                    0, scale=(0.8, 1.2), resample=PIL.Image.BICUBIC)
            ]))
    trans.append(transforms.ToTensor())
    trans.append(transforms.Normalize(mean, std))

    return transforms.Compose(trans)


def get_test_transforms(gray=True):
    return transforms.Compose([
        CLAHE_GRAY() if gray else CLAHE(),
        transforms.ToTensor(),
        transforms.Normalize((0.4715,), (0.2415,))
    ])
