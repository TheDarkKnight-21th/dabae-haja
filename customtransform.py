import albumentations as A
from hyperpar import CFG
from albumentations.pytorch.transforms import ToTensorV2
import cv2

class_augmentations = {
    0: A.Compose([ # ga gu su jung
        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3)),
        A.Rotate(limit=20, p=0.8, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.5, 0.9),rotate_limit=0, p=0.5),
        A.ChannelShuffle(p=0.6),

       ]),

    1: A.Compose([
        A.HorizontalFlip(p=0.5),

        A.Rotate(limit=20, p=0.8, border_mode=cv2.BORDER_REPLICATE),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3))
       ]),

    2: A.Compose([
        A.HorizontalFlip(p=0.5),

        A.RandomBrightnessContrast(p=0.5, brightness_limit=-0.3, contrast_limit=(0, 0.5)),
        A.Rotate(limit=45, p=0.8, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.1, 0.3), rotate_limit=0, p=0.5)
    ]),
    3: A.Compose([
        A.HorizontalFlip(p=0.5),

        A.RandomBrightnessContrast(p=0.5, brightness_limit=-0.3, contrast_limit=(-0.5, 0)),
        A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.5, 0.9), rotate_limit=0, p=0.5),

        A.ChannelShuffle(p=0.6),
    ]),
    4: A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=-0.3, contrast_limit=(0, 0.5)),
        A.Rotate(limit=90, p=0.8, border_mode=cv2.BORDER_REPLICATE),

    ]),
    5: A.Compose([
        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=-0.3, contrast_limit=(-0.5, 0.0)),
        A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.5, 0.9), rotate_limit=0, p=0.5),

        A.ChannelShuffle(p=0.6),

    ]),
    6: A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.5, 1), rotate_limit=0, p=0.5),

    ]),
    7: A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.5, 0.8), rotate_limit=0, p=0.5),
    ]),
    8: A.Compose([
        A.RandomBrightnessContrast(p=0.5, brightness_limit=-0.3, contrast_limit=(-0.5, 0.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.5, 0.9),rotate_limit=0, p=0.5),

    ]),
    9: A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=-0.3, contrast_limit=(-0.5, 0.0)),
        A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.5, 0.8), rotate_limit=0, p=0.5),

    ]),
    10: A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.5, 0.8), rotate_limit=0, p=0.5),

    ]),
    11: A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.2, 0.5), rotate_limit=0, p=0.5),

    ]),
    12: A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=-0.3, contrast_limit=(-0.5, 0.0)),
        A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.5, 0.5), rotate_limit=0, p=0.5),
    ]),
    13: A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=-0.3, contrast_limit=(-0.5, 0.0)),
        A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(1, 1.5), rotate_limit=0, p=0.5),
    ]),
    14: A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=-0.3, contrast_limit=(-0.5, 0.0)),
        A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0, 0.5), rotate_limit=0, p=0.5),
    ]),
    15: A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0, 0.5), rotate_limit=0, p=0.5),
    ]),
    16: A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_REPLICATE),
    ]),
    17: A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=-0.3, contrast_limit=(-0.5, 0.0)),
        A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0, 0.5), rotate_limit=0, p=0.5),
    ]),
    18: A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=-0.3, contrast_limit=(-0.5, 0.0)),
        A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_REPLICATE),
    ]),
}

common_transforms = A.Compose([

    #A.InvertImg(p=1),
    A.CLAHE(p=1),
    A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
    #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
    #           p=1.0),
    ToTensorV2()
])


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.CLAHE(p=1),
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.CLAHE(p=1),


    #A.ChannelShuffle(p=0.05),

    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
              p=1.0),
    ToTensorV2()
])

test_transform = A.Compose([
    A.CLAHE(p=1),
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.CLAHE(p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
                p=1.0),
    ToTensorV2()
])