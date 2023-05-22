import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torch.utils.data import Dataset
import albumentations as A

import numpy as np
import os
import pandas as pd
import glob
import cv2

import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import torch

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed °íÁ¤

class CustomDataset1(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        print(1)
        img_path = self.img_path_list[index]

        image = cv2.imread(img_path)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            print("????")
            return image

    def __len__(self):
        return len(self.img_path_list) * 5



class CustomDataset2(Dataset):
    def __init__(self, image_paths, labels, num_augmentations=1, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
        self.num_augmentations = num_augmentations


    def __getitem__(self, idx):

        image_path = self.image_paths[idx % len(self.image_paths)]
        label = self.labels[idx % len(self.labels)]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        return image, label

    def __len__(self):
        return len(self.image_paths) * self.num_augmentations

# Define the augmentations you want


if __name__ == "__main__":

    all_img_list = glob.glob('./train/*/*')
    df = pd.DataFrame(columns=['img_path', 'label'])
    df['img_path'] = all_img_list
    df['label'] = df['img_path'].apply(lambda x: str(x).split('/')[2])
    train_, val, _, _ = train_test_split(df, df['label'], test_size=0.2, stratify=df['label'], random_state=CFG['SEED'])

    le = preprocessing.LabelEncoder()
    train_['label'] = le.fit_transform(train_['label'])
    val['label'] = le.transform(val['label'])

    train_transform = A.Compose([
        A.HorizontalFlip(p=1),

        # Contrast Limited Adaptive Histogram Equalization Àû¿ë
        A.CLAHE(p=1),

        # ¹«ÀÛÀ§·Î channelÀ» ¼¯±â
        A.ChannelShuffle(p=0.5),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
                    p=1.0),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
                    p=1.0),A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
                    p=1.0),
        ToTensorV2()
    ])

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.CLAHE(p=1),
        #A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.8, 0.8), p=1),
        A.VerticalFlip(p=0.5),
        #A.ToGray(p=1),
        #A.ColorJitter(brightness=0, contrast=[0,20], saturation=[0, 0], hue= 0 ),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
    ])

    # Create the dataset
    num_augmentations_per_image = 5
    dataset = CustomDataset2(train_['img_path'].values, train_['label'].values, num_augmentations=num_augmentations_per_image, transforms=transform)
    dataset_loader =  DataLoader(dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


    #train_dataset = CustomDataset1(train_['img_path'].values, train_['label'].values, transform)
    #train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    print(len(dataset))
    #print(len(train_dataset))

    # Visualize some augmented images
    import matplotlib.pyplot as plt


    num_images = 8

    fig, axes = plt.subplots(num_images, 5, figsize=(20, 30))
    cnt = 0
    for i in range(num_images):
        # Original image
        for k in range(0,5):
            image, label = dataset[i+1]
            #image = image.permute(1, 2, 0).numpy()
            #image = 255 - image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            axes[i, k].imshow(image)
            axes[i,k].set_title(label)

    plt.show()


