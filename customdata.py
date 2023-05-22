import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm.auto import tqdm

class CustomDataset1(Dataset):  # train
    def __init__(self, image_paths, labels,class_augmentations=None, num_augmentations=1, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
        self.num_augmentations = num_augmentations
        self.class_augmentations = class_augmentations

    def __getitem__(self, idx):
        img_path = self.image_paths[idx % len(self.image_paths)]
        label = self.labels[idx % len(self.labels)]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.bitwise_not(img)

        """ alpha = 1.6
        beta = -70
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = cv2.addWeighted(inverted_img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
        #image = cv2.resize(adjusted_img, dsize=(450, 450), interpolation=cv2.INTER_AREA)"""

        # Apply specific augmentation for the class
        if self.class_augmentations is not None and label in self.class_augmentations:

            img = self.class_augmentations[label](image=img)["image"]
            #print("good")


        # Apply common transformations
        if self.transforms is not None:
            image = self.transforms(image=img)["image"]
            image = image/255
            #image = torch.from_numpy(img).permute(2, 0, 1).float()
        return image, label

    def __len__(self):
        return len(self.image_paths) * self.num_augmentations


class CustomDataset2(Dataset):  # validation ,test
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms


    def __getitem__(self, index):
        img_path = self.img_path_list[index]

        img = cv2.imread(img_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        """alpha = 1.6
        beta = -70
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inverted_img = cv2.bitwise_not(img)
        image = cv2.addWeighted(inverted_img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
        # image = cv2.resize(adjusted_img, dsize=(450, 450), interpolation=cv2.INTER_AREA)"""

        if self.transforms is not None :
            image = self.transforms(image=image)['image']
            image = image / 255
            #image = torch.from_numpy(image).permute(2, 0, 1).float()
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)

class CustomDatasetConfirm(Dataset):  # train
    def __init__(self, image_paths, labels,class_augmentations=None, num_augmentations=1, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
        self.num_augmentations = num_augmentations
        self.class_augmentations = class_augmentations

    def __getitem__(self, idx):
        label = self.labels[idx % len(self.labels)]
        return  label

    def __len__(self):
        return len(self.image_paths) * self.num_augmentations

class CustomDatasetMeta(Dataset):
    def __init__(self, images, labels, transforms=None, augmentation_counts=None,class_augmentations = None):


        self.images  = images
        self.labels = labels
        self.transform = transforms
        self.augmentation_counts = augmentation_counts
        self.class_augmentations = class_augmentations
        #print(images)
        if augmentation_counts:
            augmented_images = []
            augmented_labels = []

            for image, label in tqdm(zip(images, labels)):
                for _ in range(augmentation_counts[label]):
                    #print(image,label)
                    img = cv2.imread(image)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    if self.class_augmentations is not None and label in self.class_augmentations:
                        image_tf = self.class_augmentations[label](image=img)["image"]
                    else:
                        print('error')
                    augmented_images.append(image_tf)
                    augmented_labels.append(label)

            self.images = augmented_images
            self.labels = augmented_labels
            #print(self.images[0],self.labels[0])

        else:
            print("none")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        #print(image)
        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label