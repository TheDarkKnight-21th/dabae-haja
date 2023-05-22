import pandas as pd
import numpy as np
import glob

import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

import ttach as tta

from hyperpar import seed_everything,CFG
from customdata import CustomDataset1, CustomDataset2,CustomDatasetMeta
from evaluate import train,inference
from customtransform import class_augmentations, common_transforms
from collections import Counter
import warnings


warnings.filterwarnings(action='ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


seed_everything(CFG['SEED']) # Seed



class BaseModel(nn.Module):
    def __init__(self, num_classes=19):

        super(BaseModel, self).__init__()
        #self.backbone = models.efficientnet_b0(pretrained=True)

        self.model = timm.create_model( 'densenet161', pretrained=True, num_classes=1000)

        self.classifier = nn.Linear(1000, num_classes)

        self.mlp_head = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 19)
    )



    def forward(self, x):

        x = self.model(x)
        x = self.classifier(x)
        #x = self.mlp_head(x)
        return x


def aug_ratio(df, max_augment_count=1):
    class_image_counts = Counter(df["label"])
    print((class_image_counts))
    max_class_count = max(class_image_counts.values())  # ÃÖ´ë Áõ°­ È½¼ö ¼³Á¤

    augmentation_counts = {}
    b= {}
    for cls, count in class_image_counts.items():
        if count < 10:
            augmentation_counts[cls] = 3
        elif count < 100:
            augmentation_counts[cls] = 2
        else:
            augmentation_counts[cls] = 1

    print(augmentation_counts)
    return  augmentation_counts
    #augmentation_counts = {cls: min(max_class_count // count, max_augment_count) for cls, count in
                          # class_image_counts.items()}

if __name__ == "__main__":

    all_img_list = glob.glob('./train/*/*')
    df = pd.DataFrame(columns=['img_path', 'label'])
    df['img_path'] = all_img_list
    df['label'] = df['img_path'].apply(lambda x: str(x).split('/')[2])
    train_, val, _, _ = train_test_split(df, df['label'], test_size=0.3, stratify=df['label'], random_state=CFG['SEED'])
    print(len(df),len(train_),len(val))
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    train_['label'] = le.transform(train_['label'])
    val['label'] = le.transform(val['label'])

    #print(sorted(train_['label'].unique()),'wjm')




    augmentation_counts = aug_ratio(train_,1)
    #print(augmentation_counts)
    #class_weights = class_weight.compute_class_weight('balanced',classes= np.unique(train_["label"]),y= train_["label"])
    #class_weights = torch.tensor(class_weights).float().to(device)

    train_dataset = CustomDatasetMeta(train_['img_path'].values, train_['label'].values,
                                       class_augmentations=class_augmentations,
                                      augmentation_counts=augmentation_counts,
                                       transforms=common_transforms)
    val_dataset = CustomDatasetMeta(val['img_path'].values, val['label'].values,
                                     class_augmentations=class_augmentations,
                                    augmentation_counts=augmentation_counts,
                                     transforms=common_transforms)

    """train_dataset = CustomDataset1(train_['img_path'].values, train_['label'].values,
                                   class_augmentations,num_augmentations=2,transforms=common_transforms)

    val_dataset = CustomDataset1(val['img_path'].values, val['label'].values,
                                 class_augmentations,num_augmentations=3,transforms=common_transforms)
    """
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)



    model = BaseModel()
    model.eval()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,
                                                           threshold_mode='abs', min_lr=1e-8, verbose=True)


    infer_model = train(model, optimizer, train_loader, val_loader, scheduler ,device=device)

    test = pd.read_csv('./test.csv')

    tta_transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 5, 355]),
            tta.Multiply(factors=[0.9, 1, 1.1])

        ]
    )
    tta_model = tta.ClassificationTTAWrapper(infer_model, tta_transforms)

    test_dataset = CustomDataset2(test['img_path'].values, None, common_transforms)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    torch.save(model, f'./model.pt')

    #model1 = torch.load("model.pt", map_location=device)
    preds = inference(infer_model, test_loader,le,device)
    preds2 = inference(tta_model, test_loader,le,device)

    submit = pd.read_csv('./sample_submission.csv')
    submit2 = pd.read_csv('./sample_submission.csv')

    submit['label'] = preds
    submit2['label'] = preds2


    submit.to_csv('./submit.csv', index=False)
    submit2.to_csv('./submit2.csv', index=False)

