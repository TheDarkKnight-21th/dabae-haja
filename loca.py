import pandas as pd
import numpy as np
import glob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from hyperpar import CFG
from customdata import CustomDataset1,CustomDatasetMeta
from customtransform import class_augmentations,common_transforms
from tqdm.auto import tqdm
from collections import Counter
all_img_list = glob.glob('./train/*/*')
df = pd.DataFrame(columns=['img_path', 'label'])
df['img_path'] = all_img_list
df['label'] = df['img_path'].apply(lambda x: str(x).split('/')[2])


train_, val, _, _ = train_test_split(df, df['label'], test_size=0.3, stratify=df['label'], random_state=CFG['SEED'])
print(len(df),len(train_),len(val))
df_dict = {name:value for name, value in zip(df["label"].unique(),df["label"].value_counts(sort=False))}
train_dict = {name:value for name, value in zip(train_["label"].unique(),train_["label"].value_counts(sort=False))}
val_dict = {name:value for name, value in zip(val["label"].unique(),val["label"].value_counts(sort=False))}
le = preprocessing.LabelEncoder()
df['label'] = le.fit_transform(df['label'])
train_['label'] = le.transform(train_['label'])
val['label'] = le.transform(val['label'])

class_image_counts = Counter(df["label"])
max_class_count = max(class_image_counts.values())# ÃÖ´ë Áõ°­ È½¼ö ¼³Á¤
max_augment_count = 10
augmentation_counts = {cls: min(max_class_count // count,max_augment_count) for cls, count in class_image_counts.items()}
print(augmentation_counts)

data_info = pd.DataFrame(columns=["df",'train','val'])
print("<<<<< df   train   val   >>>>>>")
for d,t,v in zip(df_dict,train_dict,val_dict):
    data_info.loc[d]=[df_dict[d],train_dict[d],val_dict[d]]
   # print(d,"   :   ",df_dict[d],"    /    ",d," : ",train_dict[d],"     /     ",d,"   :   ",val_dict[d])

print(data_info)
train_dataset = CustomDataset1(train_['img_path'].values, train_['label'].values,
                                   class_augmentations,num_augmentations=1,transforms=common_transforms)


val_dataset = CustomDataset1(val['img_path'].values, val['label'].values,
                                 class_augmentations,num_augmentations=1,transforms=common_transforms)

train_tf = pd.DataFrame(columns=['label'])
val_tf = pd.DataFrame(columns=['label'])
for _ in tqdm(range(train_dataset.__len__())):
    x,y = train_dataset.__getitem__(_)

    train_tf.loc[_] = y

for _ in tqdm(range(val_dataset.__len__())):
    x,y = train_dataset.__getitem__(_)
    val_tf.loc[_] = y
train_tf["label"] =le.inverse_transform(np.array(train_tf["label"]))
val_tf["label"] =le.inverse_transform(np.array(val_tf["label"]))
train_dict2 = {name:value for name, value in zip(train_tf["label"].unique(),train_tf["label"].value_counts(sort=False))}
val_dict2 = {name:value for name, value in zip(val_tf["label"].unique(),val_tf["label"].value_counts(sort=False))}
data_info2 = pd.DataFrame(columns=['train','val'])
#print("<<<<< transforn   >>>>>>")
#for d,t,v in zip(df_dict,train_dict2,val_dict2):
#    data_info2.loc[d]=[train_dict2[d],val_dict2[d]]
   # print(d,"   :   ",df_dict[d],"    /    ",d," : ",train_dict[d],"     /     ",d,"   :   ",val_dict[d])
#print(data_info2)"""


augmentation_counts = {cls: min(max_class_count // count, max_augment_count) for cls, count in class_image_counts.items()}
train_dataset2 =  CustomDatasetMeta(train_['img_path'].values, train_['label'].values,
                                   class_augmentations=class_augmentations,augmentation_counts=augmentation_counts,transforms=common_transforms)
val_dataset2 = CustomDatasetMeta(val['img_path'].values, val['label'].values,
                                 class_augmentations=class_augmentations,augmentation_counts=augmentation_counts,transforms=common_transforms)
#print(class_image_counts,max_class_count,augmentation_counts)
train_tf2 = pd.DataFrame(columns=['label'])
val_tf2 = pd.DataFrame(columns=['label'])
for _ in tqdm(range(train_dataset2.__len__())):
    x,y = train_dataset2.__getitem__(_)
    train_tf2.loc[_] = y

for _ in tqdm(range(val_dataset2.__len__())):
    x,y = train_dataset2.__getitem__(_)
    val_tf2.loc[_] = y
train_tf2["label"] =le.inverse_transform(np.array(train_tf2["label"]))
val_tf2["label"] =le.inverse_transform(np.array(val_tf2["label"]))
train_dict3 = {name:value for name, value in zip(train_tf2["label"].unique(),train_tf2["label"].value_counts(sort=False))}
val_dict3 = {name:value for name, value in zip(val_tf2["label"].unique(),val_tf2["label"].value_counts(sort=False))}
data_info3 = pd.DataFrame(columns=['train','train2','val','val2'])
print("<<<<< transforn   >>>>>>")
for d,t,v in zip(df_dict,train_dict3,val_dict3):
    data_info3.loc[d]=[train_dict2[d],val_dict2[d],train_dict3[d],val_dict3[d]]
    #print(d,"   :   ",df_dict[d],"    /    ",d," : ",train_dict3[d],"     /     ",d,"   :   ",val_dict3[d])
print(data_info3)