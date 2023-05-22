from tqdm.auto import tqdm
from earlystop import EarlyStopping
import torch.nn as nn
import torch
from sklearn.metrics import f1_score
from sklearn import preprocessing
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from hyperpar import CFG

def train(model, optimizer, train_loader, val_loader, scheduler, device):
    early_stopping = EarlyStopping( verbose=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_score = 0
    best_model = None

    for epoch in range(1,CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        preds,true_labels = [],[]
        total =0
        correct = 0
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(imgs)
            preds += output.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            _, predict = torch.max(output.data, 1)
            loss = criterion(output, labels)


            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            total += labels.size(0)
            correct += (predict == labels).sum()
            _train_score = f1_score(true_labels, preds, average='weighted')
        _val_loss, _val_score,_val_acc = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        early_stopping(_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        print(
            f'\nEpoch [{epoch}], Train Loss : [{_train_loss:.5f}] , Train Accuracy : [{correct/total*100}]% , Train W F1 scire[{_train_score:.5f}]\n Val Loss : [{_val_loss:.5f}] Val Accuracy : [{_val_acc}]% , Val Weighted F1 Score : [{_val_score:.5f}]')

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_score < _val_score:
            best_score = _val_score
            best_model = model

    return best_model


def inference(model, test_loader,le,device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)

            pred = model(imgs)

            preds += pred.argmax(1).detach().cpu().numpy().tolist()

    preds = le.inverse_transform(preds)
    return preds
def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, true_labels = [], []
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            pred = model(imgs)

            loss = criterion(pred, labels)

            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            _, predict = torch.max(pred.data, 1)
            val_loss.append(loss.item())

            total += labels.size(0)
            correct += (predict == labels).sum()
        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='weighted')
        _val_acc = correct/total*100

    return _val_loss, _val_score, _val_acc