import numpy as np
import torch
from collections import deque
from hyperpar import CFG
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.cycle = 0
        self.scorebox = deque([0]*(round(CFG["EPOCHS"])-1))

    def __call__(self, val_loss, model):
        score = val_loss



        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > max(self.scorebox) + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score < min(self.scorebox):
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        self.scorebox.appendleft(score)
        self.scorebox.pop()

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss