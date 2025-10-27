import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch

def get_accuracy(preds_vec, y):
    batch_size, _ = preds_vec.shape
    # Note: For tensor is one-dimension dim=-1 does nothing
    #       For tensor is [rows, cols], then dim=-1 refers to the cols dimension
    #       For tensor is [rows, cols, widths] then dim=-1 refers to the widths dimension
    # the current operation returns a index along the cols dimension
    predicted_classes = preds_vec.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(y).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

def train_model(trainDataLoader, model, criterion, optimizer, device):
    model.train()
    batch_loss_list = []
    batch_acc_list  = []
    for batch in tqdm(trainDataLoader, desc="Training..."):
        X = batch['ids'].to(device)
        AttMasks = batch['att_masks'].to(device)
        # y is a label 0 or 1 for each sample
        y = batch['label'].to(device)
        # since it is a binary classification, pred_Vec for a sample is a 2 dimensional vector
        preds_vec = model(X, AttMasks)
        batch_loss = criterion(preds_vec, y)
        batch_acc = get_accuracy(preds_vec, y)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        batch_loss_list.append(batch_loss.item())
        batch_acc_list.append(batch_acc.item())
    epoch_loss = np.mean(batch_loss_list)
    epoch_acc = np.mean(batch_acc_list)
    return epoch_loss, epoch_acc

def evaluate(validDataLoader, model, criterion, device):
    # Note: eval() does the followings: 1) In norm-layer: no updates to mean and std
    # 2). In dropout layers: No setting zeros to emulate dropout
    # 3).
    model.eval()
    batch_loss_list = []
    batch_acc_list  = []
    with torch.no_grad():
        for batch in tqdm(validDataLoader, desc="Evaluating..."):
            X = batch['ids'].to(device)
            AttMasks = batch['att_masks'].to(device)
            y = batch['label'].to(device)
            preds_vec = model(X, AttMasks)
            batch_loss = criterion(preds_vec, y)
            batch_acc = get_accuracy(preds_vec, y)
            batch_loss_list.append(batch_loss.item())
            batch_acc_list.append(batch_acc.item())
    epoch_loss = np.mean(batch_loss_list)
    epoch_acc  = np.mean(batch_acc_list)
    return epoch_loss, epoch_acc


def train_model_under_n_epochs(trainDataLoader, validDataLoader, nEpochs, model,criterion, optimizer,device):
    best_valid_loss = np.inf
    epochMetrics = defaultdict(list)
    for epoch in range(nEpochs):
        train_loss, train_acc = train_model(trainDataLoader, model, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(validDataLoader, model, criterion, device)
        epochMetrics['train_loss_list'].append(train_loss)
        epochMetrics['train_acc_list'].append(train_acc)
        epochMetrics['valid_loss_list'].append(valid_loss)
        epochMetrics['valid_acc_list'].append(valid_acc)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "optimal_sentiment_model_state_dict.pt")
        print(f"epoch: {epoch}\n"
              f"train_loss: {train_loss:.5f}, train_acc: {train_acc*100:.3f}%\n"
              f"valid_loss: {valid_loss:.5f}, valid_acc: {valid_acc*100:.3f}%")
    return epochMetrics, best_valid_loss







