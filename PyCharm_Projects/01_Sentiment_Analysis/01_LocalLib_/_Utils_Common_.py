import numpy as np
from matplotlib import pyplot as plt
import torch

def setInitSeed(seedVal: int):
    np.random.seed(seedVal)
    torch.manual_seed(seedVal)
    torch.cuda.manual_seed(seedVal)
    # deterministic algo. for convolution operations
    torch.backends.cudnn.deterministic = True

def mapDataSample_to_numerical(dataSample: dict, tokenizer):
    # dataSample has two keys: "text" and "label"
    # tokenizer output has three keys: "input_ids", "token_type_ids", "attention_mask"
    idList = tokenizer(dataSample["text"], truncation=True)["input_ids"]
    outDict = {"ids": idList}
    return outDict

def plot_metric_vs_epoch(metrics, metricType: str):
    n_epochs = len(metrics["train_losses"])
    if metricType == "loss":
        trainKey = "train_loss_list"
        validKey = "valid_loss_list"
        trainLabel= "train loss"
        validLabel= "valid loss"
        yLabel = "loss"
    else:
        trainKey = "train_acc_list"
        validKey = "valid_acc_list"
        trainLabel = "train acc"
        validLabel = "valid acc"
        yLabel="acc"

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics[trainKey], label=trainLabel)
    ax.plot(metrics[validKey], label=validLabel)
    ax.set_xlabel("epoch")
    ax.set_ylabel(yLabel)
    ax.set_xticks(range(n_epochs))
    ax.legend()
    ax.grid()