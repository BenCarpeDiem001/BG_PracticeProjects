#------------------------------------
# %% [1] Import Standard Libraries |
#------------------------------------
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch
import os
from collections import OrderedDict
sepLen = 50

#--------------------------------------------------------------------------
# %% [2] Plotting Functions |
#--------------------------------------------------------------------------
# Function to reverse the transformations for a displayable image
def revertStandardizedImageTensor(img_tensor, imgMean, imgStd):
    # Reverse normalization
    for t, m, s in zip(img_tensor, imgMean, imgStd):
        t.mul_(s).add_(m)

    # Transpose channels (C, H, W) to (H, W, C)
    img_tensor = img_tensor.permute(1, 2, 0)

    # Convert to NumPy array and scale to 0-255
    img_numpy = img_tensor.detach().cpu().numpy()
    img_numpy = (img_numpy * 255).astype(np.uint8)

    # Convert to PIL Image
    pil_image = Image.fromarray(img_numpy)
    return pil_image

def plotImages_Dataset(dataset, classLabel, nRows, nCols, imgMean, imgStd):
    nClasses  = len(classLabel)
    myFig, myAxes = plt.subplots(nRows, nCols)
    # dataset = test_dataloader.dataset
    labelList = [item[1] for item in dataset]

    indList_UniqueClass = [labelList.index(i) for i in range(nClasses)]

    data = [dataset[idx] for idx in indList_UniqueClass]
    myAxes = myAxes.flatten()
    for curData, curAx in zip(data, myAxes):
        y = curData[1]
        # initial image shape for torch data is [3, 32,32] for cifar10
        img = curData[0]
        # imgShapeFlip = list(img.shape)
        # imgShape = [imgShapeFlip[i] for i in np.arange(len(imgShapeFlip) - 1, -1, -1)]
        # img = img.reshape(imgShape)
        # permute the channels to get the right shape [32, 32, 3]
        #img = img.permute(1, 2, 0)
        #print(type(img))
        if len(imgMean) == 3:
            img = revertStandardizedImageTensor(img, imgMean, imgStd)

        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0)
            curAx.imshow(img, cmap='gray')
        else:
            curAx.imshow(img)  # , cmap='gray')
        curAx.set_title(str(y) + ':' + classLabel[y])
    myFig.suptitle('All Class Labels in the Dataset')
    #plt.ioff()
    plt.show() #(block=False)

#----------------------------------------------------------------
# %% [3] Functions for Good to know information before training|
#----------------------------------------------------------------
def printInfo_DataLoader(curDataLoader, dataCategory):
    print("="*sepLen)
    print(f">>>  {dataCategory}  <<<")
    totalNumSamples = curDataLoader.dataset.__len__()
    totalNumBatches = curDataLoader.__len__()
    print(f"Total Number of Samples: {totalNumSamples}")
    print(f"Total Number of Batches: {totalNumBatches}")
    for X, y in curDataLoader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    print("-"*sepLen)

def printInfo_PackageVersions(info):  # info is a list
    print("=" * sepLen)
    print(">>> Package Version Info <<<")
    for item in info:
        print(item)
    print("-" * sepLen)

def printInfo_Model(model):
    print("=" * sepLen)
    print(">>> Model Parameters Info <<<")
    print(model)
    print("-" * sepLen)


#---------------------
# %% [4] Checkpoint |
#---------------------
def checkpointStruct():
    # Note save model.state_dict() and optimizer.state_dict()
    checkpoint = {'model': [], 'optimizer': [], 'best_avg_batch_train_loss': [],
                  'best_avg_batch_valid_loss': [],'EpochSaved': [],
                  'writer_step': []}
    return checkpoint

class SaveCheckPoint:
    def __init__(self, savePath="my_checkpoint.pt"):#pth.tar"):
        self.sepLen = 50
        self.savePath = savePath
        self.fileName = savePath.split(sep='\\')[-1]
        self.checkpoint = []

    def save_checkpoint(self, checkpoint):
        print('*' * self.sepLen)
        print(f"Save Checkpoint in {self.fileName}")
        print('*' * self.sepLen)
        torch.save(checkpoint, self.savePath)

    def load_checkpoint(self, model, optimizer, optSel):
        checkSel = os.path.exists(self.savePath)
        if checkSel == True:
            checkpoint = torch.load(self.savePath, weights_only=True)
            model.load_state_dict(checkpoint['model'])
            if optSel == 1:
                optimizer.load_state_dict(checkpoint['optimizer'])
            best_avg_loss = checkpoint['best_avg_batch_train_loss']
            Epoch = checkpoint['EpochSaved']
            print('*' * self.sepLen)
            print(f"Load Checkpoint from {self.fileName}")
            print(f"Last Saved Loss from Epoch {Epoch} -> {best_avg_loss}")
            print('*' * self.sepLen)
            self.checkpoint = checkpoint
        else:
            print('*' * self.sepLen)
            print(f"Load Checkpoint is no available!")
            print('*' * self.sepLen)
            best_avg_loss = np.inf
            Epoch = -1

        return model, optimizer, best_avg_loss, Epoch

#----------------------------------------
# %% [5] Modified .pt files's keywords |
#----------------------------------------
def rename_model_keys(fileName, old_str, new_str):
    # fileName ='optimal_Image_Caption_FLICKR8K_state_dict.pt'
    # old_str = 'inception'
    # new_str = 'selected_model'
    new_model  = OrderedDict()
    checkpoint = torch.load(fileName, weights_only=True)
    for old_key, value in checkpoint['model'].items():
        old_key = old_key.replace(old_str, new_str)
        new_model[old_key] = value
    checkpoint['model']= new_model
    torch.save(checkpoint, fileName)