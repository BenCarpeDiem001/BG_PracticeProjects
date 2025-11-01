#---------------------
# %% [00] Reference |
#---------------------
#[1] https://www.youtube.com/watch?v=y2BaTt1fxJU&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=22
#[2] https://stackoverflow.com/questions/68526138/torchvision-datasets-flickr8k-not-importing-datapoints-correctly
#[3] https://www.kaggle.com/code/zohaib123/image-caption-generator-using-cnn-and-lstm
#[4] https://medium.com/@zeyneptufekci.etu/cnn-lstm-for-image-captioning-with-progressive-loading-52a740705b2c
#[5] https://medium.com/@codeaigo/resnet-vs-inceptionv3-in-image-captioning-lessons-from-the-flickr8k-dataset-7811a07f907c
#[6] https://www.kaggle.com/code/zohaib123/image-caption-generator-using-cnn-and-lstm
#[7] https://learnopencv.com/image-captioning/
# Note: Inception_v3 is not good enough for encoding image info, I use resnet50.

#-------------------------------------
# %% [01] Import Standard Libraries |
#-------------------------------------
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets, transforms

import matplotlib
# Enable Plot Graphics
matplotlib.use("TkAgg")

import sys
# Mark folder as "Sources Root"
sys.path.append('01_LocalLib_')

import numpy as np
import random

#-------------------------------
# %% [02] Customized Libraries |
#-------------------------------
from _Utils_Common_ import printInfo_DataLoader, printInfo_PackageVersions
from _Utils_Common_ import SaveCheckPoint
from Custom_ImageCaption import EncoderCNN2DecoderRNN
from _Utils_ModelTraining_ import train_model_under_n_epochs
from _Utils_OnlineCode_ import get_loader

SEED = 168
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def process_main():
    # ----------------------------------
    # %% [03] Find accelerated device |
    # ----------------------------------
    torch.backends.cudnn.benchmark = True
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    # --------------------------------------------------
    # %% [04] Libraries Versions and Available Device |
    # --------------------------------------------------
    infoList = list()
    infoList.append(f'torch: {torch.__version__}')
    infoList.append(f'torchvision: {torchvision.__version__}')
    infoList.append(f"Current usable device: {device}")
    infoList.append(f"Available Plotting Graphic options: {matplotlib.backends.backend_registry.list_builtin()}")
    printInfo_PackageVersions(infoList)

    # ----------------------------------------------------
    # %% [05] Define Parameter and data transformation |
    # ----------------------------------------------------
    database_name = 'FLICKR8K'
    # CNN hyperparameters
    embedded_size = 256  # final output vector from the CNN
    # RNN hyperparameters
    hidden_size = 256  # hidden_layer of the RNN
    num_layers = 1  # number stacking layers of LSTM for a RNN
    # General Settings
    learning_rate = 3e-4
    total_epochs = 100
    batch_size = 32

    # Define load options for model weights

    saved_model_available = True  # True
    optSel = 0  # indicates don't use saved optimizer

    # Transformation setting for the incoming image
    imgMean = (0.5, 0.5, 0.5)
    imgStd = (0.5, 0.5, 0.5)
    normalize = transforms.Normalize(mean=imgMean, std=imgStd)

    my_transforms = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((224, 224)),  # was 299
        transforms.ToTensor(),
        normalize])

    #----------------------------
    # %% [06] Setup Dataloader |
    #----------------------------
    root_folder = "FLICKR8K_dataset/Images/"
    csv_file = "FLICKR8K_dataset/captions.txt"

    (train_dataloader, train_dataset,
     val_dataloader,val_dataset) = get_loader(root=root_folder, annotation_file=csv_file,
                                                 transform=my_transforms,batch_size=batch_size,
                                                 shuffle=True, num_workers=10)
    printInfo_DataLoader(train_dataloader, 'Training DataSet')
    printInfo_DataLoader(val_dataloader, 'Validation DataSet')
    # Define Vocabulary size
    vocab_size = len(train_dataset.dataset.vocab)
    # setup for writing to tensorboard
    writer_image_caption = SummaryWriter(f"runs/flickr")

    # ----------------------
    # %% [07] Build model |
    # ----------------------
    model_image_caption = EncoderCNN2DecoderRNN(vocab_size, embedded_size,
                                                   hidden_size, num_layers).to(device)


    opt_image_caption = optim.Adam(model_image_caption.parameters(),lr=learning_rate)

    # ignore loss form <PAD> which is "0"
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.dataset.vocab.stoi["<PAD>"])

    # Load Models
    if saved_model_available == True:
        saved_model = SaveCheckPoint(f"optimal_Image_Caption_{database_name}_state_dict.pt")
        (model_image_caption, tmp, best_avg_loss,
         Epoch) = saved_model.load_checkpoint(model_image_caption,opt_image_caption, optSel)

    train_model_under_n_epochs(train_dataloader,val_dataloader, model_image_caption, criterion,
                               opt_image_caption, total_epochs, writer_image_caption,
                               database_name, saved_model_available, device)
if __name__ == "__main__":
    process_main()
    print("Done!")