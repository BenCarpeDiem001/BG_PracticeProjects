'''
Note: It seems to me, at least for small image-caption paired dataset,
it might easier to learn if we use more general description of the image.
With over 100 epochs, the model somewhat improved but nowhere near perfect.
I have reason to believe that the image-caption dataset size and encoder CNN plays important
roles in a more accurate model. Moreover, for this model, I did not use any valid dataset
and no hyperparameter tuning.
'''
#----------------------------------
# %% [01] Load Standard Libraries |
#----------------------------------
import torch
import torchvision.transforms as transforms
import random
import matplotlib
from matplotlib import pyplot as plt
# Enable Plotting Graphic
matplotlib.use('TkAgg')
import os
from glob import glob
import sys
sys.path.append('01_LocalLib_')

#------------------------------------
# %% [02] Load customized Libraries |
#------------------------------------
from _Utils_Common_ import SaveCheckPoint
from _Utils_OnlineCode_ import get_loader
from _Utils_Prediction_ import predict_image_caption, predict_single_custom_image_caption
from Custom_ImageCaption import EncoderCNN2DecoderRNN

#----------------------------------
# %% [03] Find accelerated device |
#----------------------------------
torch.backends.cudnn.benchmark = True
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

#------------------------------------------
# %% [04] Selecting dataset to work with |
#------------------------------------------
database_name='FLICKR8K'
# CNN hyperparameters
embedded_size = 256   # final output vector from the CNN
# RNN hyperparameters
# Note: vocab_size -> Define vocab_size base on info from train_dataset
hidden_size = 256     # hidden_layer of the RNN
num_layers = 1        # number stacking layers of LSTM for a RNN
# General Settings
learning_rate = 3e-4 #2*3e-5 # 1) 3e-4/2   2) 3e-5 3) 3e-5/2 4) 3e-3
total_epochs = 100
batch_size = 32

load_saved_model_flag = True
# This optSel option is intended for a situation when
# I stop the training and pick up the training
# again with the same or different optimizer settings
optSel = 0 # indicates don't use save optimizer

imgMean=(0.5, 0.5, 0.5)
imgStd =(0.5, 0.5, 0.5)
normalize = transforms.Normalize(mean=imgMean,std=imgStd)

my_transforms = transforms.Compose([
    transforms.Resize((356,356)),
    transforms.RandomCrop((224, 224)),  #was 299
    transforms.ToTensor(),
    normalize])

#-------------------------------------
# %% [05] Basic info about the data |
#----------------------------------------------------------------------
# Note: FLICKR8K contains 8000 images with captions.
# However, the same image can also be matched to a different caption.
#----------------------------------------------------------------------
root_folder = "FLICKR8K_dataset/Images/"
csv_file = "FLICKR8K_dataset/captions.txt"

(train_dataloader, train_dataset,
 val_dataloader, val_dataset) = get_loader(root=root_folder, annotation_file=csv_file,
                                             transform=my_transforms,batch_size=batch_size,
                                             shuffle=True, num_workers=0) # more workers under this environment cause problem

# Define vocab_size base info from train_dataset
vocab_size = len(train_dataset.dataset.vocab)

#-----------------------
# %% [06] Build model |
#-----------------------
model_image_caption = EncoderCNN2DecoderRNN(vocab_size, embedded_size,
                                               hidden_size, num_layers).to(device)

#-----------------------
# %% [07] Load model  |
#-----------------------
# No updated optimizers
opt_image_caption = '' #optim.Adam(model_disc.parameters(),lr=learning_rate_disc)
if load_saved_model_flag == True:
    saved_model = SaveCheckPoint(f"optimal_Image_Caption_{database_name}_state_dict.pt")
    model_image_caption, tmp, best_avg_loss, Epoch = saved_model.load_checkpoint(model_image_caption,
                                                                          opt_image_caption, optSel)
    print(f"Loaded Model --> Best_Avg_Loss:{best_avg_loss:0.3f}")

#---------------------------------
# %% [08] Image Caption Testing |
#---------------------------------
# Get a batch sample from train_dataloader
sample_batch_data = next(iter(train_dataloader))
img_batch = sample_batch_data[0]
caption_truth = sample_batch_data[1]
vocabulary = train_dataset.dataset.vocab

img_captions, correct_captions, img_list = predict_image_caption(img_batch, caption_truth,
                                                                 vocabulary, device,
                                                                 imgMean, imgStd,
                                                                 model_image_caption)

# plotting images with captions from Kaggle dataset
numbers = list(range(len(img_batch)))
chosen_numbers = random.sample(numbers, k=2)

fig, axList = plt.subplots(2,1)
for i, i_ax in enumerate(axList):
    j = chosen_numbers[i]
    i_ax.imshow(img_list[j])
    pStr = ' '.join(img_captions[j])
    cStr = ' '.join(correct_captions[j])
    i_ax.set_title(f"Predicted: {pStr} \n Correct: {cStr}")


# For a single custom image (out-sample testing/random image from internet)
customHomePath = os.path.join(os.getcwd(),'CustomImage_dataset')
ind_img = 4
file_list = os.listdir(customHomePath)
fileName = file_list[ind_img]
img_path = os.path.join(customHomePath, fileName)

custom_img_caption, custom_img = predict_single_custom_image_caption(img_path,
                                                                     vocabulary, device,my_transforms
                                                                     ,imgMean, imgStd, model_image_caption)
custom_pStr = ' '.join(custom_img_caption)
plt.figure()
plt.imshow(custom_img)
plt.title(f"Predicted: {custom_pStr}")
