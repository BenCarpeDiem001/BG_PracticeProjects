#------------------------------------
# %% [1] Import Standard Libraries  |
#------------------------------------
import numpy as np
from glob import glob
import os
import torch
from _Utils_Common_ import checkpointStruct, SaveCheckPoint
sepLen = 50

#------------------------------------------
# %% Local helper function for Resume training |
#------------------------------------------
def find_writer_step():
    writer_step = 0
    epoch_num = 0
    workdir = os.getcwd()
    cur_pt_path = glob(os.path.join(workdir, '*_dict.pt'))
    if len(cur_pt_path) != 0:
        checkpoint = torch.load(cur_pt_path[-1], weights_only=True)
        writer_step = checkpoint['writer_step']
        epoch_num = checkpoint['EpochSaved']
    return writer_step, epoch_num

#-------------------------------------------------
# %% [2] Function to Evaluate Model Performances|
#-------------------------------------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            batch_size = X.shape[0]
            # y[:,-1] -> has the following dimension [32,23]
            # NEED TO WORK ON THE LAST <eos> need to remove
            outputs = model(X, y[:, :-1])

            # outputs has dimension->[batch_size, seq_len, vocab_size]
            # y has dimension -> [batch_size, seq_len]
            outputs_reshape = outputs.reshape(-1, outputs.shape[2])
            y_reshape = y.reshape(-1)
            loss = criterion(outputs_reshape, y_reshape)
            total_loss += loss.item()
    avg_val_loss = total_loss / len(dataloader)
    return avg_val_loss

#------------------------------
# %% [3] Training one epoch  |
#------------------------------
'''
Training (Note: Model need not to be return, 
all weights are updated within)
'''
def train_model(train_dataloader, model_image_caption, criterion,
                opt_image_caption, total_epochs, writer_image_caption,
                database_name, device, writer_step, ith_epoch):
    totalNumOfBatches = len(train_dataloader)

    total_loss = 0.0

    # Enable weight updates, batchnorm, and drop if any
    model_image_caption.train()

    for batch_idx, (X, y) in enumerate(train_dataloader):
        '''
        Note: X -> images (batch_size, channel, img_width, img_height)
              y-> captions (batch_size, seq_len)
        Every batch captions are padded to the same size
        and every batch has different length
        '''
        # place X and y in GPU device
        X, y = X.to(device), y.to(device)

        # Note:  y->[32,24], then y[;<:-1]->[32,23] ignoring last token
        outputs = model_image_caption(X,y[:,:-1])

        # outputs has dimension->[batch_size, seq_len, vocab_size]
        outputs_reshape = outputs.reshape(-1, outputs.shape[2])
        y_reshape = y.reshape(-1)
        '''
            outputs_reshape -> (batch_size*seq_len, vocab_size)
            y_reshape       -> (batch_size*seq_len)
        '''
        loss = criterion(outputs_reshape, y_reshape)

        writer_image_caption.add_scalar(f"{database_name}'s Training Loss", loss.item(),
                                        global_step=writer_step)
        writer_step +=1

        # Loss and weight update
        opt_image_caption.zero_grad()
        loss.backward()
        opt_image_caption.step()
        # add up all batch losses
        total_loss += loss.item()
        # print results with a step of 100 batches
        if batch_idx % 100 == 0 or batch_idx == totalNumOfBatches - 1:
            print(f"Epoch [{ith_epoch + 1}/{total_epochs}] "
                  f"Batch [{batch_idx}/{totalNumOfBatches}] "
                  f"Batch Train_Loss:{loss.item():.4f}"
                  )

    epoch_avg_loss = total_loss / totalNumOfBatches
    return epoch_avg_loss, writer_step

#------------------------------
# %% [4] Main training function  |
#------------------------------
def train_model_under_n_epochs(train_dataloader, val_dataloader, model_image_caption, criterion,
                               opt_image_caption, total_epochs,
                               writer_image_caption, database_name, saved_model_available, device):
    if saved_model_available:
        writer_step, ran_epochs = find_writer_step()
    else:
        writer_step, ran_epochs = 0, 0

    best_avg_batch_val_loss = np.inf
    best_avg_batch_train_loss = np.inf

    for ith_epoch in range(total_epochs):
        cur_epoch = ran_epochs + ith_epoch
        epoch_avg_batch_train_loss, writer_step = train_model(train_dataloader, model_image_caption, criterion,
                                     opt_image_caption,total_epochs,
                                     writer_image_caption, database_name, device, writer_step, cur_epoch)

        epoch_avg_batch_val_loss = evaluate(model_image_caption, val_dataloader, criterion, device)

        if epoch_avg_batch_train_loss < best_avg_batch_train_loss:
            best_avg_batch_train_loss = epoch_avg_batch_train_loss
            # loss and info from training
            checkpt = checkpointStruct()
            saved_model = SaveCheckPoint(f"optimal_Image_Caption_{database_name}_state_dict.pt")
            checkpt['model'] = model_image_caption.state_dict()
            checkpt['optimizer'] = opt_image_caption.state_dict()
            checkpt['best_avg_batch_train_loss'] = epoch_avg_batch_train_loss
            checkpt['best_avg_batch_valid_loss'] = epoch_avg_batch_val_loss
            checkpt['EpochSaved'] = cur_epoch
            checkpt['writer_step'] = writer_step
            saved_model.save_checkpoint(checkpt)
            print(f"Image-Caption save @ Epoch {cur_epoch} with current train_loss "
                  f"{epoch_avg_batch_train_loss: 0.3f}  ---> val_loss "
                  f"{epoch_avg_batch_val_loss: 0.3f} saved!")

        if epoch_avg_batch_val_loss < best_avg_batch_val_loss:
            best_avg_batch_val_loss = epoch_avg_batch_val_loss
            # loss and info from training
            checkpt = checkpointStruct()
            saved_model = SaveCheckPoint(f"optimal_Image_Caption_{database_name}_state_dict_Val.pt")
            checkpt['model'] = model_image_caption.state_dict()
            checkpt['optimizer'] = opt_image_caption.state_dict()
            checkpt['best_avg_batch_train_loss'] = epoch_avg_batch_train_loss
            checkpt['best_avg_batch_valid_loss'] = epoch_avg_batch_val_loss
            checkpt['EpochSaved'] = cur_epoch
            checkpt['writer_step'] = writer_step
            saved_model.save_checkpoint(checkpt)


