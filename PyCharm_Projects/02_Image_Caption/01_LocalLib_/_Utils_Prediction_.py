#------------------------------
# %% [1] Standard Libraries  |
#------------------------------
import torch
from PIL import Image
import numpy as np

#------------------------------
# %% [2] Custom Prediction Libraries  |
#------------------------------
def predict_image_caption(imgs,caption_truth, vocabulary,
                          device, imgMean, imgStd, model):
    img_captions = list()
    model.eval()
    # Get predicted captions
    with torch.no_grad():
        for i in range(len(imgs)):
            img = imgs[i].to(device).unsqueeze(0)
            cur_img_captions = model.caption_image(img, vocabulary)
            img_captions.append(cur_img_captions)

    # Format ground-truth captions
    correct_captions = list()
    for jSentence in caption_truth:
        cur_sentence = list()
        for iToken in jSentence:
            cur_word = vocabulary.itos[iToken.item()]
            if cur_word == '<PAD>':
                break
            else:
                cur_sentence.append(cur_word)
        correct_captions.append(cur_sentence)

    # Match and prent pred and ground-truth caption pairs
    for i, (pred, correct) in enumerate(zip(img_captions, correct_captions)):
        pred_sen = " ".join(pred)
        correct_sen = " ".join(correct)
        print(f"#{i} predicted: {pred_sen}")
        print(f"#{i} correct  : {correct_sen}")

    # Plot images with pred and correct captions for evaluation
    img_list = []
    for i in range(len(imgs)):
        cur_img = imgs[i]
        cur_img = np.permute_dims(cur_img, [1, 2, 0])
        cur_img_cpu = cur_img.cpu().detach().numpy()
        for j in range(3):
            cur_img_cpu[:, :, j] = cur_img_cpu[:, :, j] * imgStd[j] + imgMean[j]
        img_list.append(cur_img_cpu)

    return img_captions, correct_captions, img_list


def predict_single_custom_image_caption(img_path, vocabulary, device,
                                        my_transforms, imgMean, imgStd, model):
    img_captions = list()
    pil_img = Image.open(img_path)
    tensor_img = my_transforms(pil_img)
    model.eval()
    with torch.no_grad():
        img = tensor_img.to(device).unsqueeze(0)
        cur_img_captions = model.caption_image(img, vocabulary)
        #img_captions.append(cur_img_captions)  # <-- leave it like that

    cur_img = np.permute_dims(tensor_img, [1, 2, 0])
    cur_img_cpu = cur_img.cpu().detach().numpy()
    for j in range(3):
        cur_img_cpu[:, :, j] = cur_img_cpu[:, :, j] * imgStd[j] + imgMean[j]

    return cur_img_captions, cur_img_cpu