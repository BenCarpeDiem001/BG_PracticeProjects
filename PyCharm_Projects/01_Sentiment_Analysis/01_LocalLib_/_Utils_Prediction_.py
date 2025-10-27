import torch
from _Utils_Dataloader_ import convert_id_to_mask

def pad_sequence(token_ids, max_len, pad_index):
    tLen = len(token_ids)
    for i in range(max_len - tLen):
        token_ids.append(pad_index)
    return token_ids

def get_model_input_token_and_masks(token_ids, embedded_size,device, pad_index):
    # padded token_ids and has same size as device
    token_ids = pad_sequence(token_ids, embedded_size, pad_index)
    att_masks = convert_id_to_mask(token_ids)
    token_ids = torch.tensor(token_ids).to(device)
    att_masks = torch.tensor(att_masks).to(device)

    return token_ids, att_masks

def predict_sentiment_with_text(text_sentence, model, tokenizer, pad_index, sentimentDict, device):
    model.eval()  # Set the model to evaluation mode if you're not training
    embedded_size = tokenizer.model_max_length
    # tokenizer yields that following dict_keys(['input_ids', 'attention_mask'])
    tokenizeInput = tokenizer(text_sentence)
    token_ids = tokenizeInput['input_ids']
    # both token_ids and att_mask are a vector with size 512
    token_ids, att_masks = get_model_input_token_and_masks(token_ids, embedded_size, device,pad_index)
    pred_logit = model(token_ids, att_masks)
    # by default it has the batch dimension, need to squeeze it out
    pred_logit = pred_logit.squeeze(dim=0)
    pred_prob_logit = torch.softmax(pred_logit, dim=-1)
    pred_class = pred_prob_logit.argmax(dim=-1).item()
    pred_prob = pred_prob_logit[pred_class].item()
    print(f"---> Input Sentence: {text_sentence}")
    print(f"---> Sentiment is \" {sentimentDict[pred_class]} \" with probability of \"{pred_prob: .3f}\"")
    return pred_class, pred_prob


