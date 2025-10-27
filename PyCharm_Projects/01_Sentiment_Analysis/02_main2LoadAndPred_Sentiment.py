#----------------------------
# %% [1] Standard Libraries |
#----------------------------
import sys
import torch
import transformers
#-------------------------------
# %% [2] Customized Libraries  |
#-------------------------------
sys.path.append('01_LocalLib_')
#from _Utils_Common_ import setInitSeed
from _Utils_Prediction_ import predict_sentiment_with_text
from Customized_Transformer import Customized_Transformer

#------------------------------
# %% [3] Some basic settings |
#------------------------------
sentimentDict ={0: 'Negative', 1:'Positive'}
output_dim = len(sentimentDict)
# Get Accelerated Device from current host machine
device = "cuda" if torch.cuda.is_available() else "cpu"

#--------------------------------------
# %% [4] Load Pre-trained transformer |
#--------------------------------------
transformerType="distilbert/distilbert-base-uncased"
# Load Pretrained Transformer from HuggingFace
transformer_PreTrained = transformers.AutoModel.from_pretrained(transformerType,
                                                                attn_implementation="eager")
tokenizer = transformers.AutoTokenizer.from_pretrained(transformerType)
pad_index = tokenizer.pad_token_id

#--------------------------------------
# %% [5] Setup Customized Transformer |
#--------------------------------------
freeze = False  # don't use pre-trained parameters, allow update parameters
model = Customized_Transformer(transformer_PreTrained, output_dim, freeze)
model = model.to(device)

#-----------------------------------------
# %% [5] Load sentiment prediction model |
#-----------------------------------------
model_states = torch.load('optimal_sentiment_model_state_dict.pt')
model.load_state_dict(model_states)

#-----------------------
# %% [6] Test samples |
#-----------------------
testSentenceDict = dict()
testSentenceDict[0] = "Can you be more positive?"
testSentenceDict[1] = "This method might work. However, I still do not like the idea."
testSentenceDict[2] = "We can do better, only if we have more time... sigh.."
testSentenceDict[3] = "Are you sure it is correct? Even though it is not possible, but I believe you."
testSentenceDict[4] = "I think it might be a good idea to do!"

#-------------------------------
# %% [7] Sentiment Predictions |
#-------------------------------
for iKey, text_sentence in testSentenceDict.items():
    print(f"Sentiment Test # {str(iKey).rjust(2,'0')}")
    pred_class, pred_prob = predict_sentiment_with_text(text_sentence, model,
                                                        tokenizer, pad_index, sentimentDict, device)