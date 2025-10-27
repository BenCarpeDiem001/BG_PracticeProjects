# %% [00] References
#-------------------------------------------------------------------------------------------------
'''
[1]https://www.youtube.com/watch?v=KRgq4VnCr7I&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=34
[2]https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/4%20-%20Transformers.ipynb
[3]https://www.kaggle.com/code/vimalpillai/training-bert-custom-data-for-classification
[4]https://ai.stackexchange.com/questions/28833/isnt-attention-mask-for-bert-model-useless
[5]https://huggingface.co/docs/transformers/en/model_doc/bert
'''
#-----------------------------
# %% [01] Standard Libraries  |
#-----------------------------
import sys
import datasets
import torch.cuda
from torch import nn, optim
import transformers

#--------------------------------
# %% [02] Customized Libraries  |
#--------------------------------
sys.path.append('01_LocalLib_')
from _Utils_Common_ import setInitSeed, mapDataSample_to_numerical
from _Utils_Dataloader_ import get_data_loader
from Customized_Transformer import Customized_Transformer, count_trainable_parameters
from _Utils_ModelTraining_ import train_model_under_n_epochs, evaluate

#-----------------------------------
# %% [03] Seed for Reproducibility |
#-----------------------------------
setInitSeed(8901)

#-----------------------------------------------------------
# %% [04] Get Accelerated Device from current host machine |
#-----------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

#-----------------------------------
# %% [05] Hyperparameter settings  |
#-----------------------------------
lr = 1e-5
batch_size = 10
nEpochs = 1

#-----------------------------------
# %% [06] Load Train and Test Data |
#-----------------------------------
trainData, testData = datasets.load_dataset(path="imdb",
                                            split=["train", "test"])

#---------------------------------------------------------------------------
# %% [07] Show train and test dataset size and Example for a single sample |
#---------------------------------------------------------------------------
trainSize, testSize = len(trainData), len(testData)
dataKeys = trainData[0].keys()
print(f"---> This trainData consists of data for modeling and data for validation (aka validData")
print(f"---> trainSize: %d  TestSize: %d data-Keys: %s" %(trainSize, testSize, dataKeys))
print(f"---> One Sample of trainData\n {trainData[0]} ")
print(f"     trainData[0]'s keys: {trainData[0].keys()}")

#----------------------------------------
# %% [08] Load Pre-trained Transformer  |
#----------------------------------------
# (Website for more types: https://huggingface.co/models?sort=trending&search=bert)
transformerTypeOptions = dict()
transformerTypeOptions['FullModel'] = "google-bert/bert-base-uncased"
transformerTypeOptions['DistilledModel'] = "distilbert/distilbert-base-uncased"
transformerType= transformerTypeOptions['DistilledModel']

#----------------------------------------------------------------
# %% [09] Load and Show selected Transformer's tokenized values |
#----------------------------------------------------------------
testString = "I am learning to code!"
tokenizer = transformers.AutoTokenizer.from_pretrained(transformerType)
tokens = tokenizer.tokenize(testString)
tokens_encodings = tokenizer.encode(testString)
tokens_translated = tokenizer.convert_ids_to_tokens(tokens_encodings)
outFromTokenizer = tokenizer(testString)
print(f"---> testString: \"{testString}\" \n"
      f"     tokens: {tokens} \n"
      f"     tokens' encoding: {tokens_encodings}\n"
      f"     tokens' encoding to string: {tokens_translated}\n"
      f"     tokenized token: {outFromTokenizer}\n"
      f"     tokenized token's keys: {outFromTokenizer.keys()}\n"
      f"     token_type_ids: data type which has 0 for from-context and 1 for from-question")
print(f"---> Additional Token Info: \n"
      f"     tokenizer.pad_token -> {tokenizer.pad_token}\n"
      f"     tokenizer.pad_token_id -> {tokenizer.pad_token_id}\n"
      f"     tokenizer.vocab[tokenizer.pad_token] -> {tokenizer.vocab[tokenizer.pad_token]}")
print(f"--> Note: \"CLS\" stands for Classification task\n"
      f"          \"SEP\" stands for Separator Token")

#----------------------------
# %% [10] Text to numerical |
#----------------------------
# Add "ids" to the dataset, convert each "text" sample to numerical
# This trainData contains both data for modeling and validData
trainData = trainData.map(mapDataSample_to_numerical, fn_kwargs={"tokenizer": tokenizer})
testData  = testData.map(mapDataSample_to_numerical,   fn_kwargs={"tokenizer": tokenizer})
print(f"---> trainData[0]'s NEW Keys: {trainData[0].keys()}")

# Get validation dataset from trainData
test_size_portion = 0.25 # 25% of all trainData
train_valid_data = trainData.train_test_split(test_size=test_size_portion)
trainData = train_valid_data["train"]
validData = train_valid_data["test"]

#-----------------------------------------------------------------------------
# %% [11] Reformating all Dataset ( "text" column is not formatted to torch) |
#-----------------------------------------------------------------------------
formatCols= ["ids", "label"]  # full feature cols: ["text", "ids", "label"]
trainData = trainData.with_format(type="torch", columns=formatCols)
validData = validData.with_format(type="torch", columns=formatCols)
testData  = testData.with_format(type="torch", columns=formatCols)

#--------------------------------------------
# %% [12] Put all datasets into dataloaders |
#--------------------------------------------
pad_index = tokenizer.pad_token_id
trainDataLoader = get_data_loader(dataset=trainData, batch_size=batch_size, pad_index=pad_index, shuffle=True)
validDataLoader = get_data_loader(dataset=validData, batch_size=batch_size, pad_index=pad_index)
testDataLoader  = get_data_loader(dataset=testData, batch_size=batch_size, pad_index=pad_index)

#---------------------------------------------
# %% [13] Show each batch in each dataloader |
#---------------------------------------------
# Get one batch
one_batch_train = next(iter(trainDataLoader))
one_batch_valid = next(iter(validDataLoader))
one_batch_test  = next(iter(testDataLoader))
# Get batch size
one_batch_size_train, seq_len_train = one_batch_train['ids'].size()
one_batch_size_valid, seq_len_valid = one_batch_valid['ids'].size()
one_batch_size_test, seq_len_test = one_batch_test['ids'].size()

print(f"---> trainDataLoader -> Total # of Batch: {len(trainDataLoader)}"
      f"     Each Batch [batch_size, seq_len]: [{one_batch_size_train},{seq_len_train}]")
print(f"---> validDataLoader -> Total # of Batch: {len(validDataLoader)}"
      f"     Each Batch [batch_size, seq_len]: [{one_batch_size_valid},{seq_len_valid}]")
print(f"---> testDataLoader -> Total # of Batch: {len(testDataLoader)}"
      f"     Each Batch [batch_size, seq_len]: [{one_batch_size_test},{seq_len_test}]")
print(f"---> Additional Info -> Each batch has keys with torch type: {one_batch_train.keys()}")
print(f"--> Note: \"According to Google search, BERT typically pads input sequences to a maximum length of 512 tokens \n"
      f"           during tokenization, even when performing predictions (inference). \n"
      f"           This is a pre-processing step that is automatically handled by the BERT tokenizer\"")

#-------------------------------------------------------
# %% [14] Load Pretrained Transformer from HuggingFace |
#-------------------------------------------------------
transformer_PreTrained = transformers.AutoModel.from_pretrained(transformerType, attn_implementation="eager")
print(f"---> The pretrained Transformer has the following hidden_size: {transformer_PreTrained.config.hidden_size}")

#--------------------------------------------
# %% [15] Customized Pretrained Transformer |
#--------------------------------------------
# label is either 0 (negative sentiment) or 1 (positive sentiment)
output_dim = len(trainData["label"].unique())
freeze = False  # don't use pre-trained parameters, allow update parameters
model = Customized_Transformer(transformer_PreTrained, output_dim, freeze)
model = model.to(device)

#------------------------------------------------------
# %% [16] Show total Trainable Parameters in the model |
#------------------------------------------------------
totalTrainableParameters = count_trainable_parameters(model)
print(f"---> The model has {totalTrainableParameters:,} trainable parameters")

#--------------------------
# %% [17] Setup Optimizer |
#--------------------------
optimizer = optim.Adam(model.parameters(), lr=lr)

#-------------------------
# %% [18] Setup criterion |
#-------------------------
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

#----------------------
# %% [19] Train Model |
#----------------------
epochMetrics, best_valid_loss = train_model_under_n_epochs(trainDataLoader, validDataLoader,
                                                           nEpochs, model,criterion, optimizer,device)
#-----------------------------------------
# %% [20] Test with the best saved model |
#-----------------------------------------
# sentimentDict ={0: 'Negative', 1:'Positive'}
model.load_state_dict(torch.load("optimal_sentiment_model_state_dict.pt"))
test_loss, test_acc = evaluate(testDataLoader, model, criterion, device)
print(f"Test Data Results: \n"
      f"test loss: {test_loss:.5f}  test_acc: {test_acc*100:.3f}%")
