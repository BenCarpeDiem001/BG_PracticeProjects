#--------------------------------
# %% [1] Import Standard Libraries |
#--------------------------------
import torch
from torch import nn
import torchvision.models as models

''' 
Note: When selecting CNN architecture, it would be a good idea to select a pre-trained model
      that was trained on a lot of image data.
      When selecting RNN architecture, it might be okay to use LSTM unit.
'''
#---------------------
# %% [2] Encoder CNN |
#---------------------
class EncoderCNN(nn.Module):
    def __init__(self, embedded_size, train_model_flag=False):
        super(EncoderCNN, self).__init__()
        self.train_model_flag = train_model_flag
        # (Note logits -> logits without softmax, and axu_logits -> logits with softmax)
        self.selected_model = models.resnet50(weights="IMAGENET1K_V2")

        self.selected_model.fc = nn.Linear(in_features=self.selected_model.fc.in_features,
                                      out_features=embedded_size)
        self.batchnorm = nn.BatchNorm1d(embedded_size, momentum=0.01)
        self.leaky_relu = nn.LeakyReLU()
        #self.dropout = nn.Dropout(0.1)

        self.debug_flag = 1
        # setting to ensure what can be backpropagated
        for name, param in self.selected_model.named_parameters():
            if name == "fc.weight" or name == "fc.bias":
                param.requires_grad = True
            else:
                param.requires_grad = self.train_model_flag

    def forward(self, x):
        # x is an image [
        fea_from_encoder = self.selected_model(x)

        if self.debug_flag ==1:
            print(f"--> [1.0] EncoderCNN: Input Image Dimension -> {x.shape}")
            print(f"--> [1.1] EncoderCNN: Output Feature Dimension (logits) -> {fea_from_encoder.shape}")
            self.debug_flag = 0
        # return logits only
        fea = self.batchnorm(self.leaky_relu(fea_from_encoder))
        #fea = self.dropout(fea)

        return fea

#----------------------
# %% [3] Decoder RNN |
#----------------------
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedded_size, hidden_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedded_size)
        self.lstm = nn.LSTM(input_size=embedded_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.leaky_relu = nn.LeakyReLU()
        #self.dropout = nn.Dropout(0.1) # was 0.5
        self.debug_flag =  1

    def forward(self, fea_from_encoder, captions):
        #caption_embeddings = self.dropout(self.embedding(captions))
        caption_embeddings = self.embedding(captions)
        # feature for the image will be pass to lstm first, so it is concatenated as such
        embeddings = torch.cat((fea_from_encoder.unsqueeze(1), caption_embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)

        outputs = self.linear(hiddens)
        #outputs = self.leaky_relu(outputs)

        if self.debug_flag ==1:
            print(f"--> [2.0] DecoderRNN: Input Feature -> {fea_from_encoder.shape}")
            print(f"--> [2.1] DecoderRNN: captions -> {captions.shape}")
            print(f"--> [2.2] DecoderRNN: caption_embedding -> {caption_embeddings.shape}")
            print(f"--> [2.3] DecoderRNN: lstm input embedding -> {embeddings.shape}")
            print(f"--> [2.4] DecoderRNN: hidden from lstm -> {hiddens.shape}")
            print(f"--> [2.5] DecoderRNN: outputs -> {outputs.shape}")
            self.debug_flag = 0
        return outputs

#-------------------------
# %% Encoder to Decoder |
#-------------------------
class EncoderCNN2DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedded_size, hidden_size, num_layers):
        super(EncoderCNN2DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedded_size = int(embedded_size)
        self.hidden_size = hidden_size
        self.num_layers = int(num_layers)
        self.encoderCNN = EncoderCNN(embedded_size)
        self.decoderRNN = DecoderRNN(vocab_size, embedded_size, hidden_size, num_layers)
        self.debug_flag = 1

    def forward(self, x, captions):
        # Note: x is an image with a typical dimension of [width, height, channel=3]
        fea_from_encoder = self.encoderCNN(x)
        outputs = self.decoderRNN(fea_from_encoder, captions)
        if self.debug_flag == 1:
            print(f"--> [3.0] EncoderCNN_to_DecoderRNN: Input Image Dimension -> {x.shape}")
            print(f"--> [3.1] EncoderCNN_to_DecoderRNN: feature -> {fea_from_encoder.shape}")
            print(f"--> [3.2] EncoderCNN_to_DecoderRNN: outputs -> {outputs.shape}")
            self.debug_flag = 0
        return outputs

    def caption_image(self, x, vocabulary, max_length=50):
        if self.debug_flag == 1:
            print(f"--> [4.0] EncoderCNN_to_DecoderRNN: caption_image-> vocabulary "
                  f"--> [4.1] {len(vocabulary.itos)}")
        result_caption = []
        self.encoderCNN.eval()
        self.decoderRNN.eval()

        with torch.no_grad():
            if self.debug_flag == 1:
                print(f"--> [4.2] EncoderCNN_to_DecoderRNN: caption_image-> decoderRNN "
                      f"--> [4.3] first input (x) -> {x.shape}")

            x = self.encoderCNN(x).unsqueeze(0)

            h0 = torch.zeros(self.num_layers, 1, self.embedded_size).to(x.device)
            c0 = torch.zeros(self.num_layers, 1, self.embedded_size).to(x.device)
            states = (h0,c0)

            if self.debug_flag == 1:
                print(f"--> [4.4] EncoderCNN_to_DecoderRNN: caption_image -> decoderRNN "
                      f"--> [4.5] first output (x) -> {x.shape}")

            for i in range(max_length):
                # x is go to the hidden state in lstm not the input states ask "states"
                # x is a vector with dim [1, 256]
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens)#.unsqueeze(0))
                predicted = output.argmax(2)
                #print(f"{i} word ---> {vocabulary.itos[predicted[0].item()]}")
                if self.debug_flag == 1:
                    print(f"--> [4.6] EncoderCNN_to_DecoderRNN: caption_image -> deocderRNN "
                          f"--> [4.7] hiddens -> {hiddens.shape}")
                    print(f"--> [4.8] EncoderCNN_to_DecoderRNN: caption_image -> decoderRNN "
                          f"--> [4.9] predicted -> {predicted.shape}")

                result_caption.append(predicted.item())

                x = self.decoderRNN.embedding(predicted)
                if self.debug_flag == 1:
                    print(f"--> [4.10] EncoderCNN_to_DecoderRNN: caption_image -> decoderRNN "
                          f"--> [4.11] {i} output (x) dimension -> {x.shape}")
                    self.debug_flag = 0

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
        return [vocabulary.itos[idx] for idx in result_caption]

