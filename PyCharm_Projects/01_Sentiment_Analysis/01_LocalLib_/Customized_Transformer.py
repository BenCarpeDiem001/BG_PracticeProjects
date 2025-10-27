from torch import nn, tanh

# Local utility functions
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Customized Transformer for Sentiment Prediction
class Customized_Transformer(nn.Module):
    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer  # pre-trained transformer from hugginface
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, ids, att_masks):
        # ids = [batch_size, seq_len]  # seq_len is fixed to be the same size for each sample
        #output = self.transformer(ids, output_attentions=True)
        #output = self.transformer(ids, token_type_ids=None, attention_mask=att_masks, output_attentions=True)
        # distill bert don't have token_type_ids
        output = self.transformer(ids, attention_mask=att_masks, output_attentions=True)
        hidden = output.last_hidden_state
        # hidden = [batch_size, seq_len, hidden_dim]
        attention = output.attentions[-1]
        # attention = [batch_size, n_heads, seq_len, seq-len]
        # Selecting only the second dimension's first row
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(tanh(cls_hidden))
        # prediction = [batch_size, output_dim]
        return prediction

