import torch
from torch import nn
from torch.utils.data import DataLoader

def convert_id_to_mask(idsList):
    att_masks = []
    for id in idsList:
        # Means id is part of context in the sentence
        if id > 0:
            att_masks.append(1)
        # Means id is not part of context & it is just PADDED value (0)
        else:
            att_masks.append(0)
    return att_masks

def get_collate_fn(pad_index): # return generator
    def collate_fn(batch):
        batch_ids =[item["ids"] for item in batch]
        # Padding "0" to fill til it reaches "max_position_embeddings": 512 for bert-base-uncased
        batch_ids = nn.utils.rnn.pad_sequence(batch_ids,
                                              padding_value=pad_index,
                                              batch_first=True)
        batch_att_masks = [convert_id_to_mask(idsList) for idsList in batch_ids]
        batch_att_masks = torch.tensor(batch_att_masks)
        # batch -> <class 'torch.utils.data.dataloader._SingleProcessDataLoaderIter'>
        # batch_labels is a <class 'list'> with item as tensor([1, 0, 0, 1, 1, 0, 0, 1, 0, 0])
        batch_labels = [item["label"] for item in batch]
        # batch_labels is <class 'torch.Tensor'> and now in tensor format similar to a matrix
        batch_labels = torch.stack(batch_labels)


        batch = {"ids": batch_ids, "att_masks":batch_att_masks ,"label":batch_labels}
        return batch
    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             collate_fn=collate_fn,
                             shuffle=shuffle)
    return data_loader