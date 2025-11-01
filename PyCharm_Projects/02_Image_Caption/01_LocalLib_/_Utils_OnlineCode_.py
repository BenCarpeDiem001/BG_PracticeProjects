#-----------------------------
# %% [1] Online References  |
#-----------------------------
# [1] https://stackoverflow.com/questions/68526138/torchvision-datasets-flickr8k-not-importing-datapoints-correctly
# [2] https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/image_captioning/get_loader.py

#------------------------------
# %% [2] Standard Libraries  |
#------------------------------
from sympy.physics.units import frequency
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision.datasets import VisionDataset
import os
import csv
from PIL import Image
import spacy  # for tokenizer
spacy_eng = spacy.load("en_core_web_sm")

#-------------------------------------------
# %% [3] Build Vocabularies from dataset  |
#-------------------------------------------
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    # function is tied to the class not the object
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        # the first 4 is reserved for the self.itos definitions, more to add
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                # any word achieve threshold first then recorded
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

#------------------------------------------------
# %% [4] Load Kaggle Image and Caption dataset |
#------------------------------------------------
class KaggleFlickr8k(VisionDataset):
    def __init__(self, root, ann_file, transform=None, target_transform=None, loader=None,
                 freq_threshold = 5):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = ann_file
        self.freq_threshold = freq_threshold
        self.loader = loader if loader is not None else self._default_loader
        self._load_annotations()
        self._build_vocabulary()

    def _default_loader(self, path):
        #print(path)
        with open(path, "rb") as f:
            img = Image.open(f)
            # Use to avoid closing to early
            img.load()
            #print(img.getdata())
        return img.convert("RGB")

    def _load_annotations(self):
        self.annotations = []
        with open(self.ann_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')#'\t') # Adjust delimiter if needed
            next(reader) # Skip header if present
            for row in reader:
                #print(row)
                # Assuming format like: image_name.jpg, caption_text
                image_filename = row[0].split('#')[0] # Extract image filename
                caption = row[1].replace('.','').strip() # remove period and trail spaces
                self.annotations.append({'image': image_filename, 'caption': caption})

    def _build_vocabulary(self):
        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(self.freq_threshold)
        caption_list = [item['caption'] for item in self.annotations]
        self.vocab.build_vocabulary(caption_list)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        #print(index)
        item = self.annotations[index]
        img_path = os.path.join(self.root, item['image'])
        image = self.loader(img_path)
        caption = item['caption']

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            caption = self.target_transform(caption)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return (image, torch.tensor(numericalized_caption))

#-------------------------------------------
# %% [5] Grouping function for dataloader |
#-------------------------------------------
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return imgs,targets

#------------------------------------------------
# %% [6] convert dataset into dataloader format|
#------------------------------------------------
def get_loader(
    root,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = KaggleFlickr8k(root, annotation_file, transform=transform)

    # default value to be padded so all inputs has the same length
    pad_idx = dataset.vocab.stoi["<PAD>"]
    # 80% training and 20% validation
    train_dataset, val_dataset = random_split(
        dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return train_loader, train_dataset, val_loader, val_dataset