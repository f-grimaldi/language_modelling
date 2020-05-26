import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from functools import reduce
from torchvision import transforms
from torch import optim, nn

    # -*- coding: utf-8 -*-
class LOTRDataset(Dataset):

    def __init__(self, text, emb, transform=None):

        # Extract the sonnets (divided by empty lines and roman numerals)
        sentences = re.split('[.]', text)
        sentences = [i for i in sentences if len(i.split()) > 16]
        chapter_list = sentences
        ### Char to number
        char_to_number = {key: value for key, value in zip(emb.index, emb.values)}

        ### Store data
        self.corpus = text
        self.chapter_list = chapter_list
        self.transform = transform
        self.emb = emb
        self.char_to_number = char_to_number
        #self.number_to_char = number_to_char

    def __len__(self):
        return len(self.chapter_list)

    def __getitem__(self, idx):
        # Get sonnet text
        text = self.chapter_list[idx]
        """
        if len(text.split()) < 9:
            print(self.chapter_list[idx])
            print(text)
        """
        # Encode with numbers
        encoded = encode_text(self.char_to_number, text, self.emb)
        # Create sample
        sample = {'text': text, 'encoded': encoded}
        # Transform (if defined)
        if self.transform:
            sample = self.transform(sample)
        return sample


def encode_text(char_to_number, text, emb):
    encoded = [char_to_number[c] for c in re.findall(r"[\w']+|[.,!?;]", text) if c in emb.index]
    return encoded


def decode_text(emb, encoded):
    text = [emb.index[(emb == c).all(axis=1)][0] for c in encoded]
    #text = [number_to_char[c] for c in encoded]
    #text = reduce(lambda s1, s2: s1 + s2, text)
    return text


class RandomCrop():

    def __init__(self, crop_len):
        self.crop_len = crop_len

    def __call__(self, sample):
        text = sample['text']
        encoded = sample['encoded']
        # Randomly choose an index
        tot_words = len(text.split())
        #start_words = np.random.randint(0, tot_words - self.crop_len)
        start_words = 0
        end_words = start_words + self.crop_len
        new_text = ' '.join(text.split()[start_words:end_words])
        #print(len(text.split()))
        if len(new_text.split()) < self.crop_len:
            print(len(new_text.split()))
            print(new_text.split())
        return {**sample,
                'text': new_text,
                'encoded': encoded[start_words: end_words]}



class OneHotEncoder():

    def __init__(self, emb=None):
        self.emb = emb
        #self.alphabet_len = alphabet_len

    def __call__(self, sample):
        # Load encoded text with numbers
        encoded = np.array(sample['encoded'])
        # Create one hot matrix
        #encoded_onehot = create_one_hot_matrix(encoded, self.alphabet_len)
        encoded_onehot = encoded
        return {**sample,
                'encoded_onehot': encoded_onehot}


class ToTensor():

    def __call__(self, sample):
        # Convert one hot encoded text to pytorch tensor
        encoded_onehot = torch.tensor(sample['encoded_onehot'])
        return {'encoded_onehot': encoded_onehot}


def embedding_matrix(path, text, normalize = True):

    emb = pd.read_csv(path, sep = ' ', quotechar=None, quoting=3, header=None)
    emb.index = emb.iloc[:, 0]
    emb.drop(columns=emb.columns[0], inplace=True)
    corpus = set(word for word in text.split())
    word_in_corpus = [i for i in emb.index if i in corpus]
    emb = emb.loc[word_in_corpus, :]
    emb = pd.DataFrame(np.round(emb.values, 4), index=emb.index)

    if normalize:
        emb = emb.apply(lambda x: x/np.linalg.norm(x), axis=1)

    return emb

def clean_text(path):

    with open(path, 'r') as file:
        text = file.read()

    text = text.lower()
    text = text.replace('#', '')
    text = text.replace('*', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('`', "'")
    text = text.replace(')', '')
    text = text.replace('–', ' ')
    text = text.replace('-', ' ')
    text = text.replace('—', ' ')
    text = text.replace('»', '"')
    text = text.replace('«', '"')
    text = text.replace('_', ' ')
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    text = text.replace('ó', 'o')
    text = text.replace('{', '')
    text = text.replace('}', '')
    text = text.replace('µ', ' ')
    text = text.replace('¤', '')
    text = text.replace('¢', '')
    text = text.replace('¢', '')
    text = text.replace('®', '')
    text = text.replace('¥', '')
    text = text.replace('<br>', '')
    text = text.replace('<h4>', '')
    text = text.replace('</h4>', '')
    text = text.replace('/', '')
    text = text.replace('&', 'e')
    text = text.replace('=', 'o')
    text = text.replace('‚', ',')

    return text
