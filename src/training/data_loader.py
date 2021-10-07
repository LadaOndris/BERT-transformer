from pprint import pprint

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def to_map_style_dataset(iter_data):
#     r"""Convert iterable-style dataset to map-style dataset.
#
#     args:
#         iter_data: An iterator type object. Examples include Iterable datasets, string list, text io, generators etc.
#     """
#
#     # Inner class to convert iterable-style to map-style dataset
#     class _MapStyleDataset(torch.utils.data.Dataset):
#
#         def __init__(self, iter_data):
#             self._data = list(iter_data)
#
#         def __len__(self):
#             return len(self._data)
#
#         def __getitem__(self, idx):
#             return self._data[idx]
#
#     return _MapStyleDataset(iter_data)


class DataLoaderPreprocessor:

    def __init__(self, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle

        train_iter, test_iter = AG_NEWS()
        train_dataset = to_map_style_dataset(train_iter)
        self.test_dataset = to_map_style_dataset(test_iter)
        num_train = int(len(train_dataset) * 0.95)
        self.split_train_, self.split_valid_ = \
            random_split(train_dataset, [num_train, len(train_dataset) - num_train])

        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(self.yield_tokens(train_dataset), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

        self.num_class = len(set([label for (label, text) in train_dataset]))
        self.vocab_size = len(self.vocab)
        self.emsize = 64

    def text_pipeline(self, x):
        return self.vocab(self.tokenizer(x))

    def label_pipeline(self, x):
        return int(x) - 1

    def yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)

    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    def get_data_loader(self, iterator):
        pprint(self.text_pipeline('This is a sample sentence.'))
        pprint('10')

        return DataLoader(iterator, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_batch)

    def get_train_data_loader(self):
        return self.get_data_loader(self.split_train_)

    def get_test_data_loader(self):
        return self.get_data_loader(self.test_dataset)

    def get_valid_data_loader(self):
        return self.get_data_loader(self.split_valid_)


if __name__ == "__main__":
    ds = DataLoaderPreprocessor(batch_size=8, shuffle=False)
