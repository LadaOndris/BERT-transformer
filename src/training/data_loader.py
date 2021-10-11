import random
from pprint import pprint
from typing import Collection, Iterator

import torch
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator

from src.transformer.operations import create_padding_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BucketSampler(Sampler[int]):
    """
    Groups sentences of similar lengths to minimize padding.

    Adapted from:
    https://github.com/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb
    """

    def __init__(self, tokenizer, batch_size: int, data_source: Collection):
        super().__init__(data_source)
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        indices = [(i, len(self.tokenizer(s[1]))) for i, s in enumerate(self.data_source)]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(indices), self.batch_size * 100):
            pooled_indices.extend(sorted(indices[i:i + self.batch_size * 100], key=lambda x: x[1]))

        pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        for i in range(0, len(pooled_indices), self.batch_size):
            yield pooled_indices[i:i + self.batch_size]

    def __len__(self):
        return len(self.data_source)


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
        labels = torch.tensor(label_list, dtype=torch.int64)
        sequences_padded = pad_sequence(text_list, batch_first=True)
        pad_mask = create_padding_mask(sequences_padded)
        return labels.to(device), sequences_padded.to(device), pad_mask.to(device)

    def get_data_loader(self, iterator):
        pprint(self.text_pipeline('This is a sample sentence.'))

        batch_sampler = BucketSampler(data_source=iterator,
                                      batch_size=self.batch_size,
                                      tokenizer=self.tokenizer)
        return DataLoader(iterator,
                          batch_sampler=batch_sampler,
                          collate_fn=self.collate_batch)

    def get_train_data_loader(self):
        return self.get_data_loader(self.split_train_)

    def get_test_data_loader(self):
        return self.get_data_loader(self.test_dataset)

    def get_valid_data_loader(self):
        return self.get_data_loader(self.split_valid_)


if __name__ == "__main__":
    ds = DataLoaderPreprocessor(batch_size=8, shuffle=False)
