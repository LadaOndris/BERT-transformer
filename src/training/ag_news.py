import json
import time

import torch
from torchinfo import summary

from src.training.data_loader import DataLoaderPreprocessor
from src.training.optimizer import NoamOptimizer
from src.transformer.classifier import TransformerClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SingleGPULossCompute:
    """
    A single GPU loss computation.
    """

    def __init__(self, model, criterion, optimizer=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x, y):
        loss = self.criterion(x, y)
        loss.backward()
        if self.optimizer is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.optimizer.zero_grad()
        return loss.data


def run_epoch(data_iter, model, loss_compute, log_interval=100):
    total_loss = 0
    step_acc = 0
    step_count = 0
    epoch_total_acc = 0
    epoch_batches = 0

    for batch_idx, batch in enumerate(data_iter):
        labels, sequences, pad_masks = batch
        batch_size = sequences.size(0)

        predicted_labels = model.forward(sequences, pad_masks)
        loss = loss_compute(predicted_labels, labels)
        total_loss += loss
        acc = torch.sum(predicted_labels.argmax(1) == labels).item()
        step_acc += acc
        step_count += batch_size
        epoch_batches += 1
        epoch_total_acc += acc

        if batch_idx % log_interval == 1:
            print("Epoch step: {:5d}/{:5d} Loss: {:5.3f} Accuracy: {:8.3f}".format(
                batch_idx + 1, int(len(data_iter) / batch_size), loss / batch_size, step_acc / step_count))
            step_acc, step_count, tokens = 0, 0, 0

    epoch_acc = epoch_total_acc / epoch_batches

    return epoch_acc


def train(num_epochs, train_iter, valid_iter):
    criterion = torch.nn.CrossEntropyLoss()
    # TODO: Implement NoamOptimizer as a LR schedular!
    optimizer = NoamOptimizer(
        d_model=config['transformer']['dim_model'],
        warmup_steps=config['train']['warmup_steps'],
        optimizer=torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate']),
        lr_coeff=config['train']['lr_coeff']
    )
    epoch_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_acc = run_epoch(train_iter, model, SingleGPULossCompute(model, criterion, optimizer))
        model.eval()
        valid_acc = run_epoch(valid_iter, model, SingleGPULossCompute(model, criterion))

        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'train accuracy {:8.3f} | valid accuracy {:8.3f}'.format(epoch, time.time() - epoch_start_time, train_acc,
                                                                       valid_acc))
        print('-' * 59)
        epoch_start_time = time.time()


# def evaluate(dataloader):
#     model.eval()
#     total_acc, total_count = 0, 0
#
#     with torch.no_grad():
#         for idx, (label, text, offsets) in enumerate(dataloader):
#             predicted_label = model(text, offsets)
#             loss = criterion(predicted_label, label)
#             total_acc += (predicted_label.argmax(1) == label).sum().item()
#             total_count += label.size(0)
#     return total_acc / total_count


with open('./src/config.json', 'r') as config_file:
    config = json.load(config_file)

data_preprocessor = DataLoaderPreprocessor(batch_size=config['train']['batch_size'], shuffle=True)
model = TransformerClassifier(config, data_preprocessor.num_class, data_preprocessor.vocab_size).to(device)
summary(model)
train_dataloader = data_preprocessor.get_train_data_loader()
valid_dataloader = data_preprocessor.get_valid_data_loader()
test_dataloader = data_preprocessor.get_test_data_loader()

train(config['train']['epochs'], train_iter=train_dataloader, valid_iter=valid_dataloader)
