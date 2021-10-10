import json
import time

import torch
from torchinfo import summary

from src.training.data_loader import DataLoaderPreprocessor
from src.transformer.encoder import TransformerEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SingleGPULossCompute:
    """
    A single GPU loss computation.
    """

    def __init__(self, model, criterion, optimizer=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x, y, norm):
        x = self.model(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.optimizer is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.optimizer.zero_grad()
        return loss.data[0] * norm


def run_epoch(data_iter, model, loss_compute, log_interval=100):
    total_tokens = 0
    total_loss = 0
    tokens = 0
    step_acc = 0
    step_count = 0
    epoch_total_acc = 0
    epoch_batches = 0

    for batch_idx, batch in enumerate(data_iter):
        predicted_labels = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(predicted_labels, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        acc = (predicted_labels.argmax(1) == batch.labels).sum().item()
        step_acc += acc
        step_count += batch.labels.size(0)
        epoch_batches += 1
        epoch_total_acc += acc

        if batch_idx % log_interval == 1:
            print("Epoch step: {:5d} Loss: %f Accuracy: {:8.3f}" %
                  (batch_idx, loss / batch.ntokens, step_acc / step_count))
            step_acc, step_count, tokens = 0, 0, 0

    epoch_acc = epoch_total_acc / epoch_batches

    return epoch_acc


def train(num_epochs, train_iter, valid_iter):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = NoamOptimizer(
        torch.optim.SGD(model.parameters(), lr=config['train']['learning_rate'])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
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
conf_trans = config['transformer']

data_preprocessor = DataLoaderPreprocessor(batch_size=config['train']['batch_size'], shuffle=True)
model = TransformerEncoder(num_layers=conf_trans['num_layers'], num_heads=conf_trans['num_heads'],
                           d_model=conf_trans['dim_model'], d_ff=conf_trans['dim_ff'],
                           input_vocab_size=data_preprocessor.vocab_size,
                           max_position_encoding=conf_trans['max_position_encoding']).to(device)
summary(model)
train_dataloader = data_preprocessor.get_train_data_loader()
valid_dataloader = data_preprocessor.get_valid_data_loader()
test_dataloader = data_preprocessor.get_test_data_loader()

train(config['train']['epochs'], train_iter=train_dataloader, valid_iter=valid_dataloader)
