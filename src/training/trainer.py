import json
import os
import time

import torch
from torchinfo import summary

from src.training.data_loader import DataLoaderPreprocessor
from src.training.huggingface import create_model, get_bert_tokenizer
from src.training.losses import SingleGPULossCompute


def run_epoch(data_iter, model: torch.nn.Module, loss_compute, log_interval=50, verbose=False):
    total_loss = 0
    step_acc = 0
    step_count = 0
    epoch_total_acc = 0
    epoch_batches = 0

    for batch_idx, batch in enumerate(data_iter):
        labels, input_ids, token_type_ids, pad_masks = batch
        batch_size = input_ids.size(0)

        predicted_labels = model.forward(input_ids, token_type_ids)
        loss = loss_compute(predicted_labels, labels)
        total_loss += loss
        acc = torch.sum(predicted_labels.argmax(1) == labels).item()
        step_acc += acc
        step_count += batch_size
        epoch_batches += 1
        epoch_total_acc += acc

        if verbose and batch_idx % log_interval == 0:
            print("Epoch step: {:5d}/{:5d} Loss: {:5.3f} Accuracy: {:8.3f}".format(
                batch_idx + 1, int(len(data_iter) / batch_size), loss / batch_size, step_acc / step_count))
            step_acc, step_count, tokens = 0, 0, 0

    epoch_acc = epoch_total_acc / epoch_batches

    return epoch_acc


def train(model: torch.nn.Module, num_epochs, train_iter, valid_iter, save_dir):
    criterion = torch.nn.CrossEntropyLoss()

    epoch_start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['train']['lr_decay'])

    for epoch in range(num_epochs):
        torch.save(model.state_dict(), get_save_path(save_dir, epoch))

        model.train()
        train_acc = run_epoch(train_iter, model, SingleGPULossCompute(model, criterion, optimizer))
        model.eval()
        valid_acc = run_epoch(valid_iter, model, SingleGPULossCompute(model, criterion))

        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'train accuracy {:8.3f} | valid accuracy {:8.3f}'
              .format(epoch, time.time() - epoch_start_time, train_acc, valid_acc))
        print('-' * 59)
        epoch_start_time = time.time()
        lr_scheduler.step()

        torch.save(model.state_dict(), get_save_path(save_dir, epoch))


def get_save_path(save_dir, epoch_num):
    return os.path.join(save_dir, F"model_{epoch_num}.weights")


if __name__ == '__main__':
    with open('./src/config.json', 'r') as config_file:
        config = json.load(config_file)

    data_preprocessor = DataLoaderPreprocessor(batch_size=config['train']['batch_size'],
                                               shuffle=True,
                                               tokenizer=get_bert_tokenizer())
    bert_classifier = create_model(config)

    summary(bert_classifier)
    train_dataloader = data_preprocessor.get_train_data_loader()
    valid_dataloader = data_preprocessor.get_valid_data_loader()
    test_dataloader = data_preprocessor.get_test_data_loader()

    save_dir = 'models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train(bert_classifier, config['train']['epochs'], train_iter=train_dataloader, valid_iter=valid_dataloader,
          save_dir=save_dir)
