import argparse
import json
import os
import time

import torch
from torchinfo import summary

from src.data.loader import DataLoaderPreprocessor
from src.training.losses import SingleGPULossCompute
from src.transformer.huggingface import create_model, get_bert_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(data_iter, model: torch.nn.Module, loss_compute, verbose: bool, log_interval=500, save_model=False,
              epoch=-1, return_confusion_matrix=False):
    model = model.to(device)
    total_loss = 0
    step_acc = 0
    step_count = 0
    epoch_total_acc = 0
    epoch_count = 0
    confusion_matrix = torch.zeros(4, 4)

    for batch_idx, batch in enumerate(data_iter):
        labels, input_ids, token_type_ids, pad_masks = batch
        labels = labels.to(device)
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        pad_masks = pad_masks.to(device)

        batch_size = input_ids.size(0)

        predicted_logits = model.forward(input_ids, token_type_ids)
        loss = loss_compute(predicted_logits, labels)
        total_loss += loss
        pred_labels = predicted_logits.argmax(1)
        acc = torch.sum(pred_labels == labels).item()
        step_acc += acc
        step_count += batch_size
        epoch_total_acc += acc
        epoch_count += batch_size

        if return_confusion_matrix:
            for t, p in zip(labels, pred_labels):
                confusion_matrix[t.long(), p.long()] += 1

        if verbose and batch_idx % log_interval == 0:
            print("Epoch step: {:5d}/{:5d} Loss: {:5.3f} Accuracy: {:8.3f}".format(
                batch_idx + 1, int(len(data_iter) / batch_size), loss / batch_size, step_acc / step_count))
            step_acc, step_count, tokens = 0, 0, 0
            if save_model:
                torch.save(model.state_dict(), get_save_path(save_dir, epoch, batch_idx))

    epoch_acc = epoch_total_acc / epoch_count

    if return_confusion_matrix:
        confusion_matrix /= epoch_count
        return epoch_acc, confusion_matrix
    else:
        return epoch_acc


def train(model: torch.nn.Module, num_epochs, train_iter, valid_iter, save_dir, verbose):
    criterion = torch.nn.CrossEntropyLoss()

    epoch_start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['train']['lr_decay'])

    for epoch in range(num_epochs):
        model.train()
        train_acc = run_epoch(train_iter, model, SingleGPULossCompute(model, criterion, optimizer), verbose,
                              log_interval=10)
        model.eval()
        valid_acc = run_epoch(valid_iter, model, SingleGPULossCompute(model, criterion), verbose, log_interval=10)

        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'train accuracy {:8.3f} | valid accuracy {:8.3f}'
              .format(epoch, time.time() - epoch_start_time, train_acc, valid_acc))
        print('-' * 59)
        epoch_start_time = time.time()
        lr_scheduler.step()

        torch.save(model.state_dict(), get_save_path(save_dir, epoch, 'end'))


def get_save_path(save_dir, epoch_num, batch_num):
    return os.path.join(save_dir, F"model_{epoch_num}_{batch_num}.weights")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, action='store', default='src/config.json',
                        help='a config file name')
    parser.add_argument('--verbose', type=int, action='store', default=0,
                        help='verbose training output')
    parser.add_argument('--batch-size', type=int, action='store', default=8,
                        help='the number of samples in a batch')
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    bert_classifier = create_model(config)
    summary(bert_classifier)

    data_preprocessor = DataLoaderPreprocessor(batch_size=args.batch_size,
                                               shuffle=True,
                                               tokenizer=get_bert_tokenizer())
    train_dataloader = data_preprocessor.get_train_data_loader()
    valid_dataloader = data_preprocessor.get_valid_data_loader()
    test_dataloader = data_preprocessor.get_test_data_loader()

    save_dir = config['train']['save_weights_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train(bert_classifier,
          num_epochs=config['train']['epochs'],
          train_iter=train_dataloader,
          valid_iter=valid_dataloader,
          save_dir=save_dir,
          verbose=bool(args.verbose))
