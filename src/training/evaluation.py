import argparse
import json
import time

import torch
from torchinfo import summary

from src.data.loader import DataLoaderPreprocessor
from src.training.trainer import run_epoch
from src.transformer.huggingface import create_model, get_bert_tokenizer


def evaluate(model, data_iterator, verbose):
    start_time = time.time()

    dummy_loss = torch.nn.CrossEntropyLoss()
    accuracy = run_epoch(data_iterator, model, dummy_loss, verbose=verbose)

    print('-' * 59)
    print('| time: {:5.2f}s | test accuracy {:8.3f} |'
          .format(time.time() - start_time, accuracy))
    print('-' * 59)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, action='store', default='src/config.json',
                        help='a config file name')
    parser.add_argument('--verbose', type=int, action='store', default=0,
                        help='verbose training output')
    parser.add_argument('--batch-size', type=int, action='store', default=32,
                        help='the number of samples in a batch')
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    bert_classifier = create_model(config)
    bert_classifier.load_state_dict(torch.load(config['test']['saved_weights_path']))
    summary(bert_classifier)

    data_preprocessor = DataLoaderPreprocessor(batch_size=args.batch_size,
                                               shuffle=True,
                                               tokenizer=get_bert_tokenizer())
    test_dataloader = data_preprocessor.get_test_data_loader()

    evaluate(bert_classifier, data_iterator=test_dataloader, verbose=args.verbose)
