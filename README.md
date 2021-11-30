# Transformer encoder

Custom implementation of a transformer encoder used for text classification.
Pretrained weights are loaded from Huggingface's BERT model.

See documentation for more information.

## Requirements


## Scripts

Run the following command to fine-tune the model for text classification:
```
python3 src/training/trainer.py --batch-size 16 --verbose 1
```

Run the following to evaluate the fine-tuned model. Requires saved weights 
in a path that is specified by ``src/config.json``.
```
python3 src/training/evaluation.py --batch-size 16 --verbose 1
```


