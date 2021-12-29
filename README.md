# Transformer encoder

Custom implementation of a transformer encoder used for text classification.
Pretrained weights are loaded from Huggingface's BERT model. Atop of BERT is 
appended a linear layer which was trained to perform text classification
on the AG News dataset.

See documentation for more information.

## Setup

Create an environment with all necessary packages using conda.
```
conda env create --name zpja --file env.yml
conda activate zpja
```


## Scripts

Run the following command to fine-tune the model for text classification:
```
python3 src/training/trainer.py --batch-size 16 --verbose 1
```

Run the following to evaluate the fine-tuned model. Requires saved weights 
in a path that is specified in ``src/config.json`` file.
```
python3 src/training/evaluation.py --batch-size 16 --verbose 1
```


