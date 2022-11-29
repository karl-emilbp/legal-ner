import os
import itertools
import pandas as pd
import numpy as np
import argparse
import torch
from loguru import logger
from utils import get_root_path
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

def get_labels_and_label_ids(label_filepath):
    '''
    Function for getting labels and mapping labels to label_ids.
    '''

    with open(label_filepath, encoding='utf-8') as f:
        labels = f.readlines()

    labels = [label if label[-1:] != '\n' else label[:-1] for label in labels]

    label_encoding_dict = {}

    for i, label in enumerate(labels):
        label_encoding_dict[label] = i

    return labels, label_encoding_dict

def get_tokens_and_ner_tags(filename):
    '''
    Function for loading tokens and ner tags.
    '''
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        tokens = [[x.split('\t')[0] for x in y] for y in split_list]
        entities = [[x.split('\t')[1][:-1] for x in y] for y in split_list]

    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})
  
def get_token_dataset(filename):
    '''
    Function for creating train and test dataset.
    '''
    df = get_tokens_and_ner_tags(filename)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    return (train_dataset, test_dataset)

def tokenize_and_align_labels(examples, tokenizer, label_encoding_dict):
    '''
    Function for tokenizing dataset.
    '''
    label_all_tokens = False
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    '''
    Function for computing evaluation metrics.
    '''
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    label_list, _ = get_labels_and_label_ids('../data/labels.txt')
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    metric = load_metric("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

def main(args):
    '''
    Function for training supervised NER model.
    '''

    # Load list of labels and dict of labels mapped to label_ids
    logger.info("Loading labels and label_ids")
    label_list, label_encoding_dict = get_labels_and_label_ids(args.labels)
    
    # Load model and tokenizer from huggingface checkpoint.
    logger.info(f"Loading model and tokenizer from checkpoint: {args.checkpoint}.")
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint, num_labels=len(label_list))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Load data and create tokenized dataset.
    logger.info("Loading and preprocessing dataset.")
    train_dataset, test_dataset = get_token_dataset(args.data)
    train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "label_encoding_dict": label_encoding_dict})
    test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "label_encoding_dict": label_encoding_dict})

    # Define trainingarguments.
    training_args = TrainingArguments(
        os.path.join(get_root_path(), 'models', args.output_dir),
        evaluation_strategy = "epoch",
        save_strategy='epoch',
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir
    )

    # Define trainer.
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=test_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    logger.info("Starting training of NER model.")
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help='batch size to use in training.'
    )

    parser.add_argument(
        '--checkpoint',
        default='distilbert-base-multilingual-cased',
        type=str,
        help='Huggingface pre-trained model checkpoint to use for fine-tuning.'
    )

    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Path to a file containing all labels.",
    )

    parser.add_argument(
        "--data",
        default="",
        type=str,
        help="Path to file containing dataset. CoNLL-2003 format is expected.",
    )

    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--lr",
        default=2e-5,
        type=float,
        help="Learning rate to use in training.",
    )

    parser.add_argument(
        "--weight_decay",
        default=1e-5,
        type=float,
        help="Weight decay to use in training.",
    )

    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Directory for saving checkpoints.",
    )

    parser.add_argument(
        "--logging_dir",
        default="",
        type=str,
        help="Path to Tensorboard logs directory.",
    )

    args = parser.parse_args()
    main(args)
    