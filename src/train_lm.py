import os
import re
import argparse
import pandas as pd
from loguru import logger
from pathlib import Path
from utils import get_root_path
from datasets import load_dataset, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers_domain_adaptation import DataSelector, VocabAugmentor
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

def augment_vocab(tokenizer, model, args, ft_corpus_train):
    '''
    Function for augmenting model vocabulary to contain new tokens from in-domain dataset.
    '''

    logger.info("Augmenting vocabulary.")

    # Obtain new domain-specific terminology based on the fine-tuning corpus
    target_vocab_size = args.target_vocab_size
    augmentor = VocabAugmentor(
        tokenizer=tokenizer, 
        cased=True, 
        target_vocab_size=target_vocab_size
    )
    new_tokens = augmentor.get_new_tokens(ft_corpus_train)

    # Update model and tokenizer with new vocab terminologies
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

def load_data(tokenizer, model, dpt_corpus_train_data_selected, dpt_corpus_val):
    '''
    Function for loading and tokenizing datasets.
    '''

    logger.info("Loading and tokenizing datasets.")

    # Load datasets
    datasets = load_dataset(
        'text', 
        data_files={
            "train": dpt_corpus_train_data_selected, 
            "val": dpt_corpus_val
        }
    )

    tokenized_datasets = datasets.map(
        lambda examples: tokenizer(examples['text'], truncation=True, padding=True, max_length=model.config.max_position_embeddings),
        batched=True,
    )

    return tokenized_datasets

def remove_long_sequences(tokenized_datasets):
    '''
    Function for removing sequences longer than 512 tokens.
    '''

    # Remove sequences longer than 512 tokens.
    idx = []
    for i, row in enumerate(tokenized_datasets['train']):
        if len(row['input_ids']) > 512:
            idx.append(i)

    df = pd.DataFrame(tokenized_datasets['train'])
    df = df.drop(idx)

    tokenized_datasets_train = Dataset.from_pandas(df)

    return tokenized_datasets_train

def main(args):
    '''
    Function for Fine-Tuning Language model.
    '''

    # Domain-pre-training corpora (same as or broader than fine-tuning corpus)
    dpt_corpus_train = os.path.join(get_root_path(),"data","domain_lm", "train_dpt.txt")
    dpt_corpus_train_data_selected = os.path.join(get_root_path(),"data","domain_lm", "train_dpt_selected.txt")
    dpt_corpus_val = os.path.join(get_root_path(),"data","domain_lm", "test_dpt.txt")

    # Fine-tuning corpora
    ft_corpus_train = os.path.join(get_root_path(),"data","domain_lm", "train_ft.txt")

    # Load model and corpora
    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # Load text data into memory and clean text
    fine_tuning_texts = Path(ft_corpus_train).read_text(encoding='utf-8').splitlines()
    training_texts = Path(dpt_corpus_train).read_text(encoding='utf-8').splitlines()

    fine_tuning_texts = [re.sub('\uf095', '', fine_tuning_text) for fine_tuning_text in fine_tuning_texts]
    fine_tuning_texts = [re.sub('\uf0bd', '', fine_tuning_text) for fine_tuning_text in fine_tuning_texts]

    # Save to file
    Path(dpt_corpus_train_data_selected).write_text('\n'.join(training_texts), encoding="utf-8")

    # tokenizer, model = augment_vocab(tokenizer, model, args, ft_corpus_train)

    # Load datasets
    tokenized_datasets = load_data(tokenizer, model, dpt_corpus_train_data_selected, dpt_corpus_val)
    tokenized_datasets_train = remove_long_sequences(tokenized_datasets)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Instantiate TrainingArguments and Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        seed=42,
        dataloader_num_workers=2,
        logging_dir=args.logging_dir,
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets['val'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint',
        default='distilbert-base-multilingual-cased',
        type=str,
        help='Huggingface pre-trained model checkpoint to use for fine-tuning.'
    )

    parser.add_argument(
        '--target_vocab_size',
        default=261000,
        type=int,
        help='Target vocabulary size for Language model after vocabulary extension.'
    )

    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help='batch size to use in training.'
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
        "--output_dir",
        default="",
        type=str,
        help="Output dir for saving model checkpoints.",
    )

    parser.add_argument(
        "--logging_dir",
        default="",
        type=str,
        help="Logging dir for saving training logs.",
    )

    args = parser.parse_args()
    main(args)