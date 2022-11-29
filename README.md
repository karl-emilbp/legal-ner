# Named Entity Recognition for Danish Legal Texts
Repository for supervised and few-shot named entity recognition in the danish legal domain.

## Data
This repository provides datasets for named entity recognition in the danish legal domain in the data folder:
- danish_legal_ner_dataset.conll
  - train.conll

This dataset is used for supervised named entity recognition and consists of 2415 sentences annotated with 8 named entities:
- Organisation
- Person
- Dato
- Lokation
- Lov
- Retsinstans
- Dommer
- Advokat

The train.conll file is a processed version of danish_legal_ner_dataset.conll.

- few_shot_new_dataset.conll
  - test.conll

This dataset is used for evaluating few-shot named entity recognition algortihm in the danish legal domain and consists of 1480 sentences annotated with 5 named entities:
- Land
- By
- Retspraksis
- Litteratur
- Sagsnummer

The test.conll file is a processed version of few_shot_new_dataset.conll.

## Usage
The src folder provides scripts for training and evaluating supervised and few-named entity recognition models.

To fine-tune a transformer model for supervised named entity recognition on the dataset danish_legal_ner_dataset.conll, run the following script:
```bash
python train_ner.py --labels ../data/labels.txt --data ../data/train.conll --checkpoint <huggingface-remote-or-local-model-checkpoint> --output_dir <path-to-output-dir> --batch_size <batch_size> --lr <lr> --epochs <epochs>
```

The script datasampler.py provides an algorithm to sample arbitrary N-Way K-Shot support sets for few-shot named entity recognition. The algorithm also outputs a file with remaining sentences not sampled in support set to use as query set. For example to sample a 2-Way 5-Shot support and query set from the dataset few_shot_new_dataset.conll run the following:

```bash
python datasampler.py --n 2 --k 5 --datafile ../data/test.conll
```

To evaluate a few-shot algorithm based on StructShot using a sampled support and query set run the following:

```bash
python fewshot.py --data_dir ../data/ --labels ../data/labels.txt --target_labels ../data/labels_few_shot.txt --train_fname train --sup_fname <path-to-support-set-file> --test_fname <path-to-query-set-file> --model_name_or_path <huggingface-model-name> --checkpoint huggingface-remote-or-local-model-checkpoint<> --output_dir <path-to-output-dir> --algorithm StructShot --gpus 1 --eval_batch_size <eval_batch_size>
```