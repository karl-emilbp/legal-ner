import os
import copy
import random
import sys
import argparse
import numpy as np
from loguru import logger
from utils import get_root_path, bubble_sort

class NWayKShotDataSampler():
    '''
    Class for sampling N-Way K-Shot NER Few-Shot dataset.
    '''
    def __init__(self, n, k):
        self.n = n
        self.k = k

    def _load_dataset(self, datafile):
        '''
        Method for loading full dataset to sample N-Way K-Shot dataset from file.
        '''

        with open(datafile, encoding='utf-8') as f:
            data = f.readlines()

        # Convert to list of lists. Inner lists represents a sentence.
        dataset = []
        sentence = []
        for example in data:
            if example == '\n':
                dataset.append(sentence)
                sentence = []
            else:
                if example[:-1] != "\t":
                    sentence.append(example[:-1])

        return dataset

    def _sampling_status(self, num_examples):
        '''
        Method for checking status of support set sampling.
        '''

        bool_arr = (num_examples == self.k)
        if np.all(bool_arr):
            return True
        else:
            return False

    def _get_freqs(self, dataset):
        '''
        Method for getting frequencies of entity classes in dataset.
        '''

        freqs = {}

        for sent in dataset:
            for example in sent:
                _, entity = example.split('\t')
                if entity != 'O':
                    if entity[2:] in freqs:
                        freqs[entity[2:]] += 1
                    else:
                        freqs[entity[2:]] = 1

        return freqs

    def _process_dataset(self, dataset):
        '''
        Method for sorting sentences based on frequencies of entity classes.
        This makes sentences with entities that has lowest frequencies get sampled first.
        '''

        freqs = self._get_freqs(dataset)
        entities = list(freqs.keys())
        sorted_entities = bubble_sort(entities, freqs)

        processed_dataset = []
        cur_entity_dataset = []
        for cur_entity in sorted_entities:
            for sent in dataset[:]:
                for example in sent:
                    _, entity = example.split('\t')
                    if entity[2:] == cur_entity:
                        cur_entity_dataset.append(sent)
                        dataset.remove(sent)
                        break
        
            processed_dataset.extend(random.sample(cur_entity_dataset, k=len(cur_entity_dataset)))
            cur_entity_dataset = []

        processed_dataset.extend(dataset)
                    
        return processed_dataset

    def sample_support(self, dataset, entities):
        '''
        Method for sampling N-Way K-Shot support set.
        '''

        n_way_k_shot_support = []

        num_examples = np.zeros(self.n)

        # Sample support set.
        for sent in dataset[:]:
            cur_num_examples = copy.copy(num_examples)

            example_status = True
            for i, example in enumerate(sent):
                _, entity = example.split('\t')
                entity_status = False
                if entity != 'O':
                    entity_status = True
                    if entity[2:] not in entities:
                        example_status = False
                        break
                    if cur_num_examples[entities.index(entity[2:])] != self.k:
                        if entity[:2] != 'I-':
                            cur_num_examples[entities.index(entity[2:])] += 1
                    elif entity[2:] == sent[i-1].split('\t')[1][2:] and entity[:2] != 'B-':
                        continue
                    else:
                        example_status = False
                        break
            if example_status and entity_status:
                num_examples = cur_num_examples
                n_way_k_shot_support.append(sent)
                dataset.remove(sent)

        return n_way_k_shot_support, dataset, num_examples

    def sample_query(self, dataset, entities):
        '''
        Method for sampling query set for current N-Way K-Shot support set.
        '''

        # Sample query set.
        for sent in dataset[:]:
            entity_status = False
            for example in sent:
                _, entity = example.split('\t')
                if entity == 'O':
                    continue
                elif entity[2:] not in entities:
                    dataset.remove(sent)
                    entity_status = True
                    break
                else:
                    entity_status = True
            
            if entity_status != True:
                dataset.remove(sent)
            
        return dataset

    def _create_labels_file(self, entities):
        '''
        Method for creating file with labels used in N-Way K-Shot dataset.
        '''

        labels = ['O']
        for entity in entities:
            labels.append('B-'+entity)
            labels.append('I-'+entity)

        entities_str = "\n".join(labels)

        with open(os.path.join(get_root_path(), 'data', 'labels_few_shot.txt'), mode="w+", encoding="utf-8") as f:
            f.write(entities_str)

        return

    def sample_n_way_k_shot(self, args):
        '''
        Method for sampling a N-Way K-Shot dataset.
        '''

        logger.info(f"Sampling {self.n}-Way {self.k}-Shot support set.")

        if args.entities_file == "":
            all_entities = ['LAND', 'BY', 'RETSPRAKSIS', 'LITTERATUR', 'SAGSNUMMER']
            entities = random.sample(all_entities, args.n)
        else:
            with open(args.entities_file, encoding="utf-8") as f:
                entities = f.readlines()
                entities = [entity[:-1] if entity[-1] == '\n' else entity for entity in entities]
            if len(entities) != self.n:
                logger.info('Error: Number of entities in entities file and number of classes in datasampler does not match.')

        dataset = self._load_dataset(args.datafile)
        processed_dataset = self._process_dataset(dataset)
        
        # Sample N-Way K-Shot support set.
        support, processed_dataset, num_examples = self.sample_support(processed_dataset, entities)
        if self._sampling_status(num_examples) != True:
            logger.error(f"Unable to sample {self.n}-Way {self.k}-Shot support set.")
            sys.exit()

        logger.info(f"Sampling query set.")
        # Sample query set.
        query = self.sample_query(processed_dataset, entities)

        self._create_labels_file(entities)

        self.support = support
        self.query = query

        return

    def _format_dataset(self, dataset):
        '''
        Method formatting dataset with newlines for writing to file.
        '''

        new_lines = []
        for sent in dataset:
            for example in sent:
                new_lines.append(example+'\n')
            new_lines.append('\n')
        processed_dataset = "".join(new_lines)

        return processed_dataset

    def write_dataset_to_file(self, mode):
        '''
        Method for writing support and query datasets to file.        
        '''

        if mode.lower() == 'support':
            processed_dataset = self._format_dataset(self.support)

            with open(os.path.join(get_root_path(), 'data', f'{self.n}_way_{self.k}_shot_support.conll'), mode="w+", encoding='utf-8') as f:
                f.write(processed_dataset)

        elif mode.lower() == 'query':
            processed_dataset = self._format_dataset(self.query)

            with open(os.path.join(get_root_path(), 'data', 'query.conll'), mode="w+", encoding='utf-8') as f:
                f.write(processed_dataset)
        else:
            logger.info('Mode not supported. Please specify one of the modes: ("support", "query")')

        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n',
        default=1,
        type=int,
        help='Number of classes to sample for N-Way K-Shot support set.'
    )

    parser.add_argument(
        '--k',
        default=5,
        type=int,
        help='Number of samples for each to use in N-Way K-Shot support set.'
    )

    parser.add_argument(
        '--datafile',
        default="",
        type=str,
        help='Path to datafile to sample N-Way K-Shot dataset from.'
    )

    parser.add_argument(
        '--entities_file',
        default="",
        type=str,
        help='Path to file containing entities to sample.'
    )
    
    args = parser.parse_args()
    data_sampler = NWayKShotDataSampler(n=args.n, k=args.k)
    data_sampler.sample_n_way_k_shot(args)
    data_sampler.write_dataset_to_file('support')
    data_sampler.write_dataset_to_file('query')