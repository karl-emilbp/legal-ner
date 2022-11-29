import os
import matplotlib.pyplot as plt

def load_dataset(datafile):
    '''
    Function for loading full dataset from file.
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

def get_statistics(dataset):
    '''
    Function for computing statistics about dataset.
    '''

    num_sent = 0
    num_entities = {}

    for sent in dataset:
        num_sent += 1
        for example in sent:
            _, entity = example.split("\t")
            if entity != 'O':
                if entity[2:] in num_entities:
                    num_entities[entity[2:]] += 1
                else:
                    num_entities[entity[2:]] = 0

    return num_sent, num_entities

if __name__ == '__main__':
    datafile = '../data/train.conll'
    dataset = load_dataset(datafile)
    num_sent, num_entities = get_statistics(dataset)
    print(num_sent)
    print(num_entities)
    plt.bar(list(num_entities.keys()), list(num_entities.values()))
    plt.xlabel("Named Entities")
    plt.ylabel("Number of instances")
    plt.show()
