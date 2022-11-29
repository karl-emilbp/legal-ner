import os
import argparse
from utils import get_root_path

def process_dataset(path):
    '''
    Function that splits datset in CoNNL format into sentences instead of sections.
    '''

    new_lines = []

    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line_content = line.split(' ')
        new_lines.append(line_content[0]+'\t'+line_content[-1])
        if '.' in line_content[0] and lines[i+1].split(' ')[0][0].isupper():
            new_lines.append('\n')

    processed_dataset = "".join(new_lines)


    return processed_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--datafile',
        default="",
        type=str,
        help='Path for datafile to process.'
    )

    parser.add_argument(
        '--outputfile',
        default="",
        type=str,
        help='Path to file for saving output.'
    )
    args = parser.parse_args()

    processed_dataset = process_dataset(args.datafile)
    
    with open(args.outputfile, mode="w+", encoding="utf-8") as f:
        f.write(processed_dataset)
        