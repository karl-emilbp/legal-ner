import os
from pathlib import Path

def get_root_path():
    '''
    Get the root path of the project
    '''
    root_path = Path(os.path.abspath(__file__)).parent.parent
    return root_path

def bubble_sort(array, freqs):
    n = len(array)

    for i in range(n):
        already_sorted = True
        for j in range(n - i - 1):
            if freqs[array[j]] > freqs[array[j + 1]]:
                array[j], array[j + 1] = array[j + 1], array[j]
                already_sorted = False

        if already_sorted:
            break

    return array 