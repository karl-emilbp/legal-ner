import os
from utils import get_root_path

def get_chunks(n, s):
    pieces = s.split()
    return (" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n))

def create_mlm_dataset(domsdatabase_docs_dir, retsinformation_docs_dir, shr_docs_dir):
    '''
    Function for creating masked language model dataset for domain adaptation.
    '''

    train_ft = ""
    train_dpt = ""
    test_dpt = ""

    # Get domdasdatabase docs data. 
    num_cases = len(os.listdir(domsdatabase_docs_dir))
    num_train_ft = int(num_cases*0.3)
    num_train_dpt = int(num_cases*0.6)

    for i, case_dir in enumerate(os.listdir(domsdatabase_docs_dir)):
        for file in os.listdir(os.path.join(domsdatabase_docs_dir, case_dir)):
            if file.endswith(".txt"):
                with open(os.path.join(domsdatabase_docs_dir, case_dir, file), encoding="utf-8") as f:
                    text = f.read()
                
                chunks = get_chunks(64, text)

                if i <= num_train_ft:
                    for chunk in chunks:
                        newline_chunk = chunk + "\n"
                        train_ft += newline_chunk
                elif num_train_ft < i <= num_train_ft+num_train_dpt:
                    for chunk in chunks:
                        newline_chunk = chunk + "\n"
                        train_dpt += newline_chunk
                elif i > num_train_ft+num_train_dpt:
                    for chunk in chunks:
                        newline_chunk = chunk + "\n"
                        test_dpt += newline_chunk

    # Get retsinformation docs data.
    num_files = len(os.listdir(retsinformation_docs_dir)) / 2 # Divide by 2 as only .txt are needed and not .pdf files.
    num_train_ft = int(num_files*0.3) * 2
    num_train_dpt = int(num_files*0.6) * 2

    for i, file in enumerate(os.listdir(os.path.join(retsinformation_docs_dir))):
        if file.endswith(".txt"):
            with open(os.path.join(retsinformation_docs_dir, file), encoding="utf-8") as f:
                text = f.read()
            
            chunks = get_chunks(64, text)

            if i <= num_train_ft:
                for chunk in chunks:
                    newline_chunk = chunk + "\n"
                    train_ft += newline_chunk
            elif num_train_ft < i <= num_train_ft+num_train_dpt:
                for chunk in chunks:
                    newline_chunk = chunk + "\n"
                    train_dpt += newline_chunk
            elif i > num_train_ft+num_train_dpt:
                for chunk in chunks:
                    newline_chunk = chunk + "\n"
                    test_dpt += newline_chunk

    # Get sø- og handelsret docs data.
    num_cases = len(os.listdir(shr_docs_dir))
    num_train_ft = int(num_cases*0.3)
    num_train_dpt = int(num_cases*0.6)

    for i, case_dir in enumerate(os.listdir(shr_docs_dir)):
        for file in os.listdir(os.path.join(shr_docs_dir, case_dir)):
            if file.endswith(".txt"):
                if os.stat(os.path.join(shr_docs_dir, case_dir, file)).st_size <= 1:
                    print("Skipping empty file.")
                    continue
                
                with open(os.path.join(shr_docs_dir, case_dir, file), encoding="utf-8") as f:
                    text = f.read()
                
                chunks = get_chunks(64, text)

                if i <= num_train_ft:
                    for chunk in chunks:
                        newline_chunk = chunk + "\n"
                        train_ft += newline_chunk
                elif num_train_ft < i <= num_train_ft+num_train_dpt:
                    for chunk in chunks:
                        newline_chunk = chunk + "\n"
                        train_dpt += newline_chunk
                elif i > num_train_ft+num_train_dpt:
                    for chunk in chunks:
                        newline_chunk = chunk + "\n"
                        test_dpt += newline_chunk
                        
    train_ft = train_ft.replace(' ', '\n')      

    return train_ft, train_dpt, test_dpt


if __name__ == '__main__':

    domsdatabase_docs_dir = os.path.join(get_root_path(),'data','domsdatabase_docs')
    retsinformation_docs_dir = os.path.join(get_root_path(),'data','retsinformation_docs')
    shr_docs_dir = os.path.join(get_root_path(),'data','shr_docs')
    train_ft, train_dpt, test_dpt = create_mlm_dataset(domsdatabase_docs_dir, retsinformation_docs_dir, shr_docs_dir)

    data_dir = os.path.join(get_root_path(),'data','domain_lm')

    with open(os.path.join(data_dir, 'train_ft.txt'), mode="w+", encoding="utf-8") as f:
        f.write(train_ft)

    with open(os.path.join(data_dir, 'train_dpt.txt'), mode="w+", encoding="utf-8") as f:
        f.write(train_dpt)

    with open(os.path.join(data_dir, 'test_dpt.txt'), mode="w+", encoding="utf-8") as f:
        f.write(test_dpt)