from pathlib import Path

import pandas as pd

def load_datasets(dataset_name):
    """Load dataset for the given experiments"""
    project_dir = Path(__file__).resolve().parents[3]
    splits = ['train', 'validate', 'test']
    dataset = {}

    for split in splits:
        relative_path = 'data/processed/{}/split/raw/{}_data_{}.pkl'.format(dataset_name, split, dataset_name)
        file_path = project_dir.joinpath(relative_path)
        dataset[split] = pd.read_pickle(file_path)

    return dataset

if __name__ == '__main__':
    datasets = load_datasets('icecat')

    full_dataset = pd.DataFrame()

    for key in datasets:
        datasets[key]['dataset'] = key
        full_dataset = full_dataset.append(datasets[key])

    full_dataset['title_stripped'] = full_dataset['title'].str.strip()
    full_dataset['title_hashed'] = full_dataset['title_stripped'].map(hash).astype('str')
    grouped_dataset = full_dataset.groupby(axis=0, by=['title_hashed'])
    group = grouped_dataset.count()

    duplicate_group = group[group['title']> 1]
    duplicates = duplicate_group.index
    print(len(duplicates))

    duplicate_group = group[group['title'] > 2]
    duplicates = duplicate_group.index
    print(len(duplicates))

    duplicate_group = group[group['title'] > 3]
    duplicates = duplicate_group.index
    print(len(duplicates))
    #for duplicate in duplicates:
    #    subset = full_dataset[full_dataset['title_hashed'] == duplicate]
    #    print('-----')
    #    for index, row in subset.iterrows():
    #        print(row)
    #        print(row['description'])
    #    print('-----')