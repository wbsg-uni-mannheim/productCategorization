from pathlib import Path
import csv
import pandas as pd

def load_raw_dataset():
    """Load raw dataset for the given experiments"""
    relative_path = '../../../data/raw/webdatacommons/webdatacommons_data_raw.csv'
    dataset = pd.read_csv(relative_path, sep=',', encoding='utf-8', quotechar='"',
                                  quoting=csv.QUOTE_ALL)

    return dataset

def detect_host(url):
    value = url.replace('https://', '')
    value = value.replace('http://', '')
    value = value.replace('www.', '')

    values = value.split('/')

    return values[0]

if __name__ == '__main__':
    dataset = load_raw_dataset()

    dataset['host_url'] = dataset['url'].map(detect_host)
    print(len(dataset['host_url'].sort_values().unique()))
