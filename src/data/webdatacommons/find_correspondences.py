from pathlib import Path
import csv
import pandas as pd

def load_raw_datasets(dataset_name):
    """Load dataset for the given experiments"""
    project_dir = Path(__file__).resolve().parents[3]

    relative_path = 'data/raw/{}/{}_data_raw.csv'.format(dataset_name, dataset_name)
    file_path = project_dir.joinpath(relative_path)

    dataset = pd.read_csv(file_path.absolute(), sep=',', encoding='utf-8', quotechar='"', quoting=csv.QUOTE_ALL)
    return dataset

if __name__ == '__main__':
    ds_icecat = load_raw_datasets('icecat')
    ds_wdc = load_raw_datasets('webdatacommons')

    print(ds_wdc.columns)
    print(ds_icecat.columns)
    counter = 0


    #print(ds_wdc.head()['gtin'])
    #print(ds_icecat.head()['GTIN'])

    for index, row in ds_icecat.iterrows():
        raw_gtin = row['GTIN'].replace('[','').replace(']', '')
        raw_gtin = raw_gtin.split(',')

        for gtin in raw_gtin:
            gtin = gtin.strip()
            gtin = gtin.replace('\'', '')
            if len(gtin) > 0:
                int_gtin = int(gtin)
                ds_sub_wdc = ds_wdc[ds_wdc['gtin'] == int_gtin]

                if len(ds_sub_wdc) > 0:
                    counter = counter + 1
                    print('Icecat: Title: {} - GTIN: {} - Category: {} - Path: {}'
                        .format(row['Title'], row['GTIN'], row['Category.Name.Value'], row['pathlist_names']))
                    first_row = ds_sub_wdc.iloc[0]
                    #print(ds_sub_wdc)
                    #print(first_row)
                    print('WDC: Title: {} - GTIN: {} - Category: {} - Path: {}'
                          .format(first_row['title'], first_row['gtin'], first_row['CategoryName'], first_row['pathlist_names']))
                    print('-----------')

        if counter > 10:
            break;
