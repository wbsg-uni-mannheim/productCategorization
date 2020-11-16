"""Convert WebDataCommons Data into unified schema"""


def reduce_schema(df_original_dataset):
    df_dataset = df_original_dataset.copy()

    df_dataset['path_list'] = df_dataset['lvl1'] + '>' + df_dataset['lvl2'] + '>' + df_dataset['lvl3']

    #Use lvl3 as category for now --> Will change in the near future
    df_dataset.rename(columns={'Name': 'title', 'lvl3': 'category',
                               'Description': 'description'}, inplace=True)

    # Convert dtype to string
    df_dataset['title'] = df_dataset['title'].astype(str)
    df_dataset['category'] = df_dataset['category'].astype(str)
    df_dataset['description'] = df_dataset['description'].astype(str)

    return df_dataset[['title', 'description', 'category', 'path_list']]
