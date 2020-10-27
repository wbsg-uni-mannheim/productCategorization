"""Convert WebDataCommons Data into unified schema"""


def reduce_schema(df_original_dataset):
    df_dataset = df_original_dataset.copy()

    df_dataset.rename(columns={'CategoryID': 'category', 'pathlist': 'path_list'}, inplace=True)

    # Convert dtype to string
    df_dataset['title'] = df_dataset['title'].astype(str)
    df_dataset['category'] = df_dataset['category'].astype(str)

    return df_dataset[['title', 'category', 'path_list']]
