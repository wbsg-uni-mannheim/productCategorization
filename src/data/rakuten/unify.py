"""Convert WebDataCommons Data into unified schema"""


def reduce_schema(df_original_dataset):
    df_dataset = df_original_dataset.copy()

    df_dataset.rename(columns={'CategoryID': 'category', 'pathlist': 'path_list'}, inplace=True)

    return df_dataset[['title', 'category', 'path_list']]
