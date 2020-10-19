"""Convert WebDataCommons Data into unified schema"""


def reduce_schema(df_original_dataset):
    df_dataset = df_original_dataset.copy()

    df_dataset.rename(columns={'pathlist_names': 'path_list'}, inplace=True)

    return df_dataset[['title', 'description', 'brand', 'category', 'path_list']]
