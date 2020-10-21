"""Convert WebDataCommons Data into unified schema"""


def reduce_schema(df_original_dataset):
    df_dataset = df_original_dataset.copy()

    df_dataset.rename(columns={'pathlist_ids': 'path_list', 'CategoryName': 'category',
                               'desc': 'description'}, inplace=True)

    return df_dataset[['title', 'description', 'category', 'path_list']]
