"""Convert WebDataCommons Data into unified schema"""


def reduce_schema(df_original_dataset):
    df_dataset = df_original_dataset.copy()

    df_dataset.rename(columns={'pathlist_ids': 'path_list', 'CategoryName': 'category',
                               'desc': 'description'}, inplace=True)

    # Convert dtype to string
    df_dataset['title'] = df_dataset['title'].astype(str)
    df_dataset['category'] = df_dataset['category'].astype(str)
    df_dataset['description'] = df_dataset['description'].astype(str)

    return df_dataset[['title', 'description', 'category', 'path_list']]
