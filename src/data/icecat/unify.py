"""Convert IceCat Data into unified schema"""


def reduce_schema(df_original_dataset):
    df_dataset = df_original_dataset.copy()

    df_dataset['description'] = df_dataset['Description.LongDesc'] + \
                                df_dataset['SummaryDescription.LongSummaryDescription']

    df_dataset.rename(columns={'Title': 'title', 'Category.Name.Value': 'category', 'Brand': 'brand',
                               'pathlist_names': 'path_list'}, inplace=True)

    return df_dataset[['title', 'description', 'brand', 'category', 'path_list']]
