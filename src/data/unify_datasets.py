import src.data.icecat.unify as icecat_unify
import src.data.rakuten.unify as rakuten_unify
import src.data.webdatacommons.unify as webdatacommons_unify


def reduce_schema(df, dataset):
    if dataset == 'icecat':
        return icecat_unify.reduce_schema(df)
    elif dataset == 'rakuten' or dataset == 'subset_rakuten':
        return rakuten_unify.reduce_schema(df)
    elif dataset == 'webdatacommons':
        return webdatacommons_unify.reduce_schema(df)
    elif dataset == 'wdc_ziqi':
        return webdatacommons_unify.reduce_schema(df)
    else:
        raise ValueError('Schema of Dataset {} is not defined'.format(dataset))
