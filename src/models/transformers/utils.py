from transformers import BertTokenizerFast, BertForSequenceClassification, RobertaTokenizerFast, \
    RobertaForSequenceClassification

def provide_model_and_tokenizer(name, num_labels):
    if name == 'transformers-base-uncased':
        return bert_base_uncased(num_labels)
    elif name == 'transformers-large-uncased':
        return  bert_large_uncased(num_labels)
    elif name == 'roberta-base':
        return roberta_base(num_labels)

    raise ValueError('Unknown model name: {}!'.format(name))

def provide_tokenizer(name):
    if name == 'roberta-base':
        return roberta_base_tokenizer()

    raise ValueError('Unknown model name: {}!'.format(name))

def bert_base_uncased(num_labels):
    tokenizer = BertTokenizerFast.from_pretrained('transformers-base-uncased')
    model = BertForSequenceClassification.from_pretrained("transformers-base-uncased",num_labels=num_labels)

    return tokenizer, model

def bert_large_uncased(num_labels):
    tokenizer = BertTokenizerFast.from_pretrained('transformers-large-uncased')
    model = BertForSequenceClassification.from_pretrained("transformers-large-uncased",num_labels=num_labels)

    return tokenizer, model

def roberta_base(num_labels):
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)

    return tokenizer, model

def roberta_base_tokenizer():
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    return tokenizer
