from transformers import BertTokenizerFast, BertForSequenceClassification, RobertaTokenizerFast, \
    RobertaForSequenceClassification


def provide_model_and_tokenizer(name, num_labels):
    dict_models = {
        'bert-base-uncased': bert_base_uncased(num_labels),
        'bert-large-uncased': bert_large_uncased(num_labels),
        'roberta-base': roberta_base(num_labels),
    }
    return dict_models[name]

def bert_base_uncased(num_labels):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=num_labels)

    return tokenizer, model

def bert_large_uncased(num_labels):
    tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
    model = BertForSequenceClassification.from_pretrained("bert-large-uncased",num_labels=num_labels)

    return tokenizer, model

def roberta_base(num_labels):
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)

    return tokenizer, model
