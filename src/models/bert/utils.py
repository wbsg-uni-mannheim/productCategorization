from transformers import BertTokenizerFast, BertForSequenceClassification, RobertaTokenizerFast, \
    RobertaForSequenceClassification

def provide_model_and_tokenizer(name, num_labels):
    if name == 'bert-base-uncased':
        return bert_base_uncased(num_labels)
    elif name == 'bert-large-uncased':
        return  bert_large_uncased(num_labels)
    elif name == 'roberta-base':
        return roberta_base(num_labels)

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
