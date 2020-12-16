
LMs = ['None', 'Training Set - Title', 'Training Set - Title + Description (each sentence of description)',
      'Training Set - Title +  Path ([Title] belongs to [Path])',
      'Training Set - Title + Description + Path ([Title] belongs to [Path] - each sentence of the description)',
      'WDC Product Corpus - Selected Titles',
      'WDC Product Corpus - Selected Titles + Description (each sentence of the description)',
      'WDC Product Corpus - Selected Titles + Description + Path ([Title] belongs to [Path] - each sentence of the description)'
]

Inputs = ['Title', 'Title + Description (Just concatenate)']

NNs = ['Basic Softmax Head (RobertaForSequenceClassification)', 'Recurrent Neural Network']

with open('C://Users//alebrink//Documents//02_Research//development//productCategorization//experiments//experiment_plan.csv', 'w') as f:
    for LM in LMs:
        for Input in Inputs:
            for NN in NNs:
                f.write('{},{},{}\n'.format(LM, Input, NN))