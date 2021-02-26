Improving Hierarchical Product Classification using Domain-specific Language Modelling
==============================

Hierarchical product classification is a major challenge for product aggregators, 
who support customers in finding the right product on-line. 
To guarantee a good customer experience, product aggregators integrate heterogeneous product data from a constantly growing number of web shops into a single product hierarchy. 
Hierarchical product classification is an on-going research topic. 
Recently, pre-trained transformer models have shown state of the art performance results on many benchmark NLP tasks.
These transformer models are pre-trained on huge text corpora using self-supervised learning and fine-tuned on downstream tasks.
In this work we demonstrate how hierarchical product classification can be improved using the same transfer learning technique.
We show that by using self-supervised masked language modelling on a large domain-specific product corpus, it is possible to improve hierarchical product classification.
This large corpus of heterogeneous product offers has been extracted from the public web. 
Our best approach outperforms state of the art approaches, which rely on general pre-training and task-specific fine-tuning by 5.32%. 

All source code to reproduce our results is available in this repository.

The data needed to evaluate the results is online available as well:

1. [MWPD Data set](https://ir-ischool-uos.github.io/mwpd/)
2. [Icecat Data set](http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/categorization/)
3. [Language Modelling - Product Corpora](http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/languagemodelling/)
