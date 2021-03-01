Improving Hierarchical Product Classification using Domain-specific Language Modelling
==============================

In order to deliver a coherent user experience, product aggregators
like market places or price portals integrate product offers from
many web shops into a single product hierarchy. Recently, transformer
models have shown remarkable performance on various
NLP tasks. These models are pre-trained on huge cross-domain text
corpora using self-supervised learning and fine-tuned afterwards
for specific downstream tasks. Research from other application
domains indicates that additional self-supervised pre-training using
domain-specific text corpora can further increase downstream
performance without requiring additional task-specific training
data.

In this paper we first show that transformers outperform a more
traditional fastText-based classification technique on the task of
assigning product offers from different web shops into a single
product hierarchy. Afterwards, we investigate whether it is possible
to further improve the performance of the transformer model by
performing additional self-supervised pre-training using different
corpora of product offers which were extracted from the Common
Crawl. Our experiments show that by using large numbers of
related product offers together with the heterogeneous categorization
information from the original web shops for masked language
modelling, it is possible to further increase the performance of the
transformer model by 1.22% in wF1 and 1.36% in hF1 reaching a
performance of nearly 89% wF1.
All source code to reproduce our results is available in this repository.

The data needed to evaluate the results is online available as well:

1. [MWPD Data set](https://ir-ischool-uos.github.io/mwpd/)
2. [Icecat Data set](http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/categorization/)
3. [Language Modelling - Product Corpora](http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/languagemodelling/)
