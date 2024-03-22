# colBert_project_gsoc
Some files, ideas and other stuff for the ColBERT GSoC project.

1. Read the "efficienty tricks" paper: https://arxiv.org/pdf/2205.09707.pdf
2. The goal should be to write modular code, so that other RAG systems can reuse it.
3. https://github.com/bclavie/ragatouille
4. RAG tutorial: https://forem.julialang.org/svilupp/building-a-rag-chatbot-over-dataframesjl-documentation-hands-on-guide-449m
5. RAG survey paper: https://arxiv.org/pdf/2312.10997.pdf
6. Another RAG paper for NLP Tasks: https://arxiv.org/pdf/2005.11401.pdf
7. ColBERT paper: https://arxiv.org/pdf/2004.12832.pdf
8. ColBERTv2 paper: https://arxiv.org/abs/2112.01488 
9. ColBERT repo: https://github.com/stanford-futuredata/ColBERT

# Steps to carry out.

1. Get the pretrained ColBERT model running in Julia. Code to download the pretrained-model:

```
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz -P downloads/
tar -xvzf downloads/colbertv2.0.tar.gz -C downloads/
```

2. See how the pretrained model is applied for the indexing. For this, need to see how the `Indexer` works (lines 49-50 of `understanding_colbert/intro_notebook.py`). More specifically, we need to: 
    - See how tokenization is done from the checkpoint. This information is contained in the `modelling` module of the ColBERT repo.
    - For basic understanding, start with the `indexer.py` file containing the `Indexer` class.
    - The datasets are just `.tsv` files containing the queries and the documents. The first entry is the `qid` or the `pid`, and the second entry is the text.

## The `Queries` object

This is just an object containing the path of origin of the dataset, the data itself (in an ordered dict format, where the key is the `qid` and the value is the text). The queries are loaded using the `load_queries` method in the `evaluation.loaders` module. The `load_queries` function just scans through the dataset, collects all the `qids` in an ordered dict, and just checks whether all the `qids` are distinct. If all of this works fine, then an ordered dict is returned.

## The `Collection` object

This is similar to the `Queries` object. The collction has a field named `data`, which is just a list containing all the passages. The loading of the collection is done via the `load_collection` function from the `evaluation.loaders` module. The `load_collection` function just does the following: it gets the `pid` and the `passage` for each document. It checks if `pid` is either equal to `"id"`, or `pid` must be equal to the `line_idx` (the index of the line containing the passage). It optionally checks if the line is of the form `pid passage rest`; if so, the first string in `rest` is considered to be the `title` of the passage, and the passage is stored as `title + '|' + passage`. The final list of collections is then returned.

