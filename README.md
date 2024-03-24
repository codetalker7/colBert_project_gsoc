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

## The `RunConfig` object

The config is set up using a [*context manager*](https://realpython.com/python-with-statement/). The `Run` class (in the `infra.run` module) is a *Singleton Pattern* (i.e a class such that only one instance of the class exists at a given time). It has a class variable called `_instance` which is the only instance (if any) of the class which exists at a time. This instance has a property called `stack` to which `RunConfig`s are appended (this stack is just a list, do `Run()._instance.stack`). Default values to the `RunConfig` are assigned using the `assign_defaults` function in the `infra.config.core_config` module.

Context management is done using `contextlib`'s `@contextmanager` decorator (see [this link](https://realpython.com/python-with-statement/)). Here is how context management is done in detail:
    - A `RunConfig` object is passed to the `Run.context` function. If nothing is passed, a default `RunConfig` object is created and is stored in the `stack`. If something is passed, it is temporarily pushed to the stack, and the topmost config is used as the context for the code to run in. Finally, as the code leaves the context (i.e the `with` block is finished), the `RunConfig` is popped from the stack.


## The `ColBERTConfig` object

This object specifies the config for the ColBERT model. All of the config information is in the `infra` package. Here is a short description of the `infra.config` subpackage.
    - A `dataclass` called `DefaultVal`, which is a wrapper around any datatype. Provides functions to `__hash__` and `__eq__`, to hash the object and check for equality.
    - A `dataclass` called `CoreConfig`. This class is presumably intended to assign default values, set the configuration, and a `__post_init__` function which sets default values for keys whose values are not passed to the constructor. This is essentially a class containing a few helper functions.
    - A `dataclass` called `BaseConfig`, which inherits from `CoreConfig`. This has methods to load configs from existing ones, from checkpoints etc.
    - In the `infra.config.settings` module, we have all classes which store the actual configuration. Need to play with these to see what they are. All these are `dataclasses`.
    - The `ColBERTConfig` class inherits from all dataclasses in `infra.config.settings`, and also `BaseConfig`. `RunConfig` inherits from `BaseConfig` and `RunSettings`.

## The `Indexer` class in detail.

The constructor of this class just initializes the `config` of the indexer. The main thing in this class is the `index` function. This function just sets the configuration, the `index_path`. It checks if the `index_path` exists; for our use case, it doesn't. If it doesn't, it creates a new directory with the `index_path`. It then calls the `__launch` function, and finally returns the `index_path`.

The `__launch` function creates a `Launcher` object with the `encode` function from the `colbert.indexing.collection_indexer` module. For our purposes, we will work with `rank = 0` and `nranks = 1` (we'll deal with distributed training later). So, for this case, `shared_lists` will just contain a single list, and `shared_queues` will also be a list containing a single queue. Finally, the `launch` function of the `Launcher` object is called.

For our purposes however, we will run the code with `avoid_fork_if_possible=True` (this is to run the code without `mp`). For this, we just need to study the `launch_without_fork` function call.


The `launch` function is just running the `encode` function (for the most part).

## The `CollectionIndexer`

The constructor just sets up some boilerplate properties like `config`, `rank` and `nranks`. Remember, for our case, we assume `rank = 0` and `nranks = 1`. It is at this step that `self.checkpoint` is initialized (this is the pretrained model), moved to the GPU if necessary. Also, `self.encoder` is set to be a `CollectionEncoder`, using `self.checkpoint` as it's base model. `self.saver` is also set to an `IndexSaver`.

Again, here the main meat is in the `run` function of the indexer. The `run` function delegates it's job into four main functions: `self.setup` (computes the plan for the whole collection), `self.train` (computes the centroids from the selected passages), `self.index` (encodes and saves all tokens into residuals) and `self.finalize` (builds the metadata and the centroid-to-passage mapping).

`self.setup` calculates a plan and saves the plan in `plan.json`. It contains the following values:
    - `num_chunks = np.ceil(len(self.collection)) / self.collection.get_chunksize())`. Here, `get_chunksize` just returns `min(25000, 1 + len(collection) // Run().nranks).`
    - `num_partitions`, the number of centroids to be generated. Let's now describe how this is calculated. 
        - First, a bunch of `pids` are sampled, and stored in `sampled_pids`. The number of sampled pids is set to `16 * np.sqrt(typical_doclen * num_passages)`, where `typical_doclen = 120`. Finally, a minimum of `1` plus this number and `num_passages` is taken, and this minimum is the number of pids we sample. 
        - Next, an average of the document lengths is computed, and stored into `avg_doclen_est`. To see how this is calculated, we first need to understand the `_sample_embeddings` function. This function first calculates `self.collection.enumerate(rank=self.rank)`; for our case (with `rank = 0` and `nranks = 1`), this function just iterates over batches of `collection` of size `chunksize`, where a batch is identified by an `offset` and a list of size `chunksize`. The `offset` is just the index of the first passage in the batch. For each such batch, `enumerate` returns a tuple containing the actual index of the passage (`offset + idx` in the code) and the passage itself. This is just a very convoluted way of doing `enumerate(self.collection)`, but this becomes interesting for the case when `nranks > 1` (i.e distributed training). Then, `local_sample` refers to all the passages with `pid` in the `sampled_pids` set. For all these passages, their embeddings and doclens are computed via the `encode_passages` function of the encoder (which is a `Checkpoint`). Once these are computed and stored in `local_sample_embs` and `doclens` respectively, the average doclens is just comptuted as `sum(doclens) / len(doclens)`. Finally, all the computed embeddings are stored in half-precision (see `torch.half`) in the file named `sample.rank.pt` in the `index_path`.
        - Finally, `num_partitions` is chosen by the formula `int(2 ** np.floor(np.log2(16 * np.sqrt(self.num_embeddings_est))))`, where `self.num_embeddings_est = num_passages * avg_doclen_est`. This is the number of clusters which are picked.
        - Once this is done, all these values are saved in `plan.json`.

Next, we'll study the function `self.train` (which does the kmeans clustering).
    - `_concatenate_and_split_sample`: First, an empty `torch` tensor of the appropriate dimenion is allocated (this tensor will hold the embeddings that were saved in `sample.rank.pt`, with `rank = 0`).
