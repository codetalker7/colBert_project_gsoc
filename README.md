# ColBERTv2.0 in Julia (GSoC Proposal)

This is a proposal for a GenAI GSoC project in Julia. In this project, the goal will be to implement the [ColBERTv2.0](https://github.com/stanford-futuredata/ColBERT) neural search system in Julia. The main design inspiration of the implementation is from ColBERT's original implementation (as in the linked repository). The two key compnents of the system will be the **indexer** and the **searcher** (defined by the [`Indexer`](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/indexer.py#L15) and [`Searcher`](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/searcher.py#L22) classes in the corresponding python implementation). We now go into the details of the design.

## A small note about distributed training

The current [ColBERTv2.0](https://github.com/stanford-futuredata/ColBERT) implementation uses the `torch.multiprocessing` module for parallelizing various aspects of indexing a corpus of passages. In particular, the property [`nranks`](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/infra/config/settings.py#L27) of the `RunConfig` class controls the number of GPUs needed for indexing. In the first version of this GSoC project, we will assume that `nranks = 1` (i.e we'll use only one GPU for training the model). Once this is implemented, we can potentially use the [DaggerFlux.jl](https://github.com/FluxML/DaggerFlux.jl) package to implement distributed indexing.

## The pretrained checkpoint

We'll use the HuggingFace API from [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) to load the `colbert-ir/colbertv2.0` pretrained checkpoint released by the authors of ColBERT. For example, the following simple code snippet does the job:

```julia
using Flux, Transformers, CUDA

const PRETRAINED_MODEL = "colbert-ir/colbertv2.0"

colbert_config = Transformers.load_config(PRETRAINED_MODEL)
colbert_tokenizer = Transformers.load_tokenizer(PRETRAINED_MODEL)
colbert_model = Transformers.load_model(PRETRAINED_MODEL)
```

Alternalively, we can also directly download the checkpoint and load it via [Transformers.jl](https://github.com/chengchingwen/Transformers.jl):

```shell
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz -P downloads/
tar -xvzf downloads/lotte.tar.gz -C downloads/
```

## The dataset format

Initially, we'll assume the standard ColBERT dataset format (support for other dataset formats will be added after the initial version of the package). Specifically, we'll have a dataset 
