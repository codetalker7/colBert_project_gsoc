# ColBERTv2.0 in Julia (GSoC Proposal)

This is a proposal for a GenAI GSoC project in Julia. In this project, the goal will be to implement the [ColBERTv2.0](https://github.com/stanford-futuredata/ColBERT) neural search system in Julia. The main design inspiration of the implementation is from ColBERT's original implementation (as in the linked repository). The two key compnents of the system will be the **indexer** and the **searcher** (defined by the [`Indexer`](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/indexer.py#L15) and [`Searcher`](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/searcher.py#L22) classes in the corresponding python implementation). We now go into the details of the design.

## A small note about distributed training

The current [ColBERTv2.0](https://github.com/stanford-futuredata/ColBERT) implementation uses the `torch.multiprocessing` module for parallelizing various aspects of indexing a corpus of passages. In particular, the property [`nranks`](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/infra/config/settings.py#L27) of the `RunConfig` class controls the number of GPUs needed for indexing. In the first version of this GSoC project, we will assume that `nranks = 1` (i.e we'll use only one GPU for indexing and other operations). Once this is implemented, we can potentially use the [DaggerFlux.jl](https://github.com/FluxML/DaggerFlux.jl) package to implement distributed indexing.

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

In any case, for encoding both the queries and the documents, we will use a pretrained checkpoint.

## The LoTTE dataset and it's format
As a demonstration example, we'll use the `lifestyle` topic of the LoTTE dataset (released by the ColBERT authors).

```shell
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz -P downloads/
tar -xvzf downloads/lotte.tar.gz -C downloads/
```
The full path of the dataset can be obtained as follows:

```julia
dataroot = "download/lotte"
dataset = "lifestyle"
datasplit = "dev"

queries_path = joinpath(dataroot, dataset, datasplit, "questions.search.tsv")
collection_path = joinpath(dataroot, dataset, datasplit, "collection.tsv")
```

We'll assume the standard ColBERT dataset format. Specifically, we'll have two datasets: one for the query passages, and one for the document passages. We'll assume that both the datasets are in `.tsv` format, where each line of the query dataset is of the form `qid <tab> query_passage` (where `qid` refers to the unique ID of the query passage), and each line of the passage dataset is of the form `pid <tab> document_passage` (where `pid` refers to the unique ID of the document passage).

## The `Queries` and `Collection` types

Inspired from ColBERT's python implementation, we'll create types for efficient handling of query and document datasets. For this, we propose the types `Queries` and `Collection` for this. Roughly, the definitions of the types will go as follows:

```julia
Base.@kwdef struct Queries
    provenance::String      # path of origin of the dataset
    data::Vector{String}    # query passages
end

Base.@kwdef struct Collection
    provenance::String
    data::Vector{String}    # document passages
end
```

Both the `Queries` and `Collection` types will have standard constructors to load the datasets from the given paths. In addition, we need the following methods for a `Collection`:

```julia
enumerate_batches(collection::Collection, chunksize::UInt=nothing)
enumerate(collection::Collection)
get_chunksize(collection::Collection)
```
Here is the rough explaination of these functions:

- The `get_chunksize` function, given a `Collection`, returns the appropriate size of the chunks used for indexing the collection (please see the details about the `Indexer` to see how chunk sizes are used). This number is calculated as `min(25000, 1 + length / nranks)`, where `length` is the length of the `Collection` (the number of documents), and `nranks` is the number of GPUs to be used (which, in the initial version of the package, will be `1`). 

- The `enumerate_batches` function batches the dataset into chunks of size `chunksize` (which is calculated via the `get_chunksize` function). Each batch is of the form `(chunk_idx, offset, passage)`; here `chunk_idx` is the index of the chunk, `offset` is the `pid` of the *first* passage in the chunk, and `passage` is a list of passages contained in the chunk.

- The `enumerate` function just creates an enumeration over the passages, it's is similar to the standard `enumerate` function.

Note that, the behaviours of all these methods will slightly change when `nranks > 1`, i.e when we use multiple GPUs. This is because the logic to split the batches among different GPUs will be slightly more involved.

In addition to this, we'll need some more standard operations on `Queries` and `Collection` objects, such as getting the length of the underlying vector, loading/saving to disk etc.

## Configuration

[This](https://github.com/stanford-futuredata/ColBERT) implementation of ColBERT handles two kinds of configuration: the `RunConfig` and the `ColBERTConfig`. `RunConfig` is used to set up the environment under which the code will run (in Python terminology, it is essentially a context manager). `ColBERTConfig` is a more detailed config; it includes all the paramters in the ColBERT model and all other minor configuration settings that are needed for indexing/searching/training. For now, our focus will only be on the indexing and searching parts of the `ColBERTConfig`. For the initial version of our package, we will focus on the following configuration parameters:

- `root::String`: The path prefix wherein all files will be stored.
- `experiment::String`: The name of the experiment. Essentially an add-on to `root`.
- `index_root::String`: The root path of the index, relative to `root` and `experiment`. This is where all index-related files will be stored.
- `rank`: The GPU ID. For the initial version, we'll assume this to be `0`.
- `nranks`: The total number of GPUs to be used. For the intial version, this will be `1`.
- `query_token_id::String`: The token ID for the query token (which is appended at the beginning of each query before passing through the BERT encoder).
- `doc_token_id::String`: The token ID for the document token (which is appended at the beginning of each document before passing through the BERT encoder).
- `query_token::String`: The query token. Defaults to `"[Q]"`.
- `doc_token::String`: The document token. Defaults to `"[D]"`.
- `checkpoint::String`: The pretrained checkpoint to be used.
- `collection::Union{String, Collection}`: The underlying document dataset, or a path to it. 
- `queries::{String, Collection}`: The underlying queries dataset, or a path to it.
- `index_name::String`: The name of the folder to store the index in.
- `dim::Int`: The embedding dimension.
- `doc_maxlen::Int`: Maximum allowed value of document length (number of tokens in a document). Documents longer than this are truncated.
- `mask_punctuation::Bool`: Whether to mask punctuation tokens.
- `query_maxlen::Int`: Similar to `doc_maxlen`, but for queries.
- `attent_to_mask_tokens::Bool`: Whether to attend to mask tokens inside the query.
- `interaction::String`: Interaction style. Defaults to `"colbert"`.
- `index_path::String`: Path where the indexing files will be saved.
- `nbits::UInt`: Number of bits used to encode the residuals.
- `kmeans_niters::UInt`: Number of iterations of the kmeans algorithm for computing the centroids of the clusters.

Just like in ColBERT's Python implementation, we propose splitting these training properties into different types. For example, properties determinining the run-time environment will go into a type called `RunSettings`; those pertaining to tokenization will go into a type called `TokenizerSettings`, etc. The [`settings.py`](https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/infra/config/settings.py) file from the Python implementation specifies how these config settings could be split.

Finally, we'll also have to implement functionality to load/save these config settings in an appropriate data format.

## Indexing

We'll now describe how indexing is going to be implemented. Just like in the Python implementation, we propose a `CollectionIndexer` type (which essentially represents an indexer). Roughly, the `CollectionIndexer` type will have a similar definition as to the one below:

```julia
struct CollectionIndexer
    config::ColBERTConfig
    collection::Collection
    encoder::CollectionEncoder          # the underlying encoder model
    saver::IndexSaver                   # the underlying index saver
    num_chunks::UInt                    # number of chunks into which the documents will be indexed
    num_sample_embs::UInt               # the number of embeddings in the sampled passages (used to compute the centroids)
    avg_doclen_est::Float               # the average document length, computed over the sampled passages
    num_embeddings_est::Float           # estimated number of embeddings over all the passages
    num_partitions::UInt                # the number of clusters into which the embeddings will be clustered
    plan_path::String                   # path of the file where the indexing plan will be saved. This will be a json file
    embedding_offsets::Vector{UInt}     # list containing the offsets of the embeddings which are saved in chunks. The length of this vector is equal to the number of chunks.
    num_embeddings::UInt                # total number of embeddings saved
    metadata_path::String               # path of the file where metadata is saved.
end
```
More fields could be added to `CollectionIndex` as they are needed. As mentioned in the above definition, `CollectionEncoder` (the encoder field) is a type representing the underlying encoder model (which will be used to convert passages to encodings). Any `CollectionEncoder` object will have just two fields, namely the `checkpoint` to be used for encoding, and a `config` (i.e the `ColbertConfig` to be used). Also, we'll define the following method on a `CollectionEncoder`:

```julia
encode_passages(encoder::CollectionEncoder, passages::Vector{String})
```

The `encoder_passages` method will take an `encoder` and a list `passages` containing the passages to be encoded. It will return a tuple `(embs, doclens)`, where `embs` will be a tensor of shape `(num_embeddings, dim)`, where `num_embeddings` is the total number of embeddings (tokens) over all the `passages`, and `dim` is the embedding dimension. `doclens` will be a list containing the document lenghts (number of tokens in a document) over all the documents in the `passages` list. This will be a list of integers representing the lengths. Both the `embs` and teh `doclens` will be computed by the underlying BERT model.

Before describing the `IndexSaver` type, we'll now describe the exact process of computing the centroids for the clusters, and other minor details.

### Chunks

As we mentioned before, the index will be stored in *chunks*, i.e in batches. The `chunksize` will be obtained via the `get_chunksize(::Collection)` function, as mentioned before. The idea is then to split the document passages into chunks of this size; to that end, the `num_chunks` property of the `CollectionEncoder` object will be calculated as follows:

```julia
num_chunks = ceil(length(collection) / collection.get_chunksize())
```
Above, `collection` is the underlying `Collection` of document passages. As we will see later in this proposal, the compressed embeddings will be stored in a total of `num_chunks` chunks.

### Clustering

#### Sampling pids and calculating the number of clusters

Let's now discuss how clustering is done. The idea is to first randomly sample a bunch of passages using which we'll initialize the clustering algorithm. To do this, we first randomly sample a total of `sampled_pids` passages from the underlyging `Collection`, where `sampled_pids` is calculated as follows:

```
num_passages = length(collection)       # collection::Collection is the underlying collection
typical_doclen = 120                    # typical size of a document
num_sampled_pids = 16 * sqrt(typical_doclen * num_passages)
num_sampled_pids = min(1 + floor(num_sampled_pids), num_passages)
```

In other words, the number of `pid`s we sample is roughly proportional to the `sqrt` of the total number of embeddings in the dataset. So, assume that we have randomly sampled a total of `num_sampled_pids` number of `pid`s, and suppose all these sample `pid`s are in the set `sampled_pids`.

Next, we'll take a subset of `collection`, such that the subset contains only those passage whose `pid` is in the set `sampled_pids`. Let's assume that this subset is represented by `local_sample`, where `local_sample` has type `Vector{String}` (i.e it's a list containing all the sampled passages).

Next, we'll make the following call:

```julia
local_sample_embs, doclens = encode_passages(encoder, local_sample)
```

Above, `encoder` is the underlying encoder model. So now, for all the sampled passages, we've obtained the embeddings (stored in `local_sample_embs`) and the document lengths (stored in `doclens`).

Using this information we'll calculate the following:

- We'll set the `num_sample_embs` property of our `CollectionIndexer` to just be the number of embeddings in the `local_sample_embs` tensor. 
- The `avg_doclen_est` property of our `CollectionIndexer` will just be the mean of all document lengths in `doclens`. 
- The `num_embeddings_est` property of `CollectionIndexer`, which is the estimated number of embeddings, will be calculated as the product `len(collection) * avg_doclen_est`.
- Finally, `num_partitions`, which is the number of centroids into which we'll cluster the data, will be calculated using the following formula:
    ```julia
    num_partitions = floor(2^(floor(log(16 * sqrt(num_embeddings_est)))))
    ```
    In other words, the total number of clusters is proportional to the `sqrt` of the estimated total number of embeddings.

#### Running kmeans and getting the cluster centroids

Now, we'll discuss how the kmeans algorithm computes the centroids for the clusters. Suppose `sample` is a tensor containing all our saved sample embedings (from the previous step). We first split `sample` into two sets: a smaller set to train kmeans on, and a `heldout` set. For this, we'll randomly pick a `heldout_fraction` fraction of `sample`, and put it in a tensor called `heldout`. For instance, we can do this with a call like this:

```
sample, heldout = split(sample, heldout_fraction=0.05)      # leaving about 5% of the sampled embeddings in the heldout set
```

Now, `sample` is the set of embeddings on which we'll run the kmeans algorithm. The clustering algorithm we'll use can either be from the `faiss` module in Python (`faiss.Kmeans`) (in which case we'll might have to use the [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) package), or we can experiment with using some clustering algorithm in the [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl) package. As mentioned before, the number of dimensions will be `dim`, number of clusters `num_partitions`, and the number of iterations for kmeans `kmeans_niters` (all these properties will be part of the configuration type, say `ColBERTConfig`). 

After running kmeans, we'll have the centroids, say in the `centroids` tensor. This tensor will be of shape `(num_partitions, dim)`. Once we've obtained the `centroids`, we'll normalize them.

#### The `ResidualCodec` type

Before describing the further computations done with the centroids, we'll now describe the `ResidualCodec` type that we'll need. This type will be responsible for all the compression/decompression related tasks. Roughly, this type will have the following definition:

```julia
struct ResidualCodec
    centroids::Array{Float32}                     # the tensor containing centroids
    dim::Int                                    # the embedding dimension
    nbits::Int                                  # number of bits into which the residuals are to be compressed
    avg_residual::Float                         # the average residual
    bucket_cutoffs::Vector{Float32}             # needed in compression/decompression
    bucket_weights::Vector{Float32}             # needed in compression/decompression
end
```

A few more fields may need to be added to the `ResidualCodec` for compression/decompression of residuals. The main job of the `ResidualCodec` will be via the following methods:

```julia
load(index_path::String)
save(codec::ResidualCodec, index_path::String)
compress(codec::ResidualCodec, embs::Array{Float32})
binarize(codec::ResidualCodec, residuals::Array{Float32})
compress_into_codes(codec::ResidualCodec, embs::Array{Float32})
lookup_centroids(codec::ResidualCodec, codes::Vector{Int})
```
We'll now describe how each of these methods work.

1. `load`: This method simply loads a `ResidualCodec` into memory. See the `save` function below for format in which the codec is saved.

2. `save`: This method saves the residual codec. `centroids` are saved in a separate file (eg., `centroids.pt` in case of a `torch` tensor); the `avg_residual` is saved in it's own file (eg, `avg_residual.pt`). And the `bucket_cutoffs` and `bucket_weights` are saved in a separate file (eg., `buckets.pt`).

3. `compress`: This function takes a tensor `embs` of all the embeddings, and compresses them. Compressing the embeddings involves two steps: first, for each embedding, the nearest centroid ID is computed (where *nearest* means the centroid with the maximum inner product with the embedding. See the `compress_into_codes` function below). So suppose, `emb` is the embedding, and `centroid` is the centroid which is closest to this embedding. The residual is simply `emb - centroid`. The next step of the compression involves compressing the `residual` into `nbits` bits; this is done via the `binarize` function. Finally, the compression is just the tuple `codes, residuals_packed`, where `codes` is a tensor containing the nearest centroid IDs for each embedding, and `residuals_packed` is a tensor containing the compressed residuals. [This file](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/indexing/codecs/residual.py#L167) contains the python implementation.

4. `binarize`: This function is mainly responsible for compressing residuals. It takes a tensor `residuals` containing all the residual embeddings, and returns a tensor of type `UInt8` containing all the compressed residuals. See the [Python implementation](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/indexing/codecs/residual.py#L186) for example.
5. `compress_into_codes`: This function simply takes a tensor `embs` of embeddings, and for each embedding, computes the ID of the nearest centroid (where nearest means the centroid having the maximum inner product). The computation is done in batches for higher efficiency. See the [Python implementation](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/indexing/codecs/residual.py#L167).

6. `lookup_centroids`: This function simply takes a list of centroid IDs, and returns a tensor containing the corresponding `centroids`.


#### Computing average residuals

Once the `centroids` are computed, we then compute three quantities: `bucket_cutoffs`, `bucket_weights` and the `avg_residual`. The `bucket_cutoffs` and `bucket_weights` are used in the compression/decompression of the residuals.
