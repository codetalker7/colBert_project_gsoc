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

Now, we'll discuss how the kmeans algorithm computes the centroids for the clusters. Suppose `sample` is a tensor containing all our saved sample embeddings (from the previous step). We first split `sample` into two sets: a smaller set to train kmeans on, and a `heldout` set. For this, we'll randomly pick a `heldout_fraction` fraction of `sample`, and put it in a tensor called `heldout`. For instance, we can do this with a call like this:

```
sample, heldout = split(sample, heldout_fraction=0.05)      # leaving about 5% of the sampled embeddings in the heldout set
```

Now, `sample` is the set of embeddings on which we'll run the kmeans algorithm. The clustering algorithm we'll use can either be from the `faiss` module in Python (`faiss.Kmeans`) (in which case we'll might have to use the [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) package), or we can experiment with using some clustering algorithm in the [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl) package. As mentioned before, the number of dimensions will be `dim`, number of clusters `num_partitions`, and the number of iterations for kmeans `kmeans_niters` (all these properties will be part of the configuration type, say `ColBERTConfig`). 

After running kmeans, we'll have the centroids, say in the `centroids` tensor. This tensor will be of shape `(num_partitions, dim)`. Once we've obtained the `centroids`, we'll normalize them.

#### The `ResidualCodec` type

Before describing the further computations done with the centroids, we'll now describe the `ResidualCodec` type that we'll need. This type will be responsible for all the compression/decompression related tasks. Roughly, this type will have the following definition:

```julia
struct ResidualCodec
    centroids::Array{Float32}                   # the tensor containing centroids
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

1. `load`: This method simply loads a `ResidualCodec` into memory. See the `save` function below for the format in which the codec is saved.

2. `save`: This method saves the `ResidualCodec`. `centroids` are saved in a separate file (eg., `centroids.pt` in case of a `torch` tensor); the `avg_residual` is saved in it's own file (eg., `avg_residual.pt`). And the `bucket_cutoffs` and `bucket_weights` are saved in a separate file (eg., `buckets.pt`).

3. `compress`: This function takes a tensor `embs` of all the embeddings, and compresses them. Compressing the embeddings involves two steps: first, for each embedding, the nearest centroid ID is computed (where *nearest* means the centroid with the maximum inner product with the embedding. See the `compress_into_codes` function below). So suppose, `emb` is the embedding, and `centroid` is the centroid which is closest to this embedding. The residual is simply `emb - centroid`. The next step of the compression involves compressing the `residual` into `nbits` bits; this is done via the `binarize` function. Finally, the compression is just the tuple `codes, residuals_packed`, where `codes` is a tensor containing the nearest centroid IDs for each embedding, and `residuals_packed` is a tensor containing the compressed residuals. [This file](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/indexing/codecs/residual.py#L167) contains the Python implementation.

4. `binarize`: This function is mainly responsible for compressing residuals. It takes a tensor `residuals` containing all the residual embeddings, and returns a tensor of type `UInt8` containing all the compressed residuals. See the [Python implementation](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/indexing/codecs/residual.py#L186) for example.

5. `compress_into_codes`: This function simply takes a tensor `embs` of embeddings, and for each embedding, computes the ID of the nearest centroid (where nearest means the centroid having the maximum inner product). The computation is done in batches for higher efficiency. See the [Python implementation](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/indexing/codecs/residual.py#L204).

6. `lookup_centroids`: This function simply takes a list of centroid IDs, and returns a tensor containing the corresponding `centroids`. For instance, look at the [Python implementation](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/indexing/codecs/residual.py#L222).


#### Computing average residuals

Once the `centroids` are computed, we then compute three quantities: `bucket_cutoffs`, `bucket_weights` and the `avg_residual`. The `bucket_cutoffs` and `bucket_weights` are used in the compression/decompression of the residuals. All these quantities are computed using the `heldout` tensor we computed in the previous steps. Here is how the average residuals are computed:

1. For each embedding in `heldout`, we first compute it's nearest centroid ID. This step can be easily done with the following call:
    ```julia
    nearest_ids = compress_into_codes(heldout)      # 1D tensor containing the nearest centroid IDs
    ```

2. Then, we simply convert these IDs to a tensor containing all the corresponding centroids. This is done using the following call:
    ```julia
    nearest_centroids = lookup_centroids(nearest_ids)
    ```

3. Next, the residual embeddings are just the tensor `heldout_residual = heldout - nearest_centroids`.

4. Next, to compute the average residual tensor, we take the average of `abs(heldout_residual)` over each dimension. Suppose the resultant 1D tensor is `avg_residual_tensor`. The `avg_residual` is then just the `mean(avg_residual_tensor)`.

Next, we describe how `bucket_cutoffs` and `bucket_weights` will be computed.

1. First, we define some quantiles as follows:
    ```
    quantiles = (Vector(0:2^nbits - 1)) ./ (2^nbits)
    bucket_cutoffs_quantiles = quantiles[1:2^nbits - 1]
    bucket_weights_quantiles = quantiles .+ (0.5 / 2^nbits)
    ```

2. Then, using `bucket_cutoffs_quantiles` and `bucket_weights_quantiles`, we compute the corresponding quantiles of the `heldout_residual` tensor.

Once the average residual data is computed, we save this data in the format mentioned in the previous section.

### Encoding all passages and saving chunks

At this point, we've computed all the centroids, and all the necessary data required for compression/decompression. All of this data has been stored in the `ResidualCodec` type. Now we discuss the next step of indexing: converting all documents to embeddings and storing their compressions in chunks. This process can be sped up using multithreading, but here we just discuss a single-threaded solution (which is easily extensible to a multi-threaded solution).

First, we batch all of our passages into batches of size given by `get_chunksize(collection)` (see previous sections). A batch will just be a tuple `(chunk_idx, offset, passages)`, where `chunk_idx` is the index of the batch, `offset` is the index of the *first* passage in the batch, and `passages` is a list containing all the passages in the batch.

We then iterate over all these batches. In each iteration, we compute the `embs, doclens` for each batch (using the `encode_passages` function). Having all the `embs`, we then compute all the nearest centroid IDs for each embedding (using the `compress_into_codes` function); suppose these IDs are stored in the list `codes_`. Then, using `lookup_centroids`, we get all the corresponding centroids, say in the tensor `centroids_`. The residuals for this batch will then be given by `residuals_ = embs_ - centroids_`. The compressed residuals will then be just `binarize(residuals_)`.

Finally, we save all this data to disk. The `codes_` are saved in a file named `{chunk_idx}.codes.pt` (for the case of `torch` tensors); the `residuals_` are saved in a file called `{chunk_idx}.residuals.pt` (again, for `torch` tensors). `doclens` are saved in a file called `doclens.{chunk_idx}.json`. Finally, some metadata is stored in a file called `{chunk_idx}.metadata.json`; this metadata includes the `passage_offset` (the `offset` of the chunk), `num_passages` (number of passages in the chunk), and `num_embeddings` (number of embeddings in the chunk).

### Creating the optimized `ivf`

We'll now desribe the final steps of the indexing process. First, we'll compute the `embedding_offsets` property of our `CollectionIndexer`. As mentioned before, `embedding_offsets` is just a vector, whose length is equal to `num_chunks`, which stores, for each chunk ID, the index of the *first* embedding which is contained inside that chunk. This is easy to do, since, for each chunk, we've already stored the `num_passages` and the `num_embeddings` in it's metadata file. During this process, we also compute the total number of embeddings stored across all the chunks, and save it in the `num_embeddings` property of the `CollectionIndexer`.

Next, we'll compute the `ivf`. The `ivf` is simply a list which stores, for each centroid ID, a list of all the `pid`s of passages such that the passage has an embedding of which the closest centroid ID is this ID (in other words, for each centroid ID, store all the `pid`s that it *contains*). For our purposes, we will compute and save the following data:

1. `ivf`, a vector of type `Vector{Int}`. This vector will consecutively store all the unique `pid`s contained in some centroid ID. For instance, if there are two centroids with IDs `0` and `1`, and centroid `0` contains pids `[100, 2, 3]`, and centroids `1` contains pids `[2, 3, 4, 5]`, then `ivf` will just be the list `[100, 2, , 2, 3 ,4 ,5]`.

2. `ivf_lengths`: This is a list containing the number of unique pids stored in each centroid ID. For the above example, `ivf_lengths` will be the list `[3, 4]`.

Once these are computed, we store these two lists on disk, in a file called `ivf.pid.pt` (say, if these are `torch` tensors).

### Finishing up

As a final step, we save the `config`, `num_chunks`, `num_partitions`, `num_embeddings` and `avg_doclen` fields of our `CollectionIndexer` in a file called `metadata.json`, whose path is the `metadata_path` property of the `CollectionIndexer`. This finishes up the indexing phase.

## Searching

Next, we'll describe the implementation of the searching phase. Analogous to the indexing phase, we propose a type called `Searcher`, which will be responsible for searching the most relevant passages corresponding to a query text. Roughly, the `Searcher` will have the following structure:

```julia
struct Searcher
    index::String                       # path of the index
    index_config::ColBERTConfig         # the configuration used by the indexer
    collection::Collection              # the underlying collection
    ranker::IndexScorer                 # object used to rank passages
end
```

As before, we might add more fields to the `Searcher` type if necessary. The only new field above is the `IndexScorer`, which we'll describe in just a bit. For our use case, a `Searcher` will support the following three methods:

```julia
encode(searcher::Searcher, query::String)
search(searcher::Searcher, query::String, k::Int)
search_all(searcher::Searcher, queries::Vector{Int}, k::Int)
```

Here is a short description of the above functions:

1. `encode` simply takes up a query string, and applies the underlying BERT model to it to get the corresponding embeddings (this involves other steps too, like adding the query token `[Q]` to the beginning of the query, and padding the query with mask tokens if needed). So, the output of `encode` is a tensor (a 2D tensor) containing the embeddings for all the tokens in the query.

2. `search` takes a query, and returns a list of `k` tuples (`k` is an argument to `search`), where each tuple is of the form `(passage_id, passage_rank, passage_score)`. The top `k`-passages relevant to this query are returned, where `passage_rank` denotes the rank of the passage, and `passage_score` denotes the score of the passage against the query (we'll describe how these are computed in a bit).

3. `search_all` is similar to `search`, but it takes a batch of queries to do work on.

Next, we'll describe the `IndexScorer` type in a bit more detail, and also how the `search` function actually works.

### The `IndexScorer` and `dense_search`. 

The `Searcher` type is just a wrapper around the `IndexScorer` type, which actually does most of the searching (both types are inspired from the corresponding classes, [`Searcher`](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/searcher.py#L22) and [`IndexScorer`](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/search/index_storage.py#L20), in the Python implementation)

The `Searcher` will have support for a function called `dense_search`, which has the following template:

```
dense_search(Q, k)
```

Above, `Q` is the tensor containing all the query embeddings, and `k` is the number of passages to be retrieved. Depending on the value of `k`, the `dense_search` methods sets a bunch of configuration variables to their appropriate values, like `ncells`, `centroid_score_threshold` and `ndocs` (the Python implementation of [`dense_search`](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/searcher.py#L106) contains good defaults for these parameters, depending on the range of values `k` belongs to). Setting all these defaults, `dense_search` proceeds to compute the top `k` most relevant passages and scores as follows:

1. First, for each query embedding, the `ncells` closest centroids are computed (where closest just means the centroid with the maximum inner product with the query embedding). Suppose all these centroids are stored in a tensor called `cells`. For simplicity, we can consider the case of `ncells = 1`, in which case `cells` will just be a 1D tensor. Also, a tensor `centroid_scores` is computed, where the shape of the tensor is `(num_partitions, num_query_embeddings)`, where `num_partitions` is the number of centroids `num_query_embeddings` is the number of embeddings in the query. This tensor will just contain all the inner products of all possible combinations of centroids and query embeddings.

2. Then, a list called `pids` is computed, which is the list of all the `pid`s contained inside the centroids in `cells`.

3. Each of these `pids` then is scored to first obtain approximate scores against the query, and the top `ndocs` of these `pids` is chosen as the candidate set of passages to return. Note that these approximate scores are obtained by first using only the pruned `centroid_scores`, and then the approximate `centroid_scores` (and not the residuals; this is the algorithm used in the [PLAID](https://arxiv.org/pdf/2205.09707.pdf) paper). Then the top `ndocs/4` pids (in terms of the approximate centroid scores) are taken as the candidate set.

4. The final list of candidate `pids` is then scored using both the centroids and the residuals. More implementation details of steps 3 and 4 are contained in the [`score_pids`](https://github.com/stanford-futuredata/ColBERT/blob/852271661b22567e3720f2dd56b6d503613a3228/colbert/search/index_storage.py#L111) function of the Python implementation.

Finally, among all these `pids`, the top `k` are chosen and returned along with their computed scores. 
