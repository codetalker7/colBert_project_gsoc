import os
import sys
import pdb
sys.path.insert(0, '../')
import random
import time
import ujson

import numpy as np
import torch

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from colbert.indexing.collection_indexer import encode
from colbert.utils.utils import create_directory, print_message, flatten
from colbert.infra.launcher import Launcher
from colbert.infra.config import BaseConfig, RunConfig, RunSettings
from colbert.indexing.collection_indexer import CollectionIndexer
from colbert.infra.launcher import print_memory_stats
import faiss
from colbert.indexing.codecs.residual import ResidualCodec
import tqdm
from colbert.indexing.loaders import load_doclens
import re
from colbert.search.strided_tensor import StridedTensor
from colbert.modeling.checkpoint import Checkpoint
from colbert.search.index_storage import IndexScorer
from colbert.search.strided_tensor import StridedTensor
from colbert.search.candidate_generation import CandidateGeneration

from colbert.search.index_storage import IndexLoader

from typing import Union
TextQueries = Union[str, 'list[str]', 'dict[int, str]', Queries]

dataroot = 'downloads/lotte'
dataset = 'lifestyle'
datasplit = 'dev'

queries = os.path.join(dataroot, dataset, datasplit, 'questions.search.tsv')
queries = Queries(path=queries)

f'Loaded {len(queries)} queries'

nbits = 2
index_name = f'{dataset}.{datasplit}.{nbits}bits'

runconfig = RunConfig(experiment='notebook')
contextmanager = Run().context(runconfig)
contextmanager.__enter__()

# searcher = pdb.runcall(Searcher, index=index_name) 
searcher = Searcher(index=index_name)
# recreating the Searcher constructor call
index = index_name
checkpoint = None
collection = None
config = None
verbose = 3

searcher.verbose = verbose
if searcher.verbose > 1:
    print_memory_stats()

initial_config = ColBERTConfig.from_existing(config, Run().config)

default_index_root = initial_config.index_root_
index_root = index_root if index_root else default_index_root
searcher.index = os.path.join(index_root, index)
searcher.index_config = ColBERTConfig.load_from_index(searcher.index)

searcher.checkpoint = checkpoint or searcher.index_config.checkpoint
searcher.checkpoint_config = ColBERTConfig.load_from_checkpoint(searcher.checkpoint)
searcher.config = ColBERTConfig.from_existing(searcher.checkpoint_config, searcher.index_config, initial_config)

searcher.collection = Collection.cast(collection or searcher.config.collection)
searcher.configure(checkpoint=searcher.checkpoint, collection=searcher.collection)

searcher.checkpoint = Checkpoint(searcher.checkpoint, colbert_config=searcher.config, verbose=searcher.verbose)
use_gpu = searcher.config.total_visible_gpus > 0
if use_gpu:
    searcher.checkpoint = searcher.checkpoint.cuda()
load_index_with_mmap = searcher.config.load_index_with_mmap
if load_index_with_mmap and use_gpu:
    raise ValueError(f"Memory-mapped index can only be used with CPU!")
# self.ranker = IndexScorer(self.index, use_gpu, load_index_with_mmap)
## the IndexScorer constructor call
## super().__init__(index_path=index_path, use_gpu=use_gpu, load_index_with_mmap=load_index_with_mmap)
## The IndexLoader constructor call
searcher.ranker.index_path = searcher.index
searcher.ranker.use_gpu = use_gpu
searcher.ranker.load_index_with_mmap = load_index_with_mmap

## self._load_codec()
## the _load_codec() call
searcher.ranker.codec = ResidualCodec.load(searcher.ranker.index_path)
## _load_codec call ends here

## self._load_ivf()
## the load_ivf() call
print_message(f"#> Loading IVF...")
if os.path.exists(os.path.join(searcher.ranker.index_path, "ivf.pid.pt")):
    ivf, ivf_lengths = torch.load(os.path.join(searcher.ranker.index_path, "ivf.pid.pt"), map_location='cpu')
# else:
#     assert os.path.exists(os.path.join(self.index_path, "ivf.pt"))
#     ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pt"), map_location='cpu')
#     ivf, ivf_lengths = optimize_ivf(ivf, ivf_lengths, self.index_path)
assert ivf_lengths.sum().item() == ivf.size()[0]

# ivf = StridedTensor(ivf, ivf_lengths, use_gpu=self.use_gpu)
## the StridedTensor constructor call
## super().__init__(packed_tensor, lengths, dim=dim, use_gpu=use_gpu)
## the StridedTensorCore constructor call
packed_tensor = ivf
lengths = ivf_lengths
dim=None
use_gpu=use_gpu

searcher.ranker.ivf.dim = dim
searcher.ranker.ivf.tensor = packed_tensor
searcher.ranker.ivf.inner_dims = searcher.ranker.ivf.tensor.size()[1:]
searcher.ranker.ivf.use_gpu = use_gpu

searcher.ranker.ivf.lengths = lengths.long() if torch.is_tensor(lengths) else torch.LongTensor(lengths)

## self.strides = _select_strides(self.lengths, [.5, .75, .9, .95]) + [self.lengths.max().item()]
## the _select_strides call
lengths = searcher.ranker.ivf.lengths
quantiles = [0.5, 0.75, 0.9, 0.95] 

# if lengths.size(0) < 5_000:
#     return torch.quantile(lengths.float(), torch.tensor(quantiles, device=lengths.device)).int().tolist()

sample = torch.randint(0, lengths.size(0), size=(2_000,))   # generate 2000 random centroid ids
# return _get_quantiles(lengths[sample], quantiles)
searcher.ranker.ivf.strides = torch.quantile(lengths[sample].float(), torch.tensor(quantiles, device=lengths[sample].device)).int().tolist()

## the _select_strides call ends here
searcher.ranker.ivf.strides = searcher.ranker.ivf.strides + [searcher.ranker.ivf.lengths.max().item()]      # of all the ivf_lengths, take he [0.5, 0.75, 0.9, 0.95] quantiles, and take the maximum length

searcher.ranker.ivf.max_stride = searcher.ranker.ivf.strides[-1]        # the maximum ivf length
zero = torch.zeros(1, dtype=torch.long, device=searcher.ranker.ivf.lengths.device)
searcher.ranker.ivf.offsets = torch.cat((zero, torch.cumsum(searcher.ranker.ivf.lengths, dim=0)))       # essentially, for each centroid id, the first index of the ivf list which belongs to that centroid 
assert searcher.ranker.ivf.offsets.size()[0] == ivf_lengths.size()[0] + 1

if searcher.ranker.ivf.offsets[-2] + searcher.ranker.ivf.max_stride > searcher.ranker.ivf.tensor.size(0): ## if the second last offset plus the max stride is larger than the ivf list size
    print("Condition is true")
    padding = torch.zeros(searcher.ranker.ivf.max_stride, *searcher.ranker.ivf.inner_dims, dtype=searcher.ranker.ivf.tensor.dtype, device=searcher.ranker.ivf.tensor.device)        # a tensor of zeros, of length equal to max_stride, i.e the max possible ivf length
    searcher.ranker.ivf.tensor = torch.cat((searcher.ranker.ivf.tensor, padding))       # just pad the ivf list with a bunch of zeros

# self.views = {stride: _create_view(self.tensor, stride, self.inner_dims) for stride in self.strides}
## the _create_view call
views = {}
for stride in searcher.ranker.ivf.strides:
    outdim = searcher.ranker.ivf.tensor.size(0) - stride + 1
    inner_dims = searcher.ranker.ivf.inner_dims
    size = (outdim, stride, *inner_dims)

    inner_dim_prod = int(np.prod(inner_dims))
    multidim_stride = [inner_dim_prod, inner_dim_prod] + [1] * len(inner_dims)
    views[stride] = torch.as_strided(searcher.ranker.ivf.tensor, size=size, stride=multidim_stride)
## the _create_view call ends here

views.keys()
searcher.ranker.ivf.views.keys()

## the StridedTensorCore constructor ends here
## the StridedTensor constructor ends here
## self._load_ivf call ends here
searcher.ranker._load_doclens()     # just load the doclens
searcher.ranker._load_embeddings()  # load the saved embeddings
## both the codes and embeddings have an extra 512 entries; see the first line of the load_chunks method of the ResidualEmbeddings class

## IndexLoader constructor ends here

contextmanager.__exit__(None, None, None)

query = queries[37]
print(f"#> {query}")

# results = searcher.search(query, k=3)
# the searcher.search call
text = query
k = 3
filter_fn = None
full_length_search = False
pids = None

# Q = self.encode(text, full_length_search=full_length_search)
# the self.encode call
queries = text if type(text) is list else [text]        # just convert to list
bsize = 128 if len(queries) > 128 else None

searcher.config.query_maxlen
searcher.checkpoint.query_tokenizer.query_maxlen
searcher.checkpoint.query_tokenizer.query_maxlen = searcher.config.query_maxlen

# Q = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True, full_length_search=full_length_search)
Q = searcher.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True, full_length_search=full_length_search)         # convert query to embeddings
Q.size()        # (1, 32, 128)

Q = Q
## self.encode call ends here

## return self.dense_search(Q, k, filter_fn=filter_fn, pids=pids)
## the self.dense_search call
if searcher.config.ncells is None:
    searcher.configure(ncells=1)
if searcher.config.centroid_score_threshold is None:
    searcher.configure(centroid_score_threshold=0.5)
if searcher.config.ndocs is None:
    searcher.configure(ndocs=256)

## pids, scores = self.ranker.rank(self.config, Q, filter_fn=filter_fn, pids=pids)
## the self.ranker.rank call
config = searcher.config
Q = Q
filter_fn=None
pids = None

torch_context = torch.inference_mode()
torch_context.__enter__()

# pids, centroid_scores = self.retrieve(config, Q)
## the self.retrieve call
config = config
Q = Q

Q = Q[:, :config.query_maxlen]      # no change to Q in this case
# pids, centroid_scores = self.generate_candidates(config, Q)
## the self.generate_candidates call
## here generate_candidates = searcher.ranker.generate_candidates
## defined in CandidateGeneration.generate_candidates
ncells = config.ncells
assert isinstance(searcher.ranker.ivf, StridedTensor)

Q = Q.squeeze(0)        # get rid of the redundant 0th dimension
if searcher.ranker.use_gpu:     # true
    Q = Q.cuda().half()
assert Q.dim() == 2

# pids, centroid_scores = self.generate_candidate_pids(Q, ncells)
# the self.generate_candidate_pids call
Q, ncells = Q, ncells

## cells, scores = self.get_cells(Q, ncells)
## the self.get_cells call
scores = searcher.ranker.codec.centroids @ Q.T      # take inner product of all centroids with each token in Q
scores.shape        # (num_centroids, num_query_tokens) = (65536, 32)

if ncells == 1:
    ## for each query token, take the centroid with the largest_inner_product 
    ## return a 2 dimensional tensor of shape [1, num_query_tokens]
    ## .permute changes the shape to [num_query_tokens, 1]
    cells = scores.argmax(dim=0, keepdim=True).permute(1, 0)        # for each query token, take the centroid with the largest inner-product
else:
    cells = scores.topk(ncells, dim=0, sorted=False).indices.permute(1, 0)  # (32, ncells)

cells = cells.flatten().contiguous()        # (32 * ncells,)
cells = cells.unique(sorted=False)          # take the unique closest centroids

## self.get_cells call ends here
## pids, cell_lengths = self.ivf.lookup(cells)
## the self.ivf.lookup call

# self.ivf is a StridedTensor, created from the ivf.pid.pt file
# it was created as: ivf = StridedTensor(ivf, ivf_lengths, use_gpu=self.use_gpu)

## understanding the StridedTensor
searcher.ranker.ivf.dim is None
searcher.ranker.ivf.tensor          # just the ivf list we constructed
searcher.ranker.ivf.inner_dims
searcher.ranker.ivf.lengths
searcher.ranker.ivf.strides

torch_context.__exit__()


# Print out the top-k retrieved passages
for passage_id, passage_rank, passage_score in zip(*results):
    print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")

## Batch Search
rankings = searcher.search_all(queries, k=5).todict()
rankings[30]  # For query 30, a list of (passage_id, rank, score) for the top-k passages
