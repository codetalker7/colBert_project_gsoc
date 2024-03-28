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

searcher = Searcher(index=index_name)
searcher.index
searcher.index_config
searcher.checkpoint
searcher.checkpoint_config
searcher.config
searcher.collection
searcher.config.load_index_with_mmap
searcher.ranker     # colbert.search.index_storage.IndexScorer

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
## the self.ranker call
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


torch_context.__exit__()


# Print out the top-k retrieved passages
for passage_id, passage_rank, passage_score in zip(*results):
    print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")

## Batch Search
rankings = searcher.search_all(queries, k=5).todict()
rankings[30]  # For query 30, a list of (passage_id, rank, score) for the top-k passages
