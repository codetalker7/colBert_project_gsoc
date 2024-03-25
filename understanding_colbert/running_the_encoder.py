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
from colbert.modeling.checkpoint import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder

dataroot = 'downloads/lotte'
dataset = 'lifestyle'
datasplit = 'dev'

queries = os.path.join(dataroot, dataset, datasplit, 'questions.search.tsv')
collection = os.path.join(dataroot, dataset, datasplit, 'collection.tsv')

queries = Queries(path=queries)
collection = Collection(path=collection)

nbits = 2   # encode each dimension with 2 bits
doc_maxlen = 300   # truncate passages at 300 tokens

checkpoint = 'downloads/colbertv2.0'
index_name = f'{dataset}.{datasplit}.{nbits}bits'
f'Loaded {len(queries)} queries and {len(collection):,} passages'

runconfig = RunConfig(nranks=1, experiment='notebook', avoid_fork_if_possible=True)
contextmanager = Run().context(runconfig)
contextmanager.__enter__()

## all code should go between the __enter__() and __exit__()
config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)

indexer = Indexer(checkpoint=checkpoint, config=config)

overwrite = False

indexer.configure(collection=collection, index_name=index_name, resume=overwrite=='resume')
indexer.configure(bsize=64, partitions=None)
indexer.index_path = indexer.config.index_path_

## indexer.__launc(collection) call
launcher = Launcher(encode)
if indexer.config.nranks == 1 and indexer.config.avoid_fork_if_possible:
    # this block is true
    shared_queues = []
    shared_lists = []
    # launcher.launch_without_fork(indexer.config, collection, shared_lists, shared_queues, indexer.verbose)

## the launcher.launch_without_fork call
custom_config = indexer.config
new_config = type(custom_config).from_existing(custom_config, launcher.run_config, RunConfig(rank=0))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(12345)
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.gpus_[:config.nranks]))

inner_config = new_config
innercontextmanager = Run().context(inner_config, inherit_config = False)

innercontextmanager.__enter__()

## Embedding Code Starts

# encoder = CollectionIndexer(config=inner_config, collection=collection, verbose=indexer.verbose)

encoder_checkpoint = Checkpoint(inner_config.checkpoint, colbert_config=inner_config)
actual_encoder = CollectionEncoder(inner_config, encoder_checkpoint)

torch_context = torch.inference_mode()
torch_context.__enter__()

local_pids = collection.enumerate(rank=encoder.rank)    # enumerate all the passages

passages = [passage for pid, passage in local_pids if pid < 100]  # those passages whose id is in sampled_pids

embs, doclens = actual_encoder.encode_passages(passages)

## Embedding Code Ends

torch_context.__exit__(None, None, None)

innercontextmanager.__exit__(None, None, None)

contextmanager.__exit__(None, None, None)
