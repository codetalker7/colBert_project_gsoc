import os
import sys
import pdb
sys.path.insert(0, '../')
import random
import time

import numpy as np
import torch

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from colbert.indexing.collection_indexer import encode
from colbert.utils.utils import create_directory, print_message
from colbert.infra.launcher import Launcher
from colbert.infra.config import BaseConfig, RunConfig, RunSettings

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

# setting the context; can do it exactly once to get it right
runconfig = RunConfig(nranks=1, experiment='notebook', avoid_fork_if_possible=True)
contextmanager = Run().context(runconfig)
contextmanager.__enter__()

## all code should go between the __enter__() and __exit__()
config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)

indexer = Indexer(checkpoint=checkpoint, config=config)

## index.index call
overwrite = False
name = index_name
indexer.config.nranks == 1 and indexer.config.avoid_fork_if_possible # so launch_without_fork will be called

indexer.configure(collection=collection, index_name=name, resume=overwrite=='resume')
indexer.configure(bsize=64, partitions=None)
indexer.index_path = indexer.config.index_path_
index_does_not_exist = (not os.path.exists(indexer.config.index_path_))
create_directory(indexer.config.index_path_)
if index_does_not_exist or overwrite != 'reuse':
    # this block is true
    # indexer.__launch(collection)
    pass

## indexer.__launc(collection) call
launcher = Launcher(encode)
if indexer.config.nranks == 1 and indexer.config.avoid_fork_if_possible:
    # this block is true
    shared_queues = []
    shared_lists = []
    # launcher.launch_without_fork(indexer.config, collection, shared_lists, shared_queues, indexer.verbose)

## the launcher.launch_without_fork call
custom_config = indexer.config
## *args = (collection, shared_lists, shared_queues, indexer)
assert isinstance(custom_config, BaseConfig)
assert isinstance(custom_config, RunSettings)
assert launcher.nranks == 1
assert (custom_config.avoid_fork_if_possible or launcher.run_config.avoid_fork_if_possible)
new_config = type(custom_config).from_existing(custom_config, launcher.run_config, RunConfig(rank=0))
# return_val = run_process_without_mp(self.callee, new_config, *args)

## the run_process_without_mp call
# *args = (collection, shared_lists, shared_queues, indexer.verbose)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(12345)
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.gpus_[:config.nranks]))
# with Run().context(config, inherit_config=False):
#     return_val = callee(config, *args)
#     torch.cuda.empty_cache()
#     return return_val
inner_config = new_config
innercontextmanager = Run().context(inner_config, inherit_config = False)

innercontextmanager.__enter__()

## all the encode function code goes within the innercontextmanager
# return_val = encode(inner_config, collection, shared_lists, shared_queues, indexer.verbose)

innercontextmanager.__exit__()

contextmanager.__exit__()

return_val = encode(indexer.config, collection, [], [], indexer.verbose)
