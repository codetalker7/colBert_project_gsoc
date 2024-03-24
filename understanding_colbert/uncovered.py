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
from colbert.indexing.collection_indexer import CollectionIndexer
from colbert.infra.launcher import print_memory_stats

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
encoder = CollectionIndexer(config=inner_config, collection=collection, verbose=indexer.verbose)
# encode.run(shared_lists)

## the encode.run(shared_lists) call
torch_context = torch.inference_mode()
torch_context.__enter__()

## all encode.run code must go within torch_context
## the self.setup() call
encoder.num_chunks = int(np.ceil(len(encoder.collection) / encoder.collection.get_chunksize()))
# Saves sampled passages and embeddings for training k-means centroids later 
# sampled_pids = encoder._sample_pids()
num_passages = len(encoder.collection)
typical_doclen = 120  # let's keep sampling independent of the actual doc_maxlen
sampled_pids = 16 * np.sqrt(typical_doclen * num_passages)
sampled_pids = min(1 + int(sampled_pids), num_passages)
sampled_pids = random.sample(range(num_passages), sampled_pids)
if encoder.verbose > 1:
    Run().print_main(f"# of sampled PIDs = {len(sampled_pids)} \t sampled_pids[:3] = {sampled_pids[:3]}")
sampled_pids = set(sampled_pids)

# avg_doclen_est = encoder._sample_embeddings(sampled_pids)
local_pids = encoder.collection.enumerate(rank=encoder.rank)

# ## checking that local_pids is just normal enumerate
# mylist = []
# for idx, passage in local_pids:
#     mylist.append(passage)
# assert mylist == collection.data

local_sample = [passage for pid, passage in local_pids if pid in sampled_pids]  # those passages whose id is in sampled_pids

local_sample_embs, doclens = encoder.encoder.encode_passages(local_sample)
type(local_sample_embs)     # a tensor
local_sample_embs.shape     # two dimensional tensor
## so local_sample_embs is a matrix containing the embeddings of all tokens over all local_sample
doclens
assert len(doclens) == len(sampled_pids)

torch.cuda.is_available()
torch.distributed.is_available() and torch.distributed.is_initialized()

encoder.num_sample_embs = torch.tensor([local_sample_embs.size(0)]).cuda()
avg_doclen_est = sum(doclens) / len(doclens) if doclens else 0
avg_doclen_est = torch.tensor([avg_doclen_est]).cuda()
nonzero_ranks = torch.tensor([float(len(local_sample) > 0)]).cuda()

avg_doclen_est = avg_doclen_est.item() / nonzero_ranks.item()
encoder.avg_doclen_est = avg_doclen_est
Run().print(f'avg_doclen_est = {avg_doclen_est} \t len(local_sample) = {len(local_sample):,}')
torch.save(local_sample_embs.half(), os.path.join(encoder.config.index_path_, f'sample.{encoder.rank}.pt'))
assert local_sample_embs.half() == local_sample_embs

avg_doclen_est = avg_doclen_est

num_passages = len(encoder.collection)
encoder.num_embeddings_est = num_passages * avg_doclen_est
encoder.num_partitions = int(2 ** np.floor(np.log2(16 * np.sqrt(encoder.num_embeddings_est))))
if encoder.verbose > 0:
    Run().print_main(f'Creating {encoder.num_partitions:,} partitions.')
    Run().print_main(f'*Estimated* {int(encoder.num_embeddings_est):,} embeddings.')

encoder._save_plan()

## encoder.setup() call finishes
## encoder.train() call starts
# if not encoder.config.resume or not encoder.saver.try_load_codec():
#     encoder.train(shared_lists) # Trains centroids from selected passages
if encoder.rank > 0:
    pass

torch_context.__exit__(None, None, None)

innercontextmanager.__exit__(None, None, None)

contextmanager.__exit__(None, None, None)

return_val = encode(indexer.config, collection, [], [], indexer.verbose)
