import os
import sys
import pdb
sys.path.insert(0, '../')

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

dataroot = 'downloads/lotte'
dataset = 'lifestyle'
datasplit = 'dev'

queries = os.path.join(dataroot, dataset, datasplit, 'questions.search.tsv')
collection = os.path.join(dataroot, dataset, datasplit, 'collection.tsv')

queries = Queries(path=queries)
collection = Collection(path=collection)

f'Loaded {len(queries)} queries and {len(collection):,} passages'

# analyzing the queries object
queries.data                # ordered dict
queries.data.items()
queries.__len__()
queries.provenance()        # path of dataset
queries.toDict()            # path in dict format
queries.qas()

# analyzing the collection object
collection.data             # list
collection.__len__()
collection.__getitem__(1)   # get an item using the index
collection.provenance()     # path of dataset

# Indexing

nbits = 2   # encode each dimension with 2 bits
doc_maxlen = 300   # truncate passages at 300 tokens

checkpoint = 'downloads/colbertv2.0'
index_name = f'{dataset}.{datasplit}.{nbits}bits'

# RunConfig
Run()._instance
Run()._instance.stack
Run().config
RunConfig(nranks=1, experiment='notebook')
with Run().context(RunConfig(nranks=1, experiment='notebook')):
    print(Run()._instance.stack)        # list with two RunConfigs
    print()
    print(len(Run()._instance.stack))
Run()._instance.stack

# ColBERTConfig
with Run().context(RunConfig(nranks=1, experiment='notebook', avoid_fork_if_possible=True)):  # nranks specifies the number of GPUs to use.
    # don't use mp; for this, we use avoid_fork_if_possible
    config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)

    # ## RunSettings
    # config.overwrite
    # config.root
    # config.experiment
    # config.index_root
    # config.name
    # config.rank
    # config.nranks
    # config.amp
    # config.total_visible_gpus
    # config.gpus
    # config.avoid_fork_if_possible
    # config.gpus_
    # config.index_root_
    # config.script_name_
    # config.path_
    # config.device_
    #
    # ## TokenizerSettings
    # config.query_token_id
    # config.doc_token_id
    # config.query_token
    # config.query_token
    #
    # ## ResourceSettings
    # config.checkpoint
    # config.triples
    # config.collection
    # config.queries
    # config.index_name
    #
    # ## DocSettings
    # config.dim
    # config.doc_maxlen
    # config.mask_punctuation
    #
    # ## QuerySettings
    # config.query_maxlen
    # config.attend_to_mask_tokens
    # config.interaction
    #
    # ## TrainingSettings
    # config.similarity
    # config.bsize
    # config.accumsteps
    # config.lr
    # config.maxsteps
    # config.save_every
    # config.resume
    # config.warmup
    # config.warmup_bert
    # config.relu
    # config.nway
    # config.use_ib_negatives
    # config.reranker
    # config.distillation_alpha
    # config.ignore_scores
    # config.model_name
    #
    # ## IndexingSettings
    # config.index_path
    # config.index_bsize
    # config.nbits
    # config.kmeans_niters
    # config.resume
    # config.index_path_
    #
    # ## SearchSettings
    # config.ncells
    # config.centroid_score_threshold
    # config.ndocs
    # config.load_index_with_mmap

    # indexer = Indexer(checkpoint=checkpoint, config=config)
    indexer = pdb.runcall(Indexer, checkpoint=checkpoint, config=config)
    assert indexer.config.avoid_fork_if_possible
    # indexer.index(name=index_name, collection=collection, overwrite=True)
    pdb.runcall(indexer.index, name=index_name, collection=collection, overwrite=True)



## the enumerate_batches function from Collection
import itertools
rank = 0
nranks = 1
chunksize = collection.get_chunksize()
offset = 0
iterator = iter(collection)
batches = []
for chunk_idx, owner in enumerate(itertools.cycle(range(nranks))):
    print(chunk_idx, owner)
    L = [line for _, line in zip(range(chunksize), iterator)] # gets the first chunksize elements of collection
    if len(L) > 0 and owner == rank:
        batches.append((chunk_idx, offset, L))

    offset += len(L)

    if len(L) < chunksize:
        break

batches = []
local_pids = collection.enumerate(rank=rank)    # equivalent to normal enumerate
for idx, passage in local_pids:
    batches.append((idx, passage))
assert len(batches) == len(collection)
local_sample = [passage for pid, passage in local_pids if pid in sampled_pids]  # just get the passages for each of the sampled_pids
