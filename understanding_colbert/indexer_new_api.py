import torch
torch.cuda.is_available()

import os
import sys
sys.path.insert(0, '../')

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

dataroot = 'downloads/lotte'
dataset = 'lifestyle'
datasplit = 'dev'

collection = os.path.join(dataroot, dataset, datasplit, 'collection.tsv')
collection = Collection(path=collection)

f'Loaded {len(collection):,} passages'
print(collection[89852])
print()

nbits = 2   # encode each dimension with 2 bits
doc_maxlen = 300   # truncate passages at 300 tokens

checkpoint = 'downloads/colbertv2.0'
index_name = f'{dataset}.{datasplit}.{nbits}bits'

with Run().context(RunConfig(nranks=1, experiment='notebook')): 
    config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)
    indexer = Indexer(checkpoint=checkpoint, config=config)
    indexer.index(name=index_name, collection=collection, overwrite=True)

indexer.get_index() # You can get the absolute path of the index, if needed.
