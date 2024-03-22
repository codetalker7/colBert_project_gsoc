import os
import sys
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

