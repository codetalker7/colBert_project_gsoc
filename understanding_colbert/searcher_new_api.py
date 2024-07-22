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

###### the code that we want to uncover ############
queries = os.path.join(dataroot, dataset, datasplit, 'questions.search.tsv')
queries = Queries(path=queries)

# To create the searcher using its relative name (i.e., not a full path), set
# experiment=value_used_for_indexing in the RunConfig.
with Run().context(RunConfig(experiment='notebook')):
    searcher = Searcher(index=index_name)

# If you want to customize the search latency--quality tradeoff, you can also supply a
# config=ColBERTConfig(nprobe=.., ncandidates=..) argument. The default (2, 8192) works well,
# but you can trade away some latency to gain more extensive search with (4, 16384).
# Conversely, you can get faster search with (1, 4096).

query = queries[37]   # or supply your own query

print(f"#> {query}")

# Find the top-3 passages for this query
results = searcher.search(query, k=3)

# Print out the top-k retrieved passages
for passage_id, passage_rank, passage_score in zip(*results):
    print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")

######## code ends here ############

############# let's unpack the code now #################
### first, let's unpack the searcher.encode function
### call: searcher.encode(query)
text = query
queries = text if type(text) is list else [text]
bsize = 128 if len(queries) > 128 else None

searcher.checkpoint.query_tokenizer.query_maxlen = searcher.config.query_maxlen
Q = searcher.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True)

##### so, it suffices to understand what queryFromText is doing
