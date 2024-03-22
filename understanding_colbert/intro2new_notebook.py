import colbert

from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection

from datasets import load_dataset

dataset = 'lifestyle'
datasplit = 'dev'

collection_dataset = load_dataset("colbertv2/lotte_passages", dataset)
# collection is just a list containing all the passages
collection = [x['text'] for x in collection_dataset[datasplit + '_collection']]

queries_dataset = load_dataset("colbertv2/lotte", dataset)
# similarly, queries is just a list containing the query texts
queries = [x['query'] for x in queries_dataset['search_' + datasplit]]

f'Loaded {len(queries)} queries and {len(collection):,} passages'

print(queries[24])
print()
print(collection[19929])
print()

# Indexing
nbits = 2   # encode each dimension with 2 bits
doc_maxlen = 300 # truncate passages at 300 tokens
max_id = 10000

index_name = f'{dataset}.{datasplit}.{nbits}bits'

# answer pids is the list of all possible answer passage ids
answer_pids = [x['answers']['answer_pids'] for x in queries_dataset['search_' + datasplit]]
# get only those queries whose answer passage ids are less than 10k
filtered_queries = [q for q, apids in zip(queries, answer_pids) if any(x < max_id for x in apids)]

f'Filtered down to {len(filtered_queries)} queries'

# Running the indexer
checkpoint = 'colbert-ir/colbertv2.0'

with Run().context(RunConfig(nranks=1, experiment='notebook')):  # nranks specifies the number of GPUs to use
    config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4) # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
                                                                                # Consider larger numbers for small datasets.

    indexer = Indexer(checkpoint=checkpoint, config=config)
    indexer.index(name=index_name, collection=collection[:max_id], overwrite=True)

indexer.get_index() # You can get the absolute path of the index, if needed.

# Search
# To create the searcher using its relative name (i.e., not a full path), set
# experiment=value_used_for_indexing in the RunConfig.
with Run().context(RunConfig(experiment='notebook')):
    searcher = Searcher(index=index_name, collection=collection)


# If you want to customize the search latency--quality tradeoff, you can also supply a
# config=ColBERTConfig(ncells=.., centroid_score_threshold=.., ndocs=..) argument.
# The default settings with k <= 10 (1, 0.5, 256) gives the fastest search,
# but you can gain more extensive search by setting larger values of k or
# manually specifying more conservative ColBERTConfig settings (e.g. (4, 0.4, 4096)).

query = filtered_queries[13] # try with an in-range query or supply your own
print(f"#> {query}")

# Find the top-3 passages for this query
results = searcher.search(query, k=3)

# Print out the top-k retrieved passages
for passage_id, passage_rank, passage_score in zip(*results):
    print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")
