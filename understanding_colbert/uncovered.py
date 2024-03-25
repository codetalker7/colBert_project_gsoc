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
from colbert.utils.utils import create_directory, print_message
from colbert.infra.launcher import Launcher
from colbert.infra.config import BaseConfig, RunConfig, RunSettings
from colbert.indexing.collection_indexer import CollectionIndexer
from colbert.infra.launcher import print_memory_stats
import faiss
from colbert.indexing.codecs.residual import ResidualCodec
import tqdm

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

## _sample_embeddings call ends here

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

## sample, heldout = self._concatenate_and_split_sample()
## the _concatenate_and_split_sample call
sample = torch.empty(encoder.num_sample_embs, encoder.config.dim, dtype=torch.float16)
offset = 0
# # this for loop just unpacks the saved embeddings into a matrix
# # for our case, nranks is just 1
# for r in range(encoder.nranks):
#     sub_sample_path = os.path.join(self.config.index_path_, f'sample.{r}.pt')
#     sub_sample = torch.load(sub_sample_path)
#     os.remove(sub_sample_path)
#
#     endpos = offset + sub_sample.size(0)
#     sample[offset:endpos] = sub_sample
#     offset = endpos
offset = 0
r = 0       # nranks = 1
sub_sample_path = os.path.join(encoder.config.index_path_, f'sample.{r}.pt')
sub_sample = torch.load(sub_sample_path)    # the saved tensor
os.remove(sub_sample_path)

endpos = offset + sub_sample.size(0)
sample[offset:endpos] = sub_sample
offset = endpos

assert endpos == sample.size(0), (endpos, sample.size())

# randomly permute the sample
sample = sample[torch.randperm(sample.size(0))]

heldout_fraction = 0.05
heldout_size = int(min(heldout_fraction * sample.size(0), 50_000))
sample, sample_heldout = sample.split([sample.size(0) - heldout_size, heldout_size], dim=0)

assert sample_heldout.size()[0] + sample.size()[0] == encoder.num_sample_embs
sample, sample_heldout = sample, sample_heldout

# _concatenate_and_split_sample call ends here
# centroids = self._train_kmeans(sample, shared_lists)
# the _train_kmeans call
if encoder.use_gpu:
    torch.cuda.empty_cache()

args_ = [encoder.config.dim, encoder.num_partitions, encoder.config.kmeans_niters]

# do_fork_for_faiss is not true
# so we only include the else block of the function here
args_ = args_ + [[[sample]]]    # args now consists of 4 things

# centroids = compute_faiss_kmeans(*args_)
# the compute_faiss_kmeans call
dim, num_partitions, kmeans_niters, shared_lists = args_[0], args_[1], args_[2], args_[3]
# so shared_lists here is [sample], i.e a list containing the sample
return_value_queue=None

use_gpu = torch.cuda.is_available()
kmeans = faiss.Kmeans(dim, num_partitions, niter=kmeans_niters, gpu=use_gpu, verbose=True, seed=123)
sample = shared_lists[0][0]     # our sample
sample = sample.float().numpy()
kmeans.train(sample)

centroids = torch.from_numpy(kmeans.centroids)
# compute_faiss_kmeans call ends here

centroids = torch.nn.functional.normalize(centroids, dim=-1)
if encoder.use_gpu:
    centroids = centroids.half()
else:
    centroids = centroids.float()
## _train_kmeans call ends here
del sample      # delete the sample set

# bucket_cutoffs, bucket_weights, avg_residual = self._compute_avg_residual(centroids, heldout)
## the _compute_avg_residual call
# the average is computed over the heldout set
heldout = sample_heldout
compressor = ResidualCodec(config=encoder.config, centroids=centroids, avg_residual=None)
assert compressor.use_gpu > 0
compressor.centroids
assert compressor.dim == encoder.config.dim
assert compressor.nbits == encoder.config.nbits
assert compressor.avg_residual == None
assert compressor.bucket_cutoffs == None
assert compressor.bucket_weights == None
compressor.arange_bits # = torch.arange(0, compressor.nbits, device='cuda' if compressor.use_gpu else 'cpu', dtype=torch.uint8)
compressor.reversed_bit_map

mask = (1 << compressor.nbits) - 1  # all 1's, number of 1's = compressor.nbits
reversed_bit_map = []
for i in range(256):
    # The reversed byte
    z = 0
    for j in range(8, 0, -compressor.nbits): # [8, 6, 4, 2]
        # Extract a subsequence of length n bits
        x = (i >> (j - compressor.nbits)) & mask    # j - compressor.nbits is in [6, 4, 2, 0]

        # Reverse the endianness of each bit subsequence (e.g. 10 -> 01)
        y = 0
        for k in range(compressor.nbits - 1, -1, -1):
            y += ((x >> (compressor.nbits - k - 1)) & 1) * (2 ** k)

        # Set the corresponding bits in the output byte
        z |= y
        if j > compressor.nbits:
            z <<= compressor.nbits
    reversed_bit_map.append(z)

assert compressor.decompression_lookup_table is None
ResidualCodec.decompress_residuals  # decompress_residuals.cpp, decompress_residuals.cu
ResidualCodec.packbits # pacbits.cpp, packbits.cu 

# end of ResidualCodec constructor here
# heldout_reconstruct = compressor.compress_into_codes(heldout, out_device='cuda' if self.use_gpu else 'cpu')
## the compress_into_codes call
embs = heldout
out_device='cuda' if encoder.use_gpu else 'cpu'

## output on a single batch
bsize = (1 << 29) // compressor.centroids.size(0)
single_batch = embs.split(bsize)[0]
single_batch.size()     # [8192, 128]
single_batch.T
single_batch.T.size()   # [128, 8192]
compressor.centroids @ single_batch.T.cuda().half() # all possible inner products
(compressor.centroids @ single_batch.T.cuda().half()).size()    # [num_centroids, 8192]
(compressor.centroids @ single_batch.T.cuda().half()).max(dim=0)
indices = (compressor.centroids @ single_batch.T.cuda().half()).max(dim = 0).indices.to(device = out_device)
assert indices.size()[0] == 8192    # for each embedding in the batch, we got the index of the closest centroid

# output on all batches
codes = []
bsize = (1 << 29) // compressor.centroids.size(0)
for batch in embs.split(bsize):     # split sample_heldout into batches of size bsize
    if compressor.use_gpu:
        indices = (compressor.centroids @ batch.T.cuda().half()).max(dim=0).indices.to(device=out_device)
    else:
        indices = (compressor.centroids @ batch.T.cpu().float()).max(dim=0).indices.to(device=out_device)
    codes.append(indices)

# heldout reconstruct holds the indices of the closest centroid for each heldout embedding
heldout_reconstruct = torch.cat(codes)      ## collect all the codes together
## the compress_into_codes call ends here
assert heldout_reconstruct.size()[0] == sample_heldout.size()[0]

# heldout_reconstruct = compressor.lookup_centroids(heldout_reconstruct, out_device='cuda' if self.use_gpu else 'cpu')
## the lookup_centroids call
codes = heldout_reconstruct
inner_centroids = []

for batch in codes.split(1 << 20): # contains only one batch
    if compressor.use_gpu:
        inner_centroids.append(compressor.centroids[batch.cuda().long()].to(device=out_device))
    else:
        inner_centroids.append(compressor.centroids[batch.long()].to(device=out_device))
# so, centroids contain the centroids closest to all the embeddings in the heldout set
heldout_reconstruct = torch.cat(inner_centroids)

## lookup_centroids call ends here
if encoder.use_gpu:
    heldout_avg_residual = heldout.cuda() - heldout_reconstruct
else:
    heldout_avg_residual = heldout - heldout_reconstruct

# avg_residual is the average of all absolute values of heldout residuals over each dimension
avg_residual = torch.abs(heldout_avg_residual).mean(dim=0).cpu()    # why take abs here?
print([round(x, 3) for x in avg_residual.squeeze().tolist()])

num_options = 2 ** encoder.config.nbits
quantiles = torch.arange(0, num_options, device=heldout_avg_residual.device) * (1 / num_options)
bucket_cutoffs_quantiles, bucket_weights_quantiles = quantiles[1:], quantiles + (0.5 / num_options)

bucket_cutoffs = heldout_avg_residual.float().quantile(bucket_cutoffs_quantiles)
bucket_weights = heldout_avg_residual.float().quantile(bucket_weights_quantiles)

if encoder.verbose > 2:
    print_message(
        f"#> Got bucket_cutoffs_quantiles = {bucket_cutoffs_quantiles} and bucket_weights_quantiles = {bucket_weights_quantiles}")
    print_message(f"#> Got bucket_cutoffs = {bucket_cutoffs} and bucket_weights = {bucket_weights}")

bucket_cutoffs, bucket_weights, avg_residual = bucket_cutoffs, bucket_weights, avg_residual.mean()
#_compute_avg_residual call ends here

if encoder.verbose > 1:
    print_message(f'avg_residual = {avg_residual}')

codec = ResidualCodec(config=encoder.config, centroids=centroids, avg_residual=avg_residual, bucket_cutoffs=bucket_cutoffs, bucket_weights=bucket_weights)
# encoder.saver.save_codec(codec)
# the save_code(codec) call
# codec.save(index_path=encoder.saver.config.index_path_)
# the codec.save call
index_path = encoder.saver.config.index_path_
assert codec.avg_residual is not None
assert torch.is_tensor(codec.bucket_cutoffs), codec.bucket_cutoffs
assert torch.is_tensor(codec.bucket_weights), codec.bucket_weights

centroids_path = os.path.join(index_path, 'centroids.pt')
avgresidual_path = os.path.join(index_path, 'avg_residual.pt')
buckets_path = os.path.join(index_path, 'buckets.pt')

torch.save(codec.centroids.half(), centroids_path)
torch.save((codec.bucket_cutoffs, codec.bucket_weights), buckets_path)

if torch.is_tensor(codec.avg_residual):
    torch.save(codec.avg_residual, avgresidual_path)
else:
    torch.save(torch.tensor([codec.avg_residual]), avgresidual_path)

# codec.save call ends here
# save_codec call ends here
# train call ends here

## self.train(shared_lists) call ends here
## self.index() call
## we'll do it without threading
encoder.saver.codec = encoder.saver.load_codec()
batches = encoder.collection.enumerate_batches(rank=encoder.rank)     # batch the collection into batches of size given by collection.get_chunksize()
# each batch contains (chunk_idx, offset, batch), where offset is the id of the first passage in the batch


# # let's first try to save a single batch
# chunk_idx, offset, passages = next(batches)
# embs, doclens = encoder.encoder.encode_passages(passages)
# if encoder.use_gpu:
#     assert embs.dtype == torch.float16
# else:
#     assert embs.dtype == torch.float32
#     embs = embs.half()
# if encoder.verbose > 1:
#     Run().print_main(f"#> Saving chunk {chunk_idx}: \t {len(passages):,} passages "
#                     f"and {embs.size(0):,} embeddings. From #{offset:,} onward.")
# # encoder.saver.save_chunk(chunk_idx, offset, embs, doclens)
# # compressed_embs = self.codec.compress(embs)
# # the codec.compress call
#
# ## example on a single batch
# # single_batch = embs.split(1 << 18)[0]
# # if encoder.saver.codec.use_gpu:
# #     single_batch = single_batch.cuda().half()
# # codes_ = encoder.saver.codec.compress_into_codes(single_batch, out_device=single_batch.device)
# # centroids_ = encoder.saver.codec.lookup_centroids(codes_, out_device=single_batch.device)  # tensor containing centroids corresponding to ids in codes_
# # residuals_ = (single_batch - centroids_)
# # binarized_residuals = encoder.saver.codec.binarize(residuals_).cpu() # this is the function which quantizes the residuals
#
# codes, residuals = [], []
# for batch in embs.split(1 << 18):
#     if encoder.saver.codec.use_gpu:
#         batch = batch.cuda().half()
#     codes_ = encoder.saver.codec.compress_into_codes(batch, out_device=batch.device)  # list of nearest centroid ids
#     assert codes_.size()[0] == batch.size()[0]
#     centroids_ = encoder.saver.codec.lookup_centroids(codes_, out_device=batch.device)  # tensor containing centroids corresponding to ids in codes_
#
#     residuals_ = (batch - centroids_)
#
#     codes.append(codes_.cpu())
#     residuals.append(encoder.saver.codec.binarize(residuals_).cpu())  # this is where the residuals are quantized
#
# codes = torch.cat(codes)
# residuals = torch.cat(residuals)
#
# compressed_embs = ResidualCodec.Embeddings(codes, residuals)
# # self.codec.compress(embs) call ends here
# # after this, self.saver_queue call is made. which we don't do here. Instead, we directly call self._write_chunk_to_disk here.
# # _write_chunk_to_disk call
# path_prefix = os.path.join(encoder.saver.config.index_path_, str(chunk_idx))
# compressed_embs.save(path_prefix)
#
# doclens_path = os.path.join(encoder.saver.config.index_path_, f'doclens.{chunk_idx}.json')
# with open(doclens_path, 'w') as output_doclens:
#     ujson.dump(doclens, output_doclens)
#
# metadata_path = os.path.join(encoder.saver.config.index_path_, f'{chunk_idx}.metadata.json')
# with open(metadata_path, 'w') as output_metadata:
#     metadata = {'passage_offset': offset, 'num_passages': len(doclens), 'num_embeddings': len(compressed_embs)}
#     ujson.dump(metadata, output_metadata)
# ## _write_chunk_to_disk call ends here
# ## saving a single batch ends here
#
# del embs, doclens

for chunk_idx, offset, passages in tqdm.tqdm(batches, disable=encoder.rank > 0):
    # can ignore the if block in the code here
    embs, doclens = encoder.encoder.encode_passages(passages)
    if encoder.use_gpu:
        assert embs.dtype == torch.float16
    else:
        assert embs.dtype == torch.float32
        embs = embs.half()
    if encoder.verbose > 1:
        Run().print_main(f"#> Saving chunk {chunk_idx}: \t {len(passages):,} passages "
                        f"and {embs.size(0):,} embeddings. From #{offset:,} onward.")
    # self.saver.save_chunk(hunk_idx, offset, embs, doclens)
    
    codes, residuals = [], []
    for batch in embs.split(1 << 18):
        if encoder.saver.codec.use_gpu:
            batch = batch.cuda().half()
        codes_ = encoder.saver.codec.compress_into_codes(batch, out_device=batch.device)  # list of nearest centroid ids
        assert codes_.size()[0] == batch.size()[0]
        centroids_ = encoder.saver.codec.lookup_centroids(codes_, out_device=batch.device)  # tensor containing centroids corresponding to ids in codes_

        residuals_ = (batch - centroids_)

        codes.append(codes_.cpu())
        residuals.append(encoder.saver.codec.binarize(residuals_).cpu())  # this is where the residuals are quantized

    codes = torch.cat(codes)
    residuals = torch.cat(residuals)

    compressed_embs = ResidualCodec.Embeddings(codes, residuals)
    # self.codec.compress(embs) call ends here
    # after this, self.saver_queue call is made. which we don't do here. Instead, we directly call self._write_chunk_to_disk here.
    # _write_chunk_to_disk call
    path_prefix = os.path.join(encoder.saver.config.index_path_, str(chunk_idx))
    compressed_embs.save(path_prefix)

    doclens_path = os.path.join(encoder.saver.config.index_path_, f'doclens.{chunk_idx}.json')
    with open(doclens_path, 'w') as output_doclens:
        ujson.dump(doclens, output_doclens)

    metadata_path = os.path.join(encoder.saver.config.index_path_, f'{chunk_idx}.metadata.json')
    with open(metadata_path, 'w') as output_metadata:
        metadata = {'passage_offset': offset, 'num_passages': len(doclens), 'num_embeddings': len(compressed_embs)}
        ujson.dump(metadata, output_metadata)
    ## _write_chunk_to_disk call ends here

    del embs, doclens

## self.index() call ends here
## the self.finalize() call
encoder._check_all_files_are_saved()
## self._collect_embedding_id_offset() call
## _collect_embedding_id_offset just stores the first embedding id in a given chunk's metadata
## it also sets the total number of embeddings computed
passage_offset = 0
embedding_offset = 0

encoder.embedding_offsets = []
for chunk_idx in range(encoder.num_chunks):
    metadata_path = os.path.join(encoder.config.index_path_, f'{chunk_idx}.metadata.json')

    with open(metadata_path) as f:
        chunk_metadata = ujson.load(f)

        chunk_metadata['embedding_offset'] = embedding_offset
        encoder.embedding_offsets.append(embedding_offset)

        assert chunk_metadata['passage_offset'] == passage_offset, (chunk_idx, passage_offset, chunk_metadata)

        passage_offset += chunk_metadata['num_passages']
        embedding_offset += chunk_metadata['num_embeddings']

    with open(metadata_path, 'w') as f:
        f.write(ujson.dumps(chunk_metadata, indent=4) + '\n')

encoder.num_embeddings = embedding_offset
assert len(encoder.embedding_offsets) == encoder.num_chunks
## _collect_embedding_id_offset() call ends here



torch_context.__exit__(None, None, None)

innercontextmanager.__exit__(None, None, None)

contextmanager.__exit__(None, None, None)

return_val = encode(indexer.config, collection, [], [], indexer.verbose)
