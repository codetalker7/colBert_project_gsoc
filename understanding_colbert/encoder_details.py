from colbert.modeling.base_colbert import BaseColBERT
from colbert.modeling.colbert import ColBERT 
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra.config.base_config import BaseConfig
from colbert.infra.config.config import ColBERTConfig
from colbert.modeling.hf_colbert import class_factory
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
import os
import pdb
import string
import tqdm
import torch
from colbert.parameters import DEVICE

## HELPER FUNCTIONS

def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

    return batches

def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output

## HELPER FUNCTIONS END

dataroot = 'downloads/lotte'
dataset = 'lifestyle'
datasplit = 'dev'

nbits = 2   # encode each dimension with 2 bits
doc_maxlen = 300   # truncate passages at 300 tokens

checkpoint = 'downloads/colbertv2.0'
index_root = 'experiments/notebook/indexes'
index_name = f'{dataset}.{datasplit}.{nbits}bits'
index_path = os.path.join(index_root, index_name)

config = ColBERTConfig.load_from_index(index_path)

## BaseColBERT
baseColbert = BaseColBERT(checkpoint, config)
baseColbert.colbert_config      # the config using which the model was trained
baseColbert.name                # 'downloads/colbertv2.0', the name of the checkpoint
baseColbert.model               # HF_ColBERT model; a linear layer with input dim 768 and output dim 128 with no bias, and then the BERT model.
baseColbert.raw_tokenizer       # the underlying BERT tokenizer
baseColbert.bert                # the underlying BERT model
class_factory(baseColbert.name) # HF_ColBERT

## ColBERT
colBert = ColBERT(checkpoint, config)   # a linear layer 768=>128, and then the BERT model.
colBert.use_gpu                         # true, depends on the underlying config
colBert.colbert_config.mask_punctuation # true, depends on the underlying config
colBert.skiplist        # for each punctation, map it to true. also maps it's encoding to true
string.punctuation
colBert.raw_tokenizer
colBert.raw_tokenizer.encode('!', add_special_tokens=False)[0]
colBert.pad_token

## Checkpoint
checkPoint = Checkpoint(checkpoint, config)
checkPoint.verbose
checkPoint.query_tokenizer
checkPoint.doc_tokenizer
checkPoint.amp_manager

checkPoint.queryFromText(["hello world"])                   # this is used to convert queries to embeddings
checkPoint.docFromText(["hello world"]).size()              # this is used to convert documents to embeddings
checkPoint.doc_tokenizer
checkPoint.query_tokenizer

## docFromText details
"""
the encode_passages function of the CollectionEncoder calls docFromText in the following format:
    embs_, doclens_ = self.checkpoint.docFromText(passages_batch, bsize=self.config.index_bsize, keep_dims='flatten')
So we study this function with a batch size of 2, and keep_dims set to 'flatten'.
"""
checkPoint.docFromText(["hello world", "thank you!", "a", "this is some longer text, so length should be longer"], bsize=2, keep_dims='flatten')

## we study the above call
docs = ["hello world", "thank you!", "a", "this is some longer text, so length should be longer"]
bsize=2
keep_dims='flatten'
to_cpu=False
showprogress=False
return_tokens=False

if bsize:
    # text_batches, reverse_indices = checkPoint.doc_tokenizer.tensorize(docs, bsize=bsize)
    # we replicate the above doc_tokenizer call above
    batch_text = docs
    bsize=bsize
    assert type(batch_text) in [list, tuple], (type(batch_text))

    # add placehold for the [D] marker
    batch_text = ['. ' + x for x in batch_text]

    # convert all the texts into token ids
    obj = checkPoint.doc_tokenizer.tok(batch_text, padding='longest', truncation='longest_first',
                    return_tensors='pt', max_length=checkPoint.doc_tokenizer.doc_maxlen).to(DEVICE)
    ## observe that the second token ID is 1012, the token ID for ".".

    ids, mask = obj['input_ids'], obj['attention_mask']

    # postprocess for the [D] marker
    ids[:, 1] = checkPoint.doc_tokenizer.D_marker_token_id

    if bsize:
        ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
        """
        _sort_by_length takes in the ids, the attention mask, and the batch size
        if the number of passages (ids.size(0)) is less than bsize, a tuple 
        (ids, mask, torch.arange(ids.size(0))) is returned. So suppose the number 
        of passages is larger than bsize. In that case, the number of attended tokens
        is calculated for each passage (mask.sum(-1)), and the passages are sorted
        in increasing order of the number of tokens (indices = mask.sum(-1).sort().indices).
        so indices stores the indices of the sorted passages (in the original passage list).
        reverse_indices inverts this list, i.e for each passage, we store it's position in the 
        sorted list (i.e reverse_indices[i] stores the indice in the sorted list of the ith passage).
        then, the tensor containing the token ids (in sorted order) are returned, along with the corresponding
        mask and the reverse_indices list.
        """
        batches = _split_into_batches(ids, mask, bsize)
        """
        _split_into_batches just splits these ids and mask into batches of size bsize. recall that
        the previous step sorts these passages by the number of attended tokens (from least to highest),
        if the number of passages is larger than the batch size.
        """
        # return batches, reverse_indices

    # return ids, mask

    ## so text_batches is a batch of texts (and masks), sorted by number of tokens (in increasing order)
    ## if the batch size is greater than number of passages. otherwise, it's simply
    ## the batch of texts (unsorted), and reverse_indices is simply [0, ..., num_passages - 1].
    text_batches, reverse_indices = batches, reverse_indices            

    ## the doc_tokenizer.tensorize call ends here

    returned_text = []
    if return_tokens:
        returned_text = [text for batch in text_batches for text in batch[0]]
        returned_text = [returned_text[idx] for idx in reverse_indices.tolist()]
        returned_text = [returned_text]                     ## just the original texts, in unsorted order

    keep_dims_ = 'return_mask' if keep_dims == 'flatten' else keep_dims
    batches = [checkPoint.doc(input_ids, attention_mask, keep_dims=keep_dims_, to_cpu=to_cpu)
                for input_ids, attention_mask in tqdm(text_batches, disable=not showprogress)]

    if keep_dims is True:
        D = _stack_3D_tensors(batches)
        # return (D[reverse_indices], *returned_text)

    elif keep_dims == 'flatten':
        D, mask = [], []

        for D_, mask_ in batches:
            D.append(D_)
            mask.append(mask_)

        D, mask = torch.cat(D)[reverse_indices], torch.cat(mask)[reverse_indices]

        doclens = mask.squeeze(-1).sum(-1).tolist()

        D = D.view(-1, self.colbert_config.dim)
        D = D[mask.bool().flatten()].cpu()

        # return (D, doclens, *returned_text)

    # assert keep_dims is False

    # D = [d for batch in batches for d in batch]
    # return ([D[idx] for idx in reverse_indices.tolist()], *returned_text)

# input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
# return self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)

## checkPoint.docFromText call ends here