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

## docFromText details
