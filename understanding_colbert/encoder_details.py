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
baseColbert.name
baseColbert.model
baseColbert.raw_tokenizer
baseColbert.bert
class_factory(baseColbert.name)

## ColBERT
colBert = ColBERT(checkpoint, config)
colBert.use_gpu
colBert.colbert_config.mask_punctuation
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

checkPoint.queryFromText(["hello world"])
checkPoint.docFromText(["hello world"]).size()
