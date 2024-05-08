include("config.jl")

## let's play arond with Transformers.jl HuggingFace configs first
using Flux, Transformers, CUDA

# check if CUDA devices are available
CUDA.devices()

const PRETRAINED_MODEL = "colbert-ir/colbertv2.0"

# loading the model
bert_config = Transformers.load_config(PRETRAINED_MODEL)             # the underlying BERT config
bert_tokenizer = Transformers.load_tokenizer(PRETRAINED_MODEL)       # this is the underlying tokenizer
bert_model = Transformers.load_model(PRETRAINED_MODEL)               # this is the underlying BERT model

# testing out the tokenizer
using Transformers.TextEncoders
mask_token = encode(bert_tokenizer, "[MASK]").token
decode(bert_tokenizer, mask_token)                                   # works great! output is "[CLS][MASK][SEP]"