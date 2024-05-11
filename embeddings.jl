using Flux, Transformers, CUDA

# the documents and the batch size 
docs = [
    "hello world",
    "thank you!",
    "a",
    "this is some longer text, so length should be longer"
]
bsize = 2

# loading the model
## the huggingface checkpoint
const PRETRAINED_MODEL = "colbert-ir/colbertv2.0"

## get the underlying bert tokenizer, model and the linear layer
bert_tokenizer = Transformers.load_tokenizer(PRETRAINED_MODEL)      # this is the underlying tokenizer
bert_model = Transformers.load_model(PRETRAINED_MODEL)              # this is the underlying BERT model
                                                                    # where are these models saved