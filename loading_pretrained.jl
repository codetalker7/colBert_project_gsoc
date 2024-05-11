ENV["OPENAI_API_KEY"] = ""

using Flux, Transformers, CUDA
include("config.jl")

# paths and names
dataroot = "downloads/lotte"
dataset = "lifestyle"
datasplit = "dev"

nbits = 2   # encode each dimension with 2 bits
kmeans_niters = 20  # 20 iterations of kmeans
doc_maxlen = 300   # truncate passages at 300 tokens

checkpoint = "downloads/colbertv2.0"
index_root = "experiments/notebook/indexes"
index_name = "$dataset.$datasplit.$nbits"*"bits"
index_path = joinpath(index_root, index_name)

# setting up a config
config = ColBERTConfig(
    RunSettings(
        experiment="notebook",
        name="2024-03/29/16.34.17",
    ),
    TokenizerSettings(),
    ResourceSettings(
        checkpoint=checkpoint,
        collection="downloads/lotte/lifestyle/dev/collection.tsv",
        index_name=index_name,
    ),
    DocSettings(
        doc_maxlen=doc_maxlen,
    ),
    QuerySettings(),
    IndexingSettings(
        nbits=nbits,
        kmeans_niters=kmeans_niters,
    ),
    SearchSettings(),
)

# check if CUDA devices are available
CUDA.devices()

const PRETRAINED_MODEL = "downloads/colbertv2.0"                            # relative path to where the model is downloaded

colbert_config = Transformers.load_config(PRETRAINED_MODEL)
colbert_tokenizer = Transformers.load_tokenizer(PRETRAINED_MODEL)
colbert_model = Transformers.load_model(PRETRAINED_MODEL)

# sampling encoding and decoding of a text
text = ["this is some sentence which is huggy"]
tokens = Transformers.encode(colbert_tokenizer, text).token
decoded_text = Transformers.decode(colbert_tokenizer, tokens)
