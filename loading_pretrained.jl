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

# loading the bert model
const PRETRAINED_BERT = "colbert-ir/colbertv2.0"                            # relative path to where the model is downloaded

bert_config = Transformers.load_config(PRETRAINED_BERT)
bert_tokenizer = Transformers.load_tokenizer(PRETRAINED_BERT)
bert_model = Transformers.load_model(PRETRAINED_BERT)

# loading the linear layer
using Pickle
layers = Pickle.Torch.THload(joinpath(config.resource_settings.checkpoint, "pytorch_model.bin"))
linear_layer = Dense(Matrix(layers["linear.weight"]))

