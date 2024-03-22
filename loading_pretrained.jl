ENV["OPENAI_API_KEY"] = ""

using Flux, Transformers, CUDA

# check if CUDA devices are available
CUDA.devices()

const PRETRAINED_MODEL = "colbert-ir/colbertv2.0"

colbert_config = Transformers.load_config(PRETRAINED_MODEL)
colbert_tokenizer = Transformers.load_tokenizer(PRETRAINED_MODEL)
colbert_model = Transformers.load_model(PRETRAINED_MODEL)

# sampling encoding and decoding of a text
text = ["this is some sentence which is huggy"]
tokens = Transformers.encode(colbert_tokenizer, text).token
decoded_text = Transformers.decode(colbert_tokenizer, tokens)
