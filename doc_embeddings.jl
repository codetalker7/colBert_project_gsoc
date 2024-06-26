include("loading_pretrained.jl")
using Test

# check if CUDA devices are available
CUDA.devices()

# some utility functions
"""
    _sort_by_length(ids::AbstractMatrix, mask::AbstractMatrix, bsize::Int)
    
Sort sentences by number of attended tokens, if the number of sentences is larger than bsize. If the number of passages (first dimension of `ids`) is atmost
than `bsize`, the `ids`, `mask`, and a list `Vector(1:size(ids)[1])` is returned as a three-tuple. Otherwise,
the passages are first sorted by the number of attended tokens (figured out from `mask`), and then the the sorted arrays
`ids` and `mask` are returned, along with a reversed list of indices, i.e a mapping from passages to their indice in the sorted list.
"""
function _sort_by_length(integer_ids::AbstractMatrix, integer_mask::AbstractMatrix, bsize::Int)
    batch_size = size(integer_ids)[2]
    if batch_size <= bsize
        # if the number of passages fits the batch size, do nothing
        return integer_ids, integer_mask, Vector(1:batch_size)
    end

    lengths = vec(sum(integer_mask; dims = 1))              # number of attended tokens in each passage
    indices = sortperm(lengths)                     # get the indices which will sort lengths
    reverse_indices = sortperm(indices)             # invert the indices list
    
    integer_ids[:, indices], integer_mask[:, indices], reverse_indices
end

"""
    _split_into_batches(integer_ids::AbstractArray, integer_mask::AbstractMatrix, bsize::Int)::Vector{Tuple{AbstractArray, AbstractMatrix, Int}}

Split the given `integer_ids` and `integer_mask` into batches of size `bsize`.
"""
function _split_into_batches(integer_ids::AbstractArray, integer_mask::AbstractMatrix, bsize::Int)
    batch_size = size(integer_ids)[2]
    batches = Vector{Tuple{AbstractArray, AbstractMatrix}}()
    for offset in 1:bsize:batch_size
        push!(batches, (integer_ids[:, offset:min(batch_size, offset + bsize - 1)], integer_mask[:, offset:min(batch_size, offset + bsize - 1)]))
    end
    batches
end

function get_mask(integer_ids::AbstractArray, skiplist::Vector{Int})
    filter = token_id -> !(token_id in skiplist) && token_id != TextEncodeBase.lookup(bert_tokenizer.vocab, bert_tokenizer.padsym)
    filter.(integer_ids)
end

function doc(integer_ids::AbstractArray, integer_mask::AbstractArray, skiplist::Vector{Int})
    # run the batch through bert and the linear layer
    # integer_ids, integer_mask = Flux.gpu(integer_ids), Flux.gpu(integer_mask)
    D = bert_model((token=integer_ids, attention_mask=NeuralAttentionlib.GenericSequenceMask(integer_mask))).hidden_state
    D = linear_layer(D)

    # mask out any punctuations 
    mask = get_mask(integer_ids, skiplist)
    mask = reshape(mask, (1, size(mask)...))                    # equivalent of unsqueeze

    D = D .* mask                                               # clear out embeddings of masked tokens
    D = mapslices(v -> iszero(v) ? v : normalize(v), D, dims = 1)                 # normalize each embedding
    return D, mask
end

# the documents and the batch size 
docs = [
    "hello world",
    "thank you!",
    "a",
    "this is some longer text, so length should be longer",
]
bsize = 2

# converting docs to an embeddings tensor; the tensor will be of shape (total_num_attended_tokens, 128), where
# total_num_attended_tokens is the total number of attended tokens across all the docs

## adding a placeholder for the [D] marker in all docs
batch_text = [". " * doc for doc in docs]

## converting all docs into token ids with some options
using Transformers.TextEncoders
using OneHotArrays
using NeuralAttentionlib
using TextEncodeBase
VOCABSIZE = size(bert_tokenizer.vocab.list)[1]

encoded_text = encode(bert_tokenizer, batch_text)
## for some reason, all ids are 1 more than the corresponding python ids
## but this makes sense, since julia indices start from 1!
ids, mask = encoded_text.token, encoded_text.attention_mask
## ids has size (vocab_size, max_tokens_in_passage, num_passages)

integer_ids = reinterpret(Int32, ids)
@test isequal(bert_model((token = integer_ids, attention_mask=mask)), bert_model((token = ids, attention_mask=mask)))

## to revert integer_ids to one-hot encoding, we can do onehotbatch(integer_ids, 1:size(bert_tokenizer.vocab.list)[1])
## onehotbatch doesn't work! Transformers uses it's own OneHotArray (see PrimitiveOneHot.jl). So we need to do
## integer_ids = reinterpret(Int32, ids) and TextEncodeBase.OneHotArray{VOCABSIZE}(integer_ids) to convert the encodings!
converted_ids = TextEncodeBase.OneHotArray{VOCABSIZE}(integer_ids)
@test isequal(ids, converted_ids)
@test isequal(bert_model((token = converted_ids, attention_mask=mask)), bert_model((token = integer_ids, attention_mask=mask)))
integer_mask = NeuralAttentionlib.getmask(mask, ids)
integer_mask = integer_mask[1, :, :]
@test isequal(bert_model((token = integer_ids, attention_mask=NeuralAttentionlib.GenericSequenceMask(integer_mask))).hidden_state, bert_model((token = ids, attention_mask=mask)).hidden_state)

## add the [D] marker token id
D_marker_token_id = 3
integer_ids[2, :] .= D_marker_token_id
ids = TextEncodeBase.OneHotArray{VOCABSIZE}(integer_ids)
for i in 1:size(docs)[1]
    @test isequal(ids[:, 2, i], onehot(D_marker_token_id, 1:VOCABSIZE))
end

integer_ids, integer_mask, reverse_indices = _sort_by_length(integer_ids, integer_mask, bsize)
batches = _split_into_batches(integer_ids, integer_mask, bsize)

# writing the checkpoint.doc function
text_batches, reverse_indices = batches, reverse_indices

# ## for now, we ignore text_batches; implementing the following code for batches is straightforward
# bert_model = Flux.gpu(bert_model)
# linear_layer = Flux.gpu(linear_layer)
# integer_ids, integer_mask = Flux.gpu(integer_ids), Flux.gpu(integer_mask)
# mask = Flux.gpu(mask)
#
# ## moving one hot ids to gpu and then using bert throws an error!
# D = bert_model((token = integer_ids, attention_mask = mask)).hidden_state
# D = linear_layer(D)

## creating skiplist to skip punctuation
punctuation_list = string.(collect("!\"#\$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"))
skiplist = [TextEncodeBase.lookup(bert_tokenizer.vocab, punct) for punct in punctuation_list]

# ## trying to run doc function on a single batch
# single_input_ids = copy(batches[1][1])
# single_attention_mask = copy(batches[1][2])
# doc(single_input_ids, single_attention_mask, skiplist)          # it works!

# get all the normalized embeddings + masks (after removing tokens in skiplist) for each batch
batches = [doc(input_ids, attention_mask, skiplist) for (input_ids, attention_mask) in text_batches]

# aggregate all the embeddings
D, mask = [], []
for (_D, _mask) in batches
    push!(D, _D)
    push!(mask, _mask)
end

# concat embeddings and masks, and put them in the original order
D, mask = cat(D..., dims = 3)[:, :, reverse_indices], cat(mask..., dims = 3)[:, :, reverse_indices]
mask = reshape(mask, size(mask)[2:end])

# get doclens, i.e number of attended tokens for each passage
doclens = sum(mask, dims = 1) 

# flatten out embeddings, i.e get embeddings for each token in each passage
D = reshape(D, size(D)[1], prod(size(D)[2:end]))

# remove embeddings for masked tokens
D = D[:, reshape(mask, prod(size(mask)))]
