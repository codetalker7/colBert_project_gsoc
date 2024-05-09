import Pkg; Pkg.activate("."); Pkg.add("Transformers"); Pkg.add("Pickle"); Pkg.add("Flux");

using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace
using Flux
using Pickle


function copy_with_modification(original::T, field_to_change, new_value) where {T}
  val(field) = field == field_to_change ? new_value : getfield(original, field)
  T(val.(fieldnames(T))...)
end


const query_marker_token = 2
const document_marker_token = 3
const cls_token = 102
const sep_token = 103
const mask_token = 104
const pad_token  = 1


struct ColBERT
  bert::Transformers.HuggingFace.HGFBertModel
  linear::Dense
end

function l2_normalization(x::Array)
  l2_norm = sqrt.(sum(x.^2, dims=1))

  return x./l2_norm
end

function encode_query(query::AbstractString)
    query = ". " * query
    original_tokens = encode(textencoder, query).token
    tokens = copy_with_modification(original_tokens, :onehots, setindex!(original_tokens.onehots, query_marker_token, 2))
    return (; token = tokens)
end

function encode_document(doc::AbstractString)
    doc = ". " * doc
    original_tokens = encode(textencoder, doc).token
    tokens = copy_with_modification(original_tokens, :onehots, setindex!(original_tokens.onehots, document_marker_token, 2))
    return (; token = tokens)
end

function (m::ColBERT)(query_ids, doc_ids)
    Q = m.bert(query_ids).hidden_state
    Q = m.linear(Q)
    Q = l2_normalization(Q)
    Q_unsqueezed = reshape(Q, size(Q)[1], 1, size(Q)[2], size(Q)[3])

    D = m.bert(doc_ids).hidden_state
    D = m.linear(D)
    D = l2_normalization(D)
    D_unsqueezed = reshape(D, size(D)[1], size(D)[2], 1, size(D)[3])


    squared_diff =  (Q_unsqueezed .- D_unsqueezed).^2
    summed_squared_diff = 2 .- sum(squared_diff, dims=1)

    result =  summed_squared_diff
    max_values = maximum(result, dims=2)

    return sum(max_values, dims=3)[1]
end


textencoder, bert_model = hgf"colbert-ir/colbertv2.0"
colbert_parameters = Pickle.Torch.THload("pytorch_model.bin")
linear_layer = Dense(Matrix(colbert_parameters["linear.weight"]))

model = ColBERT(bert_model, linear_layer)


query = "What is Julia language"
documents=["Julia is a greate language",
           "I don't know",
           "Julia is my sister; she helpes with cleaning my room",
           "Harry Potter is a series of seven fantasy novels written by J. K. Rowling"]

scores = zeros(length(documents))
encoded_query = encode_query(query)
encoded_documens = [encode_document(doc) for doc in documents]

for (i, encoded_doc) in enumerate(encoded_documens)
    scores[i] = model(encoded_query, encoded_doc)
end

document_pairs = collect(zip(1:length(scores), scores))

document_order = sort(document_pairs, by = x -> x[2], rev = true)[:, 1]
ordered_documents = [(documents[i], score) for (i, score) in document_order]

display(ordered_documents)