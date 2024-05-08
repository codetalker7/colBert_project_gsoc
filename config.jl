Base.@kwdef struct RunSettings
    root::String = joinpath(pwd(), "experiments")
    experiment::String = "default"
    index_root::String = nothing
    name::String = Dates.format(now(), "yyyy/mm/dd/HH.MM.SS")
    rank::Int = 0
    nranks::Int = 0
end

Base.@kwdef struct TokenizerSettings
    query_token_id::String = "[unused0]"
    doc_token_id::String = "[unused1]"
    query_token::String = "[Q]"
    doc_token::String = "[D]"
end

Base.@kwdef struct ResourceSettings
    checkpoint::String = nothing
    collection::String = nothing
    queries::String = nothing
    index_name::String = nothing
end

Base.@kwdef struct DocSettings
    dim::Int = 128
    doc_maxlen::Int = 220
    mask_punctuation::Bool = True
end

Base.@kwdef struct QuerySettings
    query_maxlen::Int = 32
    attent_to_mask_tokens::Bool = False
    interaction::String = "colbert"
end

Base.@kwdef struct IndexingSettings
    index_path::String = nothing
    index_bsize::Int = 64
    nbits::Int = 1
    kmeans_niters = 4
end

Base.@kwdef struct SearchSettings
    ncells::Int = nothing
    centroid_score_threshold::Float = nothing
    ndocs::Int = nothing
end

Base.@kwdef struct ColBERTConfig
    run
end