# The `ColBERTConfig` type.

Just like in the [settings.py](https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/infra/config/settings.py) file, we'll split up the config into further subtypes. So we'll have the following:

```julia
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
```

```python
import datetime
def timestamp(daydir=False):
    format_str = f"%Y-%m{'/' if daydir else '-'}%d{'/' if daydir else '_'}%H.%M.%S"
    result = datetime.datetime.now().strftime(format_str)
    return result
```