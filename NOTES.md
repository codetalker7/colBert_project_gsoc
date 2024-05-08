# The `ColBERTConfig` type.

Just like in the [settings.py](https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/infra/config/settings.py) file, we'll split up the config into further subtypes. Look in the file `config.jl` for the struct definitions.

```python
import datetime
def timestamp(daydir=False):
    format_str = f"%Y-%m{'/' if daydir else '-'}%d{'/' if daydir else '_'}%H.%M.%S"
    result = datetime.datetime.now().strftime(format_str)
    return result
```