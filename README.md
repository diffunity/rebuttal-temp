# Rebuttal


## Install dependencies

* COMET

```
pip install unbabel-comet
```

* BLEURT

```
pip install --upgrade pip  # ensures that pip is current
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
```


## Usage

```
python3 rebuttal.py \
    --hypo /path/to/fairseq.output \
    --target /path/to/tsv \
    --gpus 1 --bsz 1
```

> target tsv must contain "src_text" and "tgt_text"