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

* COMET & BLEURT

```
python3 rebuttal.py \
    --hypo /path/to/fairseq.output \
    --target /path/to/tsv \
    --gpus 1 --bsz 1
```

> target tsv must contain "src_text" and "tgt_text"


* Significance test

```
python3 significance_test.py \
    --baseline /path/to/fairseq.output \
    --compare /path/to/fairseq.output \
    --label /path/to/tsv["tgt_text"] \
    --sample-size n \
    --num-samples n \
    --tokenized \
```

> note that you must extract "tgt_text" from tsv to use as input for "label" field