import argparse
from comet import download_model, load_from_checkpoint

import csv
from pathlib import Path
import zipfile
from functools import reduce
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Union
import io

import pandas as pd

from evaluate import load

def load_df_from_tsv(path: Union[str, Path]) -> pd.DataFrame:
    _path = path if isinstance(path, str) else path.as_posix()
    return pd.read_csv(
        _path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypo")
    parser.add_argument("--target")
    parser.add_argument("--tokenized", action="store_true", default=False)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--bsz", type=int)

    args = parser.parse_args()

    return args

def read_fairseq_output(file, args, filter_):
    fairseq_output = [i.strip() for i in open(file, "r")]
    fairseq_hypo = [i for i in fairseq_output if i.startswith(filter_)]
    fairseq_hypo = [(int(i.split("\t")[0].split("-")[-1]), i.split("\t")[-1]) for i in fairseq_hypo]
    fairseq_hypo = sorted(fairseq_hypo)

    return fairseq_hypo

def format_for_comet(srcs, mts, refs):
    ret = []
    for src, mt, ref in zip(srcs, mts, refs):
        ret.append(
            {"src": src, "mt": mt, "ref":ref}
        )
    return ret


def main():
    COMET_PATH = download_model("Unbabel/wmt22-comet-da")
    COMET = load_from_checkpoint(COMET_PATH)
    args = parse_args()

    hypo = read_fairseq_output(args.hypo, args, "D-" if args.tokenized else "H-")
    tgt_text = read_fairseq_output(args.hypo, args, "T-")
    tgt2hypo = {i[1]:j[1] for i,j in zip(tgt_text, hypo)}
    df = load_df_from_tsv(args.target)
    tgt2src = {i:j for i,j in zip(df['tgt_text'], df['src_text'])}

    src_text = []
    hypo = []
    tgt_text = []
    for tgt in tgt2hypo:
        if tgt in tgt2src:
            src_text.append(tgt2src[tgt])
            hypo.append(tgt2hypo[tgt])
            tgt_text.append(tgt)

    COMET_INPUT = format_for_comet(src_text, hypo, tgt_text)
    COMET_OUTPUT = COMET.predict(COMET_INPUT, batch_size=args.bsz, gpus=args.gpus)
    print("COMET SCORE: ", COMET_OUTPUT.system_score)

    BLEURT = load("bleurt", module_type="metric")
    BLEURT_OUTPUT = BLEURT.compute(predictions=hypo, references=tgt_text)
    print("BLEURT SCORE: ", sum(BLEURT_OUTPUT['scores']) / len(BLEURT_OUTPUT['scores']))


    
if __name__=="__main__":
    main()







# # Choose your model from Hugging Face Hub
# model_path = download_model("Unbabel/XCOMET-XL")
# # or for example:
# # model_path = download_model("Unbabel/wmt22-comet-da")

# # Load the model checkpoint:
# model = load_from_checkpoint(model_path)

# # Data must be in the following format:
# data = [
#     {
#         "src": "10 到 15 分钟可以送到吗",
#         "mt": "Can I receive my food in 10 to 15 minutes?",
#         "ref": "Can it be delivered between 10 to 15 minutes?"
#     },
#     {
#         "src": "Pode ser entregue dentro de 10 a 15 minutos?",
#         "mt": "Can you send it for 10 to 15 minutes?",
#         "ref": "Can it be delivered between 10 to 15 minutes?"
#     }
# ]
# # Call predict method:
# model_output = model.predict(data, batch_size=1, gpus=0)


# # fairseq tsv data = "/home/data/MUSTC/en-es-cress/en-es/dev.tsv"
# # output = ""
# en2es = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-es.single_model')
# en2es.translate('Hello world', beam=5)