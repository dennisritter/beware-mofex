import numpy as np
from pathlib import Path
from mofex.preprocessing.sequence import Sequence


def load_seqs_asf_amc(root, regex_str_asf, regex_str_amc):
    seqs = []
    print(f'Loading sequences from:\nroot: {root}\nASF pattern: {regex_str_asf}\nAMC pattern: {regex_str_amc}')
    for amc_path in Path(root).rglob(regex_str_amc):
        class_dir = '/'.join(str(amc_path).split("\\")[:-1])
        amc_file = str(amc_path).split("\\")[-1]
        asf_file = f'{amc_file[0:6]}.asf'
        asf_path = class_dir + '/' + asf_file
        seqs.append(Sequence.from_hdm05_asf_amc_files(asf_path=asf_path, amc_path=amc_path, name=amc_file, desc=class_dir.split('/')[-1]))
        print(f'loaded: {seqs[-1].name} -> {seqs[-1].desc}')
    return seqs