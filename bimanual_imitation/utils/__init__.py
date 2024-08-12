import argparse
import os
from enum import Enum

import numpy as np


def get_cpus(target_num_cpus):
    max_num_cpus = os.cpu_count()
    if max_num_cpus is None:
        max_num_cpus = 1

    # Leave some for the system :)
    num_cpus = np.clip(target_num_cpus, 1, max(int(max_num_cpus * 0.75), 1))
    return int(num_cpus)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def get_enum_value(key, enum: Enum):
    assert key in [x.value for x in enum], f'Value "{key}" not part of {enum}'
    for match_key in enum:
        if key == match_key.value:
            return match_key


def download_covering_array(k, t=3, v=2):
    import urllib
    import zipfile
    from pathlib import Path

    import requests

    file_name = f"ca.{t}.{v}^{k}.txt"  # url.split("/")[-1]
    assert t > 1
    assert v > 1 and v < 7
    url = f"https://math.nist.gov/coveringarrays/ipof/cas/t={t}/v={v}/{urllib.parse.quote(file_name.encode('utf8'))}.zip"
    file_path = Path(file_name)

    if file_path.exists():
        print(f"{file_name} already exists.")
        return np.genfromtxt(file_path, skip_header=True)

    response = requests.get(url)
    zip_file_path = Path(f"{file_name}.zip")
    zip_file_path.write_bytes(response.content)
    print(f"{zip_file_path.name} downloaded.")

    with zipfile.ZipFile(zip_file_path.as_posix(), "r") as zip_ref:
        zip_ref.extractall()

    zip_file_path.unlink()
    covering_array = np.genfromtxt(file_path, skip_header=True)
    file_path.unlink()
    return covering_array
