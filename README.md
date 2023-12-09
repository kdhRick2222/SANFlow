# SANFlow: Semantic-Aware Normalizing Flow for Anomaly Detection and Localization

This repository contains the official implementation for SANFlow introduced in the following paper:

[Paper](https://openreview.net/pdf?id=BqZ70BEtuW)

Authors: Daehyun Kim, Sungyong Baik, Tae Hyun Kim

### Abstract
Visual anomaly detection, the task of detecting abnormal characteristics in images, is challenging due to the rarity and unpredictability of anomalies. In order to reliably model the distribution of normality and detect anomalies, a few works have attempted to exploit the density estimation ability of normalizing flow (NF). However, previous NF-based methods forcibly transform the distribution of all features into a single distribution (e.g., unit normal distribution), even when the features can have locally distinct semantic information and thus follow different distributions. We claim that forcibly learning to transform such diverse distributions to a single distribution with a single network will cause the learning difficulty, thereby limiting the capacity of a network to discriminate between normal and abnormal data. As such, we propose to transform the distribution of features at each location of a given input image to different distributions. Specifically, we train NF to map the feature distributions of normal data to different distributions at each location in the given image. Furthermore, to enhance the discriminability, we also train NF to map the distribution of abnormal data to a distribution significantly different from that of normal data. The experimental results highlight the efficacy of the proposed framework in improving the density modeling and thus anomaly detection performance.

### Framework Overview
"https://github.com/kdhRick2222/SANFlow/assets/62320935/23a1085e-6da5-42bc-88d8-ea12398ab6e4"

### Running the code

**0. Preliminaries**

- For `train_liif.py` or `test.py`, use `--gpu [GPU]` to specify the GPUs (e.g. `--gpu 0` or `--gpu 0,1`).

- For `train_liif.py`, by default, the save folder is at `save/_[CONFIG_NAME]`. We can use `--name` to specify a name if needed.

- For dataset args in configs, `cache: in_memory` denotes pre-loading into memory (may require large memory, e.g. ~40GB for DIV2K), `cache: bin` denotes creating binary files (in a sibling folder) for the first time, `cache: none` denotes direct loading. We can modify it according to the hardware resources before running the training scripts.

**1. DIV2K experiments**

**Train**: `python train_liif.py --config configs/train-div2k/train_edsr-baseline-liif.yaml` (with EDSR-baseline backbone, for RDN replace `edsr-baseline` with `rdn`). We use 1 GPU for training EDSR-baseline-LIIF and 4 GPUs for RDN-LIIF.

**Test**: `bash scripts/test-div2k.sh [MODEL_PATH] [GPU]` for div2k validation set, `bash scripts/test-benchmark.sh [MODEL_PATH] [GPU]` for benchmark datasets. `[MODEL_PATH]` is the path to a `.pth` file, we use `epoch-last.pth` in corresponding save folder.

**2. celebAHQ experiments**

**Train**: `python train_liif.py --config configs/train-celebAHQ/[CONFIG_NAME].yaml`.

**Test**: `python test.py --config configs/test/test-celebAHQ-32-256.yaml --model [MODEL_PATH]` (or `test-celebAHQ-64-128.yaml` for another task). We use `epoch-best.pth` in corresponding save folder.

### Citation
If you find our work useful in your research, please cite:

```
@inproceedings{Dae2023sanflow,
  title={SANFlow: Semantic-Aware Normalizing Flow for Anomaly Detection and Localization},
  author={Daehyun Kim, Sungyong Baik, Tae Hyun Kim},
  booktitle={In Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```
