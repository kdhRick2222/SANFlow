# SANFlow: Semantic-Aware Normalizing Flow for Anomaly Detection and Localization (NeurIPS 2023)

This repository contains the official implementation for SANFlow introduced in the following paper:

[Paper](https://openreview.net/pdf?id=BqZ70BEtuW)

Authors: Daehyun Kim, Sungyong Baik, Tae Hyun Kim

### Abstract
Visual anomaly detection, the task of detecting abnormal characteristics in images, is challenging due to the rarity and unpredictability of anomalies. In order to reliably model the distribution of normality and detect anomalies, a few works have attempted to exploit the density estimation ability of normalizing flow (NF). However, previous NF-based methods forcibly transform the distribution of all features into a single distribution (e.g., unit normal distribution), even when the features can have locally distinct semantic information and thus follow different distributions. We claim that forcibly learning to transform such diverse distributions to a single distribution with a single network will cause the learning difficulty, thereby limiting the capacity of a network to discriminate between normal and abnormal data. As such, we propose to transform the distribution of features at each location of a given input image to different distributions. Specifically, we train NF to map the feature distributions of normal data to different distributions at each location in the given image. Furthermore, to enhance the discriminability, we also train NF to map the distribution of abnormal data to a distribution significantly different from that of normal data. The experimental results highlight the efficacy of the proposed framework in improving the density modeling and thus anomaly detection performance.

### Framework overview
<img width="731" alt="framework" src="https://github.com/kdhRick2222/SANFlow/assets/62320935/08d581c5-b9cb-48e7-81b3-77e742a8b3ee">

### Pre-trained Weight
[Link](https://hyu-my.sharepoint.com/:f:/g/personal/daehyun_hanyang_ac_kr/EumuRFjby9NIpX5sOpNXDjIBDoG5rlMiyz6Tg-DkQGKjPA?e=Q0fklC)


### Running the code

Train:

`python main.py --gpu 0 --inp 256 --lr 5e-4 --meta-epochs 20 --sub-epoch 4 --class-name $category -bs 4 -pl 3 --pro`

`python main.py --gpu 0 --inp 320 --lr 5e-4 --meta-epochs 20 --sub-epoch 4 --dataset mvtec --class-name $category -bs 4 -pl 3 --pro -enc 'wide_resnet101_2'`

Test:

`python main.py --gpu 0 --dataset mvtec --inp 256 --action-type norm-test --class-name $category --checkpoint './weights_WRN50/'$category'.pt' --pro --viz`

`python main.py --gpu 0 --dataset mvtec --inp 320 --action-type norm-test --class-name $category --checkpoint './weights_WRN101/'$category'.pt' -enc 'wide_resnet101_2' --pro --viz`

Run:`sh sanflow.sh`


### Source
The code is heavily borrowed from [CFLOW-AD](https://github.com/gudovskiy/cflow-ad).

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
