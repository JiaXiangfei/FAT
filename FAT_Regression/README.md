
# FAT: Frequency-Aware Pretraining for Enhanced Time-Series Representation Learning (KDD'25)

![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg?style=plastic)
![PyTorch 2.4.1](https://img.shields.io/badge/PyTorch-2.4.1-%23EE4C2C.svg?style=plastic)
![CUDA 12.2](https://img.shields.io/badge/CUDA-12.2-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)
<a href="https://doi.org/10.5281/zenodo.15586685"><img src="https://zenodo.org/badge/995535628.svg" alt="DOI"></a>


The repo is the official implementation for the paper: "FAT: Frequency-Aware Pretraining for Enhanced Time-Series Representation Learning". 

We gratefully acknowledge [Informer](https://github.com/zhouhaoyi/Informer2020), [SimMTM](https://github.com/thuml/SimMTM), and [Autoformer](https://github.com/thuml/Autoformer) for sharing their code; we primarily structured our pre‑training and fine‑tuning procedures according to their settings.


## Architecture

<p align="center">
  <img src=".\figs\fig_main_0210.png" alt="Knowledge-Guided Frequency Reformer" width="80%" />
  <br><br>
  <b>Figure 1.</b> Overview of FAT model.
</p>

FAT learns consistent and generalizable frequency patterns from time-domain signals and encodes them into representations, eliminating the need for architectural adaptations or additional modules during inference. This is achieved through three key components: the knowledge-guided frequency reformer, frequency-invariant augmentation, and frequency-similarity constraints. 


## Get Started

1、Prepare Data And Code. 

All datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1CC4ZrUD4EKncndzgy5PSTzOPSqcuyqqj/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/a238e34ff81a42878d50/?dl=1), and arrange the folder as:

```
FAT/
|-- FAT_Regression/
    |-- dataset/
        |-- ETT-small/
            |-- ETTh1.csv
            |-- ETTh2.csv
            |-- ETTm1.csv
            |-- ETTm2.csv
        |-- weather/
            |-- weather.csv
        |-- ...
    |-- exp/
        |-- exp_basic.py
        |-- exp_fat.py
    |-- layers/
    |-- models/
        |-- FAT.py
        |-- ...
    |-- results/
        |-- 
    |-- scripts/
        |-- ETTh2.sh
        |-- ETTm1.sh
    |-- utils/
    run.py
|-- FAT_Classification/
    |-- code/
        |-- layers/
        |-- scripts/
        |-- utils/
        |-- dataloader.py
    |-- dateset/
        |--...
```

2、Pretrain And Finetune. 


We have provided predictive experimental coding in `FAT`, and the experimental scripts can be found in the `. /scripts` folder. To run the code on ETTh2, simply run the following command: 


```
FAT_Regression
# Pretrain-Finetune-ETTh2
bash ./scripts/ETTh2.sh
# Pretrain-Finetune-ETTm1
bash ./scripts/weather.sh

FAT_Classification
bash ./code/scripts/EMG.sh

```




## Main Results

<p align="center">
  <img src=".\FAT_Regression\figs\full_result.jpg" alt="" align=center />
  <br><br>
  <b>Figure 2.</b> Full result of FAT model.
</p>


## Citation
If you find this repo useful, please cite our paper.

```plain
@inproceedings{2025FAT,
  title={FAT: Frequency-Aware Pretraining for Enhanced Time-Series Representation Learning},
  author={Rui Cheng, Xiangfei Jia, Qing Li, Rong Xing, Jiwen Huang, Yu Zheng, Zhilong Xie},
  booktitle={Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (SIGKDD)},
  year={2025}
}
```

## Contact

If you have any questions, please contact the author [224081200057@smail.swufe.edu.cn](mailto:224081200057@smail.suwfe.edu.cn).

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Time-Series-Library


