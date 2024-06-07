# PSMGD: Periodic Stochastic Multi-Gradient Descent for Fast Multi-Objective Optimization

Official implementation of *PSMGD: Periodic Stochastic Multi-Gradient Descent for Fast Multi-Objective Optimization*.

## Supervised Learning

The performance is evaluated under 3 scenarios:

- Regression. The QM9 dataset contains 11 tasks, which can be downloaded automatically from Pytorch Geometric.
- Image Classification. The [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains 40 tasks and the Multi-MNIST dataset contains 2 tasks.
- Dense Prediction. The [NYU-v2](https://github.com/lorenmt/mtan) dataset contains 3 tasks and the [Cityscapes](https://github.com/lorenmt/mtan) dataset contains 2 tasks.

### Setup Environment

Following [Nash-MTL](https://github.com/AvivNavon/nash-mtl) and [FAMO](https://github.com/Cranial-XIX/FAMO), we implement our method with the `MTL` library.

First, create the virtual environment:

```
conda create -n mtl python=3.9.7
conda activate mtl
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113
```

Then, install the repo:

```
git clone https://anonymous.4open.science/r/PSMG-CFE3
cd PSMG
python -m pip install -e .
```

### Run Experiment

The dataset by default should be put under `experiments/EXP_NAME/dataset/` folder where `EXP_NAME` is chosen from `{celeba, cityscapes, nyuv2, quantum_chemistry}`. To run the experiment:

```
cd experiments/EXP_NAME
sh run.sh
```
