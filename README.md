# OpenSTARLab Event Modeling package
[![Documentation Status](https://readthedocs.org/projects/openstarlab/badge/?version=latest)](https://openstarlab.readthedocs.io/en/latest/?badge=latest)
[![dm](https://img.shields.io/pypi/dm/openstarlab-event)](https://pypi.org/project/openstarlab-event/)
[![ArXiv](https://img.shields.io/badge/ArXiv-2502.02785-b31b1b?logo=arxiv)](https://arxiv.org/abs/2502.02785)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/yDrcywCs)

## Introduction
The OpenSTARLab Event package is the fundamental package for event modeling. It is designed to provide a simple and efficient way to train, inference, and simulate events. This package supports the data preprocessed by the OpenSTARLab PreProcessing package.

This package is continuously evolving to support future OpenSTARLab projects. If you have any suggestions or encounter any bugs, please feel free to open an issue.

<details>
<summary><strong>Soccer Event Modeling</strong></summary>

### Table: Comparison of model performance on soccer event prediction

**Note**: Arrows indicate whether a higher (↑) or lower (↓) value is better.   
Models are ranked by publication year. Bold values indicate the best performance (unrounded).
For more details refer to our paper [![ArXiv](https://img.shields.io/badge/ArXiv-2502.02785-b31b1b?logo=arxiv)](https://arxiv.org/abs/2502.02785)

#### Wyscout Dataset

| **Model (Year)** | **Action Acc. ↑** | **Action F1 ↑** | **Time-MAE ↓** | **X-MAE ↓** | **Y-MAE ↓** | **FLOPs** | **Num Params** |
|------------------|-------------------|------------------|----------------|-------------|-------------|-----------|----------------|
| MAJ              | 0.57              | 0.08             | 3.60           | 18.97       | 52.55       | -         | -              |
| Seq2Event (2022) | 0.67              | 0.16             | 3.41           | 7.11        | 15.72       | 112M      | 135K           |
| NMSTPP (2023)    | 0.67              | 0.17             | 3.34           | **6.94**    | **15.08**   | 296M      | 121K           |
| LEM_1 (2024)     | **0.67**          | 0.17             | 3.07           | 8.34        | 21.44       | 50M       | 98K            |
| LEM_3 (2024)     | 0.67              | **0.20**         | **2.69**       | 7.62        | 21.83       | 20M       | 39K            |
| FMS (2024)       | 0.67              | 0.16             | 3.27           | 11.27       | 24.19       | 930M      | 782K           |

#### StatsBomb Dataset

| **Model (Year)** | **Action Acc. ↑** | **Action F1 ↑** | **Time-MAE ↓** | **X-MAE ↓** | **Y-MAE ↓** | **FLOPs** | **Num Params** |
|------------------|-------------------|------------------|----------------|-------------|-------------|-----------|----------------|
| MAJ              | 0.40              | 0.06             | 2.76           | 20.72       | 33.32       | -         | -              |
| Seq2Event (2022) | 0.65              | 0.23             | 2.43           | 7.22        | 6.86        | 4.03B     | 413K           |
| NMSTPP (2023)    | 0.65              | 0.23             | 2.53           | 7.38        | **6.86**    | 2.02B     | 217K           |
| LEM_1 (2024)     | 0.65              | 0.24             | 2.23           | 7.36        | 8.21        | 66M       | 128K           |
| LEM_3 (2024)     | **0.66**          | **0.25**         | **2.07**       | **7.07**    | 8.32        | 19M       | 38K            |
| FMS (2024)       | 0.65              | 0.24             | 2.35           | 7.77        | 8.82        | 3.66B     | 1.29M          |


</details>

## Installation
- Install [pytorch](https://pytorch.org/get-started/locally/) (recommended version 2.4.0 linux pip python3.8 cuda12.1)
```
pip install torch torchvision torchaudio
```
- To install this package via PyPI
```
pip install openstarlab-event
```
- To install manually
```
git clone git@github.com:open-starlab/Event.git
cd ./Event
pip install -e .
```

## Current Features
### Sports
- [Event Model in Football/Soccer ⚽](https://openstarlab.readthedocs.io/en/latest/Event_Modeling/Sports/Soccer/index.html)

## RoadMap
- [x] Release the package
- [ ] Provide pre-trained models

## Other Information
Development torch version
```
version 2.4.0 linux pip python3.8 cuda12.1 
```

## Developer
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
<!-- [![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-) -->
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/calvinyeungck"><img src="https://github.com/calvinyeungck.png" width="100px;" alt="Calvin Yeung"/><br /><sub><b>Calvin Yeung</b></sub></a><br /><a href="#Developer-CalvinYeung" title="Lead Developer">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/keisuke198619"><img src="https://github.com/keisuke198619.png" width="100px;" alt="Keisuke Fujii"/><br /><sub><b>Keisuke Fujii</b></sub></a><br /><a href="#lead-KeisukeFujii" title="Team Leader">🧑‍💻</a></td>
    </tr>
  </tbody>
</table>
