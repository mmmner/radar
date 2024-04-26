# README

This repo provides the source codes & data for our paper, *"RADAR: Relation-Assisted Dual-graph Aligning Recognition for Grounded MNER"* .

## Overview
RADAR is proposed to efficiently utilize visual relations in scene graphs for cross-modal entity alignment and Grounded MNER reasoning. An auxiliary textual scene graph is introduced for explicit object-level alignment and bridging the modality gap. Through cross-modal dual-graph alignments assisted by relational features and structures, RADAR significantly eases the process of approaching the ground truth. The framework is shown in Figure 1.
![framework](figures/radar_framework.png)

<p align="center">
    Figure1: Our Framework of RADAR: Relation-Assisted Dual-graph Aligning Recognition. 
</p>

Our codes are modified from the baseline [GMNER](https://github.com/NUSTM/GMNER), please refer to their [README](https://github.com/NUSTM/GMNER/blob/main/README.md) for the dataset usage. 
## Dataset

The dataset released by [GMNER](https://github.com/NUSTM/GMNER)  is built on two benchmark MNER datasets, i.e., Twitter-15 (Zhang et al., 2018) and Twitter-17 (Yu et al., 2020). To get their respective results, we provide the split versions *Grounded Twitter2015* and *Grounded Twitter2017* using the GMNER dataset, needing only change the data directory into *twitter2015* or *twitter2017*  in `data/`. The original GMNER dataset(including text and image annotation)  are in `Twitter10000_v2.0`.

- The preprocessed CoNLL format files are provided in this repo. For each tweet, the first line is its image id, and the following lines are its textual contents.
- Step 1：Download each tweet's associated images via this link (<https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view>)
- Step 2:  Use [VinVL](https://github.com/pzzhang/VinVL) to identify all the candidate objects, their features and their concepts (classes).
- Step 3:  Use [IETrans](https://github.com/waxnkw/IETrans-SGG.pytorch) to extract all the relational triplets for each image with the candidate objects in Step 2 as input, and put them under the folder named "Twitter10000_IETrans" under `data/`. We provide the extracted scene graphs along with VinVL extracted objects' features and concepts via this link (<https://drive.google.com/file/d/1Hf2jSDSBydGyG4NSP-CFG4TFz7_9chn4/view?usp=sharing>) for reproduction convenience.

## Requirement

- pytorch 1.7.1
- transformers 3.4.0
- fastnlp 0.7.0


## Usage

### Training for RADAR

```
sh train.sh
```

### Evaluation

```
sh test.sh
```

## Acknowledgements

- Using the dataset means you have read and accepted the copyrights set by Twitter and original dataset providers.
- Our codes are based on the codes of  [GMNER](https://github.com/NUSTM/GMNER), thanks a lot!
