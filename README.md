# Observe before Generate: Emotion-Cause aware Video Caption for Multimodal Emotion Cause Generation in Conversations

***Fanfan Wang***, ***Heqing Ma***, ***Xiangqing Shen***, ***Jianfei Yu***\*, ***Rui Xia***\*

[![Dataset](https://img.shields.io/badge/Dataset-🤗_Hugging_Face-F0A336)](https://huggingface.co/datasets/NUSTM/ECGF) [![Conference](https://img.shields.io/badge/Paper-ACMMM_2024-4C9172)](https://doi.org/10.1145/3664647.3681601) [![Conference](https://img.shields.io/badge/Submission-OpenReview-802819)](https://openreview.net/forum?id=Pq63G43jkQ) 

This repository contains the code for **ObG**, a multimodal pipeline framework that first generates emotion-cause aware video captions (Observe) and then facilitates the generation of emotion causes (Generate).

<img src="framework.png" alt="overview" width="900"/>


## Task

**Multimodal Emotion Cause Generation in Conversations (MECGC)** aims to generate the abstractive causes of given emotions based on multimodal context.

<img src="task.png" alt="task" width="500"/>





## Dataset

[**ECGF**](https://huggingface.co/datasets/NUSTM/ECGF) is constructed by manually annotating the abstractive causes for each emotion labeled in the existing [ECF](https://huggingface.co/datasets/NUSTM/ECF) dataset.


## Requirements

```
conda create -n obg python=3.7
conda activate obg
pip install -r requirements.txt
```


## Usage


## Citation

```
@inproceedings{wang2024obg,
  title={Observe before Generate: Emotion-Cause aware Video Caption for Multimodal Emotion Cause Generation in Conversations},
  author={Wang, Fanfan and Ma, Heqing and Shen, Xiangqing and Yu, Jianfei and Xia, Rui},
  booktitle={Proceedings of the 32st ACM International Conference on Multimedia},
  pages={},
  year={2024}
}

@ARTICLE{ma2024monica,
  author={Ma, Heqing and Yu, Jianfei and Wang, Fanfan and Cao, Hanyu and Xia, Rui},
  journal={IEEE Transactions on Affective Computing}, 
  title={From Extraction to Generation: Multimodal Emotion-Cause Pair Generation in Conversations}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi={10.1109/TAFFC.2024.3446646}
}
```

## Acknowledgements

Our code benefits from [VL-T5](https://github.com/j-min/VL-T5) and [CICERO](https://github.com/declare-lab/CICERO/blob/fe728706e6faf0a1a4511e56180951174408c870/v1/experiments/nlg/evaluate.py). We appreciate their valuable contributions.
