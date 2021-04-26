# Neural Compatibility Modeling with Attentive Knowledge Distillation

Code for paper [Neural Compatibility Modeling with Attentive Knowledge Distillation](https://dl.acm.org/doi/abs/10.1145/3209978.3209996?casa_token=BMlZ2NCCuoIAAAAA:SchBFektWuZFOXcQp8-n2uA6SCgFuDB-rTP5siIjl3AP9vORv44JIbo0ne1J9uxQV-lcUHtOKgNDqQ).

## Dependencies

This project currently requires

- Python2.7

- Theano0.9

We will also provide the Pytroch version with Python 3 in the near future.

## Data Preparation

### Experiment data

You can directily run the model with the experimental data that can be downloaded from [there](https://pan.baidu.com/s/1MG36wQSAAN9uQKYlC0_1iA) with code: ttul.

### Meta data

The FashionVC dataset can be download from [there](https://drive.google.com/open?id=1lO7M-jSWb25yucaW2Jj-9j_c9NqquSVF).

### Data process

The code of data processing is in the \data_process. The processed text data utilized for rule extraction is in the \data_process\processde_text.

## Citations

```
@inproceedings{song2018neural,
  title={Neural compatibility modeling with attentive knowledge distillation},
  author={Song, Xuemeng and Feng, Fuli and Han, Xianjing and Yang, Xin and Liu, Wei and Nie, Liqiang},
  booktitle={The 41st International ACM SIGIR Conference on Research \& Development in Information Retrieval},
  pages={5--14},
  year={2018}
}

@article{han2019neural,
  title={Neural compatibility modeling with probabilistic knowledge distillation},
  author={Han, Xianjing and Song, Xuemeng and Yao, Yiyang and Xu, Xin-Shun and Nie, Liqiang},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={871--882},
  year={2019},
  publisher={IEEE}
}
```
