# transformer-sequential

This repo contains the code for three papers:

- Feedback Transformer
- Expire-Span
- Staircase Transformer

The training code is structured for long sequential modeling with Transformer-like architectures.

## Requirements

You will need a CUDA-enabled GPU to run the code.

## Setup

Run the following:

```
pip install -r requirements.txt
```

## Feedback Transformer

Introduced in [Addressing Some Limitations of Transformers with Feedback Memory](https://arxiv.org/abs/2002.09402v3).

### Running Experiments from the Paper

#### enwik8

|Model|Params|Valid|Test|
|-|-|-|-|
|Feedback Transformer|77M|0.984|0.962|

_Numbers are Bits-Per-Character_

```
bash experiments/feedback/enwik8.sh
```

#### Algorithmic

|Model|3 Variable|5 Variable|
|-|-|-|
|Transformer|33.7|37.5|
|Feedback Transformer|99.1|92.6|

_Numbers are % Accuracy on Test_

```
bash experiments/feedback/algorithmic_3var.sh
bash experiments/feedback/algorithmic_5var.sh
```

## Expire-Span

Introduced in [Not All Memories are Created Equal: Learning to Expire](https://ai.facebook.com/research/publications/not-all-memories-are-created-equal).

### Running Experiments from the Paper

#### enwik8

|Model|Params|Valid|Test|
|-|-|-|-|
|Expire-Span 12L|38M|1.014|0.994|

_Numbers are Bits-Per-Character_

```
bash experiments/expire_span/enwik8.sh
```

#### Object Collision

|Model|Maximum Span|Test Error (%)|
|-|-|-|
|Expire-Span|16k|52.2|
|Expire-Span|32k|36.7|
|Expire-Span|64k|26.7|

```
bash experiments/expire_span/object_collision_16k.sh
bash experiments/expire_span/object_collision_32k.sh
bash experiments/expire_span/object_collision_64k.sh
```

## Staircase

Introduced in [Staircase Attention for Recurrent Processing of Sequences](https://arxiv.org/pdf/2106.04279.pdf).
Note this algorithmic task in this repo is slightly different from what was used in the paper, while the number might not exactly match, it does show the same trend as in the paper. And the model implementation / hyperparameter remains the same.

### Running Experiments from the Paper

#### Algorithmic

|Model|Test|
|-|-|
|Transformer|58.44%|
|Staircase Transformer| 3.6%|

_Numbers are % error rate on Test_

```
bash experiments/staircase/algorithmic_3var.sh
```

## License

The code is licensed under CC-BY-NC license. See the LICENSE file for more details.
