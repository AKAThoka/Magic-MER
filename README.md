<h1 align="center">MAGIC-MER</h1>

<h3 align="center">Multi-agent Game-theoretic Consensus Framework for Open-Vocabulary Multimodal Emotion Recognition</h3>

[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE)

## 🚀 Data and Configuration

Please refer to AffectGPT for dataset preparation. For stable execution, more than 100 GB of storage is recommended. Environment setup is described in `environment.yml`.

Before game-theoretic training, train and configure the required models in advance. Place model files under `models/`, register their paths in `config.py`, and prepare the following configuration files:

- model parameter configuration files under `train_configs/`;
- game-training parameter configuration files under `OpenSpielGame/train_configs/`.

The repository does not include datasets or pretrained model weights.

## 🗝️ Training

```bash
python OpenSpielGame/train_cfr.py --config OpenSpielGame/train_configs/cfr_training.yaml --use_config_players
```

## 🔍 Inference

```bash
python OpenSpielGame/inference_cfr.py --config OpenSpielGame/train_configs/cfr_training.yaml
```

## 📊 Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py
```

## 📑 Citation

The citation information will be released with the accompanying paper.

## 👍 Acknowledgement

This project builds on open-source research resources. We thank their authors and maintainers.

## 🔒 License

This project is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
