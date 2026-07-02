# Hybrid Deep Learning Frameworks

Official Code Repository for the research paper

## 📄 Paper Title
Cross-Domain Convergence of Hardware-Aware Bayesian-Optimized Compact CNN-LSTM Architectures for Edge-Deployable Intrusion Detection in IoT, IIoT, and SDN

## ✨ Key Highlights

* First work demonstrating cross-domain convergence of Bayesian Optimization on heterogeneous network benchmarks.
* Achieves 8,258–12,905 trainable parameters (70–81% reduction, ~3.4–5.4× more compact) than the non-optimized baseline, while maintaining competitive performance.
* Uses a hardware-aware composite objective (Macro-F1 + latency + parameter count).
* Per-sample inference latency of 2.26–2.41 ms on a Tesla T4 GPU at batch size = 1 (true single-flow edge streaming conditions).
* Validated on three heterogeneous datasets (InSDN, ToN-IoT, Edge-IIoTset)

## 📊 Abstract
Internet of Things (IoT), Industrial IoT (IIoT), and Software-Defined Network (SDN) deployments are increasingly exposed to cyberattacks, but they always operate with limited computing resources. This increases the need for having intrusion detection systems (IDS) that are both accurate and practically deployable in environments with limited resources. Hybrid convolutional-recurrent architectures such as CNN-LSTM can capture both spatial feature interactions and temporal attack progression, yet untuned configurations tend to produce large, computationally expensive models that are impractical for edge deployment. This paper proposes a Bayesian-optimized hybrid CNN-LSTM IDS in which nine architectural and training hyperparameters are jointly tuned using the Tree-structured Parzen Estimator (TPE) under a composite objective that penalizes both inference latency and parameter count, in addition to maximizing macro-F1. The model is evaluated on three heterogeneous benchmarks: InSDN (9-class SDN traffic, 343,888 flows), ToN-IoT (10-class IoT telemetry, 190,474 flows), and Edge-IIoTset (15-class IoT/IIoT traffic, 126,117 sampled flows) against seven baselines: non-optimized CNN-LSTM, CNN-only, LSTM-only, Random Forest, LightGBM, XGBoost, and CatBoost. Bayesian optimization independently converges to identical core architectural hyperparameters across all three datasets (n_conv=1, kernel_size=3, dense_units=32), with ToN-IoT and Edge-IIoTset additionally sharing filters=24 and lstm_units=24. The resulting models contain 8,258–12,905 trainable parameters, representing a 70–81% reduction relative to the non-optimized baseline. The optimized model achieves macro-F1 of 0.8120, 0.8876, and 0.8848 on InSDN, ToN-IoT, and Edge-IIoTset, respectively, trading classification performance for substantial gains in parameter efficiency and a normalized per-sample inference latency of 2.3–2.4 ms on a Tesla T4 GPU at batch size 1, representing true single-flow edge streaming conditions and reported here as a controlled comparative benchmark rather than a direct edge-device measurement. Latency measurements at batch size 1 remove GPU-parallelism bias and enable fair cross-model comparison. Gradient boosting methods remain strong tabular baselines on all three benchmarks; a Friedman test across the three datasets confirms statistically significant performance differences among the eight evaluated methods (p = 0.0065). The convergence of BO to the same compact convolutional core across three heterogeneous domains provides a principled, hardware-aware starting point for practitioners deploying CNN-LSTM IDS in new IoT, IIoT, or SDN environments.

**Keywords** — Intrusion Detection System, CNN-LSTM, Bayesian Optimization, IoT Security, IIoT Security, Software-Defined Networks, Edge Deployment, Runtime Efficiency, Deep Learning.

## 🔁 Reproducibility

This repository accompanies the manuscript submitted to *Knowledge-Based Systems* (Elsevier) on
Bayesian-optimized hybrid CNN-LSTM architectures for network intrusion detection across
IoT, IIoT, and SDN environments.

### Datasets and corresponding code

| Dataset       | Notebook (end-to-end pipeline)         | Training script                     |
|---------------|------------------------------------------|--------------------------------------|
| InSDN         | `notebooks/cnn_lstm_insdn.ipynb`         | `src/train_cnn_lstm_insdn.py`        |
| ToN-IoT       | `notebooks/cnn_lstm_toniot.ipynb`        | `src/train_cnn_lstm_toniot.py`       |
| Edge-IIoTset  | `notebooks/cnn_lstm_edgeiiotset.ipynb`   | `src/train_cnn_lstm_edgeiiotset.py`  |

Notebooks contain the full pipeline (preprocessing, sliding-window construction, Bayesian
hyperparameter search, training, and evaluation) with executed outputs — metric tables,
confusion matrices, and training curves — included so results can be inspected without
re-running the pipeline. The `.py` scripts in `src/` are the equivalent standalone
training entry points used to produce the results reported in the paper.

### Environment

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. GPU (CUDA) recommended for training; inference-only runs work on CPU.

### Running a script directly

```bash
python src/train_cnn_lstm_insdn.py
```

Each script expects the corresponding dataset to be available locally — see comments at the
top of each script (or the notebook's data-loading cell) for the expected file paths/format.

### Data availability

- InSDN: https://www.kaggle.com/datasets/badcodebuilder/insdn-dataset
- ToN-IoT: https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset
- Edge-IIoTset: https://www.kaggle.com/datasets/sibasispradhan/edge-iiotset-dataset

### Citation

If you use this code, please cite the paper (see `CITATION.cff`).
