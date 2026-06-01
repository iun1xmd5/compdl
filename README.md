# Hybrid Deep Learning Frameworks

**Official Code Repository** for the research paper 
---
## 📄 Paper Title
**Hybrid Deep Learning Frameworks**

## ✨ Key Highlights

- First work demonstrating **cross-domain convergence** of Bayesian Optimization on heterogeneous network benchmarks.
- Achieves **~10K trainable parameters** (4× smaller than baseline) while maintaining competitive performance.
- Uses a **hardware-aware composite objective** (Macro-F1 + latency + parameter count).
- Consistent sub-**0.06 ms** inference latency on Tesla T4 GPU.
- Validated on three heterogeneous datasets

---

## 📊 Abstract

Internet of Things (IoT), Industrial IoT (IIoT), and Software-Defined Networks (SDN) operate under severe resource constraints, demanding lightweight yet accurate intrusion detection systems (IDS). This work proposes a **Bayesian-optimized compact CNN-LSTM** architecture using the Tree-structured Parzen Estimator (TPE) under a multi-objective function that jointly optimizes classification performance, inference latency, and model size. The optimized model converges to an identical compact architecture (~10K parameters) across all three domains, offering an excellent accuracy-efficiency trade-off suitable for edge deployment.
---


