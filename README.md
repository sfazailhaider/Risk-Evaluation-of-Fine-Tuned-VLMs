# Risk Evaluation of Fine-Tuned Vision-Language Models

> **Can fine-tuning a VLM on a medical task make it dangerously overconfident?**

This project fine-tunes two state-of-the-art Vision-Language Models (VLMs) on a chest X-ray pneumonia classification task using QLoRA, then rigorously evaluates whether fine-tuning improves — or quietly undermines — model reliability. We engineer 9 custom evaluation metrics to expose failure modes that standard accuracy scores miss entirely.

📄 [Report](Report.pdf) · 🎥 [Demo Video](Video.mp4) · 📓 [Notebook](Code.ipynb)

---

## The Core Finding

Fine-tuning improved task accuracy dramatically — but introduced a critical safety risk:

> **76.81% of incorrect predictions were delivered with high confidence and fluent, coherent explanations.**

A model that is wrong but sounds right is more dangerous than one that is wrong and knows it — especially in a medical context.

---

## Models

| Model | Type | Parameters |
|---|---|---|
| Qwen2.5-VL-7B | Vision-Language Model | 7B |
| Qwen3-VL-8B | Vision-Language Model | 8B |

Both evaluated under two conditions:
- **Zero-shot** — no task-specific training
- **LoRA fine-tuned** — QLoRA adapted on chest X-ray pneumonia dataset

---

## Dataset

**Chest X-Ray Images (Pneumonia)** — binary classification: `NORMAL` vs `PNEUMONIA`

- 200 test samples evaluated per model condition
- Models prompted to classify and explain their reasoning in natural language

---

## Results

### Performance

| Model | Condition | Accuracy | Pneumonia F1 | Macro F1 |
|---|---|---|---|---|
| Qwen2.5-VL-7B | Zero-shot | 51.5% | 0.1101 | 0.3884 |
| Qwen2.5-VL-7B | LoRA | 51.0% | 0.0926 | 0.3785 |
| Qwen3-VL-8B | Zero-shot | 64.0% | 0.5663 | 0.6293 |
| Qwen3-VL-8B | LoRA | **65.5%** | **0.5767** | **0.6428** |

### Safety Metrics (the concerning part)

| Metric | Qwen2.5 Zero-Shot | Qwen2.5 LoRA | Qwen3 Zero-Shot | Qwen3 LoRA |
|---|---|---|---|---|
| Overconfident Error Rate | 0.3093 | 0.6122 | 0.7361 | **0.7681** |
| Dangerous Hallucination Rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fluency Proxy, Correct Outputs | 0.3786 | 0.2745 | 0.7930 | 0.8282 |
| Fluency Proxy, Wrong Outputs | 0.4433 | 0.3214 | 0.7778 | **0.8551** |
| Fluency-Correctness Gap | -0.0717 | -0.1886 | 0.1378 | **0.2001** |
| Mean Faithfulness Score | 0.5025 | 0.5100 | **0.8900** | **0.8900** |

---

## 9 Custom Evaluation Metrics

Standard accuracy alone is insufficient for high-stakes AI. We engineer the following:

1. **Accuracy** — overall classification correctness
2. **F1 Score** — harmonic mean of precision and recall
3. **Overconfident Error Rate** — % of wrong predictions with high-confidence language
4. **Fluency Score** — linguistic coherence of generated explanations
5. **Fluency-Correctness Gap** — divergence between how well the model writes vs. how correctly it reasons
6. **Faithfulness Distribution** — whether explanations are grounded in actual image evidence
7. **Hallucination Rate** — generation of plausible but fabricated clinical details
8. **Dangerous Hallucination Count** — subset where hallucinations could cause real clinical harm
9. **Confidence-Accuracy Calibration** — whether stated confidence tracks actual accuracy

---


## How to Run

### Requirements

```bash
pip install torch transformers peft datasets pillow pandas matplotlib seaborn jupyter
```
GPU with ≥16GB VRAM recommended

### Steps
1. Clone the repo

```bash
git clone https://github.com/sfazailhaider/Risk-Evaluation-of-Fine-Tuned-VLMs.git
cd Risk-Evaluation-of-Fine-Tuned-VLMs
```
2. Open the notebook
jupyter notebook Code.ipynb

3. Run sections in order:
Dataset loading & preprocessing
Zero-shot inference
QLoRA fine-tuning
LoRA inference
Custom metric evaluation
Figure generation

## Key Takeaway
Fine-tuning VLMs on narrow medical tasks can dramatically improve task-specific performance while simultaneously degrading safety properties — producing models that are confidently, fluently, and dangerously wrong. Evaluating only accuracy misses this entirely.
