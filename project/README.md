# 基于传统与深度学习方法的虚假新闻检测项目

## 项目简介

本项目旨在对比传统机器学习（SVM）、深度学习（LSTM）与基于Transformer的BERT模型在虚假新闻检测任务中的表现。我们在两个主流数据集 LIAR1 和 LIAR2 上进行了二分类（真假）和六分类（六种真实性等级）任务，同时引入 GPT-4 对模型错误预测样本进行后验标注审查，探索大语言模型在标注质量评估中的应用潜力。

## 数据集介绍

使用的两个公开数据集：

- **LIAR1**：包含 12,836 条短文本政治陈述，标注为六个真实性等级。
- **LIAR2**：为LIAR1的扩展版本，包含约23,000条数据，增加了如发言人政党、历史可信度等结构化信息。

标签映射如下：

- 六分类：`pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`
- 二分类：将 `[pants-fire, false, barely-true]` 归为虚假类（fake），`[half-true, mostly-true, true]` 归为真实类（real）

## 模型与方法

### 1. SVM（支持向量机）

- 特征：基于 TF-IDF（5000词）构建的稀疏向量（包含 unigram + bigram）
- 核函数：RBF核
- 优点是训练快速、易于解释，但缺乏语义理解能力

### 2. LSTM（长短期记忆网络）

- 模型结构：嵌入层 + 单层 LSTM（128 单元）+ 全连接输出层
- 超参数：Dropout=0.3，学习率=1e-3，序列长度最大100
- 可建模序列关系，但对短文本效果有限

### 3. BERT（Transformer模型）

- 使用 `bert-base-uncased`，来自 HuggingFace
- 在六分类和二分类任务上进行微调
- 学习率=2e-5，批大小=16，训练3–5轮
- 表现最佳，能理解深层语义

### 4. GPT-4（后验标签审查）

- 用于重新评估 BERT 错误预测的样本
- 检查数据集中存在的模糊或错误标注，提高标签可靠性

## 实验结果

| 模型 | LIAR1 二分类 | LIAR1 六分类 | LIAR2 二分类 | LIAR2 六分类 |
|------|--------------|--------------|--------------|--------------|
| SVM  | 60%          | 24%          | 66%          | 31%          |
| LSTM | 59%          | 24%          | 68%          | 31%          |
| BERT | **63%**      | **27%**      | **70%**      | **35%**      |

- **GPT-4审查结果**：在100个BERT误判样本中，GPT-4与原标签一致的占67%，与BERT预测一致的占33%，说明原始标签存在模糊性。
- **误判分析**：
  - SVM 倾向于依赖高频词汇，容易偏向政治热词
  - LSTM 受极性词影响较大（如 "never", "only"）
  - BERT 对于区分 `half-true` 与 `barely-true` 等细分类别存在困难

## 关键发现

- 二分类任务相比六分类更容易实现高准确率
- BERT在结构更完善的LIAR2上表现优于其他模型
- 标签语义不清和分布不均会显著影响模型训练效果
- GPT-4 可作为人类辅助工具，用于标签质量评估与解释

## 未来工作展望

- 构建结合人类专家与GPT-4标注的高质量验证集，用作新基准
- 引入基于解释的训练目标，提高模型的推理能力与可解释性
- 使用对抗样本与语义接近样本，增强模型鲁棒性
- 引入“软标签”评价体系，缓解边界模糊样本的评分误差

## 参考文献

- Wang, W.Y. (2017). LIAR: A New Benchmark Dataset for Fake News Detection.
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Thorne, J. & Vlachos, A. (2022). Evidence-Based Fake News Detection with LLMs.



# Fake News Detection Using Traditional and Deep Learning Methods

## Overview

This project compares traditional machine learning, deep learning, and transformer-based approaches for fake news detection using the LIAR1 and LIAR2 datasets. We evaluate three representative models—SVM, LSTM, and BERT—on both binary (real vs fake) and six-class classification tasks. Additionally, GPT-4 is used for post-hoc label reassessment to evaluate annotation quality and potential human-AI collaboration.

## Team Members

- Caizhuangzhou Cui (z5509061)
- Hefei Fu (z5597779)
- Tingyuan Liu (z5565888)
- Yankun Wei (z5528317)
- Zihan Chen (z5527498)

## Datasets

We use the following publicly available datasets:

- **LIAR1**: 12,836 short political statements labeled with 6 truthfulness levels.
- **LIAR2**: An enhanced version of LIAR1 with ~23,000 entries, metadata (e.g. speaker party, credibility), and cleaned statements.

Labels are mapped as follows:

- Six-class: `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`
- Binary: `[pants-fire, false, barely-true]` → fake; `[half-true, mostly-true, true]` → real

## Models & Methods

### 1. SVM (Traditional)
- Input: 5000-dimensional TF-IDF (unigram + bigram)
- Kernel: RBF
- Fast and interpretable, but lacks contextual understanding

### 2. LSTM (Deep Learning)
- Embedding + 1-layer LSTM (128 units)
- Trained with dropout (0.3) and Adam optimizer
- Handles sequence modeling, but limited by short input length

### 3. BERT (Transformer)
- Model: `bert-base-uncased` from HuggingFace
- Fine-tuned on LIAR1/LIAR2 for 3–5 epochs
- Best performance across all tasks

### 4. GPT-4 (Post-Hoc Label Review)
- Used to reassess misclassified BERT outputs
- Helps identify ambiguous or mislabeled samples

## Results

| Model | LIAR1 Binary | LIAR1 Six-class | LIAR2 Binary | LIAR2 Six-class |
|-------|--------------|-----------------|--------------|-----------------|
| SVM   | 60%          | 24%             | 66%          | 31%             |
| LSTM  | 59%          | 24%             | 68%          | 31%             |
| BERT  | 63%          | 27%             | **70%**      | **35%**         |

- **GPT-4 Review**: In 100 BERT misclassifications, GPT-4 agreed with BERT 33% of the time, revealing potential label inconsistencies.
- **Qualitative analysis**: SVM biased toward frequent tokens; LSTM overreacts to polarity words; BERT confused by nuanced label boundaries.

## Key Insights

- Binary classification is significantly easier than six-class due to vague truthfulness definitions.
- BERT outperforms SVM and LSTM, especially on LIAR2 thanks to richer structure.
- Dataset quality and label clarity strongly impact classification accuracy.
- LLMs (e.g. GPT-4) enhance both label auditing and model interpretability.

## Future Work

- Develop a human-reviewed LIAR subset with GPT-4 support for benchmark testing
- Integrate explanation-based training to encourage semantically grounded decisions
- Employ contrastive training and adversarial samples to improve class boundary recognition
- Use disagreement-aware metrics to better capture near-miss classifications

## References

Key papers and datasets:
- Wang (2017): *LIAR: A New Benchmark Dataset for Fake News Detection*
- Devlin et al. (2019): *BERT: Pre-training of Deep Bidirectional Transformers*
- Thorne & Vlachos (2022): *Evidence-Based Fake News Detection with LLMs*

