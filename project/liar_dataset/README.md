# 数据处理阶段说明

主要完成以下任务：
1. **加载并解析 LIAR 数据集**
2. **文本清理（去除特殊字符、停用词等）**
3. **处理缺失值**
4. **数据可视化**
5. **特征提取**
   - **TF-IDF**（SVM）
   - **Tokenizer**（LSTM）
   - **BERT Tokenizer**（BERT）

---
### 处理好的数据在: ./liar_dataset/processed_dataset
### 生成的可视化图像在： ./liar_dataset/images
### downloads和ucsbnlp__liar是原dataset自带的

---
### LIAR 数据集字段含义

LIAR 数据集包含 **新闻声明及其真假标签**，共 14 个字段：

| **字段名** | **说明** |
|------------|----------|
| **id** | 每条数据的唯一标识符，如 `"2635.json"` |
| **label** | 新闻的真假标签，共 6 类：`true`, `false`, `half-true`, `mostly-true`, `barely-true`, `pants-fire` |
| **statement** | 新闻文本，即待分类的句子或短文 |
| **subject** | 该声明涉及的主题，如 `health-care`, `economy`, `climate-change` |
| **speaker** | 发表该声明的个人或组织，如 `Donald Trump`, `Hillary Clinton` |
| **job_title** | 发表声明者的职位，如 `President`, `Senator`, `Governor` |
| **state_info** | 发表声明者所属的州（如 `Texas`, `California`），部分数据为空 |
| **party_affiliation** | 发表声明者的政党，如 `Democrat`, `Republican`, `Independent` |
| **barely_true_counts** | 该发言者历史上被评为 `"barely-true"` 的次数 |
| **false_counts** | 该发言者历史上被评为 `"false"` 的次数 |
| **half_true_counts** | 该发言者历史上被评为 `"half-true"` 的次数 |
| **mostly_true_counts** | 该发言者历史上被评为 `"mostly-true"` 的次数 |
| **pants_on_fire_counts** | 该发言者历史上被评为 `"pants-fire"`（完全虚假）的次数 |
| **context** | 声明的背景信息，如 `"a news conference"`, `"a campaign ad"` |

**用于训练的主要字段：**  
**statement** → 主要的文本输入  
**label** → 目标分类
---

# **训练时可能用到的变量**

## **SVM**

| **变量名** | **含义**                   |
|------------|--------------------------|
| `train_feature` | 训练集的 TF-IDF 特征，表示新闻文本的词频 |
| `test_feature` | 测试集的 TF-IDF 特征           |
| `valid_feature` | 验证集的 TF-IDF 特征           |
| `train_data["label"]` | 训练集的真实分类标签               |
| `test_data["label"]` | 测试集的真实分类标签               |
| `valid_data["label"]` | 验证集的真实分类标签               |

---

## **LSTM**

| **变量名** | **含义**                  |
|------------|-------------------------|
| `train_seq` | 训练集的文本序列，每条新闻限制 `100` 词 |
| `test_seq` | 测试集的文本序列                |
| `valid_seq` | 验证集的文本序列                |
| `train_data["label"]` | 训练集的真实分类标签              |
| `test_data["label"]` | 测试集的真实分类标签              |
| `valid_data["label"]` | 验证集的真实分类标签              |

---

## **BERT 训练**

| **变量名** | **含义**             |
|------------|--------------------|
| `bert_train["input_ids"]` | 训练集的 BERT Token 序列 |
| `bert_train["attention_mask"]` | 训练集的注意力掩码，标记有效词与填充 |
| `bert_test["input_ids"]` | 测试集的 BERT Token 序列 |
| `bert_test["attention_mask"]` | 测试集的注意力掩码          |
| `bert_valid["input_ids"]` | 验证集的 BERT Token 序列 |
| `bert_valid["attention_mask"]` | 验证集的注意力掩码          |
| `train_data["label"]` | 训练集的真实分类标签         |
| `test_data["label"]` | 测试集的真实分类标签         |
| `valid_data["label"]` | 验证集的真实分类标签         |

---

### 关于特征提取限制
**长文本分类**（如完整新闻、文章） | `300-512` 词 | BERT 最大长度为 `512`，LSTM 可能需要 `300-500` 词 

**对于 LIAR 数据集：**
- **TF-IDF**：当前 5000 → **可以增加到 10,000** 
- **LSTM**：当前 100 词 → **可以尝试 150 或 200**
- **BERT**：当前 100 词 → **可以尝试 128 或 256**