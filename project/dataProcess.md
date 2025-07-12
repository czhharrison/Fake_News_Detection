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
### dataProcess.py是liar1.0的处理文件，dataProcess_liar2.py是liar2.0的处理文件
### liar2_dataset是2.0版本的dataset,两个版本的对比如下：

| **Statistics**                               | **LIAR** | **LIAR2** |
|----------------------------------------------|----------|------------|
| Training set size                            | 10,269   | 18,369     |
| Validation set size                          | 1,284    | 2,297      |
| Testing set size                             | 1,283    | 2,296      |
| Avg. statement length (tokens)               | 17.9     | 17.7       |
| Avg. speaker description length (tokens)     | \        | 39.4       |
| Avg. justification length (tokens)           | \        | 94.4       |
| **Labels**
| Pants on fire                                | 1,050    | 3,031      |
| False                                        | 2,511    | 6,605      |
| Barely-true                                  | 2,108    | 3,603      |
| Half-true                                    | 2,638    | 3,709      |
| Mostly-true                                  | 2,466    | 3,429      |
| True                                         | 2,063    | 2,585      |

#### liar2_dataset处理好的数据在./liar2_dataset/processed_dataset
#### label是数字表示的。也包含一个label_text字段与其文本对应
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true"
### LIAR2 数据集字段含义
| 字段名                    | 含义说明                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| `id`                      | 样本的唯一编号                                                           |
| `label`                   | 标签编号，取值为 0~5，表示新闻真实性等级（从假到真）                     |
| `statement`               | 新闻主语句（待判断的新闻陈述）                                           |
| `date`                    | 新闻发布或引用的日期                                                     |
| `subject`                 | 新闻主题（多个主题以分号分隔）                                           |
| `speaker`                 | 发言人名称                                                               |
| `speaker_description`     | 发言人的描述信息（身份、职业等）                                         |
| `state_info`              | 发言人所在州/地区信息                                                    |
| `true_counts`             | 发言人历史记录中被评为 "true" 的次数                                    |
| `mostly_true_counts`      | 发言人历史记录中被评为 "mostly-true" 的次数                             |
| `half_true_counts`        | 发言人历史记录中被评为 "half-true" 的次数                               |
| `mostly_false_counts`     | 发言人历史记录中被评为 "mostly-false" 的次数                            |
| `false_counts`            | 发言人历史记录中被评为 "false" 的次数                                   |
| `pants_on_fire_counts`    | 发言人历史记录中被评为 "pants-on-fire"（极端虚假）的次数                 |
| `context`                 | 新闻或引用的上下文信息（如采访、文章、演讲背景等）                       |
| `justification`           | 官方的事实验证说明文字（原始数据提供）                                   |
| `clean_statement`         | 为 LSTM 清洗后的新闻句子文本（保留否定词，去除停用词）                   |
| `clean_statement_svm`     | 为 SVM 模型处理的新闻句子文本（基于词性过滤，仅保留核心词）              |
| `clean_statement_bert`    | 为 BERT 模型清洗的原始文本（仅清理特殊字符，不去除任何语义词）           |
| `label_text`              | 标签的文本形式，如 "true", "false", "pants-fire" 等                     |
| `text_len`                | `statement` 的词数量，用于分析文本长度                                   |
| `year`                    | 从 `date` 中提取的年份，用于时间分布分析                                 |


**用于训练的主要字段：**  
**LSTM**用`clean_statement` 
**SVM**用`clean_statement_svm` 
**BERT**用`clean_statement_bert` 
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

- **TF-IDF**：当前 5000
- **LSTM**：当前 100 词
- **BERT**：当前 100 词
