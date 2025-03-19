import os
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer

# 下载NLTK停用词
nltk.download('stopwords')

# 数据路径
data_path = "./liar_dataset/downloads/extracted/ae6492a79e60a0c2bf4b97b2f29ef984fece206d21b917f538012603d0b54d9c"

# 生成文件路径
train_path = os.path.join(data_path, "train.tsv")
test_path = os.path.join(data_path, "test.tsv")
valid_path = os.path.join(data_path, "valid.tsv")

# 设置列名
column_names = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state_info",
    "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
]

# 读取数据
train_data = pd.read_csv(train_path, sep='\t', names=column_names, header=None)
test_data = pd.read_csv(test_path, sep='\t', names=column_names, header=None)
valid_data = pd.read_csv(valid_path, sep='\t', names=column_names, header=None)

# 数据清理
def clean(text):
    text = text.lower()             # 所有文本转换为小写
    text = re.sub(r'[^a-z\s]', '', text)        # 仅保留字母和空格
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # 去除停用词
    return " ".join(words)

# 对新闻主体进行文本清理
train_data["clean_statement"] = train_data["statement"].apply(clean)
test_data["clean_statement"] = test_data["statement"].apply(clean)
valid_data["clean_statement"] = valid_data["statement"].apply(clean)

# 填充空值
for d in [train_data, test_data, valid_data]:
    d["job_title"].fillna("unknown", inplace=True)
    d["state_info"].fillna("unknown", inplace=True)

# 数据可视化
# 计算 label 分布
plt.figure(figsize=(8,5))
sns.countplot(x=train_data["label"], order=train_data["label"].value_counts().index, hue=train_data["label"], palette="viridis", legend=False)
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Distribution of Labels in Training Set")
plt.xticks(rotation=20)
plt.savefig("./liar_dataset/images/label_distribution.png")


# 计算文本长度分布
train_data["text_length"] = train_data["statement"].apply(lambda x: len(x.split()))
plt.figure(figsize=(8,5))
sns.histplot(train_data["text_length"], bins=30, kde=True)
plt.xlabel("Text Length (# words)")
plt.ylabel("Frequency")
plt.title("Distribution of Statement Length")
plt.savefig("./liar_dataset/images/text_length_distribution.png")

# 假新闻的传播趋势
train_data["year"] = train_data["context"].str.extract(r"(\d{4})")      # 从context中提取年份

plt.figure(figsize=(10,6))
sns.countplot(x="year", hue="label", data=train_data, palette="rocket")
plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Fake News Trends Over Time")
plt.xticks(rotation=30)
plt.savefig("./liar_dataset/images/fake_news_trend.png")


# 真假新闻的主题分布（前10个最常见的主题）
top_subjects = train_data["subject"].value_counts().nlargest(10).index
filtered_df = train_data[train_data["subject"].isin(top_subjects)]

plt.figure(figsize=(12,6))
sns.countplot(x="subject", hue="label", data=filtered_df, palette="magma")
plt.xlabel("Subject")
plt.ylabel("Count")
plt.title("Top 10 Subjects and Their News Categories")
plt.xticks(rotation=30)
plt.savefig("./liar_dataset/images/subject_vs_news.png")


# 进行 TF-IDF 特征提取（SVM）
vectorizer = TfidfVectorizer(max_features=5000)         # 取 5000 个最重要的词, 可修改
train_feature = vectorizer.fit_transform(train_data["clean_statement"])
test_feature = vectorizer.transform(test_data["clean_statement"])
valid_feature = vectorizer.transform(valid_data["clean_statement"])

print(f"TF-IDF Train Shape: {train_feature.shape}")

# Tokenizer 处理（LSTM）
max_words = 10000     # 词汇表大小
max_len = 100         # 处理的最大文本长度, 可修改

# 创建Tokenizer，转换文本
lstm_tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
lstm_tokenizer.fit_on_texts(train_data["clean_statement"])

train_seq = pad_sequences(lstm_tokenizer.texts_to_sequences(train_data["clean_statement"]), maxlen=max_len)
test_seq = pad_sequences(lstm_tokenizer.texts_to_sequences(test_data["clean_statement"]), maxlen=max_len)
valid_seq = pad_sequences(lstm_tokenizer.texts_to_sequences(valid_data["clean_statement"]), maxlen=max_len)

print(f"LSTM Tokenized Train Shape: {train_seq.shape}")

# Tokenizer 处理（BERT）
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 对文本进行编码
def encode(texts):
    return bert_tokenizer.batch_encode_plus(
        texts.tolist(), max_length=max_len, truncation=True, padding="max_length",
        return_tensors="np"
    )

bert_train = encode(train_data["statement"])
bert_test = encode(test_data["statement"])
bert_valid = encode(valid_data["statement"])

print(f"BERT Tokenized Train Shape: {bert_train['input_ids'].shape}")

# 保存为csv文件
train_data.to_csv("./liar_dataset/processed_dataset/processed_train.csv", index=False)
test_data.to_csv("./liar_dataset/processed_dataset/processed_test.csv", index=False)
valid_data.to_csv("./liar_dataset/processed_dataset/processed_valid.csv", index=False)

print("Data processing completed. CSV files saved")
