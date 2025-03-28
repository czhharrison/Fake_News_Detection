import os
import re
import html
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer

# 下载NLTK资源
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# 数据路径
data_path = "./liar2_dataset/converted_csv"

# 生成文件路径
train_path = pd.read_csv(os.path.join(data_path, "liar2_train.csv"))
valid_path = pd.read_csv(os.path.join(data_path, "liar2_valid.csv"))
test_path = pd.read_csv(os.path.join(data_path, "liar2_test.csv"))

# 数据清理
# 清理justification
def clean_justification(text):
    if pd.isna(text):
        return text
    text = html.unescape(text)
    text = text.replace('\xa0', ' ').replace('\u00a0', ' ').replace('\ufeff', ' ')
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'window\..*?;', '', text, flags=re.DOTALL)
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r"We rate this.*", "", text)
    return text.strip()

# 移除否定词
news_stopwords = set(stopwords.words("english")) - {
    "no", "not", "nor", "never", "none", "nobody", "nothing", "neither",
    "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't",
    "don't", "doesn't", "didn't", "can't"
}

# 轻度清洗（LSTM）
def lstm_clean(text):
    if pd.isna(text):
        return ""
    text = html.unescape(text)
    text = text.replace('\xa0', ' ').replace('\u00a0', ' ').replace('\ufeff', ' ')
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in news_stopwords]
    return " ".join(tokens)

# 词性过滤（SVM）
def svm_clean(text):
    if pd.isna(text):
        return ""
    text = html.unescape(text)
    text = text.replace('\xa0', ' ').replace('\u00a0', ' ').replace('\ufeff', ' ')
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    selected = [word for word, tag in tagged if tag.startswith("NN") or tag.startswith("VB") or tag.startswith("JJ") or tag.startswith("RB")]
    return " ".join(selected)

# 对新闻主体进行文本清理
for d in [train_path, valid_path, test_path]:
    d.replace("unknown", pd.NA, inplace=True)
    d.fillna("", inplace=True)
    d["justification"] = d["justification"].apply(clean_justification)
    d["clean_statement"] = d["statement"].apply(lstm_clean)
    d["clean_statement_svm"] = d["statement"].apply(svm_clean)
    d["clean_statement_bert"] = d["statement"].apply(lambda x: re.sub(r'[\xa0\u00a0\ufeff]+', ' ', html.unescape(str(x))))
    d["date"] = d["date"].astype(str)

# 数据可视化
os.makedirs("./liar2_dataset/images", exist_ok=True)

# label的值对应
label_value = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true"
}
label_order = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
train_path = train_path[train_path["label"].isin(label_value.keys())].copy()
train_path["label"] = train_path["label"].astype(int)
train_path["label_text"] = train_path["label"].map(label_value)

# 计算 label 分布
plt.figure(figsize=(8,5))
sns.countplot(data=train_path, x="label_text", order=label_order, palette="viridis")
plt.title("Label Distribution")
plt.savefig("./liar2_dataset/images/label_distribution.png")

# 计算文本长度分布
train_path["text_len"] = train_path["statement"].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(8,5))
sns.histplot(train_path["text_len"], bins=30, kde=True)
plt.title("Text Length Distribution")
plt.savefig("./liar2_dataset/images/text_length_distribution.png")

# 假新闻的传播趋势
train_path["year"] = train_path["date"].str.extract(r"(\d{4})")      # 从context中提取年份
train_path = train_path[train_path["year"].str.isnumeric()]
train_path["year"] = train_path["year"].astype(int)
plt.figure(figsize=(12,5))
sns.countplot(data=train_path, x="year", hue="label_text", hue_order=label_order, palette="rocket")
plt.title("Fake News Over Time")
plt.xticks(rotation=45)
plt.savefig("./liar2_dataset/images/fake_news_over_time.png")

# 进行 TF-IDF 特征提取（SVM）
vectorizer = TfidfVectorizer(max_features=5000)         # 取 5000 个最重要的词, 可修改
train_feature = vectorizer.fit_transform(train_path["clean_statement_svm"])
valid_feature = vectorizer.transform(valid_path["clean_statement_svm"])
test_feature = vectorizer.transform(test_path["clean_statement_svm"])

print(f"TF-IDF Train Shape: {train_feature.shape}")

# Tokenizer 处理（LSTM）
max_words = 10000     # 词汇表大小
max_len = 100         # 处理的最大文本长度, 可修改

# 创建Tokenizer，转换文本
lstm_tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
lstm_tokenizer.fit_on_texts(train_path["clean_statement"])

train_seq = pad_sequences(lstm_tokenizer.texts_to_sequences(train_path["clean_statement"]), maxlen=max_len)
valid_seq = pad_sequences(lstm_tokenizer.texts_to_sequences(valid_path["clean_statement"]), maxlen=max_len)
test_seq = pad_sequences(lstm_tokenizer.texts_to_sequences(test_path["clean_statement"]), maxlen=max_len)

print(f"LSTM Tokenized Train Shape: {train_seq.shape}")

# Tokenizer 处理（BERT）
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode(texts):
    return bert_tokenizer.batch_encode_plus(
        texts.tolist(), max_length=max_len, truncation=True, padding="max_length", return_tensors="np"
    )

bert_train = encode(train_path["clean_statement_bert"])
bert_valid = encode(valid_path["clean_statement_bert"])
bert_test = encode(test_path["clean_statement_bert"])

print(f"BERT Tokenized Train Shape: {bert_train['input_ids'].shape}")

# 保存为csv文件
os.makedirs("./liar2_dataset/processed_dataset", exist_ok=True)
train_path.to_csv("./liar2_dataset/processed_dataset/processed_train.csv", index=False)
valid_path.to_csv("./liar2_dataset/processed_dataset/processed_valid.csv", index=False)
test_path.to_csv("./liar2_dataset/processed_dataset/processed_test.csv", index=False)

print("Data processing completed. CSV files saved!")
