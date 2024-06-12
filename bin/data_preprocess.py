import re

import spacy
from nltk.corpus import stopwords


# 读取文本文件并转换为列表
def read_file_to_list(file_path):
    data_list = []  # 初始化一个空列表来存储数据
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:  # 打开文件进行读取
        for line in file:  # 逐行读取
            line = line.strip()  # 去除每行的前后空白字符（如果有的话）
            if line:  # 如果行不为空，则添加到列表中
                data_list.append(line)
    return data_list


# 读取标签文件转换为字典列表
def read_labels_to_dict_list(file_path):
    labels_list = []  # 初始化一个空列表来存储字典
    with open(file_path, 'r', encoding='utf-8') as file:  # 打开文件进行读取
        for line in file:  # 逐行读取
            parts = line.strip().split('\t')  # 使用制表符分割每行，并去除前后空白字符
            if len(parts) == 3:  # 确保每行有三个部分
                # 创建一个字典，并以分割后的三个部分作为键值对
                label_dict = {
                    '编号': parts[0],
                    '数据集': parts[1],  # 训练集或数据集
                    '分类属性': parts[2]  # 分类属性（0或1）
                }
                labels_list.append(label_dict)  # 将字典添加到列表中
    return labels_list


# 清洗文本数据，只保留英文字母和空格
def clean_text_list(text_list):
    cleaned_text_list = []
    for text in text_list:
        # 使用正则表达式替换非英文字母和空格字符为空字符串
        cleaned_text = re.sub(r'[^\sa-zA-Z]', '', text)
        cleaned_text_list.append(cleaned_text)
    return cleaned_text_list


def tokenize_text_list(cleaned_text_list):
    # 加载英文分词模型
    nlp = spacy.load("en_core_web_sm")
    tokenized_text_list = []
    for text in cleaned_text_list:
        # 使用spaCy进行分词
        doc = nlp(text)
        tokens = [token.text for token in doc]
        tokenized_text_list.append(tokens)
    return tokenized_text_list


# 去除停用词
def remove_stopwords_from_tokens(tokens):
    # 下载并获取英文停用词列表
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    # 初始化一个空列表来存储过滤后的单词
    filtered_tokens = []
    # 遍历文档中的每个token
    for token in tokens:
        # 如果token的文本不在停用词列表中，则添加到过滤后的列表中
        if token.lower() not in stop_words:
            filtered_tokens.append(token)
    return filtered_tokens


# 对列表去除停用词
def process_text(tokenized_text_list):
    # 初始化一个空列表来存储处理后的文本
    filtered_text_list = []
    # 遍历分词后的文本列表
    for tokens in tokenized_text_list:
        # 对每个文本去除停用词
        filtered_tokens = remove_stopwords_from_tokens(tokens)
        # 将去除停用词后的单词列表添加到结果列表中
        filtered_text_list.append(filtered_tokens)
    return filtered_text_list


# 划分训练集和测试集
def divide_train_test(text_list, labels_list):
    train_text_list = []
    train_labels_list = []
    test_text_list = []
    test_labels_list = []

    train_cnt = 0
    test_cnt = 0
    for labels in labels_list:
        if labels['数据集'] == 'test':
            # print(text_list[int(labels['编号'])])
            test_text_list.append(text_list[int(labels['编号'])])
            labels['编号'] = test_cnt
            test_labels_list.append(float(labels['分类属性']))
            test_cnt += 1
        elif labels['数据集'] == 'train':
            # print(text_list[int(labels['编号'])])
            train_text_list.append(text_list[int(labels['编号'])])
            labels['编号'] = train_cnt
            train_labels_list.append(float(labels['分类属性']))
            train_cnt += 1

    return train_text_list, train_labels_list, test_text_list, test_labels_list


def data_sets_preprocess(text_path, label_path, is_join_words=True):
    text_list = read_file_to_list(text_path)
    label_dict_list = read_labels_to_dict_list(label_path)

    cleaned_text_list = clean_text_list(text_list)
    tokenized_text_list = tokenize_text_list(cleaned_text_list)
    filtered_text_list = process_text(tokenized_text_list)

    train_text_list, train_labels_list, test_text_list, test_labels_list = (
        divide_train_test(filtered_text_list, label_dict_list)
    )

    if is_join_words is False:
        return train_text_list, train_labels_list, test_text_list, test_labels_list

    texts_train = [' '.join(i) for i in train_text_list]
    texts_test = [' '.join(i) for i in test_text_list]

    return texts_train, train_labels_list, texts_test, test_labels_list


def text_preprocess(text, is_join_words=True):
    text_list = [text]

    cleaned_text_list = clean_text_list(text_list)
    tokenized_text_list = tokenize_text_list(cleaned_text_list)
    filtered_text_list = process_text(tokenized_text_list)

    if is_join_words is False:
        return filtered_text_list

    processed_text = [' '.join(i) for i in filtered_text_list]

    return processed_text


if __name__ == '__main__':
    processed_text = text_preprocess('testing this function is working or not')
    print(processed_text)
