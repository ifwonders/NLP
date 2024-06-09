from data_preprocess import data_preprocess
from 文本分类算法.log_reg import log_reg
from 特征提取方法.tf_idf import tf_idf
from 特征提取方法.word2vec import word2vec

# 文件路径
text_path = '文本分类数据集/mr.txt'
# 标签路径
label_path = '文本分类数据集/mr_labels.txt'

if __name__ == '__main__':
    # 数据预处理
    texts_train, train_labels_list, texts_test, test_labels_list =(
        data_preprocess(text_path,label_path)
    )

    # 用tf-idf方法提取特征
    train_tfidf,test_tfidf = tf_idf(texts_train, texts_test)

    # 使用逻辑回归模型训练并评估
    log_reg(train_vectors=train_tfidf,
            train_labels=train_labels_list,
            test_vectors=test_tfidf,
            test_labels=test_labels_list)
