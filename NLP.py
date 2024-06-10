from data_preprocess import data_preprocess
from 文本分类算法.logreg import log_reg
from 文本分类算法.SVM import SVM
from 文本分类算法.DTC_RFC import *
from 特征提取方法.tf_idf import tf_idf
from 特征提取方法.word2vec import word2vec
from 特征提取方法.bag_of_words import BoW
from 文本分类算法.NaiveBayes import NB

# 文件路径
text_path = '文本分类数据集/mr.txt'
# 标签路径
label_path = '文本分类数据集/mr_labels.txt'

if __name__ == '__main__':
    # 数据预处理
    texts_train, train_labels_list, texts_test, test_labels_list = (
        data_preprocess(text_path, label_path)
    )

    # 用tf-idf方法提取特征
    # train_vectors, test_vectors = tf_idf(texts_train, texts_test)

    # 用Word2Vec方法
    # train_vectors, test_vectors = word2vec(texts_train, texts_test)

    # 用Bow方法
    train_vectors, test_vectors = BoW(texts_train, texts_test)


    # 使用逻辑回归模型训练并评估
    # log_reg(train_vectors=train_vectors,
    #         train_labels=train_labels_list,
    #         test_vectors=test_vectors,
    #         test_labels=test_labels_list)

    # 使用SVM模型训练并评估
    # SVM(train_vectors=train_vectors,
    #     train_labels=train_labels_list,
    #     test_vectors=test_vectors,
    #     test_labels=test_labels_list)

    # 使用决策树模型训练并评估
    # DTC(train_vectors=train_vectors,
    #     train_labels=train_labels_list,
    #     test_vectors=test_vectors,
    #     test_labels=test_labels_list)

    # 使用随机森林模型训练并评估
    # RFC(train_vectors=train_vectors,
    #     train_labels=train_labels_list,
    #     test_vectors=test_vectors,
    #     test_labels=test_labels_list)

    # 使用NaiveBayes模型训练并评估
    NB(train_vectors=train_vectors,
        train_labels=train_labels_list,
        test_vectors=test_vectors,
        test_labels=test_labels_list)
