import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB


def NB(train_vectors, train_labels, test_vectors, test_labels):
    """
    一般来说不需要对朴素贝叶斯模型调参
    :param train_vectors:
    :param train_labels:
    :param test_vectors:
    :param test_labels:
    :return:
    """
    # 初始化Naive Bayes分类器
    nb_classifier = GaussianNB()

    if isinstance(train_vectors, csr_matrix):
        train_vectors = train_vectors.toarray()
        test_vectors = test_vectors.toarray()

    # 训练分类器
    nb_classifier.fit(train_vectors, train_labels)

    # 预测测试集标签
    predicted_labels = nb_classifier.predict(test_vectors)

    # 计算准确率
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f"\n朴素贝叶斯模型准确率: {accuracy * 100:.2f}%")

    # 输出更详细的评估报告
    print("\n朴素贝叶斯模型分类报告:")
    print(classification_report(test_labels, predicted_labels))


if __name__ == '__main__':
    # 示例：训练集和测试集的向量
    train_vectors = np.array([[0.5, 0.2, 0.1], [0.9, 0.8, 0.7]])  # 示例训练集向量
    test_vectors = np.array([[0.6, 0.1, 0.3], [0.8, 0.7, 0.5]])  # 示例测试集向量

    # 示例：训练集和测试集的标签
    train_labels = np.array([0, 1])  # 示例训练集标签
    test_labels = np.array([1, 0])  # 示例测试集标签

    NB(train_vectors, train_labels, test_vectors, test_labels)
