from sklearn.feature_extraction.text import CountVectorizer
from data_preprocess import data_preprocess


def BoW(train_set, test_set):
    vectorizer = CountVectorizer()

    # 拟合训练数据并转换
    train_matrix = vectorizer.fit_transform(train_set)
    test_matrix = vectorizer.transform(test_set)

    # 查看训练集特征矩阵
    print("训练集特征矩阵:")
    print(train_matrix.toarray())
    print(train_matrix.shape)

    # 查看测试集特征矩阵
    print("\n测试集特征矩阵:")
    print(test_matrix.toarray())
    print(test_matrix.shape)

    # 查看词汇表
    print("\n词汇表:")
    print(vectorizer.get_feature_names_out())

    return train_matrix, test_matrix


if __name__ == '__main__':
    # 文件路径
    text_path = '../文本分类数据集/mr.txt'
    # labels路径
    label_path = '../文本分类数据集/mr_labels.txt'

    # 数据预处理
    texts_train, train_labels_list, texts_test, test_labels_list = (
        data_preprocess(text_path, label_path)
    )

    BoW(texts_train, texts_test)
