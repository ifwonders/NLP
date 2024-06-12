import numpy as np
from gensim.models import Word2Vec


# 定义函数来计算文本的特征向量
def get_text_vector(text, model):
    words = [word for word in text if word in model.wv]
    if not words:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[words], axis=0)


def word2vec(train_set, test_set, vector_size=100, window=5, min_count=1, workers=4):
    # 训练Word2Vec模型
    model = Word2Vec(sentences=train_set, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

    # 提取训练集和测试集的特征
    train_vectors = np.array([get_text_vector(text, model) for text in train_set])
    test_vectors = np.array([get_text_vector(text, model) for text in test_set])

    print("训练集Word2Vec向量:")
    print(train_vectors)
    print(train_vectors.shape)

    print("\n测试集Word2Vec向量:")
    print(test_vectors)
    print(test_vectors.shape)

    return train_vectors, test_vectors


if __name__ == '__main__':
    # 假设预处理后的文本列表
    train_texts = [['this', 'is', 'a', 'sample', 'sentence'], ['another', 'example', 'sentence']]
    test_texts = [['this', 'is', 'a', 'test'], ['sample', 'test', 'text']]

    word2vec(train_texts, test_texts)
