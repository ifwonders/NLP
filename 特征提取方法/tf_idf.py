from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def tf_idf(train_set, test_set):
    vectorizer = CountVectorizer()
    # 对训练集文本进行词频统计
    train_counts = vectorizer.fit_transform(train_set)

    transformer = TfidfTransformer()

    # 使用训练集的词频统计结果来训练TF-IDF转换器
    transformer.fit(train_counts)

    # 将训练集的词频统计结果转换为TF-IDF特征向量
    train_tfidf = transformer.transform(train_counts)

    # 对测试集文本进行词频统计
    test_counts = vectorizer.transform(test_set)

    # 将测试集词频转换为TF-IDF特征向量
    test_tfidf = transformer.transform(test_counts)

    print("训练集TF-IDF特征向量:")
    print(train_tfidf.toarray())
    print(train_tfidf.shape)
    print("\n测试集TF-IDF特征向量:")
    print(test_tfidf.toarray())
    print(test_tfidf.shape)

    return train_tfidf, test_tfidf
