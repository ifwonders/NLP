from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def log_reg(train_vectors, train_labels, test_vectors, test_labels):
    logreg = LogisticRegression()

    # 训练模型
    logreg.fit(train_vectors, train_labels)

    # 使用测试集评估模型
    prediction = logreg.predict(test_vectors)

    # 计算准确率
    accuracy = accuracy_score(test_labels, prediction)
    print(f"\n准确率: {accuracy * 100:.2f}%")

    # 输出更详细的评估报告
    print("\n分类报告:")
    print(classification_report(test_labels, prediction))