from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

"""
决策树（Decision Tree）和随机森林（Random Forest）都是常用的机器学习算法，用于分类和回归任务。它们的主要区别在于它们的结构和训练方式。

决策树：
结构：决策树是一种树形结构，其中每个内部节点代表一个特征，每个分支代表一个特征的取值，每个叶节点代表一个类别标签。
工作原理：决策树通过一系列的判断规则来进行分类。从根节点开始，根据特征的取值沿着树向下遍历，直到达到一个叶节点，该叶节点给出了预测结果。
优点：决策树易于理解，不需要特征缩放，可以处理数值型和类别型数据，且生成过程快速。
缺点：决策树容易过拟合，对训练数据中的噪声敏感，且泛化能力通常不如集成方法。

随机森林：
结构：随机森林是由多个决策树组成的集成学习方法。它在训练过程中构建多棵树，每棵树都是基于随机采样得到的数据集进行训练。
工作原理：随机森林通过多数投票（分类任务）或平均预测（回归任务）来综合多棵树的预测结果。每棵树都是在不同的数据样本和特征子集上训练的，这使得随机森林具有更好的泛化能力和鲁棒性。
优点：随机森林通常比单棵决策树具有更好的泛化能力，不容易过拟合，可以对大量特征进行筛选，且对噪声和异常值不敏感。
缺点：随机森林的训练时间通常比单棵决策树长，模型解释性不如单棵决策树，且在某些数据集上可能不如其他复杂的模型。
总结来说，决策树是一种基础的分类和回归算法，而随机森林是一种基于决策树的集成学习方法。随机森林通过多棵树的组合来提高性能和泛化能力，通常在多种数据集上表现良好。相比之下，决策树更简单，更容易理解，但在复杂的数据集上可能不如随机森林表现好。
"""


def DTC(train_vectors, train_labels, test_vectors, test_labels):
    dt = DecisionTreeClassifier()
    # 训练模型
    dt.fit(train_vectors, train_labels)
    # 使用测试集评估模型
    prediction = dt.predict(test_vectors)
    # 计算准确率
    accuracy = accuracy_score(test_labels, prediction)
    print(f"\n准确率: {accuracy * 100:.2f}%")

    # 输出更详细的评估报告
    print("\n分类报告:")
    print(classification_report(test_labels, prediction))


def RFC(train_vectors, train_labels, test_vectors, test_labels):
    rf = RandomForestClassifier()
    # 训练模型
    rf.fit(train_vectors, train_labels)
    # 使用测试集评估模型
    prediction = rf.predict(test_vectors)
    # 计算准确率
    accuracy = accuracy_score(test_labels, prediction)
    print(f"\n准确率: {accuracy * 100:.2f}%")

    # 输出更详细的评估报告
    print("\n分类报告:")
    print(classification_report(test_labels, prediction))



