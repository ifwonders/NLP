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
    """
    以下是DecisionTreeClassifier参数列表的解释：
        criterion： 这是用于选择最优分割的特征评估准则。默认值为"gini"，也可以选择"entropy"。
        splitter： 这是用于选择最佳分割的特征。默认值为"best"，也可以选择"random"。
        max_depth： 这是决策树的最大深度。如果设置为None，则树会生长到所有叶子都是纯的为止。
        min_samples_split： 这是在节点分裂时所需的最小样本数。默认值为2，这意味着至少需要2个样本才能分裂一个节点。
        min_samples_leaf： 这是叶节点所需的最小样本数。默认值为1，这意味着至少需要1个样本才能形成一个叶节点。
        min_weight_fraction_leaf： 这是叶节点所需的最小权重比例。默认值为0.0，这意味着如果所有样本的权重总和为0，则该节点将被视为纯的，并成为叶节点。
        max_features： 这是在决策树生长过程中考虑的最大特征数。如果设置为None，则考虑所有特征；如果设置为一个整数，则考虑该整数个特征；如果设置为"auto"，则考虑所有特征，但不超过sqrt(n_features)；如果设置为"sqrt"，则考虑所有特征，但不超过sqrt(n_features)；如果设置为"log2"，则考虑所有特征，但不超过log2(n_features)。
        random_state： 这是用于设置随机种子，以确保结果的可重复性。如果设置为None，则使用默认的随机种子。
        max_leaf_nodes： 这是决策树生长过程中考虑的最大叶子节点数。如果设置为None，则没有限制。
        min_impurity_decrease： 这是在决策树生长过程中所需的最小不纯度减少。如果设置为0.0，则没有限制。
        class_weight： 这是用于为不同类别的样本分配不同权重。如果设置为"balanced"，则自动计算权重，以使得每个类别的样本数与整个样本数的比例相同。
        ccp_alpha： 这是在剪枝过程中使用的最小核心纯度（core purity）。如果设置为0.0，则没有限制。
        monotonic_cst： 这是用于限制决策树生长的单调性约束。如果设置为None，则没有限制。
    :param train_vectors:训练集向量
    :param train_labels:训练集标签
    :param test_vectors:测试集向量
    :param test_labels:测试集标签
    :return:null
    """
    dt = DecisionTreeClassifier()
    # 训练模型
    dt.fit(train_vectors, train_labels)
    # 使用测试集评估模型
    prediction = dt.predict(test_vectors)
    # 计算准确率
    accuracy = accuracy_score(test_labels, prediction)
    print(f"\n决策树模型准确率: {accuracy * 100:.2f}%")

    # 输出更详细的评估报告
    print("\n决策树模型分类报告:")
    print(classification_report(test_labels, prediction))


def RFC(train_vectors, train_labels, test_vectors, test_labels):
    """
    以下是RandomForestClassifier参数列表的解释：
        n_estimators： 这是随机森林中树的数量。默认值为100，你可以根据数据集的大小和复杂性调整这个值。
        criterion： 这是用于选择最优分割的特征评估准则。默认值为"gini"，也可以选择"entropy"。
        max_depth： 这是树的最大深度。如果设置为None，则树会生长到所有叶子都是纯的为止。
        min_samples_split： 这是在节点分裂时所需的最小样本数。默认值为2，这意味着至少需要2个样本才能分裂一个节点。
        min_samples_leaf： 这是叶节点所需的最小样本数。默认值为1，这意味着至少需要1个样本才能形成一个叶节点。
        min_weight_fraction_leaf： 这是叶节点所需的最小权重比例。默认值为0.0，这意味着如果所有样本的权重总和为0，则该节点将被视为纯的，并成为叶节点。
        max_features： 这是在决策树生长过程中考虑的最大特征数。如果设置为"sqrt"，则考虑所有特征，但不超过sqrt(n_features)；如果设置为"log2"，则考虑所有特征，但不超过log2(n_features)；如果设置为None，则考虑所有特征。
        max_leaf_nodes： 这是决策树生长过程中考虑的最大叶子节点数。如果设置为None，则没有限制。
        min_impurity_decrease： 这是在决策树生长过程中所需的最小不纯度减少。如果设置为0.0，则没有限制。
        bootstrap： 这是用于决定是否使用有放回的随机抽样来构建树。默认值为True，表示使用有放回的随机抽样；如果设置为False，则不使用有放回的随机抽样。
        oob_score： 这是用于决定是否使用袋外数据（Out-of-Bag data）来评估模型性能。默认值为False，表示不使用袋外数据；如果设置为True，则使用袋外数据。
        n_jobs： 这是用于指定RandomForestClassifier在构建树时应使用的并行作业数。如果设置为-1，将使用所有可用的CPU核心。
        random_state： 这是用于设置随机种子，以确保结果的可重复性。如果设置为None，则使用默认的随机种子。
        verbose： 这是用于指定是否打印随机森林的进度信息。如果设置为0，将不打印任何信息；如果设置为1，将打印基本信息；如果设置为2或更高，将打印更多详细信息。
        warm_start： 这是用于指定是否在重新训练模型时保留先前的模型参数。如果设置为True，将保留先前的模型参数；如果设置为False，将重新初始化模型参数。
        class_weight： 这是用于为不同类别的样本分配不同权重。如果设置为"balanced"，则自动计算权重，以使得每个类别的样本数与整个样本数的比例相同。
        ccp_alpha： 这是在剪枝过程中使用的最小核心纯度（core purity）。如果设置为0.0，则没有限制。
        max_samples： 这是用于指定在构建树时使用的最大样本数。如果设置为None，则使用所有样本；如果设置为一个整数，则使用该整数个样本。
        monotonic_cst： 这是用于限制决策树生长的单调性约束。如果设置为None，则没有限制。
    :param train_vectors:训练集向量
    :param train_labels:训练集标签
    :param test_vectors:测试集向量
    :param test_labels:测试集标签
    :return:null
    """
    rf = RandomForestClassifier()
    # 训练模型
    rf.fit(train_vectors, train_labels)
    # 使用测试集评估模型
    prediction = rf.predict(test_vectors)
    # 计算准确率
    accuracy = accuracy_score(test_labels, prediction)
    print(f"\n随机森林模型准确率: {accuracy * 100:.2f}%")

    # 输出更详细的评估报告
    print("\n随机森林模型分类报告:")
    print(classification_report(test_labels, prediction))



