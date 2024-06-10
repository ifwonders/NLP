from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def SVM(train_vectors, train_labels, test_vectors, test_labels):
    svm_model = SVC(kernel='linear')
    # 训练模型
    svm_model.fit(train_vectors, train_labels)

    prediction = svm_model.predict(test_vectors)

    accuracy = accuracy_score(test_labels, prediction)
    print(f"\n准确率: {accuracy * 100:.2f}%")

    # 输出更详细的评估报告
    print("\n分类报告:")
    print(classification_report(test_labels, prediction))


def SearchBestParam(train_vectors, train_labels, **params):
    """
    通过交叉验证与网格搜索优化SVM参数
    SVM参数示例
    C：正则化参数，用于控制误分类或分界面的平滑性。C值大，意味着分类器会选择一个较小的间隔，以减少训练样本上的错误分类。C值小，分类器会尝试最大化间隔，允许更多的错误分类。默认值为1.0。
    kernel：用于训练模型的核函数。它决定了数据在更高维空间中的映射方式。常用的核函数有：
        ‘linear’：线性核，适用于特征数量很多的情况或线性可分的数据。
        ‘rbf’：径向基函数（Radial Basis Function），通常用于非线性问题。
        ‘poly’：多项式核，将数据映射到更高维的空间。
        ‘sigmoid’：Sigmoid核，可以用于模拟神经元的激活函数。
        ‘precomputed’：使用预计算的距离度量作为核函数。
    degree：当选择’poly’核时，这是多项式的度。默认值为3。
    gamma：核函数的系数。对于’rbf’，'poly’和’sigmoid’核，gamma定义了单一训练样本的影响范围。值越大，影响范围越小。默认值为"scale"，它根据训练集的大小自动设置gamma。
    coef0：核函数的独立项。对于’poly’和’sigmoid’核，它是一个自由参数。
    shrinking：是否使用收缩启发式。如果为True，算法会在每次迭代中减少支持向量的数量，以提高效率。
    probability：是否启用概率估计。如果为True，SVC会采用交叉验证来估计概率。
    tol：容忍度，用于训练过程中的收敛判断。如果连续两次迭代之间的差异小于tol，训练将停止。
    cache_size：指定内核缓存的大小（以MB为单位）。默认值为200MB。
    class_weight：用于设定不同类别的权重。如果设为’balanced’，则根据训练数据的类别频率自动调整权重。
    verbose：是否启用详细输出。如果为True，模型会输出训练过程中的详细信息。
    max_iter：最大迭代次数。默认值为-1，表示不限制迭代次数。
    decision_function_shape：决定决策函数的形状。'ovr’表示一对多（One-vs-Rest），'ovo’表示一对一（One-vs-One）。
    break_ties：如果为True，则在预测时选择具有最高分数的类（如果有并列分数）。这需要额外的计算。
    random_state：随机数生成器的种子，用于概率估计的参数。在多次实验中，设置相同的random_state可以保证结果的可重复性。

    :param train_vectors:训练集向量
    :param train_labels:训练集标签
    :param params:可选参数
    :return:最佳表现与达成最佳表现的参数集
    """
    param_grid = {}
    for key, value in params.items():
        param_grid[key] = value
    # 参数集示例
    # param_grid = {
    #     'C': [0.1, 1, 10],
    #     'kernel': ['linear', 'rbf'],
    #     'gamma': ['scale', 'auto'],
    #     'degree': [2, 3, 4],
    #     'coef0': [0.0, 1.0]
    # }

    svc = SVC()

    # cv=5表示使用5折交叉验证，scoring='accuracy' 表示使用准确率作为评估标准，n_jobs=-1表示使用所有可用的CPU核心来加速计算。
    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # 使用训练数据拟合GridSearchCV模型
    grid_search.fit(train_vectors, train_labels)

    # 训练完成后，可以查看最佳的参数组合和对应的分数
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("最佳参数集:", best_params)
    print("最佳表现:", best_score)
