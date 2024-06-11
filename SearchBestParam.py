from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

"""
在通过交叉验证和网格搜索优化SVM参数的过程中，通常不需要使用测试集向量，原因如下：
1. 交叉验证的目的：交叉验证是一种评估模型泛化能力的方法，它通过将训练集分为多个折叠（folds），并在每个折叠上轮流进行训练和验证，以此来估计模型在独立数据上的性能。这个过程本身就是为了模拟测试集的效果，因此不需要额外的测试集。
2. 避免数据泄露：测试集是用于评估模型最终性能的独立数据集，不应该在模型选择或参数优化的过程中使用。如果在优化过程中使用了测试集，可能会导致对模型性能的过高估计，这是因为模型可能过度适应了测试集中的特定特征，从而失去了泛化到新数据的能力。
3. 网格搜索的作用：网格搜索是在给定的参数空间中搜索最佳参数组合的过程。它通过交叉验证来评估每个参数组合的性能，并选择最佳的一个。这个过程完全在训练集上进行，因此不需要测试集。
总结来说，交叉验证和网格搜索的过程完全在训练集上进行，用于选择最佳的模型参数。一旦找到了最佳的参数组合，你可以在训练好的模型上使用测试集向量来评估模型的性能，这样可以更准确地估计模型在实际应用中的表现。
"""


def SearchBestParam(train_vectors, train_labels, estimator_name, **params):
    """
    GridSearchCV参数列表的解释：
        estimator： 这是你想要优化的模型或估计器。你需要提供这个估计器的实例，例如SVC()。
        param_grid： 参数网格是一个字典或列表，其中包含估计器的参数及其可能的值。例如，对于SVC，param_grid可能包含C、kernel、gamma等参数及其可能的值。
        scoring： 这是用于评估每个参数组合性能的评分函数。如果你没有提供，GridSearchCV将使用accuracy_score作为默认评分函数。
        n_jobs： 这个参数指定GridSearchCV在交叉验证过程中应使用的并行作业数。如果设置为-1，将使用所有可用的CPU核心。
        refit： 这个参数指定是否在找到最佳参数组合后重新拟合模型。如果为True，找到最佳参数组合后将使用所有训练数据重新拟合模型；如果为False，则不重新拟合。
        cv： 这是交叉验证的折数。如果你没有提供，GridSearchCV将使用默认的5折交叉验证。
        verbose： 这个参数指定是否打印网格搜索的进度信息。如果为0，将不打印任何信息；如果为1，将打印基本信息；如果为2或更高，将打印更多详细信息。
        pre_dispatch： 这个参数指定在开始网格搜索之前预先分发的任务数。它是一个整数，指定在启动任何工作之前应并行执行的任务数。
        error_score： 这个参数用于指定在某些情况下，如内存不足，应该返回的错误分数。默认值为np.nan，表示模型无法训练。
        return_train_score： 这个参数指定是否在交叉验证中返回训练集的分数。如果为True，GridSearchCV将返回训练集和测试集的分数；如果为False，将只返回测试集的分数。
    :param estimator_name: 这是你想要优化的模型或估计器。你需要提供这个模型的名称，如 'DTC'。
    :param train_vectors:训练集向量
    :param train_labels:训练集标签
    :param params:可选参数
    :return:最佳表现与达成最佳表现的参数集
    """
    param_grid = {}
    for key, value in params.items():
        param_grid[key] = value
    # SVC参数集示例
    # param_grid = {
    #     'C': [0.1, 1, 10],
    #     'kernel': ['linear', 'rbf'],
    #     'gamma': ['scale', 'auto'],
    #     'degree': [2, 3, 4],
    #     'coef0': [0.0, 1.0]
    # }

    if estimator_name == 'DTC':
        estimator = DecisionTreeClassifier()
    elif estimator_name == 'RFC':
        estimator = RandomForestClassifier()
    elif estimator_name == 'LogReg':
        estimator = LogisticRegression()
    elif estimator_name == 'SVM':
        estimator = SVC()
    else:
        estimator = None

    # cv=5表示使用5折交叉验证，scoring='accuracy' 表示使用准确率作为评估标准，n_jobs=-1表示使用所有可用的CPU核心来加速计算。
    grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # 使用训练数据拟合GridSearchCV模型
    grid_search.fit(train_vectors, train_labels)

    # 训练完成后，可以查看最佳的参数组合和对应的分数
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("最佳参数集:", best_params)
    print("\n最佳表现:", best_score)


if __name__ == '__main__':
    train_vec = None
    train_labels = None

    SearchBestParam(train_vec, train_labels, estimator_name='SVM')
