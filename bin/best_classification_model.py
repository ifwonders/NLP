import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from bin.data_preprocess import text_preprocess, data_sets_preprocess
from bin.feature_extractions.bag_of_words import BoW

# 文件路径
text_path = 'bin/data_sets/mr.txt'
# 标签路径
label_path = 'bin/data_sets/mr_labels.txt'
# 训练模型保存路径
model_save_path = 'bin/text_classifier.joblib'


class classification_model:
    def __init__(self, is_new_model=True):
        self.model = None
        self.accuracy = 0.7507
        if is_new_model:
            self.init_model()
        else:
            self.load_model()

    def init_model(self):
        # 数据预处理
        texts_train, train_labels_list, texts_test, test_labels_list = (
            data_sets_preprocess(text_path, label_path)
        )

        # 选择特征提取方法 选用BoW方法
        bow_vectorizer = CountVectorizer()
        train_vectors, test_vectors = BoW(texts_train, texts_test)

        # 创建多个单独的模型
        svm_classifier = SVC(C=1.0, coef0=1.0, degree=2, gamma='scale', kernel='poly', probability=True)
        # dt_classifier = DecisionTreeClassifier(criterion='entropy',max_features='sqrt',min_samples_leaf=2,min_samples_split=2,splitter='random')
        rf_classifier = RandomForestClassifier(criterion='gini', max_depth=4, max_features='sqrt', min_samples_leaf=2,
                                               min_samples_split=2)

        # 训练模型
        svm_classifier.fit(train_vectors, train_labels_list)
        # dt_classifier.fit(train_vectors, train_labels_list)
        rf_classifier.fit(train_vectors, train_labels_list)

        # 创建一个投票集成模型
        ensemble_model = VotingClassifier(estimators=[
            ('svm', svm_classifier),
            # ('dt', dt_classifier),
            ('rf', rf_classifier)],
            voting='soft',
            weights=[0.75,
                     # 0.67,
                     0.72]
        )

        # 训练集成模型
        ensemble_model.fit(train_vectors, train_labels_list)

        # 使用make_pipeline函数创建流水线 串联向量化器和NLP模型
        self.model = make_pipeline(bow_vectorizer, ensemble_model)

        # 训练模型
        self.model.fit(texts_train, train_labels_list)

        # 使用模型进行预测
        prediction = self.model.predict(texts_test)

        # 评估模型
        self.accuracy = accuracy_score(test_labels_list, prediction)
        print(f"集成模型的准确率: {self.accuracy * 100:.2f}%")

        # 保存模型
        joblib.dump(self.model, model_save_path)

    def load_model(self):
        self.model = joblib.load(model_save_path)

    def predict_new_data(self, new_data):
        # 新文本
        new_text = text_preprocess(new_data)

        # 对新文本进行预测
        prediction = self.model.predict(new_text)

        print(prediction[0])

        return prediction[0]


if __name__ == '__main__':
    # 文件路径
    text_path = 'data_sets/mr.txt'
    # 标签路径
    label_path = 'data_sets/mr_labels.txt'
    # 训练模型保存路径
    model_save_path = 'text_classifier.joblib'

    my_model = classification_model()
    my_model.predict_new_data('an ugly-duckling tale so hideously and clumsily told it feels accidental . ')
