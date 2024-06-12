import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from data_preprocess import text_preprocess, data_sets_preprocess

# 文件路径
text_path = 'data_sets/mr.txt'
# 标签路径
label_path = 'data_sets/mr_labels.txt'
# 训练模型保存路径
model_save_path = 'text_classifier.joblib'


class classification_model:
    def __init__(self, is_new_model=True):
        self.model = None
        if is_new_model:
            self.init_model()
        else:
            self.load_model()

    def init_model(self):
        # 数据预处理
        texts_train, train_labels_list, texts_test, test_labels_list = (
            data_sets_preprocess(text_path, label_path)
        )

        # 选择特征提取方法 以TF-IDF为例
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        # 选择模型 以SVM为例
        svm_classifier = SVC(kernel='linear', C=1.0)
        # 使用make_pipeline函数创建流水线 串联向量化器和NLP模型
        self.model = make_pipeline(tfidf_vectorizer, svm_classifier)

        # 训练模型
        self.model.fit(texts_train, train_labels_list)

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
    my_model = classification_model()
    my_model.predict_new_data('an ugly-duckling tale so hideously and clumsily told it feels accidental . ')
