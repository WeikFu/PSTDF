import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



def create_model(input_path):
    acca = []
    recalla = []
    f1a = []
    for i in range(0, 1):
        source = pd.read_csv(input_path, delimiter=",", header=None, encoding='utf-8')
        source.columns = ['id0', 'id', 'code', 'label']
        data = source['code'].values
        category = source['label'].values
        x_train, x_test, y_train, y_test = train_test_split(data, category, test_size=0.4)
        # N-gram
        vectorizer = CountVectorizer(ngram_range=(2, 2))
        features = vectorizer.fit_transform(x_train)
        print("训练样本特征表长度为 " + str(features.shape))
        # print(vectorizer.get_feature_names()[3000:3050]) #特征名展示
        test_features = vectorizer.transform(x_test)
        print("测试样本特征表长度为 " + str(test_features.shape))
        # 支持向量机
        # C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0
        svmmodel = SVC()  # kernel：参数选择有rbf, linear, poly, Sigmoid, 默认的是"RBF";

        nn = svmmodel.fit(features, y_train)
        # predict = svmmodel.score(test_features ,d_test.sku)
        # print(predict)
        y_pred = svmmodel.predict(test_features)
        acc = accuracy_score(y_test, y_pred) * 100
        rcc = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        acca.append(acc)
        recalla.append(rcc)
        f1a.append(f1)
    print('average acc: {:.4f} %'.format(sum(acca) / len(acca)))
    print('average recall: {:.4f}'.format(sum(recalla) / len(recalla)))
    print('average f1: {:.4f}'.format(sum(f1a) / len(f1a)))


input_path = "C:/Users/fwk/Downloads/astnn-master/data/c/train/programs_ns.tsv"
create_model(input_path)





