import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# 构建总单词矩阵
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
acca = []
recalla = []
f1a = []
for i in range(0, 1):
    # input_path = "C:/Users/fwk/Videos/MyAST/csv/code.tsv"
    input_path = "C:/Users/fwk/Downloads/astnn-master/data/c/train/programs_ns.tsv"
    source = pd.read_csv(input_path, delimiter=",", header=None, encoding='utf-8')
    source.columns = ['id0', 'id', 'code', 'label']
    text = source['code'].values
    category = source['label'].values
    X_train, X_test, y_train, y_test = train_test_split(text, category, test_size=0.4)
    count_v0 = CountVectorizer();
    counts_all = count_v0.fit_transform(text);  # all_text为训练集+测试集语料库# 构建训练集单词矩阵
    count_v1 = CountVectorizer(vocabulary=count_v0.vocabulary_)
    n_component = 104
    counts_train = count_v1.fit_transform(X_train)  # 构建测试集单词矩阵
    count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_)
    counts_test = count_v2.fit_transform(X_test);
    from sklearn.decomposition import LatentDirichletAllocation

    lda = LatentDirichletAllocation(n_components=n_component, max_iter=300, learning_method='batch')
    x_train = lda.fit(counts_train).transform(counts_train)
    x_test = lda.fit(counts_test).transform(counts_test)
    svclf = SVC(kernel='linear')
    svclf.fit(x_train, y_train)
    y_pred = svclf.predict(x_test)
    acc = accuracy_score(y_test, y_pred) * 100
    rcc = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    acca.append(acc)
    recalla.append(rcc)
    f1a.append(f1)
print('average acc: {:.4f} %'.format(sum(acca) / len(acca)))
print('average recall: {:.4f}'.format(sum(recalla) / len(recalla)))
print('average f1: {:.4f}'.format(sum(f1a) / len(f1a)))
