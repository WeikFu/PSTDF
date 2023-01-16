# BiLSTM for sequence classification in the IMDB dataset
import time
import jieba
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from numpy import shape
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
# 随机种子
# lstm细胞数
units = 128
# 多分类问题类别数
num_class = 12
# 迭代轮数
epochs = 50
# 每批次数据大小
batch_size = 64
# 词嵌入向量的维数
embedding_vecor_length = 128
timecosta = []
acca = []
recalla = []
f1a = []
for i in range(0, 1):
    input_path = "C:/Users/fwk/Videos/MyAST/csv/code.tsv"
    # input_path = "C:/Users/fwk/Videos/MyAST/train/Experiment4/csv/code.tsv"
    source = pd.read_csv(input_path, delimiter="\t", header=None, encoding='utf-8')
    source.columns = ['id', 'code', 'label']
    text = source['code'].values
    category = source['label'].values

    corpus = []
    for t in text:
        corpus.append(list(jieba.cut(t)))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    vocab = tokenizer.word_index

    x_train, x_test, y_train, y_test = train_test_split(corpus, category, test_size=0.4)
    # 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
    x_train_word_ids = tokenizer.texts_to_sequences(x_train)
    x_test_word_ids = tokenizer.texts_to_sequences(x_test)
    # 序列模式
    # 每条样本长度不唯一，将每条样本的长度设置一个固定值
    origin_start_time = time.time()
    max_review_length = 1000
    X_train = sequence.pad_sequences(x_train_word_ids, maxlen=max_review_length)
    X_test = sequence.pad_sequences(x_test_word_ids, maxlen=max_review_length)
    print(shape(X_train))
    print(type(X_train[0]))
    # 训练词向量
    word2vec_model = Word2Vec(sentences=corpus, min_count=3, workers=16, sg=1, size=embedding_vecor_length)
    embedding_matrix = np.zeros((len(vocab) + 1, embedding_vecor_length))
    for word, i in vocab.items():
        try:
            embedding_vector = word2vec_model[str(word)]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue

    # create the model
    vocab = tokenizer.word_index
    n_timesteps = 10
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, embedding_vecor_length, input_length=max_review_length, weights=[embedding_matrix]))
    model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=(n_timesteps, 1)))
    model.add(Bidirectional(LSTM(units)))
    model.add(Dense(num_class, activation='softmax'))  # 最后一层全连接层。对于N分类问题，最后一层全连接输出个数为N个
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Final evaluation of the model
    out = model.predict(X_test)
    out = tf.nn.softmax(out)
    out = np.array(out)
    y_pred = np.argmax(out, axis=1)
    acc = accuracy_score(y_test, y_pred) * 100
    rcc = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("\nTesting Accuracy: {:.4f} %".format(acc))
    print("\nTesting Recall: {:.4f}".format(rcc))
    print("\nTesting f1 score: {:.4f}".format(f1))
    origin_end_time = time.time()
    acca.append(acc)
    recalla.append(rcc)
    f1a.append(f1)
    timecosta.append(origin_end_time - origin_start_time)
print('average timecost: {:.4f} s'.format(sum(timecosta)/len(timecosta)))
print('average acc: {:.4f} %'.format(sum(acca)/len(acca)))
print('average recall: {:.4f}'.format(sum(recalla)/len(recalla)))
print('average f1: {:.4f}'.format(sum(f1a)/len(f1a)))
