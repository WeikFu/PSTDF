import pandas as pd
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import jieba
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model

#测试TextCNN在两个数据集上的性能

# 构建TextCNN模型
def TextCNN(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, embedding_matrix, vocab, num_classes):
    acca = []
    recalla = []
    f1a = []
    for i in range(0,1):
        # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
        main_input = Input(shape=(500,), dtype='float64')
        # 词嵌入（使用预训练的词向量）
        embedder = Embedding(len(vocab) + 1, 100, input_length=500, weights=[embedding_matrix], trainable=False)
        # embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
        embed = embedder(main_input)
        # 词窗大小分别为3,4,5
        cnn1 = Conv1D(128, 1, padding='same', strides=1, activation='relu')(embed)
        cnn1 = MaxPooling1D(pool_size=5)(cnn1)
        cnn2 = Conv1D(128, 2, padding='same', strides=1, activation='relu')(embed)
        cnn2 = MaxPooling1D(pool_size=4)(cnn2)
        # 合并模型的输出向量
        cnn = concatenate([cnn1, cnn2], axis=1)
        flat = Flatten()(cnn)
        drop = Dropout(0.5)(flat)
        main_output = Dense(num_classes, activation='softmax')(drop)
        model = Model(inputs=main_input, outputs=main_output)

        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        one_hot_labels = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)  # 将标签转换为one-hot编码
        model.fit(x_train_padded_seqs, one_hot_labels, batch_size=64, epochs=50)
        # y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
        result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
        result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
        y_predict = list(map(int, result_labels))
        acc = metrics.accuracy_score(y_test, y_predict)
        rcc = metrics.recall_score(y_test, y_predict, average='weighted')
        f1 = metrics.f1_score(y_test, y_predict, average='weighted')
        acca.append(acc)
        recalla.append(rcc)
        f1a.append(f1)
    print('average acc: {:.4f} %'.format(sum(acca) / len(acca)))
    print('average recall: {:.4f}'.format(sum(recalla) / len(recalla)))
    print('average f1: {:.4f}'.format(sum(f1a) / len(f1a)))

def trainTextCNN(input_path,num_classes):
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

    x_train, x_test, y_train, y_test = train_test_split(corpus, category, test_size=0.4, random_state=1)
    # 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
    x_train_word_ids = tokenizer.texts_to_sequences(x_train)
    x_test_word_ids = tokenizer.texts_to_sequences(x_test)
    # 序列模式
    # 每条样本长度不唯一，将每条样本的长度设置一个固定值
    x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=500)  # 将超过固定值的部分截掉，不足的在最前面用0填充
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=500)

    # 训练词向量
    word2vec_model = Word2Vec(sentences=corpus, min_count=3, workers=16, sg=1, seed=1)

    embedding_matrix = np.zeros((len(vocab) + 1, 100))
    for word, i in vocab.items():
        try:
            embedding_vector = word2vec_model[str(word)]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue
    TextCNN(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, embedding_matrix, vocab, num_classes)

np.random.seed(1)
tf.random.set_seed(1)
# 源码的tsv读取路径
SARD_code_csv_path = "C:/Users/fwk/Videos/MyAST/train/Experiment4/csv/code.tsv"
num_classes = 13
trainTextCNN(SARD_code_csv_path, num_classes)

