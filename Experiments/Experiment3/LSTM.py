import os
import time

import jieba
from keras.models import Sequential
from keras.layers import *
import javalang
import csv
import xml.dom.minidom
from gensim.models.word2vec import Word2Vec
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, recall_score, f1_score
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

def code2csv(input_file, output_file):
    # 漏洞标签对应表
    vul2dic = {"none": 0}
    maxLabel = vul2dic["none"]
    # 设置输出目录及文件名
    outputf = open(output_file + "/code.tsv", "w", encoding="utf8", newline='')
    # 读取文件夹下的文件列表
    fileList = os.listdir(input_file)
    rows = []
    for i in range(0, len(fileList)):
        if fileList[i].split(".")[1] == "xml":
            continue
        else:
            # 读取java源代码
            inputf = open(input_file + "/" + fileList[i], "r")
            # 读取相应的xml描述文件，提取漏洞类型及标签
            dom = xml.dom.minidom.parse(input_file + "/" + fileList[i].split(".")[0] + ".xml")
            root = dom.documentElement
            # 提取是否存在漏洞的标记
            vulnerability = root.getElementsByTagName('vulnerability')
            # 漏洞存在则提取具体的漏洞类型
            label = vul2dic['none']
            if vulnerability[0].firstChild.data == "true":
                category = root.getElementsByTagName('category')[0].firstChild.data
                # 遍历漏洞标签对应表，若无该类型漏洞则添加，若有则给标签赋值
                noThisCategory = True
                for key in vul2dic:
                    if maxLabel < vul2dic[key]:
                        maxLabel = vul2dic[key]
                    if key == str(category):
                        label = vul2dic[key]
                        noThisCategory = False
                        break
                if noThisCategory:
                    maxLabel = maxLabel + 1
                    vul2dic[str(category)] = maxLabel
                    label = vul2dic[str(category)]
            else:
                # 无漏洞代码，标签标记为无漏洞
                label = vul2dic['none']
            s = inputf.read()
            s = s.replace("	", "    ")
            s = s.replace("\r\n", "\n")
            row = (i, s, label)
            rows.append(row)
            inputf.close()
    # 指定用制表符分割
    writer = csv.writer(outputf, delimiter='\t')
    # 写入CSV
    writer.writerows(rows)
    outputf.close()
    print(vul2dic)
    return None

# 从文件中读取java源代码，将源码中的类/函数转换成Token序列并存储为pkl文件
def csv2Token(input_path, output_path):
    def parse_program(func):
        # 对代码片段进行分析，转换成Token序列
        tokens = javalang.tokenizer.tokenize(func)
        final_token_list = ''
        for token in tokens:
            final_token_list += str(token.value)
        return final_token_list

    source = pd.read_csv(input_path, delimiter="\t", header=None, encoding='utf-8')
    source.columns = ['id', 'code', 'label']
    source['code'] = source['code'].apply(parse_program)
    # Token序列持久化存储在pkl文件中
    source.to_pickle(output_path + "/tokens.pkl")

def Token2Vec(input_path, output_path, filename):
    trees = pd.read_pickle(input_path)
    # 提取Token语句序列
    def get_token(token):
        return [token]
    word_sequences = trees['code'].apply(get_token)
    # 使用语句序列训练word2vec模型
    # size:特征向量的维度,workers:并行数,sg=1:使用skip-gram算法,min_count:词频截断,少于min_count的词则丢弃
    model = Word2Vec(word_sequences, size=128, workers=16, sg=1, min_count=1)
    if not os.path.exists(output_path + '/embedding'):
        os.mkdir(output_path + '/embedding')
    # word2vec模型持久化
    model.save(output_path + '/embedding/node_w2v_' + str(128))

    def trans2seq(token):
        v = model[token]
        v = np.array(v.tolist())
        return v
    trees['code'] = trees['code'].apply(trans2seq)
    print(trees['code'].values.shape)
    # 向量持久化到文件
    trees.to_pickle(output_path + "/" + filename)

# Seque构建方式
class SequeClassifier():
    def __init__(self, units):
        self.units = units
        self.model = None

    # 构建神经网络模型：（根据各层输入输出的shape）搭建网络结构、确定损失函数、确定优化器
    def build_model(self, loss, optimizer, metrics):
        self.model = Sequential()
        self.model.add(LSTM(self.units, return_sequences=True))
        self.model.add(LSTM(self.units))
        self.model.add(Dense(12, activation='softmax'))  # 最后一层全连接层。对于N分类问题，最后一层全连接输出个数为N个

        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics)

def token_lstm(data_path):
    data = pd.read_pickle(data_path)
    X_mid = data['code'].values
    y = data['label'].values
    X = []
    for X_item in X_mid:
        X_item = X_item.reshape(128)
        X.append(X_item)
    X = np.array(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    x_train = X_train[:, :, np.newaxis]
    x_test = X_test[:, :, np.newaxis]

    # 2 构建神经网络模型：（根据各层输入输出的shape）搭建网络结构、确定损失函数、确定优化器
    units = 128  # lstm细胞个数
    loss = "sparse_categorical_crossentropy"  # 损失函数类型
    optimizer = "adam"  # 优化器类型
    metrics = ['accuracy']  # 评估方法类型
    sclstm = SequeClassifier(units)
    sclstm.build_model(loss, optimizer, metrics)

    # 3 训练模型
    epochs = 100
    batch_size = 64
    sclstm.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # 4 模型评估
    score = sclstm.model.evaluate(x_test, y_test, batch_size=64)
    print("model score:", score)

    # 模型应用：预测
    # proba_prediction = sclstm.model.predict(x_test)

    read_model = sclstm.model
    out = read_model.predict(x_test)
    out = tf.nn.softmax(out)
    out = np.array(out)
    y_pred = np.argmax(out, axis=1)
    acc = accuracy_score(y_test, y_pred) * 100
    rcc = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print("\nTesting Accuracy: {:.3f} %".format(acc))
    print("\nTesting Recall: {:.4f}".format(rcc))
    print("\nTesting f1 score: {:.4f}".format(f1))


if __name__ == "__main__":
    # 源码读取路径
    input_file_path = "C:/Users/fwk/Videos/MyAST/train/Experiment4/SARD"
    # csv文件输出路径
    output_file_path = "C:/Users/fwk/Videos/MyAST/train/Experiment4/csv"
    # 源码的tsv读取路径
    code_csv_path = output_file_path + "/code.tsv"
    # 源码转换成Token序列的pkl文件存储路径
    token_pkl_path = "C:/Users/fwk/Videos/MyAST/OtherModel\Token+LSTM"
    # 对源码进行处理，转换成csv
    code2csv(input_file_path, output_file_path)
    # csv中的源码转换成Token序列存储到pkl文件
    # csv2Token(code_csv_path, token_pkl_path)
    # Token2Vec("C:/Users/fwk/Videos/MyAST/OtherModel/Token+LSTM/tokens.pkl", "C:/Users/fwk/Videos/MyAST/OtherModel/Token+LSTM", "data.pkl")
    # # 1 获取训练数据集，并调整为三维输入格式
    # origin_start_time = time.time()
    # token_lstm("C:/Users/fwk/Videos/MyAST/OtherModel/Token+LSTM/data.pkl")
    # origin_end_time = time.time()
    # print("Token+LSTM time cost:%.3f s" % (origin_end_time - origin_start_time))

