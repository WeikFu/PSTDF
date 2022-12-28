import os
import time

import javalang
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pandas as pd
from gensim.models.word2vec import Word2Vec
import Tools
import numpy as np
from deepforest import CascadeForestClassifier
from PSTDF import DTForest
import xml.dom.minidom
import csv
# 指定文件夹下所有源代码文件全部转换成csv
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
def csv2Ast(input_path, output_path):
    def parse_program(func):
        # 对源代码编译单元进行转换抽象语法树
        tree = javalang.parse.parse(func)
        # 对代码片段进行分析，转换成抽象语法树
        # tokens = javalang.tokenizer.tokenize(func)
        # parser = javalang.parser.Parser(tokens)
        # tree = parser.parse_member_declaration()
        return tree

    source = pd.read_csv(input_path, delimiter="\t", header=None, encoding='utf-8')
    source.columns = ['id', 'code', 'label']
    source['code'] = source['code'].apply(parse_program)
    # 抽象语法树持久化存储在pkl文件中
    source.to_pickle(output_path + "/ast.pkl")
# 将抽象语法树通过改进ASTNN编码层转换成向量
def Ast2VecImproved(input_path, output_path, filename):
    trees = pd.read_pickle(input_path)
    # AST转换成语句序列
    def Ast_to_sequences(ast):
        sequence = []
        Tools.get_sequence_BFS(ast, sequence)
        return sequence
    # 提取语句序列
    word_sequences = trees['code'].apply(Ast_to_sequences)
    str_sequences = [' '.join(c) for c in word_sequences]
    trees['code'] = pd.Series(str_sequences)
    # 使用语句序列训练word2vec模型
    # size:特征向量的维度,workers:并行数,sg=1:使用skip-gram算法,min_count:词频截断,少于min_count的词则丢弃
    model = Word2Vec(word_sequences, size=128, workers=16, sg=1, min_count=3)
    if not os.path.exists(output_path + '/embedding'):
        os.mkdir(output_path + '/embedding')
    # word2vec模型持久化
    model.save(output_path + '/embedding/' + str(128)+'improved')
    # 获取训练好的所有词
    vocab = model.wv.vocab
    max_token = model.wv.syn0.shape[0]
    def tree_to_index(node):
        token = node.token
        # 表达式子树裁剪，丢弃包引用等无用节点
        if token not in ['Import', 'PackageDeclaration']:
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result
    def trans2seq_BFS(r):
        blocks_bfs = []
        Tools.get_blocks_improved_BFS(r, blocks_bfs)
        tree = []
        for i in range(0, len(blocks_bfs)):
            btree_bfs = tree_to_index(blocks_bfs[i])
            tree.append(btree_bfs)
        return tree

    trees = pd.read_pickle(input_path)
    trees['code'] = trees['code'].apply(trans2seq_BFS)
    # 向量持久化到文件
    trees.to_pickle(output_path + "/" + filename)

def forest(Word2VecPath, dataPath):
    acca = []
    rcca = []
    f1a = []
    for num in range(0, 1):  #这个循环是计算平均性能用的，仅运行模型的话循环次数设置为1就行
        data = pd.read_pickle(dataPath)
        data_num = len(data)
        ratios = [int(r) for r in "6:4".split(':')]
        train_split = int(ratios[0] / sum(ratios) * data_num)
        data = data.sample(frac=1, random_state=1)
        train_data = data.iloc[:train_split]
        test_data = data.iloc[train_split:]
        word2vec = Word2Vec.load(Word2VecPath).wv
        embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
        embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
        ENCODE_DIM = 128
        # 标签种类数
        LABELS = 12
        MAX_TOKENS = word2vec.syn0.shape[0]
        EMBEDDING_DIM = word2vec.syn0.shape[1]
        model = DTForest(embedding_dim=EMBEDDING_DIM, vocab_size=MAX_TOKENS + 1, encode_dim=ENCODE_DIM,
                         label_size=LABELS)
        X_train, y_train = model.mg_scan(train_data)
        X_test, y_test = model.mg_scan(test_data)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        gc = CascadeForestClassifier(n_jobs=-1, n_estimators=8, n_trees=200, use_predictor=True)
        gc.fit(X_train, y_train)
        y_pred = gc.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        rcc = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        acca.append(acc)
        rcca.append(rcc)
        f1a.append(f1)
    print("\nTesting Accuracy: {:.4f} %".format(max(acca)))
    print("\nTesting Recall: {:.4f}".format(max(rcca)))
    print("\nTesting f1 score: {:.4f}".format(max(f1a)))



def main():
    # 源码读取路径
    input_file_path = "./datasets/OWASP/code"
    # tsv文件输出路径
    output_file_path = "./tsv/"
    # 源码的tsv读取路径
    code_csv_path = output_file_path + "code.tsv"
    # 训练、测试路径
    train_path = "./train"
    # 存储AST的pkl文件的路径
    ast_path = train_path + "/ast.pkl"
    # 对源码进行处理，转换成csv
    # code2csv(input_file_path, output_file_path)
    # 对csv格式的源码进行处理，转换成抽象语法树并持久化在pkl文件中
    # csv2Ast(code_csv_path, train_path)
    # 将训练集的AST转换成语句序列
    improved_start_time = time.time()
    Ast2VecImproved(ast_path, train_path, "data_improved.pkl")
    forest("C:/Users/fwk/Videos/MyAST/train/Experiment1/embedding_Experiment1/256improved", "C:/Users/fwk/Videos/MyAST/train/Experiment1/data_improved.pkl")
    improved_end_time = time.time()
    print("improved ASTNN time cost:%.4f s" % (improved_end_time - improved_start_time))
if __name__ == '__main__':
    main()
