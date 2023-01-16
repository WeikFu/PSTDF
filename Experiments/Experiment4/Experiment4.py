import csv
import os
import time
import xml.dom.minidom
from Tools import GRU as gru
import javalang
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_recall_fscore_support
import pandas as pd
from gensim.models.word2vec import Word2Vec
import Tools
import numpy as np
from deepforest import CascadeForestClassifier
from MyModel import DTForest
from torch.autograd import Variable

#实验四：在SARD数据集上验证I_ASTNN和原始ASTNN对7种不同CWE编号的多分类性能
np.random.seed(1)
# 对SARD数据集的源码进行处理，转换成csv,并返回一个标签分类字典
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
    return vul2dic
# 从文件中读取java源代码，将源码中的类/函数转换成抽象语法树并存储为pkl文件
def csv2Ast(input_path, output_path,filename):
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
    # 代码转成AST之后切割其中函数的抽象语法子树，重建数据集
    # ......
    # 抽象语法树持久化存储在pkl文件中
    source.to_pickle(output_path + "/"+filename)
# 将OWASP的抽象语法树通过I_ASTNN编码层转换成向量
# 将抽象语法树通过原始ASTNN编码层转换成向量
def Ast2Vec(input_path, output_path, filename):
    trees = pd.read_pickle(input_path)
    # AST转换成语句序列
    def Ast_to_sequences(ast):
        sequence = []
        Tools.get_sequence(ast, sequence)
        return sequence
    # 提取语句序列
    word_sequences = trees['code'].apply(Ast_to_sequences)
    str_sequences = [' '.join(c) for c in word_sequences]
    trees['code'] = pd.Series(str_sequences)
    # 使用语句序列训练word2vec模型
    # size:特征向量的维度,workers:并行数,sg=1:使用skip-gram算法,min_count:词频截断,少于min_count的词则丢弃
    model = Word2Vec(word_sequences, size=128, workers=16, sg=1, min_count=3, seed=1)
    if not os.path.exists(output_path + '/embedding_Experiment4'):
        os.mkdir(output_path + '/embedding_Experiment4')
    # word2vec模型持久化
    model.save(output_path + '/embedding_Experiment4/' +str(256))
    # 获取训练好的所有词
    vocab = model.wv.vocab
    max_token = model.wv.syn0.shape[0]
    def tree_to_index(node):
        token = node.token
        result = [vocab[token].index if token in vocab else max_token]
        children = node.children
        for child in children:
            result.append(tree_to_index(child))
        return result
    def trans2seq(r):
        blocks = []
        Tools.get_blocks(r, blocks)
        tree = []
        # 语句序列拼接
        for b in blocks:
            btree = tree_to_index(b)
            tree.append(btree)
        return tree
    trees = pd.read_pickle(input_path)
    trees['code'] = trees['code'].apply(trans2seq)
    # 向量持久化到文件
    trees.to_pickle(output_path + "/" + filename)
def GRU(Word2VecPath, dataPath):
    def get_batch(dataset, idx, bs):
        tmp = dataset.iloc[idx: idx + bs]
        data, labels = [], []
        for _, item in tmp.iterrows():
            data.append(item[1])
            labels.append(item[2])
        return data, torch.LongTensor(labels)

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
    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    # 标签种类数
    LABELS = 12
    BATCH_SIZE = 64
    USE_GPU = False
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    EPOCHS = 2
    model = gru(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE, USE_GPU, embeddings)
    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.CrossEntropyLoss()
    best_model = model
    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(train_data):
            batch = get_batch(train_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            train_inputs, train_labels = batch
            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train_inputs)
            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item() * len(train_inputs)

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
        # validation epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(train_data):
            batch = get_batch(train_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            val_inputs, val_labels = batch
            if USE_GPU:
                val_inputs, val_labels = val_inputs, val_labels.cuda()

            model.batch_size = len(val_labels)
            model.hidden = model.init_hidden()
            output = model(val_inputs)
            loss = loss_function(output, Variable(val_labels))
            # calc valing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item() * len(val_inputs)
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        end_time = time.time()
        if total_acc / total > best_acc:
            best_model = model
        print('[Epoch:%3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    predicts = []
    trues = []
    i = 0
    model = best_model

    while i < len(test_data):
        batch = get_batch(test_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        test_inputs, test_labels = batch
        if USE_GPU:
            test_inputs, test_labels = test_inputs, test_labels.cuda()
        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        output = model(test_inputs)

        loss = loss_function(output, Variable(test_labels))
        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
        trues.extend(test_labels.cpu().numpy())
        predicts.extend(predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='macro')

    print("Testing results(Acc,Recall,F1):%.4f, %.4f, %.4f" % (total_acc.item() / total, recall, f1))
def Ast2VecImproved(input_path, output_path, filename):
    trees = pd.read_pickle(input_path)

    # AST转换成语句序列
    def Ast_to_sequences(ast):
        sequence = []
        Tools.get_sequence_improved(ast, sequence)
        return sequence

    # 提取语句序列
    word_sequences = trees['code'].apply(Ast_to_sequences)
    str_sequences = [' '.join(c) for c in word_sequences]
    trees['code'] = pd.Series(str_sequences)
    # 使用语句序列训练word2vec模型
    # size:特征向量的维度,workers:并行数,sg=1:使用skip-gram算法,min_count:词频截断,少于min_count的词则丢弃
    model = Word2Vec(word_sequences, size=128, workers=16, sg=1, min_count=3, seed=1)
    if not os.path.exists(output_path + '/embedding_Experiment4'):
        os.mkdir(output_path + '/embedding_Experiment4')
    # word2vec模型持久化
    model.save(output_path + '/embedding_Experiment4/' + str(256) + 'improved')
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
    model = DTForest(embedding_dim=EMBEDDING_DIM, vocab_size=MAX_TOKENS + 1, encode_dim=ENCODE_DIM, label_size=LABELS)
    X_train, y_train = model.mg_scan(train_data)
    X_test, y_test = model.mg_scan(test_data)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # gc = gcForest(shape_1X=len(X_train[0]), window=1, min_samples_mgs=10, min_samples_cascade=7)
    gc = CascadeForestClassifier(random_state=1, n_jobs=16, n_estimators=8, n_trees=500, use_predictor=True)
    gc.fit(X_train, y_train)
    y_pred = gc.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    rcc = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("\nTesting Accuracy: {:.4f} %".format(acc))
    print("\nTesting Recall: {:.4f}".format(rcc))
    print("\nTesting f1 score: {:.4f}".format(f1))




def main():
    # 源码读取路径
    input_file_path = "C:/Users/fwk/Videos/MyAST/train/Experiment4/SARD"
    # csv文件输出路径
    output_file_path = "C:/Users/fwk/Videos/MyAST/train/Experiment4/csv"
    # 源码的tsv读取路径
    code_csv_path = output_file_path + "/code.tsv"
    # 源码转换成AST的pkl文件存储路径
    ast_pkl_path = "C:/Users/fwk/Videos/MyAST/train/Experiment4"
    # 对SARD数据集的源码进行处理，转换成csv
    # code2csv(input_file_path, output_file_path)
    # 对csv格式的SARD数据集源码进行处理，转换成抽象语法树并持久化在pkl文件中
    # csv2Ast(code_csv_path,ast_pkl_path,"ast.pkl")
    origin_start_time = time.time()
    Ast2Vec("C:/Users/fwk/Videos/MyAST/train/ast.pkl", "C:/Users/fwk/Videos/MyAST/train/Experiment4", "data_origin.pkl")
    GRU("C:/Users/fwk/Videos/MyAST/train/Experiment4/embedding_Experiment4/256",
        "C:/Users/fwk/Videos/MyAST/train/Experiment4/data_origin.pkl")
    origin_end_time = time.time()
    print("origin ASTNN in SARD Dataset time cost:%.4f s" % (origin_end_time - origin_start_time))
    improved_start_time = time.time()
    Ast2VecImproved("C:/Users/fwk/Videos/MyAST/train/Experiment4/ast.pkl", "C:/Users/fwk/Videos/MyAST/train/Experiment4", "data.pkl")
    # 训练模型
    forest("C:/Users/fwk/Videos/MyAST/train/Experiment4/embedding_Experiment4/256improved",
           "C:/Users/fwk/Videos/MyAST/train/Experiment4/data.pkl")
    improved_end_time = time.time()
    print("improved ASTNN in SARD Dataset time cost:%.4f s" % (improved_end_time - improved_start_time))
if __name__ == '__main__':
    main()
