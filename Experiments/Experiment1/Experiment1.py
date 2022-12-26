import os
import time
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from gensim.models.word2vec import Word2Vec
import Tools
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from Tools import GRU as gru
import torch
from torch.autograd import Variable

#实验目的：验证提出的I_ASTNN方法的有效性


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
    model = Word2Vec(word_sequences, size=128, workers=16, sg=1,min_count=3)
    if not os.path.exists(output_path + '/embedding_Experiment2'):
        os.mkdir(output_path + '/embedding_Experiment2')
    # word2vec模型持久化
    model.save(output_path + '/embedding_Experiment2/' +str(256))
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
# 将抽象语法树通过改进ASTNN编码层转换成向量
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
    model = Word2Vec(word_sequences, size=128, workers=16, sg=1, min_count=3)
    if not os.path.exists(output_path + '/embedding_Experiment2'):
        os.mkdir(output_path + '/embedding_Experiment2')
    # word2vec模型持久化
    model.save(output_path + '/embedding_Experiment2/' + str(256)+'improved')
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
    def trans2seq(r):
        blocks_bfs = []
        Tools.get_blocks_improved_BFS(r, blocks_bfs)
        tree = []
        # 语句序列拼接
        for i in range(0, len(blocks_bfs)):
            btree_bfs = tree_to_index(blocks_bfs[i])
            tree.append(btree_bfs)
        return tree

    trees = pd.read_pickle(input_path)
    trees['code'] = trees['code'].apply(trans2seq)
    # 向量持久化到文件
    trees.to_pickle(output_path + "/" + filename)


def GRU(Word2VecPath, dataPath):
    timecosta = []
    acca = []
    recalla = []
    f1a = []
    for i in range(0,1):
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
        EPOCHS = 3
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
        acca.append(precision)
        recalla.append(recall)
        f1a.append(f1)
        # vuldict = {0 :'none', 1:'xss', 2:'weakrand', 3:'hash', 4:'crypto', 5:'securecookie', 6:'pathtraver', 7:'ldapi',8:'cmdi', 9:'trustbound', 10:'sqli', 11:'xpathi'}
        # truess = []
        # predictss = []
        # for i in range(0,len(trues)):
        #     truess.append(vuldict[trues[i]])
        # for i in range(0,len(predicts)):
        #     predictss.append(vuldict[int(predicts[i])])
        #
        # C = confusion_matrix(trues, predicts)  # 可将'1'等替换成自己的类别，如'cat'。
        #
        # plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
        # plt.colorbar()
        # vuldicts = ['none', 'xss','weakrand','hash', 'crypto', 'securecookie', 'pathtraver',
        #            'ldapi','cmdi', 'trustbound', 'sqli','xpathi']
        # # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
        # indices = range(len(C))
        # plt.xticks(indices, vuldicts, rotation=45)  # 设置横坐标方向，rotation=45为45度倾斜
        # plt.yticks(indices, vuldicts)
        #
        # for i in range(len(C)):
        #     for j in range(len(C)):
        #         plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
        #
        # # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
        #
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        #
        # plt.show()
    print('average acc: {:.4f} %'.format(sum(acca) / len(acca)))
    print('average recall: {:.4f}'.format(sum(recalla) / len(recalla)))
    print('average f1: {:.4f}'.format(sum(f1a) / len(f1a)))


def main():
    # origin_start_time = time.time()
    # Ast2Vec("C:/Users/fwk/Videos/MyAST/train/ast.pkl", "C:/Users/fwk/Videos/MyAST/train/Experiment2", "data.pkl")
    # GRU("C:/Users/fwk/Videos/MyAST/train/Experiment2/embedding_Experiment2/256", "C:/Users/fwk/Videos/MyAST/train/Experiment2/data.pkl")
    # origin_end_time = time.time()
    # print("origin ASTNN time cost:%.3f s" % (origin_end_time - origin_start_time))
    improved_start_time = time.time()
    # Ast2VecImproved("C:/Users/fwk/Videos/MyAST/train/ast.pkl", "C:/Users/fwk/Videos/MyAST/train/Experiment2", "data_improved.pkl")
    GRU("C:/Users/fwk/Videos/MyAST/train/Experiment2/embedding_Experiment2/256improved", "C:/Users/fwk/Videos/MyAST/train/Experiment2/data_improved.pkl")
    improved_end_time = time.time()
    print("I_ASTNN time cost:%.3f s" % (improved_end_time - improved_start_time))

if __name__ == '__main__':
    main()
