import time

import numpy as np
import pandas as pd
import os

import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score
from deepforest import CascadeForestClassifier
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from PSTDF import DTForest
from tree import SingleNode, ASTNode

def get_sequences(node, sequence):
    current = SingleNode(node)
    sequence.append(current.get_token())
    for _, child in node.children():
        get_sequences(child, sequence)
    if current.get_token().lower() == 'compound':
        sequence.append('End')

def get_blocks(node, block_seq):
    children = node.children()
    name = node.__class__.__name__
    if name in ['FuncDef', 'If', 'For', 'While', 'DoWhile']:
        block_seq.append(ASTNode(node))
        if name is not 'For':
            skip = 1
        else:
            skip = len(children) - 1

        for i in range(skip, len(children)):
            child = children[i][1]
            if child.__class__.__name__ not in ['FuncDef', 'If', 'For', 'While', 'DoWhile', 'Compound']:
                block_seq.append(ASTNode(child))
            get_blocks(child, block_seq)
    elif name is 'Compound':
        block_seq.append(ASTNode(name))
        for _, child in node.children():
            if child.__class__.__name__ not in ['If', 'For', 'While', 'DoWhile']:
                block_seq.append(ASTNode(child))
            get_blocks(child, block_seq)
        block_seq.append(ASTNode('End'))
    else:
        for _, child in node.children():
            get_blocks(child, block_seq)


class CPSTDF:

    def __init__(self,  ratio, root,train_file_path, embedding_size):
        self.ratio = ratio
        self.root = root
        self.sources = None
        self.train_file_path = train_file_path
        self.test_file_path = None
        self.size = embedding_size

    # parse source code
    def parse_source(self, output_file, option):
        path = self.root+output_file
        if os.path.exists(path) and option is 'existing':
            source = pd.read_pickle(path)
        else:
            from pycparser import c_parser
            parser = c_parser.CParser()
            source = pd.read_pickle(self.root+'programs.pkl')

            source.columns = ['id', 'code', 'label']
            source['code'] = source['code'].apply(parser.parse)

            source.to_pickle(path)
        self.sources = source
        return source

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        if not input_file:
            input_file = self.train_file_path
        trees = pd.read_pickle(input_file)
        if not os.path.exists(self.root+'train/embedding'):
            os.mkdir(self.root+'train/embedding')

        def trans_to_sequences(ast):
            sequence = []
            get_sequences(ast, sequence)
            return sequence
        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        # print(str_corpus)
        trees['code'] = pd.Series(str_corpus)
        trees.to_csv(self.root+'train/programs_ns.tsv')


        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3)
        # print(w2v)
        w2v.save(self.root+'train/embedding/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self, data_path, part):
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root+'train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            get_blocks(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree
        trees = pd.read_pickle(data_path)
        trees['code'] = trees['code'].apply(trans2seq)
        trees.to_pickle(self.root+part+'/blocks.pkl')


    def forest(self,Word2VecPath,dataPath):
        def get_batch(dataset, idx, bs):
            tmp = dataset.iloc[idx: idx + bs]
            data, labels = [], []
            for _, item in tmp.iterrows():
                data.append(item[1])
                labels.append(item[2]-1)
            return data, torch.LongTensor(labels)

        data = pd.read_pickle(dataPath)
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0] / sum(ratios) * data_num)
        data = data.sample(frac=1, random_state=666)
        train_data = data.iloc[:train_split]
        word2vec = Word2Vec.load(Word2VecPath).wv
        embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
        embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
        HIDDEN_DIM = 100
        ENCODE_DIM = self.size
        # 标签种类数
        LABELS = 104
        BATCH_SIZE = 64
        USE_GPU = False
        MAX_TOKENS = word2vec.syn0.shape[0]
        EMBEDDING_DIM = word2vec.syn0.shape[1]
        EPOCHS = 3
        model = DTForest(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE, USE_GPU, embeddings)
        parameters = model.parameters()
        optimizer = torch.optim.Adamax(parameters)
        loss_function = torch.nn.CrossEntropyLoss()
        best_model = model
        matrix = []
        labels = []
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
        i = 0
        while i < len(train_data):
            batch = get_batch(train_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            train_inputs, train_labels = batch
            best_model.zero_grad()
            best_model.batch_size = len(train_labels)
            best_model.hidden = best_model.init_hidden()
            output = best_model(train_inputs)
            for j in range(0, len(output)):
                matrix.append(output.detach().numpy().tolist()[j])
                labels.append(train_labels.detach().numpy().tolist()[j])
        X_train, X_test, y_train, y_test = train_test_split(matrix, labels, random_state=1, test_size=0.2)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        gc = CascadeForestClassifier(random_state=1, n_jobs=-1)
        end_time = time.time()
        gc.fit(X_train, y_train)
        y_pred = gc.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        rcc = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        print("\nTesting Accuracy: {:.3f} %".format(acc))
        print("\nTesting Recall: {:.4f}".format(rcc))
        print("\nTesting f1 score: {:.4f}".format(f1))
        print("耗时:%.3f s" % (end_time - start_time))
    # run for processing data to train
    def run(self):
        # print('parse source code...')
        # self.parse_source(output_file='ast.pkl', option='existing')
        # print('train word embedding...')
        # self.dictionary_and_embedding(None, 128)
        # print('generate block sequences...')
        self.generate_block_seqs(self.train_file_path, 'train')
        self.forest("C:/Users/fwk/Downloads/astnn-master/data/c/train/embedding/node_w2v_128", "C:/Users/fwk/Downloads/astnn-master/data/c/train/blocks.pkl")


ppl = CPSTDF(ratio='8:2', embedding_size=128, root='C:/Users/fwk/Downloads/astnn-master/data/c/', train_file_path = 'C:/Users/fwk/Downloads/astnn-master/data/c/ast.pkl')
ppl.run()


