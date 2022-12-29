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
# Convert all source code files in the specified folder to csv
def code2tsv(input_file, output_file):
    # Vulnerability label list
    vul2dic = {"none": 0}
    maxLabel = vul2dic["none"]
    # Set the output directory and output file name
    outputf = open(output_file + "/code.tsv", "w", encoding="utf8", newline='')
    # Read the list of files under the folder
    fileList = os.listdir(input_file)
    rows = []
    for i in range(0, len(fileList)):
        if fileList[i].split(".")[1] == "xml":
            continue
        else:
            # read the java source code
            inputf = open(input_file + "/" + fileList[i], "r")
            # Read the corresponding xml description file, extract the vulnerability type and label
            dom = xml.dom.minidom.parse(input_file + "/" + fileList[i].split(".")[0] + ".xml")
            root = dom.documentElement
            # Extract flags for vulnerabilities
            vulnerability = root.getElementsByTagName('vulnerability')
            # If a vulnerability exists, extract the specific vulnerability type
            label = vul2dic['none']
            if vulnerability[0].firstChild.data == "true":
                category = root.getElementsByTagName('category')[0].firstChild.data
                # Traverse the table corresponding to the vulnerability label, if there is no such type of vulnerability, add it, if there is, assign a value to the label
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
                # Non-vulnerable code, tagged as non-vulnerable
                label = vul2dic['none']
            s = inputf.read()
            s = s.replace("	", "    ")
            s = s.replace("\r\n", "\n")
            row = (i, s, label)
            rows.append(row)
            inputf.close()
    # Specify tab-delimited
    writer = csv.writer(outputf, delimiter='\t')
    # write CSV
    writer.writerows(rows)
    outputf.close()
    print(vul2dic)
    return None
# Convert source code to AST
def tsv2Ast(input_path, output_path):
    def parse_program(func):
        # Transforms the source code compilation unit into an abstract syntax tree
        tree = javalang.parse.parse(func)
        # tokens = javalang.tokenizer.tokenize(func)
        # parser = javalang.parser.Parser(tokens)
        # tree = parser.parse_member_declaration()
        return tree

    source = pd.read_csv(input_path, delimiter="\t", header=None, encoding='utf-8')
    source.columns = ['id', 'code', 'label']
    source['code'] = source['code'].apply(parse_program)
    # The abstract syntax tree is persistently stored in the pkl file
    source.to_pickle(output_path + "/ast.pkl")
# Convert the abstract syntax tree into a vector by improved encoding layer
def Ast2VecImproved(input_path, output_path, filename):
    trees = pd.read_pickle(input_path)
    # AST is converted into a sequence of statements
    def Ast_to_sequences(ast):
        sequence = []
        Tools.get_sequence_BFS(ast, sequence)
        return sequence
    # Extract sequence of sentences
    word_sequences = trees['code'].apply(Ast_to_sequences)
    str_sequences = [' '.join(c) for c in word_sequences]
    trees['code'] = pd.Series(str_sequences)
    # Train a word2vec model with a sequence of sentences
    model = Word2Vec(word_sequences, size=128, workers=16, sg=1, min_count=3)
    if not os.path.exists(output_path + '/embedding'):
        os.mkdir(output_path + '/embedding')
    # word2vec model persistence
    model.save(output_path + '/embedding/' + str(128)+'improved')
    # Get all trained words
    vocab = model.wv.vocab
    max_token = model.wv.syn0.shape[0]
    def tree_to_index(node):
        token = node.token
        # Expression subtree pruning, discarding useless nodes such as package references
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
    # Vector persistence to file
    trees.to_pickle(output_path + "/" + filename)

def forest(Word2VecPath, dataPath):
    acca = []
    rcca = []
    f1a = []
    for num in range(0, 1):  #This loop is used to calculate the average performance. If you only run the model, set the number of loop to 1.
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
        # word embedding dimension
        ENCODE_DIM = 128
        # Number of label types
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
    # Java source code reading path
    input_file_path = "./datasets/OWASP/code"
    # tsv output path
    output_file_path = "./tsv/"
    # Source code tsv reading path
    code_tsv_path = output_file_path + "code.tsv"
    # training, testing path
    train_path = "./train"
    # The path to the pkl file where the AST is stored
    ast_path = train_path + "/ast.pkl"
    # Process the source code and convert it into tsv
    code2tsv(input_file_path, output_file_path)
    # Process the source code in tsv format, convert it into an abstract syntax tree and persist it in the pkl file
    tsv2Ast(code_tsv_path, train_path)
    # training and testing
    improved_start_time = time.time()
    Ast2VecImproved(ast_path, train_path, "data_improved.pkl")
    forest(train_path+"/embedding/128improved", train_path+"/data_improved.pkl")
    improved_end_time = time.time()
    print("improved ASTNN time cost:%.4f s" % (improved_end_time - improved_start_time))
if __name__ == '__main__':
    main()
