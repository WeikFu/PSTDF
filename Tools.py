import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from javalang.ast import Node
from tree import BlockNode
import sys
import numpy as np
sys.setrecursionlimit(10000)

def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token


def get_children(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))
# 原始ASTNN
def get_sequence(node, sequence):
    # 获得节点的token及孩子节点
    token, children = get_token(node), get_children(node)
    sequence.append(token)
    # 递归获取token
    for child in children:
        get_sequence(child, sequence)
    # 基线条件
    if token in ['ForStatement', 'WhileStatement', 'DoStatement', 'SwitchStatement', 'IfStatement']:
        sequence.append('End')

def get_sequence_BFS(root,sequence):
    queue = []
    queue.append(root)
    while queue:
        for i in range(len(queue)):
            node = queue.pop(0)
            token, children = get_token(node), get_children(node)
            if token not in ['Import', 'PackageDeclaration']:  # 表达式子树裁剪，丢弃包引用和注释等无用节点
                sequence.append(token)
            # 基线条件
            if token in ['ForStatement', 'WhileStatement', 'DoStatement', 'SwitchStatement', 'IfStatement']:
                sequence.append('End')
            for j in children:
                queue.append(j)
    return sequence

# 改进ASTNN
def get_sequence_improved(node, sequence):
    # 获得节点的token及孩子节点
    token, children = get_token(node), get_children(node)
    if token not in ['Import', 'PackageDeclaration']:  # 表达式子树裁剪，丢弃包引用和注释等无用节点
        sequence.append(token)
        # 递归获取token
        for child in children:
            get_sequence_improved(child, sequence)
        # 基线条件
        if token in ['ForStatement', 'WhileStatement', 'DoStatement', 'SwitchStatement', 'IfStatement']:
            sequence.append('End')
def bfs(root):
        res = []
        queue = []
        queue.append(root)
        while queue:
            level = []
            for i in range(len(queue)):
                node = queue.pop(0)
                level.append(node.val)
                for j in node.children:
                    queue.append(j)
            res.append(level)
        return res
def get_blocks_improved_BFS(root, block_seq):
    queue = []
    queue.append(root)
    while queue:
        for i in range(len(queue)):
            node = queue.pop(0)
            name, children = get_token(node), get_children(node)
            useless = ['Import', 'PackageDeclaration']
            if name not in useless:
                if children != None:
                    logic = ['SwitchStatement', 'IfStatement', 'ForStatement', 'WhileStatement', 'DoStatement']
                    if name in ['MethodDeclaration', 'ConstructorDeclaration']:
                        block_seq.append(BlockNode(node))
                        body = node.body
                        for child in body:
                            if get_token(child) not in logic and not hasattr(child, 'block'):
                                block_seq.append(BlockNode(child))
                            else:
                                queue.append(child)
                    elif name in logic:
                        block_seq.append(BlockNode(node))
                        for child in children[1:]:
                            token = get_token(child)
                            if not hasattr(node, 'block') and token not in logic + ['BlockStatement']:
                                block_seq.append(BlockNode(child))
                            else:
                                queue.append(child)
                            block_seq.append(BlockNode('End'))
                    elif name is 'BlockStatement' or hasattr(node, 'block'):
                        block_seq.append(BlockNode(name))
                        for child in children:
                            if get_token(child) not in logic:
                                block_seq.append(BlockNode(child))
                            else:
                                queue.append(child)
                    else:
                        for child in children:
                            queue.append(child)

def get_blocks_improved_DFS(node, block_seq):
    name, children = get_token(node), get_children(node)
    useless = ['Import', 'PackageDeclaration']
    if name not in useless:
        if children == None:
            return
        logic = ['SwitchStatement', 'IfStatement', 'ForStatement', 'WhileStatement', 'DoStatement']
        if name in ['MethodDeclaration', 'ConstructorDeclaration']:
            block_seq.append(BlockNode(node))
            body = node.body
            for child in body:
                if get_token(child) not in logic and not hasattr(child, 'block'):
                    block_seq.append(BlockNode(child))
                else:
                    get_blocks_improved_DFS(child, block_seq)
        elif name in logic:
            block_seq.append(BlockNode(node))
            for child in children[1:]:
                token = get_token(child)
                if not hasattr(node, 'block') and token not in logic + ['BlockStatement']:
                    block_seq.append(BlockNode(child))
                else:
                    get_blocks_improved_DFS(child, block_seq)
                block_seq.append(BlockNode('End'))
        elif name is 'BlockStatement' or hasattr(node, 'block'):
            block_seq.append(BlockNode(name))
            for child in children:
                if get_token(child) not in logic:
                    block_seq.append(BlockNode(child))
                else:
                    get_blocks_improved_DFS(child, block_seq)
        else:
            for child in children:
                get_blocks_improved_DFS(child, block_seq)
# 原始ASTNN
def get_blocks(node, block_seq):
    name, children = get_token(node), get_children(node)
    logic = ['SwitchStatement', 'IfStatement', 'ForStatement', 'WhileStatement', 'DoStatement']
    if name in ['MethodDeclaration', 'ConstructorDeclaration']:
        block_seq.append(BlockNode(node))
        body = node.body
        for child in body:
            if get_token(child) not in logic and not hasattr(child, 'block'):
                block_seq.append(BlockNode(child))
            else:
                get_blocks(child, block_seq)
    elif name in logic:
        block_seq.append(BlockNode(node))
        for child in children[1:]:
            token = get_token(child)
            if not hasattr(node, 'block') and token not in logic+['BlockStatement']:
                block_seq.append(BlockNode(child))
            else:
                get_blocks(child, block_seq)
            block_seq.append(BlockNode('End'))
    elif name is 'BlockStatement' or hasattr(node, 'block'):
        block_seq.append(BlockNode(name))
        for child in children:
            if get_token(child)not in logic:
                block_seq.append(BlockNode(child))
            else:
                get_blocks(child, block_seq)
    else:
        block_seq.append(BlockNode(name))
        for child in children:
            get_blocks(child, block_seq)
class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.W_l = nn.Linear(encode_dim, encode_dim)
        self.W_r = nn.Linear(encode_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        # print('batch_index', len(batch_index))
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            if node[i][0] is not -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] is not -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                batch_index[i] = -1

        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node)))))

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)
        batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]

class GRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
        super(GRU, self).__init__()
        self.stop = [vocab_size-1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.label_size = label_size
        #class "BatchTreeEncoder"
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, pretrained_weight)
        self.root2label = nn.Linear(self.encode_dim, self.label_size)
        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.label_size)
        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)

    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def forward(self, x):
        # 计算每个语句序列的长度
        lens = [len(item) for item in x]
        # 找到最长的语句序列
        max_len = max(lens)
        all = np.sum(lens)
        # print('all',all)
        encodes = []
        # 逐个语句序列进行处理
        for i in range(self.batch_size):
            # 逐个词编码进行处理
            for j in range(lens[i]):
                encodes.append(x[i][j])
        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)
        # gru
        gru_out, hidden = self.bigru(encodes, self.hidden)
        # print('lengruout', len(gru_out[0]))
        gru_out = torch.transpose(gru_out, 1, 2)
        # print('lengruout', len(gru_out[0]))
        # pooling

        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        # gru_out = gru_out[:,-1]

        # linear
        y = self.hidden2label(gru_out)
        # print('leny',len(y))
        return y
