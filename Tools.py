import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from javalang.ast import Node
from tree import BlockNode

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

def get_sequence_BFS(root,sequence):
    queue = []
    queue.append(root)
    while queue:
        for i in range(len(queue)):
            node = queue.pop(0)
            token, children = get_token(node), get_children(node)
            if token not in ['Import', 'PackageDeclaration']:  # Statement subtree pruning, discarding useless nodes such as package references and comments
                sequence.append(token)
            # base case
            if token in ['ForStatement', 'WhileStatement', 'DoStatement', 'SwitchStatement', 'IfStatement']:
                sequence.append('End')
            for j in children:
                queue.append(j)
    return sequence

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

