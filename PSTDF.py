import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
class DTForest(object):
    def __init__(self, embedding_dim, vocab_size,encode_dim,label_size,
                 n_jobs=-1, n_mgsRFtree=200, pretrained_weight=None, min_samples_mgs=0.1, batch_size=1):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.label_size = label_size
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim, pretrained_weight)
        self.n_mgsRFtree=n_mgsRFtree
        self.min_samples_mgs = min_samples_mgs
        self.n_jobs = n_jobs
    def mg_scan(self,data):
        X_mgs = []
        y_mgs = []
        def get_batch(dataset, idx, bs):
            tmp = dataset.iloc[idx: idx + bs]
            data, labels = [], []
            for _, item in tmp.iterrows():
                data.append(item[1])
                labels.append(item[2])
            return data, torch.LongTensor(labels)
        index = 0
        while index < len(data):
            batch = get_batch(data, index, self.batch_size)
            index += self.batch_size
            x, y = batch
            # Calculate the length of each sentence
            lens = [len(item) for item in x]
            # find the longest sequence of sentences
            max_len = max(lens)
            encodes = []
            # Process the sequence of statements one by one
            for i in range(self.batch_size):
                # Process word vector by word
                for j in range(lens[i]):
                    encodes.append(x[i][j])
            encodes = self.encoder(encodes, sum(lens))
            seq, start, end = [], 0, 0
            for i in range(self.batch_size):
                end += lens[i]
                if max_len - lens[i]:
                    seq.append(self.get_zeros(max_len - lens[i]))
                seq.append(encodes[start:end])
                start = end
            encodes = torch.cat(seq)
            encodes = encodes.view(self.batch_size, max_len, -1)
            encodes = torch.transpose(encodes, 1, 2)
            vec = F.max_pool1d(encodes, encodes.size(2)).squeeze(2)
            # vec = self.spatial_pyramid_pool(encodes)
            X_mgs.append(vec[0].detach().numpy())
            y_mgs.extend(y.detach().numpy().tolist())
        return X_mgs, y_mgs

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        return zeros

class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, pretrained_weight=None, batch_size=1):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.W_l = nn.Linear(encode_dim, encode_dim)
        self.W_r = nn.Linear(encode_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.node_list = []
        self.th = torch
        self.batch_node = None
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def create_tensor(self, tensor):
        return tensor

    def traverse_mul(self, node, batch_index):
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
