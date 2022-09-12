import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from sklearn.metrics import roc_auc_score, accuracy_score

from src.evaluate import get_all_metrics
from src.load_base import get_records, load_kg


class MLP(nn.Module):

    def __init__(self, int_dim, hidden_dim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(int_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim,1)

    def forward(self, x):
        x1 = t.relu(self.l1(x))
        x2 = t.sigmoid(self.l2(x1))

        return x2


class MetaPathAttention(nn.Module):

    def __init__(self, u_dim, p_dim, i_dim, hidden_dim):
        super(MetaPathAttention, self).__init__()
        self.W_u = nn.Linear(u_dim, hidden_dim)
        self.W_p = nn.Linear(p_dim, hidden_dim, bias=True)
        self.W_i = nn.Linear(i_dim, hidden_dim,bias=True)
        self.W = nn.Linear(hidden_dim, 1)

    def forward(self, x_u, y_i, c_p):
        # print(x_u.shape, y_i.shape, c_p.shape)
        a1 = t.relu(self.W_u(x_u) + self.W_i(y_i) + self.W_p(c_p))
        a2 = t.relu(self.W(a1))
        a2 = t.softmax(a2, dim=1)
        return a2


class UserItemAttention(nn.Module):

    def __init__(self, x_dim, c_dim):
        super(UserItemAttention, self).__init__()

        self.W_x = nn.Linear(x_dim, x_dim)
        self.W_c = nn.Linear(c_dim, x_dim, bias=True)

    def forward(self, x, c):
        bieta = t.relu(self.W_x(x) + self.W_c(c))
        return bieta


class MCRec(nn.Module):

    def __init__(self, n_entities, embedding_dim, predict_factor_dim, p):
        super(MCRec, self).__init__()

        self.p = p
        self.embedding_dim = embedding_dim
        self.predict_factor_dim = predict_factor_dim
        self.mlp = MLP(embedding_dim*3, predict_factor_dim)
        self.max_pool = nn.MaxPool2d(kernel_size=(p, 1))
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        self.path_attention = MetaPathAttention(embedding_dim, embedding_dim,
                                                embedding_dim, embedding_dim)
        self.user_attention = UserItemAttention(embedding_dim, embedding_dim)
        self.item_attention = UserItemAttention(embedding_dim, embedding_dim)
        self.embedding_matrix = nn.Parameter(t.randn(n_entities, embedding_dim))

    def forward(self, users, items, paths_list):

        user_embeddings = self.embedding_matrix[users]
        item_embeddings = self.embedding_matrix[items]
        meth_path_embeddings_list = []

        for meth_paths in paths_list:
            meth_path_embeddings_list.append(self.get_meta_path_embedding(meth_paths))

        # (batch_size, -1, dim)
        meth_path_embeddings = t.cat(meth_path_embeddings_list, dim=1)
        # print(meth_path_embeddings.shape)
        # (batch_size, -1, 1)
        meth_path_attention = self.path_attention(user_embeddings.reshape(-1, 1, self.embedding_dim),
                                                  item_embeddings.reshape(-1, 1, self.embedding_dim),
                                                  meth_path_embeddings)
        # (batch_size, dim)
        attention_meth_path_embeddings = (meth_path_attention * meth_path_embeddings).sum(dim=1)

        user_path_embeddings = self.user_attention(user_embeddings, attention_meth_path_embeddings) * user_embeddings
        item_path_embeddings = self.item_attention(item_embeddings, attention_meth_path_embeddings) * item_embeddings
        user_path_item_embeddings = t.cat([user_path_embeddings, attention_meth_path_embeddings, item_path_embeddings],
                                          dim=-1)
        return self.mlp(user_path_item_embeddings).reshape(-1)

    def get_meta_path_embedding(self, meta_paths):
        path_embedding_list = []
        zeros = t.zeros(4, self.embedding_dim)

        if t.cuda.is_available():
            zeros = zeros.to(self.embedding_matrix.data.device)

        for paths in meta_paths:
            if len(paths) == 0:
                path_embedding_list.extend([zeros] * self.p)
                continue
            for path in paths:
                # (len, dim)
                path_embedding_list.append(self.embedding_matrix[path])

        # (batch_size * p, len, dim)
        path_embeddings = t.cat(path_embedding_list, dim=0).reshape(-1, 4, self.embedding_dim)
        # print(path_embeddings.shape)

        # (batch_size*p, 1, dim) -> (batch_size, p, dim)
        path_instance_embeddings = self.conv1d(path_embeddings).reshape(-1, self.p, self.embedding_dim)

        # (batch_size, 1, dim)
        meta_path_embeddings = self.max_pool(path_instance_embeddings)

        return meta_path_embeddings


def get_scores(model, rec, paths_dict, p):
    model.eval()
    scores = dict()
    user_list = list(rec.keys())
    for user in rec:
        pairs = [[user, item, -1] for item in rec[user]]
        users, items, paths_list, _ = get_data(pairs, paths_dict, user_list, p)

        predict_np = model(users, items, paths_list).cpu().detach().numpy()

        i = 0
        item_scores = dict()

        for item in rec[user]:

            item_scores[item] = predict_np[i]
            i += 1

        sorted_item_scores = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        scores[user] = [i[0] for i in sorted_item_scores]
    model.train()
    return scores


def eval_ctr(model, pairs, paths_dict, p, user_list, batch_size):

    model.eval()
    pred_label = []
    users, items, paths, label_list = get_data(pairs, paths_dict, user_list, p)
    for i in range(0, len(pairs), batch_size):
        batch_label = model(users[i: i + batch_size],
                            items[i: i + batch_size],
                            [paths[0][i: i + batch_size],
                             paths[1][i: i + batch_size]]
                            ).cpu().detach().numpy().tolist()
        pred_label.extend(batch_label)
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np  = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    acc = accuracy_score(true_label, pred_label)
    return round(auc, 3), round(acc, 3)


def get_data(pairs, paths_dict, user_list, p):
    paths_list = [[], []]
    label_list = []
    users = []
    items = []
    for pair in pairs:
        uvuv_paths, uvav_paths = [], []
        if len(paths_dict[(pair[0], pair[1])]) != 0:
            paths = paths_dict[(pair[0], pair[1])]

            for path in paths:
                if path[2] in user_list:
                    uvuv_paths.append(path)
                else:
                    uvav_paths.append(path)

            if len(uvav_paths) == 0:
                uvav_paths = uvuv_paths.copy()
            if len(uvuv_paths) == 0:
                uvuv_paths = uvav_paths.copy()

            # print(len(paths_dict[(pair[0], pair[1])]), len(uvuv_paths))
            if len(uvuv_paths) >= p:
                indices = np.random.choice(len(uvuv_paths), p, replace=False)
                uvuv_paths = [uvuv_paths[i] for i in indices]
            else:
                indices = np.random.choice(len(uvuv_paths), p, replace=True)
                uvuv_paths = [uvuv_paths[i] for i in indices]

            if len(uvav_paths) >= p:
                indices = np.random.choice(len(uvav_paths), p, replace=False)
                uvav_paths = [uvav_paths[i] for i in indices]
            else:
                indices = np.random.choice(len(uvav_paths), p, replace=True)
                uvav_paths = [uvav_paths[i] for i in indices]

        paths_list[0].append(uvuv_paths)
        paths_list[1].append(uvav_paths)
        users.append(pair[0])
        items.append(pair[1])
        label_list.append(pair[2])
    return users, items, paths_list, label_list


def train(args, is_topk=False):
    np.random.seed(555)
    data_dir = './data/' + args.dataset + '/'
    train_set = np.load(data_dir + str(args.ratio) + '_train_set.npy').tolist()
    eval_set = np.load(data_dir + str(args.ratio) + '_eval_set.npy').tolist()
    test_set = np.load(data_dir + str(args.ratio) + '_test_set.npy').tolist()
    test_records = get_records(test_set)
    entity_list = np.load(data_dir + '_entity_list.npy').tolist()
    _, _, n_relation = load_kg(data_dir)
    n_entity = len(entity_list)
    rec = np.load(data_dir + str(args.ratio) + '_rec.npy', allow_pickle=True).item()
    paths_dict = np.load(data_dir + str(args.ratio) + '_3_path_dict.npy', allow_pickle=True).item()
    user_list = list(get_records(train_set).keys())
    model = MCRec(n_entity, args.embedding_dim, args.predict_factor_dim, args.p)

    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.BCELoss()

    print(args.dataset + '-----------------------------------------')
    print('embedding_dim: %d' % args.embedding_dim, end=', ')
    print('predict_factor_dim: %d' % args.predict_factor_dim, end=', ')
    print('p: %d' % args.p, end=', ')
    print('lr: %1.0e' % args.lr, end=', ')
    print('l2: %1.0e' % args.l2, end=', ')
    print('batch_size: %d' % args.batch_size)

    train_auc_list = []
    train_acc_list = []
    eval_auc_list = []
    eval_acc_list = []
    test_auc_list = []
    test_acc_list = []
    all_precision_list = []

    for epoch in range(args.epochs):
        loss_sum = 0
        start = time.clock()
        np.random.shuffle(train_set)
        users, items, paths, true_label = get_data(train_set, paths_dict, user_list, args.p)
        labels = t.tensor(true_label).float()
        if t.cuda.is_available():
            labels = labels.to(args.device)
        start_index = 0
        size = len(users)
        model.train()
        while start_index < size:
            predicts = model(users[start_index: start_index + args.batch_size],
                             items[start_index: start_index + args.batch_size],
                             [paths[0][start_index: start_index + args.batch_size],
                              paths[1][start_index: start_index + args.batch_size]])
            loss = criterion(predicts, labels[start_index: start_index + args.batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.cpu().item()

            start_index += args.batch_size

        train_auc, train_acc = eval_ctr(model, train_set, paths_dict, args.p, user_list, args.batch_size)
        eval_auc, eval_acc = eval_ctr(model, eval_set, paths_dict, args.p, user_list, args.batch_size)
        test_auc, test_acc = eval_ctr(model, test_set, paths_dict, args.p, user_list, args.batch_size)

        print('epoch: %d \t train_auc: %.3f \t train_acc: %.3f \t '
              'eval_auc: %.3f \t eval_acc: %.3f \t test_auc: %.3f \t test_acc: %.3f \t' %
              ((epoch + 1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc), end='\t')

        precision_list = []
        if is_topk:
            scores = get_scores(model, rec, paths_dict, args.p)
            precision_list = get_all_metrics(scores, test_records)[0]
            print(precision_list, end='\t')

        train_auc_list.append(train_auc)
        train_acc_list.append(train_acc)
        eval_auc_list.append(eval_auc)
        eval_acc_list.append(eval_acc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        all_precision_list.append(precision_list)
        end = time.clock()
        print('time: %d' % (end - start))

    indices = eval_auc_list.index(max(eval_auc_list))
    print(args.dataset, end='\t')
    print('train_auc: %.3f \t train_acc: %.3f \t eval_auc: %.3f \t eval_acc: %.3f \t '
          'test_auc: %.3f \t test_acc: %.3f \t' %
          (train_auc_list[indices], train_acc_list[indices], eval_auc_list[indices], eval_acc_list[indices],
           test_auc_list[indices], test_acc_list[indices]), end='\t')

    print(all_precision_list[indices])

    return eval_auc_list[indices], eval_acc_list[indices], test_auc_list[indices], test_acc_list[indices]