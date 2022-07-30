"""
training size: 120000
testing size: 7600
batch num: 1875
"""
import time
import torch
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torch.autograd import grad
import numpy as np

train_list = torch.load('./checkpoints_4000/train_list.pt')
valid_list = torch.load('./checkpoints_4000/valid_list.pt')
test_list  = torch.load('./checkpoints_4000/test_list.pt')

valid_accu_log = np.load('valid_accu_log.npy', allow_pickle=True).item()
class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


def get_gradient(grads, model):
    """
    pick the gradients by name.
    """
    return [grad for grad, (n, p) in zip(grads, model.named_parameters())]

def tracin_get(a, b):
    """ dot product between two lists"""
    return sum([torch.dot(at.flatten(), bt.flatten()) for at, bt in zip(a, b)])


criterion = torch.nn.CrossEntropyLoss()


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = train_size/4
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        # if idx == TRAIN_SIZE:
        #     break
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        # if idx % log_interval == 0 and idx > 0:
        #     elapsed = time.time() - start_time
        #     print('| epoch {:3d} | {:5d}/{:5d} batches '
        #           '| accuracy {:8.3f}'.format(epoch, idx, 4000,
        #                                       total_acc/total_count))
        #     total_acc, total_count = 0, 0
        #     start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


def SV_1(class0, class1, class2, class3):
    valid_index_list = list()
    _class0, _class1, _class2, _class3 = 0, 0, 0, 0
    for idx in valid_list:
        (label_example_valid, text_example_valid, offset_example_valid) = valid_list[idx]
        if label_example_valid == 0 and _class0 < class0:
            valid_index_list.append(idx)
            _class0 += 1
        elif label_example_valid == 1 and _class1 < class1:
            valid_index_list.append(idx)
            _class1 += 1
        elif label_example_valid == 2 and _class2 < class2:
            valid_index_list.append(idx)
            _class2 += 1
        elif label_example_valid == 3 and _class3 < class3:
            valid_index_list.append(idx)
            _class3 += 1
        elif _class0 == class0 and _class1 == class1 and _class2 == class2 and _class3 == class3:
            break
        else:
            continue
    print('SV_1: class0: {}, class1: {}, class2: {}, class3: {}'.format(_class0, _class1, _class2, _class3))
    return valid_index_list

def SV_2(difficulty, size):
    valid_index_list = list()
    easy = []
    median_easy = []
    median_hard = []
    hard = []
    for idx in valid_accu_log:
        if sum(valid_accu_log[idx][:3]) == 0:
            hard.append(idx)
        elif sum(valid_accu_log[idx][:3]) == 1:
            median_hard.append(idx)
        elif sum(valid_accu_log[idx][:3]) == 2:
            median_easy.append(idx)
        else:
            easy.append(idx)
    if difficulty == 0:
        valid_index_list = easy[:size]
    elif difficulty == 1:
        valid_index_list = median_easy[:size]
    elif difficulty == 2:
        valid_index_list = median_hard[:size]
    elif difficulty == 3:
        valid_index_list = hard[:size]
    return valid_index_list


def ST_1(valid_index_list, size_list, R):
    tracin_start_time = time.time()
    # optimizer = torch.optim.SGD(model.parameters(), lr=5)
    new_train_index_list = [set(),set(),set(),set(),set(),set()]
    for idx in valid_index_list:
        (label_example_valid, text_example_valid, offset_example_valid) = valid_list[idx]
        curr_trc = list()
        loss_function = torch.nn.CrossEntropyLoss()
        logits_test = model(text_example_valid.to(device), offset_example_valid.to(device))
        loss_test = loss_function(logits_test, label_example_valid.to(device))
        grad_z_test = grad(loss_test, model.parameters())
        grad_z_test = get_gradient(grad_z_test, model)
        for idx3, (label_example, text_example, offset_example) in enumerate(train_list):
            logits_train = model(text_example.to(device), offset_example.to(device))
            loss_train = loss_function(logits_train, label_example.to(device))
            grad_z_train = grad(loss_train, model.parameters())
            grad_z_train = get_gradient(grad_z_train, model)
            score = tracin_get(grad_z_test[1:], grad_z_train[1:]) * optimizer.state_dict()['param_groups'][0]['lr']
            curr_trc.append((idx3,float(score)))

        curr_trc.sort(key= lambda x: x[1], reverse=R)
        curr_trc = [x for x,y in curr_trc]
        new_train_index_list[0] = new_train_index_list[0].union(set(curr_trc[:size_list[0]]))
        new_train_index_list[1] = new_train_index_list[1].union(set(curr_trc[:size_list[1]]))
        new_train_index_list[2] = new_train_index_list[2].union(set(curr_trc[:size_list[2]]))
        new_train_index_list[3] = new_train_index_list[3].union(set(curr_trc[:size_list[3]]))
        new_train_index_list[4] = new_train_index_list[4].union(set(curr_trc[:size_list[4]]))
        new_train_index_list[5] = new_train_index_list[5].union(set(curr_trc[:size_list[5]]))
    print("tracin time:", time.time() - tracin_start_time)
    print("new training size:", len(new_train_index_list[0]), len(new_train_index_list[1]), len(new_train_index_list[2]),\
          len(new_train_index_list[3]),len(new_train_index_list[4]),len(new_train_index_list[5]))

    return new_train_index_list


def add_random_index(cur_set, limit):
    while len(cur_set)< limit:
        cur_set.add(np.random.randint(0,20000))


tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 1 # batch size for training
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

model.to(device)


checkpoint = torch.load('./checkpoints_4000/checkpoint_{}_epoch.pkl'.format(3))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint['epoch']
print("epoch_{} accuracy: {}".format(3, evaluate(test_list)) )

valid_index_list = SV_2(3,16)
new_train_index_list = ST_1(valid_index_list, [4000,6000,8000,10000,12000,14000], False)
# print("ST size:", len(new_train_index_set))
# add_random_index(new_train_index_set, 5000)
# valid_index_list = SV_2(3,8)
# new_train_index_set.union(ST_1(valid_index_list, 2000, False))


for new_train_index_set in new_train_index_list:
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
    new_train_list = list()
    for i in new_train_index_set:
        new_train_list.append(train_list[i])
    train_size = len(new_train_index_list)
    LR = 5
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(new_train_list)
        accu_val = evaluate(test_list)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accu_val))
        print('-' * 59)
