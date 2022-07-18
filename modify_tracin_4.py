"""
training size: 120000
testing size: 7600
batch num: 1875
"""
import time
from textaugment import Wordnet
import torch
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torch.autograd import grad
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import numpy as np
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

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
    log_interval = 2000
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        # if idx == 1000:
        #     break
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

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
optimizer = torch.optim.SGD(model.parameters(), lr=5)

model.to(device)

train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)


VALID_SIZE = 1000
TRAIN_SIZE = 20000

#attenuate the data
valid_list = []
train_list = []
for idx, (label, text, offsets) in enumerate(valid_dataloader):
    if idx == VALID_SIZE:
        break
    valid_list.append([label, text, offsets])
for idx, (label, text, offsets) in enumerate(train_dataloader):
    if idx == TRAIN_SIZE:
        break
    train_list.append([label, text, offsets])
checkpoint = torch.load('./checkpoints/checkpoint_{}_epoch.pkl'.format(3))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint['epoch']
print("epoch_{} accuracy: ".format(3), evaluate(test_dataloader))
valid_accu_log = np.load('valid_accu_log.npy', allow_pickle=True).item()
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

tracin_start_time = time.time()
new_train_index_set = set()
select_valid_list = easy[:4]
for idx in select_valid_list:
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

    curr_trc.sort(key= lambda x: x[1], reverse=True)
    # tmp_train_index_set = set()
    for idx4, score in curr_trc[:4000]:
        new_train_index_set.add(idx4)

    # if new_train_index_set is None:
    #     new_train_index_set = tmp_train_index_set
    # else:
    #     new_train_index_set = new_train_index_set.intersection(tmp_train_index_set)

select_valid_list = median_easy[:4]
for idx in select_valid_list:
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

    curr_trc.sort(key= lambda x: x[1], reverse=True)
    for idx4, score in curr_trc[5000:9000]:
        new_train_index_set.add(idx4)

select_valid_list = median_hard[:4]
for idx in select_valid_list:
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

    curr_trc.sort(key= lambda x: x[1], reverse=False)
    for idx4, score in curr_trc[5000:9000]:
        new_train_index_set.add(idx4)

select_valid_list = hard[:2]
for idx in select_valid_list:
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

    curr_trc.sort(key= lambda x: x[1], reverse=False)
    for idx4, score in curr_trc[:4000]:
        new_train_index_set.add(idx4)
    #
    # if new_train_index_set is None:
    #     new_train_index_set = tmp_train_index_set
    # else:
    #     new_train_index_set = new_train_index_set.intersection(tmp_train_index_set)

print("tracin time:", time.time()-tracin_start_time)
print("new training size:", len(new_train_index_set))
new_train_list = list()
for i in new_train_index_set:
    new_train_list.append(train_list[i])

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
epoch = 0
for i in range(10):
    epoch += 1
    epoch_start_time = time.time()
    train(new_train_list)
    accu_val = evaluate(test_dataloader)
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
