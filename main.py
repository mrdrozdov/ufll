from __future__ import print_function
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms

from tqdm import tqdm


def get_yn(y):
    ys = torch.arange(10)
    ys = ys.view(1, -1).expand(y.shape[0], 10).contiguous()
    ys[:, 0] = y
    ys[range(y.shape[0]), y] = 0
    return ys[:, 1:]


class Game(object):
    def __init__(self, args, sender, receiver, baseline, opt_s, opt_r, opt_b):
        self.sender = sender
        self.receiver = receiver
        self.baseline = baseline
        self.opt_s = opt_s
        self.opt_r = opt_r
        self.opt_b = opt_b
        self.args = args

    def step(self, samples, labels):
        if self.sender.use_cuda:
            samples = samples.cuda()
            labels = labels.cuda()

        yval, y = self.sender(samples[:, 0], labels[:, 0])
        yn = get_yn(y)
        scores = self.receiver(samples, y, yn)

        self.opt_s.zero_grad()
        self.opt_r.zero_grad()

        loss = 0

        # xent loss
        target = torch.LongTensor([0]).view(1).expand(scores.shape[0])
        if self.sender.use_cuda:
            target = target.cuda()
        lossfn = nn.MultiMarginLoss(margin=self.args.margin)
        xent_loss = lossfn(scores, target)
        xent_loss.backward()
        loss += xent_loss

        # rl loss
        if self.sender.rl:
            self.opt_b.zero_grad()

            # reward
            reward = (scores.argmax(dim=1) == 0).float()
            exprew = self.baseline(samples[:, 0]).view(*reward.shape)
            advantage = reward - exprew.detach()
            p = F.log_softmax(yval, dim=1)[range(y.shape[0]), y]
            rl_loss = (advantage * p).mean()
            rl_loss.backward()
            loss += rl_loss

            # baseline
            bas_loss = nn.MSELoss()(exprew, reward)
            bas_loss.backward()
            loss += bas_loss

        torch.nn.utils.clip_grad_norm_(self.sender.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(self.receiver.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(self.baseline.parameters(), 5.0)

        self.opt_s.step()
        self.opt_r.step()
        self.opt_b.step()

        acc = (scores.argmax(dim=1) == 0).float().mean()

        return loss.item(), acc.item()


class Sender(nn.Module):
    def __init__(self, net, rl=True):
        super(Sender, self).__init__()
        self.net = net
        self.rl = rl

    def forward(self, x, y):
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
        if self.rl:
            out = self.net(x)
            y = torch.multinomial(F.softmax(out, dim=1), 1).view(-1)
            return out, y
        return None, y


class Receiver(nn.Module):
    def __init__(self, net):
        super(Receiver, self).__init__()
        self.net = net

    def forward(self, samples, labels, negs):
        samples = samples.view(-1, *samples.shape[2:])
        if self.use_cuda:
            samples = samples.cuda()
            labels = labels.cuda()
            negs = negs.cuda()
        scores = self.net(samples, labels, negs)
        return scores


class EmbedNet(nn.Module):
    def __init__(self, nout, embed_dim=20, out_dim=20):
        super(EmbedNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, out_dim)
        self.embed = nn.Embedding(10, embed_dim)

        self.out_dim = out_dim
        self.embed_dim = embed_dim

        self.mat = nn.Parameter(torch.FloatTensor(embed_dim, out_dim))
        self.mat.data.normal_()

        self.nout = nout

    def forward(self, x, y, yn):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        pos = self.embed(y).unsqueeze(1)
        neg = self.embed(yn)
        q = torch.cat([pos, neg], 1)
        qh = torch.mm(q.view(-1, self.embed_dim), self.mat).view(q.shape[0], q.shape[1], self.out_dim)

        scores = torch.sum(qh * x.unsqueeze(1), dim=2)

        return scores


class Net(nn.Module):
    def __init__(self, nout):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nout)

        self.nout = nout

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def forever(loader):
    while True:
        for x in loader:
            yield x


def wrap(loader_positive, loader_negative, k_neg=3, verbose=False, limit=None):
    bi_neg = forever(loader_negative)

    for i, (x_pos, y_pos) in tqdm(enumerate(loader_positive), disable=not verbose):
        samples = []
        labels = []
        samples.append(x_pos.unsqueeze(1))
        labels.append(y_pos.unsqueeze(1))
        for _ in range(k_neg):
            x_neg, y_neg = next(bi_neg)
            samples.append(x_neg.unsqueeze(1))
            labels.append(y_neg.unsqueeze(1))
        samples = torch.cat(samples, 1)
        labels = torch.cat(labels, 1)

        yield samples, labels

        if limit is not None and i >= limit:
            break


def train_game(args, sender, receiver, baseline, opt_s, opt_r, opt_b, device, loader_positive, loader_negative, epoch):
    game = Game(args, sender, receiver, baseline, opt_s, opt_r, opt_b)

    arrloss, arracc = [], []
    for samples, labels in wrap(loader_positive, loader_negative, k_neg=args.k_neg, limit=args.limit, verbose=True):
        loss, acc = game.step(samples, labels)
        arrloss.append(loss)
        arracc.append(acc)

    print('loss', np.array(arrloss).mean())
    print('acc', np.array(arracc).mean())


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--k_neg', type=int, default=3,
                        help='number of negative examples for training (default: 3)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--rl', action='store_true',
                        help='enables sampled sender labels')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print(json.dumps(args.__dict__, sort_keys=True, indent=4))

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    # model = Net().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


    sender = Sender(net=Net(nout=10).to(device), rl=args.rl).to(device)
    receiver = Receiver(net=EmbedNet(nout=1).to(device)).to(device)
    baseline = Net(nout=1).to(device)
    sender.use_cuda = use_cuda
    receiver.use_cuda = use_cuda
    baseline.use_cuda = use_cuda
    opt_s = optim.Adam(sender.parameters(), lr=args.lr)
    opt_r = optim.Adam(receiver.parameters(), lr=args.lr)
    opt_b = optim.Adam(baseline.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_game(args, sender, receiver, baseline, opt_s, opt_r, opt_b, device, train_loader, train_loader, epoch)


if __name__ == '__main__':
    main()
