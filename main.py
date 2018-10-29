from __future__ import print_function
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms

from tqdm import tqdm


class Game(object):
    def __init__(self, sender, receiver, opt_s, opt_r):
        self.sender = sender
        self.receiver = receiver
        self.opt_s = opt_s
        self.opt_r = opt_r

    def step(self, samples):
        y = self.sender(samples[:, 0])
        scores = self.receiver(samples, y)

        self.opt_s.zero_grad()
        self.opt_r.zero_grad()

        target = torch.LongTensor([0]).view(1).expand(scores.shape[0])
        lossfn = nn.MultiMarginLoss(margin=1)
        loss = lossfn(scores, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.sender.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(self.receiver.parameters(), 5.0)

        self.opt_s.step()
        self.opt_r.step()

        return loss.item()


class Sender(nn.Module):
    def __init__(self):
        super(Sender, self).__init__()
        self.net = Net(nout=1) # unused for now

    def forward(self, positive):
        return torch.randint(0, 10, size=(positive.shape[0],)).long()


class Receiver(nn.Module):
    def __init__(self, net):
        super(Receiver, self).__init__()
        self.net = net

    def forward(self, samples, labels):
        nsamples = samples.shape[1]
        samples = samples.view(-1, *samples.shape[2:])
        logit = self.net(samples, labels)
        ydist = logit.view(-1, nsamples)
        return ydist


class EmbedNet(nn.Module):
    def __init__(self, nout, embed_dim=50):
        super(EmbedNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50 + embed_dim, nout)
        self.embed = nn.Embedding(10, embed_dim)

        self.nout = nout

    def forward(self, x, y):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        e = self.embed(y)
        # TODO: Remove contiguous. Be more clever.
        e = e.unsqueeze(1).expand(e.shape[0], x.shape[0]//e.shape[0], e.shape[1]).contiguous().view(x.shape[0], -1)
        x = torch.cat([x, e], 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


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


def wrap(loader, kneg=3):
    batch_iterator = iter(loader)

    while True:
        samples = []
        positive = next(batch_iterator)[0]
        samples.append(positive.unsqueeze(1))
        for _ in range(kneg):
            negative = next(batch_iterator)[0]
            samples.append(negative.unsqueeze(1))
        samples = torch.cat(samples, 1)

        yield samples


def train_game(args, sender, receiver, opt_s, opt_r, device, loader, epoch):
    game = Game(sender, receiver, opt_s, opt_r)

    losses = []
    for samples in tqdm(wrap(loader)):
        loss = game.step(samples)
        losses.append(loss)

    print('loss', np.array(losses).mean())


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

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


    sender = Sender()
    receiver = Receiver(net=EmbedNet(nout=1))
    opt_s = optim.SGD(sender.parameters(), lr=args.lr, momentum=args.momentum)
    opt_r = optim.SGD(receiver.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train_game(args, sender, receiver, opt_s, opt_r, device, train_loader, epoch)


if __name__ == '__main__':
    main()
