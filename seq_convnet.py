import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(
            4, 20, kernel_size=6, stride=1, padding="valid", dilation=1
        )
        self.conv2 = nn.Conv1d(
            20, 20, kernel_size=6, stride=1, padding="valid", dilation=1
        )
        self.dense = nn.Linear(1820, 100)
        self.dense2 = nn.Linear(100, 1)

    def forward(self, input):
        c1 = F.relu(self.conv1(input))
        c2 = F.relu(self.conv2(c1))
        m1 = F.max_pool1d(c2, kernel_size=13, stride=13)
        m1_f = torch.flatten(m1,start_dim=1,end_dim=2)
        d1 = F.sigmoid(self.dense(m1_f))
        d2 = self.dense2(d1)
        return d2


def seq_to_one_hot(seq):
    indexes = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    encoded_seq = torch.zeros([4, len(seq)]).cuda()
    for i, nuc in enumerate(seq):
        encoded_seq[indexes[nuc], i] = 1
    return encoded_seq.unsqueeze(0)


if __name__ == "__main__":
    net = Net()
    net.cuda()
    lr = 0.01
    # creating an input sequence of length 1200 for the network
    seq = "AAAATTTCCGGG" * 100
    encoded_seq = seq_to_one_hot(seq)
    output = net(encoded_seq)
    params = list(net.parameters())
    for param in params:
        if param.grad is not None:
            param.grad.zero_()

    output.backward()
    for param in params:
        print(param)
        print(param.grad)
        param = param - lr * param.grad
        print(param)
