import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from tqdm import tqdm
import wandb

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense = nn.Linear(4800, 2400)
        self.dense2 = nn.Linear(2400, 1200)
        self.dense3 = nn.Linear(1200,600)
        self.dense4 = nn.Linear(600, 300)
        self.dense5 = nn.Linear(300,1)

    def forward(self, input):
        d1 = F.sigmoid(self.dense(input.flatten(start_dim=1,end_dim=2)))
        d2 = F.sigmoid(self.dense2(d1))
        d3 = F.sigmoid(self.dense3(d2))
        d4 = F.sigmoid(self.dense4(d3))
        d5 = self.dense5(d4)
        return d5
    

def seq_to_one_hot(seq):
    indexes = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    encoded_seq = torch.zeros([4, len(seq)])
    for i, nuc in enumerate(seq):
        encoded_seq[indexes[nuc], i] = 1
    return encoded_seq.unsqueeze(0)

def count_a(seq):
    return seq.count("A")

if __name__=="__main__":
    nucs = ["A","C","G","T"]
    sequences_train = ["".join(random.choices(nucs,k=1200,weights=[random.random() for i in range(4)])) for _ in range(20000)]
    sequences_test = ["".join(random.choices(nucs,k=1200,weights=[random.random() for i in range(4)])) for _ in range(1000)]

    encoded_sequences_train = torch.tensor(np.array([seq_to_one_hot(seq) for seq in sequences_train])).squeeze().float()
    encoded_sequences_test = torch.tensor(np.array([seq_to_one_hot(seq) for seq in sequences_test])).squeeze().float()

    net = Net()
    epochs = 20
    optim = torch.optim.AdamW(params = net.parameters())
    loss_fn = nn.MSELoss(reduction='mean')
    Xtrain = encoded_sequences_train
    Ytrain = torch.tensor([count_a(seq)/1200 for seq in sequences_train]).float().unsqueeze(dim=1)
    Xtest = encoded_sequences_test
    Ytest = torch.tensor([count_a(seq)/1200 for seq in sequences_test]).float().unsqueeze(dim=1)
    dataset = torch.utils.data.TensorDataset(Xtrain,Ytrain)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=20,shuffle=True)

    wandb.init(
        project="count_a",
        config={
        "sequence_lenth": 1200,
        "batch_size": 20,
        "architecture": "MLP",
        "epochs": epochs,
        }
    )

    for e in tqdm(range(epochs)):
        net.eval()
        predTest = net(Xtest)
        predTrain = net(Xtrain)
        lossTest = loss_fn(predTest,Ytest).item()
        loss_fn.zero_grad()
        lossTrain = loss_fn(predTrain,Ytrain).item()
        loss_fn.zero_grad()

        net.train()
        for i, (batch_X, batch_Y) in enumerate(dataloader):
            pred = net(batch_X)
            loss = loss_fn(pred,batch_Y)
            loss.backward()
            wandb.log({"loss": loss.item()})
            optim.step()
            optim.zero_grad()

    torch.save(net.state_dict(), 'models/first_model.pth')