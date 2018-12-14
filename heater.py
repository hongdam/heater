import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Heater:
    def __init__(self):

        assert torch.cuda.is_available(), "Heater is not available. Please install cuda"

        self._device = "cuda:1"
        self.heater = HeaterNet().to(self._device)
        self.conduct = optim.SGD(self.heater.parameters(), lr=0.00001)

    def turn_on(self):

        self.heater.train()

        while True:
            self.conduct.zero_grad()

            fuel = torch.rand(64, 2048).to(self._device)
            heat = self.heater(fuel)

            air = F.nll_loss(heat, torch.LongTensor(64).random_(0, 2048).to(self._device))
            air.backward()

            self.conduct.step()


class HeaterNet(nn.Module):
    def __init__(self):
        super(HeaterNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(2048, 4096),
                                 nn.Linear(4096, 4096),
                                 nn.Linear(4096, 2048))

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x, dim=1)


def main():
    heater = Heater()
    heater.turn_on()


if __name__ == "__main__":
    main()
