import torch
from torch.autograd import Variable

from tqdm import tqdm


class Trainer(object):
    _cuda_available = torch.cuda.is_available()

    def __init__(self, model, optimizer, loss_f, logger, scheduler=None, log_freq=5, verb=True, use_cuda=True):
        self.model = model
        if self._cuda_available:
            model.cuda()
        self._optimizer = optimizer
        self._loss_f = loss_f
        self._logger = logger
        self._scheduler = scheduler
        self._steps = {"train": 0,
                       "test": 0}
        self._epochs = 0
        self._log_freq = log_freq
        self._verb = verb
        self._use_cuda = use_cuda

    def _loop(self, data_loader, is_train=True):
        mode = "train" if is_train else "test"
        total_size = len(data_loader.dataset)
        data_loader = tqdm(data_loader, ncols=80) if self._verb else data_loader
        loop_loss = []
        loop_correct = []
        for input, target in data_loader:
            loss, correct = self._iteration(input, target, is_train)
            loop_loss.append(loss)
            loop_correct.append(correct)
            self._steps[mode] += 1
            if is_train and self._steps[mode] % self._log_freq == 0:
                self._logger.add_scalar("iter_train_loss", loss, self._steps[mode])

        if any(loop_loss):
            self._logger.add_scalar(f"epoch_{mode}_loss", sum(loop_loss) / len(data_loader), self._epochs)

        if any(loop_correct):
            self._logger.add_scalar(f"{mode}_accuracy", sum(loop_correct) / total_size, self._epochs)

    def _iteration(self, input, target, is_train):
        input, target = self.variable(input, volatile=not is_train), self.variable(target, volatile=not is_train)
        self._optimizer.zero_grad()
        output = self.model(input)
        loss = self._loss_f(output, target)
        if is_train:
            loss.backward()
            self._optimizer.step()
        loss = loss.data[0]
        correct = self.correct(output, target)
        return loss, correct

    def train(self, data_loader):
        self.model.train()
        self._loop(data_loader)
        if self._scheduler is not None:
            self._scheduler.step()
        for name, param in self.model.named_parameters():
            self._logger.add_histogram(name, param, self._epochs, bins="sqrt")
        self._epochs += 1

    def test(self, data_loader):
        self.model.eval()
        self._loop(data_loader, is_train=False)

    def start(self, epochs, train_data, test_data):
        try:
            for ep in range(1, epochs + 1):
                print(f"epochs: {ep}")
                self.train(train_data)
                self.test(test_data)
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            self._logger.close()

    @staticmethod
    def correct(input, target):
        return (input.max(dim=1)[1] == target).sum().data[0]

    def variable(self, t, **kwargs):
        if self._cuda_available and self._use_cuda:
            t = t.cuda()
        return Variable(t, **kwargs)
