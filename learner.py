import torch.nn as nn
import torch.nn.functional as F


class Learner(nn.Module):
    def __init__(self, model, criterion, optimizer, scheduler):
        super(Learner, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0

    def forward(self, x):
        return self.model.forward(x)

    def forward_with_criterion(self, inputs, targets):
        out = self.forward(inputs)
        return self.criterion(out, targets), out

    def learn(self, inputs, targets):
        loss, out = self.forward_with_criterion(inputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.detach()
        out = out.detach()
        return loss, out


class Learner_KD(nn.Module):
    def __init__(self, model_t, model_s, criterion_s, criterion_t2s, optimizer, scheduler, beta=0.9):
        super(Learner_KD, self).__init__()
        self.model_t = model_t
        self.model_s = model_s
        self.criterion_s = criterion_s
        self.criterion_t2s = criterion_t2s
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.beta = beta
        self.epoch = 0

    def forward_t(self, x):
        return self.model_t.forward(x)

    def forward_s(self, x):
        return self.model_s.forward(x)

    def forward_with_criterion(self, inputs, targets):
        out_teacher = self.forward_t(inputs)
        out_student = self.forward_s(inputs)
        loss_hard = self.criterion_s(out_student, targets)
        loss_soft = self.criterion_t2s(F.softmax(out_teacher, dim=1), F.softmax(out_student, dim=1))
        loss = (1 - self.beta) * loss_hard + self.beta * loss_soft
        return loss, out_student

    def learn(self, inputs, targets):
        loss, out = self.forward_with_criterion(inputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.detach()
        out = out.detach()
        return loss, out

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.model_t.eval()
        return self
