import time
import torch


class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count


class Timer(object):

    def __init__(self):
        self.interval = 0
        self.time = time.time()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval


class Confusion(object):
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index
    """

    def __init__(self, k, normalized=False):
        super(Confusion, self).__init__()
        self.k = k
        self.conf = torch.LongTensor(k, k).cuda()
        self.conf.fill_(0)
        self.normalized = normalized
        self.conf_flat = None

    def reset(self):
        self.conf.fill_(0)

    def add(self, output, target):
        output = output.squeeze()
        target = target.squeeze()
        _, pred = output.max(1)

        indices = (target * self.conf.stride(0) + pred.squeeze_().type_as(target)).type_as(self.conf)
        ones = torch.ones(1).type_as(self.conf).expand(indices.size(0))
        self.conf_flat = self.conf.view(-1)
        self.conf_flat.index_add_(0, indices, ones)

    def recall(self, clsId):
        i = clsId
        TP = self.conf[i, i].sum().item()
        TPuFN = self.conf[i, :].sum().item()
        if TPuFN == 0:
            return 0
        return float(TP) / TPuFN

    def precision(self, clsId):
        i = clsId
        TP = self.conf[i, i].sum().item()
        TPuFP = self.conf[:, i].sum().item()
        if TPuFP == 0:
            return 0
        return float(TP) / TPuFP

    def f1score(self, clsId):
        r = self.recall(clsId)
        p = self.precision(clsId)
        if (p + r) == 0:
            return 0
        return 2 * float(p * r) / (p + r)

    def acc(self):
        TP = self.conf.diag().sum().item()
        total = self.conf.sum().item()
        if total == 0:
            return 0
        return float(TP) / total

    def show(self, width=6):
        print("Confusion Matrix:")
        conf = self.conf
        rows = conf.size(0)
        cols = conf.size(1)
        for i in range(0, rows):
            for j in range(0, cols):
                print(("%" + str(width) + ".d") % conf[i, j], end='')
            print()
