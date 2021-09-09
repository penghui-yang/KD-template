import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR

import models
import dataloader
from learner import Learner, Learner_KD
from train import train
from evaluate import evaluate


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    torch.cuda.empty_cache()

    train_loader, test_loader = dataloader.CIFAR10(batch_sz=200, num_workers=4)

    # teacher model & student model
    model_teacher = models.resnet50(train_loader.num_classes)
    model_teacher = nn.DataParallel(model_teacher)
    model_teacher = model_teacher.cuda()

    model_student = models.alexnet(train_loader.num_classes)
    model_student = nn.DataParallel(model_student)
    model_student = model_student.cuda()

    criterion_t = CrossEntropyLoss()

    teacher_pretrained = False

    # teacher model training

    if not teacher_pretrained:
        max_epoch_t = 40
        schedule_t = [15]
        optimizer_t = torch.optim.Adam(model_teacher.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler_t = MultiStepLR(optimizer_t, milestones=schedule_t, gamma=0.1)
        learner_t = Learner(model_teacher, criterion_t, optimizer_t, scheduler_t)
        for epoch in range(max_epoch_t):
            train(epoch, train_loader, learner_t)
            evaluate(test_loader, model_teacher)
            learner_t.scheduler.step()
        torch.save(model_teacher.state_dict(), "pretrained_models/model_teacher_resnet50.pth")
    else:
        model_teacher.load_state_dict(torch.load("pretrained_models/model_teacher_resnet50.pth"))

    model_teacher.eval()

    # student model training

    max_epoch_s = 40
    schedule_s = [15]
    criterion_s = CrossEntropyLoss()
    criterion_t2s = nn.KLDivLoss()
    optimizer_s = torch.optim.Adam(model_student.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler_s = MultiStepLR(optimizer_s, milestones=schedule_s, gamma=0.1)
    learner_s = Learner_KD(model_teacher, model_student, criterion_s, criterion_t2s, optimizer_s, scheduler_s, 1)

    for epoch in range(max_epoch_s):
        train(epoch, train_loader, learner_s)
        evaluate(test_loader, model_student)
        learner_s.scheduler.step()
