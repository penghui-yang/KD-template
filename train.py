from metric import Timer, AverageMeter, Confusion


def train(epoch, train_loader, learner):

    data_timer = Timer()
    batch_timer = Timer()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    confusion = Confusion(train_loader.num_classes)

    print('\n\n==== Epoch:{0} ===='.format(epoch))
    learner.train()
    print("LR: ", learner.optimizer.param_groups[0]['lr'])

    data_timer.tic()
    batch_timer.tic()
    print('Itr            |Batch time     |Data Time      |Loss')

    for i, (input, target) in enumerate(train_loader):

        data_time.update(data_timer.toc())  # measure data loading time

        # Prepare the inputs
        input = input.cuda()
        target = target.cuda()

        # Optimization
        loss, output = learner.learn(input, target)
        confusion.add(output, target)

        # Measure elapsed time
        batch_time.update(batch_timer.toc())
        data_timer.toc()

        # Mini-Logs
        losses.update(loss, input.size(0))

        if i % 100 == 0 or i == len(train_loader) - 1:
            print('[{0:6d}/{1:6d}]\t'
                  '{batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                  '{loss.val:.3f} ({loss.avg:.3f})'.format(
                i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    confusion.show()
    print("Train Set Accuracy: %.4f" % confusion.acc())
