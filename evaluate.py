from metric import Confusion


def evaluate(eval_loader, model):
    # Initialize all meters
    confusion = Confusion(eval_loader.num_classes)

    print('---- Evaluation ----')
    model.eval()

    for i, (input, target) in enumerate(eval_loader):
        input = input.cuda()
        target = target.cuda()

        # Inference
        output = model(input)

        # Update the performance meter
        output = output.detach()
        confusion.add(output, target)

    confusion.show()
    acc = confusion.acc()
    print("Validation Set Accuracy: %.4f" % acc)
    return acc
