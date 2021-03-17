import torch
import torch.nn as nn
import params
from utils import make_variable, normalize
from core.pgd import attack_pgd

def eval_src_robust(encoder, classifier, data_loader):

    # Set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # Init loss and accuracy
    loss, acc = 0, 0
    test_robust_loss, test_robust_acc = 0, 0

    # Set loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels)

        delta = attack_pgd(encoder, images, labels)
        delta = delta.detach()

        # Compute loss for critic
        robust_images = normalize(torch.clamp(images + delta[:images.size(0)],
                                  min=params.lower_limit, max=params.upper_limit))
        robust_preds = classifier(encoder(robust_images))
        test_robust_loss += criterion(robust_preds, labels).item()

        out = classifier(encoder(images))
        loss += criterion(out, labels).item()

        test_robust_acc += torch.sum(robust_preds.max(1)[1] == labels.data)
        acc += torch.sum(out.max(1)[1] == labels.data)

    loss /= len(data_loader)
    test_robust_loss /= len(data_loader)
    acc = acc.double() / len(data_loader.dataset)
    test_robust_acc = test_robust_acc.double() / len(data_loader.dataset)

    print("Avg Evaluation Loss: {:4f}, Avg Evaluation Accuracy: {:4f%}, Ave Evaluation Robust Loss: {:4f}, "
          "Ave Robust Accuracy: {:.4%}".format(loss, acc, test_robust_loss, test_robust_acc))


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # Set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # Init loss and accuracy
    loss, acc = 0, 0

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels)

        out = classifier(encoder(images))
        loss += criterion(out, labels).item()
        _, preds = torch.max(out, 1)
        acc += torch.sum(preds == labels.data)

    loss /= len(data_loader)
    acc = acc.double() / len(data_loader.dataset)

    print("Avg Evaluation Loss: {:4f}, Avg Evaluation Accuracy: {:.4%}".format(loss, acc))

def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # Set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # Init loss and accuracy
    loss, acc = 0, 0

    # Set loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        out = classifier(encoder(images))
        loss += criterion(out, labels).item()
        _, preds = torch.max(out, 1)

        acc += torch.sum(preds == labels.data)

    loss /= len(data_loader)
    acc = acc.double() / len(data_loader.dataset)

    print("Avg Evaluation Loss: {:4f}, Avg Accuracy: {:.4%}".format(loss, acc))