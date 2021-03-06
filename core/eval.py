import torch
import torch.nn as nn
import params
from utils.utils import make_variable, normalize
from core.pgd import attack_pgd


def eval_tgt_robust(encoder, classifier, critic, data_loader):
    """Evaluate model for target domain with attack on labels and domains"""
    # Set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()
    critic.eval()

    # Init loss and accuracy
    loss, acc = 0, 0
    test_robust_loss, test_robust_acc = 0, 0

    # Set loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images)
        labels = make_variable(labels)
        domain_tgt = make_variable(torch.ones(images.size(0)).long())

        delta_src = attack_pgd(encoder, classifier, images, labels)
        delta_domain = attack_pgd(encoder, critic, images, domain_tgt)
        delta_src = delta_src.detach()
        delta_domain = delta_domain.detach()

        # Compute loss
        robust_images = normalize(torch.clamp(images + delta_src[:images.size(0)]
                                              + delta_domain[:images.size(0)],
                                              min=params.lower_limit, max=params.upper_limit))
        robust_preds = classifier(encoder(robust_images))
        test_robust_loss += criterion(robust_preds, labels).item()

        out = classifier(encoder(images))
        loss += criterion(out, labels).item()

        test_robust_acc += torch.sum(robust_preds.max(1)[1] == labels.data).double()
        acc += torch.sum(out.max(1)[1] == labels.data).double()

    loss /= len(data_loader)
    test_robust_loss /= len(data_loader)
    acc = acc / len(data_loader.dataset)
    test_robust_acc = test_robust_acc / len(data_loader.dataset)

    print("Avg Evaluation Loss: {:.4f}, Avg Evaluation Accuracy: {:.4%}, Ave Evaluation Robust Loss: {:.4f}, "
          "Ave Robust Accuracy: {:.4%}".format(loss, acc, test_robust_loss, test_robust_acc))


def eval_tgt(encoder, classifier, data_loader):
    """Evaluate model for target domain with attack on labels only"""
    # Set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # Init loss and accuracy
    loss, acc = 0, 0
    test_robust_loss, test_robust_acc = 0, 0

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images)
        labels = make_variable(labels)

        delta_src = attack_pgd(encoder, classifier, images, labels)
        delta_src = delta_src.detach()

        # Compute loss
        robust_images = normalize(torch.clamp(images + delta_src[:images.size(0)],
                                              min=params.lower_limit, max=params.upper_limit))
        robust_preds = classifier(encoder(robust_images))
        test_robust_loss += criterion(robust_preds, labels).item()

        out = classifier(encoder(images))
        loss += criterion(out, labels).item()

        test_robust_acc += torch.sum(robust_preds.max(1)[1] == labels.data).double()
        acc += torch.sum(out.max(1)[1] == labels.data).double()

    loss /= len(data_loader)
    test_robust_loss /= len(data_loader)
    acc = acc / len(data_loader.dataset)
    test_robust_acc = test_robust_acc / len(data_loader.dataset)

    print("Avg Evaluation Loss: {:.4f}, Avg Evaluation Accuracy: {:.4%}, Ave Evaluation Robust Loss: {:.4f}, "
          "Ave Robust Accuracy: {:.4%}".format(loss, acc, test_robust_loss, test_robust_acc))
