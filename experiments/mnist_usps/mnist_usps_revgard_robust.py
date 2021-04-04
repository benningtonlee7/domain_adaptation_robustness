from models.models import Encoder, Classifier, Discriminator
from datasets.datasets import get_usps, get_mnist
from core.train import train_revgrad
from core.eval import eval_tgt_robust
from utils.utils import init_random_seed, model_init
import params


def main():
    # init random seed
    init_random_seed(params.manual_seed)

    # Load dataset
    mnist_data_loader = get_mnist(train=True, download=True)
    mnist_data_loader_eval = get_mnist(train=False, download=True)
    usps_data_loader = get_usps(train=True, download=True)
    usps_data_loader_eval = get_usps(train=False, download=True)

    # Model init revgard
    tgt_encoder = model_init(Encoder(), params.tgt_encoder_revgrad_rb_path)
    critic = model_init(Discriminator(), params.disc_revgard_rb_path)
    clf = model_init(Classifier(), params.clf_revgrad_rb_path)

    # Train models
    print("====== Robust Training source encoder and classifier in MNIST and USPS domains ======")
    if not (tgt_encoder.pretrained and clf.pretrained and critic.pretrained and params.model_trained):
        tgt_encoder, clf, critic = train_revgrad(tgt_encoder, clf, critic,
                                                 mnist_data_loader, usps_data_loader, robust=True)

    # Eval target encoder on test set of target dataset
    print("====== Evaluating classifier for encoded MNIST and USPS domain ======")
    print("-------- MNIST domain --------")
    eval_tgt_robust(tgt_encoder, clf, mnist_data_loader_eval)
    print("-------- USPS adaption --------")
    eval_tgt_robust(tgt_encoder, clf, usps_data_loader_eval)


if __name__ == '__main__':
    main()
