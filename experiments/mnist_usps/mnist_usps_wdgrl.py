from models.models import Encoder, Classifier, Discriminator
from datasets.datasets import get_usps, get_mnist
from core.train import train_critic_wdgrl, train_tgt_wdgrl
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

    # Model init WDGRL
    tgt_encoder = model_init(Encoder(), params.encoder_wdgrl_path)
    critic = model_init(Discriminator(), params.disc_wdgrl_path)
    clf = model_init(Classifier(), params.clf_wdgrl_path)

    # Train target encoder
    print("====== Training encoder for both MNIST and USPS domains ======")
    if not (tgt_encoder.pretrained and clf.pretrained and params.model_trained):
        tgt_encoder, clf = train_tgt_wdgrl(tgt_encoder, clf, critic,
                                           mnist_data_loader, usps_data_loader, usps_data_loader_eval, robust=False)

    # Eval target encoder on test set of target dataset
    print("====== Evaluating classifier for encoded MNIST and USPS domains ======")
    print("-------- MNIST domain --------")
    eval_tgt_robust(tgt_encoder, clf, mnist_data_loader_eval)
    print("-------- USPS adaption --------")
    eval_tgt_robust(tgt_encoder, clf, usps_data_loader_eval)


if __name__ == '__main__':
    main()
