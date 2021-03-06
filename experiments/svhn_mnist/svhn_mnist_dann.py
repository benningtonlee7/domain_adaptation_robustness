from models.models import Encoder, Classifier, Discriminator
from datasets.datasets import get_svhn, get_mnist
from core.train import train_dann
from core.eval import eval_tgt
from utils.utils import init_random_seed, model_init
import params


def main():
    # init random seed
    init_random_seed(params.manual_seed)

    # Load dataset
    svhn_data_loader = get_svhn(split='train', download=True)
    svhn_data_loader_eval = get_svhn(split='test', download=True)
    mnist_data_loader = get_mnist(train=True, download=True)
    mnist_data_loader_eval = get_mnist(train=False, download=True)

    # Model init DANN
    tgt_encoder = model_init(Encoder(), params.tgt_encoder_dann_rb_path)
    critic = model_init(Discriminator(in_dims=params.d_in_dims,
                                      h_dims=params.d_h_dims,
                                      out_dims=params.d_out_dims),
                        params.disc_dann_path)
    clf = model_init(Classifier(), params.clf_dann_path)

    # Train models
    print("====== Training source encoder and classifier in SVHN and MNIST domains ======")
    if not (tgt_encoder.pretrained and clf.pretrained and critic.pretrained and params.model_trained):
        tgt_encoder, clf, critic = train_dann(tgt_encoder, clf, critic,
                                              svhn_data_loader, mnist_data_loader, mnist_data_loader_eval, robust=False)

    # Eval target encoder on test set of target dataset
    print("====== Evaluating classifier for encoded SVHN and MNIST domain ======")
    print("-------- SVHN domain --------")
    eval_tgt(tgt_encoder, clf, svhn_data_loader_eval)
    print("-------- MNIST adaption --------")
    eval_tgt(tgt_encoder, clf, mnist_data_loader_eval)

if __name__ == '__main__':
    main()