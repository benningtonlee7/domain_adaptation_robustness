from models.models import Encoder, Classifier, Discriminator
from datasets.datasets import get_svhn, get_mnist
from core.train import train_src_adda, train_tgt_adda
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

    # Model init ADDA
    src_encoder = model_init(Encoder(), params.src_encoder_adda_path)
    tgt_encoder = model_init(Encoder(), params.tgt_encoder_adda_path)
    critic = model_init(Discriminator(in_dims=params.d_in_dims,
                                      h_dims=params.d_h_dims,
                                      out_dims=params.d_out_dims),
                                    params.disc_adda_path)
    clf = model_init(Classifier(), params.clf_adda_path)

    # Train source model for adda
    print("====== Training source encoder and classifier in SVHN domain ======")

    if not (src_encoder.pretrained and clf.pretrained and
            params.model_trained):
        src_encoder, clf = train_src_adda(src_encoder, clf, svhn_data_loader)

    # Eval source model
    print("====== Evaluating classifier for SVHN domain ======")
    eval_tgt(src_encoder, clf, svhn_data_loader_eval)

    # Train target encoder
    print("====== Training encoder for MNIST domain ======")
    # Initialize target encoder's weights with those of the source encoder
    if not tgt_encoder.pretrained:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.pretrained and critic.pretrained and
            params.model_trained):
        tgt_encoder = train_tgt_adda(src_encoder, tgt_encoder, critic,
                                     svhn_data_loader, mnist_data_loader, robust=True)

    # eval target encoder on test set of target dataset
    print("====== Evaluating classifier for encoded MNIST domain ======")
    print("-------- Source only --------")
    eval_tgt(src_encoder, clf, mnist_data_loader_eval)
    print("-------- Domain adaption --------")
    eval_tgt(tgt_encoder, clf, mnist_data_loader_eval)


if __name__ == '__main__':
    main()