from models.models import Encoder, Classifier, Discriminator
from datasets.datasets import get_usps, get_mnist
from core.train import train_src_adda, train_tgt_adda
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

    # Model init ADDA
    src_encoder = model_init(Encoder(), params.src_encoder_adda_path)
    tgt_encoder = model_init(Encoder(), params.tgt_encoder_adda_path)
    critic = model_init(Discriminator(), params.disc_adda_path)
    clf = model_init(Classifier(), params.clf_adda_path)

    # Train source model for adda
    print("====== Training source encoder and classifier in MNIST domain ======")
    if not (src_encoder.pretrained and clf.pretrained and
            params.model_trained):
        src_encoder, clf = train_src_adda(src_encoder, clf, mnist_data_loader)

    # Eval source model
    print("====== Evaluating classifier for MNIST domain ======")
    eval_tgt_robust(src_encoder, clf, mnist_data_loader_eval)

    # Train target encoder
    print("====== Training encoder for USPS domain ======")
    # Initialize target encoder's weights with those of the source encoder
    if not tgt_encoder.pretrained:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.pretrained and critic.pretrained and
            params.model_trained):
        tgt_encoder = train_tgt_adda(src_encoder, tgt_encoder, clf, critic,
                                     mnist_data_loader, usps_data_loader, usps_data_loader_eval, robust=False)

    # Eval target encoder on test set of target dataset
    print("====== Evaluating classifier for encoded USPS domain ======")
    print("-------- Source only --------")
    eval_tgt_robust(src_encoder, clf, usps_data_loader_eval)
    print("-------- Domain adaption --------")
    eval_tgt_robust(tgt_encoder, clf, usps_data_loader_eval)


if __name__ == '__main__':
    main()
