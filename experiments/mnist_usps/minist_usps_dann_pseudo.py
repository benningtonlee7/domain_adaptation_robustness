from models.models import Encoder, Classifier, Discriminator
from datasets.datasets import get_usps, get_mnist
from core.train import train_dann, train_src_adda, train_src_robust
from core.eval import eval_tgt_robust, eval_tgt
from utils.utils import init_random_seed, model_init, pseudo_label
import params


def main():
    # init random seed
    init_random_seed(params.manual_seed)

    # Load dataset
    mnist_data_loader = get_mnist(train=True, download=True)
    mnist_data_loader_eval = get_mnist(train=False, download=True)
    usps_data_loader = get_usps(train=True, download=True)
    usps_data_loader_eval = get_usps(train=False, download=True)

    # Model init DANN
    tgt_encoder = model_init(Encoder(), params.tgt_encoder_dann_rb_path)
    critic = model_init(Discriminator(), params.disc_dann_rb_path)
    clf = model_init(Classifier(), params.clf_dann_rb_path)

    # Train models
    print("====== Robust Training source encoder and classifier in MNIST and USPS domains ======")
    if not (tgt_encoder.pretrained and clf.pretrained and critic.pretrained and params.model_trained):
        tgt_encoder, clf, critic = train_dann(tgt_encoder, clf, critic,
                                              mnist_data_loader, usps_data_loader, usps_data_loader_eval, robust=False)

    # Eval target encoder on test set of target dataset
    print("====== Evaluating classifier for encoded MNIST and USPS domains ======")
    print("-------- MNIST domain --------")
    eval_tgt_robust(tgt_encoder, clf, critic, mnist_data_loader_eval)
    print("-------- USPS adaption --------")
    eval_tgt_robust(tgt_encoder, clf, critic, usps_data_loader_eval)

    print("====== Pseudo labeling on USPS domain ======")
    pseudo_label(tgt_encoder, clf, "usps_train_pseudo", usps_data_loader)

    # Init a new model
    tgt_encoder = model_init(Encoder(), params.tgt_encoder_path)
    clf = model_init(Classifier(), params.clf_path)

    # Load pseudo labeled dataset
    usps_pseudo_loader = get_usps(train=True, download=True, get_pseudo=True)

    print("====== Standard training on USPS domain with pseudo labels ======")
    if not (tgt_encoder.pretrained and clf.pretrained):
        train_src_adda(tgt_encoder, clf, usps_pseudo_loader, mode='ADV')
    print("====== Evaluating on USPS domain with real labels ======")
    eval_tgt(tgt_encoder, clf, usps_data_loader_eval)

    tgt_encoder = model_init(Encoder(), params.tgt_encoder_rb_path)
    clf = model_init(Classifier(), params.clf_rb_path)
    print("====== Robust training on USPS domain with pseudo labels ======")
    if not (tgt_encoder.pretrained and clf.pretrained):
        train_src_robust(tgt_encoder, clf, usps_pseudo_loader, mode='ADV')
    print("====== Evaluating on USPS domain with real labels ======")
    eval_tgt(tgt_encoder, clf, usps_data_loader_eval)

if __name__ == '__main__':
    main()
