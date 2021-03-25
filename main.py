from models.models import Encoder, Classifier, Discriminator
from datasets.datasets import get_usps, get_svhn, get_mnist
from core.train import train_src_adda, train_tgt_adda, train_src_robust, train_critic_wdgrl, train_tgt_wdgrl, train_revgard
from core.eval import eval_tgt_robust, eval_tgt
import params
from utils.utils import init_random_seed, model_init


def main():
    # init random seed
    init_random_seed(params.manual_seed)

    # Load dataset
    svhn_dataloader = get_svhn(split='train', download=True)
    # mnist_data_loader = get_mnist(train=True, download=True)
    # mnist_data_loader_eval = get_mnist(train=False, download=True)
    usps_data_loader = get_usps(train=True, download=True)
    usps_data_loader_eval = get_usps(train=False, download=True)

    tgt_encoder = model_init(Encoder(), params.tgt_encoder_revgrad_path)
    critic = model_init(Discriminator(in_dims=params.d_in_dims,
                                      h_dims=params.d_h_dims,
                                      out_dims=params.d_out_dims))
    clf = model_init(Classifier(), params.clf_revgrad_path)

    # Train source model for adda
    print("========================  Training  ========================")
    tgt_encoder, clf = train_revgard(tgt_encoder, clf, critic, usps_data_loader, usps_data_loader, robust=True)
    # eval target encoder on test set of target dataset
    print("====== Evaluating classifier for encoded target domain ======")
    print("-------- source domain --------")
    eval_tgt_robust(tgt_encoder, clf, usps_data_loader_eval)
    print("-------- domain adaption --------")
    eval_tgt_robust(tgt_encoder, clf, usps_data_loader_eval)

if __name__ == '__main__':
    main()
