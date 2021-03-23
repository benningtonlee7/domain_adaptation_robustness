from models.models import Encoder, Classifier, Discriminator
from datasets.datasets import get_usps, get_svhn, get_mnist
from core.train import train_src_adda, train_tgt_adda, train_src_robust, train_critic_wdgrl, train_tgt_wdgrl
from core.eval import eval_src_robust, eval_src, eval_tgt
import params
from utils.utils import init_random_seed, model_init


def main():
    # init random seed
    init_random_seed(params.manual_seed)

    # Load dataset
    svhn_dataloader = get_svhn(split='train', download=True)

    # src_data_loader = get_mnist(train=True, download=True)
    # src_data_loader_eval = get_mnist(train=False, download=True)
    tgt_data_loader = get_usps(train=True, download=True)
    tgt_data_loader_eval = get_usps(train=False, download=True)
    src_data_loader = get_usps(train=True, download=False)
    src_data_loader_eval = get_usps(train=False, download=False)
    # Model init ADDA
    src_encoder = model_init(Encoder(), params.src_encoder_path)
    tgt_encoder = model_init(Encoder(), params.tgt_encoder_path)
    critic = model_init(Discriminator(in_dims=params.d_in_dims,
                                             h_dims=params.d_h_dims,
                                             out_dims=params.d_out_dims),
                                             params.disc_path)
    clf = model_init(Classifier(), params.clf_path)
    
    
    # Train source model for adda
    print("====== Training source encoder and classifier in source domain ======")
    print("-------- Source Encoder --------")
    print(src_encoder)
    print("-------- Source Classifier --------")
    print(clf)

    c = train_tgt_wdgrl(src_encoder, clf, critic, src_data_loader, tgt_data_loader, robust=False)

    # if not (src_encoder.pretrained and clf.pretrained and
    #         params.src_model_trained):
    #     src_encoder, clf = train_src_adda(src_encoder, clf, src_data_loader)

    # Eval source model
    print("====== Evaluating classifier for source domain ======")
    # eval_src(src_encoder, clf, src_data_loader_eval)

    # Train target encoder by GAN
    print("====== Training encoder for target domain ======")
    print("-------- Target Encoder -------- ")
    print(tgt_encoder)
    print("-------- Critic -------- ")
    print(critic)

    # Initialize target encoder's weights with those of the source encoder
    if not tgt_encoder.pretrained:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.pretrained and critic.pretrained and
            params.tgt_model_trained):
        tgt_encoder = train_tgt_adda(src_encoder, tgt_encoder, critic,
                                 src_data_loader, tgt_data_loader, robust=True)

    # eval target encoder on test set of target dataset
    print("====== Evaluating classifier for encoded target domain ======")
    print("-------- source only --------")
    eval_tgt(src_encoder, clf, tgt_data_loader_eval)
    print("-------- domain adaption --------")
    eval_tgt(tgt_encoder, clf, tgt_data_loader_eval)

if __name__ == '__main__':
    main()
