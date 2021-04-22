import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import params
from core.eval import eval_tgt, eval_tgt_robust
from core.pgd import attack_pgd
from utils.utils import make_variable, save_model, normalize, set_requires_grad, gradient_penalty, alda_loss


def train_src_adda(encoder, classifier, data_loader, mode='ADDA'):
    """Train classifier for source domain for ADDA"""
    # Step 1: Network setup
    # Set train state for both Dropout and BN layers
    encoder.train()
    classifier.train()

    # Set up optimizer and criterion
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.learning_rate, weight_decay=params.weight_decay)

    criterion = nn.CrossEntropyLoss()
    num_epochs = params.num_epochs_pre if mode == 'ADDA' else params.num_epochs
    # Step 2: Pretrain the source model
    for epoch in range(num_epochs):
        train_acc, train_loss, train_n = 0, 0, 0
        start_time = time.time()

        for step, (images, labels) in enumerate(data_loader):

            # Make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels)

            # Zero gradients for optimizer
            optimizer.zero_grad()

            # Compute loss for classifier
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            # Optimize source classifier
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_acc += torch.sum(preds.max(1)[1] == labels.data).double()
            train_n += labels.size(0)

            # Print step info
            if (step + 1) % params.log_step_pre == 0:
                print("Epoch [{}/{}] Step [{}/{}]: Training Loss: {:.4f} Training Accuracy: {:.4%}"
                      .format(epoch + 1,
                              num_epochs,
                              step + 1,
                              len(data_loader),
                              train_loss / train_n,
                              train_acc / train_n))
        time_elapsed = time.time() - start_time

        # Eval model on test set
        if (epoch + 1) % params.eval_step_pre == 0:
            eval_tgt(encoder, classifier, data_loader)

        # Save model parameters
        if (epoch + 1) % params.save_step_pre == 0:
            print('Epoch [{}/{}] completed in {:.0f}m {:.0f}s'.format(epoch + 1,
                                                                      num_epochs, time_elapsed // 60,
                                                                      time_elapsed % 60))
            root = params.adda_root if mode == 'ADDA' else params.model_root
            save_model(encoder, root, "{}-source-encoder-{}.pt".format(mode, epoch + 1))
            save_model(classifier, root, "{}-source-classifier-{}.pt".format(mode, epoch + 1))

    # Save final model
    root = params.adda_root if mode == 'ADDA' else params.model_root
    save_model(encoder, root, "{}-source-encoder-final.pt".format(mode))
    save_model(classifier, root, "{}-source-classifier-final.pt".format(mode))

    return encoder, classifier


def train_src_robust(encoder, classifier, data_loader, mode='ADDA'):
    """Train classifier for source domain with robust training for ADDA"""

    # Step 1: Network setup
    # Set train state for both Dropout and BN layers
    encoder.train()
    classifier.train()

    # Set up optimizer and criterion
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.learning_rate, weight_decay=params.weight_decay)

    criterion = nn.CrossEntropyLoss()
    num_epochs = params.num_epochs_pre if mode == 'ADDA' else params.num_epochs
    # Step 2: Pretrain the source model
    for epoch in range(num_epochs):

        # Init accuracy and loss
        start_time = time.time()
        train_loss, train_acc, train_n = 0, 0, 0
        train_robust_loss, train_robust_acc = 0, 0

        for step, (images, labels) in enumerate(data_loader):

            # Make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels)

            # Zero gradients for optimizer
            optimizer.zero_grad()

            delta = attack_pgd(encoder, classifier, images, labels)

            # Compute loss for critic with attack img
            robust_images = normalize(torch.clamp(images + delta[:images.size(0)],
                                                  min=params.lower_limit, max=params.upper_limit))
            robust_preds = classifier(encoder(robust_images))
            robust_loss = criterion(robust_preds, labels)

            # Optimize source classifier
            robust_loss.backward()
            optimizer.step()

            # Compute loss for critic with original image
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            train_robust_loss += robust_loss.item() * labels.size(0)
            train_robust_acc += torch.sum(robust_preds.max(1)[1] == labels).double()
            train_loss += loss.item() * labels.size(0)
            train_acc += torch.sum(preds.max(1)[1] == labels.data).double()
            train_n += labels.size(0)

            # Print step info
            if (step + 1) % params.log_step_pre == 0:
                print("Epoch [{}/{}] Step [{}/{}]: Avg Training loss: {:.4f} Avg Training Accuracy: {:.4%}"
                      " Avg Robust Training Loss: {:.4f} Avg Robust Training Accuracy: {:.4%}"
                      .format(epoch + 1,
                              num_epochs,
                              step + 1,
                              len(data_loader),
                              train_loss / train_n,
                              train_acc / train_n,
                              train_robust_loss / train_n,
                              train_robust_acc / train_n))

        time_elapsed = time.time() - start_time

        # Eval model on test set
        if (epoch + 1) % params.eval_step_pre == 0:
            eval_tgt(encoder, classifier, data_loader)

        # Save model parameters
        if (epoch + 1) % params.save_step_pre == 0:
            print('Epoch [{}/{}] completed in {:.0f}m {:.0f}s'.format(epoch + 1,
                                                                      num_epochs, time_elapsed // 60,
                                                                      time_elapsed % 60))
            root = params.adda_root if mode == 'ADDA' else params.model_root
            save_model(encoder, root, "{}-source-encoder-rb-{}.pt".format(mode, epoch + 1))
            save_model(classifier, root, "{}-source-classifier-rb-{}.pt".format(mode, epoch + 1))

    # Save final model
    root = params.adda_root if mode == 'ADDA' else params.model_root

    save_model(encoder, root, "{}-source-encoder-rb-final.pt".format(mode))
    save_model(classifier, root, "{}-source-classifier-rb-final.pt".format(mode))

    return encoder, classifier


def train_tgt_adda(src_encoder, tgt_encoder, classifier, critic, src_data_loader,
                   tgt_data_loader, tgt_data_loader_eval, robust=False):
    """Train adda encoder for target domain """

    # Step 1: Network Setup
    # Set train state for Dropout and BN layers
    src_encoder.eval()
    tgt_encoder.train()
    critic.train()

    # Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.learning_rate, weight_decay=params.weight_decay)
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.learning_rate, weight_decay=params.weight_decay)

    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    # Step 2 Train network
    for epoch in range(params.num_epochs):

        start_time = time.time()
        train_disc_loss, train_disc_acc, train_n = 0, 0, 0
        # Zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))

        for step, ((images_src, _), (images_tgt, _)) in data_zip:

            # 2.1 train discriminator with fixed src_encoder
            # Make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            # Prepare real and fake label (domain labels)
            domain_src = make_variable(torch.ones(images_src.size(0)).long())
            domain_tgt = make_variable(torch.zeros(images_tgt.size(0)).long())
            domain_concat = torch.cat((domain_src, domain_tgt), 0)

            if robust:
                # Attack images with domain labels
                delta_src = attack_pgd(src_encoder, critic, images_src, domain_src)
                delta_tgt = attack_pgd(tgt_encoder, critic, images_tgt, domain_tgt)

                robust_src = normalize(torch.clamp(images_src + delta_src[:images_src.size(0)],
                                                   min=params.lower_limit, max=params.upper_limit))
                robust_tgt = normalize(torch.clamp(images_tgt + delta_tgt[:images_tgt.size(0)],
                                                   min=params.lower_limit, max=params.upper_limit))

            # Zero gradients for optimizer for the discriminator
            optimizer_critic.zero_grad()

            # Extract and concat features
            feat_src = src_encoder(images_src) if not robust else src_encoder(robust_src)
            feat_tgt = tgt_encoder(images_tgt) if not robust else tgt_encoder(robust_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # Predict on discriminator
            preds_src_domain = critic(feat_src)
            preds_tgt_domain = critic(feat_tgt)
            # pred_concat = critic(feat_concat)

            # Compute loss for critic
            l1 = criterion(preds_src_domain, domain_src)
            l2 = criterion(preds_tgt_domain, domain_tgt)
            # loss_critic = criterion((pred_concat, domain_concat)
            loss_critic = l1 + l2
            train_disc_loss += loss_critic.item() * domain_concat.size(0)
            # train_disc_acc += torch.sum(pred_concat.max(1)[1] == domain_concat.data).double()
            train_disc_acc += torch.sum(preds_src_domain.max(1)[1] == domain_src.data).double()
            train_disc_acc += torch.sum(preds_tgt_domain.max(1)[1] == domain_tgt.data).double()
            train_n += domain_concat.size(0)
            loss_critic.backward()
            # Optimize critic
            optimizer_critic.step()

            # 2.2 Train target encoder
            # Zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # Prepare fake labels (flip labels)
            domain_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

            if robust:
                # Attack the target images with domain labels
                delta_tgt = attack_pgd(tgt_encoder, critic, images_tgt, domain_tgt)
                robust_tgt = normalize(torch.clamp(images_tgt + delta_tgt[:images_tgt.size(0)],
                                                    min=params.lower_limit, max=params.upper_limit))

            # Extract target features
            feat_tgt = tgt_encoder(images_tgt) if not robust else tgt_encoder(robust_tgt)

            # Predict on discriminator
            pred_tgt = critic(feat_tgt)
            # Compute loss for target encoder
            loss_tgt = criterion(pred_tgt, domain_tgt)
            loss_tgt.backward()

            # Optimize target encoder
            optimizer_tgt.step()

            # 2.3 Print step info
            if (step + 1) % params.log_step == 0:
                print("Epoch [{}/{}] Step [{}/{}]: "
                      "Avg Discriminator Loss: {:.4f} Avg Discriminator Accuracy: {:.4%}"
                      .format(epoch + 1, params.num_epochs, step + 1, len_data_loader, train_disc_loss / train_n,
                              train_disc_acc / train_n))

        time_elapsed = time.time() - start_time

        # Eval model
        if (epoch + 1) % params.eval_step == 0:
            if not robust:
                eval_tgt(tgt_encoder, classifier, tgt_data_loader_eval)
            else:
                eval_tgt_robust(tgt_encoder, classifier, critic, tgt_data_loader_eval)

        # 2.4 Save model parameters #
        if (epoch + 1) % params.save_step == 0:
            print('Epoch [{}/{}] completec in {:.0f}m {:.0f}s'.format(epoch + 1,
                                                                      params.num_epochs, time_elapsed // 60,
                                                                      time_elapsed % 60))
            filename = "ADDA-critic-{}.pt".format(epoch + 1) if not robust \
                else "ADDA-critic-rb-{}.pt".format(epoch + 1)
            save_model(critic, params.adda_root, filename)

            filename = "ADDA-target-encoder-{}.pt".format(epoch + 1) if not robust \
                else "ADDA-target-encoder-rb-{}.pt".format(epoch + 1)
            save_model(tgt_encoder, params.adda_root, filename)



    filename = "ADDA-critic-final.pt" if not robust else "ADDA-critic-rb-final.pt"
    save_model(critic, params.adda_root, filename)

    filename = "ADDA-target-encoder-final.pt" if not robust else "ADDA-target-encoder-rb-final.pt"
    save_model(tgt_encoder, params.adda_root, filename)

    return tgt_encoder


def train_critic_wdgrl(encoder, critic, optimizer, images_src, images_tgt):
    """Train domain critic for wdgrl."""

    set_requires_grad(critic, requires_grad=True)
    set_requires_grad(encoder, requires_grad=False)
    critic_loss, train_n = 0, 0

    with torch.no_grad():
        h_s = encoder(images_src).data.view(images_src.size(0), -1)
        h_t = encoder(images_tgt).data.view(images_src.size(0), -1)

    for i in range(params.num_times_critic):
        # Zero gradients for optimizer
        optimizer.zero_grad()

        # Computer gradient penalty
        gp = gradient_penalty(critic, h_s, h_t)
        critic_s = critic(h_s)
        critic_t = critic(h_t)
        wasserstein_distance = critic_s.mean() - (1 + params.beta_ratio) * critic_t.mean()

        # Compute cost for critic
        critic_cost = -wasserstein_distance + params.wd_gp * gp
        critic_loss += critic_cost.item()

        # Optimize critic
        critic_cost.backward()
        optimizer.step()

    set_requires_grad(critic, requires_grad=False)
    set_requires_grad(encoder, requires_grad=True)

    return critic_loss / params.num_times_critic


def train_tgt_wdgrl(encoder, classifier, critic, src_data_loader, tgt_data_loader, tgt_data_loader_eval, robust=False):
    """Train encoder encoder for wdgrl """

    # Set state
    encoder.train()
    classifier.train()
    critic.train()

    # Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=params.learning_rate, weight_decay=params.weight_decay)
    critic_optimizer = optim.Adam(critic.parameters(),
                                  lr=params.learning_rate, weight_decay=params.weight_decay)
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    # Step 2 Train network
    for epoch in range(params.num_epochs):
        train_acc, train_loss, total_n = 0, 0, 0
        start_time = time.time()
        # Zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, labels_src), (images_tgt, _)) in data_zip:

            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)
            labels_src = make_variable(labels_src)

            if robust:
                # PDG attack on the source image
                delta_src = attack_pgd(encoder, classifier, images_src, labels_src)

                robust_src = normalize(torch.clamp(images_src + delta_src[:images_src.size(0)],
                                                   min=params.lower_limit, max=params.upper_limit))

            if robust:
                critic_loss = train_critic_wdgrl(encoder, critic, critic_optimizer, robust_src, images_tgt)
            else:
                critic_loss = train_critic_wdgrl(encoder, critic, critic_optimizer, images_src, images_tgt)

            feat_src = encoder(images_src) if not robust else encoder(robust_src)
            feat_tgt = encoder(images_tgt)

            preds_src = classifier(feat_src)
            clf_loss = criterion(preds_src, labels_src)
            wasserstein_distance = critic(feat_src).mean() - (1 + params.beta_ratio) * critic(feat_tgt).mean()

            loss = clf_loss + params.wd_clf * wasserstein_distance
            train_loss += loss.item() * labels_src.size(0) + critic_loss
            total_n += labels_src.size(0)
            train_acc += torch.sum(preds_src.max(1)[1] == labels_src.data).double()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]: Avg Training loss: {:.4f} Ave Training Accuracy {:.4%}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len_data_loader,
                              train_loss / total_n,
                              train_acc / total_n))

        time_elapsed = time.time() - start_time

        # Eval model
        if (epoch + 1) % params.eval_step == 0:
            eval_tgt_robust(encoder, classifier, tgt_data_loader_eval)

        # 2.4 Save model parameters #
        if ((epoch + 1) % params.save_step == 0):
            print('Epoch [{}/{}] completed in {:.0f}m {:.0f}s'.format(epoch + 1,
                                                                      params.num_epochs, time_elapsed // 60,
                                                                      time_elapsed % 60))
            filename = "WDGRL-encoder-{}.pt".format(epoch + 1) if not robust \
                else "WDGRL-encoder-rb-{}.pt".format(epoch + 1)
            save_model(encoder, params.wdgrl_root, filename)

            filename = "WDGRL-classifier-{}.pt".format(epoch + 1) if not robust \
                else "WDGRL-classifier-rb-{}.pt".format(epoch + 1)
            save_model(classifier, params.wdgrl_root, filename)

    filename = "WDGRL-classifier-final.pt" if not robust else "WDGRL-classifier-rb-final.pt"
    save_model(classifier, params.wdgrl_root, filename)

    filename = "WDGRL-encoder-final.pt" if not robust else "WDGRL-encoder-rb-final.pt"
    save_model(encoder, params.wdgrl_root, filename)

    return encoder, classifier


def train_dann(encoder, classifier, critic, src_data_loader, tgt_data_loader, tgt_data_loader_eval, robust=True):
    """Train encoder for DANN """

    # 1. Network Setup
    encoder.train()
    classifier.train()
    critic.train()

    # Set up optimizer and criterion
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()) + list(critic.parameters()),
                           lr=params.learning_rate, weight_decay=params.weight_decay)

    criterion = nn.CrossEntropyLoss()

    # 2. Train network
    for epoch in range(params.num_epochs):

        start_time = time.time()
        total_loss, train_n = 0, 0
        train_clf_loss, train_clf_acc, train_clf_n = 0, 0, 0
        train_domain_loss, train_domain_acc, train_domain_n = 0, 0, 0

        # Zip source and target data pair
        len_data_loader = min(len(src_data_loader), len(tgt_data_loader))
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))

        for step, ((images_src, labels_src), (images_tgt, _)) in data_zip:

            p = float(step + epoch * len_data_loader) / \
                params.num_epochs / len_data_loader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # Update lr
            # update_lr(optimizer, lr_scheduler(p))

            # Make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)
            labels_src = make_variable(labels_src)
            images_concat = torch.cat((images_src, images_tgt), 0)

            # Prepare real and fake label (domain labels)
            domain_src = make_variable(torch.ones(images_src.size(0)).long())
            domain_tgt = make_variable(torch.zeros(images_tgt.size(0)).long())
            domain_concat = torch.cat((domain_src, domain_tgt), 0)

            # Zero gradients for optimizer
            optimizer.zero_grad()

            if robust:
                delta_src = attack_pgd(encoder, classifier, images_src, labels_src)
                delta_domain = attack_pgd(encoder, critic, images_concat, domain_concat)

                robust_src = normalize(torch.clamp(images_src + delta_src[:images_src.size(0)],
                                                   min=params.lower_limit, max=params.upper_limit))
                robust_domain = normalize(torch.clamp(images_concat + delta_domain[:images_concat.size(0)],
                                                      min=params.lower_limit, max=params.upper_limit))

            # Train on source domain
            feats = encoder(images_src) if not robust else encoder(robust_src)
            preds_src = classifier(feats)
            feats = encoder(images_concat) if not robust else encoder(robust_domain)

            preds_domain = critic(feats, alpha=alpha)

            # Computer loss for source classification and domain classification
            loss_src = criterion(preds_src, labels_src)
            loss_domain = criterion(preds_domain, domain_concat)

            loss = loss_src + loss_domain

            train_clf_n += preds_src.size(0)
            train_domain_n += preds_domain.size(0)
            train_n += train_clf_n + train_domain_n

            total_loss += loss.item() * (preds_src.size(0) + preds_domain.size(0))
            train_clf_loss += loss_src.item() * preds_src.size(0)
            train_domain_loss += loss_domain.item()

            train_domain_acc += torch.sum(preds_domain.max(1)[1] == domain_concat.data).double()
            train_clf_acc += torch.sum(preds_src.max(1)[1] == labels_src.data).double()

            # Optimize model
            loss.backward()
            optimizer.step()

            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}] Avg total loss: {:.4f} Avg Domain Loss: {:.4f}"
                      " Avg Domain Accuracy: {:.4%} Avg Classification Loss: {:4f} "
                      "Avg Classification Accuracy: {:.4%}".format(epoch + 1,
                                                                   params.num_epochs,
                                                                   step + 1,
                                                                   len_data_loader,
                                                                   total_loss / train_n,
                                                                   train_domain_loss / train_domain_n,
                                                                   train_domain_acc / train_domain_n,
                                                                   train_clf_loss / train_clf_n,
                                                                   train_clf_acc / train_clf_n))

        time_elapsed = start_time - time.time()

        # Eval model
        if (epoch + 1) % params.eval_step == 0:
            eval_tgt_robust(encoder, classifier, tgt_data_loader_eval)

        # Save model parameters
        if (epoch + 1) % params.save_step == 0:
            print('Epoch [{}/{}] completed in {:.0f}m {:.0f}s'.format(epoch + 1,
                                                                      params.num_epochs,
                                                                      time_elapsed // 60,
                                                                      time_elapsed % 60))
            filename = "DANN-encoder-{}.pt".format(epoch + 1) if not robust \
                else "DANN-encoder-rb-{}.pt".format(epoch + 1)
            save_model(encoder, params.dann_root, filename)
            filename = "DANN-classifier-{}.pt".format(epoch + 1) if not robust \
                else "DANN-classifier-rb-{}.pt".format(epoch + 1)
            save_model(classifier, params.dann_root, filename)
            filename = "DANN-critic-{}.pt".format(epoch + 1) if not robust \
                else "DANN-critic-rb-{}.pt".format(epoch + 1)
            save_model(critic, params.dann_root, filename)

    # Save final model
    filename = "DANN-encoder-final.pt" if not robust else "DANN-encoder-rb-final.pt"
    save_model(encoder, params.dann_root, filename)
    filename = "DANN-classifier-final.pt" if not robust else "DANN-classifier-rb-final.pt"
    save_model(classifier, params.dann_root, filename)
    filename = "DANN-critic-final.pt" if not robust else "DANN-critic-rb-final.pt"
    save_model(critic, params.dann_root, filename)

    return encoder, classifier, critic


def train_alda(encoder, classifier, critic, src_data_loader, tgt_data_loader, tgt_data_loader_eval, robust=True):
    """Train encoder for DANN """

    # 1. Network Setup
    encoder.train()
    classifier.train()
    critic.train()

    # Set up optimizer and criterion
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=params.learning_rate, weight_decay=params.weight_decay)
    optimizer_critic = optim.Adam(critic.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # 2. Train network
    for epoch in range(params.num_epochs):

        start_time = time.time()
        total_loss, train_n = 0, 0
        loss_target_value = 0
        train_clf_loss, train_clf_acc, train_clf_n = 0, 0, 0

        # Zip source and target data pair
        len_data_loader = min(len(src_data_loader), len(tgt_data_loader))
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))

        for step, ((images_src, labels_src), (images_tgt, _)) in data_zip:

            p = float(step + epoch * len_data_loader) / \
                params.num_epochs / len_data_loader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # Make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)
            labels_src = make_variable(labels_src)
            images_concat = torch.cat((images_src, images_tgt), 0)

            # Prepare real and fake label (domain labels)
            domain_src = make_variable(torch.ones(images_src.size(0)).long())
            domain_tgt = make_variable(torch.zeros(images_tgt.size(0)).long())
            domain_concat = torch.cat((domain_src, domain_tgt), 0)

            # Zero gradients for optimizer
            optimizer.zero_grad()

            if robust:
                delta_src = attack_pgd(encoder, classifier, images_src, labels_src)
                robust_src = normalize(torch.clamp(images_src + delta_src[:images_src.size(0)],
                                                   min=params.lower_limit, max=params.upper_limit))

            # Train on source domain
            feats = encoder(images_src) if not robust else encoder(robust_src)
            preds_src = classifier(feats)
            preds_tgt = classifier(encoder(images_tgt))
            feats = encoder(images_concat)

            preds_critic = critic(feats, alpha=alpha)

            loss_adv, loss_reg, loss_correct = alda_loss(preds_critic, labels_src, preds_src, preds_tgt, threshold=0.6)

            # Computer loss for source classification and domain classification
            transfer_loss = loss_adv + loss_correct if epoch > 2 else 0

            # Loss_reg is only backward to the discrinminator
            set_requires_grad(encoder, requires_grad=False)
            set_requires_grad(classifier, requires_grad=False)
            loss_reg.backward(retain_graph=True)
            set_requires_grad(encoder, requires_grad=True)
            set_requires_grad(classifier, requires_grad=True)

            loss_target_value += transfer_loss.item() * (preds_src.size(0) + preds_tgt.size(0))
            train_n += preds_src.size(0) + preds_tgt.size(0)

            loss = preds_src + transfer_loss  # Loss_func.Square(softmax_output) + transfer_loss
            total_loss += loss.item() * (preds_src.size(0) + preds_tgt.size(0))

            train_clf_n += preds_src.size(0)
            train_clf_loss += loss_correct.item() * preds_src.size(0)
            train_clf_acc += torch.sum(preds_src.max(1)[1] == labels_src.data).double()

            # Optimize model
            loss.backward()
            optimizer.step()
            if epoch > 2:
                optimizer_critic.step()

            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}] Avg total loss: {:.4f} Avg Transfer Loss: {:.4f}"
                      "  Avg Classification Loss: {:4f} "
                      "Avg Classification Accuracy: {:.4%}".format(epoch + 1,
                                                                   params.num_epochs,
                                                                   step + 1,
                                                                   len_data_loader,
                                                                   total_loss / train_n,
                                                                   loss_target_value / train_n,
                                                                   train_clf_loss / train_clf_n,
                                                                   train_clf_acc / train_clf_n))

        time_elapsed = start_time - time.time()

        # Eval model
        if (epoch + 1) % params.eval_step == 0:
            eval_tgt_robust(encoder, classifier, tgt_data_loader_eval)

        # Save model parameters
        if (epoch + 1) % params.save_step == 0:
            print('Epoch [{}/{}] completed in {:.0f}m {:.0f}s'.format(epoch + 1,
                                                                      params.num_epochs,
                                                                      time_elapsed // 60,
                                                                      time_elapsed % 60))
            filename = "ALDA-encoder-{}.pt".format(epoch + 1) if not robust \
                else "ALDA-encoder-rb-{}.pt".format(epoch + 1)
            save_model(encoder, params.dann_root, filename)
            filename = "ALDA-classifier-{}.pt".format(epoch + 1) if not robust \
                else "ALDA-classifier-rb-{}.pt".format(epoch + 1)
            save_model(classifier, params.dann_root, filename)
            filename = "ALDA-critic-{}.pt".format(epoch + 1) if not robust \
                else "ALDA-critic-rb-{}.pt".format(epoch + 1)
            save_model(critic, params.dann_root, filename)

    # Save final model
    filename = "ALDA-encoder-final.pt" if not robust else "ALDA-encoder-rb-final.pt"
    save_model(encoder, params.dann_root, filename)
    filename = "ALDA-classifier-final.pt" if not robust else "ALDA-classifier-rb-final.pt"
    save_model(classifier, params.dann_root, filename)
    filename = "ALDA-critic-final.pt" if not robust else "ALDA-critic-rb-final.pt"
    save_model(critic, params.dann_root, filename)

    return encoder, classifier, critic
