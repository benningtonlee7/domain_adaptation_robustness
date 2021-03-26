import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import params
from utils.utils import make_variable, save_model, normalize, lr_schedule, gradient_penalty, update_lr
from core.eval import eval_tgt, eval_tgt_robust
from core.pgd import attack_pgd
from models.models import ReverseLayerF
import numpy as np

def train_src_adda(encoder, classifier, data_loader):
    """Train classifier for source domain for ADDA"""
    # Step 1: Network setup
    # Set train state for both Dropout and BN layers
    encoder.train()
    classifier.train()

    # Set up optimizer and criterion
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    # Step 2: Pretrain the source model
    for epoch in range(params.num_epochs_pre):
        train_acc, train_loss, train_n = 0, 0, 0
        start_time = time.time()

        for step, (images, labels) in enumerate(data_loader):

            # Make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

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
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: Training Loss: {:.4f} Training Accuracy: {:.4%}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              train_loss / train_n,
                              train_acc / train_n))
        time_elapsed = time.time() - start_time

        # Eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_tgt(encoder, classifier, data_loader)

        # Save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            print('Epoch [{}/{}] completed in {:.0f}m {:.0f}s'.format(epoch + 1,
                                                                     params.num_epochs, time_elapsed // 60,
                                                                     time_elapsed % 60))
            save_model(encoder, params.adda_root, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(classifier, params.adda_root, "ADDA-source-classifier-{}.pt".format(epoch + 1))

    # Save final model
    save_model(encoder, params.adda_root, "ADDA-source-encoder-final.pt")
    save_model(classifier, params.adda_root, "ADDA-source-classifier-final.pt")

    return encoder, classifier


def train_src_robust(encoder, classifier, data_loader):
    """Train classifier for source domain with robust training for ADDA"""

    # Step 1: Network setup
    # Set train state for both Dropout and BN layers
    encoder.train()
    classifier.train()

    # Set up optimizer and criterion
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    # Step 2: Pretrain the source model
    for epoch in range(params.num_epochs_pre):

        # Init accuracy and loss
        start_time = time.time()
        train_loss, train_acc, train_n = 0, 0, 0
        train_robust_loss, train_robust_acc = 0, 0

        for step, (images, labels) in enumerate(data_loader):

            # Make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # Zero gradients for optimizer
            optimizer.zero_grad()

            # Update lr - using piecewise lr scheduler from "Overfitting in adversarially robust deep learning"
            # lr = lr_schedule(epoch + 1, params.num_epochs)
            # update_lr(optimizer, lr)

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
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: Avg Training loss: {:.4f} Avg Training Accuracy: {:.4%}"
                      " Avg Robust Training Loss: {:.4f} Avg Robust Training Accuracy: {:.4%}".format(epoch + 1,
                                                                                     params.num_epochs_pre,
                                                                                     step + 1,
                                                                                     len(data_loader),
                                                                                     train_loss/train_n,
                                                                                     train_acc/train_n,
                                                                                     train_robust_loss/train_n,
                                                                                     train_robust_acc/train_n))
        time_elapsed = time.time() - start_time

        # Eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_tgt_robust(encoder, classifier, data_loader)

        # Save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            print('Epoch [{}/{}] completed in {:.0f}m {:.0f}s'.format(epoch + 1,
                                                                     params.num_epochs, time_elapsed // 60,
                                                                     time_elapsed % 60))
            save_model(encoder, params.adda_root, "ADDA-source-encoder-rb-{}.pt".format(epoch + 1))
            save_model(classifier, params.adda_root, "ADDA-source-classifier-rb-{}.pt".format(epoch + 1))

    # Save final model
    save_model(encoder, params.adda_root, "ADDA-source-encoder-rb-final.pt")
    save_model(classifier, params.adda_root, "ADDA-source-classifier-rb-final.pt")

    return encoder, classifier


def train_tgt_adda(src_encoder, tgt_encoder, critic, src_data_loader, tgt_data_loader, robust=False):
    """Train adda encoder for target domain for ADDA"""

    # Step 1:  Network Setup
    # Set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.c_learning_rate,
                                  betas=(params.beta1, params.beta2))
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
            label_src = make_variable(torch.ones(images_src.size(0)).long())
            label_tgt = make_variable(torch.zeros(images_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            if robust:
                # Update lr - using piecewise lr scheduler from "Overfitting in adversarially robust deep learning"
                # lr = lr_schedule(epoch + 1, params.num_epochs)
                # update_lr(optimizer_tgt, lr)
                # update_lr(optimizer_critic, lr)
                # Attack imgs with domain labels
                delta_src = attack_pgd(src_encoder, critic, images_src, label_src)
                delta_tgt = attack_pgd(tgt_encoder, critic, images_tgt, label_tgt)

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
            pred_concat = critic(feat_concat.detach())

            # Compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            train_disc_loss += loss_critic.item() * label_concat.size(0)
            train_disc_acc += torch.sum(pred_concat.max(1)[1] == label_concat.data).double()
            train_n += label_concat.size(0)
            loss_critic.backward()
            # Optimize critic
            optimizer_critic.step()

            # 2.2 Train target encoder
            # Zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # Prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

            if robust:
                # Attack the target images with domain labels
                delta_tgt = attack_pgd(tgt_encoder, critic, images_tgt, label_tgt)
                images_tgt = normalize(torch.clamp(images_tgt + delta_tgt[:images_tgt.size(0)],
                                                   min=params.lower_limit, max=params.upper_limit))

            # Extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # Predict on discriminator
            pred_tgt = critic(feat_tgt)

            # Compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # Optimize target encoder
            optimizer_tgt.step()

            # 2.3 Print step info
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "Avg Discriminator Loss: {:.4f} Avg Discriminator Accuracy: {:.4%}"
                      .format(epoch + 1, params.num_epochs, step + 1, len_data_loader, train_disc_loss / train_n,
                              train_disc_acc / train_n))

        time_elapsed = time.time() - start_time

        # 2.4 Save model parameters #
        if ((epoch + 1) % params.save_step == 0):
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


def train_critic_wdgrl(encoder, critic, src_data_loader, tgt_data_loader):
    """Train domain critic for wdgrl."""
    # Step 1: Network setup
    # Set state
    critic.train()
    encoder.eval()

    # Init optimizer and criterion
    optimizer = optim.Adam(critic.parameters(),
                           lr=params.c_learning_rate,
                           betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    for epoch in range(params.num_epochs_wdgrl_pre):
        critic_loss, train_n = 0, 0
        start_time = time.time()

        # Zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))

        for step, ((images_src, _), (images_tgt, _)) in data_zip:

            if images_src.size(0) > images_tgt.size(0):
                images_src = images_src.narrow(0, 0, images_tgt.size(0))
            elif images_src.size(0) < images_tgt.size(0):
                images_tgt = images_tgt.narrow(0, 0, images_src.size(0))

            # Make images and labels variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            # Zero gradients for optimizer
            optimizer.zero_grad()

            h_s = encoder(images_src).data.view(images_src.shape[0], -1)
            h_t = encoder(images_tgt).data.view(images_tgt.shape[0], -1)

            # Computer gradient penalty
            gp = gradient_penalty(critic, h_s, h_t)
            critic_s = critic(h_s)
            critic_t = critic(h_t)
            wasserstein_distance = critic_s.mean() - critic_t.mean()

            # Compute cost for critic
            critic_cost = -wasserstein_distance + params.wd_clf * gp
            critic_loss += critic_cost.item() * images_tgt.size(0)
            train_n += images_tgt.size(0)

            # Optimize critic
            critic_cost.backward()
            optimizer.step()

            # Print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: Training Critic Cost: {:.4f}"
                      .format(epoch + 1,
                              params.num_epochs_wdgrl_pre,
                              step + 1,
                              len_data_loader,
                              critic_loss / train_n))

        time_elapsed = time.time() - start_time

        # Save critic parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            print('Epoch [{}/{}] completed in {:.0f}m {:.0f}s'.format(epoch + 1,
                                                                     params.num_epochs_wdgrl_pre, time_elapsed // 60,
                                                                     time_elapsed % 60))
            save_model(critic, params.wdgrl_root, "WDGRL-critic-{}.pt".format(epoch + 1))

    # Save final model
    save_model(critic, params.wdgrl_root, "WDGRL-critic-final.pt")

    return critic


def train_tgt_wdgrl(encoder, classifier, critic, src_data_loader, tgt_data_loader, robust=False):
    # Set state
    encoder.train()
    critic.eval()

    # Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=params.c_learning_rate,
                           betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    # Step 2 Train network
    for epoch in range(params.num_epochs):
        train_acc, train_loss, train_n, total_n = 0, 0, 0, 0
        start_time = time.time()
        # Zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, labels_src), (images_tgt, _)) in data_zip:

            if images_src.size(0) > images_tgt.size(0):
                images_src = images_src.narrow(0, 0, images_tgt.size(0))
                labels_src = labels_src.narrow(0, 0, images_tgt.size(0))
            elif images_src.size(0) < images_tgt.size(0):
                images_tgt = images_tgt.narrow(0, 0, images_src.size(0))

            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)
            labels_src = make_variable(labels_src.squeeze_())

            if robust:
                # Update lr - using piecewise lr scheduler from "Overfitting in adversarially robust deep learning"
                # lr = lr_schedule(epoch + 1, params.num_epochs)
                # update_lr(optimizer, lr)
                # PDG attack on the source image
                delta_src = attack_pgd(encoder, classifier, images_src, labels_src)

                robust_src = normalize(torch.clamp(images_src + delta_src[:images_src.size(0)],
                                                   min=params.lower_limit, max=params.upper_limit))

            feat_src = encoder(images_src) if not robust else encoder(robust_src)
            feat_tgt = encoder(images_tgt)

            preds_src = classifier(feat_src)
            clf_loss = criterion(preds_src, labels_src)
            wasserstein_distance = critic(feat_src).mean() - critic(feat_tgt).mean()

            loss = clf_loss + params.wd_clf * wasserstein_distance
            train_loss += loss.item() * (labels_src.size(0) + images_tgt.size(0))
            total_n += (labels_src.size(0) + images_tgt.size(0))
            train_n += labels_src.size(0)
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
                              train_loss/total_n,
                              train_acc/train_n))

        time_elapsed = time.time() - start_time

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


def train_revgard(encoder, classifier, critic, src_data_loader, tgt_data_loader, robust=True):
    # Step 1: Network setup
    # Set train state for both Dropout and BN layers
    encoder.train()
    classifier.train()

    # Set up optimizer and criterion
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()) + list(critic.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))

    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))
    clf_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    # Step 2: Train model
    for epoch in range(params.num_epochs):

        # Init accuracy and loss
        start_time = time.time()
        train_clf_loss, train_clf_acc, train_n = 0, 0, 0
        train_domain_loss, train_domain_acc, train_domain_n = 0, 0, 0
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))

        for step, ((images_src, labels_src), (images_tgt, _)) in data_zip:

            # Make images and labels variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)
            labels_src = make_variable(labels_src.squeeze_())
            images_concat = torch.cat((images_src, images_tgt), 0)

            # Zero gradients for optimizer
            optimizer.zero_grad()

            # Prepare real and fake label (domain labels)
            label_src = make_variable(torch.ones(images_src.size(0)).long())
            label_tgt = make_variable(torch.zeros(images_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            if robust:
                delta_concat = attack_pgd(encoder, critic, images_concat, label_concat)
                delta_src = attack_pgd(encoder, classifier, images_src, labels_src)
                robust_concat = normalize(torch.clamp(images_concat + delta_concat[:images_concat.size(0)],
                                                      min=params.lower_limit, max=params.upper_limit))
                robust_src = normalize(torch.clamp(images_src + delta_src[:images_src.size(0)],
                                                   min=params.lower_limit, max=params.upper_limit))

            preds_domain = critic(encoder(images_concat)) if not robust else critic(encoder(robust_concat))
            preds_clf = classifier(encoder(images_src)) if not robust else classifier(encoder(robust_src))

            domain_loss = domain_criterion(preds_domain.max(1)[1].float(), label_concat.float())
            clf_loss = clf_criterion(preds_clf, labels_src)
            loss = domain_loss + clf_loss

            # Optimize model
            loss.backward()
            optimizer.step()

            train_clf_loss += clf_loss.item() * images_src.size(0)
            train_domain_loss += domain_loss.item() * images_concat.size(0)
            train_domain_acc += torch.sum(preds_domain.max(1)[1] == label_concat.data).double()
            train_clf_acc += torch.sum(preds_clf.max(1)[1] == labels_src.data).double()
            train_n += labels_src.size(0)
            train_domain_n += label_concat.size(0)
            # Print step info
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]: Avg Classifier loss: {:.4f} Avg Classifier Accuracy: {:.4%}"
                      " Avg Domain Loss: {:.4f} Avg Domain Accuracy: {:.4%}".format(epoch + 1,
                                                                                   params.num_epochs,
                                                                                   step + 1,
                                                                                   len_data_loader,
                                                                                   train_clf_loss/train_n,
                                                                                   train_clf_acc/train_n,
                                                                                   train_domain_loss/train_domain_n,
                                                                                   train_domain_acc/train_domain_n))
        time_elapsed = time.time() - start_time

        # Save model parameters
        if ((epoch + 1) % params.save_step == 0):
            print('Epoch [{}/{}] completed in {:.0f}m {:.0f}s'.format(epoch + 1,
                                                                      params.num_epochs,
                                                                      time_elapsed // 60,
                                                                      time_elapsed % 60))
            filename = "REVGRAD-encoder-{}.pt".format(epoch + 1) if not robust \
                else "REVGRAD-encoder-rb-{}.pt".format(epoch + 1)
            save_model(encoder, params.revgard_root, filename)

            filename = "REVGRAD-classifier-{}.pt".format(epoch + 1) if not robust \
                else "REVGRAD-classifier-rb-{}.pt".format(epoch + 1)
            save_model(classifier, params.revgard_root, filename)

            filename = "REVGRAD-critic-{}.pt".format(epoch + 1) if not robust \
                else "REVGRAD-critic-rb-{}.pt".format(epoch + 1)
            save_model(critic, params.revgard_root, filename)

    # Save final model
    filename = "REVGRAD-encoder-final.pt" if not robust else "REVGRAD-encoder-rb-final.pt"
    save_model(encoder, params.revgard_root, filename)
    filename = "REVGRAD-classifier-final.pt" if not robust else "REVGRAD-classifier-rb-final.pt"
    save_model(classifier, params.revgard_root, filename)
    filename = "REVGRAD-critic-final.pt" if not robust else "REVGRAD-critic-rb-final.pt"
    save_model(critic, params.revgard_root, filename)

    return encoder, classifier, critic

def train_dann(encoder, classifier, critic, src_data_loader, tgt_data_loader, tgt_data_loader_eval, robust=True):

    # 1. Network Setup
    encoder.train()
    classifier.train()
    critic.train()
    # Set up optimizer and criterion
    optimizer = optim.SGD(list(encoder.parameters()) + list(classifier.parameters()) + list(critic.parameters()),
                          lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # 2. Train network
    global_step = 0
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

            # Make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)
            labels_src = make_variable(labels_src.squeeze_())

            # Prepare real and fake label (domain labels)
            label_src = make_variable(torch.ones(images_src.size(0)).long())
            label_tgt = make_variable(torch.zeros(images_tgt.size(0)).long())

            # Zero gradients for optimizer
            optimizer.zero_grad()

            # Train on source domain
            feats = encoder(images_src)
            reversed_feats = ReverseLayerF.apply(feats.view(-1, 50 * 4 * 4), alpha)
            preds_src = classifier(feats)
            preds_src_domain = critic(reversed_feats)
            loss_src = criterion(preds_src, labels_src)
            loss_src_domain = criterion(preds_src_domain, label_src)

            # Train on target domain
            preds_tgt_domain = critic(encoder(images_tgt))
            loss_tgt_domain = criterion(preds_tgt_domain, label_tgt)

            loss = loss_src + loss_src_domain + loss_tgt_domain

            train_clf_n += preds_src.size(0)
            train_domain_n += preds_src_domain.size(0) + preds_tgt_domain.size(0)
            train_n += train_clf_n + train_domain_n

            total_loss += loss.item() * (preds_src.size(0) + preds_src_domain.size(0) + preds_tgt_domain.size(0))
            train_clf_loss += loss_src.item() * preds_src.size(0)
            train_domain_loss += loss_src_domain.item() + loss_tgt_domain.item()
            train_domain_acc += torch.sum(preds_src_domain.max(1)[1] == label_src.data).double() + \
                                    torch.sum(preds_tgt_domain.max(1)[1] == label_tgt.data).double()
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
                                                           total_loss /train_n,
                                                           train_domain_loss / train_domain_n,
                                                           train_domain_acc/ train_domain_n,
                                                           train_clf_loss/train_clf_n,
                                                           train_clf_acc/train_clf_n))
        time_elapsed = start_time - time.time()
        # Eval model
        if ((epoch + 1) % params.eval_step == 0):
            eval_tgt_robust(encoder, classifier, tgt_data_loader_eval)

        # Save model parameters
        if ((epoch + 1) % params.save_step == 0):
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

