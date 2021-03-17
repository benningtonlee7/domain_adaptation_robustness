"""Pre-train encoder and classifier for source dataset."""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import params
from utils.utils import make_variable, save_model, normalize, lr_schedule, gradient_penalty, update_lr
from core.eval import eval_src, eval_tgt, eval_src_robust
from core.pgd import attack_pgd


def train_src_adda(encoder, classifier, data_loader):
    """Train classifier for source domain."""
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
    # Alternative
    # criterion = nn.BCEWithLogitsLoss()

    # Step 2: Pretrain the source model
    for epoch in range(params.num_epochs_pre):
        train_acc, train_n = 0, 0
        for step, (images, labels) in enumerate(data_loader):

            # Make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # Zero gradients for optimizer
            optimizer.zero_grad()
            # Compute loss for critic
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            # Optimize source classifier
            loss.backward()
            optimizer.step()

            train_acc += torch.sum(preds == labels.data)

            # Print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}] Training Loss: {:.4f} Training Accuracy: {:.4%}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.item(),
                              train_acc/train_n))

        # Eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src(encoder, classifier, data_loader)

        # Save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, "{}-source-encoder-{}.pt".format("ADDA", epoch + 1))
            save_model(
                classifier, "{}-source-classifier-{}.pt".format("ADDA", epoch + 1))

    # Save final model
    save_model(encoder, "{}-source-encoder-final.pt".format("ADDA"))
    save_model(classifier, "{}-source-classifier-final.pt".format("ADDA"))

    return encoder, classifier

def train_src_robust(encoder, classifier, data_loader, model="ADDA"):
    """Train classifier for source domain with robust training"""

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
        train_loss, train_acc = 0, 0
        train_robust_loss, train_robust_acc = 0, 0
        train_n = 0

        for step, (images, labels) in enumerate(data_loader):

            # Make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # Zero gradients for optimizer
            optimizer.zero_grad()

            # Update lr - using piecewise lr scheduler from "Overfitting in adversarially robust deep learning"
            lr = lr_schedule(epoch + 1, params.num_epochs)
            update_lr(optimizer, lr)

            delta = attack_pgd(encoder, images, labels)

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
            train_robust_acc += torch.sum(robust_preds.max(1)[1] == labels)
            train_loss += loss.item() * labels.size(0)
            train_acc += torch.sum(preds.max(1)[1] == labels.data)
            train_n += labels.size(0)

            # Print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}] Avg Training loss: {:4f} Avg Training Accuracy: {:.4%}"
                      " Avg Robust Training Loss: {:4f} Avg Robust Training Accuracy: {:4%}".format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              train_loss/train_n * 1.0,
                              train_acc.double()/train_n,
                              train_robust_loss/train_n,
                              train_robust_acc.double()/train_n))
        time_elapsed = time.time() - start_time

        # Eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src_robust(encoder, classifier, data_loader)

        # Save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            print('Epoch [{}/{}] complete in {:.0f}m {:.0f}s'.format(epoch + 1,
                    params.num_epochs, time_elapsed // 60, time_elapsed % 60))
            save_model(encoder, "{}-source-encoder-robust-{}.pt".format(model, epoch + 1))
            save_model(classifier, "{}-source-classifier-robust-{}.pt".format(model, epoch + 1))

    # Save final model
    save_model(encoder, "{}-source-encoder-final-robust.pt".format(model))
    save_model(classifier, "{}-source-classifier-final-robust.pt".format(model))

    return encoder, classifier

def train_tgt_adda(src_encoder, tgt_encoder, critic, src_data_loader, tgt_data_loader, robust=False):
    """Train adda encoder for target domain."""

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
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    # Step 2 Train network
    for epoch in range(params.num_epochs):

        start_time = time.time()
        # Zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, label_src), (images_tgt, _)) in data_zip:

            # 2.1 train discriminator with fixed src_encoder
            # Make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            # Update lr - using piecewise lr scheduler from "Overfitting in adversarially robust deep learning"
            lr = lr_schedule(epoch + 1, params.num_epochs)
            update_lr(optimizer_tgt, lr)
            update_lr(optimizer_critic, lr)

            if robust:
                delta = attack_pgd(src_encoder, images_src, label_src)

                robust_imgs = normalize(torch.clamp(images_src + delta[:images_src.size(0)],
                                                      min=params.lower_limit, max=params.upper_limit))

            # Zero gradients for optimizer for the discriminator
            optimizer_critic.zero_grad()

            # Extract and concat features
            feat_src = src_encoder(robust_imgs)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # Predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # Prepare real and fake label
            label_src = make_variable(torch.ones(feat_src.size(0)).long())
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # Compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # Optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            # 2.2 Train target encoder #
            # Zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # Prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

            if robust:
                delta = attack_pgd(src_encoder, images_tgt, label_tgt)
                images_tgt = normalize(torch.clamp(images_tgt + delta[:images_src.size(0)],
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
                      "Avg Discriminator Loss: {:4f} Avg Tgt Discriminator loss: {:4f} Avg accuracy: {:.4%}"
                      .format(epoch + 1,params.num_epochs, step + 1, len_data_loader, loss_critic.item(),
                              loss_tgt.item(), acc.item()))

        time_elapsed = time.time() - start_time

        # 2.4 Save model parameters #
        if ((epoch + 1) % params.save_step == 0):
            print('Epoch [{}/{}] complete in {:.0f}m {:.0f}s'.format(epoch + 1,
                              params.num_epochs, time_elapsed // 60, time_elapsed % 60))
            torch.save(critic.state_dict(), os.path.join(
                params.model_root,
                "{}-critic-{}.pt".format("ADDA", epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                "{}-target-encoder-{}.pt".format("ADDA", epoch + 1)))

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "{}-critic-final.pt".format("ADDA")))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        params.model_root,
        "{}-target-encoder-final.pt".format("ADDA")))
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
    criterion = nn.CrossEntropyLoss()
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    for epoch in range(params.num_epochs_pre):
        critic_loss, train_n = 0, 0

        # Zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:

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

            # Optimize critic
            critic_cost.backward()
            optimizer.step()
            critic_loss += critic_cost.item()

            # Print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}] Training Critic Loss: {:.4f}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len_data_loader,
                              critic_cost.item()))


        # Save critic parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(critic, "wdgrl_critic-source-encoder-{}.pt".format(epoch + 1))

    # Save final model
    save_model(encoder, "wdgrl_critic-source-encoder-final.pt")

    return critic

def train_tgt_wdgrl(encoder, clf, critic, src_data_loader, tgt_data_loader, robust=False):

    # Set state
    encoder.train()
    critic.eval()

    # Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(clf.parameters()),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))

    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))
    # Step 2 Train network
    for epoch in range(params.num_epochs):
        ave_loss, train_n = 0, 0
        start_time = time.time()
        # Zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, labels_src), (images_tgt, _)) in data_zip:

            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)
            labels_src = make_variable(labels_src.squeeze_())

            lr = lr_schedule(epoch + 1, params.num_epochs)
            update_lr(optimizer, lr)

            feat_src = encoder(images_src)
            feat_tgt = encoder(images_tgt)

            preds_src = clf(feat_src)
            clf_loss = criterion(preds_src, labels_src)
            wasserstein_distance = critic(feat_src).mean() - critic(feat_tgt).mean()

            loss = clf_loss + params.wd_clf * wasserstein_distance
            optimizer.zero_grad()
            loss.backward()
            ave_loss += loss.item()
            train_n += images_src.size(0)
            optimizer.step()

            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]: Avg Training loss: {:4f}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len_data_loader,
                              ave_loss/train_n))

        time_elapsed = time.time() - start_time

        # 2.4 Save model parameters #
        if ((epoch + 1) % params.save_step == 0):
            print('Epoch [{}/{}] complete in {:.0f}m {:.0f}s'.format(epoch + 1,
                                                                     params.num_epochs, time_elapsed // 60,
                                                                     time_elapsed % 60))
            torch.save(encoder.state_dict(), os.path.join(
                params.model_root, "wdgrl-encoder-{}.pt".format(encoder, epoch + 1)))
            torch.save(clf.state_dict(), os.path.join(
                params.model_root, "wdgrl-clf-{}.pt".format(clf, epoch + 1)))

    torch.save(clf.state_dict(), os.path.join(
        params.model_root, "wdgrl-clf-final.pt"))
    torch.save(encoder.state_dict(), os.path.join(
        params.model_root, "wdgrl-encoder-final.pt"))

    return encoder, clf
