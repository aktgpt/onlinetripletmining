import os

import torch
from torch.nn.modules import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from loss import OnlineTripleLoss
from utils import make_weights_for_balanced_classes, save_embedding_umap


class TripletTrainer:
    def __init__(self, config):
        self.batch_size = config["batch_size"]
        self.epochs_triplet = config["epochs_triplet"]
        self.epochs_classifier = config["epochs_classifier"]
        self.learning_rate_triplet = config["learning_rate_triplet"]
        self.learning_rate_classify = config["learning_rate_classify"]
        self.triplet_margin = config["triplet_margin"]
        self.triplet_sampling_strategy = config["triplet_sampling_strategy"]
        self.exp_folder = config["exp_name"]
        if not os.path.exists(self.exp_folder):
            os.makedirs(self.temp_folder)

    def train(self, train_dataset, test_dataset, model):
        weights = make_weights_for_balanced_classes(train_dataset.targets)
        sampler = WeightedRandomSampler(weights, len(weights))
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=8,
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, num_workers=8,
        )

        criterion_triplet = OnlineTripleLoss(
            margin=self.triplet_margin,
            sampling_strategy=self.triplet_sampling_strategy,
        )
        criterion_classifier = CrossEntropyLoss()

        optimizer_triplet = Adam(
            params=model.feature_extractor.parameters(),
            lr=self.learning_rate_triplet,
        )
        optimizer_classifier = Adam(
            params=model.classifier.parameters(),
            lr=self.learning_rate_classify,
        )
        print("Training with Triplet loss")
        for i in range(self.epochs_triplet):
            self._train_epoch_triplet(
                model,
                train_dataloader,
                optimizer_triplet,
                criterion_triplet,
                i + 1,
            )
            save_embedding_umap(
                model, train_dataloader, test_dataloader, self.exp_folder, i + 1
            )
        print("Training the classifier")
        for i in range(self.epochs_classifier):
            self._train_epoch_classify(
                model,
                train_dataloader,
                optimizer_classifier,
                criterion_classifier,
                i + 1,
            )
            self._test_epoch_(
                model, test_dataloader, criterion_classifier, i + 1
            )
            # save_embedding_umap(
            #     model, train_dataloader, test_dataloader, self.exp_folder, 99
            # )

    def _train_epoch_triplet(
        self, model, data_loader, optimizer, criterion, epoch
    ):
        log_interval = 50
        model.train()
        running_loss = 0.0
        running_n_triplets = 0
        for batch_idx, sample in enumerate(data_loader):
            input = sample[0].cuda()
            labels = sample[1].cuda()

            optimizer.zero_grad()
            fv, _ = model(input)
            loss, n_triplets = criterion(fv, labels)
            loss.backward()
            optimizer.step()

            running_n_triplets += n_triplets
            running_loss += loss.item()
            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"Training: {epoch}, {batch_idx+1}\
                    Loss:{running_loss/log_interval}\
                    N_Triplets:{running_n_triplets/log_interval}"
                )
                running_loss = 0.0
                running_n_triplets = 0

    def _train_epoch_classify(
        self, model, dataloader, optimizer, criterion, epoch
    ):
        log_interval = 50
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        running_samples = 0.0
        for batch_idx, sample in enumerate(dataloader):
            input = sample[0]
            input = input.cuda()
            labels = sample[1].cuda()
            labels = labels.view(-1)
            optimizer.zero_grad()
            _, op = model(input)
            loss = criterion(op, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(op, 1)
            running_corrects += torch.sum(preds == labels.data)
            running_samples += len(labels)
            running_loss += loss.item()
            # feature_data.append(fv.detach().cpu().numpy())
            # label_data.append(labels.detach().cpu().numpy())
            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"Training:{epoch}, {batch_idx+1}\
                    Loss: {running_loss/log_interval}\
                    Accuracy:{running_corrects/running_samples}"
                )
                running_loss = 0.0
                running_corrects = 0.0
                running_samples = 0.0

    def _test_epoch_(self, model, dataloader, criterion, epoch):
        model.eval()
        running_loss = 0.0
        running_corrects = 0.0
        running_samples = 0.0
        for batch_idx, sample in enumerate(dataloader):
            input = sample[0]
            input = input.cuda()
            labels = sample[1].cuda()
            labels = labels.view(-1)
            _, op = model(input)
            loss = criterion(op, labels)
            loss.backward()
            _, preds = torch.max(op, 1)
            running_corrects += torch.sum(preds == labels.data)
            running_samples += len(labels)
            running_loss += loss.item()
            # feature_data.append(fv.detach().cpu().numpy())
            # label_data.append(labels.detach().cpu().numpy())
        val_acc = running_corrects / running_samples
        print(
            f"Testing:{epoch}, {batch_idx+1}\
                Accuracy:{val_acc}"
        )
