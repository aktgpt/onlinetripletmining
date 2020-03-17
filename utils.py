import os
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt


def make_weights_for_balanced_classes(labels):
    unique_labels, counts = torch.unique(labels, return_counts=True)
    weight_per_class = torch.sum(counts.to(torch.float)) / counts.to(
        torch.float
    )
    weights = [0] * len(labels)
    for i, val in enumerate(labels):
        weights[i] = weight_per_class[torch.where(unique_labels == val)[0][0]]
    return weights


def save_embedding_umap(
    model, train_dataloader, test_dataloader, exp_folder, iter
):
    umap_ = umap.UMAP(random_state=42, n_components=2)
    umap_folder_ = os.path.join(exp_folder, "MNIST_Umap/")
    if not os.path.exists(umap_folder_):
        os.makedirs(umap_folder_)

    feature_data = []
    label_data = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(train_dataloader):
            input = sample[0]
            input = input.cuda()
            labels = sample[1].cuda()
            labels = labels.view(-1)
            fv, _ = model(input)
            feature_data.append(fv.detach().cpu().numpy())
            label_data.append(labels.detach().cpu().numpy())
        feature_data = np.concatenate(feature_data, axis=0)
        label_data = np.concatenate(label_data, axis=0)

        trans = umap_.fit(feature_data)
        fig, ax = plt.subplots(1, figsize=(14, 10))
        plt.scatter(
            trans.embedding_[:, 0],
            trans.embedding_[:, 1],
            s=5,
            c=label_data,
            cmap="Paired",
            alpha=0.5,
        )

        plt.colorbar()
        plt.title("Training Set:UMAP Embeddings")
        plt.savefig(umap_folder_ + f"train_{iter}.png")
        plt.close()

        feature_data = []
        label_data = []
        for batch_idx, sample in enumerate(test_dataloader):
            series_tensors = sample[0]
            series_tensors = series_tensors.cuda()
            labels = sample[1].cuda()
            labels = labels.view(-1)
            fv, _ = model(series_tensors)
            # l1_loss = 0.1 * torch.norm(op, p=2, dim=1).mean()
            feature_data.append(fv.detach().cpu().numpy())
            label_data.append(labels.detach().cpu().numpy())

        feature_data = np.concatenate(feature_data, axis=0)
        label_data = np.concatenate(label_data, axis=0)
        trans = umap_.transform(feature_data)
        fig, ax = plt.subplots(1, figsize=(14, 10))
        plt.scatter(
            trans[:, 0],
            trans[:, 1],
            s=5,
            c=label_data,
            cmap="Paired",
            alpha=0.5,
        )
        plt.colorbar()
        plt.title("Testing Set:UMAP Embeddings")
        plt.savefig(umap_folder_ + f"test_{iter}.png")
        plt.close()

