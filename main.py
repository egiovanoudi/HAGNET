import os
os.environ["OMP_NUM_THREADS"] = '1'

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch
import argparse

import utils
import model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/synthetic_data.csv')
    parser.add_argument('--output_path', default='clusters.txt')
    parser.add_argument('--n_layers_g', type=int, default=1)
    parser.add_argument('--n_layers_d', type=int, default=2)
    parser.add_argument('--threshold_g', type=float, default=0.98)
    parser.add_argument('--threshold_d', type=float, default=0.5)
    parser.add_argument('--loss_t_weight', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    data = pd.read_csv(args.data_path, header=None)
    n_genes = data.shape[0]
    n_timestamps = data.shape[1]
    filter_g = utils.create_filters(args.n_layers_g, n_timestamps)
    filter_d = utils.create_filters(args.n_layers_d, n_timestamps)

    x_d = []
    edges_d = []
    sil = -2

    print('***** Creating the graphs *****')

    x_g = torch.tensor(data.values, dtype=torch.float)
    adj = utils.correlations(data)
    edges_g, weights = utils.adj_to_edge(adj, args.threshold_g)

    for i in range(n_timestamps-1):
        x_d.append(x_g)
        gene_expression = data.iloc[:, i:i + 2]
        edges_d.append(utils.nearest_neighbors(gene_expression, args.threshold_d))

    # Initialize the model
    hagnet_model = model.HAGNET(n_genes, filter_g, filter_d, args.loss_t_weight)
    optimizer = torch.optim.Adam(hagnet_model.parameters(), lr=args.lr)

    print('***** Training the model *****')

    for i in range(args.epochs):
        hagnet_model.train()
        hagnet_model.zero_grad()
        output, loss = hagnet_model(x_g, edges_g, weights, x_d, edges_d)
        loss.backward()

        if ((i + 1) % 5 == 0):
            print(f'Epoch {i + 1}: Training Loss =', loss.item())
        output = output.detach().numpy()
        clusters_pred = utils.clustering(output)
        temp = utils.evaluation(data, clusters_pred)

        if temp > sil:
            sil = temp
            best_pred = clusters_pred

        optimizer.step()

    results = utils.clusters(best_pred)
    with open(args.output_path, "w") as file:
        for cluster, gene in results:
            file.write(f"Cluster_{cluster}\tGene_{gene}\n")