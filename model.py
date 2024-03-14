import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GAT


class HAGNET(nn.Module):
    def __init__(self, n_genes, filter_g, filter_d, n_layers_ta, loss_t_weight):
        super(HAGNET, self).__init__()
        self.n_timestamps = filter_g[0]
        self.loss_t_weight = loss_t_weight

        # Global graph module
        self.gcn_encoder = GCN_Encoder(filter_g)
        self.decoder_g = Decoder_G(filter_g[::-1])

        # Dynamic graphs module
        self.ta_gat_encoder = TA_GAT_Encoder(filter_d, n_genes, self.n_timestamps, n_layers_ta)
        self.decoder_d = Decoder_D(filter_d[::-1], self.n_timestamps)

    def forward(self, x_g, edges_g, weights, x_d, edges_d):
        x_g_encoded = self.gcn_encoder(x_g, edges_g, weights)
        loss_g = self.decoder_g(x_g_encoded, x_g)
        x_d_encoded = self.ta_gat_encoder(x_d, edges_d)
        loss_d, loss_t = self.decoder_d(x_d_encoded, x_d)

        x_encoded = torch.cat((x_g_encoded, x_d_encoded[self.n_timestamps-2]), 1)
        total_loss = loss_g + loss_d + self.loss_t_weight*loss_t

        return x_encoded, total_loss


class GCN_Encoder(nn.Module):
    def __init__(self, filter):
        super(GCN_Encoder, self).__init__()

        self.n_layers = len(filter) - 1

        self.gcn_layer = nn.ModuleList([GCNConv(filter[i], filter[i+1], 1) for i in range(self.n_layers)])

        for i in range(self.n_layers):
            self.gcn_layer[i].apply(self._init_weights)

        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def _init_weights(self, m):
        if isinstance(m, GCNConv):
            bias, weight = list(m.parameters())
            nn.init.kaiming_uniform_(weight)

    def forward(self, x, edges, weights):
        for i in range(self.n_layers):
            x = self.gcn_layer[i](x, edges, weights)
            if i == (self.n_layers-1):
                x = self.sigmoid(x)
            else:
                x = self.relu(x.clone())
        return x


class Decoder_G(nn.Module):
    def __init__(self, filter):
        super(Decoder_G, self).__init__()

        self.n_layers = len(filter) - 1

        self.linear_layer = nn.ModuleList([nn.Linear(filter[i], filter[i + 1]) for i in range(self.n_layers)])

        for i in range(self.n_layers):
            self.linear_layer[i].apply(self._init_weights)

        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        #Loss functions
        self.mse = nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x, true_x):
        for i in range(self.n_layers):
            x = self.linear_layer[i](x)
            if i == (self.n_layers-1):
                x = self.sigmoid(x)
            else:
                x = self.relu(x)

        loss_g = self.mse(true_x, x) + self.kl_div(true_x, x)

        return loss_g


class TA_GAT_Encoder(nn.Module):
    def __init__(self, filter, n_genes, n_timestamps, n_layers_ta):
        super(TA_GAT_Encoder, self).__init__()

        self.n_layers = len(filter) - 1
        self.n_timestamps = n_timestamps

        self.gat_layer = nn.ModuleList([nn.ModuleList([GAT(filter[i], filter[i+1], 1) for _ in range(self.n_timestamps-1)]) for i in range(self.n_layers)])
        self.ta_layer = nn.ModuleList([nn.ModuleList([self.ta_module(n_genes, filter[i+1], n_layers_ta) for _ in range(self.n_timestamps - 2)]) for i in range(self.n_layers)])

        for i in range(self.n_layers):
            for j in range(self.n_timestamps - 1):
                self.gat_layer[i][j].apply(self._init_weights)

        for i in range(self.n_layers):
            for j in range(self.n_timestamps-2):
                self.ta_layer[i][j].apply(self._init_weights)

        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()


    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m, GAT):
            bias1, bias2, bias3, weight = list(m.parameters())
            nn.init.kaiming_uniform_(weight)

    def forward(self, x, edges):
        x_new = []

        for i in range(self.n_layers):
            for j in range(self.n_timestamps - 1):
                if i == 0 or j == 0:
                    temp = x[j]
                    out = self.gat_layer[i][j](x[j], edges[j])
                else:
                    mask = self.ta_layer[i-1][j-1](temp)
                    temp = torch.mul(x[j], mask)
                    out = self.gat_layer[i][j](temp, edges[j])
                if i == (self.n_layers-1):
                    out = self.sigmoid(out)
                else:
                    out = self.relu(out.clone())
                x_new.append(out)
            x = torch.stack(x_new)
            x_new = []

        temp = x[0]
        x_new.append(temp)
        i = self.n_layers-1
        for j in range(self.n_timestamps - 2):
            mask = self.ta_layer[i][j](temp)
            temp = torch.mul(x[j+1], mask)
            x_new.append(temp)

        x = torch.stack(x_new)
        return x

    def ta_module(self, input_dim, hidden_dim, n_layers):
        layers = []

        for _ in range(n_layers-1):
            layers.extend([
                nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, padding=0),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        layers.extend([
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid()
        ])

        ta_block = nn.Sequential(*layers)
        return ta_block


class Decoder_D(nn.Module):
    def __init__(self, filter, n_timestamps):
        super(Decoder_D, self).__init__()

        self.n_layers = len(filter) - 1
        self.n_timestamps = n_timestamps

        self.linear_layer = nn.ModuleList([nn.ModuleList([nn.Linear(filter[i], filter[i+1]) for _ in range(self.n_timestamps-1)]) for i in range(self.n_layers)])
        for i in range(self.n_layers):
            for j in range(self.n_timestamps - 1):
                self.linear_layer[i][j].apply(self._init_weights)

        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Loss functions
        self.mse = nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)

    def calculate_loss(self, x, true_x):
        loss_t = 0
        weight = 1/(self.n_timestamps-1)

        loss_d = self.mse(true_x[self.n_timestamps-2], x[self.n_timestamps-2]) + self.kl_div(true_x[self.n_timestamps-2], x[self.n_timestamps-2])

        for i in range(self.n_timestamps - 2):
            loss_t = loss_t + weight * self.kl_div(x[i], x[i+1])

        return loss_d, loss_t

    def forward(self, x, true_x):
        x_new = []

        for i in range(self.n_layers):
            for j in range(self.n_timestamps - 1):
                out = self.linear_layer[i][j](x[j])
                if i == (self.n_layers - 1):
                    out = self.sigmoid(out)
                else:
                    out = self.relu(out)
                x_new.append(out)
            x = torch.stack(x_new)
            x_new = []

        return self.calculate_loss(x, true_x)
