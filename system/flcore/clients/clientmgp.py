import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict


class clientMGP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.num_sub_protos = args.num_sub_protos
        self.triplet_margin = args.triplet_margin
        self.contrastive_weight = args.contrastive_weight
        self.warmup_rounds = args.warmup_rounds

    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.train()

        start_time = time.time()
        current_round = self.train_time_cost['num_rounds']

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for j, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[j, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                    if current_round >= self.warmup_rounds:
                        loss += self._contrastive_loss(rep, y, global_protos)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.collect_protos()
        save_item(model, self.role, 'model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _contrastive_loss(self, rep, y, global_protos):
        """Triplet-style contrastive: pull toward same-class proto, push away from nearest different-class proto."""
        loss = torch.tensor(0.0, device=self.device)
        count = 0
        for idx in range(rep.shape[0]):
            y_c = y[idx].item()
            if type(global_protos[y_c]) == type([]):
                continue

            pos_dist = torch.norm(rep[idx] - global_protos[y_c], p=2)

            min_neg_dist = None
            for k, proto in global_protos.items():
                if k != y_c and type(proto) != type([]):
                    neg_dist = torch.norm(rep[idx] - proto, p=2)
                    if min_neg_dist is None or neg_dist < min_neg_dist:
                        min_neg_dist = neg_dist

            if min_neg_dist is not None:
                triplet = pos_dist - min_neg_dist + self.triplet_margin
                loss += torch.clamp(triplet, min=0.0)
            else:
                loss += pos_dist
            count += 1

        if count > 0:
            loss = loss / count
        return loss * self.contrastive_weight

    def collect_protos(self):
        """Generate K sub-prototypes per class via lightweight K-Means."""
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.eval()

        raw_protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)

                for j, yy in enumerate(y):
                    y_c = yy.item()
                    raw_protos[y_c].append(rep[j, :].detach().data)

        result = {}
        K = self.num_sub_protos
        for label, rep_list in raw_protos.items():
            reps = torch.stack(rep_list)
            if len(rep_list) >= K * 2:
                sub_protos = self._kmeans(reps, K)
            else:
                sub_protos = [(reps.mean(0), len(rep_list))]
            result[label] = sub_protos

        save_item(result, self.role, 'protos', self.save_folder_name)

    def _kmeans(self, reps, K, n_iter=5):
        """Simple K-Means returning list of (centroid, count) tuples."""
        n = reps.shape[0]
        idx = torch.randperm(n)[:K]
        centers = reps[idx].clone()

        for _ in range(n_iter):
            dists = torch.cdist(reps, centers)
            assignments = dists.argmin(dim=1)
            new_centers = []
            for k in range(K):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centers.append(reps[mask].mean(0))
                else:
                    new_centers.append(centers[k])
            centers = torch.stack(new_centers)

        dists = torch.cdist(reps, centers)
        assignments = dists.argmin(dim=1)
        result = []
        for k in range(K):
            mask = assignments == k
            count = mask.sum().item()
            if count > 0:
                result.append((centers[k].detach(), count))
        return result

    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        model.eval()

        test_acc = 0
        test_num = 0

        if global_protos is not None:
            with torch.no_grad():
                for x, y in testloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = model.base(x)

                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in global_protos.items():
                            if type(pro) != type([]):
                                output[i, j] = self.loss_mse(r, pro)

                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
