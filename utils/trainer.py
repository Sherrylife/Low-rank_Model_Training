import torch
import torch.nn as nn
import time
from utils.model_utils import *
from utils.data_utils import *

class TrainingObject():
    def __init__(self, args, logger):
        super().__init__()
        self.dataset_name = args['dataset']
        self.model_name = args['model']
        self.args = args
        self.logger = logger

        self.train_loader, self.val_loader, self.test_loader = \
            fetch_dataset(self.args)
        self.model = generate_model(self.args)
        self.optimizer = generate_optimizer(self.model, self.args)
        self.criterion = generate_criterion(self.args)
        self.lr_scheduler = generate_scheduler(self.optimizer, self.args)
        self.metric = generate_metric(self.args)

        # this zero gradient update is needed to avoid a warning message,
        # https://github.com/ildoonet/pytorch-gradual-warmup-lr/issues/8
        self.optimizer.zero_grad()
        self.optimizer.step()

    def _cv_one_round(self, train_loader, model, criterion, metric, optimizer, args):
        device = args['device']
        regularization = args['regularization']

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            n_samples += y.size(0)

            optimizer.zero_grad()

            y_pred = model(x)

            loss_vec = criterion(y_pred, y)
            loss = loss_vec.mean()
            if regularization == 'frobenius':
                loss += args['coef_decay'] * model.frobenius_decay()
            elif regularization == 'kronecker':
                loss += args['coef_decay'] * model.kronecker_decay()
            elif regularization == 'L2':
                loss += args['coef_decay'] * model.L2_decay()
            elif regularization == 'L2-KD':
                loss += (args['coef_L2'] * model.L2_decay() + args['coef_KD'] * model.kronecker_decay())
            elif regularization == 'L2-FD':
                loss += (args['coef_L2'] * model.L2_decay() + args['coef_FD'] * model.frobenius_decay())
            elif regularization == 'FD-KD':
                loss += (args['coef_KD'] * model.kronecker_decay() + args['coef_FD'] * model.frobenius_decay())

            loss.backward()
            optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0)   # loss_vec.size(0) is batch_size
            global_metric += metric(y_pred, y).detach()

        return global_loss / n_samples, global_metric / n_samples


    def _nlp_one_round(self, train_loader, model, criterion, metric, optimizer, args):
        device = args['device']
        regularization = args['regularization']

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            n_samples += y.size(0)

            chunk_len = y.size(1)

            optimizer.zero_grad()

            y_pred = model(x)

            loss_vec = criterion(y_pred, y)
            loss = loss_vec.mean()

            if regularization == 'frobenius':
                loss += args['coef_decay'] * model.frobenius_decay()
            elif regularization == 'kronecker':
                loss += args['coef_decay'] * model.kronecker_decay()
            elif regularization == 'L2':
                loss += args['coef_decay'] * model.L2_decay()
            elif regularization == 'L2-KD':
                loss += (args['coef_L2'] * model.L2_decay() + args['coef_KD'] * model.kronecker_decay())
            elif regularization == 'L2-FD':
                loss += (args['coef_L2'] * model.L2_decay() + args['coef_FD'] * model.frobenius_decay())
            elif regularization == 'FD-KD':
                loss += (args['coef_KD'] * model.kronecker_decay() + args['coef_FD'] * model.frobenius_decay())


            loss.backward()
            optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0) / chunk_len  # loss_vec.size(0) is batch_size
            global_metric += metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples


    def _cv_model_eval(self, data_loader, model, criterion, metric, args):
        model.eval()
        device = args['device']

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                n_samples += y.size(0)
                y_pred = model(x)
                global_loss += criterion(y_pred, y).sum().item()
                global_metric += metric(y_pred, y).item()

        return global_loss / n_samples, global_metric / n_samples


    def _nlp_model_eval(self, data_loader, model, criterion, metric, args):
        model.eval()
        device = args['device']

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                n_samples += y.size(0)
                chunk_len = y.size(1)
                y_pred = model(x)
                global_loss += criterion(y_pred, y).sum().item() / chunk_len
                global_metric += metric(y_pred, y).item() / chunk_len

        return global_loss / n_samples, global_metric / n_samples


    def train(self):
        if self.dataset_name in ['svhn', 'cifar10', 'cifar100', 'tinyImageNet', 'ImageNet']:
            train_func = self._cv_one_round
            evaluate_func = self._cv_model_eval
        elif self.dataset_name in ['tinyShakespeare', 'shakespeare', 'reddit', 'wikiText2']:
            train_func = self._nlp_one_round
            evaluate_func = self._nlp_model_eval
        else:
            raise NotImplementedError("Do not support the training under this dataset and model.")

        train_logs = {'loss': [], 'metric': []}
        test_logs = {'loss': [], 'metric': []}

        total_round = self.args['T']
        eval_freq = self.args['eval_freq']

        for cur_round in range(total_round):
            start_time = time.time()
            train_loss, train_metric = train_func(
                train_loader=self.train_loader,
                model=self.model,
                criterion=self.criterion,
                metric=self.metric,
                optimizer=self.optimizer,
                args=self.args,
            )

            train_logs['loss'].append(train_loss)
            train_logs['metric'].append(train_metric)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if cur_round % eval_freq == 0:
                test_loss, test_metric = evaluate_func(
                    data_loader=self.test_loader,
                    model=self.model,
                    criterion=self.criterion,
                    metric=self.metric,
                    args=self.args,
                )
                test_logs['loss'].append(test_loss)
                test_logs['metric'].append(test_metric)

            end_time = time.time()

            self.logger.info(
                f'round = {cur_round:d}, '
                f'cost = {(end_time - start_time):.4f}s, '
                f"train_loss = {train_logs['loss'][-1]:.4f}, "
                f"train_metric = {train_logs['metric'][-1]:.4f}, "
                f"test_loss = {test_logs['loss'][-1]:.4f}, "
                f"test_metric = {test_logs['metric'][-1]:.4f}, "
            )
