import torch
import torch.nn.functional as F

def perplexity(y_pred, y):
    """
    :param y_pred: (batch_size, predicted_class, sequence_length)
    :param y: (batch_size, sequence_length)
    :return:
    """
    ce = F.cross_entropy(y_pred, y, reduction='mean')
    error = torch.exp(ce) * y.size(0) * y.size(1)
    return error

def mse(y_pred, y):
    return F.mse_loss(y_pred, y)


def binary_accuracy(y_pred, y):
    y_pred = torch.round(torch.sigmoid(y_pred))  # round predictions to the closest integer
    correct = (y_pred == y).float()
    acc = correct.sum()
    return acc


def accuracy(y_pred, y):
    """
    :param y_pred: (batch_size, predicted_class, sequence_length) for NLP task
                    (batch_size, predicted_class) for CV task
    :param y: (batch_size, sequence_length) for NLP task
                (batch_size, 1) for CV task
    :return:
    """
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y).float()
    acc = correct.sum()
    return acc
