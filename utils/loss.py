import torch
from torch.nn import Module
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.function import Function

class MixLoss(Module):

    def __init__(self, center):
        super(MixLoss,self).__init__()
        self.center = torch.autograd.Variable(center, requires_grad=False)

    def forward(self, outputs, features, targets):
        
        cross_entropy_loss = nn.functional.cross_entropy(outputs, targets)
        target_embedding = self.center.index_select(0, targets)
        target_embedding = target_embedding.float()

        a_dist = torch.sqrt(torch.sum(features * features, 1) + 0.000001)
        b_dist = torch.sqrt(torch.sum(target_embedding * target_embedding, 1) + 0.000001)
        cos_dist = torch.sum(features * target_embedding,1)

        dist_loss = torch.abs(cos_dist/(a_dist * b_dist))
        loss = cross_entropy_loss * (0.5 + 0.5 * dist_loss)
        return torch.mean(loss)


def log_sigmoid(input):
    input_sigmoid = torch.nn.functional.softmax(input)
    return torch.log(input_sigmoid + 0.0000001)


def kl_divergence(y, t):
    crossentropy = - torch.mul(t, log_sigmoid(y))
    return crossentropy


class LabelSmoothLoss(Module):

    def __init__(self, weight=None):
        super(LabelSmoothLoss, self).__init__()
        self.weight = weight
        if weight is not None:
            self.weight = torch.autograd.Variable(weight.unsqueeze(1), requires_grad=False)

    def forward(self, outputs, targets):

        per_loss = kl_divergence(outputs, targets)
        if self.weight is not None:
            loss = torch.mm(per_loss, self.weight)
        else:
            loss = torch.max(per_loss, -1)[0]

        # print(loss.data)
        # print(torch.mean(loss).data)
        return torch.mean(loss)


def kl_divergence2(y, t):
    input_softmax = torch.nn.functional.softmax(y)

    per_loss = input_softmax.gather(1, t.view(-1, 1))
    per_loss = - torch.log(per_loss)
    return per_loss


class CrossEntropyLoss(Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        if weight is not None:
            self.weight = torch.autograd.Variable(weight.unsqueeze(1), requires_grad=False)

    def forward(self, outputs, targets):
        per_loss = kl_divergence2(outputs.data, targets.data)
        if self.weight is not None:
            weight_ = self.weight.gather(0, targets.data.view(-1, 1))
            per_loss = per_loss * weight_
        return torch.mean(per_loss)


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        if feat.size(1) != self.feat_dim:
            raise ValueError(
                "Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim, feat.size(1)))
        return self.centerlossfunc(feat, label, self.centers)


class CenterlossFunc(Function):

    @staticmethod
    def forward(ctx, feature, label, centers):
        ctx.save_for_backward(feature, label, centers)
        centers_batch = centers.index_select(0, label)
        return (feature - centers_batch).pow(2).sum(1).sum(0) / 2.0

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_tensors
        centers_batch = centers.index_select(0, label)
        diff = centers_batch - feature
        counts = centers.new(centers.size(0)).fill_(1)
        ones = centers.new(label.size(0)).fill_(1)
        grad_centers = centers.new(centers.size()).fill_(0)
        counts = counts.scatter_add_(0, label, ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers / counts.view(-1, 1)

        return Variable(-grad_output.data * diff), None, Variable(grad_centers)