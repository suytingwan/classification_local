from __future__ import print_function, absolute_import

__all__ = ['accuracy','npair_accuracy', 'top_k_results']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def npair_accuracy(output, output_pos, target, topk=(1,)):

    size_row = inputs.size(0)
    size_col = inputs.size(1)

    inputs = inputs.view(size_row, size_col)
    inputs = nn.functional.normalize(inputs, dim=1)

    input_pos = input_pos.view(size_row, size_col)
    input_pos = nn.functional.normalize(input_pos, dim=1)

    dists = torch.mm(inputs, input_pos.t())


    maxk = max(topk)
    _, pred = dists.topk(maxk, 1, True, True)
    pred = pred.t()

    targets = torch.arange(size_row)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def top_k_results(output, target, topk=(1,)):

    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)

    return pred

