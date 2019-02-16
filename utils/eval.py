def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def color2label(data):
    '''
    data: 'P'模式下的Image转换来的tensor类型
    return: 分类映射的标签
    '''
    uq = torch.unique(data)
    example = torch.Tensor(data.shape)
    uqlen = len(uq)
#     print(uqlen)
    for i in range(uqlen):
        example = example.fill_(i+256)
        data = torch.where(data==uq[uqlen-1-i],example,data)
    return (data-256)