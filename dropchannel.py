import torch
import torch.nn as nn

class DropChannel(nn.Module):
    def __init__(self, wrs_ratio, p):
        super(DropChannel, self).__init__()
        self.N, self.C, self.H, self.W = 2, 16, 2, 1
        self.wrs_ratio = wrs_ratio
        self.p = p

    def forward(self, x):
        # print(x.shape)
        # self.N, self.C, self.H, self.W = x.shape
        x = x.transpose(1, 2)
        score = self.__weighted_channel_dropout(x)
        mask, alpha = self.__weighted_random_selection(score)
        rng = self.__random_number_generator()
        mask = mask & rng
        mask = mask.type(torch.FloatTensor)
        if torch.cuda.is_available():
            mask = mask.cuda()
        mask = mask * alpha
        mask = mask.view(self.N, self.C, -1)
        mask = mask.expand(self.N, self.C, self.H * self.W).view_as(x)
        x = mask * x
        x = x.transpose(1, 2)
        return x

    def __weighted_channel_dropout(self, x):
        return x.view(self.N, self.C, -1).sum(2) / (self.H * self.W)

    def __weighted_random_selection(self, score):
        # assert 0 not in torch.unique(score > torch.zeros_like(score)), \
        #     'score need greater than zero'
        p = score / score.sum(1).view(self.N, -1)
        r = torch.rand(self.N, self.C) # 生成0到1之间的随机数

        if torch.cuda.is_available():
            r = r.cuda()

        key = r ** (1 / score)

        # 选取前M个大的，mask标记为1
        M = self.wrs_ratio * self.C # 剩余通道数
        MthNum = torch.sort(key, dim=1, descending=True)[0]
        if torch.cuda.is_available():
            MthNum = torch.index_select(MthNum, 1, torch.cuda.LongTensor([M - 1]))
        else:
            MthNum = torch.index_select(MthNum, 1, torch.LongTensor([M - 1]))
        # MthNum = MthNum[:, M - 1].view(self.N, -1)
        mask = key >= MthNum

        # 计算alpha
        score_F = torch.Tensor(score.shape).copy_(score)
        if torch.cuda.is_available():
            score_F = score_F.cuda()
        score_F[key < MthNum] = 0
        alpha = score.sum(1).view(self.N, -1) / score_F.sum(1).view(self.N, -1)
        return mask, alpha

    def __random_number_generator(self):
        rng = torch.bernoulli(torch.Tensor(self.N, self.C).fill_(self.p)).type(torch.ByteTensor)
        if torch.cuda.is_available():
            rng = rng.cuda()
        return rng



if __name__ == '__main__':
    #in_planes, in_size, wrs, p,
    input = torch.FloatTensor(1, 10, 4, 4).cuda()
    input = torch.nn.ReLU(True)(input)
    print("input",input)
    import time
    start = time.time()
    dropout_layer = DropChannel(input.shape, 0.9, 0.5)
    print(f"{time.time() - start} s:\n",dropout_layer(input))
