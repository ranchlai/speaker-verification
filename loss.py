import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class AMSoftmaxLoss(nn.Layer):
    def __init__(self,
                 feature_dim: int,
                 n_classes: int,
                 m: float = 0.3,
                 s: float = 30.0):
        super(AMSoftmaxLoss, self).__init__()
        self.w = paddle.create_parameter((feature_dim, n_classes), 'float32')
        #self.w = paddle.randn((feature_dim, n_classes), 'float32')*0.01
       # self.wn = paddle.randn((feature_dim, n_classes), 'float32')
       # self.wn.stop_gradient = False
        self.s = s
        self.m = m
        self.nll_loss = nn.NLLLoss()
        self.n_classes = n_classes

    def forward(self, logit, label):
      #  logit = F.normalize(logit, p=2, axis=-1)
       # self.wn = F.normalize(self.w, p=2, axis=0)
        cosine = logit #paddle.matmul(logit, self.w)
        y = paddle.zeros((logit.shape[0], self.n_classes))
        for i in range(logit.shape[0]):
            y[i, label[i]] = self.m
        prob = F.log_softmax((cosine - y) * self.s, -1)
        return self.nll_loss(prob, label)

if __name__ == '__main__':
    loss = AMSoftmaxLoss(512,10)
    print(loss.parameters())
