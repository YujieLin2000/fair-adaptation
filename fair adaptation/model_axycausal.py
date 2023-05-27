import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn, optim

from utils import kullback_leibler, logit2proba, logsumexp, proba2logit


def joint2conditional(joint):
    marginal = np.sum(joint, axis=-1)
    conditional = joint / np.expand_dims(marginal, axis=-1)

    return CategoricalStatic(marginal, conditional)


def jointlogit2conditional(joint, model_mode=1):
    joint_ax = logsumexp(joint)
    sa = logsumexp(joint_ax)
    sa -= sa.mean(axis=1, keepdims=True)
    sxa = joint_ax - sa[:, :, np.newaxis]
    sxa -= sxa.mean(axis=2, keepdims=True)
    syax = joint - joint_ax[:, :, :, np.newaxis]
    syax -= syax.mean(axis=3, keepdims=True)
    model_mode = model_mode

    return CategoricalStatic(sa, sxa, syax, from_probas=False, model_mode=model_mode)


def sample_joint(k, n, concentration=1, dense=False, logits=True):
    """Sample n causal mechanisms of categorical variables of dimension K.

    The concentration argument specifies the concentration of the resulting cause marginal.
    """

    if logits:
        sa = stats.loggamma.rvs(concentration, size=(n, k))
        sa -= sa.mean(axis=1, keepdims=True)

        conditional_concentration = concentration if dense else concentration / k
        if conditional_concentration > 0.1:
            sxa = stats.loggamma.rvs(conditional_concentration, size=(n, k, k))
        else:
            # A loggamma with small shape parameter is well approximated
            # by a negative exponential with parameter scale = 1/ shape
            sxa = - stats.expon.rvs(scale=1 / conditional_concentration, size=(n, k, k))
        sxa -= sxa.mean(axis=2, keepdims=True)

        if conditional_concentration > 0.1:
            syax = stats.loggamma.rvs(conditional_concentration, size=(n, k , k, k))
        else:
            # A loggamma with small shape parameter is well approximated
            # by a negative exponential with parameter scale = 1/ shape
            syax = - stats.expon.rvs(scale=1 / conditional_concentration, size=(n, k, k, k))
        syax -= syax.mean(axis=3, keepdims=True)

        return CategoricalStatic(sa, sxa, syax, from_probas=False)
    else:
        pa = np.random.dirichlet(concentration * np.ones(k), size=n)
        condconcentration = concentration if dense else concentration / k
        pxa = np.random.dirichlet(condconcentration * np.ones(k), size=[n, k])
        pyax = np.random.dirichlet(condconcentration * np.ones(k), size=[n, k, k])
        return CategoricalStatic(pa, pxa, pyax, from_probas=True)


class CategoricalStatic:
    """Represent n categorical distributions of variables (a,b) of dimension k each."""

    def __init__(self, marginal, conditional, two_conditional, from_probas=True, model_mode=1):
        """The distribution is represented by a marginal p(a) and a conditional p(b|a)

        marginal is n*k array.
        conditional is n*k*k array. Each element conditional[i,j,k] is p_i(b=k |a=j)
        """
        self.n, self.k = marginal.shape
        self.model_mode = model_mode

        if not conditional.shape == (self.n, self.k, self.k):
            raise ValueError(
                f'Marginal shape {marginal.shape} and conditional '
                f'shape {conditional.shape} do not match.')

        if from_probas:
            self.marginal = marginal
            self.conditional = conditional
            self.two_conditional = two_conditional
            self.sa = proba2logit(marginal)
            self.sxa = proba2logit(conditional)
            self.syax = proba2logit(two_conditional)
        else:
            self.marginal = logit2proba(marginal)
            self.conditional = logit2proba(conditional)
            self.two_conditional = logit2proba(two_conditional)
            self.sa = marginal
            self.sxa = conditional
            self.syax = two_conditional

    # 计算分布p(a,b)
    def to_joint(self, return_probas=True):
        if return_probas:
            return self.two_conditional * \
                   (self.conditional * self.marginal[:, :, np.newaxis])[:, :, :, np.newaxis]
        else:  # return logits
            joint_xa = self.sxa \
                       + (self.sa - logsumexp(self.sxa))[:, :, np.newaxis]
            joint = self.syax \
                    + (joint_xa - logsumexp(self.syax))[:, :, :, np.newaxis]
            return joint - np.mean(joint, axis=(1, 2, 3), keepdims=True)

    # 计算p(a|b)
    def to_anticausal(self):
        """Return conditional from b to a.
        Compute marginal pb and conditional pab such that pab*pb = pba*pa.
        """
        joint = self.to_joint(return_probas=False)  # sab
        joint = np.swapaxes(joint, 2, 3)  # invert variables
        return jointlogit2conditional(joint, model_mode=2)

    # 计分布间的距离
    def probadist(self, other):
        pd = np.sum((self.marginal - other.marginal) ** 2, axis=1)
        pd += np.sum((self.conditional - other.conditional) ** 2, axis=(1, 2))
        pd += np.sum((self.two_conditional - other.conditional) ** 2, axis=(1, 2, 3))
        return pd

    # 计算score之间的距离
    def scoredist(self, other):
        sd = np.sum((self.sa - other.sa) ** 2, axis=1)
        sd += np.sum((self.sxa - other.sxa) ** 2, axis=(1, 2))
        sd += np.sum((self.syax - other.syax) ** 2, axis=(1, 2, 3))
        return sd

    # 返回布间的距离，score之间的距离
    def sqdistance(self, other):
        """Return the squared euclidean distance between self and other"""
        return self.probadist(other), self.scoredist(other)

    # 计算两个分布的KL散度
    def kullback_leibler(self, other):
        p0 = self.to_joint().reshape(self.n, self.k ** 3)
        p1 = other.to_joint().reshape(self.n, self.k ** 3)
        return kullback_leibler(p0, p1)

    def intervention(self, on, concentration=1, dense=True):
        # sample new marginal

        if on == 'independent':
            # make cause and effect independent,
            # but without changing the effect marginal.
            newmarginal = self.reverse().marginal
        elif on == 'geometric':
            newmarginal = logit2proba(self.sba.mean(axis=1))
        elif on == 'weightedgeo':
            newmarginal = logit2proba(np.sum(self.sba * self.marginal[:, :, None], axis=1))
        else:  # 从迪利克雷分布抽样构造新的pa
            newmarginal = np.random.dirichlet(concentration * np.ones(self.k), size=self.n)

        # TODO use logits of the marginal for stability certainty
        # replace the cause or the effect by this marginal
        if on == 'A':
            #np.random.seed(66)
            newmarginal1 = np.random.dirichlet(concentration * np.ones(self.k), size=self.n)
            return CategoricalStatic(newmarginal1, self.conditional, self.two_conditional)
        elif on == 'X':
            # intervention on effect
            newconditional = np.repeat(newmarginal[:, None, :],self.k , axis=1)#self.k
            return CategoricalStatic(self.marginal, newconditional, self.two_conditional)
        elif on == 'AandX':
            # intervention on effect
            newconditional = np.repeat(newmarginal[:, None, :], self.k, axis=1)
            newmarginal_A = np.random.dirichlet(concentration * np.ones(self.k), size=self.n)
            return CategoricalStatic(newmarginal_A, newconditional, self.two_conditional)
        elif on == 'Y':
            # intervention on effect
            condconcentration = concentration if dense else concentration / self.k
            newtwo_conditional = np.repeat(newmarginal[:, None, :], self.k, axis=1)
            newtwo_conditional = np.repeat(newtwo_conditional[:, :, None ,:], self.k, axis=2)
            return CategoricalStatic(self.marginal, self.conditional, newtwo_conditional)

        elif on in ['effect', 'independent', 'geometric', 'weightedgeo']:
            # intervention on effect
            newconditional = np.repeat(newmarginal[:, None, :], self.k, axis=1)
            return CategoricalStatic(self.marginal, newconditional)
        elif on == 'mechanism':
            # sample a new mechanism from the same prior
            sba = sample_joint(self.k, self.n, concentration, dense, logits=True).sba
            return CategoricalStatic(self.sa, sba, from_probas=False)
        elif on == 'gmechanism':
            # sample from a gaussian centered on each conditional
            sba = np.random.normal(self.sba, self.sba.std())
            sba -= sba.mean(axis=2, keepdims=True)
            return CategoricalStatic(self.sa, sba, from_probas=False)
        elif on == 'singlecond':
            newscores = stats.loggamma.rvs(concentration, size=(self.n, self.k))
            newscores -= newscores.mean(1, keepdims=True)
            # if 'simple':
            #     a0 = 0
            # elif 'max':
            a0 = np.argmax(self.sa, axis=1)
            sba = self.sba.copy()
            sba[np.arange(self.n), a0] = newscores
            return CategoricalStatic(self.sa, sba, from_probas=False)
        else:
            raise ValueError(f'Intervention on {on} is not supported.')

    # 对每个分布采样m个样本
    def sample(self, m, return_tensor=False):
        """For each of the n distributions, return m samples. (n*m*2 array) """
        flatjoints = self.to_joint().reshape((self.n, self.k ** 3))
        # print(flatjoints.shape)

        samples = np.array(
            [np.random.choice(self.k ** 3, size=m, p=p) for p in flatjoints])
        a = samples // (self.k ** 2)
        x = (samples - (a) * self.k * self.k) // self.k
        y = samples % self.k
        if not return_tensor:
            return a, x, y
        else:
            return torch.from_numpy(a), torch.from_numpy(x), torch.from_numpy(y)

    def to_module(self):
        if torch.cuda.is_available() == True:
            sa, sxa, syax = self.sa, self.sxa, self.syax
        return CategoricalModule(sa, sxa, syax,
                                 model_mode=self.model_mode)

    def __repr__(self):
        return (f"n={self.n} categorical of dimension k={self.k}\n"
                f"{self.marginal}\n"
                f"{self.conditional}")


class CategoricalModule(nn.Module):
    """Represent n categorical conditionals as a pytorch module"""

    def __init__(self, sa, sxa, syax, model_mode=1):
        super(CategoricalModule, self).__init__()
        self.n, self.k = tuple(sa.shape)

        sa = sa.clone().detach().cuda() if torch.is_tensor(sa) else torch.tensor(sa).cuda()
        sxa = sxa.clone().detach().cuda() if torch.is_tensor(sxa) else torch.tensor(sxa).cuda()
        syax = syax.clone().detach().cuda() if torch.is_tensor(syax) else torch.tensor(syax).cuda()
        self.sa = nn.Parameter(sa.to(torch.float32))
        self.sxa = nn.Parameter(sxa.to(torch.float32))
        self.syax = nn.Parameter(syax.to(torch.float32))
        '''self.marginal = marginal.clone().detach() if torch.is_tensor(marginal) else torch.tensor(marginal)
        self.conditional=conditional.clone().detach() if torch.is_tensor(conditional) else torch.tensor(conditional)
        self.two_conditional=two_conditional.clone().detach() if torch.is_tensor(two_conditional) else torch.tensor(two_conditional)'''
        self.marginal = self.tensor_logit2proba(self.sa)
        self.conditional = self.tensor_logit2proba(self.sxa)
        self.two_conditional = self.tensor_logit2proba(self.syax)
        self.model_mode = model_mode

    def tensor_logit2proba(self, s):
        # torch.exp(s - np.expand_dims(logsumexp(s), axis=-1))
        return torch.exp(s - torch.unsqueeze(self.tensor_logsumexp(s), -1))

    def tensor_logsumexp(self, s):
        smax = torch.amax(s, axis=-1)
        return smax + torch.log(
            torch.sum(torch.exp(s - torch.unsqueeze(smax, axis=-1)), axis=-1))

    def forward(self, a, b, c):
        """
        :param a: n*m collection of m class in {1,..., k} observed
        for each of the n models
        :param b: n*m like a
        :return: the log-probability of observing a,b,
        where model 1 explains first row of a,b,
        model 2 explains row 2 and so forth.
        """
        batch_size = a.shape[1]
        '''if self.BtoA:
            a, b = b, a'''
        if self.model_mode == 2:
            b,c = c,b
        rows = torch.arange(0, self.n).unsqueeze(1).repeat(1, batch_size)
        # print(rows.view(-1), a.view(-1), b.view(-1))
        '''print(a)
        print('......................')
        print(b)
        print('111',self.to_joint().shape)
        print(rows.view(-1).long(), a.view(-1).long(), b.view(-1).long())
        print(self.to_joint()[rows.view(-1).long(), a.view(-1).long(), b.view(-1).long()].view(self.n, batch_size))'''
        return self.to_joint()[rows.view(-1).long(), a.view(-1).long(), b.view(-1).long(), c.view(-1).long()].view(
            self.n, batch_size)

    def to_joint(self):
        # joint_ax=F.log_softmax(self.sxa, dim=2) \
        # + F.log_softmax(self.sa, dim=1).unsqueeze(dim=2)

        return F.log_softmax(self.syax, dim=3) \
               + (F.log_softmax(self.sxa, dim=2) \
                  + F.log_softmax(self.sa, dim=1).unsqueeze(dim=2)).unsqueeze(dim=3)

    def to_static(self):
        return CategoricalStatic(
            logit2proba(self.sa.detach().numpy()),
            logit2proba(self.sxa.detach().numpy()),
            logit2proba(self.syax.detach().numpy())
        )

    def kullback_leibler(self, other):
        joint = self.to_joint()

        return torch.sum((joint - other.to_joint()) * torch.exp(joint),
                         dim=(1, 2, 3))

    def scoredist(self, other):
        return torch.sum((self.sa - other.sa) ** 2, dim=1) \
               + torch.sum((self.sxa - other.sxa) ** 2, dim=(1, 2)) \
               + torch.sum((self.syax - other.syax) ** 2, dim=(1, 2, 3))

    def __repr__(self):
        return f"CategoricalModule(joint={self.to_joint().detach()})"

'''
class JointModule(nn.Module):

    def __init__(self, logits):
        super(JointModule, self).__init__()
        self.n, k3 = logits.shape  # logits is flat

        self.k = int(pow(k3, 1 / 3))
        # if self.k ** 2 != k2:
        #     raise ValueError('Logits matrix can not be reshaped to square.')

        # normalize to sum to 0
        logits = logits - logits.mean(dim=1, keepdim=True)
        self.logits = nn.Parameter(logits)

    @property
    def logpartition(self):
        return torch.logsumexp(self.logits, dim=1)

    def forward(self, a, b, c):
        batch_size = a.shape[1]
        rows = torch.arange(0, self.n).unsqueeze(1).repeat(1, batch_size).view(-1)
        index = (a * self.k * self.k + b * self.k + c).view(-1)
        return F.log_softmax(self.logits, dim=1)[rows.long(), index.long()].view(self.n, batch_size)

    def kullback_leibler(self, other):
        a = self.logpartition
        kl = torch.sum((self.logits - other.logits) * torch.exp(self.logits - a[:, None]), dim=1)
        return kl - a + other.logpartition

    def scoredist(self, other):
        return torch.sum((self.logits - other.logits) ** 2, dim=1)

    def __repr__(self):
        return f"CategoricalJoint(logits={self.logits.detach()})"
'''


class JointModule(nn.Module):

    def __init__(self, joint,n):
        super(JointModule, self).__init__()
        self.joint=joint
        self.joint = nn.Parameter(joint)
        self.logits=self.joint.detach().view(n, -1)
        self.n=n
    @property
    def logpartition(self):
        return torch.logsumexp(self.logits, dim=1)

    def forward(self, a, b, c):
        batch_size = a.shape[1]
        rows = torch.arange(0, self.n).unsqueeze(1).repeat(1, batch_size)
        return self.joint[rows.view(-1).long(), a.view(-1).long(), b.view(-1).long(), c.view(-1).long()].view(
            self.n, batch_size)

    def kullback_leibler(self, other):
        joint = self.joint
        return torch.sum((joint - other.joint) * torch.exp(joint),
                         dim=(1, 2, 3))

    def scoredist(self, other):
        return torch.sum((self.logits - other.logits) ** 2, dim=1)

    def __repr__(self):
        return f"CategoricalJoint(logits={self.logits.detach()})"



class Counter:

    def __init__(self, counts):
        self.counts = counts
        self.n, self.k, self.k2 = counts.shape

    @property
    def total(self):
        return self.counts.sum(axis=(1, 2), keepdims=True)

    # @jit
    def update(self, a: np.ndarray, b: np.ndarray):
        for aaa, bbb in zip(a.T, b.T):
            self.counts[np.arange(self.n), aaa, bbb] += 1


def test_Counter():
    c = Counter(np.zeros([1, 2, 2]))
    c.update(np.array([[0, 0, 0, 1]]), np.array([[0, 0, 1, 1]]))
    assert c.total == 4
    assert np.allclose(c.counts / c.total, [[.5, .25], [0, .25]])


class JointMAP:

    def __init__(self, prior, counter):
        self.prior = prior
        self.n0 = self.prior.sum(axis=(1, 2), keepdims=True)
        self.counter = counter

    @property
    def frequencies(self):
        return ((self.prior + self.counter.counts) /
                (self.n0 + self.counter.total))

    def to_joint(self):
        return np.log(self.frequencies)





if __name__ == "__main__":
    print("hi")
    # test_CategoricalModule()
    test_Counter()
