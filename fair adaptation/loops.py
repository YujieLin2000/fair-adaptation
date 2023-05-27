from collections import defaultdict
import joblib
import numpy as np
import torch
import tqdm
from torch import optim
from model_axycausal import *
from averaging_manager import AveragedModel
from AXY.model_axycausal import CategoricalModule, Counter, JointMAP, JointModule, sample_joint
torch.cuda.set_device(3)

def experiment_optimize(k, n, T, lr, intervention,
                        concentration=1,
                        is_init_dense=True,
                        batch_size=10, scheduler_exponent=0, n0=10,
                        log_interval=10, use_map=False):
    """Measure optimization speed and parameters distance.

    Hypothesis: initial distance to optimum is correlated to optimization speed with SGD.

    Sample n mechanisms of order k and for each of them sample an
    intervention on the desired mechanism. Use SGD to update a causal
    and an anticausal model for T steps. At each step, measure KL
    and distance in scores for causal and anticausal directions.
    """
    #static = sample_joint(k, n, concentration, is_init_dense)
    
    pa=joblib.load('AXY/mnist/pro_matrix/pa')
    pxa=joblib.load('AXY/mnist/pro_matrix/pxa')
    pyax=joblib.load('AXY/mnist/pro_matrix/pyax')
    pa,pxa,pyax=np.array(pa),np.array(pxa),np.array(pyax)
    static=CategoricalStatic(pa, pxa, pyax, from_probas=True)
    '''pa=joblib.load('AXY/adult/adult_pro/pa')
    pxa=joblib.load('AXY/adult/adult_pro/pxa')
    pyax=joblib.load('AXY/adult/adult_pro/pyax')
    pa,pxa,pyax=np.array(pa),np.array(pxa),np.array(pyax)
    static=CategoricalStatic(pa, pxa, pyax, from_probas=True)'''
    transferstatic = static.intervention(
        on=intervention,
        concentration=concentration,
        dense=is_init_dense
    )
    # MODULES
    '''a_model = astatic.to_module()

    atransfer = atransferstatic.to_module()
    #print(causal)

    x_model = astatic.to_xmodel().to_module()
    xtransfer = atransferstatic.to_xmodel().to_module()

    y_model=astatic.to_ymodule()
    ytransfer= atransferstatic.to_ymodule()'''
    causal=static.to_module()
    transfer =transferstatic.to_module()

    anticausal=static.to_anticausal().to_module()
    antitransfer = transferstatic.to_anticausal().to_module()
    joint = JointModule(causal.to_joint().detach(),n)#,.view(n, -1 )
    jointtransfer = JointModule(transfer.to_joint().detach(),n)#.view(n, -1)

    # Optimizers
    optkwargs = {'lr': lr, 'lambd': 0, 'alpha': 0, 't0': 0,
                 'weight_decay': 0}
    optimizer = optim.ASGD(causal.parameters(), **optkwargs)
    antioptimizer= optim.ASGD(anticausal.parameters(), **optkwargs)
    jointoptimizer = optim.ASGD(joint.parameters(), **optkwargs)
    optimizers = [optimizer,antioptimizer]#, antioptimizer, jointoptimizer,jointoptimizer



    steps = []
    ans = defaultdict(list)
    for t in tqdm.tqdm(range(T)):

        # EVALUATION
        if t % log_interval == 0:
            steps.append(t)

            with torch.no_grad():

                for model, optimizer, target, name in zip(
                        [causal,anticausal],#,, joint
                        optimizers,
                        [transfer,antitransfer],#, antitransfer, jointtransfer, jointtransfer
                        ['causal','anticausal'],#, 'joint'
                ):
                    # SGD
                    ans[f'kl_{name}'].append(target.kullback_leibler(model))
                    ans[f'scoredist_{name}'].append(target.scoredist(model))

                    # ASGD
                    with AveragedModel(model, optimizer) as m:
                        ans[f'kl_{name}_average'].append(
                            target.kullback_leibler(m))
                        ans[f'scoredist_{name}_average'].append(
                            target.scoredist(m))



        # UPDATE
        for opt in optimizers:
            opt.lr = lr / t ** scheduler_exponent
            opt.zero_grad()

        if batch_size == 'full':
            causalloss = transfer.kullback_leibler(causal).sum()
            #antiloss = antitransfer.kullback_leibler(anticausal).sum()
            #jointloss = jointtransfer.kullback_leibler(joint).sum()
        else:
            aa, bb ,cc = transferstatic.sample(m=batch_size)

            taa, tbb ,tcc = torch.from_numpy(aa), torch.from_numpy(bb) ,torch.from_numpy(cc)
            '''print(aa)
            print('............')
            print(bb)'''
            causalloss = - causal(taa, tbb ,tcc).sum() / batch_size
            '''if t==0 :
                print("transfer_joint",transfer.to_joint())
                print("causal_joint0", causal.to_joint())
            if t == T:
                print("causal_jointT", causal.to_joint())
            print('kl:',transfer.kullback_leibler(causal).sum())
            print('loss:' ,causalloss)'''
            antiloss = - anticausal(taa, tbb ,tcc).sum() / batch_size
            jointloss = - joint(taa, tbb ,tcc).sum() / batch_size
            #jointloss = - joint(taa, tbb).sum() / batch_size
            if t == 0 or t == T - 1:
                print('causal1111111111111111sa', causal.sa[0])
                print('anticausal1111111111111111sa', anticausal.sa[0])
                print('causal1111111111111111sxa', causal.sxa[0][0])
                print('anticausal1111111111111111sxa', anticausal.sxa[0][0])
                print('causal1111111111111111syax', causal.syax[0][0][0])
                print('anticausal1111111111111111syax', anticausal.syax[0][0][0])
        for loss, opt in zip([causalloss,antiloss], optimizers):#, antiloss, jointloss
            loss.requires_grad_(True)
            loss.backward()
            opt.step()

    for key, item in ans.items():
        ans[key] = torch.stack(item).cpu().numpy()

    return {'steps': np.array(steps), **ans}


def test_experiment_optimize():
    for intervention in ['cause', 'effect', 'gmechanism']:
        experiment_optimize(
            k=2, n=3, T=6, lr=.1, batch_size=4, log_interval=1,
            intervention=intervention, use_map=True
        )




if __name__ == "__main__":
    test_experiment_optimize()

