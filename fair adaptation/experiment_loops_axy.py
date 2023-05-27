from collections import defaultdict

import numpy as np
import torch
import tqdm
from torch import optim

from averaging_manager import AveragedModel
from AXY.models_axy import CategoricalModule, Counter, JointMAP, JointModule, sample_joint

torch.cuda.set_device(2)
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
    astatic = sample_joint(k, n, concentration, is_init_dense)
    
    atransferstatic = astatic.intervention(
        on=intervention,
        concentration=concentration,
        dense=is_init_dense
    )
    # MODULES
    a_model = astatic.to_module()

    atransfer = atransferstatic.to_module()
    #print(causal)

    x_model = astatic.to_xmodel().to_module()
    xtransfer = atransferstatic.to_xmodel().to_module()

    y_model=astatic.to_ymodule()
    ytransfer= atransferstatic.to_ymodule()
    '''joint = JointModule(causal.to_joint().detach().view(n, -1))
    jointtransfer = JointModule(transfer.to_joint().detach().view(n, -1))'''
    if torch.cuda.is_available()==True:
            a_model,x_model,y_model=a_model.cuda(),x_model.cuda(),y_model.cuda()
            atransfer,xtransfer,ytransfer=atransfer.cuda(),xtransfer.cuda(),ytransfer.cuda() 
    # Optimizers
    optkwargs = {'lr': lr, 'lambd': 0, 'alpha': 0, 't0': 0,
                 'weight_decay': 0}
    aoptimizer = optim.ASGD(a_model.parameters(), **optkwargs)
    for name, parameters in a_model.named_parameters():
        print(name, ':', parameters.size())
    xoptimizer = optim.ASGD(x_model.parameters(), **optkwargs)
    yoptimizer = optim.ASGD(y_model.parameters(), **optkwargs)
    #jointoptimizer = optim.ASGD(joint.parameters(), **optkwargs)
    optimizers = [aoptimizer,xoptimizer,yoptimizer]#, antioptimizer, jointoptimizer



    steps = []
    ans = defaultdict(list)
    for t in tqdm.tqdm(range(T)):
          
        # EVALUATION
        
        if t % log_interval == 0:
            steps.append(t)

            with torch.no_grad():
                
                for model, optimizer, target, name in zip(
                        [a_model,x_model,y_model],#, anticausal, joint
                        optimizers,
                        [atransfer,xtransfer,ytransfer],#, antitransfer, jointtransfer
                        ['a_model','x_model','y_model'],#'anti', 'joint'
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
            aloss = atransfer.kullback_leibler(a_model).sum()
            #antiloss = antitransfer.kullback_leibler(anticausal).sum()
            #jointloss = jointtransfer.kullback_leibler(joint).sum()
        else:
            aa, bb ,cc = atransferstatic.sample(m=batch_size)

            taa, tbb ,tcc = torch.from_numpy(aa), torch.from_numpy(bb) ,torch.from_numpy(cc)
            if torch.cuda.is_available()==True:
                taa, tbb ,tcc=taa.cuda(), tbb.cuda() ,tcc.cuda()
            '''print(aa)
            print('............')
            print(bb)'''
            aloss = - a_model(taa, tbb ,tcc).sum() / batch_size
            '''if t==0 :
                print("transfer_joint",transfer.to_joint())
                print("causal_joint0", causal.to_joint())
            if t == T:
                print("causal_jointT", causal.to_joint())
            print('kl:',transfer.kullback_leibler(causal).sum())
            print('loss:' ,causalloss)'''
            xloss = - x_model(taa, tbb ,tcc).sum() / batch_size
            yloss = - y_model(taa, tbb ,tcc).sum() / batch_size
            
            print('yloss:',yloss)
            #print(y_model.is_cuda)#ytransfer.is_cuda,
            print('kl:', ytransfer.kullback_leibler(y_model).sum())
            #jointloss = - joint(taa, tbb).sum() / batch_size

        if torch.cuda.is_available()==True:
            aloss,xloss,yloss=aloss.cuda(),xloss.cuda(),yloss.cuda()
        for loss, opt in zip([aloss,xloss,yloss], optimizers):#, antiloss, jointloss
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

