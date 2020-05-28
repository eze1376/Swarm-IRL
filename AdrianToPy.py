import numpy as np

def irl(mdp, IRLopts, Qopts):
    # global vis

    numOfBases = len(mdp.rewardFuns)
    Vopt = estimateValues(mdp, IRLopts.numOfSteps, IRLopts.numOfMC, IRLopts.singleAgentFlag, IRLopts.batchFlag)
    initAlphas = Vopt.T / norm(Vopt, 2)
    Vopl = np.zeros(IRLopts.numOfIter, IRLopts.numOfBases)
    alphasHistory = np.zeros(numOfBases, IRLopts.numOfIter+1)
    policies = cell(1,IRLopts.numOfIter+1)
    Qtables = cell(1,IRLopts.numOfIter+1)
    basisFuns = mdp.rewardFuns
    
    if vis:
        mdp.plotRewardFun(alphas2rewards(initAlphas, basisFuns))

    for pp in range(IRLopts.numOfIter+1):
        if pp == 1:
            alphasHistory(:,1) = initAlphas
            policies{pp} = IRLopts.initialPolicy
        else
            mdp.rewardFuns = rewardFun
            mdp = flush(mdp)
            [policies{pp}, Qtables{pp}] = swarmLearning(mdp, Qopts)
            mdp.rewardFuns = basisFuns
        mdp.policy = policies{pp}
        if vis and pp >= 2:
            visualizePolicy(mdp)
        if pp == IRLopts.numOfIter+1:
            break
        Vpol(pp,:) = estimateValues(mdp, IRLopts.numOfSteps, IRLopts.numOfMC, IRLopts.singleAgentFlag, IRLopts.batchFlag)
        alphasHistory(:,pp+1) = lp(Vopt, Vpol).T
        rewardFun = alphas2rewards(alphasHistory(:,pp+1), basisFuns)
        
        if vis:
            mdp.plotRewardFun(rewardFun)

def alphas = lp(Vopt, Vpol):

    cvx_begin quiet

        variables alphas(length(Vopt),1) y

        maximize y
        subject to
            norm(alphas,2) <= 1
            for pol in range(size(Vpol,1)):
                y <= (Vopt - Vpol(pol,:)) * alphas

    cvx_end