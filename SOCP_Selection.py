# SOCP pipeline

import cvxpy as cp
from cvxpy.lin_ops.lin_utils import transpose
import numpy as np
from numpy.core.defchararray import endswith
from scipy import sparse
from scipy.io import savemat
from scipy import io as sio
from numpy import linalg as la
import numpy as np

def socp(l,a,c,L,t,u,x):
    #Define the objective function for socp
    #objective = cp.Minimize(t + c.T@x + l*u)
    objective = cp.Minimize((a*t) - ((1-a)*(cp.sum(cp.entr(c.T@x))) + (l*u)))
    #Loop for norm
    cs = []
    for i in range(len(c.shape)):
        cs.append(cp.sum(cp.abs(x[i])))
    #Define constraints
    constraint_soc = [cp.hstack([1+t,2*L@x,1-t]) >= 0, cp.sum(cs) <= u, x >= 0]
    #Problem definition
    prob = cp.Problem(objective, constraint_soc)
    #Solvers - ECOS or MOSEK - I would recommend to use MOSEK as solver but you can comment MOSEK part and open ECOS solver.
    #prob.solve(verbose = True, solver=cp.SCS)
    prob.solve(verbose = True, solver=cp.MOSEK)
    return x.value

def nearestPD(Q):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    Q = (Q + Q.T) / 2
    _, s, V = la.svd(Q)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (Q + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(Q))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(Q.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3
def isPD(Q):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        L = la.cholesky(Q)
        return True
    except la.LinAlgError:
        return False

if __name__ == '__main__':

    F = sio.loadmat('RandModel_PRED_m.mat')
    F = F['PRED_m']
    data = sio.loadmat('RandModel_PROBs.mat')
    prob = data['softmax_dist']
    sample = 50000
    model = 300 
    category = 10
    divide = 5
    sampart = int(sample / divide)
    F = np.reshape(F,(sample,model))
    prob = np.reshape(prob,(sample,model,category))
    for iteration in range(0, 2):
        if iteration == 1:
            Fsub = F[(sampart*divide-sampart):sampart*divide, :]
            probsub = prob[(sampart*divide-sampart):sampart*divide,:,:]
            savemat("results_valid.mat", {"Fvalid" : Fsub, "probvalid" : probsub})
        else:
            Fsub = F[:sampart*(divide-1), :]
            probsub = prob[:sampart*(divide-1),:,:]
            savemat("results_train.mat", {"Ftrain" : Fsub, "probtrain" : probsub})
        #F = np.asarray(F)
        Q = np.dot(Fsub.transpose(), Fsub)
        #Q = nearestPD(Q)
        #assert (isPD(Q))
        L = np.linalg.cholesky(Q)
        x = cp.Variable(len(L[0])) #weights w_i
        rows = len(probsub)
        cols = len(probsub[0])
        c= []
        for j in range(0, cols):
            sumCol = 0
            for i in range(0, rows):
                max_prob = np.max(probsub[i][j])
                sumCol = sumCol + max_prob
            avg = sumCol/sample
            #print(avg)
            c.append(avg)
        c = np.asarray(c)
        c = np.reshape(c,(model,1))
        #c = c[:1000,:]
        #u = 300
        u = 1 #This is for norm transformation it wont take anything - stable
        t = cp.Variable() # This is for socp transormation for quadratic part - stable
        l = [0.1,0.2,0.3,0.4,0.5] # l is the lambda, this is regularization term
        a = [0.1,0.3,0.5,0.7,0.9] # a is for alpha in the function
        result = []
        result_dict = dict()
        for i in range(len(l)):
            l_new = l[i]
            #Loop for alpha in the function
            for j in range(len(a)):
                a_new = a[j]
                current_result = socp(l_new,a_new,c,L,t,u,x)
                result.append(current_result)
                result_dict[str(l_new) + ',' + str(a_new)] = current_result
        result = np.asarray(result)
        print(result.shape)
        print(result_dict.keys())
        if iteration == 1:
            results = savemat("results_valid_socp.mat", {"result_valid" : result, "dictkeys" : result_dict})
        else:
            results = savemat("results_train_socp.mat", {"result_train" : result, "dictkeys" : result_dict})
    print("end")
