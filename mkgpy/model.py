import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from GPy import Model, Param
import GPy
from paramz import ObsAr
import functools as ft
import pickle
import pandas as pd
import emcee
import jax

from numba import njit,prange,set_num_threads
import jax.numpy as jnp
from jax.config import config


import time
from multiprocessing.dummy import Process
from multiprocessing import Pool
from jax import jit, device_put

@njit(cache=True,fastmath=True,parallel=False)
def numba_dot(a,b):
    return np.dot(a,b)

@njit(cache=True,fastmath=False,parallel=False)
def numba_kron(a,b):
    return np.kron(a,b)


@jit
def kron_mvprod( A, b):
    """
    A : list of matrix
    b : array 1D
    return np.kron(A1,A2...)@b
    Do not work with non square matrices
    https://arxiv.org/pdf/1209.4120.pdf

    """
    x = b
    N = 1
    D = len(A)
    G = np.zeros((D), dtype=np.int_)
    for d in range(0, D):
        G[d] = len(A[d])
    N = np.prod(G)
    for d in range(D-1, -1, -1):
        X = np.reshape(x, (G[d], int(np.round(N/G[d]))), order='F')
        Z = jnp.dot(A[d], X) #-- This is the expensive part 
        Z = Z.T
        x = np.reshape(Z, (-1, 1), order='F')
    return x

def kron_mmprod(A,B): 
    """
    A : list of matrix
    B : matrix
    return np.kron(A1,A2,...)@B
    Faster than the usual algebra for large arrays (and can be parallelize over the loop)
    """
    res = np.empty(B.shape)
    for col in range(B.shape[1]):  ##TO PARALLELIZE
        b = B[:,col]
        res[:,col] = kron_mvprod(A,b).flatten()
    return res
    


def kron_mmprod_slice(A,B,res,l_col):
    """
    l_col : list of column index
    """
    for col in l_col:
        res[:,col] = kron_mvprod(A,B[:,col]).flatten()
    return 0


def kron_mmprod_parallel(A,B,N_thread=1): 
    """
    A : list of matrix
    B : matrix
    return np.kron(A1,A2,...)@B
    Faster than the usual algebra for large arrays (and can be parallelize over the loop)
    """
    res = np.empty(B.shape)
    #--list of list of column index
    if N_thread>B.shape[1]:
        N_thread=B.shape[1]
    L_col = np.array_split(np.arange(B.shape[1]), N_thread)
    processes = [Process(target=kron_mmprod_slice, args=(A,B,res,l_col)) for l_col in L_col]
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    return res
   

######-- Numba uselfull on CPU

@njit(cache=True,parallel=False,fastmath=False)
def kron_mvprod2( A, b):
    """
    A : list of matrix
    b : array 1D
    return np.kron(A1,A2...)@b
    Do not work with non square matrices
    https://arxiv.org/pdf/1209.4120.pdf

    """

    x = b
    N = 1
    D = len(A)
    G = np.zeros((D), dtype=np.int_)
    for d in range(0, D):
        G[d] = len(A[d])
    N = np.prod(G)
    first_it = True
    for d in range(D-1, -1, -1):
        if first_it==True:
            #X = np.reshape(x, (G[d], int(np.round(N/G[d]))), order='F')
            X = x.T.reshape(( int(np.round(N/G[d])),G[d])).T
            first_it=False
            res=0
        else:
            X = res.T.reshape(( int(np.round(N/G[d])),G[d])).T

        Z = numba_dot(A[d], X)
        #x = np.reshape(Z, (-1, 1), order='F')
        #Z = np.ascontiguousarray(Z)
        
        #print(Z.flags['C_CONTIGUOUS'],Z.flags['F_CONTIGUOUS'])
        #print(np.info(x))
        res = Z.reshape((1, -1)).T

    
        
    return res#Z.T.reshape((1, -1)).T


@njit(cache=True,parallel=True,fastmath=False)
def kron_mmprod2(A,B,N_thread): 
    """
    A : list of matrix
    B : matrix
    return np.kron(A1,A2,...)@B
    Faster than the usual algebra for large arrays (and can be parallelize over the loop)
    """
    set_num_threads(int(N_thread))
    #if B.flags['F_CONTIGUOUS']==False: #--needed for numba njit
    B = np.asfortranarray(B)

    res = np.empty(B.shape)
    for col in prange(B.shape[1]):  ##TO PARALLELIZE
        b = B[:,col]
        res[:,col] = kron_mvprod2(A,b).flatten()
    return np.asfortranarray(res)





class MultiKroModel(Model):
    
    def __init__(self,L_X,Y,L_kernel,L_noise,nugget_s=1e-15,nugget_n=None,jax_enable_x64=True,name='MultiKroModel'):
        """
        """
        super(MultiKroModel, self).__init__(name=name)
        self.L_X = L_X
        self.Y = Y
        self.L_kernel = L_kernel
        self.L_noise = L_noise        
        self.nugget_s = nugget_s
        if nugget_n==None:
            self.nugget_n = nugget_s
        else:
            self.nugget_n = nugget_n

        self.jax_enable_x64 = jax_enable_x64 #-- Speed up but decreases the precision
        config.update("jax_enable_x64", jax_enable_x64)
        
        self.L_n_samples = [X.shape[0] for X in L_X]
        
        self.L_likelihood = [] #-- To keep track of the likelihood evals
        
        # accept the construction arguments
        #self.X1 = ObsAr(X1)
        #self.X2 = ObsAr(X2)
        
        for kern,noise in zip(L_kernel,L_noise):
            self.link_parameter(kern)
            self.link_parameter(noise)
        
            
        for i,X in enumerate(L_X):
            self.__dict__[f'X_{i}'] = Param("input", X)
    
    def parameters_changed(self):
        """
        Evaluates the log_marginal_likelihood, computes and updates the gradient for every hyperparameters of the model
        Correct only when the kernels have indep parameters. See [Saatci 2012]
        """
        #-- First computed the log likelihood:
        self.log_marginal_likelihood = self.get_log_likelihood()
          
        #-- List of the projective matrix Q@S_inv@U for every dimension
        L_H_tild = [jnp.dot(jnp.dot(self.L_U[j],self.L_S_inv[j]),self.L_Q[j]) for j in range(len(self.L_Q))]

        #--loop over the subspaces to update the noise and signal parameters gradients
        for i in range(len(self.L_kernel)):
            X_i = self.__dict__[f'X_{i}']
              
             #-- First create a list with every flat parameters object to check if fixed
            L_flat_param_kernel=[]
            for el in self.L_kernel[i].flattened_parameters:
                if el.ndim==1:
                    for j in range(el.shape[0]):
                        L_flat_param_kernel.append(el[j:j+1])
                else:
                    for j in range(el.shape[0]):
                        for l in range(el.shape[1]):
                            L_flat_param_kernel.append(el[j,l:l+1])

            #-- First create a list with every flat parameters object to check if fixed
            L_flat_param_noise=[]
            for el in self.L_noise[i].flattened_parameters:
                if el.ndim==1:
                    for j in range(el.shape[0]):
                        L_flat_param_noise.append(el[j:j+1])
                else:
                    for j in range(el.shape[0]):
                        for l in range(el.shape[1]):
                            L_flat_param_noise.append(el[j,l:l+1])

                            
            #-- Compute the kernels derivatives             
            dK_dtheta = self.L_kernel[i].dK_dtheta(X_i) #-- gradients of the kernels : (nparam_i,d_i,d_i)
            dN_dtheta = self.L_noise[i].dK_dtheta(X_i)

            dL_dtheta_K = []
            dL_dtheta_N = []
            
           

                                            
            #--Need 2 diff loops for noise and signal kernels because the number of param can be different
            for p_k in range(dK_dtheta.shape[0]):
                            
                #-- if the param is fixed, don't change the gradient
                if L_flat_param_kernel[p_k]._has_fixes()==True:
                    dL_dtheta_K.append(L_flat_param_kernel[p_k].gradient[0])
                    
                else:
                    #-- compute the first part of the gradient
                    L_K = self.L_K.copy()  #--Complet list of matrix kernels
                    L_K.pop(i)  #--Remove the ith kernel (to be replaced by the gradient)
                    L_K.insert(i,dK_dtheta[p_k]) 
                    grad_dat = jnp.dot(0.5*self.alpha.T,kron_mvprod(L_K,self.alpha))

                    #--gradient for logdet
                    L_Lambda = self.L_Lambda.copy()
                    L_Lambda.pop(i) 
                    L_Lambda.insert(i, jnp.diag(jnp.dot(L_H_tild[i].T,jnp.dot(dK_dtheta[p_k],L_H_tild[i]))) ) 

                    #grad_det = -0.5 * np.sum(self.W_inv * ft.reduce(np.kron, [np.diag(np.dot(H_tild_i.T,np.dot(L_K_i,H_tild_i))) for H_tild_i,L_K_i in zip(L_H_tild,L_K)] ))
                    grad_det = -0.5 * jnp.sum(self.W_inv * ft.reduce(jnp.kron, L_Lambda ))                
                    dL_dtheta_K.append(np.float(grad_dat+grad_det))
                
            for p_n in range(dN_dtheta.shape[0]):
               
                #-- if the param is fixed, don't change the gradient
                if L_flat_param_noise[p_n]._has_fixes()==True:
                    dL_dtheta_N.append(L_flat_param_noise[p_n].gradient[0])
                                                    
                else:
                    #-- compute the first part of the gradient
                    L_N = self.L_N.copy()
                    L_N.pop(i)
                    L_N.insert(i,dN_dtheta[p_n])
                    grad_dat = 0.5*jnp.dot(self.alpha.T,kron_mvprod(L_N,self.alpha))

                    #--gradient for logdet
                    L_I = [jnp.ones(el) for el in self.L_n_samples]
                    L_I.pop(i)
                    L_I.insert(i,jnp.diag(jnp.dot(L_H_tild[i].T,jnp.dot(dN_dtheta[p_n],L_H_tild[i]))))
                    #grad_det = -0.5 * self.W_inv @ ft.reduce(np.kron, [np.diag(np.dot(H_tild_i.T,np.dot(L_N_i,H_tild_i))) for H_tild_i,L_N_i in zip(L_H_tild,L_N)] ) 
                    grad_det = -0.5 * jnp.sum(self.W_inv * ft.reduce(jnp.kron, L_I ) )
                    dL_dtheta_N.append(np.float(grad_dat+grad_det))
            
            self.L_kernel[i].dL_dtheta_K = dL_dtheta_K 

            #-- update the objective function gradient for every parameter
            self.L_kernel[i].gradient = dL_dtheta_K
            self.L_noise[i].gradient = dL_dtheta_N    
            
 
    
    def log_likelihood(self):
        return self.get_log_likelihood()

    def get_log_likelihood(self):
        #-- build the matrices
        L_K = []
        L_N = []
        for kernel,noise,X,n_samples in zip(self.L_kernel,self.L_noise,self.L_X,self.L_n_samples):
            L_K.append(device_put(kernel.K(X) + self.nugget_s*jnp.eye(n_samples))) 
            L_N.append(device_put(noise.K(X) + self.nugget_n*jnp.eye(n_samples)) ) #-- nugget term usefull to stabilize the inversion
       
        #update the computed kernels        
        self.L_K = L_K
        self.L_N = L_N
        
        #-- This part is optimized : no direct inversion or kronecker product.
        try:
            #--build list for the eigendecomposition of the noise matrices : N_i = U_i @ S_i @ U_i.T
            L_U = [] 
            L_S = []
            #--build list for the eigendecomposition of the projected kernel matrices : 
            # K_tild_i = S_i**-0.5 @ U_i.T @ K_i @ U_i @ S_i**-0.5 
            # K_tild_i = Q_i @ Lambda_i @ Q_i.T
            L_Q = []
            L_Lambda = []
            #--This loop could be parallelized
            for i in range(len(self.L_kernel)): #-- eigendecomposition takes time when subspaces become large
                eigval,eigvect = jnp.linalg.eigh(L_N[i])
                S , U = eigval, eigvect # S is a 1d vector
                L_U.append(U)
                L_S.append(S)

                tmp = jnp.dot(U , jnp.diag(S**-0.5))
                K_tild =  jnp.dot(tmp.T  , jnp.dot(L_K[i] , tmp) )
                eigval,eigvect = jnp.linalg.eigh(K_tild)
                Lambda , Q = eigval, eigvect # Lambda is a 1d vector
                L_Q.append(Q)
                L_Lambda.append(Lambda)
    
            self.L_Lambda = L_Lambda
            W = ft.reduce(jnp.kron, L_Lambda) + 1 #np.ones(np.prod(self.L_n_samples)) #1D kronecker products
            #W_inv = np.diag(W**-1)
            W_inv = W**-1

            if jnp.any(W<0):
                LogL = -np.inf
                print('Matrix not semi-positive')
                return LogL
                
            #-- Determinant part
            # eigval(A+B) = eigval(A) + eigval(B) , eigval(A@B) = eigval(A) * eigval(B) for sym matrices
            #K+N =  U@S**-0.5@ (K_tild + I) @S**-0.5@U.T ; And eig(U) = 1, eig(S**-0.5) = S**-0.5
            #=> eig(K+N) = diag(S**-0.5) * (eig(K_tild) + eig(I)) * diag(S**-0.5)
            #            = (diag(S**-1)*np.diag(W)
            
            LnDet = jnp.sum(jnp.log( ft.reduce(jnp.kron, L_S)*W ))  # = LnDet = np.sum(np.log(W)) whern N=1


            #-- Datafit part
            
            L_U_T = [U.T for U in L_U ]
            L_Q_T = [Q.T for Q in L_Q ]
            L_S_inv = [jnp.diag(S**-0.5) for S in L_S]
            
            alpha = kron_mvprod(L_U_T,self.Y)
            alpha = kron_mvprod(L_S_inv,alpha)
            alpha = kron_mvprod(L_Q_T,alpha)
            
            alpha = W_inv*alpha.flatten() #-- =np.dot(W_inv,alpha) but dot prod takes a lot of time when the total datavector becomes large
            
            alpha = kron_mvprod(L_Q,alpha)
            alpha = kron_mvprod(L_S_inv,alpha)
            alpha = kron_mvprod(L_U,alpha)
            
            self.alpha=alpha

            self.LnDet = LnDet
            LogL = -0.5 * (jnp.dot(self.Y.T, alpha) + LnDet + np.prod(self.L_n_samples) * jnp.log(2*jnp.pi))

            self.alpha = alpha # alpha = Ky**-1 @ y used in predict
            self.L_U_T = L_U_T
            self.L_Q_T = L_Q_T
            self.L_S_inv = L_S_inv
            self.L_S = L_S
            self.L_U = L_U
            self.L_Q = L_Q
            self.W_inv = W_inv

        
        
            self.L_likelihood.append(LogL)
            
            #if LogL > 0:
            #    LogL = -np.inf

        except np.linalg.LinAlgError:
            
            LogL = -jnp.inf
            print('linalg Error : inf likelihood')
            
        
        return LogL
    
    
    def predict(self,L_X_new,N_thread=1,compute_cov=True,use_M=False,jax_enable_x64=True):
        """
        use_M : bool : wether to use the intermediate method for covariance computation
        jax_enable_x64 : Bool : Only applies to the covariance estimation, the mean is always at max precision
        """
        
        #--Max precision forced for the mean
        config.update("jax_enable_x64", True) 
        
        #--Build the matrices
        L_KxX = []
        L_Kxx = []
        L_KxX_T = []
        L_Kxx_T = []        
        for kernel,X,X_new in zip(self.L_kernel,self.L_X,L_X_new):
            tmp = kernel.K(X_new,X)
            L_KxX.append(tmp)
            L_KxX_T.append(tmp.T)
            tmp = kernel.K(X_new)
            L_Kxx.append(tmp)
            L_Kxx_T.append(tmp.T)  
            
        self.L_KxX = L_KxX
        self.L_Kxx = L_Kxx
        self.L_KxX_T = L_KxX_T
        self.L_Kxx_T = L_Kxx_T       
        
        #start = time.time()
        Kxx = ft.reduce(jnp.kron,L_Kxx)
        #print('done kron:',time.time()-start)
        #start = time.time()
        KxX = ft.reduce(jnp.kron,L_KxX) #-- Takes + time
        KXx = KxX.T
        #print('done kron:',time.time()-start)
    
        #start = time.time()
        y  = jnp.dot(KxX,self.alpha)
        #print('done dot roduct for mu:',time.time()-start)

        
        if compute_cov==True:
            config.update("jax_enable_x64", jax_enable_x64)
            if use_M==False:  
                #start = time.time() #-- Takes +++ time
                A = kron_mmprod_parallel(self.L_U_T,KXx,N_thread)
                A = kron_mmprod_parallel(self.L_S_inv,A,N_thread)
                A = kron_mmprod_parallel(self.L_Q_T,A,N_thread)
                #print('done computing cov_mu step 1:',time.time()-start)
                
                WA=A*np.reshape(np.repeat(self.W_inv,A.shape[1]),A.shape) # = np.dot(W_inv,A) use the fact that W_inv is diagonal

                #if diag==True:
                #    cov_y = np.diag(Kxx) - np.sum(A*WA,axis=0)
                    
                cov_y = Kxx - jnp.dot(A.T,WA)


                #A = A*np.reshape(np.repeat(self.W_inv,A.shape[1]),A.shape) #--use the fact that W_inv is diagonal
                #A = kron_mmprod_parallel(self.L_Q,A,N_thread)
                #A = kron_mmprod_parallel(self.L_S_inv,A,N_thread)
                #A = kron_mmprod_parallel(self.L_U,A,N_thread)
                #cov_y = Kxx - np.dot(KxX,A)

            if use_M==True:
                if not hasattr(self, 'M'): #-- The computation cost is tracable for small dataset

                    M = kron_mmprod_parallel(self.L_Q,np.diag(self.W_inv),N_thread)
                    M = kron_mmprod_parallel(self.L_S_inv,M,N_thread)
                    M = kron_mmprod_parallel(self.L_U,M,N_thread)
                    M = M.T
                    M = kron_mmprod_parallel(self.L_U,M,N_thread)
                    M = kron_mmprod_parallel(self.L_S_inv,M,N_thread)
                    M = kron_mmprod_parallel(self.L_Q,M,N_thread)
                    self.M = M #save the M matrix

                cov_y = Kxx - jnp.dot(KxX, jnp.dot(self.M, KXx)) #--Once M is computed this is quite fast
                

            return y , cov_y

        if compute_cov==False:
            return y
            
        #if likelihood_err ==True:
            #cov_y += self.likelihood_err()
        
        #--Rescale to the original dataset
        #if self.data.y_scaler != None:
        #    y = y * np.std(self.data.y_train_pre) + np.mean(self.data.y_train_pre)
        #    cov_y = cov_y * np.std(self.data.y_train_pre)**2
        
    
    
    #def get_likelihood_err(self,method='bootstrap'):
    #    
    #    if method == 'bootstrap':
    #        likelihood_err = 0
    #        
    #    else method == 'propagation':
    #        likelihood_err = 0
    #        
    #    self.likelihood_err = likelihood_err
    



from GPy.kern import Kern
from GPy.core.parameterization import Param


class Add_MK(GPy.kern.Add):
    """
    Same kernel as in GPy library but add the dK_dtheta method to get the gradient of the kernel
    """
    def __init__(self, subkerns, name='sum'):
        super(Add_MK,self).__init__(subkerns, name)
        
    def dK_dtheta(self,X,X2=None):
        if X2 is None: X2 = X
        grad = ft.reduce(lambda x, y: np.concatenate((x, y), axis=0), [p.dK_dtheta(X,X2) for p in self.parts])
        return grad
        
class Prod_MK(GPy.kern.Prod):
    """
    Same kernel as in GPy library but add the dK_dtheta method to get the gradient of the kernel
    """
    def __init__( self,kernels,name='mul'):
        super(Prod_MK,self).__init__(kernels, name)
        
    def dK_dtheta(self,X,X2=None):
        if X2 is None: X2 = X
        L_grad=[]
        for i,p in enumerate(self.parts):
            tmp = self.parts.copy()
            tmp.pop(i)
            tmp = ft.reduce(np.multiply, (p.K(X,X2) for p in tmp))
            L_grad.append(tmp*p.dK_dtheta(X,X2))
        grad = ft.reduce(lambda x, y: np.concatenate((x, y), axis=0), L_grad)
        return grad
    


class WhiteHeteroscedastic_MK(GPy.kern.WhiteHeteroscedastic):
    """
    Same kernel as in GPy library but add the dK_dtheta method to get the gradient of the kernel
    """
    def __init__(self, input_dim, num_data, variance=1., active_dims=None, name='white_hetero'):
        super(WhiteHeteroscedastic_MK,self).__init__( input_dim, num_data, variance, active_dims, name)
        
    def K(self, X, X2=None):
        if (X2 is None or np.array_equal(X2,X)==True) and X.shape[0] == self.variance.shape[0]:
            return np.eye(X.shape[0]) * self.variance
        else:
            return 0.
        
    def dK_dtheta(self,X,X2=None):
        """
        The analytical derivatives come from the function update_gradients_full in GPy.kern.src.stationnary
        This function along with lengthscale_grads_pure are general to any kernel with parameters : [variance, lenghtscale(s)]. Any additional 
        parameter derivative must be added 
        return : 3D array : (nparam,Xdim,Xdim)
        """
        
        n = self.variance.size

        grad = np.zeros((n, n, n))

        # Loop through the array and set the diagonal elements to 1
        for i in range(n):
            grad[i, i, i] = 1
    
        return grad
        
    

class RBF_MK(GPy.kern.RBF):
    """
    Same kernel as in GPy library but add the dK_dtheta method to get the gradient of the kernel
    """
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='rbf', useGPU=False, inv_l=False):
        super(RBF_MK,self).__init__( input_dim, variance, lengthscale, ARD, active_dims, name, useGPU, inv_l)
        
    
    def lengthscale_grads_pure(self, tmp, X, X2):
        """
        Comes from the function _lengthscale_grads_pure in GPy.kern.src.stationnary, but modified such as it returns a 3D array containing 
        [dK_dl1,d_K_dl2,...]
        input : tmp : mixture of dL_dr and metric, see dK_dtheta() method
        """
        # sumation has been removed and lenghtscale dimesion adjusted to get a 3D array corresponding to the dK_dtheta
        # Note : transposed not considered as kernels are symmetric
        if X2 is None: X2 = X

        return -np.array([tmp * np.square(X[:,q:q+1] - X2[:,q:q+1].T) for q in range(self.input_dim)])/(self.lengthscale**3).T[:,None]

    def dK_dtheta(self,X,X2=None):
        """
        The analytical derivatives come from the function update_gradients_full in GPy.kern.src.stationnary
        This function along with lengthscale_grads_pure are general to any kernel with parameters : [variance, lenghtscale(s)]. Any additional 
        parameter derivative must be added 
        return : 3D array : (nparam,Xdim,Xdim)
        """
        
        if X2 is None: X2 = X

        ##-- The variance gradient
        #self.variance.gradient = np.sum(self.K(X, X2)* dL_dK)/self.variance
        dK_dvariance = self.K(X, X2)/self.variance

        #-- the lengthscale gradient(s)
        dL_dr = self.dK_dr_via_X(X, X2) #* dL_dK ; remove this to get dK_dtheta in the end instead of dL_dtheta

        if self.ARD:
            tmp = dL_dr*self._inv_dist(X, X2)
            #self.lengthscale.gradient = self._lengthscale_grads_pure(tmp, X, X2)
            dK_dlengthscale = self.lengthscale_grads_pure(tmp, X, X2)
            grad = np.concatenate((dK_dvariance[None,:,:],dK_dlengthscale),axis=0)
        else:
            r = self._scaled_dist(X, X2)
            #self.lengthscale.gradient = -np.sum(dL_dr*r)/self.lengthscale
            dK_dlengthscale = -dL_dr*r/self.lengthscale
            grad = np.concatenate((dK_dvariance[None,:,:],dK_dlengthscale[None,:,:]),axis=0)
        return grad
    
    
class RatQuad_MK(GPy.kern.RatQuad):
    """
    Same kernel as in GPy library but add the dK_dtheta method to get the gradient of the kernel
    """
    def __init__(self, input_dim, variance=1., lengthscale=None, power=2., ARD=False, active_dims=None, name='RatQuad'):
        super(RatQuad_MK,self).__init__( input_dim, variance, lengthscale, power, ARD, active_dims, name)
        
    
    def lengthscale_grads_pure(self, tmp, X, X2):
        """
        Comes from the function _lengthscale_grads_pure in GPy.kern.src.stationnary, but modified such as it returns a 3D array containing 
        [dK_dl1,d_K_dl2,...]
        input : tmp : mixture of dL_dr and metric, see dK_dtheta() method
        """
        # sumation has been removed and lenghtscale dimesion adjusted to get a 3D array corresponding to the dK_dtheta
        # Note : transposed not considered as kernels are symmetric
        if X2 is None: X2 = X

        return -np.array([tmp * np.square(X[:,q:q+1] - X2[:,q:q+1].T) for q in range(self.input_dim)])/(self.lengthscale**3).T[:,None]

    def dK_dtheta(self,X,X2=None):
        """
        The analytical derivatives come from the function update_gradients_full in GPy.kern.src.stationnary
        This function along with lengthscale_grads_pure are general to any kernel with parameters : [variance, lenghtscale(s)]. Any additional 
        parameter derivative must be added 
        return : 3D array : (nparam,Xdim,Xdim)
        """
        
        if X2 is None: X2 = X

        ##-- The variance gradient
        #self.variance.gradient = np.sum(self.K(X, X2)* dL_dK)/self.variance
        dK_dvariance = self.K(X, X2)/self.variance

        #-- the lengthscale gradient(s)
        dL_dr = self.dK_dr_via_X(X, X2) #* dL_dK ; remove this to get dK_dtheta in the end instead of dL_dtheta

        if self.ARD:
            tmp = dL_dr*self._inv_dist(X, X2)
            #self.lengthscale.gradient = self._lengthscale_grads_pure(tmp, X, X2)
            dK_dlengthscale = self.lengthscale_grads_pure(tmp, X, X2)
            grad = np.concatenate((dK_dvariance[None,:,:],dK_dlengthscale),axis=0)
        else:
            r = self._scaled_dist(X, X2)
            #self.lengthscale.gradient = -np.sum(dL_dr*r)/self.lengthscale
            dK_dlengthscale = -dL_dr*r/self.lengthscale
            grad = np.concatenate((dK_dvariance[None,:,:],dK_dlengthscale[None,:,:]),axis=0)
            
        #-- Power parameter gradient
        r = self._scaled_dist(X, X2)
        r2 = np.square(r)
        dK_dpow = -self.variance * np.exp(self.power*(np.log(2.)-np.log1p(r2+1)))*np.log1p(r2/2.) 
        grad = np.concatenate((grad,dK_dpow[None,:,:]),axis=0)

        return grad
    
    

        
        
class Matern32_MK(GPy.kern.Matern32):
    """
    Same kernel as in GPy library but add the dK_dtheta method to get the gradient of the kernel
    """
    def __init__(self,input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Mat32'):
        super(Matern32_MK,self).__init__( input_dim, variance, lengthscale, ARD, active_dims, name)
        
    
    def lengthscale_grads_pure(self, tmp, X, X2):
        """
        Comes from the function _lengthscale_grads_pure in GPy.kern.src.stationnary, but modified such as it returns a 3D array containing 
        [dK_dl1,d_K_dl2,...]
        input : tmp : mixture of dL_dr and metric, see dK_dtheta() method
        """
        # sumation has been removed and lenghtscale dimesion adjusted to get a 3D array corresponding to the dK_dtheta
        # Note : transposed not considered as kernels are symmetric
        if X2 is None: X2 = X

        return -np.array([tmp * np.square(X[:,q:q+1] - X2[:,q:q+1].T) for q in range(self.input_dim)])/(self.lengthscale**3).T[:,None]

    def dK_dtheta(self,X,X2=None):
        """
        The analytical derivatives come from the function update_gradients_full in GPy.kern.src.stationnary
        This function along with lengthscale_grads_pure are general to any kernel with parameters : [variance, lenghtscale(s)]. Any additional 
        parameter derivative must be added 
        return : 3D array : (nparam,Xdim,Xdim)
        """
        
        if X2 is None: X2 = X

        ##-- The variance gradient
        #self.variance.gradient = np.sum(self.K(X, X2)* dL_dK)/self.variance
        dK_dvariance = self.K(X, X2)/self.variance

        #-- the lengthscale gradient(s)
        dL_dr = self.dK_dr_via_X(X, X2) #* dL_dK ; remove this to get dK_dtheta in the end instead of dL_dtheta

        if self.ARD:
            tmp = dL_dr*self._inv_dist(X, X2)
            #self.lengthscale.gradient = self._lengthscale_grads_pure(tmp, X, X2)
            dK_dlengthscale = self.lengthscale_grads_pure(tmp, X, X2)
            grad = np.concatenate((dK_dvariance[None,:,:],dK_dlengthscale),axis=0)
        else:
            r = self._scaled_dist(X, X2)
            #self.lengthscale.gradient = -np.sum(dL_dr*r)/self.lengthscale
            dK_dlengthscale = -dL_dr*r/self.lengthscale
            grad = np.concatenate((dK_dvariance[None,:,:],dK_dlengthscale[None,:,:]),axis=0)
        return grad
    



class Exponential_MK(GPy.kern.Exponential):
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Exponential'):
        super(Exponential_MK, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)


    def lengthscale_grads_pure(self, tmp, X, X2):
        """
        Comes from the function _lengthscale_grads_pure in GPy.kern.src.stationnary, but modified such as it returns a 3D array containing 
        [dK_dl1,d_K_dl2,...]
        input : tmp : mixture of dL_dr and metric, see dK_dtheta() method
        """
        # sumation has been removed and lenghtscale dimesion adjusted to get a 3D array corresponding to the dK_dtheta
        # Note : transposed not considered as kernels are symmetric
        if X2 is None: X2 = X

        return -np.array([tmp * np.square(X[:,q:q+1] - X2[:,q:q+1].T) for q in range(self.input_dim)])/(self.lengthscale**3).T[:,None]

    def dK_dtheta(self,X,X2=None):
        """
        The analytical derivatives come from the function update_gradients_full in GPy.kern.src.stationnary
        This function along with lengthscale_grads_pure are general to any kernel with parameters : [variance, lenghtscale(s)]. Any additional 
        parameter derivative must be added 
        return : 3D array : (nparam,Xdim,Xdim)
        """
        
        if X2 is None: X2 = X

        ##-- The variance gradient
        #self.variance.gradient = np.sum(self.K(X, X2)* dL_dK)/self.variance
        dK_dvariance = self.K(X, X2)/self.variance

        #-- the lengthscale gradient(s)
        dL_dr = self.dK_dr_via_X(X, X2) #* dL_dK ; remove this to get dK_dtheta in the end instead of dL_dtheta

        if self.ARD:
            tmp = dL_dr*self._inv_dist(X, X2)
            #self.lengthscale.gradient = self._lengthscale_grads_pure(tmp, X, X2)
            dK_dlengthscale = self.lengthscale_grads_pure(tmp, X, X2)
            grad = np.concatenate((dK_dvariance[None,:,:],dK_dlengthscale),axis=0)
        else:
            r = self._scaled_dist(X, X2)
            #self.lengthscale.gradient = -np.sum(dL_dr*r)/self.lengthscale
            dK_dlengthscale = -dL_dr*r/self.lengthscale
            grad = np.concatenate((dK_dvariance[None,:,:],dK_dlengthscale[None,:,:]),axis=0)
        return grad


class Matern52_MK(GPy.kern.Matern52):
    """
    Same kernel as in GPy library but add the dK_dtheta method to get the gradient of the kernel
    """
    def __init__(self,input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Mat32'):
        super(Matern52_MK,self).__init__( input_dim, variance, lengthscale, ARD, active_dims, name)
        
    
    def lengthscale_grads_pure(self, tmp, X, X2):
        """
        Comes from the function _lengthscale_grads_pure in GPy.kern.src.stationnary, but modified such as it returns a 3D array containing 
        [dK_dl1,d_K_dl2,...]
        input : tmp : mixture of dL_dr and metric, see dK_dtheta() method
        """
        # sumation has been removed and lenghtscale dimesion adjusted to get a 3D array corresponding to the dK_dtheta
        # Note : transposed not considered as kernels are symmetric
        if X2 is None: X2 = X

        return -np.array([tmp * np.square(X[:,q:q+1] - X2[:,q:q+1].T) for q in range(self.input_dim)])/(self.lengthscale**3).T[:,None]

    def dK_dtheta(self,X,X2=None):
        """
        The analytical derivatives come from the function update_gradients_full in GPy.kern.src.stationnary
        This function along with lengthscale_grads_pure are general to any kernel with parameters : [variance, lenghtscale(s)]. Any additional 
        parameter derivative must be added 
        return : 3D array : (nparam,Xdim,Xdim)
        """
        
        if X2 is None: X2 = X

        ##-- The variance gradient
        #self.variance.gradient = np.sum(self.K(X, X2)* dL_dK)/self.variance
        dK_dvariance = self.K(X, X2)/self.variance

        #-- the lengthscale gradient(s)
        dL_dr = self.dK_dr_via_X(X, X2) #* dL_dK ; remove this to get dK_dtheta in the end instead of dL_dtheta

        if self.ARD:
            tmp = dL_dr*self._inv_dist(X, X2)
            #self.lengthscale.gradient = self._lengthscale_grads_pure(tmp, X, X2)
            dK_dlengthscale = self.lengthscale_grads_pure(tmp, X, X2)
            grad = np.concatenate((dK_dvariance[None,:,:],dK_dlengthscale),axis=0)
        else:
            r = self._scaled_dist(X, X2)
            #self.lengthscale.gradient = -np.sum(dL_dr*r)/self.lengthscale
            dK_dlengthscale = -dL_dr*r/self.lengthscale
            grad = np.concatenate((dK_dvariance[None,:,:],dK_dlengthscale[None,:,:]),axis=0)
        return grad
    

class White_MK(GPy.kern.White):
    """
    Same kernel as in GPy library but add the dK_dtheta method to get the gradient of the kernel
    """
    def __init__(self,input_dim, variance=1., active_dims=None, name='white'):
        super(White_MK,self).__init__( input_dim, variance, active_dims, name)
        
    def K(self, X, X2=None):
        if X2 is None or np.array_equal(X2,X)==True:
            return np.eye(X.shape[0])*self.variance
        else:
            return np.zeros((X.shape[0], X2.shape[0]))
        
    def dK_dtheta(self,X,X2=None):
        """
        The analytical derivatives come from the function update_gradients_full in GPy.kern.src.stationnary
        This function along with lengthscale_grads_pure are general to any kernel with parameters : [variance, lenghtscale(s)]. Any additional 
        parameter derivative must be added 
        return : 3D array : (nparam,Xdim,Xdim)
        """
        
        ##-- The variance gradient
        dK_dvariance = self.K(X)/self.variance
        
        grad = dK_dvariance[None,:,:]
    
        return grad

class Bias_MK(GPy.kern.Bias):
    def __init__(self, input_dim, variance=1., active_dims=None, name='bias'):
        super(Bias_MK, self).__init__(input_dim, variance, active_dims, name)
        
    def dK_dtheta(self,X,X2=None):

        
        ##-- The variance gradient
        dK_dvariance = self.K(X)/self.variance
        
        grad = dK_dvariance[None,:,:]
    
        return grad
    
class Poly_MK(GPy.kern.Poly):
    """
    Same kernel as in GPy library but add the dK_dtheta method to get the gradient of the kernel
    """
    def __init__(self,input_dim, variance=1., scale=1., bias=1., order=3., active_dims=None, name='poly'):
        super(Poly_MK,self).__init__( input_dim, variance, scale, bias, order, active_dims, name)
        
    
    def dK_dtheta(self,X,X2=None):
        """
        return : 3D array : (nparam,Xdim,Xdim)
        """
        
        if X2 is None: X2 = X

        ##-- The variance gradient
        #self.variance.gradient = np.sum(self.K(X, X2)* dL_dK)/self.variance
        dK_dvariance = self.K(X, X2)/self.variance
        
        dot_prod = np.dot(X, X2.T)
        A = (self.scale * dot_prod) + self.bias

        dK_dscale = self.order * dot_prod * A**(self.order-1)
        dK_dbias = self.order * A**(self.order-1)
        #dK_dorder = np.log(abs(A))* A**self.order
        grad = np.concatenate((dK_dvariance[None,:,:],dK_dscale[None,:,:],dK_dbias[None,:,:]),axis=0)
        
        return grad
    
class Coregionalize_MK(GPy.kern.Coregionalize):
    """
    Same kernel as in GPy library but add the dK_dtheta method to get the gradient of the kernel
    """
    def __init__(self, input_dim, output_dim, rank=1, W=None, kappa=None, active_dims=None, name='coregion'):
        super(Coregionalize_MK,self).__init__( input_dim, output_dim, rank, W, kappa, active_dims, name)
        
    
    def dK_dtheta(self,X,X2=None):
        """
        The analytical derivatives come from the function update_gradients_full in GPy.kern.src.stationnary
        This function along with lengthscale_grads_pure are general to any kernel with parameters : [variance, lenghtscale(s)]. Any additional 
        parameter derivative must be added 
        return : 3D array : (nparam,Xdim,Xdim)
        """
        
        #if X2 is None: X2 = X
        #-- W gradient
        n_W = np.prod(self.W.shape)
        dK_dW = np.empty((n_W,self.output_dim,self.output_dim))
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                dW = np.zeros(self.W.shape)
                dW[i,j] = 1
                tmp = jnp.dot(dW,self.W.T)
                dK_dW[i*self.W.shape[1]+j] = tmp + tmp.T

        #-- kappa gradient
        n_kappa = self.kappa.shape[0]
        dK_dkappa = np.empty((n_kappa,self.output_dim,self.output_dim))
        for i in range(n_kappa):
            tmp = np.zeros(n_kappa)
            tmp[i] = self.kappa[i]
            dK_dkappa[i] = 2*np.diag(tmp)

        grad =  np.concatenate((dK_dW,dK_dkappa),axis=0)
        
        return grad
    

class SC(Kern):
    """
    Sample covariance model with a diagonal rescaling free parameter
    """
    def __init__(self,input_dim,sample_cova , variance=1.,active_dims=None):
        super(SC, self).__init__(input_dim, active_dims, 'sample_covariance')
        #assert input_dim == 1, "For this kernel we assume input_dim=1"
        
        self.variance = Param('variance', variance)
        self.link_parameters(self.variance)
        
        self.sample_cova = sample_cova # the skeleton of the kernel, to keep fixed and/or add some small rescaling
        self.variance_dim = self.variance.size
        
    def K(self,X,X2):
        """
        In this case this method is independant of X, so can bu used only as noise kernel (no estimation at new point possible)
        Need to add a 2d interp in order to be used as a regular kernel
        """
        if X2 is None: X2 = X
        #dist2 = np.square((X-X2.T)/self.lengthscale)
     
        cov_new = self.sample_cova * np.sqrt(np.outer(self.variance,self.variance))
       
        return cov_new
    
    def dK_dtheta(self,X,X2=None):
        """
        The analytical derivatives come from the function update_gradients_full in GPy.kern.src.stationnary
        This function along with lengthscale_grads_pure are general to any kernel with parameters : [variance, lenghtscale(s)]. Any additional 
        parameter derivative must be added 
        return : 3D array : (nparam,Xdim,Xdim)
        """

        n = self.variance.size
        grad = np.zeros((n, n, n))

        # Loop through the array and set the diagonal elements to 1
        for i in range(n):
            tmp = np.zeros((n,n))   
            tmp[i,:] = self.variance
            tmp[:,i] = self.variance

            tmp *= (np.ones((n,n))+np.eye(n))
            grad[i,:,:] = self.sample_cova * tmp /(2*np.sqrt(np.outer(self.variance,self.variance)))
        
    
        return grad
    
    def Kdiag(self,X):
        return self.sigma*np.diag(self.sample_cova)
    
    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X

        dvar = self.sample_cova
 
        self.variance.gradient = np.sum(dvar*dL_dK)
    
    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = np.sum(dL_dKdiag)
        
    def Kdiag(self,X):
        return self.sigma*np.diag(self.sample_cova)
    
    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X

        dvar = self.sample_cova
 
        self.variance.gradient = np.sum(dvar*dL_dK)
    
    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = np.sum(dL_dKdiag)
        
     
    

class Gibbs_MK(GPy.kern.RBF):
    def __init__(self, input_dim, variance=1., lengthscale=None,ARD=False, active_dims=None, name='Gibbs',steep=[0],trans=[0],slope=[0],n_logistic=1):
        super().__init__(input_dim, variance, lengthscale,ARD, active_dims, name)
  
        #if input_dim>=2 or ARD==True:
        #    print('Error : This kernel operates only in 1D for now')
        self.n_logistic=n_logistic
        for i in range(n_logistic):
            setattr(self, 'steep_'+str(i), GPy.core.parameterization.Param('steep_'+str(i), steep))
            self.link_parameter(self.__dict__[f'steep_{i}'])
            setattr(self, 'trans_'+str(i), GPy.core.parameterization.Param('trans_'+str(i), trans))
            self.link_parameter(self.__dict__[f'trans_{i}'])
            setattr(self, 'slope_'+str(i), GPy.core.parameterization.Param('slope_'+str(i), slope))
            self.link_parameter(self.__dict__[f'slope_{i}'])
            
       # #for i in range(n_logistic):
        #    #globals()[f"my_variable_{i}"] = i
        #    __setattr__(self,'name',i)
            #X_i = self.__dict__[f'X_{i}']

            

        #self.steep = GPy.core.parameterization.Param('steep', steep)
        #self.trans = GPy.core.parameterization.Param('trans', trans)
        #self.slope = GPy.core.parameterization.Param('slope', slope)
        #self.link_parameter(self.steep)
        #self.link_parameter(self.trans)
        #self.link_parameter(self.slope)    

        #self.base = self.lengthscale.values[0]
        
    def wrap_func(self,x,n=0,index_theta=None):
        """
        Sigmoid
        base controls the y-value of the sigmoid's midpoint.
        steep controls the steepness of the sigmoid
        trans controls the x-value of the sigmoid's midpoint
        slope controls the scaling of the difference between the minimum and maximum values,
        slope = 0 <=> constant lenghscales equal to base
        n : 0 or 1: derivative order of the wrap function
        index_theta : if n=1, index of the derivative parameter  
        """
        if n==0:
            res=0
            for i in range(self.n_logistic):
                    res += self.__dict__[f'slope_{i}'] / (1 + np.exp(-self.__dict__[f'steep_{i}']*(x-self.__dict__[f'trans_{i}'])))  
            res += self.lengthscale
            return res
        
        if n==1:            
            if index_theta==0: #base <=> lenghtscale                    
                return np.ones(x.shape)
            elif index_theta%3==1: #steep
                res = self.__dict__[f'slope_{index_theta//3}'] * (x-self.__dict__[f'trans_{index_theta//3}']) * np.exp(-self.__dict__[f'steep_{index_theta//3}']*
                                    (x-self.__dict__[f'trans_{index_theta//3}'])) / (1 + np.exp(-self.__dict__[f'steep_{index_theta//3}']*(x-self.__dict__[f'trans_{index_theta//3}'])))**2
                return res
            
            elif index_theta%3==2:#trans
                res = -self.__dict__[f'slope_{index_theta//3}']*self.__dict__[f'steep_{index_theta//3}'] * self.__dict__[f'steep_{index_theta//3}'] * np.exp(-self.__dict__[f'steep_{index_theta//3}']*
                                            (x-self.__dict__[f'trans_{index_theta//3}'])) / (1 + np.exp(-self.__dict__[f'steep_{index_theta//3}']*(x-self.__dict__[f'trans_{index_theta//3}'])))**2
                return res
            
            elif index_theta%3==0:#slope
                res = 1 / (1 + np.exp(-self.__dict__[f'steep_{index_theta//3-1}']*(x-self.__dict__[f'trans_{index_theta//3-1}'])))
                return res


        #if n==1:
        #    if index_theta==0: #base <=> lenghtscale
        #        return x/x
        #    if index_theta==1: #steep
        #        return self.slope * (x-self.trans) * np.exp(-self.steep*(x-self.trans)) / (1 + np.exp(-self.steep*(x-self.trans)))**2
        #    if index_theta==2: #trans
        #        return -self.slope * self.steep * np.exp(-self.steep*(x-self.trans)) / (1 + np.exp(-self.steep*(x-self.trans)))**2
        #    if index_theta==3: #slope
        #        return 1 / (1 + np.exp(-self.steep*(x-self.trans)))
            

    def _scaled_dist(self, X, X2=None):
        """
        Efficiently compute the scaled distance, r.
        ..math::
            r = \sqrt( \sum_{q=1}^Q (x_q - x'q)^2/l_q^2 )
        Note that if thre is only one lengthscale, l comes outside the sum. In
        this case we compute the unscaled distance first (in a separate
        function for caching) and divide by lengthscale afterwards
        """
        
        if X2 is None:
            X2=X
            
        if self.ARD==False:    
            self.l1 = self.wrap_func(X).flatten()
            self.l2 = self.wrap_func(X2).flatten()

            self.l_dist = self.l1[:,None]**2+self.l2[None,:]**2
            self.norm = np.sqrt(2*( self.l1[:,None]*self.l2[None,:])/ self.l_dist)
            
            return self._unscaled_dist(X, X2)/np.sqrt(self.l_dist) 
        
        else :
            self.L_l1 = self.wrap_func(X) #--2D array L[:,0] are the values of the first lenghtscale
            self.L_l2 = self.wrap_func(X2)
            
            self.l_dist = (self.L_l1[:,None]**2+self.L_l2[None,:]**2)
            self.norm = np.prod(np.sqrt(2*(self.L_l1[:,None]*self.L_l2[None,:])/self.l_dist),axis=2)
            
            return self._scaled_dist_multidim(X, X2)
        
        
    
    
    def _scaled_dist_multidim(self, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        Valid for Gibbs kernel with multidimensional lenghtscales
        """
        if X2 is None:
            X2=X
        Xsq = (X[:,None]**2+X2[None,:]**2) / self.l_dist
        r2 = -2*(X[:,None]*X2[None,:])/ self.l_dist + Xsq
        r2 = np.sum(r2,axis=2)
        np.fill_diagonal(r2, 0) # force diagnoal to be zero: sometime numerically a little negative
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)

    def dK_dtheta(self,X,X2=None):
        """
        The analytical derivatives
        return : 3D array : (nparam,Xdim,Xdim)
        """
        
        if X2 is None: X2 = X

        #-- The variance gradient
        dK_dvariance = self.K(X, X2)/self.variance
        grad = dK_dvariance[None,:,:]
        
        L_l1 = self.wrap_func(X)
        L_l2 = self.wrap_func(X2)
        #--The wraping function gradient
        for i in range(len(self.parameter_names())-1): # -1 because of the variance parameter
            L_dl1 = self.wrap_func(X,n=1,index_theta=i)
            L_dl2 = self.wrap_func(X2,n=1,index_theta=i)
            
            for dim in range(self.input_dim):  #-- Deal with the dimensions one by one for each parameter type
                dl1 = L_dl1[:,dim:dim+1]
                l1 = L_l1[:,dim:dim+1]
                dl2 = L_dl2[:,dim:dim+1]
                l2 = L_l2[:,dim:dim+1]

                dK_wrap_i = ((l1**2+l2.T**2) / (2*l1*l2.T)) * ((l1**2 + l2.T**2)* ( (dl1*l2.T) +(l1*dl2.T) )- l1*l2.T *( 2*(dl1*l1) +2*(dl2*l2).T ) )/(l1**2+l2.T**2)**2 
                dK_wrap_i += (X[:,dim:dim+1] - X2[:,dim:dim+1].T)**2 * 2*((dl1*l1) + (dl2*l2).T) / (l1**2+l2.T**2)**2
                dK_wrap_i *= self.K(X,X2)

                grad = np.concatenate((grad,dK_wrap_i[None,:,:]),axis=0)

        return grad


    def K_of_r(self, r):
        return self.variance* np.exp(- r**2)*self.norm 
    

""" ANISOTROPIC PART """


class Anisotropic_Matern32_MK(GPy.kern.Matern32):
    """
    Same kernel as in GPy library but add the dK_dtheta method to get the gradient of the kernel
    
    NOTE : ARD SHOULD ALWAYS BE TRUE WITH THIS KERNEL
    """
    def __init__(self,input_dim, variance=1.,lengthscale=None, ARD=True, active_dims=None, name='Mat32',mat=None,rank=1):
        super(Anisotropic_Matern32_MK,self).__init__( input_dim, variance, lengthscale, ARD, active_dims, name)
        
       # #--initialize anisotropy to zero
       # for i in range(input_dim-1):
       #     tmp = np.random.random(input_dim-1-i)*2-1            
       #     setattr(self, 'U_row'+str(i), GPy.core.parameterization.Param('U_row'+str(i), tmp))
       #     self.link_parameter(self.__dict__[f'U_row{i}'])
        self.rank=rank    
        W = 0.5*np.random.randn(self.input_dim, self.rank)/np.sqrt(self.rank)
        self.W = Param('U_row',W)
        self.link_parameters(self.W)
        
    def _scaled_dist(self, X, X2=None):
        """
        Efficiently compute the scaled distance, r.

        ..math::
            r = \sqrt( \sum_{q=1}^Q (x_q - x'q)@precision@ (x_q - x'q).T )

       Overwrite the parent method to allow more flexible models

        U : upper triangle matrix
        [Discovering hidden features with Gaussian processes regression F.Vivarelli et al]
        """
        
        #U = np.zeros((self.input_dim,self.input_dim))
        #for i in range(self.input_dim-1):
        #    U[i,i+1:] = self.__dict__[f'U_row{i}']
        #    U[i,i] = 1 / self.lengthscale[i] # Need positive diagonal value
        #U[-1,-1] = 1 / self.lengthscale[-1]
        #self.U = U
        #precision = np.dot(self.U .T,self.U )

                
        precision = np.dot(self.W,self.W.T) + np.diag(self.lengthscale**-2)
        self.precision=precision
        
        if X2 is None:
            return sc.spatial.distance.cdist(X,X ,metric='mahalanobis',VI=precision)#numba_autocounter(X,precision)
        else:
            return sc.spatial.distance.cdist(X,X2, metric='mahalanobis',VI=precision)#numba_crosscounter(X,X2,precision)
        

    def dK_dtheta(self,X,X2=None):
        """
        The analytical derivatives come from the function update_gradients_full in GPy.kern.src.stationnary
        This function along with lengthscale_grads_pure are general to any kernel with parameters : [variance, lenghtscale(s)]. Any additional 
        parameter derivative must be added 
        return : 3D array : (nparam,Xdim,Xdim)
        """
        
        #if X2 is not None: print('gradient not implemented for X2 not None')
        if X2 is None: X2 = X
        
        K = self.K(X, X2)
        output_dim = K.shape
        
        r = self._scaled_dist(X,X2)        
    
        #--variance gradient
        dK_dvariance = K/self.variance
        
        #--lenghtscales gradient
        dK_dlengthscale = np.empty((self.input_dim,output_dim[0],output_dim[1]))
        for i in range(self.input_dim):
            dU  = np.zeros((self.input_dim,self.input_dim))
            #dU[i,i] = self.lengthscale[i]**-2#self.lengthscale[i]*np.exp(self.lengthscale[i])
            #tmp = dU.T@self.U + self.U.T@dU
            dU[i,i] = 2*self.lengthscale[i]**-3
            tmp = dU.copy()
            dr_dlengthscale = -0.5*sc.spatial.distance.cdist(X,X2, metric='mahalanobis',VI=tmp)**2 
            dK_dlengthscale[i,:,:] =  -3.*self.variance*np.exp(-np.sqrt(3.)*r) * dr_dlengthscale #dK_dr * r / r * dr_dl
          
        #--U gradient set to zero
        
        n_U = int(self.input_dim*self.rank)        
        dK_dW = np.empty((n_U,output_dim[0],output_dim[1]))
        for i in range(self.input_dim):
            for j in range(self.rank):
                #dU = np.zeros(self.U.shape)
                #dU[i,j] = 1
                #self.dU=dU
                #tmp = dU.T@self.U + self.U.T@dU
                #dr_dU = 0.5*sc.spatial.distance.cdist(X,X2, metric='mahalanobis',VI=tmp)**2
                #dK_dW[i+j-1] =  -3.*self.variance*np.exp(-np.sqrt(3.)*r) * dr_dU
                dK_dW[i*self.rank+j] = 0
        grad =  np.concatenate((dK_dvariance[None,:,:],dK_dlengthscale,dK_dW),axis=0)
        
        return grad
        

class Anisotropic_RatQuad_MK(GPy.kern.RatQuad):
    """
    Same kernel as in GPy library but add the dK_dtheta method to get the gradient of the kernel
    
    NOTE : ARD SHOULD ALWAYS BE TRUE WITH THIS KERNEL
    """
    def __init__(self, input_dim, variance=1., lengthscale=None, power=2., ARD=True, active_dims=None, name='RatQuad',rank=1):
        super(Anisotropic_RatQuad_MK,self).__init__( input_dim, variance, lengthscale,power, ARD, active_dims, name)
        
       # #--initialize anisotropy to zero
       # for i in range(input_dim-1):
       #     tmp = np.random.random(input_dim-1-i)*2-1            
       #     setattr(self, 'U_row'+str(i), GPy.core.parameterization.Param('U_row'+str(i), tmp))
       #     self.link_parameter(self.__dict__[f'U_row{i}'])
        self.rank=rank    
        W = 0.5*np.random.randn(self.input_dim, self.rank)/np.sqrt(self.rank)
        self.W = Param('U_row',W)
        self.link_parameters(self.W)
        
    def _scaled_dist(self, X, X2=None):
        """
        Efficiently compute the scaled distance, r.

        ..math::
            r = \sqrt( \sum_{q=1}^Q (x_q - x'q)@precision@ (x_q - x'q).T )

       Overwrite the parent method to allow more flexible models

        U : upper triangle matrix
        [Discovering hidden features with Gaussian processes regression F.Vivarelli et al]
        """
                
        precision = np.dot(self.W,self.W.T) + np.diag(self.lengthscale**-2)
        self.precision=precision
        
        if X2 is None:
            return sc.spatial.distance.cdist(X,X ,metric='mahalanobis',VI=precision)#numba_autocounter(X,precision)
        else:
            return sc.spatial.distance.cdist(X,X2, metric='mahalanobis',VI=precision)#numba_crosscounter(X,X2,precision)
        

    def dK_dtheta(self,X,X2=None):
        """
        The analytical derivatives come from the function update_gradients_full in GPy.kern.src.stationnary
        This function along with lengthscale_grads_pure are general to any kernel with parameters : [variance, lenghtscale(s)]. Any additional 
        parameter derivative must be added 
        return : 3D array : (nparam,Xdim,Xdim)
        """
        
        #if X2 is not None: print('gradient not implemented for X2 not None')
        if X2 is None: X2 = X
        
        K = self.K(X, X2)
        output_dim = K.shape
        
        r = self._scaled_dist(X,X2)        
    
        #--variance gradient
        dK_dvariance = K/self.variance
        
        #--lenghtscales gradient
        dK_dlengthscale = np.empty((self.input_dim,output_dim[0],output_dim[1]))
        for i in range(self.input_dim):
            dU  = np.zeros((self.input_dim,self.input_dim))
            dU[i,i] = 2*self.lengthscale[i]**-3
            tmp = dU.copy()
            dr_dlengthscale = -0.5*sc.spatial.distance.cdist(X,X2, metric='mahalanobis',VI=tmp)**2 
            dK_dlengthscale[i,:,:] =  -self.variance*self.power*np.exp(-(self.power+1)*np.log1p(r**2/2.)) * dr_dlengthscale #dK_dr * r / r * dr_dl
        
         #-- Power parameter gradient
        dK_dpow = -self.variance * np.exp(self.power*(np.log(2.)-np.log1p(r**2+1)))*np.log1p(r**2/2.) 

        #--U gradient set to zero
        
        n_U = int(self.input_dim*self.rank)        
        dK_dW = np.empty((n_U,output_dim[0],output_dim[1]))
        for i in range(self.input_dim):
            for j in range(self.rank):
                dK_dW[i*self.rank+j] = 0
        
       
        grad =  np.concatenate((dK_dvariance[None,:,:],dK_dlengthscale,dK_dpow[None,:,:],dK_dW),axis=0)

        return grad
        
        
        
        
class Anisotropic_MultiKroModel(MultiKroModel): 
    """
    Analytical gradient for anisotropic kernels requires complexe parametrization, thus the gradient for those parameters
    is computed numerically
    """
    def __init__(self,L_X,Y,L_kernel,L_noise,nugget_s=1e-15,nugget_n=None,jax_enable_x64=True,name='MultiKroModel',epsilon=1e-6):
        super(Anisotropic_MultiKroModel,self).__init__(L_X,Y,L_kernel,L_noise,nugget_s,nugget_n,jax_enable_x64,name)
        
        self.epsilon = epsilon #--ncrement to use for determining the numerical gradients
        
    def parameters_changed(self):
        """
        Evaluates the log_marginal_likelihood, computes and updates the gradient for every hyperparameters of the model
        Correct only when the kernels have indep parameters. See [Saatci 2012]
        """
        #-- First computed the log likelihood:
        self.log_marginal_likelihood = self.get_log_likelihood()

        #-- List of the projective matrix Q@S_inv@U for every dimension
        L_H_tild = [jnp.dot(jnp.dot(self.L_U[j],self.L_S_inv[j]),self.L_Q[j]) for j in range(len(self.L_Q))]

        #--loop over the subspaces to update the noise and signal parameters gradients
        for i in range(len(self.L_kernel)):
            X_i = self.__dict__[f'X_{i}']

             #-- First create a list with every flat parameters object to check if fixed
            L_flat_param_kernel=[]
            for el in self.L_kernel[i].flattened_parameters:
                if el.ndim==1:
                    for j in range(el.shape[0]):
                        L_flat_param_kernel.append(el[j:j+1])
                else:
                    for j in range(el.shape[0]):
                        for l in range(el.shape[1]):
                            L_flat_param_kernel.append(el[j,l:l+1])


            #-- First create a list with every flat parameters object to check if fixed
            L_flat_param_noise=[]
            for el in self.L_noise[i].flattened_parameters:
                if el.ndim==1:
                    for j in range(el.shape[0]):
                        L_flat_param_noise.append(el[j:j+1])
                else:
                    for j in range(el.shape[0]):
                        for l in range(el.shape[1]):
                            L_flat_param_noise.append(el[j,l:l+1])


            #-- Compute the kernels derivatives             
            dK_dtheta = self.L_kernel[i].dK_dtheta(X_i) #-- gradients of the kernels : (nparam_i,d_i,d_i)
            dN_dtheta = self.L_noise[i].dK_dtheta(X_i)

            dL_dtheta_K = []
            dL_dtheta_N = []

            #--Need 2 diff loops for noise and signal kernels because the number of param can be different
            for p_k in range(dK_dtheta.shape[0]):
                #-- if the param is fixed  don't change the gradient
                if L_flat_param_kernel[p_k]._has_fixes()==True :
                    dL_dtheta_K.append(L_flat_param_kernel[p_k].gradient[0])
                    self.L_flat_param_kernel=L_flat_param_kernel
                
                else:
                    #-- if the param is anisotropic  numerical gradient
                    if L_flat_param_kernel[p_k].name[:5]=='U_row': 
                        def tmp_grad(param):
                            self.L_kernel[i].param_array[p_k] = param
                            return self.log_likelihood()
                        
                        tmp = sc.optimize.approx_fprime(L_flat_param_kernel[p_k][0],tmp_grad,epsilon=self.epsilon)
                        self.L_kernel[i].param_array[p_k] -= self.epsilon # numerical gradient modifies the value of the parameter, need to correct
                        dL_dtheta_K.append(tmp)
                        
                    else:
                        #-- compute the first part of the gradient
                        L_K = self.L_K.copy()  #--Complet list of matrix kernels
                        L_K.pop(i)  #--Remove the ith kernel (to be replaced by the gradient)
                        L_K.insert(i,dK_dtheta[p_k]) 
                        grad_dat = jnp.dot(0.5*self.alpha.T,kron_mvprod(L_K,self.alpha))

                        #--gradient for logdet
                        L_Lambda = self.L_Lambda.copy()
                        L_Lambda.pop(i) 
                        L_Lambda.insert(i, jnp.diag(jnp.dot(L_H_tild[i].T,jnp.dot(dK_dtheta[p_k],L_H_tild[i]))) ) 

                        #grad_det = -0.5 * np.sum(self.W_inv * ft.reduce(np.kron, [np.diag(np.dot(H_tild_i.T,np.dot(L_K_i,H_tild_i))) for H_tild_i,L_K_i in zip(L_H_tild,L_K)] ))
                        grad_det = -0.5 * jnp.sum(self.W_inv * ft.reduce(jnp.kron, L_Lambda ))                
                        dL_dtheta_K.append(np.float(grad_dat+grad_det))

            for p_n in range(dN_dtheta.shape[0]):

                #-- if the param is fixed or anisotropic, don't change the gradient
                if L_flat_param_noise[p_n]._has_fixes()==True or L_flat_param_noise[p_n].name[:5]=='U_row':
                    dL_dtheta_N.append(L_flat_param_noise[p_n].gradient[0])

                else:
                    #-- if the param is anisotropic  numerical gradient
                    if L_flat_param_noise[p_n].name[:5]=='U_row':   
                        def tmp_grad(param):
                            self.L_noise[i].param_array[p_n] = param
                            return self.log_likelihood()
                        
                        tmp = sc.optimize.approx_fprime(L_flat_param_noise[p_n][0],tmp_grad,epsilon=self.epsilon)
                        self.L_noise[i].param_array[p_n] -= self.epsilon # numerical gradient modifies the value of the parameter, need to correct
                        dL_dtheta_N.append(tmp)
                    else:
                        #-- compute the first part of the gradient
                        L_N = self.L_N.copy()
                        L_N.pop(i)
                        L_N.insert(i,dN_dtheta[p_n])
                        grad_dat = 0.5*jnp.dot(self.alpha.T,kron_mvprod(L_N,self.alpha))

                        #--gradient for logdet
                        L_I = [jnp.ones(el) for el in self.L_n_samples]
                        L_I.pop(i)
                        L_I.insert(i,jnp.diag(jnp.dot(L_H_tild[i].T,jnp.dot(dN_dtheta[p_n],L_H_tild[i]))))
                        #grad_det = -0.5 * self.W_inv @ ft.reduce(np.kron, [np.diag(np.dot(H_tild_i.T,np.dot(L_N_i,H_tild_i))) for H_tild_i,L_N_i in zip(L_H_tild,L_N)] ) 
                        grad_det = -0.5 * jnp.sum(self.W_inv * ft.reduce(jnp.kron, L_I ) )
                        dL_dtheta_N.append(np.float(grad_dat+grad_det))

            self.L_kernel[i].dL_dtheta_K = dL_dtheta_K 

            #-- update the objective function gradient for every parameter
            self.L_kernel[i].gradient = dL_dtheta_K
            self.L_noise[i].gradient = dL_dtheta_N    

