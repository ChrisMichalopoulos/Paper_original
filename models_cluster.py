import numpy as np
from numba import jit,prange
from tqdm import tqdm
import model_acc as m
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
import math as ma

def  KNNtimeseries(mesured,train_set,test_set,k=15,mean=True):
    @jit(nopython=True,parallel=True)
    def KNNregressor(train_set,test_set,k=15,mean=True,n=1):
        
        #data separtation
        
        compare=train_set[:,:3]
        knowned_result=train_set[:,3]
        compared=test_set[:,:3]
        
        #size
        x=compared.shape[0]
        y=compare.shape[0]
    
        
        
        rangematrix  = np.zeros((y, x))
        nearest      = np.full((y, x), -1)
        predict      = np.zeros((x,))
    
        
        
        
        if mean == True:
            for i in prange(x):
                for ii in range(y):
                    rangematrix[ii,i]=np.linalg.norm(compare[ii,:]-compared[i,:])
                nearest[:,i]=np.argsort(rangematrix[:,i])
                predict[i]=np.sum(knowned_result[nearest[:k,i]])/k
            
            
        # else:
        #     for i in prange(x):
        #         for ii in range(y):
        #             rangematrix[ii,i]=np.linalg.norm(compare[ii,:]-compared[i,:])
        #         nearest[:,i]=np.argsort(rangematrix[:,i])
                
        #         nearest_temp=nearest[:k,i]
                
        #         predict[i]=np.sum(knowned_result[nearest_temp]*(rangematrix[nearest_temp,i]**(-n)/np.sum(rangematrix[nearest_temp,i]**(-n)) ))
        
        return predict
        
        
        
    x,y=test_set.shape
    u,v=train_set.shape
    results=np.zeros((x,y))
        
    time_series=np.concatenate((train_set,test_set),axis=1)
        
        
    for i in tqdm(range(y)):
        results[:,i]= KNNregressor(mesured[:,i+v-4:i+v],time_series[:,i+v-4:i+v],k=k,mean=mean)
        
            
        
        
    er=m.errors(test_set,results)
    
    return er.experrors()
        
def GMM(mesured,train_set,test_set,k=15,cov="spherical"):     
    def fill_matrix(X, mixture,cov):
        
    
        def log_norm_pdf_multivariate(x,mu,sigma):
            
            size=x.shape[0]
            
            if size == mu.shape[0] and (size,size)==sigma.shape:
                det=np.linalg.det(sigma)
                if det==0:
                    raise NameError("The covariance matrix cant be singular")
                
                norm_const= -np.log(    ma.pow((2*ma.pi),float(size)/2) * ma.pow(det,1/2)     )
                
                x_mu = x - mu
                
                inv = np.linalg.inv(sigma)
                result = -0.5 *(x_mu @ inv @ x_mu.T)
                return norm_const + result 
            else:
                raise NameError("The dimensions of the input dont match")
            
            
        def log_gaussian(x,mean,var):
            
            d=len(x)
            
            log_prob= - d/2.0*np.log(2*np.pi*var)
            log_prob -= 0.5*((x-mean)**2).sum()/var
            return log_prob
    
    
    
    
    
    
        n,d= X.shape
        X_pred=X.copy()
        K,_=mixture[0].shape
        
        if cov=="spherical":
            for i in range(n):
                post=np.zeros(K)
                for j in range(K):
                    
                    log_likelihood=log_gaussian(X[i,:3], mixture[0][j,:3], mixture[1][j])
                    post[j]= np.log(mixture[2][j])+log_likelihood
                
                post=np.exp(post-logsumexp(post))
                X_pred[i,3]=np.dot(post,mixture[0][:,3])
    
            return X_pred
        
        if cov=="tied":
            for i in range(n):
                post=np.zeros(K)
                for j in range(K):
                    log_likelihood=log_norm_pdf_multivariate(X[i,:3],mixture[0][j,:3],mixture[1][:3,:3])
                    post[j]=np.log(mixture[2][j])+log_likelihood
                    
                post=np.exp(post-logsumexp(post))
                X_pred[i,3]=np.dot(post,mixture[0][:,3])
    
    
            return X_pred
        
        
        if cov=="diag":
            temp=np.zeros((K,d,d))
            for i in range(K):
                temp[i,:,:]=mixture[1][i]*np.identity(d)
            mixture=(mixture[0],temp,mixture[2])
        
        if cov=="diag" or cov== "full":
            for i in range(n):
                post=np.zeros(K)
                for j in range(K):
                    log_likelihood=log_norm_pdf_multivariate(X[i,:3],mixture[0][j,:3],mixture[1][j][:3,:3])
                    post[j]=np.log(mixture[2][j])+log_likelihood
                post=np.exp(post-logsumexp(post))
                X_pred[i,3]=np.dot(post,mixture[0][:,3])
                
            return X_pred
    
        
    
    
    
    x,y=test_set.shape
    u,v=train_set.shape
    results=results=np.zeros((x,y))
    time_series=np.concatenate((train_set,test_set),axis=1)
    for i in range(y):
        
        Filled=time_series[:,i+v-3:i+v+1]
        Filled[:,3]=0
        
        gmm=GaussianMixture(n_components=k,n_init=1,max_iter=10000000000000,tol=1e-5,covariance_type=cov).fit(mesured[:,i+v-3:i+v+1])
        
        mixture=(gmm.means_,gmm.covariances_,gmm.weights_)
        rslts=fill_matrix(Filled,mixture,cov)
        
        
        results[:,i]=rslts[:,3]
        
        
    
            
    er=m.errors(test_set,results)
    
    return er.experrors()
        
    
    
    
    
    
    


