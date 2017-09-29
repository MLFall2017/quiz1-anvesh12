
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
#from sklearn.decomposition import PCA
from matplotlib.mlab import PCA

def d_PCA(x, corr_logic):
    columnMean = x.mean(axis=0)
    columnMeanAll = np.tile(columnMean, reps=(x.shape[0], 1))
    xMeanCentered = x - columnMeanAll

    # use mean_centered data or standardized mean_centered data
    if not corr_logic:
        dataForPca = xMeanCentered
    else:
        dataForPca = x

    # get covariance matrix of the data
    covarianceMatrix = np.cov(dataForPca, rowvar=False)

    # eigendecomposition of the covariance matrix
    eigenValues, eigenVectors = LA.eig(covarianceMatrix)
    II = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[II]
    eigenVectors = eigenVectors[:, II]

    # get scores
    pcaScores = np.matmul(dataForPca, eigenVectors)

    # collect PCA results
    pcaResults = {'data': x,
                   'mean_centered_data': xMeanCentered,
                   'PC_variance': eigenValues,
                   'loadings': eigenVectors,
                   'scores': pcaScores}

    return pcaResults





import pandas as pd
Data=pd.read_csv("G:\Fall 2017\ML\Datasets\SCLC_study_output_filtered.csv",header=1)
Data_matrix=pd.DataFrame.as_matrix(Data)
DataforPCA=Data.drop(Data.columns[[0 ]], axis=1)
var=np.var(DataforPCA)
total=0
for V in var: 
        total += V
total

# Total Variance of variables =3256847890374.8579

useCorr = True

PCA_bef=d_PCA(DataforPCA, useCorr)
PCA_Scores=PCA_bef['scores']
var1=np.var(PCA_Scores)

# Total Variance of PCA's before standardise =450395938060.38885

PC1=PCA_Scores[0]
PC2=PCA_Scores[1]
Cov=np.cov(PC1,PC2)
CovPC12=Cov[0, 1] 

#Covariance between PC1 and PC2=549553216190.27454+4.4383746019939983e-09j

   
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('scores plot for PC1 and PC2 before standardise ')
    ax.scatter(PC1, PC2, color='red')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.show()


PC_Var=PCA_bef['PC_variance']

percentVarianceExplained1 = 100 *( PC_Var[0] / sum(PC_Var))
#39.060720724877299+0j
percentVarianceExplained2 = 100 *( PC_Var[1] / sum(PC_Var))
#15.031786284716752+0j
percentVarianceExplained3 = 100 *( PC_Var[2] / sum(PC_Var))
#13.389976320162678+0j
percentVarianceExplained4 = 100 *( PC_Var[3] / sum(PC_Var))
#9.8077822046714704+0j

#Four PCs are needed for 75% variance

Loadings=PCA_bef['loadings']
fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('loadings plot')
    ax.scatter(PCA_bef['loadings'][:,0],PCA_bef['loadings'], color='blue')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.show()

useCor = False

PCA_aft=d_PCA(DataforPCA, useCor)
PCA_Scores1=PCA_aft['scores']
var2=np.var(PCA_Scores1)
#Total Variance after Standaridise=66466283477.037933