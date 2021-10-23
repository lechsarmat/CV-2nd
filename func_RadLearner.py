import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def DataTransformation(df):
    """Converts pandas dataframe into two numpy arrays.
       >>> DataTransformation(pd.read_csv( "data/mini_test.csv" ))[0]
       array([[-0.5,  0. ],
              [ 0.5,  0. ]])
       >>> DataTransformation(pd.read_csv( "data/mini_test.csv" ))[1]
       array([[0., 0.]])
       >>> DataTransformation(0)
       Traceback (most recent call last):
        ...
       TypeError: df must be pandas dataframe
    """
    if not isinstance( df, pd.DataFrame ):
        raise TypeError( "df must be pandas dataframe" )
    Q0 = df[df.label == 0]
    Q1 = df[df.label == 1]
    return np.array([Q0.x,Q0.y]).T, np.array([Q1.x,Q1.y]).T

def LearningAlg(seq0, seq1):
    """Implements perceptron learning algorithm on given data.
       >>> LearningAlg(np.array([[-0.5, 0.],[ 0.5, 0.]]), np.array([[0., 0.]]))
       array([ 1.,  0.,  0., -4.,  0., -1.])
       >>> LearningAlg(0, 0)
       Traceback (most recent call last):
        ...
       TypeError: seq0 and seq1 must be numpy array
       >>> LearningAlg(np.array([[ 0., 'a']]), np.array([[0., 0.]]))
       Traceback (most recent call last):
        ...
       TypeError: wrong type of elements
       >>> LearningAlg(np.array([[-0.5, 0.],[ 0.5, 0.]]), np.array([[0., 0., 0.]]))
       Traceback (most recent call last):
        ...
       ValueError: wrong shape of seq0 or seq1
    """
    if type(seq0) != np.ndarray or type(seq1) != np.ndarray:
        raise TypeError( "seq0 and seq1 must be numpy array" )
    if seq0.dtype != 'float64' or seq1.dtype != 'float64':
        raise TypeError( "wrong type of elements" )
    if seq0.shape[1] != 2 or seq1.shape[1] != 2:
        raise ValueError( "wrong shape of seq0 or seq1" )
    X = np.concatenate((seq0, seq1), axis = 0)
    Y = np.concatenate((np.ones((len(X),1)), X, np.array([X.T[0]]).T ** 2, np.array([X.T[0]]).T
                        * np.array([X.T[1]]).T, np.array([X.T[1]]).T ** 2), axis = 1)
    y_real = np.where(np.arange(len(seq0) + len(seq1)) < len(seq0), 0, 1)
    W = np.zeros(Y.shape[1])
    y_pred = (np.dot(Y,W) > 0)
    while True:
        while np.sum(y_pred != y_real) != 0:
            W = W + np.sum((y_real - y_pred).reshape((len(Y),1)) * Y, axis = 0)
            y_pred = (np.dot(Y,W) > 0)
        InvSigma = np.array([[-2 * W[3], -W[4]], [-W[4], -2 * W[5]]])
        eigval, eigvec = np.linalg.eig(InvSigma)
        EigStr = np.array([[0., 0., 0., -eigvec.T[0][0] * np.conj(eigvec.T[0][0]),
        -(eigvec.T[0][0] * np.conj(eigvec.T[0][1]) + eigvec.T[0][1] * np.conj(eigvec.T[0][0]))/2,
        -eigvec.T[0][1] * np.conj(eigvec.T[0][1])], [0., 0., 0., -eigvec.T[1][0] * np.conj(eigvec.T[1][0]),
        -(eigvec.T[1][0] * np.conj(eigvec.T[1][1]) + eigvec.T[1][1] * np.conj(eigvec.T[1][0]))/2,
        -eigvec.T[1][1] * np.conj(eigvec.T[1][1])]])
        if np.sum((np.dot(EigStr,W) > 0) != 1) != 0:
            W = W + np.sum((1. - (np.dot(EigStr,W) > 0)).reshape((2,1)) * EigStr, axis = 0)
        else:
            break
    return W

def Classifier(seq0, seq1, seq2):
    """Classifies given data.
       >>> Classifier(np.array([[-0.5, 0.],[0.5, 0.]]), np.array([[0., 0.]]), np.array([[0., -1.],[0., 0.5]]))
       array([[0]], dtype=int64)
       >>> Classifier(0, 0, 0)
       Traceback (most recent call last):
        ...
       TypeError: seq0, seq1 and seq2 must be numpy array
       >>> Classifier(np.array([[0.5, 'a']]), np.array([[0., 0.]]), np.array([[0., 0.5]]))
       Traceback (most recent call last):
        ...
       TypeError: wrong type of elements
       >>> Classifier(np.array([[0.5, 0.]]), np.array([[0., 0., 0.]]), np.array([[0., 0.5]]))
       Traceback (most recent call last):
        ...
       ValueError: wrong shape of seq0, seq1 or seq2
    """
    if type(seq0) != np.ndarray or type(seq1) != np.ndarray or type(seq2) != np.ndarray:
        raise TypeError( "seq0, seq1 and seq2 must be numpy array" )
    if seq0.dtype != 'float64' or seq1.dtype != 'float64' or seq2.dtype != 'float64':
        raise TypeError( "wrong type of elements" )
    if seq0.shape[1] != 2 or seq1.shape[1] != 2 or seq2.shape[1] != 2:
        raise ValueError( "wrong shape of seq0, seq1 or seq2" )
    W = LearningAlg(seq0, seq1)
    Y = np.concatenate((np.ones((len(seq2),1)), seq2, np.array([seq2.T[0]]).T ** 2, np.array([seq2.T[0]]).T
                        * np.array([seq2.T[1]]).T, np.array([seq2.T[1]]).T ** 2), axis = 1)
    y = (np.dot(Y,W) > 0)
    return np.argwhere(y == 0)

def DrawGraph(seq0, seq1, seq2 = np.array([])):
    """Displays classification of given data.
       >>> DrawGraph(0, 0)
       Traceback (most recent call last):
        ...
       TypeError: seq0, seq1 and seq2 must be numpy array
       >>> DrawGraph(np.array([[0.5, 'a']]), np.array([[0., 0.]]))
       Traceback (most recent call last):
        ...
       TypeError: wrong type of elements
       >>> DrawGraph(np.array([[0.5, 0.]]), np.array([[0., 0., 0.]]))
       Traceback (most recent call last):
        ...
       ValueError: wrong shape of seq0, seq1 or seq2
    """
    if type(seq0) != np.ndarray or type(seq1) != np.ndarray or type(seq2) != np.ndarray:
        raise TypeError( "seq0, seq1 and seq2 must be numpy array" )
    if seq0.dtype != 'float64' or seq1.dtype != 'float64' or seq2.dtype != 'float64':
        raise TypeError( "wrong type of elements" )
    if seq0.shape[1] != 2 or seq1.shape[1] != 2 or (np.prod(seq2.shape)/(len(seq2) + (len(seq2) == 0)) != 2
            and np.prod(seq2.shape)/(len(seq2) + (len(seq2) == 0)) != 0):
        raise ValueError( "wrong shape of seq0, seq1 or seq2" )
    fig = plt.figure(figsize = (16,9))
    with plt.style.context('seaborn'):
        ax1 = fig.add_subplot(111)
    ax1.set_title('Results', fontsize = 24)
    ax1.scatter(seq0.T[0], seq0.T[1], color = '#708090', label = 'Class 0')
    ax1.scatter(seq1.T[0], seq1.T[1], color = '#FF6347', label = 'Class 1')
    ax1.legend(loc = 'lower right', fontsize = 16)
    ax1.tick_params(labelsize = 16)
    W = LearningAlg(seq0, seq1)
    InvSigma = np.array([[-2 * W[3], -W[4]], [-W[4], -2 * W[5]]])
    mu = np.matmul(np.linalg.inv(InvSigma), np.array([W[1], W[2]]))
    eigval, eigvec = np.linalg.eig(np.linalg.inv(InvSigma))
    HW = 2 * np.sqrt(np.dot(np.matmul(InvSigma, mu), mu) + 2 * W[0]) * np.sqrt(eigval)
    ang = np.degrees(np.arccos(np.dot(eigvec.T[0], np.array([1, 0])) / np.linalg.norm(eigvec.T[0])))
    ax1.add_artist(Ellipse((mu[0], mu[1]), HW[0], HW[1], ang, facecolor = 'none', edgecolor = '#DC143C'))
    if len(seq2) != 0:
        Y = np.concatenate((np.ones((len(seq2),1)), seq2, np.array([seq2.T[0]]).T ** 2, np.array([seq2.T[0]]).T
                            * np.array([seq2.T[1]]).T, np.array([seq2.T[1]]).T ** 2), axis = 1)
        y = (np.dot(Y, W) > 0)
        for i in range(len(y)):
            if not y[i]:
                ax1.scatter(seq2.T[0][i], seq2.T[1][i], facecolors = 'none', edgecolors = '#708090')
            else:
                ax1.scatter(seq2.T[0][i], seq2.T[1][i], facecolors = 'none', edgecolors = '#FF6347')
            ax1.annotate(str(i), (seq2.T[0][i], seq2.T[1][i]), (seq2.T[0][i] + 0.007, seq2.T[1][i] + 0.007))
    plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()