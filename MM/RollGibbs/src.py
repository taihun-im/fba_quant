import numpy as np
from scipy.stats import norm, truncnorm
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd
# Assuming seed and other configurations can be set globally

def qDraw(p, q, c, varu, printLevel):
    assert p.shape[0] == q.shape[0], "p and q are of different lengths."
    assert np.isscalar(c), "c should be a scalar."
    assert np.isscalar(varu), "varu should be a scalar."
    if printLevel > 0:
        print(p, q)
    qNonzero = q != 0
    q = np.column_stack((q, q))
    p2 = np.column_stack((p, p))
    nSkip = 2
    modp = np.mod(np.arange(1, len(p) + 1), nSkip)
    ru = np.random.uniform(size=(len(p), 1))
    for iStart in range(nSkip):
        if printLevel > 0:
            print(f"qDraw. iStart: {iStart}")
        k = modp == iStart
        jnz = np.where(k & qNonzero)[0]
        if printLevel > 0:
            print(f"Drawing q's for t= {jnz}")
        q[jnz, 0] = 1
        q[jnz, 1] = -1
        cq = c * q
        v = p2 - cq
        u = v[1:, :] - v[:-1, :]
        # ... and so on

    q = q[:, 0]

def qDrawSlow(p, q, c, varu, PrintLevel):
    T = len(p)
    
    # Equivalent of "reset noname spaces=1;"
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    
    if PrintLevel > 0:
        print(f"qDraw T: {T} varu: {varu} c: {c}")
        print(q)
        
    for s in range(T):
        if PrintLevel >= 2:
            print(f"s: {s}")
            
        if q[s] == 0:  # Don't make a draw if q=0 (p is a quote midpoint)
            continue
        
        prExp = np.array([0, 0])
        
        if s < T - 1:
            uAhead = (p[s+1] - p[s]) + c * (1 - 1) - c * q[s+1]
            prExp = prExp - (uAhead**2) / (2*varu)
            if PrintLevel >= 2:
                print(f"s: {s}, p[s]: {p[s]}, p[s+1]: {p[s+1]}, uAhead: {uAhead}")
                
        if s > 0:
            uBack = (p[s] - p[s-1]) + c * q[s-1] - c * (1 - 1)
            prExp = prExp - (uBack**2) / (2*varu)
            if PrintLevel >= 2:
                print(f"s: {s}, p[s-1]: {p[s-1]}, p[s]: {p[s]}, uBack: {uBack}")
                
        logOdds = prExp[0] - prExp[1]
        if PrintLevel >= 2:
            print(f"logOdds (in favor of buy): {logOdds}")
            
        if abs(logOdds) > 100:
            q[s] = np.sign(logOdds)
        else:
            pBuy = 1 - 1 / (1 + np.exp(logOdds))
            q[s] = 1 - 2 * (np.random.uniform() > pBuy)
            
    # reset display options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.expand_frame_repr')

def BayesVarianceUpdate(priorAlpha, priorBeta, u):
    postAlpha = priorAlpha + len(u) / 2
    u2 = np.sum(np.power(u, 2))
    postBeta = priorBeta + u2 / 2
    return postAlpha, postBeta

# First function: BayesRegressionUpdate
def BayesRegressionUpdate(priorMu, priorCov, y, X, dVar):
    if priorMu.shape[1] != 1:
        print(f"BayesRegressionUpdate. priorMu is {priorMu.shape[0]}x{priorMu.shape[1]} (should be a column vector)")
        return None, None

    if X.shape[0] < X.shape[1]:
        print(f"BayesRegressionUpdate. X is {X.shape[0]}x{X.shape[1]}")
        return None, None

    if X.shape[0] != y.shape[0] or y.shape[1] != 1:
        print(f"BayesRegressionUpdate. X is {X.shape[0]}x{X.shape[1]}; y is {y.shape[0]}x{y.shape[1]}")
        return None, None

    if priorMu.shape[0] != X.shape[1]:
        print(f"BayesRegressionUpdate. X is {X.shape[0]}x{X.shape[1]}; priorMu is {priorMu.shape[0]}x{priorMu.shape[1]} (not conformable)")
        return None, None

    if priorCov.shape[0] != priorCov.shape[1] or priorCov.shape[0] != priorMu.shape[0]:
        print(f"BayesRegressionUpdate. priorMu is {X.shape[0]}x{X.shape[1]}; priorCov is {priorCov.shape[0]}x{priorCov.shape[1]}")
        return None, None

    covi = inv(priorCov)
    Di = (1/dVar) * X.T @ X + covi
    D = inv(Di)
    dd = (1/dVar) * X.T @ y + covi @ priorMu
    postMu = D @ dd
    postCov = D
    return postMu, postCov

# Second function: RandStdNormT
def RandStdNormT(zlow, zhigh):
    if zlow == float('-inf') and zhigh == float('inf'):
        return np.random.normal()

    PROBNLIMIT = 6
    if zlow > PROBNLIMIT and zhigh == float('inf'):
        return zlow + 100 * np.finfo(float).eps
    if zhigh < -PROBNLIMIT and zlow == float('-inf'):
        return zhigh - 100 * np.finfo(float).eps

    a, b = (zlow - 0) / 1, (zhigh - 0) / 1  # loc=0, scale=1 for standard normal
    return truncnorm.rvs(a, b)

def mvnrndT(mu, cov, vLower, vUpper):
    f = np.linalg.cholesky(cov).T
    n = mu.shape[0]
    eta = np.zeros((n, 1))

    low = (vLower[0] - mu[0]) / f[0, 0]
    high = (vUpper[0] - mu[0]) / f[0, 0]
    eta[0] = RandStdNormT(low, high)
    
    for k in range(1, n):
        etasum = f[k, :k].dot(eta[:k])
        low = (vLower[k] - mu[k] - etasum) / f[k, k]
        high = (vUpper[k] - mu[k] - etasum) / f[k, k]
        eta[k] = RandStdNormT(low, high)

    return mu + f.dot(eta)



def variance_decomp(dsCov: pd.DataFrame, dsCoeff: pd.DataFrame, 
                    RowLabel: str = None, Perm: list = None, PrintLevel: int = 2) -> None:
    
    # Print title
    print('Variance Decomposition')
    
    # Get number of variables
    n = dsCov.shape[0]
    
    # Extract coefficient matrix and variable names
    b = dsCoeff.to_numpy()
    
    if RowLabel is None:
        RowLabel = ['Coefficient vector' + str(i) for i in range(b.shape[0])]
    else:
        RowLabel = dsCoeff[RowLabel].tolist()
    
    print("Coefficient matrix:")
    print(pd.DataFrame(b, columns=RowLabel))
    
    # Extract covariance matrix and variable names
    cov = dsCov.to_numpy()
    VarNames = dsCov.columns.tolist()
    
    print("\nCovariance matrix:")
    print(pd.DataFrame(cov, columns=VarNames, index=VarNames))
    
    if n != cov.shape[0]:
        raise ValueError("Covariance matrix not square or not conformable with coefficients.")
    
    # Correlation matrix
    sd = np.diag(np.sqrt(1/np.diag(cov)))
    cor = sd @ cov @ sd
    print("\nCorrelation matrix:")
    print(pd.DataFrame(cor, columns=VarNames, index=VarNames))
    
    # Set vector of variable names for the permutation
    if Perm is None:
        Perm = VarNames
    
    # Set vector of indexes for permutation
    iperm = [VarNames.index(p) for p in Perm]
    varNamesP = [VarNames[i] for i in iperm]
    print("\nPermutation used in decomposition / ordering of variables:", varNamesP)
    
    # Execute permutations
    bP = b[:, iperm]
    if PrintLevel >= 2:
        print("\nPermuted coefficients:")
        print(pd.DataFrame(bP, columns=varNamesP, index=RowLabel))
    
    covP = cov[:, iperm][iperm]
    if PrintLevel >= 2:
        print("\nPermuted covariance matrix:")
        print(pd.DataFrame(covP, columns=varNamesP, index=varNamesP))
    
    # Analyze variance contributions
    cd = np.linalg.cholesky(covP).T
    print("\nCholesky factor of permuted covariance matrix:")
    print(pd.DataFrame(cd, columns=varNamesP, index=RowLabel))
    
    VarContrib = (bP @ np.linalg.cholesky(covP))**2
    print("\nVariance contributions (ordered):")
    print(pd.DataFrame(VarContrib, columns=varNamesP, index=RowLabel))
    
    VarTotal = VarContrib.sum(axis=1, keepdims=True)
    print("\nTotal variance:")
    print(pd.DataFrame(VarTotal, index=RowLabel))
    
    VarProp = VarContrib / VarTotal
    print("\nProportional contributions:")
    print(pd.DataFrame(VarProp, columns=varNamesP, index=RowLabel))
