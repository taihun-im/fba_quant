import numpy as np
import pandas as pd
from src import BayesRegressionUpdate, BayesVarianceUpdate, mvnrndT
Infinity = 1e30
eps = 1e-30

def roll_gibbs(dsIn, nSweeps=100, qDraw=True, varuDraw=True, regDraw=True, varuStart=0.001, cStart=0.01, 
               printLevel=0, cLower=0, cUpper=float('inf')):
    
    np.random.seed(1234)  # Initialize the random number generators

    # Read in data
    # Assuming dsIn is a pandas DataFrame
    if qDraw:
        p = dsIn['p'].values
        q = np.empty(p.shape)
    else:
        p = dsIn['p'].values
        q = dsIn['q'].values

    nObs = len(p)

    # Initialize output data (will be filled and then converted to pandas DataFrame later on)
    qOut_data = {'sweep': [], 't': [], 'q': []}
    parmOut_data = {'sweep': [], 'sdu': [], 'c': []}

    dp = p[1:] - p[:-1]

    if qDraw:
        qInitial = np.concatenate(([1], np.sign(dp)))
        qInitial[qInitial == 0] = 1  # Only initialize nonzero elements of q
        q = qInitial

    varu = varuStart
    c = cStart

    for sweep in range(1, nSweeps + 1):
        if sweep % 1000 == 0:
            print(f'Sweep: {sweep}')

        dq = q[1:] - q[:-1]

        if regDraw:
            priorMu = np.array([0])
            priorCov = np.array([[1]])
            postMu, postCov = BayesRegressionUpdate(priorMu, priorCov, dp, dq, varu)
            if printLevel >= 2:
                print(postMu, postCov)
            c = mvnrndT(postMu, postCov, cLower, cUpper)
            if printLevel >= 2:
                print(c)

        if varuDraw:
            u = dp - c * dq
            priorAlpha = 1e-12
            priorBeta = 1e-12
            postAlpha = np.nan
            postBeta = np.nan
            postAlpha, postBeta = BayesVarianceUpdate(priorAlpha, priorBeta, u)
            x = (1 / postBeta) * np.random.gamma(postAlpha)
            varu = 1 / x
            sdu = np.sqrt(varu)
            if printLevel >= 2:
                print(varu)

        if qDraw:
            qDrawPrintLevel = 0
            qDraw(p, q, c, varu, qDrawPrintLevel) 
            # Collect output data
            qOut_data['sweep'].extend([sweep] * nObs)
            qOut_data['t'].extend(range(1, nObs + 1))
            qOut_data['q'].extend(q)

        if regDraw or varuDraw:
            parmOut_data['sweep'].append(sweep)
            parmOut_data['sdu'].append(sdu)
            parmOut_data['c'].append(c)

    # Convert dictionaries to pandas DataFrames and save (if necessary)
    qOut_df = pd.DataFrame(qOut_data)
    parmOut_df = pd.DataFrame(parmOut_data)

    return qOut_df, parmOut_df