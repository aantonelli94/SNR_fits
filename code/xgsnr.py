import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


def dataframe():
    
    """
    Prepare a dataframe to accept variables that improve the fit.
    This function transforms the columns ['m1', 'm2', 'z', 'SNR']
    in the input dataset onto columns [['logMtot','massratio','z','logSNR']]
    that are easier to manage for the pipeline.
    
    'logMtot' = log10 of the total mass. (predictor)
    'mass ratio' = symmetric mass ratio of the binary. (predictor)
    'z' = redshift. (predictor)
    'logSNR' = log10 of the SNR. (target)
    """
    
    
    df = pd.read_pickle("./data/dataset_SNRavg.pkl") 
    df.columns = ['m1', 'm2', 'z', 'SNR']
    
    df['Mtot'] = df.loc[:,['m1','m2']].sum(axis=1)
    df['logMtot'] = np.log10(df['Mtot'])                   # Take the log of the total mass.
    df['massratio'] =  (df['m1']*df['m2'])/(df['Mtot']**2) # Take the symmetric mass ratio.
    df['logSNR'] = np.log10(df['SNR'])          # Take the log of the SNR for better predictions.
    df=df[['logMtot','massratio','z','logSNR']] # Redefine the dataframe.
    
    df = df[df.logSNR > np.log10(0.1)]      # Remove the values with very low SNRs
    
    return df



def SNR_fit(Mtot, eta, z):
    
    """
    Function that fits a 5th degree polynomial to a pre-computed dataset
    of averaged SNRs and that returns the SNR from a given 
    total mass, mass ratio, and redshift.
    """
    
    df = dataframe()
    df = df[df.logSNR > 1e-3] #input an SNR threshold to avoid numerical issues.

    
    train, test = train_test_split(df, random_state = 2, test_size=0.3) 
    variables = ['logMtot','massratio','z']
    X_train = train[variables]
    X_test = test[variables]
    y_train = train.logSNR
    y_test = test.logSNR
    
    # Call the fit.
    poly = PolynomialFeatures(degree=5)
    
    # Transform the data so that it can be accepted by a linear model fit.
    X_train_new = poly.fit_transform(X_train)
    X_test_new = poly.fit_transform(X_test)

    # Fit the model as if it was linear regression.
    model = linear_model.LinearRegression()
    polyfit = model.fit(X_train_new, y_train)
    
    
    ### Actual input
    
    logMtot = np.log10(Mtot)
    input_ = [[logMtot, eta, z]]
    X_temp = poly.fit_transform(pd.DataFrame(input_))
    
    
    
    return 10**polyfit.predict(X_temp)[0]
