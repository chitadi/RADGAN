
from pesq import pesq as pesq_fn, NoUtterancesError
import numpy as np

def nbpesq(target, est_target, fs=8000):
    try:
        return pesq_fn(fs, target, est_target, 'nb')
    except NoUtterancesError:
        # Return NaN for samples without utterances
        return np.nan