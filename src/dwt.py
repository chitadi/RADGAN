import numpy as np
import pywt
from scipy.signal import upfirdn

# tag = "zeros"
# tag = "interpolation"
# tag = "all"

def upsampler(m_t, coefficients, tag):
    if tag == "zeros":
        upsampling_coefficient = int((len(m_t)/len(coefficients))/2)
        coefficients = upfirdn([1], coefficients, upsampling_coefficient)
        #for padding to target length
        padding = np.zeros(int(len(m_t)/2)-len(coefficients),dtype=coefficients.dtype)
        coefficients = np.concatenate([coefficients,padding])

        return coefficients
    
    elif tag == "interpolation":
        upsampling_coefficient = int((len(m_t)/len(coefficients))/2)
        coefficients = upfirdn([0.5, 1.0, 0.5], coefficients, upsampling_coefficient)
        
        if len(coefficients)<(len(m_t)/2):
            #for padding to target length
            padding = np.zeros(int(len(m_t)/2)-len(coefficients),dtype=coefficients.dtype)
            coefficients = np.concatenate([coefficients,padding])
            return coefficients
        else :
            # padding = np.zeros(int(len(m_t)/2)-len(coefficients),dtype=coefficients.dtype)
            coefficients = coefficients[:int(len(m_t)/2)]
            return coefficients
    
def get_wavelet_coefficients(m_t, wavelet, level, tag):

    if (tag == "zeros" or tag == "interpolation"):
        coeffs = pywt.wavedec(m_t,wavelet,level=level, mode = 'symmetric')

        a3,d3,d2,d1 = coeffs

        a3=np.array(a3)
        d3=np.array(d3)
        d2=np.array(d2)
        d1=np.array(d1)

        a3_upsampled = upsampler(m_t,a3, tag)
        d3_upsampled = upsampler(m_t,d3, tag)
        d2_upsampled = upsampler(m_t,d2, tag)
        d1_upsampled = upsampler(m_t,d1, tag)

        return(a3_upsampled,d3_upsampled,d2_upsampled,d1_upsampled)
    
    elif(tag == "all"):
        coeffs = pywt.wavedec(m_t,wavelet,level=level, mode = 'symmetric')
        a3,d3,d2,d1 = coeffs
        a3=np.array(a3)
        d3=np.array(d3)
        d2=np.array(d2)
        d1=np.array(d1)
        return (a3,d3,d2,d1)

def normalize(m_t):
    max_abs = np.max(np.abs(m_t))
    m_t = m_t.astype(np.float64) / max_abs
    return m_t, max_abs
    
def unify(m_t, wavelet='db1', level=3, tag = "zeros"):
    #normalizing to 1
    m_t, _ = normalize(m_t)

    a3,d3,d2,d1 = get_wavelet_coefficients(m_t, wavelet, level,tag)
    row1 = m_t
    
    #input to the model
    if tag == "zeros" or tag == "interpolation":
        row2 = np.concatenate([a3,d3])
    else:
        row2 = np.concatenate([a3,d3,d2,d1])
    assert row2.shape[0] == row1.shape[0], "Shapes do not match!"
    input_array = np.vstack((row1, row2))
    return input_array

