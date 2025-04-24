import numpy as np
from scipy.stats import norm
import cupy as cp

class BlackScholes:
    @staticmethod
    def d1(S, K, r, sigma, T):
        return (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    
    @staticmethod
    def d2(S, K, r, sigma, T):
        return (np.log(S/K) + (r - sigma**2/2)*T)/(sigma*np.sqrt(T))
    
    @staticmethod
    def call_price(S, K, r, sigma, T):
        d1 = BlackScholes.d1(S, K, r, sigma, T)
        d2 = BlackScholes.d2(S, K, r, sigma, T)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, r, sigma, T):
        d1 = BlackScholes.d1(S, K, r, sigma, T)
        d2 = BlackScholes.d2(S, K, r, sigma, T)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    # Adding Greeks calculations
    @staticmethod
    def delta_call(S, K, r, sigma, T):
        return norm.cdf(BlackScholes.d1(S, K, r, sigma, T))
    
    @staticmethod
    def delta_put(S, K, r, sigma, T):
        return -norm.cdf(-BlackScholes.d1(S, K, r, sigma, T))
    
    @staticmethod
    def gamma(S, K, r, sigma, T):
        d1 = BlackScholes.d1(S, K, r, sigma, T)
        return norm.pdf(d1)/(S*sigma*np.sqrt(T))
    
    @staticmethod
    def vega(S, K, r, sigma, T):
        d1 = BlackScholes.d1(S, K, r, sigma, T)
        return S*np.sqrt(T)*norm.pdf(d1)
    
    @staticmethod
    def theta_call(S, K, r, sigma, T):
        d1 = BlackScholes.d1(S, K, r, sigma, T)
        d2 = BlackScholes.d2(S, K, r, sigma, T)
        return (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) 
                - r*K*np.exp(-r*T)*norm.cdf(d2))