from quqfin.core.options.black_scholes import BlackScholes
from quqfin.core.options.monte_carlo import MonteCarloOption
import numpy as np
import yfinance as yf

def main():
    # Get AAPL data for realistic example
    aapl = yf.Ticker("AAPL")
    hist = aapl.history(period="1y")
    current_price = hist['Close'][-1]
    historical_volatility = np.std(np.log(hist['Close']/hist['Close'].shift(1))) * np.sqrt(252)
    
    # Black-Scholes example
    bs = BlackScholes()
    S = current_price
    K = current_price  # At-the-money option
    r = 0.05  # Risk-free rate
    sigma = historical_volatility
    T = 1.0  # One year
    
    # Calculate option prices and Greeks
    call_price = bs.call_price(S, K, r, sigma, T)
    delta = bs.delta_call(S, K, r, sigma, T)
    gamma = bs.gamma(S, K, r, sigma, T)
    vega = bs.vega(S, K, r, sigma, T)
    
    print(f"AAPL Current Price: ${S:.2f}")
    print(f"Historical Volatility: {sigma:.2%}")
    print("\nBlack-Scholes Results:")
    print(f"Call Price: ${call_price:.2f}")
    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Vega: {vega:.4f}")
    
    # Monte Carlo exotic options
    mc = MonteCarloOption(n_simulations=100000)
    asian_price = mc.asian_call(S, K, r, T, sigma)
    lookback_price = mc.lookback_call(S, K, r, T, sigma)
    barrier_price = mc.barrier_call(S, K, H=1.2*S, r=r, T=T, sigma=sigma)
    
    print("\nMonte Carlo Exotic Options:")
    print(f"Asian Call: ${asian_price:.2f}")
    print(f"Lookback Call: ${lookback_price:.2f}")
    print(f"Up-and-Out Barrier Call (H=${1.2*S:.2f}): ${barrier_price:.2f}")

if __name__ == "__main__":
    main()