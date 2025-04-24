from quqfin.core.timeseries.models import TimeSeriesModels
import yfinance as yf
import numpy as np

def main():
    # Get SPY data for market analysis
    spy = yf.Ticker("SPY")
    hist = spy.history(period="2y")
    prices = hist['Close'].values
    returns = np.diff(np.log(prices))
    
    ts_model = TimeSeriesModels(prices)
    
    # GARCH volatility forecasting
    volatility = ts_model.garch(returns)
    
    print("S&P 500 ETF (SPY) Analysis")
    print("-" * 30)
    print(f"Current Price: ${prices[-1]:.2f}")
    print(f"Current Volatility (GARCH): {volatility[-1]:.2%}")
    
    # Kalman Filter for trend estimation
    state_mean, state_cov = ts_model.kalman_filter(
        returns,
        transition_matrix=1,
        observation_matrix=1,
        process_noise=0.01,
        measurement_noise=0.1
    )
    
    print(f"Trend Estimate: {state_mean[-1]:.2%}")

if __name__ == "__main__":
    main()