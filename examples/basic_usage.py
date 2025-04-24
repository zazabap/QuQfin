from quqfin.core.options.black_scholes import BlackScholes
from quqfin.core.options.monte_carlo import MonteCarloOption
from quqfin.core.portfolio.optimization import PortfolioOptimizer
from quqfin.core.risk.var import RiskMetrics
from quqfin.core.market_making.orderbook import OrderBook
from quqfin.core.fixed_income.bonds import BondPricing
from quqfin.core.timeseries.models import TimeSeriesModels
from quqfin.core.algo_trading.strategies import TradingStrategies
import numpy as np

def main():
    # Option pricing examples
    bs = BlackScholes()
    S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1
    
    # Basic pricing
    call_price = bs.call_price(S, K, r, sigma, T)
    print(f"Black-Scholes Call Price: {call_price}")
    
    # Greeks
    delta = bs.delta_call(S, K, r, sigma, T)
    gamma = bs.gamma(S, K, r, sigma, T)
    vega = bs.vega(S, K, r, sigma, T)
    print(f"Delta: {delta}, Gamma: {gamma}, Vega: {vega}")
    
    # Monte Carlo exotic options
    mc = MonteCarloOption()
    asian_price = mc.asian_call(S, K, r, T, sigma)
    lookback_price = mc.lookback_call(S, K, r, T, sigma)
    barrier_price = mc.barrier_call(S, K, H=120, r=r, T=T, sigma=sigma)
    print(f"Asian Call: {asian_price}")
    print(f"Lookback Call: {lookback_price}")
    print(f"Barrier Call: {barrier_price}")
    
    # Portfolio optimization examples
    returns = np.array([0.1, 0.15, 0.12, 0.09])
    cov_matrix = np.array([[0.1, 0.02, 0.03, 0.02],
                          [0.02, 0.15, 0.03, 0.02],
                          [0.03, 0.03, 0.12, 0.02],
                          [0.02, 0.02, 0.02, 0.1]])
    
    optimizer = PortfolioOptimizer(returns, cov_matrix)
    markowitz_weights = optimizer.markowitz_optimization(target_return=0.12)
    risk_parity_weights = optimizer.risk_parity_optimization()
    
    print(f"Markowitz Weights: {markowitz_weights}")
    print(f"Risk Parity Weights: {risk_parity_weights}")
    
    # Fixed Income examples
    bond = BondPricing()
    zcb_price = bond.zero_coupon_price(1000, 0.05, 2)
    coupon_price = bond.coupon_bond_price(1000, 0.06, 0.05, 2)
    mac_duration = bond.duration(1000, 0.06, 0.05, 2)
    print(f"Zero Coupon Bond Price: {zcb_price}")
    print(f"Coupon Bond Price: {coupon_price}")
    print(f"Macaulay Duration: {mac_duration}")
    
    # Time Series examples
    prices = np.random.random(1000)
    ts_model = TimeSeriesModels(prices)
    volatility = ts_model.garch(np.diff(np.log(prices)))
    print(f"GARCH Volatility forecast: {volatility[-1]}")
    
    # Algorithmic Trading examples
    trader = TradingStrategies(prices)
    ma_signals = trader.moving_average_crossover()
    momentum_signals = trader.momentum_strategy()
    mean_rev_signals = trader.mean_reversion()
    print(f"Latest MA Signal: {ma_signals[-1]}")
    print(f"Latest Momentum Signal: {momentum_signals[-1]}")
    print(f"Latest Mean Reversion Signal: {mean_rev_signals[-1]}")

if __name__ == "__main__":
    main()