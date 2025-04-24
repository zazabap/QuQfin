from quqfin.core.fixed_income.bonds import BondPricing
import numpy as np

def main():
    bond = BondPricing()
    
    # Example with Treasury-like bonds
    print("US Treasury-like Bond Analysis")
    print("-" * 30)
    
    # 10-year Treasury
    face_value = 1000
    coupon_rate = 0.035  # 3.5%
    market_yield = 0.04  # 4%
    maturity = 10
    
    price = bond.coupon_bond_price(face_value, coupon_rate, market_yield, maturity)
    duration = bond.duration(face_value, coupon_rate, market_yield, maturity)
    
    print(f"10-Year Treasury Bond:")
    print(f"Face Value: ${face_value}")
    print(f"Coupon Rate: {coupon_rate:.1%}")
    print(f"Market Yield: {market_yield:.1%}")
    print(f"Clean Price: ${price:.2f}")
    print(f"Duration: {duration:.2f} years")
    
    # Zero-coupon bond
    zcb_price = bond.zero_coupon_price(face_value, market_yield, maturity)
    print(f"\nZero-Coupon Bond:")
    print(f"Price: ${zcb_price:.2f}")

if __name__ == "__main__":
    main()