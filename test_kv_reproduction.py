import numpy as np
import json

# Load the constants from the JSON file
with open('constants/constants_overflow.json', 'r') as f:
    constants = json.load(f)
    Kvs = constants['Overflow']['K_vs']
    h_star = 0.05
    Kvleak_bool = constants['Overflow']['Kvleak_bool']

# Test values from line 9489
h_test = 0.03333333333333333
Kv_expected = 2.75615e-05

# Calculate Kv using the linear_valve function
def linear_valve(h, Kvs, Kvleak_bool=False):
    """
    Returns Kv value of the valve following a linear characteristic
    based on the valve displacement h in [0, 1]. 0 means fully closed, 1 means fully open.
    """
    Kv0 = Kvs / 50  # same minimum as in equal-percentage
    Kvleak = Kvs / 2000
    
    h_star = 0.05  # lower values of h make the form of the valve deviate from the equal percentage equation.
    
    if h < h_star and Kvleak_bool:
        Kvr = Kv0 + h_star * (Kvs - Kv0)
        Kv = Kvleak + h*(Kvr - Kvleak)/h_star
    else:
        Kv = Kv0 + h * (Kvs - Kv0)
    return Kv

# Calculate Kv with different scenarios
print("=" * 70)
print("KV REPRODUCTION TEST")
print("=" * 70)
print(f"\nConstants from JSON:")
print(f"  Kvs: {Kvs}")
print(f"  h_star: {h_star}")
print(f"  Kvleak_bool: {Kvleak_bool}")
print(f"\nTest case from line 9489:")
print(f"  h: {h_test}")
print(f"  Expected Kv: {Kv_expected:.15e}")

Kv0 = Kvs / 50
Kvleak = Kvs / 2000
print(f"\nIntermediate calculations:")
print(f"  Kv0 = Kvs / 50 = {Kv0:.15e}")
print(f"  Kvleak = Kvs / 2000 = {Kvleak:.15e}")

if h_test < h_star and Kvleak_bool:
    Kvr = Kv0 + h_star * (Kvs - Kv0)
    print(f"  Kvr = Kv0 + h_star * (Kvs - Kv0)")
    print(f"      = {Kv0:.15e} + {h_star} * ({Kvs:.15e} - {Kv0:.15e})")
    print(f"      = {Kv0:.15e} + {h_star} * {(Kvs - Kv0):.15e}")
    print(f"      = {Kvr:.15e}")
    
    Kv_calc = Kvleak + h_test * (Kvr - Kvleak) / h_star
    print(f"  Kv = Kvleak + h * (Kvr - Kvleak) / h_star")
    print(f"     = {Kvleak:.15e} + {h_test} * ({Kvr:.15e} - {Kvleak:.15e}) / {h_star}")
    print(f"     = {Kvleak:.15e} + {h_test} * {(Kvr - Kvleak):.15e} / {h_star}")
    print(f"     = {Kv_calc:.15e}")
else:
    Kv_calc = Kv0 + h_test * (Kvs - Kv0)
    print(f"  Kv = Kv0 + h * (Kvs - Kv0) = {Kv_calc:.15e}")

print(f"\nCalculated Kv: {Kv_calc:.15e}")
print(f"Expected Kv:   {Kv_expected:.15e}")
print(f"Difference:    {abs(Kv_calc - Kv_expected):.15e}")
print(f"Relative error: {abs(Kv_calc - Kv_expected) / Kv_expected * 100:.6f}%")

# Test with function
Kv_func = linear_valve(h_test, Kvs, Kvleak_bool)
print(f"\nUsing linear_valve function:")
print(f"  Result: {Kv_func:.15e}")
print(f"  Match expected? {np.isclose(Kv_func, Kv_expected, rtol=1e-10)}")

# Test potential issues
print(f"\n" + "=" * 70)
print("POTENTIAL ISSUES TO CHECK:")
print("=" * 70)

print(f"\n1. Kvleak_bool setting:")
print(f"   With Kvleak_bool=True: {linear_valve(h_test, Kvs, True):.15e}")
print(f"   With Kvleak_bool=False: {linear_valve(h_test, Kvs, False):.15e}")

print(f"\n2. Different Kvs values:")
for test_kvs in [0.000597, 0.003, 0.0005, 0.0006]:
    kv = linear_valve(h_test, test_kvs, Kvleak_bool)
    print(f"   Kvs={test_kvs}: Kv={kv:.15e}")

print(f"\n3. Different h values (around 0.025):")
for test_h in [0.024, 0.025, 0.026]:
    kv = linear_valve(test_h, Kvs, Kvleak_bool)
    print(f"   h={test_h}: Kv={kv:.15e}")

print(f"\n4. At h_star threshold (0.05):")
print(f"   h=0.04999 (just below): {linear_valve(0.04999, Kvs, Kvleak_bool):.15e}")
print(f"   h=0.05000 (at): {linear_valve(0.05000, Kvs, Kvleak_bool):.15e}")
print(f"   h=0.05001 (just above): {linear_valve(0.05001, Kvs, Kvleak_bool):.15e}")
