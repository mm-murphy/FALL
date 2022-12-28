import numpy as np

def visit_polynomial(x, params, n=1):
    ## This function fits a user-specified degree polynomial function
    ##    meant to be over the visit-long flux (y) vs. time (x) data
    ## Inputs:
    ##    x -- The dependent variable array, meant to be time
    ##    params -- array of coefficients to be applied to the polynomial
    ##              [length must be n + 1]
    ##    n -- the desired polynomial degree
    ##         (n=1 is linear, n=2 is quadratic, etc.)
    ## ==================================================================
    
    # -- Check that the parameter array is proper length
    if len(params) != (n+1):
        print('Length of parameter array must be n + 1')
        print('   where n = desired polynomial degree')
        return np.nan
    # -- Set up polynomial
    if n == 1:
        # if n = 1, do a linear function of y = a1*x + a0
        a1, a0 = params
        poly_y = a1*x + a0
        return poly_y
    elif n == 1:
        # if n=2, do a quadratic function of y = a2*x^2 + a1*x + a0
        a2, a1, a0 = params
        poly_y = a2*(x**2) + a1*x + a0
        return poly_y
    else:
        print('choose an allowed n value')
        return np.nan
    