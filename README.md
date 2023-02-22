# WLS_cpy
Use `C` code to realize online calculation of weighted linear regression, and pack the code to `Python` function.

### Problem
When we finish the calculation of weighted linear regression for `n1` samples, we want to update our results for other `n2` samples in the future, instead of calculating `n1 + n2` from the beginning with high time and space complexity.

### Pipeline
1. [lapack](https://netlib.org/lapack/) is used. `liblapack` can be find in package manager of Linux/Homebrew/Windows Subsystem for Linux.

2. Use `gcc calculator.c  -llapack` to compile `C` code

3. In the main file of `calculator.c`, a very large loop exists to test whether all memory used for one loop is successfully recycled. Run `a.out` to test, and check the usage of system memory.

4. Use `gcc -fPIC -llapack -shared -o a.so calculator.c` to compile `.so` for `Python`.

5. `Python` separates the functions from `.so` files. Therefore, if we call another `C` function from outside, this may lead to failures. Thus, we have to copy all functions used for `wls_iter` inside.  

6. Run `python benchmark.py` for a series of tests.
    * Check whether the `C` code agrees with the results from `numpy`.
    * Check the speed of the `statsmodels` package, my `C` code, and `numpy`, especially for online updates.
    * Check the speed and memory usage of the `statsmodels` package and my `C` code with a very large dataset.

### Conclusion

My `C` code is much more efficient than the other methods.
