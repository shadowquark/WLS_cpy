# WLS_cpy
Use C code to realize online calculation of weighted linear regression, and pack the code to Python function.

1. Use `gcc calculator.c  -llapack` to compile C code

2. In the main file of `calculator.c`, a very large loop exists to test whether all memory used for one loop is successfully recycled. Run `a.out` to test.

3. 

4. Use `gcc -fPIC -llapack -shared -o a.so calculator.c` to compile `.so` for `Python`.

5. Run `python benchmark.py` for a series of tests.

* 123
* 123
