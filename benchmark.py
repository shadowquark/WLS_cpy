from ctypes import c_void_p, Structure, c_double, c_int, cdll, cast, POINTER
from numpy.ctypeslib import ndpointer
import numpy as np
import time
import functools as ft
from functools import partial as par
import statsmodels.api as sm
def F(*z):
    z = [*z]
    z[0] = [z[0]]
    return [*ft.reduce(lambda x, y: map(y, x), z)][0]
FF = lambda *z: [*ft.reduce(lambda x, y: map(y, x), z)]
fyx = lambda f, *x: lambda *y: f(*y, *x)

lib = cdll.LoadLibrary("./a.so")

def c_add(x, y, m, n):
    lib.add.restype = ndpointer(dtype = c_double, shape = (m, n))
    return lib.add(c_void_p(x.ctypes.data), c_void_p(y.ctypes.data), m, n)

def c_times(x, y, m, p, n):
    lib.times.restype = ndpointer(dtype = c_double, shape = (m, n))
    return lib.times(c_void_p(x.ctypes.data), c_void_p(y.ctypes.data), m, p, n)

def c_xTw(x, y, m, n):
    lib.calc_xTw.restype = ndpointer(dtype = c_double, shape = (m, n))
    return lib.calc_xTw(c_void_p(x.ctypes.data), c_void_p(y.ctypes.data), m, n)

def c_wls_iter(
    y_next, x_next, w_next, n2, m,
    x = np.array([]), xTwx = None, xTwy = None,
    n1 = 0, update = False
):
    tot_len = n1 + 2 * n2 + m * (m + 2)
    if xTwx is None:
        xTwx = np.zeros((m, m))
    if xTwy is None:
        xTwy = np.zeros((m, 1))
    double = lambda x: x.astype('d')
    x_next = double(x_next)
    w_next = double(w_next)
    y_next = double(y_next)
    x = double(x)
    xTwx = double(xTwx)
    xTwy = double(xTwy)
    lib.wls_iter.restype = ndpointer(dtype = c_double, shape = (tot_len,))
    results = lib.wls_iter(
        c_void_p(x.ctypes.data),
        c_void_p(xTwx.ctypes.data),
        c_void_p(xTwy.ctypes.data),
        c_void_p(x_next.ctypes.data),
        c_void_p(w_next.ctypes.data),
        c_void_p(y_next.ctypes.data),
        n1, n2, m, update
    )
    xTwx = results[:m * m].reshape(m, m)
    xTwy = results[m * m : m * (m + 1)]
    predict = results[m * (m + 1) : m * (m + 2)]
    yhat_next = results[m * (m + 2) : m * (m + 2) + n2]
    if update:
        yhat = results[-n1 - n2:]
    else:
        yhat = None
    return xTwx, xTwy, predict, yhat_next, yhat

loop = 1
for i in range(loop):
    print('\n', i)
    m, n = 3, 2
    matrix1 = np.random.rand(m, n)
    matrix2 = np.random.rand(m, n)
    temp1 = matrix1 + matrix2
    temp2 = c_add(matrix1, matrix2, m, n)
    temp = temp1 - temp2
    print("add")
    print(len(temp[temp > 1E-3]) == 0)

    m, n, p = 3, 8, 2
    matrix1 = np.random.rand(m, p)
    matrix2 = np.random.rand(p, n)
    temp1 = np.matmul(matrix1, matrix2)
    temp2 = c_times(matrix1, matrix2, m, p, n)
    temp = temp1 - temp2
    print("times")
    print(len(temp[temp > 1E-3]) == 0)

    m, n = 3, 2
    matrix1 = np.random.rand(n, m)
    matrix2 = np.random.rand(n)
    matrix3 = np.diag(matrix2)
    temp1 = np.matmul(matrix1.T, matrix3)
    temp2 = c_xTw(matrix1, matrix2, m, n)
    temp = temp1 - temp2
    print("xTw")
    print(len(temp[temp > 1E-3]) == 0)

    n1, n2, m, update = 4, 2, 3, True
    matrix1 = np.random.rand(n1, m)
#   n1, n2, m, update = 0, 2, 3, False
#   matrix1 = np.random.rand(0,)
    matrix2 = np.random.rand(m, m)
    matrix3 = np.random.rand(m, 1)
    matrix4 = np.random.rand(n2, m)
    matrix5 = np.random.rand(n2)
    matrix6 = np.random.rand(n2, 1)
    results1, results2, results3, results4, results5 = c_wls_iter(
        matrix6, matrix4, matrix5, n2, m, 
        matrix1, matrix2, matrix3, n1, update
    )
    print("xTwx\n", results1)
    test1 = matrix2 + np.matmul(np.matmul(matrix4.T, np.diag(matrix5)), matrix4)
    test = results1.reshape(-1) - test1.reshape(-1)
    print(len(temp[temp > 1E-3]) == 0)
    print(f"{m} * {m}\n")
    print("xTwy\n", results2)
    test2 = matrix3 + np.matmul(np.matmul(matrix4.T, np.diag(matrix5)), matrix6)
    test = results2.reshape(-1) - test2.reshape(-1)
    print(len(temp[temp > 1E-3]) == 0)
    print(f"{m} * {1}\n")
    print("predict\n", results3)
    test3 = np.matmul(np.linalg.inv(test1), test2)
    test = results3.reshape(-1) - test3.reshape(-1)
    print(len(temp[temp > 1E-3]) == 0)
    print(f"{m} * {1}\n")
    print("yhat_next\n", results4)
    test4 = np.matmul(matrix4, test3)
    test = results4.reshape(-1) - test4.reshape(-1)
    print(len(temp[temp > 1E-3]) == 0)
    print(f"{n2} * {1}\n")
    print("yhat\n", results5)
    if update:
        test5 = np.matmul(np.concatenate((matrix1, matrix4)), test3)
        test = results5.reshape(-1) - test5.reshape(-1)
        print(len(temp[temp > 1E-3]) == 0)
        print(f"{n1 + n2} * {1}\n")
    else:
        print("NONE")

m, n = 5, 50000
step = 49000
X = np.random.rand(n, m)
X1 = X[:step, :]
X2 = X[step:, :]
Y = np.random.rand(n)
Y1 = Y[:step]
Y2 = Y[step:]
w = np.random.rand(n)
w1 = w[:step]
w2 = w[step:]
time0 = time.time()
test = sm.WLS(Y1, X1, weights = w1, missing = "drop").fit()
print(test.summary())
time1 = time.time()
results = c_wls_iter(Y1, X1, w1, step, m)
xTwx = results[0]
xTwy = results[1]
[print(x) for x in results[2]]
time2 = time.time()
test1 = np.matmul(np.matmul(X1.T, np.diag(w1)), X1)
test2 = np.matmul(np.matmul(X1.T, np.diag(w1)), Y1.reshape(-1, 1))
print(np.matmul(np.linalg.inv(test1), test2))
time3 = time.time()
test = sm.WLS(Y, X, weights = w, missing = "drop").fit()
print(test.summary())
time4 = time.time()
results = c_wls_iter(Y2, X2, w2, n - step, m,
                        xTwx = xTwx, xTwy = xTwy, n1 = step)
[print(x) for x in results[2]]
time5 = time.time()
test1 = xTwx + np.matmul(np.matmul(X2.T, np.diag(w2)), X2)
test2 = xTwy.reshape(-1, 1)\
        + np.matmul(np.matmul(X2.T, np.diag(w2)), Y2.reshape(-1, 1))
print(np.matmul(np.linalg.inv(test1), test2))
time6 = time.time()
print("Time Usage")
print(f"statsmodels WLS {step}:", time1 - time0)
print(f"C code {step}:", time2 - time1)
print(f"numpy {step}:", time3 - time2)
print("We get the same results for 3 methods above.")
print(f"statsmodels WLS {n}:", time4 - time3)
print(f"C code next {n - step}:", time5 - time4)
print(f"numpy next {n - step}:", time6 - time5)
print("We get the same results for 3 methods above.")
print("\n\n\n\n\n")

m, n = 5, 5000000
step = 4999500
X = np.random.rand(n, m)
X1 = X[:step, :]
X2 = X[step:, :]
Y = np.random.rand(n)
Y1 = Y[:step]
Y2 = Y[step:]
w = np.random.rand(n)
w1 = w[:step]
w2 = w[step:]
time0 = time.time()
test = sm.WLS(Y1, X1, weights = w1, missing = "drop").fit()
print(test.summary())
time1 = time.time()
results = c_wls_iter(Y1, X1, w1, step, m)
xTwx = results[0]
xTwy = results[1]
[print(x) for x in results[2]]
time2 = time.time()
time3 = time.time()
test = sm.WLS(Y, X, weights = w, missing = "drop").fit()
print(test.summary())
time4 = time.time()
results = c_wls_iter(Y2, X2, w2, n - step, m,
                        xTwx = xTwx, xTwy = xTwy, n1 = step)
[print(x) for x in results[2]]
time5 = time.time()
time6 = time.time()
print("Time Usage")
print(f"statsmodels WLS {step}:", time1 - time0)
print(f"C code {step}:", time2 - time1)
print("We get the same results for 2 methods above.")
print(f"statsmodels WLS {n}:", time4 - time3)
print(f"C code next {n - step}:", time5 - time4)
print("We get the same results for 2 methods above.")
print("\n\n\n\n\n")

