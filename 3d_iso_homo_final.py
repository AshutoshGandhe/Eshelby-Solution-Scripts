import numpy as np
import scipy.special as sp
import sympy as sym
import scipy.integrate as integrate
from matplotlib import pyplot as plt

pi = np.pi
deltaij = np.identity(3)


# Function to find Lambda (the largest root)


def find_lambda(X, axis):
    ans = sym.symbols('ans')
    eqn = sym.Eq(((X[0]) ** 2) / ((axis[0]) ** 2 + ans) + ((X[1]) ** 2) / ((axis[1]) ** 2 + ans) + ((X[2]) ** 2) / (
            (axis[2]) ** 2 + ans), 1)
    solution = sym.solve(eqn, ans)
    #print(solution)
    sol = [complex(i) for i in solution]
    #print(sol)
    LAMBDA = 0
    for i in sol:
        LAMBDA = max(LAMBDA, i.real)
    #print(LAMBDA)
    return LAMBDA


def IIJ(axis, L, flag):
    theta = np.arcsin(np.sqrt((axis[0] ** 2 - axis[2] ** 2) / (axis[0] ** 2 + L)))
    k = np.sqrt((axis[0] ** 2 - axis[1] ** 2) / (axis[0] ** 2 - axis[2] ** 2))
    F = sp.ellipkinc(theta, k ** 2)
    E = sp.ellipeinc(theta, k ** 2)
    # print("theta", theta, k, F, E)
    arr = np.zeros(3)
    dels = np.sqrt((axis[0] ** 2 + L) * (axis[1] ** 2 + L) * (axis[2] ** 2 + L))
    c1 = (4 * pi * axis[0] * axis[1] * axis[2]) / (
                (axis[0] ** 2 - axis[1] ** 2) * (np.sqrt(axis[0] ** 2 - axis[2] ** 2)))
    arr[0] = c1 * (F - E)
    c2 = (4 * pi * axis[0] * axis[1] * axis[2]) / (
                (axis[1] ** 2 - axis[2] ** 2) * (np.sqrt(axis[0] ** 2 - axis[2] ** 2)))
    d1 = ((axis[1] ** 2 + L) * (np.sqrt(axis[0] ** 2 - axis[2] ** 2))) / dels
    arr[2] = c2 * (d1 - E)

    arr[1] = (4 * pi * axis[0] * axis[1] * axis[2]) / dels - arr[0] - arr[2]

    arr1 = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                arr1[i][j] = (arr[j] - arr[i]) / (axis[i] ** 2 - axis[j] ** 2)
    for i in range(3):
        tmp = 0
        for j in range(3):
            tmp += arr1[i][j] / 3
        arr1[i][i] = (4 * pi * axis[0] * axis[1] * axis[2]) / (3 * (axis[i] ** 2 + L) * dels) - tmp

    if flag == 0:
        return arr
    else:
        return arr1


def get_Sijkl(axis, Ii, Iij):
    Sijkl = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    t1 = deltaij[i, j] * deltaij[k, l] * (
                            2 * nu * Ii[i] - Ii[k] + (axis[i] ** 2) * Iij[k][i])
                    t2 = (deltaij[i, k] * deltaij[j, l] + deltaij[i, l] * deltaij[j, k]) * (
                            (axis[i] ** 2) * Iij[i][j] - Ii[j] + (1 - nu) * (Ii[k] + Ii[l]))
                    Sijkl[i, j, k, l] = (t1 + t2) / (8 * np.pi * (1 - nu))
    Sijkl[i, j, k, l] = Sijkl[j, i, k, l]
    Sijkl[i, j, k, l] = Sijkl[i, j, l, k]

    return Sijkl


def lamb_der(x, a, lambda_):
    arr = []
    # Check if x in inside
    if (x[0] ** 2 / (a[0] ** 2)) + (x[1] ** 2 / (a[1] ** 2)) + (x[2] ** 2 / (a[2] ** 2)) <= 1:
        return [0, 0, 0]
    denom = (x[0] ** 2 / ((a[0] ** 2 + lambda_) ** 2)) + (x[1] ** 2 / ((a[1] ** 2 + lambda_) ** 2)) + (
            x[2] ** 2 / ((a[2] ** 2 + lambda_) ** 2))
    for i in range(3):
        num = (2 * x[i]) / (a[i] ** 2 + lambda_)
        arr.append(num / denom)
    return arr


# Compute double derivative matrix of lambda


def lamb_der2(x, a, lambda_, lambda_der):
    arr = np.zeros((3, 3))
    if (x[0] ** 2 / (a[0] ** 2)) + (x[1] ** 2 / (a[1] ** 2)) + (x[2] ** 2 / (a[2] ** 2)) <= 1:
        return arr
    C = x[0] ** 2 / (a[0] ** 2 + lambda_) ** 2 + x[1] ** 2 / (a[1] ** 2 + lambda_) ** 2 + x[2] ** 2 / (
                a[2] ** 2 + lambda_) ** 2
    for i in range(3):
        for j in range(3):
            n = -(2 * x[i] * lambda_der[j]) / ((a[i] ** 2 + lambda_) ** 2) + 2 * lambda_der[i] * lambda_der[j] * (
                        (x[0] ** 2) / (a[0] ** 2 + lambda_) ** 3 + (x[1] ** 2) / (a[1] ** 2 + lambda_) ** 3 + (
                            x[2] ** 2) / (a[2] ** 2 + lambda_) ** 3)
            d = C
            arr[i][j] = n / d
    return arr


# Compute derivative of Ii wrt to j direction


def Ii_j_(a, lambda_, lambda_der):
    arr = np.zeros((3, 3))
    c = -2 * np.pi * a[0] * a[1] * a[2]
    del_l = ((a[0] ** 2 + lambda_) * (a[1] ** 2 + lambda_) * (a[2] ** 2 + lambda_)) ** (0.5)
    for i in range(3):
        for j in range(3):
            arr[i, j] = c * lambda_der[j] / ((a[i] ** 2 + lambda_) * del_l)
    return arr


# Compute derivative of Iij wrt to k direction


def Iij_k_(a, lambda_, lambda_der):
    arr = np.zeros((3, 3, 3))
    c = -2 * np.pi * a[0] * a[1] * a[2]
    del_l = ((a[0] ** 2 + lambda_) * (a[1] ** 2 + lambda_) * (a[2] ** 2 + lambda_)) ** (0.5)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                arr[i, j, k] = c * lambda_der[k] / ((a[i] ** 2 + lambda_) * (a[j] ** 2 + lambda_) * del_l)
    return arr


# Compute double partial derivative of Iij wrt to k and l direction


def Iij_kl_(a, lambda_, lambda_der, lambda_der2):
    arr = np.zeros((3, 3, 3, 3))
    c = -2 * np.pi * a[0] * a[1] * a[2]
    del_l = ((a[0] ** 2 + lambda_) * (a[1] ** 2 + lambda_) * (a[2] ** 2 + lambda_)) ** (1 / 2)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    arr[i, j, k, l] = (c / ((a[i] ** 2 + lambda_) * (a[j] ** 2 + lambda_) * del_l)) * (
                                lambda_der2[k, l] - lambda_der[k] * lambda_der[l] * (
                                    1 / (a[i] ** 2 + lambda_) + 1 / (a[j] ** 2 + lambda_) + 0.5 * (
                                        1 / (a[0] ** 2 + lambda_) + 1 / (a[1] ** 2 + lambda_) + 1 / (
                                            a[2] ** 2 + lambda_))))
    return arr


# Compute derivative of Ii wrt to j and k direction


def Ii_jk_(a, lambda_, lambda_der, lambda_der2):
    arr = np.zeros((3, 3, 3))
    c = -2 * np.pi * a[0] * a[1] * a[2]
    del_l = ((a[0] ** 2 + lambda_) * (a[1] ** 2 + lambda_) * (a[2] ** 2 + lambda_)) ** (1 / 2)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                arr[i, j, k] = (c / ((a[i] ** 2 + lambda_) * del_l)) * (
                            lambda_der2[j, k] - lambda_der[j] * lambda_der[k] * (1 / (a[i] ** 2 + lambda_) + 0.5 * (
                                1 / (a[0] ** 2 + lambda_) + 1 / (a[1] ** 2 + lambda_) + 1 / (a[2] ** 2 + lambda_))))
    return arr


def get_Dijkl(s, delta, x, a, IIj, IIJk, IIJkl, IIjk):
    Dijkl = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    t1 = 8 * pi * (1 - nu) * s[i, j, k, l] + 2 * nu * delta[k, l] * x[i] * IIj[i, j]
                    t2 = (1 - nu) * (
                                delta[i, l] * x[k] * IIj[k, j] + delta[j, l] * x[k] * IIj[k, i] + delta[i, k] * x[l] *
                                IIj[l, j] + delta[j, k] * x[l] * IIj[l, i])
                    t3 = delta[i, j] * x[k] * (IIj[k, l] - (a[i] ** 2) * IIJk[k, i, l]) + (
                                delta[i, k] * x[j] + delta[j, k] * x[i]) * (IIj[j, l] - (a[i] ** 2) * IIJk[i, j, l])
                    t4 = (delta[i, l] * x[j] + delta[j, l] * x[i]) * (IIj[j, k] - (a[i] ** 2) * IIJk[i, j, k]) + x[i] * \
                         x[j] * (IIjk[j, l, k] - (a[i] ** 2) * IIJkl[i, j, l, k])
                    Dijkl[i, j, k, l] = (t1 + t2 - t3 - t4) / (8 * pi * (1 - nu))
    return Dijkl


def calc_interior():
    # Calculating theta and K

    theta = np.arcsin(np.sqrt((a ** 2 - c ** 2) / a ** 2))
    k = np.sqrt((a ** 2 - b ** 2) / (a ** 2 - c ** 2))

    F = sp.ellipkinc(theta, k ** 2)
    E = sp.ellipeinc(theta, k ** 2)
    I1 = (4 * pi * a * b * c) * (F - E) / ((a ** 2 - b ** 2) * np.sqrt(a ** 2 - c ** 2))
    I3 = ((4 * pi * a * b * c) * ((b * np.sqrt(a ** 2 - c ** 2)) / (a * c) - E)) / (
            (b ** 2 - c ** 2) * np.sqrt(a ** 2 - c ** 2))
    I2 = 4 * pi - I1 - I3

    # Solvig equations for I11 and I13

    I12 = (I2 - I1) / (a ** 2 - b ** 2)
    x, y = sym.symbols('x y')
    eq1 = sym.Eq(3 * x + I12 + y, 4 * pi / (a ** 2))
    eq2 = sym.Eq(3 * (a ** 2) * x + (b ** 2) * I12 + (c ** 2) * y, 3 * I1)
    ans = sym.solve([eq1, eq2], (x, y))
    I11 = ans[x]
    I13 = ans[y]

    # Solving equations for I21 and I22

    I23 = (I3 - I2) / (b ** 2 - c ** 2)
    x, y = sym.symbols('x y')
    eq1 = sym.Eq(3 * x + I23 + y, 4 * pi / (b ** 2))
    eq2 = sym.Eq(3 * (b ** 2) * x + (c ** 2) * I23 + (a ** 2) * y, 3 * I2)
    ans = sym.solve([eq1, eq2], (x, y))
    I21 = ans[y]
    I22 = ans[x]

    # Solving equation for I33 and I32

    I31 = (I1 - I3) / (c ** 2 - a ** 2)
    x, y = sym.symbols('x y')
    eq1 = sym.Eq(3 * x + I31 + y, 4 * pi / (c ** 2))
    eq2 = sym.Eq(3 * (c ** 2) * x + (a ** 2) * I31 + (b ** 2) * y, 3 * I3)
    ans = sym.solve([eq1, eq2], (x, y))
    I32 = ans[y]
    I33 = ans[x]

    # Solving for sigma

    # t1 t2 t3 are terms of sigma11

    t1 = (((a ** 2) * (3 * I11 - 3 * nu * I11 + nu * I21 + nu * I31)) / (8 * pi * (1 - nu) * (1 - 2 * nu)) + (
            (I1 - (nu / (1 - nu)) * (I2 + I3)) / (8 * pi)) - ((1 - nu) / (1 - 2 * nu))) * eps11

    t2 = (((b ** 2) * (I12 - I12 * nu + 3 * nu * I22 + nu * I32)) / (8 * pi * (1 - nu) * (1 - 2 * nu)) - (
            I1 - (nu / 1 - nu) * (I2 - I3)) / (8 * pi) - (nu) / (1 - 2 * nu)) * eps22

    t3 = (((c ** 2) * (I13 - nu * I13 + 3 * nu * I33 + nu * I23)) / (8 * pi * (1 - nu) * (1 - 2 * nu)) - (
            I1 - (nu / 1 - nu) * (I3 - I2)) / (8 * pi) - (nu) / (1 - 2 * nu)) * eps33

    sigma11 = 2 * mu * (t1 + t2 + t3)

    sigma12 = ((((a ** 2 + b ** 2) * I12 + (1 - 2 * nu) * (I1 + I2)) / (8 * pi * (1 - nu))) - 1) * eps12 * 2 * mu
    sigma23 = ((((b ** 2 + c ** 2) * I12 + (1 - 2 * nu) * (I2 + I3)) / (8 * pi * (1 - nu))) - 1) * eps23 * 2 * mu
    sigma31 = ((((a ** 2 + c ** 2) * I12 + (1 - 2 * nu) * (I1 + I3)) / (8 * pi * (1 - nu))) - 1) * eps31 * 2 * mu

    # q1 q2 q3 are for sigma22

    q1 = (((b ** 2) * (3 * I22 - 3 * nu * I22 + nu * I32 + nu * I12)) / (8 * pi * (1 - nu) * (1 - 2 * nu)) + (
            (I2 - (nu / (1 - nu)) * (I1 + I3)) / (8 * pi)) - ((1 - nu) / (1 - 2 * nu))) * eps22
    q2 = (((c ** 2) * (I23 - I23 * nu + 3 * nu * I33 + nu * I13)) / (8 * pi * (1 - nu) * (1 - 2 * nu)) - (
            I2 - (nu / 1 - nu) * (I3 - I1)) / (8 * pi) - (nu) / (1 - 2 * nu)) * eps33
    q3 = (((a ** 2) * (I21 - nu * I21 + 3 * nu * I11 + nu * I31)) / (8 * pi * (1 - nu) * (1 - 2 * nu)) - (
            I2 - (nu / 1 - nu) * (I1 - I3)) / (8 * pi) - (nu) / (1 - 2 * nu)) * eps11

    sigma22 = 2 * mu * (q1 + q2 + q3)

    # w1 w2 w3 are for sigma33

    w1 = (((c ** 2) * (3 * I33 - 3 * nu * I33 + nu * I13 + nu * I23)) / (8 * pi * (1 - nu) * (1 - 2 * nu)) + (
            (I3 - (nu / (1 - nu)) * (I2 + I1)) / (8 * pi)) - ((1 - nu) / (1 - 2 * nu))) * eps33
    w2 = (((a ** 2) * (I31 - I31 * nu + 3 * nu * I11 + nu * I21)) / (8 * pi * (1 - nu) * (1 - 2 * nu)) - (
            I3 - (nu / 1 - nu) * (I1 - I2)) / (8 * pi) - (nu) / (1 - 2 * nu)) * eps11
    w3 = (((b ** 2) * (I32 - nu * I32 + 3 * nu * I22 + nu * I12)) / (8 * pi * (1 - nu) * (1 - 2 * nu)) - (
            I3 - (nu / 1 - nu) * (I2 - I1)) / (8 * pi) - (nu) / (1 - 2 * nu)) * eps22

    sigma33 = 2 * mu * (w1 + w2 + w3)

    stress_inside = np.array([[sigma11, sigma12, sigma31], [sigma12, sigma22, sigma23], [sigma31, sigma23, sigma33]])

    # print(stress_inside)

    '''
    lbd = find_lambda(X, axis)
    print("Lambda is: ", lbd)
    I = IIJ(axis, lbd, 1, 1)
    Ii = IIJ(axis, lbd, 2, 2)
    Iij = IIJ(axis, lbd, 3, 3)
    print("I=", I)
    print("Ii= ", Ii)
    print("Iij= ", Iij)
    Sijkl = get_Sijkl(axis, I, Ii, Iij)
    print("Sijkl is:")
    print(Sijkl)
    lbd_der = lamb_der(X, axis, lbd)
    lbd_d_der = lamb_der2(X, axis, lbd, lbd_der)
    print("lambda der is: ")
    print(lbd_der)
    print("lambda der 2 is: ")
    print(lbd_d_der)
    IIj = Ii_j_(axis, lbd, lbd_der)
    IIJk = Iij_k_(axis, lbd, lbd_der)
    IIJkl = Iij_kl_(axis, lbd, lbd_der, lbd_d_der)
    IIjk = Ii_jk_(axis, lbd, lbd_der, lbd_d_der)
    print("IIj is: ")
    print(IIj)
    print("IIJk is: ")
    print(IIJk)
    print("IIjk is: ")
    print(IIjk)
    print("IIJkl is: ")
    print(IIJkl)
    Dijkl = get_Dijkl(Sijkl, deltaij, X, axis, IIj, IIJk, IIJkl, IIjk)
    print("Dijkl is:")
    print(Dijkl)'''

    return [stress_inside[0][0], stress_inside[1][1], stress_inside[2][2]]


def calc_exterior():
    epsilon_star = [[eps11, eps12, eps31], [eps12, eps22, eps23], [eps31, eps23, eps33]]

    lbd = find_lambda(X, axis)
    # print("Lambda is", lbd)
    # print("Lambda is: ", lbd)
    #print("Residual", X[0]**2/(axis[0]**2 + lbd) + X[1]**2/(axis[1]**2 + lbd) + X[2]**2/(axis[2]**2 + lbd) - 1)

    # Calculating I, I1, I2, I3, I11, I22, I33, etc

    Ii = IIJ(axis, lbd, 0)
    Iij = IIJ(axis, lbd, 2)
    # print(Ii)
    # print(Iij)

    Sijkl = get_Sijkl(axis, Ii, Iij)

    lbd_der = lamb_der(X, axis, lbd)
    lbd_d_der = lamb_der2(X, axis, lbd, lbd_der)

    IIj = Ii_j_(axis, lbd, lbd_der)
    IIJk = Iij_k_(axis, lbd, lbd_der)
    IIJkl = Iij_kl_(axis, lbd, lbd_der, lbd_d_der)
    IIjk = Ii_jk_(axis, lbd, lbd_der, lbd_d_der)

    Dijkl = get_Dijkl(Sijkl, deltaij, X, axis, IIj, IIJk, IIJkl, IIjk)
    #print("Dijkl :")
    #print(Dijkl)
    epsilon = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    epsilon[i][j] += Dijkl[i, j, k, l] * epsilon_star[k][l]

    #print("EPSILON")
    #print(epsilon)
    stress_outside = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    stress_outside[i][j] += Cijkl[i][j][k][l] * epsilon[k][l]
    # print(stress_outside)

    # print("EPSILON")
    # print(epsilon)
    # print("STRESS")
    # print(stress_outside)
    print(stress_outside[0][0] + stress_outside[1][1] + stress_outside[2][2])

    return [stress_outside[0][0], stress_outside[1][1], stress_outside[2][2]]


'''
    sig11 = (E/((1+nu)*(1-2*nu)))*((1-nu)*epsilon[0,0] + nu*epsilon[1,1] + nu*epsilon[2,2])
    sig22 = (E/((1+nu)*(1-2*nu)))*(nu*epsilon[0,0] + (1-nu)*epsilon[1,1] + nu*epsilon[2,2])
    sig33 = (E/((1+nu)*(1-2*nu)))*(nu*epsilon[0,0] + nu*epsilon[1,1] + (1-nu)*epsilon[2,2])
    sig12 = (E/(1+nu))*epsilon[0,1]
    sig13 = (E/(1+nu))*epsilon[0,2]
    sig23 = (E/(1+nu))*epsilon[1,2]
    stress_outside = np.array([[sig11,sig12,sig13],[sig12,sig22,sig23],[sig13,sig23,sig33]])
'''

"""
a = 24.001
b = 24
c = 23.999
eps11 = 0.001
eps22 = 0.001
eps33 = 0.001
eps12 = 0
eps23 = 0
eps31 = 0
E = 210
nu = 0.3
mu = E/(2*(1+nu))
axis = [a, b, c]
"""

a = float(input())
b = float(input())
c = float(input())
axis = [a, b, c]
eps11 = float(input())
eps22 = float(input())
eps33 = float(input())
eps12 = float(input())
eps23 = float(input())
eps31 = float(input())

E = float(input())  # Young's Modulus
nu = float(input())  # Poisson's Ratio
mu = E / (2 * (1 + nu))
lamda = 2 * mu * nu / (1 - 2 * nu)
Cijkl = np.zeros((3, 3, 3, 3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                Cijkl[i][j][k][l] = lamda * deltaij[i][j] * deltaij[k][l] + mu * (
                            deltaij[i][k] * deltaij[j][l] + deltaij[i][l] * deltaij[j][k])
                Cijkl[k][l][i][j] = Cijkl[i][j][k][l]

'''
x1 = float(input())
x2 = float(input())
x3 = float(input())
X = [x1, x2, x3]


if(x1**2/a**2 + x2**2/b**2 + x3**2/c**2 <= 1):
    print("Point is inside")
    calc_interior()

else:
    print("Point is outside")
    calc_exterior()
'''

ab = np.linspace(0, 3*a, num=100)

s = []
s1 = []
s2 = []

for x in ab:
    X = [x, 0, 0]
    #print("X is ", x)

    if x < a:
        ans = calc_interior()
        s.append(ans[0])
        s1.append(ans[1])
        s2.append(ans[2])
    else:
        ans = calc_exterior()
        s.append(ans[0])
        s1.append(ans[1])
        s2.append(ans[2])

# print(ab)
# print(s)


plt.plot(ab, s, label='sigmaXX')
plt.plot(ab, s1, label='sigmaYY')
plt.plot(ab, s2, label='sigmaZZ')
#plt.axhline(y=0)
plt.xlabel("x ( Displacement)")
plt.ylabel(" Stress")
plt.legend()
plt.show()
