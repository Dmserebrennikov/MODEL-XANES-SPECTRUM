import math
import numpy as np
import sympy as sym 
from abc import ABCMeta, abstractmethod

with open('D:\\Smexp.txt','r') as f:
    E = []
    I_exp = []
    for line in f:
        try:
            line = line.strip()
            E_value, I_exp_value = map(float, re.split('\s+', line))
        except ValueError as err:
            continue
        E.append(E_value)
        I_exp.append(I_exp_value)

E1 = 6716.9
E2 = 6723.5
initialPoint = (-1, 3, 0.7, 1.2)

def function(x): 
    f_a = 0
    for i in range(len(E)):
        f_a = f_a + ((x[2]*(x[1]**2)/(x[1]**2 + (E[i]-E1)**2))+(x[3]*(x[1]**2)/(x[1]**2 + (E[i]-E2)**2))+(x[2]*(0.5 +(0.31847*math.atan((E[i]-(E1+x[0]))/(0.5*x[1]))))/(x[2]+x[3]))+(x[3]*(0.5 + (0.31847*math.atan((E[i]-(E2+x[0]))/(0.5*x[1]))))/(x[2]+x[3])) - I_exp[i])**2 
    return (f_a)

x0 = sym.Symbol('x0')
x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2')
x3 = sym.Symbol('x3')
Ee = sym.Symbol('Ee')
Ii = sym.Symbol('Ii')
f = ((x2*(x1**2)/(x1**2 + (Ee-E1)**2))+(x3*(x1**2)/(x1**2 + (Ee-E2)**2))+(x2*(0.5 +(0.31847*sym.atan((Ee-(E1+x0))/(0.5*x1))))/(x2+x3))+(x3*(0.5 + (0.31847*sym.atan((Ee-(E2+x0))/(0.5*x1))))/(x2+x3)) - Ii)**2 
f_d_x0 = sym.diff(f, x0, 1)
f_d_x1 = sym.diff(f, x1, 1)
f_d_x2 = sym.diff(f, x2, 1)
f_d_x3 = sym.diff(f, x3, 1)
f_d_x0x0 = sym.diff(f_d_x0, x0, 1)
f_d_x0x1 = sym.diff(f_d_x0, x1, 1)
f_d_x0x2 = sym.diff(f_d_x0, x2, 1)
f_d_x0x3 = sym.diff(f_d_x0, x3, 1)
f_d_x1x0 = sym.diff(f_d_x1, x0, 1)
f_d_x1x1 = sym.diff(f_d_x1, x1, 1)
f_d_x1x2 = sym.diff(f_d_x1, x2, 1)
f_d_x1x3 = sym.diff(f_d_x1, x3, 1)
f_d_x2x0 = sym.diff(f_d_x2, x0, 1)
f_d_x2x1 = sym.diff(f_d_x2, x1, 1)
f_d_x2x2 = sym.diff(f_d_x2, x2, 1)
f_d_x2x3 = sym.diff(f_d_x2, x3, 1)
f_d_x3x0 = sym.diff(f_d_x3, x0, 1)
f_d_x3x1 = sym.diff(f_d_x3, x1, 1)
f_d_x3x2 = sym.diff(f_d_x3, x2, 1)
f_d_x3x3 = sym.diff(f_d_x3, x3, 1)

def gradient(x):
    return np.array([(derivative_x0(x)),(derivative_x1(x)), (derivative_x2(x)), (derivative_x3(x))])

def function_array(x):
    F_ar = []
    for i in range(len(E)):
        f_ar =((x[2]*(x[1]**2)/(x[1]**2 + (E[i]-E1)**2))+(x[3]*(x[1]**2)/(x[1]**2 + (E[i]-E2)**2))+(x[2]*(0.5 +(0.31847*math.atan((E[i]-(E1+x[0]))/(0.5*x[1]))))/(x[2]+x[3]))+(x[3]*(0.5 + (0.31847*math.atan((E[i]-(E2+x[0]))/(0.5*x[1]))))/(x[2]+x[3])) - I_exp[i])**2 
        F_ar.append(f_ar)
    return np.array(F_ar, dtype = float).reshape(len(E),1)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def is_sequence(arg):
    return isinstance(arg, np.ndarray) or isinstance(arg, list)

def jacobi(x):
    aa = []
    bb = []
    cc = []
    dd = []
    for i in range(len(E)):
        I_exper = I_exp[i]
        dr_i_x0 = f_d_x0.subs([(x0, x[0]), (x1, x[1]), (x2, x[2]), (x3, x[3]), (Ee, E[i]), (Ii, I_exper)])
        aa.append(dr_i_x0)
        dr_i_x1 = f_d_x1.subs([(x0, x[0]), (x1, x[1]), (x2, x[2]), (x3, x[3]), (Ee, E[i]), (Ii, I_exper)])
        bb.append(dr_i_x1)
        dr_i_x2 = f_d_x2.subs([(x0, x[0]), (x1, x[1]), (x2, x[2]), (x3, x[3]), (Ee, E[i]), (Ii, I_exper)])
        cc.append(dr_i_x2)
        dr_i_x3 = f_d_x3.subs([(x0, x[0]), (x1, x[1]), (x2, x[2]), (x3, x[3]), (Ee, E[i]), (Ii, I_exper)])
        dd.append(dr_i_x3)
    aa = np.array(aa, dtype = float).reshape(len(E),1)
    bb = np.array(bb, dtype = float).reshape(len(E),1)
    cc = np.array(cc, dtype = float).reshape(len(E),1)
    dd = np.array(dd, dtype = float).reshape(len(E),1)
    return np.hstack((aa,bb,cc,dd))



class NewtonGaussOptimizer:
        def __init__(self, function, initialPoint, gradient=None, jacobi=None, function_array=function_array, learningRate=0.05):
            self.learningRate = learningRate
            self.function = function
            self.gradient = gradient
            self.jacobi = jacobi
            self.x = initialPoint
            self.y = self.function(initialPoint)
            self.function_array = function_array
            
        def next_point(self):
            # Solve (J_t * J)d_ng = -J*f
            jacobi = self.jacobi(self.x)
            jacobisLeft = np.dot(jacobi.T, jacobi)
            jacobiLeftInverse = np.linalg.inv(jacobisLeft)
            jjj = np.dot(jacobiLeftInverse, jacobi.T)  # (J_t * J)^-1 * J_t
            nextX = self.x - self.learningRate * np.dot(jjj, self.function_array(self.x)).reshape((-1))
            return self.move_next(nextX)
        
        def move_next(self, nextX):
            nextY = self.function(nextX)
            self.y = nextY
            self.x = nextX
            return self.x, self.y

maxIteration = 100
currentIteration = 0

optimizer = NewtonGaussOptimizer(function, initialPoint, gradient, jacobi)

while currentIteration <= maxIteration:

        yBefore = optimizer.y
        xBefore = optimizer.x

        x, y = optimizer.next_point()
        if is_sequence(y):
            print('Обрати внимание',y)
            y = y[0]
        if is_sequence(yBefore):
            print('Обрати внимание',y)
            yBefore = yBefore[0]

        
        print("Iteration = {0} b = {1} G = {2} P1 = {3} P2 = {4} Z = {5}".format(currentIteration, x[0],
                                                                               x[1], x[2], x[3], y))
        currentIteration+=1
        if y < 1E-7:
            break
