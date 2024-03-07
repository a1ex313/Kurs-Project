import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import dia_matrix
from scipy import integrate
from scipy import linalg as LA
from scipy.optimize import minimize
from scipy import optimize
from mpmath import *
import time
import multiprocessing

start = time.time()

#Общие параметры
N = 4
x0 = 0
z0 = 6
l = 0.5
L = 100.0
R = 6.0


#Параметры для фи1
a1 = (0.5)*(math.sqrt(l))
b1 = (0.5)*(math.sqrt(l))
c1 = (0.5)*(math.sqrt(l))

#Параметры для фи2
a4 = 0.445
b4 = 0.445
c4 = 0.445

#Параметры для фи21
a2 = 0.2
b2 = 0.2
c2 = 0.2

#Параметры для фи22
a3 = 1.313
b3 = 1.313
c3 = 1.313

def phi1(x, y, z):
    return z*math.exp(-a1*(x-x0)**2)*math.exp(-b1*y**2)*math.exp(-c1*z**2)
def phi2(x, y, z):
    return math.exp(-a4*x**2)*math.exp(-b4*y**2)*math.exp(-c4*(z-z0)**2)
def phi21(x, y, z):
    return math.exp(-a2*x**2)*math.exp(-b2*y**2)*math.exp(-c2*(z-z0)**2)
def phi22(x, y, z):
    return math.exp(-a3*x**2)*math.exp(-b3*y**2)*math.exp(-c3*(z-z0)**2)



def phi1_dr2(x, y, z):
    return (2*a1*z*(2*a1*((x0-x)**2)-1)*math.exp((-a1*(x0-x)**2)-(b1*y**2)-(c1*z**2))) + (2*b1*z*(2*b1*(y**2)-1)*math.exp((-a1*(x0-x)**2)-(b1*y**2)-(c1*z**2))) + (2*c1*z*(2*c1*(z**2)-3)*math.exp((-a1*(x0-x)**2)-(b1*y**2)-(c1*z**2)))
def phi2_dr2(x, y, z):
    return (2 * a4 * (2 * a4 * (x ** 2) - 1) * math.exp((-a4 * x ** 2) - (b4 * y ** 2) - (c4 * (z0 - z) ** 2))) + (
                2 * b4 * (2 * b4 * (y ** 2) - 1) * math.exp((-a4 * x ** 2) - (b4 * y ** 2) - (c4 * (z0 - z) ** 2))) + (
                2 * c4 * (2 * c4 * ((z0 - z) ** 2) - 1) * math.exp(
            (-a4 * x ** 2) - (b4 * y ** 2) - (c4 * (z0 - z) ** 2)))
def phi21_dr2(x, y, z):
    return (2*a2*(2*a2*(x**2)-1)*math.exp((-a2*x**2)-(b2*y**2)-(c2*(z0-z)**2))) + (2*b2*(2*b2*(y**2)-1)*math.exp((-a2*x**2)-(b2*y**2)-(c2*(z0-z)**2))) + (2*c2*(2*c2*((z0-z)**2)-1)*math.exp((-a2*x**2)-(b2*y**2)-(c2*(z0-z)**2)))
def phi22_dr2(x, y, z):
    return (2*a3*(2*a3*(x**2)-1)*math.exp((-a3*x**2)-(b3*y**2)-(c3*(z0-z)**2))) + (2*b3*(2*b3*(y**2)-1)*math.exp((-a3*x**2)-(b3*y**2)-(c3*(z0-z)**2))) + (2*c3*(2*c3*((z0-z)**2)-1)*math.exp((-a3*x**2)-(b3*y**2)-(c3*(z0-z)**2)))



def phi1_(x, y, z):
    #if math.sqrt(x ** 2 + y ** 2 + (z - z0) ** 2) > 0.35:
    return 2 * (z * math.exp(-a1 * (x - x0) ** 2) * math.exp(-b1 * y ** 2) * math.exp(-c1 * z ** 2))/math.sqrt(x**2+y**2+(z-z0)**2) - l*(x**2+y**2+z**2) * (z * math.exp(-a1 * (x - x0) ** 2) * math.exp(-b1 * y ** 2) * math.exp(-c1 * z ** 2))/2
    #else:
    #    return 0.0
def phi2_(x, y, z):
    #if math.sqrt(x ** 2 + y ** 2 + (z - z0) ** 2) > 0.35:
    return 2 * (math.exp(-a4*x**2)*math.exp(-b4*y**2)*math.exp(-c4*(z-z0)**2))/math.sqrt(x**2+y**2+(z-z0)**2) - l*(x**2+y**2+z**2) * (math.exp(-a4*x**2)*math.exp(-b4*y**2)*math.exp(-c4*(z-z0)**2))/2
    #else:
    #    return 0.0
def phi21_(x, y, z):
    #if math.sqrt(x ** 2 + y ** 2 + (z - z0) ** 2) > 0.35:
    return 2 * (math.exp(-a2*x**2)*math.exp(-b2*y**2)*math.exp(-c2*(z-z0)**2))/math.sqrt(x**2+y**2+(z-z0)**2) - l*(x**2+y**2+z**2) * (math.exp(-a2*x**2)*math.exp(-b2*y**2)*math.exp(-c2*(z-z0)**2))/2
    #else:
    #    return 0.0
def phi22_(x, y, z):
    #if math.sqrt(x ** 2 + y ** 2 + (z - z0) ** 2) > 0.35:
    return 2 * (math.exp(-a3*x**2)*math.exp(-b3*y**2)*math.exp(-c3*(z-z0)**2))/math.sqrt(x**2+y**2+(z-z0)**2) - l*(x**2+y**2+z**2) * (math.exp(-a3*x**2)*math.exp(-b3*y**2)*math.exp(-c3*(z-z0)**2))/2
    #else:
    #    return 0.0



def phi1_r(r, teta, phi):
    x = r*math.sin(teta)*math.cos(phi)
    y = r*math.sin(teta)*math.cos(phi)
    z = r*math.cos(teta)
    return z*math.exp(-a1*(x-x0)**2)*math.exp(-b1*y**2)*math.exp(-c1*z**2)*J(r, teta)
def phi2_r(r, teta, phi):
    x = r*math.sin(teta)*math.cos(phi)
    y = r*math.sin(teta)*math.cos(phi)
    z = r*math.cos(teta)
    return math.exp(-a4*x**2)*math.exp(-b4*y**2)*math.exp(-c4*(z-z0)**2)*J(r, teta)
def phi21_r(r, teta, phi):
    x = r*math.sin(teta)*math.cos(phi)
    y = r*math.sin(teta)*math.cos(phi)
    z = r*math.cos(teta)
    return math.exp(-a2*x**2)*math.exp(-b2*y**2)*math.exp(-c2*(z-z0)**2)*J(r, teta)
def phi22_r(r, teta, phi):
    x = r*math.sin(teta)*math.cos(phi)
    y = r*math.sin(teta)*math.cos(phi)
    z = r*math.cos(teta)
    return math.exp(-a3*x**2)*math.exp(-b3*y**2)*math.exp(-c3*(z-z0)**2)*J(r, teta)



def phi1_r_(r, teta, phi):
    x = r*math.sin(teta)*math.cos(phi)
    y = r*math.sin(teta)*math.cos(phi)
    z = r*math.cos(teta)+ z0
    return 2 * (z * math.exp(-a1 * (x - x0) ** 2) * math.exp(-b1 * y ** 2) * math.exp(-c1 * z ** 2)) / 0.05 - l * (
                x ** 2 + y ** 2 + z ** 2) * (
                z * math.exp(-a1 * (x - x0) ** 2) * math.exp(-b1 * y ** 2) * math.exp(-c1 * z ** 2)) / 2
def phi2_r_(r, teta, phi):
    x = r*math.sin(teta)*math.cos(phi)
    y = r*math.sin(teta)*math.cos(phi)
    z = r*math.cos(teta)+ z0
    return 2 * (math.exp(-a4 * (x - x0) ** 2) * math.exp(-b4 * y ** 2) * math.exp(-c4 * z ** 2)) / 0.05 - l * (
                x ** 2 + y ** 2 + z ** 2) * (
                math.exp(-a4 * (x - x0) ** 2) * math.exp(-b4 * y ** 2) * math.exp(-c4 * z ** 2)) / 2
def phi21_r_(r, teta, phi):
    x = r*math.sin(teta)*math.cos(phi)
    y = r*math.sin(teta)*math.cos(phi)
    z = r*math.cos(teta)+ z0
    return 2 * (math.exp(-a2 * x ** 2) * math.exp(-b2 * y ** 2) * math.exp(-c2 * (z - z0) ** 2)) / 0.05 - l * (
                x ** 2 + y ** 2 + z ** 2) * (math.exp(-a2 * x ** 2) * math.exp(-b2 * y ** 2) * math.exp(-c2 * (z - z0) ** 2)) / 2
def phi22_r_(r, teta, phi):
    x = r*math.sin(teta)*math.cos(phi)
    y = r*math.sin(teta)*math.cos(phi)
    z = r*math.cos(teta) + z0
    return 2 * (math.exp(-a3 * x ** 2) * math.exp(-b3 * y ** 2) * math.exp(-c3 * (z - z0) ** 2)) / 0.05 - l * (
                x ** 2 + y ** 2 + z ** 2) * (math.exp(-a3 * x ** 2) * math.exp(-b3 * y ** 2) * math.exp(-c3 * (z - z0) ** 2)) / 2


def J(r, teta):
    return r * math.sin(teta)





def integrate_phi00():
    res = integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi1(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi01():
    res = integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi02():
    res = integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi21(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi03():
    res = integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi22(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res

def integrate_phi10():
    res = integrate.tplquad(lambda x, y, z: phi2(x, y, z) * phi1(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi11():
    res = integrate.tplquad(lambda x, y, z: phi2(x, y, z) * phi2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi12():
    res = integrate.tplquad(lambda x, y, z: phi2(x, y, z) * phi21(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi13():
    res = integrate.tplquad(lambda x, y, z: phi2(x, y, z) * phi22(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res

def integrate_phi20():
    res = integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi1(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi21():
    res = integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi22():
    res = integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi21(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi23():
    res = integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi22(x, y, z), 0, L, -L, L, -L, 40, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res

def integrate_phi30():
    res = integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi1(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi31():
    res = integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi32():
    res = integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi21(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_phi33():
    res = integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi22(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res



def integrate_Aphi00_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi1_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi01_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi2_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi02_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi21_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi03_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi22_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res

def integrate_Aphi10_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi2(x, y, z) * phi1_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi11_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi2(x, y, z) * phi2_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi12_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi2(x, y, z) * phi21_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi13_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi2(x, y, z) * phi22_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res

def integrate_Aphi20_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi1_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi21_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi2_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi22_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi21_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi23_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi22_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res

def integrate_Aphi30_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi1_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi31_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi2_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi32_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi21_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi33_1():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi22_dr2(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res



def integrate_Aphi00_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi1_r(r, teta, phi) * phi1_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi01_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi1_r(r, teta, phi) * phi2_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi02_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi1_r(r, teta, phi) * phi21_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi03_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi1_r(r, teta, phi) * phi22_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res

def integrate_Aphi10_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi2_r(r, teta, phi) * phi1_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi11_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi2_r(r, teta, phi) * phi2_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi12_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi2_r(r, teta, phi) * phi21_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi13_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi2_r(r, teta, phi) * phi22_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res

def integrate_Aphi20_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi21_r(r, teta, phi) * phi1_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi21_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi21_r(r, teta, phi) * phi2_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi22_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi21_r(r, teta, phi) * phi21_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi23_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi21_r(r, teta, phi) * phi22_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res

def integrate_Aphi30_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi22_r(r, teta, phi) * phi1_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi31_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi22_r(r, teta, phi) * phi2_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi32_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi22_r(r, teta, phi) * phi21_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi33_2():
    res = (-1) * integrate.tplquad(lambda r, teta, phi: phi22_r(r, teta, phi) * phi22_r_(r, teta, phi), 0.0, R, 0.0, math.pi, 0.0, 2*math.pi, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res



def integrate_Aphi00_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi1_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi01_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi2_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi02_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi21_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi03_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi1(x, y, z) * phi22_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res

def integrate_Aphi10_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi2(x, y, z) * phi1_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi11_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi2(x, y, z) * phi2_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi12_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi2(x, y, z) * phi21_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi13_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi2(x, y, z) * phi22_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res

def integrate_Aphi20_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi1_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi21_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi2_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi22_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi21_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi23_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi21(x, y, z) * phi22_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res

def integrate_Aphi30_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi1_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi31_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi2_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi32_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi21_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res
def integrate_Aphi33_3():
    res = (-1) * integrate.tplquad(lambda x, y, z: phi22(x, y, z) * phi22_(x, y, z), 0, L, -L, L, -L, L, epsabs=1.49e-5, epsrel=1.49e-3)[0]
    return res


if __name__ == '__main__':
    def get_phi(N):
        start1 = time.time()
        phi_m = np.zeros(shape=(N, N))
        pool1 = multiprocessing.Pool(processes=8)

        res00 = pool1.apply_async(integrate_phi00)
        res01 = pool1.apply_async(integrate_phi01)
        res02 = pool1.apply_async(integrate_phi02)
        res03 = pool1.apply_async(integrate_phi03)

        res10 = pool1.apply_async(integrate_phi10)
        res11 = pool1.apply_async(integrate_phi11)
        res12 = pool1.apply_async(integrate_phi12)
        res13 = pool1.apply_async(integrate_phi13)

        phi_m[0][0] = res00.get()
        phi_m[0][1] = res01.get()
        phi_m[0][2] = res02.get()
        phi_m[0][3] = res03.get()

        phi_m[1][0] = res10.get()
        phi_m[1][1] = res11.get()
        phi_m[1][2] = res12.get()
        phi_m[1][3] = res13.get()

        pool2 = multiprocessing.Pool(processes=8)

        res20 = pool2.apply_async(integrate_phi20)
        res21 = pool2.apply_async(integrate_phi21)
        res22 = pool2.apply_async(integrate_phi22)
        res23 = pool2.apply_async(integrate_phi23)

        res30 = pool2.apply_async(integrate_phi30)
        res31 = pool2.apply_async(integrate_phi31)
        res32 = pool2.apply_async(integrate_phi32)
        res33 = pool2.apply_async(integrate_phi33)

        phi_m[2][0] = res20.get()
        phi_m[2][1] = res21.get()
        phi_m[2][2] = res22.get()
        phi_m[2][3] = res23.get()

        phi_m[3][0] = res30.get()
        phi_m[3][1] = res31.get()
        phi_m[3][2] = res32.get()
        phi_m[3][3] = res33.get()

        end1 = time.time()
        print("Время на матрицу phi")
        print(end1 - start1)
        print()

        return phi_m
    def get_Aphi_1(N):
        start2 = time.time()
        Aphi_m = np.zeros(shape=(N, N))
        pool1 = multiprocessing.Pool(processes=8)

        resA00 = pool1.apply_async(integrate_Aphi00_1)
        resA01 = pool1.apply_async(integrate_Aphi01_1)
        resA02 = pool1.apply_async(integrate_Aphi02_1)
        resA03 = pool1.apply_async(integrate_Aphi03_1)

        resA10 = pool1.apply_async(integrate_Aphi10_1)
        resA11 = pool1.apply_async(integrate_Aphi11_1)
        resA12 = pool1.apply_async(integrate_Aphi12_1)
        resA13 = pool1.apply_async(integrate_Aphi13_1)

        Aphi_m[0][0] = resA00.get()
        Aphi_m[0][1] = resA01.get()
        Aphi_m[0][2] = resA02.get()
        Aphi_m[0][3] = resA03.get()

        Aphi_m[1][0] = resA10.get()
        Aphi_m[1][1] = resA11.get()
        Aphi_m[1][2] = resA12.get()
        Aphi_m[1][3] = resA13.get()

        pool2 = multiprocessing.Pool(processes=8)

        resA20 = pool2.apply_async(integrate_Aphi20_1)
        resA21 = pool2.apply_async(integrate_Aphi21_1)
        resA22 = pool2.apply_async(integrate_Aphi22_1)
        resA23 = pool2.apply_async(integrate_Aphi23_1)

        resA30 = pool2.apply_async(integrate_Aphi30_1)
        resA31 = pool2.apply_async(integrate_Aphi31_1)
        resA32 = pool2.apply_async(integrate_Aphi32_1)
        resA33 = pool2.apply_async(integrate_Aphi33_1)

        Aphi_m[2][0] = resA20.get()
        Aphi_m[2][1] = resA21.get()
        Aphi_m[2][2] = resA22.get()
        Aphi_m[2][3] = resA23.get()

        Aphi_m[3][0] = resA30.get()
        Aphi_m[3][1] = resA31.get()
        Aphi_m[3][2] = resA32.get()
        Aphi_m[3][3] = resA33.get()

        end2 = time.time()
        print("Время на матрицу Aphi_1")
        print(end2 - start2)
        print()

        return Aphi_m
    def get_Aphi_2(N):
        start2 = time.time()
        Aphi_m = np.zeros(shape=(N, N))
        pool1 = multiprocessing.Pool(processes=8)

        resA00 = pool1.apply_async(integrate_Aphi00_2)
        resA01 = pool1.apply_async(integrate_Aphi01_2)
        resA02 = pool1.apply_async(integrate_Aphi02_2)
        resA03 = pool1.apply_async(integrate_Aphi03_2)

        resA10 = pool1.apply_async(integrate_Aphi10_2)
        resA11 = pool1.apply_async(integrate_Aphi11_2)
        resA12 = pool1.apply_async(integrate_Aphi12_2)
        resA13 = pool1.apply_async(integrate_Aphi13_2)

        Aphi_m[0][0] = resA00.get()
        Aphi_m[0][1] = resA01.get()
        Aphi_m[0][2] = resA02.get()
        Aphi_m[0][3] = resA03.get()

        Aphi_m[1][0] = resA10.get()
        Aphi_m[1][1] = resA11.get()
        Aphi_m[1][2] = resA12.get()
        Aphi_m[1][3] = resA13.get()

        pool2 = multiprocessing.Pool(processes=8)

        resA20 = pool2.apply_async(integrate_Aphi20_2)
        resA21 = pool2.apply_async(integrate_Aphi21_2)
        resA22 = pool2.apply_async(integrate_Aphi22_2)
        resA23 = pool2.apply_async(integrate_Aphi23_2)

        resA30 = pool2.apply_async(integrate_Aphi30_2)
        resA31 = pool2.apply_async(integrate_Aphi31_2)
        resA32 = pool2.apply_async(integrate_Aphi32_2)
        resA33 = pool2.apply_async(integrate_Aphi33_2)

        Aphi_m[2][0] = resA20.get()
        Aphi_m[2][1] = resA21.get()
        Aphi_m[2][2] = resA22.get()
        Aphi_m[2][3] = resA23.get()

        Aphi_m[3][0] = resA30.get()
        Aphi_m[3][1] = resA31.get()
        Aphi_m[3][2] = resA32.get()
        Aphi_m[3][3] = resA33.get()

        end2 = time.time()
        print("Время на матрицу Aphi_2")
        print(end2 - start2)
        print()

        return Aphi_m
    def get_Aphi_3(N):
        start2 = time.time()
        Aphi_m = np.zeros(shape=(N, N))
        pool1 = multiprocessing.Pool(processes=8)

        resA00 = pool1.apply_async(integrate_Aphi00_3)
        resA01 = pool1.apply_async(integrate_Aphi01_3)
        resA02 = pool1.apply_async(integrate_Aphi02_3)
        resA03 = pool1.apply_async(integrate_Aphi03_3)

        resA10 = pool1.apply_async(integrate_Aphi10_3)
        resA11 = pool1.apply_async(integrate_Aphi11_3)
        resA12 = pool1.apply_async(integrate_Aphi12_3)
        resA13 = pool1.apply_async(integrate_Aphi13_3)

        Aphi_m[0][0] = resA00.get()
        Aphi_m[0][1] = resA01.get()
        Aphi_m[0][2] = resA02.get()
        Aphi_m[0][3] = resA03.get()

        Aphi_m[1][0] = resA10.get()
        Aphi_m[1][1] = resA11.get()
        Aphi_m[1][2] = resA12.get()
        Aphi_m[1][3] = resA13.get()

        pool2 = multiprocessing.Pool(processes=8)

        resA20 = pool2.apply_async(integrate_Aphi20_3)
        resA21 = pool2.apply_async(integrate_Aphi21_3)
        resA22 = pool2.apply_async(integrate_Aphi22_3)
        resA23 = pool2.apply_async(integrate_Aphi23_3)

        resA30 = pool2.apply_async(integrate_Aphi30_3)
        resA31 = pool2.apply_async(integrate_Aphi31_3)
        resA32 = pool2.apply_async(integrate_Aphi32_3)
        resA33 = pool2.apply_async(integrate_Aphi33_3)

        Aphi_m[2][0] = resA20.get()
        Aphi_m[2][1] = resA21.get()
        Aphi_m[2][2] = resA22.get()
        Aphi_m[2][3] = resA23.get()

        Aphi_m[3][0] = resA30.get()
        Aphi_m[3][1] = resA31.get()
        Aphi_m[3][2] = resA32.get()
        Aphi_m[3][3] = resA33.get()

        end2 = time.time()
        print("Время на матрицу Aphi_3")
        print(end2 - start2)
        print()

        return Aphi_m

    #Для поочередного запускания. Посмотреть время вычисления каждого из интеграллов
    def get_Aphi_4(N):
        start2 = time.time()

        start = time.time()
        resA00_1 = integrate_Aphi00_1()
        end = time.time()
        print(resA00_1)
        print("00_1: " + (end - start).__str__())

        start = time.time()
        resA00_2 = integrate_Aphi20_2()
        print(resA00_2)
        end = time.time()
        print("00_2: " + (end-start).__str__())



        start = time.time()
        resA01_1 = integrate_Aphi01_1()
        end = time.time()
        print(resA01_1)
        print("01_1: " + (end - start).__str__())

        start = time.time()
        resA01_2 = integrate_Aphi01_2()
        end = time.time()
        print(resA01_2)
        print("01_2: " + (end - start).__str__())



        start = time.time()
        resA02_1 = integrate_Aphi02_1()
        end = time.time()
        print(resA02_1)
        print("02_1: " + (end - start).__str__())

        start = time.time()
        resA02_2 = integrate_Aphi02_2()
        end = time.time()
        print(resA02_2)
        print("02_2: " + (end - start).__str__())



        start = time.time()
        resA10_1 = integrate_Aphi10_1()
        end = time.time()
        print(resA10_1)
        print("10_1: " + (end - start).__str__())

        start = time.time()
        resA10_2 = integrate_Aphi10_2()
        end = time.time()
        print(resA10_2)
        print("10_2: " + (end - start).__str__())



        start = time.time()
        resA11_1 = integrate_Aphi11_1()
        end = time.time()
        print(resA11_1)
        print("11_1: " + (end - start).__str__())

        start = time.time()
        resA11_2 = integrate_Aphi11_2()
        end = time.time()
        print(resA11_2)
        print("11_2: " + (end - start).__str__())



        start = time.time()
        resA12_1 = integrate_Aphi12_1()
        end = time.time()
        print(resA12_1)
        print("12_1: " + (end - start).__str__())

        start = time.time()
        resA12_2 = integrate_Aphi12_2()
        end = time.time()
        print(resA12_2)
        print("12_2: " + (end - start).__str__())



        start = time.time()
        resA20_1 = integrate_Aphi20_1()
        end = time.time()
        print(resA20_1)
        print("20_1: " + (end - start).__str__())

        start = time.time()
        resA20_2 = integrate_Aphi20_2()
        end = time.time()
        print(resA20_2)
        print("20_2: " + (end - start).__str__())



        start = time.time()
        resA21_1 = integrate_Aphi21_1()
        end = time.time()
        print(resA21_1)
        print("21_2: " + (end - start).__str__())



        start = time.time()
        resA21_2 = integrate_Aphi21_2()
        end = time.time()
        print(resA21_2)
        print("21_2: " + (end - start).__str__())



        start = time.time()
        resA22_1 = integrate_Aphi22_1()
        print(resA22_1)
        end = time.time()
        print("22_1: " + (end - start).__str__())

        start = time.time()
        resA22_2 = integrate_Aphi22_2()
        print(resA22_2)
        end = time.time()
        print("22_2: " + (end - start).__str__())



    phi_m = get_phi(N)
    print("Матрица phi")
    print(phi_m)
    print("\n")

    Aphi_m1 = get_Aphi_1(N)
    print("Матрица Aphi_1")
    print(Aphi_m1)
    print("\n")


    Aphi_m2 = get_Aphi_2(N)
    print("Матрица Aphi_2")
    print(Aphi_m2)
    print("\n")

    Aphi_m3 = get_Aphi_3(N)
    print("Матрица Aphi_3")
    print(Aphi_m3)
    print("\n")


    Aphi_m = Aphi_m1 + Aphi_m2 + Aphi_m3

    eigenVal, eigenVectors = LA.eig(Aphi_m, phi_m)
    print("Собственные значения")
    print(eigenVal)
    print("\n")
    print("Собственные векторы")
    print(eigenVectors)

