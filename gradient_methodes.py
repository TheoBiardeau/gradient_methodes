#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:42:44 2023

@author: theo
"""

import numpy as np
from numpy import exp
from numpy import sin
from numpy import cos
from numpy import zeros
from numpy import sqrt
from numpy import arctan

class DerichEdgeDetector :
    def __init__(self, img, alpha, omega) -> None:
        self.img = img
        self.alpha = alpha
        self.omega = omega
        
    def coefficients_deriche(self):
        alpha = self.alpha
        omega = self.omega
        c=(1-2*exp(-alpha)*cos(omega)+exp(-2*alpha))/(exp(-alpha)*sin(omega))
        k=((1-2*exp(-alpha)*cos(omega)+exp(-2*alpha))*(pow(alpha,2)+pow(omega,2)))/(2*alpha*exp(-alpha)*sin(omega)+omega-omega*exp(-2*alpha))

        c1=(k*alpha)/(pow(alpha,2)+pow(omega,2))
        c2=(k*omega)/(pow(alpha,2)+pow(omega,2))

        b1=-2*exp(-alpha)*cos(omega)
        b2=exp(-2*alpha)

        a=-c*exp(-alpha)*sin(omega)
        a0=c2
        a1=(-c2*cos(omega)+c1*sin(omega))*exp(-alpha)
        a2=a1-c2*b1
        a3=-c2*b2
        
        return c,k,c1,c2,b1,b2,a,a0,a1,a2,a3

    def filtre_deriche_recursif(self, posistion):
        c,k,c1,c2,b1,b2,a,a0,a1,a2,a3 = self.coefficients_deriche()
        image = self.img
        if posistion == 1 : 
            image = image.T
        shape = np.shape(self.img)
        n = shape[0]
        p = shape[1]
        # Gradient
        # Y+
        Yp=zeros([n,p])

        debut_j=0
        #Yp[:,debut_j]=zeros([n,1])
        debut_j=debut_j+1
        Yp[:,debut_j]=image[:,debut_j-1]-b1*Yp[:,debut_j-1]
        debut_j=debut_j+1
        for j in range(debut_j,p,1):
            Yp[:,j]=image[:,j-1]-b1*Yp[:,j-1]-b2*Yp[:,j-2]
        #endfor

        # Y-
        Ym=zeros([n,p])

        fin_j=p-1
        #Ym[:,fin_j]=zeros([n,1])
        fin_j=fin_j-1
        Ym[:,fin_j]=image[:,fin_j+1]-b1*Ym[:,fin_j+1]
        fin_j=fin_j-1
        for j in range(fin_j,-1,-1):
            Ym[:,j]=image[:,j+1]-b1*Ym[:,j+1]-b2*Ym[:,j+2]
        #endfor

        # S
        S=a*(Yp-Ym)
        
        # Lissage
        # R+
        Rp=zeros([n,p])

        debut_i=0
        Rp[debut_i,:]=a0*S[debut_i,:]
        debut_i=debut_i+1
        Rp[debut_i,:]=a0*S[debut_i,:]+a1*S[debut_i-1,:]-b1*Rp[debut_i-1,:]
        debut_i=debut_i+1
        for i in range(debut_i,n,1):
            Rp[i,:]=a0*S[i,:]+a1*S[i-1,:]-b1*Rp[i-1,:]-b2*Rp[i-2,:]
        #endfor

        # R-
        Rm=zeros([n,p])

        fin_i=n-1
        #Rm[fin_i,:]=zeros([1,p])
        fin_i=fin_i-1
        Rm[debut_i,:]=a2*S[fin_i+1,:]-b1*Rm[fin_i+1,:]
        fin_i=fin_i-1
        for i in range(fin_i,-1,-1):
            Rm[i,:]=a2*S[i+1,:]+a3*S[i+2,:]-b1*Rm[i+1,:]-b2*Rm[i+2,:]
        #endfor

        # R
        R=Rm+Rp
        
        return R

    def derich_methode (self):
        Gx = self.filtre_deriche_recursif(0)
        GyTemp = self.filtre_deriche_recursif(1)
        Gy = GyTemp.T
        norme_gradient = sqrt(Gx**2 + Gy**2)
        angle_gradient = arctan(Gy/Gx)
        return norme_gradient, angle_gradient