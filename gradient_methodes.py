#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:42:44 2023

@author: Theo Biardeau
"""

import numpy as np
from numpy import exp
from numpy import sin
from numpy import cos
from numpy import zeros
from numpy import sqrt
from numpy import arctan
import cv2 as cv

class DerichEdgeDetector :
    """ 
    DerichEdgeDetector class
    This class allows computing the gradients along X and Y axes with Derich methode
    Only grayscale img is accepted
    """
    def __init__(self, img, alpha, omega) -> None:
        self.img = img
        self.alpha = alpha
        self.omega = omega

    def coefficients_deriche(self):
        """
        Compute all deriche coeficient from aplha and omega parameters

        Returns
        -------
        c : TypE
            C coeficient.
        k : TypE
            a coeficient.
        c1 : TypE
            c1 coeficient.
        c2 : TypE
            c2 coeficient.
        b1 : TypE
            b1 coeficient.
        b2 : TypE
            b2 coeficient.
        a : TypE
            a coeficient.
        a0 : TypE
            a0 coeficient.
        a1 : TypE
            a1 coeficient.
        a2 : TypE
            a2 coeficient.
        a3 : TypE
            a3 coeficient.

        """
        alpha = self.alpha
        omega = self.omega
        c = (1 - 2 * exp(-alpha) * cos(omega) + exp(-2 * alpha)) / (exp(-alpha) * sin(omega))
        k = (( 1 - 2 * exp(-alpha) * cos(omega) + exp(-2 * alpha)) \
             * (pow(alpha, 2) + pow(omega, 2))) \
            /(2 * alpha * exp(-alpha) * sin(omega) \
              + omega - omega * exp(-2 * alpha))

        c1 = (k * alpha) / (pow(alpha, 2) + pow(omega, 2))
        c2 = (k * omega) / (pow(alpha, 2) + pow(omega, 2))

        b1 = -2 * exp(-alpha) * cos(omega)
        b2 = exp(-2 * alpha)

        a = -c * exp(-alpha) * sin(omega)
        a0 = c2
        a1 = (-c2 * cos(omega) + c1 * sin(omega)) * exp(-alpha)
        a2 = a1 - c2 * b1
        a3 = -c2 * b2

        return c, k, c1, c2, b1, b2, a, a0, a1, a2, a3

    def filtre_deriche_recursif(self, axes):
        """
        

        Parameters
        ----------
        posistion : bool
            Allows you to indicate along which axes the gradients are calculated.

        Returns
        -------
        R : TypE
            Gradient along ones axe.

        """
        c, k, c1, c2, b1, b2, a, a0, a1, a2, a3 = self.coefficients_deriche()

        image = self.img

        if axes == 1 :
            image = image.T

        shape = np.shape(self.img)
        n = shape[0]
        p = shape[1]
        # Gradient
        # Y+
        yp=zeros([n, p])

        debut_j = 0
        #yp[:,debut_j]=zeros([n,1])
        debut_j = debut_j + 1
        yp[:, debut_j] = image[:, debut_j-1] - b1 * yp[:, debut_j-1]
        debut_j = debut_j + 1
        for j in range(debut_j, p, 1):
            yp[:, j] = image[:, j-1] - b1 * yp[:, j-1]-b2 * yp[:, j-2]

        # Y-
        ym = zeros([n, p])

        fin_j = p - 1
        #ym[:,fin_j]=zeros([n,1])
        fin_j = fin_j - 1
        ym[:, fin_j] = image[:,fin_j + 1] - b1 * ym[:,fin_j + 1]
        fin_j = fin_j - 1
        for j in range(fin_j, -1, -1):
            ym[:,j] = image[:, j+1] - b1 * ym[:, j+1] - b2 * ym[:, j+2]

        # s
        s = a * (yp-ym)

        # Lissage
        # R+
        rp = zeros([n,p])

        debut_i = 0
        rp[debut_i, :] = a0 * s[debut_i, :]
        debut_i = debut_i + 1
        rp[debut_i, :] = a0 * s[debut_i, :] + a1 * s[debut_i-1, :] - b1 * rp[debut_i-1, :]
        debut_i = debut_i + 1
        for i in range(debut_i,n,1):
            rp[i, :] = a0 * s[i, :] + a1 * s[i-1, :] - b1 * rp[i-1, :] - b2 * rp[i-2, :]
        #endfor

        # R-
        rm = zeros([n, p])

        fin_i = n - 1
        #rm[fin_i,:]=zeros([1,p])
        fin_i = fin_i - 1
        rm[debut_i, :] = a2 * s[fin_i+1, :] - b1 * rm[fin_i+1, :]
        fin_i=fin_i-1
        for i in range(fin_i,-1,-1):
            rm[i, :] =a2 * s[i+1, :] + a3 * s[i+2, :] - b1 * rm[i+1, :] - b2 * rm[i+2, :]
        #endfor

        # R
        r = rm + rp

        if axes == 1 :
            return r.T

        return r

    def derich_methode (self):
        """
        Compute the gradient

        Returns
        -------
        norme_gradient : Array
            Matrix containing the norm of the gradients at each point.
        angle_gradient : Array
            matrix containing the angle of the gradients at each point.

        """
        gx = self.filtre_deriche_recursif(0)
        gy = self.filtre_deriche_recursif(1)
        norme_gradient = sqrt(gx**2 + gy**2)
        angle_gradient = arctan(gy/gx)

        return norme_gradient, angle_gradient


class CarronEdgeDetector:
    """
    CarronEdgeDetector is a edged detector methode based on the article thesis of Thierry Carron and Patrick Lambert 
    This edge detector use HSV property
    """
    def __init__(self, images):
        """

        Parameters
        ----------
        images : Array
            Images in BGR format.

        Returns
        -------
        None.

        """

        self.image_hsv = cv.cvtColor(images, cv.COLOR_BGR2HSV)

    def sigmoid(self,z):
        """

        Parameters
        ----------
        z : Long, int or double
            Sigmoid implemnentation.

        Returns
        -------
        double.

        """
        return 1/(1 + np.exp(-z)

    def carron_methode (self):
        



class ChatouxEdgeDetector:
    """
    ChatouxEdgeDetector is a edged detector methode based on the P.h.D thesis of Hermine Chatoux
    This edge detector use vectoriel approach
    """
    #def __init__(self, images, gram_matrix):
        