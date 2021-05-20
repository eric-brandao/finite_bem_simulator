# -*- coding: utf-8 -*-
"""
Created on Sun May  2 17:30:44 2021

@author: ericb
"""
class Rectangle():
    """ A class containing properties of a rectangle
    """
    def __init__(self, Lx, Ly):
        """

        Parameters
        ----------
        Lx : float
            Length of rectangular sample
        Lx : float
            Width of rectangular sample
        """
        self.Lx = Lx
        self.Ly = Ly

class Circle():
    """ A class containing properties of a circle
    """
    def __init__(self, radius = 0.5):
        """

        Parameters
        ----------
        radius : float
            Radius of rectangular sample
        """
        self.radius = radius