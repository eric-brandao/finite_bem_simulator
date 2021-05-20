import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit, prange
from scipy import integrate
import time
from tqdm import tqdm
import pickle
from controlsair import sph2cart

class ThomassonZr(object):
    """ Compute sound field above finite sample using Thomasson radiation impedance
    """
    def __init__(self, air = [], controls = [], material = [],
                 receivers = [], theta = 0, phi = 0, Lx = 1.0, Ly = 1.0):
        """

        Parameters
        ----------
        air : object (AirProperties)
            The relevant properties of the air: c0 (sound speed) and rho0 (air density)
        controls : object (AlgControls)
            Controls of the simulation (frequency spam)
        material : object (PorousAbsorber)
            Contains the material properties (surface impedance)
        sources : object (Source)
            The sound sources in the field
        receivers : object (Receiver)
            The receivers in the field
        theta : float
            elevation angle in rad
        phi : float
            azimuth angle in rad
        Lx : float
            sample length
        Ly : float
            sample width

        The objects are stored as attributes in the class (easier to retrieve).
        """

        self.air = air
        self.controls = controls
        self.material = material
        self.receivers = receivers
        self.theta = theta
        self.phi = phi
        self.Lx = Lx
        self.Ly = Ly
        #self.rectangle = []
        try:
            self.beta = (self.air.rho0 * self.air.c0) / self.material.Zs  # normalized surface admitance
        except:
            self.beta = []
        self.pres_s = []
        self.ux_s = []
        self.uy_s = []
        self.uz_s = []
        
    def radimpedance(self,):
        """ Computes the radiation impedance of the finite sample
        """
        S = self.Lx * self.Ly
        self.Zr = np.zeros(len(self.controls.k0), dtype = complex)
        bar = tqdm(total = len(self.controls.k0),
            desc = 'Calculating radiation impedance')
        for jf, k0 in enumerate(self.controls.k0):
            kx_l, ky_l, kz_l = sph2cart(k0, np.pi/2-self.theta, self.phi)
            # integrand
            fkt_r = lambda ka,ta: np.real(4*np.cos(kx_l*ka)*np.cos(ky_l*ta)*\
                (np.exp(-1j*k0*np.sqrt(ka**2+ta**2))/np.sqrt(ka**2+ta**2))*\
                    (self.Lx-ka)*(self.Ly-ta))
# =============================================================================
#             Zrr = integrate.dblquad(fkt_r, 0, self.Lx,
#                                     lambda ka: 0, lambda ka: self.Ly)
# =============================================================================
            Zrr = integrate.dblquad(fkt_r, 0, self.Lx, 0, self.Ly)
            
            fkt_i = lambda ka,ta: np.imag(4*np.cos(kx_l*ka)*np.cos(ky_l*ta)*\
                (np.exp(-1j*k0*np.sqrt(ka**2+ta**2))/np.sqrt(ka**2+ta**2))*\
                    (self.Lx-ka)*(self.Ly-ta))
# =============================================================================
#             Zri = integrate.dblquad(fkt_i, 0, self.Lx,
#                                     lambda ka: 0, lambda ka: self.Ly)
# =============================================================================
            Zri = integrate.dblquad(fkt_i, 0, self.Lx, 0, self.Ly)
            self.Zr[jf] = (-1j*k0/(2*np.pi*S))*(Zrr[0] + 1j*Zri[0])
            bar.update(1)
        bar.close()
    
    def p_fps(self,):
        """ Calculates the total sound pressure spectrum at the receivers coordinates.

        The sound pressure spectrum is calculatef for all receivers (attribute of class).
        The quantity calculated is the total sound pressure = incident + scattered.
        """
        # Loop the receivers
        self.pres_s = []
        pres_rec = np.zeros((self.receivers.coord.shape[0], 
                             len(self.controls.freq)), dtype = complex)
        bar = tqdm(total = self.receivers.coord.shape[0],
            desc = 'Calculating sound field at every receiver')
        for jrec, r_coord in enumerate(self.receivers.coord):
            for jf, k0 in enumerate(self.controls.k0):
                kx_l, ky_l, kz_l = sph2cart(k0, np.pi/2-self.theta, self.phi)
                gfun_r = lambda x0,y0: np.real((np.exp(-1j*k0*np.sqrt(
                    (r_coord[0]-x0)**2+(r_coord[1]-y0)**2+r_coord[2]**2))/\
                    np.sqrt((r_coord[0]-x0)**2+(r_coord[1]-y0)**2+r_coord[2]**2))*\
                    np.exp(-1j*(kx_l*x0+ky_l*y0)))
                Ir = integrate.dblquad(gfun_r, -self.Lx/2, self.Lx/2, -self.Ly/2, self.Ly/2)
                
                gfun_i = lambda x0,y0: np.imag((np.exp(-1j*k0*np.sqrt(
                    (r_coord[0]-x0)**2+(r_coord[1]-y0)**2+r_coord[2]**2))/\
                    np.sqrt((r_coord[0]-x0)**2+(r_coord[1]-y0)**2+r_coord[2]**2))*\
                    np.exp(-1j*(kx_l*x0+ky_l*y0)))
                Ii = integrate.dblquad(gfun_i, -self.Lx/2, self.Lx/2, -self.Ly/2, self.Ly/2)
                
                k_veci = np.array([kx_l, ky_l, -kz_l])
                k_vecr = np.array([kx_l, ky_l, kz_l])
                Zs = self.material.Zs[jf]/(self.air.c0*self.air.rho0)
                Zr = self.Zr[jf]
                pres_rec[jrec, jf] = np.exp(-1j * np.dot(k_veci, r_coord))+\
                    np.exp(-1j * np.dot(k_vecr, r_coord))-\
                    (2*1j*k0/(Zs+Zr))*(Ir[0] + 1j*Ii[0])
            bar.update(1)
        bar.close()
        self.pres_s.append(pres_rec)
        
    def p_ref(self, total_ref = True):
        """ Calculates the reflected sound pressure spectrum at the receivers coordinates.

        The sound pressure spectrum is calculatef for all receivers (attribute of class).
        The quantity calculated is the total sound pressure = incident + scattered.
        """
        p_ref = np.zeros((self.receivers.coord.shape[0], 
                             len(self.controls.freq)), dtype = complex)
        bar = tqdm(total = self.receivers.coord.shape[0],
            desc = 'Calculating reflected sound field at every receiver')
        for jrec, r_coord in enumerate(self.receivers.coord):
            for jf, k0 in enumerate(self.controls.k0):
                kx_l, ky_l, kz_l = sph2cart(k0, np.pi/2-self.theta, self.phi)
                k_veci = np.array([kx_l, ky_l, -kz_l])
                k_vecr = np.array([kx_l, ky_l, kz_l])
                if total_ref:
                    p_ref[jrec, jf] = self.pres_s[0][jrec, jf]-\
                        np.exp(-1j * np.dot(k_veci, r_coord))
                else:
                    p_ref[jrec, jf] = self.pres_s[0][jrec, jf]-\
                        np.exp(-1j * np.dot(k_veci, r_coord))-\
                            np.exp(-1j * np.dot(k_vecr, r_coord))
            bar.update(1)
        bar.close()
        return p_ref
    
    def save(self, filename = 'my_thomasson', pathname = ''):
        """ To save the simulation object as pickle

        Parameters
        ----------
        filename : str
            name of the file
        pathname : str
            path of folder to save the file
        """
        filename = filename# + '_Lx_' + str(self.Lx) + 'm_Ly_' + str(self.Ly) + 'm'
        self.path_filename = pathname + filename + '.pkl'
        f = open(self.path_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename = 'my_thomasson', pathname = ''):
        """ Load a simulation object.

        You can instantiate an empty object of the class and load a saved one.
        It will overwrite the empty object.

        Parameters
        ----------
        filename : str
            name of the file
        pathname : str
            path of folder to save the file
        """
        lpath_filename = pathname + filename + '.pkl'
        f = open(lpath_filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)


