import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit, prange
import matplotlib.tri as tri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.io as pio
# from insitu.controlsair import load_cfg
import time
#from tqdm import tqdm
#from tqdm.notebook import trange, tqdm
import pickle

# import impedance-py/C++ module and other stuff
# =============================================================================
# try:
#     import insitu_cpp
# except:
#     print("I could not find insitu_cpp. You should be able to load BEM files and add noise.")
# =============================================================================
try:
    from insitu.controlsair import plot_spk
except:
    from controlsair import plot_spk

class BEMFlushSq(object):
    """ Calculates the sound field above a finite locally reactive squared sample.

    It is used to calculate the sound pressure and particle velocity using
    the BEM formulation for an absorbent sample flush mounted  on a hard baffle
    (exact for spherical waves on locally reactive and finite samples)

    Attributes
    ----------
    beta : numpy array
        material normalized surface admitance
    Nzeta : numpy ndarray
        functions for quadrature integration (loaded from picke)
    Nweights : numpy ndarray
        wights for quadrature integration (loaded from picke)
    el_center : (Nelx2) numpy array
        coordinates of element's center (Nel is the number of elements)
    jacobian : float
        the Jacobian of the mesh
    node_x : (Nelx4) numpy array
        x-coordinates of element's vertices
    node_y : (Nelx4) numpy array
        y-coordinates of element's vertices
    p_surface : (NelxNfreq) numpy array
        The sound pressure at the center of each element on the mesh (Nfreq = len(freq)).
        Solved by the BEM C++ module.
    gij_f : list of (NelxNfreq) numpy arrays
        The BEM matrix for each frequency step.
    pres_s - list of receiver pressure spectrums for each source.
        Each element of the list has a (N_rec x N_freq) matrix for a given source.
        Each line of the matrix is a spectrum of a sound pressure for a receiver.
        Each column is a set of sound pressure at all receivers for a frequency.
    ux_s - list of receiver velocity spectrums (x-dir) for each source.
        Each element of the list has a (N_rec x N_freq) matrix for a given source.
    uy_s - list of receiver velocity spectrums (y-dir) for each source.
        Each element of the list has a (N_rec x N_freq) matrix for a given source.
    uz_s - list of receiver velocity spectrums (z-dir) for each source.
        Each element of the list has a (N_rec x N_freq) matrix for a given source.

    Methods
    ----------
    generate_mesh(Lx = 1.0, Ly = 1.0, Nel_per_wavelenth = 6)
        Generate the mesh for simulation

    def psurf()
        Calculates the surface pressure of the BEM mesh

    assemble_gij()
        Assemble the BEM matrix.

    psurf2()
        Calculate p_surface using assembled gij_f matrixes.

    p_fps()
        Calculates the total sound pressure spectrum at the receivers coordinates.

    uz_fps() - dis
        Calculates the total particle velocity spectrum at the receivers coordinates.

    add_noise(snr = 30, uncorr = False)
        Add gaussian noise to the simulated data.

    plot_scene(vsam_size = 2, mesh = True)
        Plot of the scene using matplotlib - not redered

    plot_pres()
        Plot the spectrum of the sound pressure for all receivers

    plot_uz() - dis
        Plot the spectrum of the particle velocity in zdir for all receivers

    plot_colormap(freq = 1000):
        Plots a color map of the pressure field.

    plot_intensity(self, freq = 1000)
        Plots a color map of the pressure field.

    save(filename = 'my_bemflush', path = '/home/eric/dev/insitu/data/bem_simulations/')
        To save the simulation object as pickle

    load(filename = 'my_qterm', path = '/home/eric/dev/insitu/data/bem_simulations/')
        Load a simulation object.
    """

    def __init__(self, air = [], controls = [], material = [], sources = [], receivers = [],
                 n_gauss = 36, bar_mode = 'terminal'):
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
        n_gauss : int
            number of gauss points for integration
        bar_mode : str
            Type of bar to run (useful for notebook)

        The objects are stored as attributes in the class (easier to retrieve).
        """

        self.air = air
        self.controls = controls
        self.material = material
        self.sources = sources
        self.receivers = receivers
        self.n_gauss = n_gauss
        try:
            self.beta = (self.air.rho0 * self.air.c0) / self.material.Zs  # normalized surface admitance
        except:
            self.beta = []
        self.pres_s = []
        self.ux_s = []
        self.uy_s = []
        self.uz_s = []
        #self.Nzeta, self.Nweights = ksi_weights_mtx(n_gauss = n_gauss) #zeta_weights()
        # print("pause")
        if bar_mode == 'notebook':
            from tqdm.notebook import trange, tqdm
        else:
            from tqdm import tqdm
        self.tqdm = tqdm

    def generate_mesh(self, Lx = 1.0, Ly = 1.0, el_size = 0.05, Nel_per_wavelenth = []):
        """Generate the mesh for simulation.

        The mesh will consists of rectangular elements. Their size is a function
        of the sample's size (Lx and Ly) and the maximum frequency intended and
        the number of elements per wavelength (recomended: 6)

        Parameters
        ----------
        Lx : float
            Sample's lenght
        Ly : float
            Sample's width
        Nel_per_wavelenth : int
            Number of elements per wavelength. The default value is 6.
        """
        # Get the maximum frequency to estimate element size required
        self.Lx = Lx
        self.Ly = Ly
        freq_max = self.controls.freq[-1]
        if not Nel_per_wavelenth:
            el_size = el_size #self.air.c0 / (Nel_per_wavelenth * freq_max)
        else:
            el_size = self.air.c0 / (Nel_per_wavelenth * freq_max)
        #print('The el_size is: {}'.format(el_size))
        # Number of elementes spaning x and y directions
        Nel_x = np.int(np.ceil(Lx / el_size))
        Nel_y = np.int(np.ceil(Ly / el_size))
        # x and y coordinates of element's center
        xjc = np.linspace(-Lx/2 + el_size/2, Lx/2 - el_size/2, Nel_x)
        yjc = np.linspace(-Ly/2 + el_size/2, Ly/2 - el_size/2, Nel_y)
        # A Nel_x * Nel_y by 2 matrix containing the x and y coords of element centers
        self.el_center = np.zeros((Nel_x*Nel_y, 2))
        self.jacobian = (el_size**2)/4.0
        # x and y coordinates of elements edges
        xje = np.linspace(-Lx/2, Lx/2, Nel_x+1)
        yje = np.linspace(-Ly/2, Ly/2, Nel_y+1)
        # A Nel_x * Nel_y by 4 matrix containing the x and y coords of element x and y corners
        self.node_x = np.zeros((Nel_x*Nel_y, 4))
        self.node_y = np.zeros((Nel_x*Nel_y, 4))
        # form a matrix of coordinates of centers and corners
        d = 0
        for m in np.arange(len(yjc)):
            for n in np.arange(len(xjc)):
                self.el_center[d,:]=[xjc[n], yjc[m]] # each line of matrix nodes is a node xj and yj coordinate
                self.node_x[d,:]=[xje[n], xje[n]+el_size, xje[n]+el_size, xje[n]]
                self.node_y[d,:]=[yje[m], yje[m], yje[m]+el_size, yje[m]+el_size]
                d += 1
    
    def parse_mesh(self, bemf_field_obj):
        """Parse mesh from field object.
        
        Parameters
        ----------
        bemf_field_obj : object
            BEM flush field object.
        """
        # sample size
        self.Lx = bemf_field_obj.Lx
        self.Ly = bemf_field_obj.Ly
        # element center and jacobian
        self.el_center = bemf_field_obj.el_center
        self.jacobian = bemf_field_obj.jacobian
        # A Nel_x * Nel_y by 4 matrix containing the x and y coords of element x and y corners
        self.node_x = bemf_field_obj.node_x
        self.node_y = bemf_field_obj.node_y

    def psurf(self, bar_leave = True):
        """Calculate the surface pressure of the BEM mesh.

        Uses the Python implemented module.
        The surface pressure calculation represents the first step in a BEM simulation.
        It will assemble the BEM matrix and solve for the surface pressure
        based on the incident sound pressure. Each column is
        a complex surface pressure for each element in the mesh. Each row represents
        the evolution of frequency for a single element in the mesh.
        Therefore, there are N vs len(self.controls.freq) entries in this matrix.
        This method saves memory, compared to the use of assemble_gij and psurf2.
        On the other hand, if you want to simulate with a different source(s) configuration
        you will have to re-compute the BEM simulation.
        """
        # Get shape functions and weights
        Nksi, Nweights = ksi_weights_mtx(n_gauss = self.n_gauss)     
        # Allocate memory for the surface pressure data (# each column a frequency, each line an element)
        Nel = len(self.el_center)
        self.p_surface = np.zeros((Nel, len(self.controls.k0)), dtype=complex)
        # Generate the C matrix
        c_mtx = 0.5 * np.identity(Nel)
        # Calculate the distance from source to each element center
        el_3Dcoord = np.zeros((Nel, 3))
        el_3Dcoord[:,0:2] = self.el_center
        rsel = np.repeat(np.reshape(self.sources.coord[0,:],(1,3)),Nel,axis=0)-\
            el_3Dcoord
        r_unpt = np.linalg.norm(rsel, axis = 1)
        tinit = time.time()
        bar = self.tqdm(total = len(self.controls.k0), leave = bar_leave,
            desc = 'Surf. pres. for each frequency (method 1). {} x {} m'.format(self.Lx, self.Ly))
        for jf, k0 in enumerate(self.controls.k0):
            #Version 1 (distances in loop) - most time spent here
            gij = bemflush_mtx(self.el_center, self.node_x, self.node_y,
                Nksi, Nweights, k0, self.beta[jf])
            # Calculate the unperturbed pressure
            p_unpt = 2.0 * np.exp(-1j * k0 * r_unpt) / r_unpt
            # Solve system of equations
            # print("Solving system of eqs for freq: {} Hz.".format(self.controls.freq[jf]))         
            self.p_surface[:, jf] = np.linalg.solve(c_mtx + gij, p_unpt)
            bar.update(1)
        bar.close()
        tend = time.time()
        #print("elapsed time: {}".format(tend-tinit))

    def assemble_gij(self, bar_leave = True):
        """Assemble the BEM matrix.

        Uses implemented python module.
        Assembles a Nel x Nel matrix of complex numbers for each frequency step.
        It is memory consuming. On the other hand, it is independent of the 
        material properties and the sound sources. If you store this matrix,
        you can change the material and the positions of sound sources. Then,
        the information in memory is used to calculate the p_surface attribute.
        This can save time in simulations where you vary such parametes. The calculation of
        surface pressure (based on the incident sound pressure) should be done later.
        """
        # Get shape functions and weights
        Nksi, Nweights = ksi_weights_mtx(n_gauss = self.n_gauss)
        # Allocate memory for the surface pressure data (# each column a frequency, each line an element)
        el_3Dcoord = np.zeros((len(self.el_center), 3), dtype=np.float32)
        el_3Dcoord[:,0:2] = self.el_center
        # Set a time count for performance check
        tinit = time.time()
        bar = self.tqdm(total = len(self.controls.k0), leave = bar_leave,
            desc = 'Assembling BEM matrix for each freq. {} x {} m'.format(self.Lx, self.Ly))
        self.gij_f = []
        for jf, k0 in enumerate(self.controls.k0):
            gij = bemflush_mtx(self.el_center, self.node_x, self.node_y,
            Nksi, Nweights.T, k0, 1.0)
            self.gij_f.append(gij)
            bar.update(1)
        bar.close()
        tend = time.time()
        #print("elapsed time: {}".format(tend-tinit))

    def psurf2(self, erase_gij = False, bar_leave = True):
        """Calculate p_surface using assembled gij_f matrixes.

        Uses the implemented python module.
        The surface pressure calculation represents the first step in a BEM simulation.
        It will use the assembled BEM matrix (from assemble_gij) and solve for
        the surface pressure based on the incident sound pressure and material
        properties. Each column is a complex surface pressure for each element 
        in the mesh. Each row represents the evolution of frequency for a 
        single element in the mesh. Therefore, there are N vs len(self.controls.freq)
        entries in this matrix. This method saves processing time, compared to
        the use of psurf. You need to run it if you change the material properties 
        and/or the sound source configuration (no need to run assemble_gij again).
        
        Parameters
        ----------
        erase_gij : bool
            Wheter to erase gij matrix or not. Erasing saves memory.
        """
        # Allocate memory for the surface pressure data (# each column a frequency, each line an element)
        Nel = len(self.el_center)
        self.p_surface = np.zeros((Nel, len(self.controls.k0)), dtype=np.csingle)
        # Generate the C matrix
        c_mtx = 0.5 * np.identity(len(self.el_center), dtype = np.float32)
        # Calculate the distance from source to each element center
        el_3Dcoord = np.zeros((len(self.el_center), 3), dtype=np.float32)
        el_3Dcoord[:,0:2] = self.el_center
        rsel = np.repeat(np.reshape(self.sources.coord[0,:],(1,3)),len(self.el_center),axis=0)-\
            el_3Dcoord
        r_unpt = np.linalg.norm(rsel, axis = 1)
        tinit = time.time()
        bar = self.tqdm(total = len(self.controls.k0), leave = bar_leave,
            desc = 'Surf. pres. for each frequency (method 2). {} x {} m'.format(self.Lx, self.Ly))
        for jf, k0 in enumerate(self.controls.k0):
            gij = self.beta[jf]*self.gij_f[jf]
            # Calculate the unperturbed pressure
            p_unpt = 2.0 * np.exp(-1j * k0 * r_unpt) / r_unpt
            # Solve system of equations
            self.p_surface[:, jf] = np.linalg.solve(c_mtx + gij, p_unpt)
            bar.update(1)
        bar.close()
        tend = time.time()
        if erase_gij:
            self.gij_f = []
        #print("elapsed time: {}".format(tend-tinit))

    def p_fps(self, bar_leave = True):
        """ Calculates the total sound pressure spectrum at the receivers coordinates.

        The sound pressure spectrum is calculatef for all receivers (attribute of class).
        The quantity calculated is the total sound pressure = incident + scattered.
        """
        # Get shape functions and weights
        Nksi, Nweights = ksi_weights_mtx(n_gauss = self.n_gauss)  
        # Loop the receivers
        self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)),
                                dtype = complex)
            bar = self.tqdm(total = self.receivers.coord.shape[0], leave = bar_leave,
                    desc = 'Processing spectrum at each field point')
            for jrec, r_coord in enumerate(self.receivers.coord):
                xdist = (s_coord[0] - r_coord[0])**2.0
                ydist = (s_coord[1] - r_coord[1])**2.0
                r = (xdist + ydist)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                for jf, k0 in enumerate(self.controls.k0):
                    p_scat = bemflush_pscat2(r_coord, self.node_x, self.node_y,
                        Nksi, Nweights, k0, self.beta[jf], self.p_surface[:,jf])
                    pres_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1) +\
                        (np.exp(-1j * k0 * r2) / r2) + p_scat
                bar.update(1)
            bar.close()
            self.pres_s.append(pres_rec)

# =============================================================================
#     def uz_fps(self, compute_ux = False, compute_uy = False):
#         """ Calculates the total particle velocity spectrum at the receivers coordinates.
# 
#         The particle velocity spectrum is calculatef for all receivers (attribute of class).
#         The quantity calculated is the total particle velocity = incident + scattered.
#         The z-direction of particle velocity is always computed. x and y directions are optional.
# 
#         Parameters
#         ----------
#         compute_ux : bool
#             Whether to compute x component of particle velocity or not (Default is False)
#         compute_uy : bool
#             Whether to compute y component of particle velocity or not (Default is False)
#         """
#         # Loop the receivers
#         if compute_ux and compute_uy:
#             message = 'Processing particle velocity (x,y,z dir at field point)'
#         elif compute_ux:
#             message = 'Processing particle velocity (x,z dir at field point)'
#         elif compute_uy:
#             message = 'Processing particle velocity (y,z dir at field point)'
#         else:
#             message = 'Processing particle velocity (z dir at field point)'
# 
#         for js, s_coord in enumerate(self.sources.coord):
#             hs = s_coord[2] # source height
#             uz_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
#             if compute_ux:
#                 ux_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
#             if compute_uy:
#                 uy_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
#             for jrec, r_coord in enumerate(self.receivers.coord):
#                 r = ((s_coord[0] - r_coord[0])**2.0 + (s_coord[1] - r_coord[1])**2.0)**0.5 # horizontal distance source-receiver
#                 zr = r_coord[2]  # receiver height
#                 r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
#                 r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
#                 print('Calculate particle vel. (z-dir) for source {} and receiver {}'.format(js+1, jrec+1))
#                 # bar = ChargingBar('Processing particle velocity z-dir',
#                 #     max=len(self.controls.k0), suffix='%(percent)d%%')
#                 bar = tqdm(total = len(self.controls.k0),
#                     desc = message)
#                 for jf, k0 in enumerate(self.controls.k0):
#                     uz_scat = insitu_cpp._bemflush_uzscat(r_coord, self.node_x, self.node_y,
#                         self.Nzeta, self.Nweights.T, k0, self.beta[jf], self.p_surface[:,jf])
#                     # print(uz_scat)
#                     # print('p_scat for freq {} Hz is: {}'.format(self.controls.freq[jf], p_scat))
#                     uz_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1)*\
#                         (1 + (1 / (1j * k0 * r1)))* ((hs - zr)/r1)-\
#                         (np.exp(-1j * k0 * r2) / r2) *\
#                         (1 + (1 / (1j * k0 * r2))) * ((hs + zr)/r2) - uz_scat
#                     if compute_ux:
#                         ux_scat = insitu_cpp._bemflush_uxscat(r_coord, self.node_x, self.node_y,
#                             self.Nzeta, self.Nweights.T, k0, self.beta[jf], self.p_surface[:,jf])
#                         ux_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1)*\
#                             (1 + (1 / (1j * k0 * r1)))* (-r_coord[0]/r1)-\
#                             (np.exp(-1j * k0 * r2) / r2) *\
#                             (1 + (1 / (1j * k0 * r2))) * (-r_coord[0]/r2) - ux_scat
#                     if compute_uy:
#                         uy_scat = insitu_cpp._bemflush_uyscat(r_coord, self.node_x, self.node_y,
#                             self.Nzeta, self.Nweights.T, k0, self.beta[jf], self.p_surface[:,jf])
#                         uy_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1)*\
#                             (1 + (1 / (1j * k0 * r1)))* (-r_coord[1]/r1)-\
#                             (np.exp(-1j * k0 * r2) / r2) *\
#                             (1 + (1 / (1j * k0 * r2))) * (-r_coord[1]/r2) - uy_scat
#                     # Progress bar stuff
#                     bar.update(1)
#                 bar.close()
#             self.uz_s.append(uz_rec)
#             if compute_ux:
#                 self.ux_s.append(ux_rec)
#             if compute_uy:
#                 self.uy_s.append(uy_rec)
# =============================================================================

    def add_noise(self, snr = 30, uncorr = False):
        """ Add gaussian noise to the simulated data.

        The function is used to add noise to the pressure and particle velocity data.
        it reads the clean signal and estimate its power. Then, it estimates the power
        of the noise that would lead to the target SNR. Then it draws random numbers
        from a Normal distribution with standard deviation =  noise power

        Parameters
        ----------
        snr : float
            The signal to noise ratio you want to emulate
        uncorr : bool
            If added noise to each receiver is uncorrelated or not.
            If uncorr is True the the noise power is different for each receiver
            and frequency. If uncorr is False the noise power is calculated from
            the average signal magnitude of all receivers (for each frequency).
            The default value is False
        """
        signal = self.pres_s[0]
        try:
            signal_u = self.uz_s[0]
        except:
            signal_u = np.zeros(1)
        if uncorr:
            signalPower_lin = (np.abs(signal)/np.sqrt(2))**2
            signalPower_dB = 10 * np.log10(signalPower_lin)
            noisePower_dB = signalPower_dB - snr
            noisePower_lin = 10 ** (noisePower_dB/10)
        else:
            # signalPower_lin = (np.abs(np.mean(signal, axis=0))/np.sqrt(2))**2
            signalPower_lin = ((np.mean(np.abs(signal), axis=0))/np.sqrt(2))**2
            signalPower_dB = 10 * np.log10(signalPower_lin)
            noisePower_dB = signalPower_dB - snr
            noisePower_lin = 10 ** (noisePower_dB/10)
            if signal_u.any() != 0:
                signalPower_lin_u = (np.abs(np.mean(signal_u, axis=0))/np.sqrt(2))**2
                signalPower_dB_u = 10 * np.log10(signalPower_lin_u)
                noisePower_dB_u = signalPower_dB_u - snr
                noisePower_lin_u = 10 ** (noisePower_dB_u/10)
        np.random.seed(0)
        noise = np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape) +\
                1j*np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape)
        # noise = 2*np.sqrt(noisePower_lin)*\
        #     (np.random.randn(signal.shape[0], signal.shape[1]) + 1j*np.random.randn(signal.shape[0], signal.shape[1]))
        self.pres_s[0] = signal + noise
        if signal_u.any() != 0:
            # print('Adding noise to particle velocity')
            noise_u = np.random.normal(0, np.sqrt(noisePower_lin_u), size = signal_u.shape) +\
                1j*np.random.normal(0, np.sqrt(noisePower_lin_u), size = signal_u.shape)
            self.uz_s[0] = signal_u + noise_u

    def plot_scene(self, vsam_size = 2, mesh = False):
        """ Plot of the scene using matplotlib - not redered

        Parameters
        ----------
        vsam_size : float
            Scene size. Just to make the plot look nicer. You can choose any value.
            An advice is to choose a value bigger than the sample's largest dimension.
        mesh : bool
            Whether to plot the sample mesh or not. Default is False. In this way,
            the sample is represented by a grey rectangle.
        """
        fig = plt.figure()
        fig.canvas.set_window_title("Measurement scene")
        ax = fig.gca(projection='3d')
        # vertexes plot
        if mesh:
            for jel in np.arange(len(self.el_center)):
                nodex_el = self.node_x[jel]
                nodey_el = self.node_y[jel]
                nodex = np.reshape(nodex_el.flatten(), (4, 1))
                nodey = np.reshape(nodey_el.flatten(), (4, 1))
                nodez = np.reshape(np.zeros(4), (4, 1))
                vertices = np.concatenate((nodex, nodey, nodez), axis=1)
                verts = [list(zip(vertices[:,0],
                        vertices[:,1], vertices[:,2]))]
                # mesh points
                for v in verts[0]:
                    ax.scatter(v[0], v[1], v[2],
                    color='black',  marker = "o", s=1)
                # patch plot
                collection = Poly3DCollection(verts,
                    linewidths=1, alpha=0.9, edgecolor = 'gray', zorder=1)
                collection.set_facecolor('silver')
                ax.add_collection3d(collection)
        # baffle
        else:
            vertices = np.array([[-self.Lx/2, -self.Ly/2, 0.0],
                [self.Lx/2, -self.Ly/2, 0.0],
                [self.Lx/2, self.Ly/2, 0.0],
                [-self.Lx/2, self.Ly/2, 0.0]])
            verts = [list(zip(vertices[:,0],
                    vertices[:,1], vertices[:,2]))]
            # patch plot
            collection = Poly3DCollection(verts,
                linewidths=2, alpha=0.9, edgecolor = 'black', zorder=2)
            collection.set_facecolor('silver')
            ax.add_collection3d(collection)
        # plot source
        for s_coord in self.sources.coord:
            ax.scatter(s_coord[0], s_coord[1], s_coord[2],
                color='red',  marker = "o", s=200)
        # plot receiver
        for r_coord in self.receivers.coord:
            ax.scatter(r_coord[0], r_coord[1], r_coord[2],
                color='blue',  marker = "o", alpha = 0.35)
        ax.set_xlabel('X axis')
        # plt.xticks([], [])
        ax.set_ylabel('Y axis')
        # plt.yticks([], [])
        ax.set_zlabel('Z axis')
        # ax.grid(linestyle = ' ', which='both')
        ax.set_xlim((-vsam_size/2, vsam_size/2))
        ax.set_ylim((-vsam_size/2, vsam_size/2))
        ax.set_zlim((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        # ax.set_zlim((0, 0.3))
        # ax.set_zticks((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.set_zticks((0, 0.1, 0.2, 0.3))
        ax.set_xticks((-1, -0.5, 0.0, 0.5, 1.0))
        ax.set_yticks((-1, -0.5, 0.0, 0.5, 1.0))

        ax.view_init(elev=10, azim=45)
        # ax.invert_zaxis()
        plt.show() # show plot
        
        
    def plotly_scene(self, vsam_size = 2, renderer='notebook',
                     save_state = False, path ='', filename = ''):
        """ Plot of the scene using plotly

        Parameters
        ----------
        vsam_size : float
            Scene size. Just to make the plot look nicer. You can choose any value.
            An advice is to choose a value bigger than the sample's largest dimension.
        mesh : bool
            Whether to plot the sample mesh or not. Default is False. In this way,
            the sample is represented by a grey rectangle.
        """
        baffle_vertices = np.array([[-vsam_size/2, -vsam_size/2, 0.0],
                [vsam_size/2, -vsam_size/2, 0.0],
                [vsam_size/2, vsam_size/2, 0.0],
                [-vsam_size/2, vsam_size/2, 0.0]])
        
        vertices = np.array([[-self.Lx/2, -self.Ly/2, 0.0],
                [self.Lx/2, -self.Ly/2, 0.0],
                [self.Lx/2, self.Ly/2, 0.0],
                [-self.Lx/2, self.Ly/2, 0.0]])
        
        fig = go.Figure(data=[
            # Baffle
            go.Mesh3d(
                x=baffle_vertices[:,0], y=baffle_vertices[:,1], z=baffle_vertices[:,2],
                color='grey', opacity=0.70),
            # Sample
            go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
                color='darkgoldenrod', opacity=1),
            
            go.Scatter3d(x = self.sources.coord[:,0], y = self.sources.coord[:,1],
                         z = self.sources.coord[:,2], mode='markers', name="Source",
                         marker=dict(size=12, color='red',opacity=0.5)),
            
            go.Scatter3d(x = self.receivers.coord[:,0], y = self.receivers.coord[:,1],
                         z = self.receivers.coord[:,2], mode='markers', name="Receivers",
                         marker=dict(size=4, color='blue',opacity=0.4)),
                      
            ])
        camera = dict(up=dict(x=0, y=0, z=1),
                      center=dict(x=0, y=0, z=0),
                      eye=dict(x=1.5, y=1.5, z=0.75))
        fig.update_layout(scene_camera=camera)
        #plot(fig, auto_open=True)
        pio.renderers.default = renderer
        if save_state:
            fig.write_image(path+filename+'.pdf', scale=3)
        fig.show()
        
    def plot_pres(self, figsize = (7,5)):
        """ Plot the spectrum of the sound pressure for all receivers
        """
        plot_spk(self.controls.freq, self.pres_s, ref = 20e-6, figsize = figsize)

# =============================================================================
#     def plot_uz(self):
#         """ Plot the spectrum of the particle velocity in zdir for all receivers
#         """
#         plot_spk(self.controls.freq, self.uz_s, ref = 5e-8)
# =============================================================================

    def plot_colormap(self, freq = 1000, total_pres = True,  dinrange = 20):
        """Plots a color map of the pressure field.

        Parameters
        ----------
        freq : float
            desired frequency of the color map. If the frequency does not exist
            on the simulation, then it will choose the frequency just before the target.
        total_pres : bool
            Whether to plot the total sound pressure (Default = True) or the reflected only.
            In the later case, we subtract the incident field Green's function from the total
            sound field.
        dinrange : float
            Dinamic range of the color map

        Returns
        ---------
        plt : Figure object
        """
        id_f = np.where(self.controls.freq <= freq)
        id_f = id_f[0][-1]
        # color parameter
        if total_pres:
            color_par = 20*np.log10(np.abs(self.pres_s[0][:,id_f])/np.amax(np.abs(self.pres_s[0][:,id_f])))
        else:
            r1 = np.linalg.norm(self.sources.coord - self.receivers.coord, axis = 1)
            color_par = np.abs(self.pres_s[0][:,id_f]-\
                np.exp(-1j * self.controls.k0[id_f] * r1) / r1)
            color_par = 20*np.log10(color_par/np.amax(color_par))

        # Create triangulazition
        triang = tri.Triangulation(self.receivers.coord[:,0], self.receivers.coord[:,2])
        # Figure
        fig = plt.figure() #figsize=(8, 8)
        # fig = plt.figure()
        fig.canvas.set_window_title('pressure color map')
        plt.title('Reference |P(f)| (BEM sim)')
        # p = plt.tricontourf(triang, color_par, np.linspace(-15, 0, 15), cmap = 'seismic')
        p = plt.tricontourf(triang, color_par, np.linspace(-dinrange, 0, int(dinrange)), cmap = 'seismic')
        fig.colorbar(p)
        plt.xlabel(r'$x$ [m]')
        plt.ylabel(r'$z$ [m]')
        return plt

    def plot_intensity(self, freq = 1000):
        """Plots a vector map of the intensity field.

        Parameters
        ----------
        freq : float
            desired frequency of the color map. If the frequency does not exist
            on the simulation, then it will choose the frequency just before the target.

        Returns
        ---------
        plt : Figure object
        """
        id_f = np.where(self.controls.freq <= freq)
        id_f = id_f[0][-1]
        # Intensities
        Ix = 0.5*np.real(self.pres_s[0][:,id_f] *\
            np.conjugate(self.ux_s[0][:,id_f]))
        Iy = 0.5*np.real(self.pres_s[0][:,id_f] *\
            np.conjugate(self.uy_s[0][:,id_f]))
        Iz = 0.5*np.real(self.pres_s[0][:,id_f] *\
            np.conjugate(self.uz_s[0][:,id_f]))
        I = np.sqrt(Ix**2+Iy**2+Iz**2)
        # # Figure
        fig = plt.figure() #figsize=(8, 8)
        fig.canvas.set_window_title('Intensity distribution map')
        cmap = 'viridis'
        plt.title('Reference Intensity (BEM sim)')
        # if streamlines:
        #     q = plt.streamplot(self.receivers.coord[:,0], self.receivers.coord[:,2],
        #         Ix/I, Iz/I, color=I, linewidth=2, cmap=cmap)
        #     fig.colorbar(q.lines)
        # else:
        q = plt.quiver(self.receivers.coord[:,0], self.receivers.coord[:,2],
            Ix/I, Iz/I, I, cmap = cmap, width = 0.010)
        #fig.colorbar(q)
        plt.xlabel(r'$x$ [m]')
        plt.ylabel(r'$z$ [m]')
        return plt
        # Figure
        # fig = plt.figure() #figsize=(8, 8)
        # ax = fig.gca(projection='3d')
        # cmap = 'seismic'
        # # fig = plt.figure()
        # # fig.canvas.set_window_title('Intensity distribution map')
        # plt.title('|I|')
        # q = ax.quiver(self.receivers.coord[:,0], self.receivers.coord[:,1],
        #     self.receivers.coord[:,2], Ix, Iy, Iz,
        #     cmap = cmap, length=0.01, normalize=True)
        # c = I
        # c = getattr(plt.cm, cmap)(c)
        # # fig.colorbar(p)
        # fig.colorbar(q)
        # q.set_edgecolor(c)
        # q.set_facecolor(c)
        # plt.xlabel(r'$x$ [m]')
        # plt.ylabel(r'$z$ [m]')

    def save(self, filename = 'my_bemflush', path = ''):
        """ To save the simulation object as pickle

        Parameters
        ----------
        filename : str
            name of the file
        pathname : str
            path of folder to save the file
        """
        filename = filename# + '_Lx_' + str(self.Lx) + 'm_Ly_' + str(self.Ly) + 'm'
        self.path_filename = path + filename + '.pkl'
        f = open(self.path_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename = 'my_qterm', path = ''):
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
        lpath_filename = path + filename + '.pkl'
        f = open(lpath_filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

################## Functions #################################################
def ksi_weights_mtx(n_gauss = 36):
    """ Calculates Nksi and Nweights matrices
    
    This function calculates Nksi and Nweights matrices to be used in Gaussian
    quadrature matrix integration of squared elements. It will return the 
    shape functions as a matrix of 4 x n_gauss elements and a vector of weights
    of n_gauss elements. Only certain number of gauss points are allowed: 
    4, 16 and 36 - this approach avoids singularity in the case of collocation
    point located at the center of the element being integrated.
    
    
    Parameters
    ----------
    n_gauss : int
        the number of gauss points desired. Can be 4, 16 and 36. If another
        number is choosen, 36 points are automatically selected
        
    Returns
    ----------
    Nksi : numpy ndArray
        shape functions as a matrix of 4 x n_gauss elements
    Nweights : numpy 1dArray
        vector of weights of n_gauss elements
    """
    # Initialize
    Nksi = np.zeros((4, n_gauss))
    Nweights = np.zeros(n_gauss)
    
    # Write ksi1, ksi2 and weights
    if n_gauss == 4:
        a = np.sqrt(1/3)
        ksi1 = np.array([-a, a, a, -a])
        ksi2 = np.array([-a, -a, a, a])
        Nweights += 1 
    elif n_gauss == 16:
        a = np.sqrt((3+2*np.sqrt(6/5))/7)
        b = np.sqrt((3-2*np.sqrt(6/5))/7)
        ksi1 = np.array([-a, -a, a, a, -b, -b, b, b, -a, -a, a, a, -b, -b, b, b])
        ksi2 = np.array([-a, a, a, -a, -b, b, b, -b, -b, b, -b, b, -a, a, a, -a])
        Nweights[0:4] = 0.1210029932856020
        Nweights[4:8] = 0.4252933030106942 
        Nweights[8:] = 0.2268518518518519
    else:
        ksi_line = np.array([-0.93246951, -0.66120939, -0.23861918,
                             0.23861918, 0.66120939, 0.93246951])
        ksi1g, ksi2g = np.meshgrid(ksi_line, ksi_line)
        ksi1 = ksi1g.flatten()
        ksi2 = ksi2g.flatten()
        weights = np.array([[0.17132449, 0.36076157, 0.46791393,
                            0.46791393, 0.36076157, 0.17132449]])
        weights_mtx = np.dot(weights.T, weights)
        Nweights = weights_mtx.flatten()
         
    # write shape functions
    Nksi[0,:] = 0.25 * (1-ksi1)*(1-ksi2)
    Nksi[1,:] = 0.25 * (1+ksi1)*(1-ksi2)
    Nksi[2,:] = 0.25 * (1+ksi1)*(1+ksi2)
    Nksi[3,:] = 0.25 * (1-ksi1)*(1+ksi2)
    
    return Nksi, Nweights

#@njit()
def bemflush_mtx(el_center, node_x, node_y, Nksi, Nweights, k0, beta):
    """ Forms the BEM matrix
    
    For each collocation point (element center), computes the Rayleigh integral.
    We span all elements relative to all elements. We form a symmetric matrix
    of the BEM problem to be inverted.
    
    Parameters
    ----------
    el_center : numpy ndArray
        a (Nel x 2) matrix containing the xy coordinates of element centers
    node_x : numpy ndArray
        a (Nel x 4) matrix containing the coordinates of element x vertices
    node_y : numpy ndArray
        a (Nel x 4) matrix containing the coordinates of element y vertices
    Nksi : numpy ndArray
        shape functions as a matrix of 4 x n_gauss elements
    Nweights : numpy 1dArray
        vector of weights of n_gauss elements
    k0 : float
        sound wave-number at a frequency
    beta : float complex
        boundary condition at a frequency (surface admitance)
    Returns
    ----------
    bem_mtx : numpy ndArray
        a (Nel x Nel) symmetric matrix to be inverted
    """
    Nel = el_center.shape[0]
    jacobian = ((el_center[1,0] - el_center[0,0])**2.0)/4.0 #Fix 5.26 p113
    # initialize
    bem_mtx = np.zeros((Nel, Nel), dtype = np.complex64)
    for i in np.arange(Nel):
        xy_center = el_center[i,:]
        x_center = xy_center[0] * np.ones(Nksi.shape[1])
        y_center = xy_center[1] * np.ones(Nksi.shape[1])
        for j in np.arange(i, Nel):
            xnode = node_x[j,:]
            ynode = node_y[j,:]
            xzeta = np.dot(xnode, Nksi) # xnode @ Nksi # Fix and check global dommain
            yzeta = np.dot(ynode, Nksi) #ynode @ Nksi
            # calculate the distance from el center to transformed integration points
            r = ((x_center - xzeta)**2 + (y_center - yzeta)**2)**0.5
            g = 1j * k0 * beta * (np.exp(-1j * k0 * r)/(4 * np.pi * r)) * jacobian
            #print(g.astype(np.complex64).dtype)
            bem_mtx[i,j] = np.dot(Nweights, g) #g @ Nweights
    for i in np.arange(Nel-1):
        for j in np.arange(i+1, Nel):
            bem_mtx[j,i] = bem_mtx[i,j] 
    return bem_mtx

@njit
def bemflush_pscat2(r_coord, node_x, node_y, Nksi, Nweights, k0, beta, ps):
    """ Computes the scattered part of the sound field at a field point
    
    For a given field point, computes the scattered part of the sound field.
    We span all elements' contribution to the receiver. First, we compute the
    Rayleigh's integral for each element and then combine it with the surface
    pressure (amplitude of each monopole) to get the scattering contribution.
    
    Parameters
    ----------
    r_coord : numpy ndArray
        a (3,) vector containing the coordinates of the receiver
    node_x : numpy ndArray
        a (Nel x 4) matrix containing the coordinates of element x vertices
    node_y : numpy ndArray
        a (Nel x 4) matrix containing the coordinates of element y vertices
    Nksi : numpy ndArray
        shape functions as a matrix of 4 x n_gauss elements
    Nweights : numpy 1dArray
        vector of weights of n_gauss elements
    k0 : float
        sound wave-number at a frequency
    beta : float complex
        boundary condition at a frequency (surface admitance)
    ps : numpy ndArray
        (Nel, ) vector containing the surface pressure at each element for
        a frequency
    Returns
    ----------
    p_scat : float complex
        scattered pressure at receiver
    """
    #Nweights = numba.complex64(Nweights)
    #g = np.zeros(len(Nweights), dtype = np.complex64)
    # Vector of receiver coordinates (for vectorized integration)
    x_coord = r_coord[0] * np.ones(Nksi.shape[1])
    y_coord = r_coord[1] * np.ones(Nksi.shape[1])
    z_coord = r_coord[2] * np.ones(Nksi.shape[1])
    
    
    # Number of elements and jacobian
    Nel = node_x.shape[0]
    jacobian = ((node_x[1,0] - node_x[0,0])**2.0)/4.0;
    # Initialization
    gfield = np.zeros(Nel, dtype = np.complex64) #np.complex64
    #Loop through elements once
    for j in np.arange(Nel):
        # Transform the coordinate system for integration
        xnode = node_x[j,:]
        ynode = node_y[j,:]
        xzeta = np.dot(xnode, Nksi) #xnode @ Nksi
        yzeta = np.dot(ynode, Nksi) #ynode @ Nksi
        # Calculate the distance from el center to transformed integration points
        r = ((x_coord - xzeta)**2 + (y_coord - yzeta)**2 + z_coord**2)**0.5
        # Calculate green function
        #g = -1j * k0 * beta * (np.exp(-1j * k0 * r)/(4 * np.pi * r)) * jacobian
        g = -1j *k0 * beta *(np.exp(-1j * k0 * r)/(4 * np.pi * r)) * jacobian
        # Integrate
        #g = numba.complex64(g)
        gfield[j] = np.sum((Nweights*g)) #g @ Nweights;
    p_scat = np.sum((gfield*ps)) #np.dot(gfield, ps) #gfield @ ps
    return p_scat



#@njit
def bemflush_pscat(r_coord, node_x, node_y, Nksi, Nweights, k0, beta, ps):
    """ Computes the scattered part of the sound field at a field point
    
    For a given field point, computes the scattered part of the sound field.
    We span all elements' contribution to the receiver. First, we compute the
    Rayleigh's integral for each element and then combine it with the surface
    pressure (amplitude of each monopole) to get the scattering contribution.
    
    Parameters
    ----------
    r_coord : numpy ndArray
        a (3,) vector containing the coordinates of the receiver
    node_x : numpy ndArray
        a (Nel x 4) matrix containing the coordinates of element x vertices
    node_y : numpy ndArray
        a (Nel x 4) matrix containing the coordinates of element y vertices
    Nksi : numpy ndArray
        shape functions as a matrix of 4 x n_gauss elements
    Nweights : numpy 1dArray
        vector of weights of n_gauss elements
    k0 : float
        sound wave-number at a frequency
    beta : float complex
        boundary condition at a frequency (surface admitance)
    ps : numpy ndArray
        (Nel, ) vector containing the surface pressure at each element for
        a frequency
    Returns
    ----------
    p_scat : float complex
        scattered pressure at receiver
    """
    #g = np.zeros(len(Nweights), dtype = np.complex64)
    # Vector of receiver coordinates (for vectorized integration)
    x_coord = r_coord[0] * np.ones(Nksi.shape[1])
    y_coord = r_coord[1] * np.ones(Nksi.shape[1])
    z_coord = r_coord[2] * np.ones(Nksi.shape[1])
    
    
    # Number of elements and jacobian
    Nel = node_x.shape[0]
    jacobian = ((node_x[1,0] - node_x[0,0])**2.0)/4.0;
    # Initialization
    gfield = np.zeros(Nel, dtype = np.complex64) #np.complex64
    #Loop through elements once
    for j in np.arange(Nel):
        # Transform the coordinate system for integration
        xnode = node_x[j,:]
        ynode = node_y[j,:]
        xzeta = np.dot(xnode, Nksi) #xnode @ Nksi
        yzeta = np.dot(ynode, Nksi) #ynode @ Nksi
        # Calculate the distance from el center to transformed integration points
        r = ((x_coord - xzeta)**2 + (y_coord - yzeta)**2 + z_coord**2)**0.5
        # Calculate green function
        g = -1j * k0 * beta * (np.exp(-1j * k0 * r)/(4 * np.pi * r)) * jacobian
        #g = k0 * (np.exp(- k0 * r)/(4 * np.pi * r)) * jacobian
        # Integrate
        gfield[j] = np.dot(Nweights, g) #g @ Nweights;
    p_scat = np.dot(gfield, ps) #gfield @ ps
    return p_scat

def gaussint_sq(r_coord, nodes, Nzeta, Nweights, k0, beta = 1+0*1j):
    # Vector of receiver coordinates (for vectorized integration)
    x_coord = r_coord[0] * np.ones(Nzeta.shape[1])
    y_coord = r_coord[1] * np.ones(Nzeta.shape[1])
    z_coord = r_coord[2] * np.ones(Nzeta.shape[1])
    # Jacobian of squared element
    jacobian = ((nodes[:,0][1] - nodes[:,0][0])**2.0)/4.0
    # Gauss points on local element
    xzeta = nodes[:,0] @ Nzeta
    yzeta = nodes[:,1] @ Nzeta
    # Calculate the distance from el center to transformed integration points
    r = ((x_coord - xzeta)**2 + (y_coord - yzeta)**2 + z_coord**2)**0.5
    # Calculate green function
    g = -1j * k0 * beta * (np.exp(-1j * k0 * r)/(4 * np.pi * r)) * jacobian
    #print(g.shape)
    ival = np.dot(Nweights, g) #g @ Nweights
    return ival, xzeta, yzeta

def gaussint_dbquad_sq(r_coord, nodes, k0, beta = 1+1j):
    from scipy import integrate
    gfun_r = lambda x0, y0: np.real(-1j * k0 * beta * (np.exp(-1j*k0*np.sqrt(
        (r_coord[0]-x0)**2+(r_coord[1]-y0)**2+r_coord[2]**2))/\
        (4*np.pi*np.sqrt((r_coord[0]-x0)**2+(r_coord[1]-y0)**2+r_coord[2]**2))))
    Ir = integrate.dblquad(gfun_r, nodes[:,0][0], nodes[:,0][1], nodes[:,1][0], nodes[:,1][3])
    
    gfun_i = lambda x0, y0: np.imag(-1j * k0 * beta * (np.exp(-1j*k0*np.sqrt(
        (r_coord[0]-x0)**2+(r_coord[1]-y0)**2+r_coord[2]**2))/\
        (4*np.pi*np.sqrt((r_coord[0]-x0)**2+(r_coord[1]-y0)**2+r_coord[2]**2))))
    Ii = integrate.dblquad(gfun_i, nodes[:,0][0], nodes[:,0][1], nodes[:,1][0], nodes[:,1][3])
    return Ir[0] + 1j*Ii[0]

def zeta_weights():
    """ Calculates Nzeta and Nweights - old implementation 36 gauss pts
    """
    zeta = np.array([-0.93246951, -0.66120939, -0.23861918,
    0.23861918, 0.66120939, 0.93246951])

    weigths = np.array([0.17132449, 0.36076157, 0.46791393,
        0.46791393, 0.36076157, 0.17132449])

    # Create vectors of size 1 x 36 for the zetas
    N1 = 0.25 * np.matmul(np.reshape(1-zeta, (zeta.size,1)),  np.reshape(1-zeta, (1,zeta.size)))
    N2 = 0.25 * np.matmul(np.reshape(1+zeta, (zeta.size,1)),  np.reshape(1-zeta, (1,zeta.size)))
    N3 = 0.25 * np.matmul(np.reshape(1+zeta, (zeta.size,1)),  np.reshape(1+zeta, (1,zeta.size)))
    N4 = 0.25 * np.matmul(np.reshape(1-zeta, (zeta.size,1)),  np.reshape(1+zeta, (1,zeta.size)))

    N1 = np.reshape(N1, (1,zeta.size**2))
    N2 = np.reshape(N2, (1,zeta.size**2))
    N3 = np.reshape(N3, (1,zeta.size**2))
    N4 = np.reshape(N4, (1,zeta.size**2))

    # Let each line of the following matrix be a N vector
    Nzeta = np.zeros((4, zeta.size**2))
    Nzeta[0,:] = N1
    Nzeta[1,:] = N2
    Nzeta[2,:] = N3
    Nzeta[3,:] = N4

    # Create vector of size 1 x 36 for the weights
    Nweigths = np.matmul(np.reshape(weigths, (zeta.size,1)),  np.reshape(weigths, (1,zeta.size)))
    Nweigths = np.reshape(Nweigths, (1,zeta.size**2))
    # print('I have calculated!')
    return Nzeta, Nweigths