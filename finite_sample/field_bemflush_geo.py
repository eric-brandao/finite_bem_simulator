import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit, prange
import matplotlib.tri as tri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff
# from insitu.controlsair import load_cfg
import time
from tqdm import tqdm
import pickle

try:
    from insitu.controlsair import plot_spk
except:
    from controlsair import plot_spk
from sample_geometries import Rectangle, Circle

from general_functions import add_noise, add_noise2
import gmsh
import meshio





class BEMFlushGeo(object):
    """ Calculates the sound field above a finite locally reactive sample.

    It is used to calculate the sound pressure and particle velocity using
    the BEM formulation for an absorbent sample flush mounted  on a hard baffle
    (exact for spherical waves on locally reactive and finite samples).
    Any sample geometry can be generated and meshing is done with gmsh.

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
        Calculates p_surface using assembled gij_f matrixes.

    p_fps()
        Calculates the total sound pressure spectrum at the receivers coordinates.

    uz_fps()
        Calculates the total particle velocity spectrum at the receivers coordinates.

    add_noise(snr = 30, uncorr = False)
        Add gaussian noise to the simulated data.

    plot_scene(vsam_size = 2, mesh = True)
        Plot of the scene using matplotlib - not redered

    plot_pres()
        Plot the spectrum of the sound pressure for all receivers

    plot_uz()
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
    
    def __init__(self, air = [], controls = [], material = [],
                 sources = [], receivers = [],  
                 min_max_el_size = [0.05, 0.1], Nel_per_wavelenth = [],
                 n_gauss = 6, bar_mode = 'terminal'):
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
        min_max_el_size : list or numpy 2darray
            min and max element size
        Nel_per_wavelenth : int
            Number of elements per wavelength. The default value is 6.

        The objects are stored as attributes in the class (easier to retrieve).
        """

        self.air = air
        self.controls = controls
        self.material = material
        self.sources = sources
        self.receivers = receivers
        self.min_max_el_size = min_max_el_size
        self.Nel_per_wavelenth = Nel_per_wavelenth
        self.n_gauss = n_gauss
        #self.rectangle = []
        try:
            self.beta = (self.air.rho0 * self.air.c0) / self.material.Zs  # normalized surface admitance
        except:
            self.beta = []
        self.pres_s = []
        self.ux_s = []
        self.uy_s = []
        self.uz_s = []
        #self.Nzeta, self.Nweights = zeta_weights_tri()
        if bar_mode == 'notebook':
            from tqdm.notebook import trange, tqdm
        else:
            from tqdm import tqdm
        self.tqdm = tqdm
    
    def get_min_max_elsize(self):
        """ Gets the minimum and maximum element size
        """
        if not self.Nel_per_wavelenth:
            min_el_size = np.amin(self.min_max_el_size)
            max_el_size = np.amax(self.min_max_el_size)
        else:
            freq_max = self.controls.freq[-1]
            max_el_size = self.air.c0 / (self.Nel_per_wavelenth * freq_max)
            min_el_size = 0.9*max_el_size
        return min_el_size, max_el_size
    
    def meshit(self, name):
        """ Mesh the sample
        """
        # get minimum and maximum element sizes
        min_el_size, max_el_size = self.get_min_max_elsize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_el_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_el_size)
        gmsh.model.mesh.generate(2)
        gmsh.model.occ.synchronize()
        gmsh.write(name)
        gmsh.finalize()
        msh = meshio.read(name)
        self.nodes = msh.points
        self.elem_surf = msh.cells_dict["triangle"]
        # compute centroid and area
        self.elem_area = np.zeros(self.elem_surf.shape[0])
        self.elem_center = np.zeros((self.elem_surf.shape[0], 3))
        for jel, tri in enumerate(self.elem_surf):
            vertices = self.nodes[self.elem_surf[jel,:],:]
            self.elem_area[jel] = triangle_area(vertices)
            self.elem_center[jel,:] = triangle_centroid(vertices)

    def rectangle_s(self, Lx = 1.0, Ly = 1.0):
        """ Generate a rectangular sample and its mesh

        The mesh will consists of triangular elements. Their size  can be user
        specified or it is a function of the maximum frequency intended and
        the number of elements per wavelength (recomended: 6)

        Parameters
        ----------
        Lx : float
            Sample's lenght
        Ly : float
            Sample's width
        """
        # Generate geometry
        self.rectangle = Rectangle(Lx, Ly)
        self.baffle_size = 2*Lx
        # Initialize gmsh
        gmsh.initialize()
        gmsh.model.add("rectangle.geo")
        gmsh.model.occ.addRectangle(-self.rectangle.Lx/2, -self.rectangle.Ly/2,
                                    0.0, self.rectangle.Lx,self.rectangle.Ly)
        gmsh.model.occ.synchronize()
        # mesh
        self.meshit("rectangle.vtk")
        
    def circle_s(self, radius = 0.5):
        # Generate geometry
        self.circle = Circle(radius)
        self.baffle_size = 4*radius
        # Initialize gmsh
        gmsh.initialize()
        gmsh.model.add("circle.geo")
        gmsh.model.occ.addDisk(0.0, 0.0, 0.0, self.circle.radius, self.circle.radius)
        gmsh.model.occ.synchronize()
        # mesh
        self.meshit("circle.vtk")
        
    def psurf(self,):
        """ Calculates the surface pressure of the BEM mesh.

        Uses Python implemented module.
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
        Nksi, Nweights = ksi_weights_tri_mtx(n_gauss = self.n_gauss)
        #Nksi = np.ascontiguousarray(Nksi)
        # Allocate memory for the surface pressure data (# each column a frequency, each line an element)
        Nel = self.elem_center.shape[0]
        self.p_surface = np.zeros((Nel, len(self.controls.k0)), dtype=complex)
        # Generate the C matrix
        c_mtx = 0.5 * np.identity(Nel) #, dtype = np.float32
        # Calculate the distance from source to each element center
        el_3Dcoord = np.zeros((Nel, 3)) #dtype=np.float32
        el_3Dcoord[:,0:2] = self.elem_center[:,0:2]
        rsel = np.repeat(np.reshape(self.sources.coord[0,:],(1,3)),Nel,axis=0)-\
            el_3Dcoord
        r_unpt = np.linalg.norm(rsel, axis = 1)

        tinit = time.time()
        bar = tqdm(total = len(self.controls.k0),
            desc = 'Calculating the surface pressure for each frequency step (method 1)')
        for jf, k0 in enumerate(self.controls.k0):
            #Version 1 (distances in loop) - most time spent here
            gij = bemflush_mtx_tri(self.elem_center, self.nodes, self.elem_surf,
                self.elem_area, Nksi, Nweights, k0, self.beta[jf])

            # Calculate the unperturbed pressure
            p_unpt = 2.0 * np.exp(-1j * k0 * r_unpt) / r_unpt
            # Solve system of equations
            # print("Solving system of eqs for freq: {} Hz.".format(self.controls.freq[jf]))
            
            self.p_surface[:, jf] = np.linalg.solve(c_mtx - gij, p_unpt)
            
            bar.update(1)
        bar.close()
        tend = time.time()
        #print("elapsed time: {}".format(tend-tinit))

    def p_fps(self, bar_leave = True):
        """ Calculates the total sound pressure spectrum at the receivers coordinates.

        The sound pressure spectrum is calculatef for all receivers (attribute of class).
        The quantity calculated is the total sound pressure = incident + scattered.
        """
        # Get shape functions and weights
        Nksi, Nweights = ksi_weights_tri_mtx(n_gauss = self.n_gauss)
        # Loop the receivers
        self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            bar = self.tqdm(total = self.receivers.coord.shape[0], leave = bar_leave,
                    desc = 'Processing spectrum at each field point')
            for jrec, r_coord in enumerate(self.receivers.coord):
                xdist = (s_coord[0] - r_coord[0])**2.0
                ydist = (s_coord[1] - r_coord[1])**2.0
                r = (xdist + ydist)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                #print('Calculate sound pressure for source {} at ({}) and receiver {} at ({})'.format(js+1, s_coord, jrec+1, r_coord))
# =============================================================================
#                 bar = tqdm(total = len(self.controls.k0),
#                     desc = 'Processing sound pressure at field point')
# =============================================================================
                for jf, k0 in enumerate(self.controls.k0):
                    p_scat = bemflush_pscat_tri(r_coord, self.nodes, 
                        self.elem_surf, self.elem_area, Nksi, 
                        Nweights, k0, self.beta[jf], self.p_surface[:,jf])
                    pres_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1) +\
                        (np.exp(-1j * k0 * r2) / r2) + p_scat
                bar.update(1)
            bar.close()
            self.pres_s.append(pres_rec)

    def uz_fps(self, bar_leave = True):
        """ Calculates the z component of particle vel. at the receivers coordinates.

        The z component of particle velocity is calculatef for all receivers (attribute of class).
        The quantity calculated is the uz = incident + scattered.
        """
        # Get shape functions and weights
        Nksi, Nweights = ksi_weights_tri_mtx(n_gauss = self.n_gauss)
        # Loop the receivers
        self.uz_s = []
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            uz_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            bar = self.tqdm(total = self.receivers.coord.shape[0], leave = bar_leave,
                    desc = 'Processing uz spectrum at each field point')
            for jrec, r_coord in enumerate(self.receivers.coord):
                xdist = (s_coord[0] - r_coord[0])**2.0
                ydist = (s_coord[1] - r_coord[1])**2.0
                r = (xdist + ydist)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                for jf, k0 in enumerate(self.controls.k0):
                    uz_scat = bemflush_uzscat_tri(r_coord, self.nodes, 
                        self.elem_surf, self.elem_area, Nksi, 
                        Nweights, k0, self.beta[jf], self.p_surface[:,jf])
                    uz_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1)*\
                         (1 + (1 / (1j * k0 * r1)))* ((hs - zr)/r1)-\
                         (np.exp(-1j * k0 * r2) / r2) *\
                         (1 + (1 / (1j * k0 * r2))) * ((hs + zr)/r2) - uz_scat
                bar.update(1)
            bar.close()
            self.uz_s.append(uz_rec)
    
    def ux_fps(self, bar_leave = True):
        """ Calculates the x component of particle vel. at the receivers coordinates.

        The x component of particle velocity is calculatef for all receivers (attribute of class).
        The quantity calculated is the ux = incident + scattered.
        """
        # Get shape functions and weights
        Nksi, Nweights = ksi_weights_tri_mtx(n_gauss = self.n_gauss)
        # Loop the receivers
        self.ux_s = []
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            ux_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            bar = self.tqdm(total = self.receivers.coord.shape[0], leave = bar_leave,
                    desc = 'Processing ux spectrum at each field point')
            for jrec, r_coord in enumerate(self.receivers.coord):
                xdist = (s_coord[0] - r_coord[0])**2.0
                ydist = (s_coord[1] - r_coord[1])**2.0
                r = (xdist + ydist)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                for jf, k0 in enumerate(self.controls.k0):
                    ux_scat = bemflush_uxscat_tri(r_coord, self.nodes, 
                        self.elem_surf, self.elem_area, Nksi, 
                        Nweights, k0, self.beta[jf], self.p_surface[:,jf])
                    ux_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1)*\
                         (1 + (1 / (1j * k0 * r1)))* (-r_coord[0]/r1)-\
                         (np.exp(-1j * k0 * r2) / r2) *\
                         (1 + (1 / (1j * k0 * r2))) * (-r_coord[0]/r2) - ux_scat
                bar.update(1)
            bar.close()
            self.ux_s.append(ux_rec)
     
    def uy_fps(self, bar_leave = True):
        """ Calculates the y component of particle vel. at the receivers coordinates.

        The y component of particle velocity is calculatef for all receivers (attribute of class).
        The quantity calculated is the uy = incident + scattered.
        """
        # Get shape functions and weights
        Nksi, Nweights = ksi_weights_tri_mtx(n_gauss = self.n_gauss)
        # Loop the receivers
        self.uy_s = []
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            uy_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            bar = self.tqdm(total = self.receivers.coord.shape[0], leave = bar_leave,
                    desc = 'Processing uy spectrum at each field point')
            for jrec, r_coord in enumerate(self.receivers.coord):
                xdist = (s_coord[0] - r_coord[0])**2.0
                ydist = (s_coord[1] - r_coord[1])**2.0
                r = (xdist + ydist)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                for jf, k0 in enumerate(self.controls.k0):
                    uy_scat = bemflush_uyscat_tri(r_coord, self.nodes, 
                        self.elem_surf, self.elem_area, Nksi, 
                        Nweights, k0, self.beta[jf], self.p_surface[:,jf])
                    uy_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1)*\
                         (1 + (1 / (1j * k0 * r1)))* (-r_coord[1]/r1)-\
                         (np.exp(-1j * k0 * r2) / r2) *\
                         (1 + (1 / (1j * k0 * r2))) * (-r_coord[1]/r2) - uy_scat
                bar.update(1)
            bar.close()
            self.uy_s.append(uy_rec)

    def add_noise(self, snr = 30, uncorr = False, seed = 0):
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
            signalPower_lin = np.mean((np.abs(signal)/np.sqrt(2))**2, axis=0)
            signalPower_dB = 10 * np.log10(signalPower_lin)
            noisePower_dB = signalPower_dB - snr
            noisePower_lin = 10 ** (noisePower_dB/10)
            if signal_u.any() != 0:
                signalPower_lin_u = (np.abs(np.mean(signal_u, axis=0))/np.sqrt(2))**2
                signalPower_dB_u = 10 * np.log10(signalPower_lin_u)
                noisePower_dB_u = signalPower_dB_u - snr
                noisePower_lin_u = 10 ** (noisePower_dB_u/10)
        #seed = np.random.randint(low=0, high =1000)
        np.random.seed(seed)
        noise = np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape) +\
                1j*np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape)
# =============================================================================
#         Amp = np.random.normal(np.sqrt(2*noisePower_lin), np.sqrt(2*noisePower_lin)/2, size = signal.shape)
#         noise = (np.abs(Amp))*np.exp(1j*np.random.uniform(low=-np.pi, high=np.pi, size=signal.shape))
# =============================================================================
        self.pres_s[0] = signal + noise

        meas_snr = np.mean(10 * np.log10(np.abs(signal)**2) - 10 * np.log10(np.abs(noise)**2))
        std_snr = np.std(10 * np.log10(np.abs(signal)**2) - 10 * np.log10(np.abs(noise)**2))
        print('mean -std snr {} - {}'.format(meas_snr, std_snr))
        if signal_u.any() != 0:
            # print('Adding noise to particle velocity')
            noise_u = np.random.normal(0, np.sqrt(noisePower_lin_u), size = signal_u.shape) +\
                1j*np.random.normal(0, np.sqrt(noisePower_lin_u), size = signal_u.shape)
            self.uz_s[0] = signal_u + noise_u
            
    def plot_sample(self, vsam_size = 1.2, renderer='notebook',
                   save_state = False, path ='', filename = '',
                   plot_elemcenter = False):
        """ Plot of the sample and baffle using plotly

        Parameters
        ----------
        vsam_size : float
            Scene size. Just to make the plot look nicer. You can choose any value.
            An advice is to choose a value bigger than the sample's largest dimension.
        renderer : str
            Choose a renderer to plot. if you are using jupyter, 
            then renderer = "notebook"; if you are using colab, 
            then renderer = "colab"; if you are using spyder, 
            then renderer = "browser". 
        plot_elemcenter : bool
            wether or not to plot the element center for visual inspection
        """
        self.baffle_size = vsam_size
        # baffle vertices
        baffle_vertices = np.array([[-self.baffle_size/2, -self.baffle_size/2, 0.0],
                [self.baffle_size/2, -self.baffle_size/2, 0.0],
                [self.baffle_size/2, self.baffle_size/2, 0.0],
                [-self.baffle_size/2, self.baffle_size/2, 0.0]])      
        # sample vertices mesh
        vertices = self.nodes.T#[np.unique(self.elem_surf)].T
        elements = self.elem_surf.T
        # Create the mesh
        fig = ff.create_trisurf(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            simplices=elements.T,
            color_func=elements.shape[1] * ["rgb(184, 134, 11)"],
            title="Flush sample on hard baffle"
        )
        # add the baffle
        fig.add_trace(go.Mesh3d(x=baffle_vertices[:,0], y=baffle_vertices[:,1],
                  z=baffle_vertices[:,2], color='grey', opacity=0.70))
        
        # add el center
        if plot_elemcenter:
            fig.add_trace(go.Scatter3d(x = self.elem_center[:,0],
                y = self.elem_center[:,1], z = self.elem_center[:,2], 
                name="El. centers", mode='markers',
                marker=dict(size=2, color='black',opacity=0)))
     
        pio.renderers.default = renderer
        
        return fig
    
    def plot_source(self, fig):
        """ Add source to the scene using plotly
        """
        # add the sources
        if self.sources != None:
            fig.add_trace(go.Scatter3d(x = self.sources.coord[:,0],
                                       y = self.sources.coord[:,1],
                                       z = self.sources.coord[:,2],
                                       name="Sources",mode='markers',
                                       marker=dict(size=12, color='red',opacity=0.5)))
        return fig
    
    def plot_receivers(self, fig):
        """ Add receivers to the scene using plotly
        """
        # add the receivers
        if self.receivers != None:
            fig.add_trace(go.Scatter3d(x = self.receivers.coord[:,0],
                                       y = self.receivers.coord[:,1],
                                       z = self.receivers.coord[:,2], 
                                       name="Receivers", mode='markers',
                                       marker=dict(size=4, color='blue',opacity=0.4)))
        return fig
        
    def plot_scene(self, vsam_size = 1.2, renderer='notebook',
                   save_state = False, path ='', filename = '',
                   plot_elemcenter = False):
        """ Plot of the scene using plotly

        Parameters
        ----------
        vsam_size : float
            Scene size. Just to make the plot look nicer. You can choose any value.
            An advice is to choose a value bigger than the sample's largest dimension.
        renderer : str
            Choose a renderer to plot. if you are using jupyter, 
            then renderer = "notebook"; if you are using colab, 
            then renderer = "colab"; if you are using spyder, 
            then renderer = "browser". 
        plot_elemcenter : bool
            wether or not to plot the element center for visual inspection
        """
        fig = self.plot_sample(vsam_size = vsam_size, renderer=renderer,
                   save_state = save_state, path = path, filename = filename,
                   plot_elemcenter = plot_elemcenter)
        
        # add the sources
        fig = self.plot_source(fig)
        # add the receivers
        fig = self.plot_receivers(fig)

        pio.renderers.default = renderer
        if save_state:
            fig.write_image(path+filename+'.pdf', scale=2)
        fig.show()
        
    def save(self, filename = 'my_bemflush_geo', path = ''):
        """ To save the simulation object as pickle

        Parameters
        ----------
        filename : str
            name of the file
        pathname : str
            path of folder to save the file
        """
        filename = filename
        self.path_filename = path + filename + '.pkl'
        f = open(self.path_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename = 'my_bemflush_geo', path = ''):
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
        
def triangle_area(vertices):
    """Calculate the area of a triangle.
    
    Parameters
    ----------
    vertices : numpy 1d array
        A (3,) numpy array with the vertices of the triangle
    
    Returns
    -------
    area_tri : float
        the area of the triangle
    """
    # get one side of triangle and its norm
    ab = vertices[1] - vertices[0]
    ab_norm = np.linalg.norm(ab)
    # get another side of triangle and its norm
    ac = vertices[2] - vertices[0]
    ac_norm = np.linalg.norm(ac)
    # angle between sides
    theta = np.arccos(np.dot(ab, ac)/(ab_norm * ac_norm))
    # area
    area_tri = 0.5 * ab_norm * ac_norm * np.sin(theta)
    return area_tri

def triangle_centroid(vertices):
    """ Calculates the center of a triangle
    
    Parameters
    ----------
    vertices : numpy 1d array
        A (3,) numpy array with the vertices of the triangle
    
    Returns
    ---------
    centroid : numpy 1d array
        the centroid of the triangle
    """
    return (vertices[0] + vertices[1] + vertices[2])/3        

def ksi_weights_tri_mtx(n_gauss = 6):
    """ Calculates Nksi and Nweights matrices
    
    This function calculates Nksi and Nweights matrices to be used in Gaussian
    quadrature matrix integration of triangular elements. It will return the 
    shape functions as a matrix of 3 x n_gauss elements and a vector of weights
    of n_gauss elements. Only certain number of gauss points are allowed: 
    3, 6 - this approach avoids singularity in the case of collocation
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
    Nksi = np.zeros((3, n_gauss))
    Nweights = np.zeros(n_gauss)
    
    # Write ksi1, ksi2 and weights
    if n_gauss == 3:
        a = 1/6
        b = 2/3
        ksi1 = np.array([a, a, b])
        ksi2 = np.array([a, b, a])
        Nweights += 1/6 
    else:
        a = 0.445948490915965
        b = 0.091576213509771
        c = 1-2*a
        d = 1-2*b
        ksi1 = np.array([a, c, a, b, d, b])
        ksi2 = np.array([a, a, c, b, b, d])
        Nweights[0:3] = 0.111690794839005
        Nweights[3:6] =  0.054975871827661
         
    # write shape functions
    Nksi[0,:] = ksi1
    Nksi[1,:] = ksi2
    Nksi[2,:] = 1-ksi1-ksi2
    
    return Nksi, Nweights

@njit
def gaussint_tri(r_coord, nx, ny, area, Nksi, Nweights, k0, beta = 1+0*1j):
    # Vector of receiver coordinates (for vectorized integration)
    x_coord = r_coord[0] * np.ones(Nksi.shape[1])
    y_coord = r_coord[1] * np.ones(Nksi.shape[1])
    z_coord = r_coord[2] * np.ones(Nksi.shape[1])
    # Jacobian of squared element
    jacobian = area*2 #triangle_area(nodes)*2
    #jacobian = ((nodes[:,0][1] - nodes[:,0][0])**2.0)/4.0
    # Gauss points on local element
    xksi = np.dot(nx, Nksi) #nodes[:,0] @ Nksi
    yksi = np.dot(ny, Nksi) #nodes[:,1] @ Nksi
    # Calculate the distance from el center to transformed integration points
    r = ((x_coord - xksi)**2 + (y_coord - yksi)**2 + z_coord**2)**0.5
    # Calculate green function
    g = -1j * k0 * beta * (np.exp(-1j * k0 * r)/(4 * np.pi * r)) * jacobian
    #print(g.shape)
    ival = np.sum((Nweights*g))#np.dot(Nweights, g) #g @ Nweights
    return ival, xksi, yksi


def bemflush_mtx_tri(el_center, nodes, elem_surf, areas,
                     Nksi, Nweights, k0, beta):
    """ Creates the BEM matrix
    
    Parameters
    -------------
    el_centerxy : numpy ndarray (2 x nel )
        x-y coordinates of element centers (z is always 0)
    nodesxy : numpy ndarray (2 x nnodes)
        x-y coordinates of nodes (z is always 0)
    elem_surf : numpy ndarray (nel x 3)
        indices of nodes making up the element
    areas : numpy array (nel x 1)
        area of each element
    Nzeta : numpy ndarray
        zeta matrix
    Nweights : numpy array
        weights array
    k0 : float
        wave number
    beta : complex float
        surface admitance
    
    Returns
    -------------
    bem_mtx : numpy ndarray (nel x nel)
        bem matrix (complex)
    """
    # Number of elements
    Nel = el_center.shape[0]
    # initialize
    bem_mtx = np.zeros((Nel, Nel), dtype = np.complex64)
    for i in np.arange(Nel):
        # colocation point
        r_colocation = el_center[i,:]
        for j in np.arange(Nel):
            # Get nodes of element
            nodes_of_el = nodes[elem_surf[j]]
            nx = np.array(nodes_of_el[:,0])
            ny = np.array(nodes_of_el[:,1])
            area = areas[j]
            # Integrate
            bem_mtx[i,j], _, _ = gaussint_tri(r_colocation, nx, ny, area, 
                                              Nksi, Nweights, k0, beta = beta)
    return bem_mtx

def bemflush_pscat_tri(r_coord, nodes, elem_surf, areas, 
                   Nksi, Nweights, k0, beta, ps):
    """ Calculates the pressure at a receiver
    
    Parameters
    -------------
    r_coord : numpy ndarray (1 x 3)
        The coordinates of the receiver
    nodesxy : numpy ndarray (2 x nnodes)
        x-y coordinates of nodes (z is always 0)
    elem_surf : numpy ndarray (nel x 3)
        indices of nodes making up the element
    areas : numpy array (nel x 1)
        area of each element
    Nzeta : numpy ndarray
        zeta matrix
    Nweights : numpy array
        weights array
    k0 : float
        wave number
    beta : complex float
        surface admitance
    ps : numpy array (nel x 1)
        surface sound pressure
    
    Returns
    -------------
    p_scat : float
        scattered sound pressure (complex)
    """
    # Number of elements and jacobian
    Nel = elem_surf.shape[0]
    # Initialization
    gfield = np.zeros(Nel, dtype = np.complex64)
    #Loop through elements once
    for j in np.arange(Nel):
        # Get nodes of element
        nodes_of_el = nodes[elem_surf[j]]
        nx = np.array(nodes_of_el[:,0])
        ny = np.array(nodes_of_el[:,1])
        area = areas[j]
        # Integrate
        gfield[j], _, _ = gaussint_tri(r_coord, nx, ny, area,
                                          Nksi, Nweights, k0, beta = beta)
    # Scattered pressure    
    p_scat = np.dot(gfield, ps)
    return p_scat


def bemflush_uzscat_tri(r_coord, nodes, elem_surf, areas, 
                   Nksi, Nweights, k0, beta, ps):
    """ Calculates the uz component at a receiver
    
    Parameters
    -------------
    r_coord : numpy ndarray (1 x 3)
        The coordinates of the receiver
    nodesxy : numpy ndarray (2 x nnodes)
        x-y coordinates of nodes (z is always 0)
    elem_surf : numpy ndarray (nel x 3)
        indices of nodes making up the element
    areas : numpy array (nel x 1)
        area of each element
    Nzeta : numpy ndarray
        zeta matrix
    Nweights : numpy array
        weights array
    k0 : float
        wave number
    beta : complex float
        surface admitance
    ps : numpy array (nel x 1)
        surface sound pressure
    
    Returns
    -------------
    uz_scat : float
        scattered z component of particle velocity (complex)
    """
    # Number of elements and jacobian
    Nel = elem_surf.shape[0]
    # Initialization
    gfield = np.zeros(Nel, dtype = np.complex64)
    #Loop through elements once
    for j in np.arange(Nel):
        # Get nodes of element
        nodes_of_el = nodes[elem_surf[j]]
        nx = np.array(nodes_of_el[:,0])
        ny = np.array(nodes_of_el[:,1])
        area = areas[j]
        # Integrate
        gfield[j], _, _ = gaussint_tri_uz(r_coord, nx, ny, area,
                                          Nksi, Nweights, k0, beta = beta)
        
    # Scattered pressure    
    uz_scat = np.dot(gfield, ps)
    return uz_scat

@njit
def gaussint_tri_uz(r_coord, nx, ny, area, Nksi, Nweights, k0, beta = 1+0*1j):
    # Vector of receiver coordinates (for vectorized integration)
    x_coord = r_coord[0] * np.ones(Nksi.shape[1])
    y_coord = r_coord[1] * np.ones(Nksi.shape[1])
    z_coord = r_coord[2] * np.ones(Nksi.shape[1])
    # Jacobian of squared element
    jacobian = area*2 
    # Gauss points on local element
    xksi = np.dot(nx, Nksi) #nodes[:,0] @ Nksi
    yksi = np.dot(ny, Nksi) #nodes[:,1] @ Nksi
    # Calculate the distance from el center to transformed integration points
    r = ((x_coord - xksi)**2 + (y_coord - yksi)**2 + z_coord**2)**0.5
    # Calculate green function
    g = -1j * k0 * beta * (np.exp(-1j * k0 * r)/(4 * np.pi * r)) *\
        (1/(1j*k0*r) + 1) * (r_coord[2]/r) * jacobian
    ival = np.sum((Nweights*g))#np.dot(Nweights, g) #g @ Nweights
    return ival, xksi, yksi

def bemflush_uxscat_tri(r_coord, nodes, elem_surf, areas, 
                   Nksi, Nweights, k0, beta, ps):
    """ Calculates the ux component at a receiver
    
    Parameters
    -------------
    r_coord : numpy ndarray (1 x 3)
        The coordinates of the receiver
    nodesxy : numpy ndarray (2 x nnodes)
        x-y coordinates of nodes (z is always 0)
    elem_surf : numpy ndarray (nel x 3)
        indices of nodes making up the element
    areas : numpy array (nel x 1)
        area of each element
    Nzeta : numpy ndarray
        zeta matrix
    Nweights : numpy array
        weights array
    k0 : float
        wave number
    beta : complex float
        surface admitance
    ps : numpy array (nel x 1)
        surface sound pressure
    
    Returns
    -------------
    ux_scat : float
        scattered x component of particle velocity (complex)
    """
    # Number of elements and jacobian
    Nel = elem_surf.shape[0]
    # Initialization
    gfield = np.zeros(Nel, dtype = np.complex64)
    #Loop through elements once
    for j in np.arange(Nel):
        # Get nodes of element
        nodes_of_el = nodes[elem_surf[j]]
        nx = np.array(nodes_of_el[:,0])
        ny = np.array(nodes_of_el[:,1])
        area = areas[j]
        # Integrate
        gfield[j], _, _ = gaussint_tri_ux(r_coord, nx, ny, area,
                                          Nksi, Nweights, k0, beta = beta)
        
    # Scattered pressure    
    ux_scat = np.dot(gfield, ps)
    return ux_scat

@njit
def gaussint_tri_ux(r_coord, nx, ny, area, Nksi, Nweights, k0, beta = 1+0*1j):
    # Vector of receiver coordinates (for vectorized integration)
    x_coord = r_coord[0] * np.ones(Nksi.shape[1])
    y_coord = r_coord[1] * np.ones(Nksi.shape[1])
    z_coord = r_coord[2] * np.ones(Nksi.shape[1])
    # Jacobian of squared element
    jacobian = area*2 
    # Gauss points on local element
    xksi = np.dot(nx, Nksi) #nodes[:,0] @ Nksi
    yksi = np.dot(ny, Nksi) #nodes[:,1] @ Nksi
    xsm = x_coord-xksi
    # Calculate the distance from el center to transformed integration points
    r = ((x_coord - xksi)**2 + (y_coord - yksi)**2 + z_coord**2)**0.5
    # Calculate green function
    g = -1j * k0 * beta * (np.exp(-1j * k0 * r)/(4 * np.pi * r)) *\
        (1/(1j*k0*r) + 1) * (xsm/r) * jacobian
    ival = np.sum((Nweights*g))#np.dot(Nweights, g) #g @ Nweights
    return ival, xksi, yksi

def bemflush_uyscat_tri(r_coord, nodes, elem_surf, areas, 
                   Nksi, Nweights, k0, beta, ps):
    """ Calculates the uy component at a receiver
    
    Parameters
    -------------
    r_coord : numpy ndarray (1 x 3)
        The coordinates of the receiver
    nodesxy : numpy ndarray (2 x nnodes)
        x-y coordinates of nodes (z is always 0)
    elem_surf : numpy ndarray (nel x 3)
        indices of nodes making up the element
    areas : numpy array (nel x 1)
        area of each element
    Nzeta : numpy ndarray
        zeta matrix
    Nweights : numpy array
        weights array
    k0 : float
        wave number
    beta : complex float
        surface admitance
    ps : numpy array (nel x 1)
        surface sound pressure
    
    Returns
    -------------
    uy_scat : float
        scattered x component of particle velocity (complex)
    """
    # Number of elements and jacobian
    Nel = elem_surf.shape[0]
    # Initialization
    gfield = np.zeros(Nel, dtype = np.complex64)
    #Loop through elements once
    for j in np.arange(Nel):
        # Get nodes of element
        nodes_of_el = nodes[elem_surf[j]]
        nx = np.array(nodes_of_el[:,0])
        ny = np.array(nodes_of_el[:,1])
        area = areas[j]
        # Integrate
        gfield[j], _, _ = gaussint_tri_uy(r_coord, nx, ny, area,
                                          Nksi, Nweights, k0, beta = beta)
        
    # Scattered pressure    
    uy_scat = np.dot(gfield, ps)
    return uy_scat

@njit
def gaussint_tri_uy(r_coord, nx, ny, area, Nksi, Nweights, k0, beta = 1+0*1j):
    # Vector of receiver coordinates (for vectorized integration)
    x_coord = r_coord[0] * np.ones(Nksi.shape[1])
    y_coord = r_coord[1] * np.ones(Nksi.shape[1])
    z_coord = r_coord[2] * np.ones(Nksi.shape[1])
    # Jacobian of squared element
    jacobian = area*2 
    # Gauss points on local element
    xksi = np.dot(nx, Nksi) #nodes[:,0] @ Nksi
    yksi = np.dot(ny, Nksi) #nodes[:,1] @ Nksi
    ysm = y_coord-yksi
    # Calculate the distance from el center to transformed integration points
    r = ((x_coord - xksi)**2 + (y_coord - yksi)**2 + z_coord**2)**0.5
    # Calculate green function
    g = -1j * k0 * beta * (np.exp(-1j * k0 * r)/(4 * np.pi * r)) *\
        (1/(1j*k0*r) + 1) * (ysm/r) * jacobian
    ival = np.sum((Nweights*g))#np.dot(Nweights, g) #g @ Nweights
    return ival, xksi, yksi