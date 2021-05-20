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
                 min_max_el_size = [0.05, 0.1], Nel_per_wavelenth = []):
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
        #self.rectangle = []
        try:
            self.beta = (self.air.rho0 * self.air.c0) / self.material.Zs  # normalized surface admitance
        except:
            self.beta = []
        self.pres_s = []
        self.ux_s = []
        self.uy_s = []
        self.uz_s = []
        self.Nzeta, self.Nweights = zeta_weights_tri()
    
    def get_min_max_elsize(self):
        """ Gets the minimum and maximum element size
        """
        if not self.Nel_per_wavelenth:
            min_el_size = np.amin(self.min_max_el_size)
            max_el_size = np.amax(self.min_max_el_size)
        else:
            freq_max = self.controls.freq[-1]
            max_el_size = self.air.c0 / (self.Nel_per_wavelenth * freq_max)
            min_el_size = 0.5*max_el_size
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
            gij = bemflush_mtx_tri(self.elem_center[:,0:2], self.nodes[:,0:2], self.elem_surf,
                self.elem_area, self.Nzeta, self.Nweights.T, k0, self.beta[jf])

            # Calculate the unperturbed pressure
            p_unpt = 2.0 * np.exp(-1j * k0 * r_unpt) / r_unpt
            # Solve system of equations
            # print("Solving system of eqs for freq: {} Hz.".format(self.controls.freq[jf]))
            
            self.p_surface[:, jf] = np.linalg.solve(c_mtx + gij, p_unpt)
            
            bar.update(1)
        bar.close()
        tend = time.time()
        print("elapsed time: {}".format(tend-tinit))

    def p_fps(self,):
        """ Calculates the total sound pressure spectrum at the receivers coordinates.

        The sound pressure spectrum is calculatef for all receivers (attribute of class).
        The quantity calculated is the total sound pressure = incident + scattered.
        """
        # Loop the receivers
        self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                xdist = (s_coord[0] - r_coord[0])**2.0
                ydist = (s_coord[1] - r_coord[1])**2.0
                r = (xdist + ydist)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                print('Calculate sound pressure for source {} at ({}) and receiver {} at ({})'.format(js+1, s_coord, jrec+1, r_coord))
                bar = tqdm(total = len(self.controls.k0),
                    desc = 'Processing sound pressure at field point')
                for jf, k0 in enumerate(self.controls.k0):
                    p_scat = bemflush_pscat_tri(r_coord, self.nodes[:,0:2], 
                        self.elem_surf, self.elem_area, self.Nzeta, 
                        self.Nweights.T, k0, self.beta[jf], self.p_surface[:,jf])
                    pres_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1) +\
                        (np.exp(-1j * k0 * r2) / r2) + p_scat
                    bar.update(1)
                bar.close()
            self.pres_s.append(pres_rec)
        
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
        # add the receivers
        if self.receivers != None:
            fig.add_trace(go.Scatter3d(x = self.receivers.coord[:,0],
                                       y = self.receivers.coord[:,1],
                                       z = self.receivers.coord[:,2], 
                                       name="Receivers", mode='markers',
                                       marker=dict(size=4, color='blue',opacity=0.4)))
        # add the sources
        if self.sources != None:
                fig.add_trace(go.Scatter3d(x = self.sources.coord[:,0],
                                           y = self.sources.coord[:,1],
                                           z = self.sources.coord[:,2],
                                           name="Sources",mode='markers',
                                           marker=dict(size=12, color='red',opacity=0.5)))
        
        # add el center
        if plot_elemcenter:
            fig.add_trace(go.Scatter3d(x = self.elem_center[:,0],
                y = self.elem_center[:,1], z = self.elem_center[:,2], 
                name="El. centers", mode='markers',
                marker=dict(size=2, color='black',opacity=0)))
     
        pio.renderers.default = renderer
        if save_state:
            fig.write_image(path+filename+'.pdf', scale=2)
        fig.show()
        
def triangle_area(vertices):
    """ Calculates the area of a triangle
    
    Parameters
    ----------
    vertices : numpy 1d array
        A (3,) numpy array with the vertices of the triangle
    
    Returns
    ---------
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


def zeta_weights_tri():
    """ Calculates Nzeta and Nweights matrices for a triangle
    """
    zeta = np.array([-0.93246951, -0.66120939, -0.23861918,
    0.23861918, 0.66120939, 0.93246951])

    weigths = np.array([0.17132449, 0.36076157, 0.46791393,
        0.46791393, 0.36076157, 0.17132449])

    # Create vectors of size 1 x 36 for the zetas
    N1 = np.matmul(np.reshape(zeta, (zeta.size,1)),  np.reshape(zeta, (1,zeta.size)))
    N2 = np.matmul(np.reshape(zeta, (zeta.size,1)),  np.reshape(zeta, (1,zeta.size)))
    N3 = np.matmul(np.reshape(1-zeta-zeta, (zeta.size,1)),  np.reshape(1-zeta-zeta, (1,zeta.size)))

# =============================================================================
#     N1 = (1-zeta).T @  (1-zeta) - (col @ row)
#     N1 = 0.25 * np.matmul(np.reshape(1-zeta, (zeta.size,1)),  np.reshape(1-zeta, (1,zeta.size)))
#     N2 = 0.25 * np.matmul(np.reshape(1+zeta, (zeta.size,1)),  np.reshape(1-zeta, (1,zeta.size)))
#     N3 = 0.25 * np.matmul(np.reshape(1+zeta, (zeta.size,1)),  np.reshape(1+zeta, (1,zeta.size)))
#     N4 = 0.25 * np.matmul(np.reshape(1-zeta, (zeta.size,1)),  np.reshape(1+zeta, (1,zeta.size)))
# =============================================================================
    # Flattens
    N1 = np.reshape(N1, (1,zeta.size**2))
    N2 = np.reshape(N2, (1,zeta.size**2))
    N3 = np.reshape(N3, (1,zeta.size**2))
    #N4 = np.reshape(N4, (1,zeta.size**2))

    # Let each line of the following matrix be a N vector
    Nzeta = np.zeros((3, zeta.size**2))
    Nzeta[0,:] = N1
    Nzeta[1,:] = N2
    Nzeta[2,:] = N3
    #Nzeta[3,:] = N4

    # Create vector of size 1 x 36 for the weights
    # Nweights = (w).T @  (w) - (col @ row)
    Nweigths = np.matmul(np.reshape(weigths, (zeta.size,1)),  np.reshape(weigths, (1,zeta.size)))
    # Flattens
    Nweigths = np.reshape(Nweigths, (1,zeta.size**2))
    # print('I have calculated!')
    return Nzeta, Nweigths

def bemflush_mtx_tri(el_centerxy, nodesxy, elem_surf, areas ,
                     Nzeta, Nweights, k0, beta):
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
    Nel = el_centerxy.shape[0]
    #jacobian = ((el_center[1,0] - el_center[0,0])**2.0)/4.0 #Fix 5.26 p113
    # initialize
    bem_mtx = np.zeros((Nel, Nel), dtype = np.complex64)
    for i in np.arange(Nel):
        #jacobian = areas[i]/2
        xy_center = el_centerxy[i,:]
        x_center = xy_center[0] * np.ones(Nzeta.shape[1])
        y_center = xy_center[1] * np.ones(Nzeta.shape[1])
# =============================================================================
#         jacobian = (x_center[1]-x_center[0])*(y_center[2]-y_center[0])-\
#             (x_center[2]-x_center[0])*(y_center[1]-y_center[0])
# =============================================================================
        for j in np.arange(Nel):
            xnode = nodesxy[elem_surf[j]][:,0]
            ynode = nodesxy[elem_surf[j]][:,1]
            jacobian = areas[j]/2
# =============================================================================
#             jacobian = (xnode[1]-xnode[0])*(ynode[2]-ynode[0])-\
#                 (xnode[2]-xnode[0])*(ynode[1]-ynode[0])
# =============================================================================
            xzeta = xnode @ Nzeta # Fix
            yzeta = ynode @ Nzeta
            # calculate the distance from el center to transformed integration points
            r = ((x_center - xzeta)**2 + (y_center - yzeta)**2)**0.5
            g = 1j * k0 * beta * (np.exp(-1j * k0 * r)/(4 * np.pi * r)) * jacobian
            #print(g.astype(np.complex64).dtype)
            bem_mtx[i,j] = g @ Nweights
    return bem_mtx

def bemflush_pscat_tri(r_coord, nodesxy, elem_surf, areas, 
                   Nzeta, Nweights, k0, beta, ps):
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
    # Vector of receiver coordinates (for vectorized integration)
    x_coord = r_coord[0] * np.ones(Nzeta.shape[1])
    y_coord = r_coord[1] * np.ones(Nzeta.shape[1])
    z_coord = r_coord[2] * np.ones(Nzeta.shape[1])
    
    # Number of elements and jacobian
    Nel = elem_surf.shape[0]
    # Initialization
    gfield = np.zeros(Nel, dtype = np.complex64)
    #Loop through elements once
    for j in np.arange(Nel):
        jacobian = areas[j]/2
        # Transform the coordinate system for integration between -1,1 and +1,+1
        xnode = nodesxy[elem_surf[j]][:,0]
        ynode = nodesxy[elem_surf[j]][:,1]
# =============================================================================
#         jacobian = (xnode[1]-xnode[0])*(ynode[2]-ynode[0])-\
#             (xnode[2]-xnode[0])*(ynode[1]-ynode[0])
# =============================================================================
        xzeta = xnode @ Nzeta
        yzeta = ynode @ Nzeta
        # Calculate the distance from el center to transformed integration points
        r = ((x_coord - xzeta)**2 + (y_coord - yzeta)**2 + z_coord**2)**0.5
        # Calculate green function
        g = -1j * k0 * beta * (np.exp(-1j * k0 * r)/(4 * np.pi * r)) * jacobian
        # Integrate
        gfield[j] = g @ Nweights;
    p_scat = gfield @ ps
    return p_scat
