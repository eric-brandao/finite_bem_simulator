# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 13:08:50 2021

@author: ericb
"""
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from tqdm.notebook import trange, tqdm
from tqdm import tqdm
from controlsair import AirProperties, AlgControls
from material import PorousAbsorber
from sources import Source
from receivers import Receiver
from field_bemflush import BEMFlushSq

class GenDataSet():
    """Control dataset generation for finite samples.
    """
    def __init__(self, base_folder = '/base_cases/', main_folder = 'round2/',
                 computed_df = 'log.csv'):
        self.computed_df = pd.read_csv(computed_df)
        self.computed_df_fname = computed_df
        self.size_of_computed = len(self.computed_df)
        self.base_folder = base_folder
        self.main_folder = main_folder
        
        
    def generate_test_mtx(self, Lx, Ly, resist, d1, r, theta_dl, phi_dl,
                          add_noise = False, print_allwewant = True):
        """Generate a test matrix
        
        Pass inputs as vectors and we take care of the rest. loop and create 
        combinations.
        """
        self.Lx = Lx
        self.Ly = Ly
        self.resist = resist
        self.d1 = d1
        self.add_noise = add_noise
        # exclude azimuth angles from theta = 0 deg incidence
        theta_d, phi_d = np.meshgrid(theta_dl, phi_dl)
        theta_d = theta_d.flatten()
        phi_d = phi_d.flatten()
        # find indexes of theta = 0 deg and eliminate almost all from theta and phi
        id_0deg = np.where(theta_d.flatten() == 0)
        theta_d = np.delete(theta_d, id_0deg[0][1:])
        phi_d = np.delete(phi_d, id_0deg[0][1:])
        # stack together as a pair
        self.r = r
        self.theta_phi_deg = np.hstack((np.reshape(theta_d, (len(theta_d), 1)), 
            np.reshape(phi_d, (len(phi_d), 1))))
        # convert to radians
        self.theta_phi = np.deg2rad(self.theta_phi_deg)
        # Now, we generate an empty data-frame and populate target simulations
        columns = ['Lx [m]', 'Ly [m]', 'Flow resistivity [N s/m^4]', 
                   'Thickness [m]', 'r [m]', 'Elevation [deg]', 'Azimuth [deg]', 
                   'Added noise? (bool)']
        columns = self.computed_df.columns[:8]
        self.target_df = pd.DataFrame(columns=columns)
        counter = 0
        for lx in self.Lx:
            for ly in self.Ly: #reversed(Ly)
                for res in self.resist:
                    for d in self.d1:
                        for jang in np.arange(self.theta_phi.shape[0]):
                            self.target_df.loc[counter] = [lx, ly, res, d, self.r, 
                                    np.rad2deg(self.theta_phi[jang, 0]), 
                                    np.rad2deg(self.theta_phi[jang, 1]),
                                    str(self.add_noise)]
                            counter += 1
        self.size_of_sim = len(self.Lx) * len(self.Ly) * len(self.resist) *\
            len(self.d1) * self.theta_phi_deg.shape[0]
        if print_allwewant:
            print("This is all we want!")
            print("Span of variables:" )
            print("Lx: {} [m]".format(self.Lx))
            print("Ly: {} [m]".format(self.Ly))
            print("Flow Resistivity: {} [Ns/m^4]".format(self.resist))
            print("Sample thickness: {} [m]".format(self.d1))
            print("Elevation angles: {} [deg]".format(np.unique(self.theta_phi_deg[:,0])))
            print("Azimuth angles: {} [deg]".format(np.unique(self.theta_phi_deg[:,1])))
            print("Total number of simulations is: {} cases".format(self.size_of_sim))
            print("So far we computed: {} cases".format(self.size_of_computed))
            print("Number if cases to go: {} cases".format(self.size_of_sim-self.size_of_computed))
    
    def generate_test_mtx2(self, Lx, Ly, resist, d1, r, theta_dl, phi_dl,
                          add_noise = False, print_allwewant = True):
        """Generate a test matrix
        
        Pass inputs as vectors and we take care of the rest. The vectors themselves
        form the test matrix.
        """
        self.Lx = Lx
        self.Ly = Ly
        self.resist = resist
        self.d1 = d1
        self.add_noise = np.repeat(add_noise, len(Lx))
        self.r = r
        self.theta = theta_dl
        self.phi_dl = phi_dl
        mtx = np.vstack((Lx, Ly, resist, d1, r, theta_dl, phi_dl))
        matrix = mtx.T
        # Now, we generate an empty data-frame and populate target simulations
        columns = ['Lx [m]', 'Ly [m]', 'Flow resistivity [N s/m^4]',  
                   'Thickness [m]', 'r [m]', 'Elevation [deg]', 'Azimuth [deg]', 
                   'Added noise? (bool)']
        self.target_df = pd.DataFrame(columns=columns)
        self.target_df['Lx [m]'] = Lx
        self.target_df['Ly [m]'] = Ly
        self.target_df['Flow resistivity [N s/m^4]'] = resist
        self.target_df['Thickness [m]'] = d1
        self.target_df['r [m]'] = r
        self.target_df['Elevation [deg]'] = theta_dl
        self.target_df['Azimuth [deg]'] = phi_dl
        self.target_df['Added noise? (bool)'] = self.add_noise
        self.size_of_sim = len(self.target_df)
    
    def df_difference(self,):
        """Compute unique cases to generate
        """
        target_columns = self.target_df.columns[0:7]
        computed_df = self.computed_df[target_columns]
        target_df = self.target_df[target_columns]
        df_appended = target_df.append(computed_df, ignore_index=True, sort=False)
        self.tocompute_df = df_appended.drop_duplicates(subset = target_columns, 
                                                        keep=False,inplace=False)
        self.tocompute_df.reset_index(inplace = True)
        self.tocompute_df.drop('index', axis='columns', inplace=True)
        
    def run_bemflush_rect_field(self, air, controls, receivers, base_field, r,
        res = 10000, d = 0.025, theta = 0, phi = 0, snr = 1000, n_gauss = 36):
        """Run a BEM flush rectangular case and return the field object
        
        Uses the G matrix stored in base_field to compute the surface pressure
        and evaluate field points. The G matrix is erased in the returned field
        object.
        """
        # Use air and controls to generate the material BC
        material = PorousAbsorber(air = air, controls = controls)
        material.miki(resistivity = res)
        material.layer_over_rigid(thickness = d, theta = theta)
        # Instatiate source object
        s_coord = sph2cart(r, np.pi/2-theta, phi)
        sources = Source(coord = s_coord)
        # Instantiate Field object from base field
        field = BEMFlushSq(air = air, controls = controls, material = material, 
            sources = sources, receivers = receivers, n_gauss = n_gauss)
        # parse mesh (6 el per wavelength)
        field.parse_mesh(base_field)
        field.gij_f = base_field.gij_f
        # Calculate the surface pressure
        field.psurf2(erase_gij = True, bar_leave = False)
        # Evaluate pressure at all field points
        field.p_fps(bar_leave = False)
        if add_noise:
            field.add_noise(snr = snr)
        return field
    
    def bem_dataset_gen(self, air, controls, receivers, r, 
                base_fname_base  = 'bemfbase_', base_filename  = 'bemf',
                snr = 40, n_gauss = 36, Nel_per_wavelenth = 6):
        """BEM flush database generator - rectangular samples.
        """
        # Columns
        target_columns = self.tocompute_df.columns[0:6]
        # Get all vectors to run loop
        Lx = np.unique(self.tocompute_df[target_columns[0]])
        Ly = np.unique(self.tocompute_df[target_columns[1]])
        resist = np.unique(self.tocompute_df[target_columns[2]])
        d1 = np.unique(self.tocompute_df[target_columns[3]])
        theta_deg = np.unique(self.tocompute_df[target_columns[4]])
        phi_deg = np.unique(self.tocompute_df[target_columns[5]])
        #print(Lx ,Ly, resist, d1, theta_deg, phi_deg)
        # set a bar to keep track 
        bar = tqdm(total = self.size_of_sim, desc = 'Gen. database from base...')
        bar.update(self.size_of_computed)
        lx_dummy = 0
        ly_dummy = 0
        for jc in np.arange(self.size_of_sim-self.size_of_computed):
            # Get values for simulation
            lx = self.tocompute_df[target_columns[0]][jc]
            ly = self.tocompute_df[target_columns[1]][jc]
            res = self.tocompute_df[target_columns[2]][jc]
            d = self.tocompute_df[target_columns[3]][jc]
            theta = self.tocompute_df[target_columns[4]][jc]
            phi = self.tocompute_df[target_columns[5]][jc]
            # base field
            base_filename_load = base_fname_base + 'Lx'+ str(int(100*lx)) +\
                'cm_Ly' + str(int(100*ly))+ 'cm'
            base_field = BEMFlushSq()
            base_field.load(path = self.base_folder, filename = base_filename_load)
            # file name to save
            filename = base_filename + '_Lx' + str(int(100*lx)) +\
                'cm_Ly' + str(int(100*ly)) + 'cm_res' + str(int(res))+\
                '_d' + str(int(1000*d)) + 'mm_el' +\
                str(int(theta)) + '_az' + str(int(phi))
            # Figure time stamp at begining
            start_timestamp = datetime.now()
            #subfolder
            sub_folder = 'res' + str(int(res)) +'/d' + str(int(1000*d)) + 'mm/'
            field = self.run_bemflush_rect_field(air, controls, 
                receivers, base_field, r, res = res, d = d, 
                theta = np.deg2rad(theta), phi = np.deg2rad(phi), 
                snr = snr, n_gauss = n_gauss)
            field.save(path = self.main_folder + sub_folder, filename = filename)
            # Figure time stamp at end
            stop_timestamp = datetime.now()
            # add to metadata dataframe
            self.computed_df.loc[len(self.computed_df)] = [lx, ly, res, d, 
                theta, phi, str(self.add_noise), start_timestamp, stop_timestamp,
                filename + '.pkl', self.main_folder+sub_folder]
            # Save metadata dataframe at each step
            metadata_file_folder = self.main_folder + self.computed_df_fname
            meta_df.to_csv(metadata_file_folder, index=False)
            #update bar
            bar.update(1)      
        bar.close()
        
# =============================================================================
#         for lx in Lx:
#             for ly in Ly: #reversed(Ly)
#                 # base filename and field
#                 base_filename_load = 'bemfbase_Lx'+ str(int(100*lx)) +\
#                     'cm_Ly' + str(int(100*ly))+ 'cm'
#                 base_field = BEMFlushSq()
#                 base_field.load(path = self.base_folder, filename = base_filename_load)
#                 for res in resist:
#                     for d in d1:
#                         for jang in np.arange(self.theta_phi.shape[0]):
#                             # file name to save
#                             filename = base_filename + '_Lx' + str(int(100*lx)) +\
#                             'cm_Ly' + str(int(100*ly)) + 'cm_res' + str(int(res))+\
#                             '_d' + str(int(1000*d)) + 'mm_el' +\
#                             str(int(np.rad2deg(self.theta_phi[jang, 0]))) +\
#                             '_az' + str(int(np.rad2deg(self.theta_phi[jang, 1])))
#                             # Figure time stamp at begining
#                             start_timestamp = datetime.now()
# 
#                             ############ Run BEM Field #########################
#                             # 1 - figure if file does not exists - run simu
#                             sub_folder = 'res' + str(int(res)) +'/d' + str(int(1000*d)) + 'mm/'
#                             field = self.run_bemflush_rect_field(air, controls, 
#                                 receivers, base_field, r, resist = res, d = d, 
#                                 theta = self.theta_phi[jang, 0], phi = theta_phi[jang, 1], 
#                                 snr = snr, n_gauss = n_gauss)
#                             field.save(path = self.main_folder + sub_folder, filename = filename)
#                             # Figure time stamp at end
#                             stop_timestamp = datetime.now()
#                             # add to metadata dataframe
#                             self.computed_df.loc[len(self.computed_df)] = [lx, ly, res, d, 
#                                 np.rad2deg(theta_phi[jang, 0]), np.rad2deg(theta_phi[jang, 1]),
#                                 str(self.add_noise), start_timestamp, stop_timestamp,
#                                 filename + '.pkl', main_folder+sub_folder]
#                             # Save metadata dataframe at each step
#                             metadata_file_folder = main_folder + metadata_file
#                             meta_df.to_csv(metadata_file_folder, index=False)
#                             #update bar
#                             bar.update(1)      
#         bar.close()
# =============================================================================

    def save(self, filename = 'controldataset', path = ''):
        """ To save the control object as pickle

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

    def load(self, filename = 'controldataset', path = ''):
        """ Load a control object.

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

        

        
