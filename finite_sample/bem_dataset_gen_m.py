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
import pickle
import os.path, time
from os import path
import warnings

from tqdm.notebook import trange, tqdm
from controlsair import AirProperties, AlgControls
from material import PorousAbsorber
from sources import Source
from receivers import Receiver
from field_bemflush import BEMFlushSq

class GenDataSet():
    """Control dataset generation for finite samples.
    """
    def __init__(self, main_folder = '/main_folder/', name = 'dataset', 
        base_folder = '/base_folder/', base_name = 'base_dataset'):
        """Instantiate the organizing class with a base folder and a dataset name

        Parameters
        ----------
        main_folder : str
            string with the name of the main folder
        name : str
            string with the name of the dataset
        base_folder : str
            string with the name of the base folder (load and do stuff)
        base_name : str
            string with the name of the base dataset      
        """
        try:
            self.load()
            print("Dataset already exists. I loaded it!")
        except:
            self.meta_data_names =  ['Computed? (bool)', 'Start timestamp', 
                'End timestamp', 'File name', 'Folder']
            self.main_folder = main_folder
            self.name = name
            self.base_folder = base_folder
            self.base_name = base_name
            self.locked_status = False
    
    def save(self,):
        """To save the control object as pickle
        """
        f = open(self.main_folder + self.name + '.pkl', 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self,):
        """ Load a control object.
        """
        f = open(self.main_folder + self.name + '.pkl', 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def lock_object(self,):
        """Lock certain methods of the object

        Set the status of self.locked_status to True. This is used to condition the
        execution of sensitive operations, such as generating a new test matrix and
        folder structure. This avoids deleting data unintentionally.
        """
        self.locked_status = True
    
    def get_basedataset_specs(self, ):
        base_dataset = GenDataSet(main_folder = self.base_folder, 
            name = self.base_name)
        base_dataset.load()
        self.base_length = len(base_dataset.df)
        self.base_df = base_dataset.df

    def gen_columns_testmatrix(self, list_of_columns = []):
        """Generate the columns of your test matrix

        It also creates the empty data frame with the specified columns and extra
        info and save it as a csv file in the main_folder. A column called
        'computed' was assess wether the case in the row is computed or not.

        Parameters
        ----------
        list_of_columns : list
            a list containing strings with the names of each column
        """
        if self.locked_status:
            warnings.warn("The Object is locked. You can not create new columns in a test matrix.")
        else:
            list_of_columns = list_of_columns + self.meta_data_names
            self.list_of_columns = list_of_columns
            self.df = pd.DataFrame(columns = self.list_of_columns)
            self.df.to_csv(self.main_folder + self.name + '.csv', index=False)

    def generate_test_mtx(self, *args, force_restart_csv = False):
        """Generate a test matrix with randomized inputs
        
        Pass inputs as vectors and we take care of the rest. The vectors themselves
        form the test matrix.
        """
        if self.locked_status:
            warnings.warn("The Object is locked. You can not create new data in a test matrix.")
        else:
            # Initialize the row counter
            self.which_row = 0
            # column names in the dataframe and lenght
            self.n_samples_dataset = len(args[0])
            cols_df = self.df.columns
            # looping through *args
            for jcol, col in enumerate(args):
                self.df[cols_df[jcol]]=pd.Series(args[jcol])
            # initialize metadata
            self.df[self.meta_data_names[0]] = np.repeat(False, self.n_samples_dataset)
            for jm in np.arange(1, len(self.meta_data_names)):
                self.df[self.meta_data_names[jm]] = np.repeat(None, self.n_samples_dataset)
            # save the csv
            log_exists = path.isfile(self.main_folder + self.name + '.csv')
            if (force_restart_csv) or (log_exists == False):
                self.df.to_csv(self.main_folder + self.name + '.csv', index=False)

    def generate_subfolders(self, n_files_folder = 5000):
        """Generate a number of subfolders with names 'fX'

        X is the subfolder number. The function intend to avoid a large number of
        files in a single folder, which creates difficulties when one need to locate
        the files.

        Parameters
        ----------
        n_files_folder : int
            target number of files in a given sub-folder. If the total number of simulations
            is smaller than n_files_folder, then only f0 will be created.
        """
        if self.locked_status:
            warnings.warn("The Object is locked. You can not create new folders in the dataset structure. This avoids deletion.")
        else:
            self.n_files_folder = n_files_folder
            self.folder_names = []
            if self.n_samples_dataset <= self.n_files_folder:
                self.n_folders = 1
                path = os.path.join(self.main_folder, 'f0')
                self.folder_names.append('f0')
                if os.path.isdir(path) == False:
                    os.makedirs(path)
            else:
                # determine how many folders
                if self.n_samples_dataset % self.n_files_folder == 0:
                    self.n_folders = int(self.n_samples_dataset / self.n_files_folder)
                else:
                    self.n_folders = int(self.n_samples_dataset / self.n_files_folder) + 1
                for jf in np.arange(self.n_folders):
                    sub_name = 'f' + str(int(jf))
                    self.folder_names.append(sub_name)
                    path = os.path.join(self.main_folder, sub_name)
                    if os.path.isdir(path) == False:
                        os.makedirs(path)
    
    def get_path_file_names(self,):
        """Get the path where to save a file and its name when running dataset generation
        """
        # Choose where to save the data
        folder_number = int(self.which_row/self.n_files_folder)
        subfolder_name = f'{self.folder_names[folder_number]}/'
        path = self.main_folder + subfolder_name
        file_name = f'bemf_{self.name}_{self.which_row}'
        return path, file_name

    def update_organizer(self, start_timestamp, stop_timestamp, file_name, path, cols):
        """ Updates the organizer object

        Updates the data frame metadata and saves the object and csv file

        Parameters
        ----------
        start_timestamp : str
            The initial time stamp of simulation
        stop_timestamp : str
            The final time stamp of simulation
        file_name : str
            Name of the field file
        path : str
            Path to the field file
        """
        cols_df = self.df.columns
        self.df.loc[self.which_row, [cols_df[cols[0]], cols_df[cols[1]], cols_df[cols[2]], cols_df[cols[3]], cols_df[cols[4]]]] = [True, start_timestamp, stop_timestamp, file_name, path]
        # Save object organizer and update csv file
        self.df.to_csv(self.main_folder + self.name + '.csv', index=False)
        self.save()

    def check_files(self,):
        """Check if the files in database exists and return a list of non-existing
        """
        # Initialize main bar and list
        self.list_of_failed_rows = []
        bar = tqdm(total = self.n_samples_dataset, desc = 'Checking files')
        for jf in np.arange(self.n_samples_dataset):
            # get folder and filename
            folder = self.df['Folder'][jf]
            filename = self.df['File name'][jf]
            # check for file existence
            file_exists =  path.isfile(folder + filename + '.pkl')
            # check file size
            if file_exists:
                file_size = path.getsize(folder + filename + '.pkl')
            # check if file is a bad one and update a list of rows with missing files
            if not file_exists or file_size == 0:
                self.list_of_failed_rows.append(jf)
            self.save()
            bar.update(1)        
        bar.close()
    
    def get_simu_time(self,):
        """Computes the mean simulation time
        """
        sum_time = (self.df['End timestamp']-self.df['Start timestamp']).sum()
        mean_datetime = (self.df['End timestamp']-self.df['Start timestamp']).mean()
        print(f"Mean simulation time is {mean_datetime.total_seconds()} [s]")
        print(f"Total simulation time is {sum_time}")

class GenDataSetBEMflushSq(GenDataSet):
    """Control dataset generation for finite samples.

    Inherites the organization capacities of GenDataSet and adds running capacities 
    of BEMflushSq.
    """

    def __init__(self, main_folder = '/main_folder/', name = 'dataset', 
        base_folder = '/base_folder/', base_name = 'base_dataset'):
        """

        Parameters
        ----------
        p_mtx : (N_rec x N_freq) numpy array
            A matrix containing the complex amplitudes of all the receivers
            Each column is a set of sound pressure at all receivers for a frequency.
        controls : object (AlgControls)
            Controls of the decomposition (frequency spam)
        material : object (PorousAbsorber)
            Contains the material properties (surface impedance).
        receivers : object (Receiver)
            The receivers in the field

        The objects are stored as attributes in the class (easier to retrieve).
        """
        GenDataSet.__init__(self, main_folder, name, base_folder, base_name)
        super().__init__(main_folder, name, base_folder, base_name)

    def config_sim(self, air, controls, receivers = [], base_folder = None, 
        n_gauss = 36, Nel_per_wavelenth = 6):
        """Configures the common things accross all simulations

        Parameters
        ----------
        air : obj
            Air properties.
        controls : obj
            Frequency span.
        controls : obj
            receivers in the array.
        base_folder: str or None
            Path to a base folder (where you store BEM objs with G matrix only)
        n_gauss : int
            Number of gauss points for integration
        Nel_per_wavelenth : int
            Number of elements per wavelength (for mesh generation)
        """
        if self.locked_status:
            warnings.warn("The Object is locked. You can not configure a new dataset simulation.")
        else:
            self.air = air
            self. controls = controls
            self.receivers = receivers
            #self.base_folder = base_folder
            self.n_gauss = n_gauss
            self.Nel_per_wavelenth = Nel_per_wavelenth

    def run_bemfsq(self, Lx, Ly, material, source, add_noise = False, 
        snr = 1000, from_base = False):
        """Run one BEM flush (squared elements) simulation

        Parameters
        ----------
        Lx : float
            lenght of sample
        Ly : float
            width of sample
        material : obj
            Material properties.
        source : obj
            sound source coord.
        add_noise : bool
            choose if you want to add noise or not
        snr : float
            value of Signal to Noise ratio
        from_base : bool
            choose if you want to run from base or not
        """
        # Instantiate Field object from base field
        field = BEMFlushSq(air = self.air, controls = self.controls, 
            material = material, sources = source, receivers = self.receivers, 
            n_gauss = self.n_gauss, bar_mode = 'notebook')
        field.generate_mesh(Lx = Lx, Ly = Ly, 
            Nel_per_wavelenth = self.Nel_per_wavelenth)
        field.assemble_gij(bar_leave = False)
        field.psurf2(erase_gij = True, bar_leave = False)
        # Evaluate pressure at all field points
        field.p_fps(bar_leave = False)
        if add_noise:
            field.add_noise(snr = snr)
        return field
    
    def run_base_case(self, Lx, Ly):
        """Run base case of BEM flush (squared elements) simulation

        Parameters
        ----------
        Lx : float
            lenght of sample
        Ly : float
            width of sample
        """
        # Instantiate Field object base field
        field = BEMFlushSq(air = self.air, controls = self.controls, 
            n_gauss = self.n_gauss, bar_mode = 'notebook')
        field.generate_mesh(Lx = Lx, Ly = Ly, Nel_per_wavelenth = self.Nel_per_wavelenth)
        field.assemble_gij(bar_leave = False)
        return field
    
    def run_from_base(self, base_field, material, source, 
        add_noise = False, snr = 1000):
        """Run a case of BEM flush (squared elements) simulation from base

        Parameters
        ----------
        base_field : obj
            base field object
        material : obj
            Material properties.
        source : obj
            sound source coord.
        add_noise : bool
            choose if you want to add noise or not
        snr : float
            value of Signal to Noise ratio
        """
        # Instantiate Field object from base field
        field = BEMFlushSq(air = self.air, controls = self.controls, 
            material = material, sources = source, receivers = self.receivers, 
            n_gauss = self.n_gauss, bar_mode = 'notebook')
        field.parse_mesh(base_field)
        field.gij_f = base_field.gij_f
        # Calculate the surface pressure
        field.psurf2(erase_gij = True, bar_leave = False)
        # Evaluate pressure at all field points
        field.p_fps(bar_leave = False)
        if add_noise:
            field.add_noise(snr = snr)
        return field
        
    def get_sample_size(self,):
        """Gets the sample length and width
        """
        ### Sample size
        Lx = self.df[self.df.columns[0]][self.which_row]
        Ly = self.df[self.df.columns[1]][self.which_row]
        return Lx, Ly
    
    def get_source(self, ):
        """Gets the sound source object
        """
        # Source
        r = self.df[self.df.columns[4]][self.which_row]
        theta = np.deg2rad(self.df[self.df.columns[5]][self.which_row])
        phi= np.deg2rad(self.df[self.df.columns[6]][self.which_row])
        s_coord = sph2cart(r, np.pi/2-theta, phi)
        source = Source(coord = s_coord)
        return source

    def get_material(self,):
        """Gets the material object
        """
        theta = np.deg2rad(self.df[self.df.columns[5]][self.which_row])
        resistivity = self.df[self.df.columns[2]][self.which_row]
        thickness = self.df[self.df.columns[3]][self.which_row]
        material = PorousAbsorber(air = self.air, controls = self.controls)
        material.miki(resistivity = resistivity)
        material.layer_over_rigid(thickness = thickness, theta = theta)
        return material

    def get_noise(self,):
        """Gets the noise conditions
        """
        add_noise = self.df[self.df.columns[7]][self.which_row]
        snr = self.df[self.df.columns[8]][self.which_row]
        return add_noise, snr

    def gen_dataset(self,):
        """Controls the dataset generation
        """
        # Initialize main bar
        bar = tqdm(total = self.n_samples_dataset, desc = 'Generating database', 
            initial = self.which_row + 1)
        # Main for loop
        while self.which_row < self.n_samples_dataset:
            # # Get simulation parameters from dataframe
            cols_df = self.df.columns
            # ### Sample size
            Lx, Ly = self.get_sample_size()
            # Source
            source = self.get_source()
            ### Material
            material = self.get_material()
            ### Noise
            add_noise, snr = self.get_noise()
            # run single simulation
            start_timestamp = datetime.now()
            field = self.run_bemfsq(Lx, Ly, material, source, add_noise = add_noise, 
                snr = snr, from_base = False)
            stop_timestamp = datetime.now()
            # Choose where to save the data
            path, file_name = self.get_path_file_names()
            # save field
            field.save(path = path, filename = file_name)
            # Update dataframe and csv
            self.update_organizer(start_timestamp, stop_timestamp, file_name, 
                path, cols = [9, 10, 11, 12, 13])
            # Increment row
            self.which_row += 1
            bar.update(1)        
        bar.close()

    def gen_base_dataset(self,):
        """Controls the dataset generation (base sample sizes)
        """
        # Initialize main bar
        bar = tqdm(total = self.n_samples_dataset, desc = 'Generating base sample database', 
            initial = self.which_row + 1)
        # Main for loop
        while self.which_row < self.n_samples_dataset:
            # # Get simulation parameters from dataframe
            cols_df = self.df.columns
            # ### Sample size
            Lx, Ly = self.get_sample_size()
            # run single simulation
            start_timestamp = datetime.now()
            field = self.run_base_case(Lx, Ly)
            stop_timestamp = datetime.now()
            # Choose where to save the data
            path, file_name = self.get_path_file_names()
            # save field
            field.save(path = path, filename = file_name)
            # Update dataframe and csv
            self.update_organizer(start_timestamp, stop_timestamp, file_name, 
                path, cols = [2, 3, 4, 5, 6])
            # Increment row
            self.which_row += 1
            bar.update(1)        
        bar.close()
    
    def gen_dataset_frombase(self,):
        """Controls the dataset generation from base files

        Repeating the sample sizes is necessary
        """
        self.get_basedataset_specs()
        # number of repetitions
        n_repeats = int(self.n_samples_dataset/self.base_length)
        # Initialize main bar
        bar = tqdm(total = self.n_samples_dataset, desc = 'Generating database from base', 
            initial = self.which_row + 1)
        # Main for loop
        while self.which_row < self.n_samples_dataset:
            # # Get simulation parameters from dataframe
            cols_df = self.df.columns
            # ### Sample size
            Lx, Ly = self.get_sample_size()
            # Source
            source = self.get_source()
            ### Material
            material = self.get_material()
            ### Noise
            add_noise, snr = self.get_noise()
            ### Get the base BEM file
            row_from_base = int(self.which_row / n_repeats)
            base_folder = self.base_df['Folder'][row_from_base]
            base_filename = self.base_df['File name'][row_from_base]
            # run single simulation
            start_timestamp = datetime.now()
            base_field = BEMFlushSq()
            base_field.load(path = base_folder, filename = base_filename)
            field = self.run_from_base(base_field, material, source, 
                add_noise = add_noise, snr = snr)
            stop_timestamp = datetime.now()
            # # Choose where to save the data
            path, file_name = self.get_path_file_names()
            # # save field
            field.save(path = path, filename = file_name)
            # Update dataframe and csv
            self.update_organizer(start_timestamp, stop_timestamp, file_name, 
                path, cols = [9, 10, 11, 12, 13])
            # # Increment row
            self.which_row += 1
            bar.update(1)        
        bar.close()
    
    def load_field(self, row_number):
        """Load and return a field based on the row number given

        Parameters
        ----------
        base_field : obj
            base field object
        material : obj
            Material properties.
        source : obj
            sound source coord.
        
        Returns
        ----------
        field : obj
            field object
        """
        folder = self.df['Folder'][row_number]
        filename = self.df['File name'][row_number]
        field = BEMFlushSq(bar_mode = 'notebook')
        field.load(path = folder, filename = filename)
        return field