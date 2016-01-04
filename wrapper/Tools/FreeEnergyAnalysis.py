
####################################################################################################
#                                                                                                  #
#   Script for the analysis of Free energy simulations using MBAR                                  #
#                                                                                                  #
#   author: Antonia Mey <antonia.mey@ed.ac.uk>                                                     #
#                                                                                                  #
####################################################################################################

from Sire import try_import
from Sire import try_import_from
from Sire.Units import *
import os
import sys

np = try_import("numpy")
MBAR = try_import_from("pymbar", "MBAR")
timeseries = try_import_from("pymbar", "timeseries")

class FreeEnergies(object):
    r"""This class contains all the different pmf information
    The constructor expects subsampled MBAR and TI compatible data.
    Parameters
    ----------

    u_kln : ndarray(shape=(therm_states, therm_states, nsamples), dtype=float)
        reduced perturbed energies used for MBAR estimates
    N_K : ndarray(shape=(therm_states), dtype=int)
        number of samples per thermodynamic state
    lambda_array : ndarray(shape=(therm_states), dtype=float)
        lambda thermodynamic values
    gradients_kn : ndarray(shape=(therm_state, nsamples), dtype=float)
        reduced gradients
    """

    def __init__(self, g_temp = None, u_kln=None, N_k=None, lambda_array=None, gradients_kn=None):
        r"""The data passed here is already subsampled"""

        self.T = g_temp
        self._u_kln = np.array(u_kln)
        self._N_k = N_k
        self._lambda_array = lambda_array
        self._gradients_kn = gradients_kn

        #initialise results containers
        self._deltaF_mbar = None
        self._deltaF_ti = None
        self._dDeltaF_mbar = None
        self._f_k = None
        self._pmf_ti = None


    def run_ti(self, cubic_spline=False):
        r"""Runs Thermodynamic integration free energy estimate
        Parameters
        ----------

        cubic_spline : bool
            Use cubic spline estimation instead of trapezium rule.
        """

        if cubic_spline:
            NotImplementedError("Cubic Spline TI has not been implemented yet")
        else:
            means = np.mean(self._gradients_kn, axis=1)
            self._pmf_ti = np.zeros(shape=(self._lambda_array.shape[0], 2))
            self._pmf_ti[:, 0] = self._lambda_array
            for i in range(1, self._lambda_array.shape[0]):
                self._pmf_ti[i-1][1] = np.trapz(means[0:i], self._lambda_array[0:i])
            self._pmf_ti[-1][1] = np.trapz(means, self._lambda_array)
            self._deltaF_ti = np.trapz(means, self._lambda_array)


    def run_mbar(self):
        r"""Runs MBAR free energy estimate """
        MBAR_obj = self.mute()(MBAR(self._u_kln, self._N_k, verbose=True))()
        if self.T is not None:
            self._f_k = MBAR_obj.f_k*self.T*k_boltz
        else:
            self._f_k = MBAR_obj.f_k
            print ("#Warning!, Simulation temperature is None, all results are given in reduced units!")
        (deltaF_ij, dDeltaF_ij, theta_ij) = MBAR_obj.getFreeEnergyDifferences()
        self._deltaF_mbar = deltaF_ij[0, self._lambda_array.shape[0]-1]
        self._dDeltaF_mbar = dDeltaF_ij[0, self._lambda_array.shape[0]-1]
        self._pmf_mbar = np.zeros(shape=(self._lambda_array.shape[0], 2))
        self._pmf_mbar[:, 0] = self._lambda_array
        self._pmf_mbar[:, 1] = self._f_k
        self._error_pmf_mbar = np.zeros(shape=(self._lambda_array.shape[0]))
        if self.T is not None:
            self._error_pmf_mbar = dDeltaF_ij[0,:]*self.T*k_boltz
        else:
            self._error_pmf_mbar = dDeltaF_ij[0,:]

    @property
    def pmf_ti(self):
        return self._pmf_ti

    @property
    def error_pmf_mbar(self):
        return self._error_pmf_mbar

    @property
    def pmf_mbar(self):
        return self._pmf_mbar

    @property
    def deltaF_ti(self):
        return self._deltaF_ti

    @property
    def deltaF_mbar(self):
        return self._deltaF_mbar

    @property
    def errorF_mbar(self):
        return self._dDeltaF_mbar

class SubSample(object):
    r"""This class subsamples data based on the timeseries analysis or percentage of data ready for pmf use
    Parameters
    ----------
    gradients_kn : ndarray(shape=(therm_state, nsamples), dtype=float)
        reduced gradients
    energies : ndarray(shape=(therm_state, nsamples), trype=float)
        potential energies used to find statisitical inefficiency
    u_kln : ndarray(shape=(therm_states, therm_states, nsamples), dtype=float)
        reduced perturbed energies used for MBAR estimates
    N_K : ndarray(shape=(therm_states), dtype=int)
        number of samples per thermodynamic state
    lambda_array : ndarray(shape=(therm_states), dtype=float)
        lambda thermodynamic values
    percentage : int [0,100]
        percentage of the data that should be retained from the simulation
    subsample : string
        string idenfier for subsampling method, default='timeseries' from timeseries module in MBAR
    """

    def __init__(self, gradients_kn, energies, u_kln, N_k, percentage=100, subsample='timeseries'):
        self._gradients_kn = gradients_kn
        self._N_k = N_k
        self._energies_kn = energies
        self._u_kln = u_kln
        self._subsampled_u_kln = None
        self._subsampled_N_k_energies = None
        self._subsampled_N_k_gradients = None
        self._subsampled_grad_kn = None

        if u_kln is not None:
            if N_k.shape[0]!=u_kln.shape[0]:
                RuntimeError("The number of thermodynamic states must be the same in u_kln and N_k!"
                         "u_kln has size %d and N_k has size %d" %(u_kln.shape[0], N_k.shape[0]))
        self.subsample = subsample
        self.percentage = percentage
        if percentage <0.0:
            RuntimeError("You must provide a percentage between 0 and 100")
        elif percentage>100.0:
            RuntimeError("You must provide a percentage between 0 and 100")


    def subsample_gradients(self):
        if self.subsample == False:
            print("# We are only eliminating samples from the beginning of the data and are still working with highly"
                  " correlated data!")
            if self.percentage == 0:
                RuntimeWarning("You are not subsampling your data according to the statistical inefficiency nor are"
                               "you discarding initial data. Your are trying to remove 100% of the data. "
                               "Please set percentage to another value than 0!")
                sys.exit(-1)
            percentage_removal = self._N_k*(1-self.percentage/100.0)
            self._subsampled_N_k_gradients = self._N_k-percentage_removal
            N_max = np.max(self._subsampled_N_k_gradients)
            self._subsampled_grad_kn = np.zeros(shape=(self._N_k.shape[0], N_max))
            for p in range(percentage_removal.shape[0]):
                self._subsampled_grad_kn[p,:] = self._gradients_kn[p,percentage_removal[p]:]
            if N_max <=100:
                RuntimeWarning("You have reduced your data to less than 100 samples, the results from these might not "
                               "be trustworthy. ")
        else:
            print("# We are doing a timeseries analysis using the timeseries analysis module in pymbar and will subsample"
                  " gradients according to that.")
            #first we compute statistical inefficiency
            g_k = np.zeros(shape=(self._gradients_kn.shape[0]))
            self._subsampled_N_k_gradients = np.zeros(shape=(self._gradients_kn.shape[0]))
            for i in range(g_k.shape[0]):
                g_k[i] = timeseries.statisticalInefficiency(self._gradients_kn[i,:])
            g = np.max(g_k)
            #now we need to figure out what the indices in the data are for subsampling
            indices_k = []
            for i in range(g_k.shape[0]):
                indices_k.append(timeseries.subsampleCorrelatedData(self._gradients_kn[i,:], g=g))
                self._subsampled_N_k_gradients[i]=len(indices_k[i])
            N_max = np.max(self._subsampled_N_k_gradients)
            if N_max <=100:
                RuntimeWarning("You have reduced your data to less than 100 samples, the results from these might not "
                               "be trustworthy. ")
            self._subsampled_grad_kn = np.zeros([self._gradients_kn.shape[0], N_max], np.float64)
            for k in range(self._gradients_kn.shape[0]):
                self._subsampled_grad_kn[k, :] = self._gradients_kn[k, indices_k[k]]

    def subsample_energies(self):
        if self.subsample == False:
            print("# We are only eliminating samples from the beginning of the data and are still working with highly"
                  " correlated data!")

            if self.percentage == 0:
                RuntimeWarning("You are not subsampling your data according to the statistical inefficiency nor are"
                               "you discarding initial data. Your are trying to remove 100% of the data. "
                               "Please set percentage to another value than 0!")
                sys.exit(-1)
            percentage_removal = self._N_k*(1-self.percentage/100.0)
            self._subsampled_N_k_energies = self._N_k-percentage_removal
            N_max = np.max(self._subsampled_N_k_energies)
            self._subsampled_u_kln = np.zeros(shape=(self._N_k.shape[0], self._N_k.shape[0], N_max))
            for i in range(percentage_removal.shape[0]):
                for j in range(percentage_removal.shape[0]):
                    self._subsampled_u_kln[i,j,:] = self._u_kln[i,j,percentage_removal[j]:]
            if N_max <=100:
                RuntimeWarning("You have reduced your data to less than 100 samples, the results from these might not "
                               "be trustworthy. ")
        else:
            print("# We are doing a timeseries analysis using the timeseries analysis module in pymbar and will subsample"
                  " energies according to that.")

            #first we compute statistical inefficiency
            g_k = np.zeros(shape=(self._energies_kn.shape[0]))
            for i in range(g_k.shape[0]):
                g_k[i] = timeseries.statisticalInefficiency(self._energies_kn[i,:])
            g = np.max(g_k)
            #now we need to figure out what the indices in the data are for subsampling
            indices_k = []
            self._subsampled_N_k_energies = np.zeros(shape=(self._gradients_kn.shape[0]))
            for i in range(g_k.shape[0]):
                indices_k.append(timeseries.subsampleCorrelatedData(self._energies_kn[i,:], g=g))
                self._subsampled_N_k_energies[i]=len(indices_k[i])
            #self._subsampled_N_k_energies = (np.ceil(self._N_k / g)).astype(int)
            N_max = np.max(self._subsampled_N_k_energies)
            if N_max <=100:
                RuntimeWarning("You have reduced your data to less than 100 samples, the results from these might not "
                               "be trustworthy. ")
            self._subsampled_u_kln = np.zeros([self._gradients_kn.shape[0],self._gradients_kn.shape[0], N_max], np.float64)
            for k in range(self._gradients_kn.shape[0]):
                self._subsampled_u_kln[k,:,:] = self._u_kln[k,:,indices_k[k]].transpose()
            self._subsampled_u_kln = np.array(self._subsampled_u_kln)

    @property
    def u_kln(self):
        if self._subsampled_u_kln is None:
            self.subsample_energies()
        return self._subsampled_u_kln
    @property
    def gradients_kn(self):
        if self._subsampled_grad_kn is None:
            self.subsample_gradients()
        return  self._subsampled_grad_kn
    @property
    def N_k_energies(self):
        return self._subsampled_N_k_energies
    @property
    def N_k_gradients(self):
        return self._subsampled_N_k_gradients

class SimfileParser(object):
    r""" This class will parse sim_files.dat in different lambda directories for analysis
    """
    def __init__(self, sim_files, lam, T):
        #We will load all the data now
        self._data = []
        self.sim_files = sim_files
        self.lam = lam
        self.T = T
        self._u_kln = None
        self._N_k = None
        self._grad_kn = None
        self._energies_kn = None
        self._max_l = 0

    def load_data(self):
        r"""Loads all the simfile.dat files supplied and does some sanity checks on them"""
        num_inputfiles = len(self.sim_files)
        g_temp = None #generating temeratures of the input files
        lam_array = None #lambda arrays from input files
        g_lam = None #Gnerating lambdas from all input files
        g_lam_list = []
        #Lambda sanity checks
        if self.lam is not None:
            if num_inputfiles != lam.shape[0]:
                raise Exception("The lambda array you supplied does not have the same length as the number of input files")
                sys.exit(-1)
        #sanity checking for file existance
        for i in range(num_inputfiles):
            #check if filesize is not zero:
            if not os.path.exists(self.sim_files[i]):
                raise IOError("supllied simulation file %s does not exist" %self.sim_files[i])
            if os.stat(self.sim_files[i]).st_size == 0:
                raise IOError("suplied simulation file %s does not contain any data" %self.sim_files[i])
            content = None
            with open(self.sim_files[i]) as f:
                content = f.readlines(2000)
            if i == 0:
                lam_array, g_lam, g_temp = self.analyse_headers(content)
                if self.lam is not None:
                    if not np.array_equal(lam_array, self.lam):
                        raise Exception("Alchemical array provided via the command line does not match the array found in %s" %self.sim_files[0])
                if lam_array is None:
                    print("# It seems that no lambda array was given as input for the simulation, no MBAR analysis will be possible.")
                g_lam_list.append(float(g_lam))
            else:
                la, gl, gt = self.analyse_headers(content)
                if not np.array_equal(la, lam_array):
                    raise Exception("Alchemical lambda array provided in %s does not match the array in %s" %(self.sim_files[i], self.sim_files[0]))
                    sys.exit(-1)
                if lam_array is not None:
                    if gl not in lam_array:
                        raise Exception("Generating lambda %s is not part of the alchemical array provided in %s" %(gl, self.sim_files[0]))
                        sys.exit(-1)
                if gt != g_temp:
                    raise Exception("Generating temperature %s does not match the generating temperature provided in %s" %(gt, self.sim_files[0]))
                    sys.exit(-1)
                if gt is not None and self.T != gt:
                    print ("#temperature given with a commandline argument is:\t %f" %self.T)
                    print ("#temperature in simfile is:\t\t\t\t %f" %gt)
                    raise Exception("The temperatures do not match!")
                    sys.exit(-1)
                #if everything is ok record the generating lambda. 
                g_lam_list.append(float(gl))
            #now we are convinced that the provided data files are sane, let's read the actual data
            print ("# Reading simulation file: %s" %self.sim_files[i])
            self._data.append(np.loadtxt(self.sim_files[i]))
            if lam_array is None:
                self.lam = np.array(g_lam_list)
            else:
                self.lam = lam_array


        #loading data into arrays for further processing
        #N_k is the number of samples at generating thermodynamic state (lambda) k
        self._N_k = np.zeros(self.lam.shape[0])
        for i in range(num_inputfiles):
            d = self._data[i]
            self._N_k[i] = d.shape[0]
            if self._max_l < d.shape[0]:
                self._max_l = d.shape[0]

    def analyse_headers(self, content):
        r"""reads information from comment lines in simfile.dat"""
        lam_array = None
        g_lam = None
        g_temp = None
        for i in range(12):
            l = content[i]
            if '#Generating lambda' in l:
                l = l.split()
                g_lam = float(l[-1])
            if '#Generating temperature' in l:
                l = l.split()
                if l[-1] =='SireUnits::Celsius':
                    g_temp = None
                else:
                    g_temp = l[-2]+'*'+l[-1]
                    if l[-1] == "C":
                        g_temp = float(l[-2])*celsius
                    elif l[-1] == "F":
                        g_temp = float(l[-2])*fahrenheit
                    else:
                        g_temp = float(l[-2])*kelvin
                    g_temp = g_temp.value()


            if '#Alchemical ' in l:
                l = l.split()
                lam_array = l[3:]
                round_bracket = False
                for s in filter (lambda x: '(' in x, l): round_bracket = True
                if len(lam_array) < 2:
                    lam_array = None
                else:
                    if round_bracket:
                        lam_array[0]=lam_array[0].split('(')[-1]
                        lam_array[-1]=lam_array[-1].split(')')[0]
                    else:
                        lam_array[0]=lam_array[0].split('[')[-1]
                        lam_array[-1]=lam_array[-1].split(']')[0]
                    lam_array = " ".join(lam_array)
                    lam_array = np.array(lam_array.split(',')).astype(float)
        return lam_array, g_lam, g_temp

    def _populate_e_kn(self):
        self._energies_kn = np.zeros(shape=(self.lam.shape[0], self._max_l))
        for i in range(len(self._data)):
            d = self._data[i]
            self._energies_kn[i][:self._N_k[i]] = d[:,1]

    def _populate_u_kln(self):
        if self._data[0].shape[1] > 5:
            self._u_kln = np.zeros(shape=(self.lam.shape[0], self.lam.shape[0], self._max_l))
            for i in range(len(self._data)):
                d = self._data[i]
                self._u_kln[i][:self._N_k[i]][:self._N_k[i]] = d[:,5:].transpose()
        else: 
            self._ukln = None

    def _populate_g_kn(self):
        self._grad_kn = np.zeros(shape=(self.lam.shape[0], self._max_l))
        for i in range(len(self._data)):
            d = self._data[i]
            self._grad_kn[i][:self._N_k[i]] = d[:,2]


    @property
    def u_kln(self):
        if self._u_kln is None:
            self._populate_u_kln()
        return self._u_kln

    @property
    def N_k(self):
        return self._N_k

    @property
    def energies_kn(self):
        if self._energies_kn is None:
            self._populate_e_kn()
        return self._energies_kn

    @property
    def grad_kn(self):
        if self._grad_kn is None:
            self._populate_g_kn()
        return self._grad_kn
