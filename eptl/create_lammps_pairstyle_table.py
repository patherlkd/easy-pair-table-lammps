""" Module of functions to create tables, run LAMMPS test, and generate comparisons.
"""

## Written by Dr. Luke Davis: UCL Department of Mathematics 2023 luke.davis@ucl.ac.uk
## Added class LmpExec, refactored functions to follow PEP8 standards Jennifer Clark 02/08/2024

import os
import math
import warnings
warnings.simplefilter('ignore')
from inspect import getmembers, isfunction

import autograd.numpy as np # numerical python library
from functools import partial # to pass functions into functions
import matplotlib
import matplotlib.pyplot as plt # matplotlib shorthand
matplotlib.use('pdf') # pdf backend

import eptl
MY_VERSION = eptl.__version__
import eptl.potentials_for_lammps as potentials_for_lammps
from .potentials_for_lammps import force # pair potentials file 

from pkg_resources import resource_filename
lmps_pair_write_generic_filename = resource_filename('eptl', 'resources/in.pair_write_generic')
_init_filename = resource_filename('eptl', 'resources/lmp_executable')

pair_styles = dict(getmembers( potentials_for_lammps, isfunction))
del pair_styles["force"]
del pair_styles["grad"]

class LmpExec:

    _exec_path = open(_init_filename, "r").readlines()[0]
    
    def __init__(self, exec_path=None):
        self.set_exec(exec_path)
    
    def change_exec(self, exec_path):
        if exec_path is not None and not os.path.isfile(exec_path):
            raise ValueError(f"LAMMPS executable could not be found: {exec_path}")     
        
        file = open(_init_filename, "w")
        file.write(exec_path)
        file.close()
        
    def get_exec(self):
        if self._exec_path == "EXECPATHHERE":
            raise ValueError(
                "LAMMPS Executable must be set: python -c 'from eptl.create_lammps_pairstyle_table import LmpExec; " \
                "Lmp = LmpExec(); Lmp.change_exec(\"path/to/executable/lmp\")'"
            )
        self._executable = self._exec_path
            
    def set_exec(self, exec_path):
        if exec_path is None:
            self._executable = None
        elif not os.path.isfile(exec_path):
            raise ValueError(f"LAMMPS executable could not be found: {exec_path}")
        else:
            self._executable = exec_path
            if self._exec_path == "EXECPATHHERE":
                self.change_exec(exec_path)

    def executable(self):
        if self._executable is None:
            self.get_exec()
        return self._executable
    
    def __str__(self):
        return f"LAMMPS executable: {self._executable}"

Lmp = LmpExec()    

def set_plot_rcparams(ff=3, fig_width_pt=246., inches_per_pt=0.014, golden_mean=0.66, rcparams={}):
    """ Set rc-parameters for matplotlib plot 

    Parameters
    ----------
    ff : int, default=3
        _description_
    fig_width_pt : _type_, default=246
        _description_
    inches_per_pt : float, default=0.014
        _description_
    golden_mean : float, default=0.66
        _description_
    rcparams : dict, default={}
        Additional changes to matplotlib rc parameters
        
    """
    space = ff*0.07
    fig_width = fig_width_pt*inches_per_pt
    fig_height = (fig_width*golden_mean)+space
    fig_size = [ff*fig_width, ff*fig_height]                                                          
    params = {'legend.fontsize': ff*4,
              'axes.linewidth': ff*6.5e-1,                                                           
              'axes.labelsize': ff*6,
              'axes.titlesize': ff*6,                                                                   
              'xtick.labelsize': ff*7,
              'ytick.labelsize': ff*7,
              'text.usetex': False,
              'figure.figsize': fig_size}
    params.update(rcparams)
    plt.rcParams.update(params)

def make_table_for_lammps(
    table_filename, pair_keyword, rmin=0.5, rmax=15, N=2000, pair=None, args=(), 
    kwargs={}, mode=0, prec=7, shift=False, rc=None
):
    """ Write table with lammps potential

    Parameters
    ----------
    table_filename : str
        Output path and filename for table
    pair_keyword : str
        Keyword for pair style function supported in :mod:`eptl.potentials_for_lammps` and
        to specify pair_style in LAMMPS ``pair_coeff`` definition.
    rmin : float, default=0.5
        Minimum value in distance array, ``r``
    rmax : float, default=15
        Cutoff value in distance array, ``r``, for potential. Consider adding a molecular length
        scale to the target cutoff.
    N : int, default=2000
        Number of points in table
    pair : func, default=None
        Pass a custom pair potential function, if ``None`` one of the supported potentials
        are called.
    args : tuple, default=()
        Arguments in addition to ``r`` for pair potential function
    kwargs : dict, default={}
        Additional keyword arguments for pair potential function
    mode : int, default=0
        Set parameters for lammps lookup method according to the 
        `pair_style table documentation <https://docs.lammps.org/pair_table.html>`_
        Values of ``mode`` may be set to:
        
        - ``0`` : ``"N {N}``
        - ``1`` : ``"N {N} R {rmin} {rmax}``
        - ``2`` : ``"N {N} RSQ {rmin} {rmax}``
        
    prec : int, default=7
        Default number of decimal places for output table
    shift : bool, default=False
        Shift potential to be zero at the cutoff. If ``rc is None`` then assumed to be ``rmax``
        Note that shifting the potential does not remove a discontinuity from the force.
    rc : float, default=None
        Cutoff value to set potential equal to zero.
        
    """
    
    # Check that pair_keyword is supported
    if pair is None:
        if pair_keyword not in pair_styles and f"pair_{pair_keyword}" not in pair_styles:
            raise ValueError(f"Provided pair_keyword style, {pair_keyword} is not supported choose one " \
                "of the following: {}".format(" ".join(pair_styles)))
        else:
            if pair_keyword in pair_styles:
                pair = pair_styles[pair_keyword]
            else:
                pair = pair_styles[f"pair_{pair_keyword}"]

    # Set up r-array
    if mode == 2:
        r = rmin**2
        rdelta = ( rmax**2 - rmin**2 ) / N
    else:
        r = rmin
        rdelta = ( rmax - rmin ) / N

    if shift:
        tmp_args = list(args)
        tmp_args[-1] = np.inf
        if rc is None:
            offset = -np.around( pair( rmax, *tmp_args, **kwargs), prec)
        else:
            offset = -np.around( pair( rc, *tmp_args, **kwargs), prec)
    else:
        offset = 0.0

    # Write file
    with open(table_filename,'w') as file:
        
        # Write Header
        file.write(f"# Created by [create_lammps_pairstyle_table.py version-{MY_VERSION}]\n\n")
        file.write(pair_keyword+"\n")
        if mode==0: # basic, use dist values as is
            file.write(f"N {N}\n\n")
        if mode==1: # use 'R' mode
            file.write(f"N {N} R {rmin} {rmax}\n\n")
        if mode==2: # use 'RSQ' mode
            file.write(f"N {N} RSQ {rmin} {rmax}\n\n")
            
        # Write Content    
        for n in range(1, N+1):
            rsqrt = math.sqrt(r) if mode == 2 else r
            pot = np.around( pair( rsqrt, *args, **kwargs), prec)
            if not np.isclose(pot, 0.0, 1e-12):
                pot += offset
            file.write(f"{n} {r} "+"{} {}\n".format(
                pot, 
                np.around( force( partial(pair), rsqrt, *args, **kwargs), prec)
                ))
            r += rdelta


def pair_write_lammps( 
    lmps_input_filename, lmps_table_filename, table_filename, units_string, 
    rc=12, style='linear', mode=0):
    """ Write LAMMPS Input File

    Parameters
    ----------
    lmps_input_filename : str
        Output filename and path for lammps input file
    lmps_table_filename : str
        Output filename and path for lammps tabulated values
    table_filename : str
        Output path and filename for table produced by eptl
    units_string : str
        Units for potential, choose system from `LAMMPS documentation <https://docs.lammps.org/units.html>`_
    rc : float, default=12
        Cutoff used for potential
    style : str, default="linear"
        Style of processing the table according to `LAMMPS pair_style table <https://docs.lammps.org/pair_table.html>`_
         Can be `lookup`, `linear`, `spline`, or `bitmap`.
    mode : int, default=0
        Mode used to generate table according to the 
        `pair_style table documentation <https://docs.lammps.org/pair_table.html>`_
        Values of ``mode`` may be set to:
        
        - ``0`` : ``"N {N}``
        - ``1`` : ``"N {N} R {rmin} {rmax}``
        - ``2`` : ``"N {N} RSQ {rmin} {rmax}``
        
    """
    
    if not os.path.isfile(table_filename):
        raise ValueError(f"Table file could not be found: {table_filename}")
    
    pair_keyword, N, rmin, rmax = pull_table_info(table_filename, mode=mode)

    # Pull template lammps input file
    lmpin = open(lmps_pair_write_generic_filename,'r')
    lmps_comms = lmpin.read()
    lmpin.close()
    
    if rc > rmax:
        rc = rmax

    # Replace keywords in template file
    lmps_comms = lmps_comms.replace("UNITS", units_string)
    lmps_comms = lmps_comms.replace("STYLE", style)
    lmps_comms = lmps_comms.replace("NPOINTS", str(N))
    lmps_comms = lmps_comms.replace("FILE", table_filename)
    lmps_comms = lmps_comms.replace("KEYWORD", pair_keyword)
    lmps_comms = lmps_comms.replace("RCUT", str(rc))
    lmps_comms = lmps_comms.replace("PWPOINTS", str(N))
    lmps_comms = lmps_comms.replace("RMIN", str(rmin))
    lmps_comms = lmps_comms.replace("RMAX", str(rmax))
    if mode==0 or mode==1:
        lmps_comms = lmps_comms.replace("RMODE", "r")
    else:
        lmps_comms = lmps_comms.replace("RMODE", "rsq")
    lmps_comms = lmps_comms.replace("PWFIL", lmps_table_filename)

    # Write out usable lammps input file
    lmpout = open(lmps_input_filename,'w')
    lmpout.write(lmps_comms)
    lmpout.close()
    
    # Run LAMMPS
    if os.path.isfile(lmps_table_filename):
        os.remove(lmps_table_filename)
    executable = Lmp.executable()
    os.system(f"{executable} -i {lmps_input_filename} -l {os.path.join('tables', 'log.lammps')} -screen none")


def pull_table_info(table_filename, mode=0):
    """ Pull the pair_keyword and number of points from LAMMPS table file.
    
    Parameters
    ----------
    table_filename : str
        Filename and path to LAMMPS tabulated potential produced by eptl
    mode : int, default=0
        Mode used to generate table according to the 
        `pair_style table documentation <https://docs.lammps.org/pair_table.html>`_
        Values of ``mode`` may be set to:
        
        - ``0`` : ``"N {N}``
        - ``1`` : ``"N {N} R {rmin} {rmax}``
        - ``2`` : ``"N {N} RSQ {rmin} {rmax}``
        
    Returns
    -------
    pair_keyword : str
        Keyword used to specify pair_style in LAMMPS ``pair_coeff`` definition.
    N : int
        Number of points in the provided table
    rmin : float
        Minimum value in potential table
    rmax : float
        Minimum value in potential table
        
    """ 
    if not os.path.isfile(table_filename):
        raise ValueError(f"Table file could not be found: {table_filename}")
    
    rmin = None
    
    with open(table_filename, "r") as f:
        for line in f:
            if line[0] == "#" or not line.strip():
                continue
            line_array = line.split()
            if line_array[0].isdigit():
                if rmin is None:
                    rmin = float(line_array[1])
            else:
                if line_array[0] == "N":
                    N = int(line_array[1])
                elif len(line_array) == 1:
                    pair_keyword = line_array[0]
                else:
                    raise ValueError("pair_keyword must be a single string without spaces")
        rmax = float(line_array[1])
        
    if mode == 2:
        rmin, rmax = math.sqrt(rmin), math.sqrt(rmax)
            
    return pair_keyword, N, rmin, rmax

def comparison(
    filename_with_path, lmps_table_filename, table_filename, mode=0, plot=True, show_plot=False,
    rlim=None, plim=None, flim=None, markers=True
):
    """ Compare potential written in table to original function and write the difference to a file.

    Parameters
    ----------
    filename_with_path : str
        Output file and path with difference between tabulated values
    lmps_table_filename : str
        Input filename and path for lammps tabulated values
    table_filename : str
        Input path and filename for table produced by eptl
    mode : int, default=0
        Mode used to generate table according to the 
        `pair_style table documentation <https://docs.lammps.org/pair_table.html>`_
        Values of ``mode`` may be set to:
        
        - ``0`` : ``"N {N}``
        - ``1`` : ``"N {N} R {rmin} {rmax}``
        - ``2`` : ``"N {N} RSQ {rmin} {rmax}``
        
    plot : bool, default=True
        Set whether to plot the difference
    rlim : tuple, default=None
        Minimum and maximum limits to plot x-axis
    plim : tuple, default=None
        Minimum and maximum limits to plot potential y-axis
    flim : tuple, default=None
        Minimum and maximum limits to plot force y-axis
    markers : bool, default=True
        Set whether markers are placed on the data points
        
    """
    
    lmpsdata = np.genfromtxt(lmps_table_filename, skip_header=6)
    OGdata = np.genfromtxt(table_filename, skip_header=5)
    
    if np.shape(lmpsdata)[0] != np.shape(OGdata)[0]:
        raise ValueError("Tabulated potential files, lmps_table_filename and table_filename, must be the same length.")
    
    rlmps = lmpsdata[:,1]
    rOG = OGdata[:,1]
    if mode == 2:
        rOG = np.sqrt(rOG)
    
    pairlmps = lmpsdata[:,2]
    pairOG = OGdata[:,2]
    
    forcelmps = lmpsdata[:,3]
    forceOG = OGdata[:,3]

    lenN = len(forceOG) # this will be the same as the others, so arbitrarily chosen

    with open(filename_with_path,'w') as out:
        
        for n in range(lenN):
            
            pairdiff = ( pairOG[n] - pairlmps[n] ) / pairlmps[n]
            if math.isnan(pairdiff):
                pairdiff = 0.0
            forcediff = ( forceOG[n] - forcelmps[n] ) / forcelmps[n]
            if math.isnan(forcediff):
                forcediff = 0.0
            out.write(f'{n} {rlmps[n]} {pairdiff} {forcediff}\n')
            
    if plot:
        fig, axs = plt.subplots(1,2, figsize=(14,5))
        
        axs[0].tick_params(
            which='both', direction='in', left=True, right=False, top=False, bottom=True
        )
        axs[0].tick_params( axis='both', which='major', length=10, width=2)
        axs[1].tick_params(
            which='both', direction='in', left=True, right=False, top=False, bottom=True
        )
        axs[1].tick_params( axis='both', which='major', length=10, width=2)
        
        if rlim is not None:
            axs[0].set_xlim(rlim)
            axs[1].set_xlim(rlim)
        if plim is not None:
            axs[0].set_ylim(plim)
        if flim is not None:
            axs[1].set_ylim(flim)
            
        # plot pair energy first
        plot_kwargs1 = {
            "color": "b", "linestyle": 'dotted', "label": "pair_write LAMMPS",
        }
        plot_kwargs2 = {
            "color": "k", "linestyle": 'solid', "label": "Original", "linewidth": 0.5,
        }
        if markers:
            plot_kwargs1["marker"] = "s"
            plot_kwargs2["marker"] = "v"
        
        # Pair
        axs[0].plot( rOG, pairOG, **plot_kwargs2)
        axs[0].plot( rlmps, pairlmps, **plot_kwargs1)
        axs[0].plot( [0, rOG[-1]], [0, 0], "k", linewidth=0.5, label=None)
        axs[0].set( xlabel="r", ylabel="Pair potential energy (r)")
        #axs[0].legend( loc='best', fontsize=12, markerscale=2.0, frameon=False)
        
        # force
        axs[1].plot( rOG, forceOG, **plot_kwargs2)
        axs[1].plot( rlmps, forcelmps, **plot_kwargs1)
        axs[1].plot( [0, rOG[-1]], [0, 0], "k", linewidth=0.5, label=None)
        axs[1].set(xlabel="r",ylabel="Force (r)")
        axs[1].legend(loc='best', fontsize=12, markerscale=2.0, frameon=False)
            
        fig.tight_layout(pad=2.0)
        plt.savefig(".".join(filename_with_path.split(".")[:-1]+["png"]), dpi=300)
