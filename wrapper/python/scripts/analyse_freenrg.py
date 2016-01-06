# coding=utf-8
description = """
analyse_freenrg --
is an analysis app that has been designed to analyse the output of free energy calculations in Sire.
analyse_freenrg can read in multiple file types.
1. It reads a Sire Saved Stream (.s3) file that contains a list of Sire.Analysis free energy objects (e.g.
 FEP, TI, Bennetts).
 analyse_freenrg will average and analyse these free energies according the to options you supply,
 e.g. assuming that the free energies are stored in freenrgs.s3, and you want to average iterations 100-200 from the
 simulation, and write the results to ‘results.txt’, type;

sire.app/bin/analyse_freenrg -i freenrgs.s3 -r 100 200 -o results.txt

Alternatively, if you just want to average over the last 60% of iterations, type;

sire.app/bin/analyse_freenrg -i freenrgs.s3 -o results.txt

(you can specify the percentage to average using the ‘--percent’ option)

2. analyse_freenrg can also read an ascii simulation.dat file generated with somd-freenerg. A common usage for this type of file would be
the analysis of a list of input files containing alchemical simulation data for a TI analysis and MBAR analysis. A usage example
would look like this:

sire.app/bin/analyse_freenrg -s lambda-*/simfile.dat -o results.txt

or just supplying a list of input directories

sire.app/bin/analyse_freenrg -l lambda-* -o results.txt

3. analyse_freenrg automatically knows how many free energies are contained in the s3 file, what their types are and what
should be done to analyse the results. For example, the waterswap, ligandswap and quantomm apps all output s3 files that
contain FEP, Bennetts and TI free energy data, so analyse_freenrg knows automatically to perform FEP, Bennetts and TI
analysis on that data and to report all of the results. analyse_freenrg also knows whether or not finite difference
approximations have been used, whether forwards and backwards windows were evaluated, the temperature and conditions of
the simulation etc. The aim is that it should handle everything for you, so you can concentrate on looking at the
potential of mean force (PMF) or final result.

For FEP data, analyse_freenrg will return the FEP PMF across lambda, together with errors based on statistical
convergence (95% standard error) and the difference between forwards and backwards free energies (if available).

For Bennett's data, analyse_freenrg will return the Bennett's Acceptance Ratio PMF across lambda, with errors based on
statistical convergence (95% standard error).

For TI data, analyse_free energy will return the PMF across lambda based on polynomial fitting of the gradients and
analytic integration of the resulting function. It will also return the integral across lambda using simple quadrature.
Errors are based on statistical convergence (95% standard error) and on the difference between the forwards and
backwards finite difference gradients (if available, and if finite-difference TI was used).

If you need more help understanding or interpreting the results of an analyse_freenrg analysis then please feel free to
get in touch via the Sire users mailing list, or by creating a github issue.
"""

from Sire.Analysis import *
from Sire.Maths import AverageAndStddev
import Sire.Stream
import argparse
import sys
import os
from Sire.Units import *
from Sire import try_import
from Sire.Tools.FreeEnergyAnalysis import SubSample
from Sire.Tools.FreeEnergyAnalysis import FreeEnergies
from Sire.Tools.FreeEnergyAnalysis import SimfileParser
np = try_import("numpy")


def process_free_energies(nrgs, FILE, range_start, range_end, percent):
    r"""Analysies free energies and computes PMFs
    Parameters
    ----------
    nrgs :

    FILE : filehandle
        either standard out or outputfile given in the arguments
    range_start : int
        starting point of the data for analysis
    range_end : int
        end point of the data for analysis
    Returns
    -------
    name, convergence, pmf :
        Here we need to add some info
    """
    # try to merge the free enegies - this will raise an exception
    # if this object is not a free energy collection
    nrgs.merge(0, 0)

    FILE.write("# Processing object %s\n" % nrgs)

    name = nrgs.typeName().split("::")[-1]

    nits = nrgs.count()

    # get the convergence of the free energy
    convergence = {}

    for i in range(1, nits):
        try:
            convergence[i] = nrgs[i].sum().values()[-1].y()
        except:
            try:
                convergence[i] = nrgs[i].integrate().values()[-1].y()
            except:
                pass

    # now get the averaged PMF
    if range_start:
        if range_start > nits - 1:
            start = nits - 1
        else:
            start = range_start

        if range_end > nits - 1:
            end = nits - 1
        else:
            end = range_end

    else:
        end = nits - 1
        start = end - int(percent * end / 100.0)

    FILE.write("# Averaging over iterations %s to %s\n" % (start, end))

    nrg = nrgs.merge(start, end)

    try:
        pmf = nrg.sum()
    except:
        pmf = nrg.integrate()

    return (name, convergence, pmf)

def parse_args():
    r""" function that parses all commandline arguemnts
    """
    parser = argparse.ArgumentParser(description="Analyse free energy files to calculate "
                                                 "free energies, PMFs and to view convergence.",
                                     epilog="analyse_freenrg is built using Sire and is distributed "
                                            "under the GPL. For more information please visit "
                                            "http://siremol.org/analyse_freenrg",
                                     prog="analyse_freenrg")

    parser.add_argument('--description', action="store_true",
                        help="Print a complete description of this program.")

    parser.add_argument('--author', action="store_true",
                        help="Get information about the authors of this script.")

    parser.add_argument('--version', action="store_true",
                        help="Get version information about this script.")

    parser.add_argument('-i', '--input', nargs=1,
                        help="Supply the name of the Sire Streamed Save (.s3) file containing the "
                             "free energies to be analysed.")

    parser.add_argument('-l', '--lambda_input', nargs='*',
                        help="Supply the name of all lambda directories that contain files "
                             "with free energies to be analysed.")

    parser.add_argument('-g', '--gradients', nargs='*',
                        help="Supply the name of the Sire Streamed gradients (.s3) files containing the "
                             "gradients to be analysed.")

    parser.add_argument('-s', '--simfiles', nargs='*',
                        help="Supply the name of the Ascii simulation files (.dat) files containing the "
                             "the biased energies needed for the analysis with MBAR.")

    parser.add_argument('-o', '--output', nargs=1,
                        help="""Supply the name of the file in which to write the output.""")

    parser.add_argument('-r', '--range', nargs=2,
                        help="Supply the range of iterations over which to average. "
                             "By default, this will be over the last 60 percent of iterations.")

    parser.add_argument('-p', '--percent', nargs=1,
                        help="Supply the percentage of iterations over which to average. By default "
                             "the average will be over the last 60 percent of iterations.")

    parser.add_argument('--lam', nargs='*', type=float,
                        help="The values of lambda at which a PMF should be evaluated.")

    parser.add_argument('-t', '--temperature', nargs=1, type=float, help='temperature in [Kelvin] at which the simulation was generated.')

    parser.add_argument('--no-subsampling', dest='subsampling', action='store_false', help='do not use the default subsampling')
    parser.set_defaults(subsampling=True)

    sys.stdout.write("\n")
    args = parser.parse_args()

    must_exit = False

    if args.description:
        print("%s\n" % description)
        must_exit = True

    if args.author:
        print("\nanalyse_freenrg was written by Christopher Woods and Antonia Mey (C) 2015.")
        print("\nIt is based on the analysis tools in Sire.Analysis, as well as pymbar (https://github.com/choderalab/pymbar)")
        must_exit = True

    if args.version:
        print("analyse_freenrg -- from Sire release version <%s>" % Sire.__version__)
        print("This particular release can be downloaded here: "
              "https://github.com/michellab/Sire/releases/tag/v%s" % Sire.__version__)
        must_exit = True

    if must_exit:
        sys.exit(0)

    return args, parser

def analyse_range(range):
    r""" Sets the range of data that should be analysed
    Parameters:
    -----------
    range : string array, size = 2

    Returns:
    --------
    range_start : int
        start integer for the range to be analysed
    range_end : int
        end integer for the range to be analysed
    """

    r_start = int(range[0])
    r_end = int(range[1])
    if r_end < 1:
        r_end = 1
    if r_start < 1:
        r_start = 1
    if r_start > r_end:
        tmp = r_start
        r_start = r_end
        r_end = tmp
    return r_start, r_end


def do_simfile_analysis(input_file, FILE, percent = 0, lam = None, T = None, subsample = True):
    r""" do an MBAR analysis using the pymbar external library
    Parameters
    ----------
    input_file : list(dtype, string)
        ASCII type input files, e.g. simfile.dat containing information 
    FILE : filehandle
        file handle to the output file/std out given in the commandline arguments (-o)
    lam : ndarray(dtype=double)
        lambda array at which free energies should be evaluated, free energy differences are always computed between
        lambda = 0.0 and lambda = 1.0.
    T : double
        temperature at which the simulation was simulated
    percent : double
        percentage of data that should be used for analysis. 
        Default all data will initially used, but subsampled according to statistical inefficiency.
    """

    parser = SimfileParser(input_file, lam, T)
    parser.load_data()
    subsample_obj = SubSample(parser.grad_kn, parser.energies_kn, parser.u_kln, parser.N_k, percentage=percent, subsample=subsample)
    if parser.u_kln is not None:
        print ("# We can run an MBAR analysis", flush=True)
        subsample_obj.subsample_energies()
        subsample_obj.subsample_gradients()
    else:
        print ("# We will only run a TI analysis", flush=True)
        subsample_obj.subsample_gradients()
    #Now we have our subsampled data and want to do the analysis, either TI using Sire.Analysis or MBAR
    ti = do_sire_TI(subsample_obj.gradients_kn, subsample_obj.N_k_gradients, parser.lam)
    mbar = None
    if parser.u_kln is not None:
        free_energy_obj = FreeEnergies(u_kln = subsample_obj.u_kln, N_k =subsample_obj.N_k_energies, lambda_array = parser.lam, gradients_kn = subsample_obj.gradients_kn)
        mbar = do_mbar(free_energy_obj)
        FILE.write("# PMFs MBAR\n")
        FILE.write("# Lambda  PMF  Maximum  Minimum \n")
        pmf = mbar[0]
        error_mbar = mbar[1]
        FILE.write("# Free energies MBAR \n")
        for i in range(pmf.shape[0]):
            FILE.write("%f  %f %f %f \n" % (pmf[i][0], pmf[i][1], pmf[i][1]+error_mbar[i], pmf[i][1]-error_mbar[i]))
        FILE.write("# Free energies MBAR \n")
        if T is not None:
            FILE.write("# %s = %s +/- %s kcal mol-1\n" % ("MBAR",  pmf[-1][1], error_mbar[-1]))
        else:
            print ('# If you want estimates in kcal mol-1 please provide a simulation temperature')
            FILE.write("# %s = %s +/- %s reduced units\n" % ("MBAR", pmf[-1][1], error_mbar[-1]))
    #TI results
    FILE.write("# PMFs TI\n")
    FILE.write("# Lambda  PMF  Maximum  Minimum \n")
    for p in ti.values():
        FILE.write("%.2f  %f  %f  %f \n" % \
        (p.x(),p.y(), \
        p.y()-p.yError(), \
        p.y()+p.yError()) )
    FILE.write("# Free energies TI \n")
    FILE.write("# %s = %s +/- %s kcal mol-1" % ("TI", ti.deltaG(), ti.values()[-1].yMaxError()))
    try:
        FILE.write(" (quadrature = %s kcal mol-1)\n" % ti.quadrature())
    except:
        pass
    #Now printing out free energy differences:

def do_sire_TI(gradients_kn, N_k, lam):
    r""" generates a sire.Analysis.Gradients object that allows to compute TI gradients
    Parameters
    ----------
    gradients_kn : np.array(shape=(nlambda, nsamples))
        array that contains all the subsampled gradients
    lam : np.array(shape(nlambda))
        array that contains all simulated lambda values
    """
    #adapted from script by Christopher Woods
    grad_data = {}
    for l in range(lam.shape[0]):
        grad_data[lam[l]] = gradients_kn[l,:N_k[l]]
    lam_avgs = {}
    for lamval in grad_data:
        avg = AverageAndStddev()
        for grad in grad_data[lamval]:
            avg.accumulate(grad)
        lam_avgs[lamval] = avg
    gradients = Gradients(lam_avgs)
    # Note that if you have a set of FreeEnergyAverages, then
    # you need to pass in the value of delta lambda for FDTI, e.g.
    # gradients = Gradients(lam_avgs, delta_lam)

    # Note that if you have both forwards and backwards gradients,
    # then you need to pass them both in, e.g. via
    # gradients = Gradients(lam_avg_fwds, lam_avg_bwds, delta_lam)

    pmf_ti = gradients.integrate()
    return pmf_ti

def do_mbar(free_energy_obj):
    free_energy_obj.run_mbar()
    pmf_mbar = free_energy_obj.pmf_mbar
    error_mbar = free_energy_obj.error_pmf_mbar
    return (pmf_mbar, error_mbar)


def do_sire_analysis(input_file, FILE, range_start, range_end, percent):
    r""" do sire analysis contains all the analysis script using Sire.Analysis for free energies
    Parameter
    ---------
    input_file : list(dtype: string)
        list of inputfile strings that should be read with Sire.Stream.load(), only Sire binary formats are allowed.
    FILE : FILE
        file handle to the output file/std out given in the commandline arguments
    range_start : int
    range_end : int
    """

    num_inputfiles = len(input_file)
    # Only one input file provided, assumes it contains freenrgs
    freenrgs = Sire.Stream.load(input_file[0])
    results = []

    try:
        results.append(process_free_energies(freenrgs, FILE, range_start, range_end, percent))
    except:
        for freenrg in freenrgs:
            results.append(process_free_energies(freenrg, FILE, range_start, range_end, percent))

    FILE.write("# Convergence\n")
    FILE.write("# Iteration \n")
    for result in results:
        FILE.write("# %s " % result[0])
    FILE.write("\n")

    i = 1
    has_value = True
    while has_value:
        values = []
        has_value = False
        for result in results:
            if i in result[1]:
                has_value = True
                values.append(result[1][i])
            else:
                values.append(0.0)

        if has_value:
            FILE.write("%s " % i)

            for value in values:
                FILE.write("%s " % value)

            FILE.write("\n")
            i += 1

    FILE.write("# PMFs\n")

    for result in results:
        FILE.write("# %s\n" % result[0])
        FILE.write("# Lambda  PMF  Maximum  Minimum \n")

        for value in result[2].values():
            FILE.write(
                "%.2f  %f  %f  %f\n" % (value.x(), value.y(), value.y() + value.yMaxError(), value.y() - value.yMaxError()))

    FILE.write("# Free energies \n")

    for result in results:
        FILE.write("# %s = %s +/- %s kcal mol-1" % (result[0], result[2].deltaG(), result[2].values()[-1].yMaxError()))

        try:
            FILE.write(" (quadrature = %s kcal mol-1)" % result[2].quadrature())
        except:
            pass

        FILE.write("#\n")
    #FILE.close()

def convert_gradient_files(gradient_files):
    # Multiple input files provided. Assume we have several gradients files that must be combined
    grads = {}
    fwds_grads = {}
    bwds_grads = {}
    delta_lambda = None


    for gfile in gradient_files:
        grad = Sire.Stream.load(gfile)

        analytic_data = grad.analyticData()
        fwds_data = grad.forwardsData()
        bwds_data = grad.backwardsData()

        if len(analytic_data) > 0:
            # analytic gradients
            #print(analytic_data.keys())
            lamval = list(analytic_data.keys())[0]
            grads[lamval] = analytic_data[lamval]
        else:
            # finite difference gradients
            lamval = list(fwds_data.keys())[0]
            fwds_grads[lamval] = fwds_data[lamval]
            bwds_grads[lamval] = bwds_data[lamval]
            delta_lambda = grad.deltaLambda()

    ti = None

    if len(grads) > 0:
        ti = TI(Gradients(grads))
    else:
        ti = TI(Gradients(fwds_grads, bwds_grads, delta_lambda))

    input_file = "freenrgs.s3"
    Sire.Stream.save(ti, input_file)
    return [input_file]

def do_directory_analysis(lam_dirs, FILE, range_start, range_end, percent, lam, T, subsample):
    r""" tries to autodetect based on a list of directories what kind of analysis should be done
    default simulation file names are assumed
    Parameters
    ----------
    lam_dirs : list of strings
        contains list of directories that should be analysed
    FILE : file handle
        for input and output handling
    range_start : int
        starting point of the data to be analysed
    range_end : int
        end point of the data to be analysed
    percent : float
        percentage amount of data that should be analysed
    lam  : float
        generating lambda value
    T : float
        simulation temperature passed via command line parameter
    subsample : boolean
        subsample according to a timeseries analysis
        Default = True
    """
    grd_files = True
    grd_file_list = []
    sim_files = True
    simf_file_list = []
    for l in lam_dirs:
        if not os.path.isfile(os.path.join(l, 'simfile.dat')):
            sim_files = False
        else:
            simf_file_list.append(os.path.join(l, 'simfile.dat'))
        if not os.path.isfile(os.path.join(l, 'gradients.s3')):
            grd_files = False
        else:
            grd_file_list.append(os.path.join(l, 'gradients.s3'))
    if not grd_files and not sim_files and not os.path.isfile('freenrg.s3'):
        print("The supplied directories do not contain appropriate files for analysis. Each directory must at least contain"
              " either a simfile.dat or a gradients.s3 file. Neither does the base directory contain a freenrg.s3 file."
              " If you used non-default names for your outputfiles during your simulation "
              "please use the -g options for your gradient files, the -s for simulation files and -i for a free energy "
              "binary files.")
        sys.exit(-1)
    else:
        if os.path.isfile('freenrgs.s3'):
            do_sire_analysis('freenrgs.s3',FILE, range_start, range_end, percent)
        if sim_files:
            do_simfile_analysis(simf_file_list, FILE, percent, lam, T, subsample)
        elif grd_files:
            print ("#Analysing Sire Stream gradient files %s" %grd_file_list)
            input_file = convert_gradient_files(grd_file_list)
            do_sire_analysis(input_file, FILE, range_start, range_end, percent)
            cmd = "rm freenrgs.s3"
            os.system(cmd)



if __name__ == '__main__':
    #Here is the main body of the script
    #Let's check all the arguments properly and assign things according to the arguemnts
    args, parser = parse_args()
    if args.lambda_input:
        lam_dirs = args.lambda_input
    else:
        lam_dirs = None

    if args.input:
        input_file = args.input
    else:
        input_file = None

    if args.gradients:
        gradient_files = args.gradients
    else:
        gradient_files = None

    if args.simfiles:
        sim_files = args.simfiles
    else:
        sim_files = None

    if args.output:
        output_file = args.output[0]
    else:
        output_file = None

    if args.range:
        range_start, range_end = analyse_range(args.range)
    else:
        range_start = None
        range_end = None

    if args.percent:
        percent = float(args.percent[0])
    else:
        percent = 60.0

    print (args.subsampling)

    if not input_file and not gradient_files and not sim_files and not lam_dirs:
        parser.print_help()
        print("\nPlease supply the name of the .s3 file(s) containing the free energies/gradients to be analysed, or a list of directories"
              "containing either .s3 files or simfile.dat files, and example usage is:\n"
              "analyse_freenrg -l lambda*/")
        sys.exit(-1)

    if output_file:
        print("# Writing all output to file %s" % output_file)
        FILE = open(output_file, "w")
    else:
        print("# Writing all output to stdout")
        FILE = sys.stdout

    #We have a single or set of free energy files and do a free energy analysis using Sire.Analysis
    if input_file:
        FILE.write("# Analysing free energies contained in file(s) \"%s\"\n" % input_file)
        do_sire_analysis(input_file, FILE, range_start, range_end, percent)

    #We have a bunch of gradient files in binary stream format and create an free energy object file which is then Analysed with Sire.Analysis
    if gradient_files:
        FILE.write("# Analysing free energies contained in file(s) \"%s\"\n" % gradient_files)
        input_file = convert_gradient_files(gradient_files)
        do_sire_analysis(input_file, FILE, range_start, range_end, percent)
        cmd = "rm freenrgs.s3"
        os.system(cmd)

    #We have a bunch of simfiles that are used for an MBAR analysis, but also produce output for TI, and BAR
    if sim_files:
        FILE.write("# Analysing free energies contained in file(s) \"%s\"\n" % sim_files)
        do_simfile_analysis(sim_files, FILE, percent, lam = args.lam, T = args.temperature, subsample = args.subsampling)

    #Deal with lambda directory input, where no specific input files are supplied instead look for default input files in the given directories
    if lam_dirs:
        FILE.write("# Analysing free energies contained in directory(s) \"%s\"\n" % lam_dirs)
        do_directory_analysis(lam_dirs, FILE, range_start, range_end, percent, lam = args.lam, T = args.temperature, subsample = args.subsampling)

    FILE.close()



