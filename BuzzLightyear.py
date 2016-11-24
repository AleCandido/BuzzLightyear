# ********************** IMPORTS ***************************

# generic

from numpy import array, sqrt, diag, float64
from math import log10
from matplotlib import gridspec
from scipy.stats.distributions import chi2
from pylab import loadtxt, transpose, matrix, zeros, figure, title, xlabel, ylabel, xscale, yscale, grid, errorbar, savefig, plot, clf, logspace, linspace, legend, rc

# lab flavour

from uncertainties import unumpy
from lab import mme, fit_generic_xyerr2, xe, xep

__all__ = [ # things imported when you do "from lab import *"
    'plot_fit',
    'chi2_calc',
    'pretty_print_chi2',
    'latex_table',
    'fit',
    'fast_plot',
    'umme'
]

__version__ = '1.0'

# ********************** LAB3_DEVELOPMENTS ************************

def _XYfunction(a): # default for the x-y columns from the file entries
    return a[0], a[1]

def _load_data(directory,file_):
    # load the data matrix from the data file 
    
    data = loadtxt(directory+"data/"+file_+".txt", unpack = True)    
    if type(data[0]) is float64:    # check if the first column is a column 
        data=array(transpose(matrix(data)))

    return data

def _errors(data, units, XYfun):
    # performs the error calculation, using mme

    # calculate data error with mme
    data_err = zeros((len(data),len(data[0])))
    for i in range(len(data)):
        data_err[i]=mme(data[i],*units[i])
    
    # extract from data x,y values with errors
    entries = unumpy.uarray(data,data_err)
    
    X_err = XYfun(entries)[0]
    Y_err = XYfun(entries)[1]
    
    X=unumpy.nominal_values(X_err)
    Y=unumpy.nominal_values(Y_err)
    dX=unumpy.std_devs(X_err)
    dY=unumpy.std_devs(Y_err)

    return X, Y, dX, dY, data_err

def _preplot(directory, file_, X, Y, dX, dY, title_="", fig="^^", 
             Xscale="linear", Yscale="linear", Xlab="", Ylab=""):
    # print a raw plot of the data, see fast_plot for the public function

    figure(fig+"_2")
    if (fig == file_):
        clf()
    title(title_)
    xlabel(Xlab)
    ylabel(Ylab)
    if Xscale=="log":
        xscale("log")
    if Yscale=="log":
        yscale("log")
    grid(b=True)
    errorbar(X,Y,dY,dX, fmt=",",ecolor="black",capsize=0.5)
    savefig(directory+"grafici/fast_plot_"+fig+".pdf")
    savefig(directory+"grafici/fast_plot_"+fig+".png")

def _outlier_(directory, file_, units, X, XYfun):
    # mark the outlier on the data plot, read them from a specific file

    data_ol = _load_data(directory,file_+"_ol")
    X_ol, Y_ol, dX_ol, dY_ol, data_ol_err = _errors(data_ol, units, XYfun)

    smin=min(min(X_ol),min(X))
    smax=max(max(X_ol),max(X))

    return X_ol, Y_ol, dX_ol, dY_ol, smin, smax

def _residuals(fig, gne, gs, ax1, f, par, out, X, dX, Xlab, Xscale, Y, dY, X_ol=[], Y_ol=[], dY_ol=[]):
    # performs the calculation of residuals and plot it in a fashion way

    figure(fig+"_1")

    #subplot(212)
    ax2 = gne.add_subplot(gs[3,:], sharex=ax1)
    rc('ytick', labelsize=12)
    #title("Scarti normalizzati")
    xlabel(Xlab) #
    ylabel("Scarti")
    if Xscale=="log":
        xscale("log")
    grid(b=True)

    # these stuffs is an idiot derivative, it could be dangerous
    # substitute this artigianal df with a symbolic (sympy) or numeric (scipy/numpy) calculation
    # as soon as possible!
    df = (f(X+dX/1e6,*par)-f(X,*par)) /(dX/1e6)

    plot(X, (Y-f(X,*par))/sqrt(dY**2+(df*dX)**2), ".", color="black")

    if out ==True:
        plot(X_ol, (Y_ol-f(X_ol,*par))/dY_ol, "^", color="green")

def plot_fit(directory, file_, title_, units, f, par, out, fig, residuals, xlimp, XYfun, Xscale, Yscale, Xlab, Ylab, X, Y, dX, dY):
    """
        Parameters
        ----------    

        Returns
        -------

    """    
    gs = gridspec.GridSpec(4, 1)
    gne = figure(fig+"_1")
    if (fig == file_):
        clf()
    if residuals==True:
        ax1 = gne.add_subplot(gs[:-1,:])
        setp(ax1.get_xticklabels(), visible=False)
        
        #subplot(211)
    title(title_)
    if Xscale=="log":
        xscale("log")
    if Yscale=="log":
        yscale("log")

    errorbar(X,Y,dY,dX, fmt=",",ecolor="black",capsize=0.5)

    if residuals==False :
        xlabel(Xlab)
    ylabel(Ylab)
    xlima = array(xlimp)/100

    if out ==True:
        X_ol, Y_ol, dX_ol, dY_ol, smin, smax = _outlier_(directory, file_, units, X, XYfun)
        
    else:
        smin = min(X)
        smax = max(X)
        
    if Xscale=="log":
        l=logspace(log10(smin)*xlima[0],log10(smax*xlima[1]),1000)
    else:
        l=linspace(smin*xlima[0],smax*xlima[1],1000)
    grid(b=True)
    plot(l,f(l,*par),"red")
    
    if out==True:
        outlier = errorbar(X_ol,Y_ol,dY_ol,dX_ol, fmt="g^",ecolor="black",capsize=0.5)
        legend([outlier], ['outlier'], loc="best")
    if residuals==True:
#        if out==True:          # these commented lines are useless
                                # cause the flag "out" is already passed to "_residuals" as an argument
        _residuals(fig, gne, gs, ax1, f, par, out, X, dX, Xlab, Xscale, Y, dY, X_ol, Y_ol, dY_ol)
#    _residuals(fig, gne, gs, ax1, f, par, out, X, dX, Xlab, Xscale, Y, dY)
            
    savefig(directory+"grafici/fit_"+fig+".pdf")
    savefig(directory+"grafici/fit_"+fig+".png")

def chi2_calc(f, par, X, Y, dY, dX, cov):
    """
        Parameters
        ----------    

        Returns
        -------

    """        
    # as over this df is a derivative
    # has to be substitute, watch over
    df = (f(X+dX/1e6,*par)-f(X,*par)) /(dX/1e6)

    chi = sum((Y-f(X,*par))**2/(dY**2+(df*dX)**2))

    p = chi2.sf(chi, len(X)-len(par))    
    sigma = sqrt(diag(cov))
    
    normcov = zeros((len(par),len(par)))
    
    for i in range(len(par)):
        for j in range(len(par)):
            normcov[i,j] = cov[i, j]/(sigma[i]*sigma[j])

    return chi, sigma, normcov, p

def pretty_print_chi2(file_, par, sigma, chi, X, normcov, p):
    """
        Parameters
        ----------    

        Returns
        -------

    """    
    print("________________________________")
    print("\nFIT RESULT %s\n" % file_)
    for i in range(len(par)):
        print("p%s = %s" % (i,xep(par[i],sigma[i],",")))
    
    print("\nchi / ndof = %.1f / %s" %(chi,len(X)-len(par)))
    print("p_value = %.2f %%" %(p*100))
    if len(par)>1:
        print("covarianza normalizzata=\n", normcov)

def latex_table(directory, file_, data, data_err, tab, out, data_ol=[], data_err_ol=[]):
    """
        Parameters
        ----------    

        Returns
        -------

    """

    with open(directory+"tabelle/tab_"+file_+".txt", "w") as text_file:
        text_file.write("\\begin{tabular}{c")
        for z in range (1,len(data)):
            text_file.write("|c")
        text_file.write("} \n")
        text_file.write("%s" % tab[0])
        for z in range (1,len(data)):
            text_file.write(" & %s" % tab[z])
        text_file.write("\\\\\n\hline\n")
        for i in range (len(data[0])):
            text_file.write("%s" % xe(data[0][i], data_err[0][i], "$\pm$"))
            for j in range (1,len(data)):
                text_file.write(" & %s" % xe(data[j][i], data_err[j][i], "$\pm$"))
            text_file.write("\\\\\n")
        if out==True:
            for i in range (len(data_ol[0])):
                text_file.write("%s" % xe(data_ol[0][i], data_err_ol[0][i], "$\pm$"))
                for j in range (1,len(data_ol)):
                    text_file.write(" & %s" % xe(data_ol[j][i], data_err_ol[j][i], "$\pm$"))
                text_file.write("\\\\\n")
        text_file.write("\\end{tabular}")
        text_file.close()

def fit(directory, file_, units, f, p0, 
        title_="", Xlab="", Ylab="", XYfun=_XYfunction, 
        preplot=False, Xscale="linear", Yscale="linear", 
        xlimp = array([100.,100.]), residuals=False, 
        table=False, tab=[""], fig="^^", out=False):
    
    """
        Interface for the fit functions of lab library.
        It performs the following tasks:
            - make a fast plot of the datas, with errors of course
            - make the fit of the data and print the plot
            - print the residuals plot
            - recognize the outlier and mark them on the fit plot
            - print a file with the latex tables of data, 
                ready to import in the .tex file

        Parameters
        ----------
        directory:
        file_:
        units: array of tuples, 
               each tuple must contains two elements (unit, metertype)
        f:
        p0:
        

        Returns
        -------
        1, if all is gone well.

        Notes
        -----
        
    """
    data = _load_data(directory,file_)
    X, Y, dX, dY, data_err = _errors(data, units, XYfun)

    # define a default for the figure name
    if fig=="^^":
        fig=file_
    
    # print a fast plot of the data    
    if preplot==True :
        _preplot(directory, file_, title_, fig, X, Y, dX, dY, Xscale, Yscale, Xlab, Ylab)
    
    #Fit
    par, cov = fit_generic_xyerr2(f,X,Y,dX,dY,p0)
    
    #Plotto il grafico con il fit e gli scarti
    plot_fit(directory, file_, title_, units, f, par, out, fig, residuals, xlimp, XYfun, Xscale, Yscale, Xlab, Ylab, X, Y, dX, dY)

    #Calcolo chi, errori e normalizzo la matrice di cov
    chi, sigma, normcov, p = chi2_calc(f, par, X, Y, dY, dX, cov)

    #Stampo i risultati, il chi e la matrice di cov
    pretty_print_chi2(file_, par, sigma, chi, X, normcov, p)

    if out ==True:
        data_ol = _load_data(directory,file_+"_ol")
        X_ol, Y_ol, dX_ol, dY_ol, data_err_ol = _errors(data_ol, units, XYfun)
    else:
        data_ol=[]
        data_err_ol=[]

    #Salvo la tabella formattata latex
    if table==True:
        latex_table(directory, file_, data, data_err, tab, out, data_ol, data_err_ol)

    return 1

def fast_plot(directory, file_, units, XYfun=_XYfunction, title_="",
              fig="^^", Xscale="linear", Yscale="linear", Xlab="", Ylab=""):
    """
        Parameters
        ----------    
        directory: string
            the pwd
        file_: string
            the txt file with the data to crunch
        X: (N-shaped array of) numbers
        Y: (N-shaped array of) numbers
        dX: (N-shaped array of) numbers
        dY: (N-shaped array of) numbers
        title_: string, optional
            plot title
        fig: string, optional
        Xscale: string, optional
        Yscale: string, optional
        Xlab: string, optional
        Ylab: string, optional

        Returns
        -------
        1, if all goes well

    """    
    data = _load_data(directory,file_)
    X, Y, dX, dY, data_err = _errors(data, units, XYfun)

    # define a default for the figure name
    if fig=="^^":
        fig=file_
    
    # print a fast plot of the data    
    if preplot==True :
        _preplot(directory, file_, title_, fig, X, Y, dX, dY, Xscale, Yscale, Xlab, Ylab)

    return 1
    
def umme(value, unit="volt", instrument="lab3"):
    #Shortcut to generate an ufloat type with the error given by the mme function.
    return uncertainties.ufloat(value,mme(value,unit,instrument))
