# ********************** IMPORTS ***************************

import numpy
from pylab import *
#from scipy.optimize import curve_fit
#import math
#import scipy.stats
#from scipy.stats import distributions
import uncertainties
from uncertainties import unumpy
from matplotlib import gridspec, pyplot
from math import log10, fsum, floor
#from inspect import getargspec
from numpy import array, asarray, isfinite, sqrt, diag, vectorize, number, isscalar
#import time

# ************************** FIT ***************************

def _check_finite(array): # asarray_chkfinite is absent in old numpies
	for x in array.flat:
		if not isfinite(x):
			raise ValueError("array must not contain infs or NaNs")

def curve_fit_patched(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, **kw):
	"""
		Same as curve_fit, but add absolute_sigma and check_finite if scipy is old.
		If the keyword argument force_patch=True is given, the patch is used anyway.
	"""
	from scipy.optimize import curve_fit
	force_patch = kw.pop('force_patch', False)
	args = getargspec(curve_fit).args
	if 'absolute_sigma' in args and 'check_finite' in args and not force_patch:
		rt = curve_fit(f, xdata, ydata, p0, sigma, absolute_sigma, check_finite, **kw)
	elif 'absolute_sigma' in args and not force_patch:
		if check_finite:
			_check_finite(xdata)
			_check_finite(ydata)
		rt = curve_fit(f, xdata, ydata, p0, sigma, absolute_sigma, **kw)
	else: # the case check_finite yes and absolute_sigma no does not exist
		myp0 = p0
		if p0 is None: # we need p0 to implement absolute_sigma
			args = getargspec(f).args
			if len(args) < 2:
				raise ValueError("Unable to determine number of fit parameters.")
			myp0 = [1.0] * (len(args) - (2 if 'self' in args else 1))
		if isscalar(myp0):
			myp0 = array([myp0])
		if check_finite:
			_check_finite(xdata)
			_check_finite(ydata)
		rt = curve_fit(f, xdata, ydata, p0, sigma, **kw)
		if absolute_sigma and len(ydata) > len(myp0): # invert the normalization done by curve_fit
			popt = rt[0]
			s_sq = sum(((asarray(ydata) - f(asarray(xdata), *popt)) / (asarray(sigma) if sigma != None else 1.0)) ** 2) / (len(ydata) - len(myp0))
			pcov = rt[1] / s_sq
			rt = (rt[0], pcov) + rt[2:]
	return rt

def fit_generic_xyerr(f, dfdx, x, y, sigmax, sigmay, p0=None, print_info=False, absolute_sigma=True, **kw):
	"""
		fit y = f(x, *params)
		
		Parameters
		----------
		f : callable
			the function to fit
		dfdx : callable
			derivative of f respect to x: dfdx(x, *params)
		x : M-length array
			independent data
		y : M-length array
			dependent data
		sigmax : M-length array or None
			standard deviation of x
		sigmay : M-length array
			standard deviation of y
		p0 : N-length sequence
			initial guess for parameters
		print_info : bool, optional
			If True, print information about the fit
		absolute_sigma : bool, optional
			If False, compute asymptotic errors, else standard errors for parameters
		
		Returns
		-------
		par : N-length array
			optimal values for parameters
		cov : (N,N)-shaped array
			covariance matrix of par
		
		Notes
		-----
		Algorithm: run curve_fit once ignoring sigmax, then propagate sigmax using
		dfdx and run curve_fit again with:
			sigmay = sqrt(sigmay**2 + (propagated sigmax)**2)
		until the differences between two successive estimates of the parameters are
		less than 1/1000 of the corresponding estimated errors.
	"""
	if sigmax is None:
		return curve_fit_patched(f, x, y, p0=p0, sigma=sigmay, absolute_sigma=absolute_sigma, **kw)
	x = asarray(x)
	sigmax = asarray(sigmax)
	sigmay = asarray(sigmay)
	cycles = 1
	par, cov = curve_fit_patched(f, x, y, p0=p0, sigma=sigmay, absolute_sigma=absolute_sigma, **kw)
	sigma = sqrt(diag(cov))
	error = sigma # to pass loop condition
	p0 = par
	while any(error > sigma / 1000):
		sigmayeff = sqrt(sigmay**2 + (dfdx(x, *p0) * sigmax)**2)
		par, cov = curve_fit_patched(f, x, y, p0=p0, sigma=sigmayeff, absolute_sigma=absolute_sigma, **kw)
		sigma = sqrt(diag(cov))
		error = abs(par - p0)
		p0 = par
		cycles += 1
	if print_info:
		print(fit_generic_xyerr, ": cycles: %d" % (cycles))
	return par, cov

def fit_generic_xyerr2(f, x, y, sigmax, sigmay, p0, print_info=False):
	"""
		fit y = f(x, *params)
		
		Parameters
		----------
		f : callable
			the function to fit
		x : M-length array
			independent data
		y : M-length array
			dependent data
		sigmax : M-length array
			standard deviation of x
		sigmay : M-length array
			standard deviation of y
		p0 : N-length sequence
			initial guess for parameters
		print_info : bool, optional
			If True, print information about the fit
		
		Returns
		-------
		par : N-length array
			optimal values for parameters
		cov : (N,N)-shaped array
			covariance matrix of par
		
		Notes
		-----
		This is a wrapper of scipy.odr
	"""
	from scipy.odr import Model, RealData, ODR
	f_wrap = lambda params, x: f(x, *params)
	model = Model(f_wrap)
	data = RealData(x, y, sx=sigmax, sy=sigmay)
	odr = ODR(data, model, beta0=p0)
	output = odr.run()
	par = output.beta
	cov = output.cov_beta
	if print_info:
		output.pprint()
	return par, cov

def fit_affine_yerr(x, y, sigmay):
	"""
		fit y = a * x + b
		
		Parameters
		----------
		x : M-length array
			independent data
		y : M-length array
			dependent data
		sigmay : M-length array
			standard deviation of y
		
		Returns
		-------
		a : float
			optimal value for a
		b : float
			optimal value for b
		vara : float
			variance of a
		varb : float
			variance of b
	"""
	x, y, sigmay = asarray((x, y, sigmay))
	dy2 = sigmay ** 2
	sy = fsum(y / dy2)
	sx2 = fsum(x ** 2 / dy2)
	sx = fsum(x / dy2)
	sxy = fsum(x * y / dy2)
	s1 = fsum(1 / dy2)
	denom = s1 * sx2 - sx ** 2
	a = (s1 * sxy - sy * sx) / denom
	b = (sy * sx2 - sx * sxy) / denom
	vara = s1 / denom
	varb = sx2 / denom
	return array([a, b, vara, varb])

def fit_affine_xyerr(x, y, sigmax, sigmay, print_info=False):
	"""
		fit y = a * x + b
		
		Parameters
		----------
		x : M-length array
			independent data
		y : M-length array
			dependent data
		sigmax : M-length array
			standard deviation of x
		sigmay : M-length array
			standard deviation of y
		print_info : bool, optional
			If True, print information about the fit
		
		Returns
		-------
		a : float
			optimal value for a
		b : float
			optimal value for b
		vara : float
			variance of a
		varb : float
			variance of b
		
		Notes
		-----
		Algorithm: run fit_affine_yerr once ignoring sigmax, then propagate sigmax
		using the formula:
			 sigmay = sqrt(sigmay**2 + (a * sigmax)**2)
		and run fit_affine_yerr again until the differences between two successive
		estimates of the parameters are less than 1/1000 of the corresponding
		standard deviations.
	"""
	par = fit_affine_yerr(x, y, sigmay)
	cycles = 1
	error = sqrt(par[2:]) # to pass loop condition
	while any(error > sqrt(par[2:]) / 1000):
		sigmayeff = sqrt(sigmay**2 + (par[0] * sigmax)**2)
		newpar = fit_affine_yerr(x, y, sigmayeff)
		error = abs((newpar - par)[:2])
		par = newpar
		cycles += 1
	if print_info:
		print(fit_affine_xyerr, ": cycles: %d" % (cycles))
	return par

def fit_affine_noerr(x, y):
	"""
		fit y = a * x + b
		
		Parameters
		----------
		x : M-length array
			independent data
		y : M-length array
			dependent data
		
		Returns
		-------
		a : float
			optimal value for a
		b : float
			optimal value for b
	"""
	x, y = asarray((x, y))
	sy = fsum(y)
	sx2 = fsum(x ** 2)
	sx = fsum(x)
	sxy = fsum(x * y)
	denom = len(x) * sx2 - sx ** 2
	a = (len(x) * sxy  - sx * sy) / denom
	b = (sy * sx2 - sx * sxy) / denom
	return array([a, b])

def fit_affine_xerr(x, y, sigmax):
	"""
		fit y = a * x + b
		
		Parameters
		----------
		x : M-length array
			independent data
		y : M-length array
			dependent data
		sigmax : M-length array
			standard deviation of x
		
		Returns
		-------
		a : float
			optimal value for a
		b : float
			optimal value for b
		vara : float
			variance of a
		varb : float
			variance of b
		
		Notes
		-----
		Implementation: consider the inverse relation:
			x = 1/a * y - b/a
		find 1/a and b/a using fit_affine_yerr then compute a, b and their variances
		with first-order error propagation.
	"""
	m, q, varm, varq = fit_affine_yerr(y, x, sigmax)
	a = 1 / m
	vara = varm / m**4
	b = -q / m
	varb = varq / m**2 + q**2 * vara
	return array([a, b, vara, varb])

def fit_affine_xerr2(x, y, sigmax):
	"""
		fit y = a * x + b
		
		Parameters
		----------
		x : M-length array
			independent data
		y : M-length array
			dependent data
		sigmax : M-length array
			standard deviation of x
		
		Returns
		-------
		a : float
			optimal value for a
		b : float
			optimal value for b
		vara : float
			variance of a
		varb : float
			variance of b
		
		Notes
		-----
		Algorithm: make a first estimate of <a> ignoring errors and propagate sigmax
		with the formula:
			sigmay = a * sigmax
		then run again considering errors on y until (a, b) converges
	"""
	par = fit_affine_noerr(x, y)
	sigmay = par[0] * sigmax
	newpar = fit_affine_yerr(x, y, sigmay)
	error = abs(newpar[:2] - par)
	par = newpar
	while any(error > par[2:] / 1000):
		sigmay = par[0] * sigmax
		newpar = fit_affine_yerr(x, y, sigmay)
		error = abs((newpar - par)[:2])
		par = newpar
	return par

def fit_const_yerr(y, sigmay):
	"""
		fit y = a
		
		Parameters
		----------
		y : M-length array
			dependent data
		sigmay : M-length array
			standard deviation of y
		
		Returns
		-------
		a : float
			optimal value for a
		vara : float
			variance of a
	"""
	y, sigmay = asarray((y, sigmay))
	dy2 = sigmay ** 2
	sydy2 = fsum(y / dy2)
	s1dy2 = fsum(1 / dy2)
	a = sydy2 / s1dy2
	vara = 1 / s1dy2
	return array([a, vara])

# *********************** MULTIMETERS *************************

def _find_scale(x, scales):
	# (!) scales sorted ascending
	for i in range(len(scales)):
		if x < scales[i]:
			return i
	return -1

def _find_scale_idx(scale, scales):
	# (!) scales sorted ascending
	for i in range(len(scales)):
		if scale == scales[i]:
			return i
		elif scale < scales[i]:
			return -1
	return -1

_util_mm_esr_data = dict(
	digital=dict(
		volt=dict(
			scales=[0.2, 2, 20, 200, 1000],
			perc=[0.5] * 4 + [0.8],
			digit=[1, 1, 1, 1, 2]
		),
		volt_ac=dict(
			scales=[0.2, 2, 20, 200, 700],
			perc=[1.2, 0.8, 0.8, 0.8, 1.2],
			digit=[3] * 5
		),
		ampere=dict(
			scales=[0.002,0.02,0.2,10],
			perc=[0.8,0.8,1.5,2],
			digit=[1] * 3 + [5]
		),
		ampere_ac=dict(
			scales=[0.002,0.02,0.2,10],
			perc=[1, 1, 1.8, 3],
			digit=[3] * 3 + [7]
		),
		ohm=dict(
			scales=[2 * 10**z for z in range(2, 8)],
			perc=[0.8] * 5 + [1, 5], #200MOhm 5%*(lettura-10digit) to do
			digit=[3, 1, 1, 1, 1, 2, 10]
		),
		volt_anal=dict(
			scales=[0.1, 2, 10, 50, 200, 500, 1000],
			relres=[50] * 7,
			valg=[1] * 7
		),
		volt_ac_anal=dict(
			scales=[10, 50, 250, 750],
			relres=[50] * 3 + [37.5],
			valg=[2] * 3 + [100.0 / 37.5]
		),
		ampere_anal=dict(
			scales=[50e-6, 500e-6, 5e-3, 50e-3, 500e-3, 5],
			relres=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
			valg=[1] * 6,
			cdt=[0.1, 0.294, 0.318] + [0.320] * 3
		),
		ampere_ac_anal=dict(
			scales=[250e-6, 2.5e-3, 25e-3, 250e-3, 2.5],
			relres=[50] * 5,
			valg=[2] * 5,
			cdt=[2, 1.5, 1.6, 1.6, 1.9]
		)
	)
)

def util_mm_er(x, scale, metertype='digital', unit='volt'):
	"""
		Returns the uncertainty of x and the internal resistance of the multimeter.
		
		Parameters
		----------
		x : number
			the value measured, may be negative
		metertype : string
			one of 'digital', 'analog'
			the multimeter used
		unit : string
			one of 'volt', 'volt_ac', 'ampere' 'ampere_ac', 'ohm'
			the unit of measure of x
		scale : number
			the fullscale used to measure x
		
		Returns
		-------
		e : number
			the uncertainty
		r : number or None
			the internal resistance (if applicable)
	"""
	
	x = abs(x)
	
	info = _util_mm_esr_data[metertype][unit]
	
	s = scale
	idx = _find_scale_idx(s, info['scales'])
	if idx < 0:
		raise KeyError(s)
	r = None
	
	if metertype == 'digital':
		if unit == 'volt' or unit == 'volt_ac' or unit == 'ampere' or unit == 'ampere_ac' or unit == 'ohm':
			e = x * info['perc'][idx] / 100.0 + info['digit'][idx] * 10**(idx + log10(info['scales'][0] / 2.0) - 3)
			if unit == 'volt' or unit == 'volt_ac':
				r = 10e+6
			elif unit == 'ampere' or unit == 'ampere_ac':
				r = 0.2 / s
		elif unit == 'volt_anal' or unit == 'volt_ac_anal' or unit == 'ampere_anal' or unit == 'ampere_ac_anal':
			e = x * sqrt((0.5 * info['relres'][idx]/x)**2 + (info['valg'][idx] / 100.0 )**2)
			if unit == 'volt' or unit == 'volt_ac':
				r = 20000 * s
			elif unit == 'ampere' or unit == 'ampere_ac':
				r = info['cdt'][idx] / s
			
	# elif metertype == 'analog':
	# 	e = x * sqrt((0.5 / info['relres'][idx])**2 + (info['valg'][idx] / 100.0 * s)**2)
	# 	if unit == 'volt' or unit == 'volt_ac':
	# 		r = 20000 * s
	# 	elif unit == 'ampere' or unit == 'ampere_ac':
	# 		r = info['cdt'][idx] / s
	
	return e, r

def util_mm_esr(x, metertype='digital', unit='volt'):
	"""
		determines the fullscale used to measure x with a multimeter,
		supposing the lowest possible fullscale was used, and returns the
		uncertainty, the fullscale and the internal resistance.
		
		Parameters
		----------
		x : number
			the value measured, may be negative
		metertype : string
			one of 'digital', 'analog'
			the multimeter used
		unit : string
			one of 'volt', 'volt_ac', 'ampere' 'ampere_ac', 'ohm'
			the unit of measure of x
		
		Returns
		-------
		e : number
			the uncertainty
		s : number
			the full-scale
		r : number or None
			the internal resistance (if applicable)
	"""
	
	x = abs(x)
	info = _util_mm_esr_data[metertype][unit]
	idx = _find_scale(x, info['scales'])
	s = info['scales'][idx]
	e, r = util_mm_er(x, s, metertype=metertype, unit=unit)
	return e, s, r

_util_mm_esr_vect_error = vectorize(lambda x, y, z: util_mm_esr(x, metertype=y, unit=z)[0], otypes=[number])
_util_mm_esr_vect_scale = vectorize(lambda x, y, z: util_mm_esr(x, metertype=y, unit=z)[1], otypes=[number])
_util_mm_esr_vect_res = vectorize(lambda x, y, z: util_mm_esr(x, metertype=y, unit=z)[2], otypes=[number])
_util_mm_esr2_what = dict(
	error=_util_mm_esr_vect_error,
	scale=_util_mm_esr_vect_scale,
	res=_util_mm_esr_vect_res
)

def util_mm_esr2(x, metertype='digital', unit='volt', what='error'):
	"""
		determines the fullscale used to measure x with a multimeter,
		supposing the lowest possible fullscale was used, and returns the
		uncertainty or the fullscale or the internal resistance.
		
		Parameters
		----------
		x : (X-shaped array of) number 
			the value measured, may be negative
		metertype : (X-shaped array of) string
			one of 'digital', 'analog'
			the multimeter used
		unit : (X-shaped array of) string
			one of 'volt', 'volt_ac', 'ampere' 'ampere_ac', 'ohm'
			the unit of measure of x
		what : (X-shaped array of) string
			one of 'error', 'scale', 'res'
			what to return
		
		Returns
		-------
		z : (X-shaped array of) number
			either the uncertainty, the fullscale or the internal resistance.
	"""
	if unit == 'ohm' and what == 'res':
		raise ValueError('asking internal resistance of ohmmeter')
	return _util_mm_esr2_what[what](x, metertype, unit)

# *********************** FORMATTING *************************

def _format_epositive(x, e):
	d = lambda x, n: int(("%.*e" % (n - 1, abs(x)))[0])
	ap = lambda x, n: float("%.*e" % (n - 1, x))
	if d(e, 2) < 3:
		n = 2
		e = ap(e, 2)
	elif d(e, 1) < 3:
		n = 2
		e = ap(e, 1)
	else:
		n = 1
	se = "%#.*g" % (n, e)
	dn = int(floor(log10(abs(x))) - floor(log10(e))) if x != 0 else -n
	if n + dn > 0:
		sx = "%#.*g" % (n + dn, x)
	else:
		le = 1 + floor(log10(e))
		sx = "0e%+d" % (le - n) if le - n != 0 else '0'
	return sx, se

def util_format_comp(x, e):
	"""
		format a value with its uncertainty
		
		Parameters
		----------
		x : number
			the value
		e : number
			the uncertainty
		
		Returns
		-------
		sx : string
			the formatted value
		se : string
			the formatted uncertainty
	"""
	e = abs(e)
	if not isfinite(x) or not isfinite(e) or e == 0:
		sx, se = "%.3g" % x, "%.3g" % e
	else:
		sx, se = _format_epositive(x, e)
	return sx, se

def util_format(x, e, pm='+-', percent=False):
	"""
		format a value with its uncertainty
		
		Parameters
		----------
		x : number
			the value
		e : number
			the uncertainty
		pm : string, optional
			the "plusminus" symbol
		percent : boolean, optional
			if True, also format the relative error as percentage
		
		Returns
		-------
		s : string
			the formatted value with uncertainty
	"""
	sx, se = util_format_comp(x, e)
	if not percent or float(sx) == 0:
		return "%s %s %s" % (sx, pm, se)
	else:
		ep = abs(e) / abs(x) * 100.0
		eps = "%.*g" % (2 if ep < 100.0 else 3, ep)
		return "%s %s %s (%s %%)" % (sx, pm, se, eps)

# ************************** TIME *********************************

def util_timecomp(secs):
	"""
		convert a time interval in seconds to hours, minutes, seconds
		
		Parameters
		----------
		secs : number
			the time interval expressed in seconds
		
		Returns
		-------
		hours : int
			hours, NOT bounded to 24
		minutes : int
			minutes, 0--59
		seconds : int
			seconds, 0--59
	"""
	hours = int(secs / 3600)
	minutes = int((secs - hours * 3600) / 60)
	seconds = secs - hours * 3600 - minutes * 60
	return hours, minutes, seconds

def util_timestr(secs):
	"""
		convert a time interval in seconds to a string with hours, minutes, seconds
		
		Parameters
		----------
		secs : number
			the time interval expressed in seconds
		
		Returns
		-------
		str : str
			string representing the interval
	"""
	return "%02d:%02d:%02d" % util_timecomp(secs)

_eta_start = 0

def util_etastart():
	return _time.time()

def util_etastr(eta, progress):
	interval = _time.time() - eta
	if 0 < progress <= 1:
		etastr = util_timestr((1 - progress) * interval / progress)
	elif progress == 0:
		etastr = "--:--:--"
	else:
		raise RuntimeError("progress %.2f out of bounds [0,1]")
	timestr = util_timestr(interval)
	return timestr, etastr

# ************************ SHORTCUTS ******************************

def mme(x, unit, metertype='digital'):
	"""
		determines the fullscale used to measure x with a multimeter,
		supposing the lowest possible fullscale was used, and returns the
		uncertainty of the measurement.
		
		Parameters
		----------
		x : (X-shaped array of) number 
			the value measured, may be negative
		unit : (X-shaped array of) string
			one of 'volt', 'volt_ac', 'ampere' 'ampere_ac', 'ohm'
			the unit of measure of x
		metertype : (X-shaped array of) string
			one of 'digital', 'analog'
			the multimeter used
		
		Returns
		-------
		e : (X-shaped array of) number
			the uncertainty
	"""
	return util_mm_esr2(x, metertype=metertype, unit=unit, what='error')

_util_format_vect = vectorize(util_format, otypes=[str])

def xe(x, e, pm='+-'):
	"""
		format a value with its uncertainty
		
		Parameters
		----------
		x : (X-shaped array of) number
			the value
		e : (X-shaped array of) number
			the uncertainty
		pm : string, optional
			the "plusminus" symbol
		
		Returns
		-------
		s : (X-shaped array of) string
			the formatted value with uncertainty
	"""
	return _util_format_vect(x, e, pm, False)

def xep(x, e, pm='+-'):
	"""
		format a value with its absolute and relative uncertainty
		
		Parameters
		----------
		x : (X-shaped array of) number
			the value
		e : (X-shaped array of) number
			the uncertainty
		pm : string, optional
			the "plusminus" symbol
		
		Returns
		-------
		s : (X-shaped array of) string
			the formatted value with uncertainty
	"""
	return _util_format_vect(x, e, pm, True)

#########################################################################################################
#########################################################################################################

def XYfunction(a): #formula per il calcolo della X dalle colonne del file in input
	return a[0], a[1]

def fast_plot(directory, file, units, titolo="", Xlab="", Ylab="", XYfun=XYfunction,Xscale="linear",Yscale="linear", fig="^^"):
	columns = loadtxt(directory+"data/"+file+".txt", unpack = True)
	if type(columns[0]) is numpy.float64:
		columns=array(transpose(matrix(columns)))
		
	dcolumns = zeros((len(columns),len(columns[0])))	
	for i in range(len(columns)):
		if units[i]=="volt_osc":
			dcolumns[i]=columns[i]*0.035
		elif units[i]=="volt_osc_nocal":
			dcolumns[i]=columns[i]*0.025
		elif units[i]=="tempo_osc":
			dcolumns[i]=columns[i]*0.01
		else:
			dcolumns[i]=mme(columns[i],units[i])
	
	entries = unumpy.uarray(columns,dcolumns)
	
	X_err = XYfun(entries)[0]
	Y_err = XYfun(entries)[1]
	
	X=unumpy.nominal_values(X_err)
	Y=unumpy.nominal_values(Y_err)
	dX=unumpy.std_devs(X_err)
	dY=unumpy.std_devs(Y_err)

	if fig=="^^":
		fig=file
	figure(fig+"_2")
	if (fig == file or out != True):
		clf()
	title(titolo)
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

#Load txt
def fit(directory, file, units, f, p0, titolo="", Xlab="", Ylab="", XYfun=XYfunction, preplot=False, Xscale="linear",Yscale="linear", xlimp = array([100.,100.]),scarti=False,table=False,tab=[""], fig="^^",out=False):
	
	columns = loadtxt(directory+"data/"+file+".txt", unpack = True)
	
	if type(columns[0]) is numpy.float64:
		columns=array(transpose(matrix(columns)))
	dcolumns = zeros((len(columns),len(columns[0])))
	for i in range(len(columns)):
		if units[i]=="volt_osc":
			dcolumns[i]=columns[i]*0.035
		elif units[i]=="volt_osc_nocal":
			dcolumns[i]=columns[i]*0.025
		elif units[i]=="tempo_osc":
			dcolumns[i]=columns[i]*0.01
		else:
			dcolumns[i]=mme(columns[i],units[i])
	
	entries = unumpy.uarray(columns,dcolumns)
	
	X_err = XYfun(entries)[0]
	Y_err = XYfun(entries)[1]
	
	X=unumpy.nominal_values(X_err)
	Y=unumpy.nominal_values(Y_err)
	dX=unumpy.std_devs(X_err)
	dY=unumpy.std_devs(Y_err)

	if fig=="^^":
		fig=file
	
	if preplot==True :
		figure(fig+"_2")
		if (fig == file):
			clf()
		title(titolo)
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
	
	#Fit
	par, cov = fit_generic_xyerr2(f,X,Y,dX,dY,p0)
	
	#Plotto il grafico con il fit e gli scarti

	gs = gridspec.GridSpec(4, 1)
	gne=figure(fig+"_1")
	if (fig == file):
		clf()
	if scarti==True:
		ax1 = gne.add_subplot(gs[:-1,:])
		pyplot.setp(ax1.get_xticklabels(), visible=False)
		
		#subplot(211)
	title(titolo)
	if Xscale=="log":
		xscale("log")
	if Yscale=="log":
		yscale("log")

	errorbar(X,Y,dY,dX, fmt=",",ecolor="black",capsize=0.5)

	if scarti==False :
		xlabel(Xlab)
	ylabel(Ylab)
	xlima = array(xlimp)/100
	
	#Perdonatemi per la schifezza che sto per scrivere#
	
	if out ==True:
		olcolumns = loadtxt(directory+"data/"+file+"_ol.txt", unpack = True)
		if type(olcolumns[0]) is numpy.float64:
			olcolumns=array(transpose(matrix(olcolumns)))
		oldcolumns = zeros((len(olcolumns),len(olcolumns[0])))
		for i in range(len(olcolumns)):
			if units[i]=="volt_osc":
				oldcolumns[i]=olcolumns[i]*0.035
			elif units[i]=="volt_osc_nocal":
				oldcolumns[i]=olcolumns[i]*0.025
			elif units[i]=="tempo_osc":
				oldcolumns[i]=olcolumns[i]*0.01
			else:
				oldcolumns[i]=mme(olcolumns[i],units[i])
		
		olentries = unumpy.uarray(olcolumns,oldcolumns)
		
		olX_err = XYfun(olentries)[0]
		olY_err = XYfun(olentries)[1]
		
		X_ol=unumpy.nominal_values(olX_err)
		Y_ol=unumpy.nominal_values(olY_err)
		dX_ol=unumpy.std_devs(olX_err)
		dY_ol=unumpy.std_devs(olY_err)
	
		smin=min(min(X_ol),min(X))
		smax=max(max(X_ol),max(X))
	else:
		smin = min(X)
		smax = max(X)
		
	#
	if Xscale=="log":
		l=logspace(log10(smin)*xlima[0],log10(smax*xlima[1]),1000)
	else:
		l=linspace(smin*xlima[0],smax*xlima[1],1000)
	grid(b=True)
	plot(l,f(l,*par),"red")
	
	if out==True:
		outlier = errorbar(X_ol,Y_ol,dY_ol,dX_ol, fmt="g^",ecolor="black",capsize=0.5)
		plt.legend([outlier], ['outlier'], loc="best")
	if scarti==True:
		#subplot(212)
		ax2 = gne.add_subplot(gs[3,:], sharex=ax1)
		pyplot.rc('ytick', labelsize=12)
		#title("Scarti normalizzati")
		xlabel(Xlab) #
		ylabel("Scarti")
		if Xscale=="log":
			xscale("log")
		grid(b=True)
		plot(X, (Y-f(X,*par))/dY, ".", color="black")

		if out ==True:
			plot(X_ol, (Y_ol-f(X_ol,*par))/dY_ol, "^", color="green")
			
	savefig(directory+"grafici/fit_"+fig+".pdf")
	savefig(directory+"grafici/fit_"+fig+".png")
	#Calcolo chi, errori e normalizzo la matrice di cov
	
	chi = sum((Y-f(X,*par))**2/dY**2)
	
	sigma=sqrt(diag(cov))
	
	normcov = zeros((len(par),len(par)))
	
	for i in range(len(par)):
		for j in range(len(par)):
			normcov[i,j]=cov[i, j]/(sigma[i]*sigma[j])

	#Stampo i risultati, il chi e la matrice di cov
	print("_________________________________________________________")
	print("\nFIT RESULT %s\n" % file)
	for i in range(len(par)):
		print("p%s = %s" % (i,xep(par[i],sigma[i],",")))
	
	print("\nchi / ndof =",chi,"/",len(X)-len(par))
	if len(par)>1 :
		print("covarianza normalizzata=\n",normcov)
	#Salvo la tabella formattata latex
	if table==True:
		with open(directory+"tabelle/tab_"+file+".txt", "w") as text_file:
			text_file.write("\\begin{tabular}{c")
			for z in range (1,len(columns)):
				text_file.write("|c")
			text_file.write("} \n")
			print()
			text_file.write("%s" % tab[0])
			for z in range (1,len(columns)):
				text_file.write(" & %s" % tab[z])
			text_file.write("\\\\\n\hline\n")
			for i in range (len(columns[0])):
				text_file.write("%s" % xe(columns[0][i], dcolumns[0][i], "$\pm$"))
				for j in range (1,len(columns)):
					text_file.write(" & %s" % xe(columns[j][i], dcolumns[j][i], "$\pm$"))
				text_file.write("\\\\\n")
			text_file.write("\\end{tabular}")
			text_file.close()
