#!/usr/bin/python
# 
# Program originally based on

# Copyright (C) 2013 FlowKit Ltd, Lausanne, Switzerland
# E-mail contact: contact@flowkit.com
#
# Modified by Raquel Dapena and Vicente Perez-Munuzuri
# E-mail contact: vicente.perez.munuzuri@usc.es
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License, either
# version 3 of the License, or (at your option) any later version.

#
# 2D flow around a set of cylinders
#

from numpy import *; from numpy.linalg import *
import matplotlib.pyplot as plt; from matplotlib import cm
from sys import * # para abortar o programa se hai erros no arquivo
import multiprocessing as mp
from numpy.fft import fft2, fftshift, fftfreq
from concurrent.futures import ProcessPoolExecutor

###### Flow definition #########################################################
maxIter = 300001 # Total number of time iterations.
stepIter= 10; stepVisual = 5000	# Steps to save
Re      = 200.0  # Reynolds number.
#nx = 1000; ny = 200; ly=ny-1.0; q = 9 # Lattice dimensions and populations.
nx = 2000; ny = 600; ly=ny-1.0; q = 9 # Lattice dimensions and populations.
r = 30	# Cylinder radius
margen = 5	# Margen de confianza en los bordes para calcular los parametros de turbulencia
tramos = 5	# Numero de secciones donde se van a calcular los parametros de turbulencia
num_bins = 60   # Numero de bins para Fourier

#uLB  = 0.04                       # Velocity in lattice units.
#nulb = uLB*r/Re; omega = 1.0 / (3.*nulb+0.5); # Viscosity and Relaxation parameter.

# Here, nu is kept constant while uLB is calculated as a function of r and Re. 
nulb = 1.e-03; #omega = 1.0 / (3.*nulb+0.5); # Viscosity and Relaxation parameter.  
uLB  = Re*nulb/r                # Velocity in lattice units.

cosa = zeros([int(maxIter/stepIter),4*tramos])
tiempo = arange(0.,int(maxIter/stepIter))

tke = zeros(tramos)
ener = zeros(tramos)
vort = zeros(tramos)
avg_dissipation = zeros(tramos)

# Definition of subareas to calculate mean values of the variables.

mx_max = zeros(tramos); mx_min = zeros(tramos)

my_max = ny-margen; my_min = margen

mx_max[0] = 1200;  mx_min[0] = nx/2
mx_max[1] = 1400;  mx_min[1] = mx_max[0]
mx_max[2] = 1600;  mx_min[2] = mx_max[1]
mx_max[3] = nx-200;  mx_min[3] = mx_max[2]
mx_max[4] = nx-200;  mx_min[4] = nx/2

######## Cylinders position definition ###############

NumCyl = int(argv[1])

texto = "{:03d}".format(NumCyl)

fname='FouMulti_Re'+str(int(Re))+'_r'+str(int(r))+'_'+texto

outfile = open(fname, 'w')
outfile.close()

# Area where disks may be deployed

cx_max = 800; cx_min = 400
cy_max = ny-r-margen;cy_min = r+margen

cx=zeros(NumCyl);cy=zeros(NumCyl)

# Disks are located randomly without overlapping
# First disk

ivic = 0

aleat = random.rand(2)
cx[0] = cx_min + aleat[0]*(cx_max-cx_min)
cy[0] = cy_min + aleat[1]*(cy_max-cy_min)

print (0, cx[0], cy[0], ivic)

# Rest of disks

for k1 in range(1,NumCyl):
	while True:
		aleat = random.rand(2)
		cx_try = cx_min + aleat[0]*(cx_max-cx_min)
		cy_try = cy_min + aleat[1]*(cy_max-cy_min)

# We check there is no overlapping with the rest of disks and leave some space in between them	
	
		valid = True
		for k2 in range(k1):
			dist = sqrt((cx_try-cx[k2])**2 + (cy_try-cy[k2])**2)
			if (dist<=2*r+margen):
				ivic += 1
				valid = False
				break
		
		if valid:
			break
	cx[k1] = cx_try
	cy[k1] = cy_try

	print (k1, cx[k1], cy[k1], ivic)
	
###### Lattice Constants #######################################################

c = array([(x,y) for x in [0,-1,1] for y in [0,-1,1]]) # Lattice velocities.
t = 1./36. * ones(q)                                   # Lattice weights.
t[asarray([norm(ci)<1.1 for ci in c])] = 1./9.; t[0] = 4./9.
noslip = [c.tolist().index((-c[i]).tolist()) for i in range(q)] 
i1 = arange(q)[asarray([ci[0]<0  for ci in c])] # Unknown on right wall.
i2 = arange(q)[asarray([ci[0]==0 for ci in c])] # Vertical middle.
i3 = arange(q)[asarray([ci[0]>0  for ci in c])] # Unknown on left wall.

###### Function Definitions ####################################################

sumpop = lambda fin: sum(fin,axis=0) # Helper function for density computation.

def equilibrium(rho,u):              # Equilibrium distribution function.
	cu   = 3.0 * dot(c,u.transpose(1,0,2))
	usqr = 3./2.*(u[0]**2+u[1]**2)
	feq = zeros((q,nx,ny))
	for i in range(q): feq[i,:,:] = rho*t[i]*(1.+cu[i]+0.5*cu[i]**2-usqr)
	return feq

#####     Calculate the Turbulent Kinetic Energy (TKE) for a two-dimensional flow.

def calculate_tke(u, dx_min, dx_max, dy_min, dy_max):
# Calculate the mean velocity components
	u_mean = mean(u[0,dx_min:dx_max,dy_min:dy_max])
	v_mean = mean(u[1,dx_min:dx_max,dy_min:dy_max])
    
# Calculate the fluctuating components
	u_prime = u[0] - u_mean
	v_prime = u[1] - v_mean
    
# Calculate the variances of the fluctuations
	u_prime_sq = u_prime ** 2
	v_prime_sq = v_prime ** 2
    
# Calculate the mean of the squared fluctuations
	u_prime_sq_mean = mean(u_prime_sq)
	v_prime_sq_mean = mean(v_prime_sq)
    
# Calculate TKE
	TKE = 0.5 * (u_prime_sq_mean + v_prime_sq_mean)
	return TKE
    
###### Calculate the energy of the system (kinetic energy) ############

def energy(u,rho, dx_min, dx_max, dy_min, dy_max):

	kin_energy = 0.5*rho[dx_min:dx_max,dy_min:dy_max]*(u[0,dx_min:dx_max,dy_min:dy_max]**2 + u[1,dx_min:dx_max,dy_min:dy_max]**2)
			
	energy = mean(kin_energy)
	
	return energy

###### Compute the mean vorticity field given velocity components ##############

def vorticity(u, dx_min, dx_max, dy_min, dy_max):
	dvdx = (roll(u[1,dx_min:dx_max,dy_min:dy_max], -1, axis=1) - roll(u[1,dx_min:dx_max,dy_min:dy_max], 1, axis=1))
	dudy = (roll(u[0,dx_min:dx_max,dy_min:dy_max], -1, axis=0) - roll(u[0,dx_min:dx_max,dy_min:dy_max], 1, axis=0))
   
	vort2 = abs(dvdx - dudy)   # Al calcular abs() se calcula la intensidad media de la vorticidad 
	vorticity = 0.5*mean(vort2**2)	# Media de la enstrofia en el dominio

	return vorticity
	
###### Compute the average viscous dissipation given velocity components ##############

def viscous_dissipation(u, dx_min, dx_max, dy_min, dy_max):

# Compute velocity gradients using central differences

	du_dx = (roll(u[0,:,:], -1, axis=1) - roll(u[0,:,:], 1, axis=1)) / 2
	du_dy = (roll(u[0,:,:], -1, axis=0) - roll(u[0,:,:], 1, axis=0)) / 2
	dv_dx = (roll(u[1,:,:], -1, axis=1) - roll(u[1,:,:], 1, axis=1)) / 2
	dv_dy = (roll(u[1,:,:], -1, axis=0) - roll(u[1,:,:], 1, axis=0)) / 2
    
# Compute the viscous dissipation function
	phi = 2 * nulb * (du_dx**2 + dv_dy**2) + nulb * (du_dy + dv_dx)**2
    
# Compute the average dissipation
	avg_phi = mean(phi[dx_min:dx_max,dy_min:dy_max])
    
	return avg_phi

def compute_all_metrics(ii):
	args = (int(mx_min[ii]), int(mx_max[ii]), int(my_min), int(my_max))
	return [
		calculate_tke(u, *args),
		energy(u, rho, *args),
		vorticity(u, *args),
		viscous_dissipation(u, *args)
		]

def compute_fft_spectrum(field):
# Compute 2D FFT and power spectrum
	field_k = fftshift(fft2(field))
	spec = abs(field_k)**2

# Compute k magnitudes
	nx, ny = field.shape[1], field.shape[0]
	kx = fftfreq(nx) * 2 * pi  # Wave numbers in x-direction
	ky = fftfreq(ny) * 2 * pi  # Wave numbers in y-direction
	kx, ky = meshgrid(fftshift(kx), fftshift(ky))
	k_mag = sqrt(kx**2 + ky**2)  # Compute radial wave number

# Radial averaging
	k_bins = linspace(0, amax(k_mag), num=num_bins)
#	k_mids = 0.5 * (k_bins[1:] + k_bins[:-1])
		
#	ens_spectrum  = histogram(k.flatten(), bins=k_bins, weights=power_spectrum_ens.flatten())[0]
#	ener_spectrum = histogram(k.flatten(), bins=k_bins, weights=power_spectrum_ener.flatten())[0]

# Bin into 1D radial spectrum
	spectrum, _ = histogram(k_mag.ravel(), bins=k_bins, weights=spec.ravel())
	return spectrum, k_bins
    
def compute_strain_tensor(u):
    """
    Computes the 2D strain tensor for a velocity field.

    Parameters:
    - u: 2D numpy arrays of the velocity components u(0, x, y) and u(1, x, y)
    - dx, dy: spacing in x and y directions

    Returns:
    - strain tensor components as a tuple: (exx, eyy, exy)
    """
    dx=1.; dy=1.
    
    # Compute velocity gradients
    du_dx = gradient(u[0], dx, axis=1)
    du_dy = gradient(u[0], dy, axis=0)
    dv_dx = gradient(u[1], dx, axis=1)
    dv_dy = gradient(u[1], dy, axis=0)

    # Strain tensor components
    exx = du_dx
    eyy = dv_dy
    exy = 0.5 * (du_dy + dv_dx)

    return exx, eyy, exy    
    	
    	
###### Setup: cylindrical obstacle and velocity inlet with perturbation ########

# Disks are defined for each of the positions calculated above so they donot step each other.

obstacle = False
for k in range(NumCyl):
	obstacle = obstacle + fromfunction(lambda x,y: (x-cx[k])**2+(y-cy[k])**2<r**2, (nx,ny))

vel = fromfunction(lambda d,x,y: (1-d)*uLB*(1.0+1e-4*sin(y/ly*2*pi)),(2,nx,ny))
feq = equilibrium(1.0,vel); fin = feq.copy()

# Prep figure
#fig = plt.figure(figsize=(8,5), dpi=600)

###### Main time loop ##########################################################

for time in range(maxIter):
	fin[i1,-1,:] = fin[i1,-2,:] # Right wall: outflow condition.
	rho = sumpop(fin)           # Calculate macroscopic density and velocity.
	u = dot(c.transpose(), fin.transpose((1,0,2)))/rho

	u[:,0,:] =vel[:,0,:] # Apply inlet velocity on left wall
	rho[0,:] = 1./(1.-u[0,0,:]) * (sumpop(fin[i2,0,:])+2.*sumpop(fin[i1,0,:])) # Left wall: compute density from known populations

# Relaxation parameter omega is calculated at each time step taking into account the eddy viscosity for turbulent flows 
# (Mohamad book, page 122 (8.64))
# In case the user do not want to use it comment the following lines and uncomment above the omega definition.

	exx, eyy, exy = compute_strain_tensor(u)
	second_invariant = exx**2 + eyy**2 + 2 * exy**2
	eddy_visco = 0.1*0.1*sqrt(2.*second_invariant)
	omega = 1.0 / (3.*(nulb+eddy_visco)+0.5)
	
	feq = equilibrium(rho,u) # Left wall: Zou/He boundary condition.
	fin[i3,0,:] = fin[i1,0,:] + feq[i3,0,:] - fin[i1,0,:]
	fout = fin - omega * (fin - feq)  # Collision step.
	
	for i in range(q): fout[i,obstacle] = fin[noslip[i],obstacle] # No-slip condition on obstacles (bounce-back).
	for i in range(q): # Streaming step: shift particles in direction of velocity vectors c[i]
		fin[i,:,:] = roll(roll(fout[i,:,:],c[i,0],axis=0),c[i,1],axis=1)

# In order not to save every so often, we store in an array that we then dump to a file each stepVisual
# This process is parallelized. Variables are calculated in different subareas (tramos) and for each tramos we use a different core.
 
	if (time%stepIter==0): 
		with mp.Pool(processes=tramos) as pool:
			results = pool.map(compute_all_metrics, range(tramos))
			
		for ii, (tk, en, vo, diss) in enumerate(results):
			cosa[int(time / stepIter), ii] = tk
			cosa[int(time / stepIter), tramos + ii] = en
			cosa[int(time / stepIter), 2 * tramos + ii] = vo
			cosa[int(time / stepIter), 3 * tramos + ii] = diss
						
	if (time%stepVisual==0): # Visualization
		with open(fname,'w') as outfile:
			for i in range(int(maxIter / stepIter)):
				outfile.write(" ".join(f"{x:12.3e}" for x in cosa[i, :]) + "\n")  # Convert to formatted string

# Calculation of enstrophy and energy spectrum. Parallelized.

		dx_min=int(mx_min[0]); dx_max=int(mx_max[3]); dy_min=int(my_min); dy_max=int(my_max)
		
		dvdx = (roll(u[1,dx_min:dx_max,dy_min:dy_max], -1, axis=1) - roll(u[1,dx_min:dx_max,dy_min:dy_max], 1, axis=1))
		dudy = (roll(u[0,dx_min:dx_max,dy_min:dy_max], -1, axis=0) - roll(u[0,dx_min:dx_max,dy_min:dy_max], 1, axis=0))
	   
		enstrophy = 0.5*abs(dvdx - dudy)**2   
		
		kin_energy = 0.5*rho[dx_min:dx_max,dy_min:dy_max]*(u[0,dx_min:dx_max,dy_min:dy_max]**2 + u[1,dx_min:dx_max,dy_min:dy_max]**2)
		
		with ProcessPoolExecutor() as executor:
			futures = [
				executor.submit(compute_fft_spectrum, enstrophy),
				executor.submit(compute_fft_spectrum, kin_energy),
			]

		ens_spectrum, k_bins  = futures[0].result()
		ener_spectrum, _      = futures[1].result()
				
		texto2 = "{:03d}".format(int(time/stepVisual))
		gname  = 'EnsEnerMulti_Re'+str(int(Re))+'_r'+str(int(r))+'_N'+texto+"_"+texto2

		with open(gname,'w') as outfile:
			for i in range(num_bins-1):
				outfile.write(f"{k_bins[i]:12.3e} {ens_spectrum[i]:12.3e} {ener_spectrum[i]:12.3e}\n")
				
#		tke = calculate_tke (u)
#		print(f"Turbulent Kinetic Energy (TKE): {tke:.6f}")
#		ener = energy (u,rho)
#		print(f"Energy: {ener:.6f}")
#		vort = vorticity (u)
#		print(f"Vorticity: {vort:.6f}")
#		avg_dissipation = viscous_dissipation(u, nulb)
#		print(f"Dissipation: {avg_dissipation}")

# In case yo do not want to plot anything comment the following lines.
		
		plt.clf(); 
# Plot flow velocity        
		velocity = sqrt(u[0]**2+u[1]**2)
		velocity[obstacle] = nan
		cmap = plt.cm.Reds
		cmap.set_bad('black')
		plt.imshow(velocity.transpose(),cmap=cm.Reds,aspect='equal');#plt.show()
	        
		ax = plt.gca()
		ax.invert_yaxis()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		plt.savefig('Figure_Re'+str(int(Re))+'_r'+str(int(r))+'_N'+texto+"_"+texto2+'.png',format='png',dpi=600)
        	

