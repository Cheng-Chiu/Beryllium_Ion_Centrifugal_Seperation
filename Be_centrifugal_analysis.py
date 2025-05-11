#!/usr/bin/env python3

# Origial code written by Jack
# Revived and debugged by Cheng Chiu (June 2024)
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys #for command line arguments
import os #for removing file extension from string
from scipy import signal, stats
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates

# matplotlib.font_manager._rebuild()

#mid=32768 deprecated

if len(sys.argv) != 3 and len(sys.argv) != 2:
	print("Usage: ./MCP_image_analysis_v6.1_centrifuge.py <MCP .tif file> <background subtraction>")
	print("<background subtraction> is optional, 0 (off) or 1 (on) and defaulted to 1.")
	sys.exit()
fp1 = sys.argv[1]
background_subtracted  = 1
if len(sys.argv) == 3:
	background_subtracted = int(sys.argv[2])
	if background_subtracted != 0 or background_subtracted!= 1:
		print("<background subtraction> is either 0 (off) or 1 (on).")

# fp2='mask_new.tif'
fp2 = 'MASKS/circular/MASK_1306_06.266.tif'

# set font sizes, marker point sizes, linewidth
fs1=26 # title font size
fs2=22.5 # Axis title font size
fs3=21 # Tick marker font size
fs4=21 # intermediate size
linewidth=3
pointsize=4
blue='#004299'
lblue='#4287f5'
vblue='#302480'
red='#8c303a'
pred='#ed003f'
green='#0e8006'


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['errorbar.capsize'] = 1


print("Reading file name: {}".format(fp1))
fname=str(os.path.splitext(str(fp1))[0])
run_name = fname
fname='radial_density/'+fname
I=plt.imread(fp1) #this is the read file step
Mask=plt.imread(fp2) #this is the read mask step
I = I.astype(np.float64)

### Cheng Chiu, June 2024
if background_subtracted:
	fp3 = "AT_US_MCP_background.tif"
	Background = plt.imread(fp3)
	Background = Background.astype(np.float64)
	I -= Background
	I[I < 0] = 0.0
### 

minI=np.amin(I) # minimum of I

for i in range(len(I)): # subtract minimum from whole image
	for j in range(len(I[i])):
		I[i][j]=I[i][j]-minI


meanI=np.mean(I) # average, stdev, max of I
stdevI=np.std(I)
maxI=np.amax(I)
# rangeI=maxI-minI
n=len(I) # 
m=len(I[0]) # image is an nxm matrix

b_calibration_factor=0.086/8.74 # this includes the mm/pixels calibration and
								# the fringe factor in B=1T
								# therefore converts pixels -> size in trap @B=1T

print("Image dimensions: {}x{}".format(n,m))
Antimask=[]
temp=[]

#inverted mask
for i in range(len(Mask)): # mask inversion
	for j in range(len(Mask[i])):
		temp.append(1-Mask[i][j]%2)
	Antimask.append(temp)
	temp=[]

IM=I*Mask # masked version of the image
IAM=I*Antimask # masked and antimasked versions of I


# so we've set up the image (I), masked image (IM) and antimasked image (IAM)
# now we can start working with them.

total_intensity=0
AA=[]
for i in range(len(IAM)): # average of the non-mask region
	for j in range(len(IAM[i])):
		if IAM[i][j] != 0:
			AA.append(I[i][j]) # this is just the elements of I that are outside the mask region

		total_intensity+=I[i][j] # this is sum of total intensity (whole region, not just masked)

# hot pixel removal: median filter. 
# generally leave commented out - it's not great.
#IM=scipy.signal.medfilt(IM,kernel_size=3)

bg=np.mean(AA) # mean of non-mask region
bgIntensity=(bg)*n*m # calculate bg intensity (total)
Intensity=total_intensity-bgIntensity # NNI calculation (total - bg)

# so now we have intensity (NNI)


# print("Background: {:e}\nIntensity: {:e}".format(bgIntensity,Intensity))


print("Min intensity: {}\nMax intensity: {}".format(
	np.min(I),np.max(I)))


# plot 2 - this is the unmasked version

# cutfac=1
# im=plt.imshow(I,vmax=(np.amax(I)/cutfac))
# plt.set_cmap('jet')
# plt.colorbar()
# plt.tight_layout()
# fname2=fname+'_MCP_1_v1'
# plt.savefig(fname2+'.png',format='png',dpi=500)
# plt.savefig(fname2+'.pdf',format='pdf',dpi=500)
#plt.show()


## great, so now let's try to find the middle of the plasma
# running mean function
def running_mean(x, N):
	cumsum= np.cumsum(np.insert(x, 0, 0))
	return (cumsum[N:] - cumsum[:-N]) / float(N)

# first try: weighted average
def centre_finder(a): # only argument is the 1-D list 'a'
	a_weighted=[] 
	cutfac=0.6
	for i in range(len(a)):
		a[i]=a[i]/10
	maxa=np.amax(a)

	for i in range(len(a)): # loop through list and weight by intensity
		if a[i]<cutfac*maxa:
			a[i]=0
		a_weighted.append(i*a[i])


	a_avg=np.sum(a_weighted)/np.sum(a) # weighted average position
	return (a_avg)

# flatten into both axes
flatten_i=[0]*n
for i in range(len(IM)):
	flatten_i[i]=np.sum(IM[i])

IMT=np.array(IM).T.tolist() # transpose IM
flatten_j=[0]*m
for j in range(len(IMT)):
	flatten_j[j]=np.sum(IMT[j])

x_c_rough=centre_finder(flatten_i)
y_c_rough=centre_finder(flatten_j)
#x_c_rough=257
#y_c_rough=364
print("centre is at ({:.2f}, {:.2f})".format(x_c_rough,y_c_rough))

# N_chords is the number of chords to draw, i.e. the number of points on the ellipse to 
# calculate.
N_chords=300
phi=np.linspace(0,2*math.pi,N_chords)
x_pos=x_c_rough
y_pos=y_c_rough
chord=[]
chordlength=90 # distance to travel out from centre. 150 is probably OK

# loop over phi. there is one chord drawn per loop.
for i in range(len(phi)):
	x_pos=x_c_rough
	y_pos=y_c_rough # initialise coords in the centre

	x_increment=math.cos(phi[i])
	y_increment=math.sin(phi[i]) # essentially converting from cartesian to polar
	chord.append([])

	# this loop goes along the 'chord', taking the mean of a 3x3 square each time it steps.
	# the centre of the square is simply the closest point to the current location along
	# the chord. 
	# it steps 1 pixel at a time at an angle phi. phi=0 is along +x, phi=pi/2 is along +y
	for j in range(chordlength):
		x_pos=x_pos+x_increment
		y_pos=y_pos+y_increment # increment x and y positions
		x_index=int(round(x_pos)) # round to get the indices of nearest point
		y_index=int(round(y_pos))

		# calculate the intensity. This is the mean of a 3x3 square centred on the point.
		local_intensity=np.mean([I[x_index][y_index],I[x_index+1][y_index],
			I[x_index+1][y_index+1],I[x_index][y_index+1],I[x_index-1][y_index+1],
			I[x_index-1][y_index],I[x_index-1][y_index-1],I[x_index][y_index-1],
			I[x_index+1][y_index-1],I[x_index+2][y_index],I[x_index][y_index+2],
			I[x_index-2][y_index],I[x_index][y_index-2]])
		chord[i].append(local_intensity)


# now the data is still too noisy - let's take a running mean
Nrun1=12
Nrun2=12
chord_rm=[]
chord_rm_dx=[]


#print(len(chord))
#print(chord[0])
for i in range(len(chord)):
	chord_rm.append(running_mean(chord[i],Nrun1))

print("chord:",np.shape(chord_rm))

# now we take the gradient at each step.
for i in range(len(phi)): # loop around phi
	chord_rm_dx.append([])
	for j in range(1,len(chord_rm[0])): # loop along each chord
		chord_rm_dx[i].append(chord_rm[i][j]-chord_rm[i][j-1])


# and take another running mean?
for i in range(len(phi)):
	chord_rm_dx[i]=running_mean(chord_rm_dx[i],Nrun2)


# now we can find the max gradient, and find the indices at that point.
# repeat until we have a bunch of points that trace out our ellipse

ellipse_indices=[]
cmax=[]
cmax_radius=[]

# phi loop
for i in range(len(chord_rm_dx)):
	cmax.append(0)
	cmax_radius.append(0)
	for j in range(len(chord_rm_dx[i])): # chord loop
		#print(chord_rm_dx[i][j])
		#print('{}, {}'.format(i,j))
		if chord_rm_dx[i][j]>cmax[i]:
			cmax[i]=chord_rm_dx[i][j]
			cmax_radius[i]=j

# re-calculate coordinates of the ellipse points
ex_coord=[];ey_coord=[]
ex_index=[];ey_index=[]

for i in range(len(phi)):
	ex_coord.append(x_c_rough+cmax_radius[i]*math.cos(phi[i]))
	ey_coord.append(y_c_rough+cmax_radius[i]*math.sin(phi[i]))
	ex_index.append(int(round(ex_coord[i])))
	ey_index.append(int(round(ey_coord[i])))
	#I[ex_index[i]][ey_index[i]]=maxI

fig2=plt.figure(figsize=(9,6))
plt.plot(chord[0],'o',color=blue,markersize=pointsize)
plt.plot(chord_rm[0],'D',color=green,markersize=pointsize)
#plt.plot(chord_rm_dx,'P',color=red,markersize=pointsize)



# now the ellipse fitting part
# took this code snippet from https://stackoverflow.com/questions/39693869/fitting-an-ellipse-to-a-set-of-data-points-in-python/48002645
# uses scikit-image: https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.EllipseModel
x=np.array(ex_coord)
y=np.array(ey_coord)
a_points=[]
for i in range(len(x)):
	a_points.append([x[i],y[i]])
a_points=np.array(a_points)
ell = EllipseModel()
ell.estimate(a_points)

xc, yc, a, b, theta = ell.params
print("center = ",  (yc, xc))
print("angle of rotation = ", theta)
print("axes = ", (a,b))

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
axs[0].plot(x,y,'x')

axs[1].plot(x, y,'x',color=blue)
axs[1].scatter(xc, yc, color='red', s=100)
axs[1].set_xlim(x.min(), x.max())
axs[1].set_ylim(y.min(), y.max())

# ell_patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='red', facecolor='none')
ell_patch = Ellipse((xc, yc), width=2*a, height=2*b, angle=theta*180/np.pi, edgecolor='red', facecolor='none')

axs[1].add_patch(ell_patch)

# now overlay the ellipse onto the MCP image
# note that we need to be careful with which axis is x and which is y
# (I messed this up somewhere earlier...), and need to do 90-theta for the rotation
# angle due to how the fit defines the angle vs how imshow() defines the angle.
# Ellipse is just matplotlib's ellipse function with the fit parameters thrown in
# 
fig3=plt.figure(figsize=(9,6))
cutfac=1
im=plt.imshow(I,vmax=(np.amax(I)/cutfac))
plt.set_cmap('jet')
ell_patch = Ellipse((yc, xc), width=2*a, height=2*b, angle=90. - theta*180./np.pi, edgecolor='red', facecolor='none')

plt.gca().add_patch(ell_patch)
# testing angle and axis direction
# y2 = yc + 500*np.sin(theta+np.pi/2)
# x2 = xc + 500*np.cos(theta+np.pi/2)
# plt.plot([yc, y2], [xc, x2], '-', color='red', linewidth=2)
# tick shit
ax=plt.gca()
ax.tick_params(axis="x", labelsize=fs3)
ax.tick_params(axis="y", labelsize=fs3)
plt.colorbar()
plt.tight_layout()

fname_fitted=fname+'_MCP_fitted'
plt.savefig(fname_fitted+'.png',format='png',dpi=500)
# plt.savefig(fname_fitted+'.pdf',format='pdf',dpi=500)

# write ellipse params to file
fname_temp=str(os.path.splitext(str(fp1))[0])
fname_ellipse='ellipse_params/' + fname_temp
with open(fname_ellipse + '.txt','w+') as fp_ellipse:
	fp_ellipse.write('#xc\t#yc\t#a\t#b\t#theta\n')
	fp_ellipse.write(str(xc)+'\t'+str(yc)+'\t'+str(a)+'\t'+str(b)+'\t'+str(theta)+'\n')

def circular_radial_profile():

	# now we want to calculate the intensity as a function of radius.
	I_radius=[]
	I_ang=[]

	# loop over the image (I). at each point, calculate the radius from the point to the centre
	# of the plasma
	# also rescale the radius based on ellipse coordinates, i.e. assume that the plasma has been
	# distorted into an ellipse, and undistort it. This rescaling is not done on the acual image
	# (i.e. I), but only on the radius used for plotting the density profile later.
	for i in range(len(I)):
		I_radius.append([]) 
		for j in range(len(I[i])):
			# calculate angle between point and ellipse axis
			if xc==0:
				xc=1e-7 # avoid 1/0
			I_ang.append(math.atan(abs(yc-j)/abs(xc-i)))
			if abs(xc-i)>=0 and abs(yc-j)>=0:
				pass
			if abs(xc-i)>=0 and abs(yc-j)<0:
				I_ang[-1]+=np.pi/2
			if abs(xc-i)<0 and abs(yc-j)<0:
				I_ang[-1]+=np.pi
			if abs(xc-i)<0 and abs(yc-j)>=0:
				I_ang[-1]+=3*np.pi/4

			# calculate effective radius (i.e. rescaled)
			rad=math.sqrt((xc-i)**2 + (yc-j)**2)
			anglefac=(math.cos(math.radians(theta)-I_ang[-1]))

			I_radius[i].append((rad+rad*(1-(b/a))*anglefac)*b_calibration_factor)

	# flatten both arrays. now we have a long array of intensity and its radius for every point in I
	I_radius=np.ndarray.flatten(np.asarray(I_radius))
	IMR=np.ndarray.flatten(np.asarray(I))


	# binning function.
	# start with two equal-size arrays, a and b
	# a is the array, and b is the array to bin by
	# a and b can be the same array
	# normally we're binning by the x-axis, in which case b is x

	def bin_2d(b,a,nbins):
		maxa=np.amax(a);mina=np.amin(a)
		maxb=np.amax(b);minb=np.amin(b)

		# create '_bins' array - len nbins, equally spaced
		a_bins=np.linspace(mina,maxa,nbins)
		b_bins=np.linspace(minb,maxb,nbins)

		# _cnts array - len(a) array with integer for each element, which is which bin it goes in
		a_cnts=np.digitize(a,a_bins)
		b_cnts=np.digitize(b,b_bins)
		print(len(a_cnts))
		for i in range(len(a_cnts)): # just to fix how np.digitize works...
			if a_cnts[i]==np.amax(a_cnts):
				a_cnts[i]-=1
				# print(i)
			if b_cnts[i]==np.amax(b_cnts):
				b_cnts[i]-=1
				# print(i)	

		# set up some fixed-length arrays for later
		a_binned=np.zeros(nbins)
		b_binned=np.zeros(nbins)
		a_binnedvals=[]
		b_binnedvals=[]
		a_stderr=[]
		b_stderr=[]

		while len(a_binnedvals)<nbins:
			a_binnedvals.append([])
			b_binnedvals.append([])
			a_stderr.append([])
			b_stderr.append([])

		# populate the _binned and _means arrays
		# _binned is a histogram of how many points are in each array
		# _means is the mean value for each bin

		for i in range(len(a_cnts)):
				a_binned[b_cnts[i]-1]+=1 
				b_binned[b_cnts[i]-1]+=1
				a_binnedvals[b_cnts[i]-1].append(a[i])
				b_binnedvals[b_cnts[i]-1].append(b[i])

		# cutting off the end b/c of how np.digitize works
		# (same work around as before)
		# it's acually super pepega...
		a_binnedvals=a_binnedvals[:-1]
		b_binnedvals=b_binnedvals[:-1]
		a_binned=a_binned[:-1]
		b_binned=b_binned[:-1]
		a_summed=[];b_summed=[]
		a_means=[];b_means=[]

		# finish computing _means array, and _summed array
		for i in range((len(a_binnedvals))):
			rmin=b_bins[i]
			rmax=b_bins[i+1]
			area_radius=math.pi*(rmax**2-rmin**2)
			npoints=len(b_binnedvals)
			npoints2=len(a_binnedvals)
			if npoints!=npoints2:
				print("error 37")
			a_stderr[i]=stats.sem(a_binnedvals[i])
			b_stderr[i]=stats.sem(b_binnedvals[i])
			a_summed.append(np.sum(a_binnedvals[i])/area_radius)
			b_summed.append(np.sum(a_binnedvals[i])/area_radius)
			a_means.append(np.mean(a_binnedvals[i]))
			b_means.append(np.mean(b_binnedvals[i]))
			
		return b_means, a_means, b_summed, a_summed, b_stderr, a_stderr

	# binning on the radial density.
	nbins=320
	I_radius_binned,IMR_binned,I_radius_summed,IMR_summed,I_radius_stderr,IMR_stderr=bin_2d(I_radius,IMR,nbins)

	radial_mean=[]
	for i in range(len(I_radius_binned)):
		if 1.6<I_radius_binned[i]<1.7:
			radial_mean.append(IMR_summed[i])
	radial_mean=np.mean(radial_mean)

	for i in range(len(IMR_summed)):
		IMR_summed[i]=IMR_summed[i]-radial_mean

	# write to file
	fname_data='radial_profile/' + fname_temp
	print("Saving... {}".format(fname_data))
	with open(fname_data + '.txt', 'w+') as fp_data:
		fp_data.write('#radius\tParticles per mm^2\t#Particles per mm^2 ERROR\n')
		for i in range(len(I_radius_binned)):
			fp_data.write(str(I_radius_binned[i])+'\t'+str(IMR_summed[i])+'\t'+str(IMR_stderr[i])+'\n')
	fp_data.seek(0)  # Go back to the beginning of the file
	content = fp_data.read()
	print("File content:", content)



	# the radial density profile plot.
	fig5=plt.figure(figsize=(9,6))
	xlabel='Radius from centre of plasma (mm)'
	ylabel='Particles/mm^2 in trap'
	ax=plt.gca()
	ax.grid(ls=':', color='0.65')
	ax.tick_params(axis="x", labelsize=fs3)
	ax.tick_params(axis="y", labelsize=fs3)
	ax.tick_params(direction='in',which='both',top=1,right=1)
	ax.xaxis.labelpad=15
	ax.yaxis.labelpad=15


	plt.plot(I_radius_binned,IMR_summed,'o',color=blue,markersize=pointsize)
	plt.ylabel(ylabel,fontsize=fs2) 
	plt.xlabel(xlabel,fontsize=fs2)
	plt.xlim(left=0,right=1.7) 
	plt.ylim(bottom=-20000) 
	plt.tight_layout()

	fname_radial=fname+'_radial'
	plt.savefig(fname_radial+'.png',format='png',dpi=500)
	# plt.savefig(fname_radial+'.pdf',format='pdf',dpi=500)



### @@ Cheng Chiu, June 2024
### Updated functionality on radial profile along specified angle and averaged along concentric ellipses 
### intensity to particle number calibration f_cal_Be+ = 0.799
"""
Convention: +X axis is horizontal to the left, and +Y axis is vertical downwards.
			Angle is zeroed at +x axis and increment CCW.
"""
# The xc and yc definition were incorrectly flipped, so swapping them back
xc, yc = yc, xc

# USAT pixel size calibration
unit = "[px]"
mm_per_px_calib = 0.0325

# MCP to trap calibration
mcp_trap_calib = 1.

def axis_intensity(ang, xc_set=xc, yc_set=yc, theta_set=theta, pxmm_mode=1):
	"""
	Calculate intensity along axis of specified angle with respect to major axis.
	
	ang (float): Angle of rotation with respect to major axis to plot, in degrees.
	xc_set (float, optional): X axis of center of Be+ elliptical fit, in pixels.
	yc_set (float, optional): Y axis of center of Be+ elliptical fit, in pixels.
	theta_set (float, optional): Angle of rotation of elliptical fit with respect to x axis,
								 in radians.
	pxmm_mode (int, optional): 0: plot with pixel; 1: plot with millimeter
							   Default with mode 1 (pixel).
	"""
	# Maximum distance to sample, which is the hypothanus
	max_distance = round(np.hypot(I.shape[0], I.shape[1]))

	# Define direction vectors based on theta
	# adjust step
	f = 1
	dx = f*np.sin(theta+ang*np.pi/180.)
	dy = f*np.cos(theta+ang*np.pi/180.)

	# Generate discrete coordinates
	coords = []
	# store intensity from (xc, yc), the center of the ellipse
	xcoor = xc
	ycoor = yc
	for _ in range(int(max_distance)):
		# Round to nearest integer coordinates
		ix, iy = int(round(xcoor)), int(round(ycoor))
		if 0 <= ix < I.shape[1] and 0 <= iy < I.shape[0]:
			coords.append((ix, iy))
		elif coords:
			# once it goes out of bound from valid region, it stays invalid
			break
		xcoor += dx
		ycoor += dy

	# Extract the values at the computed coordinates
	radial_intensity = [I[iy, ix] for ix, iy in coords]
	# for index, val in enumerate(radial_intensity):
	# 	radial_dist.append(np.hypot(round(xc) - coords[index][0], round(yc) - coords[index][1]))
	radial_dist = np.array([0 + i*f for i in range(len(radial_intensity))]).astype(float)
	if pxmm_mode == 1:
		radial_dist *= mm_per_px_calib
	radial_intensity = np.array(radial_intensity)
	return radial_intensity, radial_dist

# Gaussian fit function
def gaussian(x, a, b, c, d):
	'''
	Gaussian Fit
	'''
	return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

# Perform Gaussian fit
def gaussian_fitting(rIntensity, dist, pxmm_mode):
	'''
	Perform Gaussian fit to radial intensity along specified direction
	
	
	rIntensity (float array): Intensity along specified direction, in arb.
	dist (float array): Distance from the center of ellipse, in pixels.
	theta_set (float, optional): Angle of rotation of elliptical fit with respect to x axis,
								 in radians.
	pxmm_mode (int, optional): 0: plot with pixel; 1: plot with millimeter
							   Default with mode 1 (pixel).
	'''
	max_index = np.argmax(rIntensity)
	right_index = len(rIntensity) - 1
	left_index = 0
	# find fitting range, 3 consecutive points around background < 100
	for i in range(max_index, len(rIntensity) - 2):
		if all(rIntensity[j] < 100 for j in range(i, i + 3)):
			right_index = i + 4
			break
	for i in range(max_index, 1, -1):
		if all(rIntensity[j] < 100 for j in range(i, i - 3, -1)):
			left_index = i - 4
			break
	# Initial guess for parameters
	initial_guess = [1500, max_index, np.std(rIntensity), 10]
	if pxmm_mode == 0:
		initial_guess = [1500, max_index, 1, 10]
	
	params, covariance = curve_fit(gaussian, dist[left_index:right_index+1], rIntensity[left_index:right_index+1], p0=initial_guess)
	horiozntal_vals = np.arange(dist[left_index], dist[right_index]+1, 0.01)
	# print(params)
	return gaussian(horiozntal_vals, *params), horiozntal_vals, params


def plot_along_angle(ang, xc_set=xc, yc_set=yc, theta_set=theta, pxmm_mode=1):
	'''
	Plot intensity along axis of specified angle with respect to major axis.
	
	ang (float): Angle of rotation with respect to major axis to plot, in degrees.
	xc_set (float, optional): X axis of center of Be+ elliptical fit, in pixels.
	yc_set (float, optional): Y axis of center of Be+ elliptical fit, in pixels.
	theta_set (float, optional): Angle of rotation of elliptical fit with respect to x axis,
								 in radians.
	pxmm_mode (int, optional): 0: plot with pixel; 1: plot with millimeter
							   Default with mode 0 (pixel).
	'''
	rIntensity, dist = axis_intensity(ang, xc_set=xc, yc_set=yc, theta_set=theta, pxmm_mode=pxmm_mode)

	# plot with px or mm
	global unit
	if pxmm_mode == 1:
		unit = "[mm]"

	# Third argument in return is parameters of the fit
	fit_intensity, fit_dist, params = gaussian_fitting(rIntensity, dist, pxmm_mode)
	# the radial density profile plot.
	fig=plt.figure(figsize=(9,6))
	xlabel=f'Distance from center of plasma {unit}'
	ylabel='Intensity [arb]'
	ax=plt.gca()
	ax.grid(ls=':', color='0.65')
	ax.tick_params(axis="x", labelsize=fs3)
	ax.tick_params(axis="y", labelsize=fs3)
	ax.tick_params(direction='in',which='both',top=1,right=1)
	ax.xaxis.labelpad=15
	ax.yaxis.labelpad=15

	center = params[1]
	stddev = abs(params[2])
	plt.plot(dist,rIntensity,'o',color=blue,markersize=pointsize,label=f'{ang}\u00B0')
	plt.plot(fit_dist,fit_intensity, label='Fitted', color='blue')
	plt.ylabel(ylabel,fontsize=fs2) 
	plt.xlabel(xlabel,fontsize=fs2)
	plt.ylim(bottom=0)
	plt.xlim(left=max(0,center-10.*stddev), right=center+10.*stddev)
	plt.legend(fontsize='xx-large') 
	plt.tight_layout()

	fname_radial=fname+'_'+str(ang)+'_radial'
	plt.savefig(fname_radial+'.png',format='png',dpi=500)
	# plt.savefig(fname_radial+'.pdf',format='pdf',dpi=500)

	# write to file
	fname_data='radial_profile/' + fname_temp
	print("Saving... {}".format(fname_data))
	with open(fname_data + '_' + str(ang) + '.txt', 'w+') as fp_data:
		fp_data.write(f'#Distance from center {unit}\tIntensity [arb]\n')
		for i in range(len(rIntensity)):
			fp_data.write(str(dist[i])+'\t\t'+str(rIntensity[i])+'\n')


def plot_major(xc_set=xc, yc_set=yc, theta_set=theta, pxmm_mode=1):
	'''
	Plot intensity along both major axes and their averaged value.
	
	xc_set (float, optional): X axis of center of Be+ elliptical fit, in pixels.
	yc_set (float, optional): Y axis of center of Be+ elliptical fit, in pixels.
	theta_set (float, optional): Angle of rotation of elliptical fit with respect to x axis,
								 in radians.
	pxmm_mode (int, optional): 0: plot with pixel; 1: plot with millimeter
							   Default with mode 0 (pixel).
	'''
	rIntensity_0, dist_0 = axis_intensity(0, xc_set=xc, yc_set=yc, theta_set=theta, pxmm_mode=pxmm_mode)
	rIntensity_180, dist_180 = axis_intensity(180, xc_set=xc, yc_set=yc, theta_set=theta, pxmm_mode=pxmm_mode)
	min_length = len(rIntensity_0)
	dist = dist_0
	if len(rIntensity_0) > len(rIntensity_180):
		min_length = len(rIntensity_180)
		dist = dist_180
	rIntensity_avg = np.array([rIntensity_0[i] + rIntensity_180[i] for i in range(min_length)])/2.

	# plot with px or mm
	global unit
	if pxmm_mode == 1:
		unit = "[mm]"
	
	# fitting
	fit_intensity_0, fit_dist_0, params_0 = gaussian_fitting(rIntensity_0, dist, pxmm_mode)
	fit_intensity_180, fit_dist_180, params_180 = gaussian_fitting(rIntensity_180, dist, pxmm_mode)
	fit_intensity_avg, fit_dist_avg, params_avg = gaussian_fitting(rIntensity_avg, dist, pxmm_mode)

	# the radial density profile plot.
	fig=plt.figure(figsize=(9,6))
	xlabel=f'Distance from center of plasma {unit}'
	ylabel='Intensity [arb]'
	ax=plt.gca()
	ax.grid(ls=':', color='0.65')
	ax.tick_params(axis="x", labelsize=fs3)
	ax.tick_params(axis="y", labelsize=fs3)
	ax.tick_params(direction='in',which='both',top=1,right=1)
	ax.xaxis.labelpad=15
	ax.yaxis.labelpad=15

	center = params_avg[1]
	stddev = abs(params_avg[2])
	plt.plot(dist, rIntensity_0[:min_length],'o', color='blue', markersize=pointsize, label = f"0\u00B0")
	plt.plot(dist, rIntensity_180[:min_length],'o', color='red', markersize=pointsize, label = "180\u00B0")
	plt.plot(dist, rIntensity_avg[:min_length],'o', color='black', markersize=pointsize, label = "Average")
	plt.plot(fit_dist_0, fit_intensity_0, label=f'0\u00B0 Fitted', color='blue')
	plt.plot(fit_dist_180, fit_intensity_180, label=f'180\u00B0 Fitted', color='red')
	plt.plot(fit_dist_avg, fit_intensity_avg, label='Average Fitted', color='black')
	plt.ylabel(ylabel,fontsize=fs2) 
	plt.xlabel(xlabel,fontsize=fs2)
	plt.ylim(bottom=0)
	plt.xlim(left=max(0,center-10.*stddev), right=center+10.*stddev)
	plt.legend(fontsize='xx-large') 
	plt.tight_layout()

	fname_radial=fname+'_radial_major_'+unit
	plt.savefig(fname_radial+'.png',format='png',dpi=500)

	# write to file
	fname_data='radial_profile/' + fname_temp
	print("Saving... {}".format(fname_data))
	with open(fname_data + '_major_' + unit + '.txt', 'w+') as fp_data:
		fp_data.write(f'#Average distance from center {unit}\tIntensity [arb]\n')
		for i in range(min_length):
			fp_data.write(str(dist[i])+'\t\t\t\t\t\t'+str(rIntensity_avg[i])+'\n')


def plot_minor(xc_set=xc, yc_set=yc, theta_set=theta, pxmm_mode=1):
	'''
	Plot intensity along both minor axes and their averaged value.
	
	xc_set (float, optional): X axis of center of Be+ elliptical fit, in pixels.
	yc_set (float, optional): Y axis of center of Be+ elliptical fit, in pixels.
	theta_set (float, optional): Angle of rotation of elliptical fit with respect to x axis,
								 in radians.
	pxmm_mode (int, optional): 0: plot with pixel; 1: plot with millimeter
							   Default with mode 0 (pixel).
	'''
	
	rIntensity_90, dist_90 = axis_intensity(90, xc_set=xc, yc_set=yc, theta_set=theta, pxmm_mode=pxmm_mode)
	rIntensity_270, dist_270 = axis_intensity(270, xc_set=xc, yc_set=yc, theta_set=theta, pxmm_mode=pxmm_mode)
	min_length = len(rIntensity_90)
	dist = rIntensity_90
	if len(rIntensity_90) > len(rIntensity_270):
		min_length = len(rIntensity_270)
		dist = dist_270
	rIntensity_avg = np.array([rIntensity_90[i] + rIntensity_270[i] for i in range(min_length)])/2.

	# plot with px or mm
	global unit
	if pxmm_mode == 1:
		unit = "[mm]"

	# fitting
	fit_intensity_90, fit_dist_90, params_90 = gaussian_fitting(rIntensity_90, dist, pxmm_mode)
	fit_intensity_270, fit_dist_270, params_270 = gaussian_fitting(rIntensity_270, dist, pxmm_mode)
	fit_intensity_avg, fit_dist_avg, params_avg = gaussian_fitting(rIntensity_avg, dist, pxmm_mode)

	# the radial density profile plot.
	fig=plt.figure(figsize=(9,6))
	xlabel=f'Distance from center of plasma {unit}'
	ylabel='Intensity [arb]'
	ax=plt.gca()
	ax.grid(ls=':', color='0.65')
	ax.tick_params(axis="x", labelsize=fs3)
	ax.tick_params(axis="y", labelsize=fs3)
	ax.tick_params(direction='in',which='both',top=1,right=1)
	ax.xaxis.labelpad=15
	ax.yaxis.labelpad=15

	center = params_avg[1]
	stddev = abs(params_avg[2])
	plt.plot(dist, rIntensity_90[:min_length],'o', color='blue', markersize=pointsize, label = f"90\u00B0")
	plt.plot(dist, rIntensity_270[:min_length],'o', color='red', markersize=pointsize, label = "270\u00B0")
	plt.plot(dist, rIntensity_avg[:min_length],'o', color='black', markersize=pointsize, label = "Averaged")
	plt.plot(fit_dist_90, fit_intensity_90, label=f'90\u00B0 Fitted', color='blue')
	plt.plot(fit_dist_270, fit_intensity_270, label=f'270\u00B0 Fitted', color='red')
	plt.plot(fit_dist_avg, fit_intensity_avg, label='Average Fitted', color='black')
	plt.ylabel(ylabel,fontsize=fs2) 
	plt.xlabel(xlabel,fontsize=fs2)
	plt.ylim(bottom=0)
	plt.xlim(left=max(0,center-10.*stddev), right=center+10.*stddev)
	plt.legend(fontsize='xx-large') 
	plt.tight_layout()

	fname_radial=fname+'_radial_minor_'+unit
	plt.savefig(fname_radial+'.png',format='png',dpi=500)
	# plt.savefig(fname_radial+'.pdf',format='pdf',dpi=500)

	# write to file
	fname_data='radial_profile/' + fname_temp
	print("Saving... {}".format(fname_data))
	with open(fname_data + '_minor_' + unit + '.txt', 'w+') as fp_data:
		fp_data.write(f'#Average distance from center {unit}\tIntensity [arb]\n')
		for i in range(min_length):
			fp_data.write(str(dist[i])+'\t\t\t\t\t\t'+str(rIntensity_avg[i])+'\n')

def new_ellipse_fit():
	'''
	Re-fit and plot the ellipse using the coordinates where intensity peaks 
	'''
	sample_points = []
	for ang in range(0, 360, 45):
		_, _, fit_params = gaussian_fitting(*axis_intensity(ang, pxmm_mode=0), pxmm_mode=0)
		new_x = xc + fit_params[1]*np.sin(theta+ang*np.pi/180.)
		new_y = yc + fit_params[1]*np.cos(theta+ang*np.pi/180.)
		# prevent unreasonable fit
		if 0 <= new_x <= m and 0 <= new_y <= n:
			sample_points.append(np.array([new_x, new_y]))
	sample_points = np.array(sample_points)
	model_new = EllipseModel()
	model_new.estimate(sample_points)
	xc_new, yc_new, a_new, b_new, theta_new = model_new.params
	print("New Elliptical fit parameters:")
	print(f"Center = ({xc_new}, {yc_new})")
	print(f"Semi-axes = ({a_new}, {b_new})")
	print(f"Theta = {theta_new}")

	# plotting
	fig=plt.figure(figsize=(9,6))
	cutfac=1
	im=plt.imshow(I,vmax=(np.amax(I)/cutfac))
	plt.set_cmap('jet')
	ellipse_new = Ellipse((xc_new, yc_new), width=2*a_new, height=2*b_new, 
						   angle=theta_new*180./np.pi, edgecolor='red', facecolor='none')
	plt.gca().add_patch(ellipse_new)
	# print("sample points: ", sample_points)
	# plt.scatter(sample_points[:,0], sample_points[:,1], color='black', label='Data Points')
	ax=plt.gca()
	ax.tick_params(axis="x", labelsize=fs3)
	ax.tick_params(axis="y", labelsize=fs3)
	plt.xlabel(f'Horizontal Distance on MCP [mm]',fontsize=fs2)
	plt.ylabel(f'Vertical Distance on MCP [mm]',fontsize=fs2)
	plt.colorbar()
	plt.tight_layout()

	fname_fitted=fname+'_MCP_newfit'
	plt.savefig(fname_fitted+'.png',format='png',dpi=500)
	
	fname_ellipse='new_ellipse_params/' + run_name
	with open(fname_ellipse + '.txt','w+') as ellipse_fit:
		ellipse_fit.write('#xc\t#yc\t#a\t#b\t#theta\n')
		ellipse_fit.write(str(xc_new)+'\t'+str(yc_new)+'\t'+str(a_new)+'\t'+str(b_new)+'\t'+str(theta_new)+'\n')
	return model_new.params


def average_intensity_along_ellipses(xc, yc, a, b, theta, pxmm_mode=1):
	'''
	Plot intensity along both minor axes and their averaged value.
	
	xc (float): X axis of center of Be+ elliptical fit, in pixels.
	yc (float): Y axis of center of Be+ elliptical fit, in pixels.
	a (float): major axis length of Be+ elliptical fit, in pixels.
	b (float): minor axis length of Be+ elliptical fit, in pixels.
	theta (float): Angle of rotation of elliptical fit with respect to x axis,
								 in radians.
	pxmm_mode (int, optional): 0: plot with pixel; 1: plot with millimeter
							   Default with mode 0 (pixel).
	'''
	axis_ratio = float(b)/float(a)
	avg_intensity = []
	t = np.linspace(0, 2 * np.pi, 1000)
	startpoint = 1.
	endpoint = startpoint + (a - startpoint)*2.
	step = 1.
	a_array = np.arange(startpoint, endpoint, step)
	for cur_a in a_array:
		cur_b = cur_a*axis_ratio
	
		# Parametric equations of the ellipse
		x = xc + cur_a * np.cos(t) * np.cos(theta) - cur_b * np.sin(t) * np.sin(theta)
		y = yc + cur_a * np.cos(t) * np.sin(theta) + cur_b * np.sin(t) * np.cos(theta)
		
		# Round to nearest integer indices
		x_indices = np.round(x).astype(int)
		y_indices = np.round(y).astype(int)
		
		# Ensure indices are within array bounds
		valid_mask = (x_indices >= 0) & (x_indices < I.shape[1]) & (y_indices >= 0) & (y_indices < I.shape[0])
		x_indices = x_indices[valid_mask]
		y_indices = y_indices[valid_mask]
		
		# Extract intensity values
		intensity_values = I[y_indices, x_indices]

		# Calculate and return the average intensity
		avg_intensity.append(np.mean(intensity_values))
	
	avg_intensity = np.array(avg_intensity)
	# plot with px or mm
	global unit
	if pxmm_mode == 1:
		unit = "[mm]"
		a_array *= mm_per_px_calib

	# plotting
	fig=plt.figure(figsize=(9,6))
	xlabel=f'Major axis distance from center {unit}'
	ylabel='Average Intensity [arb]'
	ax=plt.gca()
	ax.grid(ls=':', color='0.65')
	ax.tick_params(axis="x", labelsize=fs3)
	ax.tick_params(axis="y", labelsize=fs3)
	ax.tick_params(direction='in',which='both',top=1,right=1)
	ax.xaxis.labelpad=15
	ax.yaxis.labelpad=15

	plt.plot(a_array, avg_intensity,'o', color='black', markersize=pointsize)
	plt.ylabel(ylabel,fontsize=fs2) 
	plt.xlabel(xlabel,fontsize=fs2)
	plt.ylim(bottom=0)
	plt.xlim(left=0)
	plt.tight_layout()

	fname_radial=fname+'_avg_major'
	plt.savefig(fname_radial+'.png',format='png',dpi=500)
	# plt.savefig(fname_radial+'.pdf',format='pdf',dpi=500)

	# write to file
	fname_data='radial_profile/'+fname_temp
	print("Saving... {}".format(fname_data))
	with open(fname_data + '_avg_major_' + unit + '.txt', 'w+') as fp_data:
		fp_data.write(f'#Distance from center along major axis {unit}\t Average Intensity [arb]\n')
		for i in range(len(avg_intensity)):
			fp_data.write(str(a_array[i])+'\t\t\t\t\t\t'+str(avg_intensity[i])+'\n')
	
	return avg_intensity, a_array


# Hollowness calculation
# H = 0.93 for constant density ellipsoid
# H -> 1 for thin annulus
def calculate_hollowness(intensity, r):
	'''
	Compute the hollowness of Be+ profile using averaged intensity along concentric ellipses
	
	intensity (float array): averaged intensity along concentric ellipses 
	r (float array): major axis of concentric ellipses
	'''
	# normalization
	r0 = (2.*np.pi*r)*intensity
	normalization_factor = 1./np.trapz(r0, r)
	r2 = r0*(r**2)*normalization_factor
	r4 = r0*(r**4)*normalization_factor
	avg_r2 = np.trapz(r2, r)
	avg_r4 = np.trapz(r4, r)
	Hollowness = (avg_r2**0.5)/(avg_r4**0.25)
	print("Hollowness H = {:.5f}".format(Hollowness))
	return Hollowness


def ellipse_to_circle_transform():
	"""
	Rotate the ellipse to align with x, y axis and scale y-coordinates to transform it into circle

	Returns:
	- Transformed image as a 2D numpy array.
	"""
	xc_new, yc_new, a_new, b_new, theta_new = new_ellipse_fit()
	rows, cols = I.shape

	# Rotate
	I_rotated = np.zeros((rows, cols))
	c = np.cos(-theta_new)
	s = np.sin(-theta_new)
	x0 = cols/2 - c*cols/2 - s*rows/2
	y0 = rows/2 - c*rows/2 + s*cols/2
	for y in range(rows):
		for x in range(cols):
			src_x = int(c*x + s*y + x0)
			src_y = int(-s*x + c*y + y0)
			if cols > src_x >= 0 and rows > src_y >= 0:
				I_rotated[y][x] = I[src_y][src_x]

	# Create the coordinate grid
	x = np.arange(cols)
	y = np.arange(rows)
	xv, yv = np.meshgrid(x, y)

	# Scale coordinates
	# f_scale > 0 is compression
	f_scale = b_new/a_new
	yv_scaled = yv*f_scale
	print("Scale Factor= ", f_scale)

	# Interpolate the values from the original image to the new coordinates
	coordinates = [yv_scaled.ravel(), xv.ravel()]
	rescaled_image = map_coordinates(I_rotated, coordinates, order=1, mode='nearest').reshape(rows, cols)
	
	# fit
	xc_circle = (c*xc_new - s*yc_new - c*x0 + s*y0)/(c*c + s*s)
	yc_circle = (s*xc_new + c*yc_new - c*y0 - s*x0)/(c*c + s*s)/f_scale
	r_circle = a_new
	print("Circular Fit:")
	print(f"Center: {xc_circle, yc_circle}")
	print(f"Radius: {r_circle}")

	# plot
	fig=plt.figure(figsize=(9,6))
	cutfac=1
	im=plt.imshow(rescaled_image,vmax=(np.amax(rescaled_image)/cutfac))
	plt.set_cmap('jet')
	circle = plt.Circle((xc_circle, yc_circle), r_circle, color='red', fill=False)
	plt.gca().add_patch(circle)
	ax=plt.gca()
	ax.tick_params(axis="x", labelsize=fs3)
	ax.tick_params(axis="y", labelsize=fs3)
	plt.ylabel("Pixel Scaled by {:.2f}".format(1/f_scale),fontsize=fs2)
	plt.colorbar()
	plt.tight_layout()

	fname_circular=fname+'_circular_fit'
	plt.savefig(fname_circular+'.png',format='png',dpi=500)

	fname_circle='new_circle_params/' + run_name
	with open(fname_circle + '.txt','w+') as circle_fit:
		circle_fit.write('#xc\t#yc\t#r\n')
		circle_fit.write(str(xc_circle)+'\t'+str(yc_circle)+'\t'+str(r_circle)+'\n')

	return rescaled_image, xc_circle, yc_circle, a

def average_intensity_along_circles(rescaled_image, xc_circle, yc_circle, r, pxmm_mode=1):
	'''
	Plot intensity along both minor axes and their averaged value.
	
	rescaled_image (float array): 2D array for circular image after rescaling
	xc_circle (float): X axis of center of Be+ circular fit, in pixels.
	yc_circle (float): Y axis of center of Be+ circular fit, in pixels.
	r (float): Radius of Be+ circular fit, in pixels.
	pxmm_mode (int, optional): 0: plot with pixel; 1: plot with millimeter
							   Default with mode 0 (pixel).
	'''
	avg_intensity = []
	t = np.linspace(0, 2 * np.pi, 1000)
	startpoint = 1.
	endpoint = startpoint + (r - startpoint)*3.
	step = 1.
	r_array = np.arange(startpoint, endpoint, step)
	for r_val in r_array:
		# Parametric equations of circle
		x = xc_circle + r_val*np.cos(t)
		y = yc_circle + r_val*np.sin(t)	

		# Round to nearest integer indices
		x_indices = np.round(x).astype(int)
		y_indices = np.round(y).astype(int)
		
		# Ensure indices are within array bounds
		valid_mask = (x_indices >= 0) & (x_indices < rescaled_image.shape[1]) & (y_indices >= 0) & (y_indices < rescaled_image.shape[0])
		x_indices = x_indices[valid_mask]
		y_indices = y_indices[valid_mask]
		
		# Extract intensity values
		intensity_values = rescaled_image[y_indices, x_indices]

		# Calculate and return the average intensity
		avg_intensity.append(np.mean(intensity_values))
	
	avg_intensity = np.array(avg_intensity)
	# plot with px or mm
	global unit
	if pxmm_mode == 1:
		unit = "[mm]"
		r_array *= mm_per_px_calib

	# plotting
	fig=plt.figure(figsize=(9,6))
	xlabel=f'Radial distance from center {unit}'
	ylabel='Average Intensity [arb]'
	ax=plt.gca()
	ax.grid(ls=':', color='0.65')
	ax.tick_params(axis="x", labelsize=fs3)
	ax.tick_params(axis="y", labelsize=fs3)
	ax.tick_params(direction='in',which='both',top=1,right=1)
	ax.xaxis.labelpad=15
	ax.yaxis.labelpad=15

	plt.plot(r_array, avg_intensity,'o', color='black', markersize=pointsize)
	plt.ylabel(ylabel,fontsize=fs2) 
	plt.xlabel(xlabel,fontsize=fs2)
	plt.ylim(bottom=0)
	plt.xlim(left=0)
	plt.tight_layout()

	fname_radial=fname+'_avg_radial'
	plt.savefig(fname_radial+'.png',format='png',dpi=500)

	# write to file
	fname_data='radial_profile/'+fname_temp
	print("Saving... {}".format(fname_data))
	with open(fname_data + '_avg_radial_' + unit + '.txt', 'w+') as fp_data:
		fp_data.write(f'#Radial Distance {unit}\t Average Intensity [arb]\n')
		for i in range(len(avg_intensity)):
			fp_data.write(str(r_array[i])+'\t\t\t\t\t\t'+str(avg_intensity[i])+'\n')
	
	return avg_intensity, r_array


## Function Calls Below

# calculate_hollowness(*average_intensity_along_ellipses(*new_ellipse_fit(), pxmm_mode=1))
# calculate_hollowness(*average_intensity_along_circles(*ellipse_to_circle_transform()))
plot_major()

