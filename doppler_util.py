import matplotlib.pyplot as plt
import numpy as np
import cataclysmic as cv
#reload(cv)
import matplotlib.cm as cm
import glob
import mynormalize
import os
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import ICRS,SkyCoord
from numpy import ma
import matplotlib.collections as mcoll
import matplotlib.path as mpath

'''
To make text files out of the fits
doppler.export('*_UVB.fits',out_dir='molly_hbeta',wave=[4711,5010])
doppler.export('*_UVB.fits',out_dir='molly_halfa',wave=[6540,6600])
'''

def make(filer,cmaps = cm.Greys_r,lc = 'white',limits=None,colorbar=False,negative=False):

	data = load_map(filer)
	data.data[data.data == 0] = np.nan
	if limits == None: limits = [data.data.min(),data.data.max()]
	fig = plt.figure(num='Doppler Map',figsize=(10,10))
	plt.clf()
	#ax=fig.add_subplot(111)
	#ax.set_aspect('equal')
	ll = ~np.isnan(data.data)
	#print data.data
	if negative:
		img = plt.imshow(-data.data/data.data[ll].max(),interpolation='nearest', cmap=cmaps,aspect='equal',origin='lower',extent=(min(data.axes[0].pos), max(data.axes[0].pos),min(data.axes[1].pos), max(data.axes[1].pos) ),vmin=-limits[1],vmax=-limits[0] )
	else:
		img = plt.imshow(data.data/data.data[ll].max(),interpolation='nearest', cmap=cmaps,aspect='equal',origin='lower',extent=(min(data.axes[0].pos), max(data.axes[0].pos),min(data.axes[1].pos), max(data.axes[1].pos) ),vmin=limits[0],vmax=limits[1] )

	axlimits=[min(data.axes[0].pos), max(data.axes[0].pos),min(data.axes[1].pos),max(data.axes[1].pos) ]
	plt.axis(axlimits)
	#plt.axvline(x=0.0,linestyle='--',color='white')

	plt.xlabel('V$_x$ / km s$^{-1}$')
	plt.ylabel('V$_y$ / km s$^{-1}$')
	plt.tight_layout()
	plt.show()
	if colorbar:
		cbar = plt.colorbar(format='%05.2f')
		cbar.set_label('Normalised Flux')
		cbar.set_norm(mynormalize.MyNormalize(vmin=limits[0],vmax=limits[1],stretch='linear'))
		cbar = cv.DraggableColorbar(cbar,img)
	else:
		cbar=1
	return cbar

def reamap(dopout = 'dop.out',cmaps = cm.Greys_r,lc = 'white',limits=None,
			colorbar=False,negative=False,remove_mean=False, corrx=0,corry=0):
	"""
	Read output files from Henk Spruit's *.out and plot a Doppler map
	"""
	f=open(dopout)
	lines=f.readlines()
	f.close()

	#READ ALL FILES
	nph,nvp,nv,w0,aa=int(lines[0].split()[0]),int(lines[0].split()[1]),int(lines[0].split()[2]),float(lines[0].split()[3]),float(lines[0].split()[4])
	gamma,abso,atm,dirin=float(lines[1].split()[0]),lines[1].split()[1],lines[1].split()[2],lines[1].split()[3]

	flag=0
	for i in np.arange(3,len(lines),1):
	    if flag==0:
	        temp=lines[i-1]+lines[i]
	        flag=1

	    else:
	        temp=temp+lines[i]
	war=temp.split()

	pha=np.array(war[:nph]).astype(np.float)/2.0/np.pi
	dum1=war[nph]
	dpha=np.array(war[nph+1:nph+1+nph]).astype(np.float)/2.0/np.pi
	last=nph+1+nph
	vp=np.array(war[last:last+nvp]).astype(np.float)
	dvp=vp[1]-vp[0]
	vp=vp-dvp/2.0
	last=last+nvp
	dm=np.array(war[last:last+nvp*nph]).astype(np.float)
	dm=dm.reshape(nvp,nph)
	last=last+nvp*nph



	ih,iw,pb0,pb1,ns,ac,al,clim,norm,wid,af=int(war[last]),int(war[last+1]),float(war[last+2]),float(war[last+3]),int(war[last+4]),float(war[last+5]),float(war[last+6]),float(war[last+7]),int(war[last+8]),float(war[last+9]),float(war[last+10])
	nv,va,dd=int(war[last+11]),float(war[last+12]),war[last+13]
	last=last+14

	im=np.array(war[last:last+nv*nv]).astype(np.float)
	im=im.reshape(nv,nv)

	last=last+nv*nv
	ndum,dum2,dum3=int(war[last]),war[last+1],war[last+2]
	last=last+3
	dmr=np.array(war[last:last+nvp*nph]).astype(np.float)
	dmr=dmr.reshape(nvp,nph)
	last=last+nvp*nph
	ndum,dum4,dum2,dum3=int(war[last]),int(war[last+1]),war[last+2],war[last+3]
	last=last+4
	dpx=np.array(war[last:last+nv*nv]).astype(np.float)
	dpx=dpx.reshape(nv,nv)
	dpx = np.array(dpx)
	vp = np.array(vp)/1e5
	data = im

	data[data == 0.0] = np.nan

	#data = ma.masked_where(dat > 0.0, dat)
	#cmaps.set_bad(alpha = 0.0)
	#cmaps.set_under('w', 0.0)
	if limits == None: limits = [data.min(),data.max()]

	# Here comes the plotting
	fig = plt.figure(num='Doppler Map',figsize=(8.57,8.57))
	plt.clf()
	ax = fig.add_subplot(111)

	ll = ~np.isnan(data)

	if remove_mean:
		rad_prof = radial_profile(data,[data[0].size/2-corrx,data[0].size/2-corry])
		meano = create_profile(data,rad_prof,[data[0].size/2-corrx,data[0].size/2-corry])
		qq = ~np.isnan(data - meano)
	if negative:
		if remove_mean:
			#print data[ll].max(),meano[qq].max()
			img = plt.imshow((data - meano)/(data - meano)[qq].max(),interpolation='nearest', cmap=cmaps,aspect='equal',origin='lower',extent=(min(vp), max(vp),min(vp), max(vp) ),vmin=limits[0],vmax=limits[1])
		else:
			img = plt.imshow(-(data)/data[ll].max(),interpolation='nearest', cmap=cmaps,aspect='equal',origin='lower',extent=(min(vp), max(vp),min(vp), max(vp)),vmin=-limits[1],vmax=-limits[0] )
	else:
		if remove_mean:
			#print data[ll].max(),meano[qq].max()
			img = plt.imshow((data - meano)/(data - meano)[qq].max(),interpolation='nearest', cmap=cmaps,aspect='equal',origin='lower',extent=(min(vp), max(vp),min(vp), max(vp) ),vmin=limits[0],vmax=limits[1])
		else:
			img = plt.imshow(data/data[ll].max(),interpolation='nearest', cmap=cmaps,aspect='equal',origin='lower',extent=(min(vp), max(vp),min(vp), max(vp) ),vmin=limits[0],vmax=limits[1] )

	axlimits=[min(vp), max(vp),min(vp), max(vp) ]
	plt.axis(axlimits)
	#plt.axvline(x=0.0,linestyle='--',color='white')

	plt.xlabel('V$_x$ / km s$^{-1}$')
	plt.ylabel('V$_y$ / km s$^{-1}$')
	plt.tight_layout()
	plt.show()
	if colorbar:
		cbar = plt.colorbar(format='%05.2f',orientation='vertical',fraction=0.046, pad=0.04)
		cbar.set_label('Normalised Flux')
		cbar.set_norm(mynormalize.MyNormalize(vmin=limits[0],vmax=limits[1],stretch='linear'))
		cbar = cv.DraggableColorbar(cbar,img)
	else:
		cbar=1


	#print vp.shape, pha,dmr.shape, dpha

	####################################################


	'''
	if remove_mean:
		#print data.size/2
		rad_prof = radial_profile(data,[data[0].size/2,data[0].size/2])
		mean = create_profile(data,rad_prof,[data[0].size/2,data[0].size/2])
		ll = ~np.isnan(mean)
		fig = plt.figure('Mean')
		plt.clf()
		fig.add_subplot(211)
		plt.plot(rad_prof)
		fig.add_subplot(212)

		plt.show()
	'''
	return cbar




def trail(dopout,lam0,bins,gamma=0.0,cmaps = cm.Greys_r,lc = 'white',
		  limits=None,xlim=None,colorbar=False,negative=False,vel=False,
		  reconstructed=False,label=None):
	"""
	Make a trail spectra from Spruit's doppler output map
	"""
	f=open(dopout)
	lines=f.readlines()
	f.close()

	#READ ALL FILES
	nph,nvp,nv,w0,aa=int(lines[0].split()[0]),int(lines[0].split()[1]),int(lines[0].split()[2]),float(lines[0].split()[3]),float(lines[0].split()[4])
	gamma,abso,atm,dirin=float(lines[1].split()[0]),lines[1].split()[1],lines[1].split()[2],lines[1].split()[3]

	flag=0
	for i in np.arange(3,len(lines),1):
	    if flag==0:
	        temp=lines[i-1]+lines[i]
	        flag=1

	    else:
	        temp=temp+lines[i]
	war=temp.split()

	pha=np.array(war[:nph]).astype(np.float)/2.0/np.pi
	dum1=war[nph]
	dpha=np.array(war[nph+1:nph+1+nph]).astype(np.float)/2.0/np.pi
	last=nph+1+nph
	vp=np.array(war[last:last+nvp]).astype(np.float)
	dvp=vp[1]-vp[0]
	vp=vp-dvp/2.0
	last=last+nvp
	dm=np.array(war[last:last+nvp*nph]).astype(np.float)
	dm=dm.reshape(nvp,nph)
	last=last+nvp*nph



	ih,iw,pb0,pb1,ns,ac,al,clim,norm,wid,af=int(war[last]),int(war[last+1]),float(war[last+2]),float(war[last+3]),int(war[last+4]),float(war[last+5]),float(war[last+6]),float(war[last+7]),int(war[last+8]),float(war[last+9]),float(war[last+10])
	nv,va,dd=int(war[last+11]),float(war[last+12]),war[last+13]
	last=last+14

	im=np.array(war[last:last+nv*nv]).astype(np.float)
	im=im.reshape(nv,nv)

	last=last+nv*nv
	ndum,dum2,dum3=int(war[last]),war[last+1],war[last+2]
	last=last+3
	dmr=np.array(war[last:last+nvp*nph]).astype(np.float)
	dmr=dmr.reshape(nvp,nph)
	last=last+nvp*nph
	ndum,dum4,dum2,dum3=int(war[last]),int(war[last+1]),war[last+2],war[last+3]
	last=last+4
	dpx=np.array(war[last:last+nv*nv]).astype(np.float)
	dpx=dpx.reshape(nv,nv)
	dpx = np.array(dpx)
	vp = np.array(vp)/1e5
	data = im

	data[data == 0.0] = np.nan

	#x,y = np.ogrid[0-.5/bins:2+.5/bins:1./bins,min(vp):max(vp):vp[-1]-vp[-2]]
	#print y
	zvals = np.zeros((bins*2+1,vp.size))
	#zvals = np.transpose(np.random.rand(len(np.transpose(y)),len(x))*0.)
	#print zvals.shape,zval.shape
	bineq=np.arange(0+.5/bins,2,1./bins)
	#print zvals.shape,len(vp),vp[-1]-vp[-2]
	if reconstructed:
		dm = dmr.T
	else:
		dm = dm.T
	histo=[]
	#print pha
	for i in np.arange(bins):
		temp=np.zeros(len(vp))
		lolo=0
		for j in np.arange(len(pha)):
			if bineq[i]>=1.0:
				flag=1
			if i==0:
				if pha[j] < bineq[0] or pha[j] >= 1-bineq[0] and flag !=1:
					#print pha[j]
					lolo=lolo+1

					temp = dm[j] + temp
					#temp_before = dm[j+bins] + temp_before
					#temp_after = dm[j+1] + temp_after
					tt=1
			if pha[j] >=bineq[i-1] and pha[j] < bineq[i] and i!=0:
				lolo=lolo+1
				temp = dm[j] + temp
				#temp_before = dm[j-1] + temp_before
				#temp_after = dm[j+1] + temp_after
		if lolo == 0: lolo=1.0
		temp = cv.savitzky_golay(temp,7,3)
		zvals[i]=temp/lolo#/np.median(temp/lolo)
		zvals[i+bins]=temp/lolo#/np.median(temp/lolo)
		if i == 0: zvals[-1]=temp/lolo#/np.median(temp/lolo)

	ll = np.isnan(zvals)
	zvals1=zvals
	zvals1[ll] = 0.0
	#print bineq
	maxi = bineq[bins]
	mini = bineq[0] - 1./bins
	#print bineq[bins],mini,maxi,maxi+2
	fig = plt.figure('Trail',figsize=(6.57,8.57))
	plt.clf()
	ax = fig.add_subplot(111)

	maxo = np.sort(zvals.flatten())
	maxo = np.median(maxo[int(maxo.size *0.98):])
	#print np.sum(zvals1/maxo > 1.0),zvals.flatten().size
	#print np.median(maxo[int(maxo.size *0.98):]),zvals.max()
	img2=plt.imshow(zvals1/maxo,interpolation='nearest', cmap=cmaps,aspect='auto',origin='lower',extent=(xlim[0], xlim[1],mini, maxi+1),vmin=limits[0],vmax=limits[1] )
	#img2=plt.imshow(zvals1/zvals.max(),interpolation='nearest', cmap=cmaps,aspect='auto',origin='lower',extent=(min(vp), max(vp),mini-2, maxi-1),vmin=limits[0],vmax=limits[1] )
	#img2=plt.imshow(zvals1/zvals.max(),interpolation='nearest', cmap=cmaps,aspect='auto',origin='lower',extent=(min(vp), max(vp),mini+2, maxi+3),vmin=limits[0],vmax=limits[1] )
	#img2=plt.imshow(dmr.T/dmr.max(),interpolation='nearest', cmap=cmaps,aspect='auto',origin='lower',extent=(min(vp), max(vp),min(pha), max(pha)),vmin=limits[0],vmax=limits[1] )
	plt.xlabel('Radial Velocity / km s$^{-1}$')
	plt.ylabel('Orbital Phase')
	#plt.ylim(-0.000000001,2.000000001)

	if colorbar:
		cbar = plt.colorbar(format='%05.2f',orientation='vertical', pad=0.04)
		cbar.set_label('Normalised Flux')
		cbar.set_norm(mynormalize.MyNormalize(vmin=limits[0],vmax=limits[1],stretch='linear'))
		cbar = cv.DraggableColorbar(cbar,img2)
	else:
		cbar=1
	if label != None: plt.text(0.16, 0.1,label, ha='center', va='center', transform=ax.transAxes)
	plt.tight_layout()

	return cbar

def trail_custom(grid,lam0,bins,delpha,flu='flux_norm',gamma=0.0,cmaps = cm.Greys_r,lc = 'white',
				limits=None, xlim=None, colorbar=False,negative=False,vel=False,
				reconstructed=False,label=None,plot_median=True):
	"""
	Make a trail spectra from Spruit's doppler output map
	"""


	if delpha > 2/bins: print("Warning: Spectrum phase is greater than bin, {:.2f}>{:.2f}".format(delpha,1/bins))
	phase = np.linspace(0,2,bins*2+1,endpoint=True) - 1./(bins)/2.

	phase = np.concatenate((phase,[2.0+1./(bins)/2.]))
	print(phase)
	if xlim == None:
	    trail = np.zeros((grid['wave'][0].size,phase.size))
	    rr = np.ones(grid['wave'][0].size,dtype='boolean')
	else:
	    rr = (grid['wave'][0]>xlim[0]) *(grid['wave'][0] < xlim[1])
	    trail = np.zeros((grid['wave'][0][rr].size,phase.size))
	tots = trail.copy()
	for i in range(grid['phase'].size):
	    #print(grid['phase'][i])
	    dist = phase - (grid['phase'][i]+delpha/2.)
	    #print(dist)
	    dist[np.abs(dist)>1./bins] = 0.
	    #print(dist)
	    dist[dist>0] = 0.0
	    #print(dist)
	    weights = np.abs(dist)/(1./bins)
	    #print(weights)
	    #print('---------------')
	    dist = phase - (grid['phase'][i]-delpha/2.)
	    #print(dist)
	    dist[np.abs(dist)>1./bins] = 0.0
	    #print(dist)
	    dist[dist>0] = 0.0
	    #print(dist)
	    dist[np.abs(dist)>0] = 1.0 - (np.abs(dist[np.abs(dist)>0]))/(1./bins)
	    weights += dist
	    #print(weights)
	    temp = trail.copy().T
	    for j in range(phase.size):
	        temp[j] =  grid[flu][i][rr] * weights[j]
	    trail+=temp.T
	    tots += weights
	    #stop
	    #rr
	    #tt = (grid['phase']+1 > phase[i])  * (grid['phase']+1 < phase[i+1])
	trail /= tots

	if plot_median:
		si = 12
		lo = 2
	else:
		si = 9
		lo = 0
	plt.figure(1,figsize=(16,si))
	plt.clf()
	if plot_median:
		ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=2)
		ax1.minorticks_on()
		plt.plot(grid['wave'][0][rr],np.nanmedian(grid[flu],axis=0)[rr],
				label='Median',color='#8e44ad')
		#plt.plot(grid['wave'][0][rr],np.nanmean(grid[flu],axis=0)[rr],
		#		label='Mean')
		lg = plt.legend(loc=2,fontsize=22)


		plt.axhline(y=0,ls='--',color='r',alpha=0.7)
		#plt.yscale('log')
		#ax.set_yticklabels([])
		ax1.set_xticklabels([])


		plt.xlim(xlim[0], xlim[1])
		#rr = (grid['wave'][0]>xlim[0]) *(grid['wave'][0] < xlim[1])
		plt.ylim(-0.35,np.nanmax(np.nanmedian(grid[flu],axis=0)[rr])*1.1)
		### Print trail spectra

		if limits == None:
			limits=[np.nanmax(np.nanmedian(grid[flu],axis=0)[rr])*0.35,
				np.nanmax(np.nanmedian(grid[flu],axis=0)[rr])*1.1]
	ax2 = plt.subplot2grid((6, 1), (lo, 0), rowspan=4+lo)
	ax2.minorticks_on()
	img = plt.imshow(trail.T,interpolation='nearest', cmap=plt.cm.magma_r,
	                 aspect='auto',origin='lower',
	                 extent=(min(grid['wave'][0][rr]),
	                         max(grid['wave'][0][rr]),phase[0],phase[-1]+1/bins),#
	                 vmin=limits[0],vmax=limits[1])
	plt.xlim(xlim[0], xlim[1])
	plt.ylim(phase[0],1+1/bins/2.)
	plt.ylabel('Orbital Phase')
	plt.xlabel('Wavelength / $\AA$')
	plt.tight_layout(h_pad=0)

	#return phase,trail.T



def export_trail(dopout,bins=10):

	f=open(dopout)
	lines=f.readlines()
	f.close()

	#READ ALL FILES
	nph,nvp,nv,w0,aa=int(lines[0].split()[0]),int(lines[0].split()[1]),int(lines[0].split()[2]),float(lines[0].split()[3]),float(lines[0].split()[4])
	gamma,abso,atm,dirin=float(lines[1].split()[0]),lines[1].split()[1],lines[1].split()[2],lines[1].split()[3]

	flag=0
	for i in np.arange(3,len(lines),1):
	    if flag==0:
	        temp=lines[i-1]+lines[i]
	        flag=1

	    else:
	        temp=temp+lines[i]
	war=temp.split()

	pha=np.array(war[:nph]).astype(np.float)/2.0/np.pi
	dum1=war[nph]
	dpha=np.array(war[nph+1:nph+1+nph]).astype(np.float)/2.0/np.pi
	last=nph+1+nph
	vp=np.array(war[last:last+nvp]).astype(np.float)
	dvp=vp[1]-vp[0]
	vp=vp-dvp/2.0
	last=last+nvp
	dm=np.array(war[last:last+nvp*nph]).astype(np.float)
	dm=dm.reshape(nvp,nph)
	last=last+nvp*nph



	ih,iw,pb0,pb1,ns,ac,al,clim,norm,wid,af=int(war[last]),int(war[last+1]),float(war[last+2]),float(war[last+3]),int(war[last+4]),float(war[last+5]),float(war[last+6]),float(war[last+7]),int(war[last+8]),float(war[last+9]),float(war[last+10])
	nv,va,dd=int(war[last+11]),float(war[last+12]),war[last+13]
	last=last+14

	im=np.array(war[last:last+nv*nv]).astype(np.float)
	im=im.reshape(nv,nv)

	last=last+nv*nv
	ndum,dum2,dum3=int(war[last]),war[last+1],war[last+2]
	last=last+3
	dmr=np.array(war[last:last+nvp*nph]).astype(np.float)
	dmr=dmr.reshape(nvp,nph)
	last=last+nvp*nph
	ndum,dum4,dum2,dum3=int(war[last]),int(war[last+1]),war[last+2],war[last+3]
	last=last+4
	dpx=np.array(war[last:last+nv*nv]).astype(np.float)
	dpx=dpx.reshape(nv,nv)
	dpx = np.array(dpx)
	vp = np.array(vp)/1e5
	data = im

	data[data == 0.0] = np.nan

	#print zvals.shape,len(vp),vp[-1]-vp[-2]

	dm = dm.T
	zvals = np.zeros((bins*2+1,vp.size))
	#zvals = np.transpose(np.random.rand(len(np.transpose(y)),len(x))*0.)
	#print zvals.shape,zval.shape
	bineq=np.arange(0+.5/bins,2,1./bins)
	for i in np.arange(bins):
		temp=np.zeros(len(vp))
		lolo=0
		for j in np.arange(len(pha)):
			if bineq[i]>=1.0:
				flag=1
			if i==0:
				if pha[j] < bineq[0] or pha[j] >= 1-bineq[0] and flag !=1:
					#print pha[j]
					lolo=lolo+1

					temp = dm[j] + temp
					#temp_before = dm[j+bins] + temp_before
					#temp_after = dm[j+1] + temp_after
					tt=1
			if pha[j] >=bineq[i-1] and pha[j] < bineq[i] and i!=0:
				lolo=lolo+1
				temp = dm[j] + temp
				#temp_before = dm[j-1] + temp_before
				#temp_after = dm[j+1] + temp_after
		if lolo == 0: lolo=1.0
		temp = cv.savitzky_golay(temp,7,3)
		zvals[i]=temp/lolo/np.median(temp/lolo)
		zvals[i+bins]=temp/lolo/np.median(temp/lolo)
		if i == 0: zvals[-1]=temp/lolo/np.median(temp/lolo)

	ll = np.isnan(zvals)
	zvals1=zvals
	zvals1[ll] = 0.0
	return vp,pha,dm,bineq,zvals1

def iso_idl(dopout = 'dop.out',cmaps = cm.Greys_r,lc = 'white',):
	'''
	Creates Iso-countours and plots on the tomograms
	'''

	f=open(dopout)
	lines=f.readlines()
	f.close()

	#READ ALL FILES
	nph,nvp,nv,w0,aa=int(lines[0].split()[0]),int(lines[0].split()[1]),int(lines[0].split()[2]),float(lines[0].split()[3]),float(lines[0].split()[4])
	gamma,abso,atm,dirin=float(lines[1].split()[0]),lines[1].split()[1],lines[1].split()[2],lines[1].split()[3]

	flag=0
	for i in np.arange(3,len(lines),1):
	    if flag==0:
	        temp=lines[i-1]+lines[i]
	        flag=1

	    else:
	        temp=temp+lines[i]
	war=temp.split()

	pha=np.array(war[:nph]).astype(np.float)/2.0/np.pi
	dum1=war[nph]
	dpha=np.array(war[nph+1:nph+1+nph]).astype(np.float)/2.0/np.pi
	last=nph+1+nph
	vp=np.array(war[last:last+nvp]).astype(np.float)
	dvp=vp[1]-vp[0]
	vp=vp-dvp/2.0
	last=last+nvp
	dm=np.array(war[last:last+nvp*nph]).astype(np.float)
	dm=dm.reshape(nvp,nph)
	last=last+nvp*nph



	ih,iw,pb0,pb1,ns,ac,al,clim,norm,wid,af=int(war[last]),int(war[last+1]),float(war[last+2]),float(war[last+3]),int(war[last+4]),float(war[last+5]),float(war[last+6]),float(war[last+7]),int(war[last+8]),float(war[last+9]),float(war[last+10])
	nv,va,dd=int(war[last+11]),float(war[last+12]),war[last+13]
	last=last+14

	im=np.array(war[last:last+nv*nv]).astype(np.float)
	im=im.reshape(nv,nv)

	last=last+nv*nv
	ndum,dum2,dum3=int(war[last]),war[last+1],war[last+2]
	last=last+3
	dmr=np.array(war[last:last+nvp*nph]).astype(np.float)
	dmr=dmr.reshape(nvp,nph)
	last=last+nvp*nph
	ndum,dum4,dum2,dum3=int(war[last]),int(war[last+1]),war[last+2],war[last+3]
	last=last+4
	dpx=np.array(war[last:last+nv*nv]).astype(np.float)
	dpx=dpx.reshape(nv,nv)
	dpx = np.array(dpx)
	vp = np.array(vp)/1e5
	data = im

	data[data == 0.0] = np.nan
	ll = ~np.isnan(data)
	fig = plt.figure( num='Doppler Map')

	#print va
	#print data.shape,X.shape,Y.shape,nvp
	vxy = (2.0 * np.arange(nv)/(nv-1)-1)*va/1e5
	dv=vxy[1]-vxy[0]
	#vxy=np.concatenate(([vxy[0]-dv/2.],vxy+dv/2.))
	#print vxy,nv

	X, Y = np.meshgrid(vxy, vxy)
	CS = plt.contour(X, Y, data/data[ll].max(), 6, colors='gray')
	plt.clabel(CS, fontsize=9, inline=1)
	plt.draw()

def iso(data):
	'''
	Creates Iso-countours and plots on the tomograms
	'''


	ll = ~np.isnan(data.data)
	fig = plt.figure( num='Doppler Map')
	X, Y = np.meshgrid(data.axes[0].pos, data.axes[1].pos)
	CS = plt.contour(X, Y, data.data/data.data[ll].max(), 6, colors='gray')
	plt.clabel(CS, fontsize=9, inline=1)
	plt.draw()


def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def create_profile(data,profile, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)
    mean = data*0.0 + 1.0
    for i in np.arange(r.max()):
    	#print i
    	ss = np.where(r == i)
    	#print profile[i]
    	mean[ss] = mean[ss] * profile[i]
    ll = ~np.isnan(mean)
    #print mean[ll]
    return mean

def savefig(dir='.',name='doppler_map',remove_mean=False):
	'''
	Saves Doppler Tomograms in png and pdf formats
	'''
	if remove_mean: name += '_mean'
	fig = plt.figure(num='Doppler Map')
	plt.savefig(dir+'/'+name+'_dopmap.png')
	plt.savefig(dir+'/'+name+'_dopmap.pdf')
	fig = plt.figure( num='Trail')
	plt.savefig(dir+'/'+name+'_trail.png')
	plt.savefig(dir+'/'+name+'_trail.pdf')

def __export_legacy__(flns, out_dir = 'molly', out_list = 'spec_ascii.lis',mol_in = 'spec_ascii.lis',mol_info = 'spec_ascii.lis',wave=[6450,6750]):
	'''
	Converts a sequence of fits files to text files.

	flns 		- Sequence of spectra to export. '*_VIS.fits'
	out_dir		- Output directory for all text files
	out_list	- List of ouput spectra. It changes the fits for txt
	wave 		- Wavelength range to extract in Angstrom [w_1,w_2], where w_2 > w_1.
	'''
	if not os.path.exists(out_dir):
		os.system('mkdir '+out_dir)

	files = glob.glob(flns)
	print ('Total Spectra: '+str(len(files)))
	counter=1
	fsock = open(out_dir+'/'+out_list,'w')
	gsock = open(out_dir+'/'+out_list,'w')
	for i in files:
		cv.Printer(('%d - %30s')%(counter,i))
		cv.export_xshooter(i,num=counter,dire = out_dir,wa=wave)
		counter+=1
		fsock.write(i[:-4]+'txt')
	fsock.close()

	### Make Parameter file
	fsock = open(out_dir+'/'+'params_molly.lis','w')
	fsock.write('OBJECT')
	fsock.write('NUM_IMAG')
	fsock.write('DATE-OBS')
	fsock.write('OPENTIME')
	fsock.write('EXPTIME')
	fsock.write('RA')
	fsock.write('DEC')
	fsock.write('EPOCH')
	fsock.close()



def map_trm(fits_file,limits=None,remove_mean=False,negative=False,cmaps=plt.cm.viridis,colorbar=True):
	"""
	Reads the output from Tom Marsh's Doppler and plots the Tomogram in a fancy way

	Attributes::

		fits_file : Input fits file generated by memit.py

		limits : 2-element list containing lower and upper limits for image scale e.g. [0.1,0.9]; Default None

		remove_mean : Removes the azhimuthal average to enchance assymetries in the disc

		negative : Inverts the color map (No use, should remove soon)

		cmaps : Color of the tomogram. Default: plt.cm.viridis (magma is better imho)

		colorbar : Show colorbar
	"""
	data = fits.getdata(fits_file)
	crpix1 = fits.getval(fits_file,'CRPIX1',1)
	cdelt1 = fits.getval(fits_file,'CDELT1',1)
	crval1 = fits.getval(fits_file,'CRVAL1',1)
	vp = (np.arange(data.shape[0]) - crpix1) * cdelt1 + crval1

	# Here comes the plotting
	fig = plt.figure(num='Doppler Map',figsize=(8.57,8.57))
	plt.clf()
	ax = fig.add_subplot(111)

	ll = ~np.isnan(data)

	if limits == None: limits = [data[ll].min(),data[ll].max()]
	if remove_mean:
		rad_prof = radial_profile(data,[data[0].size/2,data[0].size/2])
		meano = create_profile(data,rad_prof,[data[0].size/2,data[0].size/2])
		qq = ~np.isnan(data - meano)
	if negative:
		if remove_mean:
			#print data[ll].max(),meano[qq].max()
			img = plt.imshow((data - meano)/(data - meano)[qq].max(),interpolation='nearest', cmap=cmaps,aspect='equal',origin='lower',extent=(min(vp), max(vp),min(vp), max(vp) ),vmin=limits[0],vmax=limits[1])
		else:
			img = plt.imshow(-(data)/data[ll].max(),interpolation='nearest', cmap=cmaps,aspect='equal',origin='lower',extent=(min(vp), max(vp),min(vp), max(vp)),vmin=-limits[1],vmax=-limits[0] )
	else:
		if remove_mean:
			#print data[ll].max(),meano[qq].max()
			img = plt.imshow((data - meano)/(data - meano)[qq].max(),interpolation='nearest', cmap=cmaps,aspect='equal',origin='lower',extent=(min(vp), max(vp),min(vp), max(vp) ),vmin=limits[0],vmax=limits[1])
		else:
			img = plt.imshow((data-limits[0])/data[ll].max(),interpolation='nearest', cmap=cmaps,aspect='equal',origin='lower',extent=(min(vp), max(vp),min(vp), max(vp) ),vmin=limits[0],vmax=limits[1] )

	axlimits=[min(vp), max(vp),min(vp), max(vp) ]
	plt.axis(axlimits)
	#plt.axvline(x=0.0,linestyle='--',color='white')

	plt.xlabel('V$_x$ / km s$^{-1}$')
	plt.ylabel('V$_y$ / km s$^{-1}$')

	plt.show()
	if colorbar:
		cbar = plt.colorbar(format='%05.2f',orientation='vertical',fraction=0.046, pad=0.04)
		cbar.set_label('Normalised Flux')
		cbar.set_norm(mynormalize.MyNormalize(vmin=limits[0],vmax=limits[1],stretch='linear'))
		cbar = cv.DraggableColorbar(cbar,img)
		plt.tight_layout()
	else:
		cbar=1
		plt.tight_layout()

	#print vp.shape, pha,dmr.shape, dpha

	####################################################


	'''
	if remove_mean:
		#print data.size/2
		rad_prof = radial_profile(data,[data[0].size/2,data[0].size/2])
		mean = create_profile(data,rad_prof,[data[0].size/2,data[0].size/2])
		ll = ~np.isnan(mean)
		fig = plt.figure('Mean')
		plt.clf()
		fig.add_subplot(211)
		plt.plot(rad_prof)
		fig.add_subplot(212)

		plt.show()
	'''
	return cbar



def export(flns, out_dir = 'molly', mol_in = 'molly.in', mol_info ='molly.info',
		   wave=[6450,6750], lamunits='a', fluxunits='flam', ftype='xshooter',
		   num_pix=-1, ra_unit='deg', dec_unit='deg',ext=0):
	"""Export a series of fits to molly files. Serves as a first step to import
	to molly. All the spectra will be interpolated to have the same wavelength
	grid as the first one.

    Parameters
    ----------
    flns : string
       Rule to select fit files.
    outdir : string, optional
       Output directory. It will create one if it doesn't exist.
    mol_in : string, optional
       Output filename for list of molly files.
    mol_info : string, optional
       Output filename for properties of spectra.
    wave : array like, optional, [w_min,w_max]
       List of minimum and maximum wavelength to select (2D).
    lamunits : string, optional
       Units for wavelength. 'a' for Angstroms or 'mu' for micrometers.
    fluxunits : string, optional
       Units for flux array. 'flam' for erg/s/cm^2/Ang or 'mjy' for microJy.
    ftype : string, optional
       Depending on your input, its possible you need to make your own reading
	   function. Currently, it supports
	   'xshooter' : for X-Shooter data products
	   'iraf' : for 1-D spectra exported from IRAF
	   'boller' : for 1-D spectra output from the Boller & Chivens spectrograph
	   			  at SPM, Mexico.
    num_pix : function or method
       Deprecated. Don't use.
    ext : integer
       Extension to be read from fit files. Default 0.


    Returns
    -------
    None

	Notes
	-----
	This routine will create in the specified directory all the relevant
	information of a sequence of spectra to be used inside molly.
	* Please check that the fits reading function is relevant for your case *.
	If not, you will need to create it and append it.

	>>> import doppler_util
	>>> doppler_util.export('hah*.fits',out_dir='molly_halpha',wave=[6512,6610],
							ftype='iraf',ext=0)
	"""
	if not os.path.exists(out_dir):
		os.system('mkdir '+out_dir)

	files = np.sort(glob.glob(flns))
	print ('Total Spectra: '+str(len(files)))
	counter=1

	in_sock = open(out_dir+'/'+mol_in,'w')
	info_sock = open(out_dir+'/'+mol_info,'w')
	info_sock.write('Object           Record       Day  Month      Year    UTC           Dwell    RA             DEC         Equinox'+'\n')
	info_sock.write('C                     I         I      I         I    D             R        D              D           D'+'\n')
	for i in files:
		obb = i.split('.')[0]

		### write molly file
		if ftype == 'xshooter':
			waver,flux = cv.read_xshooter(i,err=False)
			waver = waver *10.
			dateobs=fits.getval(i,'DATE-OBS',0)
			t = Time(dateobs,format='isot',scale='utc')
			ra=fits.getval(i,'RA',0)
			decl=fits.getval(i,'DEC',0)
			if wave != None:
				ss = (waver >= wave[0]) * (waver <= wave[1])
			else:
				ss = waver.astype(int)*0 + 1
			fluflu = flux[ss]/1e-14
			waver0 = waver
			obj_temp=fits.getval(i,'OBJECT',0)
			obj = obj_temp.replace(' ','_')
			equinox=fits.getval(i,'EQUINOX',0)
			exptime=fits.getval(i,'EXPTIME')

		if ftype == 'boller':
			waver,flux = cv.read_iraf(i,ext=ext)
			dateobs=fits.getval(i,'JD',0)
			t = Time(dateobs,format='jd',scale='utc')
			ra=fits.getval(i,'RA',0)
			decl=fits.getval(i,'DEC',0)
			#print(ra,decl)
			coordinate = SkyCoord(ra, decl,unit=(ra_unit,dec_unit))
			ra = coordinate.ra.value
			decl = coordinate.dec.value
			if counter==1:
				waver0 = waver
			#print(waver.size,flux[0][0].size)
			fluflu = np.interp(waver0,waver[:waver.size/2],flux[0][0])

			if wave != None:
				ss = (waver0 > wave[0]) * (waver0 < wave[1])
			else:
				ss = waver0.astype(int)*0 + 1
			fluflu = fluflu[ss]
			obj_temp=fits.getval(i,'OBJECT',0)
			obj = obj_temp.replace(' ','_')
			equinox=fits.getval(i,'EPOCH',0)
			exptime=fits.getval(i,'EXPTIME')

		if ftype == 'iraf':
			waver,flux = cv.read_iraf(i,ext=ext)
			dateobs=fits.getval(i,'JD',0)
			t = Time(dateobs,format='jd',scale='utc')
			ra=fits.getval(i,'RA',0)
			decl=fits.getval(i,'DEC',0)
			coordinate = SkyCoord(ra, decl,unit=(ra_unit,dec_unit))
			ra = coordinate.ra.value
			decl = coordinate.dec.value
			if counter==1:
				waver0 = waver
			fluflu = np.interp(waver0,waver,flux)

			if wave != None:
				ss = (waver0 > wave[0]) * (waver0 < wave[1])
			else:
				ss = waver0.astype(int)*0 + 1
			fluflu = fluflu[ss]
			obj_temp=fits.getval(i,'OBJECT',0)
			obj = obj_temp.replace(' ','_')
			equinox=fits.getval(i,'EPOCH',0)
			exptime=fits.getval(i,'EXPTIME')

		if ftype == 'gtc':
			#hdulist=fits.open(i)
			dateobs=fits.getval(i,'MJD-OBS',0)
			t = Time(dateobs,format='mjd',scale='utc')
			ra=fits.getval(i,'RA',0)
			decl=fits.getval(i,'DEC',0)
			coordinate = SkyCoord(ra, decl,unit=(ra_unit,dec_unit))
			ra = coordinate.ra.value
			decl = coordinate.dec.value
			w1delta=fits.getval(i,'CD1_1',0)
			w1start=fits.getval(i,'CRVAL1',0)
			flux=fits.getdata(i,0)[0][0]
			waver = np.arange(flux.size) * w1delta + w1start
			if counter==1:
				waver0 = waver.copy()
			fluflu = np.interp(waver0,waver,flux)

			if wave != None:
				ss = (waver0 > wave[0]) * (waver0 < wave[1])
			else:
				ss = waver0.astype(int)*0 + 1
			fluflu = fluflu[ss]

			obj_temp=fits.getval(i,'OBJECT',0)
			obj = obj_temp.replace(' ','_')
			equinox=fits.getval(i,'EQUINOX',0)
			exptime=fits.getval(i,'EXPTIME')

		cv.Printer(('%2s - %20s, Pixels: %d')%(str(counter).zfill(3),i,fluflu.size))
		fsock = open(out_dir+'/'+obb+'_mol.dat','w')
		for ii,jj in zip(waver0[ss][:num_pix],fluflu[:num_pix]):
			fsock.write(str(float(ii))+'    '+str(jj)+"\n")
		fsock.close()
		if counter == 1:
			wsock = open(out_dir+'/wavelength_mol.dat','w')
			for ww in waver0[ss][:num_pix]:
				wsock.write(str(ww)+"\n")
			wsock.close

		utc = float(t.datetime.hour) + t.datetime.minute/60. + t.datetime.second/3600.
		utc+=exptime/2./3600.
		if utc >= 24.0:
			utc-=24.0
			day = t.datetime.day + 1.0
		else:
			day = t.datetime.day

		info_sock.write(('%11s%12d%10d%7d%10d%13.7f%8d%15.7f%15.7f%9.1f'+"\n")%(obj[:10],counter,
		day,t.datetime.month,t.datetime.year, utc,int(exptime),ra,decl,equinox))

		in_sock.write('lasc '+obb+'_mol.dat '+str(counter)+' 1 2 -3 '+lamunits+' '+fluxunits+' 0.5e-2'+"\n")
		counter+=1
	in_sock.close()
	info_sock.close()



def stream(q,k1,porb,m1,inc,colors='k',both_lobes=False,title=True,label=None):
	"""Calculate the Ballistic and Keplerian trajetories for a given binary
	system under Roche lobe geometry. This will be plotted directly in the
	Doppler tomogram
	"""
	xl,yl,xi,yi,wout,wkout = stream_calculate(q,ni = 100,nj = 100)

	#print''
	azms=-70
	az=np.arctan(yi/xi)

	for i in np.arange(len(az)):
		if xi[i] < 0.0:
			az[i]=az[i] + np.pi
			#print az[i],az[i]*180/np.pi
	az=az*180/np.pi
	i=0
	for j in np.arange(az.size):
		#print az[j]
		i=i+1
		if az[j] < azms:
			break

	vxi = np.real(wout)
	vyi = np.imag(wout)
	vkxi = np.real(wkout)
	vkyi = np.imag(wkout)
	#print az[0],i
	porb=24*3600*porb           # in seconds
	omega=2*np.pi/porb
	gg=6.667e-8                     # Gravitational Constant, cgs
	msun=1.989e33
	cm=q/(1.0+q)
	nvp=1000
	vxp,vyp,vkxp,vkyp,rr=[],[],[],[],[]

	xl=xl-cm
	inc=np.pi*inc/180.0
	a=(gg*m1*msun*(1.0+q))**(1./3)/omega**(2./3)     # Orbital Separation
	vfs=1e5
	vs=omega*a/vfs
	rd=0
	r=1
	vxi=vxi[:i]
	vyi=vyi[:i]
	vkxi=vkxi[:i]
	vkyi=vkyi[:i]
	az=az[:i]
	si=np.sin(inc)
	vx=vxi*si*vs
	vy=vyi*si*vs
	vkx=vkxi*si*vs
	vky=vkyi*si*vs
	xl=xl*vs*si
	yl=yl*vs*si
	npl=len(az)
	#fig = plt.figure(num='Doppler Map')
	#ax = fig.add_subplot(111)
	#dist = np.sqrt((vx - vkx)**2 + (vy - vky)**2)
	#dist = np.abs(vy-vky)
	#print np.abs(vy-vky)[:12],dist[:12]
	#ss = np.where( dist == min(dist) )[0]
	#print vy[-1],vky[-1],vx[-1],vx[-1]
	#print vx[ss],vy[ss]
	plt.plot(vx[:],vy[:],color=colors,marker='')
	plt.plot(vkx[:],vky[:],color=colors,marker='')
	plt.plot(yl[int(yl.size/4):3*int(yl.size/4)],xl[int(xl.size/4):3*int(xl.size/4)],color=colors)
	if title: plt.title(r'$i$='+str(inc/np.pi*180.)[:5]+', M$_1$='+str(m1)+' M$_{\odot}$, $q$='+str(q)+', P$_{orb}$='+str(porb/3600.)[:4]+' hr')
	## 0,0 systemic velocity, km/s
	vy1 = cm * vs * si
	plt.plot(0.,0.,'x',ms = 9,c = colors,alpha=0.3)
	## 0,-K1 systemic velocity, km/s
	plt.plot(0.,-vy1,'+',ms = 10,c = colors,alpha=0.7)
	plt.plot(0.,(1.0-cm)*vs*si,'+',ms = 10,c = colors,alpha=0.7)

	if both_lobes:
		plt.plot(np.concatenate((yl[3*int(xl.size/4):],yl[:int(yl.size/4)]),axis=0),
		np.concatenate((xl[3*int(xl.size/4):],xl[:int(yl.size/4)]),axis=0),color=colors,ls='--')
		#plt.plot(yl,xl,color=colors)
	else:
		plt.plot(yl[int(yl.size/4):3*int(yl.size/4)],xl[int(xl.size/4):3*int(xl.size/4)],color=colors)

	if label != None: plt.text(0.12, 0.1,label, ha='center', va='center', transform=ax.transAxes)
	plt.tight_layout()

def stream_calculate(qm,ni = 100,nj = 100):
	'''
	calculates Roche lobes and integrates path of stream from L1
	'''
	nmax = 10000
	xout = np.zeros(nmax)
	yout = np.zeros(nmax)
	rout = np.zeros(nmax)
	wout = np.zeros(nmax,dtype=np.complex)
	wkout = np.zeros(nmax,dtype=np.complex)
	if np.abs(qm - 1.) < 1e-4: qm = 1e-4
	rd = 0.1
	if qm <= 0.0:
		print ('Mass ratio <= 0. Does not compute. Will exit.')
		return
	rl1 = rlq1(qm)

	x,y = lobes(qm,rl1,ni,nj)
	## Center of mass relative to M1
	cm = qm / (1.0 + qm)
	## Coordinates of M1 and M2
	z1=-cm
	z2=1-cm
	wm1=np.conj(np.complex(0.,-cm))
	## Start at L1-eps with v=0
	eps=1e-3
	z = np.complex(rl1 - cm -eps,0.)
	w = 0
	zp,wp = eqmot(z,w,z1,z2,qm)
	t=0
	dt=1e-4
	isa=0
	it=0
	r=1
	ist=0
	ph=0.
	phmax=6
	while it < nmax and ph < phmax:
		dz,dw = intrk(z,w,dt,z1,z2,qm)
		z=z+dz
		w=w+dw
		t=t+dt
		if np.abs(dz)/np.abs(z) > 0.02: dt=dt/2.
		if np.abs(dz)/np.abs(z) < 0.005: dt=2.*dt

		dph= -np.imag(z*np.conj(z-dz))/np.abs(z)/np.abs(z-dz)
		ph=ph+dph
		##velocity in inertial frame
		##change by Guillaume
		wi=w+np.complex(0,1.)*z
		## unit vector normal to kepler orbit
		rold=r
		r=np.abs(z-z1)

		if ist == 0 and rold < r:
			ist=1
			rmin=rold

		# kepler velocity of circular orbit in potential of M1, rel. to M1
		vk=1.0/np.sqrt(r*(1.0+qm))
		# unit vector in r
		no = np.conj(z-z1)/r
		wk = -vk*no*np.complex(0.,1.)
		# same but rel. to cm, this is velocity in inertial frame
		wk = wk+wm1
		# velocity normal to disk edge, in rotating frame
		dot = no * w
		# velocity parallel to disk edge
		par = np.imag(no*w)
		# reflected velocity
		wr = w - 2.0*dot*no
		#        write(*,'(f8.4,1p9e11.3)')t,z,w,wk,wr,r
		xout[it] = np.real(z)+cm
		yout[it] = -np.imag(z)
		rout[it] = np.sqrt(xout[it]**2+yout[it]**2)
		# change by Guillaume
		wout[it]= wi
		wkout[it]=np.conj(wk)
		if it > 0:
			xo=xout[it]
			yo=yout[it]
			phi=np.arctan(yo/xo)
			if rout[it] < rd and rout[it-1] >  rd:
			## write(*,'('' r,x,y,phi,vs,vk,dot,par'',8f8.3)')
			##    rout(it),x,y,phi,real(w),vk,dot,par
			## write(*,'('' w,no'',4f8.3)')w,no
				xo=xout[it-1]
				yo=yout[it-1]
				phi=np.arctan(yo/xo)
			# write(*,'('' r,x,y,phi'',4f8.3)')rout(it-1),x,y,phi

		if isa == 0 and yout[it] < 0:
			isa=1
			ra=np.abs(z-z1)
			wc=np.conj(w)+np.complex(0.,1.)*np.conj(z-z1)
			ang=np.abs(np.imag((z-z1)*np.conj(wc)))
		it+=1
	return x,y,xout,yout,wout,wkout

def rlq1(q):
	'''
	Calulates roche lobe radius.
	'''
	if np.abs(1.0 - q) < 1e-4:
		rlq = 0.5
		return rlq
	rl = 0
	rn = 1.0 - q
	while np.abs(rl/rn-1.) > 1e-4:
		rl=rn
		f=q/(1.-rl)**2-1./rl**2+(1.+q)*rl-q
		fa=2.*q/(1-rl)**3+2/rl**3+(1.+q)
		rn=rl-f/fa
	rlq1 = rn
	return rlq1



def lobes(q,rs,ni,nj):
	'''
	SUBROUTINE
	'''
	r = np.zeros((ni,nj))
	ch = np.zeros(ni)
	ps = np.zeros(nj)
	x  = np.zeros(ni)
	y  = np.zeros(nj)
	x2 = np.zeros(ni)
	y2 = np.zeros(nj)
	nc = ni
	nop = nj

	r,ch,ps = surface(q,rs,nc,nop,r,ch,ps)
	j=0
	for i in np.arange(nc):
		x[i] = 1.0 -r[i,j]*np.cos(ch[i])
		y[i] = -r[i,j] * np.sin(ch[i])

	r,ch,ps = surface(1./q,1.-rs,nc,nop,r,ch,ps)
	j=0
	for i in np.arange(nc):
		x2[i] = r[i,j] * np.cos(ch[i])
		y2[i] = r[i,j] * np.sin(ch[i])
	xt = np.concatenate((x2[::-1],x,x[::-1],x2))
	yt = np.concatenate((y2[::-1],-y,y[::-1],-y2))
	return xt,yt


def pot(q,x,y,z):
	'''
	FUNCTION
	Roche potential. coordinates centered on M2,
	z along rotation axis, x toward M1
	pr is gradient in radius from M2
	first transform to polar coordinates w/r rotation axis
	'''
	r = np.sqrt(x*x+y*y+z*z)
	if (r == 0):
		print ('r=0 in pot')
		stop
	rh = np.sqrt(x*x+y*y)
	st=rh/r
	if rh == 0:
		cf=1
	else:
		cf=x/rh

	r2 = 1. / (1. + q)
	r1 = np.sqrt(1.0+r**2-2.0*r*cf*st)
	pot=-1.0/r-1.0/q/r1-0.5*(1.0/q+1.0)*(r2**2+(r*st)**2-2.0*r2*r*cf*st)
	pr=1.0/r**2+1.0/q/(r1**3)*(r-cf*st)-0.5*(1.0/q+1)*2.0*(r*st*st-r2*cf*st)
	return pot,pr


def surface(q,rs,nc,nop,r,ch,ps):
	'''
	SUBROUTINE
	Roche surface around M2, coordinates on surface are ch, ps.
	ch: polar angle from direction to M1; ps: corresponding azimuth, counting
	from orbital plane.
	q:mass ratio, rs: radius of surface at point facing M1
	nc, np: number of chi's, psi's.
	output:
	r(nf,nt): radius. ch, ps: chi and psi arrays
	'''
	r = np.zeros((100,100))
	chi = [],ps
	dc = np.pi/nc
	ch[0] = 0
	for i in np.arange(nc-1)+1:
		ch[i] = float((i-1.0))*np.pi/(nc-1.)
	ps[0] = 0
	for j in np.arange(nop-1)+1:
		ps[i] = float((j-1.0))*2.*np.pi/nop
	rs1 = 1.0 -rs
	fs,pr = pot(q,rs1,0.0,0.0)

	## max no of iterations
	im = 20

	for i in np.arange(nop):
		cp = np.cos(ps[i])
		sp = np.sin(ps[i])
		rx = (1.0 - dc) * rs1
		r[0,i] = rs1

		for k in np.arange(nc-1)+1:
			x  = np.cos(ch[k])
			sc = np.sin(ch[k])
			y  = sc * cp
			z  = sc * sp
			j  = 0
			f  = 1
			while (j < im ) and np.abs(f - fs) > 1e-4 or j == 0:
				j = j+1
				r1 = rx
				f,pr = pot(q,r1*x,r1*y,r1*z)
				rx = r1 - (f - fs)/pr
				if rx > rs1: rx = rs1
			if j >= im:
				print( 'No conv in surf',k,i,ch[k],ps[i])
				stop

			r[k,i] = rx

	return r,ch,ps


def eqmot(z,w,z1,z2,qm):
	zr1 = z-z1
	zr2 = z-z2
	## c change by Guillaume : - sign in Coriolis
	wp=-(qm*zr2/(np.abs(zr2))**3+zr1/(np.abs(zr1))**3)/(1.0+qm)-np.complex(0.,2.)*w+z
	zp = w
	return zp,wp


def intrk(z,w,dt,z1,z2,qm):
	zx=z
	wx=w
	zp,wp = eqmot(zx,wx,z1,z2,qm)
	hz0=zp*dt
	hw0=wp*dt
	zx=z+hz0/2.
	wx=w+hw0/2.
	zp,wp = eqmot(zx,wx,z1,z2,qm)
	hz1=zp*dt
	hw1=wp*dt
	zx=z+hz1/2.
	wx=w+hw1/2.
	zp,wp = eqmot(zx,wx,z1,z2,qm)
	hz2=zp*dt
	hw2=wp*dt
	zx=z+hz2
	wx=w+hw2
	zp,wp = eqmot(zx,wx,z1,z2,qm)
	hz3=zp*dt
	hw3=wp*dt
	dz=(hz0+2*hz1+2*hz2+hz3)/6.
	dw=(hw0+2*hw1+2*hw2+hw3)/6.
	return dz,dw

def xy(r,phi):
  return r*np.cos(phi), r*np.sin(phi)

def resonance(j,k,k1,q,porb,m1):
	'''Plots iso-velocity for resonance, in the general
	notation of Whitehurst & King (19XX). Eq. taken from Warner 1995
	page 206-207.
	'''
	k1 *= 1e5
	porb *= 24. * 3600.
	a_1 = k1 / q / (2.0*np.pi) * porb
	a_2 = k1 / (2.0*np.pi) * porb
	a = a_1 + a_2

	r = (j-k)**(2./3.) / j**(2./3.) / (1.0 + q)**(1./3.) * a
	r_circ = a * 0.60/(1. + q)
	velo = np.sqrt(6.67e-08 * m1 *1.989e+33 / r)/1e5
	velo_circ = np.sqrt(6.67e-08 * m1 *1.989e+33 / r_circ)/1e5
	phis=np.arange(0,6.28,0.01)

	#print r/69950000000.0,velo
	fig = plt.figure(num='Doppler Map')
	xx,yy = xy(velo,phis)
	plt.plot( xx,yy-k1/1e5,c='k',ls=':')
	#ax = fig.add_subplot(111)
	#circ = plt.Circle((0-k1,0),velo,ls='--',color='k',fill=True)


	#print velo_circ,velo
	xx_circ,yy_circ = xy(velo_circ,phis)
	plt.plot( xx_circ,yy_circ-k1/1e5,c='k',ls='-',lw=2)

	plt.draw()


def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipythonp.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments



def plot_trail(FITS_datafile, dat = 0, cmap=plt.cm.binary):
	# Read data in
	HDU_data = fits.hdu(FITS_datafile)
	if dat > len(HDU_data.data)-1:
		print( 'not a valid dataset...')
		return
	flux = HDU_data.data[dat].flux
	wave = HDU_data.data[dat].wave[0,:]
	extent = (wave[0],wave[-1],0.5,flux.shape[0]+0.5)
	plt.imshow(flux,origin='lower', aspect='auto',interpolation='nearest',extent=extent,cmap=cmap)
	plt.xlabel(r'Wavelength', fontsize = 18)
	plt.ylabel(r'Spectrum', fontsize = 18)
	plt.show()
