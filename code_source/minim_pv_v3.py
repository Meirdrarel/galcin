import numpy as np
import pyfits as pf
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ma
import os.path as osp
import os
from mpfit import mpfit
import ipdb
import pv2D_model_v3 as pv2
import pv3D_model_v3 as pv3
from pvExtract2 import pvExtract
from pvExtract2_doublet import pvExtract_doublet
from scipy import constants as cst
import tools
import tools_doublet
import astropy.stats as st
import time
from files_handle import getValue_fromTab

################# MPFIT WITH PV_2DMODEL ################################

def myfunc_pv(p,fjac=None,data=None, err=None, pix=None, dl_pix=None, rlast=None, lrange=None, l0=None, fwhm = None, lfwhm=None, slit_width = 1., cdeltxy = 0.05125,cdeltl = 0.1455, lbdaOII= 3727.425,lbda_dist=1.395, doublet=None, rc=None):
    
#    intensity profile (exponential disk)
    b0=p[0]
    rd=p[1]
    rtrunc=p[2]
#    velocity profile (exponential disk)
    vd=p[3]
    rt=p[4]
#    velocity dispersion profile (linear decreasing profile)
    sig0=p[5]
    slope=p[6]
#    galaxy parameters
    rcen=p[7]
    incl=p[8]
    pa=p[9]
    lbda=p[10]
    vs=p[11]
    if doublet is True: ratio=p[12]
    
    if doublet is True:
        model=pv2.pv_2Dmap_doublet(b0, rd, rtrunc, vd, rt,vs, sig0, slope, rcen, incl, pa, fwhm=fwhm, slitw=slit_width, lbda=lbda, lfwhm=lfwhm, pix=pix, dl_pix=dl_pix, lrange=lrange, rlast=rlast, res=cdeltxy, lres=cdeltl, l0=l0, lbdaOII= lbdaOII,lbda_dist=lbda_dist, ratio=ratio, rc=rc)
    else:
        model=pv2.pv_2Dmap(b0, rd, rtrunc, vd, rt,vs, sig0, slope, rcen, incl, pa, fwhm=fwhm, slitw=slit_width, lbda=lbda, lfwhm=lfwhm, pix=pix, dl_pix=dl_pix, lrange=lrange, rlast=rlast, res=cdeltxy, lres=cdeltl, l0=l0, rc=rc)



    return [0, np.reshape((data - model)/err, data.size)]


def pv_mpfit(file, rc='exp', doublet=False):
    
    #DATA
    data=pf.getdata('{}'.format(file))
#    data=np.rot90(data0, k=2)        #rotation of the image, so that I have North - up , East - left
    hdr=pf.getheader('{}'.format(file))
    #wavelength
    dlbda=-hdr['CDELT1']
    lbda0=hdr['CRVAL1']
    pix0= data.shape[1] -1
    lbda=lbda0 + (np.arange(data.shape[1]) - pix0) * dlbda
    
#    #*************position angle parentesis*********************
#    b=input("Slit N.:  ")
#    cy=hdr['HIERARCH ESO INS SLIT{} BEZIER CY'.format(b)]
#    cx=hdr['HIERARCH ESO INS SLIT{} BEZIER CX'.format(b)]
#    pa=round(np.degrees((np.arcsin(cy/np.sqrt(cx**2 +cy**2)))))
#    print '\nPOSITION ANGLE OF THE SLIT IS: ', pa,'\n'
#    #*************position angle parentesis*********************

    
    #ERRORS
    if doublet is True:
        vel=pvExtract_doublet(file, plotfit=False, plotcurve=False, printvalues=False)  #sign,vmax,rmax, cen, rtrunc, err, posarc, velkmscorr, velkms_err, ratio, velpix,pospix, velpix_err, err_vmax,noise2, bkg_l
        err0=vel[5]
        noise= vel[14]
        bkg_l=vel[15]
    else:
        vel=pvExtract(file, plotfit=False, plotcurve=False, printvalues=False)  #sign,vmax,rmax, cen, rtrunc, err, posarc, velkmscorr, velkms_err,velpix,pospix,velpix_err, err_vmax, noise2, bkg_l
        err0=vel[5]
        noise= vel[13]
        bkg_l=vel[14]
    try:
        gain=hdr['HIERARCH ESO DET OUT1 CONAD']     # Conversion from ADUs to electrons
    except KeyError:
        gain=0.56
    data_nobkg=np.zeros(data.shape)
    err=np.zeros(data.shape, dtype=float)
    N_el=np.zeros(data.shape, dtype=float)
    sigma_poiss=np.zeros(data.shape, dtype=float)
    for i in range(data.shape[1]):
        data_nobkg[:,i]=data[:,i]- bkg_l[i]
    data_nobkg[np.where(data_nobkg < 0.)]=0.
    for j in range(data.shape[1]):
        for k in range(data.shape[0]):
            N_el[k,j]=data_nobkg[k,j] * gain
            sigma_poiss[k,j]=np.sqrt(N_el[k,j])
            err[k,j]=np.sqrt((sigma_poiss[k,j]/gain)**2 + (err0[j])**2)

    
#    err=np.ones(data.shape, dtype=float)*err0
#    err=np.zeros(data.shape, dtype=float)

#    ipdb.set_trace()

    
    pix = hdr['HIERARCH ESO INS PIXSCALE']     # arcsec (pixel scale)
    dl_pix = round(hdr['CDELT1'],3)  # A/pixel (grism dispersion)
    rlast = pix * data.shape[0] / 2.
    lrange = dl_pix * data.shape[1]
    l0 = hdr['CRVAL1']
    mres=0.05125
    
    fwhm = hdr['HIERARCH ESO COMPUTED SEEING']
#    fwhm=0.4
    lfwhm = 0.5 #Angstrom

##### MODEL PARAMETERS ###
#    lbda_m=hdr['HIERARCH ESO MODEL LAMBDA']
#    pa_m=hdr['HIERARCH ESO MODEL PA']
#    cen_m=hdr['HIERARCH ESO MODEL CENTER']
#    inc_m=hdr['HIERARCH ESO MODEL INC']
#    b0_m=hdr['HIERARCH ESO MODEL B0']
#    rd_m=hdr['HIERARCH ESO MODEL RD']
#    rtrunc_m=hdr['HIERARCH ESO MODEL RTRUNC']
#    vd_m=hdr['HIERARCH ESO MODEL VD']
#    rt_m=hdr['HIERARCH ESO MODEL RT']
#    sig0_m=hdr['HIERARCH ESO MODEL SIG0']
#    slope_m=hdr['HIERARCH ESO MODEL SLOPE']
#    vs_m=hdr['HIERARCH ESO MODEL VS']
#    ratio_m=0.8
#    if doublet is True:
#        p0=[b0_m,rd_m,rtrunc_m,vd_m,rt_m,sig0_m,slope_m,cen_m,inc_m,pa_m,lbda_m, vs_m, ratio_m]
#    else:
#        p0=[b0_m,rd_m,rtrunc_m,vd_m,rt_m,sig0_m,slope_m,cen_m,inc_m,pa_m,lbda_m, vs_m]
#
### INITIAL GUESS PARAMETERS ###
    if doublet is True:
#        vel= pvExtract_doublet(file, plotfit=False, plotcurve=False, printvalues=False)  #sign,vmax,rmax, cen, rtrunc, err, posarc, velkmscorr, velkms_err, ratio,velpix,pospix
        intens=tools_doublet.intensity_param2(file,rtrunc=vel[4], plot=False, printfit=False)  #b0,rd,rtrunc, rcen
#        intens=tools_doublet.intensity_param2(file, plot=False, printfit=False)  #b0,rd,rtrunc, rcen
    else:
#        vel= pvExtract(file, plotfit=False, plotcurve=False, printvalues=False)  #sign,vmax,rmax, cen, rtrunc, err, posarc, velkmscorr, velkms_err,velpix,pospix
#        intens=tools.intensity_param2(file,rtrunc=vel[4], plot=False, printfit=False)  #b0,rd,rtrunc, rcen
        intens=tools.intensity_param2(file, plot=False, printfit=False)  #b0,rd,rtrunc, rcen
    b0_i=intens[0]
    rd_i=intens[1]
    rtrunc_i=intens[2]
#    rtrunc_i=vel[4]
    morph_cen=intens[3]
    sign=vel[0]
    vd_i=sign*vel[1]
    rt_i=vel[2]
    kin_cen=vel[3]
    sig0_i, slope_i = 20.,0.
    rcen_i= kin_cen
#    rcen_i= 3.45
    incl_i= hdr['HIERARCH ESO INCLINATION']
    pa_i = hdr['HIERARCH ESO PA']
    lbda_i= round(hdr['HIERARCH ESO COMPUTED CENTRAL LAMBDA'],3)
    vs_i=0.
    if doublet is True: ratio_i=vel[9]

    if doublet is True:
        p0=[b0_i,rd_i, rtrunc_i,vd_i,rt_i,sig0_i,slope_i,rcen_i,incl_i,pa_i,lbda_i,vs_i, ratio_i]
#        p0=[b0_i,rd_i, rtrunc_i,vd_i,rt_i,sig0_i,slope_i,rcen_i,inc_m,pa_m,lbda_m,vs_i, ratio_i]
        p0names=['B0','Rd', 'Rtrunc','Vd','Rt','Sig0','Slope','Rcen','Incl','PA','Lambda','Vs', 'Ratio']
    else:
#        p0=[b0_i,rd_i, rtrunc_i,vd_i,rt_i,sig0_i,slope_i,rcen_i,incl_i,pa_i,lbda_i,vs_i]
#        p0=[b0_i,rd_i, rtrunc_i,vd_i,rt_i,sig0_i,slope_i,rcen_i,inc_m,pa_m,lbda_m,vs_i]
        p0names=['B0','Rd', 'Rtrunc','Vd','Rt','Sig0','Slope','Rcen','Incl','PA','Lambda','Vs']

    parinfo=[{'value':0., 'fixed':0, 'limited':[0,0], 'limits':[0.,0.],'parname':0., 'step':0} for i in range(len(p0))]
    
    #################### B0 - P[0] #############################
#    parinfo[0]['fixed']=1
#    parinfo[0]['limited'][0]=1                      #b0
#    parinfo[0]['limits'][0]=0.                      #b0
    #################### Rd - P[1] #############################
#    parinfo[1]['fixed']=1                           #rd
#    parinfo[1]['limited']=[1,1]                     #rd
#    parinfo[1]['limits']=[p0[1]-1.,p0[1]+1.]        #rd
#    parinfo[1]['step']=1.
    #################### Rtrunc - P[2] #############################
#    parinfo[2]['fixed']=1                           #rd
#    parinfo[2]['limited'][0]=1                      #rd
#    parinfo[2]['limits'][0]=0.                      #rd
#    parinfo[2]['limited']=[1,1]                     #rd
#    parinfo[2]['limits']=[p0[2]-1.,p0[2]+1.]        #rd
    #################### Vd - P[3] #############################
#    parinfo[3]['fixed']=1                          #vd
#    parinfo[3]['limited'][0]=1                     #vd
#    parinfo[3]['limits'][0]=0                      #vd
#    parinfo[3]['limited']=[1,1]
#    parinfo[3]['limits']=[p0[3]-30.,p0[3]+30.]
    #################### Rt - P[4] #############################
#    parinfo[4]['fixed']=1                           #rt
#    parinfo[4]['limited'][0]=1                      #rt
#    parinfo[4]['limits'][0]=0.                      #rt
#    parinfo[4]['limited']=[1,1]                     #rt
##    parinfo[4]['limits']=[p0[3]-2.,p0[3]+2.]        #rt
#    parinfo[4]['step']=2.
    #################### Sig0 - P[5] #############################
#    parinfo[5]['fixed']=1                           #sig0
#    parinfo[5]['limited'][0]=1                     #sig0
#    parinfo[5]['limits'][0]=0.                     #sig0
#    parinfo[5]['limited']=[1,1]                     #sig0
#    parinfo[5]['limits']=[p0[5]-20.,p0[5]+20.]      #sig0
    #################### SLOPE - P[6] #############################
    parinfo[6]['fixed']=1                           #slope
#    parinfo[6]['limited']=[1,1]                    #slope
#    parinfo[6]['limits']=[-3.,3.]                  #slope
    #################### RCEN - P[7] ##############################
#    parinfo[7]['fixed']=1
#    parinfo[7]['limited']=[1,1]                     #rcen
#    parinfo[7]['limits']=[p0[6]-.205,p0[6]+0.205]      #rcen
    #################### INC - P[8] #############################
    parinfo[8]['fixed']=1                           #inc
##    parinfo[8]['limited']=[1,1]                    #inc
##    parinfo[8]['limits']=[0.,90.]                  #inc
    #################### PA - P[9] ##############################
    parinfo[9]['fixed']=1                           #pa
    #################### LBDA - P[10] #############################
    parinfo[10]['fixed']=1                           #lbda
#    parinfo[10]['limited']=[1,1]                    #lbda
#    parinfo[10]['limits']=[p0[10]-5.,p0[10]+5.]       #lbda
    #################### vs - P[11] #############################
#    parinfo[11]['fixed']=1                           #vs
#    parinfo[11]['limited']=[1,1]                    #vs
#    parinfo[11]['limits']=[p0[11]-500.,p0[11]+500.]       #vs

#################### RATIO - P[12] #############################
#    if doublet is True:
#        parinfo[12]['fixed']=1                           #ratio


    for i in range(len(p0)):
        parinfo[i]['value']=p0[i]
        parinfo[i]['parname']=p0names[i]
    
    
    print '\n INITIAL VALUES ARE:'
    for n in range(len(p0)):
        if parinfo[n]['fixed']==1: aa='FIXED'
        else: aa='  '
        print '\n P[{}] -- '.format(n),p0names[n],' = ',p0[n],'   ',aa
    ss=raw_input("\n *press enter to continue* \n")
    

    fa = {'data':data,'err':err,'pix':pix, 'dl_pix':dl_pix, 'rlast':rlast,'lrange':lrange, 'l0':l0, 'fwhm':fwhm, 'lfwhm':lfwhm, 'doublet':doublet, 'rc':rc}  # python dictionary


    m = mpfit.mpfit(myfunc_pv, parinfo=parinfo, functkw=fa,gtol=1.e-10)
    print "\n STATUS =" , m.status
    print "\n Reduced Chi-Square = ",m.fnorm / m.dof
#    
    print " \n PARAMETERS:"
    for j in range(m.params.shape[0]):
        print '\n {} = {} +/- {}  -- Initial_{} = {}'.format(p0names[j], round(m.params[j],3),round(m.perror[j],3),p0names[j],p0[j])
    print m.errmsg

    file_name=osp.splitext(osp.basename(file))[0]
    path='{}/Model2D_{}___TT'.format(file_name,rc)
    if not osp.exists(path): os.makedirs(path)
    text=open('{}/bestfit_2Dmodel_{}.txt'.format(path,file_name), 'w')
    text.write('#BEST FIT PARAMETERS FOR  {}'.format(file_name))
    text.write('\n')
    text.write('\n# Chi-Squared = {}     Reduced Chi-Squared = {}'.format(m.fnorm, m.fnorm / m.dof))
    text.write('\n')
    text.write('\n# PARAMETERS:')
    text.write('\n')
    for j in range(m.params.shape[0]):
        text.write('\n {} = {} +/- {}  -- Initial_{} = {}'.format(p0names[j], round(m.params[j],3),round(m.perror[j],3),p0names[j],p0[j]))
        text.write('\n')
    text.close()


    par=m.params       #[b0,rd,rtrunc,vd,rt,sig0,slope,rcen,inc,pa,lbda,vs, ratio]
    if doublet is True:
        pv_new=pv2.pv_2Dmap_doublet(par[0], par[1], par[2], par[3], par[4],par[11], par[5], par[6], par[7], par[8],par[9], fwhm=fwhm, slitw=1., lbda=par[10], lfwhm=lfwhm, pix=pix, dl_pix=dl_pix, lrange=lrange, rlast=rlast, res=0.05125, lres=0.1455, l0=l0,lbdaOII= 3727.425,lbda_dist=1.395, ratio=par[12], rc=rc)
        param_for3D=[par[0],par[1],par[2],par[3],par[4],par[5],par[6],par[7],par[7],par[8],par[9],par[10],par[11], par[12]]
    else:
        pv_new=pv2.pv_2Dmap(par[0], par[1], par[2], par[3], par[4],par[11], par[5], par[6], par[7], par[8],par[9], fwhm=fwhm, slitw=1., lbda=par[10], lfwhm=lfwhm, pix=pix, dl_pix=dl_pix, lrange=lrange, rlast=rlast, res=0.05125, lres=0.1455, l0=l0, rc=rc)
        

    if doublet is True:
        param_for3D=np.round([par[0],par[1],par[2],par[3],par[4],par[5],par[6],par[7],par[7],par[8],par[9],par[10],par[11],par[12]],2)
        param_names=['B0    ','Rd', 'Rtrunc','Vd','Rt','Sig0','Slope','Xcen','Ycen','Incl','PA','Lambda','Vs', 'Ratio']
    else:
        param_for3D=np.round([par[0],par[1],par[2],par[3],par[4],par[5],par[6],par[7],par[7],par[8],par[9],par[10],par[11]],2)
        param_names=['B0    ','Rd', 'Rtrunc','Vd','Rt','Sig0','Slope','Xcen','Ycen','Incl','PA','Lambda','Vs']

    text2=open('{}/param_for3D.txt'.format(path), 'w')
    text2.write('#')
    for k in range(len(param_for3D)):
        text2.write('{}   '.format(param_names[k]))
    text2.write('\n')
    for k in range(len(param_for3D)):
        text2.write('{}   '.format(param_for3D[k]))
    text2.close()

#    ipdb.set_trace()
    if data.shape[0] < data.shape[1]:fig=plt.figure(figsize=(7,9))
    else:fig=plt.figure(figsize=(15,7))
    if data.shape[0] < data.shape[1]: ax1=plt.subplot(311)
    else:ax1=plt.subplot(131)
    plt.imshow(data, origin='lower', interpolation='nearest',vmin=-2*noise, vmax=data.max())
    ax1.set_axis_off()
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.06)
    plt.colorbar(cax=cax1)
    plt.tight_layout()
    if data.shape[0] < data.shape[1]: ax2=plt.subplot(312)
    else: ax2=plt.subplot(132)
    plt.imshow(pv_new, origin='lower', interpolation='nearest', vmin=-2*noise, vmax=data.max())
    ax2.set_axis_off()
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.06)
    plt.colorbar(cax=cax2)
#    plt.tight_layout()
    if data.shape[0] < data.shape[1]: ax3=plt.subplot(313)
    else: ax3=plt.subplot(133)
    plt.imshow(data - pv_new, origin='lower', interpolation='nearest')
    ax3.set_axis_off()
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.06)
    plt.colorbar(cax=cax3)
    plt.tight_layout()
#    plt.axes([0.55,0.1,0.4,0.7])
#    plt.plot(x,v)
    plt.savefig('{}/minim_pv2Dmodel{}.pdf'.format(path,file_name))
    plt.show()
    if rc=='exp':
    #   **exponential_disk_velocity_1D(vd, rt, vs, rcen, incl, rlast=rlast, res=res, plot=False)**
        v= pv2.exponential_disk_velocity_1D(par[3], par[4],par[11], par[7], par[8], rlast=rlast, res=mres, plot=False)
    if rc=='flat':
        v= pv2.flat_model_velocity_1D(par[3], par[4],par[11], par[7], par[8], rlast=rlast, res=mres, plot=False)
    if rc=='arctan':
        v= pv2.arctangent_velocity_1D(par[3], par[4],par[11], par[7], par[8], rlast=rlast, res=mres, plot=False)
    r = np.arange(v.shape[0]) * mres - par[7]  # radius in arcesec
    v = (v - par[11])/np.sin(np.radians(par[8]))
    if sign==1:
        for i in range(len(vel[7])): vel[7][i]=vel[7][i] * (-1)
    y1=v - round(m.perror[3],3)
    y2=v + round(m.perror[3],3)
    ax=plt.subplot()
    plt.plot(r,v, label='Model')
    plt.fill_between(r, y1,y2, alpha=0.1, color='b')
#    plt.plot(vel[7],vel[8], '*r')
#    plt.errorbar(vel[6],vel[7], yerr=vel[8], xerr=pix/2., fmt='.', label='Data')
#    print(par[11], vel[3], par[7])
#    plt.errorbar(vel[6]+(vel[3]-par[7]),vel[7], yerr=vel[8], xerr=pix/2., fmt='.', label='Data')
    plt.errorbar(vel[6]+(vel[3]-par[7]) - 0.,vel[7], yerr=vel[8], xerr=pix/2., fmt='.', label='Data')
    if sign==1: plt.legend(framealpha=0., loc=2)
    else: plt.legend(framealpha=0., loc=1)
#    ipdb.set_trace()
    plt.xlabel('Position  (arcsec)')
    plt.ylabel('Velocity  (km/s)')
    plt.minorticks_on()
#    ax.set_rasterized(True)
    plt.savefig('{}/rotationcurve_2Dmodel{}.pdf'.format(path,file_name))
    plt.show()


    hdu_new=pf.PrimaryHDU(data=pv_new, header=hdr)
    hdulist=pf.HDUList([hdu_new])
    hdulist.writeto("{}/model2D_{}.fits".format(path,file_name),checksum=True, clobber=True)
    hdu_new=pf.PrimaryHDU(data=data-pv_new)
    hdulist=pf.HDUList([hdu_new])
    hdulist.writeto("{}/res2D_{}.fits".format(path,file_name),checksum=True, clobber=True)
    

    if doublet is True:
        pospix_data=vel[11]
        velpix_data=vel[10]
        velpix2_data=vel[16]
        velpixerr_data=vel[12]
        vmax_err=vel[13]
        vel_mod= pvExtract_doublet("{}/model2D_{}.fits".format(path,file_name), plotfit=False, plotcurve=False, printvalues=False)  #sign,vmax,rmax, cen, rtrunc, err, posarc, velkmscorr, velkms_err, ratio,velpix,pospix, velpix_err, err_vmax
        pospix_mod=vel_mod[11]
        velpix_mod=vel_mod[10]
        velpix2_mod=vel_mod[16]
        velpixerr_mod=vel_mod[12]
        vmax_err_mod=vel_mod[13]
    else:
        pospix_data=vel[10]
        velpix_data=vel[9]
        velpix2_data=vel[15]
        velpixerr_data=vel[11]
        vmax_err=vel[12]
        vel_mod= pvExtract("{}/model2D_{}.fits".format(path,file_name), plotfit=False, plotcurve=False, printvalues=False)  #sign,vmax,rmax, cen, rtrunc, err, posarc, velkmscorr, velkms_err,velpix,pospix, velpix_err, err_vmax
        pospix_mod=vel_mod[10]
        velpix_mod=vel_mod[9]
        velpix2_mod=vel_mod[15]
        velpixerr_mod=vel_mod[11]
        vmax_err_mod=vel_mod[12]
    
    fig=plt.figure(figsize=(11,9))
    if data.shape[0] < data.shape[1]: ax1=plt.subplot(211)
    else:ax1=plt.subplot(121)
    plt.title('Spectrum (rotated) - vmax={}+/-{}'.format(vel[1],vmax_err))
    plt.plot(velpix_mod,pospix_mod,'^',markeredgewidth=1,markeredgecolor='b', markerfacecolor='None', label='Model')
    plt.plot(velpix2_mod,pospix_mod,'^',markeredgewidth=1,markeredgecolor='b', markerfacecolor='None')
    plt.plot(velpix_data,pospix_data,'d',markeredgewidth=1,markeredgecolor='g', markerfacecolor='None', label='Data')
    plt.plot(velpix2_data,pospix_data,'d',markeredgewidth=1,markeredgecolor='g', markerfacecolor='None')
    ax1.axhline(round(par[7]/pix), color='b', ls='-.', lw=1)
#    plt.plot(velpix_data,pospix_data,'^',markeredgewidth=1,markeredgecolor='b', markerfacecolor='None')
#    plt.errorbar(velpix_mod,pospix_mod, xerr=velpixerr_mod,yerr=pix/2., fmt='none',elinewidth=1, ecolor='b')
    plt.errorbar(velpix_data,pospix_data, xerr=velpixerr_data,yerr=pix/2., fmt='none',elinewidth=1, ecolor='g')
    plt.errorbar(velpix2_data,pospix_data, xerr=velpixerr_data,yerr=pix/2., fmt='none',elinewidth=1, ecolor='g')
    plt.imshow(data, origin='lower', interpolation='nearest',cmap='hot', vmin=-2*noise, vmax=data.max())
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Position along the major axis  (pixels)')
    if sign==1: plt.legend(framealpha=0.4, loc=2)
    else: plt.legend(framealpha=0.4, loc=1)
    ax1.set_xticks([0,pix0/5,2*pix0/5,3*pix0/5,4*pix0/5, pix0])
    ll=np.round(lbda,2)
    ax1.set_xticklabels([ll[0],ll[pix0/5],ll[2*pix0/5],ll[3*pix0/5],ll[4*pix0/5], ll[pix0]])
    arcsec_array=np.arange(data.shape[0])
    arcsec_array2=(arcsec_array - np.int(np.round(vel[3]/pix)))*pix
    zero=np.where(np.abs(arcsec_array2) == 0.)[0][0]
#    ax1.set_yticks([zero/4, 2*zero/4, 3*zero/4, zero, zero+zero/4, zero+2*zero/4, zero+3*zero/4 ])
    ax1.set_yticks([zero/4, 2*zero/4, 3*zero/4, zero, zero+zero/4, zero+2*zero/4 ])
    aa=ax1.get_yticks()
    ax1.set_yticklabels(np.round(arcsec_array2[aa],2))
    ax1.tick_params(axis='both',labelsize='small')
    ax1.axvline(0., color='r', ls='-', lw=2)
    ax1.axvline(pix0, color='b', ls='-', lw=2)
    if data.shape[0] < data.shape[1]: ax2=plt.subplot(212)
    else:ax2=plt.subplot(122)
    plt.title('Simulated Spectrum (rotated) - vmax={}+/-{}'.format(vel_mod[1], vmax_err_mod))
    plt.plot(velpix_data,pospix_data,'d',markeredgewidth=1,markeredgecolor='g', markerfacecolor='None', label='Data')
    plt.plot(velpix2_data,pospix_data,'d',markeredgewidth=1,markeredgecolor='g', markerfacecolor='None')
    plt.plot(velpix_mod,pospix_mod,'^',markeredgewidth=1,markeredgecolor='b', markerfacecolor='None', label='Model')
    plt.plot(velpix2_mod,pospix_mod,'^',markeredgewidth=1,markeredgecolor='b', markerfacecolor='None')
    ax2.axhline(round(par[7]/pix), color='b', ls='-.', lw=1)
#    plt.errorbar(velpix_mod,pospix_mod, xerr=velpixerr_mod,yerr=pix/2., fmt='none',elinewidth=1, ecolor='b')
    plt.errorbar(velpix_data,pospix_data, xerr=velpixerr_data,yerr=pix/2., fmt='none',elinewidth=1, ecolor='g')
    plt.errorbar(velpix2_data,pospix_data, xerr=velpixerr_data,yerr=pix/2., fmt='none',elinewidth=1, ecolor='g')
    plt.imshow(pv_new, origin='lower', interpolation='nearest',cmap='hot', vmin=-2*noise, vmax=data.max())
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Position along the major axis  (pixels)')
    if sign==1: plt.legend(framealpha=0.4, loc=2)
    else: plt.legend(framealpha=0.4, loc=1)
    ax2.set_xticks([0,pix0/5,2*pix0/5,3*pix0/5,4*pix0/5, pix0])
    ax2.set_xticklabels([ll[0],ll[pix0/5],ll[2*pix0/5],ll[3*pix0/5],ll[4*pix0/5], ll[pix0]])
    arcsec_array=np.arange(data.shape[0])
    arcsec_array2=(arcsec_array - np.int(np.round(par[7]/pix)))*pix
    zero=np.where(np.abs(arcsec_array2) == 0.)[0][0]
    ax2.set_yticks([zero/4, 2*zero/4, 3*zero/4, zero, zero+zero/4, zero+2*zero/4, zero+3*zero/4 ])
    aa=ax2.get_yticks()
    ax2.set_yticklabels(np.round(arcsec_array2[aa],2))
    ax2.tick_params(axis='x',labelsize='small')
    ax2.axvline(0., color='r', ls='-', lw=2)
    ax2.axvline(pix0, color='b', ls='-', lw=2)
    plt.tight_layout()
    plt.savefig('{}/pvExtract_2Dmodel{}.pdf'.format(path,file_name))
    plt.show()




    return m, param_for3D

################# MPFIT WITH PV_3DMODEL ################################


def myfunc_pv3D(p,fjac=None,data=None, err=None, pix=None, dl_pix=None, rlast=None, lrange=None, l0=None, fwhm = None, lfwhm=None, slit_width = 1., cdeltxy = 0.05125,cdeltl = 0.1455, direction=-1, lbdaOII= 3727.425,lbda_dist=1.395, doublet=None, rc=None):
    
    #    intensity profile (exponential disk)
    b0=p[0]
    rd=p[1]
    rtrunc=p[2]
    #    velocity profile (exponential disk)
    vd=p[3]
    rt=p[4]
    #    velocity dispersion profile (linear decreasing profile)
    sig0=p[5]
    slope=p[6]
    #    galaxy parameters
    xcen=p[7]
    ycen=p[8]
    incl=p[9]
    pa=p[10]
    lbda=p[11]
    vs=p[12]
    if doublet is True: ratio=p[13]
    
    if doublet is True:
        cube=pv3.create_cube_doublet(b0, rd, rtrunc, vd, rt,vs, sig0, slope, xcen, ycen, pa, incl, fwhm, lbda, lrange=lrange, rlast=rlast, res=cdeltxy, lres=cdeltl, kernel=lfwhm, l0=l0,lbdaOII= lbdaOII,lbda_dist=lbda_dist, ratio=ratio, rc=rc)
    else:
        cube=pv3.create_cube(b0, rd, rtrunc, vd, rt,vs, sig0, slope, xcen, ycen, pa, incl, fwhm, lbda, lrange=lrange, rlast=rlast, res=cdeltxy, lres=cdeltl, kernel=lfwhm, l0=l0, rc=rc)
    pv_model=pv3.instrument(cube, cdeltxy, cdeltl, pix, dl_pix,direction=direction, slit_pa=pa, slit_x=0.0, slit_y=0., slit_width=1.)

    
    return [0,np.reshape((data - pv_model)/err, data.size)]

def pv3D_mpfit(file,rc='exp', doublet=True):
    
    
    file_name=osp.splitext(osp.basename(file))[0]
    #DATA
    data=pf.getdata('{}'.format(file))
#    data=np.rot90(data0, k=2)        #rotation of the image, so that I have North - up , East - left
    hdr=pf.getheader('{}'.format(file))
    #wavelength
    dlbda=-hdr['CDELT1']
    lbda0=hdr['CRVAL1']
    pix0= data.shape[1] - 1
    lbda=lbda0 + (np.arange(data.shape[1]) - pix0) * dlbda
    
    
    #ERRORS
    if doublet is True:
        vel=pvExtract_doublet(file, plotfit=False, plotcurve=False, printvalues=False)  #sign,vmax,rmax, cen, rtrunc, err, posarc, velkmscorr, velkms_err, ratio, velpix,pospix, velpix_err, err_vmax,noise2
        err0=vel[5]
        noise= vel[14]
        bkg_l=vel[15]
    else:
        vel=pvExtract(file, plotfit=False, plotcurve=False, printvalues=False) #sign,vmax,rmax, cen, rtrunc, err, posarc, velkmscorr, velkms_err,velpix,pospix,velpix_err, err_vmax, noise2
        err0=vel[5]
        noise= vel[13]
        bkg_l=vel[14]

    try:
        gain=hdr['HIERARCH ESO DET OUT1 CONAD']     # Conversion from ADUs to electrons
    except KeyError:
        gain=0.56
    data_nobkg=np.zeros(data.shape)
    err=np.zeros(data.shape, dtype=float)
    N_el=np.zeros(data.shape, dtype=float)
    sigma_poiss=np.zeros(data.shape, dtype=float)
    for i in range(data.shape[1]):
        data_nobkg[:,i]=data[:,i]- bkg_l[i]
    data_nobkg[np.where(data_nobkg < 0.)]=0.
    for j in range(data.shape[1]):
        for k in range(data.shape[0]):
            N_el[k,j]=data_nobkg[k,j] * gain
            sigma_poiss[k,j]=np.sqrt(N_el[k,j])
            err[k,j]=np.sqrt((sigma_poiss[k,j]/gain)**2 + (err0[j])**2)

#    err=np.ones(data.shape, dtype=float)*err0
#    err=np.ones(data.shape, dtype=float)

    pix = hdr['HIERARCH ESO INS PIXSCALE']     # arcsec (pixel scale)
#    dl_pix = hdr['CD1_1']  # A/pixel (grism dispersion)
    dl_pix = round(hdr['CDELT1'],3)  # A/pixel (grism dispersion)
    rlast = pix * data.shape[0] / 2.
    lrange = dl_pix * data.shape[1]
    l0 = hdr['CRVAL1']
    mres=0.05125
    direction=-1
    
    fwhm = hdr['HIERARCH ESO COMPUTED SEEING']
    lfwhm = 0.5 #Angstrom

#### MODEL PARAMETERS ###
#    lbda_m=hdr['HIERARCH ESO MODEL LAMBDA']
#    pa_m=hdr['HIERARCH ESO MODEL PA']
#    cen_m=hdr['HIERARCH ESO MODEL CENTER']
#    inc_m=hdr['HIERARCH ESO MODEL INC']
#    b0_m=hdr['HIERARCH ESO MODEL B0']
#    rd_m=hdr['HIERARCH ESO MODEL RD']
#    rtrunc_m=hdr['HIERARCH ESO MODEL RTRUNC']
#    vd_m=hdr['HIERARCH ESO MODEL VD']
#    rt_m=hdr['HIERARCH ESO MODEL RT']
#    sig0_m=hdr['HIERARCH ESO MODEL SIG0']
#    slope_m=hdr['HIERARCH ESO MODEL SLOPE']
#    vs_m=hdr['HIERARCH ESO MODEL VS']
#    ratio_m=0.8
#    if doublet is True:
#        p0=[b0_m,rd_m,rtrunc_m,vd_m,rt_m,sig0_m,slope_m,cen_m,cen_m,inc_m,pa_m,lbda_m, vs_m, ratio_m]
#    else:
#        p0=[b0_m,rd_m,rtrunc_m,vd_m,rt_m,sig0_m,slope_m,cen_m,cen_m,inc_m,pa_m,lbda_m, vs_m]


### INITIAL GUESS PARAMETERS ###
    if doublet is True:
#        vel= pvExtract_doublet(file, plotfit=False, plotcurve=False, printvalues=False)  #sign,vmax,rmax, cen, rtrunc, err, posarc, velkmscorr, velkms_err, ratio
#        intens=tools_doublet.intensity_param2(file,rtrunc=vel[4], plot=False, printfit=False)  #b0,rd,rtrunc, rcen
        intens=tools_doublet.intensity_param2(file, plot=False, printfit=False)  #b0,rd,rtrunc, rcen
    else:
#        vel= pvExtract(file, plotfit=False, plotcurve=False, printvalues=False)  #sign,vmax,rmax, cen, rtrunc, err, posarc, velkmscorr, velkms_err
        intens=tools.intensity_param2(file,rtrunc=vel[4], plot=False, printfit=False)  #b0,rd,rtrunc, rcen
#        intens=tools.intensity_param2(file, plot=False, printfit=False)  #b0,rd,rtrunc, rcen

    b0_i=intens[0]
    rd_i=intens[1]
#    rtrunc_i=intens[2]
    rtrunc_i= vel[4]
    morph_cen=intens[3]
    sign=vel[0]
    vd_i=sign*vel[1]
    rt_i=vel[2]
    kin_cen=vel[3]
    sig0_i, slope_i = 25.,0.
    xcen_i, ycen_i =kin_cen, kin_cen
#    xcen_i, ycen_i = 3.3, 3.3
    incl_i= hdr['HIERARCH ESO INCLINATION']
    pa_i = hdr['HIERARCH ESO PA']
    lbda_i= round(hdr['HIERARCH ESO COMPUTED CENTRAL LAMBDA'],3)
    vs_i=0.
    if doublet is True: ratio_i=vel[9]
#
    if doublet is True:
#        p0=[b0_i,rd_i, rtrunc_i,vd_i,rt_i,sig0_i,slope_i,xcen_i,ycen_i,inc_m,pa_m,lbda_m,vs_i, ratio_i]
        p0=[b0_i,rd_i, rtrunc_i,vd_i,rt_i,sig0_i,slope_i,xcen_i,ycen_i,incl_i,pa_i,lbda_i,vs_i, ratio_i]
        p0names=['B0','Rd', 'Rtrunc','Vd','Rt','Sig0','Slope','Xcen','Ycen','Incl','PA','Lambda','Vs', 'Ratio']
    else:
        p0=[b0_i,rd_i, rtrunc_i,vd_i,rt_i,sig0_i,slope_i,xcen_i,ycen_i,inc_m,pa_m,lbda_m,vs_i]
#        p0=[b0_i,rd_i, rtrunc_i,vd_i,rt_i,sig0_i,slope_i,xcen_i,ycen_i,incl_i,pa_i,lbda_i,vs_i]
        p0names=['B0','Rd', 'Rtrunc','Vd','Rt','Sig0','Slope','Xcen','Ycen','Incl','PA','Lambda','Vs']

#    if doublet is True: pv2param=pv_mpfit(file, doublet=True)[1]
#   else: pv2param=pv_mpfit(file, doublet=False)[1]
#    p0=pv2param

    path2='{}/Model2D_{}'.format(file_name,rc)
    p0=getValue_fromTab('{}/param_for3D.txt'.format(path2,file_name), col_or_row='row', n=1, printTab=True)

    parinfo=[{'value':0., 'fixed':0, 'limited':[0,0], 'limits':[0.,0.]} for i in range(len(p0))]
    
    #################### B0 - P[0] #############################
#    parinfo[0]['fixed']=1
#    parinfo[0]['limited'][0]=1                      #b0
#    parinfo[0]['limits'][0]=0.                      #b0
    #################### Rd - P[1] #############################
#    parinfo[1]['fixed']=1                           #rd
#    parinfo[1]['limited'][0]=1                      #rd
#    parinfo[1]['limits'][0]=0.                      #rd
#    parinfo[1]['limited']=[1,1]                     #rd
#    parinfo[1]['limits']=[p0[1]-1.,p0[1]+1.]        #rd
    #################### Rtrunc - P[2] #############################
#    parinfo[2]['fixed']=1                           #rd
#    parinfo[2]['limited'][0]=1                      #rd
#    parinfo[2]['limits'][0]=0.                      #rd
#    parinfo[2]['limited']=[1,1]                     #rd
#    parinfo[2]['limits']=[p0[1]-1.,p0[1]+1.]        #rd
    #################### Vd - P[3] #############################
#    parinfo[3]['fixed']=1                          #vd
#    parinfo[3]['limited'][0]=1                     #vd
#    parinfo[3]['limits'][0]=0                      #vd
    #################### Rt - P[4] #############################
#    parinfo[4]['fixed']=1                           #rt
#    parinfo[4]['limited'][0]=1                      #rt
#    parinfo[4]['limits'][0]=0.                      #rt
#    parinfo[4]['limited']=[1,1]                     #rt
#    parinfo[4]['limits']=[p0[3]-2.,p0[3]+2.]        #rt
    #################### Sig0 - P[5] #############################
#    parinfo[5]['fixed']=1                           #sig0
#    parinfo[5]['limited'][0]=1                     #sig0
#    parinfo[5]['limits'][0]=0.                     #sig0
    ##    parinfo[5]['limited']=[1,1]                     #sig0
    ##    parinfo[5]['limits']=[p0[4]-20.,p0[4]+20.]      #sig0
    #################### SLOPE - P[6] #############################
    parinfo[6]['fixed']=1                           #slope
    ##    parinfo[6]['limited']=[1,1]                    #slope
    ##    parinfo[6]['limits']=[-5.,5.]                  #slope
    #################### XCEN - P[7] ##############################
#    parinfo[7]['fixed']=1
#    parinfo[7]['limited']=[1,1]                     #xcen
#    parinfo[7]['limits']=[p0[7]-0.205,p0[7]+0.205]      #xcen
    #################### YCEN - P[8] ##############################
#    parinfo[8]['fixed']=1
#    parinfo[8]['limited']=[1,1]                     #ycen
#    parinfo[8]['limits']=[p0[8]-.205,p0[8]+0.205]      #ycen
    #################### INC - P[9] #############################
    parinfo[9]['fixed']=1                           #inc
    ##    parinfo[9]['limited']=[1,1]                    #inc
    ##    parinfo[9]['limits']=[0.,90.]                  #inc
    #################### PA - P[10] ##############################
    parinfo[10]['fixed']=1                           #pa
    #################### LBDA - P[11] #############################
    parinfo[11]['fixed']=1                           #lbda
#    parinfo[11]['limited']=[1,1]                    #lbda
#    parinfo[11]['limits']=[p0[11]-5.,p0[11]+5.]       #lbda
    #################### Vs - P[12] #############################
#    parinfo[12]['fixed']=1                           #vs
#    parinfo[12]['limited']=[1,1]                    #vs
#    parinfo[12]['limits']=[p0[12]-500.,p0[12]+500.]       #vs
    #################### RATIO - P[13] #############################
#    if doublet is True:
#        parinfo[13]['fixed']=1                           #ratio


    for i in range(len(p0)):
        parinfo[i]['value']=p0[i]
    
    print '\n INITIAL VALUES ARE:'
    for n in range(len(p0)):
        if parinfo[n]['fixed']==1: aa='FIXED'
        else: aa='  '
        print '\n P[{}] -- '.format(n),p0names[n],' = ',p0[n],'   ',aa
    ss=raw_input("\n *press enter to continue* \n")
    
    
    fa = {'data':data,'err':err,'pix':pix, 'dl_pix':dl_pix, 'rlast':rlast,'lrange':lrange, 'l0':l0, 'fwhm':fwhm, 'lfwhm':lfwhm, 'direction':direction,'doublet':doublet, 'rc':rc}  # python dictionary
    #    ipdb.set_trace()
    t1=time.time()
    m = mpfit.mpfit(myfunc_pv3D, parinfo=parinfo, functkw=fa, ftol=1.e-5, xtol=1.e-5, gtol=1.e-5)
    t2= time.time()
    print "\n STATUS =" , m.status
    print "\n Reduced Chi-Square = ",m.fnorm / m.dof
    print " \n PARAMETERS:"
    for j in range(m.params.shape[0]):
        print '\n {} = {} +/- {}  -- Initial_{} = {}'.format(p0names[j], round(m.params[j],3),round(m.perror[j],3),p0names[j],p0[j])
    print m.errmsg
    tt1=(t2-t1)
    tt2_min=int(tt1/60.)
    tt2_sec=round(tt1 - (tt2_min * 60.))
    print tt2_min, 'minutes',tt2_sec,' seconds'


    path='{}/Model3D_{}'.format(file_name,rc)
    if not osp.exists(path): os.makedirs(path)
    text=open('{}/bestfit_3Dmodel_{}.txt'.format(path,file_name), 'w')
    text.write('#BEST FIT PARAMETERS FOR  {}'.format(file_name))
    text.write('\n')
    text.write('\n# Chi-Squared = {}     Reduced Chi-Squared = {}'.format(m.fnorm, m.fnorm / m.dof))
    text.write('\n')
    text.write('\n# PARAMETERS:')
    text.write('\n')
    for j in range(m.params.shape[0]):
        text.write('\n {} = {} +/- {}  -- Initial_{} = {}'.format(p0names[j], round(m.params[j],3),round(m.perror[j],3),p0names[j],p0[j]))
        text.write('\n')
    text.write('\n time: {} minutes {} seconds'.format(tt2_min,tt2_sec))
    text.close()

    par=m.params       #[b0,rd,rtruc,vd,rt,sig0,slope,xcen,ycen,inc,pa,lbda,vs, ratio]

    if doublet is True:
        cube_new=pv3.create_cube_doublet(par[0], par[1], par[2], par[3], par[4],par[12], par[5],  par[6], par[7], par[8], par[10], par[9], fwhm, par[11], lrange=lrange, rlast=rlast, res=0.05125, lres=0.1455, kernel=lfwhm, l0=l0,lbdaOII= 3727.425,lbda_dist=1.395, ratio=par[13], rc=rc)
    else:
        cube_new=pv3.create_cube(par[0], par[1], par[2], par[3], par[4],par[12], par[5],  par[6], par[7], par[8], par[10], par[9], fwhm, par[11], lrange=lrange, rlast=rlast, res=0.05125, lres=0.1455, kernel=lfwhm, l0=l0, rc=rc)

    pv_new=pv3.instrument(cube_new, 0.05125, 0.1455, pix, dl_pix, direction=direction, slit_pa=par[10], slit_x=0.0, slit_y=0., slit_width=1.)


    if data.shape[0] < data.shape[1]:fig=plt.figure(figsize=(7,9))
    else:fig=plt.figure(figsize=(15,7))
    if data.shape[0] < data.shape[1]:ax1=plt.subplot(311)
    else:ax1=plt.subplot(131)
    im1=plt.imshow(data, origin='lower', interpolation='nearest',vmin=-2*noise, vmax=data.max())
    ax1.set_axis_off()
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.06)
    plt.colorbar(cax=cax1)
    plt.tight_layout()
    if data.shape[0] < data.shape[1]:ax2=plt.subplot(312)
    else: ax2=plt.subplot(132)
    im2=plt.imshow(pv_new, origin='lower', interpolation='nearest', vmin=-2*noise, vmax=data.max())
    ax2.set_axis_off()
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.06)
    plt.colorbar(cax=cax2)
    plt.tight_layout()
    if data.shape[0] < data.shape[1]: ax3=plt.subplot(313)
    else: ax3=plt.subplot(133)
    im3= plt.imshow(data - pv_new, origin='lower', interpolation='nearest')
    ax3.set_axis_off()
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.06)
    plt.colorbar(cax=cax3)
    plt.tight_layout()
    #    plt.axes([0.55,0.1,0.4,0.7])
    #    plt.plot(x,v)
    plt.savefig('{}/minim_pv3Dmodel{}.pdf'.format(path,file_name))
    plt.show()


    #   **exponential_disk_velocity_2D(vd, rt, vs, xcen, ycen, pa, incl, rlast=10, res=0.04, plot=True)**
#    v=pv3.exponential_disk_velocity_2D(par[3], par[4], par[12], par[7], par[8], par[10], par[9], rlast=rlast , res=mres, plot=True)
    if rc=='exp':
    #   **exponential_disk_velocity_1D(vd, rt, vs, rcen, incl, rlast=rlast, res=res, plot=False)**
        v= pv2.exponential_disk_velocity_1D(par[3], par[4],par[12], par[7], par[9], rlast=rlast, res=mres, plot=False)
    if rc=='flat':
        v= pv2.flat_model_velocity_1D(par[3], par[4],par[12], par[7], par[9], rlast=rlast, res=mres, plot=False)
    if rc=='arctan':
        v= pv2.arctangent_velocity_1D(par[3], par[4],par[12], par[7], par[9], rlast=rlast, res=mres, plot=False)

    r = np.arange(v.shape[0]) * mres - par[7]  # radius in arcesec
    v = (v - par[12])/np.sin(np.radians(par[9]))
    if sign==1:
        for i in range(len(vel[7])): vel[7][i]=vel[7][i] * (-1)
    y1=v - round(m.perror[3],3)
    y2=v + round(m.perror[3],3)
    ax=plt.subplot()
    plt.plot(r,v, label='Model')
    plt.fill_between(r, y1,y2, alpha=0.1, color='b')
#    plt.plot(vel[7],vel[8], '*r')
#    plt.errorbar(vel[6],vel[7], yerr=vel[8], xerr=pix/2., fmt='.', label='Data')
#    print(par[11], vel[3], par[7])
    plt.errorbar(vel[6]+(vel[3]-par[7]),vel[7], yerr=vel[8], xerr=pix/2., fmt='.', label='Data')
    if sign==1: plt.legend(framealpha=0., loc=2)
    else: plt.legend(framealpha=0., loc=1)
    #    ipdb.set_trace()
    plt.xlabel('Position  (arcsec)')
    plt.ylabel('Velocity  (km/s)')
    plt.minorticks_on()
    #    ax.set_rasterized(True)
    plt.savefig('{}/rotationcurve_3Dmodel{}.pdf'.format(path,file_name))
    plt.show()


    hdu_new=pf.PrimaryHDU(data=pv_new, header=hdr)
    hdulist=pf.HDUList([hdu_new])
    hdulist.writeto("{}/model3D_{}.fits".format(path,file_name),checksum=True, clobber=True)
    hdu_new=pf.PrimaryHDU(data=data-pv_new)
    hdulist=pf.HDUList([hdu_new])
    hdulist.writeto("{}/res3D_{}.fits".format(path,file_name),checksum=True, clobber=True)

    if doublet is True:
        pospix_data=vel[11]
        velpix_data=vel[10]
        velpix2_data=vel[16]
        velpixerr_data=vel[12]
        vmax_err=vel[13]
        vel_mod= pvExtract_doublet("{}/model3D_{}.fits".format(path,file_name), plotfit=False, plotcurve=False, printvalues=False)  #sign,vmax,rmax, cen, rtrunc, err, posarc, velkmscorr, velkms_err, ratio,velpix,pospix, velpix_err, err_vmax, noise2
        pospix_mod=vel_mod[11]
        velpix_mod=vel_mod[10]
        velpix2_mod=vel_mod[16]
        vmax_err_mod=vel_mod[13]
    else:
        pospix_data=vel[10]
        velpix_data=vel[9]
        velpix2_data=vel[15]
        velpixerr_data=vel[11]
        vmax_err=vel[12]
        noise=vel[13]
        vel_mod= pvExtract("{}/model3D_{}.fits".format(path,file_name), plotfit=False, plotcurve=False, printvalues=False)  #sign,vmax,rmax, cen, rtrunc, err, posarc, velkmscorr, velkms_err,velpix,pospix, velpix_err, err_vmax, noise2
        pospix_mod=vel_mod[10]
        velpix_mod=vel_mod[9]
        velpix2_mod=vel_mod[16]
        vmax_err_mod=vel_mod[12]


    fig=plt.figure(figsize=(11,9))
    if data.shape[0] < data.shape[1]: ax1=plt.subplot(211)
    else:ax1=plt.subplot(121)
    plt.title('Spectrum (rotated) - vmax={}+/-{}'.format(vel[1],vmax_err))
    plt.plot(velpix_mod,pospix_mod,'^',markeredgewidth=1,markeredgecolor='b', markerfacecolor='None', label='Model')
    plt.plot(velpix2_mod,pospix_mod,'^',markeredgewidth=1,markeredgecolor='b', markerfacecolor='None')
    plt.plot(velpix_data,pospix_data,'d',markeredgewidth=1,markeredgecolor='g', markerfacecolor='None', label='Data')
    plt.plot(velpix2_data,pospix_data,'d',markeredgewidth=1,markeredgecolor='g', markerfacecolor='None')
    ax1.axhline(round(par[7]/pix), color='b', ls='-.', lw=1)
    #    plt.plot(velpix_data,pospix_data,'^',markeredgewidth=1,markeredgecolor='b', markerfacecolor='None')
    #    plt.errorbar(velpix_mod,pospix_mod, xerr=velpixerr_mod,yerr=pix/2., fmt='none',elinewidth=1, ecolor='b')
    plt.errorbar(velpix_data,pospix_data, xerr=velpixerr_data,yerr=pix/2., fmt='none',elinewidth=1, ecolor='g')
    plt.imshow(data, origin='lower', interpolation='nearest',cmap='hot', vmin=-2*noise, vmax=data.max())
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Position along the major axis  (pixels)')
    if sign==1: plt.legend(framealpha=0.4, loc=2)
    else: plt.legend(framealpha=0.4, loc=1)
    ax1.set_xticks([0,pix0/5,2*pix0/5,3*pix0/5,4*pix0/5, pix0])
    ll=np.round(lbda,2)
    ax1.set_xticklabels([ll[0],ll[pix0/5],ll[2*pix0/5],ll[3*pix0/5],ll[4*pix0/5], ll[pix0]])
    arcsec_array=np.arange(data.shape[0])
    arcsec_array2=(arcsec_array - np.int(np.round(vel[3]/pix)))*pix
    zero=np.where(np.abs(arcsec_array2) == 0.)[0][0]
    ax1.set_yticks([zero/4, 2*zero/4, 3*zero/4, zero, zero+zero/4, zero+2*zero/4, zero+3*zero/4 ])
    aa=ax1.get_yticks()
    ax1.set_yticklabels(np.round(arcsec_array2[aa],2))
    ax1.tick_params(axis='both',labelsize='small')
    ax1.axvline(0., color='r', ls='-', lw=2)
    ax1.axvline(pix0, color='b', ls='-', lw=2)
    if data.shape[0] < data.shape[1]: ax2=plt.subplot(212)
    else:ax2=plt.subplot(122)
    plt.title('Simulated Spectrum (rotated) - vmax={}+/-{}'.format(vel_mod[1], vmax_err_mod))
    plt.plot(velpix_data,pospix_data,'d',markeredgewidth=1,markeredgecolor='g', markerfacecolor='None', label='Data')
    plt.plot(velpix2_data,pospix_data,'d',markeredgewidth=1,markeredgecolor='g', markerfacecolor='None')
    plt.plot(velpix_mod,pospix_mod,'^',markeredgewidth=1,markeredgecolor='b', markerfacecolor='None', label='Model')
    plt.plot(velpix2_mod,pospix_mod,'^',markeredgewidth=1,markeredgecolor='b', markerfacecolor='None')
    ax2.axhline(round(par[7]/pix), color='b', ls='-.', lw=1)
    #    plt.errorbar(velpix_mod,pospix_mod, xerr=velpixerr_mod,yerr=pix/2., fmt='none',elinewidth=1, ecolor='b')
    plt.errorbar(velpix_data,pospix_data, xerr=velpixerr_data,yerr=pix/2., fmt='none',elinewidth=1, ecolor='g')
    plt.imshow(pv_new, origin='lower', interpolation='nearest',cmap='hot', vmin=-2*noise, vmax=data.max())
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Position along the major axis  (pixels)')
    if sign==1: plt.legend(framealpha=0.4, loc=2)
    else: plt.legend(framealpha=0.4, loc=1)
    ax2.set_xticks([0,pix0/5,2*pix0/5,3*pix0/5,4*pix0/5, pix0])
    ax2.set_xticklabels([ll[0],ll[pix0/5],ll[2*pix0/5],ll[3*pix0/5],ll[4*pix0/5], ll[pix0]])
    arcsec_array=np.arange(data.shape[0])
    arcsec_array2=(arcsec_array - np.int(np.round(par[7]/pix)))*pix
    zero=np.where(np.abs(arcsec_array2) == 0.)[0][0]
    ax2.set_yticks([zero/4, 2*zero/4, 3*zero/4, zero, zero+zero/4, zero+2*zero/4, zero+3*zero/4 ])
    aa=ax2.get_yticks()
    ax2.set_yticklabels(np.round(arcsec_array2[aa],2))
    ax2.tick_params(axis='x',labelsize='small')
    ax2.axvline(0., color='r', ls='-', lw=2)
    ax2.axvline(pix0, color='b', ls='-', lw=2)
    plt.tight_layout()
    plt.savefig('{}/pvExtract_3Dmodel{}.pdf'.format(path,file_name))
    plt.show()




    return m







if __name__ == "__main__":
    
    file = raw_input("\n FITS FILE:  ")
#    file='CosmosHR_P04Q2_2Dspec_31slit_Halpha.fits'
#    ff= pv3D_mpfit(file,rc='exp', doublet = True)
    ff= pv_mpfit(file, rc='arctan', doublet = True)
