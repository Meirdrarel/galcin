pro fit_massiv, fito, ifix=ifix, pfix=pfix, xfix=xfix, yfix=yfix, plot=plot, file=file, option1=option, options=options, line=line, path=path, side=side;, smooth=lissage, obs=obs

if n_params() eq 0 then begin
   print,'Syntax:'
   print,'fit_massiv, fito, file=file, path=path, option=option, options=options, line=line, /ifix, /pfix, /xfix, /yfix, /plot, side=side'
   print,'fito: integer that indicates which function is used for the fitting:'
   print,'      - 0: exponential disk'
   print,'      - 1: isothermal sphere'
   print,'      - 2: flat rotation curve'
   print,'      - 3: arctangent function'
   print,'file: path/name (string) of the file that contains the initial values for the fit with the following parameters (columns)'
   print,'      - name of the object (corresponds to the name of the directory + prefix of the files)'
   print,'      - x coordinate (IDL pixel in the image)'
   print,'      - y coordinate (IDL pixel in the image)'
   print,'      - position angle of the major axis, counter-clockwise, from north (degree)'
   print,'      - inclination (degree)'
   print,'      - systemic velocity in the velocity field (km/s)'
   print,'      - half the velocity range of the velocity field, i.e. the projected maximum velocity (km/s)'
   print,'      - effective radius (pixels)'
   print,'      - mean velocity dispersion (km/s) (put 0 if the velocity dispersion map is not used for fitting)'
   print,'      - seeing of the observation (pixels)'
   print,'      - spectral psf dispersion -not FWHM- (km/s)'
   print,'      - value of the spatial smoothing (FWHM) applied for computing the kinematical maps (pixels)'
   print,'path: indicates the path where are stored the maps'
   print,'options: string that indicates the name suffixe (about cleaning for instance)'
   print,'option1: string that indicates the option (e.g. sky_ssmooth)'
   print,'line: string that indicates the line to use (in the name of the files)'
   print,'ifix: set this keyword to 1 to fix the inclination (free -0- by default)'
   print,'pfix: set this keyword to 1 to fix the position angle (free -0-  by default)'
   print,'xfix: set this keyword to 1 to fix the x coordinate (free -0-  by default)'
   print,'yfix: set this keyword to 1 to fix the y coordinate (free -0-  by default)'
   print,'plot: set this keyword to 1 to have screen plots during the process (no plot by default)'
   print,'side: set this keyword to 1 to use only the approaching side, to 2-1 to have only the receding side and to 0 for both (default is both -0-)'
   return
endif

dtr=!pi/180.
if n_elements(file) eq 0 then file='inputs_fit.txt'
if n_elements(option) eq 0 then option='_sky_ssmooth'
if n_elements(options) eq 0 then options='_clean1'
if n_elements(line) eq 0 then line='_{Ha,OIII1,OIII2,NII1,NII2,Hb}'

iso=0
expo=0
plateau=0
arctan=0
if fito le 0 then fito=0
if fito gt 3 then fito=0
if fito eq 1 then iso=1 else if fito eq 2 then plateau=1 else if fito eq 3 then arctan=1 else expo=1
fit=['_exp','_iso','_slp','_ata']
if n_elements(ifix) eq 0 then ifix=0
if ifix ne 0 then ifix=1
if n_elements(xfix) eq 0 then xfix=0
if n_elements(yfix) eq 0 then yfix=0
if n_elements(pfix) eq 0 then pfix=0
if n_elements(path) eq 0 then path='./'
if n_elements(side) eq 0 then side=0
if side gt 1 or side lt -1 then side=0

if side ne 0 and pfix eq 0 and xfix eq 0 and yfix eq 0 then print,"#### Using only one side is better if you fix both major axis and center! ####" 

if strmatch(path,'*/') eq 0 then path=path+path_sep()

suff='_'
if xfix then suff+='x'
if yfix then suff+='y'
if ifix then suff+='i'
if pfix then suff+='p'

readcol,file,nom,xxc,yyc,ppa,iinc,vvsys,vvmax,dd0,ssig0,sseeing,ppsf,lliss,format='(A,F,F,F,F,F,F,F,F,F,F,F)'

for index=0,n_elements(nom)-1 do begin
   obs=nom[index]
   print,obs
   xc=double(xxc[index])
   yc=double(yyc[index])
   pa=double(ppa[index]*dtr)
   inc=double(iinc[index]*dtr)
   vs=double(vvsys[index])
   vmax=double(vvmax[index])
   d0=double(dd0[index])
   sig0=double(ssig0[index])
   seeing=double(sseeing[index])
   psfsp=double(ppsf[index])
   lissage=double(lliss[index])
   
;    cube=file_search(path+obs+'/'+obs+option+'_cube.fits')
   rv=file_search(path+obs+'/'+obs+option+'_vel_{indep,common}'+options+'.fits')
;    mono=file_search(path+obs+'/'+obs+option+'_flux_{indep,common}'+line+'.fits')
   mono=file_search(path+obs+'/'+obs+option+'_flux_{indep,common}'+line+options+'.fits')
   disp=file_search(path+obs+'/'+obs+option+'_{sigma,disp}_{indep'+line+',common,common'+line+'}'+options+'.fits')
   err=file_search(path+obs+'/'+obs+option+'_evel_{indep,common}'+options+'.fits')
   
   m=readfits(mono)
   d=readfits(disp[0])
   v=readfits(rv,h)
   error=readfits(err)
;    c=readfits(cube,h)
   pixsiz=max([abs(sxpar(h,'CDELT1')), sqrt(sxpar(h,'CD1_1')^2+sxpar(h,'CD1_2')^2)])*3600
   
;    pixsiz=abs(sxpar(h,'CDELT1'))*3600.
;    print,'XXXXX TAILLE DE PIXEL EN DUR XXXX'
;    pixsiz=0.2
   
   ;IL FAUT NETOYER LE CHAMP DE VITESSE
   good=where(v le !values.f_infinity,comp=bad)
   if n_elements(good) lt 10 then continue

   m[bad]=0
   v[bad]=-3.1e38
   d[bad]=0
   error[bad]=max(error[good])
   baderr=where(error eq 0)
   if baderr[0] ne -1 then error[baderr]=max(error[good])
   
   if side ne 0 then theta=angell(v,pa,inc,xc,yc)
   if side eq 1 then begin
      app=where(theta ge (-!pi/2) and theta le (!pi/2))
      v[app]=-3.1e38
      d[app]=0
      error[app]=max(error[good])
   endif else if side eq -1 then begin
      rec=where(theta le (-!pi/2) or theta ge (!pi/2))
      v[rec]=-3.1e38
      d[rec]=0
      error[rec]=max(error[good])
   endif
   
   seeingas=seeing*pixsiz
   vd0=vmax/sin(inc)
   pinit=[xc,yc,vs,pa,inc,vd0,d0,sig0]
   
   cond=where(v eq -3.1e38,complement=mask)
   
   print,inc*180/!pi
   print,seeingas
   
   seeing=sqrt((seeing)^2.+lissage^2.)
   
   vori=v
   
   fit_highz_model1,v,seeing,xc=xc,yc=yc,vs=vs,pa=pa,inc=inc,vd0=vd0,d0=d0,mono=m,pra=pfinal,resv=resv,xfix=xfix,yfix=yfix,ifix=ifix,pfix=pfix,perror=perr,bestnorm=chi2,dof=dof,modv=modv,modd=modd,modhr=modhr,expo=expo,plateau=plateau,iso=iso,arctan=arctan,plot=plot,error=error,modelv_full=modv_full,modeld_full=modd_full, modelhr_full=modhr_full

   x=pfinal[0]
   y=pfinal[1]
   vs0=pfinal[2]
   pa0=pfinal[3]/dtr
   inc0=pfinal[4]/dtr
   vc=pfinal[5]
   rc=pfinal[6]
   sig=-1
   perr=[perr,-1]

   meth=fit[fito]
   openw,1,path+obs+'/'+obs+'_parameters_red'+meth+suff+options+'.txt'
   printf,1,'x','y','vs','pa','i','vc','rc','sig','chi2','dof',format='(10A15)'
   printf,1,x,y,vs0,pa0,inc0,vc,rc,sig,chi2,dof,format='(10E15.7)'
   printf,1,perr[0:2],perr[3:4]/dtr,perr[5:7],format='(8E15.7)'
   close,1

   resv[cond]=!values.f_nan
   modv[cond]=!values.f_nan
   modd[cond]=!values.f_nan
   modhr[cond]=!values.f_nan
   writefits,path+obs+'/'+obs+'_resv'+meth+suff+options+'.fits',resv,h
   writefits,path+obs+'/'+obs+'_modv'+meth+suff+options+'.fits',modv,h
   writefits,path+obs+'/'+obs+'_modhr'+meth+suff+options+'.fits',modhr,h
   writefits,path+obs+'/'+obs+'_modd'+meth+suff+options+'.fits',modd,h
   resd=resv
   c2=where(d[mask] lt modd[mask])
   resd[mask]=sqrt((d[mask]^2-modd[mask]^2-psfsp^2)>0)
   resd[cond]=!values.f_nan
   if c2[0] ne -1 then resd[mask[c2]]=0.
   writefits,path+obs+'/'+obs+'_resd'+meth+suff+options+'.fits',resd,h

   writefits,path+obs+'/'+obs+'_modv_full'+meth+suff+options+'.fits',float(modv_full),h
   writefits,path+obs+'/'+obs+'_modhr_full'+meth+suff+options+'.fits',float(modhr_full),h
   writefits,path+obs+'/'+obs+'_modd_full'+meth+suff+options+'.fits',float(modd_full),h
   
   openw,1,path+obs+'/'+obs+'_parameters_residual'+meth+suff+options+'.txt'
   printf,1,'mean(resv)','median(resv)','stdev(resv)','min(resv)','max(resv)','mean(resd)','median(resd)','stdev(resd)','min(resd)','max(resd)',format='(10A15)'
   printf,1,mean(resv[mask]),median(resv[mask]),stddev(resv[mask]),min(resv[mask]),max(resv[mask]),mean(resd[mask]),median(resd[mask]),stddev(resd[mask]),min(resd[mask]),max(resd[mask]),format='(10E15.7)'
   close,1


;   calculer le Vmax proj sur clean2 et clean3, et sur les modeles et observations
   v2=vori[good]
   v3=v2[sort(v2)]
   
   vm2=modv[good]
   vm3=vm2[sort(vm2)]
   
   vm_m=(v3[n_elements(v3)-1]-v3[0])/2.
   vm_m2=(v3[n_elements(v3)-2]-v3[1])/2.
   vm_m3=(v3[n_elements(v3)-3]-v3[2])/2.
   vm_m5=(v3[n_elements(v3)-5]-v3[4])/2.
   vmm_m=(vm3[n_elements(vm3)-1]-vm3[0])/2.
   
   dort=radius(vori,pfinal[3],x,y)*pixsiz
   r2=dort[good]
   rm_u=max(abs(r2)) ; sans contrainte
   rm_c=min([max(r2),-min(r2)]) ; avec contrainte
   
   openw,1,path+obs+'/'+obs+'_vmax_map_rlast'+options+'.txt'
   printf,1,obs,vm_m,vm_m2,vm_m3,vm_m5,vmm_m,rm_u,rm_c,format='(A15,7F15.3)'
   close,1

; stop



endfor

end

; goto,end0
; 
; fit_highz_model1,v,seeing,xc=xc,yc=yc,vs=vs,pa=pa,inc=inc,vd0=vd0,d0=d0,mono=m,disp=d,pra=pfinal,sig0=sig0,resv=resv,resd=resd,xfix=xfix,yfix=yfix,ifix=ifix,pfix=pfix,perror=perr,bestnorm=chi2,dof=dof,modv=modv,modd=modd,expo=expo,plateau=plateau,iso=iso,plot=plot,arctan=arctan
; 
; x=pfinal[0]
; y=pfinal[1]
; vs=pfinal[2]
; pa=pfinal[3]/dtr
; inc=pfinal[4]/dtr
; vc=pfinal[5]
; rc=pfinal[6]
; sig=pfinal[7]
; 
; meth=fit[fito]+'_disp'
; if ifix then meth=meth+'_ifix'
; openw,1,dir+'parameters_red'+meth+'.txt'
; printf,1,'x','y','vs','pa','i','vc','rc','sig','chi2','dof',format='(10A15)'
; printf,1,x,y,vs,pa,inc,vc,rc,sig,chi2,dof,format='(10E15.7)'
; printf,1,perr[0:2],perr[3:4]/dtr,perr[5:7],format='(8E15.7)'
; close,1
; 
; resv[cond]=!values.f_nan
; modv[cond]=!values.f_nan
; modd[cond]=!values.f_nan
; resd[cond]=!values.f_nan
; writefits,dir+'resv'+meth+'.fits',resv,h
; writefits,dir+'resd'+meth+'.fits',resd,h
; writefits,dir+'modv'+meth+'.fits',modv,h
; writefits,dir+'modd'+meth+'.fits',modd,h
; 
; openw,1,dir+'parameters_residual'+meth+'.txt'
; printf,1,'mean(resv)','median(resv)','stdev(resv)','min(resv)','max(resv)','mean(resd)','median(resd)','stdev(resd)','min(resd)','max(resd)',format='(10A15)'
; printf,1,mean(resv[mask]),median(resv[mask]),stdev(resv[mask]),min(resv[mask]),max(resv[mask]),mean(resd[mask]),median(resd[mask]),stdev(resd[mask]),min(resd[mask]),max(resd[mask]),format='(10E15.7)'
; close,1
; 
; end0:
