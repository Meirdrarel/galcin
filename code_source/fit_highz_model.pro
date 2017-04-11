function diskexpo_projected_low,x,y,p,_extra=extra

;common constante,c,g

;parameters
  xc=p[0]
  yc=p[1]
  vs=p[2]
  pa=p[3]
  inc=p[4]
  vd0=p[5]
  d0=p[6]
  a=p[7]
  g=p[8]
  
;extras
;mono=extra.mono
;seeing=extra.seeing
;tfker=extra.tfker
;tfmono=extra.tfmono
  mask=extra.mask
  
  sz=size(x)
  
  theta=angell(x,pa,inc,xc,yc)
  rj=distell(x,pa,inc,xc,yc)
  
  theta=theta[mask]
  rj=rj[mask]
  
  
  if d0 eq 0 then stop
;print,d0
  
  r1=rj/d0
  rg=rj/g
  
;v=sqrt(r1^2*mu0*d0*(beseli(0.5*r1,0,/double)*beselk(0.5*r1,0,/double)-beseli(0.5*r1,1,/double)*beselk(0.5*r1,1,/double))+a*g^2/rg*(alog(rg+sqrt((rg)^2+1))-rg*(1+(rg)^2)^(-0.5 ) ))
  
  v=vd0*(r1)^g/(1.+(r1)^a)
  
  model=vs+v*sin(inc)*cos(theta)
  
  return,model
  
end

function diskexpo_projected,x,y,p,_extra=extra
  
;common constante,c,g
  
;parameters
  xc=p[0]
  yc=p[1]
  vs=p[2]
  pa=p[3]
  inc=p[4]
  vd0=p[5]
  d0=p[6]
  
;extras
mono=extra.mono
seeing=extra.seeing
tfker=extra.tfker
tfmono=extra.tfmono
mask=extra.mask
modelv=extra.modelv
plot=extra.plot

sz=size(x)

dmin=8.
if d0 ge dmin then begin
    model=fltarr(sz[1],sz[2])

    theta=angell(model,pa,inc,xc,yc)
    rj=distell(model,pa,inc,xc,yc)
    
    if modelv eq 0 then v=besselv(rj,d0,vd0) $
    else if modelv eq 1 then v=haloisov(rj,d0,vd0) $
    else if modelv eq 2 then v=penteplateauv(rj,d0,vd0)

    proj=vs+v*sin(inc)*cos(theta)
    
;comment deconvoluer le flux mono?
    
;convolution seeing
    tfv=fft(proj*mono,-1,/double)
    tfconv=tfv*tfker
    tfconvm=tfmono*tfker
    conv_v=real_part(fft(tfconv,1,/double))
    conv_m=real_part(fft(tfconvm,1,/double))
    model=conv_v/conv_m
;stop
endif else begin
    bin_fac=ceil(dmin/d0)
    dmin=bin_fac*d0
    sz1=sz[1]*bin_fac
    sz2=sz[2]*bin_fac
    model=fltarr(sz1,sz2)
    x1=(xc+0.5)*bin_fac-0.5
    y1=(yc+0.5)*bin_fac-0.5

    theta=angell(model,pa,inc,x1,y1)
    rj=distell(model,pa,inc,x1,y1)/bin_fac

    if modelv eq 0 then v=besselv(rj,d0,vd0) $
    else if modelv eq 1 then v=haloisov(rj,d0,vd0) $
    else if modelv eq 2 then v=penteplateauv(rj,d0,vd0)
    proj=vs+v*sin(inc)*cos(theta)

;comment deconvoluer le flux mono?
;interpolation mono : les pixels morts vont poser probleme (interpolation jusqu'a 0 alors qu'on ne connait pas reellement le flux sur ces pixels)
    xx=(findgen(sz1) - (bin_fac-1.)/2.)/bin_fac
    yy=(findgen(sz2) - (bin_fac-1.)/2.)/bin_fac
    monoi=interpolate(mono,xx,yy,missing=0.,/grid)
    
;convolution seeing

    kernel=makegaussian2dc(seeing*bin_fac,sz1,sz2)
    kernel=kernel*n_elements(kernel)/total(kernel)
    tfker0=fft(kernel,-1,/double)

    tfv=fft(proj*monoi,-1,/double)
    tfmono0=fft(monoi,-1,/double)
    tfconv=tfv*tfker0
    tfconvm=tfmono0*tfker0
    conv_v=real_part(fft(tfconv,1,/double))
    conv_m=real_part(fft(tfconvm,1,/double))

;binning
    model=rebin(conv_v,sz[1],sz[2])/rebin(conv_m,sz[1],sz[2])
endelse
;stop

if plot then begin
    window,0,xsize=256,ysize=256
    imdisp,model>(vs-vd0*sin(inc))<(vs+vd0*sin(inc))
endif
return,model[mask]

end

function diskexpo_projected_disp,x,y,p,_extra=extra

;common constante,c,g

;parameters
xc=p[0]
yc=p[1]
vs=p[2]
pa=p[3]
inc=p[4]
vd0=p[5]
d0=p[6]
sig0=p[7]

;extras
mono=extra.mono
seeing=extra.seeing
tfker=extra.tfker
tfmono=extra.tfmono
mask=extra.mask
modelv=extra.modelv
plot=extra.plot

sz=size(x)

dmin=8.
if d0 ge dmin then begin
    model=fltarr(sz[1],sz[2])

    theta=angell(model,pa,inc,xc,yc)
    rj=distell(model,pa,inc,xc,yc)

    if modelv eq 0 then v=besselv(rj,d0,vd0) $
    else if modelv eq 1 then v=haloisov(rj,d0,vd0) $
    else if modelv eq 2 then v=penteplateauv(rj,d0,vd0)

    proj=vs+v*sin(inc)*cos(theta)
    proj2=proj^2
;comment deconvoluer le flux mono?
    
;convolution seeing

;rv
    tfv=fft(proj*mono,-1,/double)
    tfconv=tfv*tfker
    tfconvm=tfmono*tfker
    conv_v=real_part(fft(tfconv,1,/double))
    conv_m=real_part(fft(tfconvm,1,/double))
    model=conv_v/conv_m

;rv2
    tfv=fft(proj2*mono,-1,/double)
    tfconv=tfv*tfker
    conv_v=real_part(fft(tfconv,1,/double))
    mod2=conv_v/conv_m
    
    sigma=sqrt(sig0^2+mod2-model^2)

;stop
endif else begin
    bin_fac=ceil(dmin/d0)
;    bin_fac=9
    dmin=bin_fac*d0
    sz1=sz[1]*bin_fac
    sz2=sz[2]*bin_fac
    model=fltarr(sz1,sz2)
    x1=(xc+0.5)*bin_fac-0.5
    y1=(yc+0.5)*bin_fac-0.5

    theta=angell(model,pa,inc,x1,y1)
    rj=distell(model,pa,inc,x1,y1)/bin_fac

    if modelv eq 0 then v=besselv(rj,d0,vd0) $
    else if modelv eq 1 then v=haloisov(rj,d0,vd0) $
    else if modelv eq 2 then v=penteplateauv(rj,d0,vd0)

    proj=vs+v*sin(inc)*cos(theta)
    proj2=proj^2

;comment deconvoluer le flux mono?
;interpolation mono : les pixels morts vont poser probleme (interpolation jusqu'a 0 alors qu'on ne connait pas reellement le flux sur ces pixels)
    xx=(findgen(sz1) - (bin_fac-1.)/2.)/bin_fac
    yy=(findgen(sz2) - (bin_fac-1.)/2.)/bin_fac
    monoi=interpolate(mono,xx,yy,missing=0.,/grid)
;    monoi=mono
;convolution seeing
    kernel=makegaussian2dc(seeing*bin_fac,sz1,sz2)
    kernel=kernel*n_elements(kernel)/total(kernel)
    tfker0=fft(kernel,-1,/double)

;rv
    tfv=fft(proj*monoi,-1,/double)
    tfmono0=fft(monoi,-1,/double)
    tfconv=tfv*tfker0
    tfconvm=tfmono0*tfker0
    conv_v=real_part(fft(tfconv,1,/double))
    conv_m=real_part(fft(tfconvm,1,/double))

;binning
    model=rebin(conv_v,sz[1],sz[2])/rebin(conv_m,sz[1],sz[2])

;rv2
    tfv=fft(proj2*monoi,-1,/double)
    tfconv=tfv*tfker0
    conv_v=real_part(fft(tfconv,1,/double))

    mod2=rebin(conv_v,sz[1],sz[2])/rebin(conv_m,sz[1],sz[2])

;binning
    sigma=sqrt(sig0^2+mod2-model^2)

;    tvscl,sigma,1
;    tvscl,model,2
    ;print,inc*180/!pi
    ;stop
endelse

if plot then begin
    window,0,xsize=256,ysize=256
    imdisp,model>(vs-vd0*sin(inc))<(vs+vd0*sin(inc))
    window,1,xsize=256,ysize=256
    imdisp,sigma>sig0<(2*vd0*sin(inc))
endif

return,[model[mask],sigma[mask]]

end


pro fit_highz_model,rv,seeing,xc=xc,yc=yc,vs=vs,pa=pa,inc=inc,vd0=vd0,d0=d0,mono=mono,disp=disp,pra=pfinal,sig0=sig0,resv=resv,resd=resd,iso=iso,plateau=plateau,expo=expo,perror=paramerr,dof=dof,bestnorm=chi,xfix=xfix,yfix=yfix,vfix=vfix,pfix=pafix,ifix=ifix,modv=modv,modd=modd,plot=plot

cc = 299792.458D
;gg=6.673e-11                     ; m3 kg-1 s-2
;msol=1.989e30                   ; 1msol=1.989e30 kg
;pc=3.085678e13                  ; 1pc=3.085678e13 km
;redefinition de G pour utiliser mu0 en Msol/pc^2, r en kpc
;gg=(G*1.0e-9)*(1000.0*pc)*(msol/pc^2)

;common constante,c,g
c=cc
;g=gg

loadct,13

if n_params() eq 0 then begin
    print, 'Syntax: fit_highz_model,rv,seeing,xc=xc,yc=yc,vs=vs,pa=pa,inc=inc,vd0=vd0,d0=d0,sig0=sig0,mono=mono,disp=disp,pra=pfinal,resv=resv,resd=resd,modv=modv,modd=modd,/iso,/plateau,/expo,perror=paramerr,dof=dof,bestnorm=chi,/xfix,/yfix,/vfix,/pfix,/ifix,/plot'
    print, '    rv - velocity field'
    print, 'seeing - seeing of observation in arcsec'
    print, '    xc - center abscissa'
    print, '    yc - center ordinate'
    print, '    vs - systemic velocity'
    print, '    pa - position angle'
    print, '   inc - inclination'
    print, '   vd0 - maximum velocity of the model'
    print, '    d0 - characteristic radius of the model'
    print, '  sig0 - local velocity dispersion of the model'
    print, '  mono - monochromatic map'
    print, '  disp - velocity dispersion field'
    print, '   pra - output parameters'
    print, '  resv - residual VF'
    print, '  resd - residual VDF'
    print, '  modv - model VF'
    print, '  modd - model VDF'
    print, '   iso - set this keyword to fit a isothermal sphere'
    print, '  expo - set this keyword to fit an exponential disk'
    print, 'plateau- set this keyword to fit a pente+plateau'
    print, 'perror - error on parameters determination'
    print, '   dof - number of free parameters'
    print, 'bestnorm - chi of the best model'
    print, '  xfix - set this keyword to fix xc'
    print, '  yfix - set this keyword to fix yc'
    print, '  vfix - set this keyword to fix vs'
    print, '  pfix - set this keyword to fix pa'
    print, '  ifix - set this keyword to fix inc'
    print, '  plot - set this keyword to plot during fitting process'
    return
endif

if n_elements(mono) eq 0 then mono=1.
if n_elements(d0) eq 0 then d0=3                            ;?
if n_elements(vd0) eq 0 then vd0=200                          ;?
if n_elements(iso) eq 0 then iso=0
if n_elements(plateau) eq 0 then plateau=0
if n_elements(expo) eq 0 then expo=0
if n_elements(plot) eq 0 then plot=0

if n_elements(xfix) eq 0 then xfix=0
if n_elements(yfix) eq 0 then yfix=0
if n_elements(vfix) eq 0 then vfix=0
if n_elements(pafix) eq 0 then pafix=0
if n_elements(ifix) eq 0 then ifix=0

fix=fltarr(5)
if xfix then fix[0]=1
if yfix then fix[1]=1
if vfix then fix[2]=1
if pafix then fix[3]=1
if ifix then fix[4]=1

modelv=0
if plateau then modelv=2
if iso then modelv=1
if expo then modelv=0

comm=['Exponential Disk','Isothermal Halo','Pente + Plateau']
print,'Using ',comm[modelv],' model'

;vmax atteint pour r/d0=2.15, vmax=0.8798243*sqrt(d0*mu0)

sz=size(rv)

;model parameters

if n_elements(disp) ne 0 then begin
    func='diskexpo_projected_disp'

    p=fltarr(8)

    if n_elements(sig0) eq 0 then sig0=50.

    p[0]=xc
    p[1]=yc
    p[2]=vs
    p[3]=pa
    p[4]=inc
    p[5]=vd0
    p[6]=d0
    p[7]=sig0
    parinfo=replicate({fixed:0, limited:[0,0], limits:[0.D,0.D]},n_elements(p))

    parinfo(7).limited(0)=1
    parinfo(7).limits(0)=0.
endif else begin
    func='diskexpo_projected'

    p=fltarr(7)

    p[0]=xc
    p[1]=yc
    p[2]=vs
    p[3]=pa
    p[4]=inc
    p[5]=vd0
    p[6]=d0
    parinfo=replicate({fixed:0, limited:[0,0], limits:[0.D,0.D]},n_elements(p))
endelse
parinfo(0:4).fixed=fix
pstart=p

;parameters constraints

;limits
;parinfo(0).fixed=1
;parinfo(1).fixed=1
;parinfo(4).fixed=1

parinfo(4).limited(0)=1
parinfo(4).limits(0)=!pi/2.*1./9. ;au moins 10 degres
parinfo(4).limited(1)=1
parinfo(4).limits(1)=!pi/2.*8./9. ;au plus 80 degres

parinfo(5).limited(0)=1
parinfo(5).limits(0)=0.

parinfo(6).limited(0)=1
parinfo(6).limits(0)=min(seeing./4.,1.)

kernel=makegaussian2dc(seeing,sz[1],sz[2])
kernel=kernel*n_elements(kernel)/total(kernel)
tfker=fft(kernel,-1,/double)

mask=where(rv ne -3.1e38)
rv_f=rv[mask]
if n_elements(mono) ne 0 then begin
    cm=wherenan(mono)
    if cm[0] ne -1 then mono[cm]=0
;    mono[mask]=0
endif
tfmono=fft(mono,-1,/double)

extra={seeing:seeing,mono:mono,tfker:tfker,tfmono:tfmono,mask:mask,modelv:modelv,plot:plot}

;error=rv-rv+5.

index=findgen(sz[1],sz[2])
x=index mod sz[1]
y=index/sz[1]

if n_elements(disp) ne 0 then rv_f=[rv[mask],disp[mask]]

pfinal=mpfit2dfun(func,x,y,rv_f,error,pstart,perror=paramerr,dof=dof,ftol=1e-5,bestnorm=chi,autoderivative=1,parinfo=parinfo,/quiet,functargs=extra,yfit=model)

resv=rv
resv[mask]=rv[mask]-model[0:n_elements(mask)-1]

if n_elements(model) eq 2*n_elements(mask) then begin
    resd=rv
    resd[mask]=disp[mask]-model[n_elements(mask):2*n_elements(mask)-1]
endif
pmod=pfinal
if n_elements(pmod) eq 7 then pmod=[pmod,0]
disk_expo_projected_im,x,y,pmod,_extra=extra,model,sigma
modv=rv
modd=rv
modv[mask]=model[mask]
modd[mask]=sigma[mask]

end
