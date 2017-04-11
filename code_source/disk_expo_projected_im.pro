pro disk_expo_projected_im,x,y,p,_extra=extra,model,sigma

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

endelse

end
