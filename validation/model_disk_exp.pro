function makeGaussian2d, fwhm

  a = [0.0, 1.0, fwhm/(2.0*sqrt(2.0*alog(2.0))), fwhm/(2.0*sqrt(2.0*alog(2.0))), 0.0, 0.0, 0.0]

  nx = round(fwhm)*2.0+1.0
  ny = round(fwhm)*2.0+1.0

  x = (findgen(nx)-(nx-1.0)/2.0) # replicate(1.0, ny)
  y = replicate(1.0, nx) # (findgen(ny) - (ny-1.0)/2.0)

  u = ((x-a[4])/a[2])^2 + ((y-a[5])/a[3])^2

  z = a[0] + a[1] * exp(-u/2)

  return, z

end

; definir la taille de l image
sz=[30,30]

; definir les parametres du model
xc = 
yx =
vs = 
pa = 
inc = 
vd0 = 
d0 = 

; attribution des params dans un tableau
p=fltarr(7)
p[0]=xc
p[1]=yc
p[2]=vs
p[3]=pa
p[4]=inc
p[5]=vd0
p[6]=d0


; calcul d une psf 
kernel=makegaussian2dc(seeing,sz[1],sz[2])
kernel=kernel*n_elements(kernel)/total(kernel)
tfker=fft(kernel,-1,/double)

tfmono=fft(mono,-1,/double)

extra={seeing:seeing,mono:mono,tfker:tfker,tfmono:tfmono,mask:mask,modelv:modelv,plot:plot}

index=findgen(sz[1],sz[2])
x=index mod sz[1]
y=index/sz[1]

; calcul du model
disk_expo_projected_im,x,y,p,_extra=extra,model,sigma

writefits,'test/model_IDL.fits',model,h