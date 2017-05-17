from astropy.io import  fits


class debbuger:
    def __init__(self, name):
        self.iter = 0
        self.file = open('log_debbug_{}.log'.format(name), 'w')

    def debug(self, data, func, iter=True):
        self.file.write('\n{}\t{}'.format(str(func), data))
        hdu = fits.PrimaryHDU(data=data)
        hdulist = fits.HDUList(hdu)
        hdulist.writeto('debbug/i{}_data_{}.fits'.format(self.iter, str(func)), checksum=True, overwrite=True)
        if iter is True:
            self.iter += 1
