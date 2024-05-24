"""Saves BSVR .tfm files as a .mat file (for Cemre)


    Usage 
        python save_svr_tfms_as_pt_mat.py <indir> [prefix] [savename]
        
        prefix example: ref-Estimated-vol_%4.tfm - where 4 denotes number of digits 
"""

import re
from scipy.io import loadmat,savemat

import sys 
from compare_transforms import *



class FrameName:
    def __init__(self, name_beginning, name_ending, n_digits):
        self.name_beginning = name_beginning
        self.name_ending = name_ending
        self.n_digits = n_digits

    def filename(self):
        return self.name_beginning + "[0-9]"*self.n_digits + self.name_ending

    @classmethod
    def from_string(cls, string):
        """Disentangles digits"""
        re_res = re.search(r"%(\d+)", string)
        if re_res is None:
            print("Invalid input pattern, you must include '%' followed by a number to indicate the format of the frame number")
            exit(0)
        else:
            n_digits = int(re_res.group(1))
            span = re_res.span()
            name_beginning = string[:span[0]]
            name_ending = string[span[1]:]
            return cls(name_beginning, name_ending, n_digits)

if __name__=='__main__':
    
    indir = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv)>2 else "ref-Estimated-vol_%4.tfm"
    savename = sys.argv[3] if len(sys.argv)>3 else None
    
    #from IPython import embed; embed()
    
    input_pattern = FrameName.from_string(prefix)
    assert os.path.exists(indir)
    if savename is None:
        savename = indir+"/"+prefix.split("%")[0] +".mat"
        
    # read transforms 
    transforms_est = read_composite_transforms2(indir, prefix=input_pattern.filename())
    volumes=len(transforms_est)
    slices=transforms_est[0].shape[0]    
    
    
    # chain transforms 
    t = chain_transforms(transforms_est[:])    
    
    
    # save as .mat    
    mydict={'pt_sig':np.moveaxis(t,0,1)}
    savemat(savename,mydict)    
    
    # verbose 
    print(f"Saved to: {savename}")