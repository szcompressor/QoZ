import os
import numpy as np 
import sys

datafolder=sys.argv[1]

for file in os.listdir(datafolder):
    if file.split(".")[-1]!=".d64":
        continue
    filepath=os.path.join(datafolder,file)
    newfilename=file.split(".")[0]+".f32"
    newfilepath=os.path.join(datafolder,newfilename)
    np.fromfile(filepath,dtype=np.double).astype(np.float32).tofile(newfilepath)