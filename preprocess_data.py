import os
import numpy as np 
import sys
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--miranda','-m',type=str)
parser.add_argument('--hurricane','-u',type=str,default="")
parser.add_argument('--nyx','-n',type=str,default="")
parser.add_argument('--scale','-s',type=str,default="")
args = parser.parse_args()
for file in os.listdir(args.miranda):
    if file.split(".")[-1]!="d64":
        continue
    filepath=os.path.join(args.miranda,file)
    newfilename=file.split(".")[0]+".f32"
    newfilepath=os.path.join(args.miranda,newfilename)
    np.fromfile(filepath,dtype=np.double).astype(np.float32).tofile(newfilepath)
'''
for file in os.listdir(args.scale):
    if file.split(".")[-1]!="f32" or file[0]!="Q":
        continue
    filepath=os.path.join(args.scale,file)
    newfilename=file.split(".")[0]+".log10.f32"
    newfilepath=os.path.join(args.miranda,newfilename)
    np.log10(np.fromfile(filepath,dtype=np.float32)).tofile(newfilepath)
'''