import os
import numpy as np 
import argparse
import pandas as pd
if __name__=="__main__":

    parser = argparse.ArgumentParser()

    
    parser.add_argument('--input','-i',type=str)
    parser.add_argument('--output','-o',type=str)
    parser.add_argument('--target','-t',type=str)
    parser.add_argument('--dataset','-d',type=str)
    args = parser.parse_args()
    
    config_file=os.path.join("configs","%s_%s.config"%(args.dataset,args.target))
    if not os.path.exists(config_file):
        raise ValueError("Config file does not exists!")
    #tuning_target_dict={"psnr":"TUNING_TARGET_RD","cr":"TUNING_TARGET_CR","ssim":"TUNING_TARGET_SSIM","ac":"TUNING_TARGET_AC"}

    if args.dataset=="cesm":
        dim=2
    else:
        dim=3

    dims_dict={"cesm":["3600", "1800"],"miranda":["384", "384", "256"],"nyx":["512", "512", "512"],"scale":["1200", "1200", "98"],"hurricane":["500", "500", "100"]}
    dims=dims_dict[args.dataset]
        
    #tuning_target=tuning_target_dict[args.target]
    
    
    #parser.add_argument('--size_x','-x',type=int,default=1800)
    #parser.add_argument('--size_y','-y',type=int,default=3600)
    #parser.add_argument('--size_z','-z',type=int,default=512)
    '''
    if args.sample_blocksize==-1:
        sample_blocksize=64 if args.dim==2 else 32
    else:
        sample_blocksize=args.sample_blocksize
    '''
    
    datafolder=args.input
    datafiles=os.listdir(datafolder)
    datafiles=[file for file in datafiles if file.split(".")[-1]=="dat" or file.split(".")[-1]=="f32" or file.split(".")[-1]=="bin"]
    num_files=len(datafiles)

    ebs=[1e-5,5e-5]+[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-3 for i in range(10,21,5)]
    #ebs=[1e-4,1e-3,1e-2]
    num_ebs=len(ebs)

    total_data_size=num_files
    for d in dims:
        total_data_size*=eval(d)
    total_data_size=total_data_size*4/(1024*1024)


   
    
    
    
    cr=np.zeros((num_ebs,num_files),dtype=np.float32)
    psnr=np.zeros((num_ebs,num_files),dtype=np.float32)
    ssim=np.zeros((num_ebs,num_files),dtype=np.float32)
    #alpha=np.zeros((num_ebs,num_files),dtype=np.float32)
    #beta=np.zeros((num_ebs,num_files),dtype=np.float32)
    overall_cr=np.zeros((num_ebs,1),dtype=np.float32)
    overall_psnr=np.zeros((num_ebs,1),dtype=np.float32)
    overall_ssim=np.zeros((num_ebs,1),dtype=np.float32)
    ac=np.zeros((num_ebs,num_files),dtype=np.float32)
    overall_ac=np.zeros((num_ebs,1),dtype=np.float32)
    c_speed=np.zeros((num_ebs),dtype=np.float32)
    d_speed=np.zeros((num_ebs),dtype=np.float32)
    pid=os.getpid()
    
    


    for i,eb in enumerate(ebs):
    
        for j,datafile in enumerate(datafiles):
            
            filepath=os.path.join(datafolder,datafile)

            
            comm="qoz -z -f -a -i %s -o %s.out -M REL %f -%d %s -c %s" % (filepath,pid,eb,dim," ".join(dims),config_file)
            
            with os.popen(comm) as f:
                lines=f.read().splitlines()
                #print(lines)
                
                r=eval(lines[-3].split('=')[-1])
                p=eval(lines[-6].split(',')[0].split('=')[-1])
                n=eval(lines[-6].split(',')[1].split('=')[-1])
                ct=eval(lines[-12].split('=')[-1])
                dt=eval(lines[-2].split('=')[-1].split("s")[0])
                c_speed[i]+=ct
                d_speed[i]+=dt
                cr[i][j]=r 
                psnr[i][j]=p
                overall_psnr[i]+=n**2
                '''
                for line in lines:
                    if "alpha" in line:
                        #print(line)
                        a=eval(line.split(" ")[4][:-1])
                        b=eval(line.split(" ")[7][:-1])
                        alpha[i][j]=a
                        beta[i][j]=b
                '''
            
                
            if args.target=="ssim":
                comm="calculateSSIM -f %s %s.out %s" % (filepath,pid," ".join(dims))
                try:
                    with os.popen(comm) as f:
                        lines=f.read().splitlines()
                        #print(lines)
                        s=eval(lines[-1].split('=')[-1])
                        ssim[i][j]=max(s,0)
                except:
                    ssim[i][j]=0

            elif args.target=="ac":

                comm="computeErrAutoCorrelation -f %s %s.out " % (filepath,pid)
                try:
                    with os.popen(comm) as f:
                        lines=f.read().splitlines()
                        #print(lines)
                        a=eval(lines[-1].split(':')[-1])
                        ac[i][j]=a
                except:
                    ac[i][j]=1




            
            comm="rm -f %s.out" % pid
            os.system(comm)
    
    overall_psnr=overall_psnr/num_files
    overall_psnr=np.sqrt(overall_psnr)
    overall_psnr=-20*np.log10(overall_psnr)
    overall_cr=np.reciprocal(np.mean(np.reciprocal(cr),axis=1))
            

    cr_df=pd.DataFrame(cr,index=ebs,columns=datafiles)
    psnr_df=pd.DataFrame(psnr,index=ebs,columns=datafiles)
    overall_cr_df=pd.DataFrame(overall_cr,index=ebs,columns=["overall_cr"])
    overall_psnr_df=pd.DataFrame(overall_psnr,index=ebs,columns=["overall_psnr"])
    alpha_df=pd.DataFrame(alpha,index=ebs,columns=datafiles)
    beta_df=pd.DataFrame(beta,index=ebs,columns=datafiles)


    #cr_df.to_csv("%s_cr.tsv" % args.output,sep='\t')
    #psnr_df.to_csv("%s_psnr.tsv" % args.output,sep='\t')
    overall_cr_df.to_csv("%s_overall_cr.tsv" % args.output,sep='\t')
    overall_psnr_df.to_csv("%s_overall_psnr.tsv" % args.output,sep='\t')
    #alpha_df.to_csv("%s_alpha.tsv" % args.output,sep='\t')
    #beta_df.to_csv("%s_beta.tsv" % args.output,sep='\t')

    if args.ssim: 
        overall_ssim=np.mean(ssim,axis=1)
        ssim_df=pd.DataFrame(ssim,index=ebs,columns=datafiles)
        overall_ssim_df=pd.DataFrame(overall_ssim,index=ebs,columns=["overall_ssim"])
        #ssim_df.to_csv("%s_ssim.tsv" % args.output,sep='\t')
        overall_ssim_df.to_csv("%s_overall_ssim.tsv" % args.output,sep='\t')
        
    if (args.autocorr):
        overall_ac=np.mean(ac,axis=1)
        ac_df=pd.DataFrame(ac,index=ebs,columns=datafiles)
        overall_ac_df=pd.DataFrame(overall_ac,index=ebs,columns=["overall_ac"])
        #ac_df.to_csv("%s_ac.tsv" % args.output,sep='\t')
        overall_ac_df.to_csv("%s_overall_ac.tsv" % args.output,sep='\t')