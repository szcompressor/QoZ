import sys
import os 

#tt=sys.argv[1]
tuning_target_dict={"psnr":"TUNING_TARGET_RD","cr":"TUNING_TARGET_CR","ssim":"TUNING_TARGET_SSIM","ac":"TUNING_TARGET_AC"}

if not os.path.exists("configs"):
    os.mkdir("configs")
datasets=["cesm","miranda","nyx","scale","hurricane"]
targets=["cr","psnr","ssim","ac"]
for dataset in datasets:
    testLorenzo=0
    if dataset=="cesm":
        maxStep=64
        sampleBlockSize=64
        autoTuningRate=0.01
        predictorTuningRate=0.01
        levelwisePredictionSelection=6
    else:
        maxStep=32
        sampleBlockSize=16
        autoTuningRate=0.005
        predictorTuningRate=0.005
        levelwisePredictionSelection=4
    for target in targets:

        tuning_target=tuning_target_dict[target]
        configstr="[GlobalSettings]\nCmprAlgo = ALGO_INTERP_LORENZO \ntuningTarget = %s \n[AlgoSettings]\nautoTuningRate = %f \npredictorTuningRate= %f \nlevelwisePredictionSelection = %d \nmaxStep = %d \ntestLorenzo = %d \nsampleBlockSize = %d \nverbose = 0 \n" % \
        (tuning_target,autoTuningRate,predictorTuningRate,levelwisePredictionSelection,maxStep,testLorenzo,sampleBlockSize) 
        with open(os.path.join("configs", "%s_%s.config" % (dataset,target) ),"w") as f:
            f.write(configstr)