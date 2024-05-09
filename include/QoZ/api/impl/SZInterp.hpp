#ifndef SZ3_SZINTERP_HPP
#define SZ3_SZINTERP_HPP

#include "QoZ/compressor/SZInterpolationCompressor.hpp"

#include "QoZ/compressor/deprecated/SZBlockInterpolationCompressor.hpp"



#include "QoZ/quantizer/IntegerQuantizer.hpp"
#include "QoZ/lossless/Lossless_zstd.hpp"
#include "QoZ/utils/Iterator.hpp"
#include "QoZ/utils/Sample.hpp"
#include "QoZ/utils/Statistic.hpp"
#include "QoZ/utils/Extraction.hpp"
#include "QoZ/utils/QuantOptimization.hpp"
#include "QoZ/utils/Config.hpp"
#include "QoZ/utils/Metrics.hpp"
#include "QoZ/api/impl/SZLorenzoReg.hpp"


//#include <cunistd>
#include <cmath>
#include <memory>
#include <limits>
#include <cstring>
#include <cstdlib>


template<class T, QoZ::uint N>
char *SZ_compress_Interp(QoZ::Config &conf, T *data, size_t &outSize) {

//    std::cout << "****************** Interp Compression ****************" << std::endl;
//    std::cout << "Interp Op          = " << interpAlgo << std::endl
//              << "Direction          = " << direction << std::endl
//              << "SZ block size      = " << blockSize << std::endl
//              << "Interp block size  = " << interpBlockSize << std::endl;

    assert(N == conf.N);
    assert(conf.cmprAlgo == QoZ::ALGO_INTERP);
    QoZ::calAbsErrorBound(conf, data);

    //conf.print();
    
    auto sz = QoZ::SZInterpolationCompressor<T, N, QoZ::LinearQuantizer<T>, QoZ::HuffmanEncoder<int>, QoZ::Lossless_zstd>(
            QoZ::LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2),
            QoZ::HuffmanEncoder<int>(),
            QoZ::Lossless_zstd());

   
    //QoZ::Timer timer;

    //timer.start();
    char *cmpData = (char *) sz.compress(conf, data, outSize);
     //double incall_time = timer.stop();
    //std::cout << "incall time = " << incall_time << "s" << std::endl;
    return cmpData;
}

template<class T, QoZ::uint N>
void SZ_decompress_Interp(QoZ::Config &conf, char *cmpData, size_t cmpSize, T *decData) {
    assert(conf.cmprAlgo == QoZ::ALGO_INTERP);
    QoZ::uchar const *cmpDataPos = (QoZ::uchar *) cmpData;
   
        
     auto sz = QoZ::SZInterpolationCompressor<T, N, QoZ::LinearQuantizer<T>, QoZ::HuffmanEncoder<int>, QoZ::Lossless_zstd>(
                QoZ::LinearQuantizer<T>(),
                QoZ::HuffmanEncoder<int>(),
                QoZ::Lossless_zstd());
       
        sz.decompress(cmpDataPos, cmpSize, decData);
        
}


template<class T, QoZ::uint N>
double do_not_use_this_interp_compress_block_test(T *data, std::vector<size_t> dims, size_t num,
                                                  double eb, int interp_op, int direction_op, int block_size) {
    std::vector<T> data1(data, data + num);
    size_t outSize = 0;
    QoZ::Config conf;
    conf.absErrorBound = eb;
    conf.setDims(dims.begin(), dims.end());
    conf.blockSize = block_size;
    conf.interpMeta.interpAlgo = interp_op;
    conf.interpMeta.interpDirection = direction_op;
    auto sz = QoZ::SZBlockInterpolationCompressor<T, N, QoZ::LinearQuantizer<T>, QoZ::HuffmanEncoder<int>, QoZ::Lossless_zstd>(
            QoZ::LinearQuantizer<T>(eb),
            QoZ::HuffmanEncoder<int>(),
            QoZ::Lossless_zstd());
    char *cmpData = (char *) sz.compress(conf, data1.data(), outSize);
    delete[]cmpData;
    auto compression_ratio = num * sizeof(T) * 1.0 / outSize;
    return compression_ratio;
}



template<class T, QoZ::uint N>
inline void init_alphalist(std::vector<double> &alpha_list,const double &rel_bound, QoZ::Config &conf){

    
    /*
    if(conf.linearReduce){
        alpha_list={0,0.1,0.2,0.3,0.4,0.5};

    }
    */
    //else{
        if (conf.tuningTarget!=QoZ::TUNING_TARGET_CR){
           
            if(conf.abList==0)            
                alpha_list={1,1.25,1.5,1.75,2};
            else if(conf.abList==1)
                alpha_list={1,1.25,1.5,1.75,2,2.25,2.5};
            else
                alpha_list={1,1.25,1.5,1.75,2,2.25,2.5,2.75,3};
           
        }
        else{
            
            alpha_list={-1,1,1.25,1.5,1.75,2};
           
        }
    //}
}
template<class T, QoZ::uint N>
inline void init_betalist(std::vector<double> &beta_list,const double &rel_bound, QoZ::Config &conf){
   
    /*
    if(conf.linearReduce){
        beta_list={1,0.75,0.5,0.33,0.25};
    }
    */
    //else{
        if (conf.tuningTarget!=QoZ::TUNING_TARGET_CR){    
            
            beta_list={1.5,2,3,4};//may remove 1.5
           
        }
        else {
          
            beta_list={-1,1.5,2,3};
           
        }
    //}
}




template<class T, QoZ::uint N>
void sampleBlocks(T *data,std::vector<size_t> &dims, size_t sampleBlockSize,std::vector< std::vector<T> > & sampled_blocks,double sample_rate,int profiling ,std::vector<std::vector<size_t> > &starts,int var_first=0){
    for(int i=0;i<sampled_blocks.size();i++){
                std::vector< T >().swap(sampled_blocks[i]);                
            }
            std::vector< std::vector<T> >().swap(sampled_blocks);
    for(int i=0;i<sampled_blocks.size();i++){
        std::vector< T >().swap(sampled_blocks[i]);                  
    }
    std::vector< std::vector<T> >().swap(sampled_blocks);                               
    size_t totalblock_num=1;
    for(int i=0;i<N;i++){                        
        totalblock_num*=(int)((dims[i]-1)/sampleBlockSize);
    }               
    size_t idx=0,block_idx=0;   
    if(profiling){
        size_t num_filtered_blocks=starts.size();    
        if(var_first==0){  
            size_t sample_stride=(size_t)(num_filtered_blocks/(totalblock_num*sample_rate));
            if(sample_stride<=0)
                sample_stride=1;
            
            for(size_t i=0;i<num_filtered_blocks;i+=sample_stride){
                std::vector<T> s_block;
                QoZ::sample_blocks<T,N>(data, s_block,dims, starts[i],sampleBlockSize+1);
                sampled_blocks.push_back(s_block);
                
            }
            
        }
        else{
            std::vector< std::pair<double,std::vector<size_t> > >block_heap;
            for(size_t i=0;i<num_filtered_blocks;i++){
                double mean,sigma2,range;
                QoZ::blockwise_profiling<T>(data,dims, starts[i],sampleBlockSize+1, mean,sigma2,range);
                block_heap.push_back(std::pair<double,std::vector<size_t> >(sigma2,starts[i]));
                
            }
            std::make_heap(block_heap.begin(),block_heap.end());
          

            size_t sampled_block_num=totalblock_num*sample_rate;
            if(sampled_block_num>num_filtered_blocks)
                sampled_block_num=num_filtered_blocks;
            if(sampled_block_num==0)
                sampled_block_num=1;

            for(size_t i=0;i<sampled_block_num;i++){
                std::vector<T> s_block;
             
                QoZ::sample_blocks<T,N>(data, s_block,dims, block_heap.front().second,sampleBlockSize+1);
              
                sampled_blocks.push_back(s_block);
                std::pop_heap(block_heap.begin(),block_heap.end());
                block_heap.pop_back();
               
            }
        }
    }               
    else{
        if(var_first==0){
            size_t sample_stride=(size_t)(1.0/sample_rate);
            if(sample_stride<=0)
                sample_stride=1;
            if (N==2){                        
                for (size_t x_start=0;x_start<dims[0]-sampleBlockSize;x_start+=sampleBlockSize){                           
                    for (size_t y_start=0;y_start<dims[1]-sampleBlockSize;y_start+=sampleBlockSize){
                        if (idx%sample_stride==0){
                            std::vector<size_t> starts{x_start,y_start};
                            std::vector<T> s_block;
                            QoZ::sample_blocks<T,N>(data, s_block,dims, starts,sampleBlockSize+1);
                            sampled_blocks.push_back(s_block);
                        }
                        idx+=1;
                    }
                }
            }
            else if (N==3){                  
                for (size_t x_start=0;x_start<dims[0]-sampleBlockSize;x_start+=sampleBlockSize){                          
                    for (size_t y_start=0;y_start<dims[1]-sampleBlockSize;y_start+=sampleBlockSize){
                        for (size_t z_start=0;z_start<dims[2]-sampleBlockSize;z_start+=sampleBlockSize){
                            if (idx%sample_stride==0){
                                std::vector<size_t> starts{x_start,y_start,z_start};
                                std::vector<T> s_block;
                                QoZ::sample_blocks<T,N>(data, s_block,dims, starts,sampleBlockSize+1);
                                sampled_blocks.push_back(s_block);
                            }
                            idx+=1;
                        }
                    }
                }
            }
        }
        else{
            std::vector <std::vector<size_t> > blocks_starts;
            if (N==2){  
                for (size_t x_start=0;x_start<dims[0]-sampleBlockSize;x_start+=sampleBlockSize){                           
                    for (size_t y_start=0;y_start<dims[1]-sampleBlockSize;y_start+=sampleBlockSize){
                       
                            blocks_starts.push_back(std::vector<size_t>{x_start,y_start});
                    }
                }

            }
            else if (N==3){           
                for (size_t x_start=0;x_start<dims[0]-sampleBlockSize;x_start+=sampleBlockSize){                          
                    for (size_t y_start=0;y_start<dims[1]-sampleBlockSize;y_start+=sampleBlockSize){
                        for (size_t z_start=0;z_start<dims[2]-sampleBlockSize;z_start+=sampleBlockSize){
                            blocks_starts.push_back(std::vector<size_t>{x_start,y_start,z_start});
                        }
                    }
                }
            

                std::vector< std::pair<double,std::vector<size_t> > >block_heap;
                for(size_t i=0;i<totalblock_num;i++){
                    double mean,sigma2,range;
                    QoZ::blockwise_profiling<T>(data,dims, blocks_starts[i],sampleBlockSize+1, mean,sigma2,range);
                    block_heap.push_back(std::pair<double,std::vector<size_t> >(sigma2,blocks_starts[i]));
                }
                std::make_heap(block_heap.begin(),block_heap.end());
                size_t sampled_block_num=totalblock_num*sample_rate;
                if(sampled_block_num==0)
                    sampled_block_num=1;
                for(size_t i=0;i<sampled_block_num;i++){
                    std::vector<T> s_block;
                    QoZ::sample_blocks<T,N>(data, s_block,dims, block_heap.front().second,sampleBlockSize+1);
                    sampled_blocks.push_back(s_block);
                    std::pop_heap(block_heap.begin(),block_heap.end());
                    block_heap.pop_back();
                }

            }
        }
    }
}


template<class T, QoZ::uint N>
std::pair<double,double> CompressTest(const QoZ::Config &conf,const std::vector< std::vector<T> > & sampled_blocks,QoZ::ALGO algo = QoZ::ALGO_INTERP,
                    QoZ::TUNING_TARGET tuningTarget=QoZ::TUNING_TARGET_RD,bool useFast=true,double profiling_coeff=1,const std::vector<double> &orig_means=std::vector<double>(),
                    const std::vector<double> &orig_sigma2s=std::vector<double>(),const std::vector<double> &orig_ranges=std::vector<double>(),const std::vector<T> &flattened_sampled_data=std::vector<T>()){
    QoZ::Config testConfig(conf);
    size_t ssim_size=conf.SSIMBlockSize;    
    if(algo == QoZ::ALGO_LORENZO_REG){
        testConfig.cmprAlgo = QoZ::ALGO_LORENZO_REG;
        testConfig.dims=conf.dims;
        testConfig.num=conf.num;
        testConfig.lorenzo = true;
        testConfig.lorenzo2 = true;
        testConfig.regression = false;
        testConfig.regression2 = false;
        testConfig.openmp = false;
        testConfig.blockSize = 5;//why?
        testConfig.quantbinCnt = 65536 * 2;
    }
    double square_error=0.0;
    double bitrate=0.0;
    double metric=0.0;
    size_t sampleBlockSize=testConfig.sampleBlockSize;
    size_t num_sampled_blocks=sampled_blocks.size();
    size_t per_block_ele_num=pow(sampleBlockSize+1,N);
    size_t ele_num=num_sampled_blocks*per_block_ele_num;
    std::vector<T> cur_block(testConfig.num,0);
    std::vector<int> q_bins;
    std::vector<std::vector<int> > block_q_bins;
    std::vector<size_t> q_bin_counts;
    std::vector<T> flattened_cur_blocks;
    size_t idx=0;   
    QoZ::concepts::CompressorInterface<T> *sz;
    size_t totalOutSize=0;
    if(algo == QoZ::ALGO_LORENZO_REG){
        auto quantizer = QoZ::LinearQuantizer<T>(testConfig.absErrorBound, testConfig.quantbinCnt / 2);
        if (useFast &&N == 3 && !testConfig.regression2) {
            sz = QoZ::make_sz_general_compressor<T, N>(QoZ::make_sz_fast_frontend<T, N>(testConfig, quantizer), QoZ::HuffmanEncoder<int>(),
                                                                   QoZ::Lossless_zstd());
        }
        else{
            sz = make_lorenzo_regression_compressor<T, N>(testConfig, quantizer, QoZ::HuffmanEncoder<int>(), QoZ::Lossless_zstd());

        }
    }
    else if(algo == QoZ::ALGO_INTERP){

        sz =  new QoZ::SZInterpolationCompressor<T, N, QoZ::LinearQuantizer<T>, QoZ::HuffmanEncoder<int>, QoZ::Lossless_zstd>(
                        QoZ::LinearQuantizer<T>(testConfig.absErrorBound),
                        QoZ::HuffmanEncoder<int>(),
                        QoZ::Lossless_zstd());

    }
    else{
        std::cout<<"algo type error!"<<std::endl;
        return std::pair<double,double>(0,0);
    }
                           
    for (int k=0;k<num_sampled_blocks;k++){
        size_t sampleOutSize;
        std::vector<T> cur_block(testConfig.num);
        std::copy(sampled_blocks[k].begin(),sampled_blocks[k].end(),cur_block.begin());
        
        char *cmprData;
         
        
        cmprData = (char*)sz->compress(testConfig, cur_block.data(), sampleOutSize,1);

        delete[]cmprData;
        
        

        
        if(algo==QoZ::ALGO_INTERP){
            block_q_bins.push_back(testConfig.quant_bins);
        }

        if(tuningTarget==QoZ::TUNING_TARGET_RD){
            if(algo==QoZ::ALGO_INTERP)
                square_error+=testConfig.decomp_square_error;
            else{
               
                for(size_t j=0;j<per_block_ele_num;j++){
                    T value=sampled_blocks[k][j]-cur_block[j];
                    square_error+=value*value;
                
                }
            }
        }
        else if (tuningTarget==QoZ::TUNING_TARGET_SSIM){
            size_t ssim_block_num=orig_means.size();                       
            double mean=0,sigma2=0,cov=0,range=0;
            double orig_mean=0,orig_sigma2=0,orig_range=0;  
            std::vector<size_t>block_dims(N,sampleBlockSize+1);                      
            if(N==2){
                for (size_t i=0;i+ssim_size<sampleBlockSize+1;i+=ssim_size){
                    for (size_t j=0;j+ssim_size<sampleBlockSize+1;j+=ssim_size){
                        orig_mean=orig_means[idx];
                        orig_sigma2=orig_sigma2s[idx];
                        orig_range=orig_ranges[idx];
                        std::vector<size_t> starts{i,j};
                        QoZ::blockwise_profiling<T>(cur_block.data(),block_dims,starts,ssim_size,mean,sigma2,range);
                        cov=QoZ::blockwise_cov<T>(sampled_blocks[k].data(),cur_block.data(),block_dims,starts,ssim_size,orig_mean,mean);
                        metric+=QoZ::SSIM(orig_range,orig_mean,orig_sigma2,mean,sigma2,cov)/ssim_block_num;
                        idx++;


                    }
                }
            }
            else if(N==3){
                for (size_t i=0;i+ssim_size<sampleBlockSize+1;i+=ssim_size){
                    for (size_t j=0;j+ssim_size<sampleBlockSize+1;j+=ssim_size){
                        for (size_t kk=0;kk+ssim_size<sampleBlockSize+1;kk+=ssim_size){
                            orig_mean=orig_means[idx];
                            orig_sigma2=orig_sigma2s[idx];
                            orig_range=orig_ranges[idx];
                            std::vector<size_t> starts{i,j,kk};
                            QoZ::blockwise_profiling<T>(cur_block.data(),block_dims,starts,ssim_size,mean,sigma2,range);
                            cov=QoZ::blockwise_cov<T>(sampled_blocks[k].data(),cur_block.data(),block_dims,starts,ssim_size,orig_mean,mean);
                            //printf("%.8f %.8f %.8f %.8f %.8f %.8f %.8f\n",orig_range,orig_sigma2,orig_mean,range,sigma2,mean,cov);
                            metric+=QoZ::SSIM(orig_range,orig_mean,orig_sigma2,mean,sigma2,cov)/ssim_block_num;
                     
                            idx++;
                        }
                    }
                }
            }
        }
        else if (tuningTarget==QoZ::TUNING_TARGET_AC){
            flattened_cur_blocks.insert(flattened_cur_blocks.end(),cur_block.begin(),cur_block.end());
        }                      
    }
    if(algo==QoZ::ALGO_INTERP ){
        q_bin_counts=testConfig.quant_bin_counts;
        size_t level_num=q_bin_counts.size();
        size_t last_pos=0;
        for(int k=level_num-1;k>=0;k--){
            for (size_t l =0;l<num_sampled_blocks;l++){
                for (size_t m=last_pos;m<q_bin_counts[k];m++){
                    q_bins.push_back(block_q_bins[l][m]);
                }
            }
            last_pos=q_bin_counts[k];
        }      
    }
    size_t sampleOutSize;
    
    auto cmprData=sz->encoding_lossless(totalOutSize,q_bins);             
    delete[]cmprData;
  
    
    bitrate=8*double(totalOutSize)/ele_num;
    
    bitrate*=profiling_coeff;
    if(tuningTarget==QoZ::TUNING_TARGET_RD){
        double mse=square_error/ele_num;
        mse*=profiling_coeff;      
        
        metric=QoZ::PSNR(testConfig.rng,mse);
    }
    else if (tuningTarget==QoZ::TUNING_TARGET_AC){                       
        metric=1.0-QoZ::autocorrelation<T>(flattened_sampled_data.data(),flattened_cur_blocks.data(),ele_num);                        
    }                    
    //printf("%.2f %.2f %.4f %.2f\n",testConfig.alpha,testConfig.beta,bitrate,metric);   

    if(algo==QoZ::ALGO_LORENZO_REG)    {
        bitrate*=testConfig.lorenzoBrFix;
    }
    delete sz;
    return std::pair(bitrate,metric);
}

std::pair <double,double> setABwithRelBound(double rel_bound,int configuration=0){

    double cur_alpha=-1,cur_beta=-1;
    if(configuration==0){              
        if (rel_bound>=0.01){
            cur_alpha=2;
            cur_beta=2;
        }
        else if (rel_bound>=0.007){
            cur_alpha=1.75;
            cur_beta=2;
        }                 
        else if (rel_bound>=0.004){
            cur_alpha=1.5;
            cur_beta=2;
        }                   
        else if (rel_bound>0.001){
            cur_alpha=1.25;
            cur_beta=1.5;
        }
        else {
            cur_alpha=1;
            cur_beta=1;
        }
    }
    else if(configuration==1){                
        if (rel_bound>=0.01){
            cur_alpha=2;
            cur_beta=4;
        }
        else if (rel_bound>=0.007){
            cur_alpha=1.75;
            cur_beta=3;
        }
        else if (rel_bound>=0.004){
            cur_alpha=1.5;
            cur_beta=2;
        }           
        else if (rel_bound>0.001){
            cur_alpha=1.25;
            cur_beta=1.5;
        }
        else {
                cur_alpha=1;
                cur_beta=1;
            }
    }
    else if(configuration==2){                
        if (rel_bound>=0.01){
            cur_alpha=2;
            cur_beta=4;
        }
        else if (rel_bound>=0.007){
            cur_alpha=1.75;
            cur_beta=3;
        }                    
        else if (rel_bound>=0.004){
            cur_alpha=1.5;
            cur_beta=2;
        }
        else if (rel_bound>0.001){
            cur_alpha=1.5;
            cur_beta=1.5;
        }
        else if (rel_bound>0.0005){
            cur_alpha=1.25;
            cur_beta=1.5;
        }
        else {
            cur_alpha=1;
            cur_beta=1;
        }
    }
    return std::pair<double,double>(cur_alpha,cur_beta);
}

void setLorenzoFixRates(QoZ::Config &conf,double rel_bound){
    double e1=1e-5;
    double e2=1e-4;
    double e3=1e-3;
    //double e4=1e-1;
    /*
    double f1=conf.sampleBlockSize>=64?2: 1;
    // double f2=1.1;old
    double f2=conf.sampleBlockSize>=64?3:1.2; 
    // double f3=1.2;//old need to raise 
    //double f3=1.3;
    double f3=conf.sampleBlockSize>=64?4:1.3;
    */
    double f1=conf.sampleBlockSize>=64?2: 1;
    // double f2=1.1;old
    double f2=conf.sampleBlockSize>=64?3:1.2; 
    // double f3=1.2;//old need to raise 
    //double f3=1.3;
    double f3=conf.sampleBlockSize>=64?4:1.3;
    if(rel_bound<=e1)
        conf.lorenzoBrFix=f1;
    else if(rel_bound<=e2)
        conf.lorenzoBrFix=f1-(f1-f2)*(rel_bound-e1)/(e2-e1);
    else if (rel_bound<=e3)
        conf.lorenzoBrFix=f2-(f2-f3)*(rel_bound-e2)/(e3-e2);
    else 
        conf.lorenzoBrFix=f3;
}

template<class T, QoZ::uint N>
double Tuning(QoZ::Config &conf, T *data){
   
    T rng=conf.rng;
    double rel_bound = conf.relErrorBound>0?conf.relErrorBound:conf.absErrorBound/rng;
    if(rel_bound>1e-3 or conf.tuningTarget==QoZ::TUNING_TARGET_SSIM)//rencently changed, need to fix later
        conf.testLorenzo=0;
   // QoZ::Timer timer(true);
    //timer.stop("")
    if(conf.QoZ>0){
        
        //testLorenzo?
        //deactivate FZ-related features.
        
        //conf.var_first=0;
        
         /*
        //deactivate high-level qoz features;

        conf.dynamicDimCoeff=0;
        conf.crossBlock=0;
        conf.blockwiseTuning=0;
        conf.testLorenzo=0;
        conf.multiDimInterp=0;
        conf.naturalSpline=0;
        conf.fullAdjacentInterp=0;
        conf.freezeDimTest=0;
        */

        //activate
        conf.profiling=1;
        if(conf.autoTuningRate<=0)
            conf.autoTuningRate = (N<=2?0.01:0.005);
        if(conf.predictorTuningRate<=0)
            conf.predictorTuningRate = (N<=2?0.01:0.005);
        if (conf.maxStep<=0){
            std::array<size_t,4> anchor_strides={256,64,32,16};
            conf.maxStep = anchor_strides[N-1];
        }
        if (conf.levelwisePredictionSelection<=0)
            conf.levelwisePredictionSelection = (N<=2?5:4);
        if (conf.sampleBlockSize<=0){
            
            conf.sampleBlockSize = (N<=2?64:32);
        }

        if(conf.QoZ>=2){
            conf.testLorenzo=1;
            conf.multiDimInterp=1;
            conf.naturalSpline=1;
            conf.fullAdjacentInterp=1;
            conf.freezeDimTest=1;
            
        }
        if(conf.QoZ>=3){
            conf.dynamicDimCoeff=1;
        }
        if(conf.QoZ>=4){
            conf.crossBlock=1;
            conf.blockwiseTuning=1;
            if(conf.blockwiseSampleRate<1.0)
                conf.blockwiseSampleRate=3.0;
        }
    }   
    //Deactivate several features when dim not fit
    if(N!=2 and N!=3){
       // conf.QoZ=0; comment it for level-wise eb
        conf.autoTuningRate=0;
        conf.predictorTuningRate=0;
        conf.levelwisePredictionSelection=0;
        conf.multiDimInterp=0;
        conf.naturalSpline=0;
        conf.fullAdjacentInterp=0;
        conf.freezeDimTest=0;
        conf.dynamicDimCoeff=0;
        conf.blockwiseTuning=0;
    }
    

    if(conf.multiDimInterp==0)
        conf.dynamicDimCoeff=0;
    size_t sampling_num, sampling_block;
    double best_interp_cr=0.0;
    double best_lorenzo_ratio=0.0;
    bool useInterp=true;        
    std::vector<size_t> sample_dims(N);
    std::vector<T> sampling_data;
    double anchor_rate=0;
    int max_interp_level = -1;

    for (size_t i = 0; i < N; i++) {
        if ( max_interp_level < ceil(log2(conf.dims[i]))) {
             max_interp_level = (uint) ceil(log2(conf.dims[i]));
        }
                
    }
    
    if (conf.maxStep>0){
        anchor_rate=1/(pow(conf.maxStep,N));   
        int temp_max_interp_level=(uint)log2(conf.maxStep);//to be catious: the max_interp_level is different from the ones in szinterpcompressor, which includes the level of anchor grid.
        if (temp_max_interp_level<=max_interp_level){                  
            max_interp_level=temp_max_interp_level;
        }
        if (conf.levelwisePredictionSelection>max_interp_level)
            conf.levelwisePredictionSelection=max_interp_level;
    }
            
    /*
    std::vector<int> op_candidates={QoZ::INTERP_ALGO_LINEAR,QoZ::INTERP_ALGO_CUBIC};
    std::vector<int> dir_candidates={0,QoZ::factorial(N)-1};
     
    if(conf.multiDimInterp){
        dir_candidates.push_back(QoZ::factorial(N));
    }
    */
    /*
    std::vector<std::vector<uint8_t> > interpAlgo_lists(conf.waveletAutoTuning+1);
    std::vector<std::vector<uint8_t> > interpDirection_lists(conf.waveletAutoTuning+1);
    std::vector<std::vector<uint8_t> > cubicSplineType_lists(conf.waveletAutoTuning+1);
    std::vector<uint8_t> bestInterpAlgos(conf.waveletAutoTuning+1);
    std::vector<uint8_t> bestInterpDirections(conf.waveletAutoTuning+1);
    std::vector<uint8_t> bestCubicSplineTypes(conf.waveletAutoTuning+1);
    */

    std::vector<QoZ::Interp_Meta> bestInterpMeta_list(conf.levelwisePredictionSelection);
    QoZ::Interp_Meta bestInterpMeta;


    size_t shortest_edge=conf.dims[0];
    for (size_t i=0;i<N;i++){
        shortest_edge=conf.dims[i]<shortest_edge?conf.dims[i]:shortest_edge;
    }
 

    
    if (conf.sampleBlockSize<=0){
        conf.sampleBlockSize = (N==2?64:32);
            
    }

    
    
    size_t minimum_sbs=8;
    if (conf.sampleBlockSize<minimum_sbs)
        conf.sampleBlockSize=minimum_sbs;


    while(conf.sampleBlockSize>=shortest_edge)
        conf.sampleBlockSize/=2;


    

    while(conf.autoTuningRate>0 and conf.sampleBlockSize>=2*minimum_sbs and (pow(conf.sampleBlockSize+1,N)/(double)conf.num)>1.5*conf.autoTuningRate)
        conf.sampleBlockSize/=2;

    if (conf.sampleBlockSize<minimum_sbs){
        conf.predictorTuningRate=0.0;
        conf.autoTuningRate=0.0;
    }
    else{
        int max_lps_level=(uint)log2(conf.sampleBlockSize);//to be catious: the max_interp_level is different from the ones in szinterpcompressor, which includes the level of anchor grid.

        if (conf.levelwisePredictionSelection>max_lps_level)
            conf.levelwisePredictionSelection=max_lps_level;
    }

    std::vector< std::vector<T> > sampled_blocks;
    size_t sampleBlockSize=conf.sampleBlockSize;
    size_t num_sampled_blocks;
    size_t per_block_ele_num;
    size_t ele_num;

    
           
    size_t totalblock_num=1;  
    for(int i=0;i<N;i++){                      
        totalblock_num*=(size_t)((conf.dims[i]-1)/sampleBlockSize);
    }

    std::vector<std::vector<size_t> >starts;
    if((conf.autoTuningRate>0 or conf.predictorTuningRate>0) and conf.profiling){      
        conf.profStride=conf.sampleBlockSize/4;
        if(N==2){
            QoZ::profiling_block_2d<T,N>(data,conf.dims,starts,sampleBlockSize,conf.absErrorBound,conf.profStride);
        }
        else if (N==3){
            QoZ::profiling_block_3d<T,N>(data,conf.dims,starts,sampleBlockSize,conf.absErrorBound,conf.profStride);
        }
       
    }


    size_t num_filtered_blocks=starts.size();
    if(num_filtered_blocks<=(int)(0.3*conf.predictorTuningRate))//temp. to refine
        conf.profiling=0;
    double profiling_coeff=1;//It seems that this coefficent is useless. Need further test
  
    if(conf.profiling){// and conf.profilingFix){
        profiling_coeff=((double)num_filtered_blocks)/(totalblock_num);
    }
    std::vector<size_t> global_dims=conf.dims;
    size_t global_num=conf.num;
    if(conf.autoTuningRate>0){
        if (conf.testLorenzo>0)
            setLorenzoFixRates(conf,rel_bound);
    }
    

    bool blockwiseTuning=conf.blockwiseTuning;
    conf.blockwiseTuning=false;
    /*
    if(conf.blockwiseTuning){
        conf.predictorTuningRate=0.0;
        if(conf.adaptiveMultiDimStride>0 and N==3){//This is a very immature method. another method is to temply set conf.blockwiseTuning=0 then run a predictor tuning for dim fuse. More accurate but slower.
            std::vector<double> cubic_noknot_vars;
            QoZ::calculate_interp_error_vars<T,N>(data, global_dims,cubic_noknot_vars,1,0,conf.adaptiveMultiDimStride,conf.absErrorBound);
            size_t fused_dim=0;
            double cur_vars=cubic_noknot_vars[0];
            for(size_t i=1;i<N;i++){
                if(cubic_noknot_vars[i]>cur_vars){
                    fused_dim=i;
                    cur_vars=cubic_noknot_vars[i];
                }
            }
            double c=45.0;//need to tune
            for(size_t i=0;i<N;i++){
                if(i==fused_dim)
                    continue;
                if (cubic_noknot_vars[i]*c>=cur_vars or conf.dims[fused_dim]>10*conf.dims[i]){//second for qmcpack
                    fused_dim=-1;
                    break;
                }
            }

            conf.fused_dim=fused_dim;

        }
    }
    */

    /*
    if(conf.verbose){
        timer.stop("Prep");
        timer.start();
    }
    */

    if (conf.predictorTuningRate>0 and conf.predictorTuningRate<1){
        //int ori_sperr=conf.sperr;//temp
        //conf.sperr=0;
        if (conf.verbose)
            std::cout<<"Predictor tuning started."<<std::endl;
        double o_alpha=conf.alpha;
        double o_beta=conf.beta;
                    
        //if(!conf.waveletTest or conf.predictorTuningRate!=conf.waveletTuningRate or conf.profiling>0){
            sampleBlocks<T,N>(data,conf.dims,sampleBlockSize,sampled_blocks,conf.predictorTuningRate,conf.profiling,starts,conf.var_first);
            //std::cout<<sampleBlockSize<<std::endl;
            //std::cout<<sampled_blocks.size()<<std::endl;
        //}        
        num_sampled_blocks=sampled_blocks.size();
        per_block_ele_num=pow(sampleBlockSize+1,N);
        ele_num=num_sampled_blocks*per_block_ele_num;
        conf.dims=std::vector<size_t>(N,sampleBlockSize+1);
        conf.num=per_block_ele_num;
        std::vector<T> cur_block(per_block_ele_num,0);
        
        
        
            
        double ori_eb=conf.absErrorBound;
        std::vector<size_t> coeffs_size;
        
        if(conf.testLorenzo and conf.autoTuningRate==0 ){

            std::pair<double,double> results=CompressTest<T,N>(conf, sampled_blocks,QoZ::ALGO_LORENZO_REG,QoZ::TUNING_TARGET_CR,false);
            best_lorenzo_ratio=sizeof(T)*8.0/results.first;
            
            if(conf.verbose)
                std::cout << "lorenzo best cr = " << best_lorenzo_ratio << std::endl;

        }

        if (conf.autoTuningRate>0){

           // if(conf.pdTuningAbConf<=2){               
                std::pair<double,double> ab=setABwithRelBound(rel_bound,0);//ori pdtuningqabconf
                conf.alpha=ab.first;
                conf.beta=ab.second;
           //}               
           // else{
            //    conf.alpha=conf.pdAlpha;
           //     conf.beta=conf.pdBeta;
           // }
        }
        std::vector<uint8_t> interpAlgo_Candidates={QoZ::INTERP_ALGO_LINEAR, QoZ::INTERP_ALGO_CUBIC};
        if(conf.quadInterp){//deprecated
            interpAlgo_Candidates.push_back(QoZ::INTERP_ALGO_QUAD);
        }

        //std::vector<int> interpAlgo_Candidates={QoZ::INTERP_ALGO_CUBIC};//temp. 
        std::vector<uint8_t> interpParadigm_Candidates={0};//
        std::vector<uint8_t> cubicSplineType_Candidates={0};
        std::vector<uint8_t> interpDirection_Candidates={0, (uint8_t)(QoZ::factorial(N) -1)};
        /*
        if(N>2)
            interpDirection_Candidates={0,1, 2,3,4,QoZ::factorial(N) -1};
        */
        std::vector<uint8_t> adjInterp_Candidates={0};//

        
        if(conf.multiDimInterp>0){
            for(size_t i=1;i<=conf.multiDimInterp;i++)
                interpParadigm_Candidates.push_back(i);
            //interpParadigm_Candidates.push_back(conf.multiDimInterp);
            /*
            
            */
        }

        if (conf.naturalSpline){
            cubicSplineType_Candidates.push_back(1);
        }


        
       // std::vector<int> interpDirection_Candidates={};//temp. 
        
        if(conf.fullAdjacentInterp){
            adjInterp_Candidates.push_back(1);
            //for(size_t i=1;i<=conf.fullAdjacentInterp;i++)
            //    adjInterp_Candidates.push_back(i);
        }
        
        //if(conf.mdCrossInterp)
         //   interpDirection_Candidates.push_back(2*QoZ::factorial(N)+1);
        if(conf.levelwisePredictionSelection>0){
            std::vector<QoZ::Interp_Meta> interpMeta_list(conf.levelwisePredictionSelection);
            /*
            std::vector<uint8_t> interpDirection_list(conf.levelwisePredictionSelection,0);
            std::vector<uint8_t> cubicSplineType_list(conf.levelwisePredictionSelection,0);
            */
            auto sz = QoZ::SZInterpolationCompressor<T, N, QoZ::LinearQuantizer<T>, QoZ::HuffmanEncoder<int>, QoZ::Lossless_zstd>(
                                    QoZ::LinearQuantizer<T>(conf.absErrorBound),
                                    QoZ::HuffmanEncoder<int>(),
                                    QoZ::Lossless_zstd());   
            double best_accumulated_interp_loss_1=0;
            double best_accumulated_interp_loss_2=0;
            std::vector<std::vector<double> > linear_interp_vars(conf.levelwisePredictionSelection),cubic_noknot_vars(conf.levelwisePredictionSelection),cubic_nat_vars(conf.levelwisePredictionSelection);
            //std::cout<<"a "<<conf.dynamicDimCoeff<<" "<<conf.freezeDimTest<<std::endl;
            for(int level=conf.levelwisePredictionSelection;level>0;level--){
               // std::cout<<level<<std::endl;
                int start_level=(level==conf.levelwisePredictionSelection?9999:level);
                int end_level=level-1;
                /*
                uint8_t bestInterpAlgo = QoZ::INTERP_ALGO_CUBIC;
                uint8_t bestDirection = 0;
                uint8_t bestSplineType=0;
                */
               // std::cout<<"a "<<level<<" "<<conf.dynamicDimCoeff<<" "<<conf.freezeDimTest<<std::endl;
                if((conf.multiDimInterp>0 and conf.dynamicDimCoeff>0) or (conf.freezeDimTest>0 and level==1 and N>=3) ){
                    
                    
                    size_t interp_stride=pow(2,level-1);
                    size_t stride;
                    double cur_eb=level>=3?conf.absErrorBound/2:conf.absErrorBound;
                    if(level>=3)
                        stride=2;
                    else if(level>=2)
                        stride= conf.adaptiveMultiDimStride<=4?2:conf.adaptiveMultiDimStride/2;
                    else
                        stride= conf.adaptiveMultiDimStride;
                    if(conf.multiDimInterp>0 and conf.dynamicDimCoeff){
                        QoZ::calculate_interp_error_vars<T,N>(data, global_dims,linear_interp_vars[level-1],0,0,stride,interp_stride,cur_eb);
                        QoZ::preprocess_vars<N>(linear_interp_vars[level-1]);
                       // for(auto x:linear_interp_vars[level-1])
                       //     std::cout<<x<<" ";
                       // std::cout<<std::endl;
                        if (conf.naturalSpline){
                            QoZ::calculate_interp_error_vars<T,N>(data, global_dims,cubic_nat_vars[level-1],1,1,stride,interp_stride,cur_eb);
                            QoZ::preprocess_vars<N>(cubic_nat_vars[level-1]);
                         //   for(auto x:cubic_nat_vars[level-1])
                          //      std::cout<<x<<" ";
                          //  std::cout<<std::endl;
                        }
                    }
                    QoZ::calculate_interp_error_vars<T,N>(data, global_dims,cubic_noknot_vars[level-1],1,0,stride,interp_stride,cur_eb);
                    QoZ::preprocess_vars<N>(cubic_noknot_vars[level-1]);
                    /*
                    for(auto x:cubic_noknot_vars[level-1])
                        std::cout<<x<<" ";
                    std::cout<<std::endl;
                    */
                }


                double best_interp_absloss=std::numeric_limits<double>::max();
                //conf.cmprAlgo = QoZ::ALGO_INTERP;    
                QoZ::Interp_Meta cur_meta;
                QoZ::Interp_Meta best_meta;

                for (auto &interp_op: interpAlgo_Candidates) {
                    cur_meta.interpAlgo=interp_op;
                    for (auto &interp_pd: interpParadigm_Candidates) {
                        cur_meta.interpParadigm=interp_pd;

                        

                        for (auto &interp_direction: interpDirection_Candidates) {
                            if ((interp_pd==1 or  (interp_pd==2 and N<=2)) and interp_direction!=0)
                                continue;
                            cur_meta.interpDirection=interp_direction;
                            for(auto &cubic_spline_type:cubicSplineType_Candidates){
                                if (interp_op!=QoZ::INTERP_ALGO_CUBIC and cubic_spline_type!=0)
                                    break;
                                cur_meta.cubicSplineType=cubic_spline_type;
                                for(auto adj_interp:adjInterp_Candidates){
                                    if (interp_op!=QoZ::INTERP_ALGO_CUBIC and adj_interp!=0)
                                        break;
                                    /*
                                    if (interp_direction==2 and level<=2)//???
                                        continue;
                                    */
                                    cur_meta.adjInterp=adj_interp;
                                    
                                    if(conf.dynamicDimCoeff>0 and interp_pd>0){
                                        if(interp_op==0){
                                            for(size_t i=0;i<N;i++)
                                                cur_meta.dimCoeffs[i]=linear_interp_vars[level-1][i];
                                        }
                                        else if (cubic_spline_type==0){
                                            for(size_t i=0;i<N;i++)
                                                cur_meta.dimCoeffs[i]=cubic_noknot_vars[level-1][i];
                                        }
                                        else{
                                            for(size_t i=0;i<N;i++)
                                                cur_meta.dimCoeffs[i]=cubic_nat_vars[level-1][i];
                                        }

                                    }
                                    
                                    conf.interpMeta=cur_meta;


                                    double cur_absloss=0;
                                    for (int i=0;i<num_sampled_blocks;i++){
                                        cur_block=sampled_blocks[i];  //not so efficient              
                                        size_t outSize=0;                              
                                        auto cmprData =sz.compress(conf, cur_block.data(), outSize,2,start_level,end_level);
                                        delete []cmprData;                              
                                        cur_absloss+=conf.decomp_square_error;
                                    }
                                    //std::cout<<(int)interp_op<<" "<<(int)interp_pd<<" "<<(int)interp_direction<<" "<<(int)cubic_spline_type<<" "<<(int)adj_interp<<" "<<cur_absloss<<std::endl; 
                                    if (cur_absloss<best_interp_absloss){
                                        best_meta=cur_meta;
                                        best_interp_absloss=cur_absloss;
                                    }
                                    cur_meta.dimCoeffs={1.0/3.0,1.0/3.0,1.0/3.0};
                    
                                }
                            }
                        }   
                    }
                }
                best_accumulated_interp_loss_1+=best_interp_absloss;
      
                /*
                interpAlgo_list[level-1]=bestInterpAlgo;
                interpDirection_list[level-1]=bestDirection;
                cubicSplineType_list[level-1]=bestSplineType;
                */

                interpMeta_list[level-1]=best_meta;
                    
                if(conf.pdTuningRealComp){
                    //place to add real compression,need to deal the problem that the sampled_blocks are changed. 
                    
                    conf.interpMeta=best_meta;
                    for (int i=0;i<num_sampled_blocks;i++){

                        size_t outSize=0;
                                   
                        auto cmprData =sz.compress(conf, sampled_blocks[i].data(), outSize,2,start_level,end_level);
                        delete []cmprData;
                    }
                    
                } 
                

            }
            //conf.interpAlgo_list=interpAlgo_list;
            //conf.interpDirection_list=interpDirection_list;
            
            bestInterpMeta_list=interpMeta_list;

            //determine 
            //bool fuse_dim_test=false;



            
            if(conf.pdTuningRealComp and ((conf.autoTuningRate>0 and conf.autoTuningRate==conf.predictorTuningRate) or (conf.adaptiveMultiDimStride>0 and N==3))){
                    //recover sample if real compression used                  
                sampleBlocks<T,N>(data,global_dims,sampleBlockSize,sampled_blocks,conf.predictorTuningRate,conf.profiling,starts,conf.var_first);
            }
            
                

            //frozendim
            
            if(conf.freezeDimTest and N>=3 ){

                std::vector<QoZ::Interp_Meta> tempmeta_list=conf.interpMeta_list;
                conf.interpMeta_list=interpMeta_list;      
                std::pair<double,double> results=CompressTest<T,N>(conf,sampled_blocks,QoZ::ALGO_INTERP,QoZ::TUNING_TARGET_CR,false);
                double best_interp_cr_1=sizeof(T)*8.0/results.first;     
                conf.interpMeta_list=tempmeta_list;
                




                //std::vector<double> cubic_noknot_vars;
                //QoZ::calculate_interp_error_vars<T,N>(data, global_dims,cubic_noknot_vars,1,0,conf.adaptiveMultiDimStride,conf.absErrorBound);
                size_t frozen_dim=0;
                double cur_weight=cubic_noknot_vars[0][0];
                for(size_t i=1;i<N;i++){
                    if(cubic_noknot_vars[0][i]<cur_weight){
                        frozen_dim=i;
                        cur_weight=cubic_noknot_vars[0][i];
                    }
                }
                if(frozen_dim==0)
                    interpDirection_Candidates={6,7};
                else if (frozen_dim==1)
                    interpDirection_Candidates={8,9};
                else
                    interpDirection_Candidates={10,11};


                for(int level=conf.levelwisePredictionSelection;level>0;level--){
               // std::cout<<level<<std::endl;
                    int start_level=(level==conf.levelwisePredictionSelection?9999:level);
                    int end_level=level-1;
                  
                    double best_interp_absloss=std::numeric_limits<double>::max();
                    //conf.cmprAlgo = QoZ::ALGO_INTERP;    
                    QoZ::Interp_Meta cur_meta;
                    QoZ::Interp_Meta best_meta;

                    for (auto &interp_op: interpAlgo_Candidates) {
                        cur_meta.interpAlgo=interp_op;
                        for (auto &interp_pd: interpParadigm_Candidates) {
                            if (interp_pd>1)
                                continue;
                            cur_meta.interpParadigm=interp_pd;

                            

                            for (auto &interp_direction: interpDirection_Candidates) {
                                if (interp_pd>=1  and interp_direction%2!=0)//only dims[0] matters.
                                    continue;
                                cur_meta.interpDirection=interp_direction;
                                for(auto &cubic_spline_type:cubicSplineType_Candidates){
                                    if (interp_op!=QoZ::INTERP_ALGO_CUBIC and cubic_spline_type!=0)
                                        break;
                                    cur_meta.cubicSplineType=cubic_spline_type;
                                    for(auto adj_interp:adjInterp_Candidates){
                                        if (interp_op!=QoZ::INTERP_ALGO_CUBIC and adj_interp!=0)
                                            break;
                                       
                                        cur_meta.adjInterp=adj_interp;

                                        if(conf.dynamicDimCoeff>0 and interp_pd>0){
                                            if(interp_op==0){
                                                for(size_t i=0;i<N;i++)
                                                    cur_meta.dimCoeffs[i]=linear_interp_vars[level-1][i];
                                            }
                                            else if (cubic_spline_type==0){
                                                for(size_t i=0;i<N;i++)
                                                    cur_meta.dimCoeffs[i]=cubic_noknot_vars[level-1][i];
                                            }
                                            else{
                                                for(size_t i=0;i<N;i++)
                                                    cur_meta.dimCoeffs[i]=cubic_nat_vars[level-1][i];
                                            }

                                        }
                                        
                                        conf.interpMeta=cur_meta;

                                        
                                        double cur_absloss=0;
                                        for (int i=0;i<num_sampled_blocks;i++){
                                            cur_block=sampled_blocks[i];  //not so efficient              
                                            size_t outSize=0;                              
                                            auto cmprData =sz.compress(conf, cur_block.data(), outSize,2,start_level,end_level);
                                            delete []cmprData;                              
                                            cur_absloss+=conf.decomp_square_error;
                                        }
                                       // std::cout<<(int)interp_op<<" "<<(int)interp_pd<<" "<<(int)interp_direction<<" "<<(int)cubic_spline_type<<" "<<(int)adj_interp<<" "<<cur_absloss<<std::endl; 
                                        if (cur_absloss<best_interp_absloss){
                                            best_meta=cur_meta;
                                            best_interp_absloss=cur_absloss;
                                        }
                                        cur_meta.dimCoeffs={1.0/3.0,1.0/3.0,1.0/3.0};
                                        
                                    }
                                }
                            }   
                        }
                    }
                    best_accumulated_interp_loss_2+=best_interp_absloss;
          
                 

                    interpMeta_list[level-1]=best_meta;
                        
                    if(conf.pdTuningRealComp){
                        //place to add real compression,need to deal the problem that the sampled_blocks are changed. 
                  
                        conf.interpMeta=best_meta;
                        for (int i=0;i<num_sampled_blocks;i++){

                            size_t outSize=0;
                                       
                            auto cmprData =sz.compress(conf, sampled_blocks[i].data(), outSize,2,start_level,end_level);
                            delete []cmprData;
                        }
                        
                    } 
                }
                

                tempmeta_list=conf.interpMeta_list;
                conf.interpMeta_list=interpMeta_list;      
                results=CompressTest<T,N>(conf,sampled_blocks,QoZ::ALGO_INTERP,QoZ::TUNING_TARGET_CR,false);
                double best_interp_cr_2=sizeof(T)*8.0/results.first;     
                conf.interpMeta_list=tempmeta_list;

                //std::cout<<best_interp_cr_1<<" "<<best_interp_cr_2<<std::endl;
                if(best_interp_cr_2>best_interp_cr_1*1.05){
                    conf.frozen_dim=frozen_dim;
                    bestInterpMeta_list=interpMeta_list;
                    std::cout<<"Dim "<<frozen_dim<<" frozen"<<std::endl;
                }
            

                if(conf.pdTuningRealComp and conf.autoTuningRate>0 and conf.autoTuningRate==conf.predictorTuningRate){
                        //recover sample if real compression used                  
                    sampleBlocks<T,N>(data,global_dims,sampleBlockSize,sampled_blocks,conf.predictorTuningRate,conf.profiling,starts,conf.var_first);
                }



            }
            
            



            
            conf.interpMeta_list=bestInterpMeta_list;
            if(conf.autoTuningRate==0){ //Qustionable part.  //when adaptivemdtride>0 there's a duplication of work. To fix.
               // std::vector<QoZ::Interp_Meta> tempmeta_list=conf.interpMeta_list;
                      
                std::pair<double,double> results=CompressTest<T,N>(conf,sampled_blocks,QoZ::ALGO_INTERP,QoZ::TUNING_TARGET_CR,false);
                double cur_best_interp_cr=sizeof(T)*8.0/results.first;     
                /*
                if(wave_idx>1){
                    cur_best_interp_cr*=double(per_block_ele_num)/conf.num;//maybe incorrect.deprecated.
                }
                */
                //if(cur_best_interp_cr>best_interp_cr){
                    best_interp_cr=cur_best_interp_cr;
                    /*
                    conf.interpAlgo_list=interpAlgo_list;
                    conf.interpDirection_list=interpDirection_list;
                    conf.cubicSplineType_list=cubicSplineType_list;
                    */
                
                  

                //}
                //else{
               //     conf.interpMeta_list=tempmeta_list;
               // }
                    //if (anchor_rate>0)
                    //  best_interp_cr=1/((1-anchor_rate)/best_interp_cr+anchor_rate);   
            }
           
        }

        else{
            /*
            uint8_t bestInterpAlgo = QoZ::INTERP_ALGO_CUBIC;
            uint8_t bestDirection = 0;
            uint8_t bestCubicSplineType =0;
            */
            //frozendim and dynamic dim not added.
            QoZ::Interp_Meta best_meta,cur_meta;
                //conf.cmprAlgo == QoZ::ALGO_INTERP;
            double cur_best_interp_cr=0.0;
            for (auto &interp_op: interpAlgo_Candidates) {
                cur_meta.interpAlgo=interp_op;
                for (auto &interp_pd: interpParadigm_Candidates) {
                    cur_meta.interpParadigm=interp_pd;
                    for (auto &interp_direction: interpDirection_Candidates) {
                        if (interp_pd==1 or  (interp_pd==2 and N<=2) and interp_direction!=0)
                            continue;
                        cur_meta.interpDirection=interp_direction;
                        for(auto &cubic_spline_type:cubicSplineType_Candidates){
                            if (interp_op!=QoZ::INTERP_ALGO_CUBIC and cubic_spline_type!=0)
                                break;
                            cur_meta.cubicSplineType=cubic_spline_type;
                            for(auto adj_interp:adjInterp_Candidates){
                                if (interp_op!=QoZ::INTERP_ALGO_CUBIC and adj_interp!=0)
                                    break;
                                cur_meta.adjInterp=adj_interp;       
                                conf.interpMeta=cur_meta;
                                double cur_ratio=0;
                                std::pair<double,double> results=CompressTest<T,N>(conf, sampled_blocks,QoZ::ALGO_INTERP,QoZ::TUNING_TARGET_CR,false);
                                cur_ratio=sizeof(T)*8.0/results.first;
                                
                                if (cur_ratio>cur_best_interp_cr){
                                    cur_best_interp_cr=cur_ratio;
                                    /*
                                    bestInterpAlgo=interp_op;
                                    bestDirection=interp_direction;
                                    bestCubicSplineType=cubic_spline_type;
                                    */
                                    best_meta=cur_meta;

                                }
                            }
                        }
                    }
                }
            }
            //delete sz;
            /*
            bestInterpAlgos[wave_idx]=bestInterpAlgo;
            bestInterpDirections[wave_idx]=bestDirection;
            bestCubicSplineTypes[wave_idx]=bestCubicSplineType;
            */
            bestInterpMeta=best_meta;
            conf.interpMeta=best_meta;
            if(conf.autoTuningRate==0){
                if(cur_best_interp_cr>best_interp_cr){
                    /*
                    conf.interpAlgo=bestInterpAlgo;
                    conf.interpDirection=bestDirection;
                    conf.cubicSplineType=bestCubicSplineType;
                    */
                    best_interp_cr=cur_best_interp_cr;
                    conf.interpMeta=best_meta;
                
                }
            }
        }
        conf.absErrorBound=ori_eb;


        

        if(conf.verbose)           
            printf("Predictor tuning finished.\n");           
        conf.alpha=o_alpha;
        conf.beta=o_beta;
        conf.dims=global_dims;
        conf.num=global_num;
        //conf.sperr=ori_sperr;
        useInterp= (best_interp_cr>=best_lorenzo_ratio) or best_lorenzo_ratio>=80 or best_interp_cr>=80;//orig 0.95*lorenzo_ratio
        if(conf.verbose){
            if (conf.levelwisePredictionSelection<=0){
                std::cout << "interp best interpAlgo = " << (bestInterpMeta.interpAlgo == 0 ? "LINEAR" : (bestInterpMeta.interpAlgo == 1?"CUBIC":"QUAD")) << std::endl;
                std::cout << "interp best interpParadigm = " << (bestInterpMeta.interpParadigm == 0 ? "1D" : (bestInterpMeta.interpParadigm == 1 ? "MD" : "HD") ) << std::endl;
                if(bestInterpMeta.interpParadigm!=1)
                    std::cout << "interp best direction = " << (unsigned) bestInterpMeta.interpDirection << std::endl;
                if(bestInterpMeta.interpAlgo!=0){
                    std::cout << "interp best cubic spline = " << (unsigned) bestInterpMeta.cubicSplineType << std::endl;
                    std::cout << "interp best adj = " << (unsigned) bestInterpMeta.adjInterp << std::endl;

                }
            }
            else{
                for(int level=conf.levelwisePredictionSelection;level>0;level--){
                    std::cout << "Level: " << (unsigned) level<<std::endl;
                    std::cout << "\tinterp best interpAlgo = " << (bestInterpMeta_list[level-1].interpAlgo == 0 ? "LINEAR" : (bestInterpMeta_list[level-1].interpAlgo == 1 ? "CUBIC" : "QUAD")) << std::endl;
                    std::cout << "\tinterp best interpParadigm = " << (bestInterpMeta_list[level-1].interpParadigm == 0 ? "1D" : (bestInterpMeta_list[level-1].interpParadigm == 1 ? "MD" : "HD") ) << std::endl;
                    if(bestInterpMeta_list[level-1].interpParadigm!=1)
                        std::cout << "\tinterp best direction = " << (unsigned) bestInterpMeta_list[level-1].interpDirection << std::endl;
                    if(bestInterpMeta_list[level-1].interpAlgo!=0){
                        std::cout << "\tinterp best cubic spline = " << (unsigned) bestInterpMeta_list[level-1].cubicSplineType << std::endl;
                        std::cout << "\tinterp best adj = " << (unsigned) bestInterpMeta_list[level-1].adjInterp << std::endl;

                    }
                }
            }
            if(conf.autoTuningRate==0){
                std::cout << "interp best cr = " << best_interp_cr << std::endl;
                printf("choose %s\n", useInterp ? "interp" : "Lorenzo");
            }
        }

    }
    
    else{// if (!conf.blockwiseTuning){ //recently modified. not sure.
        QoZ::Timer timer(true);
        //size_t sampling_num, sampling_block;
        //std::vector<size_t> sample_dims(N);         
        sampling_data = QoZ::sampling<T, N>(data, conf.dims, sampling_num, sample_dims, sampling_block);
        if (sampling_num == conf.num) {
            conf.cmprAlgo = QoZ::ALGO_INTERP;
       
        }
        
        
        //lorenzo_config.quantbinCnt = 65536 * 2;
        //QoZ::writeTextFile<T>("sampled_data.dat", sampling_data.data(), lorenzo_config.num);
        else{
            QoZ::Config lorenzo_config = conf;
            lorenzo_config.cmprAlgo = QoZ::ALGO_LORENZO_REG;
            lorenzo_config.setDims(sample_dims.begin(), sample_dims.end());
            lorenzo_config.lorenzo = true;
            lorenzo_config.lorenzo2 = true;
            lorenzo_config.regression = false;
            lorenzo_config.regression2 = false;
            lorenzo_config.openmp = false;
            lorenzo_config.blockSize = 5;//why?
            size_t sampleOutSize;
            std::vector<T> cur_sampling_data=sampling_data;
            auto cmprData = SZ_compress_LorenzoReg<T, N>(lorenzo_config, cur_sampling_data.data(), sampleOutSize);
                
            delete[]cmprData;
            double ratio = sampling_num * 1.0 * sizeof(T) / sampleOutSize;
            if(conf.verbose)
                printf("Lorenzo ratio = %.4f\n", ratio);

            best_lorenzo_ratio = ratio;
            double best_interp_ratio = 0;


            for (auto &interp_op: {QoZ::INTERP_ALGO_LINEAR, QoZ::INTERP_ALGO_CUBIC}) {
                //cur_sampling_data=sampling_data;
                ratio = do_not_use_this_interp_compress_block_test<T, N>(sampling_data.data(), sample_dims, sampling_num, conf.absErrorBound,
                                                                             interp_op, conf.interpMeta.interpDirection, sampling_block);
                if (ratio > best_interp_ratio) {
                    best_interp_ratio = ratio;
                    conf.interpMeta.interpAlgo = interp_op;
                }
            }
            if(conf.verbose)
                std::cout << "interp best interpAlgo = " << (conf.interpMeta.interpAlgo == 0 ? "LINEAR" : "CUBIC") << std::endl;
                
            int direction_op = QoZ::factorial(N) - 1;
            //cur_sampling_data=sampling_data;
            ratio = do_not_use_this_interp_compress_block_test<T, N>(sampling_data.data(), sample_dims, sampling_num, conf.absErrorBound,
                                                                         conf.interpMeta.interpAlgo, direction_op, sampling_block);
            if (ratio > best_interp_ratio * 1.02) {
                best_interp_ratio = ratio;
                conf.interpMeta.interpDirection = direction_op;
            }
            useInterp=!(best_lorenzo_ratio > best_interp_ratio && best_lorenzo_ratio < 80 && best_interp_ratio < 80);
            if(conf.verbose){
                std::cout << "interp best direction = " << (unsigned) conf.interpMeta.interpDirection << std::endl;
                
                printf("Interp ratio = %.4f\n", best_interp_ratio);
                    
                printf("choose %s\n", useInterp ? "interp" : "Lorenzo");
            }
            if (useInterp){
                conf.cmprAlgo=QoZ::ALGO_INTERP;
            }
            else{
                conf.cmprAlgo=QoZ::ALGO_LORENZO_REG;
            }
        }
        if(conf.verbose)
            timer.stop("sz3 tuning");
    }
    /*
    if(conf.verbose){
        timer.stop("PredTuning");
        timer.start();
    }
    */
    if (useInterp and conf.autoTuningRate>0){
            
        if(conf.verbose)
            std::cout<<"B-M tuning started."<<std::endl;
       
        if (conf.autoTuningRate!=conf.predictorTuningRate){//} and (conf.predictorTuningRate!=0 or conf.autoTuningRate!=conf.waveletTuningRate)){
              
            sampleBlocks<T,N>(data,conf.dims,sampleBlockSize,sampled_blocks,conf.autoTuningRate,conf.profiling,starts,conf.var_first);
        }

        
        double bestalpha=1;
        double bestbeta=1;
        
        double bestb=9999;

        double bestm=0;
        size_t num_sampled_blocks=sampled_blocks.size();
        size_t per_block_ele_num=pow(sampleBlockSize+1,N);
        size_t ele_num=num_sampled_blocks*per_block_ele_num;
        //vector<double> orig_sums(num_sampled_blocks,0);
        //vector<double> orig_square_sums(num_sampled_blocks,0);
        std::vector<double> orig_means;//(num_sampled_blocks,0);
        std::vector<double> orig_sigma2s;//(num_sampled_blocks,0);
        std::vector<double> orig_ranges;//(num_sampled_blocks,0);
        conf.dims=std::vector<size_t>(N,sampleBlockSize+1);
        conf.num=per_block_ele_num;
        //size_t ssim_size=0;
        //size_t ssim_block_num=0;
        if(conf.tuningTarget==QoZ::TUNING_TARGET_SSIM){
            size_t ssim_size=conf.SSIMBlockSize;
            for (size_t k =0;k<num_sampled_blocks;k++){
                    //cur_block=sampled_blocks[k];
                    //std::cout<<cur_block.size()<<std::endl;
                double orig_mean=0,orig_sigma2=0,orig_range=0;       
                if(N==2){
                    for (size_t i=0;i+ssim_size<sampleBlockSize+1;i+=ssim_size){
                        for (size_t j=0;j+ssim_size<sampleBlockSize+1;j+=ssim_size){
                            std::vector<size_t> starts{i,j};
                            QoZ::blockwise_profiling<T>(sampled_blocks[k].data(),conf.dims,starts,ssim_size,orig_mean,orig_sigma2,orig_range);
                            orig_means.push_back(orig_mean);
                            orig_sigma2s.push_back(orig_sigma2);
                            orig_ranges.push_back(orig_range);


                        }
                    }
                }
                else if(N==3){
                    for (size_t i=0;i+ssim_size<sampleBlockSize+1;i+=ssim_size){
                        for (size_t j=0;j+ssim_size<sampleBlockSize+1;j+=ssim_size){
                            for (size_t kk=0;kk+ssim_size<sampleBlockSize+1;kk+=ssim_size){
                                std::vector<size_t> starts{i,j,kk};
                                QoZ::blockwise_profiling<T>(sampled_blocks[k].data(),conf.dims,starts,ssim_size,orig_mean,orig_sigma2,orig_range);
                                orig_means.push_back(orig_mean);
                                orig_sigma2s.push_back(orig_sigma2);
                                orig_ranges.push_back(orig_range);
                            }
                        }
                    }
                }                      
            }
           //ssim_block_num=orig_means.size();
        }
        std::vector<T> flattened_sampled_data;
           
        if (conf.tuningTarget==QoZ::TUNING_TARGET_AC){
            for(int i=0;i<num_sampled_blocks;i++)
                flattened_sampled_data.insert(flattened_sampled_data.end(),sampled_blocks[i].begin(),sampled_blocks[i].end());

        }
        double oriabseb=conf.absErrorBound;
        
        /*if(conf.verbose){
            timer.stop("B-M prep");
            timer.start();
        }*/
        
        
       
        //std::vector<double> flattened_cur_blocks;

        
        conf.dims=std::vector<size_t>(N,sampleBlockSize+1);
        conf.num=per_block_ele_num;
        /*
        //if(!conf.blockwiseTuning){
            if(conf.levelwisePredictionSelection>0){
                
                conf.interpMeta_list=bestInterpMeta_list;
            }
            else{
                

                conf.interpMeta=bestInterpMeta;
            }
        //}
        */
        std::vector<double>alpha_list;
        init_alphalist<T,N>(alpha_list,rel_bound,conf);
        size_t alpha_nums=alpha_list.size();
        std::vector<double>beta_list;
        init_betalist<T,N>(beta_list,rel_bound,conf);
        size_t beta_nums=beta_list.size();  
        
        for (size_t i=0;i<alpha_nums;i++){
            for (size_t j=0;j<beta_nums;j++){
                conf.absErrorBound=oriabseb;
                double alpha=alpha_list[i];
                double beta=beta_list[j];
                if (( (alpha>=1 and alpha>beta) or (alpha<0 and beta!=-1) ) )
                    continue;
                conf.alpha=alpha;
                conf.beta=beta; 
                
                //printf("%d %.2f %.2f %.2f\n",wave_idx,gamma,alpha,beta);                  
                std::pair<double,double> results=CompressTest<T,N>(conf, sampled_blocks,QoZ::ALGO_INTERP,(QoZ::TUNING_TARGET)conf.tuningTarget,false,profiling_coeff,orig_means,
                                                                    orig_sigma2s,orig_ranges,flattened_sampled_data);
                double bitrate=results.first;
                double metric=results.second;
                //printf("%d %.2f %.2f %.2f %.4f %.2f\n",wave_idx,gamma,alpha,beta,bitrate,metric);
                if ( (conf.tuningTarget!=QoZ::TUNING_TARGET_CR and metric>=bestm and bitrate<=bestb) or (conf.tuningTarget==QoZ::TUNING_TARGET_CR and bitrate<=bestb ) ){
                    bestalpha=alpha;
                    bestbeta=beta;
                    bestb=bitrate;
                    bestm=metric;
                    useInterp=true;
                    //printf("Best: %.2f %.2f %.2f %.4f %.2f\n",bestgamma,bestalpha,bestbeta,bestb,bestm);
                }
                else if ( (conf.tuningTarget!=QoZ::TUNING_TARGET_CR and metric<=bestm and bitrate>=bestb) or (conf.tuningTarget==QoZ::TUNING_TARGET_CR and bitrate>bestb) ){
                    if ( ((alpha>=1 and pow(alpha,max_interp_level-1)<=beta) or (alpha<1 and alpha*(max_interp_level-1)<=beta)))
                        break;

                    continue;
                }
                else{
                    double eb_fixrate;
                    /*
                    if (metric>bestm)
                        eb_fixrate=rel_bound>1e-4?1.2:1.1;
                    else
                        eb_fixrate=rel_bound>1e-4?0.8:0.9;
                        */
                    eb_fixrate=bitrate/bestb;
                    double orieb=conf.absErrorBound;
                    conf.absErrorBound*=eb_fixrate;
                        
                    std::pair<double,double> results=CompressTest<T,N>(conf, sampled_blocks,QoZ::ALGO_INTERP,(QoZ::TUNING_TARGET)conf.tuningTarget,false,profiling_coeff,orig_means,
                                                                        orig_sigma2s,orig_ranges,flattened_sampled_data);
                    conf.absErrorBound=orieb;

                    double bitrate_r=results.first;
                    double metric_r=results.second;
                    double a=(metric-metric_r)/(bitrate-bitrate_r);
                    double b=metric-a*bitrate;
                    double reg=a*bestb+b;
                        //printf("%.2f %.2f %.2f %.4f %.2f\n",gamma,alpha,beta,bitrate_r,metric_r);
                        //printf("%.2f %.2f %.2f %.4f %.2f\n",gamma,alpha,beta,bestb,reg);      
                        //conf.absErrorBound=orig_eb;
                    if (reg>bestm){
                        bestalpha=alpha;
                        bestbeta=beta;
                        bestb=bitrate;
                        bestm=metric;
                        useInterp=true;
                        //printf("Best: %.2f %.2f %.2f %.4f %.2f\n",bestgamma,bestalpha,bestbeta,bestb,bestm);
                    }
                }
                if ( ( (alpha>=1 and pow(alpha,max_interp_level-1)<=beta) or (alpha<1 and alpha*(max_interp_level-1)<=beta)) )
                    break;

            }
        }

        /*

        if(conf.fineGrainTuning ){//currently skip SPERR fgtuning
            double last_best_alpha=bestalpha,last_best_beta=bestbeta;
            //The following list building to refine.
            if(last_best_alpha==1)
                alpha_list={1.125};
            else
                alpha_list={last_best_alpha-0.125,last_best_alpha,last_best_alpha+0.125};
            if(last_best_beta==1.5)
                beta_list={1.25,1.5,1.75};
            else if (last_best_beta==2.0)
                beta_list={1.75,2.0,2.5};
            else
                beta_list={last_best_beta-0.5,last_best_beta,last_best_beta+0.5};

            
            for (auto alpha:alpha_list){
                for (auto beta:beta_list){
                    conf.absErrorBound=oriabseb;
                    
                    if (( (alpha>=1 and alpha>beta) or (alpha<0 and beta!=-1) ) )
                        continue;
                    if( alpha==last_best_alpha and beta==last_best_beta){
                        if ( ((alpha>=1 and pow(alpha,max_interp_level-1)<=beta) or (alpha<1 and alpha*(max_interp_level-1)<=beta)))
                            break;
                        continue;

                    }
                    conf.alpha=alpha;
                    conf.beta=beta; 
                    
                    //printf("%d %.2f %.2f %.2f\n",wave_idx,gamma,alpha,beta);                  
                    std::pair<double,double> results=CompressTest<T,N>(conf, sampled_blocks,QoZ::ALGO_INTERP,(QoZ::TUNING_TARGET)conf.tuningTarget,false,profiling_coeff,orig_means,
                                                                        orig_sigma2s,orig_ranges,flattened_sampled_data);
                    double bitrate=results.first;
                    double metric=results.second;
                    //printf("%d %.2f %.2f %.2f %.4f %.2f\n",wave_idx,gamma,alpha,beta,bitrate,metric);
                    if ( (conf.tuningTarget!=QoZ::TUNING_TARGET_CR and metric>=bestm and bitrate<=bestb) or (conf.tuningTarget==QoZ::TUNING_TARGET_CR and bitrate<=bestb ) ){
                        bestalpha=alpha;
                        bestbeta=beta;
                       
                        bestb=bitrate;
                        bestm=metric;
                      
                        useInterp=true;
                        //printf("Best: %.2f %.2f %.2f %.4f %.2f\n",bestgamma,bestalpha,bestbeta,bestb,bestm);
                    }
                    else if ( (conf.tuningTarget!=QoZ::TUNING_TARGET_CR and metric<=bestm and bitrate>=bestb) or (conf.tuningTarget==QoZ::TUNING_TARGET_CR and bitrate>bestb) ){
                        if ( ((alpha>=1 and pow(alpha,max_interp_level-1)<=beta) or (alpha<1 and alpha*(max_interp_level-1)<=beta)) )
                            break;

                        continue;
                    }
                    else{
                        double eb_fixrate;
                        eb_fixrate=bitrate/bestb;
                        double orieb=conf.absErrorBound;
                        conf.absErrorBound*=eb_fixrate;
                            
                        std::pair<double,double> results=CompressTest<T,N>(conf, sampled_blocks,QoZ::ALGO_INTERP,(QoZ::TUNING_TARGET)conf.tuningTarget,false,profiling_coeff,orig_means,
                                                                            orig_sigma2s,orig_ranges,flattened_sampled_data);
                        conf.absErrorBound=orieb;

                        double bitrate_r=results.first;
                        double metric_r=results.second;
                        double a=(metric-metric_r)/(bitrate-bitrate_r);
                        double b=metric-a*bitrate;
                        double reg=a*bestb+b;
                            //printf("%.2f %.2f %.2f %.4f %.2f\n",gamma,alpha,beta,bitrate_r,metric_r);
                            //printf("%.2f %.2f %.2f %.4f %.2f\n",gamma,alpha,beta,bestb,reg);      
                            //conf.absErrorBound=orig_eb;
                        if (reg>bestm){
                            bestalpha=alpha;
                            bestbeta=beta;
                                   
                            bestb=bitrate;
                            bestm=metric;
                    
                            useInterp=true;
                            //printf("Best: %.2f %.2f %.2f %.4f %.2f\n",bestgamma,bestalpha,bestbeta,bestb,bestm);
                        }
                    }
                    if ( ( (alpha>=1 and pow(alpha,max_interp_level-1)<=beta) or (alpha<1 and alpha*(max_interp_level-1)<=beta)) )
                        break;

                }
            }

        
           // delete sz;
        }
        */
        //add lorenzo
        conf.absErrorBound=oriabseb;
        if(conf.testLorenzo){    


            std::pair<double,double> results=CompressTest<T,N>(conf, sampled_blocks,QoZ::ALGO_LORENZO_REG,(QoZ::TUNING_TARGET)conf.tuningTarget,false,profiling_coeff,orig_means,
                    orig_sigma2s,orig_ranges,flattened_sampled_data);

            double bitrate=results.first;
            double metric=results.second;

            
            //printf("Lorenzo: %.4f %.2f\n",bitrate,metric);     
            if ( (conf.tuningTarget!=QoZ::TUNING_TARGET_CR and metric>=bestm and bitrate<=bestb) or (conf.tuningTarget==QoZ::TUNING_TARGET_CR and bitrate<=bestb ) ){
                
                bestb=bitrate;
                bestm=metric;
                bestalpha=-1;
                bestbeta=-1;

                useInterp=false;
                //printf("Best: %.4f %.2f\n",bestb,bestm);
                   
            }
            else if ( (conf.tuningTarget!=QoZ::TUNING_TARGET_CR and metric<=bestm and bitrate>=bestb) or (conf.tuningTarget==QoZ::TUNING_TARGET_CR and bitrate>bestb) ){
                useInterp=true;
            }
            else{
                double eb_fixrate;
                /*
                if (metric>bestm)
                    eb_fixrate=rel_bound>1e-4?1.2:1.1;
                else
                    eb_fixrate=rel_bound>1e-4?0.8:0.9;
                    */
                eb_fixrate=bitrate/bestb;
                double orieb=conf.absErrorBound;
                conf.absErrorBound*=eb_fixrate;                        
                std::pair<double,double> results=CompressTest<T,N>(conf, sampled_blocks,QoZ::ALGO_LORENZO_REG,(QoZ::TUNING_TARGET)conf.tuningTarget,false,profiling_coeff,orig_means,
                                                                    orig_sigma2s,orig_ranges,flattened_sampled_data);
                conf.absErrorBound=orieb;
                double bitrate_r=results.first;
                double metric_r=results.second;
                double a=(metric-metric_r)/(bitrate-bitrate_r);
                double b=metric-a*bitrate;
                double reg=a*bestb+b;
                       // printf("%.4f %.2f\n",bitrate_r,metric_r);
                      // printf("%.4f %.2f\n",bestb,reg);
                        //conf.absErrorBound=orig_eb;
                if (a>0 and reg>bestm){
                           // bestalpha=alpha;
                            //bestbeta=beta; 
                    bestb=bitrate;
                    bestm=metric;
                    bestalpha=-1;
                    bestbeta=-1;
                    useInterp=false;

                            //printf("Best: %.4f %.2f\n",bestb,bestm);
                }
            }          
        }
        conf.absErrorBound=oriabseb;
        /*
        if(conf.verbose){
            timer.stop("B-M step");
            timer.start();
        }
        */
        
        if(conf.tuningTarget==QoZ::TUNING_TARGET_AC){
            bestm=1-bestm;
        }
        std::string metric_name="Quality";
        if (conf.tuningTarget==QoZ::TUNING_TARGET_RD ){
            metric_name="PSNR";
        }
        else if (conf.tuningTarget==QoZ::TUNING_TARGET_SSIM ){
            metric_name="SSIM";
        }
        else if (conf.tuningTarget==QoZ::TUNING_TARGET_AC ){
            metric_name="AutoCorrelation";
        }
        if(conf.verbose){
            printf("Autotuning finished.\n");
            if (useInterp)
                printf("Interp selected. Selected alpha: %f. Selected beta: %f. Best bitrate: %f. Best %s: %f.\n",bestalpha,bestbeta,bestb, const_cast<char*>(metric_name.c_str()),bestm);
            else
                printf("Lorenzo selected. Best bitrate: %f. Best %s: %f.\n",bestb, const_cast<char*>(metric_name.c_str()),bestm);

        }
        conf.alpha=bestalpha;
        conf.beta=bestbeta;
        conf.dims=global_dims;
        conf.num=global_num;  
        
        //if(use_sperr<T,N>(conf))
        //   conf.sperr_eb_coeff=bestalpha;
        /*
        if(useInterp){ 

            if(conf.levelwisePredictionSelection>0){
                
                conf.interpMeta_list=interpMeta_lists[bestWave];
            }
            else{
              
                conf.interpMeta=bestInterpMetas[bestWave];
            }
        }*/
    }
    else if(useInterp and conf.QoZ){
        std::pair<double,double> ab=setABwithRelBound(rel_bound,2);
        conf.alpha=ab.first;
        conf.beta=ab.second;


    }
    conf.blockwiseTuning=blockwiseTuning;
    if (useInterp){
        conf.cmprAlgo=QoZ::ALGO_INTERP;
    }
    else{
         conf.cmprAlgo=QoZ::ALGO_LORENZO_REG;
    } 
        
    for(int i=0;i<sampled_blocks.size();i++){
        std::vector< T >().swap(sampled_blocks[i]);              
    }
    std::vector< std::vector<T> >().swap(sampled_blocks);     
    return best_lorenzo_ratio;    
}



template<class T, QoZ::uint N>
char *SZ_compress_Interp_lorenzo(QoZ::Config &conf, T *data, size_t &outSize) {
    assert(conf.cmprAlgo == QoZ::ALGO_INTERP_LORENZO);
    QoZ::calAbsErrorBound(conf, data);

    if(N!=2&&N!=3){
        conf.autoTuningRate=0;
        conf.predictorTuningRate=0;
        conf.maxStep=0;
        conf.levelwisePredictionSelection=0;
        conf.multiDimInterp=0;
        conf.QoZ=0;
        

    }//merge this part to other branches
   
   
    
  //  T *origdata,*coeffData;
    if (conf.rng<0)
        conf.rng=QoZ::data_range<T>(data,conf.num);

    
    if (conf.relErrorBound<=0)
        conf.relErrorBound=conf.absErrorBound/conf.rng;
   // T* coeffs;
  
    //conf.sperr=0;
    


    if(conf.verbose)
        std::cout << "====================================== BEGIN TUNING ================================" << std::endl;
    QoZ::Timer timer(true);
    
     
    double best_lorenzo_ratio=Tuning<T,N>(conf,data);
    
   // char * compress_output;


    
  
//    printf("%lu %lu %lu %lu %lu\n", sampling_data.size(), sampling_num, sample_dims[0], sample_dims[1], sample_dims[2]);
   // bool useInterp = !(best_lorenzo_ratio > best_interp_ratio && best_lorenzo_ratio < 80 && best_interp_ratio < 80);
    
//    printf("\nLorenzo compression ratio = %.2f\n", best_lorenzo_ratio);
//    printf("Interp compression ratio = %.2f\n", best_interp_ratio);  

    if (conf.cmprAlgo == QoZ::ALGO_INTERP) {
    
        std::vector<int>().swap(conf.quant_bins);
        double tuning_time = timer.stop();
        if(conf.verbose){
            std::cout << "Tuning time = " << tuning_time << "s" << std::endl;
            std::cout << "====================================== END TUNING ======================================" << std::endl;
        }

        //if (conf.predictorTuningRate<1){      
            
        return SZ_compress_Interp<T, N>(conf, data, outSize);        
            
       // }
        /*
        else {
            std::vector<int> op_candidates={QoZ::INTERP_ALGO_LINEAR,QoZ::INTERP_ALGO_CUBIC};
            std::vector<int> dir_candidates={0,QoZ::factorial(N)-1};
            if(conf.multiDimInterp){
                dir_candidates.push_back(QoZ::factorial(N));
            }
            if(conf.wavelet >1){
                compress_output = SZ_compress_AutoSelectiveInterp<T,N>(conf,coeffData,outSize,op_candidates,dir_candidates,0);
            }
            else
                compress_output = SZ_compress_AutoSelectiveInterp<T,N>(conf,data,outSize,op_candidates,dir_candidates,0);
        }
        */
    } 

    else {
        QoZ::Config lorenzo_config = conf;
        size_t sampling_num, sampling_block;        
        std::vector<size_t> sample_dims(N);
        std::vector<T> sampling_data;

        size_t sampleOutSize;
        double ratio;  
            //size_t sampling_num, sampling_block;
            
        sampling_data = QoZ::sampling<T, N>(data, conf.dims, sampling_num, sample_dims, sampling_block);    
        lorenzo_config.cmprAlgo = QoZ::ALGO_LORENZO_REG;
        
        lorenzo_config.lorenzo = true;
        lorenzo_config.lorenzo2 = true;
        lorenzo_config.regression = false;
        lorenzo_config.regression2 = false;
        lorenzo_config.openmp = false;
        lorenzo_config.blockSize = 5;//why?
        if (sampling_num != conf.num) {
            lorenzo_config.setDims(sample_dims.begin(), sample_dims.end());
       
        //lorenzo_config.quantbinCnt = 65536 * 2;
                    
            if(conf.autoTuningRate>0 or conf.predictorTuningRate>0){
                auto cmprData = SZ_compress_LorenzoReg<T, N>(lorenzo_config, sampling_data.data(), sampleOutSize);
                delete[]cmprData;
                ratio = sampling_num * 1.0 * sizeof(T) / sampleOutSize;
                //printf("Lorenzo ratio = %.2f\n", ratio);
                best_lorenzo_ratio = ratio;
            }
          
            //further tune lorenzo
            if (N == 3 ) {
                lorenzo_config.quantbinCnt = QoZ::optimize_quant_invl_3d<T>(data, conf.dims[0], conf.dims[1], conf.dims[2], conf.absErrorBound);
                lorenzo_config.pred_dim = 2;
                auto cmprData = SZ_compress_LorenzoReg<T, N>(lorenzo_config, sampling_data.data(), sampleOutSize);
                delete[]cmprData;
                ratio = sampling_num * 1.0 * sizeof(T) / sampleOutSize;
                //printf("Lorenzo, pred_dim=2, ratio = %.4f\n", ratio);
                if (ratio > best_lorenzo_ratio * 1.02) {
                    best_lorenzo_ratio = ratio;
                } else {
                    lorenzo_config.pred_dim = 3;
                }
            }
            if (conf.relErrorBound < 1.01e-6 && best_lorenzo_ratio > 5 && lorenzo_config.quantbinCnt != 16384) {
                auto quant_num = lorenzo_config.quantbinCnt;
                lorenzo_config.quantbinCnt = 16384;
                auto cmprData = SZ_compress_LorenzoReg<T, N>(lorenzo_config, sampling_data.data(), sampleOutSize);
                delete[]cmprData;
                ratio = sampling_num * 1.0 * sizeof(T) / sampleOutSize;
    //            printf("Lorenzo, quant_bin=8192, ratio = %.2f\n", ratio);
                if (ratio > best_lorenzo_ratio * 1.02) {
                    best_lorenzo_ratio = ratio;
                } else {
                    lorenzo_config.quantbinCnt = quant_num;
                }
            }
            lorenzo_config.setDims(conf.dims.begin(), conf.dims.end());
        }
     
        
        conf = lorenzo_config;
        /*
        if(conf.useCoeff){
            
            if (conf.lorenzo){
                size_t num_coeff=int(pow(2,N)-1);
                std::vector <double> A;
                std::vector<double> b;
                if(N==2)
                    QoZ::extract_lorenzoreg_2d<T,N>(data, A, b, conf.dims,1,conf.regSampleStep);
                else if (N==3)
                    QoZ::extract_lorenzoreg_3d<T,N>(data, A, b, conf.dims,1,conf.regSampleStep);
                //std::cout<<"step1"<<std::endl;
                size_t num_points=b.size();
                double * coeff_array=QoZ::Regression(A.data(),num_points,num_coeff,b.data());
                //std::cout<<"step2"<<std::endl;
                conf.lorenzo1_coeffs=std::vector<double>(coeff_array,coeff_array+num_coeff);
                delete [] coeff_array;

            }
            if (conf.lorenzo2){
                size_t num_coeff=int(pow(3,N)-1);
                std::vector <double> A;
                std::vector<double> b;
                if(N==2)
                    QoZ::extract_lorenzoreg_2d<T,N>(data, A, b, conf.dims,2,conf.regSampleStep);
                else if (N==3)
                    QoZ::extract_lorenzoreg_3d<T,N>(data, A, b, conf.dims,2,conf.regSampleStep);
                //std::cout<<"step3"<<std::endl;
                size_t num_points=b.size();
                double * coeff_array=QoZ::Regression(A.data(),num_points,num_coeff,b.data());
                //std::cout<<"step4"<<std::endl;
                conf.lorenzo2_coeffs=std::vector<double>(coeff_array,coeff_array+num_coeff);
                delete [] coeff_array;

            }
        }
        */

        double tuning_time = timer.stop();
        if(conf.verbose){
            std::cout << "Tuning time = " << tuning_time << "s" << std::endl;
            std::cout << "====================================== END TUNING ======================================" << std::endl;
        }
        return SZ_compress_LorenzoReg<T, N>(conf, data, outSize);
    }
  
}


#endif