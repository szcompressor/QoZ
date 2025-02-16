#ifndef SZ_GENERAL_COMPRESSOR_HPP
#define SZ_GENERAL_COMPRESSOR_HPP

#include "QoZ/compressor/Compressor.hpp"
#include "QoZ/frontend/Frontend.hpp"
#include "QoZ/encoder/Encoder.hpp"
#include "QoZ/lossless/Lossless.hpp"
#include "QoZ/utils/FileUtil.hpp"
#include "QoZ/utils/Config.hpp"
#include "QoZ/utils/Timer.hpp"
#include "QoZ/def.hpp"
#include <cstring>

namespace QoZ {
    template<class T, uint N, class Frontend, class Encoder, class Lossless>
    class SZGeneralCompressor : public concepts::CompressorInterface<T> {
    public:


        SZGeneralCompressor(Frontend frontend, Encoder encoder, Lossless lossless) :
                frontend(frontend), encoder(encoder), lossless(lossless) {
            static_assert(std::is_base_of<concepts::FrontendInterface<T, N>, Frontend>::value,
                          "must implement the frontend interface");
            static_assert(std::is_base_of<concepts::EncoderInterface<int>, Encoder>::value,
                          "must implement the encoder interface");
            static_assert(std::is_base_of<concepts::LosslessInterface, Lossless>::value,
                          "must implement the lossless interface");
        }

        uchar *compress( Config &conf, T *data, size_t &compressed_size,int tuning=0) {

            Timer timer(true);

            std::vector<int> new_quant_inds = frontend.compress(data);
            quant_inds.insert(quant_inds.end(),new_quant_inds.begin(),new_quant_inds.end());
             if (tuning){
                uchar *buffer = new uchar[1];
                buffer[0]=0;
                return buffer;

            }

            encoder.preprocess_encode(quant_inds, 0);
            size_t bufferSize = 1.5 * (frontend.size_est() + encoder.size_est() + sizeof(T) * quant_inds.size());//todo: lower the 1.5.
            uchar *buffer = new uchar[bufferSize];
            uchar *buffer_pos = buffer;

            frontend.save(buffer_pos);

            timer.start();
            
            encoder.save(buffer_pos);
            encoder.encode(quant_inds, buffer_pos);
            encoder.postprocess_encode();
            assert(buffer_pos - buffer < bufferSize);

            uchar *lossless_data = lossless.compress(buffer, buffer_pos - buffer, compressed_size);
            lossless.postcompress_data(buffer);

            return lossless_data;
        }
        uchar *encoding_lossless(size_t &compressed_size,const std::vector<int> &q_inds=std::vector<int>()){

            if(q_inds.size()>0)
                quant_inds=q_inds;
          
            size_t bufferSize = 2 * quant_inds.size()*sizeof(T);//original is 3
            uchar *buffer = new uchar[bufferSize];
            uchar *buffer_pos = buffer;

            frontend.save(buffer_pos);

            encoder.preprocess_encode(quant_inds, 0);
            encoder.save(buffer_pos);
            encoder.encode(quant_inds, buffer_pos);
            encoder.postprocess_encode();
            assert(buffer_pos - buffer < bufferSize);

            uchar *lossless_data = lossless.compress(buffer, buffer_pos - buffer, compressed_size);
            lossless.postcompress_data(buffer);

            return lossless_data;

        }

        T *decompress(uchar const *cmpData, const size_t &cmpSize, size_t num) {
            T *dec_data = new T[num];
            return decompress(cmpData, cmpSize, dec_data);
        }

        T *decompress(uchar const *cmpData, const size_t &cmpSize, T *decData) {
            size_t remaining_length = cmpSize;

            Timer timer(true);
            auto compressed_data = lossless.decompress(cmpData, remaining_length);
            uchar const *compressed_data_pos = compressed_data;

            frontend.load(compressed_data_pos, remaining_length);

            encoder.load(compressed_data_pos, remaining_length);

            timer.start();
            auto quant_inds = encoder.decode(compressed_data_pos, frontend.get_num_elements());
            encoder.postprocess_decode();

            lossless.postdecompress_data(compressed_data);

            timer.start();
            frontend.decompress(quant_inds, decData);
            return decData;
        }


    private:
        Frontend frontend;
        std::vector<int> quant_inds;
        Encoder encoder;
        Lossless lossless;
    };

    template<class T, uint N, class Frontend, class Encoder, class Lossless>
    std::shared_ptr<SZGeneralCompressor<T, N, Frontend, Encoder, Lossless>>
    make_sz_general_compressor(Frontend frontend, Encoder encoder, Lossless lossless) {
        return std::make_shared<SZGeneralCompressor<T, N, Frontend, Encoder, Lossless>>(frontend, encoder, lossless);
    }


}
#endif
