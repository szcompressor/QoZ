#ifndef _SZ_LORENZO_PREDICTOR_HPP
#define _SZ_LORENZO_PREDICTOR_HPP

#include "QoZ/def.hpp"
#include "QoZ/predictor/Predictor.hpp"
#include "QoZ/utils/Iterator.hpp"
#include <cassert>
#include<cmath>

namespace QoZ {

    // N-dimension L-layer lorenzo predictor
    template<class T, uint N, uint L>
    class LorenzoPredictor : public concepts::PredictorInterface<T, N> {
    public:
        static const uint8_t predictor_id = 0b00000001;
        using Range = multi_dimensional_range<T, N>;
        using iterator = typename multi_dimensional_range<T, N>::iterator;

        LorenzoPredictor() {
            this->noise = 0;
            
        }

        LorenzoPredictor(double eb) {
            this->noise = 0;
            
            if (L == 1) {
                if (N == 1) {
                    this->noise = 0.5 * eb;
                } else if (N == 2) {
                    this->noise = 0.81 * eb;
                } else if (N == 3) {
                    this->noise = 1.22 * eb;
                } else if (N == 4) {
                    this->noise = 1.79 * eb;
                }
            } else if (L == 2) {
                if (N == 1) {
                    this->noise = 1.08 * eb;
                } else if (N == 2) {
                    this->noise = 2.76 * eb;
                } else if (N == 3) {
                    this->noise = 6.8 * eb;
                }
            }
        }

        void precompress_data(const iterator &) const {}

        void postcompress_data(const iterator &) const {}

        void predecompress_data(const iterator &) const {}

        void postdecompress_data(const iterator &) const {}

        bool precompress_block(const std::shared_ptr<Range> &) { return true; }

        void precompress_block_commit() noexcept {}

        bool predecompress_block(const std::shared_ptr<Range> &) { return true; }

        /*
         * save doesn't need to store anything except the id
         */
        // std::string save() const {
        //   return std::string(1, predictor_id);
        // }
        void save(uchar *&c) const {
            //std::cout << "save Lorenzo predictor" << std::endl;
            //uchar *buffer_pos = c;
            //c[0] = predictor_id;
            //c += sizeof(uint8_t);
            write(predictor_id, c);
            
        }

        /*
         * just verifies the ID, increments
         */
        // static LorenzoPredictor<T,N> load(const unsigned char*& c, size_t& remaining_length) {
        //   assert(remaining_length > sizeof(uint8_t));
        //   c += 1;
        //   remaining_length -= sizeof(uint8_t);
        //   return LorenzoPredictor<T,N>{};
        // }
        void load(const uchar *&c, size_t &remaining_length) {
            //std::cout << "load Lorenzo predictor" << std::endl;

            //read(predictor_id, c,remaining_length);
            c += sizeof(uint8_t);
            remaining_length-=sizeof(uint8_t);

            
            
        }

        void print() const {
            std::cout << L << "-Layer " << N << "D Lorenzo predictor, noise = " << noise << "\n";
        }

        inline T estimate_error(const iterator &iter) const noexcept {
            return fabs(*iter - predict(iter)) + this->noise;
        }

        inline T predict(const iterator &iter) const noexcept {
            return do_predict(iter);
        }

        void clear() {}

    protected:
        T noise = 0;
       

    private:
        template<uint NN = N, uint LL = L>
        inline typename std::enable_if<NN == 1 && LL == 1, T>::type do_predict(const iterator &iter) const noexcept {
            return iter.prev(1);
        }

        template<uint NN = N, uint LL = L>
        inline typename std::enable_if<NN == 2 && LL == 1, T>::type do_predict(const iterator &iter) const noexcept {
            //if(!useCoeff)
                return iter.prev(0, 1) + iter.prev(1, 0) - iter.prev(1, 1);
            /*else{
                
                T value=0;
                size_t idx=0;
                for(int i=1;i>=0;i--){
                    for(int j=1;j>=0;j--){
                        if(i==0 and j==0)
                            break;
                        value+=iter.prev(i,j)*coeffs[idx++];

                    }

                }
                return value;
                
               // return (int)coeffs[2]*iter.prev(0, 1) + (int)coeffs[1]*iter.prev(1, 0) - (int)coeffs[0]*iter.prev(1, 1);
            }*/
        }

        template<uint NN = N, uint LL = L>
        inline typename std::enable_if<NN == 3 && LL == 1, T>::type do_predict(const iterator &iter) const noexcept {
           // if (!useCoeff)
                return iter.prev(0, 0, 1) + iter.prev(0, 1, 0) + iter.prev(1, 0, 0)
                       - iter.prev(0, 1, 1) - iter.prev(1, 0, 1) - iter.prev(1, 1, 0)
                       + iter.prev(1, 1, 1);
            /*
            else{
                T value=0;
                size_t idx=0;
                for(int i=1;i>=0;i--){
                    for(int j=1;j>=0;j--){
                        for(int k=1;k>=0;k--){
                            if(i==0 and j==0 and k==0)
                                break;
                            value+=iter.prev(i,j,k)*coeffs[idx++];
                        }

                    }

                }
                return value;

            }
            */



        }

        template<uint NN = N, uint LL = L>
        inline typename std::enable_if<NN == 4, T>::type do_predict(const iterator &iter) const noexcept {
            return iter.prev(0, 0, 0, 1) + iter.prev(0, 0, 1, 0) - iter.prev(0, 0, 1, 1) + iter.prev(0, 1, 0, 0)
                   - iter.prev(0, 1, 0, 1) - iter.prev(0, 1, 1, 0) + iter.prev(0, 1, 1, 1) + iter.prev(1, 0, 0, 0)
                   - iter.prev(1, 0, 0, 1) - iter.prev(1, 0, 1, 0) + iter.prev(1, 0, 1, 1) - iter.prev(1, 1, 0, 0)
                   + iter.prev(1, 1, 0, 1) + iter.prev(1, 1, 1, 0) - iter.prev(1, 1, 1, 1);
        }

        template<uint NN = N, uint LL = L>
        inline typename std::enable_if<NN == 1 && LL == 2, T>::type do_predict(const iterator &iter) const noexcept {
            return 2 * iter.prev(1) - iter.prev(2);
        }

        template<uint NN = N, uint LL = L>
        inline typename std::enable_if<NN == 2 && LL == 2, T>::type do_predict(const iterator &iter) const noexcept {
            //if(!useCoeff)
                return 2 * iter.prev(0, 1) - iter.prev(0, 2) + 2 * iter.prev(1, 0)
                       - 4 * iter.prev(1, 1) + 2 * iter.prev(1, 2) - iter.prev(2, 0)
                       + 2 * iter.prev(2, 1) - iter.prev(2, 2);
            /*           
            else{
                
                T value=0;
                size_t idx=0;
                for(int i=2;i>=0;i--){
                    for(int j=2;j>=0;j--){
                        if(i==0 and j==0)
                            break;
                        value+=iter.prev(i,j)*coeffs[idx++];

                    }

                }
                return value;

                
                
            }
            */
        }

        template<uint NN = N, uint LL = L>
        inline typename std::enable_if<NN == 3 && LL == 2, T>::type do_predict(const iterator &iter) const noexcept {
            //if (!useCoeff)
                return 2 * iter.prev(0, 0, 1) - iter.prev(0, 0, 2) + 2 * iter.prev(0, 1, 0)
                       - 4 * iter.prev(0, 1, 1) + 2 * iter.prev(0, 1, 2) - iter.prev(0, 2, 0)
                       + 2 * iter.prev(0, 2, 1) - iter.prev(0, 2, 2) + 2 * iter.prev(1, 0, 0)
                       - 4 * iter.prev(1, 0, 1) + 2 * iter.prev(1, 0, 2) - 4 * iter.prev(1, 1, 0)
                       + 8 * iter.prev(1, 1, 1) - 4 * iter.prev(1, 1, 2) + 2 * iter.prev(1, 2, 0)
                       - 4 * iter.prev(1, 2, 1) + 2 * iter.prev(1, 2, 2) - iter.prev(2, 0, 0)
                       + 2 * iter.prev(2, 0, 1) - iter.prev(2, 0, 2) + 2 * iter.prev(2, 1, 0)
                       - 4 * iter.prev(2, 1, 1) + 2 * iter.prev(2, 1, 2) - iter.prev(2, 2, 0)
                       + 2 * iter.prev(2, 2, 1) - iter.prev(2, 2, 2);
            /*
            else{
                T value=0;
                size_t idx=0;
                for(int i=2;i>=0;i--){
                    for(int j=2;j>=0;j--){
                        for(int k=2;k>=0;k--){
                            if(i==0 and j==0 and k==0)
                                break;
                            value+=iter.prev(i,j,k)*coeffs[idx++];
                        }

                    }

                }
                return value;

            }*/
        
        }
    };
}
#endif
