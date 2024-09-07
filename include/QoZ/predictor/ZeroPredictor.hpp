#ifndef _SZ_ZERO_PREDICTOR_HPP
#define _SZ_ZERO_PREDICTOR_HPP

#include "QoZ/def.hpp"
#include "QoZ/predictor/Predictor.hpp"
#include "QoZ/utils/Iterator.hpp"
#include <cassert>
#include<cmath>
//This does no prediction, returns 0 for each prediction.

namespace QoZ {

    
    template<class T, uint N>
    class ZeroPredictor : public concepts::PredictorInterface<T, N> {
    public:
        using Range = multi_dimensional_range<T, N>;
        using iterator = typename multi_dimensional_range<T, N>::iterator;

        ZeroPredictor() {
            
        }


        void precompress_data(const iterator &) const {}

        void postcompress_data(const iterator &) const {}

        void predecompress_data(const iterator &) const {}

        void postdecompress_data(const iterator &) const {}

        bool precompress_block(const std::shared_ptr<Range> &) { return true; }

        void precompress_block_commit() noexcept {}

        bool predecompress_block(const std::shared_ptr<Range> &) { return true; }

        
        void save(uchar *&c) const {
            
        }

        
        void load(const uchar *&c, size_t &remaining_length) {
           
        }

        void print() const {
           
        }

        inline T estimate_error(const iterator &iter) const noexcept {
            return *iter;
        }

        inline T predict(const iterator &iter) const noexcept {
            return 0;
        }

        void clear() {}

    protected:
       

    private:
        

    };
}
#endif
