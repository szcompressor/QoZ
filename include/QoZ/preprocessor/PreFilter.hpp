//
// Created by Kai Zhao on 1/29/21.
//

#ifndef QoZ_PREFILTER_HPP
#define QoZ_PREFILTER_HPP

#include "QoZ/preprocessor/PreProcessor.hpp"

namespace QoZ {
    template<class T, uint N>

    class PreFilter : public concepts::PreprocessorInterface<T, N> {

        void preprocess(T *data, std::array<size_t, N> dims, std::pair<T, T> range, T defaultValue) {
            for (T &d : data) {
                if (d > range.second || d < range.first) {
                    d = defaultValue;
                }
            }
        }
    };
}
#endif //QoZ_PRETRANSPOSE_H
