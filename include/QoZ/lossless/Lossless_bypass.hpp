
#ifndef SZ_LOSSLESS_BYPASS_HPP
#define SZ_LOSSLESS_BYPASS_HPP

#include "zstd.h"
#include "QoZ/def.hpp"
#include "QoZ/utils/MemoryUtil.hpp"
#include "QoZ/utils/FileUtil.hpp"
#include "QoZ/lossless/Lossless.hpp"

namespace QoZ {
    class Lossless_bypass : public concepts::LosslessInterface {

    public:

        void postcompress_data(uchar *data) {};

        void postdecompress_data(uchar *data) {};

        uchar *compress(uchar *data, size_t dataLength, size_t &outSize) {
            outSize = dataLength;
            return data;
        }

        uchar *decompress(const uchar *data, size_t &compressedSize) {
            return (uchar *) data;
        }
    };
}
#endif //SZ_LOSSLESS_BYPASS_HPP
