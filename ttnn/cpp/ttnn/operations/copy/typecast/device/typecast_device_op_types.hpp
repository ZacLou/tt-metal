// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct TypecastParams {
    const tt::tt_metal::DataType input_dtype;
    const tt::tt_metal::DataType output_dtype;
    const tt::tt_metal::MemoryConfig output_memory_config;
    const bool fp32_dest_acc_en = false;
    const bool preserve_fp32_precision = false;
    const bool bfp8_pack_precise = false;
    const std::optional<CoreRangeSet> sub_core_grids = std::nullopt;
};

struct TypecastInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

// Dataflow-buffer format for a typecast operand.
//
// INT8 tensors hold raw two's complement bytes, but the Int8 unpacker/packer decode and emit
// sign-magnitude, so configuring the buffers with tt::DataFormat::Int8 would corrupt every
// negative value. They are configured as UInt8 instead - a raw byte pass-through - and the
// typecast LLK, which is still selected from the true DataType, does the sign handling in the
// SFPU. This mirrors the int8 quantization path in binary_ng.
//
// The LLK selection (make_typecast_defines) must keep using datatype_to_dataformat_converter.
inline tt::DataFormat typecast_buffer_data_format(tt::tt_metal::DataType dtype) {
    return dtype == tt::tt_metal::DataType::INT8 ? tt::DataFormat::UInt8
                                                 : tt::tt_metal::datatype_to_dataformat_converter(dtype);
}

}  // namespace ttnn::prim
