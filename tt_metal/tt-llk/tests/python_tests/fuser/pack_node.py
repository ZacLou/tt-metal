# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List, Optional

import torch
from helpers.golden_generators import apply_l1_accumulation
from helpers.tilize_untilize import tilize_block, untilize_block

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig

from helpers.llk_params import L1Accumulation, PackerReluType, PerfRunType

from .arch_common import pack_common
from .base_packer import Packer
from .block_data import (
    BlockData,
    KernelInvocation,
    NodeBlockPlan,
    wrap_generated_calls,
)
from .operand import Operand


class PackNode:
    """Wraps a packer with its output operand and pack settings.

    Analogous to FpuNode on the math side. Each PackNode represents
    one pack destination within an operation. Multiple PackNodes allow a
    single math result to be packed to different output buffers with
    independent relu or L1 accumulation configs.
    """

    def __init__(
        self,
        packer: Packer,
        output: Operand,
        pack_relu: PackerReluType = PackerReluType.NoRelu,
        relu_threshold: float = 0.0,
        pack_l1_accumulation: L1Accumulation = L1Accumulation.No,
        blocks: Optional[List[NodeBlockPlan]] = None,
    ):
        self.packer = packer
        self.output = output
        self.pack_relu = pack_relu
        self.relu_threshold = relu_threshold
        self.pack_l1_accumulation = pack_l1_accumulation
        self.blocks = blocks

    def init(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ) -> str:
        code = self.packer.init(self, operation, config, block)
        code += pack_common.relu_config(config, operation, self)
        code += pack_common.l1_accumulation_config(config, operation, self)
        return code

    def pack_loop(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ) -> str:
        if config.perf_run_type in (
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
        ):
            return ""
        code = ""
        plans = self.codegen_blocks if block.codegen else self.blocks
        for call in plans[block.block_index].calls:
            block.tile_id_block = call.dest
            block.tile_id_global = call.out
            code += self.packer.pack(self, operation, config, block)
        if block.codegen:
            code = wrap_generated_calls(code, self.packer.granularity, block)
        return code

    def uninit(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> str:
        return self.packer.uninit(self, operation, config, None)

    def golden(
        self,
        tensor: torch.Tensor,
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        return self.packer.golden(tensor, self, operation, config)

    def block_golden(
        self,
        call: KernelInvocation,
        tensor_dst: torch.Tensor,
        output: torch.Tensor,
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        tile_shape = self.output.tile_shape
        tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())
        num_faces = tile_shape.total_num_faces()
        tile = untilize_block(
            tensor_dst[call.dest].flatten(),
            config.sentinel.golden_math_format,
            tile_dims,
            tile_dimensions=tile_dims,
            num_faces=num_faces,
        )
        if self.pack_relu != PackerReluType.NoRelu:
            tile = self.packer.relu_golden(tile, config, operation, self)
        packed = tilize_block(
            tile,
            tile_dims,
            self.output.data_format,
            num_faces=num_faces,
            tile_dimensions=tile_dims,
        )[0]
        if self.pack_l1_accumulation == L1Accumulation.Yes:
            output[call.out] = apply_l1_accumulation(
                [output[call.out], packed], self.output.data_format
            )
        else:
            output[call.out] = packed
        return output

    def get_headers(self) -> List[str]:
        return self.packer.get_headers()

    def __str__(self):
        return f"PackNode({self.packer}, output={self.output})"
