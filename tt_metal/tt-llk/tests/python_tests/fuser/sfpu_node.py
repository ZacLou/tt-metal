# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig

from helpers.golden_generators import (
    BinarySFPUGolden,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.tilize_untilize import tilize_block, untilize_block

from .base_sfpu import Sfpu
from .block_data import BlockData, KernelInvocation, NodeBlockPlan


class SfpuNode:
    def __init__(self, sfpu: Sfpu, blocks: Optional[List[NodeBlockPlan]] = None):
        self.sfpu = sfpu
        self.blocks = blocks

    def sfpu_init(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if config.skip_math_init:
            return ""
        return self.sfpu.init(operation, config, self, block)

    def sfpu_run(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if config.skip_math_init:
            return ""
        code = ""
        plans = self.codegen_blocks if block.codegen else self.blocks
        for call in plans[block.block_index].calls:
            block.tile_id_src_a = call.src0
            block.tile_id_src_b = call.src1
            block.tile_id_block = call.dest
            code += self.sfpu.calculate(operation, config, self, block)
        return code

    def sfpu_uninit(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if config.skip_math_init:
            return ""
        return self.sfpu.uninit(operation, config, self, block)

    def golden(
        self,
        input_tensor_a,
        input_tensor_b,
        tensor_a,
        tensor_b,
        tensor_dst,
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tile_dims = (
            operation.tile_shape.total_row_dim(),
            operation.tile_shape.total_col_dim(),
        )
        num_faces = operation.tile_shape.total_num_faces()
        tilized_dst = tilize_block(
            tensor_dst,
            operation.max_output_dimensions,
            config.sentinel.golden_math_format,
            num_faces=num_faces,
            tile_dimensions=tile_dims,
        )

        tile_count_x = (
            operation.max_output_dimensions[1] // operation.tile_shape.total_col_dim()
        )
        tile_count_y = (
            operation.max_output_dimensions[0] // operation.tile_shape.total_row_dim()
        )
        block_tiles_x = operation.block_tiles_x
        block_tiles_y = operation.block_tiles_y

        full_blocks_x = tile_count_x // block_tiles_x
        full_blocks_y = tile_count_y // block_tiles_y
        remaining_tiles_x = tile_count_x % block_tiles_x
        remaining_tiles_y = tile_count_y % block_tiles_y

        full_x_limit = full_blocks_x * block_tiles_x
        full_y_limit = full_blocks_y * block_tiles_y

        tile_size = tilized_dst.shape[1]

        def process_block(block_x, block_y, block_tiles_x_eff, block_tiles_y_eff):
            block_tile_ids = []
            for tile_y in range(block_tiles_y_eff):
                for tile_x in range(block_tiles_x_eff):
                    tile_id = tile_count_x * (block_y + tile_y) + (block_x + tile_x)
                    block_tile_ids.append(tile_id)

            block_tile_cnt = len(block_tile_ids)
            if block_tile_cnt == 0:
                return

            block_tensor = tilized_dst[block_tile_ids, :].clone().flatten()
            block_dims = (
                block_tile_cnt * operation.tile_shape.total_row_dim(),
                operation.tile_shape.total_col_dim(),
            )

            block_tensor = self.sfpu.golden(
                block_tensor,
                operation,
                config,
                self,
                block_dims,
                block_tile_cnt,
            )

            tilized_dst[block_tile_ids, :] = block_tensor.view(
                block_tile_cnt, tile_size
            )

        if full_blocks_x > 0 and full_blocks_y > 0:
            for block_x in range(0, full_x_limit, block_tiles_x):
                for block_y in range(0, full_y_limit, block_tiles_y):
                    process_block(block_x, block_y, block_tiles_x, block_tiles_y)

        if remaining_tiles_y > 0 and full_blocks_x > 0:
            for block_x in range(0, full_x_limit, block_tiles_x):
                process_block(block_x, full_y_limit, block_tiles_x, remaining_tiles_y)

        if remaining_tiles_x > 0 and full_blocks_y > 0:
            for block_y in range(0, full_y_limit, block_tiles_y):
                process_block(full_x_limit, block_y, remaining_tiles_x, block_tiles_y)

        if remaining_tiles_x > 0 and remaining_tiles_y > 0:
            process_block(
                full_x_limit, full_y_limit, remaining_tiles_x, remaining_tiles_y
            )

        tensor_dst = untilize_block(
            tilized_dst.flatten(),
            config.sentinel.golden_math_format,
            operation.max_output_dimensions,
            tile_dimensions=tile_dims,
            num_faces=num_faces,
        ).reshape(operation.max_output_dimensions)

        return (
            tensor_a,
            tensor_b,
            tensor_dst.reshape(operation.max_output_dimensions),
        )

    def block_golden(
        self,
        call: KernelInvocation,
        tensor_dst: torch.Tensor,
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        tile_dims = (
            operation.tile_shape.total_row_dim(),
            operation.tile_shape.total_col_dim(),
        )
        tile_size = operation.tile_shape.total_tile_size()
        data_format = config.sentinel.golden_math_format
        if call.src0 is None:
            result = get_golden_generator(UnarySFPUGolden)(
                self.sfpu.operation,
                tensor_dst[call.dest].clone(),
                data_format,
                config.dest_acc,
                data_format,
                tile_dims,
                self.sfpu.iterations,
                0,
                self.sfpu.fill_const_value,
                skip_tilize=True,
            )
            tensor_dst[call.dest] = result.view(tile_size)
            return tensor_dst

        work = torch.stack(
            (
                tensor_dst[call.src0].clone(),
                tensor_dst[call.src1].clone(),
                tensor_dst[call.dest].clone(),
            )
        )
        result = get_golden_generator(BinarySFPUGolden)(
            self.sfpu.operation,
            work.flatten(),
            0,
            1,
            2,
            self.sfpu.iterations,
            (tile_dims[0] * 3, tile_dims[1]),
            data_format,
            skip_tilize=True,
        )
        tensor_dst[call.dest] = result.view(3, tile_size)[2]
        return tensor_dst

    def get_headers(self):
        return self.sfpu.get_headers()

    def __str__(self):
        return f"{self.sfpu}"
