# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import TYPE_CHECKING, List, Union

import torch
from helpers.tilize_untilize import untilize_block

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig

from helpers.llk_params import GoldenType, format_dict

from .arch_common import fpu_common, pack_common, unpack_common
from .base_fpu import Fpu
from .base_sfpu import Sfpu
from .base_unpacker import Unpacker
from .block_data import (
    BlockData,
    InvocationGranularity,
    KernelInvocation,
    NodeBlockPlan,
)
from .fpu_node import FpuNode
from .pack_node import PackNode
from .sfpu_node import SfpuNode


class ComputePipeline:
    math_nodes: List[Union[FpuNode, SfpuNode]]
    pack_nodes: List[Union[PackNode, SfpuNode]]

    def __init__(
        self,
        math_nodes: List[Union[FpuNode, SfpuNode]],
        pack_nodes: List[Union[PackNode, SfpuNode]],
        explicit_blocks: bool = False,
    ):
        self.math_nodes = math_nodes
        self.pack_nodes = pack_nodes
        self.explicit_blocks = explicit_blocks
        self.block_data = []
        self.codegen_block_data = []

    @staticmethod
    def _positions(granularity: InvocationGranularity, block: BlockData):
        if granularity == InvocationGranularity.TILE:
            return (
                (tile_x, tile_y)
                for tile_x in range(block.block_tiles_x)
                for tile_y in range(block.block_tiles_y)
            )
        if granularity == InvocationGranularity.ROW:
            return ((0, tile_y) for tile_y in range(block.block_tiles_y))
        if granularity == InvocationGranularity.BLOCK:
            return ((0, 0),)
        return ()

    @staticmethod
    def _call(block: BlockData, tile_x: int, tile_y: int) -> KernelInvocation:
        if any(
            isinstance(value, str)
            for value in (block.block_x, block.block_y, tile_x, tile_y)
        ):
            global_id = f"({block.tile_count_x} * ({block.block_y} + {tile_y}) + ({block.block_x} + {tile_x}))"
            dest_id = f"({tile_y} * {block.block_tiles_x} + {tile_x})"
        else:
            global_id = (
                block.tile_count_x * (block.block_y + tile_y) + block.block_x + tile_x
            )
            dest_id = tile_y * block.block_tiles_x + tile_x
        return KernelInvocation(
            in0=global_id,
            in1=global_id,
            src0=dest_id,
            src1=dest_id,
            dest=dest_id,
            out=global_id,
        )

    @staticmethod
    def _with_defaults(
        call: KernelInvocation, defaults: KernelInvocation
    ) -> KernelInvocation:
        return replace(
            call,
            **{
                name: value
                for name, value in vars(defaults).items()
                if value is not None
            },
        )

    @classmethod
    def _fpu_call(
        cls,
        block: BlockData,
        tile_x: int,
        tile_y: int,
        node: FpuNode,
    ) -> KernelInvocation:
        call = cls._call(block, tile_x, tile_y)
        if node.src_b is None:
            call = replace(call, in1=None)
        return call

    @staticmethod
    def _sfpu_call(node: SfpuNode) -> KernelInvocation:
        sfpu = node.sfpu
        if hasattr(sfpu, "dst_index_in0"):
            return KernelInvocation(
                src0=sfpu.dst_index_in0,
                src1=sfpu.dst_index_in1,
                dest=sfpu.dst_index_out,
            )
        return KernelInvocation(dest=sfpu.dest_idx)

    @staticmethod
    def _symbolic_position(granularity: InvocationGranularity):
        if granularity == InvocationGranularity.TILE:
            return "tile_x", "tile_y"
        if granularity == InvocationGranularity.ROW:
            return 0, "tile_y"
        if granularity == InvocationGranularity.BLOCK:
            return 0, 0
        return None

    def plan(self, operation: "L1Operation"):
        self.block_data = []
        self.codegen_block_data = []
        nodes = self.math_nodes + self.pack_nodes
        if self.explicit_blocks:
            tile_count_x = (
                operation.max_output_dimensions[1]
                // operation.tile_shape.total_col_dim()
            )
            tile_count_y = (
                operation.max_output_dimensions[0]
                // operation.tile_shape.total_row_dim()
            )
            for block_index in range(len(nodes[0].blocks)):
                self.block_data.append(
                    BlockData(
                        block_x=0,
                        block_y=0,
                        block_tiles_x=1,
                        block_tiles_y=1,
                        tile_count_x=tile_count_x,
                        tile_count_y=tile_count_y,
                        full_x_limit=tile_count_x,
                        full_y_limit=tile_count_y,
                        tile_id_global=0,
                        tile_id_block=0,
                        block_index=block_index,
                    )
                )
            for node in self.math_nodes:
                if isinstance(node, FpuNode):
                    node.unpack_blocks = node.blocks
            return

        tile_count_x = (
            operation.max_output_dimensions[1] // operation.tile_shape.total_col_dim()
        )
        tile_count_y = (
            operation.max_output_dimensions[0] // operation.tile_shape.total_row_dim()
        )
        full_x_limit = tile_count_x // operation.block_tiles_x * operation.block_tiles_x
        full_y_limit = tile_count_y // operation.block_tiles_y * operation.block_tiles_y

        def add_blocks(x_origins, y_origins, tiles_x, tiles_y):
            for block_x in x_origins:
                for block_y in y_origins:
                    self.block_data.append(
                        BlockData(
                            block_x=block_x,
                            block_y=block_y,
                            block_tiles_x=tiles_x,
                            block_tiles_y=tiles_y,
                            tile_count_x=tile_count_x,
                            tile_count_y=tile_count_y,
                            full_x_limit=full_x_limit,
                            full_y_limit=full_y_limit,
                            tile_id_global=0,
                            tile_id_block=0,
                            block_index=len(self.block_data),
                        )
                    )

        full_x = range(0, full_x_limit, operation.block_tiles_x)
        full_y = range(0, full_y_limit, operation.block_tiles_y)
        add_blocks(
            full_x,
            full_y,
            operation.block_tiles_x,
            operation.block_tiles_y,
        )
        if full_y_limit < tile_count_y:
            add_blocks(
                full_x,
                (full_y_limit,),
                operation.block_tiles_x,
                tile_count_y - full_y_limit,
            )
        if full_x_limit < tile_count_x:
            add_blocks(
                (full_x_limit,),
                full_y,
                tile_count_x - full_x_limit,
                operation.block_tiles_y,
            )
        if full_x_limit < tile_count_x and full_y_limit < tile_count_y:
            add_blocks(
                (full_x_limit,),
                (full_y_limit,),
                tile_count_x - full_x_limit,
                tile_count_y - full_y_limit,
            )

        def add_codegen_block(
            block_x, block_y, tiles_x, tiles_y, loop_x=False, loop_y=False
        ):
            self.codegen_block_data.append(
                BlockData(
                    block_x=block_x,
                    block_y=block_y,
                    block_tiles_x=tiles_x,
                    block_tiles_y=tiles_y,
                    tile_count_x=tile_count_x,
                    tile_count_y=tile_count_y,
                    full_x_limit=full_x_limit,
                    full_y_limit=full_y_limit,
                    tile_id_global=0,
                    tile_id_block=0,
                    block_index=len(self.codegen_block_data),
                    codegen=True,
                    loop_x=loop_x,
                    loop_y=loop_y,
                )
            )

        if full_x_limit > 0 and full_y_limit > 0:
            add_codegen_block(
                "block_x",
                "block_y",
                operation.block_tiles_x,
                operation.block_tiles_y,
                loop_x=True,
                loop_y=True,
            )
        if full_x_limit > 0 and full_y_limit < tile_count_y:
            add_codegen_block(
                "block_x",
                full_y_limit,
                operation.block_tiles_x,
                tile_count_y - full_y_limit,
                loop_x=True,
            )
        if full_x_limit < tile_count_x and full_y_limit > 0:
            add_codegen_block(
                full_x_limit,
                "block_y",
                tile_count_x - full_x_limit,
                operation.block_tiles_y,
                loop_y=True,
            )
        if full_x_limit < tile_count_x and full_y_limit < tile_count_y:
            add_codegen_block(
                full_x_limit,
                full_y_limit,
                tile_count_x - full_x_limit,
                tile_count_y - full_y_limit,
            )

        for node in self.math_nodes:
            if isinstance(node, FpuNode):
                node.unpack_blocks = [
                    NodeBlockPlan(
                        tuple(
                            self._with_defaults(
                                self._fpu_call(block, x, y, node),
                                replace(node.block_defaults, dest=None),
                            )
                            for x, y in self._positions(
                                (
                                    node.unpacker.granularity
                                    if node.unpacker is not None
                                    else InvocationGranularity.NONE
                                ),
                                block,
                            )
                        )
                    )
                    for block in self.block_data
                ]
                node.blocks = [
                    NodeBlockPlan(
                        tuple(
                            self._with_defaults(
                                self._fpu_call(block, x, y, node),
                                node.block_defaults,
                            )
                            for x, y in self._positions(node.fpu.granularity, block)
                        )
                    )
                    for block in self.block_data
                ]
                unpack_granularity = (
                    node.unpacker.granularity
                    if node.unpacker is not None
                    else InvocationGranularity.NONE
                )
                node.codegen_unpack_blocks = []
                node.codegen_blocks = []
                for block in self.codegen_block_data:
                    position = self._symbolic_position(unpack_granularity)
                    calls = ()
                    if position is not None:
                        calls = (
                            self._with_defaults(
                                self._fpu_call(block, *position, node),
                                replace(node.block_defaults, dest=None),
                            ),
                        )
                    node.codegen_unpack_blocks.append(NodeBlockPlan(calls))

                    position = self._symbolic_position(node.fpu.granularity)
                    calls = ()
                    if position is not None:
                        calls = (
                            self._with_defaults(
                                self._fpu_call(block, *position, node),
                                node.block_defaults,
                            ),
                        )
                    node.codegen_blocks.append(NodeBlockPlan(calls))
            else:
                call = self._sfpu_call(node)
                node.blocks = [NodeBlockPlan((call,)) for _ in self.block_data]
                node.codegen_blocks = [
                    NodeBlockPlan((call,)) for _ in self.codegen_block_data
                ]

        for node in self.pack_nodes:
            if isinstance(node, SfpuNode):
                call = self._sfpu_call(node)
                node.blocks = [NodeBlockPlan((call,)) for _ in self.block_data]
                node.codegen_blocks = [
                    NodeBlockPlan((call,)) for _ in self.codegen_block_data
                ]
                continue
            node.blocks = []
            for block in self.block_data:
                calls = []
                for x, y in self._positions(node.packer.granularity, block):
                    call = self._call(block, x, y)
                    if (
                        node.pack_l1_accumulation.value
                        and node.packer.granularity == InvocationGranularity.TILE
                    ):
                        call = KernelInvocation(
                            dest=call.dest, out=y * block.tile_count_x + x
                        )
                    calls.append(call)
                node.blocks.append(NodeBlockPlan(tuple(calls)))
            node.codegen_blocks = []
            for block in self.codegen_block_data:
                position = self._symbolic_position(node.packer.granularity)
                calls = ()
                if position is not None:
                    x, y = position
                    call = self._call(block, x, y)
                    if (
                        node.pack_l1_accumulation.value
                        and node.packer.granularity == InvocationGranularity.TILE
                    ):
                        call = KernelInvocation(
                            dest=call.dest,
                            out=f"{y} * {block.tile_count_x} + {x}",
                        )
                    calls = (call,)
                node.codegen_blocks.append(NodeBlockPlan(calls))

    def _get_pack_nodes(self) -> List[PackNode]:
        return [pn for pn in self.pack_nodes if isinstance(pn, PackNode)]

    def get_unpackers(self) -> List["Unpacker"]:
        unpackers: List["Unpacker"] = []

        for operation in self.math_nodes:
            if isinstance(operation, FpuNode) and operation.unpacker is not None:
                unpackers.append(operation.unpacker)

        return unpackers

    def get_math_units(self) -> List[Union["Fpu", "Sfpu"]]:
        math_units = []

        for operation in self.math_nodes:
            if isinstance(operation, FpuNode):
                math_units.append(operation.fpu)
            elif isinstance(operation, SfpuNode):
                math_units.append(operation.sfpu)

        return math_units

    def _all_same_operand_formats(self, ops: List[FpuNode]) -> bool:
        def signature(op: FpuNode):
            return (
                op.src_a.data_format if op.src_a is not None else None,
                op.src_b.data_format if op.src_b is not None else None,
            )

        return len({signature(op) for op in ops}) <= 1

    def _batch_loop(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        body_fn,
        init_fn=None,
        uninit_fn=None,
    ) -> str:
        code = ""
        if not self.explicit_blocks:
            for block in self.codegen_block_data:
                body = body_fn(block)
                if not body:
                    continue
                if block.loop_y:
                    body = (
                        f"for (std::uint32_t block_y = 0; block_y < {block.full_y_limit}; "
                        f"block_y += {block.block_tiles_y}) {{\n{body}}}\n"
                    )
                if block.loop_x:
                    body = (
                        f"for (std::uint32_t block_x = 0; block_x < {block.full_x_limit}; "
                        f"block_x += {block.block_tiles_x}) {{\n{body}}}\n"
                    )
                if init_fn is not None:
                    code += init_fn(block)
                code += body
                if uninit_fn is not None:
                    code += uninit_fn(block)
            return code

        groups = []
        for block in self.block_data:
            shape = (block.block_tiles_x, block.block_tiles_y)
            if not groups or groups[-1][0] != shape:
                groups.append((shape, []))
            groups[-1][1].append(block)
        for _, blocks in groups:
            bodies = [(block, body_fn(block)) for block in blocks]
            bodies = [(block, body) for block, body in bodies if body]
            if not bodies:
                continue
            if init_fn is not None:
                code += init_fn(bodies[0][0])
            code += "".join(body for _, body in bodies)
            if uninit_fn is not None:
                code += uninit_fn(bodies[-1][0])
        return code

    def _zone(self, config: "GlobalConfig", name: str, body: str) -> str:
        if not config.profiler_enabled:
            return body
        code = "{\n"
        code += f'ZONE_SCOPED("{name}")\n'
        code += body
        code += "PROFILER_SYNC();\n"
        code += "}\n"
        return code

    def _zone_loop(self, config: "GlobalConfig", name: str, body: str) -> str:
        if not config.profiler_enabled:
            return body
        code = "{\n"
        code += f'ZONE_SCOPED("{name}")\n'
        code += f"for(int loop = 0; loop < {config.loop_factor}; loop++)\n"
        code += "{\n"
        code += body
        code += "}\n"
        code += "PROFILER_SYNC();\n"
        code += "}\n"
        return code

    def unpack_body(self, operation: "L1Operation", config: "GlobalConfig") -> str:
        unpack_ops = [
            cu
            for cu in self.math_nodes
            if isinstance(cu, FpuNode) and cu.unpacker is not None
        ]
        hoist = len(unpack_ops) == 1
        hoist_reconfig = hoist or self._all_same_operand_formats(unpack_ops)

        init_code = ""
        init_code += unpack_common.dvalid_init(config=config, operation=operation)
        init_code += config.sentinel.hw_configure_unpack(config, operation)
        if hoist_reconfig and unpack_ops and not config.skip_unpack_init:
            init_code += config.sentinel.configure_unpack(
                config, operation, unpack_ops[0]
            )
        if hoist and not unpack_ops[0].unpacker.per_block_init:
            init_code += unpack_ops[0].unpack_init(operation, config, None)
        code = self._zone(config, "INIT", init_code)

        code += unpack_common.sync_with_packer(config, operation)

        init_fn = None
        uninit_fn = None
        if hoist and unpack_ops[0].unpacker.per_block_init:
            init_fn = lambda block: unpack_ops[0].unpack_init(operation, config, block)
            uninit_fn = lambda block: unpack_ops[0].unpack_uninit(
                operation, config, block
            )

        def batch_body(block: BlockData):
            body = ""
            for cu in self.math_nodes:
                if not isinstance(cu, FpuNode):
                    continue
                if (
                    not hoist_reconfig
                    and cu.unpacker is not None
                    and not config.skip_unpack_init
                ):
                    body += config.sentinel.configure_unpack(config, operation, cu)
                if not hoist:
                    body += cu.unpack_init(operation, config, block)
                body += cu.unpack_run(operation, config, block)
                if not hoist:
                    body += cu.unpack_uninit(operation, config, block)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(operation, config, batch_body, init_fn, uninit_fn),
        )

        uninit_code = ""
        if hoist and not unpack_ops[0].unpacker.per_block_init:
            uninit_code += unpack_ops[0].unpack_uninit(operation, config, None)
        code += self._zone(config, "INIT", uninit_code)

        return code

    def math_body(self, operation: "L1Operation", config: "GlobalConfig") -> str:
        code = f"// Operation {operation.stage_id}: Math Setup\n"
        fpu_ops = [cu for cu in self.math_nodes if isinstance(cu, FpuNode)]
        hoist = len(fpu_ops) == 1
        hoist_reconfig = hoist or self._all_same_operand_formats(fpu_ops)

        init_code = config.sentinel.hw_configure_math(config, operation)
        init_code += fpu_common.math_pack_sync_init(config, operation)
        init_code += fpu_common.math_dest_remap_config(
            any(pn.packer.requires_dest_remap for pn in self._get_pack_nodes())
        )
        if hoist_reconfig and fpu_ops and not config.skip_math_init:
            init_code += config.sentinel.configure_math(config, operation, fpu_ops[0])
        if hoist and not fpu_ops[0].fpu.per_block_init:
            init_code += fpu_ops[0].fpu_init(operation, config, None)
        code += self._zone(config, "INIT", init_code)

        init_fn = None
        uninit_fn = None
        if hoist and fpu_ops[0].fpu.per_block_init:
            init_fn = lambda block: fpu_ops[0].fpu_init(operation, config, block)
            uninit_fn = lambda block: fpu_ops[0].fpu_uninit(operation, config, block)

        def batch_body(block: BlockData):
            body = fpu_common.math_wait_for_dest(config, operation)
            for cu in self.math_nodes:
                if isinstance(cu, FpuNode):
                    if not hoist_reconfig and not config.skip_math_init:
                        body += config.sentinel.configure_math(config, operation, cu)
                    if not hoist:
                        body += cu.fpu_init(operation, config, block)
                    body += cu.fpu_run(operation, config, block)
                    if not hoist:
                        body += cu.fpu_uninit(operation, config, block)
                elif isinstance(cu, SfpuNode):
                    body += cu.sfpu_init(operation, config, block)
                    body += cu.sfpu_run(operation, config, block)
                    body += cu.sfpu_uninit(operation, config, block)
            body += fpu_common.math_dest_section_done(config, operation)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(operation, config, batch_body, init_fn, uninit_fn),
        )

        uninit_code = ""
        if hoist and not fpu_ops[0].fpu.per_block_init:
            uninit_code += fpu_ops[0].fpu_uninit(operation, config, None)
        code += self._zone(config, "INIT", uninit_code)

        return code

    def _all_same_pack_formats(self) -> bool:
        pack_only = self._get_pack_nodes()
        if len(pack_only) <= 1:
            return True
        first_fmt = pack_only[0].output.data_format
        return all(pn.output.data_format == first_fmt for pn in pack_only[1:])

    def pack_body(self, operation: "L1Operation", config: "GlobalConfig") -> str:
        code = f"// Operation {operation.stage_id}: Packer\n"
        pack_only = self._get_pack_nodes()
        hoist = len(pack_only) == 1 and len(self.pack_nodes) == 1
        hoist_reconfig = hoist or self._all_same_pack_formats()

        init_code = config.sentinel.hw_configure_pack(config, operation, pack_only)
        if hoist_reconfig and pack_only:
            init_code += config.sentinel.configure_pack(config, operation, pack_only[0])
        init_code += pack_common.pack_reduce_mask_config(operation)
        init_code += pack_common.pack_dest_init(config, operation, pack_only[0])
        if hoist and not pack_only[0].packer.per_block_init:
            init_code += pack_only[0].init(operation, config, None)
        code += self._zone(config, "INIT", init_code)

        init_fn = None
        uninit_fn = None
        if hoist and pack_only[0].packer.per_block_init:
            init_fn = lambda block: pack_only[0].init(operation, config, block)
            uninit_fn = lambda block: pack_only[0].uninit(operation, config)

        def batch_body(block: BlockData):
            body = pack_common.packer_wait_for_math(config, operation)
            if not hoist_reconfig:
                config.sentinel.reset_pack_formats()
            prev_was_pack = False
            for pack_node in self.pack_nodes:
                if isinstance(pack_node, SfpuNode):
                    if prev_was_pack:
                        body += "TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK);\n"
                    body += pack_node.sfpu_init(operation, config, block)
                    body += pack_node.sfpu_run(operation, config, block)
                    body += pack_node.sfpu_uninit(operation, config, block)
                    prev_was_pack = False
                elif isinstance(pack_node, PackNode):
                    if not hoist_reconfig:
                        body += config.sentinel.configure_pack(
                            config, operation, pack_node
                        )
                    if not hoist:
                        body += pack_node.init(operation, config, block)
                    body += pack_node.pack_loop(operation, config, block)
                    if not hoist:
                        body += pack_node.uninit(operation, config)
                    prev_was_pack = True
            body += pack_common.packer_dest_section_done(config, operation)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(operation, config, batch_body, init_fn, uninit_fn),
        )

        uninit_code = pack_common.packer_sync_with_unpacker(config, operation)
        if hoist and not pack_only[0].packer.per_block_init:
            uninit_code += pack_only[0].uninit(operation, config)
        uninit_code += pack_common.pack_reduce_mask_clear(operation)
        code += self._zone(config, "INIT", uninit_code)

        return code

    def golden(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        golden_type: GoldenType,
    ):
        if self.explicit_blocks:
            return self._block_golden(operation, config, golden_type)

        first_fpu = next(
            (
                op
                for op in self.math_nodes
                if isinstance(op, FpuNode) and op.src_a is not None
            ),
            None,
        )
        if first_fpu is not None:
            tensor_a = torch.zeros(first_fpu.src_a.dimensions)
            tensor_b = torch.zeros(
                first_fpu.src_b.dimensions
                if first_fpu.src_b is not None
                else first_fpu.src_a.dimensions
            )
        else:
            tensor_a = torch.zeros(operation.max_output_dimensions)
            tensor_b = torch.zeros(operation.max_output_dimensions)
        tensor_dst = torch.zeros(operation.max_output_dimensions)
        for op in self.math_nodes:
            config.sentinel.configure_golden(config, operation, op)
            if isinstance(op, FpuNode) and op.src_a is not None:
                input_tensor_a = (
                    op.src_a.raw_data
                    if golden_type == GoldenType.L1_GOLDEN
                    else op.src_a.master_golden
                )
                input_tensor_b = (
                    (
                        op.src_b.raw_data
                        if golden_type == GoldenType.L1_GOLDEN
                        else op.src_b.master_golden
                    )
                    if op.src_b is not None
                    else None
                )
            else:
                input_tensor_a = None
                input_tensor_b = None
            tensor_a, tensor_b, tensor_dst = op.golden(
                input_tensor_a,
                input_tensor_b,
                tensor_a,
                tensor_b,
                tensor_dst,
                operation,
                config,
            )

        for pack_node in self.pack_nodes:
            if isinstance(pack_node, SfpuNode):
                tensor_a, tensor_b, tensor_dst = pack_node.golden(
                    None, None, tensor_a, tensor_b, tensor_dst, operation, config
                )
                continue

            config.sentinel.configure_golden(
                config, operation, output_format=pack_node.output.data_format
            )

            dimensions = pack_node.output.dimensions
            cropped = tensor_dst.reshape(operation.max_output_dimensions)[
                : dimensions[0], : dimensions[1]
            ]
            result = pack_node.golden(cropped, operation, config)

            if golden_type == GoldenType.L1_GOLDEN:
                pack_node.output.l1_golden = result
            else:
                pack_node.output._master_golden = result

    def _block_golden(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        golden_type: GoldenType,
    ):
        pack_nodes = self._get_pack_nodes()
        outputs = {
            id(node): torch.zeros(
                (node.output.tile_count, node.output.tile_shape.total_tile_size()),
                dtype=format_dict[node.output.data_format],
            )
            for node in pack_nodes
        }
        config.sentinel.configure_golden(
            config, operation, output_format=pack_nodes[0].output.data_format
        )
        master = golden_type == GoldenType.MASTER_GOLDEN
        nodes = self.math_nodes + self.pack_nodes
        for block_index in range(len(nodes[0].blocks)):
            dest_indices = [
                index
                for node in nodes
                for call in node.blocks[block_index].calls
                for index in (call.src0, call.src1, call.dest)
                if index is not None
            ]
            tensor_dst = torch.zeros(
                (
                    max(dest_indices, default=0) + 1,
                    operation.tile_shape.total_tile_size(),
                ),
                dtype=format_dict[config.sentinel.golden_math_format],
            )
            for node in self.math_nodes:
                config.sentinel.configure_golden(config, operation, node)
                for call in node.blocks[block_index].calls:
                    tensor_dst = (
                        node.block_golden(call, tensor_dst, operation, config, master)
                        if isinstance(node, FpuNode)
                        else node.block_golden(call, tensor_dst, operation, config)
                    )
            for node in self.pack_nodes:
                if isinstance(node, SfpuNode):
                    for call in node.blocks[block_index].calls:
                        tensor_dst = node.block_golden(
                            call, tensor_dst, operation, config
                        )
                    continue
                config.sentinel.configure_golden(
                    config, operation, output_format=node.output.data_format
                )
                for call in node.blocks[block_index].calls:
                    outputs[id(node)] = node.block_golden(
                        call, tensor_dst, outputs[id(node)], operation, config
                    )

        for node in pack_nodes:
            tile_shape = node.output.tile_shape
            result = untilize_block(
                outputs[id(node)].flatten(),
                node.output.data_format,
                node.output.dimensions,
                tile_dimensions=(
                    tile_shape.total_row_dim(),
                    tile_shape.total_col_dim(),
                ),
                num_faces=tile_shape.total_num_faces(),
            )
            if golden_type == GoldenType.L1_GOLDEN:
                node.output.l1_golden = result
            else:
                node.output._master_golden = result

    def __str__(self):
        result = "Math:"
        for op in self.math_nodes:
            result += "\n    "
            result += op.__str__()
        result += "\n  Pack:"
        for pn in self.pack_nodes:
            result += "\n    "
            if isinstance(pn, PackNode):
                result += pn.output.__str__()
            else:
                result += str(pn)
        return result
