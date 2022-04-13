# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ZeRO-3: Partition parameters."""
from tvm.relay import TensorType
from raf._core.ndarray import ndarray
from raf.frontend.model import FrameworkModel
from .. import distributed as dist
from .._ffi.pass_ import DistParams, EraseType, InferType
from .utils import split_ndarray_with_padding


def zero3(model, args):
    """ZeRO-3: Partition parameters. This should be applied by optimizers after AutoDiff.

    Returns
    -------
    model : raf.model.Model
        Model to be partitioned. Note that this model MUST NOT have optimizer.

    args: List[raf.ndarray]
        The input data of the model.
    """
    comm = dist.get_communicator()
    state = {}
    partitioned_types = {}
    for name in model.state().keys():
        param = model.state()[name]

        # For each tensor "param" that requires gradient (i.e., training weights),
        # partition the tensor based on dist rank.
        if param.requires_grad and "float" in param.dtype:
            assert isinstance(param, ndarray), "Only `raf.ndarray` can be optimized!"

            # The first axis of the parameter is partitioned to 1/n.
            # Pad and copy a slice of weight to be the SGD statues.
            param_nd = param.to(device="cpu")
            slice_param = split_ndarray_with_padding(param_nd, comm.size)[comm.rank]
            part_param = ndarray(
                slice_param,
                device=param.device,
                name=f"{name}.part",
                dtype=param.dtype,
            )
            # Release the original parameter.
            del param
            # Override the parameter with partitioned one.
            state[name] = part_param
            partitioned_types[name] = TensorType(part_param.shape, dtype=param_nd.dtype)
        else:
            state[name] = param

    if not state:
        return model

    train_mod = model._internal(*args).mod
    train_mod = DistParams(partitioned_types)(train_mod)
    train_mod = InferType()(EraseType()(train_mod))
    return FrameworkModel(train_mod, train_mod, state, dict())
