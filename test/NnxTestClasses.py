# Luka Macan <luka.macan@unibo.it>
# NEUREKA-TX: Lionnus Kesting <lkesting@ethz.ch>
#
# Copyright 2023 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Callable, Optional, Set, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, PositiveInt, field_validator, model_validator

from HeaderWriter import HeaderWriter
from NeuralEngineFunctionalModel import NeuralEngineFunctionalModel
from TestClasses import IntegerType, KernelShape, Padding, Stride, implies
from NeurekaPPolyApproxModel import PiecewisePolyApproxModel


class NnxTestConf(BaseModel):
    # Input / output shapes
    in_height: PositiveInt
    in_width: PositiveInt
    in_channel: PositiveInt
    out_channel: PositiveInt

    # Layer parameters
    padding: Padding
    kernel_shape: KernelShape
    depthwise: bool
    stride: Stride

    # Datatypes
    in_type: IntegerType
    out_type: IntegerType
    weight_type: IntegerType
    scale_type: Optional[IntegerType] = None
    bias_type: Optional[IntegerType] = None

    # Feature flags
    has_norm_quant: bool
    has_bias: bool
    has_relu: bool
    
    # GEMM Operation mode flag
    is_gemm: bool = False
    polyapprox_degree: int = 0  # Poly approximation, 0 for none, 1 for linear, 2 for quadratic
    polyapprox_segments: int = 16  # Number of segments for piecewise approximation
    polyapprox_bounds_bitwidth: int = 8 # Number of bits for each bound value
    polyapprox_coeffs_mul_bitwidth: int = 8 # Number of bits for each multiplication coefficient value
    polyapprox_coeffs_add_bitwidth: int = 32 # Number of bits for each addition coefficient value

    polyapprox_func: str = "gelu"  # Function to approximate: check NeurekaPPolyApproxModel for available functions

    # Test‑data generation helpers
    synthetic_weights: bool = False
    synthetic_inputs: bool = False

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")  # type: ignore
    def check_valid_depthwise_channels(self) -> NnxTestConf:
        assert implies(self.depthwise, self.in_channel == self.out_channel), (
            f"Input and output channel should be the same in a depthwise layer. "
            f"input channel: {self.in_channel}, output channel: {self.out_channel}"
        )
        return self

    @model_validator(mode="after")  # type: ignore
    def check_valid_padding_with_kernel_shape_1x1(self) -> NnxTestConf:
        assert implies(
            self.kernel_shape == KernelShape(height=1, width=1),
            self.padding == Padding(top=0, bottom=0, left=0, right=0),
        ), f"No padding on 1x1 kernel. Given padding {self.padding}"
        return self

    @model_validator(mode="after")  # type: ignore
    def check_valid_norm_quant_types_when_has_norm_qunat(self) -> NnxTestConf:
        if self.has_norm_quant:
            assert self.scale_type is not None, "Scale type was not provided."
            if self.has_bias:
                assert self.bias_type is not None, "Bias type was not provided."
        return self

    @model_validator(mode="after")  # type: ignore
    def check_has_relu_with_norm_quant(self) -> NnxTestConf:
        assert implies(self.has_relu, self.has_norm_quant), (
            f"Relu flag can only be enabled when norm_quant is enabled. "
            f"Given has_relu {self.has_relu} and has_norm_quant {self.has_norm_quant}"
        )
        return self

    @model_validator(mode="after")  # type: ignore
    def check_has_bias_with_norm_quant(self) -> NnxTestConf:
        # GEMM tests are allowed to request a bias type, but the bias will be
        # hard‑clamped to zero in the generator, so the original constraint
        # is still valid.
        assert implies(self.has_bias, self.has_norm_quant), (
            f"Bias flag can only be enabled when norm_quant is enabled. "
            f"Given has_bias {self.has_bias} and has_norm_quant {self.has_norm_quant}"
        )
        return self

    @model_validator(mode="after")  # type: ignore
    def check_valid_out_type_with_relu(self) -> NnxTestConf:
        assert self.has_relu ^ self.out_type._signed, (
            f"Output type has to be unsigned when there is relu, otherwise signed. "
            f"Given output type {self.out_type} and has_relu {self.has_relu}"
        )
        return self


class NnxTest:
    _CONF_NAME = "conf.json"
    _INPUT_NAME = "input.pt"
    _OUTPUT_NAME = "output.pt"
    _WEIGHT_NAME = "weight.pt"
    _SCALE_NAME = "scale.pt"
    _BIAS_NAME = "bias.pt"
    _GLOBAL_SHIFT_NAME = "global_shift.pt"
    _SCALE2_NAME = "scale2.pt"
    _BIAS2_NAME = "bias2.pt"
    _GLOBAL_SHIFT2_NAME = "global_shift2.pt"
    _PPOLY_PARAMS_NAME = "ppolyapprox.pt"

    def __init__(
        self,
        conf: NnxTestConf,
        input: Optional[torch.Tensor],
        output: Optional[torch.Tensor],
        weight: Optional[torch.Tensor],
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        global_shift: Optional[torch.Tensor] = torch.Tensor([0]),
        scale2: Optional[torch.Tensor] = None,
        bias2: Optional[torch.Tensor] = None,
        global_shift2: Optional[torch.Tensor] = None,
        is_gemm: Optional[bool] = False,
        polyapprox_degree: int = 0,
        ppoly_params: Optional[torch.Tensor] = None,
        ppoly_model: Optional[PiecewisePolyApproxModel] = None,
        synthetic_weights: Optional[bool] = False,
        synthetic_inputs: Optional[bool] = False,
    ) -> None:
        self.conf = conf
        self.input = input
        self.output = output
        self.weight = weight
        self.scale = scale
        self.bias = bias
        self.global_shift = global_shift
        self.scale2 = scale2
        self.bias2 = bias2
        self.global_shift2 = global_shift2
        self.is_gemm = is_gemm
        self.polyapprox_degree = polyapprox_degree
        self.ppoly_params = ppoly_params
        self.ppoly_model = ppoly_model
        self.synthetic_weights = synthetic_weights
        self.synthetic_inputs = synthetic_inputs

    def is_valid(self) -> bool:
        return all(
            [
                self.input is not None,
                self.output is not None,
                self.weight is not None,
                implies(self.conf.has_norm_quant, self.scale is not None),
                implies(self.conf.has_bias, self.bias is not None),
                implies(self.conf.has_norm_quant, self.global_shift is not None),
            ]
        )

    def save_conf(self, path: Union[str, os.PathLike]) -> None:
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, NnxTest._CONF_NAME), "w") as fp:
            fp.write(self.conf.model_dump_json(indent=4))

    def save_data(self, path: Union[str, os.PathLike]) -> None:
        os.makedirs(path, exist_ok=True)

        torch.save(self.input, os.path.join(path, NnxTest._INPUT_NAME))
        torch.save(self.output, os.path.join(path, NnxTest._OUTPUT_NAME))
        torch.save(self.weight, os.path.join(path, NnxTest._WEIGHT_NAME))
        if self.scale is not None:
            torch.save(self.scale, os.path.join(path, NnxTest._SCALE_NAME))
        if self.bias is not None:
            torch.save(self.bias, os.path.join(path, NnxTest._BIAS_NAME))
        if self.global_shift is not None:
            torch.save(
                self.global_shift, os.path.join(path, NnxTest._GLOBAL_SHIFT_NAME)
            )
        if self.scale2 is not None:
            torch.save(self.scale2, os.path.join(path, NnxTest._SCALE2_NAME))
        if self.bias2 is not None:
            torch.save(self.bias2, os.path.join(path, NnxTest._BIAS2_NAME))
        if self.global_shift2 is not None:
            torch.save(
                self.global_shift2, os.path.join(path, NnxTest._GLOBAL_SHIFT2_NAME)
            )
        
        # Pack parameters only when saving
        if self.ppoly_model is not None:
            ppoly_params_packed = torch.from_numpy(
                self.ppoly_model.get_packed_parameters(
                    output_bits=10,  # TODO: make configurable
                    mul_bw=self.conf.polyapprox_coeffs_mul_bitwidth,
                    add_bw=self.conf.polyapprox_coeffs_add_bitwidth
                )
            ).to(torch.int32)
            torch.save(ppoly_params_packed, os.path.join(path, NnxTest._PPOLY_PARAMS_NAME))
        elif self.ppoly_params is not None:
            torch.save(self.ppoly_params, os.path.join(path, NnxTest._PPOLY_PARAMS_NAME))

    def save(self, path: Union[str, os.PathLike]) -> None:
        self.save_conf(path)
        self.save_data(path)

    @staticmethod
    def is_test_dir(path: Union[str, os.PathLike]) -> bool:
        fileset = set(os.listdir(path))
        required_fileset = set([NnxTest._CONF_NAME])
        return required_fileset.issubset(fileset)

    @classmethod
    def load(cls: Type[NnxTest], confCls: Type[NnxTestConf], path: Union[str, os.PathLike]) -> NnxTest:
        assert NnxTest.is_test_dir(
            path
        ), f"ERROR: Test {path} does not contain the necessary files."

        with open(os.path.join(path, NnxTest._CONF_NAME), "r") as fp:
            conf = confCls.model_validate_json(fp.read())

        def load_if_exist(filename: str) -> Optional[torch.Tensor]:
            filepath = os.path.join(path, filename)
            return torch.load(filepath) if os.path.isfile(filepath) else None

        input = load_if_exist(NnxTest._INPUT_NAME)
        output = load_if_exist(NnxTest._OUTPUT_NAME)
        weight = load_if_exist(NnxTest._WEIGHT_NAME)
        scale = load_if_exist(NnxTest._SCALE_NAME)
        bias = load_if_exist(NnxTest._BIAS_NAME)
        global_shift = load_if_exist(NnxTest._GLOBAL_SHIFT_NAME)
        scale2 = load_if_exist(NnxTest._SCALE2_NAME)
        bias2 = load_if_exist(NnxTest._BIAS2_NAME)
        global_shift2 = load_if_exist(NnxTest._GLOBAL_SHIFT2_NAME)
        ppoly_params = load_if_exist(NnxTest._PPOLY_PARAMS_NAME)
        
        return cls(
            conf, input, output, weight, scale, bias, global_shift,
            scale2, bias2, global_shift2, ppoly_params
        )


class NnxTestGenerator:
    _DEFAULT_SEED = 0
    _DEFAULT_WEIGHT_MEAN = 0.5 # as we use torch.floor(), this makes the generation unbiased
    _DEFAULT_WEIGHT_STDEV = 0.27
    _DEFAULT_SCALE_MAX_BIT_32BIT = 17
    _DEFAULT_SCALE_MAX_BIT_16BIT = 11
    _DEFAULT_SCALE_MAX_BIT_8BIT = 5
    _DEFAULT_BIAS_MAX_BIT = 18

    # ---------------------- Helper methods ----------------------

    @staticmethod
    def _calculate_global_shift(
        tensor: torch.Tensor, out_type: IntegerType
    ) -> torch.Tensor:
        """Calculate global shift so that the output values are in the range of out_type"""
        s = tensor.type(torch.float64).std()
        target_s = 2 ** (out_type._bits - 1)
        shift = torch.ceil(torch.log2(s / target_s))
        return torch.clamp(shift, 0, 255).type(torch.uint8)


    @staticmethod
    def _random_data(_type: IntegerType, shape: Tuple, extremes: Tuple = None):
        if extremes is None:
            return torch.randint(_type.min, _type.max, size=shape)
        else:
            return torch.randint(max(_type.min, extremes[0]), min(_type.max, extremes[1]), size=shape)

    @staticmethod
    def _random_data_normal(_type: IntegerType, shape: Tuple, mean: float = 0.5, std: float=0.27):
        return torch.floor(torch.clip(torch.normal(mean, std, size=shape), _type.min, _type.max)).type(torch.int64)

    @staticmethod
    def _get_activation_function(func_name: str) -> Callable: # TODO: Add I-BERT/make more flexible for other models
        """Get activation function by name."""
        functions = {
            "gelu": lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "tanh": lambda x: np.tanh(x),
            "relu": lambda x: np.maximum(0, x),
            "swish": lambda x: x * (1 / (1 + np.exp(-x))),
        }
        return functions.get(func_name, functions["gelu"])

    @staticmethod
    def from_conf(
        conf: NnxTestConf,
        input: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        global_shift: Optional[torch.Tensor] = None,
        scale2: Optional[torch.Tensor] = None,
        bias2: Optional[torch.Tensor] = None,
        global_shift2: Optional[torch.Tensor] = None,
        ppoly_params: Optional[torch.Tensor] = None, 
        verbose: bool = False,
    ) -> NnxTest:
        """Generate (or regenerate) a test‑vector bundle from a configuration."""
        torch.manual_seed(NnxTestGenerator._DEFAULT_SEED)

        # Initialize ppoly_model to None
        ppoly_model = None

        # ------------------------------------------------------------------
        # Shape bookkeeping
        # ------------------------------------------------------------------
        if conf.is_gemm:
            input_shape = (conf.in_channel, conf.in_height*conf.in_width)
            weight_shape = (conf.out_channel, conf.in_channel)
            scale_shape = (conf.out_channel,1) # TODO: How many diff scales?
            bias_shape = (conf.out_channel,1)
        else:
            #Convolution mode
            input_shape = (1, conf.in_channel, conf.in_height, conf.in_width)
            weight_shape = (
                conf.out_channel,
                1 if conf.depthwise else conf.in_channel,
                conf.kernel_shape.height,
                conf.kernel_shape.width,
            )
            scale_shape = (1, conf.out_channel, 1, 1)
            bias_shape = (1, conf.out_channel, 1, 1)

        # ------------------------------------------------------------------
        # Input tensor
        # ------------------------------------------------------------------
        if input is None:
            if conf.synthetic_inputs:
                input = torch.zeros(input_shape, dtype=torch.int64)
                for i in range(conf.in_channel):
                    input[:, i,0,0] = i
            else:
                input = NnxTestGenerator._random_data(
                    _type=conf.in_type,
                    shape=input_shape,
                )

        # ------------------------------------------------------------------
        # Weight tensor
        # ------------------------------------------------------------------
        if weight is None:
            if conf.synthetic_weights:
                weight = torch.zeros(weight_shape, dtype=torch.int64)
                for i in range(0, min(weight.shape[0], weight.shape[1])):
                    weight[i,i,0,0] = 1
            else:
                weight_mean = NnxTestGenerator._DEFAULT_WEIGHT_MEAN
                weight_std  = NnxTestGenerator._DEFAULT_WEIGHT_STDEV * (1<<(conf.weight_type._bits-1)-1)
                weight = NnxTestGenerator._random_data_normal(
                    mean = weight_mean,
                    std = weight_std,
                    _type=conf.weight_type,
                    shape=weight_shape,
                )

        # ------------------------------------------------------------------
        # Scale & bias (only if norm‑quant is enabled)
        # ------------------------------------------------------------------
        if conf.has_norm_quant:
            if scale is None:
                assert conf.scale_type is not None
                # same limits as in old NE16 generator
                scale_extremes = (1, (1<<NnxTestGenerator._DEFAULT_SCALE_MAX_BIT_32BIT)-1) if conf.scale_type._bits == 32 else \
                                 (1, (1<<NnxTestGenerator._DEFAULT_SCALE_MAX_BIT_16BIT)-1) if conf.scale_type._bits == 16 else \
                                 (1, (1<<NnxTestGenerator._DEFAULT_SCALE_MAX_BIT_8BIT)-1)  if conf.scale_type._bits == 8  else (1, (1<<18)-1)
                scale = NnxTestGenerator._random_data(
                    conf.scale_type, shape=scale_shape, extremes=scale_extremes
                )
            if conf.has_bias and bias is None:
                assert conf.bias_type is not None
                # same limits as in old NE16 generator
                bias_extremes = (-(1<<NnxTestGenerator._DEFAULT_BIAS_MAX_BIT), (1<<NnxTestGenerator._DEFAULT_BIAS_MAX_BIT)-1)
                bias = NnxTestGenerator._random_data(
                    conf.bias_type, shape=bias_shape, extremes=bias_extremes
                ).type(torch.int32)
                
            # Calculate global_shift for first norm-quant
            if global_shift is None:
                global_shift = torch.Tensor([0]).type(torch.int32)
                conv_kwargs = {
                    **conf.__dict__,
                    "norm1_out_type": NeuralEngineFunctionalModel.ACCUMULATOR_TYPE,
                    "norm2_out_type": NeuralEngineFunctionalModel.ACCUMULATOR_TYPE, # doesnt matter since not used in this pass
                    "skip_polyapprox_degree": True, # Skip to calculate first norm quant shift
                }
                if conf.is_gemm:
                    output = NeuralEngineFunctionalModel().gemm(
                        input,weight,scale,bias,global_shift,verbose=False,**conv_kwargs)
                else:
                    output = NeuralEngineFunctionalModel().convolution(
                        input, weight, scale, bias, global_shift, verbose=False, **conv_kwargs)
                global_shift = NnxTestGenerator._calculate_global_shift(
                    output, conf.out_type
                )
            
            # For GEMM with polynomial approximation, handle second norm-quant parameters
            if conf.is_gemm and conf.polyapprox_degree > 0:
                 # Get the activation function to approximate
                activation_func = NnxTestGenerator._get_activation_function(conf.polyapprox_func)
                
                # Create piecewise approximation model - keep it unpacked
                ppoly_model = PiecewisePolyApproxModel(
                    target_func=activation_func,
                    num_segments=conf.polyapprox_segments,
                    input_range=(-4.0, 4.0),  # TODO set from higher up
                    input_quantization=32
                )
                
                # Get quantized parameters and store them in the model for direct use
                boundaries_q, slopes_q, intercepts_q = ppoly_model.get_quantized_lut(
                    output_bits=10,  # TODO: make configurable
                    slope_bits=conf.polyapprox_coeffs_mul_bitwidth,
                    intercept_bits=conf.polyapprox_coeffs_add_bitwidth
                )

                # Store quantized parameters in the model for direct access
                ppoly_model.boundaries_q = boundaries_q
                ppoly_model.slopes_q = slopes_q
                ppoly_model.intercepts_q = intercepts_q
                
                if verbose:
                    print(f"Generated piecewise polynomial approximation for {conf.polyapprox_func}")
                    print(f"  Segments: {conf.polyapprox_segments}")
                    
                # Generate scale2 if not provided
                if scale2 is None and conf.scale_type is not None:
                    scale_extremes = (1, (1<<NnxTestGenerator._DEFAULT_SCALE_MAX_BIT_32BIT)-1) if conf.scale_type._bits == 32 else \
                                     (1, (1<<NnxTestGenerator._DEFAULT_SCALE_MAX_BIT_16BIT)-1) if conf.scale_type._bits == 16 else \
                                     (1, (1<<NnxTestGenerator._DEFAULT_SCALE_MAX_BIT_8BIT)-1)  if conf.scale_type._bits == 8  else (1, (1<<18)-1)
                    scale2 = NnxTestGenerator._random_data(
                        conf.scale_type, shape=scale_shape, extremes=scale_extremes
                    )
                
                # Generate bias2 if not provided and has_bias is True
                if conf.has_bias and bias2 is None and conf.bias_type is not None:
                    bias_extremes = (-(1<<NnxTestGenerator._DEFAULT_BIAS_MAX_BIT), (1<<NnxTestGenerator._DEFAULT_BIAS_MAX_BIT)-1)
                    bias2 = NnxTestGenerator._random_data(
                        conf.bias_type, shape=bias_shape, extremes=bias_extremes
                    ).type(torch.int32)
                
                # Calculate global_shift2 for second norm-quant
                if global_shift2 is None:
                    
                    # First run to get intermediate result after polynomial approximation
                    global_shift2 = torch.Tensor([0]).type(torch.int32)
                    conv_kwargs = {
                        **conf.__dict__,
                        "scale2": scale2,
                        "bias2": bias2,
                        "global_shift2": global_shift2,
                        "norm1_out_type": IntegerType(name="int8"),
                        "norm2_out_type": NeuralEngineFunctionalModel.ACCUMULATOR_TYPE,
                        "ppoly_model": ppoly_model,  # Pass the model directly
                    }
                    output = NeuralEngineFunctionalModel().gemm(
                        input, weight, scale, bias, global_shift, verbose=False, **conv_kwargs
                    )
                    global_shift2 = NnxTestGenerator._calculate_global_shift(
                        output, conf.out_type
                    )

        # Generate final output with all parameters
        if conf.is_gemm:
            conv_kwargs = {
                **conf.__dict__,
                "scale2": scale2,
                "bias2": bias2,
                "global_shift2": global_shift2,
                "norm1_out_type": IntegerType(name="int8"),
                "norm2_out_type": IntegerType(name="int8"),
            }
            if ppoly_model is not None:
                conv_kwargs["ppoly_model"] = ppoly_model
            output = NeuralEngineFunctionalModel().gemm(
                input, weight, scale, bias, global_shift, verbose=verbose, **conv_kwargs
            )
        else:
            output = NeuralEngineFunctionalModel().convolution(
                input, weight, scale, bias, global_shift, verbose=verbose, **conf.__dict__
            )

        return NnxTest(
            conf=conf,
            input=input,
            output=output,
            weight=weight,
            scale=scale,
            bias=bias,
            global_shift=global_shift,
            scale2=scale2,
            bias2=bias2,
            global_shift2=global_shift2,
            ppoly_params=ppoly_params,
            ppoly_model=ppoly_model,  # Store the model
            synthetic_inputs=conf.synthetic_inputs,
            synthetic_weights=conf.synthetic_weights,
        )

    # -----------------------------------------------------------------------
    # Regenerate test
    # -----------------------------------------------------------------------
    @staticmethod
    def regenerate(test: NnxTest, regen_tensors: Set[str]) -> NnxTest:
        test_tensors = set(["input", "output", "weight", "scale", "bias", "scale2", "bias2"])
        load_tensors = test_tensors - regen_tensors
        kwargs = {tensor: getattr(test, tensor) for tensor in load_tensors if hasattr(test, tensor)}
        return NnxTestGenerator.from_conf(test.conf, **kwargs)


class NnxTestHeaderGenerator:
    DEFAULT_HEADERS_DIR = "app/gen"

    def __init__(
        self,
        weightEncode: Callable[
            [npt.NDArray[np.uint8], int, bool], npt.NDArray[np.uint8]
        ],
        headers_dir: Optional[Union[str, os.PathLike]] = None,
    ):
        if headers_dir is None:
            headers_dir = NnxTestHeaderGenerator.DEFAULT_HEADERS_DIR
        self.header_writer = HeaderWriter(headers_dir)
        # function that takes the weights in CoutCinK format, bitwidth, and a depthwise flag,
        # and returns a numpy array of dtype=np.uint8 of data in a layout correct for the accelerator
        self.weightEncode = weightEncode

    def generate(self, test_name: str, test: NnxTest):
        assert test.input is not None and test.output is not None
        if test.conf.is_gemm:
            out_channel, in_channel = test.weight.shape
            in_height = test.conf.in_height
            in_width = test.conf.in_width
            out_height = in_height
            out_width = in_width
        else:
            _, in_channel, in_height, in_width = test.input.shape
            _, out_channel, out_height, out_width = test.output.shape

        # ------------------------------------------------------------------
        # Render input tensor
        # ------------------------------------------------------------------
        if test.conf.is_gemm:
            in_ctype = test.conf.in_type.ctype()
            in_signed = test.conf.in_type._signed
            in_data = test.input.permute(1,0).ravel()
        else:
            in_ctype = test.conf.in_type.ctype()
            in_signed = test.conf.in_type._signed
            in_data = test.input.permute(0, 2, 3, 1).ravel()
        self.header_writer.generate_vector_files(
            "input", _type=in_ctype, size=in_data.numel(), init=in_data
        )

        # ------------------------------------------------------------------
        # Render golden output tensor
        # ------------------------------------------------------------------
        if test.conf.is_gemm:
            out_ctype = test.conf.out_type.ctype()
            out_data_golden = test.output.permute(1,0).ravel()
        else:
            out_ctype = test.conf.out_type.ctype()
            out_data_golden = test.output.permute(0, 2, 3, 1).ravel()
        self.header_writer.generate_vector_files(
            "output",
            _type=out_ctype,
            size=out_data_golden.numel(),
            golden=out_data_golden,
        )
        # ------------------------------------------------------------------
        # Render weights (CoutCinK)
        # ------------------------------------------------------------------
        assert test.weight is not None
        weight_type = test.conf.weight_type
        weight_bits = weight_type._bits
        assert weight_bits > 1 and weight_bits <= 8
        if test.synthetic_weights:
            weight_offset = 0
        else:
            weight_offset = weight_bits - 1 #Changed from absolute value to shift value
        
        if test.conf.is_gemm:
            # GEMM mode
            weight_out_ch, weight_in_ch = test.weight.shape
            weight_ks_h = 1
            weight_ks_w = 1
            weight_ctype = test.conf.weight_type.ctype()
            # Cout, Cin, kernel_h, kernel_w shape
            weight_init = test.weight.permute(0, 1).contiguous().view(-1).numpy() #TODO Remove permute
            self.header_writer.generate_vector_files(
                "weight", _type=weight_ctype, size=weight_init.size, init=weight_init
            )
        else:
            weight_out_ch, weight_in_ch, weight_ks_h, weight_ks_w = test.weight.shape
            weight_data: np.ndarray = test.weight.numpy() + (2 ** (weight_bits - 1))
            weight_init = self.weightEncode(
            weight_data.astype(np.uint8),
            weight_type._bits,
            test.conf.depthwise,
            )
            self.header_writer.generate_vector_files(
                "weight", _type="uint8_t", size=weight_init.size, init=weight_init
            )
            
        # ------------------------------------------------------------------
        # Render scale
        # ------------------------------------------------------------------
        if test.scale is not None:
            assert test.conf.scale_type is not None
            scale_ctype = test.conf.scale_type.ctype()
            self.header_writer.generate_vector_files(
                "scale",
                _type=scale_ctype,
                size=test.scale.numel(),
                init=test.scale.ravel(),
            )

        # ------------------------------------------------------------------
        # Render bias
        # ------------------------------------------------------------------
        if test.bias is not None:
            assert test.conf.bias_type is not None
            bs_ctype = test.conf.bias_type.ctype()
            self.header_writer.generate_vector_files("bias", _type=bs_ctype, size=test.bias.numel(), init=test.bias.ravel()
            )

        # ------------------------------------------------------------------
        # Render scale2 (for GEMM post-polyapprox)
        # ------------------------------------------------------------------
        if test.scale2 is not None:
            assert test.conf.scale_type is not None
            scale2_ctype = test.conf.scale_type.ctype()
            self.header_writer.generate_vector_files(
                "scale2",
                _type=scale2_ctype,
                size=test.scale2.numel(),
                init=test.scale2.ravel(),
            )

        # ------------------------------------------------------------------
        # Render bias2 (for GEMM post-polyapprox)
        # ------------------------------------------------------------------
        if test.bias2 is not None:
            assert test.conf.bias_type is not None
            bs2_ctype = test.conf.bias_type.ctype()
            self.header_writer.generate_vector_files(
                "bias2", 
                _type=bs2_ctype, 
                size=test.bias2.numel(), 
                init=test.bias2.ravel()
            )

        global_shift = 0 if test.global_shift is None else int(test.global_shift.item())
        global_shift2 = 0 if test.global_shift2 is None else int(test.global_shift2.item())

        # ------------------------------------------------------------------
        # Render piecewise polynomial approximation parameters
        # ------------------------------------------------------------------
        if test.ppoly_model is not None:
            # Pack parameters only when generating headers
            ppoly_data = test.ppoly_model.get_packed_parameters(
                output_bits=10,
                mul_bw=test.conf.polyapprox_coeffs_mul_bitwidth,
                add_bw=test.conf.polyapprox_coeffs_add_bitwidth
            )
            self.header_writer.generate_vector_files(
                "ppolyapprox",
                _type="uint32_t",
                size=ppoly_data.size,
                init=ppoly_data
            )
        elif test.ppoly_params is not None:
            ppoly_data = test.ppoly_params.numpy()
            self.header_writer.generate_vector_files(
                "ppolyapprox",
                _type="uint32_t",
                size=ppoly_data.size,
                init=ppoly_data
            )

        # ------------------------------------------------------------------
        # Layer configuration header
        # ------------------------------------------------------------------
        self.header_writer.generate_defines_header(
            "layer_conf",
            {
                "test_name": test_name,
                "input": {
                    "height": in_height,
                    "width": in_width,
                    "channel": in_channel,
                    "signed": in_signed,
                    "bits": test.conf.in_type._bits,
                },
                "output": {
                    "height": out_height,
                    "width": out_width,
                    "channel": out_channel,
                    "bits": test.conf.out_type._bits,
                },
                "weight": {
                    "height": weight_ks_h,
                    "width": weight_ks_w,
                    "channel_in": weight_in_ch,
                    "channel_out": weight_out_ch,
                    "bits": weight_bits,
                    "offset": weight_offset,
                },
                "scale": {
                    "bits": test.conf.scale_type._bits
                    if test.conf.scale_type is not None
                    else 0
                },
                "bias": {
                    "bits": test.conf.bias_type._bits
                    if test.conf.bias_type is not None
                    else 0
                },
                "scale2": {
                    "bits": test.conf.scale_type._bits
                    if test.conf.scale_type is not None
                    else 0
                },
                "bias2": {
                    "bits": test.conf.bias_type._bits
                    if test.conf.bias_type is not None
                    else 0
                },
                "padding": {
                    "top": test.conf.padding.top,
                    "bottom": test.conf.padding.bottom,
                    "left": test.conf.padding.left,
                    "right": test.conf.padding.right,
                    "value": 0,
                },
                "stride": test.conf.stride.model_dump(),
                "groups": test.conf.in_channel if test.conf.depthwise else 1,
                "outshift": global_shift,
                "outshift2": global_shift2,
                "has_norm_quant": test.conf.has_norm_quant,
                "has_bias": test.conf.has_bias,
                "has_relu": test.conf.has_relu,
                "is_gemm": test.conf.is_gemm,
                "polyapprox_degree": test.conf.polyapprox_degree,
                "polyapprox": {
                    "nr_parts": test.conf.polyapprox_segments,
                    "bounds_bitwidth": test.conf.polyapprox_bounds_bitwidth,
                    "coeffs_mul_bitwidth": test.conf.polyapprox_coeffs_mul_bitwidth,
                    "coeffs_add_bitwidth": test.conf.polyapprox_coeffs_add_bitwidth,
                    "params_size": (
                        len(test.ppoly_model.get_packed_parameters(10, test.conf.polyapprox_coeffs_mul_bitwidth, test.conf.polyapprox_coeffs_add_bitwidth)) 
                        if test.ppoly_model is not None 
                        else (test.ppoly_params.numel() if test.ppoly_params is not None else 0)
                    ),
                }
            },
        )