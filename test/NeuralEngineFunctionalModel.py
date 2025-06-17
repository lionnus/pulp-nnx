from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from TestClasses import IntegerType, Padding, Stride
from NeurekaPPolyApproxModel import PiecewisePolyApproxModel

class NeuralEngineFunctionalModel:
    ACCUMULATOR_TYPE = IntegerType(name="int32")

    @staticmethod
    def _tensor_to_hex(tensor):
        int_tensor = np.asarray(torch.floor(tensor).to(torch.int64))
        int_tensor[int_tensor < 0] = 0xffffffff + (int_tensor[int_tensor < 0]+1)
        hex_tensor = np.empty(int_tensor.shape, dtype=object)
        for idx in np.ndindex(int_tensor.shape):
            hex_tensor[idx] = hex(int_tensor[idx].item())
        return hex_tensor

    @staticmethod
    def _cast(
        tensor: torch.Tensor, _type: IntegerType, saturate: bool = False
    ) -> torch.Tensor:
        if saturate:
            return tensor.clamp(_type.min, _type.max)
        else:
            return tensor & ((1 << _type._bits) - 1)

    def _norm_quant(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        global_shift: torch.Tensor,
        out_type: IntegerType,
        bias_type: Optional[IntegerType],
        has_bias: bool,
        has_relu: bool,
        verbose: bool,
    ) -> torch.Tensor:
        # Scale accumulators are in 48bit, so keeping the data in 64bit
        tensor = tensor * scale
        assert tensor.dtype == torch.int64

        if verbose:
            print("INTERMEDIATE RESULTS (after scale):")
            current_threshold = np.get_printoptions()['threshold']
            np.set_printoptions(threshold=np.inf)
            print(NeuralEngineFunctionalModel._tensor_to_hex(tensor))
            np.set_printoptions(threshold=current_threshold)

        if has_bias:
            assert bias is not None
            assert bias_type is not None

            tensor = NeuralEngineFunctionalModel._cast(
                tensor, bias_type, saturate=False
            ).type(torch.int32)

            tensor = tensor + bias

            tensor = NeuralEngineFunctionalModel._cast(
                tensor, bias_type, saturate=True
            ).type(torch.int32)

            if verbose:
                print("INTERMEDIATE RESULTS (after bias):")
                current_threshold = np.get_printoptions()['threshold']
                np.set_printoptions(threshold=np.inf)
                print(NeuralEngineFunctionalModel._tensor_to_hex(tensor))
                np.set_printoptions(threshold=current_threshold)

        if has_relu:
            tensor = F.relu(tensor)

        # Global shift
        tensor = tensor >> global_shift

        if verbose:
            print("INTERMEDIATE RESULTS (after shift):")
            current_threshold = np.get_printoptions()['threshold']
            np.set_printoptions(threshold=np.inf)
            print(NeuralEngineFunctionalModel._tensor_to_hex(tensor))
            np.set_printoptions(threshold=current_threshold)

        # Saturate into out_type
        tensor = NeuralEngineFunctionalModel._cast(tensor, out_type, saturate=True)

        return tensor
    
    def _apply_piecewise_poly_approx(
        self, 
        tensor: torch.Tensor,
        ppoly_model: PiecewisePolyApproxModel,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Apply piecewise polynomial approximation using the model directly.
        
        Args:
            tensor: Input tensor (int32)
            ppoly_model: PiecewisePolyApproxModel
            verbose: Print intermediate results
        
        Returns:
            Approximated tensor (int32)
        """
        # Check if quantized parameters are available, otherwise use float parameters
        if hasattr(ppoly_model, 'boundaries_q') and hasattr(ppoly_model, 'slopes_q') and hasattr(ppoly_model, 'intercepts_q'):
            # Use quantized parameters for integer arithmetic
            boundaries = ppoly_model.boundaries_q
            slopes = ppoly_model.slopes_q
            intercepts = ppoly_model.intercepts_q
        else:
            # Fallback: convert float parameters to integers on-the-fly
            boundaries = (ppoly_model.boundaries * ppoly_model.input_quantization).round().astype(int)
            slopes = (ppoly_model.slopes * 1024).round().astype(int)  # Use a reasonable scale
            intercepts = (ppoly_model.intercepts * 1024).round().astype(int)  # Use a reasonable scale
        
        if verbose:
            print("PIECEWISE POLY APPROX PARAMETERS:")
            print(f"  Boundaries: {boundaries}")
            print(f"  Slopes: {slopes}")
            print(f"  Intercepts: {intercepts}")
        
        # Apply piecewise linear approximation with integer arithmetic
        output = torch.zeros_like(tensor, dtype=torch.int32)
        tensor_np = tensor.numpy()
        
        for i in range(len(slopes)):
            if i == 0:
                mask = tensor_np <= boundaries[i + 1]
            elif i == len(slopes) - 1:
                mask = tensor_np > boundaries[i]
            else:
                mask = (tensor_np > boundaries[i]) & (tensor_np <= boundaries[i + 1])
            
            # Apply linear transformation: y = slope * x + intercept (all integers)
            output[mask] = int(slopes[i]) * tensor[mask] + int(intercepts[i])
        
        if verbose:
            print("INTERMEDIATE RESULTS (after piecewise poly approx):")
            current_threshold = np.get_printoptions()["threshold"]
            np.set_printoptions(threshold=np.inf)
            print(NeuralEngineFunctionalModel._tensor_to_hex(output))
            np.set_printoptions(threshold=current_threshold)
        
        return output


    def convolution(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        global_shift: Optional[torch.Tensor],
        padding: Padding,
        stride: Stride,
        depthwise: bool,
        out_type: IntegerType,
        bias_type: Optional[IntegerType],
        has_norm_quant: bool,
        has_bias: bool,
        has_relu: bool,
        verbose: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        _ = kwargs

        input_padded = F.pad(
            input,
            (
                padding.left,
                padding.right,
                padding.top,
                padding.bottom,
            ),
            "constant",
            0,
        )

        if verbose:
            print("INPUTS (padded):")
            current_threshold = np.get_printoptions()['threshold']
            np.set_printoptions(threshold=np.inf)
            print(NeuralEngineFunctionalModel._tensor_to_hex(input_padded))
            print("WEIGHTS (padded):")
            print(NeuralEngineFunctionalModel._tensor_to_hex(weight))
            np.set_printoptions(threshold=current_threshold)

        # Accumulators are 32bit non-saturating.
        # Calculate in higher precision (int64)
        output = F.conv2d(
            input=input_padded,
            weight=weight,
            stride=(stride.height, stride.width),
            groups=weight.shape[0] if depthwise else 1,
        ).type(torch.int64)

        # Cast to accumulator type
        output = NeuralEngineFunctionalModel._cast(
            output, NeuralEngineFunctionalModel.ACCUMULATOR_TYPE, saturate=False
        ).type(torch.int32)

        if verbose:
            print("INTERMEDIATE RESULTS (pre-normalization/requant):")
            current_threshold = np.get_printoptions()['threshold']
            np.set_printoptions(threshold=np.inf)
            print(NeuralEngineFunctionalModel._tensor_to_hex(output))
            np.set_printoptions(threshold=current_threshold)

        if has_norm_quant:
            assert scale is not None
            assert global_shift is not None
            output = self._norm_quant(
                output,
                scale,
                bias,
                global_shift,
                out_type,
                bias_type,
                has_bias,
                has_relu,
                verbose,
            )

        return output

    def gemm(
        self,
        b_matrix: torch.Tensor, # activations
        a_matrix: torch.Tensor, # weights
        scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        global_shift: Optional[torch.Tensor],
        norm1_out_type: IntegerType,
        norm2_out_type: IntegerType,
        bias_type: Optional[IntegerType],
        has_norm_quant: bool,
        has_bias: bool,
        has_relu: bool,
        scale2: Optional[torch.Tensor] = None,
        bias2: Optional[torch.Tensor] = None,   
        global_shift2: Optional[torch.Tensor] = None,
        is_gemm: bool = True,
        polyapprox_degree: int = 0,
        polyapprox_segments: int = 16,
        ppoly_params: Optional[torch.Tensor] = None,  # Keep for backward compatibility
        ppoly_model: Optional[PiecewisePolyApproxModel] = None,  # Preferred way
        verbose: bool = False,
        skip_polyapprox_degree: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        _ = kwargs
        """
        Performs A @ B, then optionally applies per-channel quantization:
         - scale (int64 accum -> int64 scaled)
         - bias (int32), should not be used for GEMM
         - ReLU, should not be used for GEMM
         - global_shift (right-shift)
         - saturation to out_type

        Args mirror convolution's quant options but for GEMM.
        """
        if verbose:
            print("INPUTS (B matrix):")
            current_threshold = np.get_printoptions()['threshold']
            np.set_printoptions(threshold=np.inf)
            print(NeuralEngineFunctionalModel._tensor_to_hex(b_matrix))
            # Print again with folding, so +2^(bits-1) is added to negative numbers
            b_matrix_folded = b_matrix + (1 << (8 - 1))
            print("INPUTS (B matrix, with folding):")
            print(NeuralEngineFunctionalModel._tensor_to_hex(b_matrix_folded))
            print("WEIGHTS (A matrix):")
            print(NeuralEngineFunctionalModel._tensor_to_hex(a_matrix))
            # Print again with folding, so +2^(bits-1) is added to negative numbers
            a_matrix_folded = a_matrix + (1 << (8 - 1))
            print("WEIGHTS (A matrix, with folding):")
            print(NeuralEngineFunctionalModel._tensor_to_hex(a_matrix_folded))
            np.set_printoptions(threshold=current_threshold)

        # matrix multiply in 64bit
        output = torch.matmul(a_matrix, b_matrix).type(torch.int64)

        # cast into the 32-bit accumulator
        output = NeuralEngineFunctionalModel._cast(
            output, NeuralEngineFunctionalModel.ACCUMULATOR_TYPE, saturate=False
        ).type(torch.int32)

        if verbose:
            print("INTERMEDIATE RESULTS (A*B, pre-normalization/requant):")
            curr = np.get_printoptions()['threshold']
            np.set_printoptions(threshold=np.inf)
            print(self._tensor_to_hex(output))
            np.set_printoptions(threshold=curr)

        # First normalization + requant (always done if has_norm_quant)
        if has_norm_quant:
            assert scale is not None and global_shift is not None
            output = self._norm_quant(
                output,
                scale,
                bias,
                global_shift,
                norm1_out_type,
                bias_type,
                has_bias,
                has_relu,
                verbose,
            )
            
        # Polynomial approximation (only if not skipped and polyapprox_degree > 0)
        if polyapprox_degree > 0 and not skip_polyapprox_degree:
            assert polyapprox_degree in [1, 2], "polyapprox_degree must be 1 (linear) or 2 (quadratic)"
            
            if ppoly_model is not None:
                # Use the model directly - much cleaner!
                output = self._apply_piecewise_poly_approx(
                    output, 
                    ppoly_model,
                    verbose=verbose
                )
            elif ppoly_params is not None:
                # Fallback to packed parameters for backward compatibility
                # Get bitwidths from kwargs if available, otherwise use defaults
                mul_bw = kwargs.get('polyapprox_coeffs_mul_bitwidth', 8)
                add_bw = kwargs.get('polyapprox_coeffs_add_bitwidth', 16)
                output = self._apply_piecewise_poly_approx_packed(
                    output, 
                    ppoly_params,
                    polyapprox_segments,
                    mul_bw=mul_bw,
                    add_bw=add_bw,
                    verbose=verbose
                )
            else:
                raise ValueError("Either ppoly_model or ppoly_params must be provided for polynomial approximation.")
        
            # Second normalization + requant after polynomial approximation
            if has_norm_quant:
                # Use separate scale/bias/shift parameters if provided for post-polyapprox
                post_scale = scale2 if scale2 is not None else scale
                post_bias = bias2 if bias2 is not None else bias
                post_shift = global_shift2 if global_shift2 is not None else global_shift
                
                output = self._norm_quant(
                    output,
                    post_scale,
                    post_bias,
                    post_shift,
                    norm2_out_type,
                    bias_type,
                    has_bias,
                    has_relu,
                    verbose,
                )

        return output

    def _apply_piecewise_poly_approx_packed(
        self, 
        tensor: torch.Tensor,
        ppoly_params: torch.Tensor,
        num_segments: int,
        mul_bw: int = 8,
        add_bw: int = 16,
        verbose: bool = False
    ) -> torch.Tensor:
        """Backward compatibility method for packed parameters."""
        output = PiecewisePolyApproxModel.apply_packed_approximation(
            tensor, ppoly_params, num_segments, mul_bw, add_bw, verbose
        )
        
        if verbose:
            print("INTERMEDIATE RESULTS (after piecewise poly approx):")
            current_threshold = np.get_printoptions()["threshold"]
            np.set_printoptions(threshold=np.inf)
            print(NeuralEngineFunctionalModel._tensor_to_hex(output))
            np.set_printoptions(threshold=current_threshold)
        
        return output