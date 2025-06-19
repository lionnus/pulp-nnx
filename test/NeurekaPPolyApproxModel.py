import numpy as np
import torch
from typing import Tuple, Callable, Optional
import matplotlib.pyplot as plt

class PiecewisePolyApproxModel:
    """General piecewise polynomial approximation model for arbitrary functions."""
    
    def __init__(self, 
                 target_func: Callable[[np.ndarray], np.ndarray],
                 num_segments: int = 16, 
                 input_range: Tuple[float, float] = (-4.0, 4.0),
                 input_quantization: int = 32,
                 slope_bits: int = 8,
                 intercept_bits: int = 32,
                 output_bits: int = 24):
        """
        Initialize piecewise polynomial approximation model.
        
        Args:
            target_func: Function to approximate
            num_segments: Number of piecewise segments
            input_range: Float range for input values
            input_quantization: Quantization factor for input (x_float = x_int / input_quantization)
            slope_bits: Bit width for slope coefficients
            intercept_bits: Bit width for intercept coefficients
            output_bits: Output scale as 2^output_bits
        """
        self.target_func = target_func
        self.num_segments = num_segments
        self.input_range = input_range
        self.input_quantization = input_quantization
        self.slope_bits = slope_bits
        self.intercept_bits = intercept_bits
        self.output_bits = output_bits
        
        # Fit the piecewise linear approximation
        self.boundaries, self.slopes, self.intercepts = self._fit_piecewise_linear()
        
        # Pre-compute quantized parameters for use in forward pass
        self.boundaries_q, self.slopes_q, self.intercepts_q = self.get_params()
    
    def _fit_piecewise_linear(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit linear approximation for each segment."""
        # Evenly spaced segment boundaries in float domain
        boundaries_float = np.linspace(self.input_range[0], self.input_range[1], self.num_segments + 1)
        slopes = np.zeros(self.num_segments)
        intercepts = np.zeros(self.num_segments)
        
        # Fit linear approximation for each segment
        for i in range(self.num_segments):
            x = np.linspace(boundaries_float[i], boundaries_float[i + 1], 100)
            y = self.target_func(x)
            # Linear least squares: y = slope * x + intercept
            A = np.vstack([x, np.ones(len(x))]).T
            slopes[i], intercepts[i] = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return boundaries_float, slopes, intercepts
    
    def apply(self, tensor: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Apply piecewise linear approximation to input tensor.
        
        Args:
            tensor: Input tensor (torch.Tensor, int32)
            verbose: Print intermediate results
            
        Returns:
            Approximated tensor (torch.Tensor, int32)
        """
        if verbose:
            print("PIECEWISE POLY APPROX PARAMETERS:")
            print(f"  Boundaries (quantized): {self.boundaries_q}")
            print(f"  Slopes (quantized): {self.slopes_q}")
            print(f"  Intercepts (quantized): {self.intercepts_q}")
        
        # Apply piecewise linear approximation
        output = torch.zeros_like(tensor, dtype=torch.int32)
        tensor_np = tensor.numpy()
        
        for i in range(self.num_segments):
            if i == 0:
                mask = tensor_np <= self.boundaries_q[i + 1]
            elif i == self.num_segments - 1:
                mask = tensor_np > self.boundaries_q[i]
            else:
                mask = (tensor_np > self.boundaries_q[i]) & (tensor_np <= self.boundaries_q[i + 1])
            
            # Apply linear transformation: y = slope * x + intercept
            output[mask] = self.slopes_q[i] * tensor[mask] + self.intercepts_q[i]
        
        return output
    
    def get_packed_parameters(self,
                             output_bits: Optional[int] = None,
                             mul_bw: Optional[int] = None,
                             add_bw: Optional[int] = None,
                             boundary_bits: int = 8,
                             max_nr_parts: int = 16) -> np.ndarray:
        """
        Get parameters packed in the format expected by neural engine.
        
        Format:
        - First boundary_total_bits: up to max_nr_parts+1 boundary values
        - Then num_segments * (mul_bw + add_bw) bits: slopes and intercepts
        Returns:
            Packed parameters as uint32 array
        """
        # Use instance defaults if not provided
        output_bits = self.output_bits
        mul_bw = self.slope_bits
        add_bw = self.intercept_bits
            
        # Get quantized values
        boundaries_q, slopes_q, intercepts_q = self.get_params()
        
        # Pack slopes and intercepts together
        total_coeff_bits = self.num_segments * (mul_bw + add_bw)
        boundary_total_bits = (max_nr_parts + 1) * boundary_bits
        total_bits = boundary_total_bits + total_coeff_bits
        
        # Calculate number of uint32s needed
        num_uint32 = (total_bits + 31) // 32
        packed = np.zeros(num_uint32, dtype=np.uint32)
        
        # Generic bit packing function
        def pack_bits(value, num_bits, bit_pos, packed_array):
            bits_to_write = num_bits
            val_to_pack = int(value) & ((1 << num_bits) - 1)
            
            while bits_to_write > 0:
                uint32_idx = bit_pos // 32
                bit_offset = bit_pos % 32
                bits_this_round = min(bits_to_write, 32 - bit_offset)
                
                mask = (1 << bits_this_round) - 1
                packed_array[uint32_idx] |= ((val_to_pack & mask) << bit_offset)
                
                val_to_pack >>= bits_this_round
                bits_to_write -= bits_this_round
                bit_pos += bits_this_round
            
            return bit_pos
        
        # Pack boundaries
        bit_pos = 0
        for bound in boundaries_q:
            if bit_pos < boundary_total_bits:
                bit_pos = pack_bits(bound, boundary_bits, bit_pos, packed)
        
        # Skip to coefficient section
        bit_pos = boundary_total_bits
        
        # Pack coefficients (slope + intercept for each segment)
        for i in range(self.num_segments):
            bit_pos = pack_bits(slopes_q[i], mul_bw, bit_pos, packed)
            bit_pos = pack_bits(intercepts_q[i], add_bw, bit_pos, packed)
        
        return packed
    
    def get_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate quantized lookup table for integer arithmetic.
        
        Returns:
            Quantized boundaries, slopes, and intercepts
        """
        output_bits = self.output_bits
        slope_bits = self.slope_bits
        intercept_bits = self.intercept_bits
            
        output_scale = 2**output_bits
        
        # Quantize parameters
        slopes_q = np.round(self.slopes * output_scale / self.input_quantization).astype(np.int32)
        intercepts_q = np.round(self.intercepts * output_scale).astype(np.int32)
        boundaries_q = np.round(self.boundaries * self.input_quantization).astype(np.int32)
        
        # Clamp boundaries to valid 8-bit signed range [-128, 127]
        boundaries_q = np.clip(boundaries_q, -128, 127)
        
        # Check bit constraints
        slope_max = 2**(slope_bits - 1) - 1
        slope_min = -2**(slope_bits - 1)
        intercept_max = 2**(intercept_bits - 1) - 1
        intercept_min = -2**(intercept_bits - 1)
        
        # Warn if values exceed bit width
        if np.any(slopes_q > slope_max) or np.any(slopes_q < slope_min):
            overflow_indices = np.where((slopes_q > slope_max) | (slopes_q < slope_min))[0]
            print(f"WARNING: Slope overflow in segments {overflow_indices} for {slope_bits}-bit storage")
            print(f"         Max magnitude: {np.max(np.abs(slopes_q))}, allowed range: [{slope_min}, {slope_max}]")
            print(f"         Clipped to max value of {slope_max} and min value of {slope_min}")
        slopes_q = np.clip(slopes_q, slope_min, slope_max)
        
        if np.any(intercepts_q > intercept_max) or np.any(intercepts_q < intercept_min):
            overflow_indices = np.where((intercepts_q > intercept_max) | (intercepts_q < intercept_min))[0]
            print(f"WARNING: Intercept overflow in segments {overflow_indices} for {intercept_bits}-bit storage")
            print(f"         Max magnitude: {np.max(np.abs(intercepts_q))}, allowed range: [{intercept_min}, {intercept_max}]")
            print(f"         Clipped to max value of {intercept_max} and min value of {intercept_min}")
        intercepts_q = np.clip(intercepts_q, intercept_min, intercept_max)

        return boundaries_q, slopes_q, intercepts_q

    def print_lut(self) -> None:
        """Print the lookup table in a readable format."""
        # Use quantized values if not provided
        boundaries = self.boundaries_q
        slopes = self.slopes_q
        intercepts = self.intercepts_q

        print("\nLookup Table:")
        print("Seg | Slope | Intercept | X Range")
        print("-" * 40)
        for i in range(len(slopes)):
            x_start = boundaries[i]
            x_end = boundaries[i+1] if i < len(slopes)-1 else boundaries[-1]
            print(f"{i:3d} | {slopes[i]:5d} | {intercepts[i]:9d} | [{x_start:4d}, {x_end:4d}]")