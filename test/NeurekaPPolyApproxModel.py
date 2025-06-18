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
                 input_quantization: int = 32):
        """
        Initialize piecewise polynomial approximation model.
        
        Args:
            target_func: Function to approximate
            num_segments: Number of piecewise segments
            input_range: Float range for input values
            input_quantization: Quantization factor for input (x_float = x_int / input_quantization)
        """
        self.target_func = target_func
        self.num_segments = num_segments
        self.input_range = input_range
        self.input_quantization = input_quantization
        self.boundaries, self.slopes, self.intercepts = self._fit_piecewise_linear()
    
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
    
    def get_packed_parameters(self,
                             output_bits: int = 40,
                             mul_bw: int = 8,
                             add_bw: int = 32,
                             boundary_bits: int = 8,
                             max_nr_parts: int = 16) -> np.ndarray:
        """
        Get parameters packed in the format expected by neural engine.
        
        Format:
        - First boundary_total_bits: up to 32 8-bit boundaries
        - Then num_segments * (mul_bw + add_bw) bits: slopes and intercepts
        
        Args:
            output_bits: Output scale as 2^output_bits
            mul_bw: Bit width for multiply coefficients (slopes)
            add_bw: Bit width for add coefficients (intercepts)
            boundary_bits: Bits per boundary (if boundary_total_bits not specified)
            boundary_total_bits: Total bits for all boundaries (e.g., 256 for Neureka-TX mem bandwidth)
            
        Returns:
            Packed parameters as uint32 array
        """
        # Get quantized values
        boundaries_q, slopes_q, intercepts_q = self.get_quantized_lut(
            output_bits=output_bits,
            slope_bits=mul_bw,
            intercept_bits=add_bw
        )
        
        # Pack slopes and intercepts together
        total_coeff_bits = self.num_segments * (mul_bw + add_bw)
        boundary_total_bits = (max_nr_parts + 1) * boundary_bits if boundary_bits else 0
        total_bits = boundary_total_bits + total_coeff_bits
        
        # Calculate number of uint32s needed
        num_uint32 = (total_bits + 31) // 32
        packed = np.zeros(num_uint32, dtype=np.uint32)
        
        # Pack boundaries first
        bit_pos = 0
        for i, bound in enumerate(boundaries_q):
            if bit_pos < boundary_total_bits:  # Only pack if within allocated space
                bound_val = int(bound) & ((1 << boundary_bits) - 1)
                bits_to_write = min(boundary_bits, boundary_total_bits - bit_pos)
                while bits_to_write > 0:
                    uint32_idx = bit_pos // 32
                    bit_offset = bit_pos % 32
                    bits_available = 32 - bit_offset
                    bits_this_round = min(bits_to_write, bits_available)

                    mask = (1 << bits_this_round) - 1
                    packed[uint32_idx] |= ((bound_val & mask) << bit_offset)

                    bound_val >>= bits_this_round
                    bits_to_write -= bits_this_round
                    bit_pos += bits_this_round
                    
        # Skip to coefficient section if needed
        bit_pos = boundary_total_bits

        # Pack coefficients (slope + intercept for each segment)
        for i in range(self.num_segments):
            # Pack slope first, then intercept (matching unpacking order)
            slope_val = int(slopes_q[i]) & ((1 << mul_bw) - 1)  # Mask to ensure proper bit width
            intercept_val = int(intercepts_q[i]) & ((1 << add_bw) - 1)  # Mask to ensure proper bit width
            
            # Pack slope (mul_bw bits)
            bits_to_write = mul_bw
            val_to_pack = slope_val
            while bits_to_write > 0:
                uint32_idx = bit_pos // 32
                bit_offset = bit_pos % 32
                bits_available = 32 - bit_offset
                bits_this_round = min(bits_to_write, bits_available)

                mask = (1 << bits_this_round) - 1
                packed[uint32_idx] |= ((val_to_pack & mask) << bit_offset)

                val_to_pack >>= bits_this_round
                bits_to_write -= bits_this_round
                bit_pos += bits_this_round
            
            # Pack intercept (add_bw bits)
            bits_to_write = add_bw
            val_to_pack = intercept_val
            while bits_to_write > 0:
                uint32_idx = bit_pos // 32
                bit_offset = bit_pos % 32
                bits_available = 32 - bit_offset
                bits_this_round = min(bits_to_write, bits_available)

                mask = (1 << bits_this_round) - 1
                packed[uint32_idx] |= ((val_to_pack & mask) << bit_offset)

                val_to_pack >>= bits_this_round
                bits_to_write -= bits_this_round
                bit_pos += bits_this_round
                
        return packed
    
    def get_quantized_lut(self, 
                         output_bits: int = 24,
                         slope_bits: int = 8,
                         intercept_bits: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate quantized lookup table for integer arithmetic.
        
        Args:
            output_bits: Output scale as 2^output_bits
            slope_bits: Bit width for slope coefficients
            intercept_bits: Bit width for intercept coefficients
            
        Returns:
            Quantized boundaries, slopes, and intercepts
        """
        output_scale = 2**output_bits
        
        # For y = slope * x + intercept with x_float = x_int / input_quantization:
        # y_scaled = slope * x_int / input_quantization * output_scale + intercept * output_scale
        # y_int = (slope * output_scale / input_quantization) * x_int + intercept * output_scale
        
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
            print(f"         Consider using output_bits < {output_bits} or more slope bits")
        
        if np.any(intercepts_q > intercept_max) or np.any(intercepts_q < intercept_min):
            overflow_indices = np.where((intercepts_q > intercept_max) | (intercepts_q < intercept_min))[0]
            print(f"WARNING: Intercept overflow in segments {overflow_indices} for {intercept_bits}-bit storage")
            print(f"         Max magnitude: {np.max(np.abs(intercepts_q))}, allowed range: [{intercept_min}, {intercept_max}]")
        
        return boundaries_q, slopes_q, intercepts_q
    
    def apply_quantized(self, x_int: np.ndarray, boundaries: np.ndarray, 
                       slopes: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
        """
        Apply quantized piecewise linear approximation.
        
        Args:
            x_int: Integer input values
            boundaries: Quantized segment boundaries
            slopes: Quantized slopes
            intercepts: Quantized intercepts
            
        Returns:
            Integer output values
        """
        y = np.zeros_like(x_int, dtype=np.int32)
        
        for i in range(len(slopes)):
            if i == 0:
                mask = x_int < boundaries[i + 1]
            elif i == len(slopes) - 1:
                mask = x_int >= boundaries[i]
            else:
                mask = (x_int >= boundaries[i]) & (x_int < boundaries[i + 1])
            
            y[mask] = slopes[i] * x_int[mask] + intercepts[i]
        
        return y
    
    def apply_torch(self, x: torch.Tensor, input_scale: float = 1.0, 
                   output_scale: float = 1.0) -> torch.Tensor:
        """
        Apply piecewise linear approximation to PyTorch tensor.
        
        Args:
            x: Input tensor
            input_scale: Scale to convert input to float range
            output_scale: Scale to convert output back
            
        Returns:
            Output tensor
        """
        output = torch.zeros_like(x, dtype=torch.float32)
        x_float = x.float() / input_scale
        
        for i in range(self.num_segments):
            if i == 0:
                mask = x_float <= self.boundaries[i + 1]
            elif i == self.num_segments - 1:
                mask = x_float > self.boundaries[i]
            else:
                mask = (x_float > self.boundaries[i]) & (x_float <= self.boundaries[i + 1])
            
            output[mask] = self.slopes[i] * x_float[mask] + self.intercepts[i]
        
        return (output * output_scale).round().long()
    
    def print_lut(self, boundaries: np.ndarray, slopes: np.ndarray, 
                  intercepts: np.ndarray) -> None:
        """Print the lookup table in a readable format."""
        print("\nLookup Table:")
        print("Seg | Slope | Intercept | X Range")
        print("-" * 40)
        for i in range(len(slopes)):
            x_start = boundaries[i]
            x_end = boundaries[i+1] if i < len(slopes)-1 else boundaries[-1]
            print(f"{i:3d} | {slopes[i]:5d} | {intercepts[i]:9d} | [{x_start:4d}, {x_end:4d}]")
    
    def evaluate_approximation(self, x_int: np.ndarray, y_int: np.ndarray, 
                             output_scale: int, plot: bool = True) -> dict:
        """
        Evaluate approximation quality and optionally plot results.
        
        Args:
            x_int: Integer input values
            y_int: Integer output values from approximation
            output_scale: Output scaling factor
            plot: Whether to generate plots
            
        Returns:
            Dictionary with error statistics
        """
        # Reference values
        x_float = x_int / self.input_quantization
        y_true = self.target_func(x_float)
        y_approx = y_int / output_scale
        
        # Calculate errors
        error = y_true - y_approx
        max_error = np.max(np.abs(error))
        rms_error = np.sqrt(np.mean(error**2))
        
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Comparison plot
            ax1.plot(x_int, y_true, 'b-', label='True Function', linewidth=2)
            ax1.plot(x_int, y_approx, 'r--', label='Piecewise Approx', linewidth=2)
            ax1.set_xlabel('Input (integer)')
            ax1.set_ylabel('Output (normalized)')
            ax1.set_title(f'Piecewise Linear Approximation ({self.num_segments} segments)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Mark segment boundaries
            boundaries_int = np.round(self.boundaries * self.input_quantization).astype(int)
            for b in boundaries_int[1:-1]:
                ax1.axvline(x=b, color='gray', linestyle=':', alpha=0.5)
            
            # Error plot
            ax2.plot(x_int, error, 'g-', linewidth=1.5)
            ax2.set_xlabel('Input (integer)')
            ax2.set_ylabel('Error')
            ax2.set_title(f'Approximation Error (Max: {max_error:.6f}, RMS: {rms_error:.6f})')
            ax2.grid(True, alpha=0.3)
            
            # Mark segment boundaries
            for b in boundaries_int[1:-1]:
                ax2.axvline(x=b, color='gray', linestyle=':', alpha=0.5)
            
            plt.tight_layout()
            plt.show()
        
        return {
            'max_error': max_error,
            'rms_error': rms_error,
            'output_range': (np.min(y_int), np.max(y_int))
        }
    
    def apply_approximation(self, tensor, verbose: bool = False):
        """
        Apply piecewise polynomial approximation directly using unpacked parameters.
        
        Args:
            tensor: Input tensor (torch.Tensor, int32)
            verbose: Print intermediate results
        
        Returns:
            Approximated tensor (torch.Tensor, int32)
        """
        import torch
        import numpy as np
        
        if verbose:
            print("PIECEWISE POLY APPROX PARAMETERS:")
            print(f"  Boundaries: {self.boundaries}")
            print(f"  Slopes: {self.slopes}")
            print(f"  Intercepts: {self.intercepts}")
        
        # Apply piecewise linear approximation
        output = torch.zeros_like(tensor, dtype=torch.int32)
        tensor_np = tensor.numpy()
        
        for i in range(self.num_segments):
            if i == 0:
                mask = tensor_np <= self.boundaries[i + 1]
            elif i == self.num_segments - 1:
                mask = tensor_np > self.boundaries[i]
            else:
                mask = (tensor_np > self.boundaries[i]) & (tensor_np <= self.boundaries[i + 1])
            
            # Apply linear transformation: y = slope * x + intercept
            output[mask] = self.slopes[i] * tensor[mask] + self.intercepts[i]
        
        return output


def demo_256bit_boundaries():
    """Demonstrate usage with 256-bit boundary packing."""
    # Define GELU function
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    print("256-bit Boundary Packing Demo")
    print("=" * 50)
    
    # Create model
    model = PiecewisePolyApproxModel(
        target_func=gelu,
        num_segments=16,
        input_range=(-4.0, 4.0),
        input_quantization=32
    )
    
    # Get packed parameters with 256 bits for boundaries
    output_bits = 10
    mul_bw = 8
    add_bw = 16
    
    ppoly_bounds_coeffs = model.get_packed_parameters(
        output_bits=output_bits,
        mul_bw=mul_bw,
        add_bw=add_bw,
        boundary_total_bits=256  # Fixed 256 bits for all boundaries
    )
    
    print(f"\nPacked parameters with 256-bit boundaries:")
    print(f"  Number of segments: {model.num_segments}")
    print(f"  Number of boundaries: {model.num_segments + 1}")
    print(f"  Bits per boundary: {256 // (model.num_segments + 1)} ({256} total bits)")
    print(f"  Coefficient bits: {model.num_segments} * ({mul_bw} + {add_bw}) = {model.num_segments * (mul_bw + add_bw)} bits")
    
    total_bits = 256 + model.num_segments * (mul_bw + add_bw)
    print(f"  Total bits: {total_bits}")
    print(f"  Packed into {len(ppoly_bounds_coeffs)} uint32 values")
    
    print(f"\nPacked data (hex):")
    for i, val in enumerate(ppoly_bounds_coeffs):
        print(f"  ppoly_bounds_coeffs[{i:2d}] = 0x{val:08x}")
    
    # Also get the unpacked values for reference
    boundaries, slopes, intercepts = model.get_quantized_lut(
        output_bits=output_bits,
        slope_bits=mul_bw,
        intercept_bits=add_bw
    )
    
    print(f"\nUnpacked values for reference:")
    model.print_lut(boundaries, slopes, intercepts)
    
    return ppoly_bounds_coeffs


def apply_packed_approximation(
    tensor,
    ppoly_params,
    num_segments: int,
    mul_bw: int = 8,
    add_bw: int = 16,
    boundary_total_bits: int = None,
    boundary_bits: int = 8,
    boundary_packing_info: dict = None,
    verbose: bool = False
):
    """
    Apply piecewise polynomial approximation using packed parameters.
    
    Args:
        tensor: Input tensor (torch.Tensor, int32)
        ppoly_params: Packed parameters (torch.Tensor)
        num_segments: Number of piecewise segments
        mul_bw: Bit width for multiply coefficients
        add_bw: Bit width for add coefficients
        boundary_total_bits: Total bits for boundaries (if None, use boundary_bits per boundary)
        boundary_bits: Bits per boundary (if boundary_total_bits not specified)
        boundary_packing_info: Dictionary with packing info (min_bound, max_bound, etc.)
        verbose: Print intermediate results
    
    Returns:
        Approximated tensor (torch.Tensor, int32)
    """
    import torch
    import numpy as np
    
    # Unpack parameters from ppoly_params
    ppoly_np = ppoly_params.numpy().astype(np.uint32)
    
    # Determine boundary unpacking parameters
    num_boundaries = num_segments + 1
    if boundary_total_bits is not None:
        boundary_bits = boundary_total_bits // num_boundaries
        total_boundary_bits = boundary_total_bits
    else:
        boundary_bits = boundary_bits
        total_boundary_bits = num_boundaries * boundary_bits
    
    # Extract boundaries
    boundaries = np.zeros(num_boundaries, dtype=np.int32)
    
    bit_pos = 0
    for i in range(num_boundaries):
        if bit_pos < total_boundary_bits:
            bits_to_read = min(boundary_bits, total_boundary_bits - bit_pos)
            boundary_val = 0
            bits_read = 0
            
            while bits_read < bits_to_read:
                uint32_idx = bit_pos // 32
                bit_offset = bit_pos % 32
                bits_available = min(32 - bit_offset, bits_to_read - bits_read)
                
                mask = (1 << bits_available) - 1
                extracted = (ppoly_np[uint32_idx] >> bit_offset) & mask
                boundary_val |= extracted << bits_read
                
                bits_read += bits_available
                bit_pos += bits_available
            
            boundaries[i] = boundary_val
    
    # Map boundaries back to input range with proper 8-bit signed clamping
    if boundary_packing_info:
        max_val = boundary_packing_info['max_val']
        min_bound = boundary_packing_info['min_bound']
        max_bound = boundary_packing_info['max_bound']
        boundaries = (boundaries / max_val * (max_bound - min_bound) + min_bound).astype(np.int32)
    else:
        # Default mapping with proper 8-bit signed range [-128, 127]
        max_val = (1 << boundary_bits) - 1
        boundaries = (boundaries / max_val * 255 - 127.5).astype(np.int32)
        # Clamp to valid 8-bit signed range
        boundaries = np.clip(boundaries, -128, 127)
    
    # Skip to coefficient section
    bit_pos = total_boundary_bits
    
    # Extract slopes and intercepts
    slopes = np.zeros(num_segments, dtype=np.int32)
    intercepts = np.zeros(num_segments, dtype=np.int32)
    
    for i in range(num_segments):
        # Extract slope (mul_bw bits)
        bits_read = 0
        slope_val = 0
        while bits_read < mul_bw:
            uint32_idx = bit_pos // 32
            bit_offset = bit_pos % 32
            bits_available = min(32 - bit_offset, mul_bw - bits_read)
            
            mask = (1 << bits_available) - 1
            extracted = (ppoly_np[uint32_idx] >> bit_offset) & mask
            slope_val |= extracted << bits_read
            
            bits_read += bits_available
            bit_pos += bits_available
        
        # Sign extend if necessary
        if slope_val & (1 << (mul_bw - 1)):
            slope_val |= ~((1 << mul_bw) - 1)
        slopes[i] = slope_val
        
        # Extract intercept (add_bw bits)
        bits_read = 0
        intercept_val = 0
        while bits_read < add_bw:
            uint32_idx = bit_pos // 32
            bit_offset = bit_pos % 32
            bits_available = min(32 - bit_offset, add_bw - bits_read)
            
            mask = (1 << bits_available) - 1
            extracted = (ppoly_np[uint32_idx] >> bit_offset) & mask
            intercept_val |= extracted << bits_read
            
            bits_read += bits_available
            bit_pos += bits_available
        
        # Sign extend if necessary
        if intercept_val & (1 << (add_bw - 1)):
            intercept_val |= ~((1 << add_bw) - 1)
        intercepts[i] = intercept_val
    
    if verbose:
        print("PIECEWISE POLY APPROX PARAMETERS:")
        print(f"  Boundaries: {boundaries}")
        print(f"  Slopes: {slopes}")
        print(f"  Intercepts: {intercepts}")
    
    # Apply piecewise linear approximation
    output = torch.zeros_like(tensor, dtype=torch.int32)
    tensor_np = tensor.numpy()
    
    for i in range(num_segments):
        if i == 0:
            mask = tensor_np <= boundaries[i + 1]
        elif i == num_segments - 1:
            mask = tensor_np > boundaries[i]
        else:
            mask = (tensor_np > boundaries[i]) & (tensor_np <= boundaries[i + 1])
        
        # Apply linear transformation: y = slope * x + intercept
        output[mask] = slopes[i] * tensor[mask] + intercepts[i]
    
    return output

# Add it as a static method to the class for compatibility
PiecewisePolyApproxModel.apply_packed_approximation = staticmethod(apply_packed_approximation)

if __name__ == "__main__":
    # Original demos
    # demo_gelu()
    # demo_custom_function()
    
    # New demo for 256-bit boundary packing
    ppoly_params = demo_256bit_boundaries()