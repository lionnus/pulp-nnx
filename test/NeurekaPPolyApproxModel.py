import numpy as np
import torch
from typing import Tuple, Callable, Optional

class PiecewisePolyApproxModel:
    """
    General piecewise polynomial approximation model for arbitrary functions.
    
    This model is designed specifically for hardware accelerator compatibility.
    
    Hardware Accelerator Coefficient Format:
    - All coefficients are stored in [c2, c1, c0] order regardless of polynomial degree
    - For degree 1: evaluation uses (c2*x + c1), c0 is unused but still packed
    - For degree 2: evaluation uses (c2*x + c1)*x + c0
    - Bit packing always follows {c2, c1, c0} order in hardware memory layout
    """
    
    def __init__(self, 
                 target_func: Callable[[np.ndarray], np.ndarray],
                 num_segments: int = 16, 
                 input_range: Tuple[float, float] = (-4.0, 4.0),
                 input_quantization: int = 32,
                 degree: int = 2,
                 c2_bits: int = 8,
                 c1_bits: int = 24,
                 c0_bits: int = 16,
                 output_bits: int = 24,
                 max_nr_parts: int = 16) -> None:
        """
        Initialize piecewise polynomial approximation model.
        
        Args:
            target_func: Function to approximate
            num_segments: Number of piecewise segments
            input_range: Float range for input values
            input_quantization: Quantization factor for input (x_float = x_int / input_quantization)
            degree: Polynomial degree (1 for linear, 2 for quadratic)
            c2_bits: Bit width for quadratic coefficients (c2)
            c1_bits: Bit width for linear coefficients (c1)
            c0_bits: Bit width for constant coefficients (c0)
            output_bits: Output scale as 2^output_bits
        """
        self.target_func = target_func
        self.num_segments = num_segments
        self.input_range = input_range
        self.input_quantization = input_quantization
        self.degree = degree
        self.c2_bits = c2_bits
        self.c1_bits = c1_bits
        self.c0_bits = c0_bits
        self.output_bits = output_bits
        self.max_nr_parts = max_nr_parts
        
        # Fit the piecewise polynomial approximation
        self.boundaries, self.horner_coefficients = self._fit_piecewise_polynomial()
        
        # Pre-compute quantized parameters for use in forward pass
        self.boundaries_q, self.coefficients_q = self.get_params()
    
    def _fit_piecewise_polynomial(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fit polynomial approximation for each segment and convert to Horner form for hardware accelerator."""
        # Evenly spaced segment boundaries in float domain
        boundaries_float = np.linspace(self.input_range[0], self.input_range[1], self.num_segments + 1)
        
        # Hardware accelerator coefficient storage format:
        # - Always stores 3 coefficients per segment: [c2, c1, c0]
        # - For degree 1: uses c2 and c1 in evaluation ((c2*x + c1))
        # - For degree 2: uses c2, c1, c0 in evaluation ((c2*x + c1)*x + c0)
        horner_coefficients = np.zeros((self.num_segments, 3))  # Always 3 coefficients for HW
        
        # Fit polynomial approximation for each segment
        for i in range(self.num_segments):
            x = np.linspace(boundaries_float[i], boundaries_float[i + 1], 100)
            y = self.target_func(x)
            
            # np.polyfit returns coefficients for standard polynomial form:
            # For degree 2: y = p2*x^2 + p1*x + p0
            # Returns in descending order: [p2, p1, p0]
            poly_coeffs = np.polyfit(x, y, self.degree)
            
            # Convert to hardware accelerator format [c2, c1, c0]
            if self.degree == 2:
                p2, p1, p0 = poly_coeffs  # Unpack for clarity
                horner_coefficients[i] = [p2, p1, p0]  # c2, c1, c0
            elif self.degree == 1:
                p1, p0 = poly_coeffs
                # For degree 1, hardware uses ((c2*x + c1)), so:
                # c2 = p1 (linear coefficient), c1 = p0 (constant), c0 = 0 (unused)
                horner_coefficients[i] = [p1, p0, 0]  # c2, c1, c0
                
        return boundaries_float, horner_coefficients
    
    def apply(self, tensor: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Apply piecewise polynomial approximation to input tensor using Horner's method.
        
        Args:
            tensor: Input tensor (torch.Tensor, int32)
            verbose: Print intermediate results
            
        Returns:
            Approximated tensor (torch.Tensor, int32)
        """
        if verbose:
            print("PIECEWISE POLY APPROX PARAMETERS:")
            print(f"  Boundaries (quantized): {self.boundaries_q}")
            print(f"  Coefficients (quantized): {self.coefficients_q}")
        
        # Apply piecewise polynomial approximation
        output = torch.zeros_like(tensor, dtype=torch.int32)
        tensor_np = tensor.numpy()
        
        # Create tensors to store intermediate results for the whole input tensor
        if verbose:
            if self.degree == 1:
                step1_full = torch.zeros_like(tensor, dtype=torch.int32)  # c2*x
                step2_full = torch.zeros_like(tensor, dtype=torch.int32)  # c2*x + c1
            elif self.degree == 2:
                step1_full = torch.zeros_like(tensor, dtype=torch.int32)  # c2*x
                step2_full = torch.zeros_like(tensor, dtype=torch.int32)  # c2*x + c1
                step3_full = torch.zeros_like(tensor, dtype=torch.int32)  # (c2*x + c1)*x

        for i in range(self.num_segments):
            if i == 0:
                mask_np = tensor_np <= self.boundaries_q[i + 1]
            elif i == self.num_segments - 1:
                mask_np = tensor_np > self.boundaries_q[i]
            else:
                mask_np = (tensor_np > self.boundaries_q[i]) & (tensor_np <= self.boundaries_q[i + 1])
            
            if np.any(mask_np):
                # Convert numpy mask to torch tensor for indexing
                mask = torch.from_numpy(mask_np)
                x = tensor[mask]
                coeffs = self.coefficients_q[i]
                
                # Apply hardware accelerator Horner's method with coefficients [c2, c1, c0]
                if self.degree == 1:
                    # Linear: hardware uses (c2*x + c1)
                    if verbose:
                        # First multiplication: c2*x
                        step1 = coeffs[0] * x
                        step1_full[mask] = step1
                        
                        # First addition: c2*x + c1
                        result = step1 + coeffs[1]
                        step2_full[mask] = result
                    else:
                        # Normal computation without verbose output
                        result = coeffs[0] * x + coeffs[1]  # c2*x + c1
                elif self.degree == 2:
                    # Quadratic: hardware uses (c2*x + c1)*x + c0
                    if verbose:
                        # First multiplication: c2*x
                        step1 = coeffs[0] * x
                        step1_full[mask] = step1
                        
                        # First addition: c2*x + c1
                        step2 = step1 + coeffs[1]
                        step2_full[mask] = step2
                        
                        # Second multiplication: (c2*x + c1)*x
                        step3 = step2 * x
                        step3_full[mask] = step3
                        
                        # Second addition: (c2*x + c1)*x + c0
                        result = step3 + coeffs[2]
                    else:
                        # Normal computation without verbose output
                        result = (coeffs[0] * x + coeffs[1]) * x + coeffs[2]  # (c2*x + c1)*x + c0
                    
                output[mask] = result
        
        if verbose:
            from NeuralEngineFunctionalModel import NeuralEngineFunctionalModel
            current_threshold = np.get_printoptions()['threshold']
            np.set_printoptions(threshold=np.inf)

            print("INTERMEDIATE RESULTS (c2*x, first mult):")
            print(NeuralEngineFunctionalModel._tensor_to_hex(step1_full))
            
            print("INTERMEDIATE RESULTS (c2*x + c1, first add):")
            print(NeuralEngineFunctionalModel._tensor_to_hex(step2_full))

            if self.degree == 2:
                print("INTERMEDIATE RESULTS ((c2*x + c1)*x, second mult):")
                print(NeuralEngineFunctionalModel._tensor_to_hex(step3_full))

                print("INTERMEDIATE RESULTS ((c2*x + c1)*x + c0, second add):")
                print(NeuralEngineFunctionalModel._tensor_to_hex(output))
            else:
                print("INTERMEDIATE RESULTS (final result):")
                print(NeuralEngineFunctionalModel._tensor_to_hex(output))

            np.set_printoptions(threshold=current_threshold)

        return output
    
    def get_packed_parameters(self,
                             output_bits: Optional[int] = None,
                             c2_bits: Optional[int] = None,
                             c1_bits: Optional[int] = None,
                             c0_bits: Optional[int] = None,
                             boundary_bits: int = 8) -> np.ndarray:
        """
        Get parameters packed in the format expected by neural engine.
        
        Hardware accelerator bit packing format:
        - First boundary_total_bits: up to max_nr_parts+1 boundary values
        - Then num_segments * (c2_bits + c1_bits + c0_bits) bits: polynomial coefficients
        - Coefficients are always packed as {c2, c1, c0} regardless of polynomial degree
        - For degree 1: uses c2 and c1 (c0 is set to 0 but still packed)
        - For degree 2: uses c2, c1, and c0
        
        Returns:
            Packed parameters as uint32 array
        """
        # Get max number of parts, fixed in HW
        max_nr_parts = self.max_nr_parts
        # Use instance defaults if not provided
        output_bits = output_bits or self.output_bits
        c2_bits = c2_bits or self.c2_bits
        c1_bits = c1_bits or self.c1_bits
        c0_bits = c0_bits or self.c0_bits
            
        # Get quantized values
        boundaries_q, coefficients_q = self.get_params()
        
        # Calculate total coefficient bits per segment
        coeff_bits_per_segment = c2_bits + c1_bits + c0_bits
        
        total_coeff_bits = self.num_segments * coeff_bits_per_segment
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
        
        # Pack coefficients for each segment
        for i in range(self.num_segments):
            coeffs = coefficients_q[i]
            # Hardware accelerator always expects coefficients packed as {c2, c1, c0}
            # coeffs array is stored as [c2, c1, c0] format
            bit_pos = pack_bits(coeffs[0], c2_bits, bit_pos, packed)  # c2
            bit_pos = pack_bits(coeffs[1], c1_bits, bit_pos, packed)  # c1
            bit_pos = pack_bits(coeffs[2], c0_bits, bit_pos, packed)  # c0
        
        return packed
    
    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate quantized lookup table for integer arithmetic.
        
        Converts floating-point Horner coefficients to quantized integer coefficients
        that account for the input quantization scale.
        
        Hardware accelerator coefficient format: [c2, c1, c0] for all segments
        
        Returns:
            Quantized boundaries and coefficients in [c2, c1, c0] format
        """
        output_bits = self.output_bits
        c2_bits = self.c2_bits
        c1_bits = self.c1_bits
        c0_bits = self.c0_bits
            
        output_scale = 2**output_bits
        
        # Quantize boundaries
        boundaries_q = np.round(self.boundaries * self.input_quantization).astype(np.int32)
        # Clamp boundaries to valid 8-bit signed range [-128, 127]
        boundaries_q = np.clip(boundaries_q, -128, 127)
        
        # Quantize coefficients - horner_coefficients is already in [c2, c1, c0] format
        coefficients_q = np.zeros_like(self.horner_coefficients, dtype=np.int32)
        for i in range(self.num_segments):
            # Process c2 coefficient (index 0)
            if self.degree >= 1:  # c2 exists for both degree 1 and 2
                # c2 multiplies x_q (for degree 1) or x_q^2 (for degree 2)
                if self.degree == 1:
                    # For degree 1, c2 is the linear coefficient, scale by output_scale / input_quantization
                    coeff_q = np.round(self.horner_coefficients[i, 0] * output_scale / self.input_quantization).astype(np.int32)
                else:  # degree == 2
                    # For degree 2, c2 is the quadratic coefficient, scale by output_scale / input_quantization^2
                    coeff_q = np.round(self.horner_coefficients[i, 0] * output_scale / (self.input_quantization**2)).astype(np.int32)
                coeff_max = 2**(c2_bits - 1) - 1
                coeff_min = -2**(c2_bits - 1)
                if coeff_q > coeff_max or coeff_q < coeff_min:
                    print(f"WARNING: c2 overflow in segment {i} for {c2_bits}-bit storage")
                    print(f"         Value: {coeff_q}, allowed range: [{coeff_min}, {coeff_max}]")
                    print(f"         Clipped to allowed range")
                coefficients_q[i, 0] = np.clip(coeff_q, coeff_min, coeff_max)
            
            # Process c1 coefficient (index 1)
            if self.degree >= 1:  # c1 exists for both degree 1 and 2
                if self.degree == 1:
                    # For degree 1, c1 is the constant term, scale by output_scale only
                    coeff_q = np.round(self.horner_coefficients[i, 1] * output_scale).astype(np.int32)
                else:  # degree == 2
                    # For degree 2, c1 multiplies x_q, scale by output_scale / input_quantization
                    coeff_q = np.round(self.horner_coefficients[i, 1] * output_scale / self.input_quantization).astype(np.int32)
                coeff_max = 2**(c1_bits - 1) - 1
                coeff_min = -2**(c1_bits - 1)
                if coeff_q > coeff_max or coeff_q < coeff_min:
                    print(f"WARNING: c1 overflow in segment {i} for {c1_bits}-bit storage")
                    print(f"         Value: {coeff_q}, allowed range: [{coeff_min}, {coeff_max}]")
                    print(f"         Clipped to allowed range")
                coefficients_q[i, 1] = np.clip(coeff_q, coeff_min, coeff_max)
            
            # Process c0 coefficient (index 2)
            if self.degree == 2:  # c0 only exists for degree 2
                # c0 is the constant term, scale by output_scale only
                coeff_q = np.round(self.horner_coefficients[i, 2] * output_scale).astype(np.int32)
                coeff_max = 2**(c0_bits - 1) - 1
                coeff_min = -2**(c0_bits - 1)
                if coeff_q > coeff_max or coeff_q < coeff_min:
                    print(f"WARNING: c0 overflow in segment {i} for {c0_bits}-bit storage")
                    print(f"         Value: {coeff_q}, allowed range: [{coeff_min}, {coeff_max}]")
                    print(f"         Clipped to allowed range")
                coefficients_q[i, 2] = np.clip(coeff_q, coeff_min, coeff_max)
            else:
                # For degree 1, c0 is unused (set to 0)
                coefficients_q[i, 2] = 0

        return boundaries_q, coefficients_q

    def print_lut(self) -> None:
        """Print the lookup table in decimal and hexadecimal format."""
        # Use quantized values
        boundaries = self.boundaries_q
        coefficients = self.coefficients_q

        print("\nLookup Table (Decimal) - Hardware accelerator format [c2, c1, c0]:")
        if self.degree == 1:
            print("Seg |   c2   |   c1   | X Range")
            print("-" * 40)
            for i in range(len(coefficients)):
                x_start = boundaries[i]
                x_end = boundaries[i+1] if i < len(coefficients)-1 else boundaries[-1]
                print(f"{i:3d} | {coefficients[i, 0]:6d} | {coefficients[i, 1]:6d} | [{x_start:4d}, {x_end:4d}]")
        elif self.degree == 2:
            print("Seg | c2 |   c1   |    c0    | X Range")
            print("-" * 50)
            for i in range(len(coefficients)):
                x_start = boundaries[i]
                x_end = boundaries[i+1] if i < len(coefficients)-1 else boundaries[-1]
                print(f"{i:3d} | {coefficients[i, 0]:2d} | {coefficients[i, 1]:6d} | {coefficients[i, 2]:8d} | [{x_start:4d}, {x_end:4d}]")

        print("\nLookup Table (Hexadecimal) - Hardware accelerator format [c2, c1, c0]:")
        if self.degree == 1:
            print("Seg |   c2   |   c1   | X Range")
            print("-" * 40)
            for i in range(len(coefficients)):
                x_start = boundaries[i]
                x_end = boundaries[i+1] if i < len(coefficients)-1 else boundaries[-1]
                c2_val = coefficients[i, 0] & ((1 << self.c2_bits) - 1)  # Mask to bit width
                c1_val = coefficients[i, 1] & ((1 << self.c1_bits) - 1)  # Mask to bit width
                print(f"{i:3d} | 0x{c2_val:0{(self.c2_bits + 3) // 4}x} | 0x{c1_val:0{(self.c1_bits + 3) // 4}x} | [{x_start:4d}, {x_end:4d}]")
        elif self.degree == 2:
            print("Seg | c2  |   c1   |    c0    | X Range")
            print("-" * 55)
            for i in range(len(coefficients)):
                x_start = boundaries[i]
                x_end = boundaries[i+1] if i < len(coefficients)-1 else boundaries[-1]
                c2_val = coefficients[i, 0] & ((1 << self.c2_bits) - 1)  # Mask to bit width
                c1_val = coefficients[i, 1] & ((1 << self.c1_bits) - 1)  # Mask to bit width
                c0_val = coefficients[i, 2] & ((1 << self.c0_bits) - 1)  # Mask to bit width
                print(f"{i:3d} | 0x{c2_val:0{(self.c2_bits + 3) // 4}x} | 0x{c1_val:0{(self.c1_bits + 3) // 4}x} | 0x{c0_val:0{(self.c0_bits + 3) // 4}x} | [{x_start:4d}, {x_end:4d}]")