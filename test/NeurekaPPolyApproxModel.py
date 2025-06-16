import numpy as np
import torch
from typing import Tuple


class NeurekaPiecewisePolyApproxModel:
    """Piecewise linear approximation of GELU activation function."""
    
    def __init__(self, num_segments: int = 16, input_range: Tuple[float, float] = (-4.0, 4.0)):
        self.num_segments = num_segments
        self.input_range = input_range
        self.boundaries, self.slopes, self.intercepts = self._generate_piecewise_linear_gelu()
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation: x * Φ(x)"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def _generate_piecewise_linear_gelu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate linear approximation coefficients for each segment."""
        # Evenly spaced segment boundaries
        boundaries = np.linspace(self.input_range[0], self.input_range[1], self.num_segments + 1)
        slopes = np.zeros(self.num_segments)
        intercepts = np.zeros(self.num_segments)
        
        # Fit linear approximation for each segment
        for i in range(self.num_segments):
            x = np.linspace(boundaries[i], boundaries[i + 1], 100)
            y = self._gelu(x)
            # Linear least squares: y = slope * x + intercept
            A = np.vstack([x, np.ones(len(x))]).T
            slopes[i], intercepts[i] = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return boundaries, slopes, intercepts
    
    def apply_piecewise_linear(self, x: torch.Tensor, input_scale: float = 1.0, output_scale: float = 1.0) -> torch.Tensor:
        """
        Apply piecewise linear approximation to input tensor.
        
        Args:
            x: Input tensor (quantized values)
            input_scale: Scale to convert input to float range
            output_scale: Scale to convert output back to quantized range
        """
        output = torch.zeros_like(x, dtype=torch.float32)
        x_float = x.float() / input_scale
        
        # Apply linear transformation for each segment
        for i in range(self.num_segments):
            if i == 0:
                mask = x_float <= self.boundaries[i + 1]
            elif i == self.num_segments - 1:
                mask = x_float > self.boundaries[i]
            else:
                mask = (x_float > self.boundaries[i]) & (x_float <= self.boundaries[i + 1])
            
            output[mask] = self.slopes[i] * x_float[mask] + self.intercepts[i]
        
        return (output * output_scale).round().long()
    
    def get_quantized_coefficients(self, slope_scale: float, intercept_scale: float, 
                                   scale_bits: int = 3, bias_bits: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize slopes and intercepts with specified scaling factors.
        
        Args:
            slope_scale: Scaling factor for slopes
            intercept_scale: Scaling factor for intercepts
            scale_bits: Bit width for slope storage (for overflow checking)
            bias_bits: Bit width for intercept storage (for overflow checking)
        """
        quantized_slopes = np.round(self.slopes * slope_scale).astype(np.int32)
        quantized_intercepts = np.round(self.intercepts * intercept_scale).astype(np.int32)
        
        # Check for overflow
        slope_max = 2**(scale_bits - 1) - 1
        slope_min = -2**(scale_bits - 1)
        intercept_max = 2**(bias_bits - 1) - 1
        intercept_min = -2**(bias_bits - 1)
        
        # Warn if values exceed bit width
        if np.any(quantized_slopes > slope_max) or np.any(quantized_slopes < slope_min):
            overflow_indices = np.where((quantized_slopes > slope_max) | (quantized_slopes < slope_min))[0]
            print(f"WARNING: Slope overflow detected in segments {overflow_indices} for {scale_bits}-bit storage")
            print(f"         Max slope value: {np.max(np.abs(quantized_slopes))}, allowed range: [{slope_min}, {slope_max}]")
        
        if np.any(quantized_intercepts > intercept_max) or np.any(quantized_intercepts < intercept_min):
            overflow_indices = np.where((quantized_intercepts > intercept_max) | (quantized_intercepts < intercept_min))[0]
            print(f"WARNING: Intercept overflow detected in segments {overflow_indices} for {bias_bits}-bit storage")
            print(f"         Max intercept value: {np.max(np.abs(quantized_intercepts))}, allowed range: [{intercept_min}, {intercept_max}]")
        
        return quantized_slopes, quantized_intercepts
    
    def get_quantized_boundaries(self, boundary_scale: float, boundary_bits: int = 8) -> np.ndarray:
        """
        Quantize segment boundaries with specified scaling factor.
        
        Args:
            boundary_scale: Scaling factor for boundaries
            boundary_bits: Bit width for boundary storage (for overflow checking)
        """
        # Normalize boundaries to [-1, 1] then scale
        min_val, max_val = self.input_range
        normalized = (self.boundaries - min_val) / (max_val - min_val) * 2 - 1
        quantized_boundaries = np.round(normalized * boundary_scale).astype(np.int32)
        
        # Check for overflow
        boundary_max = 2**(boundary_bits - 1) - 1
        boundary_min = -2**(boundary_bits - 1)
        
        if np.any(quantized_boundaries > boundary_max) or np.any(quantized_boundaries < boundary_min):
            print(f"WARNING: Boundary overflow detected for {boundary_bits}-bit storage")
            print(f"         Max boundary value: {np.max(np.abs(quantized_boundaries))}, allowed range: [{boundary_min}, {boundary_max}]")
        
        return quantized_boundaries


def main():
    """Test piecewise GELU approximation with custom scaling."""
    import matplotlib.pyplot as plt
    
    # Initialize model
    model = NeurekaPiecewisePolyApproxModel(num_segments=16, input_range=(-4.0, 4.0))
    
    # Test range: 8-bit signed integers
    x_int = np.arange(-128, 128)
    x_float = x_int * (4.0 / 128)  # Map to [-4, 4]
    
    # Compute GELU and approximation
    y_true = model._gelu(x_float)

    # Test quantization with different scaling factors
    print("GELU Piecewise Linear Approximation (16 segments)")
    print("=" * 50)
    
    # Try different scaling factors
    slope_scale = 2**6  # Custom scaling factor for slopes
    intercept_scale = 2**8  # Custom scaling factor for intercepts
    
    print(f"\nUsing scaling factors: slope_scale={slope_scale}, intercept_scale={intercept_scale}")
    quant_slopes, quant_intercepts = model.get_quantized_coefficients(
        slope_scale, intercept_scale, scale_bits=8, bias_bits=16
    )
    
    print("\nQuantized coefficients:")
    for i in range(model.num_segments):
        print(f"Seg {i:2d}: slope={quant_slopes[i]:4d}, intercept={quant_intercepts[i]:6d}")
    
    # Plot comparison using quantized coefficients
    y_quant = np.zeros_like(x_float)
    for i in range(model.num_segments):
        if i == 0:
            mask = x_float <= model.boundaries[i + 1]
        elif i == model.num_segments - 1:
            mask = x_float > model.boundaries[i]
        else:
            mask = (x_float > model.boundaries[i]) & (x_float <= model.boundaries[i + 1])
        y_quant[mask] = (quant_slopes[i] / slope_scale).round() * x_float[mask] \
                       + (quant_intercepts[i] / intercept_scale).round()

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x_int, y_true, 'b-', label='True GELU', linewidth=2)
    plt.plot(x_int, y_quant, 'r--', label='Quantized Approx', linewidth=2)
    plt.xlabel('Input (8-bit range)')
    plt.ylabel('Output')
    plt.title('GELU vs Piecewise Linear Approximation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot error **for quantized approximation**
    plt.subplot(2, 1, 2)
    error = y_true - y_quant
    plt.plot(x_int, error, 'g-')
    plt.xlabel('Input (8-bit range)')
    plt.ylabel('Error')
    plt.title(f'Approximation Error (Max: {np.max(np.abs(error)):.4f}, '
              f'RMS: {np.sqrt(np.mean(error**2)):.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()