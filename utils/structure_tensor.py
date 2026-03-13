import torch
import torch.nn.functional as F
import math

def gaussian_kernel_2d(sigma: float, kernel_size: int = None, device=None, dtype=torch.float32):
    """
    Creates a 2D Gaussian kernel.
    """
    if kernel_size is None:
        kernel_size = int(math.ceil(3 * sigma)) * 2 + 1
    
    # Create a 1D Gaussian vector
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device, dtype=dtype)
    gauss1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss1d = gauss1d / gauss1d.sum()
    
    # Outer product to get 2D kernel
    gauss2d = gauss1d.view(-1, 1) @ gauss1d.view(1, -1)
    
    # Reshape for conv2d: (out_channels, in_channels, H, W)
    return gauss2d.view(1, 1, kernel_size, kernel_size)

def compute_gradients(image: torch.Tensor, sigma: float = 1.0):
    """
    Computes Gaussian smoothed gradients Ix, Iy.
    image shape: (B, C, H, W)
    """
    B, C, H, W = image.shape
    device = image.device
    dtype = image.dtype
    
    # Apply Gaussian smoothing if sigma > 0
    if sigma > 0:
        kernel = gaussian_kernel_2d(sigma, device=device, dtype=dtype)
        kernel = kernel.repeat(C, 1, 1, 1) # Support C channels via groups=C
        # Padding
        pad = kernel.size(-1) // 2
        smoothed = F.conv2d(image, kernel, padding=pad, groups=C)
    else:
        smoothed = image
        
    # Compute gradients using central differences (Sobel approx or plain differences)
    # Using simple finite differences here
    kx = torch.tensor([[-0.5, 0.0, 0.5]], device=device, dtype=dtype).view(1, 1, 1, 3)
    ky = torch.tensor([[-0.5], [0.0], [0.5]], device=device, dtype=dtype).view(1, 1, 3, 1)
    
    kx = kx.repeat(C, 1, 1, 1)
    ky = ky.repeat(C, 1, 1, 1)
    
    Ix = F.conv2d(smoothed, kx, padding=(0, 1), groups=C)
    Iy = F.conv2d(smoothed, ky, padding=(1, 0), groups=C)
    
    return Ix, Iy

def structure_tensor(image: torch.Tensor, sigma: float = 1.0, rho: float = 2.0):
    """
    Computes the structure tensor components for an image.
    J = G_rho * [Ix^2, IxIy; IxIy, Iy^2]
    """
    B, C, H, W = image.shape
    assert C == 1, "Currently implemented for 1 channel (grayscale) images"
    
    Ix, Iy = compute_gradients(image, sigma)
    
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    
    # Smooth the components with scale rho
    if rho > 0:
        kernel = gaussian_kernel_2d(rho, device=image.device, dtype=image.dtype)
        pad = kernel.size(-1) // 2
        J11 = F.conv2d(Ixx, kernel, padding=pad)
        J12 = F.conv2d(Ixy, kernel, padding=pad)
        J22 = F.conv2d(Iyy, kernel, padding=pad)
    else:
        J11, J12, J22 = Ixx, Ixy, Iyy
        
    return J11, J12, J22

def eigen_decomposition_2x2(J11: torch.Tensor, J12: torch.Tensor, J22: torch.Tensor):
    """
    Analytic eigen decomposition of a symmetric 2x2 matrix field.
    Returns:
    lambda1, lambda2: The eigenvalues (lambda1 >= lambda2)
    V1, V2: The eigenvectors corresponding to lambda1 and lambda2
            V1 = (v1x, v1y), V2 = (v2x, v2y)
    """
    trace = J11 + J22
    det = J11 * J22 - J12 * J12
    
    # Half the difference of the trace squared and determinant
    gap = torch.sqrt(torch.clamp((trace / 2)**2 - det, min=1e-12))
    
    lambda1 = trace / 2 + gap
    lambda2 = trace / 2 - gap
    
    # Eigenvectors:
    # (J11 - lambda)x + J12 y = 0  =>  [-J12, J11-lambda] or [J22-lambda, -J12]
    # For lambda1:
    v1x = J12
    v1y = lambda1 - J11
    
    # Norm handling to avoid division by zero
    norm1 = torch.sqrt(torch.clamp(v1x**2 + v1y**2, min=1e-12))
    v1x = v1x / norm1
    v1y = v1y / norm1
    
    # The matrix is symmetric, so v2 is orthogonal to v1
    v2x = -v1y
    v2y = v1x
    
    return lambda1, lambda2, (v1x, v1y), (v2x, v2y)

def compute_beltrami_metric(image: torch.Tensor, alpha: float = 2.0, sigma: float = 1.0, rho: float = 2.0):
    """
    Computes the Beltrami inverse metric tensor components g^ij for Beltrami flow.
    G_alpha = V * diag(1, 1 + alpha * (lambda1 - lambda2)) * V^T
    Returns the inverse metric G_alpha^{-1} = [gxx, gxy; gxy, gyy].
    """
    J11, J12, J22 = structure_tensor(image, sigma, rho)
    lambda1, lambda2, V1, V2 = eigen_decomposition_2x2(J11, J12, J22)
    
    # lambda_edge = lambda1 - lambda2
    lambda_edge = torch.relu(lambda1 - lambda2)
    
    # The metric eigenvalues (for G_alpha, NOT G_alpha^-1)
    # L1 represents direction across edges
    # L2 represents direction along edges
    L1 = 1.0 + alpha * lambda_edge
    L2 = torch.ones_like(L1) # 1.0
    
    # For the inverse metric G_alpha^-1, the eigenvalues are 1/L1 and 1/L2
    inv_L1 = 1.0 / torch.clamp(L1, min=1e-6)
    inv_L2 = 1.0 / L2 # Since L2 is 1.0, this is just 1.0
    
    v1x, v1y = V1
    v2x, v2y = V2
    
    # G_alpha^-1 = V * diag(1/L1, 1/L2) * V^T
    # gxx = v1x^2 * inv_L1 + v2x^2 * inv_L2
    # gyy = v1y^2 * inv_L1 + v2y^2 * inv_L2
    # gxy = v1x*v1y * inv_L1 + v2x*v2y * inv_L2
    
    gxx = v1x**2 * inv_L1 + v2x**2 * inv_L2
    gyy = v1y**2 * inv_L1 + v2y**2 * inv_L2
    gxy = v1x * v1y * inv_L1 + v2x * v2y * inv_L2
    
    return gxx, gxy, gyy, lambda_edge
