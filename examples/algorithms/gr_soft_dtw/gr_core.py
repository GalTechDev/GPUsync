import torch

def compute_cost_matrix(X_v, Y_v):
    """
    Computes the squared Euclidean distance matrix between two 1D tensors.
    X_v: (N,)
    Y_v: (M,)
    Returns C: (N, M)
    """
    return (X_v.unsqueeze(1) - Y_v.unsqueeze(0)) ** 2

def soft_dtw_forward(C, gamma):
    """
    Computes the Soft-DTW distance using DP.
    Uses a list of lists to store tensors, avoiding dictionary lookups
    while preserving double differentiation compatibility.
    """
    N, M = C.shape
    R = [[C.new_tensor(float('inf')) for _ in range(M + 1)] for _ in range(N + 1)]
    R[0][0] = C.new_tensor(0.0)
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            a = R[i-1][j-1]
            b = R[i-1][j]
            c = R[i][j-1]
            
            # Handle inf explicitly to avoid nan in logsumexp
            if torch.isinf(a) and torch.isinf(b) and torch.isinf(c):
                sm = C.new_tensor(float('inf'))
            else:
                stacked = torch.stack([-a/gamma, -b/gamma, -c/gamma])
                sm = -gamma * torch.logsumexp(stacked, dim=0)
                
            R[i][j] = C[i-1, j-1] + sm
            
    return R[N][M]

def get_alignment_matrix_connected(C, gamma):
    """
    Computes A directly from the connected graph C.
    """
    loss = soft_dtw_forward(C, gamma)
    A = torch.autograd.grad(loss, C, create_graph=True)[0]
    return loss, A

def shift_matrix_cols(A, delta):
    """
    Shifts matrix A by delta along columns (axis 1).
    Equivalent to A * P_delta
    """
    N, M = A.shape
    A_shifted = torch.zeros_like(A)
    if delta > 0:
        if delta < M:
            A_shifted[:, delta:] = A[:, :-delta]
    elif delta < 0:
        d = -delta
        if d < M:
            A_shifted[:, :-d] = A[:, d:]
    else:
        A_shifted = A
    return A_shifted

def gr_soft_dtw_loss(X, Y, gamma, edges, lambd):
    """
    Computes the Graph-Regularized Soft-DTW loss.
    X: (D, N)
    Y: (D, M)
    edges: list of tuples (u, v, delta) representing the graph and shifts.
    lambd: regularization strength.
    """
    D = X.shape[0]
    
    total_soft_dtw = 0.0
    A_matrices = []
    
    for v in range(D):
        C_v = compute_cost_matrix(X[v], Y[v])
        loss_v, A_v = get_alignment_matrix_connected(C_v, gamma)
        total_soft_dtw = total_soft_dtw + loss_v
        A_matrices.append(A_v)
        
    reg_loss = 0.0
    for u, v, delta in edges:
        A_u = A_matrices[u]
        A_v = A_matrices[v]
        A_v_shifted = shift_matrix_cols(A_v, delta)
        
        diff = A_u - A_v_shifted
        
        # Crop to evaluate Frobenius norm only on overlapping valid regions
        if delta > 0:
            diff_valid = diff[:, delta:]
        elif delta < 0:
            d = -delta
            diff_valid = diff[:, :-d]
        else:
            diff_valid = diff
            
        reg_loss = reg_loss + torch.sum(diff_valid ** 2)
        
    return total_soft_dtw + lambd * reg_loss
