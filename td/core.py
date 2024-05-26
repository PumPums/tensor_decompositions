import numpy as np


def unfold(tensor, dim):
    """Unfold tensor along the dim, dimension of the tensor size must be greater than 2

    Args:
        tensor: ndarray
        dim: int

    Returns:
        2darray
    """
    assert len(tensor.shape) > 2, "dimension of input tensor must be greater than 1"
    assert dim < len(
        tensor.shape
    ), "input dim must be less than dimension of input tensor"
    return np.reshape(np.moveaxis(tensor, dim, 0), (tensor.shape[dim], -1))


def fold(unfolded_tensor, dim, new_shape):
    """Fold tensor along the dim, dimension of the tensor must be equal 2

    Args:
        tensor: ndarray
        dim: int

    Returns:
        4darray
    """
    assert (
        len(unfolded_tensor.shape) == 2
    ), "dimension of input unfolded tensor must be equal 2"
    new_shape = list(new_shape)
    mode_dim = new_shape.pop(dim)
    new_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(unfolded_tensor, new_shape), 0, dim)


def dims_dot(tensor, factors, dims, ranks, skip=None, transpose=True):
    """Tensor dot product through some dimensions"""
    shape = list(tensor.shape)
    for i, dim in enumerate(dims):
        if skip is not None and i == skip:
            continue
        if transpose:
            step_tensor = np.dot(factors[i].T, unfold(tensor, dim))
        else:
            step_tensor = np.dot(factors[i], unfold(tensor, dim))
        shape[dim] = ranks[i]
        tensor = fold(step_tensor, dim, shape)
    return tensor


def initialize_tucker(tensor, dims, ranks):
    """Initialize core for partial_tucker"""
    factors = []
    for i, dim in enumerate(dims):
        u, _, _ = np.linalg.svd(unfold(tensor, dim))
        factors.append(u[:, : ranks[i]])
    core = dims_dot(tensor, factors, dims, ranks)
    return core, factors


def core_approx(
    tensor, init_core, init_factors, dims, ranks, n_iters=100, tol=1e-4, verbose=False
):
    """Core approximation for equality of norms"""
    core, factors = init_core.copy(), init_factors.copy()
    errs = []
    tensor_norm = np.linalg.norm(tensor)
    for itr in range(n_iters):
        for i, dim in enumerate(dims):
            core_approx_ = dims_dot(tensor, factors, dims, ranks, skip=i)
            u, _, _ = np.linalg.svd(unfold(core_approx_, dim))
            factors[i] = u[:, : ranks[i]]
        core = dims_dot(tensor, factors, dims=dims, ranks=ranks)
        errs.append(
            np.sqrt(np.abs(tensor_norm**2 - np.linalg.norm(core) ** 2)) / tensor_norm
        )
        if itr > 0:
            curr_err = np.abs(errs[-2] - errs[-1])
            if verbose:
                print(f"Iter {itr}: err={errs[-1]}, variation error={curr_err}")
            if tol and curr_err < tol:
                if verbose:
                    print(f"Finished in {itr} iterations")
                break
    return core, factors


def partial_tucker(tensor, dims, ranks, n_iters=100, tol=1e-4, verbose=False):
    """Decompose 4d tensor to 3 small tensors

    Args:
        tensor: ndarray conv2d weight has shape of (ch_out, ch_in, ker_h, ker_w)
        dims: dims of decompose # TODO check dims instead [0, 1]
        ranks: ranks of intermediate shape
        n_iters: amount of iterations for core approximation
        tol: err for early stop of iterations

    Returns:
        3 tensors (core, [last, first])

    Example:
        (ch_out, ch_in, ker_h, ker_w) to (ch_out, ranks[0], 1, 1) +
        (ranks[0], ranks[1], ker_h, ker_w) + (rank[1], ch_in, 1, 1)
    """
    assert len(dims) == len(ranks), "len(dims) != len(ranks)"
    assert len(tensor.shape) == 4, "tensor dim is not equal 4"
    assert (
        tensor.shape[0] % 16 == 0 and tensor.shape[1] % 16 == 0
    ), "dim of in/output tensor doesnt div by 16"
    shape = list(tensor.shape)
    for dim, rank in zip(dims, ranks):
        if rank > shape[dim]:
            raise Exception(
                f"New rank {rank} is more than exist rank {shape[dim]} (dim: {dim}"
            )

    core, factors = initialize_tucker(tensor, dims=dims, ranks=ranks)
    core, factors = core_approx(
        tensor,
        init_core=core,
        init_factors=factors,
        dims=dims,
        ranks=ranks,
        n_iters=n_iters,
        tol=tol,
        verbose=verbose,
    )
    return core, factors


def tucker_stick(tensor, dim=1, rank=None):
    """Decompose 4d tensor to 2 small tensors

    Args:
        tensor: ndarray conv2d weight has shape of (ch_out, ch_in, ker_h, ker_w)
        dim: dim of decompose <#TODO other dims instead of 1>
        rank: rank of the intermediate dim

    Returns:
        list of tensors ([first, last])

    Example:
        (ch_out, ch_in, ker_h, ker_w) to (ch_out, rank, ker_h, 1) + (rank, ch_in, 1, ker_w)
    """
    assert len(tensor.shape) == 4, "dim of input tensor must be equal 4"
    assert (
        tensor.shape[0] % 16 == 0 and tensor.shape[1] % 16 == 0
    ), "dim of in/output tensor doesnt div by 16"
    init_shape = list(tensor.shape)
    if init_shape[1] > init_shape[0]:
        tensor = tensor.transpose(1, 0, 2, 3)

    if rank is None:
        rank = tensor.shape[0] // 2

    sh = list(tensor.shape)
    tensor = tensor.reshape(sh[0], sh[1] * sh[2], sh[3])
    sh = list(tensor.shape)
    u, _, _ = np.linalg.svd(unfold(tensor, dim))
    u = u[:, :rank]
    core = dims_dot(tensor, factors=[u], dims=[dim], ranks=[rank])

    res = []
    if init_shape[1] > init_shape[0]:
        res.append(np.expand_dims(core.transpose(1, 0, 2), axis=2))
        res.append(
            np.expand_dims(
                u.reshape(init_shape[0], -1, rank).transpose(0, 2, 1), axis=3
            )
        )
    else:
        res.append(
            np.expand_dims(u.transpose(1, 0).reshape(rank, init_shape[1], -1), axis=3)
        )
        res.append(np.expand_dims(core, axis=2))
    return res  # [first, last]
