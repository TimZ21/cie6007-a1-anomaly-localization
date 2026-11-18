"""Small shared implementation of B1/B2/P and exact nearest-neighbor scoring."""
import math
import torch


def projection(input_dim, output_dim, seed, device):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return (torch.randn(input_dim, output_dim, generator=generator) / math.sqrt(output_dim)).to(device)


@torch.inference_mode()
def nearest_distance(queries, bank, chunk_size=1024):
    """Exact search over every bank vector; float32 Euclidean distances."""
    if len(bank) == 0 or chunk_size < 1:
        raise ValueError('Nonempty bank and positive chunk size required')
    return torch.cat([torch.cdist(chunk, bank, p=2,
                      compute_mode='use_mm_for_euclid_dist').min(dim=1).values
                      for chunk in queries.split(chunk_size)])


@torch.inference_mode()
def farthest_first(points, count, initial_centers=None):
    """Distinct candidate IDs; seed 0 or farthest from supplied centers.

    Supplying centers initializes distances against ALL those centers, then
    every new center updates distances. This makes P's second stage complementary.
    """
    if not 0 < count <= len(points):
        raise ValueError('Count must be within candidate population')
    norms = points.square().sum(dim=1)
    if initial_centers is None:
        distances = torch.full_like(norms, float('inf'))
        index = torch.zeros((), dtype=torch.long, device=points.device)
    else:
        distances = nearest_distance(points, initial_centers).square()
        index = distances.argmax()
    selected = torch.empty(count, dtype=torch.long, device=points.device)
    for step in range(count):
        selected[step] = index
        center = points[index]
        new_distances = (norms + center.square().sum() - 2 * torch.mv(points, center)).clamp_min_(0)
        torch.minimum(distances, new_distances, out=distances)
        distances[index] = -float('inf')  # Preserve exclusion through subsequent minima.
        index = distances.argmax()  # PyTorch uses the first index to break exact ties.
    return selected


@torch.inference_mode()
def select_bank(projected, original_count, method, bank_size, rho=0.5):
    if method == 'B1':
        return farthest_first(projected[:original_count], bank_size)
    if method == 'B2':
        return farthest_first(projected, bank_size)
    if method == 'P':
        quota = int(bank_size * rho)
        if quota != bank_size * rho or not 0 < quota < bank_size:
            raise ValueError('P needs an integral original quota between 0 and bank size')
        original = farthest_first(projected[:original_count], quota)
        augmented = farthest_first(projected[original_count:], bank_size-quota,
                                   initial_centers=projected[original]) + original_count
        return torch.cat([original, augmented])
    raise ValueError(f'Unknown method: {method}')
