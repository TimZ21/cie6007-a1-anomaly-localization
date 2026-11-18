"""Spatial and initialization checks without training/test images or labels."""

import copy
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import unittest

import torch
from torch.nn import functional as F

from src.capacity_adapter import (SpatialAdapter, center_features,
                                  gather_neighborhoods, make_head, transform_grid)
from src.defect_adapter import DefectAdapter, loss_components, objective


class CapacityAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def randomize(head):
        # Nonzero biases/weights expose errors that zero-initialization hides.
        with torch.no_grad():
            for p in head.parameters():
                p.normal_(0, .13)
        return head

    def test_counts_shared_initialization_and_identity(self):
        counts = {'MLP32-16': 24992, 'MLP64-16': 49600,
                  'MLP32-32': 24992, 'MLP64-32': 49600,
                  'CNN1-32': 86528, 'CNN3-32': 160384}
        heads = {arm: make_head(arm, 2, self.device) for arm in counts}
        reference = heads['MLP64-32']
        z = F.normalize(torch.randn(4, 7, 384, device=self.device), dim=-1)
        for arm, head in heads.items():
            self.assertEqual(sum(p.numel() for p in head.parameters()), counts[arm])
            torch.testing.assert_close(transform_grid(head, z), z, atol=2e-7, rtol=0)
            if isinstance(head, SpatialAdapter):
                torch.testing.assert_close(head.down.weight[:, :, 0, 0], reference.down.weight, atol=0, rtol=0)
                torch.testing.assert_close(head.down.bias, reference.down.bias, atol=0, rtol=0)
                for block in head.blocks:
                    self.assertEqual(int(torch.count_nonzero(block.weight)), 0)
                    self.assertEqual(int(torch.count_nonzero(block.bias)), 0)
        for width in (32, 64):
            for key, value in heads[f'MLP{width}-16'].state_dict().items():
                torch.testing.assert_close(value, heads[f'MLP{width}-32'].state_dict()[key], atol=0, rtol=0)

    def test_pointwise_mlp_matches_one_by_one_convolutions(self):
        mlp = self.randomize(DefectAdapter(dimension=11, bottleneck=5)).to(self.device)
        cnn = SpatialAdapter(1, dimension=11, hidden=5, prototype=mlp).to(self.device)
        z = F.normalize(torch.randn(2, 11, 5, 9, device=self.device), dim=1)
        expected = mlp(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        torch.testing.assert_close(cnn(z), expected, atol=3e-7, rtol=2e-6)

    def test_neighborhoods_keep_image_view_and_rectangular_coordinates(self):
        first = torch.arange(3 * 5 * 2).reshape(3, 5, 2).float()
        second = 1000 + torch.arange(6 * 4 * 2).reshape(6, 4, 2).float()
        patches, valid = gather_neighborhoods([first, second], [1, 0, 1], [0, 2, 3], [3, 0, 2], 1)
        for i, (grid, row, col) in enumerate([(second, 0, 3), (first, 2, 0), (second, 3, 2)]):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    inside = 0 <= row + dy < grid.shape[0] and 0 <= col + dx < grid.shape[1]
                    self.assertEqual(bool(valid[i, 0, dy + 1, dx + 1]), inside)
                    target = grid[row + dy, col + dx] if inside else torch.zeros(2)
                    torch.testing.assert_close(patches[i, :, dy + 1, dx + 1], target, atol=0, rtol=0)
        uniform = torch.stack([first, first + 1000])
        a, am = gather_neighborhoods(uniform, [0, 1], [0, 2], [4, 1], 3)
        b, bm = gather_neighborhoods(list(uniform), [0, 1], [0, 2], [4, 1], 3)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertTrue(torch.equal(am, bm))

    def test_crop_outputs_and_gradients_match_full_grids_at_edges(self):
        for depth in (1, 3):
            # Double precision isolates exact spatial logic from CUDA rounding.
            full = self.randomize(SpatialAdapter(depth, dimension=7, hidden=4)).double().to(self.device)
            crop = copy.deepcopy(full)
            grids = F.normalize(torch.randn(2, 5, 8, 7, device=self.device, dtype=torch.float64), dim=-1)
            vi = torch.tensor([0, 1, 0, 1, 0, 1], device=self.device)
            yy = torch.tensor([0, 4, 0, 2, 4, 2], device=self.device)
            xx = torch.tensor([0, 7, 7, 0, 2, 4], device=self.device)
            patches, valid = gather_neighborhoods(grids, vi, yy, xx, depth)
            actual = center_features(crop, patches, valid)
            expected = full(grids.permute(0, 3, 1, 2))[vi, :, yy, xx]
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-10)
            target = torch.randn_like(actual)
            (actual * target).sum().backward()
            (expected * target).sum().backward()
            for name, p in full.named_parameters():
                other = dict(crop.named_parameters())[name]
                torch.testing.assert_close(p.grad, other.grad, atol=2e-11, rtol=2e-9)

    def test_gradient_accumulation_preserves_center_loss_normalization(self):
        for depth in (1, 3):
            full = self.randomize(SpatialAdapter(depth, dimension=9, hidden=5)).to(self.device)
            split = copy.deepcopy(full)
            z = F.normalize(torch.randn(2, 6, 8, 9, device=self.device), dim=-1)
            zt = F.normalize(z + .1 * torch.randn_like(z), dim=-1)
            views, rows, cols = [0, 1, 1, 0, 0, 1, 0], [0, 1, 5, 3, 4, 2, 5], [7, 4, 0, 2, 3, 6, 7]
            p, mask = gather_neighborhoods(z, views, rows, cols, depth)
            pt, mt = gather_neighborhoods(zt, views, rows, cols, depth)
            teacher = p[:, :, depth, depth]

            def objective(model, indices):
                h = center_features(model, p[indices], mask[indices])
                ht = center_features(model, pt[indices], mt[indices])
                return ((ht - h).square().sum() + .1 * (h - teacher[indices]).square().sum()) / len(p)

            loss = objective(full, slice(None))
            loss.backward()
            accumulated = 0.
            for start in range(0, len(p), 3):
                value = objective(split, slice(start, start + 3))
                accumulated += float(value.detach())
                value.backward()
            self.assertAlmostEqual(float(loss.detach()), accumulated, places=6)
            for name, param in full.named_parameters():
                other = dict(split.named_parameters())[name]
                torch.testing.assert_close(param.grad, other.grad, atol=3e-7, rtol=5e-5)

    def test_zero_initialization_then_spatial_gradient_activation(self):
        for depth in (1, 3):
            head = SpatialAdapter(depth, dimension=8, hidden=4).to(self.device)
            z = F.normalize(torch.randn(4, 8, 7, 7, device=self.device), dim=1)
            target = F.normalize(z + .2 * torch.randn_like(z), dim=1)
            opt = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001, foreach=False)
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                loss = (head(z) - target).square().sum(1).mean()
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                self.assertGreater(float(head.up.weight.grad.norm()), 0)
                for block in head.blocks:
                    grad_norm = float(block.weight.grad.norm())
                    self.assertEqual(grad_norm, 0) if step == 0 else self.assertGreater(grad_norm, 0)
                self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in head.parameters()))
                opt.step()

    def test_active_centers_match_legacy_zero_weight_branches(self):
        cases = [('cpu', torch.float64, 16, 11, 5)]
        if torch.cuda.is_available():
            cases.append(('cuda', torch.float32, 512, 384, 64))
        for device, dtype, count, dimension, width in cases:
            with self.subTest(device=device, effective_centers=count):
                legacy = self.randomize(DefectAdapter(dimension, width)).to(device=device, dtype=dtype)
                active = copy.deepcopy(legacy)
                batch = [F.normalize(torch.randn(2 * count, dimension, device=device, dtype=dtype), dim=-1)
                         for _ in range(4)]
                before = [x.clone() for x in batch]
                parts = loss_components(legacy, batch, count)
                old_loss = objective(parts, 'A1', anchor_weight=.1, defect_weight=1.)
                h, ht = active(batch[0][:count]), active(batch[1][:count])
                new_loss = ((ht - h).square().sum() + .1 * (h - batch[0][:count]).square().sum()) / count
                old_loss.backward(); new_loss.backward()
                atol, rtol = (2e-12, 2e-10) if dtype == torch.float64 else (3e-7, 3e-5)
                torch.testing.assert_close(old_loss, new_loss, atol=atol, rtol=rtol)
                for name, p in legacy.named_parameters():
                    torch.testing.assert_close(p.grad, dict(active.named_parameters())[name].grad,
                                               atol=atol, rtol=rtol)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(batch, before)))

    @unittest.skipUnless(torch.cuda.is_available(), 'Production CUDA chunk check needs CUDA')
    def test_production_float32_cnn_chunks_keep_the_512_center_objective(self):
        for depth in (1, 3):
            with self.subTest(depth=depth):
                full = SpatialAdapter(depth).cuda()
                with torch.no_grad():
                    for p in full.parameters():
                        p.normal_(0, .02)
                chunked = copy.deepcopy(full)
                grids = F.normalize(torch.randn(2, 32, 32, 384, device='cuda'), dim=-1)
                shifted = F.normalize(grids + .01 * torch.randn_like(grids), dim=-1)
                views = torch.randint(2, (512,), device='cuda')
                rows, cols = (torch.randint(32, (512,), device='cuda') for _ in range(2))
                # Ensure the full-image edges and corners enter the production probe.
                rows[:4] = torch.tensor([0, 0, 31, 31], device='cuda')
                cols[:4] = torch.tensor([0, 31, 0, 31], device='cuda')
                p, valid = gather_neighborhoods(grids, views, rows, cols, depth)
                pt, valid_t = gather_neighborhoods(shifted, views, rows, cols, depth)

                def center_objective(head, start, end):
                    h = center_features(head, p[start:end], valid[start:end])
                    ht = center_features(head, pt[start:end], valid_t[start:end])
                    z = p[start:end, :, depth, depth]
                    return ((ht - h).square().sum() + .1 * (h - z).square().sum()) / 512

                loss = center_objective(full, 0, 512)
                expected = float(loss.detach()); loss.backward()
                actual = 0.
                for start in range(0, 512, 128):
                    part = center_objective(chunked, start, start + 128)
                    actual += float(part.detach()); part.backward()
                self.assertAlmostEqual(expected, actual, places=6)
                for name, p in full.named_parameters():
                    other = dict(chunked.named_parameters())[name]
                    self.assertTrue(bool(torch.isfinite(p.grad).all()))
                    torch.testing.assert_close(p.grad, other.grad, atol=2e-7, rtol=2e-4)

    def test_rejects_invalid_centers_and_incomplete_halos(self):
        grids = torch.randn(2, 3, 5, 4)
        with self.assertRaises(IndexError):
            gather_neighborhoods(grids, [0], [3], [2], 1)
        with self.assertRaises(IndexError):
            gather_neighborhoods(grids, [2], [0], [2], 1)
        with self.assertRaises(ValueError):
            center_features(SpatialAdapter(3, dimension=4, hidden=2), torch.randn(1, 4, 3, 3), torch.ones(1, 1, 3, 3))


if __name__ == '__main__':
    unittest.main()
