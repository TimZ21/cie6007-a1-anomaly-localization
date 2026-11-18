"""Mechanism/data-isolation tests using synthetic tensors only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from src.defect_adapter import DefectAdapter, quadruplets, loss_components, objective, batch_indices
from src.features import photometric


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_initial_identity_and_gradient_flow(self):
        model = DefectAdapter().to(self.device)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 24992)
        z = F.normalize(torch.randn(48, 384, device=self.device), dim=-1)
        before = z.clone()
        torch.testing.assert_close(model(z), z, atol=1e-6, rtol=0)
        target = F.normalize(z + .1 * torch.randn_like(z), dim=-1)
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001, foreach=False)
        losses = []
        for _ in range(20):
            opt.zero_grad(set_to_none=True)
            loss = (model(z) - target).square().sum(-1).mean()
            loss.backward()
            self.assertGreater(float(model.up.weight.grad.norm()), 0.)
            opt.step(); losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(torch.equal(z, before))
        self.assertIsNone(z.grad)

    def test_quadruplets_alignment_and_reproducibility(self):
        rgb = torch.rand(3, 448, 448)
        cond = [{'kind': 'brightness', 'factor': .85}, {'kind': 'contrast', 'factor': 1.15}]
        first = list(quadruplets(rgb, 'cable/train/good/001.png', 3, cond))
        second = list(quadruplets(rgb, 'cable/train/good/001.png', 3, cond))
        self.assertEqual(len(first), 4)
        for (views, mask, meta), (again, othermask, othermeta) in zip(first, second):
            self.assertEqual(meta, othermeta)
            self.assertTrue(torch.equal(mask, othermask) and mask.any())
            self.assertEqual(mask.shape, (1024,))
            for a, b in zip(views, again):
                self.assertTrue(torch.equal(a, b))
                self.assertTrue((a >= 0).all() and (a <= 1).all())
            x, xt, xa, xta = views
            t, l, h, w = meta['rectangle_tlhw']
            outside = torch.ones(448, 448, dtype=torch.bool); outside[t:t+h, l:l+w] = False
            self.assertTrue(torch.equal(x[:, outside], xa[:, outside]))
            c = meta['condition']
            self.assertTrue(torch.equal(xta, photometric(xa, c['kind'], c['factor'])))
            self.assertTrue(torch.equal(xt, photometric(x, c['kind'], c['factor'])))
            # Every selected patch has >=80% occupancy within the inset rectangle.
            for p in mask.nonzero().flatten().tolist():
                y, x0 = divmod(p, 32)
                overlap = max(0, min((y+1)*14, t+h-14)-max(y*14, t+14))
                overlap *= max(0, min((x0+1)*14, l+w-14)-max(x0*14, l+14))
                self.assertGreaterEqual(overlap / 196, .8)

    def test_clean_and_cross_objectives_are_distinct(self):
        z = F.normalize(torch.randn(16, 8), dim=-1)
        za = F.normalize(z + .4 * torch.randn_like(z), dim=-1)
        zt = F.normalize(z + .2 * torch.randn_like(z), dim=-1)
        zta = F.normalize(za + .3 * torch.randn_like(z), dim=-1)
        p = loss_components(torch.nn.Identity(), [z, zt, za, zta], 8)
        self.assertEqual(float(p['anchor']), 0.)
        self.assertEqual(float(p['clean_defect']), 0.)
        self.assertGreater(float(p['cross_defect']), 0.)
        self.assertAlmostEqual(float(objective(p, 'A1')), float(p['illumination']))
        self.assertAlmostEqual(float(objective(p, 'A2')), float(p['illumination']))
        self.assertAlmostEqual(float(objective(p, 'A3')), float(p['illumination'] + p['cross_defect']))
        # Same lighting on both views reduces cross preservation to clean preservation.
        same = loss_components(torch.nn.Identity(), [z, z, za, za], 8)
        self.assertEqual(float(objective(same, 'A3')), 0.)

    def test_identical_seeded_training_batches(self):
        interior = np.array([3, 7, 12, 55])
        a, b = np.random.default_rng(9), np.random.default_rng(9)
        for _ in range(20):
            i, j = batch_indices(a, 64, interior, 32), batch_indices(b, 64, interior, 32)
            np.testing.assert_array_equal(i, j)
            self.assertTrue(np.isin(i[16:], interior).all())
            self.assertTrue(((i >= 0) & (i < 64)).all())


if __name__ == '__main__':
    unittest.main()
