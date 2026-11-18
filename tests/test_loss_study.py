"""Loss and gradient checks on synthetic tensors, without dataset access."""

import copy
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import unittest

import torch
from torch.nn import functional as F

from src.defect_adapter import DefectAdapter, loss_components, objective
from src.loss_study import loss_parts, make_adapter, total


class LossStudyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def nonidentity(head):
        # Nonzero output projection exposes teacher/anchor gradient mistakes.
        with torch.no_grad():
            head.up.weight.normal_(0, .08)
            head.up.bias.normal_(0, .04)
        return head

    def test_original_initialization_identity_and_finite_gradients(self):
        for seed in (0, 1, 2):
            head = make_adapter(seed, self.device)
            torch.manual_seed(5000 + seed)
            original = DefectAdapter().to(self.device)
            self.assertEqual(sum(p.numel() for p in head.parameters()), 24992)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, head.state_dict()[name], atol=0, rtol=0)
            z = F.normalize(torch.randn(32, 384, device=self.device), dim=-1)
            zt = F.normalize(z + .03 * torch.randn_like(z), dim=-1)
            torch.testing.assert_close(head(z), z, atol=2e-7, rtol=0)
            for variant in ('A1', 'L1', 'L2'):
                head.zero_grad(set_to_none=True)
                value = total(loss_parts(head, z, zt), variant)
                self.assertTrue(bool(torch.isfinite(value)))
                value.backward()
                self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                    for p in head.parameters()))
                self.assertGreater(float(head.up.weight.grad.norm()), 0.)

    def test_a1_matches_original_loss_and_all_parameter_gradients(self):
        cases = [('cpu', torch.float64, 16, 11, 5)]
        if torch.cuda.is_available():
            cases.append(('cuda', torch.float32, 512, 384, 32))
        for device, dtype, count, dimension, width in cases:
            with self.subTest(device=device, centers=count):
                legacy = self.nonidentity(DefectAdapter(dimension, width)).to(device=device, dtype=dtype)
                current = copy.deepcopy(legacy)
                batch = [F.normalize(torch.randn(2 * count, dimension, device=device, dtype=dtype), dim=-1)
                         for _ in range(4)]
                old_loss = objective(loss_components(legacy, batch, count), 'A1')
                new_loss = total(loss_parts(current, batch[0][:count], batch[1][:count]), 'A1')
                old_loss.backward(); new_loss.backward()
                atol, rtol = (2e-12, 2e-10) if dtype == torch.float64 else (3e-7, 3e-5)
                torch.testing.assert_close(old_loss, new_loss, atol=atol, rtol=rtol)
                for name, parameter in legacy.named_parameters():
                    torch.testing.assert_close(parameter.grad, dict(current.named_parameters())[name].grad,
                                               atol=atol, rtol=rtol)

    def test_l1_changes_only_anchor_coefficient(self):
        head = self.nonidentity(DefectAdapter(11, 5)).double()
        z = F.normalize(torch.randn(17, 11, dtype=torch.float64), dim=-1)
        zt = F.normalize(z + .1 * torch.randn_like(z), dim=-1)
        parts = loss_parts(head, z, zt)
        delta = total(parts, 'L1') - total(parts, 'A1')
        torch.testing.assert_close(delta, .1 * parts['anchor'], atol=2e-12, rtol=2e-10)
        actual = torch.autograd.grad(delta, tuple(head.parameters()), retain_graph=True)
        expected = torch.autograd.grad(.1 * parts['anchor'], tuple(head.parameters()))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-10)

    def test_l2_detaches_fixed_teacher_and_preserves_clean_anchor_gradient(self):
        head = self.nonidentity(DefectAdapter(11, 5)).double()
        z = F.normalize(torch.randn(19, 11, dtype=torch.float64), dim=-1).requires_grad_()
        zt = F.normalize(torch.randn(19, 11, dtype=torch.float64), dim=-1).requires_grad_()
        parts = loss_parts(head, z, zt)
        # The one-way teacher term has no path to the clean input. The input
        # deliberately requires grad here solely to verify this isolation.
        clean_teacher_grad = torch.autograd.grad(parts['fixed_teacher_alignment'], z,
                                                 allow_unused=True, retain_graph=True)[0]
        self.assertIsNone(clean_teacher_grad)
        shifted_grad = torch.autograd.grad(parts['fixed_teacher_alignment'], zt, retain_graph=True)[0]
        self.assertGreater(float(shifted_grad.norm()), 0.)
        anchor_grads = torch.autograd.grad(parts['anchor'], tuple(head.parameters()), retain_graph=True)
        self.assertGreater(sum(float(g.norm()) for g in anchor_grads), 0.)
        clean_total_grad = torch.autograd.grad(total(parts, 'L2'), z, retain_graph=True)[0]
        clean_anchor_grad = torch.autograd.grad(.1 * parts['anchor'], z, retain_graph=True)[0]
        torch.testing.assert_close(clean_total_grad, clean_anchor_grad, atol=2e-12, rtol=2e-10)
        self.assertGreater(float(clean_total_grad.norm()), 0.)
        ht, h = head(zt), head(z)
        fixed = (ht - z.detach()).square().sum(-1).mean()
        moving = (ht - h.detach()).square().sum(-1).mean()
        torch.testing.assert_close(parts['fixed_teacher_alignment'], fixed, atol=2e-12, rtol=2e-10)
        self.assertGreater(float((fixed - moving).detach().abs()), 1e-5)
        torch.testing.assert_close(total(parts, 'L2'), fixed + .1 * parts['anchor'], atol=2e-12, rtol=2e-10)

    def test_sum_over_features_mean_over_effective_centers(self):
        z = torch.zeros(7, 384, dtype=torch.float64); z[:, 0] = 1
        zt = -z
        parts = loss_parts(torch.nn.Identity(), z, zt)
        self.assertEqual(float(parts['illumination']), 4.)
        self.assertEqual(float(parts['fixed_teacher_alignment']), 4.)
        self.assertEqual(float(parts['anchor']), 0.)
        chunk = loss_parts(torch.nn.Identity(), z, zt, total_centers=512)
        self.assertEqual(float(chunk['illumination']), 7 * 4 / 512)
        self.assertNotEqual(float(parts['illumination']), float(F.mse_loss(z, zt)))

    def test_chunks_preserve_full_512_center_losses_and_gradients(self):
        z = F.normalize(torch.randn(512, 384, device=self.device), dim=-1)
        zt = F.normalize(z + .03 * torch.randn_like(z), dim=-1)
        before = (z.clone(), zt.clone())
        for variant in ('A1', 'L1', 'L2'):
            with self.subTest(variant=variant):
                full = self.nonidentity(make_adapter(0, self.device))
                split = copy.deepcopy(full)
                value = total(loss_parts(full, z, zt), variant)
                value.backward(); accumulated = 0.
                # Uneven final chunk catches a mean-of-chunk-means bug.
                for start in range(0, 512, 127):
                    part = total(loss_parts(split, z[start:start + 127], zt[start:start + 127],
                                            total_centers=512), variant)
                    accumulated += float(part.detach()); part.backward()
                self.assertAlmostEqual(float(value.detach()), accumulated, places=6)
                for name, parameter in full.named_parameters():
                    torch.testing.assert_close(parameter.grad, dict(split.named_parameters())[name].grad,
                                               atol=3e-7, rtol=5e-5)
        self.assertTrue(torch.equal(z, before[0]) and torch.equal(zt, before[1]))

    def test_rejects_bad_variants_shapes_and_denominators(self):
        head = torch.nn.Identity(); z = torch.randn(4, 7)
        with self.assertRaises(ValueError):
            total(loss_parts(head, z, z), 'unknown')
        for denominator in (0, 3, 4.5, True):
            with self.assertRaises(ValueError):
                loss_parts(head, z, z, total_centers=denominator)
        with self.assertRaises(ValueError):
            loss_parts(head, z, z[:2])
        with self.assertRaises(ValueError):
            loss_parts(head, z[:0], z[:0])


if __name__ == '__main__':
    unittest.main()
