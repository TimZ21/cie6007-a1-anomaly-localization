"""CPU checks for the width-only study, without datasets or encoder weights.

The frozen DefectAdapter and original A1 objective are the reference. These
checks use synthetic normalized inputs, never test images or defect labels.
"""
import copy
import unittest

import torch
from torch.nn import functional as F

from src.defect_adapter import DefectAdapter, loss_components, objective
from src.loss_study import loss_parts


def make_cpu(width, seed=0):
    torch.manual_seed(5000 + seed)
    return DefectAdapter(dimension=384, bottleneck=width).cpu()


def a1_loss(parts):
    return parts['illumination'] + .1 * parts['anchor']


class ExpandedMLPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    @staticmethod
    def nonidentity(head):
        # Expose loss/gradient differences hidden by the zero initial output.
        with torch.no_grad():
            head.up.weight.normal_(0, .008)
            head.up.bias.normal_(0, .004)
        return head

    def test_counts_seed_rule_and_initial_unit_identity(self):
        z = F.normalize(torch.randn(37, 384), dim=-1)
        for width, count in ((32, 24992), (768, 590976)):
            for seed in (0, 1, 2):
                with self.subTest(width=width, seed=seed):
                    head = make_cpu(width, seed)
                    repeat = make_cpu(width, seed)
                    self.assertEqual(sum(p.numel() for p in head.parameters()), count)
                    self.assertEqual(tuple(head.down.weight.shape), (width, 384))
                    self.assertEqual(tuple(head.up.weight.shape), (384, width))
                    self.assertEqual(torch.count_nonzero(head.up.weight).item(), 0)
                    self.assertEqual(torch.count_nonzero(head.up.bias).item(), 0)
                    self.assertGreater(torch.count_nonzero(head.down.weight).item(), 0)
                    for name, value in head.state_dict().items():
                        torch.testing.assert_close(value, repeat.state_dict()[name], atol=0, rtol=0)
                    output = head(z)
                    self.assertEqual(tuple(output.shape), (37, 384))
                    torch.testing.assert_close(output, z, atol=2e-7, rtol=0)
                    torch.testing.assert_close(output.norm(dim=-1), torch.ones(37), atol=2e-7, rtol=0)
            different = make_cpu(width, 1)
            self.assertFalse(torch.equal(make_cpu(width, 0).down.weight, different.down.weight))
        # The two widths follow the same seed rule, not an elementwise identity
        # claim between differently shaped parameter tensors.
        self.assertNotEqual(make_cpu(32).down.weight.shape, make_cpu(768).down.weight.shape)

    def test_two_updates_have_finite_gradients_and_leave_inputs_unchanged(self):
        z = F.normalize(torch.randn(512, 384), dim=-1)
        zt = F.normalize(z + .03 * torch.randn_like(z), dim=-1)
        before = (z.clone(), zt.clone())
        for width in (32, 768):
            with self.subTest(width=width):
                head = make_cpu(width)
                optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001,
                                              foreach=False)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss = a1_loss(loss_parts(head, z, zt))
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                                        for p in head.parameters()))
                    self.assertGreater(head.up.weight.grad.norm().item(), 0)
                    if step == 0:
                        self.assertEqual(torch.count_nonzero(head.down.weight.grad).item(), 0)
                        self.assertEqual(torch.count_nonzero(head.down.bias.grad).item(), 0)
                    else:
                        self.assertGreater(head.down.weight.grad.norm().item(), 0)
                    optimizer.step()
                    self.assertTrue(all(torch.isfinite(p).all().item() for p in head.parameters()))
                self.assertGreater(torch.count_nonzero(head.up.weight).item(), 0)
        self.assertFalse(z.requires_grad or zt.requires_grad)
        self.assertIsNone(z.grad)
        self.assertIsNone(zt.grad)
        torch.testing.assert_close(z, before[0], atol=0, rtol=0)
        torch.testing.assert_close(zt, before[1], atol=0, rtol=0)

    def test_active_a1_loss_matches_original_four_view_loss_and_gradients(self):
        count = 32
        batch = [F.normalize(torch.randn(2 * count, 384), dim=-1) for _ in range(4)]
        for width in (32, 768):
            with self.subTest(width=width):
                original = self.nonidentity(make_cpu(width))
                current = copy.deepcopy(original)
                old = objective(loss_components(original, batch, count), 'A1', .1, 1.)
                new = a1_loss(loss_parts(current, batch[0][:count], batch[1][:count]))
                old.backward()
                new.backward()
                torch.testing.assert_close(old, new, atol=3e-7, rtol=3e-5)
                for name, p in original.named_parameters():
                    torch.testing.assert_close(p.grad, dict(current.named_parameters())[name].grad,
                                               atol=3e-7, rtol=3e-5)

    def test_both_illumination_branches_receive_the_analytic_gradient(self):
        class TrackedHead(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner, self.outputs = inner, []

            def forward(self, value):
                output = self.inner(value)
                output.retain_grad()
                self.outputs.append(output)
                return output

        z = F.normalize(torch.randn(17, 384), dim=-1)
        zt = F.normalize(z + .04 * torch.randn_like(z), dim=-1)
        head = TrackedHead(self.nonidentity(make_cpu(768)))
        a1_loss(loss_parts(head, z, zt)).backward()
        clean, light = head.outputs
        expected_clean = (2 * (clean.detach() - light.detach()) + .2 * (clean.detach() - z)) / len(z)
        expected_light = 2 * (light.detach() - clean.detach()) / len(z)
        torch.testing.assert_close(clean.grad, expected_clean, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(light.grad, expected_light, atol=2e-7, rtol=2e-5)
        self.assertGreater(clean.grad.norm().item(), 0)
        self.assertGreater(light.grad.norm().item(), 0)

    def test_loss_sums_coordinates_and_averages_512_effective_centers(self):
        z = torch.zeros(512, 384, dtype=torch.float64)
        z[:, 0] = 1
        parts = loss_parts(torch.nn.Identity(), z, -z)
        self.assertEqual(parts['illumination'].item(), 4.)
        self.assertEqual(parts['anchor'].item(), 0.)
        self.assertNotEqual(parts['illumination'].item(), F.mse_loss(z, -z).item())
        chunk = loss_parts(torch.nn.Identity(), z[:127], -z[:127], total_centers=512)
        self.assertEqual(chunk['illumination'].item(), 127 * 4 / 512)

    def test_512_center_forward_loss_and_gradient_chunk_equivalence(self):
        z = F.normalize(torch.randn(512, 384), dim=-1)
        zt = F.normalize(z + .03 * torch.randn_like(z), dim=-1)
        for width in (32, 768):
            with self.subTest(width=width):
                full = self.nonidentity(make_cpu(width))
                split = copy.deepcopy(full)
                with torch.no_grad():
                    whole_output = full(z)
                    chunk_output = torch.cat([split(x) for x in z.split(127)])
                torch.testing.assert_close(whole_output, chunk_output, atol=2e-6, rtol=2e-6)
                value = a1_loss(loss_parts(full, z, zt))
                value.backward()
                accumulated = 0.
                # Uneven last chunk detects accidental averaging of chunk means.
                for start in range(0, len(z), 127):
                    chunk_loss = a1_loss(loss_parts(split, z[start:start + 127],
                        zt[start:start + 127], total_centers=512))
                    accumulated += chunk_loss.detach().item()
                    chunk_loss.backward()
                self.assertAlmostEqual(value.detach().item(), accumulated, places=6)
                for name, p in full.named_parameters():
                    torch.testing.assert_close(p.grad, dict(split.named_parameters())[name].grad,
                                               atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
