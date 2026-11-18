"""Check the release wrapper against the unchanged loss functions on synthetic data."""
import copy
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import reproduce as release
from src.capacity_adapter import SpatialAdapter, center_features, gather_neighborhoods
from src.defect_adapter import DefectAdapter
from src.loss_study import loss_parts, total


class ReleaseLossTests(unittest.TestCase):
    def test_pointwise_losses_and_chunked_gradients_match_original(self):
        # No download, labels or real images; double precision isolates reductions.
        release.torch = torch
        release.SpatialAdapter = SpatialAdapter
        release.center_features = center_features
        release.gather_neighborhoods = gather_neighborhoods
        torch.manual_seed(17)
        grids = F.normalize(torch.randn(10, 5, 8, 9, dtype=torch.float64), dim=-1)
        ids = torch.randint(16*5*8, (512,))
        tuples = ids//40
        z = grids[tuples//8*5, ids%40//8, ids%8]
        zt = grids[tuples//8*5+1+tuples%4, ids%40//8, ids%8]
        for arm in ('MLP32-16','L1','L2','A1-no-anchor'):
            expected = DefectAdapter(9,5).double()
            with torch.no_grad():
                for parameter in expected.parameters():
                    parameter.normal_(0,.13)
            actual = copy.deepcopy(expected)
            parts = loss_parts(expected,z,zt)
            loss = parts['illumination'] if arm=='A1-no-anchor' else total(parts,arm if arm in ('L1','L2') else 'A1')
            loss.backward()
            observed = release.backward_centers(actual,grids,ids,arm,128)
            self.assertAlmostEqual(observed,float(loss.detach()),places=12)
            for a,b in zip(actual.parameters(),expected.parameters()):
                torch.testing.assert_close(a.grad,b.grad,atol=2e-12,rtol=2e-10)


if __name__=='__main__':
    unittest.main()
