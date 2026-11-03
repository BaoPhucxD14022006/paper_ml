"""Unit tests kiểm tra các thước đo KDM (SE và JS divergence - Phương trình 7 và 8)."""
import math

import torch

from src.memory.kdm import JensenShannonKDM, SquaredErrorKDM, linear_alpha_bar


def test_se_matches_eq7_bruteforce():
    """Kiểm tra SquaredErrorKDM khớp với tính khoảng cách bình phương thủ công."""
    torch.manual_seed(0)
    x, y = torch.randn(5, 3, 4, 4), torch.randn(7, 3, 4, 4)
    d = SquaredErrorKDM().pairwise(x, y)
    for i in range(5):
        for j in range(7):
            ref = ((x[i] - y[j]) ** 2).sum()
            assert torch.allclose(d[i, j], ref, rtol=1e-4, atol=1e-4)


def test_se_zero_on_identical_and_symmetric():
    """Khoảng cách tới chính nó bằng 0 và ma trận khoảng cách đối xứng."""
    x = torch.randn(6, 10)
    d = SquaredErrorKDM().pairwise(x, x)
    assert torch.allclose(d.diag(), torch.zeros(6), atol=1e-4)
    assert torch.allclose(d, d.T, atol=1e-4)


def test_alpha_bar_linear_schedule():
    """Kiểm tra giá trị alpha_bar tại t=100 theo lịch trình tuyến tính (~0.897)."""
    ab = linear_alpha_bar(100)
    assert 0.85 < ab < 0.95


def test_js_properties():
    """Kiểm tra các tính chất cơ bản của phân kỳ JS: không âm, <= ln 2, đối xứng."""
    kdm = JensenShannonKDM(t_star=100)
    x = torch.randn(8, 20)
    d = kdm.pairwise(x, x)
    assert torch.allclose(d.diag(), torch.zeros(8), atol=1e-6)
    assert (d >= 0).all() and (d <= math.log(2) + 1e-6).all()
    assert torch.allclose(d, d.T, atol=1e-6)


def test_js_1d_matches_monte_carlo():
    """Đối chiếu tích phân Gauss-Hermite với ước lượng độc lập bằng Monte Carlo."""
    kdm = JensenShannonKDM()
    torch.manual_seed(0)
    for delta in (0.3, 1.0, 2.5):
        p = torch.distributions.Normal(0.0, 1.0)
        q = torch.distributions.Normal(delta, 1.0)
        zp, zq = p.sample((400_000,)), q.sample((400_000,))

        def log_m(z):
            return torch.logsumexp(torch.stack([p.log_prob(z), q.log_prob(z)]), 0) - math.log(2)

        mc = 0.5 * (p.log_prob(zp) - log_m(zp)).mean() + 0.5 * (q.log_prob(zq) - log_m(zq)).mean()
        val = kdm.js_1d(torch.tensor([delta]))[0]
        assert abs(float(val) - float(mc)) < 5e-3, (delta, float(val), float(mc))


def test_js_exact_reduces_to_1d_in_multivariate_case():
    """Kiểm chứng toán học: Phân kỳ JS đa chiều giữa 2 Gaussian cùng phương sai
    đẳng hướng tương đương với phân kỳ JS 1 chiều dọc theo hiệu hai vector kỳ vọng.
    """
    kdm = JensenShannonKDM(alpha_bar=0.5)
    xg, xh = torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[0.6, -0.8, 0.0]])
    P = torch.distributions.MultivariateNormal(math.sqrt(0.5) * xg[0], 0.5 * torch.eye(3))
    Q = torch.distributions.MultivariateNormal(math.sqrt(0.5) * xh[0], 0.5 * torch.eye(3))
    torch.manual_seed(1)
    zp, zq = P.sample((300_000,)), Q.sample((300_000,))
    lm = lambda z: torch.logsumexp(torch.stack([P.log_prob(z), Q.log_prob(z)]), 0) - math.log(2)
    mc = 0.5 * (P.log_prob(zp) - lm(zp)).mean() + 0.5 * (Q.log_prob(zq) - lm(zq)).mean()
    val = kdm.pairwise(xg, xh)[0, 0]
    assert abs(float(val) - float(mc)) < 5e-3


def test_js_monotone_in_se():
    """Tính đơn điệu: Phân kỳ JS tăng đơn điệu theo khoảng cách bình phương L2."""
    kdm = JensenShannonKDM(alpha_bar=0.5)
    base = torch.zeros(1, 4)
    pts = torch.stack([torch.tensor([r, 0.0, 0.0, 0.0]) for r in (0.1, 0.5, 1.0, 2.0)])
    d = kdm.pairwise(base, pts)[0]
    assert (d[1:] > d[:-1]).all()


def test_js_dimwise_mode_runs_and_is_bounded():
    """Kiểm tra chế độ dimwise hoạt động đúng và luôn nằm trong khoảng [0, ln 2]."""
    kdm = JensenShannonKDM(t_star=100, mode="dimwise")
    x, y = torch.rand(4, 3, 8, 8) * 2 - 1, torch.rand(3, 3, 8, 8) * 2 - 1
    d = kdm.pairwise(x, y)
    assert d.shape == (4, 3)
    assert (d >= 0).all() and (d <= math.log(2) + 1e-5).all()
    assert torch.allclose(kdm.pairwise(x, x).diag(), torch.zeros(4), atol=1e-5)
