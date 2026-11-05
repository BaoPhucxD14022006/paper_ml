"""Unit tests kiểm tra tính đúng đắn của dòng dữ liệu OnlineStream và các giao diện."""
import inspect

import torch

from src.baselines import ReservoirMemory
from src.data import OnlineStream, Preprocess, RawDataset
from src.models import GaussianDiffusion, UNet, make_betas
from src.models.classifier import MLP, ReducedResNet18


def fake_raw(n_per_class=30, n_classes=10):
    """Tạo bộ dữ liệu giả lập cho việc test nhanh."""
    labels = torch.arange(n_classes).repeat_interleave(n_per_class)
    imgs = (labels.view(-1, 1, 1, 1) * 20).to(torch.uint8).expand(-1, 1, 28, 28).contiguous()
    return RawDataset(imgs, labels, "mnist")


def test_stream_is_task_ordered_and_covers_each_sample_once():
    """Đảm bảo dòng dữ liệu sắp xếp theo đúng thứ tự task và mỗi mẫu chỉ xuất hiện đúng 1 lần."""
    raw = fake_raw()
    st = OnlineStream(raw, n_tasks=5, batch_size=7, preprocess=Preprocess(), seed=0)
    seen, last_task = [], -1
    for i, (x, y) in enumerate(st):
        assert x.shape[0] <= 7
        seen.extend(y.tolist())
        t = st.oracle_task_of_step(i)
        assert t >= last_task
        last_task = t
    assert sorted(seen) == sorted(raw.labels.tolist())
    first_task = [l for l in seen[:60]]
    assert set(first_task) == {0, 1}


def test_stream_yields_no_task_information():
    """Kiểm tra dòng dữ liệu chỉ trả về (x, y), tuyệt đối không để lộ task ID hay ranh giới task."""
    st = OnlineStream(fake_raw(), n_tasks=5, batch_size=10, preprocess=Preprocess(), seed=0)
    item = next(iter(st))
    assert isinstance(item, tuple) and len(item) == 2


def test_preprocess_shapes_and_range():
    """Kiểm tra tiền xử lý ảnh: đúng kích thước, số kênh và dải giá trị chuẩn hóa."""
    raw = fake_raw(2)
    x = Preprocess(image_size=32, channels=3, value_range="pm1")(raw.images[:4])
    assert x.shape == (4, 3, 32, 32) and x.min() >= -1 and x.max() <= 1
    x = Preprocess(flatten=True)(raw.images[:4])
    assert x.shape == (4, 784)


def test_memory_interfaces_take_no_model_or_task_argument():
    """Đảm bảo hàm observe của các bộ nhớ chỉ nhận (x, y), không nhận model hay task ID."""
    from src.memory import DynamicClusterMemory
    for cls in (DynamicClusterMemory, ReservoirMemory):
        params = list(inspect.signature(cls.observe).parameters)
        assert params == ["self", "x", "y"]


def test_reservoir_capacity():
    """Kiểm tra bộ nhớ Reservoir không bao giờ vượt quá sức chứa capacity."""
    m = ReservoirMemory(20)
    for _ in range(10):
        m.observe(torch.randn(10, 3), torch.zeros(10, dtype=torch.long))
    assert len(m) == 20 and m.n_seen == 100


def test_classifiers_forward():
    """Kiểm tra chiều đầu ra của các mạng phân loại MLP và ReducedResNet18."""
    assert MLP()(torch.randn(2, 784)).shape == (2, 10)
    assert ReducedResNet18()(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_iddpm_loss_and_sampling_smoke():
    """Kiểm tra tính loss và sinh ảnh nhanh của mô hình khuếch tán IDDPM."""
    torch.manual_seed(0)
    net = UNet(in_channels=3, base=16, mults=(1, 2), n_res=1, attn_res=(8,), image_size=16, dropout=0.0)
    diff = GaussianDiffusion(make_betas("linear", 1000))
    x0 = torch.rand(4, 3, 16, 16) * 2 - 1
    out = diff.training_loss(net, x0)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    img = diff.respaced(5).sample(net, (2, 3, 16, 16))
    assert img.shape == (2, 3, 16, 16) and torch.isfinite(img).all()


def test_respacing_preserves_alpha_bar():
    """Kiểm tra việc rút gọn số bước sinh ảnh (respacing) bảo toàn lịch trình alpha_bar."""
    d = GaussianDiffusion(make_betas("linear", 1000))
    r = d.respaced(50)
    orig = torch.tensor(d.abar)[r.timestep_map]
    assert torch.allclose(torch.tensor(r.abar), orig, rtol=1e-8)
