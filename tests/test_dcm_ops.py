"""Unit tests kiểm thử các thao tác của cụm bộ nhớ DCM (Phương trình 9 đến 14)."""
import torch

from src.memory import DynamicClusterMemory, MemoryCluster, SquaredErrorKDM

SE = SquaredErrorKDM()


def blobs(centers, n, std=0.05, seed=0):
    """Hàm phụ trợ sinh các cụm điểm giả lập quanh các tâm centers."""
    g = torch.Generator().manual_seed(seed)
    return torch.cat([c + std * torch.randn(n, c.numel(), generator=g) for c in centers])


# ----------------------------------------------------------------- Phương trình (9)
def test_medoid_eq9_bruteforce():
    """Kiểm tra tìm medoid: kết quả phải trùng khớp với duyệt cạn thủ công."""
    torch.manual_seed(0)
    x = torch.randn(15, 6)
    idx = MemoryCluster.find_medoid_index(x, SE)
    sums = [sum(float(((x[c] - x[j]) ** 2).sum()) for j in range(15) if j != c) for c in range(15)]
    assert idx == min(range(15), key=lambda c: sums[c])


# ----------------------------------------------------------------- Phương trình (12)
def test_intra_cluster_removal_modes():
    """Kiểm tra 3 chế độ cắt tỉa nội cụm: farthest, nearest, quantile."""
    proto = torch.zeros(2)
    pts = torch.stack([proto, torch.tensor([0.1, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([5.0, 0.0])])
    c = MemoryCluster(pts, None, capacity=10, kdm=SE, prototype_index=0)
    assert c.removal_index("farthest") == 3          # Eq. 12 in: xóa mẫu xa nhất
    assert c.removal_index("nearest") == 1           # Lời văn: xóa mẫu gần nhất (trừ prototype)
    assert c.removal_index("quantile") in (1, 2)     # Thuật toán quantile: xóa mẫu có khoảng lân cận hẹp nhất


def test_capacity_enforced_and_prototype_kept():
    """Kiểm tra dung lượng cụm luôn <= capacity và prototype không bao giờ bị xóa."""
    c = MemoryCluster(torch.zeros(1, 2), None, capacity=3, kdm=SE, prototype_index=0)
    for v in (0.1, 0.2, 0.3, 0.4, 5.0):
        c.add(torch.tensor([v, 0.0]), None, "nearest")
        assert len(c) <= 3
    assert any(torch.equal(s, torch.zeros(2)) for s in c.samples)
    assert torch.equal(c.prototype, torch.zeros(2))


def test_labels_follow_samples():
    """Đảm bảo nhãn luôn đi kèm tương ứng với mẫu dữ liệu khi thêm/xóa."""
    x = torch.arange(4.0).reshape(4, 1)
    c = MemoryCluster(x, torch.tensor([0, 1, 2, 3]), capacity=4, kdm=SE, prototype_index=0)
    c.add(torch.tensor([10.0]), torch.tensor(7), "farthest")
    assert len(c) == 4
    assert 7 not in c.labels.tolist()  # Mẫu xa nhất (10.0, nhãn 7) bị loại bỏ ngay
    for s, l in zip(c.samples[:, 0].tolist(), c.labels.tolist()):
        assert int(s) == l


# --------------------------------------------------------- Phương trình (10) & (11)
def test_expansion_creates_one_cluster_per_distribution():
    """Kiểm tra mở rộng cụm: mỗi phân phối mới tự động kích hoạt tạo 1 cụm mới."""
    centers = [torch.tensor([0.0, 0.0]), torch.tensor([5.0, 0.0]), torch.tensor([0.0, 5.0])]
    mem = DynamicClusterMemory(SE, cluster_capacity=20, max_clusters=10, lambda_val=4.0)
    for c in centers:
        data = blobs([c], 32)
        for s in range(0, 32, 8):
            mem.observe(data[s:s + 8])
    assert mem.k == 3
    protos = mem.prototypes()
    for c in centers:
        assert ((protos - c) ** 2).sum(1).min() < 0.5


def test_sample_selection_routes_to_nearest_prototype():
    """Kiểm tra định tuyến: mẫu mới phải được gán vào cụm có prototype gần nhất."""
    mem = DynamicClusterMemory(SE, cluster_capacity=50, max_clusters=10, lambda_val=4.0)
    mem.observe(blobs([torch.tensor([0.0, 0.0])], 8))
    mem.observe(blobs([torch.tensor([5.0, 5.0])], 8, seed=1))
    assert mem.k == 2
    before = mem.cluster_sizes()
    mem.observe(torch.tensor([[4.9, 5.1]]))
    after = mem.cluster_sizes()
    far = int(((mem.prototypes() - torch.tensor([5.0, 5.0])) ** 2).sum(1).argmin())
    assert after[far] == before[far] + 1


def test_large_lambda_never_expands():
    """Nếu đặt lambda cực lớn thì bộ nhớ không bao giờ tạo cụm thứ 2."""
    mem = DynamicClusterMemory(SE, cluster_capacity=16, max_clusters=10, lambda_val=1e9)
    for c in (0.0, 10.0, 20.0):
        mem.observe(blobs([torch.tensor([c, c])], 8))
    assert mem.k == 1 and len(mem) <= 16


def test_per_sample_expansion_mode():
    """Kiểm tra chế độ mở rộng theo từng mẫu đơn lẻ."""
    mem = DynamicClusterMemory(SE, cluster_capacity=16, max_clusters=10, lambda_val=4.0,
                               expansion_mode="sample")
    mem.observe(blobs([torch.tensor([0.0, 0.0])], 4))
    mem.observe(torch.tensor([[0.0, 0.1], [9.0, 9.0], [-9.0, 9.0]]))
    assert mem.k == 3


# --------------------------------------------------------- Phương trình (13) & (14)
def _mem_with_prototypes(protos, mode):
    mem = DynamicClusterMemory(SE, cluster_capacity=4, max_clusters=len(protos), lambda_val=1.0, mpp_mode=mode)
    for p in protos:
        mem.clusters.append(MemoryCluster(p[None], None, 4, SE, prototype_index=0))
    return mem


def test_mpp_consistent_removes_redundant_cluster():
    """Kiểm tra MPP consistent: xóa cụm dư thừa trong cặp gần nhau nhất."""
    protos = [torch.tensor([0.0, 0.0]), torch.tensor([0.1, 0.0]), torch.tensor([10.0, 0.0]), torch.tensor([0.0, 10.0])]
    mem = _mem_with_prototypes(protos, "consistent")
    g, h, victim = mem.select_pair_and_victim()
    assert {g, h} == {0, 1}
    assert victim in (0, 1)


def test_mpp_literal_follows_printed_equations():
    """Kiểm tra MPP literal: công thức in trong bài (argmax) chọn cặp xa nhất."""
    protos = [torch.tensor([0.0, 0.0]), torch.tensor([0.1, 0.0]), torch.tensor([10.0, 0.0]), torch.tensor([-10.0, 0.0])]
    mem = _mem_with_prototypes(protos, "literal")
    g, h, _ = mem.select_pair_and_victim()
    assert {g, h} == {2, 3}


def test_mpp_keeps_k_at_most_rho():
    """Đảm bảo số cụm không bao giờ vượt quá rho khi kích hoạt MPP."""
    mem = DynamicClusterMemory(SE, cluster_capacity=8, max_clusters=3, lambda_val=1.0)
    for i in range(8):
        mem.observe(blobs([torch.tensor([10.0 * i, 0.0])], 4, seed=i))
        assert mem.k <= 3
    assert mem.n_prunes == 5


def test_memory_is_label_free():
    """Tính chất Label-Free: nhãn khác nhau (thậm chí đối nghịch) không làm thay đổi trạng thái bộ nhớ."""
    data = blobs([torch.tensor([0.0, 0.0]), torch.tensor([6.0, 0.0])], 24)
    m1 = DynamicClusterMemory(SE, cluster_capacity=10, max_clusters=5, lambda_val=4.0)
    m2 = DynamicClusterMemory(SE, cluster_capacity=10, max_clusters=5, lambda_val=4.0)
    y1 = torch.zeros(48, dtype=torch.long)
    y2 = torch.randint(0, 10, (48,))
    for s in range(0, 48, 8):
        m1.observe(data[s:s + 8], y1[s:s + 8])
        m2.observe(data[s:s + 8], y2[s:s + 8])
    assert m1.cluster_sizes() == m2.cluster_sizes()
    assert torch.equal(m1.all_samples()[0], m2.all_samples()[0])
