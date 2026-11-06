# Tái hiện bài báo: Online Task-Free Continual Learning via Dynamic Cluster Memory (CVPR 2024)

Repo này cài đặt lại thuật toán **Dynamic Cluster Memory (DCM)** dựa trên bài báo:  
> **"Online Task-Free Continual Generative and Discriminative Learning via Dynamic Cluster Memory"**  
> *Fei Ye, Adrian G. Bors — CVPR 2024* ([Link bài báo](https://openaccess.thecvf.com/content/CVPR2024/html/Ye_Online_Task-Free_Continual_Generative_and_Discriminative_Learning_via_Dynamic_Cluster_CVPR_2024_paper.html))

Dự án tập trung giải quyết bài toán **Học liên tục trực tuyến không nhãn tác vụ (Online Task-Free Continual Learning - OTFCL)** trên cả hai bài toán:
- **Phân loại ảnh (Discriminative):** Dùng mạng MLP trên Split MNIST và Reduced ResNet-18 trên Split CIFAR-10.
- **Sinh ảnh không giám sát (Generative):** Dùng mô hình khuếch tán Improved DDPM (IDDPM).

Ghi chú chi tiết về phân tích lý thuyết và kết quả thực nghiệm: [docs/spec.md](docs/spec.md).

---

## 1. Cấu trúc thư mục

```text
├── configs/               # File cấu hình thí nghiệm (YAML)
├── docs/
│   └── spec.md            # Ghi chú phân tích bài báo & kết quả chi tiết
├── src/
│   ├── memory/            # Thuật toán bộ nhớ DCM (KDM, cụm bộ nhớ, cắt tỉa MPP)
│   ├── baselines/         # Bộ nhớ Reservoir Sampling làm đối chứng
│   ├── models/            # Mạng phân loại (MLP, ResNet) và mạng sinh (U-Net, IDDPM)
│   ├── trainers/          # Vòng lặp huấn luyện trực tuyến cho phân loại & sinh ảnh
│   └── utils/             # Hàm đo Accuracy, FID và vẽ biểu đồ
├── tests/                 # Unit tests kiểm tra các công thức toán (Eq. 1–14)
├── tools/                 # Script chạy thống kê khoảng cách & chạy toàn bộ thực nghiệm
├── run_experiment.py      # File chạy thực nghiệm chính từ dòng lệnh
└── requirements.txt
```

---

## 2. Cài đặt

Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

Chạy kiểm thử các module:
```bash
pytest
```

---

## 3. Cách chạy thực nghiệm

### 3.1. Chạy nhanh toàn bộ thực nghiệm & xuất biểu đồ
```bash
python tools/run_suite.py
```
Kết quả tổng hợp sẽ được lưu tại `results/suite/full_report.json` và các biểu đồ tại `results/figures/`:
* `fig3a_lambda_sweep.png`: Ảnh hưởng của ngưỡng $\lambda$ tới số lượng cụm và độ chính xác (tái hiện Hình 3a).
* `fig3b_dynamic_expansion_lam90.png`: Động thái mở rộng cụm khi dữ liệu đổi phân phối (tái hiện Hình 3b).
* `fig3c_memory_time.png`: So sánh thời gian xử lý bộ nhớ giữa DCM-SE và DCM-JS (tái hiện Hình 3c).

### 3.2. Chạy từng thí nghiệm phân loại (Split MNIST)

```bash
# 1. Finetune thông thường (không dùng bộ nhớ - baseline cận dưới)
python run_experiment.py --config configs/classification_mnist.yaml --set memory.type=none --seeds 0,1,2

# 2. Dùng bộ nhớ Reservoir Sampling (chọn mẫu ngẫu nhiên)
python run_experiment.py --config configs/classification_mnist.yaml --set memory.type=reservoir --seeds 0,1,2

# 3. Dùng DCM-SE (đo khoảng cách Euclidean bình phương)
python run_experiment.py --config configs/classification_mnist.yaml --set memory.type=dcm memory.kdm=se memory.lambda=75 --seeds 0,1,2

# 4. Dùng DCM-JS (đo phân kỳ Jensen-Shannon)
python run_experiment.py --config configs/classification_mnist.yaml --set memory.type=dcm memory.kdm=js memory.js_mode=dimwise memory.lambda=0.115 --seeds 0,1,2
```

### 3.3. Thử nghiệm sinh ảnh với IDDPM (Split MNIST)
```bash
# Chạy cấu hình kiểm tra nhanh trên CPU
python run_experiment.py --config configs/generation_mnist.yaml --set data.image_size=16 data.channels=1 data.max_samples_per_task=640 model.unet.base=32 eval.compute_fid=false --seeds 0
```

---

## 4. Tóm tắt kết quả chính (Split MNIST)

| Phương pháp | Bộ nhớ | Số cụm ($k$) | Thời gian quản lý bộ nhớ | Độ chính xác (Accuracy) |
| :--- | :---: | :---: | :---: | :---: |
| **Finetune (Không bộ nhớ)** | 0 | - | 0.0 s | 19.22% ± 0.19% |
| **Reservoir Sampling** | 2000 | - | ~1.0 s | 91.32% ± 0.91% |
| **DCM-SE ($\lambda=75$, Eq. 12 in trong bài)** | 1748 | 15 | ~5.5 s | 55.58% ± 6.33% |
| **DCM-SE ($\lambda=75$, lời văn trang 6)** | 1748 | 15 | ~6.5 s | 62.99% ± 12.83% |
| **DCM-SE + Quantile Pruning (nhóm đề xuất)** | 1748 | 15 | ~14.3 s | **72.41% ± 11.94%** *(Max 80.25%)* |

> Chi tiết phân tích nguyên nhân sự chênh lệch và các hiện tượng thú vị trong bài báo được trình bày trong [docs/spec.md](docs/spec.md).

---

## 5. Tài liệu tham khảo

```bibtex
@inproceedings{ye2024online,
  title={Online Task-Free Continual Generative and Discriminative Learning via Dynamic Cluster Memory},
  author={Ye, Fei and Bors, Adrian G.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={26202--26212},
  year={2024}
}
```
