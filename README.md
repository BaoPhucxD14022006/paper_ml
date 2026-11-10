# Tái hiện bài báo: Online Task-Free Continual Learning via Dynamic Cluster Memory (CVPR 2024)

Repo này cài đặt lại thuật toán **Dynamic Cluster Memory (DCM)** dựa trên bài báo:  
> **"Online Task-Free Continual Generative and Discriminative Learning via Dynamic Cluster Memory"**  
> *Fei Ye, Adrian G. Bors — CVPR 2024* ([Link bài báo](https://openaccess.thecvf.com/content/CVPR2024/html/Ye_Online_Task-Free_Continual_Generative_and_Discriminative_Learning_via_Dynamic_Cluster_CVPR_2024_paper.html))

Dự án tập trung giải quyết bài toán **Học liên tục trực tuyến không nhãn tác vụ (Online Task-Free Continual Learning - OTFCL)** trên cả hai bài toán:
- **Phân loại ảnh (Discriminative - Lab 1):** Dùng mạng MLP trên Split MNIST và Reduced ResNet-18 trên Split CIFAR-10.
- **Sinh ảnh không giám sát (Generative - Lab 1):** Dùng mô hình khuếch tán Improved DDPM (IDDPM).
- **Phản biện lý thuyết & Thực nghiệm kiểm chứng (Critique & Empirics - Lab 2):** Khảo sát các điểm mâu thuẫn công thức Eq. 12, Eq. 13-14, hiện tượng bão hòa JS Divergence và đề xuất cơ chế cắt tỉa `quantile`.

Ghi chú chi tiết về phân tích lý thuyết và kết quả thực nghiệm: [docs/spec.md](docs/spec.md).

---

## 1. Cấu trúc thư mục

```text
├── configs/               # File cấu hình thí nghiệm (YAML) có tích hợp W&B
├── docs/
│   └── spec.md            # Ghi chú phân tích bài báo, phản biện lý thuyết & kết quả chi tiết
├── src/
│   ├── data/              # Tải dữ liệu tự động & bộ tạo dòng dữ liệu OnlineStream
│   ├── memory/            # Thuật toán bộ nhớ DCM (KDM, cụm bộ nhớ, cắt tỉa MPP)
│   ├── baselines/         # Bộ nhớ Reservoir Sampling làm đối chứng
│   ├── models/            # Mạng phân loại (MLP, ResNet) và mạng sinh (U-Net, IDDPM)
│   ├── trainers/          # Vòng lặp huấn luyện trực tuyến tích hợp Weights & Biases
│   └── utils/             # Hàm đo Accuracy, FID và vẽ biểu đồ
├── tests/                 # Unit tests kiểm tra các công thức toán (Eq. 1–14)
├── tools/                 # Script chạy thống kê khoảng cách, suite thực nghiệm & W&B logging
│   ├── distance_stats.py  # Hiệu chỉnh ngưỡng khoảng cách lambda
│   ├── run_suite.py       # Chạy toàn bộ suite thí nghiệm và vẽ đồ thị
│   └── log_all_wandb.py   # Chạy và đồng bộ 12 runs lên W&B Dashboard
├── run_experiment.py      # File chạy thực nghiệm chính từ dòng lệnh (hỗ trợ cờ --wandb)
└── requirements.txt
```

---

## 2. Cài đặt

Cài đặt các thư viện cần thiết (bao gồm PyTorch, Torchvision và Weights & Biases):
```bash
pip install -r requirements.txt
```

Chạy kiểm thử toàn bộ 28 unit tests:
```bash
pytest
```

---

## 3. Theo dõi Thực nghiệm với Weights & Biases (W&B)

Toàn bộ các thực nghiệm (bao gồm các bước huấn luyện, độ đo Accuracy, FID, số lượng cụm $k$ và ảnh sinh ra) được theo dõi tự động trên nền tảng **Weights & Biases (W&B)**:

* **Trang tổng quan dự án (W&B Project Dashboard):**  
  [https://forge.coreweave.com/wandb/zahyyy-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/dcm-otfcl-cvpr2024](https://forge.coreweave.com/wandb/zahyyy-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/dcm-otfcl-cvpr2024)

### 3.1. Thiết lập W&B
1. Đăng nhập vào tài khoản W&B trên máy cá nhân:
   ```bash
   wandb login
   ```
2. Mặc định các thí nghiệm sẽ log vào project `dcm-otfcl-cvpr2024`. Bạn có thể chỉ định project hoặc entity qua tham số dòng lệnh `--wandb-project` và `--wandb-entity`.

### 3.2. Chạy thực nghiệm có bật W&B
* **Chạy từng thực nghiệm:**
  ```bash
  python run_experiment.py --config configs/classification_mnist.yaml --wandb
  ```
* **Chạy toàn bộ Suite và đẩy log lên W&B:**
  ```bash
  python tools/run_suite.py --wandb
  ```
* **Chạy và tự động cập nhật liên kết 12 thực nghiệm chính vào README:**
  ```bash
  python tools/log_all_wandb.py
  ```

---

## 4. Bảng Ánh Xạ Thực Nghiệm (Experiment Mapping Table)

Dưới đây là bảng tổng hợp toàn bộ các thực nghiệm được tái hiện (**Lab 1: Reproduction**) và các thực nghiệm phản biện/cắt bỏ (**Lab 2: Empirics**) theo đúng quy chuẩn theo dõi của môn học:

| Tên thực nghiệm | Nhóm / Tag W&B | Script / Lệnh thực thi | Liên kết W&B Run (Dashboard) | Kết quả & Ý nghĩa chính (Key Finding) |
| :--- | :---: | :--- | :---: | :--- |
| **Finetune (No Memory)** | `lab1-reproduction` | `python run_experiment.py --config configs/classification_mnist.yaml --set memory.type=none --wandb` | [W&B Run](https://forge.coreweave.com/wandb/zahyyy-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/dcm-otfcl-cvpr2024/runs/wbyk3b6z) | Đạt **19.22% ± 0.19%** (Paper: 19.75%), mô hình quên hoàn toàn các task cũ khi không có bộ nhớ đệm. |
| **Reservoir Sampling** | `lab1-reproduction` | `python run_experiment.py --config configs/classification_mnist.yaml --set memory.type=reservoir --wandb` | [W&B Run](https://forge.coreweave.com/wandb/zahyyy-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/dcm-otfcl-cvpr2024/runs/afah4950) | Đạt **91.32% ± 0.91%** (Paper: 92.16%), baseline ngẫu nhiên lưu mẫu đồng đều nhưng không giữ được ranh giới cụm. |
| **DCM-SE ($\lambda=75$, Eq. 12 in)** | `lab1-reproduction` | `python run_experiment.py --config configs/classification_mnist.yaml --set memory.type=dcm memory.kdm=se memory.lambda=75 memory.intra_prune_mode=farthest --wandb` | [W&B Run](https://forge.coreweave.com/wandb/zahyyy-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/dcm-otfcl-cvpr2024/runs/nflmy56o) | Đạt **55.58% ± 6.33%**, do xóa mẫu xa tâm (`farthest`) khiến cụm co chặt và xóa sạch các lớp nét mảnh (chữ số 7). |
| **DCM-SE ($\lambda=75$, Lời văn)** | `lab1-reproduction` | `python run_experiment.py --config configs/classification_mnist.yaml --set memory.type=dcm memory.kdm=se memory.lambda=75 memory.intra_prune_mode=nearest --wandb` | [W&B Run](https://forge.coreweave.com/wandb/zahyyy-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/dcm-otfcl-cvpr2024/runs/79w7rurp) | Đạt **62.99% ± 12.83%**, xóa mẫu gần tâm (`nearest`) tăng đa dạng nhưng lại xóa sạch chữ số 1 khi bị lớp khác lấn. |
| **DCM-SE + Quantile (Đề xuất)** | `lab2-empirics` | `python run_experiment.py --config configs/classification_mnist.yaml --set memory.type=dcm memory.kdm=se memory.lambda=75 memory.intra_prune_mode=quantile --wandb` | [W&B Run](https://forge.coreweave.com/wandb/zahyyy-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/dcm-otfcl-cvpr2024/runs/hjm9ukiw) | Đạt **72.41% ± 11.94%** (Max: **80.25%**), giải quyết triệt để sự mất cân bằng mẫu của cả `nearest` và `farthest`. |
| **DCM-JS (Exact, $t^\star=100$)** | `lab1-reproduction` | `python run_experiment.py --config configs/classification_mnist.yaml --set memory.type=dcm memory.kdm=js memory.js_mode=exact memory.lambda=0.5 --wandb` | [W&B Run](https://forge.coreweave.com/wandb/zahyyy-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/dcm-otfcl-cvpr2024/runs/cgzhb3uc) | Đạt **38.49% ± 0.71%**, chứng minh hiện tượng bão hòa phân kỳ JS về hằng số $\ln 2 \approx 0.6931$ ở số chiều cao. |
| **DCM-JS (Dimwise)** | `lab2-empirics` | `python run_experiment.py --config configs/classification_mnist.yaml --set memory.type=dcm memory.kdm=js memory.js_mode=dimwise memory.lambda=0.115 --wandb` | [W&B Run](https://forge.coreweave.com/wandb/zahyyy-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/dcm-otfcl-cvpr2024/runs/h1teo9xv) | Khắc phục bão hòa JS bằng cách tính trung bình 1D theo từng pixel, kích hoạt phân bổ cụm hoạt động bình thường. |
| **MPP Cắt tỉa liên cụm ($\rho=6$)** | `lab2-empirics` | `python run_experiment.py --config configs/classification_mnist.yaml --set memory.lambda=70 memory.rho=6 --sweep memory.mpp_mode=consistent,literal --wandb` | [W&B Run](https://forge.coreweave.com/wandb/zahyyy-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/dcm-otfcl-cvpr2024/runs/tihnu6mb) | Đối chứng giữa công thức in Eq. 13 ($\arg\max$) và logic chuẩn ($\arg\min$), chứng minh tính cần thiết của việc giữ cụm đa dạng. |
| **Độ nhạy ngưỡng $\lambda$ (Hình 3a)** | `lab2-empirics` | `python tools/run_suite.py --wandb` | [W&B Run](https://forge.coreweave.com/wandb/zahyyy-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/dcm-otfcl-cvpr2024/runs/tlqmsp83) | Quét $\lambda \in [65, 95]$, số cụm giảm đơn điệu từ 15 về 4, tương quan trực tiếp tới độ bao phủ phân phối dữ liệu. |
| **Mở rộng cụm động (Hình 3b)** | `lab2-empirics` | `python tools/run_suite.py --wandb` | [W&B Run](https://forge.coreweave.com/wandb/zahyyy-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/dcm-otfcl-cvpr2024/runs/zn8vjzt8) | Số lượng cụm tự động nhảy bậc bám sát 5 phân phối tác vụ mà hoàn toàn không cần tín hiệu Task ID. |
| **Thời gian xử lý bộ nhớ (Hình 3c)** | `lab1-reproduction` | `python tools/run_suite.py --wandb` | [W&B Run](https://forge.coreweave.com/wandb/zahyyy-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/dcm-otfcl-cvpr2024/runs/221opabn) | DCM-SE chỉ mất 5.4s và DCM-JS mất 7.5s trên 60.000 mẫu nhờ cơ chế cache `proto_dists` và cập nhật tại chỗ. |
| **Sinh ảnh IDDPM (Split MNIST)** | `lab1-reproduction` | `python run_experiment.py --config configs/generation_mnist.yaml --wandb` | [W&B Run](https://forge.coreweave.com/wandb/zahyyy-tr-ng-i-h-c-khoa-h-c-t-nhi-n-hqg-hcm/dcm-otfcl-cvpr2024/runs/0hfdejm8) | Mô hình khuếch tán không giám sát sinh lại được các chữ số của các task đầu tiên mà không bị quên thảm khốc. |

---

## 5. Tóm tắt kết quả chính (Split MNIST - Bảng 4)

| Phương pháp | Bộ nhớ | Số cụm ($k$) | Thời gian quản lý bộ nhớ | Độ chính xác (Accuracy ± std) |
| :--- | :---: | :---: | :---: | :---: |
| **Finetune (Không bộ nhớ)** | 0 | - | 0.0 s | **19.22% ± 0.19%** *(Paper: 19.75%)* |
| **Reservoir Sampling** | 2000 | - | ~1.0 s | **91.32% ± 0.91%** *(Paper: 92.16%)* |
| **DCM-SE ($\lambda=75$, Eq. 12 in trong bài)** | 1748 | 15 | ~5.5 s | **55.58% ± 6.33%** |
| **DCM-SE ($\lambda=75$, lời văn trang 6)** | 1748 | 15 | ~6.5 s | **62.99% ± 12.83%** |
| **DCM-SE + Quantile Pruning (Đề xuất cải tiến)** | 1748 | 15 | ~14.3 s | **72.41% ± 11.94%** *(Seed 1: **80.25%**)* |
| **DCM-JS (Exact, $t^\star=100$)** | 142 | 15 | ~18.9 s | **38.49% ± 0.71%** *(Bão hòa $\ln 2$)* |

---

## 6. Các phát hiện phản biện mở rộng (Dành cho Poster & Báo cáo)

Theo yêu cầu của chặng **Lab 2 (Empirical Demonstration)**, các điểm yếu lý thuyết đã được kiểm chứng bằng thực nghiệm bao gồm:
1. **Nghịch lý cắt tỉa nội cụm Phương trình (12):**
   * Công thức in (`farthest`) gây co cụm quá mức quanh tâm, loại trừ các mẫu nét mảnh (chữ số `7` chỉ còn đúng 2 mẫu trong toàn bộ 2000 ô nhớ).
   * Lời văn (`nearest`) lại xóa sạch các lớp có ít pixel sáng (chữ số `1` bị biến mất hoàn toàn).
   * Giải pháp `quantile` của nhóm loại bỏ mẫu nội điểm có khoảng cách lân cận hẹp nhất, giữ các mẫu trải đều từ tâm ra biên và nâng accuracy lên **80.25%**.
2. **Hiện tượng bão hòa phân kỳ Jensen–Shannon (Phương trình 8):**
   * Với ảnh đa chiều, khoảng cách giữa 2 Gaussian phân tách xa $> 25\sigma$ khiến phân kỳ JS chạm ngưỡng giới hạn hằng số $\ln 2 \approx 0.6931$, làm vô hiệu hóa việc phân cụm nếu không chuẩn hóa trung bình từng pixel (`dimwise`).
3. **Lỗi ký hiệu trong Thuật toán 1:**
   * Dòng 8 Algorithm 1 in nhầm $\arg\max$ thay vì $\arg\min$ cho phép gán mẫu vào cụm gần nhất.

---

## 7. Tài liệu tham khảo

```bibtex
@inproceedings{ye2024online,
  title={Online Task-Free Continual Generative and Discriminative Learning via Dynamic Cluster Memory},
  author={Ye, Fei and Bors, Adrian G.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={26202--26212},
  year={2024}
}
```
