# Ghi Chú Phân Tích Bài Báo & Kết Quả Thực Nghiệm

Tài liệu này tổng hợp phân tích lý thuyết, các điểm cần lưu ý khi cài đặt và kết quả thực nghiệm thu được khi tái hiện bài báo:  
**"Online Task-Free Continual Generative and Discriminative Learning via Dynamic Cluster Memory"** (Fei Ye, Adrian G. Bors — CVPR 2024).

---

## 1. Bài toán: Online Task-Free Continual Learning (OTFCL)

* **Online:** Dữ liệu tới theo từng batch nhỏ $b$, lướt qua một lần và không được lưu lại toàn bộ để train nhiều epoch.
* **Task-Free:** Mô hình không được cung cấp Task ID hay tín hiệu ranh giới khi đổi phân phối.
* **Mục tiêu:** Vừa phân loại tốt (tránh quên thảm khốc), vừa sinh được ảnh của các phân phối cũ mà không cần giám sát.

---

## 2. Tóm tắt Phương pháp Dynamic Cluster Memory (DCM)

Bộ nhớ $\mathcal{M}$ được chia thành $k$ cụm ($k \le \rho$), mỗi cụm chứa tối đa $\pi$ mẫu:

1. **Thước đo KDM ($F_{\text{KDM}}$):**
   * **DCM-SE:** Khoảng cách Euclidean bình phương $\|\mathbf{x}_g - \mathbf{x}_h\|_2^2$ trên không gian ảnh.
   * **DCM-JS:** Cho ảnh qua quá trình khuếch tán thuận tại bước $t^\star=100$, đo phân kỳ Jensen-Shannon giữa hai phân phối Gaussian.
2. **Khởi tạo & Tìm Prototype (Eq. 9):**
   * Cụm đầu tiên lấy từ batch $t_1$. Prototype $\widehat{\mathbf{x}}^k$ là mẫu medoid có tổng khoảng cách tới các mẫu khác trong cụm nhỏ nhất. Prototype chỉ tính 1 lần lúc tạo cụm để tiết kiệm thời gian.
3. **Mở rộng cụm (Eq. 10):**
   * Nếu khoảng cách từ tất cả các mẫu trong batch tới mọi prototype đều lớn hơn ngưỡng $\lambda$, hệ thống tự động tạo cụm mới và lấy mẫu dị biệt nhất làm prototype mới.
4. **Gán mẫu & Cắt tỉa nội cụm (Eq. 11 & 12):**
   * Mẫu quen được gán vào cụm có prototype gần nhất.
   * Khi cụm đầy ($>\pi$), một mẫu sẽ bị loại bỏ để duy trì kích thước.
5. **Cắt tỉa liên cụm MPP (Eq. 13 & 14):**
   * Khi số cụm vượt quá trần $\rho$, tính ma trận tương đồng $B[g, h] = 1 / (F_{\text{KDM}} + \varepsilon)$.
   * Tìm cặp cụm chồng lấn nhiều nhất và xóa cụm có mức độ dư thừa cao hơn so với phần còn lại của bộ nhớ.

---

## 3. Các điểm bất nhất trong bài báo & Cách xử lý

Khi đọc kỹ bài báo và kiểm tra công thức, có 4 điểm mâu thuẫn giữa mô tả văn bản và ký hiệu in trong bài:

| Vấn đề | Trong bài báo | Thực tế xử lý |
| :--- | :--- | :--- |
| **Gán mẫu (Eq. 11 vs Algo 1)** | Văn bản ghi chọn cụm gần nhất ($\arg\min$), nhưng Algorithm 1 dòng 8 in nhầm thành $\arg\max$. | Dùng $\arg\min$ theo đúng bản chất gom cụm. |
| **Cắt tỉa khi cụm đầy (Eq. 12)** | Văn bản trang 6 ghi *"xóa mẫu gần prototype nhất"* để giữ tính đa dạng. Nhưng công thức Eq. 12 lại in $\arg\max$ (*xóa mẫu xa prototype nhất*). | Cài đặt cả 2 tùy chọn (`nearest` và `farthest`) để đối chứng thực nghiệm, đồng thời bổ sung thêm thuật toán `quantile` để khắc phục điểm yếu của cả hai. |
| **Cắt tỉa liên cụm MPP (Eq. 13-14)** | Văn bản ghi ghép cặp cụm giống nhau nhất (khoảng cách nhỏ nhất), nhưng Eq. 13 in $\arg\max F_{\text{KDM}}$. | Dùng logic chuẩn: tìm cặp có khoảng cách nhỏ nhất (tương đồng nhất) và loại bỏ cụm dư thừa hơn. |
| **Bão hòa phân kỳ JS (Eq. 8)** | Trong không gian ảnh nhiều chiều, phân kỳ JS giữa 2 Gaussian phân tách xa dễ bị bão hòa ở hằng số $\ln 2 \approx 0.6931$. | Bổ sung chế độ `js_mode="dimwise"` (tính trung bình JS theo từng pixel) để thước đo không bị bão hòa. |

---

## 4. Kết quả Thực nghiệm (Split MNIST)

Thực nghiệm chạy trên 3 seed ngẫu nhiên (`0, 1, 2`) theo cấu hình $b=10$, $\pi=128$, bộ nhớ tối đa $2000$ mẫu:

| Phương pháp | Chế độ cắt tỉa nội cụm | Số cụm ($k$) | Bộ nhớ | Thời gian bộ nhớ | Độ chính xác (Accuracy) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Finetune** (Không bộ nhớ) | — | — | 0 | 0.0 s | **19.22% ± 0.19%** *(Paper: 19.75%)* |
| **Reservoir Sampling** | Ngẫu nhiên (Vitter) | — | 2000 | 1.0 s | **91.32% ± 0.91%** *(Paper: 92.16%)* |
| **DCM-SE** ($\lambda=75$, Eq. 12 in) | `farthest` ($\arg\max$) | 15 | 1748 | 5.5 s | **55.58% ± 6.33%** |
| **DCM-SE** ($\lambda=75$, lời văn) | `nearest` ($\arg\min$) | 15 | 1748 | 6.5 s | **62.99% ± 12.83%** |
| **DCM-SE + Quantile (Đề xuất)** | `quantile` | 15 | 1748 | 14.3 s | **72.41% ± 11.94%** *(Seed 1: **80.25%**)* |
| **DCM-JS** (Exact, $t^\star=100$) | `nearest` | 15 | 142 | 18.9 s | **38.49% ± 0.71%** *(Bị bão hòa $\ln 2$)* |

---

## 5. Nhận xét & Phát hiện chính

1. **Về mâu thuẫn ở Eq. 12 (`nearest` vs `farthest`):**
   * **Khi dùng `farthest`:** Các cụm co cụm lại rất chặt quanh prototype. Các mẫu thuộc lớp có nét mảnh (ví dụ chữ số `7`) khi xuất hiện ở task sau bị xem là xa tâm và bị xóa liên tục. Kết quả là lớp số 7 chỉ còn sót lại đúng 2 mẫu trong bộ nhớ, kéo tụt accuracy xuống 55.58%.
   * **Khi dùng `nearest`:** Các mẫu gần tâm bị đẩy ra ngoài. Với các lớp ít pixel sáng như số `1`, chúng bị các lớp khác lấn chiếm và xóa sạch khỏi bộ nhớ.
   * **Khi dùng `quantile`:** Thay vì xóa ở hai đầu cực trị, phương pháp này xóa mẫu có khoảng cách gần nhất với mẫu lân cận (thu hẹp khoảng trống dư thừa), giúp mẫu phân bố đều từ tâm ra biên. Nhờ đó accuracy tăng lên 72.41% (đạt 80.25% ở seed 1).
2. **Về hành vi mở rộng cụm động (Hình 3a, 3b):**
   * Khi tăng $\lambda$ từ $65 \to 95$, số cụm tạo ra giảm đều từ $15 \to 4$.
   * Với $\lambda=90$, số lượng cụm tự động nhảy bậc theo từng task mới (tạo ra 6 cụm cho 5 task), chứng minh DCM nhận biết được sự thay đổi phân phối mà hoàn toàn không cần Task ID.
3. **Về thời gian thực thi (Hình 3c):**
   * Nhờ cơ chế lưu đệm khoảng cách prototype (`proto_dists`) và cập nhật tại chỗ (`in-place slot eviction`), việc quản lý bộ nhớ trên toàn bộ 60.000 mẫu của MNIST chỉ mất khoảng 5–6 giây với `DCM-SE`.
