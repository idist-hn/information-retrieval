# Bảng Phân Công Công Việc — ViT CBIR

**Đề tài:** Investigating the Vision Transformer Model for Image Retrieval Tasks (arXiv:2101.03771)
**Môn học:** Truy xuất thông tin
**Nhóm:** 5 người

---

## Danh sách thành viên

| STT | Họ và tên            | MSSV       | Vai trò chính                     |
| --- | -------------------- | ---------- | --------------------------------- |
| 1   | Vũ Minh Toàn         | B25CHHT056 | Trưởng nhóm / Lập trình (code)    |
| 2   | Đinh Quang Hiến      | B25CHHT018 | Lập trình (code) / Viết báo cáo  |
| 3   | Phan Tiến Thành      | B25CHHT053 | Thực nghiệm / Slides thuyết trình |
| 4   | Đỗ Thanh Hường       | NCS2025.4  | Nghiên cứu lý thuyết / Báo cáo   |
| 5   | Dương Thị Thu Phương | B25CHHT047 | Viết báo cáo / Chuẩn bị dữ liệu  |

---

## Bảng phân công chi tiết

| STT | Công việc                          | Người phụ trách      | MSSV       | Mô tả chi tiết                                                                                                                                                 |
| --- | ---------------------------------- | -------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Nghiên cứu bài báo gốc             | Đỗ Thanh Hường       | NCS2025.4  | Nắm rõ cơ chế hoạt động của Vision Transformer (ViT), cách áp dụng vào bài toán CBIR; hiểu các phương pháp chuẩn hóa descriptor và các độ đo khoảng cách được so sánh trong paper |
| 2   | Cài đặt ViT Feature Extractor      | Đinh Quang Hiến      | B25CHHT018 | Xây dựng pipeline trích xuất đặc trưng ảnh: tiền xử lý ảnh (resize 384×384, chuẩn hóa [-1,1]), đưa qua mô hình ViT lấy vector descriptor, hỗ trợ cache kết quả để tránh trích xuất lại |
| 3   | Cài đặt Normalizer, Retrieval & Demo | Đinh Quang Hiến    | B25CHHT018 | Cài đặt các phương pháp chuẩn hóa descriptor (ROBUST, L2, L1); xây dựng chức năng tìm kiếm ảnh tương tự theo nhiều độ đo khoảng cách; xây dựng giao diện demo trực quan |
| 4   | Thiết kế kiến trúc & tích hợp      | Vũ Minh Toàn         | B25CHHT056 | Thiết kế cấu trúc tổng thể của hệ thống; kết nối các thành phần (trích xuất, chuẩn hóa, tìm kiếm, đánh giá) thành pipeline hoàn chỉnh; cài đặt các độ đo đánh giá mAP và N-S score |
| 5   | Chuẩn bị dữ liệu                   | Dương Thị Thu Phương | B25CHHT047 | Tải về và tổ chức các bộ dataset Oxford5K, INRIA Holidays, UKBench, Paris6K; kiểm tra tính đầy đủ của ảnh và file ground truth; ghi chép thống kê cơ bản của từng bộ dữ liệu |
| 6   | Thực nghiệm & ghi kết quả          | Phan Tiến Thành      | B25CHHT053 | Chạy đánh giá hệ thống trên từng bộ dataset với các cấu hình khác nhau (model, normalization, metric); ghi lại kết quả mAP và N-S score; so sánh với kết quả công bố trong paper |
| 7   | Viết báo cáo (lý thuyết)           | Đỗ Thanh Hường       | NCS2025.4  | Trình bày tổng quan bài toán CBIR, kiến trúc ViT và cơ chế self-attention, lý do ViT phù hợp với truy xuất ảnh, các phương pháp chuẩn hóa và độ đo khoảng cách |
| 8   | Viết báo cáo (thực nghiệm)         | Dương Thị Thu Phương | B25CHHT047 | Trình bày thiết lập thực nghiệm, mô tả các bộ dataset và giao thức đánh giá, phân tích kết quả thu được, so sánh và rút ra nhận xét; tổng hợp hoàn thiện toàn bộ báo cáo |
| 9   | Viết báo cáo (phần hệ thống)       | Đinh Quang Hiến      | B25CHHT018 | Mô tả kiến trúc tổng thể của hệ thống đã xây dựng, giải thích từng bước trong pipeline từ ảnh đầu vào đến kết quả truy xuất                                     |
| 10  | Slides & thuyết trình              | Phan Tiến Thành      | B25CHHT053 | Xây dựng bộ slide thuyết trình đủ nội dung (lý thuyết, hệ thống, kết quả, demo); phân chia phần trình bày cho từng thành viên; chuẩn bị chạy demo trực tiếp    |

---

## Phân chia theo module source code

> Chỉ **Đinh Quang Hiến** và **Vũ Minh Toàn** thực hiện công việc lập trình.

| Module / File              | Người phụ trách | MSSV       |
| -------------------------- | --------------- | ---------- |
| `config.py`                | Vũ Minh Toàn   | B25CHHT056 |
| `core/types.py`            | Vũ Minh Toàn   | B25CHHT056 |
| `core/metrics.py`          | Vũ Minh Toàn   | B25CHHT056 |
| `run_evaluate.py`          | Vũ Minh Toàn   | B25CHHT056 |
| `core/extractor.py`        | Đinh Quang Hiến | B25CHHT018 |
| `core/pipeline.py`         | Đinh Quang Hiến | B25CHHT018 |
| `core/normalizer.py`       | Đinh Quang Hiến | B25CHHT018 |
| `core/retrieval.py`        | Đinh Quang Hiến | B25CHHT018 |
| `datasets/inria.py`        | Đinh Quang Hiến | B25CHHT018 |
| `datasets/ukbench.py`      | Đinh Quang Hiến | B25CHHT018 |
| `datasets/oxford_paris.py` | Đinh Quang Hiến | B25CHHT018 |
| `utils/visualization.py`   | Đinh Quang Hiến | B25CHHT018 |
| `demo.py`                  | Đinh Quang Hiến | B25CHHT018 |

---

## Tiến độ dự kiến

| Giai đoạn   | Công việc                              | Deadline |
| ----------- | -------------------------------------- | -------- |
| Giai đoạn 1 | Nghiên cứu bài báo, thiết kế kiến trúc | Tuần 1   |
| Giai đoạn 2 | Cài đặt tất cả các module              | Tuần 2–3 |
| Giai đoạn 3 | Tích hợp, chạy thực nghiệm, debug      | Tuần 4   |
| Giai đoạn 4 | Viết báo cáo, làm slides               | Tuần 5   |
| Giai đoạn 5 | Hoàn thiện, thuyết trình               | Tuần 6   |
