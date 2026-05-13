# Hướng dẫn cài đặt và chạy

Implement lại hệ thống CBIR từ paper **"Investigating the Vision Transformer Model for Image Retrieval Tasks"** (arXiv:2101.03771).

Pipeline tốt nhất theo paper: **ViT-B16 → ROBUST normalization → Cosine distance**

---

## 1. Cài đặt môi trường

```bash
# Tạo virtualenv (khuyến nghị Python 3.8–3.10)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# hoặc: venv\Scripts\activate   # Windows

# Cài dependencies
pip install -r requirements.txt
```

> **Lưu ý phiên bản:** `vit-keras` yêu cầu TensorFlow 2.8–2.12.  
> Nếu dùng TF mới hơn, cài thêm: `pip install tensorflow==2.12`

---

## 2. Tải dataset

### INRIA Holidays
```bash
mkdir -p data/inria/jpg
# Tải từ: http://lear.inrialpes.fr/~jegou/data.php
# File: jpg1.tar.gz + jpg2.tar.gz → giải nén vào data/inria/jpg/
```

### UKBench
```bash
mkdir -p data/ukbench/full
# Tải từ: https://archive.org/details/ukbench
# File: ukbench.zip → giải nén, đặt tất cả ảnh vào data/ukbench/full/
```

### Oxford5K
```bash
mkdir -p data/oxford5k/images data/oxford5k/ground_truth
# Ảnh: https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
# GT:  https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz
# → giải nén vào data/oxford5k/images/ và data/oxford5k/ground_truth/
```

### Paris6K
```bash
mkdir -p data/paris6k/images data/paris6k/ground_truth
# Ảnh: https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz + paris_2.tgz
# GT:  https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_120310.tgz
# → giải nén vào data/paris6k/images/ và data/paris6k/ground_truth/
```

---

## 3. Kiểm tra nhanh với dữ liệu mẫu (100 ảnh)

Dùng để xác minh pipeline chạy đúng mà không cần tải toàn bộ dataset.

### Tạo dữ liệu mẫu
```bash
python3 create_sample.py
```

Kết quả tạo ra thư mục `data_sample/` với symlink (không tốn thêm ổ đĩa):

| Dataset  | Số ảnh mẫu | Cấu trúc                                      |
|----------|-----------|-----------------------------------------------|
| UKBench  | 100 ảnh   | 25 nhóm × 4 ảnh, giữ nguyên cấu trúc nhóm    |
| Oxford5k | 100 ảnh   | 5 query đầu + ảnh relevant + distractor       |
| Paris6k  | 100 ảnh   | 5 query đầu + ảnh relevant + distractor       |

### Chạy đánh giá trên dữ liệu mẫu
```bash
python3 run_evaluate.py --dataset ukbench  --data_dir data_sample/ukbench
python3 run_evaluate.py --dataset oxford5k --data_dir data_sample/oxford5k
python3 run_evaluate.py --dataset paris6k  --data_dir data_sample/paris6k
```

> **Lưu ý:** Kết quả trên dữ liệu mẫu sẽ khác so với kết quả paper vì số lượng ảnh ít hơn nhiều. Mục đích chỉ để kiểm tra pipeline hoạt động đúng.

---

## 4. Đánh giá trên dataset

### INRIA Holidays (mAP)
```bash
python run_evaluate.py \
    --dataset inria \
    --data_dir data/inria \
    --model vit_b16 \
    --norm robust \
    --metric cosine \
    --cache cache/inria_vit_b16.npy
```
**Kết quả kỳ vọng:** mAP ≈ 88.0

### UKBench (N-S score)
```bash
python run_evaluate.py \
    --dataset ukbench \
    --data_dir data/ukbench \
    --model vit_b16 \
    --norm robust \
    --metric cosine \
    --cache cache/ukbench_vit_b16.npy
```
**Kết quả kỳ vọng:** N-S ≈ 3.759

### Paris6K (mAP)
```bash
python run_evaluate.py \
    --dataset paris6k \
    --data_dir data/paris6k \
    --model vit_b16 \
    --norm robust \
    --metric cosine \
    --cache cache/paris6k_vit_b16.npy
```
**Kết quả kỳ vọng:** mAP ≈ 87.83

### Oxford5K (mAP)
```bash
python run_evaluate.py \
    --dataset oxford5k \
    --data_dir data/oxford5k \
    --model vit_b16 \
    --norm robust \
    --metric cosine \
    --cache cache/oxford5k_vit_b16.npy
```
**Kết quả kỳ vọng:** mAP ≈ 64.68

---

## 5. Demo nhanh với ảnh bất kỳ

```bash
python demo.py \
    --query path/to/query.jpg \
    --db_dir path/to/image_folder \
    --top_k 5
```

Kết quả được lưu vào `demo_result.png`.

---

## 6. Cấu trúc dự án

```
vit_cbir/
├── config.py           # Hằng số cấu hình toàn cục
├── run_evaluate.py     # CLI đánh giá
├── demo.py             # Demo trực quan
├── create_sample.py    # Tạo dữ liệu mẫu 100 ảnh để test nhanh
├── requirements.txt
├── core/
│   ├── extractor.py    # Load ViT model, trích xuất feature vector
│   ├── normalizer.py   # ROBUST, L1/L2 normalization
│   ├── retrieval.py    # Cosine/Euclidean distance, ranking
│   └── metrics.py      # mAP và N-S score
├── datasets/
│   ├── inria.py        # INRIA Holidays loader
│   ├── ukbench.py      # UKBench loader
│   └── oxford_paris.py # Oxford5K / Paris6K loader
├── data/               # Dataset đầy đủ (không commit lên git)
│   ├── ukbench/full/
│   ├── oxford5k/images/ và ground_truth/
│   └── paris6k/images/ và ground_truth/
└── data_sample/        # Dữ liệu mẫu 100 ảnh (symlink, không commit)
    ├── ukbench/
    ├── oxford5k/
    └── paris6k/
```

---

## 7. Thay đổi cấu hình thực nghiệm

Có thể thử các cấu hình khác bằng cách thay đổi tham số:

| Tham số   | Giá trị                                              |
|-----------|------------------------------------------------------|
| `--model` | `vit_b16`, `vit_b32`, `vit_l16`, `vit_l32`         |
| `--norm`  | `robust`, `l2_axis1`, `l1_axis1`, `l2_axis0`, `l1_axis0` |
| `--metric`| `cosine`, `euclidean`, `cityblock`, `braycurtis`, `canberra`, `chebyshev`, `correlation` |

Các kết quả đầy đủ cho tất cả cấu hình xem tại Bảng I–IV trong paper.
