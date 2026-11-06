# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202604100
- **Họ và tên:** Lê Bá Nguyên 
- **SEED** (in ở ô đầu notebook):8616

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.797      0.887        0.889   
2. BoW + MLP (deep learning)         1.000   0.911      0.954        0.974   
3a. Encoder (MiniLM) + kNN           0.893   0.949      0.920        1.000   
3b. Encoder (MiniLM) + LogReg        0.974   0.949      0.962        0.974   
3c. Encoder (MiniLM) + MLP head      0.962   0.962      0.962        1.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)           10.476      11105.199       0.050   
2. BoW + MLP (deep learning)        13.898      16102.938       0.379   
3a. Encoder (MiniLM) + kNN         125.653         20.445      50.276   
3b. Encoder (MiniLM) + LogReg      126.199         27.923      26.560   
3c. Encoder (MiniLM) + MLP head    126.706         28.168      27.141   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.090           0.0  
2. BoW + MLP (deep learning)          6.475           0.0  
3a. Encoder (MiniLM) + kNN           68.179           0.0  
3b. Encoder (MiniLM) + LogReg        39.533           0.0  
3c. Encoder (MiniLM) + MLP head      40.339           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

1.ml_lr (learning rate): Dự đoán lr=50 làm loss dao động mạnh và loss cuối cao. Thực tế loss nhảy lên ~2,7 ở vài epoch đầu rồi rơi xuống 0.007 (thấp nhất trong ba mức), không dao động. Lệch vì gradient của mỗi từ rất nhỏ (mỗi từ chỉ có trong ít tin, lại chia cho n) và dữ liệu train gần như tách được, nên sigmoid bão hòa và loss thấp; đây là loss train nên có thể là học thuộc.
2.dl_curve (MLP so với LogReg): Dự đoán MLP chỉ hơn LogReg một chút. Trên SUB 40 tin, MLP đạt F1 0.974 so với 0.889 của LogReg, chênh hơn dự đoán. Phần val loss tăng lại sau epoch 9 thì đúng dự đoán (overfit nhẹ). Lệch vì SUB chỉ có 40 tin, một tin sai đổi 1 − F1 khoảng 2,5 điểm, nên chênh lệch này có thể chỉ là nhiễu.
3.tradeoff: Dự đoán F1 của các cách khá gần nhau. Thực tế BoW + LogReg lỗi ~11% (1 − F1) trong khi các cách còn lại ~0-2,5%, tức rõ rệt kém hơn. Lệch vì mình đánh giá thấp điểm yếu của BoW ở các tin không có từ khoá quen.

## 4. Phân tích 50 case khó

BoW + LogReg tụt nhiều nhất: F1 từ 0.889 xuống 0.333; BoW + MLP từ 0.974 xuống 0.429. Các model embedding tụt ít hơn (0.64-0.70). Test thường quá lạc quan vì phần lớn tin dễ và cùng phân phối với tập train.

Case #12 (Spam giọng tự nhiên): "hi there, sorry to bother you, my friend gave me your number, I'm new in town..." Cả 5 model đều sai (cùng 5 case spam giọng tự nhiên #12, 13, 14, 16, 19). Không có từ khoá hay mẫu spam quen thuộc nên BoW không có tín hiệu, còn embedding đặt nó gần cụm tin nhắn đời thường vì giọng văn thân thiện.
Case #31 (Né từ khoá): "U have been s.e.l.e.c.t.e.d for a c.a.s.h r.e.w.a.r.d..." BoW sai vì dấu chấm chen giữa làm các token không còn khớp từ điển; Emb + kNN và Emb + LogReg vẫn đúng vì embedding vẫn nắm được nghĩa tổng thể; Emb + MLP head sai.

BoW đạt 100% ở nhóm "Nói về spam" chỉ vì nó gần như luôn đoán ham, không phải vì hiểu ngữ cảnh.

## 5. Thử thách cá nhân

Đề #1 theo SEED 8616 (8616 mod 6 = 0): đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1, báo precision/recall/F1 từng ngưỡng, và chọn ngưỡng cho ngân hàng sợ chặn nhầm tin OTP.

Cách làm: huấn luyện lại BoW + LogReg (lr=0.5, 300 epoch), tính xác suất p_te trên tập test một lần, rồi chỉ đổi ngưỡng cắt p_te >= th cho th ∈ {0.2, 0.5, 0.8} và gọi metrics.

Kết quả (test): ngưỡng 0.2: P=0.892, R= 0.937, F1=0.914; ngưỡng 0.5: P= 1.000, R=0.797, F1=0.887; ngưỡng 0.8: P= 1.000, R=0.633, F1=0.775.

Giải thích: hạ ngưỡng thì recall tăng nhưng precision giảm; nâng ngưỡng thì ngược lại. Ngân hàng sợ chặn nhầm OTP (tin ham quan trọng) nên chọn ngưỡng cao (0.8) để ưu tiên precision, chấp nhận lọt thêm một số spam vì chặn nhầm OTP làm khách không đăng nhập được, thiệt hại lớn hơn.

## 6. Phương án cho sản phẩm thật

Phương án mới: Encoder + MLP head (3c). p95 40.3 ms đạt SLA, F1 test 0.962, F1 SUB 1.000. Encoder + LogReg (3b) gần như ngang.
Encoder + kNN bị loại: p95 là 68.2 ms, vượt SLA 50 ms, và chỉ chạy ~20 tin/s. Số ~40 ms trước đó mình đọc từ biểu đồ Pareto của một lần chạy khác. Lab cũng cảnh báo số đo dao động theo Colab, nên dùng bảng này.
Mình thêm lưu ý về thông lượng: mỗi máy chỉ ~28 tin/s, cần khoảng 5 máy cho tải trung bình 116 tin/s.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
Hiểu vectorize 0/1 và sơ đồ mạng neural | Giải thích bag-of-words, ReLU, 64 feature tự học | Đối chiếu với sơ đồ trong notebook
Điều hiểu ra hôm nay: accuracy dễ đánh lừa khi dữ liệu lệch (luôn đoán ham đã ~87%) nên cần F1; learning rate lớn không chắc làm loss dao động vì còn phụ thuộc dữ liệu; điểm cao trên test thường chưa chắc tốt ở các ca khó như spam né từ khoá hay giọng tự nhiên.
