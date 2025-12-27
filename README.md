# Phân Tích Trầm Cảm Ở Sinh Viên

Đường dẫn đến repository: [Student Depression Analysis](https://github.com/p1neapplechoco/StudentDepression)

## Tổng Quan Dự Án

Dự án này phân tích các yếu tố góp phần gây ra trầm cảm ở sinh viên sử dụng các phương pháp thống kê và học máy. Phân tích điều tra 10 câu hỏi nghiên cứu (RQ) xoay quanh ba chủ đề chính: Lối sống, Áp lực Tâm lý, và Các biến số Học tập & Công việc.

## Bộ Dữ Liệu

### Nguồn

- **Nền tảng:** Kaggle
- **Liên kết:** [Student Depression Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset/data)
- **Tác giả:** Adil Shamim
- **Giấy phép:** Apache License 2.0

### Mô tả Dữ liệu

- **Quy mô:** 27,901 bản ghi x 18 đặc trưng
- **Phạm vi địa lý:** 52 thành phố tại Ấn Độ
- **Loại dữ liệu:** Khảo sát tự báo cáo
- **Chất lượng:** Không có giá trị thiếu, không có bản ghi trùng lặp

### Biến Mục tiêu

- **Trầm cảm (Depression)** (Phân loại nhị phân)
  - Lớp 0 (Không trầm cảm): 11,565 sinh viên (41.5%)
  - Lớp 1 (Trầm cảm): 16,336 sinh viên (58.5%)

## Câu Hỏi Nghiên Cứu & Phát Hiện Chính

### Chủ đề 1: Yếu tố Lối sống (Lifestyle)

**RQ1: Nghịch lý Giấc ngủ 7-8 Giờ**

- **Câu hỏi:** Tại sao sinh viên ngủ đủ 7-8 giờ (theo khuyến nghị y tế) lại có tỷ lệ trầm cảm CAO HƠN (60.67%) so với nhóm ngủ 5-6 giờ (58.72%)?
- **Phát hiện:** Đây là hiện tượng "ngược chiều nhân quả" (reverse causation). Sinh viên trầm cảm thường có xu hướng ngủ nhiều hơn (triệu chứng ngủ rũ - hypersomnia).
- **Kết luận:** Chất lượng giấc ngủ quan trọng hơn số lượng.

**RQ2: Chế độ ăn uống như sự bù đắp**

- **Câu hỏi:** Chế độ ăn uống lành mạnh có thể bù đắp cho việc thiếu ngủ không?
- **Phát hiện:** Chế độ ăn lành mạnh giúp giảm tỷ lệ trầm cảm từ 25-33%. Tuy nhiên, ngay cả khi ăn uống lành mạnh, nhóm ngủ dưới 5 giờ vẫn có tỷ lệ trầm cảm cao (50.38%) hơn nhiều so với nhóm ngủ trên 8 giờ kết hợp ăn uống lành mạnh (36.64%).
- **Kết luận:** Ăn uống tốt không thể thay thế hoàn toàn cho giấc ngủ đủ.

**RQ3: Ngưỡng Lối sống Tối ưu**

- **Câu hỏi:** Đâu là "điểm ngọt" (sweet spot) kết hợp giữa Giấc ngủ và Chế độ ăn để giảm thiểu rủi ro trầm cảm?
- **Phát hiện:**
  - **Tốt nhất:** Ngủ trên 8 giờ + Ăn uống lành mạnh (Tỷ lệ trầm cảm: 36.64%)
  - **Tệ nhất:** Ngủ dưới 5 giờ + Ăn uống không lành mạnh (Tỷ lệ trầm cảm: 75.96%)
- **Chiến lược ROI:** Đối với nhóm tệ nhất, cải thiện chế độ ăn uống là bước đầu tiên hiệu quả nhất (giảm 12.79 điểm phần trăm).

### Chủ đề 2: Áp lực Tâm lý (Psychological Pressure)

**RQ4: Hiệu ứng Áp lực Tích lũy**

- **Câu hỏi:** Sự kết hợp của Áp lực học tập, Áp lực công việc và Căng thẳng tài chính có làm tăng suy nghĩ tự tử theo cấp số nhân không?
- **Phát hiện:** Có tác động tích lũy rõ rệt.
  - Áp lực học tập: Tăng 46% nguy cơ mỗi đơn vị tăng thêm.
  - Căng thẳng tài chính: Tăng 31% nguy cơ mỗi đơn vị tăng thêm.
  - Áp lực công việc: Không có ý nghĩa thống kê (do phần lớn sinh viên không đi làm).

**RQ5: Sự hài lòng là Yếu tố Bảo vệ**

- **Câu hỏi:** Sự hài lòng cao với việc học có thể giảm suy nghĩ tự tử ở sinh viên có tiền sử gia đình về bệnh tâm thần không?
- **Phát hiện:** Có. Khi mức độ hài lòng đạt đỉnh (5/5), rủi ro của nhóm có tiền sử bệnh giảm xuống ngang bằng với nhóm không có tiền sử bệnh. Sự hài lòng cao có thể "bù đắp" cho rủi ro di truyền.

**RQ6: Ngưỡng Báo động Áp lực Học tập**

- **Câu hỏi:** Tại mức áp lực học tập nào thì xác suất có suy nghĩ tự tử vượt quá 50%?
- **Phát hiện:** Ngưỡng báo động là **1.57** (trên thang điểm 5).
- **Khuyến nghị:** Cần can thiệp ngay khi sinh viên báo cáo mức áp lực >= 2.

### Chủ đề 3: Biến số Học tập & Công việc

**RQ7: Yếu tố Tác động Mạnh nhất**

- **Câu hỏi:** Giữa Điểm số (CGPA) và Giờ học/làm việc, yếu tố nào tác động mạnh hơn đến trầm cảm?
- **Phát hiện:** Giờ học/làm việc tác động mạnh gấp **4 lần** so với CGPA. Điểm số (CGPA) có tác động rất yếu và không đáng kể về mặt thực tiễn.

**RQ8: Nhóm Ngành Rủi ro Cao**

- **Câu hỏi:** Ngành học nào có tỷ lệ trầm cảm cao nhất?
- **Phát hiện:** Nhóm **Học sinh lớp 12 (Class 12)** có tỷ lệ cao nhất (70.8%), cao hơn mức trung bình 12.3 điểm phần trăm. Tiếp theo là BSc, B.Arch, BBA, MBBS.
- **Khuyến nghị:** Cần hỗ trợ tâm lý đặc biệt cho học sinh cuối cấp phổ thông.

**RQ9: Ngưỡng Thời gian An toàn**

- **Câu hỏi:** Ngưỡng giờ học/làm việc mỗi ngày bao nhiêu là an toàn?
- **Phát hiện:** Ngưỡng an toàn là dưới **4.12 giờ/ngày**.
  - Dưới 4 giờ: Tương đối an toàn (dưới 50% xác suất trầm cảm).
  - Trên 6 giờ: Vùng nguy hiểm (trên 55% xác suất trầm cảm).

**RQ10: Nghịch lý Thành tích - Hài lòng**

- **Câu hỏi:** Sinh viên điểm cao nhưng không hài lòng có rủi ro cao hơn sinh viên điểm thấp nhưng hài lòng không?
- **Phát hiện:** Có.
  - Điểm cao + Hài lòng thấp: 67.4% trầm cảm.
  - Điểm thấp + Hài lòng cao: 52.9% trầm cảm.
- **Kết luận:** Sự hài lòng với việc học quan trọng hơn điểm số. "Học giỏi mà không vui" rủi ro hơn "Học kém mà vui".

## Kết Quả Mô Hình Hóa

**Mô hình tốt nhất:** Logistic Regression

- **Độ chính xác (Accuracy):** 84.39%
- **F1-Score:** 86.41%
- **AUC-ROC:** 92.09%

**Top 5 Đặc trưng Quan trọng nhất (theo SHAP values):**

1. **Suy nghĩ tự tử (Suicidal Thoughts):** Yếu tố dự báo mạnh nhất.
2. **Áp lực học tập (Academic Pressure):** Tác động lớn thứ hai.
3. **Điểm rủi ro tổng hợp (Risk Score):** Biến tổng hợp từ feature engineering.
4. **Tuổi (Age):** Yếu tố nhân khẩu học quan trọng.
5. **Tổng mức độ Stress (Total Stress):** Tổng hợp các loại áp lực.

*Lưu ý: Điểm số (CGPA) nằm ở cuối bảng xếp hạng độ quan trọng, khẳng định lại kết quả của RQ7.*

## Khuyến Nghị Thực Tiễn

1. **Ưu tiên Sức khỏe Tâm thần:** Tập trung vào việc giảm áp lực học tập và tài chính hơn là chỉ thúc đẩy điểm số.
2. **Cải thiện Lối sống:** Khuyến khích sinh viên ngủ đủ trên 8 giờ và duy trì chế độ ăn uống lành mạnh. Đối với sinh viên đang gặp khó khăn, cải thiện chế độ ăn là bước khởi đầu tốt nhất.
3. **Giới hạn Thời gian:** Khuyến cáo sinh viên cân bằng thời gian học/làm việc dưới 4-6 giờ mỗi ngày (ngoài giờ lên lớp chính thức) để tránh kiệt sức.
4. **Tăng cường Sự hài lòng:** Nhà trường nên chú trọng tạo môi trường học tập tích cực, giúp sinh viên tìm thấy niềm vui trong việc học thay vì chỉ chạy theo thành tích.
5. **Nhóm Cần Quan tâm Đặc biệt:** Học sinh lớp 12 và sinh viên có suy nghĩ tự tử cần được theo dõi và hỗ trợ sát sao.

## Cấu Trúc Thư Mục

```
StudentDepression/
├── data/
│   └── student_depression_dataset.csv       # Dữ liệu thô
├── notebooks/
│   ├── 01_exploration.ipynb                 # Khám phá dữ liệu (EDA)
│   ├── 02_preprocessing.ipynb               # Tiền xử lý dữ liệu
│   ├── 03_lifestyle.ipynb                   # Phân tích Lối sống (RQ1-RQ3)
│   ├── 04_pressure.ipynb                    # Phân tích Áp lực (RQ4-RQ6)
│   ├── 05_work.ipynb                        # Phân tích Học tập & Công việc (RQ7-RQ10)
│   └── 06_modeling.ipynb                    # Mô hình hóa dự đoán
├── src/
│   ├── preprocessing.py                     # Các hàm tiền xử lý
│   ├── features.py                          # Các hàm tạo đặc trưng
│   ├── models.py                            # Các hàm mô hình hóa
│   └── run_pipeline.py                      # Script chạy toàn bộ pipeline
├── results/
│   ├── best_model_metrics.csv               # Kết quả mô hình tốt nhất
│   ├── model_comparison.csv                 # So sánh các mô hình
│   ├── shap_importance.csv                  # Độ quan trọng của các biến
│   └── processed_data_notebook.csv          # Dữ liệu đã qua xử lý
└── README.md
```

## Disclaimer

Dự án này chỉ nhằm mục đích giáo dục và nghiên cứu. Các mô hình và phát hiện không nên được sử dụng thay thế cho chẩn đoán hoặc điều trị sức khỏe tâm thần chuyên nghiệp. Nếu bạn hoặc ai đó bạn biết đang trải qua trầm cảm hoặc có suy nghĩ tự tử, hãy tìm kiếm sự giúp đỡ từ các chuyên gia y tế ngay lập tức.
