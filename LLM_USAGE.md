# Nhật ký Sử dụng LLM (LLM Usage Log)

Theo quy định của môn học (Mục 6 của đề bài), bảng dưới đây ghi lại các lần sử dụng công cụ LLM đáng kể trong quá trình thực hiện đồ án.

| Ngày | Người | Công cụ | Đã hỏi gì | Dùng hoặc sửa thế nào |
| :--- | :--- | :--- | :--- | :--- |
| 2026-10-21 | Đức Khải | LLM Assistant | Gợi ý biểu thức chính quy (regex) trong Python để tách cấu hình kit (`2x16GB`, `16GBx2`, `Kit of 2`), tốc độ MHz, độ trễ CL và chuẩn PCIe từ `product_name`. | Đưa vào `src/parse_titles.py`, tự kiểm tra trên mẫu 100 tên sản phẩm thực tế và bổ sung quy tắc xử lý ngoại lệ (như lỗi chính tả `CMSX32GX5M1A5600C48` và các thiết bị NAS). |
| 2026-11-05 | Đức Khải | LLM Assistant | Cách thiết kế kiểm định độ nhạy (sensitivity check) với bộ lọc ngoại lai IQR 1.5×, IQR 3.0×, Z-score và khoảng tin cậy Bootstrap cho chênh lệch trung vị ở Q5. | Áp dụng trong `notebooks/02_q5_kit_comparison_clean.ipynb`, tự đối chiếu kết quả theo từng phân khúc UDIMM/SODIMM và từng thương hiệu. |
| 2026-11-09 | Đức Khải | Antigravity | Cách thiết kế `Pipeline` scikit-learn kết hợp `GroupShuffleSplit` / `GroupKFold` theo `product_name`, ước lượng Bootstrap theo cụm sản phẩm và Grouped Permutation Importance cho bài toán Q6 (ML). | Hoàn thiện code trong `notebooks/03_ml_question.ipynb`, bổ sung bước gom các sản phẩm không ghi tên hãng ở đầu về nhãn `GENERIC`, tắt `early_stopping` nội bộ của `HistGradientBoostingRegressor` để tránh rò rỉ thực thể, và tự kiểm tra lại toàn bộ bảng số liệu/biểu đồ. |

