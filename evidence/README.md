# Phân tích Đánh giá RAGAS — A/B Testing Prompt V1 vs Prompt V2

## 1. Bảng điểm tổng hợp (RAGAS Evaluation Scores)

| Chỉ số (Metric) | Prompt V1 (Ngắn gọn) | Prompt V2 (Chuyên gia / Cấu trúc) | Winner |
|---|:---:|:---:|:---:|
| **Faithfulness** | 0.8359 | **0.9772** | **← V2 (+14.13%)** |
| **Answer Relevancy** | 0.9274 | **0.9503** | **← V2 (+2.29%)** |
| **Context Recall** | 1.0000 | 1.0000 | Hòa (100%) |
| **Context Precision** | 0.9444 | **0.9500** | **← V2 (+0.56%)** |

> **Mục tiêu đề bài:** Faithfulness ≥ 0.8 cho ít nhất một phiên bản.  
> **Kết quả:** Cả 2 phiên bản đều vượt ngưỡng yêu cầu, trong đó **Prompt V2 xuất sắc đạt 0.9772 (tiệm cận tuyệt đối)**.

---

## 2. Phân tích nguyên nhân chênh lệch hiệu năng giữa V1 và V2

### 2.1. Tại sao Faithfulness của V2 (0.9772) cao hơn vượt trội so với V1 (0.8359)?
- **Prompt V1:** Được thiết kế theo phong cách *"trợ lý AI thân thiện, trả lời ngắn gọn 2-4 câu"*. Khi bị ép độ dài ngắn và văn phong thân mật, LLM có xu hướng nén thông tin, paraphrase hoặc khái quát hóa. Quá trình này dễ dẫn đến hiện tượng trôi dạt ngữ nghĩa (semantic drift) hoặc vô tình đưa ra các nhận định chung chung không có căn cứ cụ thể trong context, làm giảm điểm Faithfulness.
- **Prompt V2:** Được thiết kế theo phong cách *"chuyên gia phân tích thông tin: đọc kỹ context, xác định các facts liên quan, câu trả lời rõ ràng có tổ chức (3-5 câu), không suy đoán ngoài context"*. 
  - Chỉ thị rõ ràng về việc trích xuất **"facts liên quan"** đã đóng vai trò như một kỹ thuật *Chain-of-Thought định hướng facts*.
  - Ràng buộc tiêu cực mạnh mẽ **"Không suy đoán ngoài context"** triệt tiêu triệt để tình trạng hallucination.
  - Dung lượng 3-5 câu vừa đủ để trình bày trọn vẹn luận điểm mà không bị ép cắt gọt thông tin.

### 2.2. Answer Relevancy (V2: 0.9503 vs V1: 0.9274)
- Prompt V2 có cấu trúc mạch lạc, giải thích rõ ràng các khái niệm phức tạp (như bias-variance tradeoff, vanishing gradient, transformer self-attention), giúp câu trả lời khớp sát và đầy đủ với trọng tâm câu hỏi gốc hơn so với câu trả lời ngắn của V1.

### 2.3. Context Recall & Context Precision
- Cả hai phiên bản đều sử dụng chung cơ chế tìm kiếm FAISS với $k=3$ và embedding `BAAI/bge-small-en-v1.5`, do đó Context Recall đạt mức tối đa 1.0 (toàn bộ thông tin cần thiết trong câu trả lời mẫu đều được tìm thấy trong context).

---

## 3. Kết luận và Khuyến nghị Production
- **Prompt V2 là phiên bản chiến thắng toàn diện (Winner across all metrics)**.
- Đối với các ứng dụng RAG trong doanh nghiệp đòi hỏi độ tin cậy và chính xác cao (tài chính, y tế, kỹ thuật), nên ưu tiên áp dụng cấu trúc System Prompt theo phong cách V2 (phân tích facts, nghiêm cấm suy đoán, quy định dung lượng câu trả lời rõ ràng).
