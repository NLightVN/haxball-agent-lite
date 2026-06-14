# Haxball Multi-Agent - Reward & Possession Algorithm

Tài liệu này mô tả chi tiết về thuật toán theo dõi quyền kiểm soát bóng (Possession) và cơ chế phân phối phần thưởng (Investment Algorithm) được sử dụng trong môi trường Multi-Agent (2v2).

## 1. Cơ chế theo dõi kiểm soát bóng (0.3s Possession Logic)
Trong các pha tranh chấp, bóng thường xuyên nảy qua lại giữa 2 đội khiến việc xác định quyền kiểm soát bị nhiễu. Để khắc phục, môi trường áp dụng cơ chế "cướp bóng tạm thời" (Tentative Steal).

- **Theo dõi tương tác:** Mỗi tick vật lý, môi trường ghi nhận những đội nào chạm hoặc sút trúng bóng.
- **Duy trì kiểm soát:** Nếu đội đang giữ bóng tiếp tục chạm hoặc sút bóng, quyền kiểm soát được duy trì và trạng thái cướp bóng của đối phương bị hủy bỏ (`tentative_team = 0`).
- **Cướp bóng (Cần 0.3s):** Khi đội đối phương chạm hoặc sút bóng lần đầu tiên, họ chỉ được tính là "cướp bóng tạm thời" (`tentative_start_tick` được ghi nhận). Để chính thức giành được quyền kiểm soát (`possession_team` đổi chủ), họ phải thỏa mãn 1 trong 2 điều kiện sau:
  1. **Tiếp tục tương tác:** Khoảng thời gian giữa lần tương tác đầu tiên và lần tương tác cuối cùng (chạm/sút bóng) đạt ít nhất 18 ticks vật lý (0.3 giây).
  2. **Bám sát bóng:** Sau lần chạm đầu tiên, nếu họ không chạm được bóng thêm lần nào nhưng **luôn là phe đứng gần bóng nhất** liên tục trong 18 ticks (0.3 giây), họ vẫn sẽ chính thức cướp được Possession. (Điều kiện này xử lý các tình huống một cầu thủ rướn người chạm bóng nhưng bóng lăn đi trước mặt và họ bám theo sát sao).
- **Mất kiểm soát do bóng giam/chết (2 giây):** Nếu tốc độ bóng giảm xuống dưới 0.3 và duy trì tình trạng này liên tục trong 2 giây (120 ticks vật lý), quyền kiểm soát bóng sẽ bị reset về 0 (trung lập). Cơ chế này kết hợp với phần phạt giam bóng để ép các Agent phải liên tục di chuyển và luân chuyển bóng, chống lại các hành vi ôm bóng đứng yên.

## 2. Thuật toán Đầu tư (Investment Algorithm)
Đây là thuật toán cốt lõi để giải quyết bài toán Credit Assignment trong Multi-Agent: Làm sao để thưởng cho những đường chuyền kiến tạo thay vì chỉ thưởng cho người ghi bàn cuối cùng. Thuật toán duy trì một bể cổ phần (`_inv_pool`).

### 2.1 Pha loãng Cổ phần và Ghi nhận Đường chuyền
- **Mất bóng:** Nếu quyền kiểm soát bóng rơi vào tay đối phương, toàn bộ bể cổ phần (`_inv_pool`) sẽ bị xóa sạch.
- **Chạm bóng đầu tiên:** Cầu thủ đầu tiên của đội chạm bóng sẽ nhận được `1.0` (100%) cổ phần trong bể.
- **Sự kiện Chuyền bóng (Pass Event):** Khi một đồng đội *khác* nhận được bóng:
  1. Cổ phần của tất cả những người đang có trong bể sẽ bị **pha loãng đi, chỉ còn giữ lại 60%**.
  2. Người vừa nhận bóng sẽ được cộng thêm **40% (0.4)** cổ phần mới. (Tỷ lệ 60/40)
  3. Nếu Agent đang train (Agent 0) chính là người thực hiện đường chuyền (Passer), môi trường sẽ ghi lại mốc thời gian (step) và lượng cổ phần mà Agent 0 đang nắm giữ ngay tại thời điểm đó.

### 2.2 Trả thưởng ngược về quá khứ (Retroactive Reward Distribution)
Khi đội nhà ghi bàn:
- **Phần thưởng cơ bản cho người ghi bàn (Scorer):** Người chạm bóng cuối cùng sẽ luôn nhận được một mức thưởng cứng bằng với tỷ lệ cổ phần nhận bóng ban đầu (40% của tổng 20.0 điểm thưởng = 8.0 điểm).
- **Chia chác Cổ tức:** Phần điểm thưởng còn lại (60% = 12.0 điểm) sẽ được chia cho các đồng đội dựa trên tỉ lệ cổ phần cuối cùng của họ trong `_inv_pool`.
- **Cộng điểm xuyên không gian:** Số điểm cổ tức của Agent 0 sẽ được chia tỉ lệ ngược về các đường chuyền mà nó đã thực hiện trước đó (dựa trên trọng số cổ phần lúc chuyền). Môi trường xuất ra thông tin này qua biến `info["investment_credit"] = [(số_step_lùi_lại, lượng_phần_thưởng)]`.
- **Investment Callback:** Callback này bên trong SB3 sẽ bắt lấy thông tin trên và lùi lại quá khứ, **cộng trực tiếp phần thưởng vào đúng vị trí `rollout_buffer.rewards` nơi đường chuyền đã diễn ra**. Điều này giúp thuật toán PPO nhận ra giá trị của đường chuyền ngay lập tức thay vì phải chờ mòn mỏi đến lúc có bàn thắng.
