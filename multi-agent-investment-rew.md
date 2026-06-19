# Hệ thống Phần thưởng Đa tác vụ và Chuỗi Đầu tư (Multi-Agent Investment Reward)

Tài liệu này mô tả logic tính toán phần thưởng (reward) dựa trên quyền kiểm soát bóng (possession) và cơ chế phân phối phần thưởng theo chuỗi phối hợp (investment sequence).  
Phiên bản hiện tại: **env.py MARL branch** — áp dụng cho Phase A0 (1v0) và A1 (1v1+).

---

## 1. Hệ Thống Possession (Quyền kiểm soát bóng)

### 1.1. Possession hiện tại — `last_touch_team`
Là **công tắc đơn giản**: Possession ĐỘC QUYỀN được tính hoàn toàn dựa trên lần chạm cuối cùng (last touch).
**Đặc biệt lưu ý:** Sự kiện chạm bóng (touch event) được ghi nhận và cập nhật liên tục ngay cả khi **chính agent đó tự chạm lại quả bóng mà họ vừa chạm** ở frame trước. Mỗi một frame có va chạm vật lý hoặc cú sút, hệ thống đều tính đó là một touch mới, qua đó reset lại bộ đếm thời gian `last_touch_time`. Biến này lật sang team đó và **giữ nguyên** cho đến khi có người khác chạm.

### 1.2. Previous Possession Among 0.25s — `prev_poss_at_touch`
- **Được cập nhật duy nhất một lần: tại khoảnh khắc có touch mới.**
- Đo trong cửa sổ 0.25 giây tính **ngược từ thời điểm chạm**, không phải từ "bây giờ".
- **Chỉ có 2 giá trị hợp lệ:**
  - `None` — không có chủ sở hữu trước đó khác team hiện tại (bóng tự do hoặc phe mình cầm liên tục).
  - `opp_id` — phe đối thủ đã chạm bóng trong vòng 0.25s trước lần chạm hiện tại.

**Quy tắc tính toán khi touch xảy ra (tại thời điểm T):**
```
dt = T - last_touch_time
if last_touch_team != new_tid AND dt <= 0.25:
    prev_poss_at_touch = last_touch_team  # phe trước đó, khác team mới
else:
    prev_poss_at_touch = None             # liên tục hoặc bóng tự do quá lâu
```

**Ví dụ:**
- Đối thủ đá → bóng bay 3 giây → mình chạm: `dt = 3s > 0.25s` → `prev_poss_at_touch = None` ✅
- Đối thủ chạm → 0.1s sau mình chạm: `dt = 0.1s ≤ 0.25s`, khác team → `prev_poss_at_touch = opp_id` ✅
- Mình cầm bóng 1s → sút: không có touch mới → `prev_poss_at_touch` giữ nguyên từ lần chạm trước ✅

> **Không còn rolling deque clock.** Trước đây dùng deque 0.25s luôn trượt theo thời gian hiện tại → lỗ hổng. Nay dùng `last_touch_time` so sánh tại đúng khoảnh khắc touch.

---

## 2. Phần thưởng Quỹ đạo Bóng (Ball Movement Reward)

### 2.1. Điều kiện Thưởng và Phạt

```
prev_poss_our_side = (prev_poss_at_touch is None) or (prev_poss_at_touch == team_id)

has_possession_reward = (last_touch_team == team_id) OR (last_touch_team == opp_id AND prev_poss_at_touch == team_id)
has_possession_penalty = True   # luôn phạt khi bóng lùi — không có ân xá
```

- **Thưởng** khi bóng tiến về phía gôn đối thủ, nếu `has_possession_reward = True`.
- **Phạt** khi bóng lùi về phía gôn nhà, **luôn luôn** (không phân biệt ai đang cầm).

### 2.2. Ngoại lệ

**Quy tắc bắt buộc:** Khi `no_pass_possession = True` HOẶC `self_pass = True`, nhân thưởng phải **bằng** nhân phạt (không được phép thưởng nhiều hơn phạt hay ngược lại).

| Tình huống | Nhân thưởng | Nhân phạt | Điều kiện |
|---|---|---|---|
| **Sút thật / Sút bừa** | `1.0` | `1.0` | `real_pass = True` (ngay cả khi bóng bay tự do, mult vẫn giữ 1.0) |
| **No-pass possession** (cầm bóng không chuyền) | `0.333` | `0.333` | Agent đang possess bóng nhưng chưa thực hiện real_pass cho đồng đội (bao gồm tự chạm hoặc không sút). Không dùng timer. |
| **Self-pass** (sút về mình) | `0.333` | `0.333` | `self_pass = True` (hết hiệu lực sau 2.5s) |
| **Pass lùi đồng đội** (MARL)| `0.333` | `0.333` | `has_teammates AND prev_poss_our_side AND last_touch == team` |
| **Đối thủ pass lùi** | `0` | `0` | `prev_poss == opp AND last_touch == opp` |

> **Lưu ý A0 (1v0):** `has_teammates = False` → ngoại lệ "pass lùi" không bao giờ kích hoạt.

> **Tóm tắt mult logic:** Chỉ `real_pass = True` mới được mult = 1.0. Mọi trường hợp còn lại đều dùng mult = 0.333 cho **cả thưởng lẫn phạt** để triệt tiêu khả năng farm điểm.

### 2.3. Phân chia Zone (Khu vực sân)

Sân chia 3 vùng theo trục X (`zone_width = goal_y * 2.0`):

| Zone | Vị trí | Công thức `delta_dist` |
|---|---|---|
| **Sân nhà** | `adv_x ≤ -HW + zone_width` | Bóng xa gôn nhà = `+` (tốt) |
| **Giữa sân** | khoảng còn lại | `(bx - px) * atk` |
| **Sân khách** | `adv_x ≥ HW - zone_width` | Bóng gần gôn đối thủ = `+` (tốt) |

```
reward += ADVANCE_REWARD  × delta_dist × mult × invest_share   (nếu delta > 0)
reward -= BACKWARD_PENALTY × |delta_dist| × mult × penalty_share  (nếu delta < 0)
```
`ADVANCE_REWARD = BACKWARD_PENALTY = 0.0003` (đã chia 10 để giảm biến động quá mức)

---

## 3. Chuỗi Đầu Tư (Investment Sequence)

### 3.1. Niềm tin Chủ quan (Subjective Investment Sequences)
Trong MARL, nếu team có N người, hệ thống sẽ duy trì **N chuỗi investment_sequence khác nhau** (`marl_investment_sequences = [[], [], ...]`). Mỗi người sẽ có một "niềm tin" riêng về việc ai đã đóng góp vào pha bóng này:

- **Góc nhìn của bản thân (`i == pid`):** Khi một agent tự chạm bóng, họ tin rằng mọi thứ từ đây là do họ tự làm. Chuỗi của họ **bị reset** thành `[pid]`.
- **Góc nhìn của đồng đội (`i != pid`):** Khi thấy đồng đội chạm bóng, họ tin rằng họ đã có công luân chuyển bóng đến đó. Họ sẽ **append** `pid` vào chuỗi của họ (nếu `pid` đã có, đẩy `pid` xuống cuối).
- **Bị reset toàn bộ:** Khi đối thủ giữ bóng liên tục ≥ 2 giây, TẤT CẢ các chuỗi đều bị xóa sạch.

> **Tối thượng:** Agent `i` chỉ nhận reward dựa trên **ĐÚNG chuỗi của agent `i`**, không quan tâm người khác nghĩ gì hay có chuỗi như thế nào.

### 3.2. Phân bổ phần thưởng (`invest_share`)

| Vai trò | `invest_share` |
|---|---|
| Người cầm bóng (cuối sequence) | `1.0` |
| Người trước 1 pass | `0.30` |
| Người trước 2 pass | `0.15` |
| Người trước N pass | `0.30 × 0.5^(N-1)` |
| Không trong sequence | `0.0` (Không được hưởng thưởng tiến bóng nếu chưa đầu tư) |

### 3.3. Phần thưởng ghi bàn

```
Ghi bàn:       +20.0 (base)
                +3.0  (bonus_pool, chia theo invest_share cho assisters)
Bị ghi bàn:    -20.0
```

---

## 4. Cơ chế Chống Farm Solo

### 4.1. No-pass Possession (cầm bóng không chuyền)
- Cờ `no_pass[i]` là một boolean đơn giản, **bật lên** khi agent `i` đang là người cuối cầm bóng (holder) nhưng chưa thực hiện `real_pass` cho đồng đội.
- Bao gồm cả trường hợp: tự chạm bóng (va chạm vật lý), sút nhưng bóng không đến đồng đội, hoặc đơn giản chỉ đứng cạnh bóng.
- **Phá vỡ no_pass:** Reset về `False` ngay lập tức khi `real_pass = True` (agent sút và bóng được predict đến đồng đội khác). Không dùng timer.
- **Hậu quả:** mult_adv = mult_back = 0.333 (thưởng bằng phạt, không thể farm).

### 4.2. Chống tự chuyền bóng (Self-pass)
- Khi sút, hệ thống predict quỹ đạo bóng 150 frames.
- Nếu người nhận là **chính agent vừa sút** → `self_pass[i] = True`.
- Cờ này sẽ **tự động hết hạn sau 2.5 giây**, sau đó bóng thành bóng tự do nếu không ai nhặt.
- **Cân bằng thưởng phạt:** Self-pass áp dụng mult `0.333` cho **cả phần thưởng lẫn hình phạt** để triệt tiêu hoàn toàn khả năng farm điểm âm/dương.

### 4.3. Phạt Turnover
```
Turnover penalty = -0.1 × invest_share  (nếu trong sequence)
                  = -0.1               (nếu không trong sequence)
```
Kích hoạt khi phe mình đang cầm bóng mà để đối thủ chạm được.

---

## 5. Obs liên quan đến Possession

| Dim | Giá trị | Ý nghĩa |
|---|---|---|
| `possession_current` | `[1,0,0]` / `[0,1,0]` / `[0,0,1]` | None / Team / Opp (last_touch_team) |
| `prev_poss_at_touch` | `[1,0,0]` / `[0,0,1]` | None / Opp (chỉ 2 giá trị hợp lệ) |
| `opp_possession_time` | `[0, 1]` | Thời gian đối thủ giữ bóng / 2s |
| `agent_invest_share` | `[0, 1]` | % phần thưởng agent nhận |
