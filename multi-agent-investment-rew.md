# Hệ thống Phần thưởng Đa tác vụ và Chuỗi Đầu tư (Multi-Agent Investment Reward)

Tài liệu này mô tả logic tính toán phần thưởng (reward) dựa trên quyền kiểm soát bóng (possession) và cơ chế phân phối phần thưởng theo chuỗi phối hợp (investment sequence).  
Phiên bản hiện tại: **env.py MARL branch** — áp dụng cho Phase A0 (1v0) và A1 (1v1+).

---

## 1. Hệ Thống Possession (Quyền kiểm soát bóng)

### 1.1. Possession hiện tại — `last_touch_team`
Là **công tắc đơn giản**: mỗi khi ai đó chạm bóng, biến này lật sang team đó và **giữ nguyên** cho đến khi có người khác chạm.

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

has_possession_reward = (last_touch_team == team_id) OR prev_poss_our_side
has_possession_penalty = True   # luôn phạt khi bóng lùi — không có ân xá
```

- **Thưởng** khi bóng tiến về phía gôn đối thủ, nếu `has_possession_reward = True`.
- **Phạt** khi bóng lùi về phía gôn nhà, **luôn luôn** (không phân biệt ai đang cầm).

### 2.2. Ngoại lệ

| Tình huống | Nhân (mult) | Điều kiện |
|---|---|---|
| Bình thường (kick hoặc đang tiến) | `1.0` | mặc định |
| **Dribble không sút** | `0.333` | `is_dribbling AND NOT kick` |
| **Self-pass** (sút về chính mình) | `0.333` | `self_pass_active = True` |
| **Pass lùi cho đồng đội** (MARL) | phạt `× 0.333` | `has_teammates AND prev_poss_our_side AND last_touch == team` |
| **Đối thủ pass lùi** | `0` (không thưởng) | `prev_poss == opp AND last_touch == opp` |

> **Lưu ý A0 (1v0):** `has_teammates = False` → ngoại lệ "pass lùi" không bao giờ kích hoạt.

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
`ADVANCE_REWARD = BACKWARD_PENALTY = 0.003`

---

## 3. Chuỗi Đầu Tư (Investment Sequence)

### 3.1. Quy tắc hình thành
- Mỗi khi ai chạm bóng (phe mình), họ được **append** vào `investment_sequence`.
- Nếu người đó **đã có trong sequence**, những người sau họ bị cắt bỏ.
- **Bị reset** nếu đối thủ giữ bóng liên tục ≥ 2 giây.

### 3.2. Phân bổ phần thưởng (`invest_share`)

| Vai trò | `invest_share` |
|---|---|
| Người cầm bóng (cuối sequence) | `1.0` |
| Người trước 1 pass | `0.30` |
| Người trước 2 pass | `0.15` |
| Người trước N pass | `0.30 × 0.5^(N-1)` |
| Không trong sequence | `min_share = 0.3 × 0.5^(num_teammates)` |

### 3.3. Phần thưởng ghi bàn

```
Ghi bàn:       +10.0 (base)
                +3.0  (bonus_pool, chia theo invest_share cho assisters)
Bị ghi bàn:    -10.0
```

---

## 4. Cơ chế Chống Farm Solo

### 4.1. Giảm thưởng khi tự dẫn bóng (Dribble)
- Nếu agent liên tục cầm bóng không chuyền (không kick), phần thưởng đưa bóng lên giảm còn **1/3**.
- Ngay khi sút thực sự (`kick = 1`), nhân trở về `1.0`.

### 4.2. Chống tự chuyền bóng (Self-pass)
- Khi sút, hệ thống predict quỹ đạo bóng 150 frames.
- Nếu người có khả năng nhận bóng sớm nhất là **chính agent vừa sút** → `self_pass_active = True` → phần thưởng giảm còn **1/3**.
- Reset khi bóng chạm đồng đội hoặc đối thủ.

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
