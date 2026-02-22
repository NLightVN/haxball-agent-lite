# Haxball Agent Lite — Project Context

## Phiên bản: Offline với Engine tích hợp sẵn

Project này là bản **offline hoàn toàn** của Haxball, tích hợp sẵn physics engine (port từ Wazarr94's Haxball clone).  
Không cần internet, không cần kết nối `haxball.com`. Chạy bằng `npm start` trên local HTTP server.

---

## Engine nằm ở đâu?

### 📄 `script.js` — File chính chứa engine

Toàn bộ physics engine nằm trong file này (~2285 dòng), bao gồm:

| Thành phần | Mô tả |
|---|---|
| `Disc`, `Player`, `Game` | Các object cơ bản |
| `ballPhysics`, `playerPhysics` | Cập nhật vận tốc, damping mỗi tick |
| `resolveDDCollision` | Collision giữa 2 disc (player-player, player-ball) |
| `resolveDPCollision` | Collision disc vs plane (tường ngang/dọc) |
| `resolveDSCollision` | Collision disc vs segment (tường xiên, lưới) |
| `resolveDVCollision` | Collision disc vs vertex (cột, góc) |
| `checkGoal` | Kiểm tra bóng qua vạch goal |
| `setInterval + requestAnimationFrame` | Game loop cố định 60 FPS |

> **Lưu ý:** Physics và render hiện đang nằm lẫn trong cùng 1 file, chưa tách module.

---

## Các file quan trọng khác

| File | Vai trò |
|---|---|
| `agent-api.js` | API cho AI: `getState()`, `setPlayerInput()`, `predictBallPosition()` |
| `bot.js` | Bot AI cơ bản |
| `enhanced-bot.js` | Bot AI nâng cao |
| `map-loader.js` | Load file `.hbs` (map Haxball) |
| `maps/` | Chứa các file map `.hbs` |
| `legacy/` | Phiên bản cũ dùng `HBInit` (cần internet — không dùng nữa) |

---

## State format (`AgentAPI.getState()`)

```js
{
  ball: { x, y, xs, ys, radius, invMass, damping, cMask, cGroup },
  players: [{ team, disc: { x, y, xs, ys, radius, ... }, inputs, bot }],
  stadium: { width, height, spawnDistance, segments, goals, discs, planes, traits, playerPhysics, ballPhysics }
}
```

---

## Bounding box (`AgentAPI.getBoundingBox()`)

Tính kích thước **vùng chơi thực tế** từ các `segments` có trait `cMask: ['ball']` (ballArea).

```js
{ W, H, minX, maxX, minY, maxY }
```

> **Điều kiện mới:** Luôn đảm bảo `W > H` (tự động hoán đổi nếu cần).

---

## Hằng số vật lý
```
// --- Từ valn-v4.hbs ---
ball_radius    = 5.8      // ballPhysics.radius
ball_damping   = 0.99     // ballPhysics.damping
ball_bCoef     = 0.443    // ballPhysics.bCoef
ball_invMass   = 1.5      // ballPhysics.invMass

player_radius  = 15       // playerPhysics.radius
player_damping = 0.96     // playerPhysics.damping
acceleration   = 0.11     // playerPhysics.acceleration
kickStrength   = 4.545    // playerPhysics.kickStrength

// --- Tính từ physics ---
player_max_speed = acceleration / (1 - damping)
               = 0.11 / 0.04 = 2.75  // px/tick (terminal velocity)

// Ball max speed: sau 1 cú kick từ player đang full speed
// kickStrength + transfer từ va chạm ≈ 4.545 + ~2 ≈ 6-7 px/tick
// Đặt max_speed = 10 để có buffer, đủ clamp [-1,1]
max_speed      = 10       // px/tick (hệ số scale, không phải giới hạn vật lý)

// --- Từ enhanced-bot.js ---
kick_range     = 25       // px (center-to-center), surface gap = 25 - 15 - 5.8 = 4.2

// --- Normalize ---
NORM           = 800      // chia tọa độ
DIAG           = 1132     // sqrt(800² + 800²), chia khoảng cách diagonal
```

## Observation format (`AgentAPI.getObs(agentTeam)`)

Observation dạng **object** (không còn là flat array).  
`agentTeam`: 1 = RED, 2 = BLUE.

```js
obs = {
  ## 1. Hằng số sân (mỗi episode)
| Feature | Công thức | Ý nghĩa |
|---|---|---|
| `goal_y` | `goal_y / NORM` | Cột goal trên (goal dưới = -goal_y) |
| `HH_norm` | `HH / NORM` | Tỉ lệ chiều cao sân |
| `HW_norm` | `HW / NORM` | Tỉ lệ chiều rộng sân |
| `agentTeam` | `0 hoặc 1` | Team của agent |

---

## 2. Agent — Bóng
| Feature | Công thức | Ý nghĩa |
|---|---|---|
| `d_to_ball_x` | `(ball_x - my_x) / NORM` | Vector hướng đến bóng (x) |
| `d_to_ball_y` | `(ball_y - my_y) / NORM` | Vector hướng đến bóng (y) |
| `dist_to_ball` | `(dist(ball, my) - player_radius - ball_radius) / DIAG` | Khoảng cách bề mặt đến bóng |
| `can_kick` | `1.0 if dist_to_ball <= kick_range/DIAG else 0.0` | Có thể sút không |
| `path_to_ball_blocked_opp` | `1.0 nếu có opponent cắt ngang đường đến bóng` | Đường đến bóng bị chặn bởi opponent |
| `path_to_ball_blocked_wall` | `1.0 nếu có tường giữa player và bóng` | Đường đến bóng bị chặn bởi tường |

---

## 3. Trạng thái động
| Feature | Công thức | Ý nghĩa |
|---|---|---|
| `ball_x, ball_y` | `/ NORM` | Vị trí bóng |
| `ball_xs, ball_ys` | `/ max_speed` | Vận tốc bóng |
| `my_x, my_y` | `/ NORM` | Vị trí agent |
| `my_xs, my_ys` | `/ max_speed` | Vận tốc agent |
| `my_speed` | `sqrt(my_xs² + my_ys²) / max_speed` | Tổng tốc độ agent |

---

## 4. Game state
| Feature | Công thức | Ý nghĩa |
|---|---|---|
| `time_remaining` | `time_left / max_time` | Thời gian còn lại → [0, 1] |
| `possession` | `sign(opp[0].dist_to_ball - tm[0].dist_to_ball)` | -1 agent team, 0 neutral, +1 opponent |

---

## 5. Teammate — `tm[0..3]`
*(pad zeros nếu không có, sort theo dist_to_ball tăng dần)*

| Feature | Công thức | Ý nghĩa |
|---|---|---|
| `x, y` | `/ NORM` | Vị trí |
| `xs, ys` | `/ max_speed` | Vận tốc |
| `d_to_me_x` | `(tm_x - my_x) / NORM` | Vector hướng đến agent (x) |
| `d_to_me_y` | `(tm_y - my_y) / NORM` | Vector hướng đến agent (y) |
| `dist_to_me` | `(dist(tm, my) - 2*player_radius) / DIAG` | Khoảng cách bề mặt đến agent |
| `d_to_ball_x` | `(ball_x - tm_x) / NORM` | Vector hướng đến bóng (x) |
| `d_to_ball_y` | `(ball_y - tm_y) / NORM` | Vector hướng đến bóng (y) |
| `dist_to_ball` | `(dist(tm, ball) - player_radius - ball_radius) / DIAG` | Khoảng cách bề mặt đến bóng |
| `can_kick` | `1.0 if dist_to_ball <= kick_range/DIAG else 0.0` | Có thể sút không |

**4 teammate**

---

## 6. Opponent — `opp[0..4]`
*(pad zeros nếu không có, sort theo dist_to_ball tăng dần)*

| Feature | Công thức | Ý nghĩa |
|---|---|---|
| `x, y` | `/ NORM` | Vị trí |
| `xs, ys` | `/ max_speed` | Vận tốc |
| `d_to_me_x` | `(opp_x - my_x) / NORM` | Vector hướng đến agent (x) |
| `d_to_me_y` | `(opp_y - my_y) / NORM` | Vector hướng đến agent (y) |
| `dist_to_me` | `(dist(opp, my) - 2*player_radius) / DIAG` | Khoảng cách bề mặt đến agent |
| `d_to_ball_x` | `(ball_x - opp_x) / NORM` | Vector hướng đến bóng (x) |
| `d_to_ball_y` | `(ball_y - opp_y) / NORM` | Vector hướng đến bóng (y) |
| `dist_to_ball` | `(dist(opp, ball) - player_radius - ball_radius) / DIAG` | Khoảng cách bề mặt đến bóng |
| `can_kick` | `1.0 if dist_to_ball <= kick_range/DIAG else 0.0` | Có thể sút không |

}
```

---

## Normalize Observation

Mọi **tọa độ** chia cho `800`, mọi **vận tốc** chia cho `MAX_SPEED`.

### Hằng số normalize

```js
const NORM_POS   = 800    // mọi tọa độ x, y chia cho giá trị này
const MAX_SPEED  = 30     // px/tick — hệ số scale vận tốc (clamp vào [-1, 1])
```

### Công thức normalize

| Loại biến | Công thức |
|---|---|
| **Tọa độ X** | `x_norm = x / 800` → `[-1, 1]` |
| **Tọa độ Y** | `y_norm = y / 800` → `[-1, 1]` |
| **Vận tốc X** | `vx_norm = vx / MAX_SPEED` → `[-1, 1]` |
| **Vận tốc Y** | `vy_norm = vy / MAX_SPEED` → `[-1, 1]` |
| **Giới hạn goal Y** | `goal_norm = goal_y / 800` → `[-1, 1]` |
| **HH_norm, HW_norm** | `HH / 800`, `HW / 800` → `[0.3, 1.0]` |

> **Lưu ý:**  
> - Gốc tọa độ `(0, 0)` là **tâm sân** → `left_limit/800 = -HW/800`, `right_limit/800 = HW/800`.  
> - Vì `HW ≤ 800`, tọa độ x luôn nằm trong `[-1, 1]`. Tương tự cho y.  
> - Vận tốc được **clamp** vào `[-1, 1]` nếu vượt `MAX_SPEED`.  
> - `MAX_SPEED` là hệ số scale cố định, không phải tốc độ vật lý thực tế.



---

## Build Field Segments (Physics)

Khi khởi tạo episode với kích thước `HW × HH`, cần build 4 tường nảy theo đúng physics futsal **và** khoét gap cho goal ở 2 cạnh trái/phải.

### Cấu trúc sân

```
      (-HW, -HH) ─────────── top ─────────── (HW, -HH)
           │                                       │
       left│  [gap: -goal_y .. +goal_y = goal]  right│
       wall│                                    wall│
           │                                       │
      (-HW, +HH) ─────────── bot ─────────── (HW, +HH)
```

- **Top / Bottom:** 1 segment liên tục từ `-HW` → `+HW`, nảy bình thường (`bCoef = ball_bCoef = 0.443`)
- **Left / Right:** mỗi bên **2 segment**, khoét gap `[-goal_y, +goal_y]`:
  - segment trên: `(-HW, -HH)` → `(-HW, -goal_y)`
  - segment dưới: `(-HW, +goal_y)` → `(-HW, +HH)`
  - Phần giữa `[-goal_y, +goal_y]` **không có segment** → bóng xuyên qua = ghi bàn

### bCoef của tường

Futsal thật: tường cứng, ít mất tốc độ → dùng `bCoef = 1.0` cho 4 tường.

| Cạnh | bCoef |
|---|---|
| Top / Bottom | `1.0` |
| Left / Right (phần ngoài goal) | `1.0` |

> Khác với `ball_bCoef = 0.443` là hệ số **ball-to-ball** khi bóng va vào goalpost, không áp cho tường phẳng.

### Goal detection (đúng chuẩn HaxBall)

HaxBall xác định goal bằng **goal line segment** — một đường thẳng vô hình nằm ngay trên cạnh sân, bóng "vượt qua" khi center bóng băng qua đường này.

Implement trong physics loop:

```js
// Mỗi tick, sau khi cập nhật vị trí bóng:
function checkGoal(ball, prevBall, HW, goal_y) {
  // Kiểm tra goal bên TRÁI (Red goal, Blue ghi bàn)
  if (prevBall.x >= -HW && ball.x < -HW) {
    // Nội suy y tại x = -HW
    const t = (-HW - prevBall.x) / (ball.x - prevBall.x);
    const crossY = prevBall.y + t * (ball.y - prevBall.y);
    if (Math.abs(crossY) <= goal_y) return 'BLUE';  // Blue scores
  }
  // Kiểm tra goal bên PHẢI (Blue goal, Red ghi bàn)
  if (prevBall.x <= HW && ball.x > HW) {
    const t = (HW - prevBall.x) / (ball.x - prevBall.x);
    const crossY = prevBall.y + t * (ball.y - prevBall.y);
    if (Math.abs(crossY) <= goal_y) return 'RED';   // Red scores
  }
  return null;  // không có goal
}
```

> **Logic:** Nội suy vị trí y tại thời điểm bóng băng qua `x = ±HW`. Nếu `|crossY| ≤ goal_y` thì bóng đi qua cửa goal → ghi bàn.  
> Phương pháp này chính xác ở mọi tốc độ (không bị bỏ qua khi bóng bay nhanh qua goal trong 1 tick).

---

## State Pipeline


Hệ thống duy trì **2 dạng state song song** tại mỗi tick:

```
raw_state  ──────────────────────────────►  physics engine (px, px/tick)
                 │
                 ▼ get_normalized_state()
norm_state ──────────────────────────────►  neural network input ([-1,1])
```

### Raw state (đơn vị vật lý)

Lấy trực tiếp từ `AgentAPI.getState()`, đơn vị **px / px per tick**:

```js
raw = {
  ball:   { x, y, xs, ys },
  my:     { x, y, xs, ys },
  opp:    { x, y, xs, ys },
  HW, HH,           // nửa chiều rộng/cao sân (px)
  goal_y,           // cột goal trên (px), goal dưới = -goal_y
}
```

> Raw state dùng cho physics simulation, tính reward, predict trajectory.

---

### `get_normalized_state(raw, agentTeam)`

Chuyển raw → flat array `[-1, 1]` đưa thẳng vào network:

```js
function get_normalized_state(raw, agentTeam) {
  const { ball, my, opp, HW, HH, goal_y } = raw;

  // Flip x nếu BLUE → agent luôn "đá sang phải" từ góc nhìn network
  const flip = (agentTeam === 2) ? -1 : 1;

  return [
    // --- Sân: 3 features ---
    HW     / NORM,                   // nửa chiều rộng sân
    HH     / NORM,                   // nửa chiều cao sân
    goal_y / NORM,                   // cột goal (>0 = phía dưới)

    // --- Bóng: 4 features ---
    flip * ball.x  / NORM,
           ball.y  / NORM,
    flip * ball.xs / MAX_SPEED,
           ball.ys / MAX_SPEED,

    // --- Agent: 4 features ---
    flip * my.x  / NORM,
           my.y  / NORM,
    flip * my.xs / MAX_SPEED,
           my.ys / MAX_SPEED,

    // --- Opponent: 4 features ---
    flip * opp.x  / NORM,
           opp.y  / NORM,
    flip * opp.xs / MAX_SPEED,
           opp.ys / MAX_SPEED,
  ];  // shape: (15,)
}
```

> **BLUE flip:** Flip trục x để cả 2 team dùng chung 1 policy — agent luôn thấy goal đối thủ bên phải.

---

### Action loop (1 step)

```
agent ──► action { dx, dy ∈ [-1,1], kick ∈ {0,1} }
               │
               ▼  setPlayerInput(action)
          physics engine ──► tick() × 1  (1/60 s)
               │
               ▼  getState()
          raw_state ──► get_normalized_state() ──► next obs
               │
               ▼
          reward(), done()
```

- **Mỗi step = đúng 1 tick** — action được nạp vào input, engine chạy một bước
- **Reward & done** tính từ **raw state** (đơn vị px, không normalize)
- **Normalized state** chỉ dùng làm input cho network, không ảnh hưởng physics

---

## Kích thước sân ngẫu nhiên (Domain Randomization)

Để agent tổng quát hóa, kích thước sân được random mỗi episode:

```js
HH = rand(240, 800)      // nửa chiều cao sân (px), range [240, 800]
HW = rand(HH, 800)       // nửa chiều rộng sân (px), đảm bảo HW ≥ HH (sân ngang hoặc vuông)

// Từ đó:
left_limit  = -HW,  right_limit = HW
upper_limit = -HH,  lower_limit = HH
```

> **Điều kiện:** `HH ≤ HW ≤ 800` — sân luôn ngang hoặc vuông, không bao giờ dọc.

### Encode vào observation

`HH_norm` và `HW_norm` được thêm vào `obs` để agent biết sân đang to nhỏ ra sao:

```js
HH_norm = HH / 800   // ∈ [0.3, 1.0]
HW_norm = HW / 800   // ∈ [HH/800, 1.0]
```

> Agent có thể suy ra tỉ lệ sân từ `HH_norm / HW_norm` nếu cần.

---

## Giới hạn hiện tại

- **Không thể tăng tốc training** — game loop ghim 60 FPS qua `requestAnimationFrame`
- **Physics chưa tách khỏi render** — cần refactor `script.js` để chạy headless
- **`spawnDistance`** là bán kính vòng tròn spawn, không thể quy định vị trí cụ thể qua HBS

---

## Mục tiêu: Train AI Agent cho `bot.js`

AI được train theo **curriculum learning** — từ đơn giản đến phức tạp.

> **⚠️ Quan trọng:** Khi train AI, luôn dùng **physics của map futsal đang deploy** chứ không phải Classic.
>
> ### Map đang dùng: `valn-v4.hbs`
>
> **Ball physics:**
>
> | Thuộc tính | Giá trị |
> |---|---|
> | `radius` | **5.8** |
> | `bCoef` | 0.443 |
> | `invMass` | 1.5 |
> | `damping` | 0.99 |
> | `cGroup` | `["ball"]` |
> | `cMask` | `["all"]` |
>
> **Player physics:**
>
> | Thuộc tính | Giá trị |
> |---|---|
> | `radius` | **15** |
> | `bCoef` | 0 |
> | `invMass` | 0.5 |
> | `damping` | 0.96 |
> | `acceleration` | 0.11 |
> | `kickingAcceleration` | 0.083 |
> | `kickingDamping` | 0.96 |
> | `kickStrength` | 4.545 |
> | `kickback` | 0 |
> | `cGroup` | `["red", "blue"]` |
>
> **Kích thước sân (từ ballArea segments):**
>
> | Giới hạn | Giá trị |
> |---|---|
> | `left_limit` | **-700** |
> | `right_limit` | **700** |
> | `upper_limit` | **-320** |
> | `lower_limit` | **320** |
> | `upper_goal_limit` | **-85** (cột trên) |
> | `lower_goal_limit` | **85** (cột dưới) |
> | Chiều rộng W | 1400 |
> | Chiều cao H | 640 |

---

## Hàm hỗ trợ: `evaluate_shot`

Gọi sau mỗi lần agent thực hiện cú shot. Giả lập toàn bộ player đều cắt đường bóng theo hướng **tối ưu (intercept path)**, sau đó so sánh thời gian chạm bóng:

| Kết quả | Reward |
|---|---|
| Shot thẳng vào goal, không ai cắt được | ✅ Thưởng lớn |
| Team mình chạm bóng trước với khoảng cách đáng kể | ✅ Thưởng |
| Team địch chạm bóng trước với khoảng cách đáng kể | ❌ Phạt |
| Hai bên chạm nhau xít xìn xịt | ➖ Vô thưởng vô phạt |

---

## Curriculum

### A0 — Ghi bàn cơ bản (map cố định)

- **Setup:** 1 player, không đối thủ, không vật cản
- **Random:** vị trí player, phe, vị trí bóng
- **Reward:**
  - ✅ Ghi bàn đúng phe → thưởng, càng sớm càng cao
  - ❌ Tự ghi vào goal mình → phạt
  - ❌ Hết time → 0

---

### A1 — Ghi bàn + chất lượng shot (map cố định)

- **Setup:** giống A0
- **Thêm:** reward từ `evaluate_shot` sau mỗi cú sút

---

### A2 — Train thủ (A2 vs A1, 1 goal)

- **Setup:** A2 (đang train) vs A1 (frozen), **chỉ có goal của phe thủ**
- **Vị trí:** bóng + A1 đặt ở vị trí **thuận lợi tấn công** (gần goal, đủ để A1 ghi bàn nhanh)
- **Flow mỗi episode:** A1 tấn công → A2 cản → reset (dù cản được hay bị goal)
- **Reward A2:**
  - ✅ Cản được → thưởng
  - ❌ Bị ghi bàn → phạt
- A1 không update weights, chỉ A2 học

---

### A3 — Self-play (2 goal)

- **Điều kiện:** A2 đủ mạnh ở cả công và thủ
- **Setup:** self-play, sân đầy đủ 2 goal
- **Reward:** chưa xác định chi tiết — sẽ bổ sung sau
