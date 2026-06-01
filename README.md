# Haxball Agent Lite ⚽🤖

Một dự án huấn luyện AI (Reinforcement Learning) chơi Haxball sử dụng Python, **Stable Baselines3 (PPO)** và **Pygame**.
Dự án bao gồm một môi trường vật lý (physics environment) giả lập chính xác gameplay của Haxball, cùng với các kịch bản huấn luyện nâng cao như Self-Play (PFSP) và Curriculum Learning.

---

## 🚀 Hướng Dẫn Bắt Đầu (Quick Start)

### 1. Cài đặt yêu cầu (Requirements)
Dự án yêu cầu Python 3.8+ và các thư viện sau:
```bash
pip install numpy pygame stable-baselines3 tensorboard
```

### 2. Chơi với AI (Interactive Play)
Bạn có thể tự mình thi đấu với mô hình AI đã được huấn luyện hoặc xem AI tự đá bằng cách chạy file `play.py`.
```bash
python play.py
```

**🎮 Điều khiển khi chơi (Human vs AI):**
- **Cụm phím Mũi tên**: Di chuyển (Lên/Xuống/Trái/Phải)
- **Enter / Ctrl Phải**: Sút bóng (Kick)
- **Space**: Tạm dừng (Pause) / Tiếp tục (Resume)
- **R**: Chơi lại ván mới ngay lập tức
- **Q / ESC**: Thoát game

---

## 📁 Cấu Trúc Dự Án (Project Structure)

```
haxball-agent-lite/
├── play.py               # Script chơi game tương tác (Human vs AI hoặc AI vs AI)
├── eval.py               # Script đánh giá hiệu suất của mô hình
├── eval_render.py        # Đánh giá mô hình có giao diện hiển thị
├── training/             # Chứa mã nguồn môi trường và huấn luyện
│   ├── env.py            # Môi trường giả lập vật lý Haxball (Gym Environment)
│   ├── train_a0.py       # Script huấn luyện PPO cơ bản
│   └── train_a1_pfsp.py  # Script huấn luyện nâng cao bằng Self-Play (PFSP)
├── models/               # Thư mục chứa các model checkpoints đã lưu
├── tensorboard/          # Chứa log huấn luyện cho Tensorboard
└── js_env/               # Bản gốc hoặc giao diện web LITE (HTML/JS)
```

---

## 🤖 Huấn Luyện AI (Training)

Môi trường AI được thiết kế theo quy chuẩn Gym Env, hỗ trợ **Curriculum Learning** và **Prioritized Fictitious Self-Play (PFSP)**.

### Huấn luyện Cơ bản (Phase A0)
```bash
python training/train_a0.py
```
Dùng để huấn luyện agent những kỹ năng cơ bản ban đầu (vd: sút bóng vào lưới trống).

### Huấn luyện Self-Play (Phase A1)
```bash
python training/train_a1_pfsp.py
```
Sử dụng phương pháp **PFSP** để Agent tự đấu với chính các phiên bản cũ của mình (checkpoints) hoặc các bot cơ bản (Random, Wanderer...) nhằm nâng cao khả năng thi đấu đối kháng.

**Theo dõi quá trình huấn luyện bằng TensorBoard:**
```bash
tensorboard --logdir tensorboard/
```

---

## ✨ Tính Năng Nổi Bật (Features)
✅ **Môi trường vật lý (Physics Env) hoàn chỉnh bằng Python**: Chạy cực nhanh, không cần server hay browser.  
✅ **Thuật toán PPO tối ưu**: Tích hợp sẵn Stable-Baselines3.  
✅ **Curriculum Learning**: Tự động tăng độ khó mục tiêu, giúp AI học nhanh và hiệu quả hơn.  
✅ **Self-Play (PFSP)**: Quản lý đối thủ thông minh, giúp Agent học hỏi từ điểm yếu của đối thủ cũ mà không bị "bỏ quên" (forgetting).  
✅ **Giao diện Pygame mượt mà**: Debug trực quan, tốc độ 60 FPS chuẩn Haxball.

---

> 💡 **Tip:** Nếu bạn muốn tinh chỉnh cấu hình của trận đấu (Agent nào đấu với Agent nào, thay đổi thành Human vs AI, v.v.), hãy chỉnh sửa các hằng số ở phần `Configuration` trong file `play.py`.
