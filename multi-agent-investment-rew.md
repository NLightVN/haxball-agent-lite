# Hệ thống Phần thưởng Đa tác vụ và Chuỗi Đầu tư (Multi-Agent Investment Reward)

Tài liệu này mô tả logic tính toán phần thưởng (reward) dựa trên quyền kiểm soát bóng (possession) và cơ chế phân phối phần thưởng theo chuỗi phối hợp (investment sequence).

## 1. Hệ Thống Possession (Quyền kiểm soát bóng)

Có 2 loại possession cần theo dõi đồng thời:
1. **Possession hiện tại**: Đội/cầu thủ chạm bóng gần nhất.
2. **Possession liền trước (trong vòng $\le 0.25$ giây)**: Trạng thái kiểm soát bóng trong khoảng thời gian lùi lại 0.25s kể từ thời điểm hiện tại.

### 1.1. Quy tắc cập nhật Possession
Possession được cập nhật mỗi lần có người chạm bóng.
- **Tình huống bóng bật ra**: Nếu đối thủ chạm bóng, sau đó sút đập vào người tôi:
  - Possession hiện tại: Team mình.
  - Truy nguoc lai 0.25 s ve truoc xem co doi thu khong, neu co thi update Possession liền trước 0.25s: Đối thủ.
- **Tình huống chuyền bóng**: Nếu tôi chạm bóng, chuyền cho đồng đội và đồng đội chạm bóng, possession được cập nhật lại:
  - Nếu trong 0.25s trước đó không có đối thủ chạm bóng $\rightarrow$ Possession liền trước 0.25s cập nhật thành team mình.
  - Nếu trong 0.25s trước đó vẫn có đối thủ chạm bóng $\rightarrow$ Possession liền trước 0.25s cập nhật thành đối thủ.
- **Tình huống tự dẫn bóng**: Một cầu thủ team mình chạm bóng và giữ bóng (không ai khác chạm) trong $\ge 0.25$ giây. Khi chính cầu thủ đó chạm bóng lần tiếp theo trong vòng 0.25s kể từ lần chạm trước: tính là lần chạm gần nhất. Lùi về trước 0.25s kể từ lúc đó, do không có possession của đối thủ $\rightarrow$ Possession liền trước 0.25s thuộc về team mình. 
  - *Sự khác biệt*: Mốc truy vết lùi lại 0.25s sẽ di chuyển theo thời gian của các lần chạm, tương tác.

## 2. Phần thưởng dựa trên quỹ đạo bóng (Ball Movement Reward)

### 2.1. Điều kiện Thưởng (Reward) và Phạt (Penalty)
- **Thưởng (Reward)** khi bóng tiến lên theo trục X nếu:
  - Possession hiện tại đang thuộc team mình, **HOẶC**
  - Possession liền trước $\le 0.25$ giây thuộc team mình.
- **Phạt (Penalty)** khi bóng quay về theo trục X nếu:
  - Possession hiện tại đang thuộc team đối thủ, **HOẶC**
  - Possession liền trước $\le 0.25$ giây thuộc team mình (tình huống mất bóng về phía sau).

### 2.2. Ngoại lệ (Exceptions)
- **Phạt nhẹ khi chuyền về**: Nếu possession liền trước 0.25s **VÀ** possession hiện tại đều là team mình $\rightarrow$ **Bị phạt bằng 1/3** (so với khi để mất bóng vào tay đối thủ) khi bóng bay về phía đội mình.
- **Không thưởng khi đối thủ chuyền về**: Nếu possession liền trước 0.25s **VÀ** possession hiện tại đều là đối thủ $\rightarrow$ **Không thưởng** khi bóng bay về phía  đối thủ.

## 3. Phân chia Zone (Khu vực sân)

Sân được chia làm 3 khu vực. Việc tính khoảng cách để làm cơ sở thưởng/phạt (áp dụng các điều kiện possession ở Mục 2) được tính riêng cho từng vùng:

1. **Zone Teammate (Khu vực sân nhà: từ `x_goal_team_minh` tới `x1`)**
   - Khoảng cách được tính **từ bóng tới tâm goal đội mình**.
   - Bóng càng xa tâm goal $\rightarrow$ Càng thưởng (nếu có ít nhất 1 trong 2 loại possesion ở mục 2).
   - Bóng càng gần tâm goal $\rightarrow$ Càng phạt (nếu có ít nhất 1 trong 2 loại possesion ở mục 2).
2. **Zone Mid (Khu vực giữa sân: từ `x1` đến `x2`)**
   - Khoảng cách chỉ tính theo **trục X**.
   - Bóng càng tiến về phía `x2` $\rightarrow$ Càng thưởng (nếu có ít nhất 1 trong 2 loại possesion ở mục 2).
   - Bóng càng lùi về phía `x1` $\rightarrow$ Càng phạt (nếu có ít nhất 1 trong 2 loại possesion ở mục 2).
3. **Zone Opponent (Khu vực sân khách: từ `x2` đến `x_goal_team_ban`)**
   - Bóng càng gần goal đối thủ $\rightarrow$ Càng thưởng (nếu có ít nhất 1 trong 2 loại possesion ở mục 2).
   - Bóng càng xa goal đối thủ $\rightarrow$ Càng phạt (nếu có ít nhất 1 trong 2 loại possesion ở mục 2).

## 4. Chuỗi đầu tư (Investment Sequence)

Cơ chế Investment Sequence giúp chia sẻ phần thưởng cho những cầu thủ tham gia vào chuỗi phối hợp, ngay cả khi họ không trực tiếp là người cầm bóng cuối cùng.

### 4.1. Quy tắc hình thành và biến đổi chuỗi
- **Thêm vào chuỗi**: Khi chuyền cho 1 người và người đó chạm bóng hoặc sút thành công, người đó sẽ trở thành người tiếp theo trong Investment Sequence.
- **Vòng lặp trong chuỗi**: Khi ai đó nhận lại bóng mà bản thân đã có sẵn trong Investment Sequence, những người đứng sau vị trí của người đó trong chuỗi sẽ bị loại bỏ.
- **Đối với Agent được train**: Khi **Tôi** (Agent được train và nhận observation) nhận bóng $\rightarrow$ Investment Sequence **trở về 0** (reset).

### 4.2. Cắt đứt chuỗi (Interruption)
- Nếu đối thủ có possession **$\ge 2$ giây**: Investment Sequence biến mất hoàn toàn. Đội không còn được hưởng lợi từ chuỗi đầu tư này nữa.
- Nếu đối thủ có possession **$< 2$ giây** (mất bóng chóng vánh): Investment Sequence **chưa bị xóa** mà tiếp tục được cập nhật.
- *Lưu ý*:
  - Phần đếm ngược 2 giây này sẽ được đưa vào Observation (obs).
  - Investment Sequence chỉ tính theo possession hiện tại, không liên quan đến possession liền trước 0.25s.

### 4.3. Phân bổ Phần thưởng
- **Người cầm bóng (Trực tiếp)**: Là người ảnh hưởng trực tiếp đến lợi ích liên quan đến việc bóng đi về goal đối phương, tạo cơ hội ghi bàn... và nhận phần thưởng chính.
- **Người đầu tư (Investors)**: Những cầu thủ nằm trong chuỗi phối hợp trước đó sẽ nhận được phần thưởng chia sẻ theo công thức:
  
  **$\text{Reward của người Invest} = 30\% \times \left(\frac{1}{2}\right)^{\text{Số lượng người trong investment sequence} - 1}$**
  
  *(Tức là người chuyền gần nhất được hưởng 30%, người trước đó nữa hưởng 15%, người trước nữa hưởng 7.5%,...)*
  Khi ghi bàn sẽ có bonus ghi bàn sớm: nếu ai có người ghi bàn trong investment sequence sẽ đc nhận bonus theo phần trăm, người ghi bàn ăn 100% bonus.
  Reward ghi bàn gốc và bị ghi bàn gốc là như nhau

## 5. Cơ chế Cân bằng Đặc biệt (Chống Farm Solo)

Nhằm tránh tình trạng agent lạm dụng việc tự dẫn bóng (dribble) hoặc tự chuyền cho chính mình (đập tường) để "farm" reward dâng bóng liên tục, trong bối cảnh mức phạt bị giảm chỉ còn 1/3, các cơ chế sau được áp dụng:

### 5.1. Giảm thưởng khi tự rê dắt (Dribble)
- Nếu 1 agent liên tục hold bóng, tự đi bóng (dribble) mà không chịu chuyền cho đồng đội:
  - **Phần thưởng cho việc đưa bóng tiến lên (advance/gaining advantage) bị giảm xuống chỉ còn 1/3** (tức là phần thưởng giảm xuống đúng bằng với mức bị phạt).
  - **Ngoại lệ:** Khi agent tung ra cú sút thực sự (shoot), phần thưởng cho cú sút vẫn được giữ nguyên (vẫn gấp 3 lần mức phạt bình thường).

### 5.2. Chống tự chuyền bóng (Self-pass)
- Khi 1 agent sút bóng, hệ thống sẽ đánh giá quỹ đạo sút để dự đoán ai là người có khả năng cắt đường chuyền tốt nhất (tính cả các pha sút đập tường nảy lại).
- Nếu người nhiều khả năng nhận được bóng nhất (nhận gần nhất) lại **chính là agent vừa sút**, thì phần thưởng cho đường đi của bóng cũng **chỉ còn 1/3**.
- **Điều kiện gỡ bỏ (Reset):** Trong quá trình sút/chuyền đó, nếu bóng va chạm với bất kỳ **đồng đội (teammate)** hay **đối thủ (opponent)** nào, cơ chế hạn chế đặc biệt này sẽ lập tức biến mất (trở lại mức thưởng 100%).
