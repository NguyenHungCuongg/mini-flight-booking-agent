# Báo cáo BTVN#4: Persistent Memory cho agent đặt vé máy bay

|           |                                                         |
| --------- | ------------------------------------------------------- |
| Môn học   | SE373: Kỹ thuật xây dựng hệ thống Agentic AI            |
| Bài       | BTVN#4: Challenge A, Persistent Memory (Buổi 7)         |
| Họ và tên | `Nguyễn Hùng Cường`                                     |
| MSSV      | `23520201`                                              |
| Lớp       | `SE373.R11`                                             |
| Repo      | `https://github.com/NguyenHungCuongg/mini-flight-booking-agent` |

## Tổng quan

### Agent được chọn: agent đặt vé máy bay của BTVN#3

Đề Challenge A cho phép "tái sử dụng agent đã có". Bài này dùng lại **agent đặt vé máy bay** em đã xây dựng ở BTVN#3, cùng repo ở trên. Tóm tắt agent đó:

- **Nhiệm vụ:** người dùng nhờ đặt một vé máy bay theo yêu cầu, ví dụ "SGN đi DAD ngày 2026-10-07, bay trước 12:00, giá không quá 2.000.000 VND". Agent tìm chuyến, kiểm tra ghế, giữ chỗ, thanh toán, rồi đọc lại booking để xác nhận.
- **Công nghệ:** Python, LangChain (`create_agent` cùng middleware). Model là LLM thật; bài này dùng Gemini `gemini-3.5-flash` (riêng session S4 của demo dùng `gemini-3.5-flash-lite` vì free tier chỉ cho 20 request/ngày mỗi model) qua thư viện `langchain-google-genai` (BTVN#3 dùng Qwen qua API tương thích OpenAI).
- **Công cụ (tool):** 5 tool giả lập với dữ liệu cố định, gồm `search_flights`, `check_seat`, `book_seat`, `pay`, `get_booking`. Không có hệ thống bán vé thật nào bị gọi.
- **Kiến trúc `Agent = Model + Harness`:** model chỉ *đề xuất* lời gọi tool. Harness là phần code bao quanh model, quyết định lời gọi đó có được chạy hay không, và tự kiểm tra bằng code xem việc đặt vé đã thật sự xong chưa. Harness gồm:
  - Ràng buộc của người dùng được lưu thành dữ liệu (`Constraints`).
  - Kiểm tra quyền (permission check) trước mỗi tool: chặn đặt vé sai ràng buộc, hỏi người duyệt khi thanh toán vượt hạn mức.
  - Kiểm tra hoàn thành bằng cách đọc lại booking.
  - Gói bàn giao (handoff) cho người khi agent phải dừng.
  - Thêm giới hạn số lần gọi model (budget) và phát hiện lặp.
- **Ba mẫu thiết kế:** ReAct, Plan-then-Execute và Lai (Hybrid). Cả ba chạy trên cùng harness và được đánh giá trên 8 kịch bản (xem `REPORT.md`).

### Vì sao agent này hợp với Challenge A

Ở BTVN#3, mỗi lần chạy là độc lập: người dùng phải nhắc lại đầy đủ "bay trước 12:00, giá tối đa 2 triệu" trong mọi yêu cầu. Trong khi đó, một người hay đi máy bay thường có **thói quen cố định**, ví dụ "tôi chỉ bay buổi sáng" hay "không mua vé quá 2 triệu". Đó chính là "quy ước" mà đề yêu cầu agent phải nhớ: người dùng nói một lần, agent áp dụng cho các lần đặt vé sau, và cập nhật khi người dùng đổi ý.

Agent này còn có sẵn một điểm thuận lợi: ràng buộc vốn đã là dữ liệu mà harness dùng để chặn bằng code. Vì vậy quy ước lấy từ memory không chỉ được "nhắc" cho model trong prompt, mà còn được harness **kiểm tra bằng code** trước mỗi lần đặt vé. Nếu model quên hoặc bị lừa, harness vẫn không cho đặt vé sai quy ước.

### Bài này bổ sung những gì

| Thành phần mới                  | Vai trò                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| `flight_agent/memory.py`        | Memory Store (`MemoryStore`): lưu quy ước vào `memory/preferences.json`, đánh dấu bản cũ khi quy ước đổi |
| Tool `remember_preference`      | Agent ReAct tự gọi để lưu quy ước khi người dùng nói "từ giờ ...", có người dùng xác nhận          |
| Thay đổi trong `run()`          | Mở session mới thì nạp quy ước vào cả context của model lẫn ràng buộc của harness                 |
| `flight_agent/memory_demo.py`   | Demo: mỗi lần chạy là một session mới, in ra phần memory đã inject                               |
| `tests/test_memory.py`          | 25 test tự động, gồm test cho Chứng minh 1 và Chứng minh 2                                        |

Toàn bộ phần của BTVN#3 vẫn giữ nguyên: 26 test cũ pass mà không phải sửa, và khi chạy không có memory thì agent hoạt động y hệt trước.

### Cấu trúc báo cáo

Mục 1 nêu yêu cầu của đề. Mục 2 và 3 định nghĩa "quy ước" và các khái niệm State, History, Memory, Context trong agent này. Mục 4 và 5 trình bày thiết kế theo năm câu gợi ý của đề. Mục 6 là demo với LLM thật, gồm hai chứng minh. Mục 7 là kiểm thử tự động. Mục 8 và 9 là hạn chế và kết luận. Mã nguồn đầy đủ nằm trong repo; báo cáo chỉ dẫn link tới từng file và hàm liên quan.

## 1. Yêu cầu bài

Theo slide Buổi 7, mục I.9:

> **Mục tiêu:** Xây dựng cơ chế giúp agent nhớ được quy ước dự án qua nhiều session và tự cập nhật khi quy ước thay đổi (có thể tái sử dụng agent đã có).
>
> 1. _Chứng minh 1:_ Mở session mới, agent vẫn áp dụng đúng quy ước đã thống nhất ở session trước.
> 2. _Chứng minh 2:_ Khi quy ước thay đổi, các session tiếp theo phải sử dụng quy ước mới.

Mức độ hoàn thành:

| Yêu cầu                                   | Trạng thái                          | Nằm ở                                                         |
| ----------------------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| Lưu quy ước xuống đĩa                     | Xong                                | `MemoryStore` trong `flight_agent/memory.py`                  |
| Agent tự ghi quy ước khi người dùng nói   | Xong, có người dùng xác nhận        | `Harness.remember_preference` trong `flight_agent/core.py`    |
| Nạp quy ước khi mở session mới            | Xong, inject + harness kiểm tra    | `run()` trong `flight_agent/run.py`, `flight_agent/react.py`  |
| Quy ước đổi: bản cũ thành `superseded`    | Xong                                | `MemoryStore.remember`                                        |
| Log phần memory đã inject                 | Xong                                | `RunResult.memory_injected`, `flight_agent/memory_demo.py`    |
| Chứng minh 1 và 2                         | Test tự động: xong; demo LLM thật: mục 6 | `tests/test_memory.py`, mục 6                            |
| Không làm hỏng BTVN#3                     | 26 test cũ vẫn pass, tổng 52 test   | `uv run pytest -q`                                            |

## 2. "Quy ước" trong bài này

Với agent đặt vé, quy ước là **sở thích lâu dài của người dùng**, nói một lần và áp dụng cho mọi lần đặt sau. Bài chọn hai quy ước:

| Khoá            | Ý nghĩa                   | Ví dụ người dùng nói                    | Dạng lưu    |
| --------------- | ------------------------- | --------------------------------------- | ----------- |
| `depart_before` | Chỉ bay trước giờ này     | "Từ giờ tôi chỉ bay trước 11:00"        | `"11:00"`   |
| `max_price`     | Giá vé tối đa (VND)       | "giá vé tối đa 2.000.000 VND"           | `2000000`   |

Lý do chọn hai khoá này: chúng chính là hai trường của `Constraints` trong [flight_agent/core.py](https://github.com/NguyenHungCuongg/mini-flight-booking-agent/blob/main/flight_agent/core.py). Ở BTVN#3, `Constraints` là dữ liệu mà harness dùng để **chặn bằng code** các booking vi phạm (`Harness.check_permission`) và để **kiểm tra hoàn thành** (`Harness.is_done`). Vì quy ước được đổ vào `Constraints`, nên nó không chỉ là một dòng nhắc trong prompt: dù model quên hay bị lừa, harness vẫn không cho đặt vé sai quy ước.

## 3. State, History, Memory và Context trong agent

Ánh xạ bảng ở slide I.2 vào code của bài:

| Thành phần  | Trả lời câu hỏi                | Trong agent này                                                                                       | Sống bao lâu       |
| ----------- | ------------------------------ | ----------------------------------------------------------------------------------------------------- | ------------------ |
| **State**   | Đang làm đến bước nào?         | Các trường của `Harness`: `model_calls`, `stop_reason`, `world.bookings`, `scenario.constraints`      | Một session        |
| **History** | Đã có những tương tác gì?      | Tin nhắn của LangChain trong một lần `run_react`, và `Harness.trace` (mọi lời gọi tool kèm kết quả)    | Một session        |
| **Memory**  | Điều gì cần ghi nhớ lâu dài?   | File `memory/preferences.json` do `MemoryStore` quản lý                                               | Qua nhiều session  |
| **Context** | LLM đang nhìn thấy gì lúc này? | `SYSTEM_PROMPT` + khối memory (`MemoryStore.to_prompt()`) + yêu cầu của người dùng + history          | Một lần gọi model  |

Một **session** = một lần gọi `run()` = một `Harness` mới, `World` mới và lịch sử hội thoại rỗng. Thứ duy nhất được giữ lại giữa các session là file memory. Trong demo, mỗi session còn là một tiến trình Python riêng, nên không thể có chuyện "nhớ nhờ biến còn trong RAM".

## 4. Thiết kế

Phần này trả lời lần lượt năm câu gợi ý của đề.

### 4.1. Lưu ở đâu: một file JSON

Memory là một danh sách các mục trong `memory/preferences.json`. Mỗi mục có dạng:

```json
{
  "key": "depart_before",
  "value": "11:00",
  "status": "active",
  "since": "2026-10-22T10:02:41",
  "source": "Từ giờ tôi chỉ bay trước 11:00 ... Đặt cho tôi một vé SGN -> DAD ngày 2026-10-07, chọn chuyến rẻ nhất.",
  "superseded_at": null
}
```

- `status`: `active` (đang hiệu lực) hoặc `superseded` (đã bị thay thế).
- `since`, `superseded_at`: thời điểm ghi và thời điểm bị thay thế.
- `source`: nguyên văn yêu cầu của người dùng trong session đã sinh ra quy ước này.

Lý do: theo bảng so sánh ở slide I.8, JSON/Markdown "dễ cài đặt, dễ đọc và kiểm tra bằng mắt", còn điểm yếu "khó tìm kiếm khi dữ liệu phình to" không áp dụng vì bài chỉ có vài mục. SQLite phải thiết kế schema và migration; vector retrieval tốn embedding và dễ gặp kết quả nhiễu (false matches). Cả hai đều thừa ở quy mô này. File JSON còn có lợi cho chính việc chứng minh: mở file ra là thấy đủ lịch sử thay đổi.

`MemoryStore` nằm trong [flight_agent/memory.py](https://github.com/NguyenHungCuongg/mini-flight-booking-agent/blob/main/flight_agent/memory.py), gồm các hàm:

| Hàm                                  | Việc làm                                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------------------ |
| `entries()`                          | Đọc mọi mục, kể cả mục đã `superseded`. Chưa có file thì trả danh sách rỗng.                |
| `active()`                           | Trả `{khoá: giá trị}` của các mục đang `active`.                                           |
| `remember(key, value, source)`       | Kiểm tra và chuẩn hoá giá trị, đánh dấu bản cũ `superseded`, ghi bản mới. Trả kết quả cùng dạng với kết quả tool (`status: ok` hoặc `invalid_param`). |
| `to_prompt()`                        | Sinh đúng khối văn bản được inject vào context.                                            |
| `apply(constraints)`                 | Trả `Constraints` mới, trong đó các trường tương ứng lấy giá trị từ memory.                                    |

`remember` chuẩn hoá giá trị trước khi lưu, vì model có thể truyền `"9:00"`, `"2,000,000"` hay `"2.000.000"`. Giờ luôn được lưu dạng `HH:MM` có số 0 ở đầu (`"09:00"`), vì `Constraints.is_ok` so sánh giờ như chuỗi: `"9:00" < "12:00"` là **sai** theo thứ tự chuỗi, nên nếu lưu nguyên `"9:00"` thì harness sẽ chặn nhầm mọi chuyến. Giá có phần thập phân (`"2000000.0"`, `"1.5"`) bị từ chối thay vì bỏ dấu chấm, vì bỏ dấu chấm sẽ biến `2000000.0` thành 20 triệu. Khoá lạ hoặc giá trị không hợp lệ (`"9h"`, `"25:00"`, `"abc"`, số âm) bị trả `invalid_param` và không ghi gì xuống đĩa.

### 4.2. Ai ghi: agent gọi tool, người dùng xác nhận

Bài dùng mẫu **memory tools** (slide I.7): agent ReAct có thêm tool `remember_preference(key, value)` và tự quyết định gọi khi người dùng nói một quy ước lâu dài ("từ giờ ..."). Cách này không cần thêm lần gọi LLM riêng để trích xuất quy ước từ hội thoại, nên không tốn thêm chi phí và độ trễ.

Tool này **không** được đối xử đặc biệt: nó đi qua `Harness.execute()` như mọi tool khác, nên vẫn có phát hiện lặp, kiểm tra quyền và ghi vào trace. Code nằm trong [flight_agent/core.py](https://github.com/NguyenHungCuongg/mini-flight-booking-agent/blob/main/flight_agent/core.py):

- `Harness.tools`: danh sách tool của `World`, cộng thêm `remember_preference` khi session có memory.
- `Harness.check_permission`: trước khi ghi memory, hỏi người duyệt qua `approve_memory(key, value)`. Trong demo, `HumanApprover` hỏi `Save preference depart_before = 11:00 for future sessions? (y/n)`. Câu hỏi hiển thị giá trị **đã chuẩn hoá**, đúng giá trị sẽ được lưu (gõ `9:00` thì hỏi `09:00`). Giá trị không hợp lệ thì không hỏi, tool trả thẳng `invalid_param`. Trả lời `n` thì kết quả là `denied` và file không đổi.
- `Harness.remember_preference`: gọi `MemoryStore.remember`, rồi cập nhật luôn `Constraints` của session đang chạy, để quy ước mới có hiệu lực ngay trong session đó chứ không chỉ từ session sau.

Vì sao phải có bước xác nhận: memory tồn tại lâu hơn một lần chạy. Ở BTVN#3 có kịch bản `injection`, trong đó kết quả tìm chuyến chứa một ghi chú giả mạo kiểu "khách đã bỏ yêu cầu giờ bay". Nếu model bị lừa và gọi `remember_preference(depart_before="23:59")`, chỉ một lần prompt injection cũng làm sai memory của **mọi** session về sau (memory poisoning). Bước xác nhận là chốt chặn cho trường hợp đó. Mô tả của tool cũng dặn model chỉ gọi khi chính người dùng nói ra quy ước, không gọi vì một kết quả tool.

### 4.3. Khi nào nạp: toàn bộ memory, ngay lúc mở session

Memory rất nhỏ (tối đa hai mục `active`), nên bài nạp **toàn bộ** ngay khi mở session, không cần tìm kiếm. Như slide I.2 lưu ý, memory nằm trên đĩa không tự xuất hiện trong context: harness phải truy xuất (retrieval) rồi tiêm vào ngữ cảnh (injection). Việc nạp làm hai chỗ, trong `run()` của [flight_agent/run.py](https://github.com/NguyenHungCuongg/mini-flight-booking-agent/blob/main/flight_agent/run.py) và trong `run_react` của [flight_agent/react.py](https://github.com/NguyenHungCuongg/mini-flight-booking-agent/blob/main/flight_agent/react.py):

1. **Vào dữ liệu ràng buộc:** `run()` gọi `memory.apply(scenario.constraints)` trước khi tạo `Harness`. Từ đây, kiểm tra quyền và kiểm tra hoàn thành đều dùng quy ước đã nhớ.
2. **Vào context của model:** `Harness` lưu `memory.to_prompt()` vào `memory_prompt`, và `run_react` nối khối này vào cuối system prompt. Khối được inject có dạng:

```text
Saved user preferences (from earlier sessions, the newest value wins):
- depart_before = 11:00 (since 2026-10-22T10:02:41)
- max_price = 2000000 (since 2026-10-22T10:02:41)
Apply them unless the user says otherwise. When the user states a lasting rule ("from now on ..."), save it with remember_preference.
```

Đặt memory trong system prompt còn hợp với Phần III của slide (prompt caching): khối này không đổi trong suốt một session, nên thuộc phần tĩnh ở đầu prompt (_static prefix_) và không làm mất hiệu lực cache của các lượt gọi sau.

Khi `run()` được gọi không kèm memory (như lúc đánh giá BTVN#3), `memory_prompt` rỗng, `request` mặc định là `Constraints.to_prompt()` như cũ, và agent chạy y hệt trước.

### 4.4. Khi quy ước đổi: đánh dấu đã bị thay thế, không xoá

Theo slide I.6, cách sai là giữ cả hai giá trị như đang hiệu lực (LLM bị mâu thuẫn) hoặc xoá trắng giá trị cũ (mất lịch sử). `MemoryStore.remember` làm như sau:

- Nếu khoá đã có một mục `active` **cùng giá trị**: không ghi gì thêm, tránh sinh mục trùng.
- Nếu **khác giá trị**: mục cũ chuyển sang `status: "superseded"` và nhận `superseded_at`; mục mới được thêm với `status: "active"`, `since` và `source`.

Mỗi khoá vì vậy luôn có **đúng một** giá trị đang hiệu lực, và chỉ giá trị đó được đưa vào context. Mục cũ vẫn nằm trong file để truy vết: quy ước cũ là gì, người dùng đổi lúc nào, bằng câu nói nào.

Ví dụ file sau khi đổi giờ bay từ 11:00 sang 09:00 (rút gọn `source`):

```json
[
  {"key": "depart_before", "value": "11:00", "status": "superseded",
   "since": "2026-10-22T10:02:41", "source": "Từ giờ tôi chỉ bay trước 11:00 ...", "superseded_at": "2026-10-22T10:20:15"},
  {"key": "max_price", "value": 2000000, "status": "active",
   "since": "2026-10-22T10:02:41", "source": "Từ giờ tôi chỉ bay trước 11:00 ...", "superseded_at": null},
  {"key": "depart_before", "value": "09:00", "status": "active",
   "since": "2026-10-22T10:20:15", "source": "Từ giờ tôi chỉ bay trước 09:00 ...", "superseded_at": null}
]
```

### 4.5. Kiểm chứng: log phần memory đã inject

- `RunResult` có thêm trường `memory_injected`, chứa đúng khối văn bản đã đưa vào context. Test dùng trường này để khẳng định memory thực sự có trong context.
- Demo [flight_agent/memory_demo.py](https://github.com/NguyenHungCuongg/mini-flight-booking-agent/blob/main/flight_agent/memory_demo.py) in khối memory **trước khi** gọi model, rồi in trace, kết quả và nội dung file memory sau session.

## 5. Luồng một session

Vòng đời memory ở slide I.5 (Observe → Decide → Write → Store → Search → Retrieve → Use) gắn vào code như sau:

```
Session N (một lần run(), một tiến trình)
 ├─ Retrieve  run(): MemoryStore.active()  ← đọc memory/preferences.json
 ├─ Use       memory.apply(constraints)            → harness kiểm tra quy ước bằng code
 │            Harness.memory_prompt = to_prompt()  → nối vào system prompt (run_react)
 ├─ Observe   model đọc yêu cầu: "Từ giờ tôi chỉ bay trước 09:00. Đặt vé ..."
 ├─ Decide    model gọi remember_preference(depart_before, "09:00")
 │            Harness.execute → check_permission → approve_memory? (người dùng y/n)
 ├─ Write     MemoryStore.remember: chuẩn hoá, bản cũ → superseded, bản mới → active
 ├─ Store     ghi lại toàn bộ file JSON
 │            Harness.scenario.constraints cập nhật luôn cho phần còn lại của session
 └─ đặt vé như BTVN#3: search → book (bị chặn nếu sai quy ước) → pay → get_booking

Session N+1: bắt đầu lại từ Retrieve, chỉ thấy giá trị active mới nhất
```

Bước "Search" không cần thiết ở quy mô hiện tại vì toàn bộ memory được nạp. Mục 8 bàn về lúc nào cần đến nó.

Phạm vi: chỉ mẫu ReAct có tool ghi memory. Hai mẫu Plan-then-Execute và Lai vẫn **đọc** được memory qua `Constraints` mà `run()` đã cập nhật từ memory, nhưng không ghi được, vì kế hoạch của chúng chỉ gồm các bước đặt vé.

## 6. Demo với LLM thật

<!-- HƯỚNG DẪN CHUNG (xoá comment này trước khi nộp):
     1. Demo dùng Gemini (gemini-3.5-flash) qua langchain-google-genai; .env đã cấu hình sẵn. Free tier: 20 request/ngày cho mỗi model.
     2. Mỗi session tốn khoảng 5 request. S0..S4 tốn khoảng 25 request.
     3. Bắt đầu sạch:  uv run python -m flight_agent.memory_demo --reset
     4. Mỗi session là MỘT lệnh riêng:  uv run python -m flight_agent.memory_demo
        Tới dòng "Your request:" thì dán đúng câu ở bảng dưới. Khi được hỏi "Save preference ...? (y/n)" thì gõ y.
     5. S0 phải chạy TRƯỚC S1 (lúc memory còn trống).
     6. Tạo thư mục docs-img/ ở gốc repo để lưu ảnh (KHÔNG lưu trong docs/, thư mục đó bị gitignore).
     7. Nếu model thật làm khác kịch bản (ví dụ không gọi remember_preference, hoặc đặt theo thứ tự khác),
        cứ chụp đúng những gì xảy ra và sửa phần "Nhận xét" cho khớp. Điều cần thấy trong ảnh là:
        khối memory đã inject + trace có denied/ok + booking cuối cùng.
        Phần "Nhận xét" dưới mỗi ảnh hiện mô tả kết quả MONG ĐỢI; đối chiếu với ảnh thật và sửa lại trước khi nộp.
     8. Chụp xong ảnh nào thì thay dòng "> 📸 ..." bằng ![Ảnh N](docs-img/....png) -->

Lệnh: `uv run python -m flight_agent.memory_demo`. Mỗi lần chạy là một session mới. Các chuyến bay cố định (`MEMORY_SCENARIO` trong `flight_agent/memory.py`):

| Chuyến | Giờ đi | Giá (VND) | Ghi chú                       |
| ------ | ------ | --------- | ----------------------------- |
| VJ610  | 14:20  | 1.150.000 | Rẻ nhất, nhưng bay buổi chiều |
| QH112  | 10:30  | 1.600.000 | Trước 11:00                   |
| VN122  | 08:10  | 1.950.000 | Trước 09:00                   |

Ràng buộc gốc của kịch bản này được để rất lỏng (không giới hạn giờ và giá), nên **mọi** giới hạn giờ hay giá đều phải đến từ memory. Không có memory thì đặt VJ610 là hợp lệ.

| Session                 | Yêu cầu gõ vào                                                                                                | Mong đợi                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| S0 (đối chứng)          | `Đặt cho tôi một vé SGN -> DAD ngày 2026-10-07, chọn chuyến rẻ nhất.`                                                              | Memory trống, đặt VJ610 (rẻ nhất) là hợp lệ                          |
| S1 (thống nhất quy ước) | `Từ giờ tôi chỉ bay trước 11:00 và giá vé tối đa 2.000.000 VND. Đặt cho tôi một vé SGN -> DAD ngày 2026-10-07, chọn chuyến rẻ nhất.` | Gọi `remember_preference` 2 lần (gõ y), đặt QH112                    |
| S2 (Chứng minh 1)       | `Đặt cho tôi một vé SGN -> DAD ngày 2026-10-07, chọn chuyến rẻ nhất.`                                                              | Memory inject có 11:00 và 2.000.000, đặt QH112, không đặt VJ610      |
| S3 (đổi quy ước)        | `Từ giờ tôi chỉ bay trước 09:00. Đặt cho tôi một vé SGN -> DAD ngày 2026-10-07, chọn chuyến rẻ nhất.`                               | Lưu 09:00, mục 11:00 thành `superseded`, đặt VN122                   |
| S4 (Chứng minh 2)       | `Đặt cho tôi một vé SGN -> DAD ngày 2026-10-07, chọn chuyến rẻ nhất.`                                                              | Memory inject chỉ có 09:00, đặt VN122                                |

Câu yêu cầu ở S2 và S4 giống hệt S0 và **không nhắc gì** đến giờ hay giá. Cụm "chọn chuyến rẻ nhất" có mặt trong mọi câu để model tự quyết định thay vì hỏi lại người dùng; nhờ vậy cũng thấy rõ memory làm thay đổi chuyến "rẻ nhất hợp lệ": VJ610 (không có quy ước), QH112 (trước 11:00), VN122 (trước 09:00). Nếu kết quả khác S0 thì sự khác biệt chỉ có thể đến từ memory.

### 6.1. Session đối chứng (S0)

> 📸 **Ảnh 0: S0, memory trống, agent được phép đặt VJ610**
<!-- CHỤP: sau --reset, chạy demo, dán câu S0. Chụp từ "=== New session" (thấy "- none yet")
     đến hết "--- result ---". Lưu: docs-img/a-00-control.png -->

Nhận xét: khối memory là `- none yet`, và vì ràng buộc gốc không giới hạn giờ nên agent đặt chuyến rẻ nhất VJ610 lúc 14:20 với kết quả `done`.

### 6.2. Thống nhất quy ước (S1)

> 📸 **Ảnh 1: S1, agent gọi remember_preference, người dùng xác nhận, file memory được tạo**
<!-- CHỤP: chạy demo, dán câu S1, gõ y cho mỗi câu "Save preference ...". Có thể cần 2 ảnh:
     (a) đoạn có 2 câu hỏi y/n + trace (thấy [1] remember_preference ... -> ok);
     (b) đoạn "--- Memory file now" với 2 mục active.
     Lưu: docs-img/a-01-session1.png (và a-01b-memory-file.png nếu tách) -->

Nhận xét: model nhận ra câu "Từ giờ ..." là quy ước lâu dài và gọi `remember_preference` cho cả hai khoá. Harness hỏi xác nhận trước khi ghi. Sau session, file có hai mục `active` với `source` là câu yêu cầu S1.

### 6.3. Chứng minh 1: session mới vẫn áp dụng quy ước (S2)

> 📸 **Ảnh 2: S2, memory được inject, agent đặt QH112 chứ không đặt VJ610**
<!-- CHỤP: chạy demo (tiến trình mới), dán câu S2. Chụp từ "--- Memory injected into context ---"
     (thấy depart_before = 11:00 và max_price = 2000000) đến hết "--- result ---".
     Lưu: docs-img/a-02-proof1.png -->

Nhận xét:

- Đây là tiến trình mới, lịch sử hội thoại rỗng, câu yêu cầu giống hệt S0. Thứ duy nhất khác S0 là khối memory được in ở đầu.
- Agent đặt QH112 (10:30, 1.600.000) thay vì VJ610. Nếu model vẫn thử đặt VJ610, trace sẽ có dòng `book_seat({'flight': 'VJ610'}) -> denied ... breaks the constraints`: quy ước được harness kiểm tra bằng code chứ không chỉ nhờ model nhớ.

### 6.4. Đổi quy ước (S3)

> 📸 **Ảnh 3: S3, quy ước 11:00 bị thay bằng 09:00, mục cũ được đánh dấu superseded**
<!-- CHỤP: chạy demo, dán câu S3, gõ y. Chụp trace (thấy remember_preference(... '09:00') -> ok,
     và booking VN122) + toàn bộ "--- Memory file now" (thấy mục 11:00 có "status": "superseded"
     và "superseded_at"). Lưu: docs-img/a-03-change.png -->

Nhận xét: mục `depart_before = 11:00` không bị xoá mà chuyển sang `superseded`, có `superseded_at`. Mục mới `09:00` là `active`. Ngay trong session này, `Harness.remember_preference` đã cập nhật ràng buộc, nên QH112 (10:30) không còn hợp lệ và agent đặt VN122 (08:10).

### 6.5. Chứng minh 2: session sau dùng quy ước mới (S4)

> 📸 **Ảnh 4: S4, memory inject chỉ còn 09:00, agent đặt VN122**
<!-- CHỤP: chạy demo, dán câu S4. Chụp từ "--- Memory injected into context ---"
     (thấy depart_before = 09:00, KHÔNG có 11:00) đến hết "--- result ---".
     Lưu: docs-img/a-04-proof2.png -->

Nhận xét: khối inject chỉ có `depart_before = 09:00`. Mục 11:00 vẫn nằm trong file (xem Ảnh 3) nhưng không được đưa vào context vì đã `superseded`. Với cùng câu yêu cầu như S0 và S2, agent giờ đặt VN122.

## 7. Kiểm thử tự động

Toàn bộ test chạy với model giả có kịch bản sẵn (`ScriptedModel` trong `tests/test_run.py`), không gọi mạng và không tốn quota. Test của bài nằm trong [tests/test_memory.py](https://github.com/NguyenHungCuongg/mini-flight-booking-agent/blob/main/tests/test_memory.py):

| Test                                                                 | Chứng minh điều gì                                                                    |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `test_empty_memory_has_no_preferences`                               | Lần đầu chạy, chưa có file: không lỗi, prompt ghi `none yet`                          |
| `test_remember_persists_on_disk_for_a_new_store`                     | Ghi xong, một `MemoryStore` mới (như session mới) đọc lại được                        |
| `test_changing_a_preference_supersedes_the_old_one`                  | Đổi quy ước: mục cũ `superseded` có `superseded_at`, mục mới `active`                 |
| `test_same_preference_twice_adds_no_entry`                           | Nói lại quy ước cũ không sinh mục trùng                                               |
| `test_values_are_normalised` (6 trường hợp)                          | `"9:00"` → `"09:00"`, `"2,000,000"` / `"2.000.000"` / `2000000.0` → `2000000`         |
| `test_bad_preferences_are_rejected_and_not_written` (8 trường hợp)   | Khoá lạ, giờ sai, giá sai hoặc có phần thập phân bị trả `invalid_param`, không ghi gì |
| `test_prompt_lists_active_values_only_and_apply_overlays_constraints`| Chỉ giá trị `active` vào context; `apply` thay đúng trường của `Constraints`            |
| `test_without_memory_the_cheap_afternoon_flight_is_allowed`          | Đối chứng: không có memory thì đặt VJ610 hợp lệ                                       |
| `test_a_session_saves_a_preference_the_user_states`                  | Agent gọi tool trong một session, quy ước nằm lại trên đĩa                            |
| `test_proof_1_new_session_applies_the_saved_preference`              | **Chứng minh 1:** session mới có memory trong context, VJ610 bị chặn, đặt QH112       |
| `test_proof_2_changed_preference_applies_now_and_in_later_sessions`  | **Chứng minh 2:** đổi sang 09:00 có hiệu lực ngay và ở session sau; 11:00 không còn trong context |
| `test_saving_a_preference_needs_the_users_yes`                       | Người dùng từ chối thì tool trả `denied`, file không đổi (chống memory poisoning)       |
| `test_the_user_confirms_the_value_that_will_be_stored_and_is_not_asked_about_invalid_ones` | Người dùng được hỏi đúng giá trị sẽ lưu (`09:00`), không bị hỏi về giá trị sai (`9h`) |

Kết quả: 52 test pass, gồm 26 test cũ của BTVN#3 (không phải sửa test nào), 25 test memory mới ở trên, và 1 test mới trong `tests/test_run.py` kiểm tra lỗi hết quota của Gemini được tính là `infra_error`.

> 📸 **Ảnh 5: kết quả `uv run pytest -q`**
<!-- CHỤP: chạy  uv run pytest -q -v tests/test_memory.py  (thấy tên từng test PASSED),
     rồi  uv run pytest -q  (thấy "52 passed"). Lưu: docs-img/a-05-tests.png -->

## 8. Hạn chế và hướng phát triển

- **Chỉ hai loại quy ước.** Thêm quy ước mới (hãng bay ưa thích, sân bay nhà, hạng ghế) cần thêm trường vào `Constraints` và thêm luật kiểm tra trong `_normalise`, để harness vẫn kiểm tra được bằng code.
- **Chỉ ReAct ghi được memory.** Plan-then-Execute và Lai chỉ đọc. Muốn hai mẫu này ghi, cần một bước riêng trước khi lập kế hoạch để trích quy ước từ yêu cầu của người dùng (mẫu "harness tự trích xuất", tốn thêm một lần gọi model).
- **Chưa có quên và hết hạn.** Slide I.5 nêu ba chính sách Update, Forget, Expire; bài mới làm Update (qua `superseded`). Chưa có lệnh "quên quy ước" hay quy ước tạm thời tự hết hạn. `--reset` của demo chỉ xoá toàn bộ.
- **JSON hợp với vài chục mục.** Khi memory lớn lên (nhiều người dùng, nhiều loại quy ước), nên chuyển sang SQLite có metadata, hoặc mô hình index + topic files như `MEMORY.md` ở slide I.4: index nhỏ luôn được nạp, chi tiết chỉ nạp khi cần. Lúc đó mới cần bước "Search" trong vòng đời.
- **Người dùng xác nhận là chốt chặn duy nhất chống memory poisoning.** Nếu người dùng bấm y theo thói quen, một injection vẫn có thể lọt vào. Cách chặt hơn: chỉ cho ghi khi giá trị xuất hiện trong chính câu người dùng nói, hoặc không cho ghi memory sau khi agent đã đọc kết quả từ nguồn ngoài.
- **Chưa an toàn khi nhiều session ghi cùng lúc.** Hai session chạy song song có thể ghi đè nhau. Demo chạy tuần tự nên chưa gặp; dùng SQLite sẽ giải quyết luôn.

## 9. Kết luận

Agent đặt vé đã nhớ được quy ước của người dùng qua nhiều session nhờ một file JSON do `MemoryStore` quản lý, được nạp vào cả context của model lẫn dữ liệu ràng buộc của harness mỗi khi mở session. Khi quy ước đổi, bản cũ được đánh dấu `superseded` kèm thời điểm và nguồn, và chỉ bản mới vào context. Cả hai yêu cầu chứng minh đều được kiểm bằng test tự động với model giả, và được minh hoạ bằng demo với LLM thật ở mục 6. Điểm khác so với chỉ "nhắc trong prompt" là quy ước được harness kiểm tra bằng code, nên vẫn đúng ngay cả khi model quên.

## Phụ lục: chạy lại

```bash
git clone https://github.com/NguyenHungCuongg/mini-flight-booking-agent && cd mini-flight-booking-agent
uv sync
uv run pytest -q                                      # 52 test, model giả, không cần key
cp .env.example .env                                  # rồi điền OPENAI_API_KEY
uv run python -m flight_agent.memory_demo --reset     # xoá memory, bắt đầu lại
uv run python -m flight_agent.memory_demo             # một session; chạy lại lệnh này cho mỗi session
```
