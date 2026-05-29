# image-analysis

OpenAI Batch API를 활용한 얼굴 마스크 세그멘테이션 메타데이터 자동 어노테이션 도구.

이미지를 개별 API 호출 없이 일괄 처리하여 **비용 약 50% 절감** 및 대용량 데이터셋 처리를 지원합니다.

---

## Features

- **Batch API 기반** — 실시간 호출 대신 비동기 일괄 처리로 비용 절감
- **Responses API 사용** — `input_image` 타입으로 멀티모달 이미지 전송
- **Structured Output** — JSON Schema 강제 적용으로 일관된 어노테이션 보장
- **재개(Resume)** — 중단 시 `batch_state.json`을 통해 기존 배치 이어서 실행
- **중복 스킵** — 이미 완료된 이미지는 자동으로 제외
- **MIME 타입 자동 감지** — JPG / PNG / WebP 확장자별 올바른 타입 전송
- **용량 경고** — OpenAI 200MB 제한 초과 시 사전 오류 발생
- **에러 로그** — 실패 건을 `batch/errors.jsonl`에 별도 저장

---

## Requirements

- Python 3.9+
- OpenAI API Key (`OPENAI_API_KEY` 환경변수)

```bash
uv add openai python-dotenv tqdm
```

---

## Project Structure

```
image-analysis/
├── annotate_batch.py       # 메인 실행 스크립트
│
├── images/                 # 입력 이미지 디렉토리
│   ├── image_001.jpg
│   ├── image_002.png
│   └── ...
│
├── annotations/            # 출력 어노테이션 디렉토리 (자동 생성)
│   ├── image_001.json
│   ├── image_002.json
│   └── ...
│
└── batch/                  # 배치 작업 임시 파일 (자동 생성)
    ├── requests.jsonl      # 배치 요청 파일
    ├── batch_state.json    # 재개용 상태 파일 (완료 시 자동 삭제)
    └── errors.jsonl        # 실패 건 에러 로그 (실패 시 생성)
```

---

## Configuration

`annotate_batch.py` 상단의 CONFIG 섹션에서 설정합니다.

```python
IMAGE_DIR    = "./images"       # 입력 이미지 경로
OUTPUT_DIR   = "./annotations"  # 출력 JSON 경로
BATCH_DIR    = "./batch"        # 배치 임시 파일 경로

MODEL_NAME   = "gpt-5.4-mini"  # 사용 모델 ("gpt-5.4"로 교체 가능)
TEMPERATURE  = 0               # 재현성을 위해 0 권장

POLL_INTERVAL = 60             # 배치 상태 확인 주기 (초)
RESET_BATCH   = False          # True: 기존 배치 무시하고 새로 실행
```

---

## Usage

**1. API Key 환경변수 설정**

```bash
export OPENAI_API_KEY="sk-..."
```

**2. 이미지 준비**

```
images/
├── face_001.jpg
├── face_002.png
└── ...
```

지원 포맷: `.jpg` `.jpeg` `.png` `.webp`

**3. 실행**

```bash
python annotate_batch.py
```

**4. 출력 확인**

```
annotations/
├── face_001.json
├── face_002.json
└── ...
```

---

## How It Works

```
[Step 1] JSONL 생성
  이미지 → base64 인코딩 → batch/requests.jsonl

[Step 2] 배치 제출
  requests.jsonl → OpenAI Files API 업로드
  → Batch Job 생성 → batch/batch_state.json 저장

[Step 3] 완료 대기
  60초 간격 폴링
  [validating] → [in_progress] → [completed]

[Step 4] 결과 저장
  결과 파일 다운로드 → 이미지별 .json 파싱 및 저장
```

중단 후 재실행 시 `RESET_BATCH = False`(기본값)이면 Step 3부터 자동으로 재개됩니다.

---

## Output Schema

이미지 1장당 하나의 JSON 파일이 생성됩니다.

```json
{
  "num_people": 1,

  "scene": {
    "background": "indoor",
    "lighting": ["warm", "dim"],
    "image_quality": ["medium"]
  },

  "person": {
    "pose": "front",
    "yaw": 6.0,
    "pitch": -2.0
  },

  "mask": {
    "state": "transition_taking_off",
    "type": "cloth",
    "boundary_quality": "stretched",
    "deformation": true
  },

  "occlusion": {
    "types": ["hair", "hand"],
    "hand_overlap_ratio": 0.43,
    "hair_overlap_ratio": 0.17
  },

  "visibility": {
    "mask_visible_ratio": 0.68,
    "nose_visible_ratio": 0.84,
    "mouth_visible_ratio": 0.12,
    "chin_visible_ratio": 0.18
  },

  "difficulty": {
    "segmentation_difficulty": 0.89,
    "difficulty_reasons": [
      "hand_overlap",
      "deformation",
      "ambiguous_boundary"
    ]
  }
}
```

### 필드 상세

**`scene`**

| 필드 | 타입 | 값 |
|---|---|---|
| `background` | string | `indoor` `outdoor` `studio` `vehicle` `crowd` `plain` `complex` `unknown` |
| `lighting` | array | `bright` `dim` `warm` `cool` `uneven` `backlit` |
| `image_quality` | array | `high` `medium` `low` `blurry` `compressed` `noisy` |

**`person`**

| 필드 | 타입 | 값 |
|---|---|---|
| `pose` | string | `front` `left` `right` `up` `down` `tilted` `profile_left` `profile_right` |
| `yaw` | float | -90 ~ 90 (좌우 회전각) |
| `pitch` | float | -90 ~ 90 (상하 회전각) |

**`mask`**

| 필드 | 타입 | 값 |
|---|---|---|
| `state` | string | `normal` `partial_removed` `under_chin` `nose_exposed` `mouth_exposed` `fully_removed_holding` `hanging_one_side` `covering_one_side_only` `transition_putting_on` `transition_taking_off` |
| `type` | string | `surgical` `kf94` `cloth` `transparent` `respirator` `unknown` |
| `boundary_quality` | string | `sharp` `soft` `blurry` `folded` `stretched` `wrinkled` `ambiguous` |
| `deformation` | bool | 마스크 변형 여부 |

**`occlusion`**

| 필드 | 타입 | 값 |
|---|---|---|
| `types` | array | `none` `hair` `hand` `shadow` `glasses` `phone` `object` `clothing` |
| `hand_overlap_ratio` | float | 0.0 ~ 1.0 (손에 의한 마스크 가림 비율) |
| `hair_overlap_ratio` | float | 0.0 ~ 1.0 (머리카락에 의한 마스크 가림 비율) |

**`visibility`**

| 필드 | 타입 | 설명 |
|---|---|---|
| `mask_visible_ratio` | float | 마스크 가시 비율 (0.0 ~ 1.0) |
| `nose_visible_ratio` | float | 코 가시 비율 (0.0 ~ 1.0) |
| `mouth_visible_ratio` | float | 입 가시 비율 (0.0 ~ 1.0) |
| `chin_visible_ratio` | float | 턱 가시 비율 (0.0 ~ 1.0) |

**`difficulty`**

| 필드 | 타입 | 값 |
|---|---|---|
| `segmentation_difficulty` | float | 세그멘테이션 난이도 (0.0 ~ 1.0) |
| `difficulty_reasons` | array | `hand_overlap` `hair_overlap` `low_resolution` `motion_blur` `extreme_pose` `ambiguous_boundary` `deformation` `shadow` `low_contrast` `partial_visibility` `multiple_occlusion` `tiny_mask_region` |

---

## Notes

**OpenAI Batch API 제한 사항**

| 항목 | 제한 |
|---|---|
| 최대 파일 크기 | 200 MB |
| 최대 요청 수 | 50,000건 / 배치 |
| 완료 보장 시간 | 24시간 이내 |
| 비용 | 표준 가격 대비 약 50% |

이미지 수가 많거나 해상도가 높을 경우 JSONL 파일이 200MB를 초과할 수 있습니다.
이 경우 `IMAGE_DIR` 내 이미지를 분할하여 여러 번 실행하세요.
이미 완료된 이미지는 자동으로 스킵되므로 순서대로 실행하면 됩니다.

**배치 재개**

실행 중 중단되더라도 재실행 시 `batch/batch_state.json`이 존재하면
자동으로 기존 배치를 이어서 모니터링합니다.

처음부터 다시 실행하려면 `RESET_BATCH = True`로 설정하거나
`batch/batch_state.json` 파일을 삭제하세요.