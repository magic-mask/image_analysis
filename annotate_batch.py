import os
import json
import base64
import time

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ============================================
# LOAD ENV
# ============================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")


# ============================================
# CONFIG
# ============================================

IMAGE_DIR    = "./images"
OUTPUT_DIR   = "./annotations"
BATCH_DIR    = "./batch"

MODEL_NAME   = "gpt-5.4"
# MODEL_NAME = "gpt-5.4-mini"

TEMPERATURE  = 0

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

# [내 코드에서 가져옴] 확장자별 MIME 타입 — 새 코드는 항상 image/jpeg로 잘못 전송했음
MIME_TYPES = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
}

# [내 코드에서 가져옴] 운영 설정
POLL_INTERVAL = 60     # 배치 상태 확인 주기 (초)
RESET_BATCH   = False  # True: 기존 배치 무시하고 새로 실행

STATE_FILE    = os.path.join(BATCH_DIR, "batch_state.json")
REQUESTS_FILE = os.path.join(BATCH_DIR, "requests.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BATCH_DIR,  exist_ok=True)

client = OpenAI(api_key=API_KEY)


# ============================================
# SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = r"""
You are an expert computer vision dataset annotator specializing in semantic segmentation and edge-case analysis for face mask segmentation datasets.

Your task is to analyze images for segmentation-relevant metadata.

You are NOT a general captioning model.

You MUST focus on:

- mask visibility
- mask deformation
- segmentation boundary ambiguity
- occlusion
- pose
- lighting
- image quality
- difficulty estimation

Return ONLY valid JSON.

Do NOT return markdown.
Do NOT explain.
Do NOT include commentary.
Do NOT include additional keys.
Do NOT output null unless explicitly required.
"""


# ============================================
# JSON SCHEMA
# ============================================

SCHEMA = {
    "name": "mask_segmentation_annotation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "num_people": {
                "type": "integer"
            },
            "scene": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "background": {
                        "type": "string",
                        "enum": [
                            "indoor", "outdoor", "studio", "vehicle",
                            "crowd", "plain", "complex", "unknown"
                        ]
                    },
                    "lighting": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["bright", "dim", "warm", "cool", "uneven", "backlit"]
                        }
                    },
                    "image_quality": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["high", "medium", "low", "blurry", "compressed", "noisy"]
                        }
                    }
                },
                "required": ["background", "lighting", "image_quality"]
            },
            "person": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pose": {
                        "type": "string",
                        "enum": [
                            "front", "left", "right", "up", "down",
                            "tilted", "profile_left", "profile_right"
                        ]
                    },
                    "yaw":   {"type": "number", "minimum": -90, "maximum": 90},
                    "pitch": {"type": "number", "minimum": -90, "maximum": 90}
                },
                "required": ["pose", "yaw", "pitch"]
            },
            "mask": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": [
                            "normal", "partial_removed", "under_chin", "nose_exposed",
                            "mouth_exposed", "fully_removed_holding", "hanging_one_side",
                            "covering_one_side_only", "transition_putting_on",
                            "transition_taking_off"
                        ]
                    },
                    "type": {
                        "type": "string",
                        "enum": ["surgical", "kf94", "cloth", "transparent", "respirator", "unknown"]
                    },
                    "boundary_quality": {
                        "type": "string",
                        "enum": ["sharp", "soft", "blurry", "folded", "stretched", "wrinkled", "ambiguous"]
                    },
                    "deformation": {"type": "boolean"}
                },
                "required": ["state", "type", "boundary_quality", "deformation"]
            },
            "occlusion": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "none", "hair", "hand", "shadow",
                                "glasses", "phone", "object", "clothing"
                            ]
                        }
                    },
                    "hand_overlap_ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "hair_overlap_ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["types", "hand_overlap_ratio", "hair_overlap_ratio"]
            },
            "visibility": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "mask_visible_ratio":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "nose_visible_ratio":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "mouth_visible_ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "chin_visible_ratio":  {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": [
                    "mask_visible_ratio", "nose_visible_ratio",
                    "mouth_visible_ratio", "chin_visible_ratio"
                ]
            },
            "difficulty": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "segmentation_difficulty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "difficulty_reasons": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "hand_overlap", "hair_overlap", "low_resolution",
                                "motion_blur", "extreme_pose", "ambiguous_boundary",
                                "deformation", "shadow", "low_contrast",
                                "partial_visibility", "multiple_occlusion", "tiny_mask_region"
                            ]
                        }
                    }
                },
                "required": ["segmentation_difficulty", "difficulty_reasons"]
            }
        },
        "required": [
            "num_people", "scene", "person", "mask",
            "occlusion", "visibility", "difficulty"
        ]
    }
}


# ============================================
# FEW-SHOT EXAMPLES
# [새 코드에서 가져옴] Responses API 네이티브 포맷 (input_text / output_text) 유지
# ============================================

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "Edge-case example:\n"
                    "Person is pulling mask downward with hand.\n"
                    "Nose exposed.\n"
                    "Boundary stretched.\n"
                    "Hair overlaps left mask edge."
                )
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": json.dumps({
                    "num_people": 1,
                    "scene": {
                        "background": "indoor",
                        "lighting": ["warm"],
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
                        "deformation": True
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
                })
            }
        ]
    }
]


# ============================================
# IMAGE ENCODER
# ============================================

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================
# STEP 1: PREPARE BATCH JSONL
# ============================================

def prepare_batch_file(image_files: list) -> None:
    """
    [새 코드] /v1/responses + input_image 포맷 사용
    [내 코드] MIME 타입 분기 / 200MB 초과 경고 추가
    """
    print(f"\n[Step 1] Preparing JSONL — {len(image_files)} images")

    with open(REQUESTS_FILE, "w", encoding="utf-8") as f:
        for image_path in tqdm(image_files, desc="  Encoding"):
            custom_id    = Path(image_path).stem
            base64_image = encode_image(str(image_path))

            # [내 코드에서 가져옴] 확장자에 맞는 MIME 타입 지정
            mime = MIME_TYPES.get(Path(image_path).suffix.lower(), "image/jpeg")

            line = {
                "custom_id": custom_id,
                "method": "POST",

                # [새 코드에서 가져옴] Responses API 엔드포인트 사용
                "url": "/v1/responses",

                "body": {
                    "model": MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "input": [
                        {
                            "role": "system",
                            "content": [
                                {"type": "input_text", "text": SYSTEM_PROMPT}
                            ]
                        },
                        *FEW_SHOT_EXAMPLES,
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        "Analyze this image for segmentation metadata.\n"
                                        "Return ONLY strict JSON."
                                    )
                                },
                                {
                                    # [새 코드에서 가져옴] 멀티모달 전용 input_image 타입
                                    # [내 코드에서 가져옴] 올바른 MIME 타입 적용
                                    "type": "input_image",
                                    "image_url": f"data:{mime};base64,{base64_image}"
                                }
                            ]
                        }
                    ],
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": SCHEMA["name"],
                            "strict": True,
                            "schema": SCHEMA["schema"]
                        }
                    }
                }
            }

            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(REQUESTS_FILE) / 1024 / 1024
    print(f"  → {REQUESTS_FILE} ({size_mb:.1f} MB)")

    # [내 코드에서 가져옴] OpenAI Batch API 파일 크기 제한 (200 MB)
    if size_mb > 200:
        raise ValueError(
            f"JSONL 파일이 200 MB 제한 초과 ({size_mb:.1f} MB). "
            "이미지를 더 작은 단위로 분할하세요."
        )


# ============================================
# STEP 2: UPLOAD AND SUBMIT BATCH
# ============================================

def submit_batch() -> str:
    """
    [내 코드에서 가져옴] 배치 제출 후 STATE_FILE에 상태 저장 (재개 지원)
    """
    print("\n[Step 2] Uploading to OpenAI Files API...")

    with open(REQUESTS_FILE, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    print(f"  → File ID : {uploaded.id}")
    print("  Submitting batch job...")

    batch = client.batches.create(
        input_file_id=uploaded.id,

        # [새 코드에서 가져옴] Responses API 엔드포인트
        endpoint="/v1/responses",

        completion_window="24h",
        metadata={"description": "mask_segmentation_annotation"}
    )

    print(f"  → Batch ID: {batch.id}")
    print(f"  → Status  : {batch.status}")

    # [내 코드에서 가져옴] 중단 후 재실행을 위한 상태 저장
    with open(STATE_FILE, "w") as f:
        json.dump(
            {
                "batch_id":     batch.id,
                "file_id":      uploaded.id,
                "submitted_at": time.time()
            },
            f, indent=2
        )

    return batch.id


# ============================================
# STEP 3: POLL UNTIL COMPLETE
# ============================================

def wait_for_batch(batch_id: str):
    """
    [내 코드에서 가져옴] 경과 시간 표시 / 60초 간격 폴링
    """
    print(f"\n[Step 3] Monitoring batch {batch_id}")
    print(f"  Polling every {POLL_INTERVAL}s...\n")

    while True:
        batch  = client.batches.retrieve(batch_id)
        counts = batch.request_counts

        total     = counts.total     if counts else "?"
        completed = counts.completed if counts else 0
        failed    = counts.failed    if counts else 0

        elapsed_str = ""
        if batch.created_at:
            elapsed_mins = int((time.time() - batch.created_at) / 60)
            elapsed_str  = f" | elapsed {elapsed_mins}m"

        print(
            f"  [{batch.status}] "
            f"{completed}/{total} completed, "
            f"{failed} failed"
            f"{elapsed_str}"
        )

        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch

        time.sleep(POLL_INTERVAL)


# ============================================
# STEP 4: DOWNLOAD AND SAVE RESULTS
# ============================================

def save_results(batch) -> None:
    """
    [새 코드에서 가져옴] output[0].content[0].text 파싱 방식 (Responses API 구조)
    [내 코드에서 가져옴] 에러 파일 다운로드 / 완료 후 STATE_FILE 삭제
    """
    print(f"\n[Step 4] Final status: {batch.status}")

    # [내 코드에서 가져옴] 실패 건 에러 로그 저장
    if batch.error_file_id:
        err_path = os.path.join(BATCH_DIR, "errors.jsonl")
        err_resp = client.files.content(batch.error_file_id)
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(err_resp.text)
        print(f"  Error log: {err_path}")

    if batch.status != "completed":
        print("  Batch가 정상 완료되지 않았습니다. 저장 중단.")
        return

    print(f"  Downloading: {batch.output_file_id}")

    file_resp = client.files.content(batch.output_file_id)
    lines = [l for l in file_resp.text.splitlines() if l.strip()]

    success, fail = 0, 0

    for line in tqdm(lines, desc="  Saving"):
        item      = json.loads(line)
        custom_id = item["custom_id"]
        out_path  = os.path.join(OUTPUT_DIR, f"{custom_id}.json")

        if item.get("error"):
            print(f"\n  [ERROR] {custom_id}: {item['error']['message']}")
            fail += 1
            continue

        try:
            # [새 코드에서 가져옴] Responses API 응답 구조로 파싱
            response_text = item["response"]["body"]["output"][0]["content"][0]["text"]
            parsed = json.loads(response_text)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)

            success += 1

        except Exception as e:
            print(f"\n  [PARSE ERROR] {custom_id}: {e}")
            fail += 1

    # [내 코드에서 가져옴] 완료 후 상태 파일 삭제
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    print(f"\n{'─' * 40}")
    print(f"  ✓ Success : {success}")
    print(f"  ✗ Failed  : {fail}")
    print(f"  Output    : {OUTPUT_DIR}/")
    print(f"{'─' * 40}")


# ============================================
# MAIN
# ============================================

def main():

    # [내 코드에서 가져옴] 기존 배치 재개
    if not RESET_BATCH and os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)

        batch_id = state["batch_id"]
        print(f"Resuming existing batch: {batch_id}")

        batch = wait_for_batch(batch_id)
        save_results(batch)
        return

    # 이미지 수집
    image_files = sorted(
        img
        for ext in SUPPORTED_EXTENSIONS
        for img in Path(IMAGE_DIR).glob(f"*{ext}")
    )

    # [내 코드에서 가져옴] 이미 완료된 이미지 스킵
    pending = [
        img for img in image_files
        if not os.path.exists(os.path.join(OUTPUT_DIR, img.stem + ".json"))
    ]

    print(f"Total   : {len(image_files)}")
    print(f"Done    : {len(image_files) - len(pending)}")
    print(f"Pending : {len(pending)}")

    if not pending:
        print("All images already processed.")
        return

    prepare_batch_file(pending)
    batch_id = submit_batch()
    batch    = wait_for_batch(batch_id)
    save_results(batch)


if __name__ == "__main__":
    main()