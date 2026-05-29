from pathlib import Path
from collections import defaultdict

IMAGE_DIR = Path("./images")

groups = defaultdict(list)

# 같은 prefix끼리 그룹화
for path in IMAGE_DIR.glob("*.jpg"):

    # 예:
    # image_0_jpg.rf.74b78567.jpg
    # -> image_0_jpg
    prefix = path.stem.split(".rf.")[0]

    groups[prefix].append(path)

deleted_count = 0

# 각 그룹에서 파일명 기준 첫 번째만 남김
for prefix, files in groups.items():

    # 파일명 오름차순 정렬
    files.sort(key=lambda p: p.name)

    # 첫 번째 유지
    keep = files[0]

    print(f"\nKEEP: {keep.name}")

    # 나머지 삭제
    for f in files[1:]:
        print(f"DELETE: {f.name}")
        f.unlink()
        deleted_count += 1

print(f"\n완료: {deleted_count}개 삭제")