from pathlib import Path
import requests

FILELIST = "subset_GPM_3IMERGDF_07_20251219_161207_.txt"       # file .txt chứa danh sách link .nc4
OUTDIR = Path(r"C:\BKHN\Data Science\subset_GPM_3IMERGDE_2019_2024") # thư mục lưu file tải về
OUTDIR.mkdir(exist_ok=True)
# Session dùng chung, requests sẽ tự dùng .netrc để login Earthdata
session = requests.Session()

def download_one(url: str):
    url = url.strip()
    if not url:
        return

    filename = url.split("/")[-1]
    out_path = OUTDIR / filename

    if out_path.exists():
        print(f"Skip (đã có): {out_path}")
        return

    print(f"Đang tải: {url}")
    resp = session.get(url, stream=True)
    resp.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"  -> Xong: {out_path}")

def main():
    with open(FILELIST, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                download_one(line)
            except Exception as e:
                print(f"Lỗi tải {line}: {e}")

if __name__ == "__main__":
    main()
