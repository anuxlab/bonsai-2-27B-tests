#!/usr/bin/env bash
# Download model (+ projector for vision shards) and runtime with resume, verify SHA-256, record everything.
# usage: prepare_model.sh <out_metadata_dir> <need_mmproj: true|false>
# env overrides (optional): MODEL_URL MODEL_SHA256 MMPROJ_URL MMPROJ_SHA256 RUNTIME_URL
set -uo pipefail
META="${1:?metadata dir}"; NEED_MMPROJ="${2:-false}"
CFG=config/model.json
mkdir -p "$META" models bin
cfg() { python3 -c "import json,sys; d=json.load(open('$CFG')); print(eval('d'+sys.argv[1]))" "$1"; }

MODEL_FILE="$(cfg "['model']['filename']")";   MODEL_URL="${MODEL_URL:-$(cfg "['model']['url']")}";   MODEL_SHA="${MODEL_SHA256:-$(cfg "['model']['sha256']")}"
MM_FILE="$(cfg "['mmproj']['filename']")";     MM_URL="${MMPROJ_URL:-$(cfg "['mmproj']['url']")}";   MM_SHA="${MMPROJ_SHA256:-$(cfg "['mmproj']['sha256']")}"
RT_URL="${RUNTIME_URL:-$(cfg "['runtime']['url']")}";   RT_SHA="$(cfg "['runtime']['sha256']")"
PINNED=true; [ -n "${MODEL_URL_OVERRIDE:-}" ] && PINNED=false

download() { # url dest ; echo seconds
  local url="$1" dest="$2" t0 t1
  t0=$(date +%s.%N)
  curl -fL --retry 6 --retry-all-errors --retry-delay 5 --connect-timeout 30 -C - -o "$dest" "$url" || return 1
  t1=$(date +%s.%N); echo "$t1 - $t0" | bc -l
}
fetch_verified() { # label file url expected_sha -> writes $META/dl_<label>.json
  local label="$1" file="$2" url="$3" want="$4" secs size got
  echo "::group::download $label"
  secs=$(download "$url" "$file") || { echo "::error::download failed for $label"; echo "download_failure" > "$META/failure_class.txt"; return 10; }
  size=$(stat -c %s "$file"); got=$(sha256sum "$file" | cut -d' ' -f1)
  echo "$label size=$size sha256=$got seconds=$secs"
  python3 - "$label" "$file" "$size" "$got" "$want" "$secs" "$url" "$META" <<'PY'
import json, sys
label, file, size, got, want, secs, url, meta = sys.argv[1:]
secs, size = float(secs), int(size)
json.dump({"label": label, "file": file, "bytes": size, "sha256": got, "expected_sha256": want or None,
           "verified": (got == want) if want else None, "download_seconds": round(secs, 1),
           "throughput_MBps": round(size / 1e6 / secs, 1) if secs else None, "url": url.split("?")[0]},
          open(f"{meta}/dl_{label}.json", "w"), indent=2)
PY
  echo "::endgroup::"
  if [ -n "$want" ] && [ "$got" != "$want" ]; then
    echo "::error::INTEGRITY FAILURE for $label: expected $want got $got"; echo "integrity_failure" > "$META/failure_class.txt"; rm -f "$file"; return 11
  fi
}

fetch_verified model "models/$MODEL_FILE" "$MODEL_URL" "$MODEL_SHA" || exit $?
if [ "$NEED_MMPROJ" = "true" ]; then fetch_verified mmproj "models/$MM_FILE" "$MM_URL" "$MM_SHA" || exit $?; fi

echo "::group::download runtime"
t0=$(date +%s.%N)
curl -fL --retry 6 --retry-all-errors --connect-timeout 30 -o runtime.tar.gz "$RT_URL" || { echo "::error::runtime download failed"; echo download_failure > "$META/failure_class.txt"; exit 10; }
RT_GOT=$(sha256sum runtime.tar.gz | cut -d' ' -f1)
if [ -n "$RT_SHA" ] && [ "$RT_GOT" != "$RT_SHA" ]; then echo "::error::runtime hash mismatch"; echo integrity_failure > "$META/failure_class.txt"; exit 11; fi
tar -xzf runtime.tar.gz -C bin --strip-components=1 && rm -f runtime.tar.gz   # never uploaded, never committed
chmod +x bin/llama-server
echo "::endgroup::"

python3 - "$META" "$MODEL_FILE" "$MM_FILE" "$NEED_MMPROJ" "$RT_URL" "$RT_GOT" "$PINNED" <<'PY'
import json, sys, os
meta, mf, mm, need, rt_url, rt_sha, pinned = sys.argv[1:]
d = json.load(open(f"{meta}/dl_model.json"))
info = {"model_file": mf, "model": d, "mmproj": json.load(open(f"{meta}/dl_mmproj.json")) if need == "true" else None,
        "runtime": {"url": rt_url, "tarball_sha256": rt_sha}, "pinned_hashes": pinned == "true",
        "config": json.load(open("config/model.json"))}
json.dump(info, open(f"{meta}/model_info.json", "w"), indent=2)
PY
echo "model + runtime ready"
