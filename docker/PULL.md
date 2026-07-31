# 사내 반입·실행 절차 — kernel-kit 도커 2티어 (PLAN §S2 전달 모델)

작성: 2026-07-29 · 대상: 사내 A100 노드 러너(Jimmy). 빌드는 전부 개발 호스트에서 끝났고,
사내에서는 **pull(또는 load) → run** 만 한다. 이미지에 repo 는 없다 — repo 는 볼륨 마운트.

| 이미지 | 내용 | 실행 대상 | 크기(압축 전송량 추정) |
|---|---|---|---|
| `ghcr.io/jimmylegendary/k3-sm80-kernelkit:triton` | torch 2.13.0(cu12.6)+triton 3.7.1 | k01~k04·k06·k07 (vendored 커널, vLLM 없음) | ~3.7 GB |
| `ghcr.io/jimmylegendary/k3-sm80-kernelkit:full` | + vLLM kimi-k3 `0498dc7e` SM80 소스빌드 + **compute-sanitizer** | k01~k07 전부(k05 포함) + S4 | ~7–9 GB |

날짜태그(`:triton-YYYYMMDD`, `:full-YYYYMMDD`)는 같은 이미지의 스냅샷 별칭이다.

드라이버 요구: triton 티어 = CUDA 12.6 유저랜드(r560+ 권장, 데이터센터 minor-compat 시 그 이하 가능) ·
full 티어 = CUDA 13.0 유저랜드(**r580+**, RUNBOOK §1.1/§6.1). `nvidia-smi` 의 driver_version 먼저 확인.

## 0. 사전조건

```bash
docker --version                          # docker + nvidia-container-toolkit (--gpus all 동작해야 함)
docker run --rm --gpus all nvidia/cuda:13.0.3-base-ubuntu22.04 nvidia-smi   # GPU 패스스루 확인(선택)
git clone git@github.com:jimmylegendary/k3-sm80.git && cd k3-sm80           # repo (마운트 대상)
```

## 1. 반입 경로 ① — ghcr pull (기본)

ghcr 이미지는 private → pull 에도 로그인이 필요하다.

```bash
# gh CLI 있으면:
gh auth token | docker login ghcr.io -u <github-계정> --password-stdin
# gh 없으면: read:packages 스코프 PAT 로
echo <PAT> | docker login ghcr.io -u <github-계정> --password-stdin

docker pull ghcr.io/jimmylegendary/k3-sm80-kernelkit:triton
docker pull ghcr.io/jimmylegendary/k3-sm80-kernelkit:full
```

## 2. 반입 경로 ② — GitHub Release (registry 차단 시 폴백)

`docker save` 를 zstd 압축·1900MB 분할로 올려둔 것을 받아 결합한다.
릴리스 태그는 `kernelkit-YYYYMMDD` (푸시 로그/릴리스 페이지에서 확인).

```bash
TAG=kernelkit-20260729   # 실제 태그로 치환
gh release download "$TAG" -R jimmylegendary/k3-sm80 -p 'k3-sm80-kernelkit-full-*'
sha256sum -c k3-sm80-kernelkit-full-*.sha256          # 무결성 검증 (전부 OK 여야 함)
cat k3-sm80-kernelkit-full-*.tar.zst.part-* | unzstd | docker load
# triton 티어도 같은 방식: -p 'k3-sm80-kernelkit-triton-*'
```

`gh` 가 없으면 릴리스 페이지(웹)에서 part 파일들을 받아 같은 `cat | unzstd | docker load` 를 쓴다.
(웹 다운로드 후 파일명이 바뀌지 않았는지 확인 — part 순서는 `part-aa`, `part-ab`, … 사전순.)

주의(결합 오염 방지): `cat *.part-*` 는 글롭에 걸리는 **모든** 파일을 결합한다 — 이전에 받다 만
파일이나 다른 날짜의 part 가 같은 디렉터리에 있으면 이미지가 손상된다. **빈 디렉터리에서**
받고, 결합 전 `ls *.part-*` 목록이 `.sha256` 파일에 나열된 목록과 정확히 일치하는지 확인할 것.
(push.sh 는 릴리스당 tier 1세트 불변식을 강제하지만, 로컬 잔재까지는 막지 못한다.)

### 2.1 Kimi-K3 OpenAI serving image v3

5개 serving lane은 kernel-kit `:full`이 아니라
`k3-sm80-openai:0498dc7e-sm80-v3`를 사용한다. GHCR이 막힌 환경에서는 빈 NFS
디렉터리에 GitHub Release asset을 한 번 받은 뒤, 네 노드의 로컬 Docker daemon에
각각 load한다.

```bash
TAG=openai-0498dc7e-sm80-v3
mkdir -p /mnt/nas/home/ubuntu/service/large_model/Kimi-K3/images/openai-v3
cd /mnt/nas/home/ubuntu/service/large_model/Kimi-K3/images/openai-v3
gh release download "$TAG" -R jimmylegendary/k3-sm80 \
  -p 'k3-sm80-openai-0498dc7e-sm80-v3.tar.zst.part-*' \
  -p SHA256SUMS \
  -p image-inspect.json
sha256sum -c SHA256SUMS
cat k3-sm80-openai-0498dc7e-sm80-v3.tar.zst.part-* | zstd -d | docker load
docker image inspect k3-sm80-openai:0498dc7e-sm80-v3 --format '{{.Id}} {{index .Config.Labels "io.k3.patchset"}}'
```

같은 `cat ... | zstd -d | docker load`를 worker1–3에서도 실행한다. 디렉터리가 NFS라
part 파일을 다시 받을 필요는 없다. 네 노드의 image ID가 같아야 하며, 각 lane의
`make preflight`가 이를 다시 검증한다.

개발 호스트에서 image를 다시 publish할 때는 다음을 사용한다.

```bash
K3_OPENAI_GITHUB_UPLOAD=1 bash docker/release-openai.sh
```

이 명령은 1900MB part, `SHA256SUMS`, `image-inspect.json`을
`openai-0498dc7e-sm80-v3` Release에 생성/갱신한다. Release 업로드는 image의 CPU
contract만 의미하며 A100 kernel/service PASS를 의미하지 않는다.

## 3. 실행 (repo = /work 볼륨 마운트)

```bash
cd k3-sm80    # §0 에서 clone 한 repo 루트

# 전체 실행 (k05 포함 → full 이미지):
docker run --rm --gpus all -v "$PWD":/work \
  ghcr.io/jimmylegendary/k3-sm80-kernelkit:full \
  bash /work/kernel-kit/run_all.sh

# 부분 실행 (KERNELS 환경변수) — 예: triton 이미지로 k05 제외 전부:
docker run --rm --gpus all -v "$PWD":/work \
  -e KERNELS="k01_kda_chunk k02_kda_recurrent k03_causal_conv1d k04_mla_decode k06_dequant_bench k07_situ_act" \
  ghcr.io/jimmylegendary/k3-sm80-kernelkit:triton \
  bash /work/kernel-kit/run_all.sh

# 결과 반출: run_all 이 kernel-kit/ 에 k3-sm80-kernel-report-<ts>.tgz 를 만든다.
# /work 가 호스트 repo 마운트이므로 컨테이너 종료 후 그대로 호스트에 남아 있다 →
# 그 tgz 를 개발 호스트로 회신하면 된다 (PLAN "반복 루프").
ls -t kernel-kit/k3-sm80-kernel-report-*.tgz | head -1
```

메모:
- **SANITIZER 게이트**는 compute-sanitizer 가 있는 **full 이미지에서만 실제 실행**된다.
  triton 이미지에서는 해당 게이트가 SKIP(사유 기록)으로 남는다 — env_check 가 처리.
- 루트 소유 파일이 싫으면 비루트 실행: `-u $(id -u):$(id -g) -e HOME=/tmp` 를 추가
  (HOME=/tmp 는 run_all 의 venv 자동탐지를 안전하게 비활성화한다). 루트로 돌렸다면
  반출 tgz 가 root 소유가 되니 필요 시 `sudo chown $USER` 후 회신.
- GPU 지정: `--gpus '"device=0"'` 처럼 한 장만 물릴 수 있다 (S2 는 단일 GPU 로 충분).
- 컨테이너 안 python 은 이미지 동봉 python 이다 (venv 불필요 — run_all 이 자동 인지).
  RUNBOOK §2/§3 의 venv 절차는 **도커 미사용(베어메탈) 경로**의 이야기다.
- full 이미지 provenance: `/opt/vllm-build/` 에 VLLM_COMMIT.txt·build-k3.log.gz·pip-freeze.txt 동봉.

### exit code (run_all.sh)

| rc | 뜻 | 다음 행동 |
|---|---|---|
| 0 | FAIL 없음 | kernel-report tgz 회신 |
| 1 | FAIL/CRASH 존재 | 그대로 tgz 회신 (FAIL 로그도 S3 입력물) |
| 2 | 사전조건 미충족 skip | 컨테이너에서 나왔다면 대부분 `--gpus all` 누락 또는 드라이버 부족 — §4 |
| 3 | 프레임워크 오류 | 로그와 함께 회신 |

## 4. 트러블슈팅

| 증상 | 조치 |
|---|---|
| `could not select device driver "nvidia"` | nvidia-container-toolkit 미설치/미설정 — 사내 표준 절차로 설치 후 docker 재시작 |
| full 이미지에서 `CUDA driver version is insufficient` | 노드 드라이버 < r580. **triton 티어로 k01~k04·k06·k07 을 먼저 진행**하고 드라이버 업그레이드를 요청 (RUNBOOK §6.1) |
| env_check `gpu_visible FAIL` (컨테이너 안) | `--gpus all` 빠짐. docker run 옵션 확인 |
| ghcr pull `denied` | §1 로그인 누락 또는 계정에 repo/package 권한 없음 → 경로② 사용 |
| release 다운로드가 느림/끊김 | `gh release download` 는 재실행 시 이어받지 않음 — part 단위로 `-p '...part-aa'` 지정해 나눠 받기 |
| `docker load` 후 이미지가 안 보임 | `docker image ls ghcr.io/jimmylegendary/k3-sm80-kernelkit` 로 확인 — load 는 태그를 보존한다 |
