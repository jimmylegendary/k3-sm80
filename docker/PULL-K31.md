# 사내 반입·실행 절차: K3.1 OpenAI 서빙 이미지 (K3 용 [PULL.md](/home/jimmy/repos/k3-sm80/docker/PULL.md) 와 별개)

| 이미지 | 태그 | 내용 | 반입 |
|---|---|---|---|
| K3.1 full | `k31-sm80-kernelkit:full-a3c14efb` | vLLM main `a3c14efb` SM80 소스빌드(패치 전) + compute-sanitizer | 개발 호스트 로컬 태그(Release 미포함) |
| K3.1 openai | `k31-sm80-openai:a3c14efb-sm80-v1` | full + marlin `.so` + TRITON_MLA_SPARSE + DSA glue (패치 3개) | GitHub Release `openai-k31-a3c14efb-sm80-v1` (1900MB part, `SHA256SUMS`, `image-inspect.json`) |

K3 이미지(`k3-sm80-openai:0498dc7e-sm80-v3`, kernelkit `:full`, `:triton`)와 lane 01~05 는 그대로다.

드라이버: 두 이미지 모두 CUDA 13.0.3 유저랜드라 **r580+** 가 필요하다(K3 full 티어와 동일, K3 가 이미 동작 중이면 충족). 확인은 `python3 docker/verify_k31_image.py --gpu` 가 하며, 드라이버가 580 미만이면 WARN 을 낸다.

## load (노드별 1회, 멱등)
    cd <k3-sm80 clone>
    bash run/11-load-k31-image.sh        # gh 있으면 자동 다운로드, 없으면 안내

수동:

    TAG=openai-k31-a3c14efb-sm80-v1      # 빈 디렉터리에서
    gh release download "$TAG" -R jimmylegendary/k3-sm80 -p 'k31-sm80-openai-a3c14efb-sm80-v1.tar.zst.part-*' -p SHA256SUMS -p image-inspect.json
    sha256sum -c SHA256SUMS
    cat k31-sm80-openai-a3c14efb-sm80-v1.tar.zst.part-* | zstd -d | docker load
    docker image inspect k31-sm80-openai:a3c14efb-sm80-v1 --format '{{.Id}} {{index .Config.Labels "io.k3.patchset"}}'

네 노드의 image ID 가 같아야 한다. `cat *.part-*` 는 글롭에 걸린 모든 파일을 이어 붙이므로 빈 디렉터리에서 받는다([run/11-load-k31-image.sh](/home/jimmy/repos/k3-sm80/run/11-load-k31-image.sh) 는 part 목록과 `SHA256SUMS` 의 일치를 검사한다).

## 확인과 커널 gate
    python3 docker/verify_k31_image.py --gpu        # [static]/[import]/[cli]/[gpu] 구역별 출력
    bash run/21-k31-kernel-test.sh                  # auto: full 이 없으면 openai 이미지로, k08 은 패치 후 회귀 기준

## 개발 호스트 publish
    bash docker/build-k31.sh && K31_OPENAI_GITHUB_UPLOAD=1 bash docker/release-k31.sh

Release 업로드는 이미지의 CPU 계약만 의미하며 A100 kernel/service PASS 가 아니다.
