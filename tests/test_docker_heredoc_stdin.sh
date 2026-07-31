#!/usr/bin/env bash
# Regression: docker run must keep stdin open when Python source is supplied
# through a heredoc. Without -i, python3 exits 0 after reading immediate EOF
# and the caller sees no output.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${K3_TEST_IMAGE:-ghcr.io/jimmylegendary/k3-sm80-kernelkit:triton}"

for script in run/26-k01-ab.sh run/27-perf-calib.sh; do
  python3 - "$ROOT/$script" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
match = re.search(
    r"docker\s+run(?P<options>.*?)"
    r'"\$IMG"\s+python3\s+(?P<python_options>(?:-\w+\s+)*)-\s*<<',
    source,
    flags=re.DOTALL,
)
assert match, f"{path}: docker heredoc Python invocation not found"
tokens = match.group("options").replace("\\\n", " ").split()
assert "-i" in tokens or "--interactive" in tokens, (
    f"{path}: docker run must include -i/--interactive for heredoc stdin"
)
assert "-u" in match.group("python_options").split(), (
    f"{path}: python3 must include -u so progress is visible before completion"
)
PY
done

if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  output="$(
    docker run --rm -i "$IMAGE" python3 -u - <<'PY'
print("SENTINEL-HEREDOC", flush=True)
PY
  )"
  [[ "$output" == "SENTINEL-HEREDOC" ]] || {
    echo "dynamic heredoc test failed: <$output>" >&2
    exit 1
  }
else
  echo "SKIP dynamic test: image unavailable ($IMAGE)"
fi

echo "PASS docker heredoc stdin regression"
