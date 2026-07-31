# Kimi K3 Five-Lane Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build five independently operable four-node Kimi K3 serving lanes, their A100 kernel/communication gates, and reproducible runbooks.

**Architecture:** Each lane is a self-contained directory with one `.env`, one four-profile Compose file, one Makefile, scripts, and one Runbook. A standard-library Python generator owns repeated artifacts and a validator prevents drift; a patched source-built OpenAI image supplies KDA, MXFP4, FP8-KV, and DSpark paths. Kernel gates use the existing `KernelReport` schema and are promoted into lane-specific digest-bound stamps.

**Tech Stack:** Python 3.11+, Bash, GNU Make, Docker Compose v2, PyTorch 2.13/CUDA 13, Triton 3.7, pinned vLLM `0498dc7ea9204f1863b46515816f5d8d458e01e7`, NCCL, compute-sanitizer.

## Global Constraints

- Hardware is fixed at four nodes with eight A100-80GB SM80 GPUs per node; head node participates.
- Use vLLM native multi-node `mp`, never Ray or `external_launcher`.
- Target checkpoint is original `moonshotai/Kimi-K3` MXFP4 from shared `HF_HOME=/mnt/nas/home/models`.
- Serving root is `/mnt/nas/home/ubuntu/service/large_model/Kimi-K3` on the same NFS mount on all nodes.
- Every lane has exactly one `.env`; its head and three workers share that file. No node-specific or override env files.
- Every lane Compose has profiles `head`, `worker1`, `worker2`, and `worker3`.
- Unresolved site values use `CHANGE_ME_*` and make `preflight` fail.
- GPU-less host runs must report GPU gates as not executed; they may not report PASS.
- GPU completion can only be claimed from fresh A100 evidence bound to image digest, repository commit, GPU/driver, and lane env hash.
- Lane 01 is TP8×PP4; lanes 02–05 are TP8×DP4×EP32.
- Lanes 01–03 use 256k maximum context; lanes 04–05 use 1M and require k09.
- Lanes 03 and 05 use `Inferact/Kimi-K3-DSpark` and require k10.
- Runbooks never require returning a tarball; failures emit a stable code plus at most 20 useful lines.

---

## File Map

### Deployment control

- `deploy/tooling/lane_schema.py`: parse and validate one lane `.env`.
- `deploy/tooling/lane_specs.py`: immutable definitions for lanes 01–05.
- `deploy/tooling/render_lanes.py`: render committed lane artifacts deterministically.
- `deploy/tooling/validate_lanes.py`: enforce count, schema, Compose, command, and drift invariants.
- `deploy/tooling/stamps.py`: digest-bound gate/calibration stamp API.
- `deploy/tooling/templates/`: Compose, Makefile, shell/Python script, and Runbook templates.
- `deploy/01-*` through `deploy/05-*`: generated, self-contained runtime directories.
- `tests/deploy/`: unit, snapshot, shell syntax, and Compose rendering tests.

### Image integration

- `docker/Dockerfile.openai`: pinned source build and OpenAI entrypoint.
- `docker/build-openai.sh`: deterministic SM80 build and host-side static checks.
- `docker/release-openai.sh`: GitHub Release split archive producer.
- `docker/patches/kda-sm80-int64.patch`: apply the proven KDA offset fix to actual vLLM.
- `docker/patches/mxfp4-marlin-moe-sm80.patch`: k08-driven Marlin changes.
- `docker/patches/mla-fp8-kv-sm80.patch`: k09-driven K3 MLA/Triton changes.
- `docker/verify_image.py`: source revision, patch, architecture, and CLI assertions.

### New gates

- `kernel-kit/k08_mxfp4_marlin_moe/harness.py`: original K3 MXFP4 TP/EP arms.
- `kernel-kit/k09_mla_fp8_kv_sm80/harness.py`: SM80 software FP8 cache kernel gate.
- `kernel-kit/k10_k3_dspark_sm80/harness.py`: DSpark draft/verify/state gate.
- `kernel-kit/c01_ep32_a2a/harness.py`: 32-rank communication gate.
- `kernel-kit/common/mxfp4_ref.py`: MXFP4/E8M0 unpack and MoE reference.
- `kernel-kit/common/fp8_ref.py`: bit-exact E4M3 encode/decode reference.
- `kernel-kit/common/gate_stamp.py`: report-to-stamp conversion shared with deployment.
- `kernel-kit/vendor/mla_fp8_kv_sm80.py`: standalone Triton pack/load kernels.
- `tests/kernel/`: CPU reference and report contract tests.

---

### Task 1: Lane schema and immutable specifications

**Files:**
- Create: `deploy/tooling/__init__.py`
- Create: `deploy/tooling/lane_schema.py`
- Create: `deploy/tooling/lane_specs.py`
- Create: `tests/deploy/test_lane_schema.py`

**Interfaces:**
- Produces: `ALLOWED_KEYS: frozenset[str]`
- Produces: `LaneEnv.load(path: Path) -> LaneEnv`
- Produces: `LaneEnv.parse(text: str) -> LaneEnv`
- Produces: `LaneEnv.validate(spec: LaneSpec) -> list[str]`
- Produces: `LaneSpec(id: str, tp: int, pp: int, dp: int, ep: bool, max_len: int, dspark: bool, fp8_kv: bool)`
- Produces: `LANES: tuple[LaneSpec, ...]`

- [ ] **Step 1: Write the failing schema tests**

```python
class LaneSchemaTest(unittest.TestCase):
    def test_exact_lane_contract(self):
        self.assertEqual(
            [(x.id, x.tp, x.pp, x.dp, x.ep, x.max_len, x.dspark, x.fp8_kv)
             for x in LANES],
            [
                ("01-tp8-pp4-256k-safe", 8, 4, 1, False, 262144, False, False),
                ("02-tp8-dp4-ep32-256k-safe", 8, 1, 4, True, 262144, False, False),
                ("03-tp8-dp4-ep32-256k-dspark", 8, 1, 4, True, 262144, True, False),
                ("04-tp8-dp4-ep32-1m-fp8kv", 8, 1, 4, True, 1048576, False, True),
                ("05-tp8-dp4-ep32-1m-fp8kv-dspark", 8, 1, 4, True, 1048576, True, True),
            ],
        )

    def test_rejects_unknown_and_duplicate_keys(self):
        env = LaneEnv.parse("VLLM_MAX_MODEL_LEN=262144\nVLLM_MAX_MODEL_LEN=1\nBOGUS=1\n")
        self.assertIn("duplicate key: VLLM_MAX_MODEL_LEN", env.errors)
        self.assertIn("unknown key: BOGUS", env.errors)
```

- [ ] **Step 2: Run the schema test and observe the import failure**

Run: `python3 -m unittest tests.deploy.test_lane_schema -v`

Expected: FAIL because `deploy.tooling.lane_schema` and `lane_specs` do not exist.

- [ ] **Step 3: Implement strict standard-library parsing and lane constants**

Implement `LaneEnv.parse()` without `source` or `eval`; accept comments, blank lines, and one
`KEY=VALUE` per line. Define a closed key set containing all `K3_*` and `VLLM_*` names from the
approved spec. Validate integer, boolean (`0|1`), JSON, port, topology product, model path, and
`CHANGE_ME_*` rules.

```python
@dataclass(frozen=True)
class LaneSpec:
    id: str
    tp: int
    pp: int
    dp: int
    ep: bool
    max_len: int
    dspark: bool
    fp8_kv: bool

@dataclass
class LaneEnv:
    values: dict[str, str]
    errors: list[str]

    @classmethod
    def parse(cls, text: str) -> "LaneEnv":
        values: dict[str, str] = {}
        errors: list[str] = []
        for line_number, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if not separator:
                errors.append(f"line {line_number}: expected KEY=VALUE")
            elif key in values:
                errors.append(f"duplicate key: {key}")
            else:
                values[key] = value
        errors.extend(f"unknown key: {key}" for key in values.keys() - ALLOWED_KEYS)
        return cls(values, errors)
```

- [ ] **Step 4: Run schema tests**

Run: `python3 -m unittest tests.deploy.test_lane_schema -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add deploy/tooling tests/deploy/test_lane_schema.py
git commit -m "feat(deploy): define five-lane env schema"
```

### Task 2: Deterministic lane artifact generator

**Files:**
- Create: `deploy/tooling/render_lanes.py`
- Create: `deploy/tooling/templates/compose.yaml.tmpl`
- Create: `deploy/tooling/templates/Makefile.tmpl`
- Create: `deploy/tooling/templates/preflight.py.tmpl`
- Create: `deploy/tooling/templates/cluster.sh.tmpl`
- Create: `deploy/tooling/templates/smoke.py.tmpl`
- Create: `deploy/tooling/templates/RUNBOOK.md.tmpl`
- Create: `tests/deploy/test_render_lanes.py`

**Interfaces:**
- Consumes: `LANES`, `LaneSpec`
- Produces: `render_template(name: str, context: dict[str, object]) -> str`
- Produces: `render_env(context: dict[str, object]) -> str`
- Produces: `render_lane(spec: LaneSpec, root: Path) -> list[Path]`
- Produces: `render_all(root: Path, check: bool = False) -> int`

- [ ] **Step 1: Write failing deterministic-render tests**

```python
def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

def test_rendered_lane_is_self_contained(self):
    render_lane(LANES[0], self.tmp)
    lane = self.tmp / LANES[0].id
    self.assertEqual(
        {p.name for p in lane.iterdir()},
        {".env", "compose.yaml", "Makefile", "RUNBOOK.md", "scripts"},
    )
    self.assertEqual((lane / "compose.yaml").read_text().count("profiles:"), 4)

def test_render_is_byte_deterministic(self):
    render_all(self.tmp)
    first = tree_hash(self.tmp)
    render_all(self.tmp)
    self.assertEqual(first, tree_hash(self.tmp))
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m unittest tests.deploy.test_render_lanes -v`

Expected: FAIL because `render_lanes` and templates do not exist.

- [ ] **Step 3: Implement generator and exact lane `.env` defaults**

Render all five lane directories. Each `.env` contains the same variable names and lane-specific
values. Use unique ports: API `8101..8105`, MP `29601..29605`, and DP RPC `29701..29705`.
Set `K3_IMAGE=k3-sm80-openai:0498dc7e-sm80`,
`K3_TARGET_MODEL=moonshotai/Kimi-K3`,
`K3_DRAFT_MODEL=Inferact/Kimi-K3-DSpark`,
`K3_HF_HOME=/mnt/nas/home/models`, and the approved serving root.
Use `CHANGE_ME_*` only for node, SSH, and interface values.

```python
def render_lane(spec: LaneSpec, root: Path) -> list[Path]:
    lane = root / spec.id
    scripts = lane / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    number = int(spec.id[:2])
    context = {
        "spec": spec,
        "api_port": 8100 + number,
        "master_port": 29600 + number,
        "dp_rpc_port": 29700 + number,
    }
    outputs = {
        lane / ".env": render_env(context),
        lane / "compose.yaml": render_template("compose.yaml.tmpl", context),
        lane / "Makefile": render_template("Makefile.tmpl", context),
        lane / "RUNBOOK.md": render_template("RUNBOOK.md.tmpl", context),
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
    return sorted(outputs)
```

- [ ] **Step 4: Run render tests and generator check mode**

Run:

```bash
python3 -m unittest tests.deploy.test_render_lanes -v
python3 -m deploy.tooling.render_lanes
python3 -m deploy.tooling.render_lanes --check
```

Expected: tests PASS and `--check` exits 0 with `drift=0`.

- [ ] **Step 5: Commit**

```bash
git add deploy tests/deploy/test_render_lanes.py
git commit -m "feat(deploy): render five self-contained serving lanes"
```

### Task 3: Compose command construction and static validation

**Files:**
- Create: `deploy/tooling/command.py`
- Create: `deploy/tooling/validate_lanes.py`
- Create: `tests/deploy/test_commands.py`
- Create: `tests/deploy/test_static_layout.py`

**Interfaces:**
- Produces: `validate_extra_args(args: list[str]) -> list[str]`
- Produces: `build_vllm_args(env: LaneEnv, node_rank: int) -> list[str]`
- Produces: `validate_repository(root: Path) -> list[str]`

- [ ] **Step 1: Write failing command tests**

```python
def contains_in_order(values: list[str], expected: list[str]) -> bool:
    index = -1
    for value in expected:
        try:
            index = values.index(value, index + 1)
        except ValueError:
            return False
    return True

def test_lane_01_mp_command(self):
    env = LaneEnv.load(Path("deploy/01-tp8-pp4-256k-safe/.env"))
    args = build_vllm_args(env, node_rank=3)
    self.assertTrue(contains_in_order(args, [
        "--distributed-executor-backend", "mp",
        "--nnodes", "4", "--node-rank", "3",
        "--tensor-parallel-size", "8",
        "--pipeline-parallel-size", "4",
        "--max-model-len", "262144",
        "--max-num-seqs", "1",
    ]))
    self.assertNotIn("--enable-expert-parallel", args)

def test_lane_05_has_ep_fp8_and_dspark(self):
    env = LaneEnv.load(Path("deploy/05-tp8-dp4-ep32-1m-fp8kv-dspark/.env"))
    args = build_vllm_args(env, node_rank=0)
    self.assertIn("--enable-expert-parallel", args)
    self.assertEqual(args[args.index("--kv-cache-dtype") + 1], "fp8")
    self.assertEqual(
        args[args.index("--speculative-config") + 1],
        env.values["VLLM_SPECULATIVE_CONFIG"],
    )
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m unittest tests.deploy.test_commands tests.deploy.test_static_layout -v`

Expected: FAIL because command and validator modules do not exist.

- [ ] **Step 3: Implement token-safe argument construction and repository checks**

Use arrays, never `eval`. Parse `VLLM_EXTRA_ARGS` with `shlex.split`; reject duplication of
topology, quantization, cache, context, model, and speculative flags. Validate exactly five `.env`
files, no override env, four profiles per Compose, unique ports/projects, and world size 32.

```python
def build_vllm_args(env: LaneEnv, node_rank: int) -> list[str]:
    v = env.values
    args = [
        v["K3_TARGET_MODEL"],
        "--distributed-executor-backend", "mp",
        "--nnodes", "4",
        "--node-rank", str(node_rank),
        "--master-addr", v["K3_HEAD_NODE_IP"],
        "--master-port", v["VLLM_MASTER_PORT"],
        "--tensor-parallel-size", v["VLLM_TENSOR_PARALLEL_SIZE"],
        "--pipeline-parallel-size", v["VLLM_PIPELINE_PARALLEL_SIZE"],
        "--data-parallel-size", v["VLLM_DATA_PARALLEL_SIZE"],
        "--max-model-len", v["VLLM_MAX_MODEL_LEN"],
        "--max-num-seqs", v["VLLM_MAX_NUM_SEQS"],
        "--quantization", v["VLLM_QUANTIZATION"],
        "--dtype", v["VLLM_DTYPE"],
        "--kv-cache-dtype", v["VLLM_KV_CACHE_DTYPE"],
    ]
    if v["VLLM_ENABLE_EXPERT_PARALLEL"] == "1":
        args.append("--enable-expert-parallel")
    if v["VLLM_SPECULATIVE_CONFIG"]:
        args.extend(["--speculative-config", v["VLLM_SPECULATIVE_CONFIG"]])
    args.extend(validate_extra_args(shlex.split(v["VLLM_EXTRA_ARGS"])))
    return args
```
- [ ] **Step 4: Run unit and Compose rendering tests**

Run:

```bash
python3 -m unittest tests.deploy.test_commands tests.deploy.test_static_layout -v
for d in deploy/0*-*; do (cd "$d" && docker compose --env-file .env config --quiet); done
python3 -m deploy.tooling.validate_lanes
```

Expected: all commands exit 0 and validator prints `lanes=5 env_files=5 errors=0`.

- [ ] **Step 5: Commit**

```bash
git add deploy/tooling deploy/0*-* tests/deploy
git commit -m "feat(deploy): validate lane-specific vllm commands"
```

### Task 4: Gate stamps and four-node orchestration

**Files:**
- Create: `deploy/tooling/stamps.py`
- Create: `deploy/tooling/remote.py`
- Modify: `deploy/tooling/templates/Makefile.tmpl`
- Modify: `deploy/tooling/templates/preflight.py.tmpl`
- Modify: `deploy/tooling/templates/cluster.sh.tmpl`
- Create: `tests/deploy/test_stamps.py`
- Create: `tests/deploy/test_make_dry_run.sh`

**Interfaces:**
- Produces: `StampKey(repo_commit, image_digest, env_sha256, gate_id)`
- Produces: `RemoteNode(profile: str, ssh_user: str, ssh_host: str)`
- Produces: `RemoteLane(shared_directory: str, workers: tuple[RemoteNode, ...])`
- Produces: `write_stamp(path: Path, key: StampKey, evidence: dict[str, object]) -> None`
- Produces: `validate_stamp(path: Path, expected: StampKey) -> list[str]`
- Produces: CLI `python3 -m deploy.tooling.remote {up,down,status,logs} --lane PATH --dry-run`

- [ ] **Step 1: Write failing stamp and dry-run tests**

Test that changing one `.env` byte invalidates a stamp; test remote dry-run emits exactly four
profile commands and never invokes SSH.

```python
def test_env_change_invalidates_stamp(self):
    key = StampKey("abc", "sha256:image", "sha256:env-a", "k08")
    write_stamp(self.path, key, {"overall": "PASS"})
    changed = StampKey("abc", "sha256:image", "sha256:env-b", "k08")
    self.assertEqual(validate_stamp(self.path, changed), ["env_sha256 mismatch"])
```

```bash
out="$(python3 -m deploy.tooling.remote up \
  --lane deploy/01-tp8-pp4-256k-safe --dry-run)"
test "$(printf '%s\n' "$out" | grep -c 'profile=')" -eq 4
! printf '%s\n' "$out" | grep -q 'ssh executed'
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python3 -m unittest tests.deploy.test_stamps -v
bash tests/deploy/test_make_dry_run.sh
```

Expected: both fail because orchestration APIs are absent.

- [ ] **Step 3: Implement NFS lock, stamp binding, and rollback-safe startup**

Start head, then workers in parallel over SSH using the shared lane directory. Wait for 32 ranks
and `/health`; on failure, stop only containers started by the current run id. Make `down`
idempotent and continue across unreachable workers. Never disable SSH host-key checking.

```python
def validate_stamp(path: Path, expected: StampKey) -> list[str]:
    actual = json.loads(path.read_text())
    return [
        f"{field} mismatch"
        for field in dataclasses.fields(StampKey)
        if actual[field.name] != getattr(expected, field.name)
    ]
```

```python
for node in lane.workers:
    commands.append([
        "ssh", f"{node.ssh_user}@{node.ssh_host}",
        "cd", lane.shared_directory, "&&",
        "docker", "compose", "--env-file", ".env",
        "--profile", node.profile, "up", "-d",
    ])
```
- [ ] **Step 4: Regenerate and run orchestration tests**

Run:

```bash
python3 -m deploy.tooling.render_lanes
python3 -m unittest tests.deploy.test_stamps -v
bash tests/deploy/test_make_dry_run.sh
for d in deploy/0*-*; do make -C "$d" dry-run; done
```

Expected: PASS; each dry-run reports ranks `0,1,2,3` and performs no mutation.

- [ ] **Step 5: Commit**

```bash
git add deploy tests/deploy
git commit -m "feat(deploy): orchestrate four-node mp lanes"
```

### Task 5: Patched OpenAI serving image

**Files:**
- Create: `docker/Dockerfile.openai`
- Create: `docker/build-openai.sh`
- Create: `docker/release-openai.sh`
- Create: `docker/verify_image.py`
- Create: `docker/patches/kda-sm80-int64.patch`
- Create: `tests/docker/test_openai_image_contract.py`

**Interfaces:**
- Produces image: `k3-sm80-openai:0498dc7e-sm80`
- Produces labels: `org.opencontainers.image.revision`, `io.k3.cuda_arch`, `io.k3.patchset`
- Produces CLI: `python3 docker/verify_image.py IMAGE`

- [ ] **Step 1: Write failing Dockerfile contract test**

Assert pinned commit, `TORCH_CUDA_ARCH_LIST=8.0`, actual KDA source patch, OpenAI entrypoint,
healthcheck, and immutable labels.

```python
def test_openai_image_is_pinned_and_patched(self):
    dockerfile = Path("docker/Dockerfile.openai").read_text()
    self.assertIn("0498dc7ea9204f1863b46515816f5d8d458e01e7", dockerfile)
    self.assertIn("TORCH_CUDA_ARCH_LIST=8.0", dockerfile)
    self.assertIn("kda-sm80-int64.patch", dockerfile)
    self.assertIn('ENTRYPOINT ["vllm", "serve"]', dockerfile)
    self.assertIn("HEALTHCHECK", dockerfile)
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m unittest tests.docker.test_openai_image_contract -v`

Expected: FAIL because the OpenAI Dockerfile and verifier do not exist.

- [ ] **Step 3: Implement source build and KDA patch assertion**

Build vLLM at the pinned commit, apply KDA int64 fixes to the installed source rather than the
standalone vendor copy, compile SM80 extensions, and make `vllm serve --help` the build gate.
Package `docker save | zstd | split` artifacts without requiring GHCR.

```dockerfile
ARG VLLM_COMMIT=0498dc7ea9204f1863b46515816f5d8d458e01e7
ENV TORCH_CUDA_ARCH_LIST=8.0
RUN git clone https://github.com/vllm-project/vllm.git /src/vllm \
 && git -C /src/vllm checkout --detach "${VLLM_COMMIT}"
COPY docker/patches/kda-sm80-int64.patch /tmp/patches/kda.patch
RUN git -C /src/vllm apply --check /tmp/patches/kda.patch \
 && git -C /src/vllm apply /tmp/patches/kda.patch
RUN python3 -m pip install --no-build-isolation /src/vllm
RUN vllm serve --help >/dev/null
ENTRYPOINT ["vllm", "serve"]
```
- [ ] **Step 4: Run static contract and CPU image inspection**

Run:

```bash
python3 -m unittest tests.docker.test_openai_image_contract -v
bash -n docker/build-openai.sh docker/release-openai.sh
docker build --target contract -f docker/Dockerfile.openai .
python3 docker/verify_image.py k3-sm80-openai:0498dc7e-sm80
```

Expected: static tests PASS; contract target verifies source/CLI without claiming GPU execution.

- [ ] **Step 5: Commit**

```bash
git add docker tests/docker
git commit -m "feat(docker): build patched K3 OpenAI image for SM80"
```

### Task 6: k08 CPU reference and report contract

**Files:**
- Create: `kernel-kit/common/mxfp4_ref.py`
- Create: `kernel-kit/k08_mxfp4_marlin_moe/harness.py`
- Modify: `kernel-kit/common/shapes.py`
- Modify: `kernel-kit/common/tolerance.py`
- Modify: `kernel-kit/run_all.sh`
- Create: `tests/kernel/test_mxfp4_ref.py`
- Create: `tests/kernel/test_k08_dry_run.py`

**Interfaces:**
- Produces: `decode_e2m1(packed: bytes) -> list[float]`
- Produces: `decode_e8m0(scales: bytes) -> list[float]`
- Produces: `mxfp4_moe_reference(x, packed_w, scales, routing) -> torch.Tensor`
- Produces: `build_dry_report() -> dict[str, object]`
- Produces report id: `k08_mxfp4_marlin_moe`

- [ ] **Step 1: Write bit-pattern and dry-run failing tests**

Use all 16 E2M1 values, E8M0 boundary exponents, one tiny two-expert fixture, and report schema
checks requiring TP and EP arms.

```python
def test_all_e2m1_codes_are_defined(self):
    decoded = [decode_e2m1(bytes([code]))[0] for code in range(16)]
    self.assertEqual(len(decoded), 16)
    self.assertTrue(all(math.isfinite(x) for x in decoded))

def test_k08_dry_run_requires_both_layout_arms(self):
    report = build_dry_report()
    self.assertEqual(report["overall"], "INCOMPLETE")
    self.assertEqual(report["evidence"]["layout_arms"], ["tp8", "ep32"])
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m unittest tests.kernel.test_mxfp4_ref tests.kernel.test_k08_dry_run -v`

Expected: FAIL because reference and harness do not exist.

- [ ] **Step 3: Implement reference, shapes, tolerance, and dry-run**

Register K3 shapes `(hidden=3584, intermediate=3072, experts=896, topk=16)` for token counts
1, 8, 32, plus representative prefill. Dry-run validates metadata only and reports INCOMPLETE.

```python
REQUIRED_SHAPES["k08_mxfp4_marlin_moe"] = {
    "m1": {"m": 1, "k": 3584, "n": 3072, "e": 896, "topk": 16},
    "m8": {"m": 8, "k": 3584, "n": 3072, "e": 896, "topk": 16},
    "m32": {"m": 32, "k": 3584, "n": 3072, "e": 896, "topk": 16},
    "prefill": {"m": 8192, "k": 3584, "n": 3072, "e": 896, "topk": 16},
}
```
- [ ] **Step 4: Run CPU tests and aggregate dry-run**

Run:

```bash
python3 -m unittest tests.kernel.test_mxfp4_ref tests.kernel.test_k08_dry_run -v
python3 kernel-kit/k08_mxfp4_marlin_moe/harness.py --cpu-dry-run
bash kernel-kit/run_all.sh --cpu-dry-run
```

Expected: references PASS; k08 overall INCOMPLETE; aggregate has no false GPU PASS.

- [ ] **Step 5: Commit**

```bash
git add kernel-kit tests/kernel
git commit -m "test(kernel): add original MXFP4 MoE gate"
```

### Task 7: k08 GPU arms and Marlin patch

**Files:**
- Modify: `kernel-kit/k08_mxfp4_marlin_moe/harness.py`
- Create: `docker/patches/mxfp4-marlin-moe-sm80.patch`
- Modify: `docker/Dockerfile.openai`
- Create: `tests/kernel/test_k08_source_contract.py`

**Interfaces:**
- Consumes: `_moe_C.fused_marlin_moe`
- Produces gate arms: `tp8`, `ep32`
- Produces: `classify_child(case: str, returncode: int, child_json: str, stderr: str) -> dict[str, object]`
- Produces sanitizer evidence with memory-violation count separated from allocator OOM

- [ ] **Step 1: Write failing source-contract tests**

Assert the patch touches the pinned MXFP4 MoE instantiation/build target and the harness invokes
the same op signature used by K3 compressed tensors.

```python
def test_marlin_patch_and_harness_share_mxfp4_contract(self):
    patch = Path("docker/patches/mxfp4-marlin-moe-sm80.patch").read_text()
    harness = Path("kernel-kit/k08_mxfp4_marlin_moe/harness.py").read_text()
    self.assertIn("sm_80", patch)
    self.assertIn("fused_marlin_moe", patch)
    self.assertIn("weight_packed", harness)
    self.assertIn("weight_scale", harness)
    self.assertIn("fused_marlin_moe", harness)
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m unittest tests.kernel.test_k08_source_contract -v`

Expected: FAIL because the patch and GPU arms are absent.

- [ ] **Step 3: Implement the minimal SM80 instantiation/bug fix and isolated child cases**

Each case runs in a subprocess so illegal access cannot poison later cases. Use a crash ladder
`m1,m8,m32,prefill`, BF16 reference, canaries, and compute-sanitizer. Do not classify
`cudaErrorMemoryAllocation` alone as an invalid access.

```python
for case in ("m1", "m8", "m32", "prefill"):
    child_json = str(report_dir / f"{case}.json")
    result = subprocess.run(
        [sys.executable, __file__, "--child-case", case, "--child-out", child_json],
        env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
        text=True,
        capture_output=True,
    )
    records.append(classify_child(case, result.returncode, child_json, result.stderr))
```
- [ ] **Step 4: Run host checks and record the exact A100 command**

Host:

```bash
python3 -m unittest tests.kernel.test_k08_source_contract -v
python3 kernel-kit/k08_mxfp4_marlin_moe/harness.py --cpu-dry-run
```

A100:

```bash
python3 kernel-kit/k08_mxfp4_marlin_moe/harness.py \
  --report-dir /mnt/nas/home/ubuntu/service/large_model/Kimi-K3/runs/gates/k08
```

Expected on host: PASS/INCOMPLETE. Expected on A100 before lane promotion: all four gates PASS.

- [ ] **Step 5: Commit**

```bash
git add docker kernel-kit tests/kernel
git commit -m "fix(kernel): enable K3 MXFP4 Marlin MoE on SM80"
```

### Task 8: c01 EP32 communication gate

**Files:**
- Create: `kernel-kit/c01_ep32_a2a/harness.py`
- Create: `kernel-kit/c01_ep32_a2a/launch.sh`
- Create: `tests/kernel/test_c01_contract.py`

**Interfaces:**
- Produces report id: `c01_ep32_a2a`
- Consumes environment: `MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK`, `NNODES=4`
- Produces: `rank_layout(node_rank: int, local_rank: int) -> tuple[int, int, int]`
- Produces: `validate_backend(name: str) -> None`

- [ ] **Step 1: Write failing topology and dry-run tests**

Assert 32 global ranks, eight local ranks, deterministic routed-token fixtures, skew fixture,
timeout classification, and no SM90 backend.

```python
def test_rank_layout_covers_32_unique_ranks(self):
    rows = [rank_layout(node, local) for node in range(4) for local in range(8)]
    self.assertEqual(sorted(global_rank for global_rank, _, _ in rows), list(range(32)))

def test_sm90_backends_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "SM80 correctness backend"):
        validate_backend("deepep_v2")
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m unittest tests.kernel.test_c01_contract -v`

Expected: FAIL because c01 files do not exist.

- [ ] **Step 3: Implement torch.distributed NCCL correctness and timing**

Use `all_gather_into_tensor` plus `reduce_scatter_tensor` to mirror the generic backend. Verify
round-trip payload and counts for 1, 8, and 32 routed tokens and a single-hot expert skew.

```python
def rank_layout(node_rank: int, local_rank: int) -> tuple[int, int, int]:
    global_rank = node_rank * 8 + local_rank
    tp_rank = local_rank
    dp_rank = node_rank
    return global_rank, tp_rank, dp_rank
```

```python
dist.all_gather_into_tensor(gathered, local_payload, group=ep_group)
dist.reduce_scatter_tensor(combined, gathered, op=dist.ReduceOp.SUM, group=ep_group)
torch.testing.assert_close(combined, expected)
```
- [ ] **Step 4: Run host dry-run and publish four-node command in the harness help**

Run:

```bash
python3 -m unittest tests.kernel.test_c01_contract -v
python3 kernel-kit/c01_ep32_a2a/harness.py --cpu-dry-run
```

Expected: contract PASS and GPU gates INCOMPLETE.

- [ ] **Step 5: Commit**

```bash
git add kernel-kit/c01_ep32_a2a tests/kernel
git commit -m "test(comm): add four-node EP32 gate"
```

### Task 9: k10 DSpark SM80 gate

**Files:**
- Create: `kernel-kit/k10_k3_dspark_sm80/harness.py`
- Create: `kernel-kit/common/dspark_ref.py`
- Modify: `kernel-kit/common/shapes.py`
- Create: `tests/kernel/test_k10_contract.py`

**Interfaces:**
- Produces report id: `k10_k3_dspark_sm80`
- Produces cases: `greedy_accept_0..7`, `probabilistic_block`, `ctx256k`, `ctx1m`
- Produces: `accepted_state_prefix(states: Sequence[T], accepted: int) -> Sequence[T]`

- [ ] **Step 1: Write failing DSpark state-transition tests**

Create CPU fixtures proving accepted token count 0–7 advances KDA recurrent and conv state by the
accepted count only; verify target auxiliary layer ids `(2,23,47,71,89)` and block size 7.

```python
def test_every_accept_count_commits_only_accepted_states(self):
    states = tuple(f"state-{i}" for i in range(8))
    for accepted in range(8):
        self.assertEqual(
            accepted_state_prefix(states, accepted),
            states[:accepted],
        )

def test_k3_dspark_contract(self):
    self.assertEqual(TARGET_AUX_LAYER_IDS, (2, 23, 47, 71, 89))
    self.assertEqual(DSPARK_BLOCK_SIZE, 7)
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m unittest tests.kernel.test_k10_contract -v`

Expected: FAIL because DSpark reference and harness do not exist.

- [ ] **Step 3: Implement reference and isolated vLLM integration arms**

Load the 4B draft offline, force `TRITON_MLA`, compare greedy target tokens, validate block
rejection, measure draft memory delta, and run sanitizer on non-causal MLA kernels.

```python
T = TypeVar("T")
TARGET_AUX_LAYER_IDS = (2, 23, 47, 71, 89)
DSPARK_BLOCK_SIZE = 7

def accepted_state_prefix(states: Sequence[T], accepted: int) -> Sequence[T]:
    if not 0 <= accepted <= DSPARK_BLOCK_SIZE:
        raise ValueError(f"accepted must be in [0, 7], got {accepted}")
    return states[:accepted]
```
- [ ] **Step 4: Run host contract and A100 command**

Host:

```bash
python3 -m unittest tests.kernel.test_k10_contract -v
python3 kernel-kit/k10_k3_dspark_sm80/harness.py --cpu-dry-run
```

A100:

```bash
python3 kernel-kit/k10_k3_dspark_sm80/harness.py \
  --target moonshotai/Kimi-K3 --draft Inferact/Kimi-K3-DSpark
```

Expected before lanes 03/05: k10 all gates PASS.

- [ ] **Step 5: Commit**

```bash
git add kernel-kit tests/kernel
git commit -m "test(kernel): add K3 DSpark SM80 gate"
```

### Task 10: k09 bit-exact FP8 reference and Triton standalone kernel

**Files:**
- Create: `kernel-kit/common/fp8_ref.py`
- Create: `kernel-kit/vendor/mla_fp8_kv_sm80.py`
- Create: `kernel-kit/k09_mla_fp8_kv_sm80/harness.py`
- Modify: `kernel-kit/common/shapes.py`
- Create: `tests/kernel/test_fp8_ref.py`
- Create: `tests/kernel/test_k09_contract.py`

**Interfaces:**
- Produces: `encode_e4m3fn(x: float) -> int`
- Produces: `decode_e4m3fn(bits: int) -> float`
- Produces: `pack_reference(values: Sequence[float], dst: memoryview, scale: float) -> None`
- Produces Triton helper: `e4m3_bits_to_fp32(bits: tl.tensor) -> tl.tensor`
- Produces Triton wrappers: `pack_bf16_to_e4m3(src, dst, scale)` and `load_e4m3_to_bf16(src, scale, dst)`
- Produces report id: `k09_mla_fp8_kv_sm80`

- [ ] **Step 1: Write exhaustive failing FP8 tests**

Check all 256 byte patterns against PyTorch float8 conversion on CPU where representable, explicit
NaN/saturation policy, round-to-nearest-even ties, per-tensor scale, and page-boundary fixtures.

```python
def test_decode_is_defined_for_all_bytes(self):
    values = [decode_e4m3fn(code) for code in range(256)]
    self.assertEqual(len(values), 256)

def test_encode_roundtrip_for_all_finite_codes(self):
    for code in range(256):
        value = decode_e4m3fn(code)
        if math.isfinite(value):
            self.assertEqual(decode_e4m3fn(encode_e4m3fn(value)), value)

def test_page_boundary_does_not_touch_canary(self):
    page = bytearray([0xAA] + [0x00] * 576 + [0x55])
    pack_reference([0.0] * 576, memoryview(page)[1:-1], scale=1.0)
    self.assertEqual((page[0], page[-1]), (0xAA, 0x55))
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m unittest tests.kernel.test_fp8_ref tests.kernel.test_k09_contract -v`

Expected: FAIL because reference and kernel do not exist.

- [ ] **Step 3: Implement reference, Triton byte-storage kernels, and harness**

Do not invoke native FP8 matmul. Store E4M3 bytes and dequantize to BF16 on load. Cover contiguous
and non-contiguous block tables, partial final block, 8k/256k/1M logical lengths, canaries,
sanitizer, and measured HBM footprint.

```python
@triton.jit
def e4m3_bits_to_fp32(bits):
    sign = (bits >> 7) & 1
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    subnormal = mantissa.to(tl.float32) * (2.0 ** -9)
    normal = (1.0 + mantissa.to(tl.float32) / 8.0) * tl.exp2(
        exponent.to(tl.float32) - 7.0
    )
    value = tl.where(exponent == 0, subnormal, normal)
    value = tl.where(sign == 0, value, -value)
    return tl.where((exponent == 0xF) & (mantissa == 0x7), float("nan"), value)

@triton.jit
def load_e4m3_to_bf16_kernel(src, scale, dst, n_elements: tl.constexpr):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < n_elements
    bits = tl.load(src + offsets, mask=mask, other=0).to(tl.uint8)
    values = e4m3_bits_to_fp32(bits) * tl.load(scale)
    tl.store(dst + offsets, values.to(tl.bfloat16), mask=mask)
```
- [ ] **Step 4: Run CPU/dry tests and A100 gate**

Host:

```bash
python3 -m unittest tests.kernel.test_fp8_ref tests.kernel.test_k09_contract -v
python3 kernel-kit/k09_mla_fp8_kv_sm80/harness.py --cpu-dry-run
```

A100:

```bash
python3 kernel-kit/k09_mla_fp8_kv_sm80/harness.py \
  --report-dir /mnt/nas/home/ubuntu/service/large_model/Kimi-K3/runs/gates/k09
```

Expected before lanes 04/05: numeric, shapes, sanitizer, and memory gates PASS.

- [ ] **Step 5: Commit**

```bash
git add kernel-kit tests/kernel
git commit -m "feat(kernel): add SM80 software FP8 MLA cache"
```

### Task 11: k09 vLLM backend integration

**Files:**
- Create: `docker/patches/mla-fp8-kv-sm80.patch`
- Modify: `docker/Dockerfile.openai`
- Modify: `docker/verify_image.py`
- Create: `tests/docker/test_fp8_patch_contract.py`

**Interfaces:**
- Consumes: standalone k09 conversion semantics
- Produces: K3 `TRITON_MLA` FP8 KV support on SM80 only when `VLLM_K3_SM80_FP8_KV=1`

- [ ] **Step 1: Write failing source-patch tests**

Assert the patch removes no SM89 safety globally, gates only K3 SM80, uses byte storage, and falls
back to the original code when the opt-in is absent.

```python
def test_fp8_patch_is_k3_sm80_opt_in(self):
    patch = Path("docker/patches/mla-fp8-kv-sm80.patch").read_text()
    self.assertIn("VLLM_K3_SM80_FP8_KV", patch)
    self.assertIn("uint8", patch)
    self.assertIn("Kimi-K3", patch)
    self.assertNotIn("-    if self.kv_cache_dtype.startswith(\"fp8\")", patch)
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m unittest tests.docker.test_fp8_patch_contract -v`

Expected: FAIL because the integration patch is absent.

- [ ] **Step 3: Implement opt-in K3 MLA cache insert/load integration**

Reuse k09 conversion logic in pinned vLLM Triton kernels. Preserve BF16 queries and BF16
accumulation. Emit a startup banner with backend, storage dtype, conversion mode, and patch hash.

```python
use_k3_sm80_fp8 = (
    model_config.hf_text_config.model_type == "kimi_k3"
    and current_platform.is_device_capability(80)
    and envs.VLLM_K3_SM80_FP8_KV
)
if kv_cache_dtype.startswith("fp8") and not (
    current_platform.has_device_capability(89) or use_k3_sm80_fp8
):
    raise ValueError("FP8 KV cache requires SM89+ or the K3 SM80 software path")
```
- [ ] **Step 4: Run static/image tests and A100 model gate**

Host:

```bash
python3 -m unittest tests.docker.test_fp8_patch_contract -v
python3 docker/verify_image.py k3-sm80-openai:0498dc7e-sm80
```

A100: run lane 04 `make gate-load` at 256k, 512k, 768k, and 1M.

Expected: host contract PASS; A100 1M completes with at least 5% free HBM.

- [ ] **Step 5: Commit**

```bash
git add docker tests/docker
git commit -m "feat(vllm): integrate SM80 FP8 KV for K3 MLA"
```

### Task 12: Lane Runbooks and gate dependency wiring

**Files:**
- Modify: `deploy/tooling/templates/RUNBOOK.md.tmpl`
- Modify: `deploy/tooling/templates/Makefile.tmpl`
- Modify: `deploy/tooling/templates/preflight.py.tmpl`
- Create: `tests/deploy/test_runbooks.py`
- Create: `tests/deploy/test_gate_matrix.py`

**Interfaces:**
- Produces lane gate matrix:
  - 01: `k08,kda,pp`
  - 02: `k08,c01,ep`
  - 03: `k08,c01,k10,ep,dspark`
  - 04: `k08,c01,k09,ep,1m`
  - 05: `k08,c01,k09,k10,ep,1m,dspark`
- Produces: `GATE_MATRIX: dict[str, tuple[str, ...]]`

- [ ] **Step 1: Write failing Runbook/matrix tests**

Assert every Runbook contains preparation, preflight, exact commands, stable error keys, maximum
20-line relay, down/cleanup, and next-lane promotion. Assert Make dependencies match the matrix.

```python
EXPECTED = {
    "01": ("k08", "kda", "pp"),
    "02": ("k08", "c01", "ep"),
    "03": ("k08", "c01", "k10", "ep", "dspark"),
    "04": ("k08", "c01", "k09", "ep", "1m"),
    "05": ("k08", "c01", "k09", "k10", "ep", "1m", "dspark"),
}

def test_gate_matrix_is_exact(self):
    self.assertEqual(GATE_MATRIX, EXPECTED)

def test_runbooks_have_operator_contract(self):
    for path in Path("deploy").glob("0*-*/RUNBOOK.md"):
        text = path.read_text()
        for heading in ("Preflight", "실행", "오류 코드", "정리", "승격"):
            self.assertIn(heading, text, path)
        self.assertIn("최대 20줄", text)
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m unittest tests.deploy.test_runbooks tests.deploy.test_gate_matrix -v`

Expected: FAIL until templates and Make targets are complete.

- [ ] **Step 3: Implement lane-specific Runbooks and digest-bound dependencies**

Document user-visible outputs instead of tarball return. Add `make gates`, `gate-load`,
`gate-context`, `gate-dspark`, and `calibrate`; reject stale image/env/repository stamps.

```python
GATE_MATRIX = {
    "01": ("k08", "kda", "pp"),
    "02": ("k08", "c01", "ep"),
    "03": ("k08", "c01", "k10", "ep", "dspark"),
    "04": ("k08", "c01", "k09", "ep", "1m"),
    "05": ("k08", "c01", "k09", "k10", "ep", "1m", "dspark"),
}
```
- [ ] **Step 4: Regenerate and run all host verification**

Run:

```bash
python3 -m deploy.tooling.render_lanes
python3 -m deploy.tooling.render_lanes --check
python3 -m deploy.tooling.validate_lanes
python3 -m unittest discover -s tests -v
for f in deploy/0*-*/scripts/*.sh docker/*.sh kernel-kit/*/launch.sh; do bash -n "$f"; done
for d in deploy/0*-*; do (cd "$d" && docker compose --env-file .env config --quiet); done
```

Expected: all host tests PASS, generated drift 0, exactly five env files, no GPU PASS claims.

- [ ] **Step 5: Commit**

```bash
git add deploy tests
git commit -m "docs(deploy): add executable runbooks and gate matrix"
```

### Task 13: A100 execution sequence and final evidence

**Files:**
- Modify: `SUMMARY.md`
- Create: `docs/A100-5LANE-RESULTS.md`

**Interfaces:**
- Consumes: lane reports and stamps from Tasks 6–12
- Produces: one PASS/FAIL/INCOMPLETE row per lane with exact blocking gate

- [ ] **Step 1: Add an initially INCOMPLETE evidence table**

Create rows for k08, c01, k10, k09, and lanes 01–05. Each row must include command, image digest,
env hash, GPU/driver, result, and artifact path.

```markdown
| id | command | image digest | env hash | GPU/driver | result | artifact |
|---|---|---|---|---|---|---|
| k08 | `make -C deploy/01-tp8-pp4-256k-safe gate-k08` | not-run | not-run | not-run | INCOMPLETE | not-run |
| c01 | `make -C deploy/02-tp8-dp4-ep32-256k-safe gate-c01` | not-run | not-run | not-run | INCOMPLETE | not-run |
| k10 | `make -C deploy/03-tp8-dp4-ep32-256k-dspark gate-k10` | not-run | not-run | not-run | INCOMPLETE | not-run |
| k09 | `make -C deploy/04-tp8-dp4-ep32-1m-fp8kv gate-k09` | not-run | not-run | not-run | INCOMPLETE | not-run |
| lane01 | `make -C deploy/01-tp8-pp4-256k-safe smoke` | not-run | not-run | not-run | INCOMPLETE | not-run |
| lane02 | `make -C deploy/02-tp8-dp4-ep32-256k-safe smoke` | not-run | not-run | not-run | INCOMPLETE | not-run |
| lane03 | `make -C deploy/03-tp8-dp4-ep32-256k-dspark smoke` | not-run | not-run | not-run | INCOMPLETE | not-run |
| lane04 | `make -C deploy/04-tp8-dp4-ep32-1m-fp8kv smoke` | not-run | not-run | not-run | INCOMPLETE | not-run |
| lane05 | `make -C deploy/05-tp8-dp4-ep32-1m-fp8kv-dspark smoke` | not-run | not-run | not-run | INCOMPLETE | not-run |
```

- [ ] **Step 2: Run host evidence validator**

Run: `python3 -m deploy.tooling.validate_lanes --evidence docs/A100-5LANE-RESULTS.md`

Expected: exits 0 while reporting all GPU rows INCOMPLETE, never PASS.

- [ ] **Step 3: Execute gates and lanes on A100 in dependency order**

Run:

```bash
make -C deploy/01-tp8-pp4-256k-safe gates up smoke status down
make -C deploy/02-tp8-dp4-ep32-256k-safe gates up smoke status down
make -C deploy/03-tp8-dp4-ep32-256k-dspark gates up smoke status down
make -C deploy/04-tp8-dp4-ep32-1m-fp8kv gates up calibrate smoke status down
make -C deploy/05-tp8-dp4-ep32-1m-fp8kv-dspark gates up calibrate smoke status down
```

Stop promotion at the first FAIL but retain later rows as INCOMPLETE. Do not relabel blocked lanes
as FAIL unless their own command ran.

- [ ] **Step 4: Validate fresh evidence and cleanup**

Run:

```bash
python3 -m deploy.tooling.validate_lanes --evidence docs/A100-5LANE-RESULTS.md
for d in deploy/0*-*; do make -C "$d" status; done
nvidia-smi
```

Expected: evidence validator agrees with reports; all deployment containers/processes are absent
after down; no stale GPU allocations remain.

- [ ] **Step 5: Commit final evidence**

```bash
git add SUMMARY.md docs/A100-5LANE-RESULTS.md
git commit -m "docs: record five-lane A100 serving results"
```

---

## Final Verification

Run on the development host:

```bash
python3 -m deploy.tooling.render_lanes --check
python3 -m deploy.tooling.validate_lanes
python3 -m unittest discover -s tests -v
bash kernel-kit/run_all.sh --cpu-dry-run
git diff --check
git status --short
```

Required result:

- renderer drift 0;
- five lanes and exactly five `.env` files;
- all static/unit/shell/Compose checks PASS;
- GPU-only gates INCOMPLETE on the development host;
- clean working tree after intentional commits.

Run on A100:

```bash
python3 -m deploy.tooling.validate_lanes --evidence docs/A100-5LANE-RESULTS.md
```

Only this evidence can promote k08/k09/k10/c01 or lanes 01–05 to PASS.
