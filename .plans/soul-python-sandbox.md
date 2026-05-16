# Plan: Soul Python Sandbox

> **Status:** Not started.
> **Last audited:** 2026-05-16

## Goal

Give `src/voxbot/plugins/soul` a tightly scoped Python execution tool for small,
stdlib-only programs that help answer Discord user questions.

The sandbox must:

- Run untrusted model/user-derived Python outside the bot process.
- Have no network access.
- Have no access to bot secrets, source tree, release directories, Redis, Discord
  tokens, or host files.
- Support basic stdlib calculations, parsing, date math, simulations, and
  formatting.
- Return stdout, stderr, exceptions, timeout state, and duration in a predictable
  shape.
- Minimize steady-state resource drain on the Mac/Colima deployment.

The sandbox is not meant to run packages, scrape the web, persist files, spawn
services, or perform long-running computation.

---

## Recommendation

Start with **one fresh Docker container per sandbox run**, using a prebuilt slim
Python image and strict Docker runtime limits.

This should be the default because it gives the best practical security and the
lowest idle resource cost:

- No long-lived sandbox process sits around consuming memory.
- `--network=none` is simple and enforceable.
- `--rm` resets filesystem and process state after every run.
- Any orphan processes die with the container.
- The image is already local after deploy, so per-run overhead is mostly process
  startup, not image pull/build.

Add a **warm container mode** only if metrics prove container startup is too slow
or too CPU-heavy for real usage. Warm mode should still use `network_mode: none`
and communicate with `docker exec`, not HTTP, because an HTTP sandbox service
would need a network namespace and would weaken the "no network" guarantee for
the code being executed.

---

## Current Repo Context

Relevant code:

- `src/voxbot/plugins/soul/ai.py`
  - Defines the Pydantic AI `soul_agent`.
  - Existing tools include reactions, display-name changes, remember, and forget.
- `src/voxbot/plugins/soul/cog.py`
  - Routes whitelisted Discord messages through `soul_agent.run`.
  - Dispatches returned actions after the model finishes.
- `src/voxbot/plugins/soul/settings.py`
  - Existing `SOUL_`-prefixed plugin settings.
- `deploy/macos/infra/compose.yaml`
  - Docker/Colima is already part of production infrastructure.
- `deploy/macos/apps/deploy.sh`
  - Builds/tests releases before flipping the `current` symlink.
- `deploy/macos/apps/run-bot.sh`
  - Runs the bot on the host via `uv run --frozen voxbot bot`.

Important deployment shape:

- The bot runs on the host under launchd, not inside Docker.
- Colima/Docker already runs Redis and Agent Memory Server.
- The `voxbot` user is expected to have Docker CLI access on the server.

---

## Architecture

```text
Discord message
  SoulCog.on_message
    soul_agent.run(...)
      run_python tool
        DockerPythonSandbox.run(code)
          docker run --rm --network=none ...
            python -I -S /opt/voxbot-sandbox/runner.py
              parse request JSON from stdin
              AST/import preflight
              execute with safe builtins and resource limits
              emit result JSON
        formatted tool result
      model final response
    BotAIAction dispatch
```

The sandbox should be a narrow tool, not general model autonomy. The model asks
for a run, receives bounded output, and then decides how to answer the user.

---

## Runtime Mode Decision

| Mode | Pros | Cons | Recommendation |
|---|---|---|---|
| Per-job `docker run --rm` | Best reset semantics, zero idle memory, no shared state, easy timeout cleanup | Pays container startup on every run | **Default** |
| Warm `network_mode: none` container + `docker exec` | Lower per-run startup, still no container network | Shared container state, harder orphan cleanup, idle memory | Add only after metrics |
| Long-lived HTTP sandbox service | Easy job API, low per-run latency | Requires a network namespace, so user code also has container network unless extra kernel policy is added | Avoid |
| Pure Python subprocess on host | Simple and fast | Not a hard sandbox; cannot reliably block file/network access | Do not use for untrusted code |

Decision gate for warm mode:

- Keep per-job mode if p95 sandbox wall time is under 2 seconds and usage stays
  below roughly 30 runs/hour.
- Consider warm mode if p95 wall time exceeds 2 seconds due mostly to Docker
  startup, or if repeated short runs create visible CPU spikes.
- Prefer answering fewer questions with a stronger sandbox over weakening network
  isolation for convenience.

---

## New Files

```text
deploy/sandbox/Dockerfile
deploy/sandbox/runner.py
src/voxbot/plugins/soul/sandbox.py
tests/plugins/soul/test_sandbox.py
tests/plugins/soul/test_sandbox_runner.py
```

Optional if warm mode is implemented later:

```text
deploy/macos/infra/sandbox-compose.md        # notes or copied compose snippet
```

Do not mount the Voxbot repo into the sandbox container.

---

## Sandbox Image

Create `deploy/sandbox/Dockerfile`:

```dockerfile
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin sandbox

COPY runner.py /opt/voxbot-sandbox/runner.py
RUN chmod 0555 /opt/voxbot-sandbox/runner.py

USER 10001:10001
WORKDIR /tmp

ENTRYPOINT ["python", "-I", "-S", "/opt/voxbot-sandbox/runner.py"]
```

Why `python:3.14-slim`:

- Matches the project Python target.
- Keeps behavior close to normal CPython.
- Avoids Alpine/musl surprises.
- Still small enough for this use.

Image hardening notes:

- Do not install project dependencies.
- Do not copy source code except `runner.py`.
- Do not configure package indexes or credentials.
- Use `-I -S` to isolate Python and skip site initialization.
- Run as non-root.

---

## Docker Run Contract

The bot should execute something equivalent to:

```bash
docker run \
  --rm \
  --name "voxbot-sandbox-<uuid>" \
  --interactive \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=8m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 32 \
  --memory 64m \
  --memory-swap 64m \
  --cpus 0.25 \
  --ulimit nofile=32:32 \
  --ulimit fsize=1048576:1048576 \
  --log-driver none \
  --pull never \
  voxbot-python-sandbox:current
```

Input is JSON on stdin:

```json
{
  "code": "print(sum(range(10)))",
  "stdout_limit": 6000,
  "stderr_limit": 2000,
  "allowed_imports": ["calendar", "collections", "datetime", "decimal", "fractions", "functools", "itertools", "json", "math", "random", "re", "statistics", "string", "textwrap", "time", "zoneinfo"]
}
```

Output is JSON on stdout:

```json
{
  "ok": true,
  "stdout": "45\n",
  "stderr": "",
  "error": null,
  "duration_ms": 14
}
```

Timeouts are enforced by the bot process, not trusted to the container:

1. Start `docker run` with a unique container name.
2. Wait with `asyncio.wait_for`.
3. On timeout, run `docker rm -f <name>`.
4. Return a structured timeout result.

---

## Runner Behavior

`deploy/sandbox/runner.py` should be stdlib-only.

Responsibilities:

- Read request JSON from stdin.
- Reject malformed JSON.
- Enforce max code length.
- Parse code with `ast.parse`.
- Reject imports outside the allowed stdlib subset.
- Reject common foot-gun builtins and direct dangerous calls.
- Apply Linux resource limits with `resource.setrlimit`.
- Execute with a small builtin namespace.
- Capture stdout and stderr separately.
- Truncate output deterministically.
- Return JSON to original stdout.

The AST checks are **not** the security boundary. They are there to keep normal
model-generated code focused and understandable. Docker/OS isolation is the
security boundary.

Suggested allowed imports:

```python
ALLOWED_IMPORTS = {
    "calendar",
    "collections",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "json",
    "math",
    "random",
    "re",
    "statistics",
    "string",
    "textwrap",
    "time",
    "zoneinfo",
}
```

Suggested rejected imports:

```python
BLOCKED_IMPORTS = {
    "asyncio",
    "ctypes",
    "http",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "platform",
    "socket",
    "subprocess",
    "sys",
    "threading",
    "urllib",
}
```

Suggested safe builtins:

```python
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "int": int,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
```

Do not include:

- `open`
- `input`
- `eval`
- `exec`
- `compile`
- `breakpoint`
- `globals`
- `locals`
- `vars`
- `dir`
- `help`
- `__import__` except for a custom allowlisted import function

Resource limits inside `runner.py`:

```python
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
resource.setrlimit(resource.RLIMIT_AS, (48 * 1024 * 1024, 48 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
```

The Docker memory limit should be slightly above `RLIMIT_AS` so Python can start
and return a clean failure when possible.

---

## Bot Integration

Add `src/voxbot/plugins/soul/sandbox.py`.

Suggested data models:

```python
class SandboxResult(pydantic.BaseModel):
    ok: bool
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    timed_out: bool = False
    exit_code: int | None = None
    duration_ms: int


class DockerPythonSandbox:
    async def run(self, code: str) -> SandboxResult:
        ...
```

Implementation notes:

- Use `asyncio.create_subprocess_exec`; do not shell out through `sh -c`.
- Pass request JSON via stdin.
- Capture stdout/stderr from the Docker CLI process.
- Use a unique container name for cleanup.
- On timeout, call `docker rm -f <container_name>` and consume the original
  process result.
- Use a module-level `asyncio.Semaphore` to cap concurrent runs.
- Treat Docker CLI failure as a normal tool failure, not a bot crash.
- Log with structlog but never log full user code by default.

Suggested settings in `src/voxbot/plugins/soul/settings.py`:

```python
python_sandbox_enabled: bool = False
python_sandbox_mode: Literal["per_run", "warm_exec"] = "per_run"
python_sandbox_image: str = "voxbot-python-sandbox:current"
python_sandbox_timeout_seconds: float = 2.0
python_sandbox_max_code_chars: int = 4000
python_sandbox_stdout_chars: int = 6000
python_sandbox_stderr_chars: int = 2000
python_sandbox_max_concurrency: int = 1
python_sandbox_memory: str = "64m"
python_sandbox_cpus: str = "0.25"
```

Keep the feature disabled until the image exists and local tests pass.

---

## Pydantic AI Tool

Add the tool in `src/voxbot/plugins/soul/ai.py`:

```python
@soul_agent.tool
async def run_python(ctx: RunContext[DiscordDeps], code: str, purpose: str = "") -> str:
    """
    Run a small stdlib-only Python program in a locked-down sandbox.

    Use this only for deterministic calculations, data transformations,
    date/time reasoning, small simulations, or checking simple Python behavior.
    Do not use it for network access, filesystem work, package installation,
    secrets, Discord operations, or long-running tasks.
    """
```

The tool should:

- Check `soul_settings.python_sandbox_enabled`.
- Reject empty or oversized code before Docker.
- Call `DockerPythonSandbox.run`.
- Return a concise text summary to the model:
  - success/failure
  - stdout
  - stderr if present
  - exception if present
  - timeout if present
  - elapsed time

Tool output should be capped lower than the raw runner output so the model does
not flood context.

Suggested prompt guidance in `prompts/personality.mdc`:

- Use Python only when it materially improves correctness.
- Prefer mental math for trivial questions.
- Never claim the sandbox has internet or package access.
- Explain results in normal prose instead of dumping code unless the user asked
  for code.

---

## Deployment Changes

### Build image during deploy

Update `deploy/macos/apps/deploy.sh` after `uv run --frozen pytest` succeeds:

```bash
docker build \
  -t "voxbot-python-sandbox:$SHA" \
  -t "voxbot-python-sandbox:current" \
  -f "$REL/deploy/sandbox/Dockerfile" \
  "$REL/deploy/sandbox"
```

Reason:

- The bot should not pull images during runtime.
- `--pull never` in the run command prevents surprise network use.
- Tagging by SHA makes rollback/debugging easier.
- `:current` keeps runtime configuration simple.

### Rollback behavior

On deployment rollback, either:

- retag the previous release's sandbox image as `voxbot-python-sandbox:current`,
  or
- configure the bot with `VOXBOT_RELEASE_SHA` and have it use
  `voxbot-python-sandbox:<sha>`.

Preferred: use the release SHA tag in bot settings at startup.

Example:

```python
python_sandbox_image: str = f"voxbot-python-sandbox:{settings.voxbot_release_sha or 'current'}"
```

If that introduces import-order awkwardness between global settings and soul
settings, keep `:current` for v1 and add retag-on-rollback later.

### Server bootstrap

Update `deploy/macos/README.md` with:

- Confirm `voxbot` user can run `docker ps`.
- Build the sandbox image manually once during bootstrap if needed.
- Set `SOUL_PYTHON_SANDBOX_ENABLED=true` only after the image build succeeds.

---

## Warm Container Mode

Do not implement this until per-run mode has metrics.

If needed, add a compose-managed sandbox container:

```yaml
python-sandbox:
  image: voxbot-python-sandbox:current
  restart: unless-stopped
  network_mode: "none"
  read_only: true
  tmpfs:
    - /tmp:rw,nosuid,nodev,noexec,size=8m
  cap_drop:
    - ALL
  security_opt:
    - no-new-privileges
  pids_limit: 32
  mem_limit: 64m
  cpus: 0.25
  command: ["sleep", "infinity"]
```

The bot would run:

```bash
docker exec -i voxbot-python-sandbox python -I -S /opt/voxbot-sandbox/runner.py
```

Warm mode rules:

- Keep `network_mode: none`.
- Keep `python_sandbox_max_concurrency=1`.
- Restart the warm container after every timeout.
- Restart the warm container after N successful jobs, for example 100.
- Restart on malformed runner output.
- Never switch to HTTP unless a separate kernel-level network block is added for
  the child code.

Warm mode tradeoff:

- It reduces container startup latency.
- It weakens per-job reset semantics.
- It consumes idle memory.
- It is an optimization, not the default design.

---

## Safety Boundaries

Hard boundaries:

- Docker network namespace: `--network none`.
- No host mounts.
- Read-only root filesystem.
- Tiny tmpfs.
- Non-root user.
- Dropped capabilities.
- No new privileges.
- CPU, memory, PID, file-size, and file-descriptor limits.
- Bot-side timeout and forced container removal.

Soft controls:

- AST import allowlist.
- Safe builtin namespace.
- Code length limit.
- Output length limit.
- Tool prompt guidance.
- Concurrency limit.

Threats this should handle:

- Prompt injection asking to read files.
- Code trying to import networking modules.
- Infinite loops.
- Huge allocations.
- Fork/process storms.
- Large output spam.
- Attempts to inspect environment variables.
- Attempts to persist files for the next run.

Threats this does not fully solve:

- Docker/container runtime vulnerabilities.
- Linux kernel vulnerabilities.
- Side channels.
- Abuse by a user allowed to trigger many runs unless rate limits are added.

---

## Rate Limiting

Add simple bot-side limits before enabling broadly:

- Global sandbox concurrency: `1`.
- Per-user cooldown: one run every 30 seconds.
- Per-channel cooldown: one run every 10 seconds.
- Max executions per agent response: rely on Pydantic AI tool-call limits if
  available; otherwise track in deps or wrapper state.

If Pydantic AI exposes model/tool usage limits in the current installed version,
configure the soul agent to allow at most one or two `run_python` calls per user
message.

---

## Observability

Log structured events:

- `python_sandbox_started`
  - container name
  - mode
  - code length
- `python_sandbox_completed`
  - duration
  - exit code
  - stdout length
  - stderr length
- `python_sandbox_failed`
  - duration
  - error kind
- `python_sandbox_timeout`
  - duration
  - cleanup success

Do not log full code or full output by default. If debugging needs code capture,
gate it behind an explicit owner-only debug setting.

Future admin health surface:

- Add sandbox enabled/mode/image to `/admin health`.
- Add recent sandbox failure count if Redis health storage grows a generic stats
  mechanism.

---

## Tests

### Runner tests

Test `deploy/sandbox/runner.py` directly without Docker:

- `print(2 + 2)` returns ok with stdout.
- Allowed imports work, e.g. `math`, `statistics`, `datetime`.
- Blocked imports return a clean error, e.g. `socket`, `subprocess`, `os`.
- Runtime exceptions return a clean error.
- stdout truncates.
- stderr truncates.
- Oversized code is rejected.
- Infinite loop is covered by Docker/integration tests, not direct runner tests.

### Bot wrapper tests

Mock `asyncio.create_subprocess_exec`:

- Docker command includes `--network none`.
- Docker command includes `--read-only`.
- Docker command includes `--cap-drop ALL`.
- Docker command includes memory, CPU, PID, and tmpfs limits.
- Request JSON is sent on stdin.
- Valid runner JSON becomes `SandboxResult`.
- Invalid runner JSON becomes a failure result.
- Timeout calls `docker rm -f <container_name>`.

### Integration tests

Mark Docker tests separately so normal CI can skip if Docker is unavailable:

```python
pytestmark = pytest.mark.docker
```

Cases:

- Basic calculation succeeds.
- Network attempt fails:
  - `socket.create_connection(("example.com", 80), timeout=1)`
- File read attempt fails or cannot access host:
  - `open("/etc/passwd").read()` should fail if `open` is unavailable.
- Infinite loop times out and container is removed.
- Memory bomb fails without destabilizing test runner.

Update pytest config if needed so Docker tests are opt-in locally or skipped in
GitHub Actions when Docker is not available.

---

## Implementation Phases

### Phase 1 - Runner and Image

- Add `deploy/sandbox/Dockerfile`.
- Add `deploy/sandbox/runner.py`.
- Add direct runner unit tests.
- Build the image locally:

```bash
docker build -t voxbot-python-sandbox:current -f deploy/sandbox/Dockerfile deploy/sandbox
```

Success criteria:

- Runner tests pass.
- Docker image builds.
- Manual `docker run` can execute `print(2 + 2)`.
- Manual network attempt cannot reach the internet.

### Phase 2 - Bot Executor

- Add `src/voxbot/plugins/soul/sandbox.py`.
- Add `SoulSettings` fields.
- Implement per-run Docker executor.
- Add mocked executor tests.

Success criteria:

- Executor builds the expected Docker command.
- Timeout cleanup is tested.
- Docker CLI failures do not crash the bot process.

### Phase 3 - Soul Tool

- Add `run_python` Pydantic AI tool.
- Add prompt guidance.
- Format sandbox results for the model.
- Keep feature disabled by default.

Success criteria:

- With `SOUL_PYTHON_SANDBOX_ENABLED=false`, tool returns unavailable.
- With the setting enabled and image present, the tool returns calculation output.
- The model can use the tool once and answer with a normal Discord text action.

### Phase 4 - Deployment

- Build/tag the image in `deploy/macos/apps/deploy.sh`.
- Document bootstrap and enabling in `deploy/macos/README.md`.
- Decide whether to use `:current` or release-SHA image tags.

Success criteria:

- Deploy builds the sandbox image before flipping `current`.
- Rollback leaves the bot with a usable sandbox image.
- The production `voxbot` user can run the Docker sandbox command.

### Phase 5 - Hardening and Metrics

- Add rate limits.
- Add structured sandbox logs.
- Add Docker integration tests.
- Run a small real-world trial in one whitelisted channel.

Success criteria:

- No secrets or host paths appear in sandbox output.
- Infinite loops time out reliably.
- Repeated sandbox runs do not leave containers behind.
- Latency and CPU usage are known well enough to decide on warm mode.

### Phase 6 - Optional Warm Mode

Only start this phase if measured per-run overhead is a real problem.

- Add compose service or launchd-managed `network_mode: none` container.
- Add `warm_exec` executor path.
- Restart warm container on timeout or malformed output.
- Compare p50/p95 latency and CPU against per-run mode.

Success criteria:

- Warm mode preserves no-network behavior.
- Warm mode is materially faster under real load.
- Warm mode does not leave orphan processes after timeouts.
- Per-run remains the default fallback.

---

## Open Questions

1. Should the tool be owner-only at first, or enabled in all whitelisted `soul`
   channels?

   Recommended answer: owner-only or one test channel first, then expand.

2. Should code be shown back to users?

   Recommended answer: only when the user asks. The normal response should
   explain the result, not dump tool internals.

3. Should sandbox failures be visible to users?

   Recommended answer: summarize briefly, e.g. "I tried a quick Python check,
   but it timed out." Do not expose Docker errors directly.

4. Should the sandbox allow randomness?

   Recommended answer: allow `random` for simulations, but instruct the model to
   set a seed when reproducibility matters.

5. Should the sandbox allow file writes inside `/tmp`?

   Recommended answer: not in v1. Most useful snippets can stay in memory. If
   later needed, keep `/tmp` tiny and per-run only.

---

## Non-Goals

- Running arbitrary packages.
- Installing dependencies at runtime.
- Giving the model a shell.
- Giving the model Docker access.
- Giving code access to Discord, Redis, memory server, or the Voxbot source tree.
- Replacing normal reasoning with Python for trivial questions.
- Supporting long-running jobs.

---

## Practical Default Values

Initial settings:

```env
SOUL_PYTHON_SANDBOX_ENABLED=false
SOUL_PYTHON_SANDBOX_MODE=per_run
SOUL_PYTHON_SANDBOX_IMAGE=voxbot-python-sandbox:current
SOUL_PYTHON_SANDBOX_TIMEOUT_SECONDS=2
SOUL_PYTHON_SANDBOX_MAX_CODE_CHARS=4000
SOUL_PYTHON_SANDBOX_STDOUT_CHARS=6000
SOUL_PYTHON_SANDBOX_STDERR_CHARS=2000
SOUL_PYTHON_SANDBOX_MAX_CONCURRENCY=1
SOUL_PYTHON_SANDBOX_MEMORY=64m
SOUL_PYTHON_SANDBOX_CPUS=0.25
```

Initial Docker limits:

```text
network: none
memory: 64m
swap: 64m
cpus: 0.25
pids: 32
tmpfs: /tmp size 8m, noexec, nosuid, nodev
rootfs: read-only
capabilities: none
privilege escalation: disabled
timeout: 2s
concurrency: 1
```

These are intentionally small. Increase only when a real allowed stdlib use case
fails and the security/resource tradeoff is acceptable.
