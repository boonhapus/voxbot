---
name: code-rules
description: Contains core principles that allow us to write maintainable code. Typed public APIs, small functions, shallow nesting, isolated I/O, explicit errors, tested contracts, minimal globals, no speculative abstraction, and one reliable quality gate.
---
# Code Rules

## Core Principle

Write code that is boring, explicit, typed, testable, and easy to delete.
Prefer local clarity over clever abstraction.

## Quality Gate

1. Every repository should define one command that runs formatting, linting, type checking, and tests.
2. Code is not done because it runs locally. It is done when the quality gate passes.
3. New behavior needs tests. A passing smoke test is not a test suite.
4. Tool suppressions like `# type: ignore`, `# noqa`, or linter ignores must be rare and explained.

## Code Style

1. Use consistent formatting from the repository formatter. Do not hand-format around it.
2. Keep functions small: roughly 50 lines max unless there is a strong reason.
3. Keep nesting shallow. Prefer early returns over nested `if`/`else` blocks.
4. Avoid long parameter lists. If a function needs many inputs, use a small config or model object.
5. Public functions should have complete type annotations, including return types.
6. Avoid positional booleans in public APIs. Prefer keyword-only options:

```python
def render_report(*, include_metadata: bool = False) -> str:
    ...
```

7. Prefer named constants for repeated literals when they carry domain meaning.
8. Use comments only to explain non-obvious decisions, not to narrate obvious code.
9. Avoid `Any` unless crossing an untyped external boundary. Convert unknown data into typed structures quickly.
10. Use `Enum`, `Literal`, Pydantic models, dataclasses, or typed aliases when they make valid states clearer.

## Code Semantics

1. A function should do one conceptual thing.
2. Function names should describe the semantic operation, not the implementation detail.
3. Avoid hidden side effects. If a function writes to disk, mutates state, sends a request, or updates global state, that should be obvious from its name or placement.
4. Separate pure logic from I/O. Parsing, validation, scoring, formatting, and transformation should be easy to test without network, files, databases, or framework objects.
5. Treat persistence shapes as contracts. JSON formats, database schemas, cache keys, message payloads, and external API models need tests when changed.
6. Preserve invariants close to the data model. Validation belongs in the model or constructor path, not scattered across callers.
7. Use explicit domain errors. Do not make callers inspect strings.
8. Chain exceptions when translating layers:

```python
try:
    ...
except ExternalError as exc:
    raise StorageError("failed to load records") from exc
```

9. Do not swallow exceptions silently. If failure is acceptable, log it or return a typed result that makes the failure visible.
10. Broad `except Exception` blocks should exist only at process, job, request, or task boundaries.

## Module Design

1. A module should have one clear reason to exist.
2. Avoid god files that mix models, storage, networking, formatting, and orchestration.
3. Keep framework glue thin. Route inputs to services; keep business logic elsewhere.
4. Keep dependency direction clean: low-level utilities should not import high-level application modules.
5. Avoid import-time work beyond defining constants, types, and lightweight objects.
6. Use module-level singletons sparingly. They are convenient but make tests and lifecycle management harder.

## Async and I/O

1. Do not block an event loop with file, network, subprocess, or CPU-heavy work.
2. Use timeouts for external calls.
3. Retry only operations that are safe to retry.
4. Keep retry logic separate from parsing and business logic.
5. Isolate external SDKs behind small wrapper classes or functions.

## Testing Rules

1. Add tests around every bug fix.
2. Prefer focused tests for pure logic first.
3. Mock external systems at the boundary, not deep inside business logic.
4. Test invalid inputs, empty inputs, and boundary cases.
5. Storage backends that implement the same interface should share behavioral tests.
6. If code is hard to test, that is usually a design signal.

## Maintainability Rules

1. Do not add abstraction for one caller.
2. Do not refactor unrelated code while making a feature change.
3. Prefer deleting code over preserving unused flexibility.
4. Keep error messages useful to developers, but sanitize anything user-facing.
5. Make state transitions explicit.
6. Choose readability over cleverness, especially in code that handles persistence, concurrency, money, security, or external APIs.
