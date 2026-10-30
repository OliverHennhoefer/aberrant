# ABERRANT 1.4 and off-key configuration integration

This execution plan tracks the approved change across
`OliverHennhoefer/aberrant` and `DataDrivenSustainabilitySolutions/off-key`.
ABERRANT defines algorithm contracts. off-key owns model selection, presets,
resource ceilings, feature binding, calibration, and deployment policy. The
offline catalog remains the control plane's dependency boundary.

## Delivery sequence

1. Start from the reviewed Python 3.11 dependency baseline (ABERRANT PR #95,
   commit `7b878c8a36abf8350bb4afecf0597c90f28edad6`). Keep the feature changes
   independently reviewable and preserve the remaining dependency PRs.
2. Deliver a transaction-capability PR: shared learner protocol, compatibility
   specialization, both engines, success/failure/cancellation coverage.
3. Deliver a catalog PR: per-operation parameter descriptors, resolved detector
   configuration, structured issues, evaluator reuse, compatibility tests and
   documentation. Target release version 1.4.0.
4. Deliver one coordinated off-key PR: RADAR resolution reuse, field metadata,
   generated contracts, shared controls, structured errors, migration and docs.
5. Test off-key with the candidate wheel before publication. After library CI
   and the candidate integration gates pass, publish 1.4.0 and update off-key's
   exact pin, frozen lockfile and generated catalog against the published wheel.
6. Execute deployment rehearsal and rollback gates. Serve new backend metadata
   before activating the browser bundle. Preserve historical records and
   predecessor-run links; upgraded runtimes relearn and recalibrate rather than
   load incompatible checkpoints.

## Contracts and acceptance

- Resolve each component occurrence once, share inspection only within an
  operation, and construct independent detectors on repeated builds. Do not
  introduce a process-wide cache or cache learned objects.
- Keep normalization/fingerprinting validation boundaries, canonical JSON and
  fingerprint rules, defaults, IDs, configuration format, optional-import
  isolation and detached public metadata.
- Preserve tuple/list conversion, integer mapping keys, unions, nulls, nested
  dependency paths and independent field diagnostics. Constructor-only
  invariants attach to the parameter block.
- Preserve eager linear learning and staged graph readiness. Test terminal
  rollback, downstream failure, rollback failure, nontransactional learners and
  cancellation without claiming atomicity for nontransactional state.
- off-key fields distinguish omission, explicit null and defaults, including
  exclusive bounds. Both editors support numeric pairs/lists, adjacency and
  nested FAISS parameters with the same controls and field errors.
- Carry structured JSON Schema, Pydantic and ABERRANT locations through existing
  HTTP 422 details and bounded pipeline reports. No parsing messages for paths.
- Keep presentation metadata out of executable fingerprints and checkpoints.
  Review deployed 1.3.0 launch normalization, upgrade and rollback explicitly.

## Verification gates

- ABERRANT: targeted compatibility and transaction tests; Ruff, mypy, full tests,
  wheel build; Linux/Windows CI on Python 3.11 and 3.12; core-only import checks.
- off-key: backend Python 3.12/3.14, lint/type/tests; catalog and TypeScript
  generation checks; frontend lint/tests/build; real RADAR and API/browser lanes
  with no skipped required tests; deployment and rollback rehearsal.
- Real flows: nested KNN, adjacency, omission/null/default, several simultaneous
  field errors, correction/publication, static/adaptive monitoring, desktop/mobile.

## Progress and evidence

- 2026-10-30: verified the seven reviewed dependency PRs remain open; created
  `codex/transactional-learners` on the tested Python 3.11 baseline.
- 2026-10-30: created an isolated off-key checkout on
  `codex/aberrant-configuration-integration`, preserving the existing checkout
  and local runtime data.
- Transaction capability is delivered in PR #103. Local focused tests (74),
  Ruff and mypy passed. Linux/Windows Python 3.11/3.12 CI, core/extras/integration,
  Security and Docs are green for commit `5d46f1512f3b179fce6e6c34f819147fd48da5f2`.
- Catalog resolution and structured diagnostics are implemented. Targeted tests
  verify one resolution per component, operation-local inspection reuse, fresh
  builds, unchanged core schemas and representative canonical fingerprints,
  nested FAISS diagnostics and independent engine state. Full gates in progress.
- No release has been published and no deployment has been changed.

## Deliberately deferred

Standalone `Self` sweeps, `StrEnum` conversion, file splitting alone, and a public
exception-hierarchy redesign. No new configuration language or default changes.
