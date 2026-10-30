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
- At the initial catalog handoff, no release had been published and no deployment
  had been changed.
- Catalog PR #104 passed Linux/Windows Python 3.11/3.12 CI, Security and Docs.
  The local full suite passed 1,995 tests (3 existing skips, 5 expected failures,
  109 subtests); Ruff, mypy and the candidate wheel build passed.
- off-key now uses the candidate wheel in its isolated environment. Resolution
  reuse, shared field metadata, generated contracts, the shared editor and
  structured diagnostics are implemented; candidate integration and migration
  gates remain in progress. The browser production build passes.
- The user explicitly approved the library merges and publication. Dependency
  PRs #95, #96, #97, #99, #100, #101 and #102 landed with catalog PR #104. GitHub
  closed transaction PR #103 when its dependency base branch was removed; its
  complete commit is included in #104. PR #105 aligns the documentation example
  with the merged Ruff version. Linux/Windows Python 3.11/3.12, core/extras,
  integration, Security and Docs checks passed.
- ABERRANT 1.4.0 was published from `c61114f368400da910d0187925ff55e8ad6d454a`.
  [Release workflow](https://github.com/OliverHennhoefer/aberrant/actions/runs/41053480790)
  succeeded. The candidate wheel had already passed 206 real off-key RADAR tests
  with zero skips, including database recovery, MQTT and Swarm lifecycle checks.
- The coordinated platform implementation is
  [off-key PR #384](https://github.com/DataDrivenSustainabilitySolutions/off-key/pull/384).
  It installs the published exact pin, frozen lock and regenerated offline catalog.
  Its first complete CI run passed backend 3.12/3.14, frontend, generated contracts,
  all production images, collection and RADAR. Infrastructure documentation
  navigation and rehearsal evidence fixes are included in the updated run.
- A checkpoint captured from off-key's actual 1.3.0 production image verifies
  authentication followed by rejection before deserialization. Saved launch
  normalization preserves its source configuration; the new detector performs
  fresh warm-up and calibration. Historical records and predecessor links remain
  governed by the existing real database/release tests.
- Backend readiness now fences activation of a changed browser bundle. The
  binary rollback reference is previous platform code with reviewed 1.4.0
  numerics at `9c010bda9831648f1db8c395dc0633f1302df757`; strict migration,
  catalog and numerical-version guards remain enabled. A numerical rollback to
  1.3.0 is outside this reviewed binary rollback floor.
- Required production browser, staged rollout and binary rollback gates are
  running. Current evidence and migration details continue in
  [off-key's execution record](https://github.com/DataDrivenSustainabilitySolutions/off-key/blob/codex/aberrant-configuration-integration/docs/development/aberrant-1.4-integration.md).
  No production deployment has been changed.

## Deliberately deferred

Standalone `Self` sweeps, `StrEnum` conversion, file splitting alone, and a public
exception-hierarchy redesign. No new configuration language or default changes.
