# Acme Robotics — Engineering Handbook

## Code Review
Every pull request requires at least two approvals before it can be merged into the main branch. At least one approval must come from a designated code owner of the affected module. Pull requests must be smaller than 400 lines of changed code; larger changes should be split into a stack of smaller PRs. The continuous integration suite must pass before any merge.

## Deployment
Acme Robotics deploys to production using a blue-green strategy. All deployments to production must occur between 10:00 and 16:00 on business days — a policy known internally as the "daylight deploy" rule — so that engineers are available to respond to issues. Friday afternoon deployments after 14:00 are frozen. Every production deployment must be accompanied by a rollback plan documented in the release ticket.

## On-Call
Engineering teams operate a weekly on-call rotation. The on-call engineer must acknowledge any critical page within 10 minutes. On-call shifts run Monday to Monday and are compensated with an additional day of leave per completed week. No engineer may be on call for more than one week in any four-week period.

## Technology Standards
Backend services are written in Go and Python. The standard database is PostgreSQL; new services must not introduce a new database technology without Architecture Review Board approval. All services must expose a /health endpoint and emit structured JSON logs. The robot fleet firmware is written in Rust.
