# Roadmap

Features described in `deep-research-report.md` that are not yet implemented, organized by priority tier.

## Collection — Additional Sources

- [ ] **Crossref REST API collector** — DOI-level metadata, publication status (preprint → published), funding/license/ORCID/ROR fields. Free API with polite-usage rate limits (changed Dec 2025). Gives the strongest "is this published?" signal for journal papers.
- [ ] **OpenAlex collector** — Unified scholarly graph (CC0 metadata). Works/authors/institutions/sources in one API. Credits/key system; Premium tier unlocks `from_updated_date` filter. Abstracts come as inverted index (needs reconstruction to plain text).
- [ ] **OpenReview collector** — Best source for acceptance decisions at conferences hosted on OpenReview (ICLR, NeurIPS workshops, etc.). Official `openreview-py` client exists. Beware v1/v2 API differences.
- [ ] **CVF Open Access scraper** — Accepted-version PDFs for CVPR/ICCV/ECCV. Strong acceptance signal. No official API; requires HTML parsing.
- [ ] **IEEE Xplore API** — Final proceedings for IEEE conferences/journals. Requires API key application and approval. Enterprise-oriented.
- [ ] **ACL Anthology** — NLP venue metadata from public GitHub repo + Python module. Useful if topic overlaps with NLP (language-guided navigation, etc.).
- [ ] **arXiv OAI-PMH / RSS** — Alternative to the search API for bulk daily collection. RSS updates daily per category; OAI-PMH gives incremental harvest.

## Enrichment — Additional Signals

- [ ] **GitHub API enrichment** — Stars, commit frequency, issues, releases for code repos. Adds depth beyond PWC's binary "has code" signal. Requires token; rate limits apply per endpoint.
- [ ] **Unpaywall enrichment** — Canonical OA status + best OA PDF link. REST API (~100k req/day free). Data Feed (daily changefiles) available by subscription.
- [ ] **OpenAlex citation enrichment** — `cited_by_count` for citation data, especially for DOI-registered works.
- [ ] **Author affiliation scoring** — Crossref ROR/affiliations + OpenAlex institutions. Build an `AuthorCredibilityScore` with bonus for top-tier affiliations (configurable list).
- [ ] **OpenCitations** — Open citation graph as an independent citation data source. Useful for bibliometric analysis layer.

## Scoring — Advanced Relevance

- [ ] **Embedding-based relevance** — Compute title+abstract embeddings and measure cosine similarity against a seed set of known-relevant papers. More robust than keyword matching for catching novel terminology.
- [ ] **Publication status score** — Factor acceptance/publication status into quality: accepted > preprint. Sources: CVF pages, OpenReview decisions, Crossref venue field.
- [ ] **Influential citation velocity** — Track citation growth at 7/30/90 days. More useful than absolute count for fresh papers where citations ≈ 0.
- [ ] **Configurable venue prior** — Per-venue relevance prior for conferences that reliably publish AD/VLM/VLA work (e.g., CVPR perception track, CoRL).

## Scoring — Incremental Re-evaluation

- [ ] **Re-enrichment scheduler** — Periodically re-fetch venue/code for existing papers (e.g., weekly for papers < 30 days old, monthly for < 90 days). Requires tracking `last_enriched_at` and a re-enrichment CLI command/cron job.
- [ ] **Status transition tracking** — Detect and log when a paper moves from preprint → accepted → published. Store events in a `paper_events` table for audit.

## Summarization — Full-Text & Validation

- [ ] **Summary template versioning** — Store `SummaryTemplate` as a versioned object in the DB. Each generated summary is linked to a template version for auditability. Templates as Jinja2 or YAML, editable without code changes.
- [ ] **Schema validation on LLM output** — Strict JSON schema check (field types, max lengths, required fields) on every LLM response. Reject and retry on malformed output.
- [ ] **Anti-hallucination cross-checks** — If LLM claims "accepted at CVPR", verify against CVF/OpenReview/Crossref data. If LLM claims "code available", verify against PWC/GitHub. Flag unverified claims in the digest.
- [ ] **Provenance tracking** — Record which API source provided each enrichment field (e.g., "code_url from PWC 2026-02-20"). Stored per-field, queryable for debugging and trust assessment.

## Delivery — Additional Channels & Formats

- [ ] **Email delivery (SMTP)** — HTML email with clickable table of contents, top-N highlighted, full list below.
- [ ] **Slack delivery** — Post digest to a Slack channel via `chat.postMessage` Web API.
- [ ] **HTML report generation** — Rendered HTML digest with styling, anchor links, expandable summaries.
- [ ] **PDF report generation** — Archival-quality PDF from HTML template.

## Infrastructure — Production Hardening

- [ ] **PostgreSQL migration** — Replace SQLite with Postgres for concurrent access, better full-text search, and production reliability.
- [ ] **Vector index** — pgvector or dedicated vector store for embedding-based search and relevance scoring.
- [ ] **Object storage** — S3-compatible storage for downloaded PDFs/TEI XML.
- [ ] **Task queue** — Redis + Celery/RQ to parallelize network I/O and LLM calls across workers.
- [ ] **Orchestration** — Airflow or Prefect DAGs instead of flat cron. Enables retries, dependency graphs, backfill, and monitoring.
- [ ] **Centralized logging & metrics** — Structured logs, API call latency/error tracking, scoring distribution histograms.
- [ ] **Secrets management** — Vault/KMS for API keys instead of environment variables.

## Enterprise

- [ ] **RBAC / SSO / SCIM** — Role-based access, corporate SSO integration.
- [ ] **Data isolation & compliance** — Tenant isolation, retention policies, GDPR considerations.
- [ ] **On-prem LLM endpoint** — Private LLM deployment for sensitive/proprietary research contexts.

## Quality of Life

- [ ] **Zotero integration** — Export digest entries to a Zotero collection via Web API.
- [ ] **Hugging Face Hub integration** — Check for associated models/datasets on HF Hub.
- [ ] **Configurable digest frequency** — Support weekly/biweekly digests in addition to daily.
- [ ] **Interactive score tuning** — CLI or notebook for adjusting weights and previewing how rankings change.
