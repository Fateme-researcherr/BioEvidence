BioEvidence Review

A provenance-first evidence reasoning engine for biomedical research questions.

Given a disease, a gene/target and a therapeutic approach (for example Rett syndrome / MECP2 / gene therapy), the system:

builds a small knowledge graph of the entities from curated public databases,
retrieves candidate evidence from the literature, trial registries and target-association databases,
converts each source record into a structured, scored Evidence Object,
reasons over the graph and the evidence using one of four explicit analysis frameworks, and
returns an Intelligence Brief in which every claim is traceable to a primary-source identifier (PMID, NCT ID, ontology/database ID), a confidence score and the graph path used to reach it.

If any claim in the result cannot be traced in this way, the system does not return a degraded report. It returns a structured gap notice instead.

Research prototype. This project is an experimental research tool. It is not a medical device and its output must not be used for clinical decisions.

Table of contents
Design principles
Architecture
Workflow
Evidence Objects and confidence scoring
Reasoning frameworks
Failure intelligence
Data sources
HTTP API
Repository layout
Getting started
Configuration
Testing
Current status and known limitations
Design principles

These rules are enforced in code, not just described in documentation.

Principle	Where it is enforced
Evidence is a first-class entity. Evidence Objects are stored independently and linked to graph nodes, relationships, hypotheses, questions, etc. through a polymorphic evidence_link table. One object can support several targets without being duplicated.	repositories/evidence_object_repository.py, migrations/0002_create_evidence_link.sql
Only primary scientific sources are valid origins. source_database must belong to a fixed allow-list; library/wrapper names (e.g. Biopython, LangChain) are explicitly rejected. The same list is enforced again by a SQL CHECK constraint.	schemas/evidence_object_schema.py, migrations/0001_create_evidence_object.sql
The LLM never judges whether biology is true. In claim extraction the LLM only reports whether a text supports / conflicts with / is neutral toward a claim, and how confident it is that it read the text correctly. The extraction schema has no field for confidence_score or source_authority_level.	pipeline/claim_extraction_pipeline.py
Confidence is a composite, not an LLM opinion. Only one of six components (extraction_confidence) comes from the LLM.	engine/confidence_engine.py
No unsupported hypotheses. A hypothesis can only be created by binding it to at least one supporting Evidence Object ID; an empty list raises an error.	reasoning/hypothesis_generation.py
No "try again" recommendations. Failed findings are turned into reframed research questions; a regex guard rejects any reframed question that amounts to repeating the failed therapy.	reasoning/failure_pipeline.py
Provenance gate. Every report section must have a non-empty evidence trace whose sources are all in the allow-list, a non-empty reasoning path and a named framework. If any section fails, the whole run returns a gap notice.	agents/explainability_agent.py, orchestration/workflow_controller.py
Fixed orchestration, no autonomous planning. The controller runs one fixed sequence; the only branches are explicit, named rules.	orchestration/workflow_controller.py
Per-source fault isolation. Each external source call is wrapped individually; a failure in one source is logged/recorded and the rest continue.	agents and clients/
Architecture

Services (Docker Compose)

Service	Role
api	FastAPI app. Authenticates requests, enqueues jobs, serves job status/results.
worker	Polls the job table (SELECT … FOR UPDATE SKIP LOCKED) and runs the workflow controller for each pending job.
postgres (16)	Evidence Objects, evidence links, jobs, API keys (plus question / hypothesis tables).
neo4j (5 community)	Knowledge graph (Disease, Gene, Protein, Pathway, …).

Agents

Agent	Responsibility
Ingestion & Curation	Resolves the disease and gene, then writes Disease → Gene → Protein → Pathway nodes and edges to Neo4j. Every node must carry at least one resolvable primary-source ID or the write is rejected. Records a source_manifest noting which sources succeeded or failed.
Evidence	Retrieves candidate records from PubMed, Europe PMC, ClinicalTrials.gov, Open Targets and ChEMBL; runs claim extraction; computes confidence; validates and stores Evidence Objects.
Reasoning	Read-only. Classifies the question into a framework, traverses the graph, selects evidence, generates hypotheses (hard-gated) and processes conflicting evidence through the failure-intelligence pipeline. The only agent with a Neo4j dependency besides ingestion.
Explainability	Resolves every claim into a four-part trace (evidence + source + confidence + reasoning path) using Postgres only, and applies the provenance gate.
Workflow

WorkflowController.receive_query(disease, gene, therapy) executes this fixed sequence:

Resolve or ingest – look up the disease/gene in the graph (if a lookup function is supplied); otherwise call the Ingestion & Curation Agent.
Structural check – GapRules.kg_ids_resolve() confirms the disease/gene node IDs actually exist in Neo4j. Postgres cannot enforce a foreign key against Neo4j, so this is an application-level check. On failure the run halts with a StructuralError.
Ensure evidence – for each target (disease, gene) with fewer than one supporting Evidence Object, call the Evidence Agent.
Reason – call the Reasoning Agent.
Explain – call the Explainability Agent.
Provenance gate – if trace_complete is false (including the case of zero report sections), return a GapNotice; otherwise assemble and return the IntelligenceBrief.

The controller returns exactly one of: IntelligenceBrief, StructuralError, or GapNotice.

Evidence Objects and confidence scoring

An Evidence Object records one claim extracted from one source record:

claim_summary, source_database, source_id, retrieval_method, retrieved_at, optional claim_asserted_date
source_authority_level, primary_or_secondary
evidence_type: genetic_association, functional_mechanistic, clinical_trial_result, case_report, target_disease_association_score, review_meta_analysis
study_design: RCT, observational, preclinical_in_vitro, preclinical_in_vivo, case_series, not_applicable
supports_or_conflicts: supports / conflicts / neutral
confidence_score and the full confidence_components

An Evidence Object with any missing confidence component is rejected, not defaulted.

source_authority_level is computed from source metadata (never by the LLM):

Source	Authority level
ClinicalTrials.gov	registered_trial_record
Open Targets	database_aggregated_score
ChEMBL	curated_database
PubMed / Europe PMC	peer_reviewed_primary (or peer_reviewed_secondary when the extracted type is review_meta_analysis)
Confidence formula
confidence_score = clamp(0, 1,
      0.30 × evidence_type_weight
    + 0.25 × study_design_weight
    + 0.20 × source_authority_weight
    + 0.15 × extraction_confidence          ← the only LLM-derived component
    + replication_boost
    − conflict_penalty )

replication_boost = min(0.15, 0.05 × replication_count)
conflict_penalty  = min(0.30, 0.10 × conflicting_count)
replication_count = number of existing supporting Evidence Objects on the same primary target.
conflicting_count = number of existing conflicting Evidence Objects on that target.
The per-category weight tables are configurable in backend/configs/confidence_weights.yaml:
evidence_type	weight	study_design	weight	authority	weight
clinical_trial_result	0.9	RCT	1.0	curated_database	0.9
genetic_association	0.8	observational	0.6	registered_trial_record	0.9
functional_mechanistic	0.7	preclinical_in_vivo	0.5	peer_reviewed_primary	0.8
target_disease_association_score	0.6	preclinical_in_vitro	0.4	database_aggregated_score	0.7
review_meta_analysis	0.5	case_series	0.3	peer_reviewed_secondary	0.6
case_report	0.3	not_applicable	0.5	preprint	0.4

The outer formula and its coefficients are fixed constants in code (versioned with the engine), not configuration.

Supporting and conflicting evidence are stored as separate objects; they are never averaged together.

Reasoning frameworks

The Reasoning Agent classifies each question into exactly one of four frameworks. Classification combines a rule-based keyword pre-filter (comparator / causal-claim / target-feasibility / mechanism signals, passed to the LLM as advisory input) with an LLM call constrained to one of the four labels plus a rationale and a list of missing information.

Framework	Intended for	Graph traversal	Evidence types considered
PICO	clinical / comparative questions	(Disease)<-[STUDIES]-(ClinicalTrial)-[TESTS]->(Therapy)	clinical_trial_result
MechanisticBiology	pathway / mechanism questions	Gene → ENCODES → Protein → PARTICIPATES_IN → Pathway → AFFECTS → Phenotype	functional_mechanistic, genetic_association
BradfordHill	causal inference, especially without RCT-style evidence	Disease → HAS_CAUSAL_GENE → Gene	genetic_association, functional_mechanistic, case_report, clinical_trial_result
TargetValidation	drug-target feasibility	(Gene)-[ENCODES]->(Protein)<-[TARGETS]-(Therapy)	target_disease_association_score, functional_mechanistic

Every traversal step is denormalized at traversal time into a self-describing record (node_or_edge_type, id, display_name, source_id), so the Explainability Agent never needs to query Neo4j.

Failure intelligence

Evidence tagged conflicts is not discarded. Each conflicting Evidence Object is passed through a four-stage pipeline (all LLM stages operate only on a real, already-stored Evidence Object):

Limitation extraction – was the limit technological, a biological assumption, or design-related?
Technology-delta check – has that specific limiting factor changed since the finding?
Alternative perspective – which adjacent approach, scale or mechanism was not tried?
Reframed research question – a new, narrower question stating what was tried, why it failed, what is different now and what to ask instead.

A hard guard (RETRY_RECOMMENDATION_PATTERNS) raises RetryRecommendationError if the reframed question matches patterns such as "try again", "retry" or "repeat the same trial/study/therapy/treatment".

Data sources

Clients live in backend/clients/. All use file-based caching (30-day TTL), retry up to three times with exponential backoff, and self-throttle.

Source	Client	Used by	Notes
Orphanet (Orphadata)	orphanet_client.py	Ingestion	Bulk XML export, weekly refresh, falls back to the last good copy. Downloaded in 4 MB HTTP Range chunks.
NCBI Gene	ncbi_client.py (NCBIGeneClient)	Ingestion	E-utilities esearch/efetch. 3 req/s, 10 req/s with an API key.
UniProt	uniprot_client.py	Ingestion	REST /uniprotkb/{accession}, 5 req/s.
Reactome	reactome_client.py	Ingestion	Content Service /data/pathways/low/entity/{id}, 5 req/s.
PubMed	ncbi_client.py (PubMedClient)	Evidence	E-utilities, batches of ≤ 200 IDs.
Europe PMC	europepmc_client.py	Evidence	REST search + full-text XML, 1 req/s.
ClinicalTrials.gov	ctgov_client.py	Evidence	API v2 /studies, 5 req/s.
Open Targets	opentargets_client.py	Evidence	GraphQL target–disease association, 1 req/s. If retrieval fails the score is marked unavailable, not defaulted.
ChEMBL	chembl_client.py	Evidence	REST /target, /activity, 5 req/s.
Ensembl	ensembl_client.py	(not yet used by an agent)	Intended for ID cross-referencing only, never as primary evidence.

The source_database allow-list in the schema also contains MONDO and HPO; no client for those is included yet.

HTTP API

Authentication: send X-API-Key: <key>. Keys are stored only as SHA-256 hashes in the api_key table; missing, unknown or revoked keys return 401.

POST /query
json
{
  "disease": "Rett syndrome",
  "gene_target": "MECP2",
  "therapeutic_approach": "gene therapy"
}

Enqueues a job and returns immediately:

json
{ "job_id": "<uuid>", "status": "pending" }
GET /query/{job_id}

Poll for status. Job statuses: pending, running, complete, failed.

complete → { job_id, status, result } where result is the Intelligence Brief: question_id, framework_selected, framework_rationale, trace_complete, and report_sections[]. Each section has claim_or_hypothesis, evidence_trace[] (evidence_id, source_database, source_id, confidence_score, primary_or_secondary), reasoning_path[] and framework_used.
failed with a reason (gap notice or structural error) → { job_id, status, reason, failed_sections }.
failed by an unhandled exception → { job_id, status, result: { "error": "…" } }.
Unknown job_id → 404.

Structured JSON logs (timestamp, job_id, stage, status, source_manifest_delta, duration_ms) are emitted by the API.

Frontend

frontend/index.html is a single static page (no build step) that submits a query, polls the job and renders the brief — claims with numbered citations, the graph path as a rail of nodes, the evidence rows with confidence scores — or the gap notice. It calls /query on the same origin; the FastAPI app does not serve static files or configure CORS, so serve the page and the API behind one host (for example via a reverse proxy).

Repository layout
.
├── backend/
│   ├── agents/            # ingestion_curation, evidence, reasoning, explainability
│   ├── api/main.py        # FastAPI app (POST /query, GET /query/{job_id}, API-key auth)
│   ├── clients/           # one client per external source
│   ├── configs/           # confidence_weights.yaml
│   ├── engine/            # confidence_engine.py
│   ├── orchestration/     # workflow_controller.py, gap_rules.py, worker.py
│   ├── pipeline/          # claim_extraction_pipeline.py
│   ├── reasoning/         # framework_classifier, kg_traversal, hypothesis_generation, failure_pipeline
│   ├── repositories/      # evidence_object_repository.py (Postgres)
│   ├── schemas/           # evidence_object_schema.py (pydantic models + validation gates)
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/index.html
├── migrations/            # 0001 … 0008 SQL migrations
├── neo4j_constraints_indexes.cypher
├── docker-compose.yml
└── .env.example
Getting started

Prerequisites: Docker with Compose; an NCBI API key (optional but raises the rate limit); an Anthropic API key for the LLM stages.

1. Configure
bash
cp .env.example .env
# edit .env: set passwords and API keys (see Configuration below)
2. Start the stack
bash
docker compose up --build

This starts api (port 8000), worker, postgres and neo4j.

3. Apply the database schema

The migrations and the Neo4j constraints are not applied automatically by docker-compose.yml. Apply them once, in order:

bash
for f in migrations/*.sql; do
  docker compose exec -T postgres psql -U app -d biomed_evidence < "$f"
done

docker compose exec -T neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  < neo4j_constraints_indexes.cypher
4. Create an API key

There is no endpoint for creating keys; insert the SHA-256 hash of a key you generate:

bash
KEY=$(openssl rand -hex 32)
HASH=$(printf %s "$KEY" | sha256sum | cut -d' ' -f1)
docker compose exec -T postgres psql -U app -d biomed_evidence -c \
  "INSERT INTO api_key (key_id, key_hash, owner_label) VALUES (gen_random_uuid(), '$HASH', 'local');"
echo "Your API key: $KEY"
5. Submit a query
bash
curl -s -X POST http://localhost:8000/query \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"disease":"Rett syndrome","gene_target":"MECP2","therapeutic_approach":"gene therapy"}'

curl -s http://localhost:8000/query/<job_id> -H "X-API-Key: $KEY"
Configuration

Environment variables (see .env.example for the full annotated list):

Variable	Purpose
PG_PASSWORD, DATABASE_URL	PostgreSQL credentials / connection string
NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER	Neo4j credentials / connection
NCBI_API_KEY	Raises the NCBI rate limit from 3 to 10 req/s
LLM_API_KEY	Passed to the api and worker containers by docker-compose.yml
LLM_EXTRACTION_MODEL, LLM_CLASSIFIER_MODEL, LLM_FAILURE_PIPELINE_MODEL	Model for claim extraction, framework classification and failure intelligence (default claude-sonnet-4-6)
WORKER_POLL_INTERVAL_SECONDS	Worker polling interval
*_CACHE_DIR, ORPHANET_STORE_DIR	Cache / local-store directories for each client
CONFIDENCE_WEIGHTS_PATH	Optional override of the weights YAML

Note on the LLM key: the LLM stages call anthropic.Anthropic() without passing a key explicitly, so the Anthropic SDK reads its standard ANTHROPIC_API_KEY environment variable. docker-compose.yml currently forwards LLM_API_KEY only; make sure the key reaches the containers under the name the SDK expects.

Testing

Run from the backend/ directory (imports are relative to it):

bash
cd backend
pip install -r requirements.txt
pytest
Test module	What it covers
test_confidence_engine_mecp2_worked_example.py	Weight math against a worked MECP2 example (supporting Open Targets association vs. conflicting toxicity study), replication/conflict caps, clamping, unknown-weight error.
test_claim_extraction_pipeline_mecp2_rett.py	Schema-constrained extraction, schema-repair retry and drop behaviour over a fixture set of MECP2/Rett excerpts (LLM mocked).
test_reasoning_agent_failed_hypothesis_reframe.py	A documented failed hypothesis (constitutive AAV-MECP2 overexpression toxicity) is reframed rather than retried; traversal path is denormalized; hypotheses require supporting evidence; the retry guard rejects retry phrasing.
test_end_to_end_rett_mecp2_gene_therapy.py	The full controller sequence (ingestion → evidence → reasoning → explainability) against fakes/fixtures for Rett/MECP2/gene therapy plus a second DMD exon-skipping case; asserts every section passes validate_trace_complete.
test_ingestion_curation_agent_rett_mecp2.py	Ingestion write path against a live Neo4j instance, with source-client HTTP responses mocked.

External HTTP and LLM calls are mocked or faked in the tests. The ingestion test needs a reachable Neo4j (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD).

Current status and known limitations

This is an MVP-stage research prototype. The unit/integration-style tests exercise each component and the controller sequence with fixtures and fakes; a fully live end-to-end run against all real services is not demonstrated in this repository. Specifically, based on the current code:

Graph coverage. Ingestion writes Disease, Gene, Protein and Pathway nodes with HAS_CAUSAL_GENE, ENCODES and PARTICIPATES_IN edges. Phenotype, Therapy and ClinicalTrial nodes and the AFFECTS, TARGETS, TESTS and STUDIES edges used by some framework traversals are defined in the schema/queries but nothing in the codebase writes them yet.
UniProt resolution. The protein lookup currently uses the gene symbol as the accession key (a documented MVP placeholder); Ensembl cross-referencing is not yet wired in.
Optional evidence sources. Open Targets and ChEMBL retrieval run only if ensembl_id / efo_id / chembl_target_id are present in the query context; the controller does not currently supply them.
Identifiers passed between stages. The controller passes Neo4j node id values as the question's disease/gene identifiers, whereas the traversal queries match on orpha_id / hgnc_id, and the evidence search term is built from the values in the query context. These hand-offs should be reviewed before running against live data.
Hypothesis text. The hypothesis statement is a template string bound to the selected supporting Evidence Object IDs; it is not LLM-written.
Unused tables. question and hypothesis tables exist in the migrations, but the current code does not write to them.
Design-document references. Code comments refer to numbered sections (e.g. "Section 24") and "Patches" of a design specification that is not included in this repository.