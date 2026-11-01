"""
Week 1 completion-criterion validation (Section 30):

"Ingestion & Curation Agent v0 — validated against Rett syndrome + MECP2
populating Neo4j end-to-end (Disease, Gene, Protein, Pathway nodes)."

External source-client HTTP calls are mocked with fixture data representing
Rett syndrome / MECP2 (consistent with the Integration Tests strategy,
Section 29 — mocked external HTTP responses per primary-source client);
the Neo4j write path itself is exercised end-to-end against a live Neo4j
instance (as started via docker-compose, Section 28).
"""

import os

import pytest
from neo4j import GraphDatabase

from agents.ingestion_curation_agent import IngestionCurationAgent
from clients.ncbi_client import NCBIRecord
from clients.orphanet_client import OrphanetBulkLoad
from clients.reactome_client import ReactomePathwayRecord
from clients.uniprot_client import UniProtRecord

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

RETT_ORPHADATA_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<JDBOR>
  <DisorderList>
    <Disorder id="1">
      <OrphaCode>778</OrphaCode>
      <Name lang="en">Rett syndrome</Name>
    </Disorder>
  </DisorderList>
</JDBOR>"""

MECP2_GENE_XML_FIXTURE = """<Entrezgene>
  <Gene-ref_syn>MECP2</Gene-ref_syn>
  <Dbtag_db>HGNC</Dbtag_db>
  <Dbtag_tag>HGNC:6990</Dbtag_tag>
  <note>Xref HGNC:6990</note>
</Entrezgene>"""

MECP2_UNIPROT_FIXTURE = """{"primaryAccession": "P51608", "uniProtkbId": "MECP2_HUMAN"}"""

MECP2_REACTOME_FIXTURE = """[{"stId": "R-HSA-2559580", "displayName": "Oxidative Stress Induced Senescence"}]"""


@pytest.fixture
def agent():
    instance = IngestionCurationAgent()
    yield instance
    instance.close()


@pytest.fixture(autouse=True)
def _cleanup_rett_mecp2_nodes():
    yield
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run(
            "MATCH (d:Disease {orpha_id: 'ORPHA:778'}) DETACH DELETE d"
        )
        session.run(
            "MATCH (g:Gene {ncbi_gene_id: '4204'}) DETACH DELETE g"
        )
        session.run(
            "MATCH (p:Protein {uniprot_id: 'P51608'}) DETACH DELETE p"
        )
        session.run(
            "MATCH (pw:Pathway {reactome_id: 'R-HSA-2559580'}) DETACH DELETE pw"
        )
    driver.close()


def test_rett_mecp2_populates_neo4j_end_to_end(agent, monkeypatch):
    monkeypatch.setattr(
        agent._orphanet,
        "load",
        lambda: OrphanetBulkLoad(
            raw_content=RETT_ORPHADATA_FIXTURE,
            source_database="Orphanet",
            retrieval_method="Orphadata bulk XML export (scheduled weekly refresh)",
            loaded_at=0.0,
        ),
    )
    monkeypatch.setattr(agent._ncbi_gene, "search", lambda term: ["4204"])
    monkeypatch.setattr(
        agent._ncbi_gene,
        "fetch",
        lambda ids: [
            NCBIRecord(
                identifier="4204",
                raw_xml=MECP2_GENE_XML_FIXTURE,
                source_database="NCBIGene",
                retrieval_method="NCBI E-utilities efetch via requests (db=gene)",
            )
        ],
    )
    monkeypatch.setattr(
        agent._uniprot,
        "fetch",
        lambda accession: UniProtRecord(
            accession="P51608",
            raw_content=MECP2_UNIPROT_FIXTURE,
            source_database="UniProt",
            retrieval_method="UniProt REST /uniprotkb/{accession}",
        ),
    )
    monkeypatch.setattr(
        agent._reactome,
        "fetch_pathways_for_uniprot_accession",
        lambda uniprot_id: ReactomePathwayRecord(
            entity_id=uniprot_id,
            raw_content=MECP2_REACTOME_FIXTURE,
            source_database="Reactome",
            retrieval_method="Reactome Content Service /data/pathways/low/entity/{id}",
        ),
    )

    result = agent.run(disease_query="Rett syndrome", gene_filter="MECP2")

    written_labels = {node.label for node in result["kg_nodes_written"]}
    assert written_labels == {"Disease", "Gene", "Protein", "Pathway"}

    written_edge_types = {edge.type for edge in result["kg_edges_written"]}
    assert written_edge_types == {"HAS_CAUSAL_GENE", "ENCODES", "PARTICIPATES_IN"}

    for node in result["kg_nodes_written"]:
        assert any(v for v in node.source_id_fields.values()), (
            f"{node.label} node missing resolvable primary-source ID"
        )

    failed_entries = [entry for entry in result["source_manifest"] if entry.note]
    assert not failed_entries, f"unexpected source failures: {failed_entries}"

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        disease_count = session.run(
            "MATCH (d:Disease {orpha_id: 'ORPHA:778'}) RETURN count(d) AS c"
        ).single()["c"]
        gene_count = session.run(
            "MATCH (g:Gene {ncbi_gene_id: '4204'}) RETURN count(g) AS c"
        ).single()["c"]
        protein_count = session.run(
            "MATCH (p:Protein {uniprot_id: 'P51608'}) RETURN count(p) AS c"
        ).single()["c"]
        pathway_count = session.run(
            "MATCH (pw:Pathway {reactome_id: 'R-HSA-2559580'}) RETURN count(pw) AS c"
        ).single()["c"]
    driver.close()

    assert disease_count == 1
    assert gene_count == 1
    assert protein_count == 1
    assert pathway_count == 1
