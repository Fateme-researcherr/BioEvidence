"""
Ingestion & Curation Agent v0 (Section 21.1).

INPUT:  { disease_query: string, gene_filter?: string, therapy_filter?: string }
OUTPUT: { kg_nodes_written: [NodeRef], kg_edges_written: [EdgeRef],
          source_manifest: [{source_database, source_id, retrieval_method}] }

Validation gate: every node/edge written must carry >=1 resolvable
primary-source ID (Section 7); write rejected otherwise.

Error handling: each source client call wrapped individually — a failure
in one source logs and continues rather than aborting the whole
ingestion; partial results are acceptable and flagged in `source_manifest`.

Week 1 completion criterion: validated against Rett syndrome + MECP2,
populating Neo4j end-to-end (Disease, Gene, Protein, Pathway nodes).
"""

import os
import re
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from neo4j import GraphDatabase

from clients.ncbi_client import NCBIGeneClient
from clients.orphanet_client import OrphanetClient
from clients.reactome_client import ReactomeClient
from clients.uniprot_client import UniProtClient

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")


@dataclass
class NodeRef:
    label: str  # enum[Disease, Gene, Protein, Pathway, Phenotype, Therapy, ClinicalTrial]
    id: str
    source_id_fields: Dict[str, Optional[str]] = field(default_factory=dict)


@dataclass
class EdgeRef:
    type: str  # enum[HAS_CAUSAL_GENE, ENCODES, PARTICIPATES_IN, AFFECTS, PRESENTS_WITH, TARGETS, TESTS, STUDIES]
    id: str
    from_: NodeRef
    to: NodeRef


@dataclass
class SourceManifestEntry:
    source_database: str
    source_id: str
    retrieval_method: str
    note: Optional[str] = None


class IngestionCurationAgentError(Exception):
    pass


class IngestionCurationAgent:
    """
    Orchestrates Disease -> Gene -> Protein -> Pathway population of the
    Knowledge Graph from official sources, per the fixed ingestion contract.
    """

    def __init__(self):
        self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self._orphanet = OrphanetClient()
        self._ncbi_gene = NCBIGeneClient()
        self._uniprot = UniProtClient()
        self._reactome = ReactomeClient()

    def close(self):
        self._driver.close()

    # -- public contract -----------------------------------------------

    def run(
        self,
        disease_query: str,
        gene_filter: Optional[str] = None,
        therapy_filter: Optional[str] = None,
    ) -> dict:
        kg_nodes_written: List[NodeRef] = []
        kg_edges_written: List[EdgeRef] = []
        source_manifest: List[SourceManifestEntry] = []

        disease_node = self._ingest_disease(disease_query, source_manifest)
        if disease_node is not None:
            kg_nodes_written.append(disease_node)

        gene_node = None
        if gene_filter is not None:
            gene_node = self._ingest_gene(gene_filter, source_manifest)
            if gene_node is not None:
                kg_nodes_written.append(gene_node)

                if disease_node is not None:
                    edge = self._write_edge(
                        "HAS_CAUSAL_GENE", disease_node, gene_node, source_manifest
                    )
                    if edge is not None:
                        kg_edges_written.append(edge)

        protein_node = None
        if gene_node is not None:
            protein_node = self._ingest_protein(gene_filter, source_manifest)
            if protein_node is not None:
                kg_nodes_written.append(protein_node)

                edge = self._write_edge("ENCODES", gene_node, protein_node, source_manifest)
                if edge is not None:
                    kg_edges_written.append(edge)

        pathway_node = None
        if protein_node is not None:
            pathway_node = self._ingest_pathway(protein_node, source_manifest)
            if pathway_node is not None:
                kg_nodes_written.append(pathway_node)

                edge = self._write_edge(
                    "PARTICIPATES_IN", protein_node, pathway_node, source_manifest
                )
                if edge is not None:
                    kg_edges_written.append(edge)

        return {
            "kg_nodes_written": kg_nodes_written,
            "kg_edges_written": kg_edges_written,
            "source_manifest": source_manifest,
        }

    # -- per-source ingestion (each wrapped individually) ----------------

    def _ingest_disease(
        self, disease_query: str, source_manifest: List[SourceManifestEntry]
    ) -> Optional[NodeRef]:
        try:
            bulk_load = self._orphanet.load()
            if bulk_load is None:
                source_manifest.append(
                    SourceManifestEntry(
                        source_database="Orphanet",
                        source_id=disease_query,
                        retrieval_method="Orphadata bulk XML export",
                        note="failed: no bulk load available, skipped",
                    )
                )
                return None

            orpha_id, name = self._parse_disease_from_bulk(bulk_load.raw_content, disease_query)
            if orpha_id is None:
                source_manifest.append(
                    SourceManifestEntry(
                        source_database="Orphanet",
                        source_id=disease_query,
                        retrieval_method=bulk_load.retrieval_method,
                        note="failed: disease not resolvable to a primary-source ID, skipped",
                    )
                )
                return None

            source_id_fields = {"orpha_id": orpha_id, "mondo_id": None}
            node = self._write_node(
                label="Disease",
                match_key="orpha_id",
                match_value=orpha_id,
                properties={"name": name, "orpha_id": orpha_id},
                source_id_fields=source_id_fields,
            )
            source_manifest.append(
                SourceManifestEntry(
                    source_database="Orphanet",
                    source_id=orpha_id,
                    retrieval_method=bulk_load.retrieval_method,
                )
            )
            return node
        except Exception as exc:  # per-source failure logs and continues
            source_manifest.append(
                SourceManifestEntry(
                    source_database="Orphanet",
                    source_id=disease_query,
                    retrieval_method="Orphadata bulk XML export",
                    note=f"failed: {exc}, skipped",
                )
            )
            return None

    def _ingest_gene(
        self, gene_symbol: str, source_manifest: List[SourceManifestEntry]
    ) -> Optional[NodeRef]:
        try:
            gene_ids = self._ncbi_gene.search(f"{gene_symbol}[sym] AND human[orgn]")
            if not gene_ids:
                source_manifest.append(
                    SourceManifestEntry(
                        source_database="NCBIGene",
                        source_id=gene_symbol,
                        retrieval_method="NCBI E-utilities esearch (db=gene)",
                        note="failed: no gene ID resolved, skipped",
                    )
                )
                return None

            records = self._ncbi_gene.fetch(gene_ids[:1])
            if not records:
                source_manifest.append(
                    SourceManifestEntry(
                        source_database="NCBIGene",
                        source_id=gene_ids[0],
                        retrieval_method="NCBI E-utilities efetch (db=gene)",
                        note="failed: efetch returned no record, skipped",
                    )
                )
                return None

            record = records[0]
            ncbi_gene_id = record.identifier
            hgnc_id = self._extract_hgnc_id(record.raw_xml)

            if hgnc_id is None and ncbi_gene_id is None:
                source_manifest.append(
                    SourceManifestEntry(
                        source_database=record.source_database,
                        source_id=ncbi_gene_id,
                        retrieval_method=record.retrieval_method,
                        note="failed: no resolvable primary-source ID, write rejected",
                    )
                )
                return None

            source_id_fields = {
                "hgnc_id": hgnc_id,
                "ncbi_gene_id": ncbi_gene_id,
                "ensembl_id": None,
            }
            node = self._write_node(
                label="Gene",
                match_key="ncbi_gene_id",
                match_value=ncbi_gene_id,
                properties={"symbol": gene_symbol, "ncbi_gene_id": ncbi_gene_id, "hgnc_id": hgnc_id},
                source_id_fields=source_id_fields,
            )
            source_manifest.append(
                SourceManifestEntry(
                    source_database=record.source_database,
                    source_id=ncbi_gene_id,
                    retrieval_method=record.retrieval_method,
                )
            )
            return node
        except Exception as exc:
            source_manifest.append(
                SourceManifestEntry(
                    source_database="NCBIGene",
                    source_id=gene_symbol,
                    retrieval_method="NCBI E-utilities (db=gene)",
                    note=f"failed: {exc}, skipped",
                )
            )
            return None

    def _ingest_protein(
        self, gene_symbol: str, source_manifest: List[SourceManifestEntry]
    ) -> Optional[NodeRef]:
        try:
            # Week 1: direct accession lookup post ID-resolution. Ensembl
            # cross-reference is not yet wired (scheduled Week 2); the gene
            # symbol is used as the accession-resolution key for MVP.
            accession = gene_symbol
            record = self._uniprot.fetch(accession)
            if record is None:
                source_manifest.append(
                    SourceManifestEntry(
                        source_database="UniProt",
                        source_id=accession,
                        retrieval_method="UniProt REST /uniprotkb/{accession}",
                        note="failed: no UniProt record resolved, skipped",
                    )
                )
                return None

            uniprot_id = record.accession
            if not uniprot_id:
                source_manifest.append(
                    SourceManifestEntry(
                        source_database=record.source_database,
                        source_id=accession,
                        retrieval_method=record.retrieval_method,
                        note="failed: no resolvable primary-source ID, write rejected",
                    )
                )
                return None

            source_id_fields = {"uniprot_id": uniprot_id}
            node = self._write_node(
                label="Protein",
                match_key="uniprot_id",
                match_value=uniprot_id,
                properties={"uniprot_id": uniprot_id, "name": None, "function_summary": None},
                source_id_fields=source_id_fields,
            )
            source_manifest.append(
                SourceManifestEntry(
                    source_database=record.source_database,
                    source_id=uniprot_id,
                    retrieval_method=record.retrieval_method,
                )
            )
            return node
        except Exception as exc:
            source_manifest.append(
                SourceManifestEntry(
                    source_database="UniProt",
                    source_id=gene_symbol,
                    retrieval_method="UniProt REST /uniprotkb/{accession}",
                    note=f"failed: {exc}, skipped",
                )
            )
            return None

    def _ingest_pathway(
        self, protein_node: NodeRef, source_manifest: List[SourceManifestEntry]
    ) -> Optional[NodeRef]:
        try:
            uniprot_id = protein_node.source_id_fields.get("uniprot_id")
            if not uniprot_id:
                return None

            record = self._reactome.fetch_pathways_for_uniprot_accession(uniprot_id)
            if record is None:
                source_manifest.append(
                    SourceManifestEntry(
                        source_database="Reactome",
                        source_id=uniprot_id,
                        retrieval_method="Reactome Content Service /data/pathways/low/entity/{id}",
                        note="failed: no pathway resolved, skipped",
                    )
                )
                return None

            reactome_id = self._extract_first_pathway_id(record.raw_content)
            if reactome_id is None:
                source_manifest.append(
                    SourceManifestEntry(
                        source_database=record.source_database,
                        source_id=uniprot_id,
                        retrieval_method=record.retrieval_method,
                        note="failed: no resolvable primary-source ID, write rejected",
                    )
                )
                return None

            source_id_fields = {"reactome_id": reactome_id}
            node = self._write_node(
                label="Pathway",
                match_key="reactome_id",
                match_value=reactome_id,
                properties={"reactome_id": reactome_id, "name": None},
                source_id_fields=source_id_fields,
            )
            source_manifest.append(
                SourceManifestEntry(
                    source_database=record.source_database,
                    source_id=reactome_id,
                    retrieval_method=record.retrieval_method,
                )
            )
            return node
        except Exception as exc:
            source_manifest.append(
                SourceManifestEntry(
                    source_database="Reactome",
                    source_id=protein_node.id,
                    retrieval_method="Reactome Content Service",
                    note=f"failed: {exc}, skipped",
                )
            )
            return None

    # -- Neo4j write helpers ---------------------------------------------

    def _write_node(
        self,
        label: str,
        match_key: str,
        match_value: str,
        properties: dict,
        source_id_fields: Dict[str, Optional[str]],
    ) -> Optional[NodeRef]:
        # Validation gate: reject write if no resolvable primary-source ID.
        if not any(v for v in source_id_fields.values()):
            raise IngestionCurationAgentError(
                f"{label} node rejected: no resolvable primary-source ID"
            )

        node_id = str(uuid.uuid4())
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        props = dict(properties)
        props["id"] = node_id
        props["created_at"] = now
        props["updated_at"] = now

        query = (
            f"MERGE (n:{label} {{{match_key}: $match_value}}) "
            f"ON CREATE SET n += $props "
            f"RETURN n.id AS id"
        )
        with self._driver.session() as session:
            result = session.run(query, match_value=match_value, props=props)
            record = result.single()
            written_id = record["id"] if record else node_id

        return NodeRef(label=label, id=written_id, source_id_fields=source_id_fields)

    def _write_edge(
        self,
        edge_type: str,
        from_node: NodeRef,
        to_node: NodeRef,
        source_manifest: List[SourceManifestEntry],
    ) -> Optional[EdgeRef]:
        try:
            edge_id = str(uuid.uuid4())
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            query = (
                f"MATCH (a:{from_node.label} {{id: $from_id}}), "
                f"(b:{to_node.label} {{id: $to_id}}) "
                f"MERGE (a)-[r:{edge_type}]->(b) "
                f"ON CREATE SET r.id = $edge_id, r.created_at = $now, r.source_agent = $agent "
                f"RETURN r.id AS id"
            )
            with self._driver.session() as session:
                result = session.run(
                    query,
                    from_id=from_node.id,
                    to_id=to_node.id,
                    edge_id=edge_id,
                    now=now,
                    agent="IngestionCurationAgent",
                )
                record = result.single()
                written_id = record["id"] if record else edge_id

            return EdgeRef(type=edge_type, id=written_id, from_=from_node, to=to_node)
        except Exception as exc:
            source_manifest.append(
                SourceManifestEntry(
                    source_database="KG",
                    source_id=f"{from_node.id}->{to_node.id}",
                    retrieval_method=f"Neo4j edge write ({edge_type})",
                    note=f"failed: {exc}, skipped",
                )
            )
            return None

    # -- parsing helpers ---------------------------------------------------

    def _parse_disease_from_bulk(self, raw_xml: str, disease_query: str):
        """Resolve a disease name/query to its ORPHA ID from the Orphadata bulk export."""
        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError:
            return None, None

        query_lower = disease_query.strip().lower()
        for disorder in root.iter("Disorder"):
            name_elem = disorder.find("Name")
            orpha_elem = disorder.find("OrphaCode")
            if name_elem is None or orpha_elem is None:
                continue
            if name_elem.text and query_lower in name_elem.text.strip().lower():
                return f"ORPHA:{orpha_elem.text.strip()}", name_elem.text.strip()

        return None, None

    def _extract_hgnc_id(self, raw_xml: str) -> Optional[str]:
        match = re.search(r"HGNC:(\d+)", raw_xml)
        if match:
            return f"HGNC:{match.group(1)}"
        return None

    def _extract_first_pathway_id(self, raw_content: str) -> Optional[str]:
        match = re.search(r'"stId"\s*:\s*"([^"]+)"', raw_content)
        if match:
            return match.group(1)
        return None
