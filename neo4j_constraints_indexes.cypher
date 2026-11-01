CREATE CONSTRAINT disease_id_unique FOR (d:Disease) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT gene_id_unique FOR (g:Gene) REQUIRE g.id IS UNIQUE;
CREATE CONSTRAINT protein_id_unique FOR (p:Protein) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT pathway_id_unique FOR (pw:Pathway) REQUIRE pw.id IS UNIQUE;
CREATE CONSTRAINT phenotype_id_unique FOR (ph:Phenotype) REQUIRE ph.id IS UNIQUE;
CREATE CONSTRAINT therapy_id_unique FOR (t:Therapy) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT trial_id_unique FOR (ct:ClinicalTrial) REQUIRE ct.id IS UNIQUE;

CREATE INDEX gene_hgnc_idx FOR (g:Gene) ON (g.hgnc_id);
CREATE INDEX disease_orpha_idx FOR (d:Disease) ON (d.orpha_id);
CREATE INDEX trial_nct_idx FOR (ct:ClinicalTrial) ON (ct.nct_id);
