"""BIA-02.6: migrações transacionais e verificáveis, sem depender de APIs externas.

0001 marca o esquema histórico; 0002 adiciona avaliações e snapshots append-only;
0003 adiciona governança de retenção sem remover fontes nem direitos históricos.
Não modifica colunas antigas nem elimina dados. Instalações antigas sem tabela de versão
são reconhecidas somente quando o conjunto de tabelas histórico está completo.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

VERSIONS = ("0001", "0002", "0003")
VERSION_TABLE = "bia_schema_versions"


def upgrade_schema(engine: Engine, *, auto: bool = True) -> str:
    from .db import Base
    new_tables = {"dataset_snapshots", "institute_assessments", "document_governance"}
    old_tables = [t for t in Base.metadata.sorted_tables if t.name not in new_tables]
    with engine.connect() as conn:
        try:
            # SQLite: serializa múltiplos processos de migração desde o primeiro comando.
            # PostgreSQL: lock transacional nomeado para evitar duas instâncias competindo.
            if engine.dialect.name == "sqlite":
                conn.exec_driver_sql("BEGIN IMMEDIATE")
            elif engine.dialect.name == "postgresql":
                conn.execute(text("SELECT pg_advisory_xact_lock(802026)"))
            existing = set(inspect(conn).get_table_names())
            if VERSION_TABLE not in existing:
                if not auto:
                    raise RuntimeError("Schema sem versão; execute python -m bia_core.schema")
                historical = {t.name for t in old_tables}
                found = historical.intersection(existing)
                if found and found != historical:
                    raise RuntimeError("Schema parcial não identificado; faça backup e migração supervisionada")
                if not found:
                    Base.metadata.create_all(conn, tables=old_tables, checkfirst=True)
                conn.execute(text(
                    "CREATE TABLE bia_schema_versions (version VARCHAR(8) PRIMARY KEY, installed_utc VARCHAR(40) NOT NULL)"
                ))
                conn.execute(text("INSERT INTO bia_schema_versions (version, installed_utc) VALUES (:version, :at)"),
                             {"version": VERSIONS[0], "at": datetime.now(timezone.utc).isoformat()})
            recorded = [r[0] for r in conn.execute(text("SELECT version FROM bia_schema_versions ORDER BY version"))]
            if not recorded or recorded[0] != VERSIONS[0] or any(v not in VERSIONS for v in recorded):
                raise RuntimeError("Versão de schema inesperada; downgrade/upgrade não autorizado")
            if recorded != list(VERSIONS[:len(recorded)]):
                raise RuntimeError("Histórico de migrações descontínuo")
            if VERSIONS[1] not in recorded:
                if not auto:
                    raise RuntimeError("Schema desatualizado; execute python -m bia_core.schema")
                Base.metadata.create_all(conn, tables=[Base.metadata.tables[n] for n in ("dataset_snapshots", "institute_assessments")], checkfirst=True)
                # Imutabilidade no próprio banco, inclusive contra escrita fora da API.
                if engine.dialect.name == "sqlite":
                    conn.exec_driver_sql("""CREATE TRIGGER IF NOT EXISTS bia_snapshot_no_update
                        BEFORE UPDATE ON dataset_snapshots BEGIN SELECT RAISE(ABORT, 'snapshot imutavel'); END""")
                    conn.exec_driver_sql("""CREATE TRIGGER IF NOT EXISTS bia_snapshot_no_delete
                        BEFORE DELETE ON dataset_snapshots BEGIN SELECT RAISE(ABORT, 'snapshot imutavel'); END""")
                elif engine.dialect.name == "postgresql":
                    conn.exec_driver_sql("""CREATE OR REPLACE FUNCTION bia_snapshot_immutable() RETURNS trigger
                        LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'snapshot imutavel'; END $$""")
                    conn.exec_driver_sql("""CREATE TRIGGER bia_snapshot_no_mutation BEFORE UPDATE OR DELETE
                        ON dataset_snapshots FOR EACH ROW EXECUTE FUNCTION bia_snapshot_immutable()""")
                else:
                    raise RuntimeError("Snapshots imutáveis não homologados para este dialeto")
                conn.execute(text("INSERT INTO bia_schema_versions (version, installed_utc) VALUES (:version, :at)"),
                             {"version": VERSIONS[1], "at": datetime.now(timezone.utc).isoformat()})
            if VERSIONS[2] not in recorded:
                if not auto:
                    raise RuntimeError("Schema desatualizado; execute python -m bia_core.schema")
                Base.metadata.create_all(conn, tables=[Base.metadata.tables["document_governance"]], checkfirst=True)
                conn.execute(text("INSERT INTO bia_schema_versions (version, installed_utc) VALUES (:version, :at)"),
                             {"version": VERSIONS[2], "at": datetime.now(timezone.utc).isoformat()})
            if engine.dialect.name == "sqlite":
                installed = {r[0] for r in conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ('bia_snapshot_no_update', 'bia_snapshot_no_delete')")}
                if installed != {"bia_snapshot_no_update", "bia_snapshot_no_delete"}:
                    raise RuntimeError("Integridade de snapshots ausente; restaure os triggers antes de iniciar")
            conn.commit()
            return VERSIONS[-1]
        except Exception:
            conn.rollback()
            raise


def main():
    from sqlalchemy import create_engine
    from .db import database_url
    parser = argparse.ArgumentParser(description="Migrações de banco bIA")
    parser.add_argument("--database-url", default=database_url())
    args = parser.parse_args()
    engine = create_engine(args.database_url)
    print("Schema bIA:", upgrade_schema(engine))


if __name__ == "__main__":
    main()
