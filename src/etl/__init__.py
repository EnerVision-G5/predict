"""Ingestion des mesures énergétiques vers TimescaleDB.

Le pipeline est découpé en trois étages indépendants et testables séparément :
extraction depuis l'API Mock IoT, transformation en tableau normalisé,
chargement idempotent en base. `pipeline` les enchaîne.
"""
