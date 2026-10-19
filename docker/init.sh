#!/bin/bash
# Sourced by the PostgreSQL entrypoint while initialising an empty data
# directory. Renders docker/init.sql with the HNSW parameters taken from the
# environment.
#
# Why not mount init.sql into docker-entrypoint-initdb.d directly: the index
# parameters have to come from somewhere, and an entrypoint script is the only
# place in this container where environment variables and SQL meet. psql's
# :variable substitution is the templating mechanism; mounting the .sql on its
# own would leave `:hnsw_m` unsubstituted and fail to parse.
#
# Defaults are repeated here as well as in docker-compose.yml so the schema
# still initialises correctly if the image is run without compose.

: "${APEX_HNSW_M:=16}"
: "${APEX_HNSW_EF_CONSTRUCTION:=64}"

# Fatal on failure, deliberately: a half-built schema that starts anyway is much
# worse than a container that refuses to come up.
psql -v ON_ERROR_STOP=1 \
     -v hnsw_m="$APEX_HNSW_M" \
     -v hnsw_ef_construction="$APEX_HNSW_EF_CONSTRUCTION" \
     --username "$POSTGRES_USER" \
     --dbname "$POSTGRES_DB" \
     -f /opt/apex/init.sql || exit 1
