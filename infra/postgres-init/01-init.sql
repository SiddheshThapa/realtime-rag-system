-- infra/postgres-init/01-init.sql
--CREATE USER rag WITH PASSWORD 'ragpwd';
--CREATE DATABASE ragdb OWNER rag;
--GRANT ALL PRIVILEGES ON DATABASE ragdb TO rag;
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);