
# Real-Time RAG System with Kafka and Observability

A complete, end-to-end system that ingests a real-time data stream, processes it using a Retrieval-Augmented Generation (RAG) pipeline, and provides a queryable API with a full observability stack.

---

##  Core Features

* **Real-Time Ingestion:** Ingests data from a live Wikimedia SSE stream using a resilient `producer`.
* **Event-Driven Architecture:** Uses **Kafka** as a durable message bus to decouple ingestion from processing.
* **Asynchronous Processing:** A scalable `processor` service consumes data, creates vector embeddings (`all-MiniLM-L6-v2`), and stores them.
* **Flexible Vector Store:** Supports both **Redis** and **FAISS** as vector databases using an Adapter Pattern.
* **Intelligent RAG API:** A secure FastAPI (`/v1/ask`) endpoint retrieves relevant context from the vector store to provide grounded, factual answers from an LLM.
* **Full Observability Stack:** Pre-configured monitoring with **Prometheus**, **Grafana**, and **Jaeger** for metrics, dashboards, and distributed tracing.
* **Containerized:** The entire multi-service application is containerized with **Docker** for consistent and reliable deployments.

---

##  System Architecture

The system follows a decoupled, event-driven architecture. Data flows from the producer, through the Kafka message bus, to the processor for storage, and is finally queried by the API.


```mermaid
graph TD
    A[Wikimedia SSE Stream] --> B(Producer Service);
    B --> C{Kafka Topic: wikipedia.edits};
    C --> D(Processor Service);
    D -- Stores Vector --> E[Vector DB <br>(Redis / FAISS)];
    F(User) --> G(API Service);
    G -- Queries Vector --> E;
    G -- Sends Prompt to --> H(LLM <br>(Ollama));
    H --> G;
    G --> F;

    


## Tech Stack
Backend: Python, FastAPI
Data Streaming: Kafka
Database: Redis (Vector Store), PostgreSQL (Metadata - future use)
AI/ML: Sentence-Transformers, PyTorch, Ollama
Observability: Prometheus, Grafana, Jaeger (OpenTelemetry)
Containerization: Docker, Docker Compose


## External endpoints 
**On Windows:** You will need to use a Bash-compliant terminal like **Git Bash** (which comes with Git) or **WSL** to run the helper scripts (`.sh` files)

API Documentation (Swagger UI): http://localhost:8000/docs
Grafana Dashboards: http://localhost:3000 (login: admin / admin)
Prometheus Targets: http://localhost:9090/targets
Jaeger Traces: http://localhost:16686
./start.sh: Starts the entire application stack.
./stop.sh: Stops all running containers.
./logs.sh <service_name>: Follows the logs for a specific service (e.g., ./logs.sh api).
./test.sh: Runs a quick smoke test against the API.

