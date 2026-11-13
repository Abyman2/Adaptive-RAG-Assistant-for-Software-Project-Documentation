# UniServe Software Project Documentation

## 1. System Overview and Value Proposition
UniServe is an enterprise distributed platform connecting field service providers with business clients in real-time. It coordinates technician dispatching, job tracking, asynchronous notifications, invoice generation, and settlement reconciliation. The system operates on a cloud-native microservices architecture designed for high throughput, sub-second dispatch latency, and 99.99% availability.

## 2. Technology Stack & Frameworks
The core system leverages the following enterprise stack:
- Frontend Client: React 18 single-page application with Vite tooling and TailwindCSS styling.
- Backend Core Service: Spring Boot 3.2 built on Java 17 LTS runtime with Spring Data JPA.
- Persistence Layer: PostgreSQL 15 relational database cluster with read-replicas.
- In-Memory Cache: Redis 7 cluster for session state and rapid geospatial provider indexing.
- Asynchronous Event Bus: Apache Kafka distributed broker with Zookeeper coordination.
- Security Subsystem: Spring Security 6 with stateless JSON Web Tokens (JWT).
- Database Schema Versioning: Flyway automated migration manager.
- Containerization & Orchestration: Docker multi-stage builds and Kubernetes deployments.
- Staging Platform: Railway cloud infrastructure with automated PR preview environments.

## 3. Frontend Architecture and Client Configuration
The frontend user interface is structured as an optimized single-page application (SPA) powered by React 18 and Vite.
The development server is booted using the script `npm run dev` and serves client assets on port 5173 (`http://localhost:5173`).
Production build artifacts are compiled via `npm run build`.
The frontend utilizes an Axios HTTP client instance preconfigured with an API base URL directed at `http://localhost:8080/api/v1`.
State management is handled using Zustand for client-side state and TanStack Query for server-state caching, background revalidation, and optimistic UI mutations.

## 4. Backend Architecture and Layered Design
The Spring Boot application follows strict domain-driven layered architecture:
- Web/Controller Tier: Exposes RESTful endpoints, handles HTTP request validation, and converts payloads.
- Service Tier: Encapsulates transactional business logic, domain invariants, and inter-service dispatch events.
- Persistence Tier: Interfaces with PostgreSQL using Spring Data JPA and Hibernate ORM.
- Data Transfer Tier: Strict Request/Response DTO records to prevent entity leakage.
The Spring Boot backend listens locally on port 8080. Core application settings are externalized into `application.yml` and environment variables.

## 5. Security, Authentication, and Access Control
Security is enforced by Spring Security using stateless JWT bearer tokens.
Clients authenticate via `POST /api/v1/auth/login`. Upon credential verification, the authentication service issues an access token with a 60-minute expiry window alongside a refresh token valid for 7 days.
The `JwtAuthenticationFilter` intercepts each incoming HTTP request, validates the cryptographic signature of the Bearer token in the `Authorization` header, and populates the `SecurityContextHolder`.
Role-based access control (RBAC) enforces granular permissions across three distinct user roles:
- `ROLE_CUSTOMER`: Access to request service quotes, view personal orders, and submit reviews.
- `ROLE_PROVIDER`: Access to accept jobs, update job status, and query payment settlements.
- `ROLE_ADMIN`: Global access across tenant accounts, platform audit logs, and infrastructure metrics.

## 6. Database Architecture and Schema Migrations
PostgreSQL 15 serves as the source of truth for all persistent transactional data.
Database schema evolutions are governed through Flyway versioned migration scripts located under `src/main/resources/db/migration`.
To execute pending migrations outside normal startup, developers run the command `./mvnw flyway:migrate`.
High-performance connection pooling is provided by HikariCP, tuned with a default maximum pool size of 20 connections per container instance.
Database health probes and connection metrics are monitored via Spring Boot Actuator.

## 7. Event-Driven Messaging and Real-Time Communication
UniServe incorporates Apache Kafka to decouple high-volume business workflows from synchronous HTTP request cycles.
Direct bi-directional real-time communication between customers and active field providers is orchestrated through the `MessageController`.
Critical domain events are published to dedicated Kafka event topics:
- `orders.events`: Emitted when new service requests are submitted or modified.
- `dispatches.events`: Emitted when field technicians are matched, routed, or arrive on site.
- `notifications.events`: Consumed by notification microservices to dispatch SMS and email updates.

## 8. Deployment, CI/CD, and Observability
The application repository contains multi-stage Dockerfiles that assemble minimal container images based on Eclipse Temurin 17 Alpine Linux.
Continuous integration workflows run automated unit and integration tests via GitHub Actions on every pull request.
Staging builds are automatically deployed to Railway.
Application health status is accessible via the HTTP endpoint `/actuator/health`.
Application performance telemetry, JVM memory statistics, and Prometheus metrics are exposed at `/actuator/prometheus`.
