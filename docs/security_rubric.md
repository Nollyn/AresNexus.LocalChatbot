# Ares-Nexus (NEAI) - Rúbrica de Gobernanza de Seguridad

**Versión de la Política:** `v1.2.0-SEC`  
**Framework:** Ares-Nexus AI Governance (NEAI)  
**Clasificación:** Zero-Trust Autonomous Control Plane  

---

## 1. Propósito y Alcance
Esta rúbrica establece los estándares de seguridad deterministas y de Zero-Trust exigidos para todo código, pipeline agéntico o componente de inferencia evaluado en el marco del ecosistema Ares-Nexus (NEAI).

---

## 2. Dimensiones de Evaluación de Seguridad

### SEC-01: Cero Secretos Hardcodeados y Gestión de Credenciales (Peso: 35%)
- **Criterio:** Queda estrictamente prohibido incluir tokens, API keys (ej. OpenAI, Anthropic, AWS, HuggingFace), contraseñas o certificados en el código fuente o archivos de configuración.
- **Requisitos Obligatorios:**
  - Inyección exclusiva mediante variables de entorno seguras (`os.environ`) o gestores de secretos (ej. GitHub Secrets, AWS Secrets Manager).
  - Archivos `.env`, `.pem` o credenciales deben estar explícitamente ignorados en `.gitignore`.
- **Penalizaciones:**
  - Detección de cualquier secreto o token en texto plano (-1.00 -> Disparo inmediato de `BLOCK` en Circuit Breaker).

### SEC-02: Sanitización de Entradas y Defensas contra Prompt Injection (Peso: 25%)
- **Criterio:** Todo dato proveniente de usuarios o fuentes externas debe ser validado, tipado y sanitizado antes de interactuar con el LLM o con capas del sistema operativo.
- **Requisitos Obligatorios:**
  - Validación de esquemas mediante `Pydantic` o validadores de tipos estrictos.
  - Mitigación de inyecciones de prompt (delimitadores claros, aislamiento de instrucciones del sistema frente a datos de contexto no confiables).
  - Prevención de Path Traversal y Command Injection al interactuar con el sistema de archivos o subprocesos.
- **Penalizaciones:**
  - Inyección directa de entradas no sanitizadas en prompts de sistema o llamadas al OS (-0.50 -> `BLOCK`).
  - Falta de tipado o validación de límites en entradas de usuario (-0.25).

### SEC-03: Principio de Mínimo Privilegio y Ejecución Segura (Peso: 20%)
- **Criterio:** Los agentes autónomos y herramientas auxiliares solo deben poseer los permisos mínimos necesarios para su función.
- **Requisitos Obligatorios:**
  - Prohibición de ejecución de código dinámico no controlado (`eval()`, `exec()`, o subprocesos arbitrarios).
  - Restricción de acceso de solo lectura en herramientas de análisis estático o evaluación.
- **Penalizaciones:**
  - Uso de `eval()` o `exec()` no aislado (-0.80 -> `BLOCK`).
  - Herramientas con privilegios elevados no justificados (-0.30).

### SEC-04: Registro de Auditoría, Trazabilidad e Integridad Criptográfica (Peso: 20%)
- **Criterio:** Cada ejecución del control plane debe generar una traza inmutable y reproducible.
- **Requisitos Obligatorios:**
  - Vinculación de dictámenes al `Commit SHA` del PR y a la versión hash de las políticas aplicadas.
  - Registro de llamadas a herramientas, tiempos de ejecución y dictamen final sin exponer datos sensibles ni secretos.
- **Penalizaciones:**
  - Omisión de metadatos de auditoría (SHA, versión de política, veredicto) (-0.30).
  - Filtración de datos sensibles o PII en logs de auditoría (-0.60 -> `HUMAN_REVIEW` o `BLOCK`).

---

## 3. Umbrales de Puntuación y Mapeo de Decisiones

| Rango de Puntuación | Clasificación | Acción del Decision Gate |
| :--- | :--- | :--- |
| **0.90 - 1.00** | Seguridad Verificada | **PASS** (Supera el Gate de Seguridad) |
| **0.70 - 0.89** | Observaciones Menores / Riesgo Medio | **HUMAN_REVIEW** (Requiere auditoría manual de Seguridad) |
| **0.00 - 0.69** | Vulnerabilidad o Riesgo Alto | **BLOCK** (Rechazo inmediato del PR) |

*Nota Crítica: Cualquier hallazgo en SEC-01 (secreto expuesto) o uso inseguro de `eval()` / `exec()` anula la puntuación y activa automáticamente un veredicto determinista de **BLOCK**.*
