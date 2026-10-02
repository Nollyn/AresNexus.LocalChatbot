# Ares-Nexus (NEAI) - Rúbrica de Gobernanza Arquitectónica

**Versión de la Política:** `v1.2.0-ARCH`  
**Framework:** Ares-Nexus AI Governance (NEAI)  
**Clasificación:** Zero-Trust Autonomous Control Plane  

---

## 1. Propósito y Alcance
Esta rúbrica formaliza los criterios de evaluación arquitectónica estricta para todo código o artefacto sometido a Pull Request (PR). El pipeline de evaluación de IA (Evaluator-Optimizer) y los gates deterministas auditan el cumplimiento de estos principios antes de autorizar cualquier despliegue o fusión.

---

## 2. Dimensiones de Evaluación Arquitectónica

### ARCH-01: Separación de Responsabilidades y Clean Architecture (Peso: 25%)
- **Criterio:** El código debe mantener una separación nítida entre capas: Dominio (`domain`), Aplicación (`application`), Infraestructura (`infrastructure`) y Presentación/CLI (`presentation`).
- **Requisitos Obligatorios:**
  - Las entidades de dominio e interfaces no deben depender de frameworks de infraestructura externa (ej. ChromaDB, SDKs de LLM específicos).
  - La inversión de dependencias debe realizarse mediante interfaces abstractas (`Protocol` o `ABC`).
- **Penalizaciones:**
  - Acoplamiento directo en dominio/aplicación a librerías de terceros no abstractas (-0.30).
  - Lógica de negocio mezclada en scripts CLI o endpoints de infraestructura (-0.20).

### ARCH-02: Determinismo en Decisiones y Control de Estado (Peso: 30%)
- **Criterio:** La inteligencia artificial nunca es el sistema de registro ni la autoridad final de aceptación. Las decisiones críticas deben resolverse mediante gates lógicos deterministas.
- **Requisitos Obligatorios:**
  - Toda transición de estado debe modelarse explícitamente (ej. `StateGraph` de LangGraph o máquinas de estado finitas).
  - Estados auditables y tipados con `TypedDict` o esquemas `Pydantic`.
  - Ausencia de efectos secundarios no rastreables durante las mutaciones de estado.
- **Penalizaciones:**
  - Auto-aprobación no supervisada por parte de modelos generativos (-0.50 -> Disparador de `HUMAN_REVIEW` o `BLOCK`).
  - Mutaciones de estado globales o no declaradas (-0.25).

### ARCH-03: Resiliencia, Fallbacks y Circuit Breakers (Peso: 25%)
- **Criterio:** Todo componente de inferencia o comunicación distribuida debe contar con límites de reintento estrictos y mecanismos de salida segura (circuit breakers).
- **Requisitos Obligatorios:**
  - Límite máximo de iteraciones en bucles de optimización / reintento (máximo 2-3 ciclos).
  - Respuestas o fallbacks seguros predeterminados ante fallas de proveedor, timeouts o estancamiento de score.
  - Manejo defensivo de deserialización de respuestas no estructuradas (JSON malformado).
- **Penalizaciones:**
  - Bucles infinitos potenciales o sin condición de parada explícita (-0.60 -> `BLOCK`).
  - Ausencia de manejo de excepciones en clientes de red o LLM (-0.30).

### ARCH-04: Trazabilidad, Observabilidad y Modularidad (Peso: 20%)
- **Criterio:** Cada acción del sistema debe ser rastreable, reproducible y auditable.
- **Requisitos Obligatorios:**
  - Registro estructurado de eventos que incluya `Commit SHA`, versión de política, nodo ejecutor y veredicto.
  - Formato modular reutilizable, permitiendo testing unitario e integración sin dependencias vivas.
- **Penalizaciones:**
  - Falta de logs estructurados o métricas de trazabilidad (-0.20).
  - Funciones monolíticas con alta complejidad ciclomática (> 15) (-0.15).

---

## 3. Umbrales de Puntuación y Mapeo de Decisiones

| Rango de Puntuación | Clasificación | Acción del Decision Gate |
| :--- | :--- | :--- |
| **0.85 - 1.00** | Cumplimiento Alto | **PASS** (Aprobación automática de Fase 2) |
| **0.60 - 0.84** | Requiere Remediación / Revisión | **HUMAN_REVIEW** (Requiere aprobación manual de Arquitecto) |
| **0.00 - 0.59** | Incumplimiento Crítico | **BLOCK** (Rechazo inmediato del PR) |

*Nota: Una violación crítica en ARCH-02 (ej. auto-aprobación del LLM sin gate determinista) o ARCH-03 (bucle sin acotar) forza automáticamente el dictamen a `BLOCK` o `HUMAN_REVIEW` independientemente del score ponderado.*
