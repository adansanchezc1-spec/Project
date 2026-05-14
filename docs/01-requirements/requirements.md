# Especificacion de Requerimientos de Software

**Proyecto**: Sistema de Carga ETL  
**Version**: 1.0  
**Fecha**: 2026-05-14  
**Fase PDCO**: PLAN  
**Estandar guia**: IEEE 830 / ISO 29148  

## 1. Proposito

El sistema debe permitir la carga, validacion, limpieza, transformacion y unificacion de datasets inmobiliarios para producir datos consistentes y listos para analitica posterior.

## 2. Alcance

El alcance funcional cubre un pipeline ETL academico con arquitectura MVC y capas de aplicacion, dominio e infraestructura. El flujo principal inicia en `AppView`, es coordinado por `DatasetController` y ejecutado por `ETLPipelineFacade`.

Fuera de alcance para esta version:

- Entrenamiento de modelos predictivos.
- Despliegue en produccion.
- Integracion con servicios reales de correo externos.
- Persistencia en bases de datos relacionales.

## 3. Actores

| Actor | Descripcion | Interes |
| --- | --- | --- |
| Usuario | Persona que selecciona y procesa datasets. | Obtener datos limpios y listos para analisis. |
| Sistema ETL | Aplicacion que coordina el pipeline. | Validar, transformar y persistir datasets. |
| Servicio de notificacion | Componente que informa eventos del pipeline. | Comunicar errores o finalizacion del proceso. |

## 4. Requerimientos Funcionales

| ID | Descripcion | Prioridad | Entidad/Componente |
| --- | --- | --- | --- |
| RF-001 | El sistema debe permitir seleccionar una ruta de dataset para procesar. | Alta | `AppView`, `DatasetController` |
| RF-002 | El sistema debe validar si la ruta del dataset existe antes de cargarlo. | Alta | `IngestionService`, `IDataRepository` |
| RF-003 | El sistema debe cargar datasets desde archivos soportados mediante un repositorio desacoplado. | Alta | `PandasRepository` |
| RF-004 | El sistema debe registrar el estado del dataset durante el pipeline. | Alta | `Dataset`, `DatasetStatus` |
| RF-005 | El sistema debe ejecutar estrategias de limpieza sobre el dataset. | Alta | `CleaningService`, `IDataCleaner` |
| RF-006 | El sistema debe eliminar o tratar valores nulos. | Media | `NullValueCleaner` |
| RF-007 | El sistema debe normalizar formatos de datos. | Media | `FormatCleaner` |
| RF-008 | El sistema debe eliminar registros duplicados. | Media | `DuplicateCleaner` |
| RF-009 | El sistema debe analizar caracteristicas del dataset. | Media | `FeatureEngineeringService`, `IFeatureAnalyzer` |
| RF-010 | El sistema debe generar estadisticas y perfilado basico del dataset. | Media | `DataExplorer`, `PandasFeatureAnalyzer` |
| RF-011 | El sistema debe unificar datasets mediante operaciones MDM. | Alta | `MDMService`, `IMDMService` |
| RF-012 | El sistema debe guardar el resultado limpio o unificado. | Alta | `IDataRepository` |
| RF-013 | El sistema debe notificar eventos importantes del pipeline. | Media | `INotificationService` |
| RF-014 | El sistema debe exponer una fachada para ejecutar el pipeline completo. | Alta | `ETLPipelineFacade` |
| RF-015 | El sistema debe mostrar resultados del procesamiento al usuario. | Media | `AppView` |

## 5. Requerimientos No Funcionales

| ID | Tipo | Descripcion | Metrica/Criterio |
| --- | --- | --- | --- |
| RNF-001 | Mantenibilidad | El codigo debe separar responsabilidades por capas. | Estructura `presentation`, `application`, `domain`, `infrastructure`. |
| RNF-002 | Extensibilidad | Nuevas estrategias de limpieza deben poder agregarse sin modificar `CleaningService`. | Uso de `IDataCleaner`. |
| RNF-003 | Testabilidad | Los servicios deben depender de interfaces cuando aplique. | Interfaces en `domain/interfaces`. |
| RNF-004 | Calidad | El codigo Python debe seguir PEP 8. | Revision estatica/manual. |
| RNF-005 | Confiabilidad | Los errores de carga o validacion deben dejar el dataset en estado `ERROR`. | Estado controlado por `DatasetStatus`. |
| RNF-006 | Documentacion | Cada fase debe tener artefactos tecnicos en Markdown. | Carpeta `docs` organizada por PDCO. |

## 6. Restricciones

| ID | Descripcion |
| --- | --- |
| R-001 | El proyecto debe implementarse en Python. |
| R-002 | La arquitectura debe corresponder al UML del proyecto. |
| R-003 | El sistema debe aplicar POO, SOLID y patrones Repository, Facade y Controller. |
| R-004 | La persistencia inicial se limita a archivos locales mediante implementaciones de infraestructura. |
| R-005 | Los commits deben realizarse en la rama `dev_asanchez00@unisalle.edu.co`. |

## 7. Criterios de Aceptacion

- El pipeline se puede ejecutar desde el controlador o la fachada.
- Un dataset invalido produce notificacion o estado de error.
- Un dataset valido pasa por ingesta, limpieza, analisis y MDM.
- Las clases del dominio no dependen de pandas directamente.
- Las implementaciones concretas quedan aisladas en `infrastructure`.

