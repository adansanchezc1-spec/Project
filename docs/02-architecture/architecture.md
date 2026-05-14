# Arquitectura del Sistema

**Proyecto**: Sistema de Carga ETL  
**Fase PDCO**: PLAN -> DEVELOPMENT  
**Referencia**: UML del proyecto entregado por el equipo  

## 1. Vision General

El sistema usa una arquitectura MVC en la capa de presentacion y una separacion por capas para aislar dominio, casos de uso e infraestructura.

```text
presentation -> application -> domain <- infrastructure
```

La dependencia principal va desde las capas externas hacia contratos del dominio. Las implementaciones concretas viven en `infrastructure`.

## 2. Capas

| Capa | Responsabilidad | Paquetes |
| --- | --- | --- |
| Presentacion | Entrada/salida con el usuario y coordinacion de solicitudes. | `presentation/views`, `presentation/controllers` |
| Aplicacion | Orquestacion de casos de uso del pipeline ETL. | `application/services` |
| Dominio | Entidades, estados e interfaces estables del negocio. | `domain/entities`, `domain/enums`, `domain/interfaces` |
| Infraestructura | Implementaciones concretas basadas en pandas, archivos y notificaciones. | `infrastructure/*` |
| Pruebas | Validacion automatizada de comportamiento. | `tests` |

## 3. Flujo Principal

1. `AppView` muestra el menu y recibe la ruta del dataset.
2. `DatasetController` recibe la solicitud de procesamiento.
3. `ETLPipelineFacade` coordina el flujo completo.
4. `IngestionService` valida y carga el dataset mediante `IDataRepository`.
5. `CleaningService` aplica estrategias `IDataCleaner`.
6. `FeatureEngineeringService` delega analisis en `IFeatureAnalyzer`.
7. `MDMService` unifica datasets mediante `IMDMService`.
8. La vista muestra resultados o errores.

## 4. Componentes del UML

### Presentacion

- `AppView`
  - `show_menu()`
  - `show_results()`
- `DatasetController`
  - Depende de `ETLPipelineFacade`.
  - Expone `process_dataset(path)` y `get_status(dataset_id)`.

### Aplicacion

- `ETLPipelineFacade`
  - Coordina ingesta, limpieza, feature engineering y MDM.
  - Expone `execute_pipeline(path)`.
- `IngestionService`
  - Depende de `IDataRepository` e `INotificationService`.
- `CleaningService`
  - Ejecuta una lista de estrategias `IDataCleaner`.
- `FeatureEngineeringService`
  - Depende de `IFeatureAnalyzer`.
- `MDMService`
  - Depende de `IMDMService`.

### Dominio

- `Dataset`
- `Feature`
- `CleaningReport`
- `DatasetStatus`
- `IDataRepository`
- `INotificationService`
- `IDataCleaner`
- `IFeatureAnalyzer`
- `IMDMService`

### Infraestructura

- `PandasRepository`
- `EmailNotificationService`
- `NullValueCleaner`
- `FormatCleaner`
- `DuplicateCleaner`
- `DataExplorer`
- `PandasFeatureAnalyzer`
- `PandasMDMService`

## 5. Decisiones Arquitectonicas

| Decision | Justificacion |
| --- | --- |
| Separar interfaces en `domain` | Protege la logica de negocio frente a cambios de pandas, archivos o correo. |
| Usar una fachada de pipeline | Simplifica el caso de uso principal y evita que el controlador conozca todos los servicios. |
| Usar estrategias de limpieza | Permite agregar reglas de limpieza sin modificar el orquestador. |
| Mantener pandas en infraestructura | Evita acoplar entidades del dominio a una libreria especifica. |

## 6. Validacion SOLID

| Principio | Aplicacion |
| --- | --- |
| SRP | Cada servicio tiene una responsabilidad del pipeline. |
| OCP | Nuevos limpiadores pueden agregarse implementando `IDataCleaner`. |
| LSP | Implementaciones concretas deben poder reemplazar interfaces del dominio. |
| ISP | Cada interfaz modela una capacidad puntual. |
| DIP | Servicios de alto nivel dependen de abstracciones, no de detalles concretos. |

