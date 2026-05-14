# Casos de Uso

**Proyecto**: Sistema de Carga ETL  
**Fase PDCO**: PLAN  

## UC-001: Procesar Dataset

- **Actor principal**: Usuario
- **Componentes**: `AppView`, `DatasetController`, `ETLPipelineFacade`
- **Precondicion**: El usuario conoce la ruta del dataset.
- **Flujo principal**:
  1. El usuario selecciona o ingresa la ruta del dataset.
  2. `AppView` envia la ruta a `DatasetController`.
  3. `DatasetController` delega el procesamiento en `ETLPipelineFacade`.
  4. La fachada ejecuta ingesta, limpieza, feature engineering y MDM.
  5. El sistema retorna el resultado del procesamiento.
- **Flujo alternativo**:
  - Si la ruta no existe, el sistema detiene el proceso, notifica el error y marca el dataset como `ERROR`.
- **Postcondicion**: El dataset queda procesado o con estado de error.
- **Requerimientos relacionados**: RF-001, RF-014, RF-015.

## UC-002: Ingestar Dataset

- **Actor principal**: Sistema ETL
- **Componentes**: `IngestionService`, `IDataRepository`, `PandasRepository`
- **Precondicion**: Existe una ruta entregada por el controlador.
- **Flujo principal**:
  1. `IngestionService` valida si la ruta existe.
  2. El repositorio carga el archivo.
  3. El sistema crea o actualiza la entidad `Dataset`.
  4. El dataset cambia a estado `STORED`.
- **Flujo alternativo**:
  - Si el archivo no existe, se notifica el error y no continua el pipeline.
- **Postcondicion**: Dataset cargado y disponible para limpieza.
- **Requerimientos relacionados**: RF-002, RF-003, RF-004.

## UC-003: Limpiar Dataset

- **Actor principal**: Sistema ETL
- **Componentes**: `CleaningService`, `IDataCleaner`, `NullValueCleaner`, `FormatCleaner`, `DuplicateCleaner`
- **Precondicion**: El dataset fue cargado correctamente.
- **Flujo principal**:
  1. `CleaningService` recibe un dataframe.
  2. Ejecuta cada estrategia de limpieza registrada.
  3. Genera datos limpios para las siguientes etapas.
- **Flujo alternativo**:
  - Si una estrategia falla, el sistema detiene el flujo y registra el error.
- **Postcondicion**: Dataset limpio o pipeline detenido por error.
- **Requerimientos relacionados**: RF-005, RF-006, RF-007, RF-008.

## UC-004: Analizar Caracteristicas

- **Actor principal**: Sistema ETL
- **Componentes**: `FeatureEngineeringService`, `IFeatureAnalyzer`, `PandasFeatureAnalyzer`, `DataExplorer`
- **Precondicion**: Existe un dataset limpio.
- **Flujo principal**:
  1. El servicio recibe el dataframe limpio.
  2. El analizador genera estadisticas y perfilado.
  3. El sistema identifica o descarta caracteristicas irrelevantes.
- **Postcondicion**: Dataset preparado para MDM o analitica posterior.
- **Requerimientos relacionados**: RF-009, RF-010.

## UC-005: Unificar Datos con MDM

- **Actor principal**: Sistema ETL
- **Componentes**: `MDMService`, `IMDMService`, `PandasMDMService`
- **Precondicion**: Existe uno o mas datasets transformados.
- **Flujo principal**:
  1. `MDMService` recibe datasets candidatos.
  2. Aplica estandarizacion de estructura.
  3. Elimina duplicados.
  4. Fusiona los datasets en una salida consistente.
- **Postcondicion**: Dataset maestro generado.
- **Requerimientos relacionados**: RF-011, RF-012.

## UC-006: Notificar Evento del Pipeline

- **Actor principal**: Sistema ETL
- **Componentes**: `INotificationService`, `EmailNotificationService`
- **Precondicion**: Ocurre un evento relevante durante el pipeline.
- **Flujo principal**:
  1. El servicio genera un mensaje.
  2. La implementacion de notificacion envia o registra el mensaje.
- **Eventos**:
  - Ruta invalida.
  - Dataset guardado.
  - Limpieza completada.
  - Pipeline finalizado.
- **Requerimientos relacionados**: RF-013.

