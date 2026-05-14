# Arquitectura

Esta carpeta contiene los artefactos de arquitectura del sistema.

El diseno base sigue el UML entregado para el proyecto, con las capas:

- `presentation`
- `application`
- `domain`
- `infrastructure`

El flujo principal inicia en `AppView`, pasa por `DatasetController`, usa `ETLPipelineFacade` y coordina servicios de ingesta, limpieza, feature engineering y MDM.
