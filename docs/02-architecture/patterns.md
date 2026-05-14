# Patrones y Principios Aplicados

**Proyecto**: Sistema de Carga ETL  
**Fase PDCO**: PLAN -> DEVELOPMENT  

## Patrones Principales

| Patron | Componente | Motivo |
| --- | --- | --- |
| MVC | `AppView`, `DatasetController`, servicios | Separa interaccion, coordinacion y logica del sistema. |
| Facade | `ETLPipelineFacade` | Expone un punto simple para ejecutar todo el pipeline. |
| Repository | `IDataRepository`, `PandasRepository` | Desacopla persistencia y carga de datos. |
| Strategy | `IDataCleaner`, limpiadores concretos | Permite intercambiar reglas de limpieza. |
| Adapter | Implementaciones `Pandas*` | Adapta pandas a contratos propios del dominio. |
| Controller | `DatasetController` | Recibe acciones del usuario y coordina el caso de uso. |
| Polymorphism | Interfaces del dominio | Evita condicionales por tipo de implementacion. |

## GRASP

| Principio | Aplicacion |
| --- | --- |
| Controller | `DatasetController` concentra la entrada del caso de uso. |
| Creator | Servicios crean o actualizan objetos cuando tienen los datos necesarios. |
| Low Coupling | La aplicacion depende de interfaces. |
| High Cohesion | Cada paquete agrupa una responsabilidad clara. |
| Protected Variations | Las interfaces protegen ante cambios de persistencia, limpieza, analitica o MDM. |

## Antipatrones Prevenidos

| Antipatron | Mitigacion |
| --- | --- |
| God Object | El pipeline se divide en servicios especializados. |
| Spaghetti Code | La fachada define una secuencia explicita. |
| Hard Coding | Los detalles concretos quedan en infraestructura. |
| Copy-Paste Programming | Estrategias reutilizables para limpieza. |
| Switch por tipo | Polimorfismo mediante interfaces. |

