# Setup local Python environment

Para trabajar aquí como lo harías en Google Colab, sigue estos pasos:

1. Abre PowerShell en `c:\Users\ADAN\Project`
2. Crea un entorno virtual:

```powershell
python -m venv .venv
```

3. Actívalo:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Actualiza pip e instala dependencias:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Abre VS Code y selecciona el kernel de Python del entorno `.venv` en cualquier notebook `.ipynb`.

6. Para cargar datos, usa rutas locales en lugar de `google.colab`:

```python
import pandas as pd

df = pd.read_csv('data/raw/archivo.csv')
```

## Notas

- Puedes usar `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `torch`, `matplotlib`, etc.
- No todas las funciones de Colab funcionan aquí; los módulos `google.colab` son específicos de la nube.
- Si quieres guardar los paquetes instalados después, ejecuta:

```powershell
pip freeze > requirements.txt
```
