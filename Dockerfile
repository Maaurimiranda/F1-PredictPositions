FROM astrocrpublic.azurecr.io/runtime:3.3-4

# El código del proyecto vive en include/f1 y se importa como `f1`.
# Esto permite que los DAGs puedan hacer, por ejemplo:
# from f1.jolpica import ...
ENV PYTHONPATH="/usr/local/airflow/include:${PYTHONPATH}"

# Este proyecto no requiere Chromium/Playwright para su flujo principal.
# Si en el futuro se agrega scraping por navegador, ahí sí conviene instalar
# dependencias adicionales dentro de la imagen base.
