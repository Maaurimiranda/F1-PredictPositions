FROM astrocrpublic.azurecr.io/runtime:3.3-4

# include/ contiene el paquete f1/ — agregarlo al PYTHONPATH para que
# los DAGs puedan hacer `from f1.jolpica import ...`.
ENV PYTHONPATH="/usr/local/airflow/include:${PYTHONPATH}"
