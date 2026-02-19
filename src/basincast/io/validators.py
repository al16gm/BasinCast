import pandas as pd

def normalize_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza el canonical_df:
    - Asegura que la columna 'date' está en el formato correcto.
    - Asegura que 'point_id' es un string.
    - Elimina filas duplicadas o nulas en 'date', 'point_id', y 'value'.
    """
    # Convertir 'date' a datetime, y luego a período mensual
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    
    # Asegurarse que 'point_id' es de tipo str
    df["point_id"] = df["point_id"].astype(str)
    
    # Eliminar valores nulos en 'date', 'point_id' y 'value'
    df = df.dropna(subset=["date", "point_id", "value"]).reset_index(drop=True)
    
    return df

def validate_canonical(df: pd.DataFrame) -> tuple[bool, list]:
    """
    Valida que el canonical_df tiene las columnas necesarias y que los valores clave son correctos.
    Devuelve un tuple (validación_ok, lista_de_errores).
    """
    required_columns = ['date', 'point_id', 'value', 'unit', 'resource_type', 'lat', 'lon', 
                        'precip_mm_month_est', 't2m_c', 'tmax_c', 'tmin_c', 'source', 'demand']
    
    errors = []
    
    # Comprobar que todas las columnas requeridas están presentes
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    
    # Validar si hay puntos con 'NaN' en las columnas clave
    missing_data = df[['date', 'point_id', 'value']].isna().any(axis=1)
    if missing_data.any():
        errors.append(f"Some rows have missing values in required columns.")

    # Validar latitudes y longitudes (si no son NaN)
    if not df[['lat', 'lon']].notna().all().all():
        errors.append("Some latitude or longitude values are missing.")

    return len(errors) == 0, errors