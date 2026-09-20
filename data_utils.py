import pandas as pd


def get_schema_description(df: pd.DataFrame, max_rows: int = 3) -> str:
    schema_lines = ["Columns:"]
    for col, dtype in df.dtypes.items():
        if dtype == 'object' and df[col].nunique() < 20:
            unique_vals = df[col].dropna().unique()[:5]
            schema_lines.append(f"- {col} ({dtype}) - sample values: {', '.join(map(str, unique_vals))}")
        else:
            schema_lines.append(f"- {col} ({dtype})")
    sample = df.head(max_rows).to_markdown(index=False)
    return "\n".join(schema_lines) + "\n\nSample rows:\n" + sample


def preprocess_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_columns = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue
        if any(x in col.lower() for x in ['date', 'time', 'day']):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if df[col].notna().sum() > 0:
                    date_columns.append(col)
            except Exception:
                pass
    for col in date_columns:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_year_month'] = df[col].dt.to_period('M').astype(str)
    return df


def apply_filters(df: pd.DataFrame, filters: list) -> pd.DataFrame:
    if not filters:
        return df
    filtered = df.copy()
    for f in filters:
        col = f.get("column")
        op = f.get("op")
        val = f.get("value")
        if col not in filtered.columns:
            continue

        # Cast the filter value to match the column's dtype so comparisons
        # actually match instead of silently returning zero rows.
        if pd.api.types.is_datetime64_any_dtype(filtered[col]):
            val_cast = pd.to_datetime(val, errors='coerce')
        else:
            try:
                val_cast = pd.to_numeric(val)
            except Exception:
                val_cast = val

        if op == "==":
            filtered = filtered[filtered[col] == val_cast]
        elif op == "!=":
            filtered = filtered[filtered[col] != val_cast]
        elif op == ">":
            filtered = filtered[filtered[col] > val_cast]
        elif op == "<":
            filtered = filtered[filtered[col] < val_cast]
        elif op == ">=":
            filtered = filtered[filtered[col] >= val_cast]
        elif op == "<=":
            filtered = filtered[filtered[col] <= val_cast]
        elif op == "contains":
            filtered = filtered[filtered[col].astype(str).str.contains(str(val_cast), case=False, na=False)]
    return filtered


def run_analysis_plan(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    op = plan.get("operation")
    target_col = plan.get("target_column")
    group_by = plan.get("group_by") or []
    filters = plan.get("filters") or []
    metric = plan.get("metric", "sum")

    if filters:
        df = apply_filters(df, filters)

    if not op or not target_col or target_col not in df.columns:
        return df.head(20)

    # Drop any group_by columns the LLM invented that aren't in the schema,
    # instead of letting groupby() raise a raw KeyError downstream.
    group_by = [g for g in group_by if g in df.columns]

    if op in ("aggregate_compare", "group_by_summary", "filter_then_aggregate"):
        if not group_by:
            if metric == "sum":
                value = df[target_col].sum()
            elif metric == "mean":
                value = df[target_col].mean()
            elif metric == "count":
                value = df[target_col].count()
            elif metric == "max":
                value = df[target_col].max()
            elif metric == "min":
                value = df[target_col].min()
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            return pd.DataFrame({f"{metric}_{target_col}": [value]})

        if metric == "sum":
            agg_df = df.groupby(group_by)[target_col].sum().reset_index()
        elif metric == "mean":
            agg_df = df.groupby(group_by)[target_col].mean().reset_index()
        elif metric == "count":
            agg_df = df.groupby(group_by)[target_col].count().reset_index()
        elif metric == "max":
            agg_df = df.groupby(group_by)[target_col].max().reset_index()
        elif metric == "min":
            agg_df = df.groupby(group_by)[target_col].min().reset_index()
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        agg_df = agg_df.sort_values(by=target_col, ascending=False)
        return agg_df

    return df.head(20)