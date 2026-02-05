"""Data loading functions for historical and forecasted sales."""
import logging
import pandas as pd
import streamlit as st
from pathlib import Path

logger = logging.getLogger(__name__)


def load_historical_data() -> pd.DataFrame:
    """
    Load historical sales data from CSV file.

    :return: DataFrame with historical sales data
    """
    try:
        csv_path = Path(__file__).parent / "sales.csv"

        if not csv_path.exists():
            error_msg = f"CSV file not found: {csv_path}"
            logger.error(error_msg)
            st.error(error_msg)
            return pd.DataFrame()

        df = pd.read_csv(csv_path, low_memory=False)

        columns_needed = [
            'product_id', 'product_name', 'year', 'calendar_week',
            'sales_t0', 'offer_volume_provision', 'offer_volume_provision_lead1'
        ]

        df = df[columns_needed].copy()
        df.rename(columns={
            'offer_volume_provision': 'offer_volume_prov',
            'offer_volume_provision_lead1': 'offer_volume_lead1'
        }, inplace=True)

        df['product_id'] = df['product_id'].astype(str)

        if not df.empty:
            df['sales_t0'] = pd.to_numeric(df['sales_t0'], errors='coerce')
            df['offer_volume_prov'] = pd.to_numeric(
                df['offer_volume_prov'],
                errors='coerce'
            )
            df['offer_volume_lead1'] = pd.to_numeric(
                df['offer_volume_lead1'],
                errors='coerce'
            )

        return df
    except Exception as e:
        error_msg = f"Error loading historical data: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        st.exception(e)
        return pd.DataFrame()


def _transform_forecasts_to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform forecasts from long to wide format.

    :param df: Long format with forecasted_at_date, forecast_week,
               product_id, quantity
    :return: Wide format with product_id, year, calendar_week, t0-t5
    """
    if df.empty:
        return pd.DataFrame()

    df['forecast_week'] = pd.to_datetime(df['forecast_week'])
    df['forecasted_at_date'] = pd.to_datetime(df['forecasted_at_date'])

    results = []

    for (product_id, forecasted_at), group in df.groupby(
        ['product_id', 'forecasted_at_date']
    ):
        group = group.sort_values('forecast_week')

        if len(group) < 6:
            continue

        first_week = group.iloc[0]
        year = first_week['forecast_week'].isocalendar()[0]
        calendar_week = first_week['forecast_week'].isocalendar()[1]

        row = {
            'product_id': str(product_id),
            'year': year,
            'calendar_week': calendar_week
        }

        for i, (_, forecast_row) in enumerate(group.head(6).iterrows()):
            row[f't{i}'] = float(forecast_row['quantity'])
            row[f'mae_t{i}'] = None
            row[f'mape_t{i}'] = None
            row[f'mse_t{i}'] = None
            row[f'rmse_t{i}'] = None
            row[f'r2_t{i}'] = None

        results.append(row)

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        for i in range(6):
            result_df[f't{i}'] = pd.to_numeric(
                result_df[f't{i}'],
                errors='coerce'
            )

    return result_df


def load_predictions_data() -> pd.DataFrame:
    """
    Load forecasted sales data from CSV file.

    :return: DataFrame with forecasted sales data in wide format
    """
    try:
        csv_path = Path(__file__).parent / "forecasted_sales.csv"

        if not csv_path.exists():
            error_msg = f"CSV file not found: {csv_path}"
            logger.error(error_msg)
            st.error(error_msg)
            return pd.DataFrame()

        df = pd.read_csv(csv_path)
        df = _transform_forecasts_to_wide_format(df)
        return df
    except Exception as e:
        error_msg = f"Error loading predictions data: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        st.exception(e)
        return pd.DataFrame()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing historical data load")
    hist_df = load_historical_data()
    logger.info(f"Historical Data Shape: {hist_df.shape}")
    logger.info(f"Historical Data Columns: {hist_df.columns.tolist()}")
    logger.info(f"Historical Data Head:\n{hist_df.head()}")

    logger.info("\nTesting predictions data load")
    pred_df = load_predictions_data()
    logger.info(f"Predictions Data Shape: {pred_df.shape}")
    logger.info(f"Predictions Data Columns: {pred_df.columns.tolist()}")
    logger.info(f"Predictions Data Head:\n{pred_df.head()}")
