import streamlit as st
import io
import numpy as np
import time
st.set_page_config(page_title="Absatzprognose Tool", page_icon="📈")
import pandas as pd
import plotly.graph_objects as go
from load_sheet_data import load_historical_data, load_predictions_data

# Initialize session state variables
if 'product_id' not in st.session_state:
    st.session_state.product_id = None
if 'hist_year' not in st.session_state:
    st.session_state.hist_year = '2024'
if 'start_week' not in st.session_state:
    st.session_state.start_week = '1'
if 'pred_year' not in st.session_state:
    st.session_state.pred_year = '2024'
if 'prediction_week' not in st.session_state:
    st.session_state.prediction_week = '1'
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None
if 'adjusted_predictions' not in st.session_state:
    st.session_state.adjusted_predictions = None
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = '6-Wochen-Prognose'  # Default view mode
if 'set_to_today' not in st.session_state:
    st.session_state.set_to_today = False


@st.cache_data(ttl=3600, show_spinner="Lade Daten...")
def load_data():
    historical_df = load_historical_data()
    predictions_df = load_predictions_data()
    return historical_df, predictions_df


# Load both dataframes
hist_df, pred_df = load_data()

# # ============================================================================
# # DEBUG SECTION - Shows data structure at the very top
# # ============================================================================
# st.title('Absatzprognose-Tool')

# st.write("---")
# st.subheader("🔍 Debug Information")
# st.write("**Historical DataFrame:**")
# st.write(f"Shape: {hist_df.shape}")
# st.write(f"Columns: {hist_df.columns.tolist()}")
# if not hist_df.empty:
#     st.dataframe(hist_df.head(3))
# else:
#     st.error("⚠️ Historical DataFrame is EMPTY! Check the error messages above.")

# st.write("**Predictions DataFrame:**")
# st.write(f"Shape: {pred_df.shape}")
# st.write(f"Columns: {pred_df.columns.tolist()}")
# if not pred_df.empty:
#     st.dataframe(pred_df.head(3))
# else:
#     st.error("⚠️ Predictions DataFrame is EMPTY!")
# st.write("---")

# # ============================================================================


# Get unique products that have predictions
@st.cache_data(ttl=3600)
def get_products_with_predictions():
    # Get unique product IDs from the prediction dataframe
    products_with_pred = pred_df['product_id'].unique()
    # Sort them based on their first appearance in the prediction dataframe
    sorted_products = []
    for product in products_with_pred:
        first_appearance = pred_df[pred_df['product_id'] == product].index.min()
        sorted_products.append((product, first_appearance))
    sorted_products.sort(key=lambda x: x[1])  # Sort by first appearance index
    return [p[0] for p in sorted_products]  # Return just the product IDs in order


def get_product_name(product_id):
    product_data = hist_df[hist_df['product_id'] == product_id]
    if not product_data.empty and 'product_name' in product_data.columns:
        return product_data['product_name'].iloc[0]
    return "Unknown Product"


def get_actual_sales(product_id, start_year, start_week, end_year, end_week):
    if end_week > 52:
        end_year += 1
        end_week -= 52

    product_data = hist_df[(hist_df['product_id'] == product_id) &
                           (((hist_df['year'] == start_year) & (hist_df['calendar_week'] >= start_week)) |
                            (hist_df['year'] > start_year) & (hist_df['year'] <= end_year) |
                            ((hist_df['year'] == end_year) & (hist_df['calendar_week'] <= end_week)))]
    return product_data[['year', 'calendar_week', 'sales_t0', 'offer_volume_prov']].dropna()


def get_predictions(product_id, year, prediction_week):
    pred_data = pred_df[
        (pred_df['product_id'] == product_id) &
        (pred_df['year'] == year) &
        (pred_df['calendar_week'] == prediction_week)
        ]

    if pred_data.empty:
        st.warning("Keine Prognosen für diese Kombination verfügbar.")
        return [], [], [], [], [], [], [], [], []

    pred_weeks = range(prediction_week, prediction_week + 6)
    pred_values = [pred_data[f't{i}'].iloc[0] for i in range(6)]
    mse_values = [pred_data[f'mse_t{i}'].iloc[0] for i in range(6)]
    rmse_values = [pred_data[f'rmse_t{i}'].iloc[0] for i in range(6)]
    mae_values = [pred_data[f'mae_t{i}'].iloc[0] for i in range(6)]
    r2_values = [pred_data[f'r2_t{i}'].iloc[0] for i in range(6)]
    mape_values = [pred_data[f'mape_t{i}'].iloc[0] for i in range(6)]

    # Get previous year sales
    prev_year_sales = []
    pred_offer_volume = []
    for week in pred_weeks:
        adj_week = week
        adj_year = year
        if week > 52:
            adj_week -= 52
            adj_year += 1

        # Fix: Check if filtered DataFrame is empty before accessing .iloc[0]
        filtered_prev = hist_df[
            (hist_df['product_id'] == product_id) &
            (hist_df['year'] == adj_year - 1) &
            (hist_df['calendar_week'] == adj_week)
        ]
        prev_year = filtered_prev['sales_t0'].iloc[0] if not filtered_prev.empty else None

        filtered_offer = hist_df[
            (hist_df['product_id'] == product_id) &
            (hist_df['year'] == adj_year) &
            (hist_df['calendar_week'] == adj_week)
        ]
        offer_vol = filtered_offer['offer_volume_lead1'].iloc[0] if not filtered_offer.empty else None

        prev_year_sales.append(prev_year)
        pred_offer_volume.append(offer_vol)

    return list(
        pred_weeks), pred_values, mse_values, rmse_values, mae_values, r2_values, mape_values, prev_year_sales, pred_offer_volume


# New function to get all t0 predictions for a product
def get_t0_prediction_timeline(product_id):
    product_predictions = pred_df[pred_df['product_id'] == product_id].copy()
    
    if product_predictions.empty:
        st.warning(f"Keine Prognosen für Produkt {product_id} verfügbar.")
        return None
    
    # Create a date column for sorting and visualization
    product_predictions['date'] = pd.to_datetime(
        product_predictions['year'].astype(str) + '-' + 
        product_predictions['calendar_week'].astype(str) + '-1', 
        format='%Y-%W-%w')
    
    # Sort by date for proper timeline visualization
    product_predictions = product_predictions.sort_values('date')
    
    # Add t0_target_date column (the date for which the t0 prediction is made)
    product_predictions['t0_target_date'] = product_predictions['date']
    
    # Add columns for t1-t5 target dates
    for i in range(1, 6):
        product_predictions[f't{i}_target_date'] = product_predictions['date'] + pd.Timedelta(weeks=i)
    
    return product_predictions


# ============================================================================
# SIDEBAR AND MAIN UI
# ============================================================================
st.sidebar.image('logo_garmo.png', width=150)
st.sidebar.header('Eingabe')

# Add view mode selector at the top of the sidebar
view_mode = st.sidebar.radio(
    "Ansichtsmodus",
    options=["6-Wochen-Prognose", "Prognose-Timeline"],
    key="view_mode"
)

# Only show products that have predictions, in the same order
products_with_pred = get_products_with_predictions()
product_options = [(pid, get_product_name(pid)) for pid in products_with_pred]
product_labels = [f"{name} (ID: {pid})" for pid, name in product_options]

product_index = 0
if st.session_state.product_id in products_with_pred:
    product_index = products_with_pred.index(st.session_state.product_id)

product_id = st.sidebar.selectbox(
    'Produkt ID auswählen',
    options=products_with_pred,
    format_func=lambda x: x,  
    index=product_index,
    key='product_id'
)

if view_mode == "6-Wochen-Prognose":
    if st.session_state.set_to_today:
        from datetime import datetime
        today = datetime.now()
        st.session_state.pred_year = str(today.isocalendar()[0])
        st.session_state.prediction_week = str(today.isocalendar()[1])
        st.session_state.set_to_today = False

    st.sidebar.write("Start Beobachtungszeitraum")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        hist_year = st.text_input('Jahr',
                                 value=str(int(hist_df['year'].max()) - 1),
                                 key='hist_year')
    st.sidebar.write("Prognosezeitpunkt")
    with col2:
        start_week = st.text_input('KW',
                                   value=st.session_state.start_week,
                                   key='start_week')

    col3, col4 = st.sidebar.columns(2)
    with col3:
        pred_year = st.text_input('Jahr',
                                  value=str(int(hist_df['year'].max())) if st.session_state.pred_year is None
                                  else st.session_state.pred_year,
                                  key='pred_year')

    with col4:
        prediction_week = st.text_input('KW',
                                        value=st.session_state.prediction_week,
                                        key='prediction_week')

    if st.sidebar.button('📅 Heute', help='Auf aktuelle KW setzen'):
        st.session_state.set_to_today = True
        st.rerun()

    start_button = st.sidebar.button('Start')
else:  # Timeline view - No inputs needed for timeline mode
    start_button = st.sidebar.button('Timeline anzeigen')

if st.sidebar.button("🔄"):
    # Clear all caches
    st.cache_data.clear()
    # Reset session state
    for key in ['predictions_df', 'adjusted_predictions', 'current_selection']:
        if key in st.session_state:
            del st.session_state[key]
    st.success("Cache wurde geleert!")
    time.sleep(1)
    st.rerun()

# Process the input based on the view mode
if start_button:
    if view_mode == "6-Wochen-Prognose":
        try:
            hist_year = int(hist_year)
            start_week = int(start_week)
            pred_year = int(pred_year)
            prediction_week = int(prediction_week)
        except ValueError:
            st.error("Bitte geben Sie für Jahr und KW nur ganze Zahlen ein.")
            st.stop()

        if not (1 <= start_week <= 53) or not (1 <= prediction_week <= 53):
            st.error("KW muss zwischen 1 und 53 liegen.")
            st.stop()

        if hist_year < int(hist_df['year'].min()) or pred_year > int(hist_df['year'].max()) + 1:
            st.error(f"Jahr muss zwischen {int(hist_df['year'].min())} und {int(hist_df['year'].max()) + 1} liegen.")
            st.stop()

        # Store the current state
        st.session_state.current_selection = {
            'product_id': product_id,
            'hist_year': hist_year,
            'start_week': start_week,
            'pred_year': pred_year,
            'prediction_week': prediction_week,
            'view_mode': '6-Wochen-Prognose'
        }
        # Reset predictions DataFrame when clicking Start
        st.session_state.predictions_df = None
    
    else:  # Timeline view - Simply store the product ID
        # Store the current timeline selection - no date range needed
        st.session_state.current_selection = {
            'product_id': product_id,
            'view_mode': 'Prognose-Timeline'
        }

# Only show visualizations if we have a current selection
if hasattr(st.session_state, 'current_selection'):
    current = st.session_state.current_selection
    product_name = get_product_name(current['product_id'])
    
    # Render visualization based on view mode
    if current.get('view_mode') == '6-Wochen-Prognose':
        end_week = current['prediction_week']
        end_year = current['pred_year']

        pred_weeks, pred_values, mse_values, rmse_values, mae_values, r2_values, mape_values, prev_year_sales, pred_offer_volume = get_predictions(
            current['product_id'], current['pred_year'], current['prediction_week'])

        if pred_weeks:
            end_week = pred_weeks[-1]
            if end_week > 52:
                end_year += 1
                end_week -= 52

        actual_sales = get_actual_sales(current['product_id'], current['hist_year'], current['start_week'], end_year, end_week)

        fig = go.Figure()

        actual_sales['date'] = pd.to_datetime(
            actual_sales['year'].astype(str) + '-' + actual_sales['calendar_week'].astype(str) + '-1', format='%Y-%W-%w')
        actual_sales = actual_sales.sort_values('date')

        start_date = pd.to_datetime(f"{current['hist_year']}-{current['start_week']}-1", format='%Y-%W-%w')
        end_date = pd.to_datetime(f"{end_year}-{end_week}-1", format='%Y-%W-%w')

        num_weeks = (end_date - start_date).days // 7 + 1

        pred_dates = []
        if pred_weeks:
            for week in pred_weeks:
                year = current['pred_year']
                if week > 52:
                    year += 1
                    week -= 52
                pred_dates.append(pd.to_datetime(f"{year}-{week}-1", format='%Y-%W-%w'))

        fig.add_trace(go.Scatter(
            x=actual_sales['date'],
            y=actual_sales['sales_t0'],
            mode='lines+markers',
            name='Absätze (tatsächlich)',
            line=dict(color='blue'),
            hovertemplate='Jahr: %{text}<br>KW: %{customdata}<br>Absatz: %{y:.2f}',
            text=actual_sales['year'],
            customdata=actual_sales['calendar_week']
        ))

        if pred_weeks and pred_values:
            # If we have adjusted predictions, use those instead of the original pred_values
            if st.session_state.predictions_df is not None:
                pred_values = st.session_state.predictions_df['Prognose'].tolist()

            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=pred_values,
                mode='lines+markers',
                name='Absatzprognose Normalmenge',
                line=dict(color='red', dash='dash'),
                hovertemplate='Jahr: %{text}<br>KW: %{customdata}<br>Prognose: %{y:.2f}',
                text=[d.year for d in pred_dates],
                customdata=[d.isocalendar()[1] for d in pred_dates]
            ))

            # Only show total predictions where there is an Aktionsmenge
            total_predictions = []
            total_pred_dates = []
            total_pred_years = []
            total_pred_weeks = []

            for i, (p, o) in enumerate(zip(pred_values, pred_offer_volume)):
                if o is not None and o > 0:
                    total_predictions.append(p + o)
                    total_pred_dates.append(pred_dates[i])
                    total_pred_years.append(pred_dates[i].year)
                    total_pred_weeks.append(pred_dates[i].isocalendar()[1])

            if total_predictions:  # Only add trace if there are total predictions to show
                fig.add_trace(go.Scatter(
                    x=total_pred_dates,
                    y=total_predictions,
                    mode='markers',
                    name='Absatzprognose Gesamt',
                    marker=dict(
                        size=8,
                        symbol='diamond',
                        color='#FF69B4',
                        line=dict(color='DarkSlateGrey')
                    ),
                    hovertemplate='Jahr: %{text}<br>KW: %{customdata}<br>Gesamtprognose: %{y:.2f}',
                    text=total_pred_years,
                    customdata=total_pred_weeks
                ))

        all_offer_volumes = pd.concat([
            actual_sales[['date', 'offer_volume_prov', 'year', 'calendar_week']],
            pd.DataFrame({'date': pred_dates, 'offer_volume_prov': pred_offer_volume,
                          'year': [d.year for d in pred_dates], 'calendar_week': [d.isocalendar()[1] for d in pred_dates]})
        ])
        non_zero_offers = all_offer_volumes[all_offer_volumes['offer_volume_prov'] > 0]

        if not non_zero_offers.empty:
            fig.add_trace(go.Scatter(
                x=non_zero_offers['date'],
                y=non_zero_offers['offer_volume_prov'],
                mode='markers',
                name='Aktionsmenge',
                marker=dict(
                    size=8,
                    symbol='diamond',
                    color='#66C2A5',
                    line=dict(color='DarkSlateGrey')
                ),
                hovertemplate='Jahr: %{text}<br>KW: %{customdata}<br>Aktionsmenge: %{y:.2f}',
                text=non_zero_offers['year'],
                customdata=non_zero_offers['calendar_week']
            ))

        st.subheader(f" {product_name} (ID: {current['product_id']})")
        fig.update_layout(
            title=f'Absätze und Prognosen für Produkt {current["product_id"]}',
            xaxis_title='Jahr-KW',
            yaxis_title='Absätze',
            legend_title='Datentypen',
            hovermode="x unified"
        )

        if num_weeks <= 10:
            custom_ticks = [start_date + pd.Timedelta(weeks=i) for i in range(num_weeks)]
            fig.update_xaxes(
                tickmode='array',
                tickvals=custom_ticks,
                ticktext=[d.strftime('%Y-KW%V') for d in custom_ticks],
                range=[start_date, end_date]
            )
        else:
            fig.update_xaxes(
                range=[start_date, end_date]
            )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Prognostizierte Absätze")
        if pred_weeks:
            if st.session_state.predictions_df is None:
                actual_sales_pred_period = get_actual_sales(current['product_id'], current['pred_year'],
                                                          current['prediction_week'], end_year, end_week)
                actual_sales_dict = dict(zip(actual_sales_pred_period['calendar_week'],
                                           actual_sales_pred_period['sales_t0']))

                adjusted_pred_weeks = []
                actual_values = []
                diff_values = []
                diff_percent_values = []
                for week in pred_weeks:
                    adjusted_week = week if week <= 52 else week - 52
                    adjusted_pred_weeks.append(f"{adjusted_week:02d}")

                    actual_sale = actual_sales_dict.get(adjusted_week, None)
                    actual_values.append(actual_sale)

                    if actual_sale is not None and pred_values[len(diff_values)] is not None:
                        diff = actual_sale - pred_values[len(diff_values)]
                        diff_values.append(diff)
                        diff_percent = (diff / actual_sale) * 100 if actual_sale != 0 else None
                        diff_percent_values.append(diff_percent)
                    else:
                        diff_values.append(None)
                        diff_percent_values.append(None)

                st.session_state.predictions_df = pd.DataFrame({
                    'KW': adjusted_pred_weeks,
                    'Vorjahr': prev_year_sales,
                    'Aktionsmenge': pred_offer_volume,
                    'Prognose': pred_values,
                    'Absätze Normalmenge (tatsächlich)': actual_values,
                    'Differenz (%)': diff_percent_values,
                    'MAE': mae_values,
                    'MAPE': mape_values
                }).round(2)
                st.session_state.predictions_df['MAPE'] = st.session_state.predictions_df['MAPE'].replace(
                    [float('inf'), float('-inf')], 'n/a')

            def format_or_na(value, format_str):
                if pd.isna(value) or value is None:
                    return 'n/a'
                return format_str.format(value)

            # Display single table
            styled_df = st.session_state.predictions_df.style.format({
                "KW": "{}",
                "Vorjahr": lambda x: format_or_na(x, "{:.2f}"),
                "Aktionsmenge": lambda x: format_or_na(x, "{:.2f}"),
                "Absätze Normalmenge (tatsächlich)": lambda x: format_or_na(x, "{:.2f}"),
                "Prognose": lambda x: format_or_na(x, "{:.2f}"),
                "Differenz (%)": lambda x: format_or_na(x, "{:.2f}%"),
                "MAE": lambda x: format_or_na(x, "{:.2f}"),
                "MAPE": lambda x: format_or_na(x, "{:.2f}") if isinstance(x, (int, float)) else x
            })
            st.dataframe(styled_df, hide_index=True)


            if pred_weeks:
                # Get product details
                product_name = get_product_name(current['product_id'])

                # Get historical data for the selected period, but only up to the last prediction week
                last_pred_week = pred_weeks[-1]
                last_pred_year = current['pred_year']
                if last_pred_week > 52:
                    last_pred_week -= 52
                    last_pred_year += 1

                historical_data = hist_df[
                    (hist_df['product_id'] == current['product_id']) &
                    (hist_df['year'] == current['pred_year']) &
                    (hist_df['calendar_week'] <= last_pred_week)
                ].copy()

                # Create base dataframe with all required columns
                export_df = pd.DataFrame({
                    'Produkt ID': historical_data['product_id'],
                    'Produktname': product_name,
                    'Jahr': historical_data['year'],
                    'KW': historical_data['calendar_week'],
                    'Tatsächlicher Absatz': historical_data['sales_t0'],
                    'Aktionsmenge': historical_data['offer_volume_prov'],
                    'Prognostizierte Normalmenge': None,  # Will be filled with predictions
                    'Absatzprognose Gesamt': None  # Will be filled with total predictions
                })

                # Add month column based on calendar week
                export_df['Monat'] = export_df['KW'].apply(lambda x: ((x-1) // 4) + 1)

                # Sort columns in the desired order
                export_df = export_df[['Produkt ID', 'Produktname', 'Jahr', 'Monat', 'KW',
                                      'Tatsächlicher Absatz', 'Prognostizierte Normalmenge',
                                      'Aktionsmenge', 'Absatzprognose Gesamt']]

                # Add predictions to the dataframe
                for week, pred, offer in zip(pred_weeks, pred_values, pred_offer_volume):
                    adjusted_week = week if week <= 52 else week - 52
                    adjusted_year = current['pred_year'] if week <= 52 else current['pred_year'] + 1

                    # Check if the row exists, if not add it
                    if not any(export_df['KW'] == adjusted_week):
                        new_row = pd.DataFrame({
                            'Produkt ID': [current['product_id']],
                            'Produktname': [product_name],
                            'Jahr': [adjusted_year],
                            'Monat': [((adjusted_week-1) // 4) + 1],
                            'KW': [adjusted_week],
                            'Tatsächlicher Absatz': [None],
                            'Prognostizierte Normalmenge': [pred],
                            'Aktionsmenge': [offer],
                            'Absatzprognose Gesamt': [pred + offer if offer and offer > 0 else pred]
                        })
                        export_df = pd.concat([export_df, new_row], ignore_index=True)
                    else:
                        mask = export_df['KW'] == adjusted_week
                        export_df.loc[mask, 'Prognostizierte Normalmenge'] = pred
                        if offer and offer > 0:
                            export_df.loc[mask, 'Absatzprognose Gesamt'] = pred + offer
                        else:
                            export_df.loc[mask, 'Absatzprognose Gesamt'] = pred

                # Sort by calendar week
                export_df = export_df.sort_values(['Jahr', 'KW'])

                # Download button
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f'prognose_{current["product_id"]}_{current["pred_year"]}_KW{current["prediction_week"]}.csv',
                    mime='text/csv'
                )

            # Adjustment interface
            st.subheader("Prognose anpassen")
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                selected_week = st.selectbox('KW auswählen', options=st.session_state.predictions_df['KW'].tolist())

            with col2:
                adjustment_type = st.selectbox('Anpassungsart', options=['Absolut', 'Prozentual'])

            with col3:
                adjustment_value = st.text_input('Wert', value="0")

            if st.button('Anpassen'):
                try:
                    adj_value = float(adjustment_value)
                    week_idx = st.session_state.predictions_df[
                        st.session_state.predictions_df['KW'] == selected_week].index[0]
                    original_value = st.session_state.predictions_df.loc[week_idx, 'Prognose']

                    if adjustment_type == 'Absolut':
                        new_value = original_value + adj_value
                    else:
                        percentage_factor = 1 + (adj_value / 100)
                        new_value = original_value * percentage_factor

                    st.session_state.predictions_df.loc[week_idx, 'Prognose'] = new_value
                    st.rerun()

                except ValueError:
                    st.error("Bitte geben Sie eine gültige Zahl ein.")
    
    else:  # Timeline View
        # Get t0 predictions timeline
        timeline_data = get_t0_prediction_timeline(current['product_id'])
        
        if timeline_data is not None:
            # No filtering by date range - show all available data
            if timeline_data.empty:
                st.warning(f"Keine Prognosedaten für dieses Produkt verfügbar.")
            else:
                st.subheader(f"Prognose-Timeline für {product_name} (ID: {current['product_id']})")
                
                # Create figure for timeline visualization
                fig_timeline = go.Figure()
                
                # Get full date range of predictions
                min_date = timeline_data['date'].min()
                max_date = timeline_data['date'].max()
                
                # Get actual sales for the entire timeline period
                start_year = min_date.year
                start_week = min_date.isocalendar()[1]
                end_year = max_date.year
                end_week = max_date.isocalendar()[1]
                
                # Get actual sales for comparison - for the full timeline
                actual_sales = get_actual_sales(
                    current['product_id'], 
                    start_year, 
                    start_week, 
                    end_year, 
                    end_week
                )
                
                if not actual_sales.empty:
                    actual_sales['date'] = pd.to_datetime(
                        actual_sales['year'].astype(str) + '-' + 
                        actual_sales['calendar_week'].astype(str) + '-1', 
                        format='%Y-%W-%w'
                    )
                    actual_sales = actual_sales.sort_values('date')
                    
                    # Add actual sales to the plot
                    fig_timeline.add_trace(go.Scatter(
                        x=actual_sales['date'],
                        y=actual_sales['sales_t0'],
                        mode='lines+markers',
                        name='Tatsächliche Absätze',
                        line=dict(color='blue'),
                        hovertemplate='Jahr: %{text}<br>KW: %{customdata}<br>Absatz: %{y:.2f}',
                        text=actual_sales['year'],
                        customdata=actual_sales['calendar_week']
                    ))
                
                # Add all t0 predictions - only t0 as requested
                fig_timeline.add_trace(go.Scatter(
                    x=timeline_data['date'],
                    y=timeline_data['t0'],
                    mode='lines+markers',
                    name='t0 Prognosen',
                    line=dict(color='red', dash='dash'),
                    hovertemplate='Prognosedatum: %{x|%Y-KW%V}<br>t0 Prognose: %{y:.2f}',
                ))
                
                fig_timeline.update_layout(
                    title=f'Absatzprognose-Timeline für Produkt {current["product_id"]}',
                    xaxis_title='Datum',
                    yaxis_title='Absätze',
                    legend_title='Datentypen',
                    hovermode="closest"
                )
                
                # Format x-axis with custom date format
                fig_timeline.update_xaxes(
                    tickformat="%Y-KW%V",
                    range=[min_date, max_date]
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Add prediction metrics table
                st.subheader("Prognosegenauigkeit über Zeit")
                
                metrics_df = timeline_data[[
                    'date', 'calendar_week', 'year', 't0', 'mae_t0', 'mape_t0', 'r2_t0'
                ]].copy()
                
                metrics_df['date_str'] = metrics_df['date'].dt.strftime('%Y-KW%V')
                metrics_df = metrics_df.rename(columns={
                    'date_str': 'Datum',
                    't0': 'Prognose',
                    'mae_t0': 'MAE',
                    'mape_t0': 'MAPE',
                    'r2_t0': 'R²'
                })
                
                # Format and display metrics table
                st.dataframe(
                    metrics_df[['Datum', 'Prognose', 'MAE', 'MAPE', 'R²']].round(2), 
                    hide_index=True
                )
                
                # Download button for timeline data - only include t0 predictions
                timeline_csv = timeline_data[[
                    'date', 'year', 'calendar_week', 't0', 'mae_t0', 'mape_t0', 'r2_t0'
                ]].to_csv(index=False)
                
                st.download_button(
                    label="Download Timeline CSV",
                    data=timeline_csv,
                    file_name=f'prognose_timeline_{current["product_id"]}.csv',
                    mime='text/csv'
                )
