#!/usr/bin/env python3
"""
Unified Intraday Energy Analysis Script
Analyzes any energy source for EU countries with 15-minute resolution
"""

from entsoe import EntsoePandasClient
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pytz
import numpy as np
import os
import sys
import argparse

# EU country codes
EU_COUNTRIES = [
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
    'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
    'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
]

# Energy source mapping (matching data collection script)
ENERGY_SOURCE_MAPPING = {
    'solar': ['Solar'],
    'wind': ['Wind Onshore', 'Wind Offshore'],
    'hydro': ['Hydro', 'Hydro Water Reservoir', 'Hydro Run-of-river', 'Hydro Pumped Storage',
              'Water Reservoir', 'Run-of-river', 'Poundage', 'Hydro Run-of-river and poundage'],
    'biomass': ['Biomass', 'Biogas', 'Biofuel'],
    'geothermal': ['Geothermal'],
    'gas': ['Fossil Gas', 'Natural Gas', 'Gas', 'Fossil Coal-derived gas'],
    'coal': ['Fossil Hard coal', 'Fossil Brown coal', 'Fossil Brown coal/Lignite', 
             'Hard Coal', 'Brown Coal', 'Coal', 'Lignite', 'Fossil Peat', 'Peat'],
    'nuclear': ['Nuclear'],
    'oil': ['Fossil Oil', 'Oil', 'Petroleum'],
    'waste': ['Waste', 'Other non-renewable', 'Other'],
    'all-renewables': ['Solar', 'Wind Onshore', 'Wind Offshore',
                       'Hydro', 'Hydro Water Reservoir', 'Hydro Run-of-river', 'Hydro Pumped Storage',
                       'Water Reservoir', 'Run-of-river', 'Poundage', 'Hydro Run-of-river and poundage',
                       'Geothermal', 'Biomass', 'Biogas', 'Biofuel', 'Other renewable'],
    'all-non-renewables': ['Fossil Gas', 'Natural Gas', 'Gas', 'Fossil Coal-derived gas',
                           'Fossil Hard coal', 'Fossil Brown coal', 'Fossil Brown coal/Lignite',
                           'Hard Coal', 'Brown Coal', 'Coal', 'Lignite', 'Fossil Peat', 'Peat',
                           'Nuclear', 'Fossil Oil', 'Oil', 'Petroleum',
                           'Waste', 'Other non-renewable', 'Other']
}

# Display names for sources
DISPLAY_NAMES = {
    'solar': 'Solar',
    'wind': 'Wind',
    'hydro': 'Hydro',
    'biomass': 'Biomass',
    'geothermal': 'Geothermal',
    'gas': 'Gas',
    'coal': 'Coal',
    'nuclear': 'Nuclear',
    'oil': 'Oil',
    'waste': 'Waste',
    'all-renewables': 'All Renewables',
    'all-non-renewables': 'All Non-Renewables'
}


def interpolate_country_data(country_data, country, mark_extrapolated=False):
    """
    Interpolate country data to 15-minute resolution with proper timezone handling
    """
    if country_data.empty:
        return country_data
    
    # Ensure timezone
    if country_data.index.tz is None:
        country_data.index = country_data.index.tz_localize('Europe/Brussels')
    else:
        country_data.index = country_data.index.tz_convert('Europe/Brussels')
    
    # Create 15-minute index
    start_time = country_data.index.min().floor('H')
    end_time = country_data.index.max().ceil('H') + timedelta(hours=1)
    
    freq_15min = pd.date_range(start=start_time, end=end_time, freq='15min', tz='Europe/Brussels')
    
    # Reindex
    interpolated = country_data.reindex(freq_15min)
    
    # Mark original data points
    if mark_extrapolated:
        interpolated['is_original'] = interpolated.index.isin(country_data.index)
    
    # Interpolate using cubic method (requires scipy)
    try:
        interpolated = interpolated.interpolate(method='cubic', limit_area='inside')
    except:
        # Fallback to linear if scipy not available
        interpolated = interpolated.interpolate(method='linear', limit_area='inside')
    
    # Filter to remove overflow timestamps (next day 00:00+)
    max_time = country_data.index.max()
    if max_time.hour == 23 and max_time.minute >= 45:
        max_allowed = max_time.floor('D') + timedelta(days=1)
        interpolated = interpolated[interpolated.index < max_allowed]
    
    return interpolated


def aggregate_eu_data(country_data_dict, source_type, mark_extrapolated=False):
    """
    Aggregate and interpolate EU data from all countries
    """
    all_country_series = []
    countries_with_data = []
    
    for country, country_data in country_data_dict.items():
        if country_data is not None and not country_data.empty:
            interpolated = interpolate_country_data(country_data, country, mark_extrapolated)
            
            if not interpolated.empty:
                # Sum columns if multiple
                if isinstance(interpolated, pd.DataFrame):
                    country_series = interpolated.sum(axis=1)
                else:
                    country_series = interpolated
                
                all_country_series.append(country_series)
                countries_with_data.append(country)
    
    if not all_country_series:
        return pd.Series(dtype=float), {}, []
    
    # Combine all countries
    combined_df = pd.concat(all_country_series, axis=1)
    combined_df.columns = countries_with_data
    
    # Sum across countries
    eu_total = combined_df.sum(axis=1)
    
    # Store individual country data
    country_dict = {country: combined_df[country] for country in countries_with_data}
    
    return eu_total, country_dict, countries_with_data


def load_intraday_data(source_type, api_key):
    """
    Load intraday data for specified energy source
    """
    client = EntsoePandasClient(api_key=api_key)
    
    # Define time ranges
    brussels_tz = pytz.timezone('Europe/Brussels')
    now = datetime.now(brussels_tz)
    
    # Today: from 00:00 to now+1h
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = now + timedelta(hours=1)
    
    # Yesterday: full day
    yesterday_start = today_start - timedelta(days=1)
    yesterday_end = today_start
    
    # Historical: same day of week, last 4 weeks
    historical_starts = [today_start - timedelta(weeks=i) for i in range(1, 5)]
    
    print(f"\nTODAY: {today_start.date()} to {today_end.date()}")
    
    # Get source keywords
    source_keywords = ENERGY_SOURCE_MAPPING.get(source_type, [])
    if not source_keywords:
        raise ValueError(f"Unknown source type: {source_type}")
    
    # Query all countries
    def query_country_period(country, start, end, period_name):
        try:
            data = client.query_generation(country, start=start, end=end)
            
            if data.empty:
                return None
            
            # Filter for relevant columns
            relevant_cols = []
            for keyword in source_keywords:
                matching = [col for col in data.columns if keyword in col]
                relevant_cols.extend(matching)
            
            relevant_cols = list(set(relevant_cols))
            
            if not relevant_cols:
                return None
            
            if len(relevant_cols) == 1:
                return data[relevant_cols[0]]
            else:
                return data[relevant_cols].sum(axis=1)
                
        except Exception as e:
            return None
    
    # Query today
    print("Loading today's data...")
    today_data = {}
    for country in EU_COUNTRIES:
        today_data[country] = query_country_period(country, today_start, today_end, "today")
    
    # Query yesterday
    print("Loading yesterday's data...")
    yesterday_data = {}
    for country in EU_COUNTRIES:
        yesterday_data[country] = query_country_period(country, yesterday_start, yesterday_end, "yesterday")
    
    # Query historical
    print("Loading historical data...")
    historical_data_list = []
    for i, hist_start in enumerate(historical_starts):
        hist_end = hist_start + timedelta(days=1)
        hist_data = {}
        for country in EU_COUNTRIES:
            hist_data[country] = query_country_period(country, hist_start, hist_end, f"hist_{i}")
        historical_data_list.append(hist_data)
    
    # Aggregate
    today_eu, today_by_country, today_countries = aggregate_eu_data(today_data, source_type, mark_extrapolated=False)
    yesterday_eu, yesterday_by_country, yesterday_countries = aggregate_eu_data(yesterday_data, source_type)
    
    historical_eu_list = []
    for hist_data in historical_data_list:
        hist_eu, _, _ = aggregate_eu_data(hist_data, source_type)
        historical_eu_list.append(hist_eu)
    
    # Calculate projection (estimate for countries not reporting)
    all_countries_set = set(EU_COUNTRIES)
    reporting_countries = set(today_countries)
    missing_countries = all_countries_set - reporting_countries
    
    if missing_countries and reporting_countries:
        # Use historical average to estimate missing
        reporting_total = today_eu
        # Simple projection: scale up proportionally
        projection_factor = len(all_countries_set) / len(reporting_countries)
        today_projected = reporting_total * projection_factor
    else:
        today_projected = today_eu
    
    return {
        'today': today_eu,
        'today_projected': today_projected,
        'yesterday': yesterday_eu,
        'historical': historical_eu_list,
        'today_countries': today_countries,
        'missing_countries': list(missing_countries)
    }


def create_intraday_plot(data, source_type, output_file):
    """
    Create intraday comparison plot
    """
    source_name = DISPLAY_NAMES.get(source_type, source_type.capitalize())
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Apply 2-hour cutoff for data quality
    cutoff_time = datetime.now(pytz.timezone('Europe/Brussels')) - timedelta(hours=2)
    
    # Plot today (actual data)
    today_data = data['today']
    if not today_data.empty:
        valid_today = today_data[today_data.index <= cutoff_time]
        if not valid_today.empty:
            ax.plot(valid_today.index, valid_today / 1000, 
                   label='Today (Actual)', color='#2E86AB', linewidth=2.5, zorder=3)
    
    # Plot today projected (dashed)
    today_projected = data['today_projected']
    if not today_projected.empty:
        valid_proj = today_projected[today_projected.index <= cutoff_time]
        if not valid_proj.empty:
            ax.plot(valid_proj.index, valid_proj / 1000,
                   label='Today (Projected)', color='#2E86AB', linewidth=2.5, 
                   linestyle='--', alpha=0.7, zorder=2)
    
    # Plot yesterday
    yesterday_data = data['yesterday']
    if not yesterday_data.empty:
        # Shift yesterday's index to today's date for comparison
        today_date = datetime.now(pytz.timezone('Europe/Brussels')).date()
        yesterday_shifted_index = yesterday_data.index.map(
            lambda x: x.replace(year=today_date.year, month=today_date.month, day=today_date.day)
        )
        ax.plot(yesterday_shifted_index, yesterday_data / 1000,
               label='Yesterday', color='#A23B72', linewidth=2, alpha=0.7, zorder=2)
    
    # Plot historical average
    if data['historical']:
        historical_dfs = [hist for hist in data['historical'] if not hist.empty]
        if historical_dfs:
            # Align and average
            combined_hist = pd.concat(historical_dfs, axis=1)
            hist_avg = combined_hist.mean(axis=1)
            
            # Shift to today's date
            today_date = datetime.now(pytz.timezone('Europe/Brussels')).date()
            hist_shifted_index = hist_avg.index.map(
                lambda x: x.replace(year=today_date.year, month=today_date.month, day=today_date.day)
            )
            ax.plot(hist_shifted_index, hist_avg / 1000,
                   label='Historical Avg (4 weeks)', color='#8B8B8B', 
                   linewidth=1.5, alpha=0.6, zorder=1)
    
    # Formatting
    ax.set_xlabel('Time (CET/CEST)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (GW)', fontsize=12, fontweight='bold')
    ax.set_title(f'EU {source_name} Energy Production - Intraday Analysis', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone('Europe/Brussels')))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    
    # Info text
    info_text = f"Reporting: {len(data['today_countries'])}/{len(EU_COUNTRIES)} countries"
    if data['missing_countries']:
        info_text += f"\nMissing: {', '.join(sorted(data['missing_countries']))}"
    
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
           fontsize=8, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_file}")
    plt.close()


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='EU Energy Intraday Analysis')
    parser.add_argument('--source', required=True, 
                       choices=list(ENERGY_SOURCE_MAPPING.keys()),
                       help='Energy source to analyze')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"{DISPLAY_NAMES[args.source].upper()} INTRADAY ANALYSIS")
    print("=" * 60)
    
    # Get API key
    api_key = os.environ.get('ENTSOE_API_KEY')
    if not api_key:
        print("ERROR: ENTSOE_API_KEY environment variable not set!")
        sys.exit(1)
    
    try:
        # Load data
        data = load_intraday_data(args.source, api_key)
        
        # Create plot
        output_file = f'plots/{args.source.replace("-", "_")}_analysis.png'
        create_intraday_plot(data, args.source, output_file)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
