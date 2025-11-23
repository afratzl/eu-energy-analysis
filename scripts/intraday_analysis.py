#!/usr/bin/env python3
"""
Unified Intraday Energy Analysis Script - FIXED VERSION
Analyzes any energy source for EU countries with 15-minute resolution
"""

from entsoe import EntsoePandasClient
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import sys
import argparse
import time

warnings.filterwarnings('ignore')

# Create plots directory
os.makedirs('plots', exist_ok=True)

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


def get_intraday_data_for_country(client, country, start_date, end_date, max_retries=3):
    """
    Get intraday generation data for a specific country with retry logic
    """
    start = pd.Timestamp(start_date, tz='Europe/Brussels')
    end = pd.Timestamp(end_date, tz='Europe/Brussels') + timedelta(hours=1)

    for attempt in range(max_retries):
        try:
            data = client.query_generation(country, start=start, end=end)

            if data.empty:
                return pd.DataFrame()

            # Convert to Brussels timezone
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC').tz_convert('Europe/Brussels')
            elif str(data.index.tz) != 'Europe/Brussels':
                data.index = data.index.tz_convert('Europe/Brussels')

            time.sleep(0.2)  # Rate limiting protection
            return data

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 0.5 * (2 ** attempt)
                print(f"  Retry {country} (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                time.sleep(0.5)
                return pd.DataFrame()

    return pd.DataFrame()


def extract_source_from_generation_data(generation_data, source_keywords):
    """
    Extract specific energy source data using keyword matching
    Copied from working wind_analysis.py logic
    """
    relevant_columns = []
    for keyword in source_keywords:
        matching_cols = [col for col in generation_data.columns if keyword in col]
        relevant_columns.extend(matching_cols)
    relevant_columns = list(set(relevant_columns))

    if relevant_columns:
        if len(relevant_columns) == 1:
            energy_series = generation_data[relevant_columns[0]]
        else:
            energy_series = generation_data[relevant_columns].sum(axis=1)
        return energy_series, relevant_columns
    else:
        return pd.Series(0, index=generation_data.index), []


def interpolate_country_data(country_series, country_name, mark_extrapolated=False):
    """
    Interpolate to 15-minute resolution
    Copied from working wind_analysis.py
    """
    if len(country_series) == 0:
        return None

    time_diffs = country_series.index.to_series().diff().dt.total_seconds() / 60
    most_common_interval = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else 15

    start_time = country_series.index.min().floor('15T')
    end_time = country_series.index.max().ceil('15T')
    complete_index = pd.date_range(start_time, end_time, freq='15T')

    last_actual_time = country_series.index.max() if mark_extrapolated else None

    if most_common_interval >= 45:  # Hourly data
        interpolated = country_series.reindex(complete_index)
        interpolated = interpolated.interpolate(method='cubic', limit_area='inside')
        interpolated = interpolated.fillna(method='ffill').fillna(method='bfill')

        if mark_extrapolated:
            mask = complete_index > last_actual_time
            interpolated.loc[mask] = np.nan
    else:  # Already 15-min
        interpolated = country_series.reindex(complete_index)
        interpolated = interpolated.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')

        if mark_extrapolated:
            mask = complete_index > last_actual_time
            interpolated.loc[mask] = np.nan

    return interpolated


def aggregate_eu_data(client, countries, start_date, end_date, source_keywords, mark_extrapolated=False):
    """
    Aggregate energy data across EU countries
    Adapted from working wind_analysis.py
    """
    all_interpolated_data = []
    successful_countries = []

    for country in countries:
        country_data = get_intraday_data_for_country(client, country, start_date, end_date)

        if not country_data.empty:
            country_energy, energy_columns = extract_source_from_generation_data(country_data, source_keywords)

            if energy_columns:  # Only process if columns found
                country_energy.name = country
                interpolated = interpolate_country_data(country_energy, country, mark_extrapolated=mark_extrapolated)

                if interpolated is not None:
                    all_interpolated_data.append(interpolated)
                    successful_countries.append(country)

    if not all_interpolated_data:
        return pd.Series(dtype=float), pd.DataFrame(), []

    combined_df = pd.concat(all_interpolated_data, axis=1)
    eu_total = combined_df.sum(axis=1, skipna=True)

    return eu_total, combined_df, successful_countries


def load_intraday_data(source_type, api_key):
    """
    Load intraday data for specified energy source
    """
    client = EntsoePandasClient(api_key=api_key)
    
    # Get source keywords
    source_keywords = ENERGY_SOURCE_MAPPING.get(source_type)
    if not source_keywords:
        raise ValueError(f"Unknown source type: {source_type}")
    
    print(f"\nSource keywords: {source_keywords}")
    
    # Define time ranges
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    
    week_ago_end = yesterday
    week_ago_start = week_ago_end - timedelta(days=7)
    
    periods = {
        'today': (today, today + timedelta(days=1)),
        'yesterday': (yesterday, yesterday + timedelta(days=1)),
        'week_ago': (week_ago_start, week_ago_end)
    }
    
    all_data = {}
    all_country_data = {}
    
    for period_name, (start_date, end_date) in periods.items():
        print(f"\n{period_name.upper()}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        mark_extrap = (period_name in ['today', 'yesterday'])
        
        energy_data, energy_by_country, energy_countries = aggregate_eu_data(
            client, EU_COUNTRIES, start_date, end_date, source_keywords, mark_extrapolated=mark_extrap
        )
        
        if not energy_data.empty:
            all_data[period_name] = energy_data
            all_country_data[period_name] = energy_by_country
            
            avg_value = energy_data.mean() / 1000  # Convert to GW
            print(f"  ✓ {len(energy_data)} points, avg: {avg_value:.1f} GW")
            print(f"  ✓ {len(energy_countries)}/{len(EU_COUNTRIES)} countries reporting")
        else:
            print(f"  ✗ No data")
    
    # Calculate projected values for missing countries
    if 'today' in all_data and 'week_ago' in all_country_data:
        print("\nCreating projected values for TODAY...")
        all_data['today_projected'] = create_projected_data(
            all_data['today'],
            all_country_data.get('today', pd.DataFrame()),
            all_country_data.get('week_ago', pd.DataFrame()),
            'today'
        )
    
    if 'yesterday' in all_data and 'week_ago' in all_country_data:
        print("Creating projected values for YESTERDAY...")
        all_data['yesterday_projected'] = create_projected_data(
            all_data['yesterday'],
            all_country_data.get('yesterday', pd.DataFrame()),
            all_country_data.get('week_ago', pd.DataFrame()),
            'yesterday'
        )
    
    return all_data, all_country_data


def create_projected_data(actual_series, actual_countries, week_ago_countries, period_name='today'):
    """
    Create projected values by filling missing country data with week_ago averages
    """
    if actual_countries.empty or week_ago_countries.empty:
        return actual_series
    
    print(f"  Creating projected data for {period_name}...")
    
    week_ago_countries_with_time = week_ago_countries.copy()
    week_ago_countries_with_time['time'] = week_ago_countries_with_time.index.strftime('%H:%M')
    week_ago_avg = week_ago_countries_with_time.groupby('time').mean(numeric_only=True)
    
    # Find missing countries
    week_ago_list = set(week_ago_countries.columns)
    today_list = set(actual_countries.columns)
    completely_missing = week_ago_list - today_list
    
    if completely_missing:
        print(f"  Missing countries: {', '.join(sorted(completely_missing))}")
    
    # Create projected series by adding weekly averages for missing countries
    projected_series = actual_series.copy()
    
    for timestamp in projected_series.index:
        time_str = timestamp.strftime('%H:%M')
        
        if time_str in week_ago_avg.index:
            additional_power = 0
            for country in completely_missing:
                if country in week_ago_avg.columns:
                    val = week_ago_avg.loc[time_str, country]
                    if not pd.isna(val):
                        additional_power += val
            
            projected_series.loc[timestamp] += additional_power
    
    return projected_series


def create_intraday_plot(all_data, all_country_data, source_type, output_file):
    """
    Create intraday comparison plot
    """
    source_name = DISPLAY_NAMES.get(source_type, source_type.capitalize())
    
    fig, ax = plt.subplots(figsize=(20, 9))
    
    colors = {
        'today': '#FF4444',
        'yesterday': '#FF8800',
        'week_ago': '#4444FF',
        'today_projected': '#FF4444',
        'yesterday_projected': '#FF8800'
    }
    
    linestyles = {
        'today': '-',
        'yesterday': '-',
        'week_ago': '-',
        'today_projected': '--',
        'yesterday_projected': '--'
    }
    
    labels = {
        'today': 'Today',
        'yesterday': 'Yesterday',
        'week_ago': 'Previous Week (avg)',
        'today_projected': 'Today (Projected)',
        'yesterday_projected': 'Yesterday (Projected)'
    }
    
    # Plot order
    plot_order = ['week_ago', 'yesterday', 'today', 'yesterday_projected', 'today_projected']
    
    # Create time axis (15-minute bins for 24 hours)
    time_labels = []
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            time_labels.append(f"{hour:02d}:{minute:02d}")
    
    # Cutoff time (2 hours ago)
    cutoff_time = pd.Timestamp.now(tz='Europe/Brussels') - timedelta(hours=2)
    cutoff_time = cutoff_time.floor('15T')
    
    max_power = 0
    
    for period_name in plot_order:
        if period_name not in all_data:
            continue
        
        data_series = all_data[period_name]
        if data_series.empty:
            continue
        
        # Group by time of day
        data_with_time = pd.DataFrame({'power': data_series})
        data_with_time['time'] = data_with_time.index.strftime('%H:%M')
        time_indexed = data_with_time.groupby('time')['power'].mean()
        
        # Align to standard times
        aligned = time_indexed.reindex(time_labels).interpolate()
        aligned = aligned.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Apply cutoff for today/projected
        if period_name in ['today', 'today_projected']:
            cutoff_time_str = cutoff_time.strftime('%H:%M')
            try:
                cutoff_idx = time_labels.index(cutoff_time_str)
            except ValueError:
                cutoff_idx = len([t for t in time_labels if t <= cutoff_time_str])
            
            # Keep data up to cutoff, NaN after
            x_values = np.arange(cutoff_idx)
            y_values = aligned.iloc[:cutoff_idx].values / 1000  # Convert to GW
        else:
            x_values = np.arange(len(aligned))
            y_values = aligned.values / 1000  # Convert to GW
        
        if len(y_values) > 0:
            max_power = max(max_power, np.nanmax(y_values))
            
            ax.plot(x_values, y_values,
                   color=colors[period_name],
                   linestyle=linestyles[period_name],
                   linewidth=3 if period_name in ['today', 'today_projected'] else 2,
                   label=labels[period_name],
                   alpha=0.7 if 'projected' in period_name else 1.0)
    
    # Formatting
    ax.set_title(f'EU {source_name} Energy Production\n15-minute Resolution',
                fontsize=18, fontweight='bold')
    ax.set_ylabel('Power Production (GW)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time of Day (Brussels)', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(time_labels))
    
    if max_power > 0:
        ax.set_ylim(0, max_power * 1.05)
    
    # X-axis labels (every 2 hours)
    tick_positions = np.arange(0, len(time_labels), 8)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([time_labels[i] for i in tick_positions], rotation=45)
    
    # Legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # Info text
    today_countries = all_country_data.get('today')
    if today_countries is not None and not today_countries.empty:
        n_reporting = len(today_countries.columns)
    else:
        n_reporting = 0
    
    info_text = f"Reporting: {n_reporting}/{len(EU_COUNTRIES)} countries"
    
    if today_countries is not None and not today_countries.empty:
        week_ago_countries = all_country_data.get('week_ago')
        if week_ago_countries is not None and not week_ago_countries.empty:
            missing = set(week_ago_countries.columns) - set(today_countries.columns)
            if missing:
                info_text += f"\nMissing: {', '.join(sorted(missing))}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
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
        all_data, all_country_data = load_intraday_data(args.source, api_key)
        
        if not all_data:
            print("\n✗ No data retrieved. Check:")
            print("  - API key is valid")
            print("  - Source keywords are correct")
            print("  - ENTSO-E API is accessible")
            sys.exit(1)
        
        # Create plot
        output_file = f'plots/{args.source.replace("-", "_")}_analysis.png'
        create_intraday_plot(all_data, all_country_data, args.source, output_file)
        
        # Create timestamp file
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        with open(f'plots/{args.source.replace("-", "_")}_last_update.txt', 'w') as f:
            f.write(timestamp)
        
        print("\n" + "=" * 60)
        print("✓ ANALYSIS COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
