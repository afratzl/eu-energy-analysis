from entsoe import EntsoePandasClient
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for automated runs
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Get API key from environment variable (SECURE - not hardcoded!)
api_key = os.environ.get('ENTSOE_API_KEY')

if not api_key:
    raise ValueError(
        "ENTSOE_API_KEY environment variable not set!\n"
        "For GitHub Actions: Add it as a repository secret.\n"
        "For local testing: export ENTSOE_API_KEY='your-key-here'"
    )

# Initialize the client
client = EntsoePandasClient(api_key=api_key)

# EU country codes
eu_countries = [
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
    'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
    'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
]

# EXACT naming conventions from the first script
WIND_KEYWORDS = {
    'onshore': ['Wind Onshore'],
    'offshore': ['Wind Offshore'],
    'total': ['Wind Onshore', 'Wind Offshore']
}


def get_intraday_data_for_country(country, start_date, end_date, data_type='generation', max_retries=3):
    """
    Get intraday data for a specific country and date range
    """
    import time

    start = pd.Timestamp(start_date, tz='Europe/Brussels')
    end = pd.Timestamp(end_date, tz='Europe/Brussels') + timedelta(hours=1)

    for attempt in range(max_retries):
        try:
            if data_type == 'generation':
                data = client.query_generation(country, start=start, end=end)
            elif data_type == 'load':
                data = client.query_load(country, start=start, end=end)
            else:
                return pd.DataFrame()

            if data.empty:
                return pd.DataFrame()

            # Convert to Brussels timezone
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC').tz_convert('Europe/Brussels')
            elif str(data.index.tz) != 'Europe/Brussels':
                data.index = data.index.tz_convert('Europe/Brussels')

            time.sleep(0.2)
            return data

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 0.5 * (2 ** attempt)
                time.sleep(wait_time)
            else:
                time.sleep(0.5)
                return pd.DataFrame()

    return pd.DataFrame()


def extract_wind_from_generation_data(generation_data, wind_type='total'):
    """
    Extract wind data using EXACT logic from first script
    """
    source_keywords = WIND_KEYWORDS[wind_type]

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
    """
    if len(country_series) == 0:
        return None

    time_diffs = country_series.index.to_series().diff().dt.total_seconds() / 60
    most_common_interval = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else 15

    start_time = country_series.index.min().floor('15T')
    end_time = country_series.index.max().ceil('15T')
    complete_index = pd.date_range(start_time, end_time, freq='15T')

    last_actual_time = country_series.index.max() if mark_extrapolated else None

    if most_common_interval >= 45:  # Hourly
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


def aggregate_eu_data(countries, start_date, end_date, data_type='generation', wind_type='total',
                      mark_extrapolated=False):
    """
    Aggregate wind data across EU countries
    """
    all_interpolated_data = []
    successful_countries = []

    for country in countries:
        country_data = get_intraday_data_for_country(country, start_date, end_date, data_type)

        if not country_data.empty:
            if data_type == 'generation':
                country_wind, wind_columns = extract_wind_from_generation_data(country_data, wind_type)

                if wind_columns:
                    country_wind.name = country
                    interpolated = interpolate_country_data(country_wind, country, mark_extrapolated=mark_extrapolated)

                    if interpolated is not None:
                        all_interpolated_data.append(interpolated)
                        successful_countries.append(country)

            elif data_type == 'load':
                if isinstance(country_data, pd.Series):
                    country_series = country_data
                elif isinstance(country_data, pd.DataFrame) and len(country_data.columns) == 1:
                    country_series = country_data.iloc[:, 0]
                else:
                    country_series = country_data.sum(axis=1)

                country_series.name = country
                interpolated = interpolate_country_data(country_series, country, mark_extrapolated=mark_extrapolated)

                if interpolated is not None:
                    all_interpolated_data.append(interpolated)
                    successful_countries.append(country)

    if not all_interpolated_data:
        return pd.DataFrame(), pd.DataFrame(), []

    combined_df = pd.concat(all_interpolated_data, axis=1)
    eu_total = combined_df.sum(axis=1, skipna=True)

    return eu_total, combined_df, successful_countries


def load_intraday_wind_data(wind_type='total'):
    """
    Load 15-minute wind and load data from ENTSO-E
    """
    print("=" * 60)
    print(f"LOADING {wind_type.upper()} WIND DATA")
    print("=" * 60)

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)

    week_ago_end = yesterday
    week_ago_start = week_ago_end - timedelta(days=7)

    year_ago_end = datetime(today.year - 1, yesterday.month, yesterday.day)
    year_ago_start = year_ago_end - timedelta(days=7)

    two_years_ago_end = datetime(today.year - 2, yesterday.month, yesterday.day)
    two_years_ago_start = two_years_ago_end - timedelta(days=7)

    periods = {
        'today': (today, today + timedelta(days=1)),
        'yesterday': (yesterday, yesterday + timedelta(days=1)),
        'week_ago': (week_ago_start, week_ago_end),
        'year_ago': (year_ago_start, year_ago_end),
        'two_years_ago': (two_years_ago_start, two_years_ago_end)
    }

    all_data = {}
    all_country_data = {}

    for period_name, (start_date, end_date) in periods.items():
        print(f"\n{period_name.upper()}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        mark_extrap = (period_name in ['today', 'yesterday'])

        wind_data, wind_by_country, wind_countries = aggregate_eu_data(
            eu_countries, start_date, end_date, 'generation', wind_type, mark_extrapolated=mark_extrap
        )
        load_data, load_by_country, load_countries = aggregate_eu_data(
            eu_countries, start_date, end_date, 'load', wind_type, mark_extrapolated=mark_extrap
        )

        if not wind_data.empty and not load_data.empty:
            common_times = wind_data.index.intersection(load_data.index)

            if len(common_times) > 0:
                period_df = pd.DataFrame({
                    'timestamp': common_times,
                    'wind_production': wind_data.loc[common_times].values,
                    'total_load': load_data.loc[common_times].values
                })

                period_df = period_df[
                    (period_df['timestamp'] >= pd.Timestamp(start_date, tz='Europe/Brussels')) &
                    (period_df['timestamp'] < pd.Timestamp(end_date, tz='Europe/Brussels'))
                    ]

                period_df['wind_percentage'] = np.clip(
                    (period_df['wind_production'] / period_df['total_load']) * 100, 0, 100
                )

                period_df['date'] = period_df['timestamp'].dt.strftime('%Y-%m-%d')
                period_df['time'] = period_df['timestamp'].dt.strftime('%H:%M')

                all_data[period_name] = period_df

                if not wind_by_country.empty:
                    all_country_data[period_name] = wind_by_country.loc[period_df['timestamp']]
                if not load_by_country.empty:
                    all_country_data[f'{period_name}_load'] = load_by_country.loc[period_df['timestamp']]

                avg_pct = period_df['wind_percentage'].mean()
                print(f"  ✓ {len(period_df)} points, avg wind: {avg_pct:.1f}%")

    # Create projected values
    if 'week_ago' in all_country_data and 'today' in all_country_data and 'today' in all_data:
        if 'today_load' in all_country_data and 'week_ago_load' in all_country_data:
            print("\nCreating projected values for TODAY...")
            all_data['today_projected'] = create_projected_data(
                all_data['today'],
                all_country_data['today'], all_country_data['week_ago'],
                all_country_data['today_load'], all_country_data['week_ago_load'],
                'today'
            )

    if 'week_ago' in all_country_data and 'yesterday' in all_country_data and 'yesterday' in all_data:
        if 'yesterday_load' in all_country_data and 'week_ago_load' in all_country_data:
            print("Creating projected values for YESTERDAY...")
            all_data['yesterday_projected'] = create_projected_data(
                all_data['yesterday'],
                all_country_data['yesterday'], all_country_data['week_ago'],
                all_country_data['yesterday_load'], all_country_data['week_ago_load'],
                'yesterday'
            )

    return all_data


def create_projected_data(actual_df, actual_wind_countries, week_ago_wind_countries,
                          actual_load_countries, week_ago_load_countries, period_name='today'):
    """
    Create projected values by filling missing country data with week_ago averages
    """
    print("  Creating projected data...")

    week_ago_wind_list = set(week_ago_wind_countries.columns)
    today_wind_list = set(actual_wind_countries.columns)

    week_ago_load_list = set(week_ago_load_countries.columns)
    today_load_list = set(actual_load_countries.columns)

    completely_missing_wind = week_ago_wind_list - today_wind_list
    completely_missing_load = week_ago_load_list - today_load_list

    week_ago_wind_with_time = week_ago_wind_countries.copy()
    week_ago_wind_with_time['time'] = week_ago_wind_with_time.index.strftime('%H:%M')
    week_ago_wind_avg = week_ago_wind_with_time.groupby('time').mean(numeric_only=True)

    week_ago_load_with_time = week_ago_load_countries.copy()
    week_ago_load_with_time['time'] = week_ago_load_with_time.index.strftime('%H:%M')
    week_ago_load_avg = week_ago_load_with_time.groupby('time').mean(numeric_only=True)

    projected_df = actual_df.copy()
    missing_by_time = {}

    for i in range(len(actual_df)):
        time_str = actual_df.iloc[i]['time']
        timestamp = actual_df.iloc[i]['timestamp']

        missing_countries = []
        projected_wind_total = 0
        projected_load_total = 0
        has_missing_data = False

        if timestamp in actual_wind_countries.index:
            actual_wind_row = actual_wind_countries.loc[timestamp]

            for country in today_wind_list:
                actual_val = actual_wind_row[country]

                if pd.isna(actual_val) or actual_val < 1:
                    missing_countries.append(country)
                    has_missing_data = True
                    if time_str in week_ago_wind_avg.index and country in week_ago_wind_avg.columns:
                        proj_val = week_ago_wind_avg.loc[time_str, country]
                        if not pd.isna(proj_val):
                            projected_wind_total += proj_val
                else:
                    projected_wind_total += actual_val

        for country in completely_missing_wind:
            if country not in missing_countries:
                missing_countries.append(country)
            has_missing_data = True
            if time_str in week_ago_wind_avg.index and country in week_ago_wind_avg.columns:
                proj_val = week_ago_wind_avg.loc[time_str, country]
                if not pd.isna(proj_val):
                    projected_wind_total += proj_val

        if timestamp in actual_load_countries.index:
            actual_load_row = actual_load_countries.loc[timestamp]

            for country in today_load_list:
                actual_val = actual_load_row[country]

                if pd.isna(actual_val) or actual_val < 1:
                    has_missing_data = True
                    if time_str in week_ago_load_avg.index and country in week_ago_load_avg.columns:
                        proj_val = week_ago_load_avg.loc[time_str, country]
                        if not pd.isna(proj_val):
                            projected_load_total += proj_val
                else:
                    projected_load_total += actual_val

        for country in completely_missing_load:
            has_missing_data = True
            if time_str in week_ago_load_avg.index and country in week_ago_load_avg.columns:
                proj_val = week_ago_load_avg.loc[time_str, country]
                if not pd.isna(proj_val):
                    projected_load_total += proj_val

        if has_missing_data:
            if missing_countries:
                missing_by_time[time_str] = sorted(set(missing_countries))

            projected_df.iloc[i, projected_df.columns.get_loc('wind_production')] = projected_wind_total
            projected_df.iloc[i, projected_df.columns.get_loc('total_load')] = projected_load_total

            if projected_load_total > 0:
                projected_df.iloc[i, projected_df.columns.get_loc('wind_percentage')] = np.clip(
                    (projected_wind_total / projected_load_total) * 100, 0, 100
                )

    if missing_by_time:
        all_missing = set()
        for countries in missing_by_time.values():
            all_missing.update(countries)
        print(f"  Missing: {', '.join(sorted(all_missing))}")

    return projected_df


def calculate_daily_statistics(data_dict):
    """
    Calculate daily statistics
    """
    standard_times = []
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            standard_times.append(f"{hour:02d}:{minute:02d}")

    stats = {}

    for period_name, df in data_dict.items():
        if len(df) == 0:
            continue

        if period_name in ['today', 'yesterday', 'today_projected', 'yesterday_projected']:
            time_indexed = df.groupby('time')[['wind_production', 'total_load', 'wind_percentage']].mean()

            aligned_wind = time_indexed['wind_production'].reindex(standard_times)
            aligned_load = time_indexed['total_load'].reindex(standard_times)
            aligned_percentage = time_indexed['wind_percentage'].reindex(standard_times)

            if period_name in ['today', 'today_projected']:
                current_time = pd.Timestamp.now(tz='Europe/Brussels')
                cutoff_time = current_time - timedelta(hours=2)
                cutoff_time = cutoff_time.floor('15T')

                try:
                    cutoff_time_str = cutoff_time.strftime('%H:%M')
                    cutoff_idx = standard_times.index(cutoff_time_str)
                except ValueError:
                    cutoff_idx = len([t for t in standard_times if t <= cutoff_time_str])

                aligned_wind.iloc[:cutoff_idx] = aligned_wind.iloc[:cutoff_idx].interpolate()
                aligned_load.iloc[:cutoff_idx] = aligned_load.iloc[:cutoff_idx].interpolate()
                aligned_percentage.iloc[:cutoff_idx] = aligned_percentage.iloc[:cutoff_idx].interpolate()

                aligned_wind.iloc[cutoff_idx:] = np.nan
                aligned_load.iloc[cutoff_idx:] = np.nan
                aligned_percentage.iloc[cutoff_idx:] = np.nan
            else:
                aligned_wind = aligned_wind.interpolate()
                aligned_load = aligned_load.interpolate()
                aligned_percentage = aligned_percentage.interpolate()

            aligned_wind = aligned_wind.fillna(0.1)
            aligned_load = aligned_load.fillna(method='ffill').fillna(method='bfill').fillna(50000)
            aligned_percentage = aligned_percentage.fillna(0)

            stats[period_name] = {
                'time_bins': standard_times,
                'wind_mean': aligned_wind.values,
                'wind_std': np.zeros(len(standard_times)),
                'percentage_mean': aligned_percentage.values,
                'percentage_std': np.zeros(len(standard_times)),
            }

        else:
            unique_dates = df['date'].unique()

            daily_wind_data = []
            daily_percentage_data = []

            for date in unique_dates:
                day_data = df[df['date'] == date]

                if len(day_data) > 0:
                    time_indexed = day_data.set_index('time')
                    time_indexed = time_indexed[['wind_production', 'wind_percentage']].groupby(
                        time_indexed.index).mean()

                    aligned_wind = time_indexed['wind_production'].reindex(standard_times).interpolate()
                    aligned_percentage = time_indexed['wind_percentage'].reindex(standard_times).interpolate()

                    aligned_wind = aligned_wind.fillna(0.1)
                    aligned_percentage = aligned_percentage.fillna(0)

                    daily_wind_data.append(aligned_wind.values)
                    daily_percentage_data.append(aligned_percentage.values)

            if daily_wind_data:
                wind_array = np.array(daily_wind_data)
                percentage_array = np.array(daily_percentage_data)

                stats[period_name] = {
                    'time_bins': standard_times,
                    'wind_mean': np.mean(wind_array, axis=0),
                    'wind_std': np.std(wind_array, axis=0),
                    'percentage_mean': np.mean(percentage_array, axis=0),
                    'percentage_std': np.std(percentage_array, axis=0),
                }

    return stats


def create_time_axis():
    """
    Create time axis for 15-minute bins
    """
    times = []
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            times.append(f"{hour:02d}:{minute:02d}")
    return times


def plot_wind_analysis(stats_data, wind_type='total'):
    """
    Create plots - SAVES to file instead of showing
    """
    if not stats_data:
        print("No data for plotting")
        return None

    print("\nCreating plots...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    colors = {
        'today': '#FF4444',
        'yesterday': '#FF8800',
        'week_ago': '#4444FF',
        'year_ago': '#44AA44',
        'two_years_ago': '#AA44AA',
        'today_projected': '#FF4444',
        'yesterday_projected': '#FF8800'
    }

    linestyles = {
        'today': '-',
        'yesterday': '-',
        'week_ago': '-',
        'year_ago': '-',
        'two_years_ago': '-',
        'today_projected': '--',
        'yesterday_projected': '--'
    }

    labels = {
        'today': 'Today',
        'yesterday': 'Yesterday',
        'week_ago': 'Previous Week (avg ± std)',
        'year_ago': 'Same Period Last Year (avg ± std)',
        'two_years_ago': 'Same Period 2 Years Ago (avg ± std)',
        'today_projected': 'Today (Projected)',
        'yesterday_projected': 'Yesterday (Projected)'
    }

    time_labels = create_time_axis()

    wind_type_title = wind_type.capitalize() if wind_type != 'total' else 'Total'
    ax1.set_title(f'EU {wind_type_title} Wind Energy Production\n15-minute Resolution',
                  fontsize=18, fontweight='bold')
    ax1.set_ylabel('Wind Production (MW)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Time of Day (Brussels)', fontsize=14)

    max_wind = 0

    plot_order = ['week_ago', 'year_ago', 'two_years_ago', 'yesterday', 'today', 
                  'yesterday_projected', 'today_projected']

    for period_name in plot_order:
        if period_name not in stats_data:
            continue
            
        data = stats_data[period_name]
        if 'wind_mean' not in data or len(data['wind_mean']) == 0:
            continue

        color = colors.get(period_name, 'gray')
        linestyle = linestyles.get(period_name, '-')
        label = labels.get(period_name, period_name)

        x_values = np.arange(len(data['wind_mean']))
        y_values = data['wind_mean'].copy()

        max_wind = max(max_wind, np.nanmax(y_values))

        if period_name in ['today', 'today_projected']:
            mask = ~np.isnan(y_values)
            if np.any(mask):
                x_values = x_values[mask]
                y_values = y_values[mask]
            else:
                continue

        ax1.plot(x_values, y_values, color=color, linestyle=linestyle, linewidth=3, label=label)

        if period_name in ['week_ago', 'year_ago', 'two_years_ago'] and 'wind_std' in data:
            std_values = data['wind_std']
            if period_name == 'today':
                std_values = std_values[mask] if np.any(mask) else std_values

            upper_bound = y_values + std_values[:len(x_values)]
            lower_bound = y_values - std_values[:len(x_values)]
            max_wind = max(max_wind, np.nanmax(upper_bound))

            ax1.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.2)

    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_xlim(0, len(time_labels))
    ax1.set_ylim(0, max_wind * 1.05)

    ax2.set_title(f'{wind_type_title} Wind as Percentage of Total EU Load\n15-minute Resolution',
                  fontsize=18, fontweight='bold')
    ax2.set_xlabel('Time of Day (Brussels)', fontsize=14)
    ax2.set_ylabel('Wind Percentage (%)', fontsize=16, fontweight='bold')

    max_percentage = 0

    for period_name in plot_order:
        if period_name not in stats_data:
            continue
            
        data = stats_data[period_name]
        if 'percentage_mean' not in data or len(data['percentage_mean']) == 0:
            continue

        color = colors.get(period_name, 'gray')
        linestyle = linestyles.get(period_name, '-')
        label = labels.get(period_name, period_name)

        x_values = np.arange(len(data['percentage_mean']))
        y_values = data['percentage_mean'].copy()

        max_percentage = max(max_percentage, np.nanmax(y_values))

        if period_name in ['today', 'today_projected']:
            mask = ~np.isnan(y_values)
            if np.any(mask):
                x_values = x_values[mask]
                y_values = y_values[mask]
            else:
                continue

        ax2.plot(x_values, y_values, color=color, linestyle=linestyle, linewidth=3, label=label)

        if period_name in ['week_ago', 'year_ago', 'two_years_ago'] and 'percentage_std' in data:
            std_values = data['percentage_std']
            if period_name == 'today':
                std_values = std_values[mask] if np.any(mask) else std_values

            upper_bound = y_values + std_values[:len(x_values)]
            lower_bound = y_values - std_values[:len(x_values)]
            max_percentage = max(max_percentage, np.nanmax(upper_bound))

            ax2.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.2)

    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_xlim(0, len(time_labels))

    if max_percentage > 0:
        ax2.set_ylim(0, max_percentage * 1.05)
    else:
        ax2.set_ylim(0, 50)

    tick_positions = np.arange(0, len(time_labels), 8)
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([time_labels[i] for i in tick_positions], rotation=45)

    handles1, labels1 = ax1.get_legend_handles_labels()
    fig.legend(handles1, labels1, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=4, fontsize=14, frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 0.98])

    # Save the plot
    output_file = f'plots/wind_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()

    return output_file


def main():
    """
    Main function - automated version
    """
    print("=" * 60)
    print("WIND INTRADAY ANALYSIS - AUTOMATED")
    print("=" * 60)

    try:
        # Always run total wind for automation
        wind_type = 'total'
        
        print(f"\n{'=' * 60}")
        print(f"{wind_type.upper()} WIND - EU COMBINED")
        print(f"{'=' * 60}")

        raw_data = load_intraday_wind_data(wind_type)

        if not raw_data:
            print(f"✗ No data")
            return

        stats_data = calculate_daily_statistics(raw_data)

        plot_file = plot_wind_analysis(stats_data, wind_type)

        # Create timestamp file
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        with open('plots/last_update.html', 'w') as f:
            f.write(f'<p>Last updated: {timestamp}</p>')

        print(f"\n✓ COMPLETE! Plot saved to {plot_file}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
