#!/usr/bin/env python3
"""
Aggregate yesterday's statistics from all energy sources
Creates a summary JSON for the homepage
"""

import json
import os
from datetime import datetime

# Energy source categories
RENEWABLE_SOURCES = ['solar', 'wind', 'hydro', 'biomass', 'geothermal']
NON_RENEWABLE_SOURCES = ['gas', 'coal', 'nuclear', 'oil', 'waste']

def load_source_stats():
    """
    Load statistics from all individual source JSON files
    """
    stats = {}
    stats_dir = 'plots/stats'
    
    if not os.path.exists(stats_dir):
        print("Stats directory not found!")
        return None
    
    # Load individual source files
    for filename in os.listdir(stats_dir):
        if filename.endswith('.json'):
            source_id = filename.replace('.json', '')
            filepath = os.path.join(stats_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    stats[source_id] = data
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return stats

def create_summary(stats):
    """
    Create summary with renewable/non-renewable categories
    """
    renewables = []
    non_renewables = []
    
    total_renewable_pct = 0
    total_non_renewable_pct = 0
    
    # Categorize sources
    for source_id, data in stats.items():
        source_info = {
            'id': source_id,
            'name': data['display_name'],
            'percentage': data['yesterday_percentage'],
            'production_gw': data['yesterday_production_gw']
        }
        
        # Check if renewable or non-renewable
        clean_id = source_id.replace('_', '-')
        if clean_id in RENEWABLE_SOURCES or clean_id == 'all-renewables':
            if clean_id != 'all-renewables':  # Don't double count
                renewables.append(source_info)
                total_renewable_pct += data['yesterday_percentage']
        elif clean_id in NON_RENEWABLE_SOURCES or clean_id == 'all-non-renewables':
            if clean_id != 'all-non-renewables':  # Don't double count
                non_renewables.append(source_info)
                total_non_renewable_pct += data['yesterday_percentage']
    
    # If we have totals, use them (more accurate)
    if 'all_renewables' in stats:
        total_renewable_pct = stats['all_renewables']['yesterday_percentage']
    if 'all_non_renewables' in stats:
        total_non_renewable_pct = stats['all_non_renewables']['yesterday_percentage']
    
    # Sort by percentage (highest first)
    renewables.sort(key=lambda x: x['percentage'], reverse=True)
    non_renewables.sort(key=lambda x: x['percentage'], reverse=True)
    
    summary = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'renewables': renewables,
        'non_renewables': non_renewables,
        'total_renewable_pct': round(total_renewable_pct, 1),
        'total_non_renewable_pct': round(total_non_renewable_pct, 1)
    }
    
    return summary

def main():
    """
    Main aggregator function
    """
    print("=" * 60)
    print("AGGREGATING YESTERDAY'S ENERGY STATISTICS")
    print("=" * 60)
    
    # Load all source stats
    stats = load_source_stats()
    
    if not stats:
        print("✗ No statistics found!")
        return
    
    print(f"✓ Loaded {len(stats)} source statistics")
    
    # Create summary
    summary = create_summary(stats)
    
    # Save to JSON
    output_file = 'plots/yesterday_summary.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to {output_file}")
    print(f"\nSummary:")
    print(f"  Renewables:     {summary['total_renewable_pct']}%")
    print(f"  Non-Renewables: {summary['total_non_renewable_pct']}%")
    print(f"  Sources: {len(summary['renewables'])} renewable, {len(summary['non_renewables'])} non-renewable")
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
