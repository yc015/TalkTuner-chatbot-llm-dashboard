#!/usr/bin/env python3
"""
Migration script to reorganize extra probes into the new structure:
- Folders: {meta_attribute}_{target}/
- Files: {meta_attribute}_at_layer_N.pth and {meta_attribute}_at_layer.json

This script:
1. Reads existing JSON stats files to extract meta_attribute and target
2. Creates subdirectories: {meta_attribute}_{target}/
3. Renames probe files to use meta_attribute
4. Moves all files to their respective subdirectories
"""

import os
import json
import shutil
from pathlib import Path

# Directories to migrate
PROBE_DIRS = [
    "llama3_control_probes_extra",
    "llama3_read_probes_extra",
    "gemma2_control_probes_extra",
    "gemma2_read_probes_extra",
    "mistral_control_probes_extra",
    "mistral_read_probes_extra",
]

STATS_DIRS = [
    "llama3_control_probes_stats_extra",
    "llama3_read_probes_stats_extra",
    "gemma2_control_probes_stats_extra",
    "gemma2_read_probes_stats_extra",
    "mistral_control_probes_stats_extra",
    "mistral_read_probes_stats_extra",
]


def migrate_directory(probe_dir, stats_dir, dry_run=False):
    """
    Migrate probes from flat structure to organized subdirectories.
    
    Args:
        probe_dir: Path to probe directory
        stats_dir: Path to stats directory
        dry_run: If True, only print what would be done without making changes
    """
    if not os.path.exists(probe_dir):
        print(f"  Directory not found: {probe_dir} (skipping)")
        return
    
    if not os.path.exists(stats_dir):
        print(f"  Stats directory not found: {stats_dir} (skipping)")
        return
    
    print(f"\n{'='*70}")
    print(f"Migrating: {probe_dir}")
    print(f"Stats dir: {stats_dir}")
    print(f"{'='*70}")
    
    # Find all JSON stats files at root level (not in subdirectories)
    stats_files = []
    try:
        for item in os.listdir(stats_dir):
            if item.endswith('_at_layer.json') and os.path.isfile(os.path.join(stats_dir, item)):
                stats_files.append(item)
    except Exception as e:
        print(f"  Error listing stats directory: {e}")
        return
    
    if not stats_files:
        print("  No stats files found at root level (already migrated or no probes)")
        return
    
    print(f"  Found {len(stats_files)} stats file(s) to process\n")
    
    # Process each stats file
    for stats_filename in stats_files:
        stats_path = os.path.join(stats_dir, stats_filename)
        
        # Extract attribute name from filename
        attribute = stats_filename.replace('_at_layer.json', '')
        print(f"  Processing attribute: {attribute}")
        
        # Load metadata from JSON
        try:
            with open(stats_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"    ✗ Failed to read stats file: {e}")
            continue
        
        # Extract meta_attribute and target
        meta_attribute = metadata.get('meta_attribute', attribute)
        target = metadata.get('target', 'user')
        
        print(f"    Meta attribute: {meta_attribute}")
        print(f"    Target: {target}")
        
        # Create folder name: {meta_attribute}_{target}
        folder_name = f"{meta_attribute}_{target}"
        
        # Create subdirectories
        probe_subdir = os.path.join(probe_dir, folder_name)
        stats_subdir = os.path.join(stats_dir, folder_name)
        
        if not dry_run:
            os.makedirs(probe_subdir, exist_ok=True)
            os.makedirs(stats_subdir, exist_ok=True)
            print(f"    ✓ Created directories: {folder_name}/")
        else:
            print(f"    [DRY RUN] Would create directories: {folder_name}/")
        
        # Find and migrate probe files
        probe_files = []
        try:
            for item in os.listdir(probe_dir):
                # Match files like: {attribute}_at_layer_N.pth
                if item.startswith(f"{attribute}_at_layer_") and item.endswith('.pth') and os.path.isfile(os.path.join(probe_dir, item)):
                    probe_files.append(item)
        except Exception as e:
            print(f"    ✗ Error listing probe directory: {e}")
            continue
        
        if not probe_files:
            print(f"    ⚠ No probe files found for {attribute}")
        else:
            print(f"    Found {len(probe_files)} probe file(s)")
            
            # Rename and move probe files
            for probe_filename in probe_files:
                old_probe_path = os.path.join(probe_dir, probe_filename)
                
                # Extract layer number
                layer_part = probe_filename.replace(f"{attribute}_at_layer_", "")
                
                # Create new filename with meta_attribute
                new_probe_filename = f"{meta_attribute}_at_layer_{layer_part}"
                new_probe_path = os.path.join(probe_subdir, new_probe_filename)
                
                if not dry_run:
                    shutil.move(old_probe_path, new_probe_path)
                    print(f"      ✓ Moved: {probe_filename} -> {folder_name}/{new_probe_filename}")
                else:
                    print(f"      [DRY RUN] Would move: {probe_filename} -> {folder_name}/{new_probe_filename}")
        
        # Rename and move stats file
        new_stats_filename = f"{meta_attribute}_at_layer.json"
        new_stats_path = os.path.join(stats_subdir, new_stats_filename)
        
        if not dry_run:
            shutil.move(stats_path, new_stats_path)
            print(f"    ✓ Moved stats: {stats_filename} -> {folder_name}/{new_stats_filename}")
        else:
            print(f"    [DRY RUN] Would move stats: {stats_filename} -> {folder_name}/{new_stats_filename}")
        
        print(f"    ✓ Completed migration for {attribute}\n")


def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate extra probes to new subdirectory structure')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without making changes')
    parser.add_argument('--base-dir', type=str, default='.',
                       help='Base directory containing probe folders (default: current directory)')
    
    args = parser.parse_args()
    
    base_dir = os.path.abspath(args.base_dir)
    
    print("\n" + "="*70)
    print("EXTRA PROBE MIGRATION SCRIPT")
    print("="*70)
    print(f"Base directory: {base_dir}")
    print(f"Mode: {'DRY RUN (no changes will be made)' if args.dry_run else 'LIVE (files will be moved)'}")
    print("="*70)
    
    if not args.dry_run:
        response = input("\n⚠️  This will modify your probe files. Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Migration cancelled.")
            return
    
    # Migrate each pair of directories
    for probe_dir_name, stats_dir_name in zip(PROBE_DIRS, STATS_DIRS):
        probe_dir = os.path.join(base_dir, probe_dir_name)
        stats_dir = os.path.join(base_dir, stats_dir_name)
        
        migrate_directory(probe_dir, stats_dir, dry_run=args.dry_run)
    
    print("\n" + "="*70)
    if args.dry_run:
        print("DRY RUN COMPLETE")
        print("Run without --dry-run to actually perform the migration")
    else:
        print("MIGRATION COMPLETE!")
        print("All probes have been organized into subdirectories")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

