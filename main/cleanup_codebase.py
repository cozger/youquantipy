#!/usr/bin/env python3
"""
Cleanup script to organize the YouQuantiPy codebase.
This script will:
1. Move deprecated files to a 'deprecated' folder
2. Move test files to a 'tests' folder
3. Create a summary of changes
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Files to move to deprecated folder
DEPRECATED_FILES = [
    'tileddetector.py',  # Replaced by retinaface_detector.py
    'landmark_worker_pool_mp.py',  # Experimental variant
    'landmark_worker_pool_threaded.py',  # Experimental variant
    'parallelworker_advanced_enhanced.py',  # Not used in main flow
    'parallelworker_enhanced_integration.py',  # Not used in main flow
    'parallelworker_advanced_wrapper.py',  # Not used in main flow
]

# Files to move to tests folder
TEST_FILES = [
    'test_enhanced.py',
    'test_retinaface_model.py',
]

# Files that are part of enhanced architecture but not currently active
ENHANCED_EXPERIMENTAL = [
    'camera_worker_enhanced.py',
    'frame_router.py',
    'roi_manager.py',
    'result_aggregator.py',
    'landmark_worker_pool_adaptive.py',
    'use_enhanced_architecture.py',
]

def create_folder_if_not_exists(folder_path):
    """Create folder if it doesn't exist."""
    Path(folder_path).mkdir(exist_ok=True)

def move_files(files, source_dir, target_dir, category_name):
    """Move files from source to target directory."""
    moved_files = []
    for file in files:
        source_path = os.path.join(source_dir, file)
        if os.path.exists(source_path):
            target_path = os.path.join(target_dir, file)
            try:
                shutil.move(source_path, target_path)
                moved_files.append(file)
                print(f"✓ Moved {file} to {category_name}")
            except Exception as e:
                print(f"✗ Error moving {file}: {e}")
        else:
            print(f"- {file} not found (may already be moved)")
    return moved_files

def create_readme(folder_path, description, files):
    """Create a README in the folder explaining its contents."""
    readme_path = os.path.join(folder_path, 'README.md')
    content = f"""# {os.path.basename(folder_path).title()} Folder

{description}

## Files in this folder:

"""
    for file in files:
        if file in DEPRECATED_FILES:
            reason = "Replaced by newer implementation"
            if file == 'tileddetector.py':
                reason = "Replaced by retinaface_detector.py"
            elif 'landmark_worker_pool' in file:
                reason = "Experimental variant of landmark worker pool"
            elif 'parallelworker' in file:
                reason = "Experimental variant of parallel worker"
            content += f"- **{file}**: {reason}\n"
        elif file in TEST_FILES:
            content += f"- **{file}**: Test script\n"
        elif file in ENHANCED_EXPERIMENTAL:
            content += f"- **{file}**: Part of experimental enhanced architecture\n"
    
    content += f"\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    with open(readme_path, 'w') as f:
        f.write(content)

def main():
    """Main cleanup function."""
    print("YouQuantiPy Codebase Cleanup Script")
    print("=" * 50)
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create folders
    deprecated_dir = os.path.join(current_dir, 'deprecated')
    tests_dir = os.path.join(current_dir, 'tests')
    enhanced_experimental_dir = os.path.join(current_dir, 'experimental_enhanced')
    
    print("\n1. Creating folders...")
    create_folder_if_not_exists(deprecated_dir)
    create_folder_if_not_exists(tests_dir)
    create_folder_if_not_exists(enhanced_experimental_dir)
    
    # Move files
    print("\n2. Moving deprecated files...")
    deprecated_moved = move_files(DEPRECATED_FILES, current_dir, deprecated_dir, "deprecated")
    
    print("\n3. Moving test files...")
    test_moved = move_files(TEST_FILES, current_dir, tests_dir, "tests")
    
    print("\n4. Moving experimental enhanced files...")
    enhanced_moved = move_files(ENHANCED_EXPERIMENTAL, current_dir, enhanced_experimental_dir, "experimental_enhanced")
    
    # Create READMEs
    print("\n5. Creating README files...")
    if deprecated_moved:
        create_readme(deprecated_dir, 
                     "This folder contains deprecated files that have been replaced by newer implementations.",
                     deprecated_moved)
    
    if test_moved:
        create_readme(tests_dir,
                     "This folder contains test scripts for various components.",
                     test_moved)
    
    if enhanced_moved:
        create_readme(enhanced_experimental_dir,
                     "This folder contains experimental enhanced architecture components that are not currently active in the main flow.",
                     enhanced_moved)
    
    # Summary
    print("\n" + "=" * 50)
    print("CLEANUP SUMMARY:")
    print(f"- Moved {len(deprecated_moved)} files to deprecated/")
    print(f"- Moved {len(test_moved)} files to tests/")
    print(f"- Moved {len(enhanced_moved)} files to experimental_enhanced/")
    print("\nCleanup complete! The codebase is now better organized.")
    print("\nNote: If you need any of these files, they are safely stored in their respective folders.")

if __name__ == "__main__":
    response = input("\nThis script will reorganize files in the codebase. Continue? (y/n): ")
    if response.lower() == 'y':
        main()
    else:
        print("Cleanup cancelled.")