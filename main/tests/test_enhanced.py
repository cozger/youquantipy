#!/usr/bin/env python3
"""Test if enhanced architecture loads correctly"""

import os
os.environ['USE_ENHANCED_ARCHITECTURE'] = 'true'

print("Testing enhanced architecture loading...")

try:
    from parallelworker_enhanced_integration import parallel_participant_worker_enhanced
    print("✓ Enhanced integration imported successfully")
except ImportError as e:
    print(f"✗ Failed to import enhanced integration: {e}")

try:
    from camera_worker_enhanced import camera_worker_enhanced
    print("✓ Camera worker enhanced imported successfully")
except ImportError as e:
    print(f"✗ Failed to import camera worker enhanced: {e}")

try:
    from advanced_detection_integration import parallel_participant_worker_auto
    print("✓ Advanced detection integration imported successfully")
    
    # Check which worker it would use
    from advanced_detection_integration import should_use_advanced_detection
    print(f"should_use_advanced_detection() = {should_use_advanced_detection()}")
    
except ImportError as e:
    print(f"✗ Failed to import advanced detection integration: {e}")

print("\nDone.")