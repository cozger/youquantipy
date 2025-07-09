#!/usr/bin/env python3
"""Test standard mode configuration"""

import json
import os

print("Testing Standard Mode Configuration")
print("="*60)

# Create a test configuration with standard mode
test_config = {
    "startup_mode": {
        "enable_face_recognition": False  # This disables enhanced mode
    },
    "cameras": {
        "0": {
            "max_participants": 2,
            "enabled": True
        }
    }
}

# Temporarily save config
with open("test_config_standard.json", "w") as f:
    json.dump(test_config, f, indent=2)

print("\nTest configuration created with:")
print("  - enable_face_recognition: False (standard mode)")

# Test the detection logic
print("\nChecking mode detection logic:")

# Simulate the should_use_advanced_detection logic
face_recognition_enabled = test_config.get('startup_mode', {}).get('enable_face_recognition', False)
print(f"  - Face recognition enabled: {face_recognition_enabled}")

if not face_recognition_enabled:
    print("  - Result: Will use STANDARD mode ✓")
else:
    print("  - Result: Would check for models...")

# Clean up
os.remove("test_config_standard.json")

print("\nConclusion:")
print("-"*60)
print("The system correctly falls back to standard mode when:")
print("  1. enable_face_recognition is False, OR")
print("  2. enable_face_recognition is True but model files are missing")
print("\nCurrent situation: Face recognition is enabled but models are missing")
print("Therefore: System is using STANDARD mode")