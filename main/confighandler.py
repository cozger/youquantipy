import json
import os
from pathlib import Path


class ConfigHandler:
    """Handles loading and saving configuration for YouQuantiPy"""
    
    DEFAULT_CONFIG = {
        "video_recording": {
            "enabled": False,
            "save_directory": "./recordings",
            "filename_template": "{participant}_{timestamp}",
            "codec": "MJPG"
        },
        "camera_settings": {
            "target_fps": 30,
            "resolution": "720p"
        },
        "startup_mode": {
            "multi_face": False,
            "participant_count": 2,
            "camera_count": 2,
            "enable_mesh": False,
            "enable_pose": True
        },
        "paths": {
            "model_path": "D:/Projects/youquantipy/face_landmarker.task",
            "pose_model_path": "D:/Projects/youquantipy/pose_landmarker_heavy.task"
        },
        "advanced_detection": {
            "retinaface_model": "D:/Projects/youquantipy/retinaface.onnx",
            "arcface_model": "D:/Projects/youquantipy/arcface.onnx",
            "tile_size": 640,
            "tile_overlap": 0.2,
            "detection_confidence": 0.3,
            "nms_threshold": 0.4,
            "max_detection_workers": 4,
            "landmark_worker_count": 4,
            "tracker_settings": {
                "max_age": 30,
                "min_hits": 3,
                "iou_threshold": 0.3,
                "max_drift": 50.0,
                "drift_correction_rate": 0.1
            },
            "roi_settings": {
                "target_size": [256, 256],
                "padding_ratio": 0.3,
                "min_quality_score": 0.5,
                "max_roi_workers": 4
            },
            "recognition_settings": {
                "embedding_dim": 512,
                "max_embeddings_per_person": 50,
                "similarity_threshold": 0.5,
                "update_threshold": 0.7
            },
            "enrollment_settings": {
                "min_samples": 10,
                "min_quality": 0.7,
                "min_consistency": 0.85,
                "min_stability": 0.8,
                "collection_timeout": 30.0,
                "improvement_window": 20
            }
        },
        "audio_recording": {
            "enabled": False,
            "standalone_audio": False,
            "audio_with_video": False,
            "sample_rate": 44100,
            "channels": 1
        },
        "audio_devices": {
            # Will be populated with device assignments like "cam0": device_index
        },
        "camera_resolutions": {
            "480p": [640, 480],
            "720p": [1280, 720],
            "1080p": [1920, 1080],
            "4K": [3840, 2160]
        }
    }
    
    def __init__(self, config_file="./youquantipy_config.json"):
        # If relative path, make it relative to this file's directory
        config_path = Path(config_file)
        if not config_path.is_absolute():
            # Get the directory where this confighandler.py file is located
            this_dir = Path(__file__).parent
            config_path = this_dir / config_file
        
        self.config_file = config_path
        print(f"[ConfigHandler] Initializing with config file: {self.config_file.absolute()}")
        print(f"[ConfigHandler] Config file exists: {self.config_file.exists()}")
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                merged = self._merge_configs(self.DEFAULT_CONFIG, loaded_config)
                return merged
            except Exception as e:
                print(f"[Config] Error loading config: {e}")
                return self.DEFAULT_CONFIG.copy()
        else:
            # Create default config file
            self.save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default, loaded):
        """Recursively merge loaded config with defaults, preserving loaded values"""
        # Start with loaded config to preserve all user values
        result = loaded.copy()
        
        # Add missing keys from defaults
        for key, value in default.items():
            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result[key], dict):
                # Recursively merge nested dicts
                result[key] = self._merge_configs(value, result[key])
        
        return result
    
    def save_config(self, config=None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"[Config] Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"[Config] Error saving config: {e}")
    
    def get(self, key_path, default=None):
        """Get config value using dot notation (e.g., 'video_recording.save_directory')"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path, value):
        """Set config value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self.save_config()