# main.py
import cv2
from pylsl import resolve_streams
from multiprocessing import Process
from participant import Participant
from pygrabber.dshow_graph import FilterGraph


def run_participant(config, model_path):
    """
    Spawned in its own process: builds a Participant from simple data
    and calls its .run() method.
    """
    p = Participant(
        participant_id=config["id"],
        camera_index=config["video_source"],
        model_path=model_path,
        stream_name=config["landmark_stream_name"],
        source_id=f"{config['landmark_stream_name']}_uid",
        enable_raw_facemesh=enable_full_mesh
    )
    p.run()


def list_video_devices(max_devices=10):
    """
    Prints available video camera indices along with their names and current resolution.
    """
    graph = FilterGraph()
    names = graph.get_input_devices()  # list of device names in index order

    print("Available video devices:")
    for i in range(min(len(names), max_devices)):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue

        # Query current resolution
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  [{i}] {names[i]!r}: {width}×{height}")
        cap.release()

def list_eeg_streams(timeout=5):
    try:
        """Return a list of available EEG LSL streams."""
        print("Searching for EEG streams...")
        streams = resolve_streams(timeout=timeout)
        eegs = [s for s in streams if s.type() == "EEG"]
        for idx, s in enumerate(eegs):
            print(f"[{idx}] {s.name()}  (uid={s.source_id()})")
        return eegs
    except Exception as e:
        print(f"Error resolving EEG streams: {e}. Are you sure the LSL streams are running?")
        return []


def collect_participant_info():
    """Interactively build a list of participant configs."""
    configs = []
    n = int(input("Enter number of participants: "))
    use_eeg = input("Enable EEG? (y/n): ").strip().lower() == 'y'

    video_devices = list_video_devices()

    eeg_streams = []
    if use_eeg:
        eeg_streams = list_eeg_streams()

    for i in range(n):
        print(f"\n--- Participant {i+1} ---")
        pid = input("  Participant ID: ").strip()

        # Choose camera
        cam = int(input(f"  Select video device index for {pid}: "))

        # Landmark stream name with default
        default_lsl = f"{pid}_landmarks"
        prompt = (
            f"  LSL stream name for {pid}'s landmarks "
            f"[default: {default_lsl}]: "
        )
        lsl_name = input(prompt).strip()
        if not lsl_name:
            lsl_name = default_lsl

        # EEG stream selection (if enabled)
        eeg_name = None
        if use_eeg:
            idx = int(input(f"  Select EEG stream index for {pid}: "))
            eeg_name = eeg_streams[idx].name()

        configs.append({
            "id": pid,
            "video_source": cam,
            "landmark_stream_name": lsl_name,
            "eeg_stream_name": eeg_name
        })
    return configs

def main():
    # Path to your Mediapipe blendshape model
    model_path = r"D:\Projects\MovieSynchrony\face_landmarker.task"
    full_mesh = input("Enable full FaceMesh output in addition to blendshapes? (y/N): ")\
                    .strip().lower() == 'y'
    


    configs = collect_participant_info()

    processes = []
    for cfg in configs:
        source_id = f"{cfg['landmark_stream_name']}_uid"
        p = Participant(
            participant_id=cfg["id"],
            camera_index=cfg["video_source"],
            model_path=model_path,
            stream_name=cfg["landmark_stream_name"],
            source_id=source_id
        )
        proc = Process(
        target=run_participant,
        args=(cfg, model_path,full_mesh),
    )

        proc.start()
        processes.append(proc)

    print("\nStreaming started. Hit ESC in any video window to stop that participant.")

    # Wait for all to finish
    for proc in processes:
        proc.join()

if __name__ == "__main__":
    main()
