import math

class FaceTracker:
    """
    Simple greedy tracker for multiple face centroids.
    On each update, assign detections to existing tracks (by distance),
    or create new tracks. Tracks dropped after max_missed frames.
    """
    def __init__(self, track_threshold=50, max_missed=150):
        # parameters
        self.track_threshold = track_threshold  # px
        self.max_missed = max_missed

        # internal state
        self.next_id = 1
        self.tracks = []  # list of dicts: {'id', 'centroid':(x,y), 'missed'}

    def update(self, detections):
        """
        Update tracks with new detections.
        :param detections: list of (x, y) tuples
        :return: list of track IDs in same order as detections
        """
        new_tracks = []
        used_ids = set()
        assigned_ids = []

        # Greedy match each detection
        for det in detections:
            best = None
            best_dist = float('inf')
            for tr in self.tracks:
                if tr['id'] in used_ids:
                    continue
                dx = tr['centroid'][0] - det[0]
                dy = tr['centroid'][1] - det[1]
                d = math.hypot(dx, dy)
                if d < best_dist:
                    best_dist = d
                    best = tr

            if best and best_dist < self.track_threshold:
                # Assign to existing track
                best['centroid'] = det
                best['missed'] = 0
                new_tracks.append(best)
                used_ids.add(best['id'])
                assigned_ids.append(best['id'])
            else:
                # Create new track
                tid = self.next_id
                self.next_id += 1
                tr = {'id': tid, 'centroid': det, 'missed': 0}
                new_tracks.append(tr)
                used_ids.add(tid)
                assigned_ids.append(tid)

        # Increment miss count for unmatched
        for tr in self.tracks:
            if tr['id'] not in used_ids:
                tr['missed'] += 1
                if tr['missed'] < self.max_missed:
                    new_tracks.append(tr)

        # Replace old tracks
        self.tracks = new_tracks
        return assigned_ids

    def reset(self):
        """Clear all tracks and reset IDs."""
        self.next_id = 1
        self.tracks.clear()
