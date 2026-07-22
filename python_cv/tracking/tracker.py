from supervision import ByteTrack, Detections
from utils.logger import get_logger

log = get_logger("tracker")

class Tracker:
    def __init__(self):
        self._tracker = ByteTrack()
        log.info("ByteTrack initialized")

    def update(self, detections: Detections) -> Detections:
        mask = detections.class_id == 0
        person_dets = detections[mask]
        if len(person_dets) == 0:
            return person_dets
        return self._tracker.update_with_detections(person_dets)

    def reset(self):
        self._tracker = ByteTrack()
