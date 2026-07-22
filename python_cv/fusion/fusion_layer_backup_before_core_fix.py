from collections import deque
from datetime import datetime
import threading
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from config.settings import (
    UID_MAP, FUSION_TIME_WINDOW, FUSION_THRESHOLD,
    FUSION_WEIGHT_TIME, FUSION_WEIGHT_REID,
)
from reid.reid_engine  import ReIDEngine
from reid.gallery      import Gallery
from database.database import Database
from utils.logger      import get_logger

log = get_logger("fusion")
RFID_EXPIRE_WINDOW = 15.0
INTRUDER_WINDOW = 15.0
PROXY_THRESHOLD    = 0.45


class FusionLayer:
    def __init__(self, reid, gallery, db):
        self.reid    = reid
        self.gallery = gallery
        self.db      = db
        self._pending             = {}
        self._confirmed           = {}
        self._crossings           = deque(maxlen=50)
        self._unmatched_crossings = deque(maxlen=50)
        threading.Thread(target=self._cleanup_loop, daemon=True).start()

    def on_rfid_event(self, event):
        uid    = event["uid"]
        ts     = event["dt"]
        emp_id = UID_MAP.get(uid.upper())
        self.db.log_rfid_event(uid, emp_id, ts.isoformat(),
                               event.get("device",""), event.get("zone",1))
        if emp_id is None:
            log.warning(f"Unknown UID: {uid}")
            self.db.log_anomaly("unknown_uid", None, None, f"uid={uid}", ts)
            return
        if emp_id not in self._pending:
            self._pending[emp_id] = deque(maxlen=5)
        self._pending[emp_id].append({
            "ts": ts, "uid": uid,
            "expires": time.time() + RFID_EXPIRE_WINDOW,
        })
        log.info(f"Pending: {emp_id} @ {ts.strftime('%H:%M:%S')}")
        self._hungarian_match()

    def on_entry_crossing(self, track_id, crop_bgr, timestamp):
        emb = self.reid.get_embedding(crop_bgr)
        entry = {
            "track_id": track_id, "ts": timestamp,
            "time": time.time(), "embedding": emb, "assigned": False,
        }
        self._crossings.append(entry)
        self._unmatched_crossings.append(entry)
        self._hungarian_match()

    def identify_zone2(self, track_id, crop_bgr):
        emp_id = self.reid.identify(crop_bgr, self.gallery.all())
        if emp_id:
            self._confirmed[track_id] = emp_id
            log.info(f"Zone2 re-id: track_{track_id} -> {emp_id}")
            self.db.update_presence(emp_id, 2, track_id)
        return emp_id

    def get_label(self, track_id):
        return self._confirmed.get(track_id, f"track_{track_id}")

    def is_confirmed(self, track_id):
        return track_id in self._confirmed

    def _hungarian_match(self):
        now = time.time()
        pending_list = [
            (emp_id, q[-1])
            for emp_id, q in self._pending.items()
            if q and now < q[-1]["expires"]
        ]
        unassigned = [c for c in self._crossings if not c["assigned"]]
        if not pending_list or not unassigned:
            return
        N, M = len(pending_list), len(unassigned)
        cost = np.full((N, M + N), 1e6)
        for i, (emp_id, rfid) in enumerate(pending_list):
            cost[i][M + i] = 1.0 - FUSION_THRESHOLD
            for j, crossing in enumerate(unassigned):
                dt = abs((crossing["ts"] - rfid["ts"]).total_seconds())
                if dt > FUSION_TIME_WINDOW:
                    continue
                gallery_embeds = self.gallery.get(emp_id)
                if gallery_embeds and crossing["embedding"] is not None:
                    app_score = self.reid.match_score(
                        crossing["embedding"], gallery_embeds)
                else:
                    app_score = 0.5
                time_score  = 1.0 - (dt / FUSION_TIME_WINDOW)
                total_score = (FUSION_WEIGHT_TIME * time_score
                               + FUSION_WEIGHT_REID * app_score)
                cost[i][j] = 1.0 - total_score
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            emp_id = pending_list[i][0]
            if j < M:
                score = 1.0 - cost[i][j]
                if score >= FUSION_THRESHOLD:
                    crossing = unassigned[j]
                    tid = crossing["track_id"]
                    self._confirmed[tid] = emp_id
                    crossing["assigned"] = True
                    self._unmatched_crossings = deque(
                        [c for c in self._unmatched_crossings
                         if c["track_id"] != tid], maxlen=50)
                    self._pending[emp_id].clear()
                    if crossing["embedding"] is not None:
                        self.gallery.update(emp_id, crossing["embedding"])
                    self.db.log_entry(emp_id, tid, crossing["ts"], score)
                    log.info(f"CONFIRMED: track_{tid} -> {emp_id} (score={score:.3f})")
                    self.db.update_presence(emp_id, 1, tid)
                    if score < PROXY_THRESHOLD:
                        log.warning(f"PROXY SWIPE: {emp_id} score={score:.3f}")
                        self.db.log_anomaly("proxy_swipe", emp_id, tid,
                            f"Low score={score:.3f}", crossing["ts"])
                else:
                    log.debug(f"No match for {emp_id} (best={score:.3f})")

    def _cleanup_loop(self):
        while True:
            time.sleep(3)
            now = time.time()
            remove_list = []
            for crossing in list(self._unmatched_crossings):
                if crossing["assigned"]:
                    remove_list.append(crossing)
                    continue
                age = now - crossing["time"]
                if age > INTRUDER_WINDOW:
                    tid = crossing["track_id"]
                    log.warning(f"INTRUDER ALERT: track_{tid} no RFID ({age:.0f}s)")
                    self.db.log_anomaly("no_rfid_intruder", None, tid,
                        f"No RFID after {age:.0f}s", crossing["ts"])
                    remove_list.append(crossing)
            for c in remove_list:
                try:
                    self._unmatched_crossings.remove(c)
                except ValueError:
                    pass
            for emp_id, q in list(self._pending.items()):
                if not q:
                    continue
                if now > q[-1]["expires"]:
                    log.warning(f"RFID TIMEOUT: {emp_id} no crossing")
                    self.db.log_anomaly("rfid_no_crossing", emp_id, None,
                        f"No crossing within {RFID_EXPIRE_WINDOW}s", q[-1]["ts"])
                    q.clear()