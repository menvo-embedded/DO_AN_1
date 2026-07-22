from collections import deque
import os
import threading
import time
from datetime import datetime as _dt
import numpy as np
from scipy.optimize import linear_sum_assignment

import cv2 as _cv2

from config.settings import (
    UID_MAP,
    ENABLE_MQTT,
    EMPLOYEE_NAMES,
    REID_MATCH_THRESHOLD,
    REID_MATCH_MARGIN,
    FUSION_TIME_WINDOW,
    FUSION_THRESHOLD,
    FUSION_WEIGHT_TIME,
    FUSION_WEIGHT_REID,
    RFID_VISUAL_MATCH_THRESHOLD,
    ZONE1_BEST_PERSON_MAX_AGE_SEC,
    ZONE1_FACE_REQUIRED,
    ZONE1_FACE_UPSCALE,
    ZONE1_FACE_UPSCALE_MIN_W,
    FACE_MATCH_THRESHOLD,
)
from utils.logger import get_logger
from utils.display_text import (
    vi, PUSH_ALERT_TYPES, alert_title, alert_status, alert_severity, zone_name,
)

try:
    from alerts.ntfy_alert import NtfyAlert
except Exception:
    NtfyAlert = None

log = get_logger("fusion")

RFID_EXPIRE_WINDOW      = 15.0
INTRUDER_WINDOW         = 15.0
INTRUDER_ALERT_COOLDOWN = 60.0   # throttle cùng track_key không spam alert
PROXY_THRESHOLD         = 0.45

# Thư mục lưu ảnh evidence khi quẹt thẻ (đưa vào báo cáo)
ZONE1_EVIDENCE_DIR = os.path.join("outputs", "tc_evidence")

# ── Zone 2 tuning ─────────────────────────────────────────────────────────────
# Body chỉ được lock khi đủ cả score VÀ margin
BODY_LOCK_MIN_SCORE  = 0.92
BODY_LOCK_MIN_MARGIN = 0.05

# Track lock TTL — reset khi face override
ZONE2_TRACK_LOCK_TTL = 8.0

# Face override: chỉ cho phép relock khi face score đủ mạnh
FACE_OVERRIDE_MIN_SCORE  = 0.35
FACE_OVERRIDE_MIN_MARGIN = 0.10

# Throttle log
ZONE2_LOG_INTERVAL = 2.0


class FusionLayer:
    def __init__(self, reid, gallery, db, face_engine=None):
        self.reid        = reid
        self.gallery     = gallery
        self.db          = db
        self.face_engine = face_engine

        self._zone1_camera        = None   # inject sau bằng set_zone1()
        self._pending             = {}
        self._confirmed           = {}
        self._zone2_track_locks   = {}
        self._zone2_last_decision = {}   # track_key -> last logged (decision, time)
        self._crossings           = deque(maxlen=50)
        self._unmatched_crossings = deque(maxlen=50)
        self._intruder_alerted    = {}   # track_key -> last alert time (throttle)
        self._ntfy_alert          = NtfyAlert() if NtfyAlert is not None else None

        threading.Thread(target=self._cleanup_loop, daemon=True).start()

    # ── Key helpers ───────────────────────────────────────────────────────────
    def _track_key(self, zone: str, track_id) -> str:
        return f"{zone}:{track_id}"

    def _send_ntfy_alert(
        self,
        alert_type: str,
        zone: str,
        timestamp,
        track_id=None,
        employee_id: str | None = None,
        detail: str = "",
        frame=None,
        prefix: str | None = None,
    ):
        if self._ntfy_alert is None:
            return False

        # Chỉ đẩy các loại cảnh báo; ca hợp lệ đã lưu ở dashboard, không đẩy về app.
        if alert_type not in PUSH_ALERT_TYPES:
            return False

        if hasattr(timestamp, "strftime"):
            time_text = timestamp.strftime("%d/%m/%Y %H:%M")
        else:
            time_text = str(timestamp)

        severity = alert_severity(alert_type)
        emoji = "🚨" if severity == "Nghiêm trọng" else "⚠️"

        lines = [
            f"{emoji} {alert_title(alert_type)}",
            "",
            f"Khu vực: {zone_name(zone)}",
            f"Thời gian: {time_text}",
            f"Mức độ: {severity}",
            f"Trạng thái: {alert_status(alert_type)}",
        ]
        if employee_id:
            lines.append(
                f"Nhân viên (thẻ): {employee_id} ({EMPLOYEE_NAMES.get(employee_id, employee_id)})"
            )
        message = "\n".join(lines)

        image = frame.copy() if hasattr(frame, "copy") else frame
        image_prefix = prefix or alert_type

        def _worker():
            try:
                if image is not None:
                    self._ntfy_alert.send_frame(image, message=message, prefix=image_prefix)
                else:
                    self._ntfy_alert.send_text(message)
            except Exception as e:
                log.warning(f"ntfy alert failed: {e}")

        threading.Thread(target=_worker, name=f"ntfy-{alert_type}", daemon=True).start()
        return True

    # ── Zone 1 injection ──────────────────────────────────────────────────────
    def set_zone1(self, camera):
        """Inject CameraZone1 sau khi khởi tạo (tránh circular dependency)."""
        self._zone1_camera = camera
        log.info("FusionLayer: Zone 1 camera injected for RFID-trigger verify")

    # ── RFID ──────────────────────────────────────────────────────────────────
    def on_rfid_event(self, event):
        uid    = event["uid"]
        ts     = event["dt"]
        emp_id = UID_MAP.get(uid.upper())

        self.db.log_rfid_event(
            uid, emp_id, ts.isoformat(),
            event.get("device", ""), event.get("zone", 1),
        )

        if emp_id is None:
            log.warning(f"Unknown UID: {uid} | {vi('unknown_uid')}")
            self.db.log_anomaly("unknown_uid", None, None, f"uid={uid}", ts)
            self._save_anomaly_evidence("unknown_uid", f"uid_{uid}", ts, f"uid={uid}")
            self._send_ntfy_alert(
                "unknown_uid",
                f"zone{event.get('zone', 1)}",
                ts,
                detail=f"uid={uid}",
                prefix="unknown_uid",
            )
            return

        if emp_id not in self._pending:
            self._pending[emp_id] = deque(maxlen=5)

        self._pending[emp_id].append({
            "ts": ts, "uid": uid,
            "expires": time.time() + RFID_EXPIRE_WINDOW,
        })
        log.info(f"Pending RFID: {emp_id} @ {ts.strftime('%H:%M:%S')} | Đã nhận sự kiện quẹt thẻ của {emp_id}")

        # RFID-trigger: verify ngay với Zone 1 camera
        if self._zone1_camera is not None:
            self._rfid_trigger_verify(emp_id, ts)

        self._hungarian_match()

    # ── Lưu ảnh evidence khi quẹt thẻ phát hiện người ─────────────────────────
    def _save_zone1_evidence(self, emp_id, key, ts, result_tag, score, person):
        """Lưu full annotated frame Zone 1 + banner kết quả để paste vào báo cáo.

        Fallback về crop của person nếu chưa có annotated frame.
        Trả về đường dẫn file đã lưu, hoặc None nếu thất bại.
        """
        try:
            frame = None
            if self._zone1_camera is not None:
                frame = self._zone1_camera.get_latest_annotated()
            if frame is None and person is not None:
                frame = person.get("crop_bgr")
            if frame is None:
                return None

            frame = frame.copy()
            h = frame.shape[0]
            emp_name = EMPLOYEE_NAMES.get(emp_id, emp_id or "?")
            ts_str = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                f"{ts_str}",
                f"Card: {emp_id} ({emp_name})  Track: {key}",
                f"Result: {result_tag}  Score: {score:.3f}",
            ]
            # nền tối cho dễ đọc chữ
            _cv2.rectangle(frame, (0, h - 78), (frame.shape[1], h), (0, 0, 0), -1)
            y = h - 56
            for ln in lines:
                _cv2.putText(frame, ln, (12, y), _cv2.FONT_HERSHEY_SIMPLEX,
                             0.6, (0, 255, 0), 2, _cv2.LINE_AA)
                y += 24

            day_dir = os.path.join(ZONE1_EVIDENCE_DIR, _dt.now().strftime("%Y%m%d"))
            os.makedirs(day_dir, exist_ok=True)
            fname = f"{_dt.now().strftime('%H%M%S')}_{result_tag}_{emp_id}.jpg"
            path = os.path.join(day_dir, fname)
            _cv2.imwrite(path, frame, [_cv2.IMWRITE_JPEG_QUALITY, 90])
            log.info(f"Evidence saved: {path} | Đã lưu ảnh bằng chứng quẹt thẻ")
            return path
        except Exception as e:
            log.warning(f"Evidence save failed: {e}")
            return None

    def _save_anomaly_evidence(self, atype, key, ts, detail, frame=None):
        """Lưu ảnh evidence cho anomaly (frame Zone 1 + banner song ngữ).

        Dùng PIL để hiện đúng dấu tiếng Việt. frame=None → lấy latest annotated.
        """
        try:
            if frame is None and self._zone1_camera is not None:
                frame = self._zone1_camera.get_latest_annotated()
            if frame is None:
                return None

            from PIL import Image, ImageDraw, ImageFont
            rgb = _cv2.cvtColor(frame.copy(), _cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
            except Exception:
                font = ImageFont.load_default()

            ts_str = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                ts_str,
                f"ANOMALY: {atype} | {vi(atype)}",
                f"Track: {key}   {detail}",
            ]
            w, h = img.size
            draw.rectangle([0, h - 92, w, h], fill=(0, 0, 0))
            y = h - 86
            for ln in lines:
                draw.text((12, y), ln, fill=(255, 170, 0), font=font)
                y += 28

            out = _cv2.cvtColor(np.array(img), _cv2.COLOR_RGB2BGR)
            day_dir = os.path.join(ZONE1_EVIDENCE_DIR, _dt.now().strftime("%Y%m%d"))
            os.makedirs(day_dir, exist_ok=True)
            fname = f"{_dt.now().strftime('%H%M%S')}_{atype}_{key.replace(':', '_')}.jpg"
            path = os.path.join(day_dir, fname)
            _cv2.imwrite(path, out, [_cv2.IMWRITE_JPEG_QUALITY, 90])
            log.info(f"Anomaly evidence saved: {path} | Đã lưu ảnh bằng chứng cảnh báo")
            return path
        except Exception as e:
            log.warning(f"Anomaly evidence save failed: {e}")
            return None

    # ── RFID-trigger visual verify (face-first) ───────────────────────────────
    def _rfid_trigger_verify(self, emp_id: str, ts):
        """
        Gọi ngay sau khi nhận RFID event.
        Flow: get_best_person → InsightFace identify → quyết định confirm/anomaly.

        Nếu không có face engine: fallback body Re-ID (giữ backward compat).
        Nếu không có person tại gate: log no_person_at_gate, fallback Hungarian.
        """
        person = self._zone1_camera.get_best_person(
            max_age_sec=ZONE1_BEST_PERSON_MAX_AGE_SEC
        )

        if person is None:
            log.warning(
                f"RFID-trigger: {emp_id} — no_person_at_gate "
                f"(buffer empty or stale > {ZONE1_BEST_PERSON_MAX_AGE_SEC}s), "
                f"fallback to Hungarian | {vi('no_person_at_gate')}"
            )
            return

        track_id = person["track_id"]
        key      = f"zone1:track_{track_id}"

        # ── Fallback: không có InsightFace → dùng body Re-ID như cũ ──────────
        if self.face_engine is None:
            log.warning(
                f"RFID-trigger: {emp_id} — InsightFace not available, "
                f"fallback to body Re-ID"
            )
            self._rfid_trigger_body_fallback(emp_id, ts, person, key)
            return

        # ── Face-first verify ─────────────────────────────────────────────────
        face_crop = person["crop_bgr"]
        if ZONE1_FACE_UPSCALE:
            h_c, w_c = face_crop.shape[:2]
            if w_c < ZONE1_FACE_UPSCALE_MIN_W:
                scale     = ZONE1_FACE_UPSCALE_MIN_W / w_c
                face_crop = _cv2.resize(
                    face_crop,
                    (int(w_c * scale), int(h_c * scale)),
                    interpolation=_cv2.INTER_CUBIC,
                )
                log.info(
                    f"RFID-trigger face upscale: {w_c}x{h_c} "
                    f"-> {face_crop.shape[1]}x{face_crop.shape[0]} "
                    f"(bbox_area_ratio={person['bbox_area_ratio']:.3f})"
                )
        face_result = self.face_engine.identify_image(face_crop)
        status      = face_result.get("status", "UNKNOWN")
        face_id     = face_result.get("employee_id")
        face_score  = float(face_result.get("score", 0.0))

        log.info(
            f"RFID-trigger face: card={emp_id} track={track_id} "
            f"face_status={status} face_id={face_id} face_score={face_score:.3f} "
            f"bbox_area_ratio={person['bbox_area_ratio']:.3f} conf={person['confidence']:.2f}"
        )

        # ── Case 1: Face match đúng employee_id ──────────────────────────────
        if status == "MATCH" and face_id == emp_id:
            self._confirmed[key] = emp_id
            self._pending[emp_id].clear()
            self.db.log_entry(emp_id, key, ts, face_score)
            self.db.update_presence(emp_id, 1, key, source="rfid_face_verified")
            log.info(
                f"RFID-trigger CONFIRMED (face_verified): "
                f"{key} -> {emp_id} face_score={face_score:.3f} | {vi('confirmed_face_verified')}"
            )
            self._save_zone1_evidence(emp_id, key, ts, "confirmed_face_verified", face_score, person)
            return

        # ── Case 2: Face match nhưng ra người khác → proxy_swipe ─────────────
        if status == "MATCH" and face_id is not None and face_id != emp_id:
            log.warning(
                f"RFID-trigger PROXY SWIPE: card={emp_id} "
                f"but face matched {face_id} score={face_score:.3f} | {vi('proxy_swipe')}"
            )
            self.db.log_anomaly(
                "proxy_swipe", emp_id, key,
                f"card={emp_id} but face={face_id} score={face_score:.3f}", ts,
            )
            self._send_ntfy_alert(
                "proxy_swipe", "zone1", ts,
                track_id=key,
                employee_id=emp_id,
                detail=(
                    f"Card={emp_id} nhưng face nhận ra {face_id} "
                    f"score={face_score:.3f}"
                ),
                frame=person["crop_bgr"],
                prefix="proxy_swipe",
            )
            self._save_zone1_evidence(emp_id, key, ts, "proxy_swipe", face_score, person)
            # Đã xử lý xong cú quẹt này (proxy) → clear pending để không bắn rfid_no_crossing dư
            self._pending[emp_id].clear()
            return

        # ── Case 3: Face detected nhưng UNKNOWN / score thấp ─────────────────
        if status in ("MATCH", "UNKNOWN") and status != "MATCH":
            log.warning(
                f"RFID-trigger MISMATCH (face_unknown): card={emp_id} "
                f"face_score={face_score:.3f} | {vi('visual_mismatch_low_confidence')}"
            )
            self.db.log_anomaly(
                "visual_mismatch_low_confidence", emp_id, key,
                f"card={emp_id} face_status={status} score={face_score:.3f}", ts,
            )
            return

        # ── Case 4: No face detected ──────────────────────────────────────────
        # status == "NO_FACE" hoặc không detect được
        if status == "NO_FACE" or face_id is None:
            log.warning(
                f"RFID-trigger: {emp_id} — no_face_at_gate "
                f"(ZONE1_FACE_REQUIRED={ZONE1_FACE_REQUIRED}) | {vi('no_face_at_gate')}"
            )
            if not ZONE1_FACE_REQUIRED:
                # Confirm bằng RFID + person presence
                self._confirmed[key] = emp_id
                self._pending[emp_id].clear()
                self.db.log_entry(emp_id, key, ts, 0.0)
                self.db.update_presence(
                    emp_id, 1, key, source="rfid_presence_only_fallback"
                )
                log.info(
                    f"RFID-trigger CONFIRMED (presence_only_fallback): "
                    f"{key} -> {emp_id} | {vi('presence_only_fallback')}"
                )
                self._save_zone1_evidence(emp_id, key, ts, "presence_only_fallback", 0.0, person)
            else:
                self.db.log_anomaly(
                    "no_face_at_gate", emp_id, key,
                    f"card={emp_id} face not detected, entry pending", ts,
                )
            return

        # ── Catch-all: trạng thái không xác định, fallback Hungarian ─────────
        log.warning(
            f"RFID-trigger: {emp_id} unhandled face status={status}, "
            f"fallback to Hungarian"
        )

    def _rfid_trigger_body_fallback(self, emp_id: str, ts, person: dict, key: str):
        """Body Re-ID fallback khi InsightFace không có."""
        emb = self.reid.get_embedding(person["crop_bgr"])
        if emb is None:
            log.warning(f"RFID-trigger body fallback: {emp_id} embedding failed")
            return

        gallery_embeds = self.gallery.get(emp_id)
        if not gallery_embeds:
            log.warning(f"RFID-trigger body fallback: {emp_id} no gallery embeddings")
            return

        score = float(self.reid.match_score(emb, gallery_embeds))
        log.info(
            f"RFID-trigger body fallback: {emp_id} "
            f"score={score:.3f} threshold={RFID_VISUAL_MATCH_THRESHOLD}"
        )

        if score >= RFID_VISUAL_MATCH_THRESHOLD:
            self._confirmed[key] = emp_id
            self._pending[emp_id].clear()
            self.gallery.update(emp_id, emb)
            self.db.log_entry(emp_id, key, ts, score)
            self.db.update_presence(emp_id, 1, key, source="rfid_body_fallback")
            log.info(
                f"RFID-trigger CONFIRMED (body_fallback): "
                f"{key} -> {emp_id} score={score:.3f}"
            )
        else:
            log.warning(
                f"RFID-trigger body fallback MISMATCH: card={emp_id} "
                f"score={score:.3f} < {RFID_VISUAL_MATCH_THRESHOLD}"
            )
            self.db.log_anomaly(
                "visual_mismatch", emp_id, key,
                f"card={emp_id} body score={score:.3f}", ts,
            )
            self._send_ntfy_alert(
                "visual_mismatch", "zone1", ts,
                track_id=key,
                employee_id=emp_id,
                detail=(
                    f"Card={emp_id} body score={score:.3f} "
                    f"< threshold={RFID_VISUAL_MATCH_THRESHOLD}"
                ),
                frame=person["crop_bgr"],
                prefix="visual_mismatch",
            )

    # ── Zone 1 entry crossing ─────────────────────────────────────────────────
    def on_entry_crossing(self, zone: str, track_id, crop_bgr, timestamp, frame_bgr=None):
        key = self._track_key(zone, track_id)
        emb = self.reid.get_embedding(crop_bgr)

        entry = {
            "zone": zone, "track_id": track_id, "track_key": key,
            "ts": timestamp, "time": time.time(),
            "embedding": emb, "assigned": False,
            "alert_frame": frame_bgr.copy() if hasattr(frame_bgr, "copy") else crop_bgr.copy(),
        }
        self._crossings.append(entry)
        self._unmatched_crossings.append(entry)
        log.info(f"Entry crossing: {key}")
        self._hungarian_match()

    # ── Zone generic identify ─────────────────────────────────────────────────
    def identify_zone(self, zone: str, zone_number: int, track_id, crop_bgr, frame_bgr=None):
        key    = self._track_key(zone, track_id)
        emp_id = self.reid.identify(crop_bgr, self.gallery.all())
        if emp_id:
            self._confirmed[key] = emp_id
            log.info(f"{zone} re-id: {key} -> {emp_id}")
            self.db.update_presence(emp_id, zone_number, key, source=f"cv_{zone}")
        else:
            self._send_ntfy_alert(
                "unknown_person",
                zone,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                track_id=key,
                detail="Person not matched with gallery",
                frame=frame_bgr if frame_bgr is not None else crop_bgr,
                prefix=f"{zone}_unknown",
            )
        return emp_id

    # ── Zone 2 main entry point ───────────────────────────────────────────────
    def identify_zone2(self, track_id, crop_bgr, crop_quality_ok: bool = True, frame_bgr=None):
        if not crop_quality_ok:
            # Crop xấu: giữ lock cũ nếu có
            return self._get_zone2_track_lock(track_id)

        # Có face engine: LUÔN chạy hybrid để face có cơ hội override lock cũ
        if self.face_engine is not None:
            return self._identify_zone2_hybrid(track_id, crop_bgr, frame_bgr=frame_bgr)

        # Không có face engine: lock thì giữ, chưa lock thì body-only
        locked_id = self._get_zone2_track_lock(track_id)
        if locked_id:
            return locked_id
        return self._identify_zone2_body_only(track_id, crop_bgr, frame_bgr=frame_bgr)

    # ── Body-only path (không có face engine) ────────────────────────────────
    def _identify_zone2_body_only(self, track_id, crop_bgr, frame_bgr=None):
        body = self._rank_body(crop_bgr)
        key  = self._track_key("zone2", track_id)

        body_lockable = (
            body["best_score"] >= BODY_LOCK_MIN_SCORE
            and body["margin"]  >= BODY_LOCK_MIN_MARGIN
        )

        if body_lockable:
            final_id = body["best_id"]
            self._lock_zone2_track(track_id, final_id, "body_strict")
            self._log_zone2_decision(key, final_id, "body_strict",
                                     body["best_score"], body["margin"])
            return final_id

        self._log_zone2_decision(key, None, "body_low_margin",
                                 body["best_score"], body["margin"])
        self._send_ntfy_alert(
            "unknown_person",
            "zone2",
            time.strftime("%Y-%m-%d %H:%M:%S"),
            track_id=key,
            detail=(
                f"Body Re-ID low confidence: best={body['best_id']} "
                f"score={body['best_score']:.3f} margin={body['margin']:.3f}"
            ),
            frame=frame_bgr if frame_bgr is not None else crop_bgr,
            prefix="zone2_unknown",
        )
        return None

    # ── Hybrid path (có face engine) ─────────────────────────────────────────
    def _identify_zone2_hybrid(self, track_id, crop_bgr, frame_bgr=None):
        key         = self._track_key("zone2", track_id)
        face_result = self.face_engine.identify_image(crop_bgr)
        face        = self._face_summary(face_result)
        body        = self._rank_body(crop_bgr)

        locked_id = self._zone2_track_locks.get(key, {}).get("employee_id")
        final_id  = None
        reason    = "no_confirm"

        # ── Rule 1: Face MATCH rõ ràng ────────────────────────────────────────
        if face["status"] == "MATCH":
            face_strong = (
                face["score"]  >= FACE_OVERRIDE_MIN_SCORE
                and face["margin"] >= FACE_OVERRIDE_MIN_MARGIN
            )
            if face_strong:
                final_id = face["id"]
                reason   = "face_match"

                # Face override lock cũ nếu khác ID
                if locked_id and locked_id != final_id:
                    log.info(
                        f"zone2 relock: {key} {locked_id} -> {final_id} "
                        f"reason=face_override "
                        f"face_score={face['score']:.3f} face_margin={face['margin']:.3f}"
                    )
                elif body["best_id"] and body["best_id"] != final_id:
                    log.warning(
                        f"zone2 body-face conflict: {key} "
                        f"body={body['best_id']}({body['best_score']:.3f}/"
                        f"{body['margin']:.3f}) "
                        f"face={face['id']}({face['score']:.3f}) → keep face"
                    )
            else:
                # Face MATCH nhưng score yếu → treat như UNKNOWN
                reason = "face_match_weak"
                if locked_id:
                    final_id = locked_id
                    reason   = "face_match_weak_keep_lock"

        # ── Rule 2: Đang locked + không có face mới → giữ lock ───────────────
        elif locked_id and face["status"] in ("NO_FACE", "UNKNOWN"):
            # Refresh lock TTL thông qua _get_zone2_track_lock
            final_id = self._get_zone2_track_lock(track_id)
            reason   = f"locked_keep_{face['status'].lower()}"

        # ── Rule 3: Chưa locked + face UNKNOWN → body strict ─────────────────
        elif face["status"] == "UNKNOWN":
            body_strict = (
                body["best_score"] >= BODY_LOCK_MIN_SCORE
                and body["margin"]  >= BODY_LOCK_MIN_MARGIN
            )
            if body_strict:
                final_id = body["best_id"]
                reason   = "face_unknown_body_strict"
            else:
                reason   = "face_unknown_body_margin_low"

        # ── Rule 4: Chưa locked + NO_FACE → body strict ──────────────────────
        elif face["status"] == "NO_FACE":
            body_strict = (
                body["best_score"] >= BODY_LOCK_MIN_SCORE
                and body["margin"]  >= BODY_LOCK_MIN_MARGIN
            )
            if body_strict:
                final_id = body["best_id"]
                reason   = "no_face_body_strict"
            else:
                reason   = "no_face_body_margin_low"

        # ── Rule 5: Face trạng thái khác → body fallback ─────────────────────
        else:
            if locked_id:
                final_id = self._get_zone2_track_lock(track_id)
                reason   = f"face_{face['status'].lower()}_keep_lock"
            else:
                body_strict = (
                    body["best_score"] >= BODY_LOCK_MIN_SCORE
                    and body["margin"]  >= BODY_LOCK_MIN_MARGIN
                )
                if body_strict:
                    final_id = body["best_id"]
                    reason   = f"face_{face['status'].lower()}_body_strict"
                else:
                    reason   = f"face_{face['status'].lower()}_body_margin_low"

        # ── Lock / relock nếu có final_id từ quyết định mới ──────────────────
        if final_id and reason in (
            "face_match", "face_unknown_body_strict",
            "no_face_body_strict",
        ) or (final_id and face["status"] == "MATCH" and final_id != locked_id):
            lock_reason = "face_match" if "face_match" in reason else "body_strict"
            self._lock_zone2_track(track_id, final_id, lock_reason)

        # ── Log throttled ─────────────────────────────────────────────────────
        self._log_zone2_decision(
            key, final_id, reason,
            body["best_score"], body["margin"],
            face_id=face["id"], face_score=face["score"],
            face_margin=face["margin"], face_status=face["status"],
        )

        if final_id is None:
            self._send_ntfy_alert(
                "unknown_person",
                "zone2",
                time.strftime("%Y-%m-%d %H:%M:%S"),
                track_id=key,
                detail=(
                    f"Hybrid Re-ID unknown: reason={reason}; "
                    f"body={body['best_id']} score={body['best_score']:.3f} "
                    f"margin={body['margin']:.3f}; "
                    f"face={face['id']} status={face['status']} "
                    f"score={face['score']:.3f} margin={face['margin']:.3f}"
                ),
                frame=frame_bgr if frame_bgr is not None else crop_bgr,
                prefix="zone2_unknown",
            )

        return final_id

    # ── Body ranker ───────────────────────────────────────────────────────────
    def _rank_body(self, crop_bgr) -> dict:
        emb = self.reid.get_embedding(crop_bgr)
        empty = {
            "best_id": None, "best_score": 0.0,
            "second_id": None, "second_score": 0.0,
            "margin": 0.0, "identified": None,
        }
        if emb is None:
            return empty

        rows = [
            (emp_id, self.reid.match_score(emb, embeds))
            for emp_id, embeds in self.gallery.all().items()
            if embeds
        ]
        if not rows:
            return empty

        rows.sort(key=lambda x: x[1], reverse=True)
        best_id, best_score     = rows[0]
        second_id, second_score = rows[1] if len(rows) > 1 else (None, 0.0)
        margin = float(best_score - second_score)

        identified = (
            best_id
            if best_score >= REID_MATCH_THRESHOLD and margin >= REID_MATCH_MARGIN
            else None
        )

        return {
            "best_id":      best_id,
            "best_score":   float(best_score),
            "second_id":    second_id,
            "second_score": float(second_score),
            "margin":       margin,
            "identified":   identified,
        }

    # ── Face summary helper ───────────────────────────────────────────────────
    def _face_summary(self, face_result) -> dict:
        status  = face_result.get("status", "UNKNOWN")
        ranking = face_result.get("ranking", [])

        if status == "MATCH":
            face_id = face_result.get("employee_id")
        elif ranking:
            face_id = ranking[0].get("employee_id")
        else:
            face_id = None

        return {
            "id":       face_id,
            "score":    float(face_result.get("score", 0.0)),
            "margin":   float(face_result.get("margin", 0.0)),
            "status":   status,
            "decision": f"MATCH:{face_id}" if status == "MATCH" else status,
        }

    # ── Lock helpers ──────────────────────────────────────────────────────────
    def _get_zone2_track_lock(self, track_id) -> str | None:
        key  = self._track_key("zone2", track_id)
        lock = self._zone2_track_locks.get(key)
        if lock is None:
            return None

        now = time.time()
        if now > lock["expires_at"]:
            self._zone2_track_locks.pop(key, None)
            self._confirmed.pop(key, None)
            log.info(f"zone2 lock expired: {key} -> {lock['employee_id']}")
            return None

        # Refresh TTL
        lock["last_seen"]  = now
        lock["expires_at"] = now + ZONE2_TRACK_LOCK_TTL

        # Throttle log cho locked track
        last_log = lock.get("last_log_at", 0.0)
        if now - last_log >= ZONE2_LOG_INTERVAL:
            log.info(f"zone2 locked: {key} -> {lock['employee_id']} "
                     f"reason={lock['reason']}")
            lock["last_log_at"] = now

        self.db.update_presence(
            lock["employee_id"], 2, key,
            source=lock.get("source", "cv_zone2"),
        )
        return lock["employee_id"]

    def _lock_zone2_track(self, track_id, emp_id: str, reason: str):
        key    = self._track_key("zone2", track_id)
        now    = time.time()
        source = "cv_zone2_face_match" if reason == "face_match" else "cv_zone2"

        self._confirmed[key] = emp_id
        self._zone2_track_locks[key] = {
            "employee_id": emp_id,
            "reason":      reason,
            "source":      source,
            "locked_at":   now,
            "last_seen":   now,
            "last_log_at": now,
            "expires_at":  now + ZONE2_TRACK_LOCK_TTL,
        }
        self.db.update_presence(emp_id, 2, key, source=source)
        log.info(f"zone2 lock new: {key} -> {emp_id} reason={reason}")

    # ── Throttled decision log ────────────────────────────────────────────────
    def _log_zone2_decision(
        self, key, final_id, reason,
        body_score, body_margin,
        face_id=None, face_score=0.0, face_margin=0.0, face_status=None,
    ):
        now  = time.time()
        prev = self._zone2_last_decision.get(key)

        # Log khi decision đổi HOẶC đủ interval
        decision_changed = (prev is None or prev[0] != final_id)
        time_ok          = (prev is None or now - prev[1] >= ZONE2_LOG_INTERVAL)

        if not (decision_changed or time_ok):
            return

        self._zone2_last_decision[key] = (final_id, now)

        if face_status is not None:
            log.info(
                f"zone2 hybrid: {key} "
                f"body=({body_score:.3f}/{body_margin:.3f}) "
                f"face={face_id}({face_score:.3f}/{face_margin:.3f})[{face_status}] "
                f"→ {final_id} [{reason}] | {vi(reason)}"
            )
        else:
            log.info(
                f"zone2 body: {key} "
                f"score={body_score:.3f} margin={body_margin:.3f} "
                f"→ {final_id} [{reason}] | {vi(reason)}"
            )

    # ── Public helpers ────────────────────────────────────────────────────────
    def _confirm_zone_identity(self, zone, zone_number, track_id, emp_id, source=None):
        key = self._track_key(zone, track_id)
        self._confirmed[key] = emp_id
        self.db.update_presence(emp_id, zone_number, key, source=source or f"cv_{zone}")
        return key

    def get_label(self, track_id, zone: str | None = None):
        if zone is not None:
            key = self._track_key(zone, track_id)
            return self._confirmed.get(key, f"{zone}:track_{track_id}")
        return self._confirmed.get(str(track_id), f"track_{track_id}")

    def is_confirmed(self, track_id, zone: str | None = None):
        if zone is not None:
            key = self._track_key(zone, track_id)
            return key in self._confirmed
        return str(track_id) in self._confirmed

    # ── Hungarian RFID+CV match ───────────────────────────────────────────────
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

        n    = len(pending_list)
        m    = len(unassigned)
        cost = np.full((n, m + n), 1e6)

        for i, (emp_id, rfid) in enumerate(pending_list):
            cost[i][m + i] = 1.0 - FUSION_THRESHOLD
            for j, crossing in enumerate(unassigned):
                dt = abs((crossing["ts"] - rfid["ts"]).total_seconds())
                if dt > FUSION_TIME_WINDOW:
                    continue
                gallery_embeds = self.gallery.get(emp_id)
                app_score = (
                    self.reid.match_score(crossing["embedding"], gallery_embeds)
                    if gallery_embeds and crossing["embedding"] is not None
                    else 0.5
                )
                time_score  = 1.0 - (dt / FUSION_TIME_WINDOW)
                total_score = FUSION_WEIGHT_TIME * time_score + FUSION_WEIGHT_REID * app_score
                cost[i][j]  = 1.0 - total_score

        row_ind, col_ind = linear_sum_assignment(cost)

        for i, j in zip(row_ind, col_ind):
            emp_id = pending_list[i][0]
            if j >= m:
                continue
            score = 1.0 - cost[i][j]
            if score < FUSION_THRESHOLD:
                log.debug(f"No match for {emp_id} best={score:.3f}")
                continue

            crossing = unassigned[j]
            key      = crossing["track_key"]

            self._confirmed[key] = emp_id
            crossing["assigned"] = True

            self._unmatched_crossings = deque(
                [c for c in self._unmatched_crossings if c["track_key"] != key],
                maxlen=50,
            )
            self._pending[emp_id].clear()

            if crossing["embedding"] is not None:
                self.gallery.update(emp_id, crossing["embedding"])

            self.db.log_entry(emp_id, key, crossing["ts"], score)
            self.db.update_presence(emp_id, 1, key, source="rfid_cv_fusion")
            log.info(f"CONFIRMED: {key} -> {emp_id} score={score:.3f} | Vào kho được xác nhận")

            if score < PROXY_THRESHOLD:
                log.warning(f"PROXY SWIPE: {emp_id} score={score:.3f} | {vi('proxy_swipe')}")
                self.db.log_anomaly("proxy_swipe", emp_id, key,
                                    f"Low score={score:.3f}", crossing["ts"])
                self._send_ntfy_alert(
                    "proxy_swipe",
                    crossing["zone"],
                    crossing["ts"],
                    track_id=key,
                    employee_id=emp_id,
                    detail=f"Low fusion score={score:.3f}",
                    frame=crossing.get("alert_frame"),
                    prefix="proxy_swipe",
                )

    # ── Cleanup loop ──────────────────────────────────────────────────────────
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
                    key        = crossing["track_key"]
                    # Track đã được xác nhận entry qua RFID-trigger (B-lite) → không phải intruder.
                    # B-lite lưu _confirmed dạng "zone1:track_N", còn crossing dùng "zone1:N".
                    blite_key = f"{crossing['zone']}:track_{crossing['track_id']}"
                    if key in self._confirmed or blite_key in self._confirmed:
                        remove_list.append(crossing)
                        continue
                    last_alert = self._intruder_alerted.get(key, 0.0)
                    if now - last_alert >= INTRUDER_ALERT_COOLDOWN:
                        log.warning(f"INTRUDER ALERT: {key} no RFID after {age:.0f}s | {vi('no_rfid_intruder')}")
                        self.db.log_anomaly("no_rfid_intruder", None, key,
                                            f"No RFID after {age:.0f}s", crossing["ts"])
                        self._send_ntfy_alert(
                            "no_rfid_intruder",
                            crossing["zone"],
                            crossing["ts"],
                            track_id=key,
                            detail=f"No RFID after {age:.0f}s",
                            frame=crossing.get("alert_frame"),
                            prefix="no_rfid_intruder",
                        )
                        self._save_anomaly_evidence(
                            "no_rfid_intruder", key, crossing["ts"],
                            f"No RFID after {age:.0f}s",
                            frame=crossing.get("alert_frame"),
                        )
                        self._intruder_alerted[key] = now
                    else:
                        log.debug(
                            f"INTRUDER throttled: {key} "
                            f"(last alerted {now - last_alert:.0f}s ago)"
                        )
                    remove_list.append(crossing)
            for c in remove_list:
                try:
                    self._unmatched_crossings.remove(c)
                except ValueError:
                    pass

            for key, lock in list(self._zone2_track_locks.items()):
                if now > lock["expires_at"]:
                    emp_id = lock["employee_id"]
                    self._zone2_track_locks.pop(key, None)
                    self._confirmed.pop(key, None)
                    log.info(f"zone2 lock expired (cleanup): {key} -> {emp_id}")

            for key in list(self._intruder_alerted):
                if now - self._intruder_alerted[key] > INTRUDER_ALERT_COOLDOWN:
                    self._intruder_alerted.pop(key, None)

            for emp_id, q in list(self._pending.items()):
                if not q:
                    continue
                if now > q[-1]["expires"]:
                    log.warning(f"RFID TIMEOUT: {emp_id} no crossing | Thẻ {emp_id} đã quẹt nhưng không phát hiện người đi qua cửa")
                    self.db.log_anomaly("rfid_no_crossing", emp_id, None,
                                        f"No crossing within {RFID_EXPIRE_WINDOW}s",
                                        q[-1]["ts"])
                    self._send_ntfy_alert(
                        "rfid_no_crossing",
                        "zone1",
                        q[-1]["ts"],
                        employee_id=emp_id,
                        detail=f"No crossing within {RFID_EXPIRE_WINDOW}s",
                        prefix="rfid_no_crossing",
                    )
                    self._save_anomaly_evidence(
                        "rfid_no_crossing", f"zone1:{emp_id}", q[-1]["ts"],
                        f"No crossing within {RFID_EXPIRE_WINDOW}s",
                    )
                    q.clear()
