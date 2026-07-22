"""
Mapping status code / alert type -> chuoi hien thi tieng Viet.

Dung cho: NTFY message, CSV export, Dashboard JS (copy thu cong).
Code noi bo (log ky thuat, DB schema, API key) khong doi.
"""

_VI_TEXT: dict[str, str] = {
    # Alert / anomaly types (cot type trong anomaly_log)
    "no_rfid_intruder": "Phat hien nguoi di qua nhung khong co RFID",
    "unknown_uid": "UID the khong co trong he thong",
    "proxy_swipe": "Nghi ngo quet the ho",
    "visual_mismatch": "The va dang nguoi khong khop",
    "visual_mismatch_low_confidence": "Khuon mat phat hien nhung do tin cay thap",
    "rfid_no_crossing": "The da quet nhung khong phat hien nguoi di qua cua",
    "no_face_at_gate": "Khong thay khuon mat tai cong khi quet the",
    "unknown_person": "Phat hien nguoi chua xac dinh",
    "no_person_at_gate": "Khong phat hien nguoi tai cua khi quet the",

    # Entry / presence sources
    "rfid_face_verified": "Xac nhan bang khuon mat (RFID-trigger)",
    "rfid_presence_only_fallback": "Xac nhan bang hien dien tai cua",
    "rfid_body_fallback": "Xac nhan bang dang nguoi (RFID-trigger)",
    "rfid_cv_fusion": "Xac nhan bang RFID + CV (Hungarian)",
    "cv_zone2": "Re-ID Zone 2",
    "cv_zone2_face_match": "Khop khuon mat Zone 2",

    # Zone 2 decision reasons
    "face_match": "Khop khuon mat",
    "body_strict": "Khop nhan dang dang nguoi chac chan",
    "no_face": "Khong thay khuon mat",
    "locked_keep_no_face": "Giu dinh danh cu do tam thoi khong thay khuon mat",
    "locked_keep_unknown": "Giu dinh danh cu do khuon mat chua du chac chan",
    "face_unknown_body_strict": "Khuon mat khong ro, khop bang dang nguoi",
    "face_unknown_body_margin_low": "Khuon mat khong ro, do chenh dang nguoi chua du chac chan",
    "no_face_body_strict": "Khong thay khuon mat, khop bang dang nguoi",
    "no_face_body_margin_low": "Khong thay khuon mat, do chenh dang nguoi chua du chac chan",
    "face_match_weak": "Khuon mat khop nhung diem so yeu",
    "face_match_weak_keep_lock": "Khuon mat khop yeu, giu dinh danh dang khoa",

    # Entry flow statuses (dung trong NTFY / bao cao)
    "presence_only_fallback": "Xac nhan bang hien dien tai cua",
    "confirmed_face_verified": "Xac nhan thanh cong bang khuon mat",
    "rfid_timeout": "The da quet nhung khong phat hien nguoi di qua cua",

    # Event types (cho CSV export)
    "rfid_scan": "Quet the RFID",
    "entry_confirmed": "Vao kho duoc xac nhan",
    "presence_update": "Cap nhat vi tri trong kho",
}


def vi(code: str) -> str:
    """Tra ve chuoi tieng Viet cho status/alert code; fallback ve chinh code."""
    return _VI_TEXT.get(str(code), code)


# NTFY push notification: chi day cac loai canh bao.
PUSH_ALERT_TYPES: set[str] = {
    "no_rfid_intruder",
    "unknown_uid",
    "proxy_swipe",
    "visual_mismatch",
    "visual_mismatch_low_confidence",
    "rfid_no_crossing",
    "no_face_at_gate",
    "unknown_person",
    "no_person_at_gate",
}

_ALERT_TITLE: dict[str, str] = {
    "no_rfid_intruder": "Canh bao nguoi la xam nhap",
    "unknown_uid": "Canh bao the khong hop le",
    "proxy_swipe": "Canh bao nghi quet the ho",
    "visual_mismatch": "Canh bao the va nguoi khong khop",
    "visual_mismatch_low_confidence": "Canh bao nhan dien do tin cay thap",
    "rfid_no_crossing": "Canh bao the quet nhung khong qua cua",
    "no_face_at_gate": "Canh bao khong thay khuon mat tai cong",
    "unknown_person": "Canh bao nguoi chua xac dinh",
    "no_person_at_gate": "Canh bao quet the khong co nguoi",
}

_ALERT_STATUS: dict[str, str] = {
    "no_rfid_intruder": "Nguoi di qua cua nhung khong quet the",
    "unknown_uid": "The khong co trong he thong",
    "proxy_swipe": "The va khuon mat khong khop - nghi quet ho",
    "visual_mismatch": "Dang nguoi khong khop voi the",
    "visual_mismatch_low_confidence": "Khuon mat phat hien nhung do tin cay thap",
    "rfid_no_crossing": "Da quet the nhung khong thay nguoi di qua cua",
    "no_face_at_gate": "Khong thay khuon mat khi quet the",
    "unknown_person": "Chua xac dinh duoc nhan vien",
    "no_person_at_gate": "Khong thay nguoi tai cua khi quet the",
}

_ALERT_SEVERITY: dict[str, str] = {
    "no_rfid_intruder": "Nghiem trong",
    "proxy_swipe": "Nghiem trong",
}

_ZONE_NAME: dict[str, str] = {
    "zone1": "Zone 1 - Cua vao",
    "zone2": "Zone 2 - Trong kho",
}


def alert_title(code: str) -> str:
    return _ALERT_TITLE.get(str(code), "Canh bao bat thuong")


def alert_status(code: str) -> str:
    return _ALERT_STATUS.get(str(code), vi(code))


def alert_severity(code: str) -> str:
    return _ALERT_SEVERITY.get(str(code), "Can kiem tra")


def zone_name(code: str) -> str:
    return _ZONE_NAME.get(str(code), str(code))
