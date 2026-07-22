import os
import threading
import time

import cv2
import requests


class NtfyAlert:
    def __init__(self):
        self.enabled = os.getenv("ENABLE_NTFY_ALERT", "false").lower() == "true"
        self.server = os.getenv("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        self.topic = os.getenv("NTFY_TOPIC", "")
        self.alert_dir = os.getenv("NTFY_ALERT_DIR", "outputs/ntfy_alerts")
        self.cooldown_seconds = int(os.getenv("NTFY_COOLDOWN_SECONDS", "5"))
        self.last_sent_time = 0.0
        self._lock = threading.Lock()

        os.makedirs(self.alert_dir, exist_ok=True)

    @staticmethod
    def _enc(value: str) -> str:
        """Cho phép UTF-8 (emoji + tiếng Việt) trong HTTP header.

        requests mã hoá header value bằng latin-1 → lỗi với emoji/dấu tiếng Việt.
        Ta gửi raw UTF-8 bytes (ntfy server decode UTF-8) bằng roundtrip latin-1.
        """
        return value.encode("utf-8").decode("latin-1")

    def can_send(self):
        if not self.enabled:
            print("[NTFY] Disabled")
            return False

        if not self.topic:
            print("[NTFY] Missing topic")
            return False

        with self._lock:
            now = time.time()
            if now - self.last_sent_time < self.cooldown_seconds:
                return False

            self.last_sent_time = now
            return True

    def save_frame(self, frame, prefix="alert"):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(self.alert_dir, f"{prefix}_{timestamp}.jpg")
        cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return image_path

    def send_text(self, message, title="RFID-CV Cảnh báo", priority="urgent"):
        if not self.can_send():
            return False

        url = f"{self.server}/{self.topic}"
        safe_title = title.replace("\n", " | ").replace("\r", "")
        headers = {
            "Title": self._enc(safe_title),
            "Priority": priority,
            "Tags": "warning,camera",
        }

        try:
            response = requests.post(
                url,
                data=message.encode("utf-8"),
                headers=headers,
                timeout=10,
            )
            print("[NTFY] Text status:", response.status_code)
            return response.status_code in (200, 201)
        except Exception as e:
            print("[NTFY] Text error:", e)
            return False

    def send_photo(self, image_path, message="Phát hiện cảnh báo từ hệ thống", title="RFID-CV Cảnh báo ảnh"):
        if not self.can_send():
            return False

        return self._post_photo(image_path, message=message, title=title)

    def send_frame(self, frame, message="Phát hiện cảnh báo từ hệ thống", prefix="alert"):
        if not self.can_send():
            return False

        image_path = self.save_frame(frame, prefix=prefix)
        return self._post_photo(image_path, message=message, title="RFID-CV Image Alert")

    def _post_photo(self, image_path, message, title):
        url = f"{self.server}/{self.topic}"
        safe_title = title.replace("\n", " | ").replace("\r", "")
        safe_message = message.replace("\n", " | ").replace("\r", "")
        headers = {
            "Title": self._enc(safe_title),
            "Message": self._enc(safe_message),
            "Priority": "urgent",
            "Tags": "warning,camera",
            "Filename": os.path.basename(image_path),
            "Content-Type": "image/jpeg",
        }

        try:
            with open(image_path, "rb") as f:
                response = requests.post(
                    url,
                    data=f,
                    headers=headers,
                    timeout=15,
                )

            print("[NTFY] Photo status:", response.status_code)
            if response.status_code not in (200, 201):
                print("[NTFY] Response:", response.text)

            return response.status_code in (200, 201)

        except Exception as e:
            print("[NTFY] Photo error:", e)
            return False
