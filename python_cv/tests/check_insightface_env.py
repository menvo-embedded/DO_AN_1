# tests/check_insightface_env.py

import sys
import inspect

print("========== PYTHON ==========")
print("Python:", sys.executable)
print("Version:", sys.version)

print("\n========== INSIGHTFACE ==========")
try:
    import insightface
    print("insightface version:", getattr(insightface, "__version__", "UNKNOWN"))
    print("insightface file:", getattr(insightface, "__file__", "UNKNOWN"))
except Exception as e:
    print("import insightface ERROR:", repr(e))
    raise SystemExit

print("\n========== FACE ANALYSIS API ==========")
try:
    from insightface.app import FaceAnalysis

    print("FaceAnalysis class:", FaceAnalysis)
    print("FaceAnalysis.__init__ signature:")
    print(inspect.signature(FaceAnalysis.__init__))

    print("\nFaceAnalysis.prepare signature:")
    print(inspect.signature(FaceAnalysis.prepare))

except Exception as e:
    print("FaceAnalysis API ERROR:", repr(e))
    raise SystemExit

print("\n========== ONNXRUNTIME ==========")
try:
    import onnxruntime as ort
    print("onnxruntime version:", ort.__version__)
    print("available providers:", ort.get_available_providers())
except Exception as e:
    print("onnxruntime ERROR:", repr(e))

print("\n========== TEST CONSTRUCTORS ==========")

construct_tests = [
    ("FaceAnalysis()", lambda: FaceAnalysis()),
    ('FaceAnalysis(name="buffalo_sc")', lambda: FaceAnalysis(name="buffalo_sc")),
    ('FaceAnalysis("buffalo_sc")', lambda: FaceAnalysis("buffalo_sc")),
]

working_app = None

for name, fn in construct_tests:
    try:
        app = fn()
        print("[OK]", name)
        working_app = app
        break
    except Exception as e:
        print("[FAIL]", name, "=>", repr(e))

if working_app is None:
    print("\n[STOP] Không constructor nào chạy được.")
    raise SystemExit

print("\n========== TEST PREPARE ==========")

prepare_tests = [
    ('prepare(ctx_id=0, det_size=(640,640), det_thresh=0.3)', lambda: working_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)),
    ('prepare(ctx_id=0, det_size=(640,640))', lambda: working_app.prepare(ctx_id=0, det_size=(640, 640))),
    ('prepare(ctx_id=-1, det_size=(640,640), det_thresh=0.3)', lambda: working_app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)),
    ('prepare(ctx_id=-1, det_size=(640,640))', lambda: working_app.prepare(ctx_id=-1, det_size=(640, 640))),
]

for name, fn in prepare_tests:
    try:
        fn()
        print("[OK]", name)
        print("\n========== RESULT ==========")
        print("Use this prepare mode:", name)
        print("InsightFace environment check DONE.")
        raise SystemExit
    except Exception as e:
        print("[FAIL]", name, "=>", repr(e))

print("\n[STOP] Không prepare mode nào chạy được.")