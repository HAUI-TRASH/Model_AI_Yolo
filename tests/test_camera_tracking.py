import threading
import unittest

try:
    from trashsmart.pipeline.roi import BboxStabilizer, BBOX_DISAPPEAR_FRAMES, InferenceThread
except ModuleNotFoundError as error:
    raise unittest.SkipTest(str(error))


def detection(label, bbox):
    return {"label": label, "bbox": bbox, "object_label": "chai", "material_label": label,
            "label_resnet": label, "conf_resnet": 0.9, "label_yolo": "chai",
            "raw_yolo_label": "Bottle"}


class CameraTrackingTests(unittest.TestCase):
    def test_two_detections_cannot_share_one_existing_track(self):
        stabilizer = BboxStabilizer()
        stabilizer.update([detection("old", (0, 0, 100, 100))])
        stabilizer.update([detection("glass", (0, 0, 100, 100)),
                           detection("plastic", (10, 0, 110, 100))])
        self.assertEqual(len(stabilizer.get_stable()), 2)
        self.assertEqual({d["label"] for d in stabilizer.get_stable()}, {"glass", "plastic"})

    def test_missing_track_expires_after_completed_inferences(self):
        stabilizer = BboxStabilizer()
        stabilizer.update([detection("glass", (0, 0, 100, 100))])
        for _ in range(BBOX_DISAPPEAR_FRAMES - 1):
            stabilizer.update([])
        self.assertEqual(len(stabilizer.get_stable()), 1)
        stabilizer.update([])
        self.assertEqual(stabilizer.get_stable(), [])

    def test_strongest_overlap_wins_regardless_of_detection_order(self):
        stabilizer = BboxStabilizer()
        stabilizer.update([detection("old", (0, 0, 100, 100))])
        stabilizer.update([detection("weaker", (20, 0, 120, 100)),
                           detection("exact", (0, 0, 100, 100))])
        stable = stabilizer.get_stable()
        self.assertEqual(len(stable), 2)
        self.assertEqual(stable[0]["label"], "exact")

    def test_completed_result_is_consumed_once_including_empty_result(self):
        worker = InferenceThread.__new__(InferenceThread)
        worker._lock = threading.Lock()
        worker._detections = []
        worker._result_pending = True
        self.assertEqual(worker.get_detections(), [])
        self.assertIsNone(worker.get_detections())
