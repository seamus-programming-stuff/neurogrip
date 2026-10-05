"""Stereo tests use actual OpenCV fitting/matching and independently known XYZ."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from stereo_core import StereoCalibration, StereoDepth, check_epipolar_alignment
from stereo_calibrate_core import calibrate_stereo


def calibration(size=(384, 192), baseline=60., rotation=None, translation=None):
    width, height = size
    K = np.array([[220., 0, width / 2], [0, 220., height / 2], [0, 0, 1]])
    R = np.eye(3) if rotation is None else rotation
    T = np.array([-baseline, 0., 0.]) if translation is None else np.asarray(translation, np.float64)
    D = np.zeros(5)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K, D, K, D, size, R, T.reshape(3, 1),
                                                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    return StereoCalibration(size, "synthetic:left", "synthetic:right", K, D, K, D,
                             R, T, R1, R2, P1, P2, Q, {"simulation": True}, roi1, roi2)


def textured_pair(disparity=20, size=(384, 192)):
    rng = np.random.default_rng(9812)
    left = cv2.GaussianBlur(rng.integers(0, 256, (size[1], size[0]), np.uint8), (3, 3), .6)
    right = rng.integers(0, 256, left.shape, np.uint8)
    right[:, :size[0]-disparity] = left[:, disparity:]
    return left, right


def projected_boards(noise=.04):
    """Board coordinates mm, two independently defined physical cameras."""
    rng = np.random.default_rng(56)
    size = (640, 480)
    K1 = np.array([[510., 0, 321.], [0, 506., 239.], [0, 0, 1.]])
    K2 = np.array([[500., 0, 318.], [0, 504., 242.], [0, 0, 1.]])
    D1 = np.array([-.04, .015, .0003, -.0002, 0.])
    D2 = np.array([-.03, .012, -.0002, .0001, 0.])
    R = cv2.Rodrigues(np.array([.006, .025, -.008]))[0]
    T = np.array([-65., 1.2, .8])
    board = np.zeros((9 * 6, 3), np.float32)
    board[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 15.
    objects, left, right = [], [], []
    while len(objects) < 18:
        rvec = rng.uniform([-.45, -.45, -.18], [.45, .45, .18])
        tvec = rng.uniform([-140., -105., 350.], [50., 55., 950.])
        board_rotation = cv2.Rodrigues(rvec)[0]
        right_rvec = cv2.Rodrigues(R @ board_rotation)[0]
        right_tvec = R @ tvec + T
        p1 = cv2.projectPoints(board, rvec, tvec, K1, D1)[0]
        p2 = cv2.projectPoints(board, right_rvec, right_tvec, K2, D2)[0]
        if any(np.any(p[:, 0, 0] < 8) or np.any(p[:, 0, 0] > size[0]-8)
               or np.any(p[:, 0, 1] < 8) or np.any(p[:, 0, 1] > size[1]-8) for p in (p1, p2)):
            continue
        p1 = (p1 + rng.normal(0, noise, p1.shape)).astype(np.float32)
        p2 = (p2 + rng.normal(0, noise, p2.shape)).astype(np.float32)
        objects.append(board.copy())
        left.append(p1)
        right.append(p2)
    return objects, left, right, size, R, T


class StereoCalibrationTests(unittest.TestCase):
    def test_q_reconstructs_known_xyz_in_rotated_left_frame(self):
        rotation = cv2.Rodrigues(np.array([.004, .035, -.012]))[0]
        rig = calibration(rotation=rotation, translation=[-70, 2, 1])
        xyz = np.array([[-65., -30., 450.], [20., 25., 800.], [100., -40., 1400.]])
        left = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), rig.K1, rig.D1)[0]
        right = cv2.projectPoints(xyz, cv2.Rodrigues(rig.R)[0], rig.T_mm, rig.K2, rig.D2)[0]
        a = cv2.undistortPoints(left, rig.K1, rig.D1, R=rig.R1, P=rig.P1).reshape(-1, 2)
        b = cv2.undistortPoints(right, rig.K2, rig.D2, R=rig.R2, P=rig.P2).reshape(-1, 2)
        np.testing.assert_allclose(a[:, 1], b[:, 1], atol=1e-7)
        self.assertTrue(np.all(a[:, 0] > b[:, 0]))
        homogeneous = np.column_stack((a, a[:, 0]-b[:, 0], np.ones(len(a)))) @ rig.Q.T
        reconstructed_mm = homogeneous[:, :3] / homogeneous[:, 3, None]
        np.testing.assert_allclose(reconstructed_mm, xyz @ rig.R1.T, atol=1e-6)

    def test_schema_round_trip_and_corrupt_q_rejected(self):
        rig = calibration()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.json"
            rig.save(path)
            loaded = StereoCalibration.load(path)
            self.assertEqual(loaded.image_size, rig.image_size)
            self.assertEqual((loaded.left_source, loaded.right_source), (rig.left_source, rig.right_source))
            np.testing.assert_allclose(loaded.K_rect, rig.K_rect)
            self.assertAlmostEqual(loaded.baseline_mm, 60.)
            values = json.loads(path.read_text())
            values["Q"][3][2] *= 2
            path.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "Q is inconsistent"):
                StereoCalibration.load(path)

    def test_vertical_order_small_baseline_and_projection_corruption_rejected(self):
        with self.assertRaisesRegex(ValueError, "Vertical"):
            calibration(translation=[0, -60, 0])
        with self.assertRaisesRegex(ValueError, "Camera order"):
            calibration(translation=[60, 0, 0])
        with self.assertRaisesRegex(ValueError, "baseline"):
            calibration(baseline=5)
        rig = calibration()
        corrupt = rig.P2.copy()
        corrupt[0, 3] *= 1.1
        with self.assertRaisesRegex(ValueError, "P2 translation"):
            replace(rig, P2=corrupt)
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(rig, Q=np.full((4, 4), np.nan))
        with self.assertRaisesRegex(ValueError, "metadata baseline"):
            replace(rig, metadata={"baseline_mm": 100})
        with self.assertRaisesRegex(ValueError, "quality limit"):
            replace(rig, metadata={"stereo_rms_px": 3})

    def test_resolution_or_channel_changes_are_not_silently_adapted(self):
        rig = calibration()
        with self.assertRaisesRegex(ValueError, "resolution"):
            rig.rectify(np.zeros((96, 192, 3), np.uint8), np.zeros((96, 192, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "channel"):
            rig.rectify(np.zeros((192, 384, 3), np.uint8), np.zeros((192, 384), np.uint8))

    def test_real_calibrator_recovers_independently_projected_rig(self):
        objects, left, right, size, true_rotation, true_translation = projected_boards()
        fitted = calibrate_stereo(objects, left, right, size, "left-device", "right-device",
                                  metadata={"square_mm": 15})
        self.assertLess(abs(fitted.baseline_mm - np.linalg.norm(true_translation)), .5)
        self.assertLess(np.linalg.norm(cv2.Rodrigues(fitted.R @ true_rotation.T)[0]), .005)
        self.assertLess(fitted.metadata["stereo_rms_px"], .2)
        self.assertLess(fitted.metadata["epipolar_p95_px"], .3)
        self.assertEqual(fitted.metadata["paired_views"], 18)
        self.assertFalse(fitted.metadata["timing_synchronized"])

    def test_wrong_board_pairing_and_too_few_views_rejected(self):
        objects, left, right, size, _, _ = projected_boards()
        with self.assertRaisesRegex(ValueError, "matched board views"):
            calibrate_stereo(objects[:5], left[:5], right[:5], size, "L", "R")
        with self.assertRaisesRegex(ValueError, "RMS"):
            calibrate_stereo(objects, left, right[1:]+right[:1], size, "L", "R")


class StereoMatchingTests(unittest.TestCase):
    def test_actual_sgbm_recovers_textured_plane_in_metres(self):
        rig = calibration()
        left, right = textured_pair()
        result = StereoDepth(rig, num_disparities=64).compute(left, right)
        region = result.depth_m[20:-20, 100:300]
        self.assertGreater(np.isfinite(region).mean(), .9)
        expected_metres = 220 * 60 / 20 / 1000
        self.assertAlmostEqual(float(np.nanmedian(region)), expected_metres, delta=.003)
        self.assertAlmostEqual(float(np.nanmedian(result.disparity_px)), 20., delta=.1)
        self.assertTrue(result.metric)
        self.assertEqual(result.depth_m.dtype, np.float32)
        self.assertTrue(np.isnan(result.depth_m[~result.valid_mask]).all())
        self.assertFalse(result.valid_mask[:, :64].any())

    def test_occluded_texture_rejected_by_reverse_matching(self):
        rig = calibration()
        left, right = textured_pair()
        # Left pixels x=160..205 have no corresponding right observations.
        right[25:165, 140:185] = 0
        result = StereoDepth(rig, num_disparities=64).compute(left, right)
        self.assertLess(result.valid_mask[40:150, 168:195].mean(), .15)
        self.assertGreater(result.valid_mask[40:150, 100:140].mean(), .85)

    def test_textureless_and_depth_outside_working_range_rejected(self):
        rig = calibration()
        empty = np.zeros((192, 384), np.uint8)
        self.assertFalse(StereoDepth(rig, num_disparities=64).compute(empty, empty).valid_mask.any())
        left, right = textured_pair()
        result = StereoDepth(rig, num_disparities=64, min_depth_mm=800, max_depth_mm=1200).compute(left, right)
        self.assertFalse(result.valid_mask.any())

    def test_declared_rectification_roi_is_enforced(self):
        rig = replace(calibration(), roi1=(90, 20, 210, 130), roi2=(30, 20, 270, 130))
        result = StereoDepth(rig, num_disparities=64).compute(*textured_pair())
        self.assertFalse(result.valid_mask[:20].any())
        self.assertFalse(result.valid_mask[150:].any())
        self.assertFalse(result.valid_mask[:, 300:].any())
        self.assertGreater(result.valid_mask[35:130, 110:250].mean(), .9)

    def test_alignment_detects_vertical_shift_without_claiming_blank_certainty(self):
        left, right = textured_pair()
        good = check_epipolar_alignment(left, right)
        self.assertTrue(good["aligned"], good)
        shifted = np.zeros_like(right)
        shifted[9:] = right[:-9]
        bad = check_epipolar_alignment(left, shifted)
        self.assertFalse(bad["aligned"], bad)
        self.assertGreater(bad["median_vertical_error_px"], 7)
        blank = check_epipolar_alignment(np.zeros_like(left), np.zeros_like(right))
        self.assertIsNone(blank["aligned"])


if __name__ == "__main__":
    unittest.main()
