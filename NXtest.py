# -*- coding: utf-8 -*-
"""
TensorRT 加速版 作物/杂草 检测系统
基于 YOLOv8 TensorRT + 卡尔曼滤波跟踪

★★★ 从 PC 版移植的全部高级功能 ★★★
  - 增强型卡尔曼跟踪器（轨迹拟合垂直速度 vy + 线前基准点预测 + 状态继承）
  - 多目标关联跟踪器（IOU + 中心距离混合关联 + 轨迹恢复）
  - 预测撞线检测（目标消失后仍可通过垂直速度外推预测触发）
  - 触发高亮动画
  - 多类别可视化（crop绿色 / weeds红色）
  - 接近警告
  - 串口通信
  - 丰富的视频叠加信息与诊断建议
  - 完整命令行参数
  - 可配置水平线段段数

速度拟合: 只拟合垂直方向 vy，预测时 X 坐标锁定消失前位置
撞线检测: X 轴方向的重叠判断完整保留

TensorRT 推理管线（YoLov8TRT / CUDA）保持不变。
"""

import ctypes
import os
import shutil
import random
import sys
import math
import threading
import time
import traceback
import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import serial
import argparse
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment

# ============================================================================
# 全局参数
# ============================================================================
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4
POSE_NUM = 17 * 3
DET_NUM = 6
SEG_NUM = 32
OBB_NUM = 1

# 检测框过滤参数
MIN_BOX_AREA = 500
MAX_BOX_AREA = 200000
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0

# 中心线 / 撞线 / 调试
CENTER_LINE_CLASS_KEYWORDS = ['crop', 'corn']
COLLISION_Y_TOLERANCE = 5
APPROACH_WARN_DISTANCE = 50
MIN_TRAJECTORY_POINTS = 3

DEBUG_ASSOC = False
DEBUG_TRACK = False
DEBUG_PREDICT = True


# ============================================================================
# 工具函数
# ============================================================================
def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox[:4]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox[:4]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def center_distance(b1, b2):
    c1 = bbox_center(b1)
    c2 = bbox_center(b2)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def compute_center_line_from_bbox(bbox, image_width,
                                  center_line_ratio=0.5, extend_ratio=0.0):
    """从检测框计算水平中心线段"""
    x1, y1, x2, y2 = map(int, bbox[:4])
    cy = (y1 + y2) // 2
    bw = x2 - x1
    length = int(bw * center_line_ratio)
    sx = x1 + (bw - length) // 2
    ex = sx + length
    if extend_ratio > 0:
        el = int(bw * extend_ratio)
        sx = max(0, sx - el)
        ex = min(image_width - 1, ex + el)
    return (sx, cy, ex, cy)


# ============================================================================
# 颜色配置
# ============================================================================
CLASS_COLORS = {
    'weeds': (0, 0, 255),
    'crop': (0, 255, 0),
    'corn': (0, 255, 0),
    'default': (255, 0, 255),
}


def get_class_color(class_name):
    name = class_name.lower()
    if 'weed' in name:
        return CLASS_COLORS['weeds']
    elif 'crop' in name or 'corn' in name:
        return CLASS_COLORS['crop']
    return CLASS_COLORS.get(name, CLASS_COLORS['default'])


def should_draw_center_line(class_name):
    name = class_name.lower()
    for keyword in CENTER_LINE_CLASS_KEYWORDS:
        if keyword in name:
            return True
    return False


# ============================================================================
# 串口通信
# ============================================================================
def init_serial(port="/dev/ttyTHS0", baudrate=115200):
    if port is None:
        print("[串口] 未指定串口，跳过串口初始化")
        return None
    try:
        com = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
        )
        print(f"[串口] 连接成功: {port} @ {baudrate}")
        return com
    except Exception as e:
        print(f"[串口] 连接失败: {e}（将以无串口模式运行）")
        return None


# ============================================================================
# 增强型卡尔曼滤波跟踪器（只拟合 vy）
# ============================================================================
class KalmanBoxTracker:
    """
    增强型卡尔曼滤波跟踪器
    状态: [x, y, w, h, vx, vy, vw, vh]
    速度拟合: 只拟合垂直方向 vy
    预测时: X 坐标锁定消失前位置，只有 Y 外推
    """
    count = 0
    MAX_VELOCITY = 100

    def __init__(self, bbox, class_id, conf,
                 image_width=None, cl_ratio=0.5, cl_extend=0.0):
        self.kf = cv2.KalmanFilter(8, 4)

        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0

        x, y, w, h = self._convert_bbox_to_xywh(bbox)
        self.kf.statePost = np.array(
            [x, y, w, h, 0, 0, 0, 0], dtype=np.float32
        ).reshape(8, 1)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history = deque(maxlen=10)
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

        self.class_id = class_id
        self.conf = conf
        self.conf_history = deque([conf], maxlen=10)

        # 中心线跟踪
        self.prev_center_line = None
        self.center_line_velocity = None  # 存储格式: vy (标量)

        if image_width is not None and image_width > 0:
            self.last_center_line = compute_center_line_from_bbox(
                bbox, image_width, cl_ratio, cl_extend
            )
        else:
            self.last_center_line = None

        self._saved_velocity = None
        self._saved_last_center_line = None

        # 轨迹拟合（只记录 Y）
        self._trajectory = deque(maxlen=30)
        self._frame_counter = 0

        # 线前基准点
        self._before_line_base = {}
        self._segment_ys = {}

        # 恢复标记
        self.just_recovered = False
        self.recovered_from_id = None

    def _convert_bbox_to_xywh(self, bbox):
        x1, y1, x2, y2 = bbox[:4]
        return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

    def _convert_xywh_to_bbox(self, x, y, w, h):
        return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

    def update(self, bbox, conf):
        if self.time_since_update > 1:
            if self.center_line_velocity is not None:
                self._saved_velocity = self.center_line_velocity
                self._saved_last_center_line = self.last_center_line
            self.prev_center_line = None
            self.last_center_line = None

        self.time_since_update = 0
        self.history.clear()
        self.hits += 1
        self.hit_streak += 1
        self.conf_history.append(conf)
        self.conf = float(np.mean(self.conf_history))

        x, y, w, h = self._convert_bbox_to_xywh(bbox)
        measurement = np.array([x, y, w, h], dtype=np.float32).reshape(4, 1)
        self.kf.correct(measurement)

    def predict(self):
        self.kf.predict()
        self.age += 1
        self._frame_counter += 1

        if self.time_since_update > 1:
            self.hit_streak = 0

        self.time_since_update += 1

        state = self.kf.statePost
        bbox = self._convert_xywh_to_bbox(
            state[0][0], state[1][0], state[2][0], state[3][0]
        )
        self.history.append(bbox)
        return bbox

    def get_state(self):
        state = self.kf.statePost
        return self._convert_xywh_to_bbox(
            state[0][0], state[1][0], state[2][0], state[3][0]
        )

    def update_center_line(self, center_line, segment_ys=None):
        """更新中心线并拟合垂直速度"""
        self.prev_center_line = self.last_center_line
        self.last_center_line = center_line

        self._trajectory.append((self._frame_counter, center_line))
        self._fit_velocity()

        if segment_ys is not None:
            self._segment_ys = segment_ys
            cy = center_line[1]
            vy = self.center_line_velocity  # 标量或 None

            for sid, seg_y in segment_ys.items():
                moving_down = (vy is not None and vy > 0)
                moving_up = (vy is not None and vy < 0)

                if moving_down:
                    if cy < seg_y:
                        self._before_line_base[sid] = (
                            self._frame_counter, center_line
                        )
                elif moving_up:
                    if cy > seg_y:
                        self._before_line_base[sid] = (
                            self._frame_counter, center_line
                        )
                else:
                    self._before_line_base[sid] = (
                        self._frame_counter, center_line
                    )

    def _fit_velocity(self):
        """只拟合垂直方向速度 vy（线性回归）"""
        if len(self._trajectory) < MIN_TRAJECTORY_POINTS:
            return

        frames = np.array([t[0] for t in self._trajectory], dtype=np.float64)
        cys = np.array([t[1][1] for t in self._trajectory], dtype=np.float64)

        n = len(frames)
        sum_f = frames.sum()
        sum_f2 = (frames * frames).sum()
        denom = n * sum_f2 - sum_f * sum_f

        if abs(denom) < 1e-6:
            return

        vy = (n * (frames * cys).sum() - sum_f * cys.sum()) / denom

        if abs(vy) > self.MAX_VELOCITY:
            return

        self.center_line_velocity = vy  # ★ 标量
        self._saved_velocity = vy
        self._saved_last_center_line = self.last_center_line

    def get_effective_velocity(self):
        """返回有效的 vy（标量或 None）"""
        if self.center_line_velocity is not None:
            return self.center_line_velocity
        return self._saved_velocity

    def get_effective_last_center_line(self):
        if self.last_center_line is not None:
            return self.last_center_line
        return self._saved_last_center_line

    def predict_center_line_for_segment(self, seg_id):
        """
        针对特定线段的中心线预测
        X 坐标锁定基准点位置不变，只有 Y 根据 vy 外推
        返回: (pred_cl, base_cl, dt) 或 None
        """
        vy = self.get_effective_velocity()
        if vy is None:
            return None

        base_info = self._before_line_base.get(seg_id)
        if base_info is not None:
            base_frame, base_cl = base_info
            dt = self._frame_counter - base_frame
        else:
            base_cl = self.get_effective_last_center_line()
            if base_cl is None:
                return None
            dt = self.time_since_update

        if dt <= 0:
            return None

        # ★ X 不动，只有 Y 外推
        pred_sx = base_cl[0]
        pred_cy = int(base_cl[1] + vy * dt)
        pred_ex = base_cl[2]
        return (pred_sx, pred_cy, pred_ex, pred_cy), base_cl, dt

    def predict_center_line(self):
        """通用中心线预测（X 锁定，Y 外推）"""
        vy = self.get_effective_velocity()
        base_cl = self.get_effective_last_center_line()
        if base_cl is None or vy is None:
            return None
        dt = self.time_since_update
        if dt <= 0:
            return None

        pred_sx = base_cl[0]
        pred_cy = int(base_cl[1] + vy * dt)
        pred_ex = base_cl[2]
        return (pred_sx, pred_cy, pred_ex, pred_cy)

    def inherit_state_from(self, old_tracker):
        """从旧轨迹继承历史状态"""
        self._trajectory = deque(old_tracker._trajectory, maxlen=30)
        self._saved_velocity = old_tracker.get_effective_velocity()
        self._saved_last_center_line = old_tracker.get_effective_last_center_line()
        self.center_line_velocity = old_tracker.get_effective_velocity()
        self._before_line_base = dict(old_tracker._before_line_base)
        self._segment_ys = dict(old_tracker._segment_ys)
        self.just_recovered = True
        self.recovered_from_id = old_tracker.id

    def get_trajectory_info(self):
        n = len(self._trajectory)
        if n < 2:
            return f"pts={n}"

        first = self._trajectory[0][1]
        last = self._trajectory[-1][1]
        total_frames = self._trajectory[-1][0] - self._trajectory[0][0]

        if total_frames > 0:
            overall_vy = (last[1] - first[1]) / total_frames
        else:
            overall_vy = 0

        vy = self.center_line_velocity
        fit_vy = vy if vy is not None else 0

        base_info = []
        for sid, (bf, bcl) in self._before_line_base.items():
            base_info.append(f"L{sid + 1}:base_y={bcl[1]}@f{bf}")
        base_str = " ".join(base_info) if base_info else "无基准"

        return (f"pts={n} "
                f"y:{first[1]}->{last[1]} "
                f"fit_vy={fit_vy:.1f} "
                f"| {base_str}")


# ============================================================================
# 多目标关联跟踪器
# ============================================================================
class MultiObjectTracker:
    """IOU + 中心距离混合关联，类别感知，轨迹恢复继承"""

    def __init__(self,
                 max_age=50,
                 min_hits=1,
                 iou_threshold=0.2,
                 predict_frames=180,
                 cl_ratio=0.5,
                 cl_extend=0.0,
                 dist_threshold=120,
                 recover_max_age=45):
        self.max_age = max(max_age, predict_frames)
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.predict_frames = predict_frames
        self.cl_ratio = cl_ratio
        self.cl_extend = cl_extend
        self.dist_threshold = dist_threshold
        self.recover_max_age = max(recover_max_age, predict_frames)

        self.trackers = []
        self.frame_count = 0

    def reset(self):
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def _iou(self, bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = max(0, x1_max - x1_min) * max(0, y1_max - y1_min)
        area2 = max(0, x2_max - x2_min) * max(0, y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _associate(self, detections):
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(self.trackers)))

        num_dets = len(detections)
        num_trks = len(self.trackers)

        cost_matrix = np.full((num_dets, num_trks), 1e6, dtype=np.float32)
        iou_matrix = np.zeros((num_dets, num_trks), dtype=np.float32)
        dist_matrix = np.full((num_dets, num_trks), 1e6, dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                if int(det[5]) != int(trk.class_id):
                    continue

                trk_bbox = trk.get_state()
                iou = self._iou(det[:4], trk_bbox)
                dist = center_distance(det[:4], trk_bbox)

                iou_matrix[d, t] = iou
                dist_matrix[d, t] = dist

                if iou >= self.iou_threshold:
                    cost = (1.0 - iou) + 0.002 * dist
                elif dist <= self.dist_threshold:
                    cost = 0.8 + 0.004 * dist
                else:
                    cost = 1e6

                cost_matrix[d, t] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = list(range(num_dets))
        unmatched_trks = list(range(num_trks))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= 1e5:
                continue

            iou = iou_matrix[r, c]
            dist = dist_matrix[r, c]

            if iou >= self.iou_threshold or dist <= self.dist_threshold:
                matched.append([r, c])
                if r in unmatched_dets:
                    unmatched_dets.remove(r)
                if c in unmatched_trks:
                    unmatched_trks.remove(c)

                if DEBUG_ASSOC:
                    print(f"[ASSOC] det={r} -> trk={self.trackers[c].id} "
                          f"iou={iou:.3f} dist={dist:.1f}")

        return matched, unmatched_dets, unmatched_trks

    def _try_recover_old_tracker(self, det, image_width):
        best_old = None
        best_score = 1e9

        for trk in self.trackers:
            if int(trk.class_id) != int(det[5]):
                continue
            if trk.time_since_update <= 0:
                continue
            if trk.time_since_update > self.recover_max_age:
                continue

            dist = center_distance(det[:4], trk.get_state())
            if dist > self.dist_threshold:
                continue

            score = dist + trk.time_since_update * 2.0
            if score < best_score:
                best_score = score
                best_old = trk

        new_trk = KalmanBoxTracker(
            det[:4], det[5], det[4],
            image_width=image_width,
            cl_ratio=self.cl_ratio,
            cl_extend=self.cl_extend,
        )

        if best_old is not None and best_old.get_effective_velocity() is not None:
            new_trk.inherit_state_from(best_old)
            if DEBUG_TRACK:
                print(f"[RECOVER] new_id={new_trk.id} inherit from "
                      f"old_id={best_old.id} dist={best_score:.1f}")

        return new_trk

    def update(self, detections, image_width=0):
        self.frame_count += 1

        for tracker in self.trackers:
            tracker.predict()

        matched, unmatched_dets, unmatched_trks = self._associate(detections)

        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(
                detections[det_idx][:4], detections[det_idx][4]
            )
            self.trackers[trk_idx].class_id = detections[det_idx][5]

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            trk = self._try_recover_old_tracker(det, image_width)
            self.trackers.append(trk)

        visible_results = []
        predicted_trackers = []
        to_remove = []

        for i, tracker in enumerate(self.trackers):
            if (tracker.time_since_update == 0
                    and tracker.hit_streak >= self.min_hits):
                bbox = tracker.get_state()
                visible_results.append(
                    bbox + [tracker.conf, tracker.class_id, tracker.id]
                )
            elif (0 < tracker.time_since_update <= self.predict_frames
                  and tracker.get_effective_velocity() is not None):
                predicted_trackers.append(tracker)

            if tracker.time_since_update > self.max_age:
                to_remove.append(i)

            if DEBUG_TRACK:
                print(f"[TRACK] id={tracker.id} cls={tracker.class_id} "
                      f"tsu={tracker.time_since_update} "
                      f"hits={tracker.hits} streak={tracker.hit_streak} "
                      f"vy={'Y' if tracker.get_effective_velocity() is not None else 'N'}")

        for i in reversed(to_remove):
            if DEBUG_TRACK:
                print(f"[REMOVE] tracker id={self.trackers[i].id} "
                      f"tsu={self.trackers[i].time_since_update}")
            self.trackers.pop(i)

        return visible_results, predicted_trackers

    def get_tracker_by_id(self, track_id):
        for t in self.trackers:
            if t.id == track_id:
                return t
        return None


# ============================================================================
# 虚拟线段配置（增加段数配置）
# ============================================================================
class LineConfig:
    def __init__(self):
        self.layout_mode = 'horizontal'
        self.num_segments = 4  # 默认4段，可配置1-8段

        self.horizontal_config = {
            'y_position_ratio': 0.5,
            'segment_width_ratio': 0.175,
            'gap_ratio': 0.15,
            'start_offset_ratio': 0.0,
        }

        self.vertical_config = {
            'x_positions_ratio': [0.2, 0.4, 0.6, 0.8],
            'start_y_ratio': 0.2,
            'end_y_ratio': 0.8,
        }

        self.custom_segments = [
            (0.1, 0.3, 0.3, 0.3),
            (0.4, 0.4, 0.6, 0.4),
            (0.7, 0.5, 0.9, 0.5),
            (0.2, 0.7, 0.8, 0.7),
        ]

        # 预定义颜色数组（最多8种颜色）
        self.colors = [
            (0, 0, 255),  # 红
            (255, 0, 255),  # 紫
            (255, 0, 0),  # 蓝
            (0, 255, 255),  # 青
            (0, 255, 0),  # 绿
            (255, 255, 0),  # 黄
            (128, 0, 128),  # 紫罗兰
            (255, 165, 0),  # 橙
        ]
        self.thickness = 3
        # 预定义串口命令（最多8个）
        self.commands = [b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8']

        self.center_line_config = {
            'enabled': True,
            'color': (0, 255, 255),
            'thickness': 2,
            'line_type': cv2.LINE_AA,
            'extend_ratio': 0.0,
            'center_line_ratio': 0.5,
        }


# ============================================================================
# YoLov8TRT 类（完全保持不变）
# ============================================================================
class YoLov8TRT(object):
    """YOLOv8 TensorRT wrapper - 推理管线原封不动"""

    def __init__(self, engine_file_path):
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
        self.det_output_length = host_outputs[0].shape[0]

    def infer(self, raw_image_generator):
        threading.Thread.__init__(self)
        self.ctx.push()
        stream = self.stream
        context = self.context
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(
            shape=[self.batch_size, 3, self.input_h, self.input_w])

        for i, image_raw in enumerate(raw_image_generator):
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(batch_size=self.batch_size,
                              bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        end = time.time()
        self.ctx.pop()

        output = host_outputs[0]
        result_boxes_list = []
        result_scores_list = []
        result_classid_list = []

        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.post_process(
                output[i * self.det_output_length:(i + 1) * self.det_output_length],
                batch_origin_h[i], batch_origin_w[i]
            )
            result_boxes_list.append(result_boxes)
            result_scores_list.append(result_scores)
            result_classid_list.append(result_classid)

        return batch_image_raw, end - start, result_boxes_list, result_scores_list, result_classid_list

    def is_corn_class(self, class_id):
        corn_class_ids = [0]
        return class_id in corn_class_ids

    def destroy(self):
        self.ctx.pop()

    def get_raw_image(self, image_path_batch):
        for img_path in image_path_batch:
            yield cv2.imread(img_path)

    def get_raw_image_zeros(self, image_path_batch=None):
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
        )
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0]
            y[:, 2] = x[:, 2]
            y[:, 1] = x[:, 1] - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 3] - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 2] - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1]
            y[:, 3] = x[:, 3]
            y /= r_h
        return y

    def post_process(self, output, origin_h, origin_w):
        num_values_per_detection = DET_NUM + SEG_NUM + POSE_NUM + OBB_NUM
        num = int(output[0])
        pred = np.reshape(output[1:], (-1, num_values_per_detection))[:num, :]
        boxes = self.non_max_suppression(
            pred, origin_h, origin_w,
            conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD
        )

        if len(boxes) > 0:
            filtered_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                w = x2 - x1
                h = y2 - y1
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                if (MIN_BOX_AREA <= area <= MAX_BOX_AREA and
                        MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
                    filtered_boxes.append(box)
            boxes = np.array(filtered_boxes) if filtered_boxes else np.array([])

        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        inter_area = (np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None)
                      * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None))
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w,
                            conf_thres=0.5, nms_thres=0.4):
        boxes = prediction[prediction[:, 4] >= conf_thres]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        confs = boxes[:, 4]
        boxes = boxes[np.argsort(-confs)]
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(
                np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color1 = color or (0, 255, 0)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color1, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2_label = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2_label, color1, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# ============================================================================
# 增强型基础检测处理器（vy-only 预测 + 完整 X 轴撞线检测）
# ============================================================================
class BaseDetectionProcessor:
    TRIGGER_HIGHLIGHT_FRAMES = 5

    def __init__(self, yolov8_wrapper, config=None, tracker_cfg=None,
                 serial_conn=None, categories=None):
        self.yolov8_wrapper = yolov8_wrapper
        self.config = config if config is not None else LineConfig()
        self.serial_conn = serial_conn
        self.categories = categories or ["corn"]
        self.triggered_records = defaultdict(lambda: defaultdict(bool))

        tk_cfg = tracker_cfg or {}
        cl_cfg = self.config.center_line_config
        self.tracker = MultiObjectTracker(
            max_age=tk_cfg.get('max_age', 60),
            min_hits=tk_cfg.get('min_hits', 1),
            iou_threshold=tk_cfg.get('iou_threshold', 0.2),
            predict_frames=tk_cfg.get('predict_frames', 60),
            cl_ratio=cl_cfg.get('center_line_ratio', 0.5),
            cl_extend=cl_cfg.get('extend_ratio', 0.0),
            dist_threshold=tk_cfg.get('dist_threshold', 120),
            recover_max_age=tk_cfg.get('recover_max_age', 60),
        )

        self.segment_highlight = {}
        self.all_visible_y = []

    def _get_class_name(self, class_id):
        cid = int(class_id)
        if 0 <= cid < len(self.categories):
            return self.categories[cid]
        return str(cid)

    def _get_segment_ys(self, segments):
        seg_ys = {}
        for seg in segments:
            ss, se = seg['start'], seg['end']
            if ss[1] == se[1]:
                seg_ys[seg['id']] = ss[1]
            else:
                seg_ys[seg['id']] = (ss[1] + se[1]) // 2
        return seg_ys

    def get_segment_lines(self, width, height):
        segments = []
        cfg = self.config
        num_segments = min(cfg.num_segments, 8)  # 限制最大8段

        if cfg.layout_mode == 'horizontal':
            hc = cfg.horizontal_config
            center_y = int(height * hc['y_position_ratio'])
            seg_w = int(width * hc['segment_width_ratio'])
            gap = int(width * hc['gap_ratio'])
            offset = int(width * hc['start_offset_ratio'])

            # 计算总宽度，如果需要的话调整gap
            total_seg_width = num_segments * seg_w + (num_segments - 1) * gap
            if total_seg_width > width:
                # 如果总宽度超过画面，按比例缩小gap
                gap = max(5, int((width - num_segments * seg_w) / (num_segments - 1)))

            for i in range(num_segments):
                sx = offset + i * (seg_w + gap)
                ex = min(sx + seg_w, width)
                if sx < width:
                    segments.append({
                        'start': (sx, center_y), 'end': (ex, center_y),
                        'color': cfg.colors[i % len(cfg.colors)], 'id': i,
                    })

        elif cfg.layout_mode == 'vertical':
            vc = cfg.vertical_config
            # 确保x_positions有足够的元素
            x_positions = vc['x_positions_ratio'][:num_segments]
            # 如果不够，线性插值生成更多位置
            if len(x_positions) < num_segments:
                x_positions = np.linspace(0.2, 0.8, num_segments).tolist()

            for i, xr in enumerate(x_positions):
                xp = int(width * xr)
                sy = int(height * vc['start_y_ratio'])
                ey = int(height * vc['end_y_ratio'])
                segments.append({
                    'start': (xp, sy), 'end': (xp, ey),
                    'color': cfg.colors[i % len(cfg.colors)], 'id': i,
                })

        elif cfg.layout_mode == 'custom':
            # 对于custom模式，只使用配置好的线段数
            for i, (sx, sy, ex, ey) in enumerate(cfg.custom_segments[:num_segments]):
                segments.append({
                    'start': (int(width * sx), int(height * sy)),
                    'end': (int(width * ex), int(height * ey)),
                    'color': cfg.colors[i % len(cfg.colors)], 'id': i,
                })

        return segments

    def draw_segment_lines(self, frame, segments):
        for seg in segments:
            sid = seg['id']
            highlight = self.segment_highlight.get(sid)

            if highlight and highlight['remaining'] > 0:
                remaining = highlight['remaining']
                tid = highlight['track_id']
                trigger_type = highlight['type']

                if remaining % 2 == 1:
                    line_color = (255, 255, 255)
                else:
                    line_color = (0, 255, 255)

                cv2.line(frame, seg['start'], seg['end'],
                         line_color, self.config.thickness + 4, cv2.LINE_AA)
                cv2.line(frame, seg['start'], seg['end'],
                         seg['color'], self.config.thickness, cv2.LINE_AA)

                mx = (seg['start'][0] + seg['end'][0]) // 2
                my = (seg['start'][1] + seg['end'][1]) // 2

                cv2.putText(frame, f"Line {sid + 1}", (mx, my - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, seg['color'], 2)

                if trigger_type == 'predict':
                    alert_text = f"PREDICT TRIGGER! ID:{tid}"
                    alert_color = (0, 0, 255)
                else:
                    alert_text = f"TRIGGER! ID:{tid}"
                    alert_color = (0, 200, 0)

                (tw, th), _ = cv2.getTextSize(
                    alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_x = mx - tw // 2
                text_y = my - 15
                cv2.rectangle(frame,
                              (text_x - 4, text_y - th - 4),
                              (text_x + tw + 4, text_y + 4),
                              (0, 0, 0), -1)
                cv2.putText(frame, alert_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            alert_color, 2, cv2.LINE_AA)

                highlight['remaining'] -= 1
            else:
                cv2.line(frame, seg['start'], seg['end'],
                         seg['color'], self.config.thickness, cv2.LINE_AA)
                lx = (seg['start'][0] + seg['end'][0]) // 2
                ly = (seg['start'][1] + seg['end'][1]) // 2 - 20
                cv2.putText(frame, f"Line {seg['id'] + 1}", (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, seg['color'], 2)

    def _activate_segment_highlight(self, seg_id, track_id, trigger_type='normal'):
        self.segment_highlight[seg_id] = {
            'remaining': self.TRIGGER_HIGHLIGHT_FRAMES,
            'track_id': track_id,
            'type': trigger_type,
        }

    def _compute_center_line(self, box, image_width):
        cl_cfg = self.config.center_line_config
        return compute_center_line_from_bbox(
            box, image_width,
            cl_cfg.get('center_line_ratio', 0.5),
            cl_cfg.get('extend_ratio', 0.0),
        )

    def _get_min_distance_to_segments(self, center_line, segments):
        _, cy, _, _ = center_line
        min_dist = float('inf')
        nearest_sid = -1

        for seg in segments:
            ss, se = seg['start'], seg['end']
            if ss[1] == se[1]:
                dist = abs(cy - ss[1])
            elif ss[0] == se[0]:
                seg_y1 = min(ss[1], se[1])
                seg_y2 = max(ss[1], se[1])
                if seg_y1 <= cy <= seg_y2:
                    dist = 0
                else:
                    dist = min(abs(cy - seg_y1), abs(cy - seg_y2))
            else:
                seg_y_mid = (ss[1] + se[1]) // 2
                dist = abs(cy - seg_y_mid)

            if dist < min_dist:
                min_dist = dist
                nearest_sid = seg['id']

        return min_dist, nearest_sid

    def _is_prediction_still_useful_for_segment(self, tracker, pred_cl, seg,
                                                extra_margin=10):
        """
        只基于 vy 判断预测线是否已越过撞线区域
        vy > 0 向下移动，vy < 0 向上移动
        """
        vy = tracker.get_effective_velocity()
        if vy is None:
            return True

        ss, se = seg['start'], seg['end']
        _, cy, _, _ = pred_cl
        tol = COLLISION_Y_TOLERANCE + extra_margin

        # 取线段的 Y 中点作为参考
        seg_mid_y = (ss[1] + se[1]) / 2.0

        if vy > 0 and cy > seg_mid_y + tol:
            return False  # 向下移动，已越过
        if vy < 0 and cy < seg_mid_y - tol:
            return False  # 向上移动，已越过

        return True

    def _check_collision(self, center_line, segments, track_id):
        """
        撞线检测（含去重）
        中心线: (csx, cy, cex, cy)  — X 范围 + Y 位置
        水平线段: 检查 Y 容差 + X 重叠
        垂直线段: 检查 X 包含 + Y 范围
        斜线段:   检查 X 重叠 + Y 范围
        """
        triggered = []
        csx, cy, cex, _ = center_line
        tol = COLLISION_Y_TOLERANCE

        for seg in segments:
            sid = seg['id']
            ss, se = seg['start'], seg['end']

            # ---- 水平线段 ----
            if ss[1] == se[1]:
                seg_y = ss[1]
                seg_x1 = min(ss[0], se[0])
                seg_x2 = max(ss[0], se[0])
                if abs(cy - seg_y) <= tol:
                    if not (cex < seg_x1 or csx > seg_x2):
                        if not self.triggered_records[track_id][sid]:
                            triggered.append(sid)
                            self.triggered_records[track_id][sid] = True
                    else:
                        if self.triggered_records[track_id][sid]:
                            if cex < seg_x1 - 20 or csx > seg_x2 + 20:
                                self.triggered_records[track_id][sid] = False

            # ---- 垂直线段 ----
            elif ss[0] == se[0]:
                seg_x = ss[0]
                seg_y1 = min(ss[1], se[1])
                seg_y2 = max(ss[1], se[1])
                if (csx <= seg_x <= cex) and (seg_y1 <= cy <= seg_y2):
                    if not self.triggered_records[track_id][sid]:
                        triggered.append(sid)
                        self.triggered_records[track_id][sid] = True
                else:
                    if self.triggered_records[track_id][sid]:
                        if cex < seg_x - 20 or csx > seg_x + 20:
                            self.triggered_records[track_id][sid] = False

            # ---- 斜线段 ----
            else:
                seg_x1 = min(ss[0], se[0])
                seg_x2 = max(ss[0], se[0])
                seg_y1 = min(ss[1], se[1])
                seg_y2 = max(ss[1], se[1])
                if (not (cex < seg_x1 or csx > seg_x2)
                        and seg_y1 <= cy <= seg_y2):
                    if not self.triggered_records[track_id][sid]:
                        triggered.append(sid)
                        self.triggered_records[track_id][sid] = True

        return triggered

    def _send_serial(self, seg_id):
        if self.serial_conn is not None:
            try:
                self.serial_conn.write(self.config.commands[seg_id])
                print(f"[触发] 线段 {seg_id + 1}，发送串口命令: {seg_id + 1}")
            except Exception as e:
                print(f"[串口] 发送失败: {e}")
        else:
            print(f"[触发] 线段 {seg_id + 1}（串口未连接）")

    def process_frame(self, frame, send_serial=True, conf_thresh=0.5):
        """处理单帧 — 完整流水线"""
        height, width = frame.shape[:2]
        segments = self.get_segment_lines(width, height)
        segment_ys = self._get_segment_ys(segments)

        # --- TensorRT 推理 ---
        t0 = time.time()
        batch_image_raw, inference_time, result_boxes_list, \
            result_scores_list, result_classid_list = \
            self.yolov8_wrapper.infer([frame])
        total_inference_time = time.time() - t0

        if not batch_image_raw:
            return frame, 0, 0, 0, {
                'normal_triggers': [], 'predicted_triggers': []}

        processed_frame = batch_image_raw[0]
        result_boxes = result_boxes_list[0]
        result_scores = result_scores_list[0]
        result_classid = result_classid_list[0]

        # --- 构建检测列表 ---
        detections = []
        if len(result_boxes) > 0:
            for i in range(len(result_boxes)):
                x1, y1, x2, y2 = result_boxes[i]
                conf = float(result_scores[i])
                cls_id = int(result_classid[i])
                detections.append([x1, y1, x2, y2, conf, cls_id])

        raw_count = len(detections)

        # --- 多目标跟踪 ---
        visible_objects, predicted_trackers = self.tracker.update(
            detections, image_width=width
        )
        tracked_count = len(visible_objects)

        all_triggered = []
        active_ids = set()
        trigger_info = {
            'normal_triggers': [],
            'predicted_triggers': [],
        }

        # ======== 处理真实可见目标 ========
        for obj in visible_objects:
            x1, y1, x2, y2, conf, cls_id, tid = obj
            tid = int(tid)
            cls_id = int(cls_id)
            active_ids.add(tid)

            class_name = self._get_class_name(cls_id)
            color = get_class_color(class_name)
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(processed_frame, (ix1, iy1), (ix2, iy2), color, 2)

            label = f'ID:{tid} {class_name} {conf:.2f}'
            tracker_obj = self.tracker.get_tracker_by_id(tid)
            if tracker_obj is not None and tracker_obj.just_recovered:
                label += f' <-R({tracker_obj.recovered_from_id})'

            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(processed_frame, (ix1, iy1 - lh - 10),
                          (ix1 + lw, iy1), color, -1)
            cv2.putText(processed_frame, label, (ix1, iy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if should_draw_center_line(class_name):
                cl_cfg = self.config.center_line_config
                if cl_cfg['enabled']:
                    cl = self._compute_center_line([x1, y1, x2, y2], width)
                    cv2.line(processed_frame,
                             (cl[0], cl[1]), (cl[2], cl[3]),
                             cl_cfg['color'], cl_cfg['thickness'],
                             cl_cfg['line_type'])

                    if tracker_obj is not None:
                        tracker_obj.update_center_line(cl, segment_ys)

                        vy = tracker_obj.center_line_velocity
                        if vy is not None:
                            traj_n = len(tracker_obj._trajectory)
                            vel_text = f"vy={vy:.1f} n={traj_n}"
                            cv2.putText(processed_frame, vel_text,
                                        (ix1, iy2 + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                        (0, 255, 255), 1, cv2.LINE_AA)

                            row_y = iy2 + 28
                            for sid, (bf, bcl) in tracker_obj._before_line_base.items():
                                base_text = f"L{sid + 1}:base_y={bcl[1]}"
                                cv2.putText(processed_frame, base_text,
                                            (ix1, row_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                            (0, 200, 200), 1, cv2.LINE_AA)
                                row_y += 12

                    dist, nearest = self._get_min_distance_to_segments(
                        cl, segments)
                    dist_text = f"d={int(dist)}px->L{nearest + 1}"
                    cv2.putText(processed_frame, dist_text,
                                (ix1, iy2 + 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (255, 200, 0), 1, cv2.LINE_AA)

                    self.all_visible_y.append(cl[1])

                    if dist <= APPROACH_WARN_DISTANCE:
                        vy_info = tracker_obj.center_line_velocity if tracker_obj else None
                        vy_str = f"fit_vy={vy_info:.1f}" if vy_info else "vy=?"
                        print(f"  ★[接近] ID:{tid} y={cl[1]} "
                              f"距L{nearest + 1}:{int(dist)}px {vy_str}")

                    triggered = self._check_collision(cl, segments, tid)
                    all_triggered.extend(triggered)

                    if triggered:
                        trigger_info['normal_triggers'].extend(
                            [{'track_id': tid, 'seg_id': s}
                             for s in triggered])
                        for sid in triggered:
                            self._activate_segment_highlight(
                                sid, tid, 'normal')

        # ======== 处理预测中心线（vy-only，X 锁定） ========
        for tracker in predicted_trackers:
            tid = tracker.id
            pred_cls_id = tracker.class_id
            tsu = tracker.time_since_update
            active_ids.add(tid)

            pred_class_name = self._get_class_name(pred_cls_id)
            if not should_draw_center_line(pred_class_name):
                continue

            best_pred_cl = None
            best_base_y = None
            best_dt = None
            has_useful_segment = False

            for seg in segments:
                sid = seg['id']
                result = tracker.predict_center_line_for_segment(sid)
                if result is None:
                    continue

                pred_cl, base_cl, dt = result

                if not self._is_prediction_still_useful_for_segment(
                        tracker, pred_cl, seg):
                    if DEBUG_PREDICT:
                        vy = tracker.get_effective_velocity()
                        vy_str = f"{vy:.2f}" if vy is not None else "N/A"
                        print(f"[预测终止] ID:{tid} L{sid + 1} "
                              f"pred_y={pred_cl[1]} base_y={base_cl[1]} "
                              f"dt={dt} vy={vy_str} 原因:已越过撞线")
                    continue

                has_useful_segment = True

                # ★ 撞线检测：X 范围仍然完整参与判断
                triggered = self._check_collision(pred_cl, [seg], tid)
                if triggered:
                    all_triggered.extend(triggered)
                    print(f"  └─[预测触发!] ID:{tid} 消失{tsu}帧 "
                          f"基准y={base_cl[1]} dt={dt} "
                          f"预测y={pred_cl[1]} "
                          f"X范围=[{pred_cl[0]},{pred_cl[2]}] "
                          f"触发L{sid + 1}")

                    trigger_info['predicted_triggers'].append({
                        'track_id': tid,
                        'seg_ids': triggered,
                        'lost_frames': tsu,
                        'base_y': base_cl[1],
                    })
                    for s in triggered:
                        self._activate_segment_highlight(s, tid, 'predict')

                dist, _ = self._get_min_distance_to_segments(pred_cl, [seg])
                if best_pred_cl is None:
                    best_pred_cl = pred_cl
                    best_base_y = base_cl[1]
                    best_dt = dt
                else:
                    best_dist, _ = self._get_min_distance_to_segments(
                        best_pred_cl, segments)
                    if dist < best_dist:
                        best_pred_cl = pred_cl
                        best_base_y = base_cl[1]
                        best_dt = dt

            # 绘制预测中心线
            if has_useful_segment and best_pred_cl is not None:
                cv2.line(processed_frame,
                         (best_pred_cl[0], best_pred_cl[1]),
                         (best_pred_cl[2], best_pred_cl[3]),
                         (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(processed_frame,
                           (best_pred_cl[0], best_pred_cl[1]),
                           5, (0, 0, 255), -1)
                cv2.circle(processed_frame,
                           (best_pred_cl[2], best_pred_cl[3]),
                           5, (0, 0, 255), -1)

                mid_x = (best_pred_cl[0] + best_pred_cl[2]) // 2
                dist, nearest = self._get_min_distance_to_segments(
                    best_pred_cl, segments)

                pred_label = (f"PRED ID:{tid} lost:{tsu} "
                              f"d={int(dist)}px base:{best_base_y}")
                (plw, plh), _ = cv2.getTextSize(
                    pred_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
                cv2.rectangle(processed_frame,
                              (mid_x - plw // 2 - 2,
                               best_pred_cl[1] - plh - 12),
                              (mid_x + plw // 2 + 2,
                               best_pred_cl[1] - 2),
                              (0, 0, 180), -1)
                cv2.putText(processed_frame, pred_label,
                            (mid_x - plw // 2, best_pred_cl[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 2, cv2.LINE_AA)

                if dist <= 100 and DEBUG_PREDICT:
                    vy_val = tracker.get_effective_velocity()
                    vy_str = f"fit_vy={vy_val:.1f}" if vy_val else "N/A"
                    traj_info = tracker.get_trajectory_info()
                    print(f"[预测→接近!] ID:{tid} | "
                          f"消失:{tsu}帧 | {vy_str} | "
                          f"base_y={best_base_y} dt={best_dt} | "
                          f"pred_y={best_pred_cl[1]} | "
                          f"X=[{best_pred_cl[0]},{best_pred_cl[2]}] | "
                          f"距L{nearest + 1}:{int(dist)}px | {traj_info}")

        # 绘制线段
        self.draw_segment_lines(processed_frame, segments)

        # 清理已消失目标的触发记录
        for track_id in list(self.triggered_records.keys()):
            if track_id not in active_ids:
                if self.tracker.get_tracker_by_id(track_id) is None:
                    del self.triggered_records[track_id]

        # 发送串口
        if send_serial:
            for sid in set(all_triggered):
                self._send_serial(sid)

        return (processed_frame, inference_time, raw_count,
                tracked_count, trigger_info)

    def print_line_position_suggestion(self, height):
        if not self.all_visible_y:
            print("\n[诊断] 没有收集到任何crop目标的y坐标")
            return

        ys = np.array(self.all_visible_y)
        y_min, y_max = int(ys.min()), int(ys.max())
        y_median = int(np.median(ys))
        y_mean = int(np.mean(ys))

        hist, bin_edges = np.histogram(ys, bins=20)
        densest_bin = int(np.argmax(hist))
        dense_y_start = int(bin_edges[densest_bin])
        dense_y_end = int(bin_edges[densest_bin + 1])
        dense_y_mid = (dense_y_start + dense_y_end) // 2

        seg_y = int(
            height * self.config.horizontal_config['y_position_ratio'])

        print(f"\n{'=' * 50}")
        print(f"[线段位置诊断]")
        print(f"  当前线段Y     : {seg_y} "
              f"(比例: {self.config.horizontal_config['y_position_ratio']:.2f})")
        print(f"  crop目标Y范围 : {y_min} ~ {y_max}")
        print(f"  crop目标Y中位 : {y_median}")
        print(f"  crop目标Y均值 : {y_mean}")
        print(f"  目标最密集区间: {dense_y_start} ~ {dense_y_end} "
              f"(中点: {dense_y_mid})")
        print(f"  采样数        : {len(ys)}")

        suggested_ratio = dense_y_mid / height
        print(f"\n  ★ 建议: --line-y {suggested_ratio:.2f}")
        print(f"    这会把线段放到 y={dense_y_mid}，正好在目标最密集的区域")

        if abs(seg_y - y_median) > 100:
            print(f"\n  ⚠ 警告: 当前线段(y={seg_y})离目标活动区域"
                  f"(y≈{y_median})相差{abs(seg_y - y_median)}px")
            print(f"    目标不经过线段位置，触发可能性低！")
        print(f"{'=' * 50}\n")


# ============================================================================
# 摄像头检测线程
# ============================================================================
class CameraThread(threading.Thread):
    def __init__(self, yolov8_wrapper, camera_id=0, config=None,
                 tracker_cfg=None, serial_conn=None, categories=None,
                 conf_thresh=0.5):
        threading.Thread.__init__(self)
        self.yolov8_wrapper = yolov8_wrapper
        self.camera_id = camera_id
        self.conf_thresh = conf_thresh
        self.capture = cv2.VideoCapture(self.camera_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.capture.isOpened():
            raise RuntimeError("Could not open camera")

        self.processor = BaseDetectionProcessor(
            yolov8_wrapper, config, tracker_cfg,
            serial_conn, categories
        )

    def run(self):
        total_normal = 0
        total_predict = 0

        while True:
            ret, frame = self.capture.read()
            if not ret:
                print("无法读取摄像头画面")
                break

            try:
                processed_frame, inference_time, raw_count, \
                    tracked_count, trigger_info = \
                    self.processor.process_frame(
                        frame, send_serial=True,
                        conf_thresh=self.conf_thresh)

                total_normal += len(trigger_info['normal_triggers'])
                total_predict += len(trigger_info['predicted_triggers'])

                predicting = sum(
                    1 for t in self.processor.tracker.trackers
                    if 0 < t.time_since_update <= self.processor.tracker.predict_frames
                    and t.get_effective_velocity() is not None
                )

                info_text = [
                    f"Inference: {inference_time * 1000:.1f}ms",
                    f"Tracked: {tracked_count}  Predicting: {predicting}",
                    f"Triggers: normal={total_normal} pred={total_predict}",
                ]

                for i, text in enumerate(info_text):
                    cv2.putText(processed_frame, text,
                                (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)

                cv2.imshow('Detection - Camera', processed_frame)

            except Exception as e:
                print(f"处理帧时出错: {e}")
                traceback.print_exc()
                cv2.imshow('Detection - Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.capture.release()
        cv2.destroyAllWindows()


# ============================================================================
# 视频文件检测器
# ============================================================================
class VideoDetector:
    def __init__(self, yolov8_wrapper, config=None, tracker_cfg=None,
                 serial_conn=None, categories=None):
        self.yolov8_wrapper = yolov8_wrapper
        self.config = config if config is not None else LineConfig()
        self.processor = BaseDetectionProcessor(
            yolov8_wrapper, config, tracker_cfg,
            serial_conn, categories
        )

    def detect_video(self, video_path, output_path=None,
                     show_display=True, send_serial=False,
                     conf_thresh=0.5):
        if not os.path.exists(video_path):
            print(f"错误：视频文件不存在 - {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 - {video_path}")
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        seg_y = int(
            h * self.processor.config.horizontal_config['y_position_ratio'])

        print(f"\n{'=' * 50}")
        print(f"视频信息: {video_path}")
        print(f"  分辨率: {w}x{h}  帧率: {fps}  总帧数: {total}")
        print(f"  时长: {total / fps:.2f} 秒")
        print(f"  线段Y位置: {seg_y}")
        print(f"  线段数量: {self.config.num_segments}")
        tk = self.processor.tracker
        print(f"跟踪器: max_age={tk.max_age}  min_hits={tk.min_hits}  "
              f"iou={tk.iou_threshold}  predict_frames={tk.predict_frames} "
              f"dist_threshold={tk.dist_threshold}")
        print(f"撞线容差: {COLLISION_Y_TOLERANCE}px")
        print(f"速度拟合: 仅 vy（垂直方向），X 坐标锁定")
        print(f"{'=' * 50}\n")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        self.processor.tracker.reset()
        self.processor.triggered_records.clear()
        self.processor.segment_highlight.clear()
        self.processor.all_visible_y.clear()

        frame_count = 0
        total_inf = 0
        total_det = 0
        total_pred_triggers = 0
        total_normal_triggers = 0
        start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            try:
                processed, inf_time, raw_cnt, trk_cnt, trigger_info = \
                    self.processor.process_frame(
                        frame, send_serial=send_serial,
                        conf_thresh=conf_thresh)
                total_inf += inf_time
                total_det += trk_cnt

                total_pred_triggers += len(
                    trigger_info['predicted_triggers'])
                total_normal_triggers += len(
                    trigger_info['normal_triggers'])

                predicting = sum(
                    1 for t in self.processor.tracker.trackers
                    if 0 < t.time_since_update <= self.processor.tracker.predict_frames
                    and t.get_effective_velocity() is not None
                )

                elapsed = time.time() - start
                avg_fps = frame_count / elapsed if elapsed > 0 else 0
                progress = frame_count / total * 100 if total > 0 else 0

                info_lines = [
                    f"Frame: {frame_count}/{total} ({progress:.1f}%)",
                    f"Inference: {inf_time * 1000:.1f}ms  FPS: {avg_fps:.1f}",
                    f"Tracked: {trk_cnt}  Predicting: {predicting}",
                    f"Line Y: {seg_y}  Tol: {COLLISION_Y_TOLERANCE}px  Segments: {self.config.num_segments}",
                    f"Triggers: normal={total_normal_triggers} "
                    f"pred={total_pred_triggers}",
                ]

                for i, txt in enumerate(info_lines):
                    txt_color = (
                        (0, 0, 255)
                        if total_pred_triggers > 0 and 'pred=' in txt
                        else (0, 255, 0)
                    )
                    cv2.putText(processed, txt, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                txt_color, 2)

                # 图例
                ly = 30 + len(info_lines) * 25 + 10
                cv2.rectangle(processed, (5, ly - 5), (240, ly + 95),
                              (50, 50, 50), -1)
                cv2.putText(processed, 'Legend:', (10, ly + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                cv2.rectangle(processed, (10, ly + 25), (30, ly + 40),
                              CLASS_COLORS['weeds'], -1)
                cv2.putText(processed, 'Weeds', (35, ly + 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
                cv2.rectangle(processed, (10, ly + 45), (30, ly + 60),
                              CLASS_COLORS['crop'], -1)
                cv2.putText(processed, 'Crop/Corn', (35, ly + 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
                cv2.line(processed, (10, ly + 73), (30, ly + 73),
                         (0, 0, 255), 3)
                cv2.putText(processed, 'Predict CL', (35, ly + 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
                cv2.line(processed, (10, ly + 90), (30, ly + 90),
                         (0, 255, 255), 2)
                cv2.putText(processed, 'Visible CL', (35, ly + 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

                if writer:
                    writer.write(processed)
                if show_display:
                    cv2.imshow('Detection', processed)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)

                if frame_count % 100 == 0:
                    print(f"进度: {frame_count}/{total} ({progress:.1f}%) "
                          f"FPS: {avg_fps:.1f}  "
                          f"正常: {total_normal_triggers}  "
                          f"预测: {total_pred_triggers}")

            except Exception as e:
                print(f"第 {frame_count} 帧出错: {e}")
                traceback.print_exc()
                if writer:
                    writer.write(frame)

        cap.release()
        if writer:
            writer.release()
        if show_display:
            cv2.destroyAllWindows()

        total_time = time.time() - start
        avg_inf = (total_inf / frame_count * 1000) if frame_count > 0 else 0
        avg_fps = frame_count / total_time if total_time > 0 else 0

        stats = {
            'video_path': video_path,
            'output_path': output_path,
            'total_frames': frame_count,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'avg_inference_ms': avg_inf,
            'total_detections': total_det,
            'total_normal_triggers': total_normal_triggers,
            'total_predicted_triggers': total_pred_triggers,
            'num_segments': self.config.num_segments,
        }

        print(f"\n{'=' * 50}")
        print(f"处理完成!")
        print(f"  帧数: {frame_count}  用时: {total_time:.2f}s")
        print(f"  平均FPS: {avg_fps:.1f}  平均推理: {avg_inf:.1f}ms")
        print(f"  线段数量: {self.config.num_segments}")
        print(f"  正常触发: {total_normal_triggers}")
        print(f"  预测触发: {total_pred_triggers}")
        print(f"{'=' * 50}")

        self.processor.print_line_position_suggestion(h)

        return stats

    def detect_video_batch(self, video_paths, output_dir=None,
                           show_display=False, send_serial=False,
                           conf_thresh=0.5):
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_stats = []
        for idx, vp in enumerate(video_paths):
            print(f"\n{'#' * 50}")
            print(f"视频 {idx + 1}/{len(video_paths)}: {vp}")
            print(f"{'#' * 50}")

            op = None
            if output_dir:
                base = os.path.splitext(os.path.basename(vp))[0]
                op = os.path.join(output_dir, f"{base}_detected.mp4")

            KalmanBoxTracker.count = 0
            stats = self.detect_video(
                vp, output_path=op,
                show_display=show_display,
                send_serial=send_serial,
                conf_thresh=conf_thresh,
            )
            if stats:
                all_stats.append(stats)

        if all_stats:
            print(f"\n{'=' * 50}")
            print(f"批量完成: {len(all_stats)}/{len(video_paths)} 成功")
            print(f"{'=' * 50}\n")

        return all_stats


# ============================================================================
# 预热线程（保持不变）
# ============================================================================
class warmUpThread(threading.Thread):
    def __init__(self, yolov8_wrapper):
        threading.Thread.__init__(self)
        self.yolov8_wrapper = yolov8_wrapper

    def run(self):
        batch_image_raw, use_time, _, _, _ = self.yolov8_wrapper.infer(
            self.yolov8_wrapper.get_raw_image_zeros()
        )
        print('warm_up->{}, time->{:.2f}ms'.format(
            batch_image_raw[0].shape, use_time * 1000))


# ============================================================================
# 命令行参数
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='作物/杂草检测系统 (TensorRT加速 | vy-only预测)')

    p.add_argument('--mode', type=str, default='video',
                   choices=['camera', 'video'])
    p.add_argument('--engine', type=str,
                   default='/home/jetson/Desktop/tensorrtx-master/'
                           'yolov8/build/best.engine')
    p.add_argument('--plugin', type=str, default='./build/libmyplugins.so')
    p.add_argument('--video', type=str, default=None)
    p.add_argument('--video-dir', type=str, default=None)
    p.add_argument('--output', type=str, default=None)
    p.add_argument('--camera-id', type=int, default=0)
    p.add_argument('--conf', type=float, default=0.6)
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--categories', type=str, nargs='+', default=['corn'],
                   help='类别名称列表，如: corn 或 crop weeds')

    # 串口
    p.add_argument('--serial-port', type=str, default=None)
    p.add_argument('--serial-baud', type=int, default=115200)
    p.add_argument('--send-serial', action='store_true')
    p.add_argument('--no-serial', action='store_true')

    # 线段
    p.add_argument('--line-mode', type=str, default='horizontal',
                   choices=['horizontal', 'vertical', 'custom'])
    p.add_argument('--num-segments', type=int, default=4, choices=range(1, 9),
                   help='水平线段段数 (1-8)')
    p.add_argument('--line-y', type=float, default=0.6)
    p.add_argument('--line-width', type=float, default=0.15)
    p.add_argument('--vert-x', type=float, nargs='+',
                   default=[0.2, 0.4, 0.6, 0.8])
    p.add_argument('--vert-y-start', type=float, default=0.2)
    p.add_argument('--vert-y-end', type=float, default=0.8)

    # 跟踪器
    p.add_argument('--max-age', type=int, default=60)
    p.add_argument('--min-hits', type=int, default=1)
    p.add_argument('--iou-thresh', type=float, default=0.2)
    p.add_argument('--predict-frames', type=int, default=60)
    p.add_argument('--dist-threshold', type=float, default=120.0)
    p.add_argument('--recover-max-age', type=int, default=60)

    # 撞线
    p.add_argument('--collision-tolerance', type=int, default=50)
    p.add_argument('--min-traj-points', type=int, default=3)

    # 调试
    p.add_argument('--debug-assoc', action='store_true')
    p.add_argument('--debug-track', action='store_true')
    p.add_argument('--debug-predict', action='store_true')

    return p.parse_args()


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    args = parse_args()

    COLLISION_Y_TOLERANCE = args.collision_tolerance
    MIN_TRAJECTORY_POINTS = args.min_traj_points
    DEBUG_ASSOC = args.debug_assoc
    DEBUG_TRACK = args.debug_track
    DEBUG_PREDICT = args.debug_predict
    CONF_THRESH = args.conf

    if args.max_age < args.predict_frames:
        print(f"[警告] max_age({args.max_age}) < "
              f"predict_frames({args.predict_frames})，"
              f"自动提升到 {args.predict_frames}")
        args.max_age = args.predict_frames

    if args.recover_max_age < args.predict_frames:
        print(f"[警告] recover_max_age({args.recover_max_age}) < "
              f"predict_frames({args.predict_frames})，"
              f"自动提升到 {args.predict_frames}")
        args.recover_max_age = args.predict_frames

    # 加载插件
    PLUGIN_LIBRARY = args.plugin
    engine_file_path = args.engine

    try:
        ctypes.CDLL(PLUGIN_LIBRARY)
        print(f"[插件] 加载成功: {PLUGIN_LIBRARY}")
    except Exception as e:
        print(f"[插件] 加载失败: {e}")

    categories = args.categories

    if os.path.exists('output/'):
        shutil.rmtree('output/')
    os.makedirs('output/')

    # 线段配置
    line_cfg = LineConfig()
    line_cfg.layout_mode = args.line_mode
    line_cfg.num_segments = args.num_segments
    line_cfg.horizontal_config['y_position_ratio'] = args.line_y
    line_cfg.horizontal_config['segment_width_ratio'] = args.line_width
    line_cfg.vertical_config['x_positions_ratio'] = args.vert_x
    line_cfg.vertical_config['start_y_ratio'] = args.vert_y_start
    line_cfg.vertical_config['end_y_ratio'] = args.vert_y_end

    # 跟踪器配置
    tracker_cfg = {
        'max_age': args.max_age,
        'min_hits': args.min_hits,
        'iou_threshold': args.iou_thresh,
        'predict_frames': args.predict_frames,
        'dist_threshold': args.dist_threshold,
        'recover_max_age': args.recover_max_age,
    }

    # 串口
    serial_conn = None
    if not args.no_serial and (args.serial_port or args.send_serial):
        port = args.serial_port or '/dev/ttyTHS0'
        serial_conn = init_serial(port, args.serial_baud)

    # 初始化模型
    yolov8_wrapper = YoLov8TRT(engine_file_path)

    try:
        print('batch size is', yolov8_wrapper.batch_size)

        print("开始预热...")
        for i in range(5):
            thread1 = warmUpThread(yolov8_wrapper)
            thread1.start()
            thread1.join()
        print("预热完成")

        print(f"\n{'=' * 60}")
        print("★ 作物/杂草检测系统（TensorRT加速 | vy-only预测）★")
        print(f"{'=' * 60}")
        print(f"运行模式       : {args.mode.upper()}")
        print(f"类别           : {categories}")
        print(f"置信度阈值     : {CONF_THRESH}")
        print(f"线段数量       : {args.num_segments}")
        print(f"线段Y位置      : {args.line_y * 100:.0f}%")
        print(f"预测帧数       : {args.predict_frames}")
        print(f"max_age        : {args.max_age}")
        print(f"dist_threshold : {args.dist_threshold}")
        print(f"recover_max_age: {args.recover_max_age}")
        print(f"撞线容差       : {COLLISION_Y_TOLERANCE}px")
        print(f"轨迹拟合       : 仅 vy（垂直方向），最少{MIN_TRAJECTORY_POINTS}点")
        print(f"预测X坐标      : 锁定消失前位置（不外推）")
        print(f"撞线X检测      : 完整保留（X 范围重叠判断）")
        print(f"{'=' * 60}\n")

        if args.mode == 'camera':
            print("启动摄像头检测...")
            camera_thread = CameraThread(
                yolov8_wrapper,
                camera_id=args.camera_id,
                config=line_cfg,
                tracker_cfg=tracker_cfg,
                serial_conn=serial_conn,
                categories=categories,
                conf_thresh=args.conf,
            )
            camera_thread.start()
            camera_thread.join()

        elif args.mode == 'video':
            video_detector = VideoDetector(
                yolov8_wrapper,
                config=line_cfg,
                tracker_cfg=tracker_cfg,
                serial_conn=serial_conn,
                categories=categories,
            )

            if args.video_dir:
                exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
                files = [
                    os.path.join(args.video_dir, f)
                    for f in os.listdir(args.video_dir)
                    if os.path.splitext(f)[1].lower() in exts
                ]
                if files:
                    print(f"找到 {len(files)} 个视频文件")
                    video_detector.detect_video_batch(
                        files,
                        output_dir=args.output,
                        show_display=not args.no_display,
                        send_serial=args.send_serial,
                        conf_thresh=args.conf,
                    )
                else:
                    print(f"未在目录中找到视频文件: {args.video_dir}")
            else:
                video_path = (args.video
                              or '/home/jetson/Desktop/crop2.mp4')
                output_path = (args.output
                               or '/home/jetson/Desktop/result.mp4')
                video_detector.detect_video(
                    video_path,
                    output_path=output_path,
                    show_display=not args.no_display,
                    send_serial=args.send_serial,
                    conf_thresh=args.conf,
                )

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"运行出错: {e}")
        traceback.print_exc()
    finally:
        yolov8_wrapper.destroy()
        if serial_conn is not None:
            serial_conn.close()
        print("程序结束")