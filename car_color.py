#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WINDOWS_OCR_SCRIPT = r"""param([Parameter(Mandatory = $true)][string]$ManifestPath)

# Local OCR only. No installation, network request, or visible window.
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime]

$taskMethod = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and
    $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
} | Select-Object -First 1

function Wait-OcrOperation($Operation, [Type]$ResultType) {
    $task = $taskMethod.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.GetAwaiter().GetResult()
}

$language = [Windows.Globalization.Language]::new('en-US')
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
if ($null -eq $engine) {
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
}
if ($null -eq $engine) { throw 'Windows OCR language is unavailable.' }

$paths = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$texts = @()
foreach ($path in $paths) {
    $stream = $bitmap = $null
    try {
        $file = Wait-OcrOperation ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
        $stream = Wait-OcrOperation ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
        $decoder = Wait-OcrOperation ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap = Wait-OcrOperation ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        $result = Wait-OcrOperation ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
        $texts += [string]$result.Text
    }
    finally {
        if ($null -ne $bitmap) { $bitmap.Dispose() }
        if ($null -ne $stream) { $stream.Dispose() }
    }
}
ConvertTo-Json -InputObject $texts -Compress
"""

# --- CAU HINH ---

PROJECT_ROOT = Path(__file__).resolve().parent
DETECTION_MODEL = str(PROJECT_ROOT / 'yolo11s.pt')
PP_MODEL_DIR = PROJECT_ROOT / 'models' / 'pp_vehicle_attribute'
COLOR_NAMES = ['yellow', 'orange', 'green', 'gray', 'red', 'blue', 'white', 'golden', 'brown', 'black']
VEHICLE_TYPE_NAMES = ['sedan', 'suv', 'van', 'hatchback', 'mpv', 'pickup', 'bus', 'truck', 'estate']
PASSENGER_TYPES = {'sedan', 'suv', 'hatchback', 'mpv', 'estate'}
COLOR_VI = {'yellow': 'VÀNG', 'orange': 'CAM', 'green': 'XANH LÁ', 'gray': 'XÁM / BẠC', 'red': 'ĐỎ', 'blue': 'XANH DƯƠNG', 'white': 'TRẮNG', 'golden': 'VÀNG KIM', 'brown': 'NÂU', 'black': 'ĐEN'}
CLASSES = {2: 'Ô TÔ', 5: 'XE KHÁCH', 7: 'XE TẢI'}


@dataclass
class RuntimeConfig:
    """Cấu hình triển khai ổn định; CLI chỉ mở các tùy chọn vận hành."""

    source: str
    output: str | None = None
    paddle_device: str = 'auto'
    render_width: int = 1280
    no_show: bool = False
    roi_file: str | None = None
    redraw_roi: bool = False
    no_roi: bool = False
    camera_start: str | None = None
    clock_box: tuple[int, int, int, int] | None = None
    day_mode: bool = False

    model: str = DETECTION_MODEL
    model_dir: str = str(PP_MODEL_DIR)
    screen_w: int = 1280
    screen_h: int = 800
    roi_confirm_seconds: float = 0.3
    roi_exit_grace: float = .8
    roi_boundary_grace: float = 14.0
    conf: float = .15
    new_conf: float = .30
    iou: float = .50
    imgsz: int = 960
    min_area: int = 650
    min_hits: int = 2
    hold_frames: int = 6
    no_flow_hold_frames: int = 2
    reid_frames: int = 90
    box_alpha: float = .78
    recover_suv: bool = True
    suv_type_conf: float = .40
    suv_type_every: int = 8
    suv_min_side: int = 26
    vote_window: int = 9
    min_color_samples: int = 3
    color_conf: float = .46
    color_every: int = 1
    color_every_confirmed: int = 4
    switch_hits: int = 8
    switch_hits_neutral: int = 10
    min_color_side: int = 28
    white_gray_assist: bool = True
    two_colors: bool = True
    two_color_conf: float = .65
    second_share: float = .08
    night_from: int = 18
    night_until: int = 6
    night_alt_conf: float = .48
    night_check_every: int = 2
    scene_light_guard: bool = True

    video_fps: float = 25.0
    roi_enabled: bool = True
    scene_green_shift: bool = False
    night_active: bool = False


# --- NHAN VA HINH HOC ---

def output_path(source):
    """Tạo tên video xuất mặc định: outputs/<tên-video>-ketqua.mp4."""
    source_name = str(source)
    name = f"camera_{source_name}" if source_name.isdigit() else Path(source_name).stem
    safe_name = re.sub('[^\\w-]+', '_', name, flags=re.UNICODE).strip('_')
    return Path('outputs') / f"{safe_name or 'video'}-ketqua.mp4"

@lru_cache(maxsize=16)
def get_font(size=19):
    for path in ('C:/Windows/Fonts/arialbd.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf'):
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()

def clip_box(box, width, height):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = (max(0, min(x1, width - 1)), max(0, min(y1, height - 1)))
    x2, y2 = (max(x1 + 1, min(x2, width)), max(y1 + 1, min(y2, height)))
    return (x1, y1, x2, y2)

def format_vehicle_label(car_id, detection_conf, color):
    """ID ít nhất 2 chữ số; Conf YOLO theo %, ví dụ 0.89 -> 089."""
    confidence = int(round(float(np.clip(detection_conf, 0.0, 1.0)) * 100))
    color_label = ' và '.join((part.capitalize() for part in color.split(' và ')))
    return f"ID:{int(car_id):02d}-Conf:{confidence:03d}-{color_label}"

def argsafe_hold(fps):
    return max(1, int(round(0.22 * fps)))

def rectangles_overlap(a, b, min_pad=3):
    return not (a[2] + min_pad <= b[0] or b[2] + min_pad <= a[0] or a[3] + min_pad <= b[1] or (b[3] + min_pad <= a[1]))

def box_iou(a, b):
    xa = max(float(a[0]), float(b[0]))
    ya = max(float(a[1]), float(b[1]))
    xb = min(float(a[2]), float(b[2]))
    yb = min(float(a[3]), float(b[3]))
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1.0, aa + bb - inter)

def center_size(box):
    b = np.asarray(box, dtype=np.float32)
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])], dtype=np.float32)

def deduplicate_detections(detections, iou_threshold=0.72):
    """Tránh tạo hai track khi cùng xe nhận cả car/bus và truck."""
    ordered = sorted(detections, key=lambda d: (d['class_id'] in (2, 5), d['conf']), reverse=True)
    kept = []
    for det in ordered:
        if any((box_iou(det['box'], other['box']) >= iou_threshold for other in kept)):
            continue
        kept.append(det)
    return kept

def feature_points_in_box(gray, box, max_points=45):
    """Lấy đặc trưng KLT trong phần giữa bbox, chỉ dùng để giữ khung ngắn hạn."""
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = clip_box(box, w, h)
    bw, bh = (x2 - x1, y2 - y1)
    if min(bw, bh) < 17:
        return None
    mx = max(2, int(0.12 * bw))
    my = max(2, int(0.12 * bh))
    lx, rx = (x1 + mx, x2 - mx)
    ty, by = (y1 + my, y2 - my)
    if rx - lx < 8 or by - ty < 8:
        return None
    # Find corners locally; scanning the whole camera frame for every car is
    # expensive and lets unrelated high-contrast objects set the threshold.
    region = gray[y1:y2, x1:x2]
    mask = np.zeros(region.shape, dtype=np.uint8)
    mask[ty-y1:by-y1, lx-x1:rx-x1] = 255
    points = cv2.goodFeaturesToTrack(region, maxCorners=max_points,
                                   qualityLevel=0.015, minDistance=4,
                                   blockSize=5, mask=mask)
    if points is not None:
        points += np.array([x1, y1], dtype=np.float32)
    return points


# --- GIO CAMERA (THAM KHAO) ---

def parse_camera_clock(value):
    """Chấp nhận HH:MM, HH:MM:SS hoặc YYYY-MM-DD HH:MM:SS."""
    value = str(value).strip()
    for fmt in ('%Y-%m-%d %H:%M:%S', '%H:%M:%S', '%H:%M'):
        try:
            dt = datetime.datetime.strptime(value, fmt)
            return dt.time().replace(microsecond=0)
        except ValueError:
            pass
    raise ValueError('Thời gian camera phải là HH:MM:SS (VD 18:06:37) hoặc YYYY-MM-DD HH:MM:SS')

def clock_from_text(raw):
    """Đọc HH:MM:SS; kiểm tra khoảng giờ và từ chối nhiều giờ khác nhau."""
    pattern = '(?<![\\dA-Za-z])([0-9OoIl]{1,2})\\s*[:.]\\s*([0-9OoIl]{2})\\s*[:.;iIl]\\s*([0-9OoIl]{2})(?![\\dA-Za-z])'
    values = set()
    substitutions = str.maketrans({'O': '0', 'o': '0', 'I': '1', 'l': '1'})
    for match in re.finditer(pattern, str(raw)):
        try:
            values.add(parse_camera_clock(':'.join(match.groups()).translate(substitutions)))
        except ValueError:
            pass
    return next(iter(values)) if len(values) == 1 else None

def clock_ocr_images(frame, clock_box=None):
    """Chuẩn bị vùng timestamp, tách chữ trắng khỏi nền camera."""
    h, w = frame.shape[:2]
    if clock_box is None:
        x1, y1, x2, y2 = (int(0.7 * w), 0, w, int(0.13 * h))
    else:
        x1, y1, x2, y2 = clock_box
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError('--clock-box phải nằm trong kích thước frame gốc')
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return []
    variants = [cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)]
    if clock_box is None:
        loose = frame[int(0.035 * h):max(int(0.09 * h), int(0.035 * h) + 1), int(0.855 * w):int(0.98 * w)]
        tight = frame[int(0.04 * h):max(int(0.078 * h), int(0.04 * h) + 1), int(0.875 * w):int(0.98 * w)]
    else:
        loose = tight = roi
    if loose.size and tight.size:
        white = cv2.inRange(cv2.cvtColor(loose, cv2.COLOR_BGR2HSV), (0, 0, 200), (179, 80, 255))
        variants.append(255 - white)
        _, binary = cv2.threshold(cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY_INV)
        variants.append(binary)
        white = cv2.inRange(cv2.cvtColor(tight, cv2.COLOR_BGR2HSV), (0, 0, 190), (179, 100, 255))
        variants.append(cv2.resize(255 - white, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        variants.append(cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY))
        _, binary = cv2.threshold(cv2.cvtColor(loose, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY_INV)
        variants.append(binary)
    return [cv2.copyMakeBorder(v, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255) for v in variants]

def ocr_clock_images(images):
    """Windows OCR có sẵn; Tesseract là phương án phụ nếu đã được cài."""
    powershell = shutil.which('powershell.exe') if sys.platform == 'win32' else None
    if powershell and images:
        try:
            with tempfile.TemporaryDirectory(prefix='vehicle_clock_') as directory:
                paths = []
                for index, img in enumerate(images):
                    path = Path(directory) / f"ocr_{index}.png"
                    if not cv2.imwrite(str(path), img):
                        raise OSError('Không ghi được ảnh tạm cho OCR')
                    paths.append(str(path.resolve()))
                manifest = Path(directory) / 'images.json'
                manifest.write_text(json.dumps(paths), encoding='utf-8')
                manifest_literal=str(manifest).replace("'","''")
                command="$ManifestPath = '"+manifest_literal+"'\n"+WINDOWS_OCR_SCRIPT.split('\n',1)[1]
                result = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-Command', command], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if result.returncode == 0:
                    texts = json.loads(result.stdout.lstrip('\ufeff'))
                    if isinstance(texts, list) and len(texts) == len(images):
                        clocks = [clock_from_text(text) for text in texts]
                        if any((clock is not None for clock in clocks)):
                            return texts
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    try:
        import pytesseract
        return [pytesseract.image_to_string(img, config='--psm 6', timeout=3) for img in images]
    except (ImportError, OSError, RuntimeError):
        return []

def clock_from_overlay(frame, clock_box=None):
    """OCR các phiên bản cùng một frame; không nhận nếu có nhiều giờ khác nhau."""
    values = {clock for text in ocr_clock_images(clock_ocr_images(frame, clock_box)) if (clock := clock_from_text(text)) is not None}
    return next(iter(values)) if len(values) == 1 else None

def camera_clock_consensus(observations, fps):
    """Suy giờ frame 0 từ ít nhất hai frame, chấp nhận sai số lượng tử 1 giây."""
    if len(observations) < 2 or len({clock for _, clock in observations}) < 2:
        return None
    initial = [(clock.hour * 3600 + clock.minute * 60 + clock.second - frame / max(1.0, fps)) % 86400 for frame, clock in observations]
    groups = [[index for index, value in enumerate(initial) if abs((value - base + 43200) % 86400 - 43200) <= 1.1] for base in initial]
    groups = [group for group in groups if len({observations[index][1] for index in group}) >= 2]
    if not groups:
        return None
    best = max(groups, key=len)
    if len(best) < max(2, int(np.ceil(2 * len(observations) / 3))):
        return None
    base = initial[best[0]]
    aligned = [base + ((initial[index] - base + 43200) % 86400 - 43200) for index in best]
    seconds = int(round(float(np.median(aligned)))) % 86400
    return datetime.time(seconds // 3600, seconds % 3600 // 60, seconds % 60)

def get_camera_start(cap, args):
    """Ưu tiên --camera-start, rồi OCR ba frame; không yêu cầu nhập tay."""
    if args.camera_start:
        clock = parse_camera_clock(args.camera_start)
        print(f"[CLOCK] Dùng giờ camera do người dùng nhập: {clock}")
        return clock
    is_file = not str(args.source).isdigit() and (not str(args.source).lower().startswith(('rtsp://', 'http://', 'https://')))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    except (ValueError, TypeError):
        fps = 25.0
    if not 1 <= fps <= 240:
        fps = 25.0
    positions = [0, round(fps), round(2 * fps)] if is_file else [0]
    observations = []
    try:
        for position in positions:
            if is_file and (not cap.set(cv2.CAP_PROP_POS_FRAMES, position)):
                break
            ok, frame = cap.read()
            if not ok:
                break
            clock = clock_from_overlay(frame, args.clock_box)
            if clock is not None:
                observations.append((position, clock))
    finally:
        if is_file and (not cap.set(cv2.CAP_PROP_POS_FRAMES, 0)):
            raise RuntimeError('Không tua lại được video sau khi kiểm tra đồng hồ camera')
    candidate = camera_clock_consensus(observations, fps)
    if candidate is not None:
        print(f"[CLOCK] OCR xác nhận giờ frame đầu: {candidate} từ nhiều frame; dùng tính mốc 18:00")
        return candidate
    print('[LIGHT] Chưa xác nhận được giờ camera; tự kiểm tra độ sáng ảnh, không cần nhập giờ.')
    return None

def frame_brightness(frame):
    """Độ sáng trung vị ở giữa ảnh; chỉ dùng khi không có giờ camera."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(np.median(gray[9:81, 16:144]))

def automatic_night_mode(frame, previous_brightness=None, previous_night=False):
    """Ảnh tối bật kiểm tra đèn; làm mượt và dùng hai ngưỡng chống bật/tắt liên tục."""
    measured = frame_brightness(frame)
    brightness = measured if previous_brightness is None else 0.9 * previous_brightness + 0.1 * measured
    threshold = 95.0 if previous_night else 80.0
    return (brightness < threshold, brightness)

def is_night_clock(start_clock, frame_number, fps, night_hour=18, morning_hour=6):
    """Đổi giờ camera theo thời lượng VIDEO, không dựa vào tốc độ chạy model."""
    initial = start_clock.hour * 3600 + start_clock.minute * 60 + start_clock.second
    seconds = (initial + max(0, frame_number - 1) / max(1.0, fps)) % 86400
    return seconds >= night_hour * 3600 or seconds < morning_hour * 3600


# --- ROI ---

def point_in_vehicle_roi(box, polygon):
    """Kiểm tra điểm giữa cạnh ĐÁY bbox, phù hợp vùng đường, không dùng tâm xe."""
    if polygon is None:
        return True
    x1, _, x2, y2 = box
    return cv2.pointPolygonTest(polygon, ((float(x1) + float(x2)) / 2, float(y2)), False) >= 0

def vehicle_roi_signed_distance(box, polygon):
    """Khoảng cách có dấu từ điểm giữa cạnh đáy bbox đến mép ROI (pixel)."""
    if polygon is None:
        return float('inf')
    x1, _, x2, y2 = box
    return float(cv2.pointPolygonTest(polygon, ((float(x1) + float(x2)) / 2, float(y2)), True))


def roi_search_box(polygon, width, height, boundary_grace=14.0, imgsz=960):
    """Chỉ cắt ROI khi kích thước tensor YOLO thực sự nhỏ hơn toàn frame."""
    if polygon is None:
        return (0, 0, width, height)
    x, y, box_width, box_height = cv2.boundingRect(
        np.asarray(polygon, dtype=np.int32))
    pad_x = max(32, round(.04 * width), round(2 * boundary_grace))
    pad_top = max(48, round(.15 * height))
    pad_bottom = max(32, round(.04 * height))
    candidate = clip_box((x - pad_x, y - pad_top,
                          x + box_width + pad_x, y + box_height + pad_bottom),
                         width, height)

    def tensor_area(box):
        box_w, box_h = box[2] - box[0], box[3] - box[1]
        scale = min(imgsz / box_w, imgsz / box_h)
        tensor_w = int(np.ceil(box_w * scale / 32.0) * 32)
        tensor_h = int(np.ceil(box_h * scale / 32.0) * 32)
        return tensor_w * tensor_h

    full = (0, 0, width, height)
    return candidate if tensor_area(candidate) < .85 * tensor_area(full) else full


def read_roi_file(filename, width, height):
    path = Path(filename)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        old_width, old_height = (int(data['frame_width']), int(data['frame_height']))
        pts = np.asarray(data['points'], dtype=np.float32)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Không đọc được ROI file {path}: {exc}") from exc
    if old_width < 1 or old_height < 1 or pts.ndim != 2 or (pts.shape[1] != 2) or (len(pts) < 3):
        raise RuntimeError('ROI file không hợp lệ: cần >=3 điểm và kích thước video hợp lệ')
    if not np.all(np.isfinite(pts)) or np.any(pts < 0) or np.any(pts[:, 0] >= old_width) or np.any(pts[:, 1] >= old_height):
        raise RuntimeError('ROI file có tọa độ điểm nằm ngoài frame')
    old_ratio, new_ratio = (old_width / old_height, width / height)
    if abs(old_ratio / new_ratio - 1) > 0.025:
        raise RuntimeError('Tỉ lệ khung hình ROI file khác video hiện tại. Vẽ lại ROI cho camera này.')
    pts[:, 0] *= width / old_width
    pts[:, 1] *= height / old_height
    result = np.round(pts).astype(np.int32)
    result[:, 0] = np.clip(result[:, 0], 0, width - 1)
    result[:, 1] = np.clip(result[:, 1], 0, height - 1)
    if abs(cv2.contourArea(result)) < 20:
        raise RuntimeError('ROI quá nhỏ hoặc các điểm trùng nhau')
    return result

def write_roi_file(filename, polygon, width, height):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {'version': 1, 'frame_width': width, 'frame_height': height, 'points': np.asarray(polygon, dtype=np.int32).tolist()}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[INFO] Đã lưu ROI tại: {path}")

def select_vehicle_roi(frame_bgr, screen_w=1280, screen_h=800):
    """Chọn ROI đa giác trên frame đầu bằng cửa sổ OpenCV."""
    height, width = frame_bgr.shape[:2]
    scale = min(max(100, screen_w - 60) / width,
                max(100, screen_h - 100) / height, 1.0)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    background = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
    window = 'Chon ROI | Click diem - Enter: xac nhan - Z: lui - R: xoa - Esc: huy'
    points = []

    def on_mouse(event, x, y, _flags, _data):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))

    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, on_mouse)
        while True:
            canvas = background.copy()
            if len(points) >= 3:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [np.asarray(points, np.int32)], (255, 200, 40))
                canvas = cv2.addWeighted(overlay, .18, canvas, .82, 0)
            if len(points) >= 2:
                cv2.polylines(canvas, [np.asarray(points, np.int32)], False, (65, 205, 255), 2)
            for index, point in enumerate(points, 1):
                cv2.circle(canvas, point, 5, (90, 110, 255), -1)
                cv2.putText(canvas, str(index), (point[0] + 7, point[1] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xff
            if key in (ord('z'), ord('Z')) and points:
                points.pop()
            elif key in (ord('r'), ord('R')):
                points.clear()
            elif key == 27:
                raise RuntimeError('Bạn đã hủy bước chọn ROI.')
            elif key in (10, 13):
                polygon = np.round(np.asarray(points, np.float32) / scale).astype(np.int32)
                if len(points) >= 3 and abs(cv2.contourArea(polygon)) >= 20:
                    break
                print('[WARN] ROI cần ít nhất 3 điểm và phải có diện tích hợp lệ.')
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError('Bạn đã đóng cửa sổ chọn ROI.')
    except cv2.error as exc:
        raise RuntimeError('Không mở được cửa sổ ROI. Dùng --no-roi khi chạy không có màn hình.') from exc
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    print(f"[INFO] Đã chọn ROI {len(polygon)} đỉnh. ROI chỉ dùng để lọc, không vẽ lên video.")
    return polygon

def setup_roi(cap, source, args, width, height):
    """Mặc định LUÔN vẽ ROI trước khi chạy, trừ khi --no-roi."""
    if getattr(args, 'no_roi', False):
        print('[INFO] Bỏ bước chọn ROI theo yêu cầu --no-roi')
        return None
    if args.roi_file and Path(args.roi_file).is_file() and (not getattr(args, 'redraw_roi', False)):
        polygon = read_roi_file(args.roi_file, width, height)
        print(f"[INFO] Sử dụng ROI từ {args.roi_file} ({len(polygon)} đỉnh)")
        return polygon
    ok, first = cap.read()
    if not ok:
        raise RuntimeError('Không lấy được frame đầu để vẽ ROI.')
    if not str(source).isdigit() and (not str(source).lower().startswith(('rtsp://', 'http://', 'https://'))):
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            raise RuntimeError('Video không hỗ trợ tua về frame đầu sau khi vẽ ROI.')
    polygon = select_vehicle_roi(first, args.screen_w, args.screen_h)
    if args.roi_file:
        write_roi_file(args.roi_file, polygon, width, height)
    return polygon


# --- MODEL PP-VEHICLE ---

def find_paddle_model(directory):
    """Tìm cặp model + params, hỗ trợ cả pdmodel (Paddle 2.x) và json (Paddle 3.x)."""
    directory = Path(directory)
    if not directory.exists():
        return None
    for pattern in ('**/*.pdmodel', '**/*.json'):
        for model_file in sorted(directory.glob(pattern)):
            if model_file.name in {'infer_cfg.json', 'config.json'}:
                continue
            param_file = model_file.with_suffix('.pdiparams')
            if param_file.is_file():
                return (model_file, param_file)
            candidates = list(model_file.parent.glob('*.pdiparams'))
            if len(candidates) == 1:
                return (model_file, candidates[0])
    return None

class PPVehicleColorClassifier:
    """Đưa ảnh xe qua PP-LCNet inference của PaddlePaddle."""

    def __init__(self, model_dir=PP_MODEL_DIR, device='cpu'):
        files = find_paddle_model(model_dir)
        if not files:
            raise FileNotFoundError(
                f"Không tìm thấy model PP-Vehicle cục bộ trong '{Path(model_dir)}'. "
                "Cần có model.pdmodel (hoặc model.json) và model.pdiparams."
            )
        self.model_file, self.param_file = files
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='No ccache found.*')
                import paddle
                import paddle.inference as paddle_infer
        except ImportError as exc:
            raise RuntimeError('Thiếu paddlepaddle. Cài: python -m pip install paddlepaddle\nNếu cài lỗi, hãy dùng Python 3.10/3.11 trên hệ điều hành được hỗ trợ.') from exc
        has_cuda = bool(paddle.device.is_compiled_with_cuda())
        if device == 'auto':
            device = 'gpu' if has_cuda else 'cpu'
        elif device == 'gpu' and not has_cuda:
            raise RuntimeError(
                'PaddlePaddle hiện tại không hỗ trợ GPU. Cài paddlepaddle-gpu '
                'hoặc dùng --paddle-device cpu.')
        config = paddle_infer.Config(str(self.model_file), str(self.param_file))
        if device == 'gpu':
            config.enable_use_gpu(500, 0)
            if hasattr(config, 'switch_ir_optim'):
                config.switch_ir_optim(True)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(4)
            if hasattr(config, 'disable_onednn'):
                config.disable_onednn()
            elif hasattr(config, 'disable_mkldnn'):
                config.disable_mkldnn()
            else:
                raise RuntimeError('PaddlePaddle thiếu API disable_onednn/disable_mkldnn. Hãy kiểm tra phiên bản Paddle và dùng Python 3.10/3.11 với paddlepaddle==2.6.2 nếu cần.')
            if hasattr(config, 'switch_ir_optim'):
                config.switch_ir_optim(False)
            print('[INFO] Paddle CPU safe mode: oneDNN=OFF, IR optimization=OFF')
        if hasattr(config, 'switch_use_feed_fetch_ops'):
            config.switch_use_feed_fetch_ops(False)
        if hasattr(config, 'disable_glog_info'):
            config.disable_glog_info()
        self.predictor = paddle_infer.create_predictor(config)
        input_names = self.predictor.get_input_names()
        output_names = self.predictor.get_output_names()
        if not input_names or not output_names:
            raise RuntimeError('Paddle model không có input/output tensor')
        self.input_handle = self.predictor.get_input_handle(input_names[0])
        self.output_handle = self.predictor.get_output_handle(output_names[0])
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        try:
            self.predict_proba(np.full((192, 256, 3), 128, dtype=np.uint8))
        except Exception as exc:
            raise RuntimeError(f"PP-Vehicle chưa suy luận được trên máy này, dù đã tắt oneDNN. Kiểm tra PaddlePaddle/Python; nếu dùng Windows, thử môi trường Python 3.10/3.11 + paddlepaddle==2.6.2. Lỗi gốc: {exc}") from exc
        print(f"[INFO] PP-Vehicle: {self.model_file} | paddle device={device}")
        print('[READY] PP-Vehicle initialized and warmed up.')

    def preprocess(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError('Ảnh xe rỗng')
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (256, 192), interpolation=cv2.INTER_LINEAR)
        norm = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        return np.transpose(norm, (2, 0, 1)).copy()[None, ...]

    def predict_attributes(self, crop_bgr):
        """19 đầu ra: 10 điểm màu + 9 điểm kiểu xe (không phải xác suất diện tích sơn)."""
        x = self.preprocess(crop_bgr)
        self.input_handle.reshape(x.shape)
        self.input_handle.copy_from_cpu(x)
        self.predictor.run()
        output = np.asarray(self.output_handle.copy_to_cpu(), dtype=np.float32).reshape(-1)
        if output.size != 19:
            raise RuntimeError(f"PP-Vehicle cần output 19 chiều; model trả {output.shape}")
        if not np.all(np.isfinite(output)):
            raise RuntimeError('PP-Vehicle sinh NaN/Inf')
        if np.any(output < -0.0001) or np.any(output > 1.0001):
            raise RuntimeError('PP-Vehicle trả điểm ngoài [0,1]; cần kiểm tra trọng số/hậu xử lý')
        return (np.clip(output[:10], 0.0, 1.0), np.clip(output[10:], 0.0, 1.0))

    def predict_proba(self, crop_bgr):
        return self.predict_attributes(crop_bgr)[0]

    def predict_type_proba(self, crop_bgr):
        return self.predict_attributes(crop_bgr)[1]


# --- BANG CHUNG THAN XE / DEN ---

def red_paint_pixels(hsv):
    """Include maroon paint and its shadows, not only vivid red lamp pixels."""
    hue, saturation, value = cv2.split(hsv)
    minimum_value = max(28., .45*float(np.median(value)))
    return ((hue <= 12) | (hue >= 160)) & (saturation >= 75) & (value >= minimum_value)


def red_lamp_candidates(hsv, include_upper=False):
    """Find compact hot red components, independently of reflected body color."""
    hue, saturation, value = cv2.split(hsv)
    h, w = value.shape
    median=float(np.median(value))
    hot_value = max(65. if median<85 else 175.,1.15*median)
    hot = ((hue <= 12) | (hue >= 170)) & (saturation >= 110) & (value >= hot_value)
    if not include_upper:
        hot[:int(.30*h)] = False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8),8)
    mask = np.zeros((h,w),np.uint8)
    for index in range(1,count):
        if 2 <= stats[index,cv2.CC_STAT_AREA] <= max(3,int(.085*h*w)):
            mask[labels == index] = 255
    return mask


def get_tail_light_mask(crop):
    """Tạo mask các đốm đèn hậu đỏ, nhỏ và sáng ở phần dưới của xe."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if min(h, w) < 22:
        return np.zeros((h, w), np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    upper = hsv[int(.06*h):max(int(.48*h), int(.06*h)+1),
                int(.12*w):max(int(.88*w), int(.12*w)+1)]
    paint_share = float(np.mean(red_paint_pixels(upper))) if upper.size else 0.0
    hot = ((H <= 12) | (H >= 170)) & (S >= 110) & (V >= 175)
    hot[:int(0.4 * h)] = False
    num, labels, stats, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8) * 255, 8)
    mask = np.zeros((h, w), np.uint8)
    area = h * w
    for i in range(1, num):
        comp_area = int(stats[i, cv2.CC_STAT_AREA])
        if 2 <= comp_area <= max(3, int(area * 0.085)):
            mask[labels == i] = 255
    frac = float(np.count_nonzero(mask) / area)
    if paint_share < .1 and .001 <= frac <= .16:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        mask[:] = 0
    return mask

def night_red_spatial_evidence(crop):
    """Đo phân bố đỏ phía TRÊN và DƯỚI của ảnh xe, chỉ là heuristic."""
    empty = dict(valid=False, tail_dominant=False, red_body_supported=False,
                 diffuse_cast_suspected=False, red_upper=0.0, red_middle=0.0,
                 red_lower=0.0, neutral_upper=0.0, bright_upper=0.0,
                 neutral_roof=0.0, red_roof=0.0, lamp_share=0.0,
                 strong_red_middle=0.0, rear_paint_share=0.0,
                 rear_paint_patch=0.0)
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 24:
        return empty
    h, w = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    red = red_paint_pixels(hsv)
    left, right = (int(0.12 * w), max(int(0.88 * w), int(0.12 * w) + 1))
    upper = red[int(0.06 * h):max(int(0.45 * h), int(0.06 * h) + 1), left:right]
    lower = red[int(0.54 * h):, left:right]
    middle = red[int(.38*h):max(int(.80*h),int(.38*h)+1),
                 int(.20*w):max(int(.80*w),int(.20*w)+1)]
    upper_sat = ss[int(0.06 * h):max(int(0.45 * h), int(0.06 * h) + 1), left:right]
    upper_v = vv[int(0.06 * h):max(int(0.45 * h), int(0.06 * h) + 1), left:right]
    red_upper = float(upper.mean()) if upper.size else 0.0
    red_lower = float(lower.mean()) if lower.size else 0.0
    red_middle = float(middle.mean()) if middle.size else 0.0
    neutral_upper = float((upper_sat <= 65).mean()) if upper_sat.size else 0.0
    bright_upper = float(((upper_sat <= 65) & (upper_v >= 145)).mean()) if upper.size else 0.0
    core = red[int(0.06 * h):, left:right].astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    largest = float(np.max(stats[1:, cv2.CC_STAT_AREA])) / (h * w) if count > 1 else 0.0
    roof_slice = (slice(int(.08*h),max(int(.30*h),int(.08*h)+1)),
                  slice(int(.20*w),max(int(.80*w),int(.20*w)+1)))
    neutral_roof = float(np.mean(ss[roof_slice] <= 65))
    red_roof = float(np.mean(red[roof_slice]))
    lamps = red_lamp_candidates(hsv)
    lamp_share = float(np.mean(lamps > 0))
    middle_slice = (slice(int(.38*h),max(int(.80*h),int(.38*h)+1)),
                    slice(int(.20*w),max(int(.80*w),int(.20*w)+1)))
    strong_red_middle = float(np.mean(
        (red & (ss >= 125) & (lamps == 0))[middle_slice]))
    rear_region = np.zeros((h, w), dtype=bool)
    rear_region[int(.38*h):max(int(.90*h), int(.38*h)+1),
                int(.12*w):max(int(.88*w), int(.12*w)+1)] = True
    excluded_lamps = cv2.dilate(lamps, np.ones((5, 5), np.uint8)) > 0
    rear_paint = cv2.morphologyEx(
        (red & rear_region & ~excluded_lamps).astype(np.uint8),
        cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    rear_size = max(1, np.count_nonzero(rear_region))
    rear_paint_share = float(np.count_nonzero(rear_paint)) / rear_size
    rear_count, _, rear_stats, _ = cv2.connectedComponentsWithStats(rear_paint, 8)
    rear_paint_patch = (float(np.max(rear_stats[1:, cv2.CC_STAT_AREA])) / rear_size
                        if rear_count > 1 else 0.0)
    side_region = rear_region.copy()
    side_region[:, int(.36*w):int(.64*w)] = False
    side_red_share = float(np.mean(rear_paint[side_region])) if np.any(side_region) else 0.0
    strong_ratio = strong_red_middle / max(red_middle, .01)
    rear_body_supported = (red_lower >= .22 and red_middle >= .16
                           and rear_paint_share >= .14 and rear_paint_patch >= .09
                           and side_red_share >= .11 and strong_red_middle >= .024
                           and strong_ratio >= .11
                           and (lamp_share <= .035 or strong_ratio >= .17))
    diffuse_cast = (not rear_body_supported and neutral_roof >= .65
                    and red_roof <= .12 and lamp_share >= .002
                    and red_middle >= .20 and strong_red_middle < .35
                    and strong_red_middle < .50*max(red_middle,.01))
    body_supported = (red_upper >= .135 or (red_lower >= .38 and largest >= .18)
                      or (red_middle >= .38 and largest >= .12)
                      or rear_body_supported)
    body_supported = body_supported and not diffuse_cast
    dominant = diffuse_cast or (not body_supported and red_lower >= 0.065 and
        red_upper <= 0.095 and red_lower >= max(0.065,2.3*red_upper) and neutral_upper >= 0.46)
    return dict(valid=True, tail_dominant=bool(dominant), red_body_supported=bool(body_supported),
                diffuse_cast_suspected=bool(diffuse_cast), red_upper=red_upper,
                red_middle=red_middle, red_lower=red_lower, neutral_upper=neutral_upper,
                bright_upper=bright_upper, neutral_roof=neutral_roof, red_roof=red_roof,
                lamp_share=lamp_share, strong_red_middle=strong_red_middle,
                rear_paint_share=rear_paint_share, rear_paint_patch=rear_paint_patch)


def red_floodlight_observation(crop):
    """Phát hiện ám đỏ LAN RỘNG từ đèn, không đơn thuần phát hiện pixel đỏ."""
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 32:
        return False
    h, w = crop.shape[:2]
    evidence = night_red_spatial_evidence(crop)
    if evidence['lamp_share'] < .012 or evidence['red_upper'] < .25:
        return False
    body = crop[int(.20*h):max(int(.82*h), int(.20*h)+1),
                int(.14*w):max(int(.86*w), int(.14*w)+1)]
    if body.size == 0:
        return False
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red_span = ((hue <= 14) | (hue >= 155)) & (saturation >= 58) & (value >= 42)
    red_coverage = float(np.mean(red_span))
    b, g, r = cv2.split(body.astype(np.float32))
    red_excess = float(np.median(r-g))
    return bool(red_coverage >= .46 and red_excess >= 27
                and evidence['red_middle'] >= .32)

def flood_neutral_observation(crop, raw_scores, classifier, args):
    """Conservative neutral-color fallback when lights are always on."""
    if classifier is None or min(crop.shape[:2]) < 48:
        return None
    original = validated_scores(raw_scores)
    raw_name = COLOR_NAMES[int(np.argmax(original))]
    if raw_name not in {'black', 'gray', 'white'} or float(original.max()) < .65:
        return None
    h, w = crop.shape[:2]
    pixels = crop.astype(np.float32)
    corrected = crop.copy()
    corrected[:, :, 2] = np.round(np.minimum(
        pixels[:, :, 2], .5*(pixels[:, :, 0]+pixels[:, :, 1]))).astype(np.uint8)
    body = corrected[int(.24*h):max(int(.88*h), int(.24*h)+1),
                     int(.18*w):max(int(.82*w), int(.18*w)+1)]
    if body.size == 0:
        return None
    base = validated_scores(classifier.predict_proba(corrected))
    partial = validated_scores(classifier.predict_proba(body))
    i = int(np.argmax(base))
    name = COLOR_NAMES[i]
    if name not in {'black', 'gray', 'white'} or name != raw_name or i != int(np.argmax(partial)):
        return None
    if min(float(base[i]), float(partial[i])) < max(.58, args.color_conf):
        return None
    if min(float(base[i] - np.max(np.delete(base, i))),
           float(partial[i] - np.max(np.delete(partial, i)))) < .14:
        return None
    bg = .5*(pixels[:, :, 0]+pixels[:, :, 1])
    center_bg = bg[int(.3*h):int(.83*h), int(.18*w):int(.82*w)]
    if name == 'black' and float(np.median(center_bg)) > 135:
        return None
    return name, float(.5*(base[i]+partial[i]))


def appearance_signature(crop):
    """Chữ ký màu sắc nhẹ để ghép lại xe khi ByteTrack đổi ID."""
    if crop is None or crop.size == 0:
        return np.zeros(18, dtype=np.float32)
    h, w = crop.shape[:2]
    roi = crop[int(0.12 * h):max(int(0.88 * h), int(0.12 * h) + 1), int(0.1 * w):max(int(0.9 * w), int(0.1 * w) + 1)]
    if roi.size == 0:
        roi = crop
    hsv = cv2.cvtColor(cv2.resize(roi, (48, 36)), cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    lamp_mask = get_tail_light_mask(crop)
    head_mask = get_headlight_mask(crop)
    lamps = cv2.bitwise_or(lamp_mask, head_mask)
    lamps = lamps[int(.12*h):max(int(.88*h),int(.12*h)+1),
                  int(.10*w):max(int(.90*w),int(.10*w)+1)]
    valid = cv2.resize(lamps, (48,36), interpolation=cv2.INTER_NEAREST) == 0
    reference = float(np.percentile(V[valid],90)) if np.any(valid) else 1.
    normalized_v = np.clip(V.astype(np.float32)*200/max(1.,reference),0,255)
    colorful = (S > 60) & (V > max(20., .4*float(np.median(V)))) & valid
    hue_hist = np.histogram(H[colorful], bins=12, range=(0, 180))[0].astype(np.float32)
    neutral = ~colorful & valid
    value_hist = np.histogram(normalized_v[neutral], bins=6, range=(0, 256))[0].astype(np.float32)
    hist = np.concatenate([hue_hist, value_hist])
    return hist / max(1e-06, float(np.linalg.norm(hist)))

def get_headlight_mask(crop):
    """Compact bright white lights at the sides; broad white paint is retained."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if min(h, w) < 24:
        return mask
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    value, saturation = hsv[:, :, 2], hsv[:, :, 1]
    hot = (value >= 245) & (saturation <= 55)
    hot[:int(.30*h)] = False
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        hot.astype(np.uint8), 8)
    area = h*w
    for index in range(1, count):
        x, y, width, height, size = stats[index]
        cx, _ = centers[index]
        if not (max(3, int(.0004*area)) <= size <= .08*area
                and height <= .30*h and width <= .45*w
                and (cx <= .40*w or cx >= .60*w)):
            continue
        pad = max(3, int(.04*min(h, w)))
        neighborhood = value[max(0,y-pad):min(h,y+height+pad),
                             max(0,x-pad):min(w,x+width+pad)]
        if float(np.percentile(neighborhood, 30)) > 220:
            continue
        mask[labels == index] = 255
    if np.any(mask):
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    return mask


def observed_color_fractions(crop):
    """Đo bằng chứng pixel cho 10 màu; tỷ lệ trên ROI, không phải diện tích sơn."""
    if crop is None or crop.size == 0:
        return {}
    h, w = crop.shape[:2]
    if min(h, w) < 20:
        return {}
    top, bottom = (int(0.16 * h), max(int(0.84 * h), int(0.16 * h) + 1))
    left, right = (int(0.12 * w), max(int(0.88 * w), int(0.12 * w) + 1))
    roi = crop[top:bottom, left:right]
    if roi.size == 0:
        return {}
    hsv = cv2.cvtColor(cv2.resize(roi, (90, 60), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    colorful = (S >= 60) & (V >= 55)
    neutral = S < 60
    warm = colorful & (H >= 10) & (H < 35)
    n = float(hsv.shape[0] * hsv.shape[1])
    masks = {'red': colorful & ((H < 10) | (H >= 160)), 'orange': warm & (H < 20) & (V >= 145), 'yellow': warm & (H >= 20) & (V >= 180), 'golden': warm & (H >= 20) & (V >= 90) & (V < 180), 'brown': warm & ((H < 20) & (V < 145) | (H >= 20) & (V < 90)), 'green': colorful & (H >= 35) & (H < 85), 'blue': colorful & (H >= 85) & (H < 138), 'white': neutral & (V >= 190), 'gray': neutral & (V >= 55) & (V < 190), 'black': neutral & (V < 55)}
    lamp_mask = get_tail_light_mask(crop)
    if lamp_mask is not None:
        lamps = cv2.resize(lamp_mask[top:bottom, left:right], (90, 60), interpolation=cv2.INTER_NEAREST) > 0
        masks['red'] &= ~lamps
    headlights = get_headlight_mask(crop)
    if headlights is not None:
        glare = cv2.resize(headlights[top:bottom, left:right], (90,60),
                           interpolation=cv2.INTER_NEAREST) > 0
        masks['white'] &= ~glare
    kernel = np.ones((3, 3), dtype=np.uint8)
    return {name: float(np.count_nonzero(cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel))) / n for name, mask in masks.items()}


def dominant_body_paint(crop):
    """Ứng viên màu sơn phủ phần thân giữa/dưới, không lấy kính/nóc làm sơn."""
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 36:
        return None
    h, w = crop.shape[:2]
    top, bot = int(.36*h), max(int(.90*h), int(.36*h)+1)
    left, right = int(.12*w), max(int(.88*w), int(.12*w)+1)
    roi = crop[top:bot, left:right]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(cv2.resize(roi, (120,96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    chromatic = (S >= 48) & (V >= 42) & (V <= 248)
    warm = chromatic & (H >= 12) & (H < 36)
    masks = {
        'red': chromatic & ((H <= 14) | (H >= 150)),
        'orange': warm & (H < 20) & (V >= 125),
        'yellow': warm & (H >= 20) & (V >= 165),
        'golden': warm & (H >= 20) & (V >= 100) & (V < 165),
        'brown': warm & (((H < 20) & (V < 125)) | ((H >= 20) & (V < 100))),
        'green': chromatic & (H >= 36) & (H < 85),
        'blue': chromatic & (H >= 85) & (H < 145),
    }
    original_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    light_mask = cv2.bitwise_or(red_lamp_candidates(original_hsv), get_headlight_mask(crop))
    part = light_mask[top:bot,left:right]
    if part.size:
        invalid = cv2.resize(part,(120,96),interpolation=cv2.INTER_NEAREST)>0
        for name in masks:
            masks[name] &= ~invalid
    kernel = np.ones((3,3),np.uint8)
    strengths = {}
    for name,mask in masks.items():
        smooth = cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,kernel)
        n,_,stats,_=cv2.connectedComponentsWithStats(smooth,8)
        largest = float(np.max(stats[1:,cv2.CC_STAT_AREA]))/smooth.size if n>1 else 0.
        share = float(np.mean(smooth))
        strengths[name]=(share,largest)
    order = sorted(strengths,key=lambda n:strengths[n][0],reverse=True)
    name=order[0]
    share,patch = strengths[name]
    other=max((strengths[n][0] for n in order[1:]),default=0.)
    if share < .43 or patch < .105 or share-other < .20:
        return None
    if name=='red':
        e=night_red_spatial_evidence(crop)
        if e['tail_dominant'] and not e['red_body_supported']:
            return None
    return {'name':name,'share':share,'patch':patch, 'runner':other}


def strong_neutral_body_paint(crop):
    """White-paint corroboration from mid/lower body, avoiding dark roof/glass."""
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 44:
        return None
    h, w = crop.shape[:2]
    roi = crop[int(.36*h):max(int(.90*h), int(.36*h)+1),
               int(.12*w):max(int(.88*w), int(.12*w)+1)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(cv2.resize(roi, (120, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    chromatic = (sat >= 53) & (val >= 40)
    white = (sat <= 45) & (val >= 172)
    # Sparse bright patches and rear lamps are never sufficient to identify white.
    headlight_mask = get_headlight_mask(crop)
    if headlight_mask is not None:
        piece = headlight_mask[int(.36*h):max(int(.90*h),int(.36*h)+1),
                               int(.12*w):max(int(.88*w),int(.12*w)+1)]
        if piece.size:
            white &= cv2.resize(piece, (120,96), interpolation=cv2.INTER_NEAREST) == 0
    cleaned = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    largest = float(np.max(stats[1:, cv2.CC_STAT_AREA])) / cleaned.size if count > 1 else 0.
    neutral = float(np.mean(sat <= 55))
    share = float(cleaned.mean())
    value_80 = float(np.percentile(val, 80))
    # Require a broad visible bright surface, not just a highlight.
    if (share >= .32 and largest >= .11 and neutral >= .73
            and float(chromatic.mean()) < .22 and value_80 >= 207):
        return {'name': 'white', 'share': share, 'patch': largest}
    return None


def spatial_paint_candidates(crop, raw_scores=None, vehicle_class=2,
                             min_share=.08, min_model_score=.65):
    """Find two separate paint regions in original pixels, excluding glass/lights."""
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 44:
        return None
    # A flooded red frame is not adequate evidence for a second paint layer.
    if red_floodlight_observation(crop):
        return None
    image = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, value = cv2.split(hsv)
    bus = vehicle_class == 5
    body = np.zeros(value.shape, bool)
    body[88:138, 32:128] = True
    if bus:
        body[58:140, 18:142] = True
    chroma_region = body.copy()
    if bus:
        chroma_region[18:140, 18:142] = True
    lamps = red_lamp_candidates(hsv, include_upper=bus) | get_headlight_mask(image)
    lamps = cv2.dilate(lamps, np.ones((3, 3), np.uint8)) > 0
    valid = ~lamps & (value >= 28) & (value < 246)
    chromatic = valid & chroma_region & (sat >= 65) & (value >= 42)
    warm = chromatic & (hue >= 12) & (hue < 36)
    masks = {
        'red': chromatic & ((hue < 12) | (hue >= 150)),
        'orange': warm & (hue < 20) & (value >= 135),
        'yellow': warm & (hue >= 20) & (value >= 165),
        'golden': warm & (hue >= 20) & (value >= 100) & (value < 165),
        'brown': warm & (((hue < 20) & (value < 135)) |
                         ((hue >= 20) & (value < 100))),
        'green': chromatic & (hue >= 36) & (hue < 85),
        'blue': chromatic & (hue >= 85) & (hue < 145),
    }
    kernel = np.ones((3, 3), np.uint8)
    strengths = {}
    def surface(mask, region):
        clean = cv2.morphologyEx((mask & region).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
        size = max(1, np.count_nonzero(region))
        share = float(clean.sum()) / size
        largest = float(np.max(stats[1:, cv2.CC_STAT_AREA])) / size if count > 1 else 0.
        return share, largest, clean
    for name, mask in masks.items():
        share, largest, clean = surface(mask, chroma_region)
        if share >= (.035 if bus else .16) and largest >= (.018 if bus else .075):
            strengths[name] = (share, largest)
    red_info = night_red_spatial_evidence(crop)
    if not bus and not red_info['red_body_supported']:
        strengths.pop('red', None)
    neutral = valid & body & (sat <= (60 if bus else 32))
    if not bus:
        for low, high in ((28, 125), (125, 195), (195, 246)):
            full = ((sat <= 32) & (value >= low) & (value < high)).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(full, 8)
            for label in range(1, count):
                x, y, width, height, _ = stats[label]
                edges = (int(x == 0) + int(y == 0)
                         + int(x + width == 160) + int(y + height == 160))
                if edges >= 2:
                    neutral[labels == label] = False
    share, patch, clean = surface(neutral, body)
    if share >= (.38 if bus else .24) and patch >= (.20 if bus else .14):
        values = value[clean > 0]
        middle = float(np.median(values))
        neutral_name = 'black' if middle < 125 else ('white' if middle >= 195 else 'gray')
        if raw_scores is not None:
            scores = validated_scores(raw_scores)
            top = COLOR_NAMES[int(np.argmax(scores))]
            if top in {'black', 'gray', 'white'} and scores.max() >= .75:
                neutral_name = top
        strengths[neutral_name] = (share, patch)
    ordered = sorted(strengths, key=lambda name: strengths[name][0], reverse=True)
    if len(ordered) < 2:
        return None
    first, second = ordered[:2]
    neutral_names = {'black', 'gray', 'white'}
    warm_names = {'yellow', 'orange', 'golden', 'brown'}
    if ({first, second} <= neutral_names or {first, second} <= warm_names):
        return None
    if not bus and ({first, second} & neutral_names):
        neutral_name = next(iter({first, second} & neutral_names))
        lower_panel = body.copy()
        lower_panel[:112] = False
        lower_share, lower_patch, _ = surface(neutral, lower_panel)
        if lower_share < .17 or lower_patch < .10:
            return None
        dominant = dominant_body_paint(crop)
        if dominant and dominant['name'] in {first, second} and dominant['share'] >= .43:
            if (strengths[neutral_name][0] < .42 or
                    strengths[neutral_name][1] < .32):
                return None
    total = strengths[first][0] + strengths[second][0]
    secondary_ratio = strengths[second][0] / max(total, 1e-6)
    required_share = min(min_share, .05) if bus else max(min_share, .20)
    required_patch = .018 if bus else .085
    if secondary_ratio < required_share or strengths[second][1] < required_patch:
        return None
    if raw_scores is not None:
        scores = validated_scores(raw_scores)
        if max(float(scores[COLOR_NAMES.index(name)]) for name in (first, second)) < min_model_score:
            return None
    return first, second


def white_gray_evidence(crop):
    """Tín hiệu phụ hỗ trợ phân biệt trắng/xám khi chính model đang phân vân."""
    if crop is None or crop.size == 0:
        return (0.0, 0.0, 0.0)
    h, w = crop.shape[:2]
    if min(h, w) < 18:
        return (0.0, 0.0, 0.0)
    roi = crop[int(0.08 * h):max(int(0.74 * h), int(0.08 * h) + 1), int(0.08 * w):max(int(0.92 * w), int(0.08 * w) + 1)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S, V = (hsv[:, :, 1], hsv[:, :, 2])
    neutral = S < 60
    bright = neutral & (V > 185)
    return (float(np.mean(bright)), float(np.mean(neutral)), float(np.percentile(V[neutral], 75)) if neutral.sum() > 20 else 0.0)

def warm_chroma_evidence(crop):
    """Warm hue support remains measurable when illumination drops."""
    h, w = crop.shape[:2]
    roi = crop[int(.17*h):max(int(.80*h),int(.17*h)+1),
               int(.13*w):max(int(.87*w),int(.13*w)+1)]
    hue, saturation, value = cv2.split(cv2.cvtColor(roi,cv2.COLOR_BGR2HSV))
    return float(np.mean((hue >= 12) & (hue <= 35) & (saturation >= 60) & (value >= 28)))


def body_chroma_support(crop):
    """Hue evidence on the lower body, with relative exposure and lamps excluded."""
    h,w = crop.shape[:2]
    top,bottom = int(.35*h),max(int(.92*h),int(.35*h)+1)
    left,right = int(.12*w),max(int(.88*w),int(.12*w)+1)
    hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    region = hsv[top:bottom,left:right]
    hue,saturation,value = cv2.split(region)
    chromatic = (saturation >= 65) & (value >= max(28.,.40*float(np.median(value))))
    valid = red_lamp_candidates(hsv)[top:bottom,left:right] == 0
    valid &= get_headlight_mask(crop)[top:bottom,left:right] == 0
    chromatic &= valid
    masks = {
        'red':(hue <= 12)|(hue >= 160),
        'blue':(hue >= 85)&(hue < 140),
        'green':(hue >= 35)&(hue < 85),
        'warm':(hue >= 8)&(hue < 35),
    }
    result = {name:float(np.mean(mask & chromatic)) for name,mask in masks.items()}
    for name in ('yellow','orange','golden','brown'):result[name]=result['warm']
    return result


# --- DANH GIA ANH SANG ---

@dataclass(frozen=True)
class LightingContext:
    median_value: float
    contrast: float
    clipped_share: float
    headlight_share: float
    weight: float


@dataclass(frozen=True)
class ColorAssessment:
    scores: np.ndarray
    status: str
    context: LightingContext
    blocked_colors: frozenset
    evidence_crop: np.ndarray
    red_body_supported: bool = False
    paint_verified: bool = False


def validated_scores(values):
    scores = np.asarray(values, dtype=np.float32).reshape(-1)
    if scores.shape != (10,) or not np.isfinite(scores).all():
        raise ValueError('Bộ phân loại phải trả 10 điểm màu hữu hạn')
    if np.any(scores < -1e-4) or np.any(scores > 1.0001):
        raise ValueError('Điểm màu phải nằm trong [0,1]')
    return np.clip(scores, 0., 1.)


def lighting_context(crop):
    """Measure visibility on the crop, independent of hour and weather names."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    value, saturation = hsv[:, :, 2], hsv[:, :, 1]
    h, w = value.shape
    body = value[int(.12*h):max(int(.82*h), int(.12*h)+1),
                 int(.12*w):max(int(.88*w), int(.12*w)+1)]
    median = float(np.median(body))
    low, high = np.percentile(body, [10,90])
    clipped = float(np.mean((value >= 250) & (saturation <= 30)))
    mask = get_headlight_mask(crop)
    lamps = float(np.count_nonzero(mask)/value.size)
    contrast = float(high-low)
    weight = max(.25, 1. - .45*(median < 65) - .20*(contrast < 28)
                 - .40*clipped - .6*lamps)
    return LightingContext(median, contrast, clipped, lamps, weight)


@lru_cache(maxsize=3)
def gamma_lookup(gamma):
    return np.round(255 * (np.arange(256) / 255.) ** gamma).astype(np.uint8)


@lru_cache(maxsize=1)
def visibility_clahe():
    return cv2.createCLAHE(clipLimit=2., tileGridSize=(4, 4))


def enhance_visibility(crop):
    """Raise value/contrast while keeping HSV hue and saturation channels."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    median = float(np.median(value))
    gamma = .65 if median < 60 else (.75 if median < 85 else 1.)
    table = gamma_lookup(gamma)
    adjusted = cv2.LUT(value, table)
    local = visibility_clahe().apply(adjusted)
    hsv[:, :, 2] = cv2.addWeighted(adjusted, .70, local, .30, 0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def prepare_detection_frame(frame):
    """Enhance only genuinely dark scenes; output coordinates are unchanged."""
    sample = cv2.resize(frame, (160,90), interpolation=cv2.INTER_AREA)
    value = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)[:, :, 2]
    return enhance_visibility(frame) if np.median(value) < 65 else frame

def reduce_diffuse_red_cast(crop, evidence):
    """Remove excess R only for a measured lamp cast; do not brighten dark paint."""
    neutral_tail = (evidence.get('tail_dominant',False)
                    and evidence.get('neutral_roof',0.) >= .75
                    and evidence.get('strong_red_middle',1.) < .28)
    if not (evidence.get('diffuse_cast_suspected',False) or neutral_tail):
        return None
    pixels = crop.astype(np.float32)
    corrected = crop.copy()
    corrected[:,:,2] = np.round(np.minimum(
        pixels[:,:,2],.5*(pixels[:,:,0]+pixels[:,:,1]))).astype(np.uint8)
    return corrected


def diffuse_cast_scores(crop, classifier, args, evidence):
    """Both corrected views must agree on a neutral color; never force black."""
    corrected = reduce_diffuse_red_cast(crop,evidence)
    if corrected is None:
        return None
    h, w = crop.shape[:2]
    inner = corrected[int(.04*h):max(int(.94*h),int(.04*h)+1),
                      int(.04*w):max(int(.96*w),int(.04*w)+1)]
    full_scores = validated_scores(classifier.predict_proba(corrected))
    inner_scores = validated_scores(classifier.predict_proba(inner))
    index = int(np.argmax(full_scores))
    candidate = COLOR_NAMES[index]
    consensus = .70*full_scores + .30*inner_scores
    runners = np.delete(consensus,index)
    if (candidate in {'black','gray','white'} and index == int(np.argmax(inner_scores))
            and full_scores[index] >= max(.50,args.color_conf,args.night_alt_conf)
            and (inner_scores[index] >= max(.55,args.color_conf,args.night_alt_conf)
                 or (full_scores[index] >= .70 and inner_scores[index] >= .25))
            and consensus[index] >= max(.58,args.color_conf,args.night_alt_conf)
            and consensus[index]-float(np.max(runners)) >= .10):
        consensus[COLOR_NAMES.index('red')] = 0.
        return consensus
    neutral = {'black','gray','white'}
    first,second = candidate,COLOR_NAMES[int(np.argmax(inner_scores))]
    conflicting_neutrals = first in neutral and second in neutral and first != second
    check_padding = (not conflicting_neutrals and
                     ((first in neutral) != (second in neutral)
                      or (first == second and first in neutral
                          and max(full_scores.max(),inner_scores.max()) < .70))
                     and max(full_scores.max(),inner_scores.max()) >= .48
                     and max(full_scores.max()-float(np.partition(full_scores,-2)[-2]),
                             inner_scores.max()-float(np.partition(inner_scores,-2)[-2])) >= .10)
    if check_padding:
        views = [full_scores,inner_scores]
        for margin in (.02,.06):
            view = corrected[int(margin*h):max(int((1-margin)*h),int(margin*h)+1),
                             int(margin*w):max(int((1-margin)*w),int(margin*w)+1)]
            views.append(validated_scores(classifier.predict_proba(view)))
        votes = {}
        for values in views:
            j = int(np.argmax(values));name=COLOR_NAMES[j]
            runner = float(np.max(np.delete(values,j)))
            if (name in neutral and values[j] >= max(.48,args.color_conf,args.night_alt_conf)
                    and values[j]-runner >= .10):
                votes.setdefault(name,[]).append(values)
        if len(votes)==1:
            supporting = next(iter(votes.values()))
            if len(supporting)>=2:
                result=np.mean(supporting,axis=0).astype(np.float32)
                result[COLOR_NAMES.index('red')]=0.
                return result
    return None


def night_color_scores(crop, scores, classifier, args, allow_alternate=True):
    """Kiểm tra màu đỏ trên thân/đèn trước voting, cả ban ngày và buổi tối."""
    scores = np.asarray(scores, dtype=np.float32).copy()
    ridx = COLOR_NAMES.index('red')
    evidence = night_red_spatial_evidence(crop)
    tail = evidence['tail_dominant']
    best = int(np.argmax(scores))
    if best != ridx:
        if tail:
            scores[ridx] = 0.0
            return (scores, 'tail_nonred')
        if evidence['valid'] and evidence['red_body_supported']:
            return (scores, 'red_body_supported')
        scores[ridx] = 0.0
        return (scores, 'normal')
    if evidence['valid'] and evidence['red_body_supported'] and (not tail):
        return (scores, 'red_body_supported')
    if not tail or classifier is None or (not allow_alternate):
        return (np.zeros(10, dtype=np.float32), 'red_uncertain')
    if reduce_diffuse_red_cast(crop,evidence) is not None:
        corrected_scores = diffuse_cast_scores(crop,classifier,args,evidence)
        if corrected_scores is not None:
            return (corrected_scores,'reflection_corrected')
        return (np.zeros(10,dtype=np.float32),'red_uncertain')
    mask = get_tail_light_mask(crop)
    if mask is None or not np.any(mask):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hh, ss, vv = cv2.split(hsv)
        lamp = ((hh <= 12) | (hh >= 170)) & (ss >= 85) & (vv >= 130)
        lamp[:int(0.5 * crop.shape[0])] = False
        mask = lamp.astype(np.uint8) * 255
    if mask is None:
        return (np.zeros(10, dtype=np.float32), 'red_uncertain')
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    masked = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
    h, w = crop.shape[:2]
    body_crop = crop[int(0.06 * h):max(int(0.49 * h), int(0.06 * h) + 1), int(0.08 * w):max(int(0.92 * w), int(0.08 * w) + 1)]
    if body_crop.size == 0:
        return (np.zeros(10, dtype=np.float32), 'red_uncertain')
    a = np.asarray(classifier.predict_proba(masked), dtype=np.float32)
    b = np.asarray(classifier.predict_proba(body_crop), dtype=np.float32)
    if a.shape != (10,) or b.shape != (10,) or (not (np.isfinite(a).all() and np.isfinite(b).all())):
        return (np.zeros(10, dtype=np.float32), 'red_uncertain')
    consensus = 0.55 * a + 0.45 * b
    consensus[ridx] = 0.0
    j = int(np.argmax(consensus))
    support = float(consensus[j])
    candidate = COLOR_NAMES[j]
    paint_support = body_chroma_support(crop)
    supported = (candidate in {'white','gray','black'}
                 or paint_support.get(candidate,0.) >= .12)
    if (support >= max(.60,args.night_alt_conf) and supported
            and j == int(np.argmax(a)) and j == int(np.argmax(b))):
        return (np.clip(consensus, 0.0, 1.0).astype(np.float32), 'alternate_supported')
    return (np.zeros(10, dtype=np.float32), 'red_uncertain')


def classify_color_observation(crop, raw_scores, classifier, args, allow_alternate=True):
    """Use measured light conditions, not a day/night switch, to validate paint."""
    scores = validated_scores(raw_scores)
    context = lighting_context(crop)
    evidence_crop = crop
    status = None
    blocked = set()
    if context.clipped_share > .70:
        return ColorAssessment(np.zeros(10,np.float32), 'overexposed', context,
                               frozenset(), crop)
    if context.median_value < 28:
        return ColorAssessment(np.zeros(10,np.float32), 'underexposed', context,
                               frozenset(), crop)

    main = COLOR_NAMES[int(np.argmax(scores))]
    weak = float(np.max(scores)) < args.color_conf
    raw_pixels = observed_color_fractions(crop)
    colorful = sum(raw_pixels.get(name,0.) for name in
                   ('red','blue','green','yellow','orange','golden','brown'))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    shadow_color = float(np.mean((hsv[:,:,1] >= 80) & (hsv[:,:,2] >= 28)))
    poor_visibility = context.median_value < 85 or (
        context.median_value < 150 and context.contrast < 32)
    warm_names = {'yellow','orange','golden','brown'}
    neutral_share = white_gray_evidence(crop)[1]
    paint_support = body_chroma_support(crop)
    red_evidence = night_red_spatial_evidence(crop)
    warm_mismatch = (main in warm_names
                     and neutral_share >= .65 and warm_chroma_evidence(crop) < .06)
    if warm_mismatch and not poor_visibility:
        blocked.update(warm_names)
        if allow_alternate and classifier is not None:
            corrected = diffuse_cast_scores(crop,classifier,args,red_evidence)
            if corrected is not None:
                return ColorAssessment(corrected,'reflection_corrected',context,
                                       frozenset(blocked|{'red'}),crop)
        return ColorAssessment(np.zeros(10,np.float32),'warm_uncertain',context,
                               frozenset(blocked),crop)
    if warm_mismatch:
        blocked.update(warm_names)
        if not allow_alternate:
            return ColorAssessment(np.zeros(10,np.float32), 'warm_uncertain',
                                   context, frozenset(blocked), crop)
    suspect_dark_color = (main == 'black' or (main == 'red'
                          and not red_evidence['red_body_supported']))
    if (allow_alternate and context.median_value >= 28 and poor_visibility
            and not red_evidence['diffuse_cast_suspected']
            and (weak or warm_mismatch
                 or (suspect_dark_color and max(colorful, shadow_color) >= .30))):
        enhanced = enhance_visibility(crop)
        enhanced_scores = validated_scores(classifier.predict_proba(enhanced))
        candidate = COLOR_NAMES[int(np.argmax(enhanced_scores))]
        pixels = observed_color_fractions(enhanced)
        supported = pixels.get(candidate,0.) >= .18
        if candidate in {'black','white','gray'}:
            supported = (white_gray_evidence(enhanced)[1] >= .65
                         and sum(pixels.get(name,0.) for name in
                                 ('red','blue','green','yellow','orange','golden','brown')) < .20)
        if warm_mismatch:
            supported = supported and candidate in {'black','white','gray'}
        if enhanced_scores.max() >= .65 and supported:
            if weak or warm_mismatch or (main == 'black' and candidate not in {'black','white','gray'}):
                scores = enhanced_scores
            else:
                scores = .35*scores + .65*enhanced_scores
            evidence_crop = enhanced
            status = 'low_light_enhanced'
        elif warm_mismatch:
            return ColorAssessment(np.zeros(10,np.float32), 'warm_uncertain',
                                   context, frozenset(blocked), crop)

    main = COLOR_NAMES[int(np.argmax(scores))]
    if (main in {'blue','green'} and float(scores.max()) >= args.color_conf
            and paint_support.get(main,0.) < .12):
        return ColorAssessment(np.zeros(10,np.float32),'paint_uncertain',context,
                               frozenset(blocked|{main}),crop)
    bright_body, _, _ = white_gray_evidence(crop)
    suspicious_white = (main == 'white' and context.headlight_share >= .003
                        and bright_body < .24)
    if suspicious_white:
        blocked.add('white')
        if not allow_alternate:
            return ColorAssessment(np.zeros(10,np.float32), 'headlight_uncertain',
                                   context, frozenset(blocked), crop)
        mask = get_headlight_mask(evidence_crop)
        masked = cv2.inpaint(evidence_crop,mask,3,cv2.INPAINT_TELEA)
        h, w = crop.shape[:2]
        body = masked[:max(1,int(.65*h)), int(.08*w):max(int(.92*w),int(.08*w)+1)]
        full_scores = validated_scores(classifier.predict_proba(masked))
        body_scores = validated_scores(classifier.predict_proba(body))
        index = int(np.argmax(full_scores))
        candidate = COLOR_NAMES[index]
        pixels = observed_color_fractions(crop)
        neutral_share = sum(pixels.get(name,0.) for name in ('black','white','gray'))
        chromatic_share = sum(pixels.get(name,0.) for name in
                              ('blue','green','yellow','orange','golden','brown'))
        pixel_support = (neutral_share >= .65 and chromatic_share < .20
                         if candidate in {'black','white','gray'}
                         else pixels.get(candidate,0.) >= .15)
        if (index == int(np.argmax(body_scores)) and pixel_support
                and full_scores[index] >= .60 and body_scores[index] >= .45):
            scores = .65*full_scores + .35*body_scores
            if candidate == 'white':
                blocked.discard('white')
                status = 'headlights_body_supported'
            else:
                scores[COLOR_NAMES.index('white')] = 0.
                status = 'headlights_corrected'
        else:
            return ColorAssessment(np.zeros(10,np.float32), 'headlight_uncertain',
                                   context, frozenset(blocked), crop)

    scores, red_status = night_color_scores(
        evidence_crop, scores, classifier, args, allow_alternate)
    if red_status in {'red_uncertain','tail_nonred','alternate_supported','reflection_corrected'}:
        blocked.add('red')
        status = red_status
    else:
        status = status or red_status
    return ColorAssessment(validated_scores(scores), status, context,
                           frozenset(blocked), evidence_crop,
                           red_status == 'red_body_supported',
                           main in {'red','blue','green','yellow','orange','golden','brown'}
                           and paint_support.get(main,0.) >= .20
                           and (main != 'red' or red_evidence['lamp_share'] < .002
                                or red_evidence['strong_red_middle'] >= .35))


# --- CANH BAO DOI ANH SANG TOAN CANH (KHONG THAY MODEL MAU) ---

class SceneLightShiftGuard:
    """Detect a *new*, widespread blue/green cast relative to this video's"""

    def __init__(self):
        self.reference = None
        self.consecutive = 0
        self.active = False

    @staticmethod
    def _measure(frame):
        small = cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hh, ss, vv = cv2.split(hsv)
        chroma = ((hh >= 35) & (hh < 140) & (ss >= 58)
                  & (vv >= 40) & (vv < 250))
        return np.array([
            [float(np.mean(chroma[y*36:(y+1)*36, x*48:(x+1)*48]))
             for x in range(4)] for y in range(3)
        ], dtype=np.float32)

    def update(self, frame):
        current = self._measure(frame)
        if self.reference is None:
            self.reference = current.copy()
            return False
        delta = current - self.reference
        elevated = (current >= .33) & (delta >= .25)
        adjacent = bool(np.any(elevated[:, :-1] & elevated[:, 1:]) or
                        np.any(elevated[:-1, :] & elevated[1:, :]))
        shifted = adjacent and float(np.mean(delta)) >= .055
        self.consecutive = min(8, self.consecutive + 1) if shifted else max(0, self.consecutive - 1)
        self.active = self.consecutive >= 2 or (self.active and self.consecutive > 0)
        if not self.active and not shifted:
            self.reference = .995 * self.reference + .005 * current
        return self.active


# --- LICH SU MAU THEO ID ---

class TrackColorState:
    """Một luồng bỏ phiếu cho cả một/hai màu, cập nhật bằng mọi quan sát mới."""

    def __init__(self, track_id, args):
        self.id = track_id
        self.vehicle_class = 2
        self.suv_type_votes = deque(maxlen=5)
        self.bus_type_votes = deque(maxlen=6)
        self.verified_suv = False
        self.votes = deque(maxlen=max(7, args.vote_window))
        self.color = 'CHƯA RÕ'
        self.status = 'unverified'
        self.label_confirmed = False
        self.label_paint_verified = False
        self.paint_pair_confirmed = False
        self.pair_red_refutations = deque(maxlen=max(24, 2 * args.switch_hits_neutral))
        self.challenger = None
        self.challenger_hits = 0
        self.observation_attempts = 0
        self.last_frame = None

    def _commit(self, colors, verified):
        self.color = ' và '.join(COLOR_VI[name] for name in colors)
        self.label_confirmed = True
        self.label_paint_verified = verified
        self.paint_pair_confirmed = len(colors) == 2
        self.pair_red_refutations.clear()
        self.challenger = None
        self.challenger_hits = 0

    def _consider_red_pair_refutation(self, evidence, context,
                                     neutral_scores, args):
        """Drop a red second color only after repeated non-paint observations."""
        old = frozenset(self.color.split(' và ')) if self.label_confirmed else frozenset()
        neutral_survivor = old - {COLOR_VI['red']}
        neutral_labels = {COLOR_VI[name] for name in ('black', 'gray', 'white')}
        if (not self.paint_pair_confirmed or self.vehicle_class == 5
                or COLOR_VI['red'] not in old or len(neutral_survivor) != 1
                or not neutral_survivor <= neutral_labels):
            self.pair_red_refutations.clear()
            return False
        if evidence['red_body_supported']:
            self.pair_red_refutations.clear()
            return False
        no_red_paint = (context.weight >= .60 and context.median_value >= 60
                        and context.clipped_share < .10
                        and evidence['diffuse_cast_suspected']
                        and not evidence['red_body_supported'])
        neutral_conf = 0.0
        if no_red_paint and neutral_scores is not None:
            candidate = validated_scores(neutral_scores)
            name = COLOR_NAMES[int(np.argmax(candidate))]
            if name in {'black', 'gray', 'white'}:
                neutral_conf = float(candidate.max())
        self.pair_red_refutations.append((bool(no_red_paint), neutral_conf))
        needed = max(12, args.switch_hits_neutral)
        recent = list(self.pair_red_refutations)
        if (len(recent) < needed
                or sum(no_red for no_red, _ in recent) < needed
                or sum(score >= max(.58, args.color_conf) for _, score in recent) < 2
                or not all(no_red for no_red, _ in recent[-3:])):
            return False
        survivor = next(iter(neutral_survivor))
        survivor_name = next(name for name in ('black', 'gray', 'white')
                             if COLOR_VI[name] == survivor)
        self._commit((survivor_name,), False)
        self.votes.clear()
        self.status = 'pair_red_refuted'
        return True

    def update(self, crop, classifier, frame_number, args):
        self.observation_attempts += 1
        if self.last_frame is not None and frame_number-self.last_frame > max(3, round(getattr(args, 'video_fps', 25.))):
            self.votes.clear()
            self.pair_red_refutations.clear()
            self.challenger = None
            self.challenger_hits = 0
        self.last_frame = frame_number
        if hasattr(classifier, 'predict_attributes'):
            raw, vehicle_types = classifier.predict_attributes(crop)
            vehicle_types = np.asarray(vehicle_types, dtype=np.float32)
            if vehicle_types.shape != (9,) or not np.isfinite(vehicle_types).all():
                raise ValueError('Bộ phân loại phải trả 9 điểm kiểu xe hữu hạn')
            suv_index = VEHICLE_TYPE_NAMES.index('suv')
            bus_index = VEHICLE_TYPE_NAMES.index('bus')
            self.suv_type_votes.append(int(np.argmax(vehicle_types)) == suv_index
                                       and vehicle_types[suv_index] >= .80)
            self.bus_type_votes.append(int(np.argmax(vehicle_types)) == bus_index
                                       and vehicle_types[bus_index] >= .70)
            if (len(self.suv_type_votes) >= 3
                    and sum(self.suv_type_votes) >= .80*len(self.suv_type_votes)):
                self.verified_suv = True
            if self.verified_suv:
                self.vehicle_class = 2
            elif (len(self.bus_type_votes) >= 4 and
                  sum(self.bus_type_votes) >= .75*len(self.bus_type_votes)):
                self.vehicle_class = 5
        else:
            raw = classifier.predict_proba(crop)
        raw = validated_scores(raw)
        if red_floodlight_observation(crop):
            if self.paint_pair_confirmed and self.vehicle_class != 5:
                evidence = night_red_spatial_evidence(crop)
                context = lighting_context(crop)
                corrected = (diffuse_cast_scores(crop, classifier, args, evidence)
                             if evidence['diffuse_cast_suspected'] else None)
                if self._consider_red_pair_refutation(
                        evidence, context, corrected, args):
                    return
            self.votes.append(None)
            if not self.label_confirmed:
                rescue = flood_neutral_observation(crop, raw, classifier, args)
                if rescue is not None:
                    name, conf = rescue
                    self.votes.pop()
                    self.votes.append((frozenset((name,)), .70, conf, False))
                    key = frozenset((name,))
                    matching = [v for v in self.votes if v is not None and v[0] == key]
                    total = sum(v[1] if v is not None else .60 for v in self.votes)
                    if (len(matching) >= args.min_color_samples
                            and sum(v[1] for v in matching) / max(total, 1e-6) >= .70):
                        self._commit((name,), False)
                        self.status = 'low_light_neutral_confirmed'
                        return
            self.status = 'held_red_floodlight' if self.label_confirmed else 'red_floodlight_uncertain'
            return
        allow_alternate = (self.observation_attempts-1) % args.night_check_every == 0
        assessment = classify_color_observation(crop, raw, classifier, args, allow_alternate)
        context = assessment.context
        scores = assessment.scores
        self.status = assessment.status
        primary = COLOR_NAMES[int(np.argmax(scores))] if scores.max() else None
        confidence = float(scores.max())
        colors = (primary,) if primary and confidence >= args.color_conf else None
        verified = assessment.paint_verified
        body = dominant_body_paint(crop)
        white_body = strong_neutral_body_paint(crop)
        clear = (context.weight >= .65 and context.median_value >= 60
                 and context.clipped_share < .10 and context.headlight_share < .09)
        if body and clear and body['name'] not in assessment.blocked_colors:
            name = body['name'];index = COLOR_NAMES.index(name)
            if body['share'] >= .43 and raw[index] >= max(.65, args.color_conf):
                colors = (name,)
                confidence = float(raw[index])
                verified = (body['share'] >= .48 and
                            (name != 'red' or assessment.red_body_supported))
                self.status = 'body_model_supported'
        white_index = COLOR_NAMES.index('white')
        white_supported = (args.white_gray_assist and white_body is not None and clear
                            and (raw[white_index] >= .20 and white_index in np.argsort(raw)[-2:]
                                 or (white_body['share'] >= .37 and white_body['patch'] >= .20
                                     and raw[COLOR_NAMES.index('gray')] >= .48
                                     and raw[white_index] >= .10)))
        if white_supported:
            colors = ('white',)
            confidence = float(raw[white_index])
            verified = bool(raw[white_index] >= .20)
            self.status = 'white_body_supported'
        pair = None
        if args.two_colors and clear and context.clipped_share < .08:
            minimum_pair_model_score = (min(args.two_color_conf, .40)
                                        if self.vehicle_class == 5
                                        else args.two_color_conf)
            pair = spatial_paint_candidates(crop, raw, self.vehicle_class,
                                            args.second_share, minimum_pair_model_score)
        if pair is not None and not red_floodlight_observation(crop):
            denied = set(pair) & set(assessment.blocked_colors)
            separate_paint = (
                self.vehicle_class == 5
                and denied == {'red'}
                and assessment.status == 'red_uncertain'
                and 'red' in pair
                and body_chroma_support(crop).get('red', 0.0) >= .065
                and night_red_spatial_evidence(crop)['lamp_share'] < .045
            )
            if not denied or separate_paint:
                colors = pair
                confidence = max(float(raw[COLOR_NAMES.index(name)]) for name in pair)
                verified = True
                self.status = 'two_paint_regions' if not denied else 'two_paint_lamp_recovery'
        if (colors is not None and self.label_confirmed
                and getattr(args, 'scene_green_shift', False)):
            previous = set(self.color.split(' và '))
            suggested = {COLOR_VI[name] for name in colors}
            introduced = suggested - previous
            if introduced & {COLOR_VI['green'], COLOR_VI['blue']}:
                colors = None
        if assessment.status in {'underexposed', 'overexposed'}:
            colors = None
        old = frozenset(self.color.split(' và ')) if self.label_confirmed else frozenset()
        proposed = frozenset(COLOR_VI[name] for name in colors) if colors else frozenset()
        if self.paint_pair_confirmed:
            if self._consider_red_pair_refutation(
                    night_red_spatial_evidence(crop), context,
                    scores if assessment.status == 'reflection_corrected' else None,
                    args):
                return
        corrected = assessment.status in {'reflection_corrected', 'alternate_supported', 'headlights_corrected'}
        neutral_labels = {COLOR_VI[name] for name in ('black', 'gray', 'white')}
        if old and proposed != old and colors:
            if corrected and not white_supported and (
                    old <= neutral_labels and proposed <= neutral_labels
                    or self.label_paint_verified):
                colors = None
            if context.weight < .60 or context.clipped_share >= .12:
                colors = None
            if colors and colors == ('red',) and old <= neutral_labels:
                evidence = night_red_spatial_evidence(crop)
                if evidence['lamp_share'] >= .003 and evidence['strong_red_middle'] < .28:
                    colors = None
            if colors and self.paint_pair_confirmed and len(colors) == 1:
                single_clear = ((body and body['name'] == colors[0]
                                 and body['share'] >= (.80 if self.vehicle_class == 5 else .43)
                                 and raw[COLOR_NAMES.index(colors[0])] >= .80)
                                or (white_supported and white_body['share'] >= .60))
                if not single_clear:
                    colors = None
        if colors is None:
            self.votes.append(None)
            self.challenger = None
            self.challenger_hits = 0
            self.status = ('held_' if self.label_confirmed else '') + assessment.status
            return
        key = frozenset(colors)
        self.votes.append((key, context.weight, confidence, verified))
        supporting = [vote for vote in self.votes if vote is not None and vote[0] == key]
        minimum = max(5, args.min_color_samples) if len(colors) == 2 else args.min_color_samples
        missing_weight = .25 if len(colors) == 2 else .60
        total_weight = sum(vote[1] if vote is not None else missing_weight for vote in self.votes)
        support = sum(vote[1] for vote in supporting) / max(total_weight, 1e-6)
        if len(supporting) < minimum or support < .70:
            self.challenger = None
            self.challenger_hits = 0
            return
        confidence = float(np.median([vote[2] for vote in supporting]))
        verified = sum(vote[3] for vote in supporting) >= .70*len(supporting)
        if not self.label_confirmed:
            self._commit(colors, verified)
        elif proposed == old:
            self.label_paint_verified = self.label_paint_verified or verified
            self.challenger = None
            self.challenger_hits = 0
        else:
            required = args.switch_hits
            if old & neutral_labels or proposed & neutral_labels or self.paint_pair_confirmed:
                required = max(required, args.switch_hits_neutral)
            self.challenger_hits = self.challenger_hits+1 if self.challenger == key else 1
            self.challenger = key
            if self.challenger_hits >= required:
                self._commit(colors, verified)


# --- THEO DOI XE ---

class StableCar:
    """Một ID duy nhất do chương trình quản lý, không dùng raw ByteTrack ID để hiển thị."""

    def __init__(self, car_id, detection, frame_number, args):
        self.id = car_id
        self.raw_id = detection.get('raw_id')
        self.box = np.asarray(detection['box'], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.appearance = np.asarray(detection['appearance'], dtype=np.float32)
        self.last_frame = frame_number
        self.conf = float(detection['conf'])
        self.hits = 0
        self.confirmed = False
        self.in_roi = bool(detection.get('in_roi', True))
        self.color_state = TrackColorState(car_id, args)
        self.class_history = deque(maxlen=15)
        self.roi_entry_frame = frame_number if self.in_roi else None
        self.roi_last_inside_frame = frame_number if self.in_roi else -10 ** 8
        self.roi_inside_hits = 0
        self.roi_ready = False
        self.flow_points = None
        self.flow_offset = np.zeros(2, dtype=np.float32)
        self.flow_last_good_frame = -10 ** 8
        self.flow_static_streak = 0

    @property
    def color(self):
        return self.color_state.color

    def predicted_box(self, frame_number, fps):
        dt = max(0, frame_number - self.last_frame)
        b = self.box.copy()
        if dt == 0:
            return b
        if self.flow_last_good_frame >= self.last_frame + 1:
            shift = self.flow_offset.copy()
            missing_flow = max(0, frame_number - self.flow_last_good_frame)
            shift += self.velocity * min(missing_flow, 3)
        else:
            shift = self.velocity * min(dt, min(argsafe_hold(fps), 7))
        shift = np.clip(shift, -max(20.0, min(b[2] - b[0], b[3] - b[1]) * 0.9), max(20.0, min(b[2] - b[0], b[3] - b[1]) * 0.9))
        b[[0, 2]] += shift[0]
        b[[1, 3]] += shift[1]
        return b

    def advance_flow(self, prev_gray, gray, frame_number, args):
        """Theo dịch chuyển KLT giữa hai frame; tuyệt đối không coi là detection mới."""
        if frame_number - self.last_frame > args.hold_frames:
            return
        if self.flow_points is None or len(self.flow_points) < 5:
            return
        try:
            next_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, self.flow_points, None, winSize=(21, 21), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
            if next_pts is None or st is None:
                self.flow_points = None
                return
            back_pts, bst, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, next_pts, None, winSize=(21, 21), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
            if back_pts is None or bst is None:
                self.flow_points = None
                return
            p0 = self.flow_points.reshape(-1, 2)
            p1 = next_pts.reshape(-1, 2)
            pb = back_pts.reshape(-1, 2)
            valid = (st.reshape(-1) > 0) & (bst.reshape(-1) > 0) & (np.linalg.norm(pb - p0, axis=1) < 1.8)
            if np.count_nonzero(valid) < 5:
                self.flow_points = None
                return
            shifts = p1[valid] - p0[valid]
            delta = np.median(shifts, axis=0)
            dev = np.median(np.linalg.norm(shifts - delta, axis=1))
            if not np.all(np.isfinite(delta)) or np.linalg.norm(delta) > max(28.0, min(self.box[2] - self.box[0], self.box[3] - self.box[1]) * 0.6) or dev > max(5.0, np.linalg.norm(delta) * 0.85):
                self.flow_points = None
                return
            flow_speed = float(np.linalg.norm(delta))
            vehicle_speed = float(np.linalg.norm(self.velocity))
            if vehicle_speed > 1.5 and flow_speed < 0.35:
                self.flow_static_streak += 1
            else:
                self.flow_static_streak = 0
            if self.flow_static_streak >= 2:
                self.flow_points = None
                self.flow_last_good_frame = -10 ** 8
                return
            self.flow_offset += delta.astype(np.float32)
            self.flow_points = p1[valid].reshape(-1, 1, 2).astype(np.float32)
            self.flow_last_good_frame = frame_number
        except cv2.error:
            self.flow_points = None

    def update_roi_status(self, detection, frame_number, args):
        inside = bool(detection.get('in_roi', True))
        self.in_roi = inside
        if not getattr(args, 'roi_enabled', False):
            self.roi_ready = True
            return
        grace = max(1, round(args.roi_exit_grace * args.video_fps))
        if inside:
            if self.roi_entry_frame is None or frame_number - self.roi_last_inside_frame > grace:
                self.roi_entry_frame = frame_number
                self.roi_inside_hits = 0
            self.roi_last_inside_frame = frame_number
            self.roi_inside_hits += 1
            if self.roi_inside_hits >= 3:
                self.roi_ready = True
        elif frame_number - self.roi_last_inside_frame > grace:
            self.roi_ready = False
            self.roi_entry_frame = None
            self.roi_inside_hits = 0

    def match_score(self, det, frame_number, args, fps):
        age = frame_number - self.last_frame
        if age > args.reid_frames:
            return None
        raw = np.asarray(det['box'], dtype=np.float32)
        pred = self.predicted_box(frame_number, fps)
        p, q = (center_size(pred), center_size(raw))
        ratio = q[2] * q[3] / max(1.0, p[2] * p[3])
        raw_same = self.raw_id is not None and det['raw_id'] is not None and (self.raw_id == det['raw_id'])
        if ratio < 0.45 or ratio > 2.35:
            return None
        scale = max(28.0, float(np.hypot(p[2], p[3])), float(np.hypot(q[2], q[3])))
        dist = float(np.linalg.norm(p[:2] - q[:2]))
        overlap = box_iou(pred, raw)
        gate = min(1.25, 0.55 + 0.025 * age)
        if dist > scale * gate:
            return None
        sim = float(np.dot(self.appearance, det['appearance']))
        if age > args.hold_frames and (not raw_same) and (sim < 0.73):
            return None
        if overlap < 0.05 and (dist > scale * 0.42 or (not raw_same and sim < 0.62)):
            return None
        if not raw_same and overlap < 0.12 and (sim < 0.55):
            return None
        score = 3.7 * overlap + 1.35 * max(0.0, 1.0 - dist / (scale * gate)) + 0.55 * sim
        if raw_same:
            score += 1.1
        score -= 0.022 * age
        return score if score >= 0.9 else None

    def update(self, detection, frame, frame_number, classifier, args, gray=None):
        raw = np.asarray(detection['box'], dtype=np.float32)
        dt = max(1, frame_number - self.last_frame)
        pred_before = self.predicted_box(frame_number, args.video_fps)
        prev_center = center_size(self.box)[:2]
        new_center = center_size(raw)[:2]
        measured_vel = (new_center - prev_center) / dt
        measured_vel = np.clip(measured_vel, -24.0, 24.0)
        if dt <= 4:
            self.velocity = 0.75 * self.velocity + 0.25 * measured_vel
        else:
            self.velocity = 0.3 * measured_vel
        self.box = args.box_alpha * raw + (1 - args.box_alpha) * pred_before
        self.appearance = 0.7 * self.appearance + 0.3 * detection['appearance']
        self.appearance /= max(1e-06, float(np.linalg.norm(self.appearance)))
        if detection['raw_id'] is not None:
            self.raw_id = detection['raw_id']
        self.last_frame = frame_number
        self.conf = float(detection['conf'])
        self.update_roi_status(detection, frame_number, args)
        self.flow_offset[:] = 0
        self.flow_last_good_frame = -10 ** 8
        self.flow_static_streak = 0
        self.flow_points = feature_points_in_box(gray, raw) if gray is not None else None
        self.hits += 1
        self.class_history.append(detection['class_id'])
        bus_hits = self.class_history.count(5)
        car_hits = self.class_history.count(2)
        if (bus_hits >= 3 and bus_hits >= .60*(bus_hits + car_hits)
                and not self.color_state.verified_suv):
            self.color_state.vehicle_class = 5
        self.confirmed = self.confirmed or self.hits >= args.min_hits
        if not self.in_roi:
            return
        interval = (args.color_every_confirmed
                    if self.color_state.label_confirmed else args.color_every)
        if classifier is None or (self.hits - 1) % interval != 0:
            return
        x1, y1, x2, y2 = clip_box(raw, frame.shape[1], frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < args.min_color_side:
            return
        if not color_crop_is_complete((x1, y1, x2, y2), frame.shape):
            self.color_state.status = 'partial_vehicle'
            return
        self.color_state.update(crop, classifier, frame_number, args)

    def annotated(self, frame_number, args, fps, roi_polygon=None):
        age = frame_number - self.last_frame
        if not self.confirmed or not self.roi_ready or age > args.hold_frames:
            return None
        if args.roi_enabled:
            grace = max(1, round(args.roi_exit_grace * fps))
            if frame_number - self.roi_last_inside_frame > grace:
                return None
        no_flow_age = frame_number - self.flow_last_good_frame if self.flow_last_good_frame >= self.last_frame + 1 else age
        if age > 0 and no_flow_age > args.no_flow_hold_frames:
            return None
        box = self.predicted_box(frame_number, fps) if age > 0 else self.box
        if roi_polygon is not None:
            if vehicle_roi_signed_distance(box, roi_polygon) < -args.roi_boundary_grace:
                return None
        text = format_vehicle_label(self.id, self.conf, self.color)
        return (box, text, age > 0)

class CarTrackManager:

    def __init__(self, args, fps, roi_polygon=None):
        self.roi_polygon = roi_polygon
        self.args = args
        self.fps = fps
        self.args.video_fps = fps
        self.args.roi_enabled = roi_polygon is not None
        self.prev_gray = None
        self.cars = {}
        self.next_id = 1
        self.type_cache = []

    def is_passenger_mislabeled_truck(self, det, frame_number, frame, classifier):
        """Cứu SUV bị YOLO gán truck bằng *kiểu xe* từ PP-Vehicle."""
        args = self.args
        if not args.recover_suv:
            return False
        if self.roi_polygon is not None and (not det.get('in_roi', False)):
            return False
        box = det['box']
        x1, y1, x2, y2 = clip_box(box, frame.shape[1], frame.shape[0])
        if min(x2 - x1, y2 - y1) < args.suv_min_side:
            return False
        self.type_cache = [r for r in self.type_cache if frame_number - r['frame'] <= args.suv_type_every * 2]
        for recent in self.type_cache:
            same_id = det['raw_id'] is not None and recent['raw_id'] == det['raw_id']
            same_box = box_iou(box, recent['box']) >= 0.48
            if (same_id or same_box) and frame_number - recent['frame'] < args.suv_type_every:
                return recent['accept']
        if classifier is None or not hasattr(classifier, 'predict_type_proba'):
            raise RuntimeError('Bộ phân loại PP-Vehicle cần hỗ trợ predict_type_proba để xét SUV bị gán truck')
        types = np.asarray(classifier.predict_type_proba(frame[y1:y2, x1:x2]), dtype=np.float32)
        if types.shape != (9,):
            raise RuntimeError(f"PP-Vehicle cần 9 điểm kiểu xe, thực tế {types.shape}")
        top = int(np.argmax(types))
        name = VEHICLE_TYPE_NAMES[top]
        score = float(types[top])
        accept = name in PASSENGER_TYPES and score >= args.suv_type_conf
        self.type_cache.append({'frame': frame_number, 'raw_id': det['raw_id'], 'box': box, 'accept': accept})
        return accept

    def step(self, detections, frame_number, frame, classifier):
        args = self.args
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_ids = {det['raw_id'] for det in detections if det['raw_id'] is not None}
        if self.prev_gray is not None:
            for car in self.cars.values():
                if car.raw_id is None or car.raw_id not in detected_ids:
                    car.advance_flow(self.prev_gray, gray, frame_number, args)
        self.prev_gray = gray
        pairs = []
        for di, det in enumerate(detections):
            for car_id, car in self.cars.items():
                score = car.match_score(det, frame_number, args, self.fps)
                if score is not None:
                    pairs.append((score, di, car_id))
        pairs.sort(reverse=True)
        used_dets = set()
        used_cars = set()
        for _, di, car_id in pairs:
            if di in used_dets or car_id in used_cars:
                continue
            used_dets.add(di)
            used_cars.add(car_id)
            car = self.cars[car_id]
            car.update(detections[di], frame, frame_number, classifier, args, gray)
        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            if det['conf'] < args.new_conf:
                continue
            if any((box_iou(det['box'], c.predicted_box(frame_number, self.fps)) > 0.42 and frame_number - c.last_frame <= args.hold_frames for c in self.cars.values())):
                continue
            if det['class_id'] == 7:
                if not self.is_passenger_mislabeled_truck(
                        det, frame_number, frame, classifier):
                    continue
            elif det['class_id'] not in (2, 5):
                continue
            car = StableCar(self.next_id, det, frame_number, args)
            self.cars[self.next_id] = car
            self.next_id += 1
            car.update(det, frame, frame_number, classifier, args)
        expired = [cid for cid, car in self.cars.items() if frame_number - car.last_frame > args.reid_frames]
        for cid in expired:
            del self.cars[cid]
        out = []
        for car in self.cars.values():
            a = car.annotated(frame_number, args, self.fps, self.roi_polygon)
            if a is not None:
                out.append(a)
        return out


# --- VE NHAN ---

def annotate_frame(frame_bgr, annotations, render_width=1280):
    """Vẽ nhãn ID:01-Conf:089-Màu; xếp nhãn tránh va chạm và tràn ảnh."""
    h, w = frame_bgr.shape[:2]
    if render_width > 0 and w > render_width:
        new_h = max(2, round(h * render_width / w))
        out = cv2.resize(frame_bgr, (render_width, new_h), interpolation=cv2.INTER_AREA)
        sx = render_width / w
        sy = new_h / h
    else:
        out = frame_bgr.copy()
        sx = sy = 1.0
    ph, pw = out.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = get_font(18)
    boxes = []
    for box, label, held in annotations:
        x1, y1, x2, y2 = box
        x1 = max(0, min(pw - 2, int(round(x1 * sx))))
        x2 = max(x1 + 1, min(pw - 1, int(round(x2 * sx))))
        y1 = max(0, min(ph - 2, int(round(y1 * sy))))
        y2 = max(y1 + 1, min(ph - 1, int(round(y2 * sy))))
        outline = (242, 192, 70) if not held else (170, 174, 180)
        draw.rectangle((x1, y1, x2, y2), outline=outline, width=2)
        boxes.append(((x1, y1, x2, y2), label, held))
    boxes.sort(key=lambda item: (item[0][1], item[0][0]))
    occupied = []
    for (x1, y1, x2, y2), label, held in boxes:
        text_box = draw.textbbox((0, 0), label, font=font)
        tw = text_box[2] - text_box[0]
        th = text_box[3] - text_box[1]
        label_w = min(pw, max(18, tw + 12))
        label_h = th + 12
        base_x = max(0, min(x1, pw - label_w))
        choices = []
        for yy in (y1 - label_h - 3, y2 + 3, y1 + 3):
            for xx in (base_x, max(0, min(pw - label_w, x2 - label_w)), max(0, min(pw - label_w, base_x - label_w // 2))):
                if 0 <= yy <= ph - label_h:
                    choices.append((xx, yy, xx + label_w, yy + label_h))
        chosen = None
        for candidate in choices:
            if not any((rectangles_overlap(candidate, old, 2) for old in occupied)):
                chosen = candidate
                break
        if chosen is None:
            for offset in range(0, min(160, ph), label_h + 3):
                cy = max(0, min(ph - label_h, y1 + offset))
                candidate = (base_x, cy, base_x + label_w, cy + label_h)
                if not any((rectangles_overlap(candidate, old, 2) for old in occupied)):
                    chosen = candidate
                    break
        if chosen is None:
            chosen = (base_x, max(0, min(ph - label_h, y1 - label_h - 3)), base_x + label_w, max(0, min(ph - label_h, y1 - label_h - 3)) + label_h)
        occupied.append(chosen)
        draw.rectangle(chosen, fill=(18, 21, 29))
        draw.text((chosen[0] + 6, chosen[1] + 4), label, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# --- XU LY VIDEO ---

def color_crop_is_complete(box, shape):
    """Tránh phân loại chỉ một góc xe khi bbox bị cắt sát biên hình."""
    x1, y1, x2, y2 = box
    height, width = shape[:2]
    border = max(2, round(.004*min(height,width)))
    at_edge = x1 <= border or y1 <= border or x2 >= width-border or y2 >= height-border
    return not (at_edge and y2-y1 < .75*(x2-x1))


def run(args, detector=None, classifier=None):
    """Chạy nhận diện màu từng frame; điều phối ROI, YOLO, tracker và MP4."""
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {args.source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not 1 <= fps <= 240:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        roi_polygon = setup_roi(cap, args.source, args, width, height)
    except Exception:
        cap.release()
        raise
    search_box = roi_search_box(
        roi_polygon, width, height, args.roi_boundary_grace, args.imgsz)
    search_x1, search_y1, search_x2, search_y2 = search_box
    box_offset = np.array(
        [search_x1, search_y1, search_x1, search_y1], dtype=np.float32)
    if args.day_mode:
        camera_start = None
    else:
        try:
            camera_start = get_camera_start(cap, args)
        except Exception:
            cap.release()
            raise
    output = Path(args.output) if args.output else output_path(args.source)
    try:
        if detector is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError('Thiếu Ultralytics. Cài: python -m pip install ultralytics') from exc
            if not Path(args.model).is_file():
                raise FileNotFoundError(f"Không tìm thấy model YOLO cục bộ: {args.model}")
            detector = YOLO(args.model)
        if classifier is None:
            classifier = PPVehicleColorClassifier(args.model_dir, args.paddle_device)
    except Exception:
        cap.release()
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    render_width = round(width * min(1.0, args.render_width / width)) if args.render_width > 0 else width
    render_height = round(height * render_width / width)
    render_width += render_width % 2
    render_height += render_height % 2
    output_size = (render_width, render_height)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Không ghi được MP4: {output}")
    preview_window = 'Nhan dien mau o to | Q: dung'
    show = not args.no_show
    if show:
        try:
            cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(preview_window, render_width, render_height)
        except cv2.error as exc:
            print(f"[WARN] Không mở được cửa sổ xem: {exc}. Vẫn xuất MP4.")
            show = False
    manager = CarTrackManager(args, fps, roi_polygon=roi_polygon)
    scene_light = SceneLightShiftGuard() if args.scene_light_guard else None
    args.scene_green_shift = False
    frame_no = 0
    last_night_active = None
    brightness = None
    args.night_active = False
    print(f"[INFO] Video: {width}x{height}, {fps:.2f} FPS, tổng {total_frames} frame")
    print(f"[INFO] YOLO class 2+5+7; nhận ô tô, xe khách và SUV được xác minh; giữ khung tối đa {args.hold_frames} frame khi optical flow còn tốt")
    if roi_polygon is not None:
        print(f"[INFO] ROI: xác nhận sau {args.roi_confirm_seconds:.1f}s; dung sai ra vùng {args.roi_exit_grace:.1f}s")
        if search_box != (0, 0, width, height):
            print(f"[INFO] YOLO chỉ xử lý vùng ROI mở rộng: "
                  f"{search_x2-search_x1}x{search_y2-search_y1}")
    print(f"[INFO] Màu: {('tối đa 2 màu' if args.two_colors else '1 màu chính')} | SUV rescue: {('ON' if args.recover_suv else 'OFF')}")
    print('[INFO] Màu từ model và vùng thân; xác nhận qua nhiều lượt quan sát theo ID.')
    print(f"[INFO] Video kết quả: {output}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            args.scene_green_shift = bool(scene_light.update(frame)) if scene_light else False
            if args.day_mode:
                args.night_active = False
            elif camera_start is not None:
                args.night_active = is_night_clock(camera_start, frame_no, fps, args.night_from, args.night_until)
            else:
                args.night_active, brightness = automatic_night_mode(frame, brightness, args.night_active)
            if frame_no == 1 or args.night_active != last_night_active:
                print(f"[COLOR] Frame {frame_no}: " + ('CHẾ ĐỘ ĐÊM/ẢNH TỐI' if args.night_active else 'CHẾ ĐỘ NGÀY/ẢNH SÁNG'))
                last_night_active = args.night_active
            search_frame = frame[search_y1:search_y2, search_x1:search_x2]
            detection_frame = prepare_detection_frame(search_frame)
            result = detector.track(
                detection_frame, persist=True, tracker='bytetrack.yaml',
                classes=list(CLASSES), conf=args.conf, iou=args.iou,
                imgsz=args.imgsz, verbose=False)[0]
            detections = []
            boxes = result.boxes
            if boxes is not None:
                positions = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
                for index, (raw_box, confidence, class_id) in enumerate(zip(positions, confidences, class_ids)):
                    if class_id not in CLASSES:
                        continue
                    box = clip_box(np.asarray(raw_box, dtype=np.float32) + box_offset,
                                   width, height)
                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                    if box_area < args.min_area:
                        continue
                    crop = frame[box[1]:box[3], box[0]:box[2]]
                    if crop.size == 0:
                        continue
                    detections.append({'box': box, 'conf': float(confidence), 'class_id': int(class_id), 'raw_id': int(track_ids[index]) if track_ids is not None else None, 'appearance': appearance_signature(crop), 'in_roi': point_in_vehicle_roi(box, roi_polygon)})
            detections = deduplicate_detections(detections)
            annotations = manager.step(detections, frame_no, frame, classifier)
            rendered = annotate_frame(frame, annotations, args.render_width)
            if rendered.shape[:2] != (render_height, render_width):
                rendered = cv2.resize(rendered, output_size, interpolation=cv2.INTER_AREA)
            writer.write(rendered)
            if show:
                cv2.imshow(preview_window, rendered)
                if (cv2.waitKey(1) & 0xff) in (ord('q'), ord('Q')):
                    break
                if cv2.getWindowProperty(preview_window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if frame_no % 100 == 0:
                print(f"[INFO] Đã xử lý {frame_no}/{total_frames} frame")
    finally:
        cap.release()
        writer.release()
        if show:
            try:
                cv2.destroyWindow(preview_window)
            except cv2.error:
                pass
    print(f"[DONE] {frame_no} frame -> {output.resolve()}")
    return str(output)


# --- THAM SO DONG LENH ---

def parse_args(argv=None):
    """Chỉ nhận các tùy chọn cần thiết khi vận hành hệ thống."""
    parser = argparse.ArgumentParser(
        description='Nhận dạng màu ô tô/xe khách bằng YOLO11 và PP-Vehicle',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', required=True, help='Video, RTSP hoặc webcam 0')
    parser.add_argument('--output', help='MP4 kết quả; bỏ trống để tự đặt tên')
    parser.add_argument('--paddle-device', choices=('auto', 'cpu', 'gpu'),
                        default='auto', help='Tự dùng GPU khi Paddle hỗ trợ')
    parser.add_argument('--render-width', type=int, default=1280,
                        help='Chiều rộng video kết quả')
    parser.add_argument('--no-show', action='store_true',
                        help='Không mở cửa sổ xem trực tiếp')
    parser.add_argument('--roi-file', help='Đọc hoặc lưu vùng ROI dạng JSON')
    parser.add_argument('--redraw-roi', action='store_true',
                        help='Vẽ lại ROI dù đã có ROI JSON')
    parser.add_argument('--no-roi', action='store_true',
                        help='Xử lý toàn khung hình')
    parser.add_argument('--camera-start', help='Giờ của frame đầu, dạng HH:MM:SS')
    parser.add_argument('--clock-box', type=int, nargs=4,
                        metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='Vùng đồng hồ camera dùng cho OCR')
    parser.add_argument('--day-mode', action='store_true',
                        help='Buộc dùng chế độ ban ngày')
    values = parser.parse_args(argv)
    if values.render_width < 320:
        parser.error('--render-width phải từ 320 trở lên')
    if values.no_roi and values.redraw_roi:
        parser.error('--no-roi không thể đi cùng --redraw-roi')
    if values.clock_box:
        x1, y1, x2, y2 = values.clock_box
        if x2 <= x1 or y2 <= y1:
            parser.error('--clock-box cần X2 > X1 và Y2 > Y1')
    if values.camera_start:
        try:
            parse_camera_clock(values.camera_start)
        except ValueError as exc:
            parser.error(str(exc))
    return RuntimeConfig(**vars(values))


def configure_console():
    """Giữ log tiếng Việt đọc được trên terminal Windows và Linux."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except (OSError, ValueError):
                pass


def main():
    configure_console()
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("\n[INFO] Đã dừng theo yêu cầu")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
