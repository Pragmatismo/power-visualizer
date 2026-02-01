import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict

from PySide6.QtCore import (
    Qt, QRect, QRectF, QPoint, QPointF, QSize, QSettings, QTimer
)
from PySide6.QtGui import (
    QAction, QBrush, QColor, QFont, QPainter, QPen, QCursor
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTableWidget, QTableWidgetItem, QPushButton, QLabel,
    QFileDialog, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QDialog, QFormLayout, QDialogButtonBox, QGroupBox, QCheckBox,
    QLineEdit, QHeaderView, QListWidget, QListWidgetItem
)

# =========================
# Models
# =========================

MINUTES_PER_DAY = 24 * 60


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def time_to_minutes(hhmm: str) -> int:
    # "HH:MM"
    try:
        hh, mm = hhmm.strip().split(":")
        return clamp(int(hh) * 60 + int(mm), 0, MINUTES_PER_DAY)
    except Exception:
        return 0


def minutes_to_time(m: int) -> str:
    m = clamp(int(m), 0, MINUTES_PER_DAY)
    hh = m // 60
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"


@dataclass
class Interval:
    start_min: int
    end_min: int  # exclusive-ish; we treat as end boundary

    def normalized(self) -> "Interval":
        s = clamp(self.start_min, 0, MINUTES_PER_DAY)
        e = clamp(self.end_min, 0, MINUTES_PER_DAY)
        if e < s:
            s, e = e, s
        return Interval(s, e)


@dataclass
class Event:
    start_min: int
    duration_min: int
    # if energy_wh is set, energy is fixed; otherwise power_w is used (device power)
    energy_wh: Optional[float] = None

    def normalized(self) -> "Event":
        s = clamp(self.start_min, 0, MINUTES_PER_DAY - 1)
        d = max(1, int(self.duration_min))
        # split at midnight is enforced by clamping duration if needed
        d = min(d, MINUTES_PER_DAY - s)
        return Event(s, d, self.energy_wh)


class DeviceType:
    ALWAYS = "Always on"
    SCHEDULED = "Scheduled blocks"
    EVENTS = "Per-use events"


@dataclass
class Device:
    name: str = "New device"
    dtype: str = DeviceType.SCHEDULED
    power_w: float = 20.0
    quantity: int = 1

    intervals: List[Interval] = field(default_factory=list)  # for scheduled
    events: List[Event] = field(default_factory=list)        # for per-use

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "power_w": self.power_w,
            "quantity": self.quantity,
            "intervals": [asdict(i) for i in self.intervals],
            "events": [asdict(e) for e in self.events],
        }

    @staticmethod
    def from_dict(d: dict) -> "Device":
        dev = Device(
            name=d.get("name", "Device"),
            dtype=d.get("dtype", DeviceType.SCHEDULED),
            power_w=float(d.get("power_w", 20.0)),
            quantity=int(d.get("quantity", 1)),
        )
        dev.intervals = [Interval(**i).normalized() for i in d.get("intervals", [])]
        dev.events = [Event(**e).normalized() for e in d.get("events", [])]
        return dev


@dataclass
class TariffWindow:
    start_min: int
    end_min: int
    rate_per_kwh: float  # £/kWh

    def normalized(self) -> "TariffWindow":
        s = clamp(self.start_min, 0, MINUTES_PER_DAY)
        e = clamp(self.end_min, 0, MINUTES_PER_DAY)
        if e < s:
            s, e = e, s
        return TariffWindow(s, e, max(0.0, float(self.rate_per_kwh)))


@dataclass
class FreeRule:
    enabled: bool = False
    # free within [start, end)
    start_min: int = 9 * 60
    end_min: int = 16 * 60
    # free power threshold in kW (instantaneous). If total_kW <= threshold, cost is free for that minute.
    free_kw_threshold: float = 5.0

    def normalized(self) -> "FreeRule":
        s = clamp(self.start_min, 0, MINUTES_PER_DAY)
        e = clamp(self.end_min, 0, MINUTES_PER_DAY)
        if e < s:
            s, e = e, s
        return FreeRule(
            enabled=bool(self.enabled),
            start_min=s,
            end_min=e,
            free_kw_threshold=max(0.0, float(self.free_kw_threshold))
        )


@dataclass
class SettingsModel:
    currency_symbol: str = "£"
    month_days: int = 30
    step_minutes: int = 1

    # Default UK-ish flat rate (roughly; user can change)
    base_rate_flat: float = 0.30  # £/kWh

    # Optional time-of-day schedule
    use_time_of_day: bool = False
    offpeak_rate: float = 0.22
    offpeak_start_min: int = 0
    offpeak_end_min: int = 7 * 60

    # Optional free window rule (solar simulation-ish)
    free_rule: FreeRule = field(default_factory=FreeRule)

    def to_qsettings(self, qs: QSettings):
        qs.setValue("currency_symbol", self.currency_symbol)
        qs.setValue("month_days", self.month_days)
        qs.setValue("step_minutes", self.step_minutes)

        qs.setValue("base_rate_flat", self.base_rate_flat)

        qs.setValue("use_time_of_day", self.use_time_of_day)
        qs.setValue("offpeak_rate", self.offpeak_rate)
        qs.setValue("offpeak_start_min", self.offpeak_start_min)
        qs.setValue("offpeak_end_min", self.offpeak_end_min)

        qs.setValue("free_enabled", self.free_rule.enabled)
        qs.setValue("free_start_min", self.free_rule.start_min)
        qs.setValue("free_end_min", self.free_rule.end_min)
        qs.setValue("free_kw_threshold", self.free_rule.free_kw_threshold)

        qs.setValue("has_run_before", True)

    @staticmethod
    def from_qsettings(qs: QSettings) -> "SettingsModel":
        sm = SettingsModel()
        sm.currency_symbol = qs.value("currency_symbol", sm.currency_symbol)
        sm.month_days = int(qs.value("month_days", sm.month_days))
        sm.step_minutes = int(qs.value("step_minutes", sm.step_minutes))

        sm.base_rate_flat = float(qs.value("base_rate_flat", sm.base_rate_flat))

        sm.use_time_of_day = (str(qs.value("use_time_of_day", sm.use_time_of_day)).lower() == "true")
        sm.offpeak_rate = float(qs.value("offpeak_rate", sm.offpeak_rate))
        sm.offpeak_start_min = int(qs.value("offpeak_start_min", sm.offpeak_start_min))
        sm.offpeak_end_min = int(qs.value("offpeak_end_min", sm.offpeak_end_min))

        fr = FreeRule(
            enabled=(str(qs.value("free_enabled", sm.free_rule.enabled)).lower() == "true"),
            start_min=int(qs.value("free_start_min", sm.free_rule.start_min)),
            end_min=int(qs.value("free_end_min", sm.free_rule.end_min)),
            free_kw_threshold=float(qs.value("free_kw_threshold", sm.free_rule.free_kw_threshold)),
        ).normalized()
        sm.free_rule = fr
        # enforce the constraints you chose
        sm.month_days = 30
        sm.step_minutes = 1
        return sm

    @staticmethod
    def has_run_before(qs: QSettings) -> bool:
        return (str(qs.value("has_run_before", "false")).lower() == "true")


@dataclass
class Project:
    devices: List[Device] = field(default_factory=list)
    custom_periods: List[Tuple[str, float]] = field(default_factory=lambda: [("Lettuce", 45.0)])

    def to_dict(self) -> dict:
        return {
            "devices": [d.to_dict() for d in self.devices],
            "custom_periods": [{"name": n, "days": days} for (n, days) in self.custom_periods],
        }

    @staticmethod
    def from_dict(d: dict) -> "Project":
        p = Project()
        p.devices = [Device.from_dict(x) for x in d.get("devices", [])]
        cps = []
        for item in d.get("custom_periods", []):
            try:
                cps.append((item.get("name", "Period"), float(item.get("days", 1.0))))
            except Exception:
                pass
        if cps:
            p.custom_periods = cps
        return p


# =========================
# Simulation
# =========================

@dataclass
class SimResult:
    total_kwh_day: float
    total_cost_day: float
    per_device_kwh_day: List[float]
    per_device_cost_day: List[float]
    per_device_on_minutes: List[int]
    minute_total_w: List[float]  # length 1440


def tariff_rate_for_minute(settings: SettingsModel, minute: int) -> float:
    """Return £/kWh rate for the minute (base schedule only)."""
    if not settings.use_time_of_day:
        return settings.base_rate_flat
    # offpeak window can be any range; for MVP we keep it simple and assume no wrap
    if settings.offpeak_start_min <= minute < settings.offpeak_end_min:
        return settings.offpeak_rate
    return settings.base_rate_flat


def is_free_this_minute(settings: SettingsModel, minute: int, total_kw: float) -> bool:
    fr = settings.free_rule.normalized()
    if not fr.enabled:
        return False
    if fr.start_min <= minute < fr.end_min:
        return total_kw <= fr.free_kw_threshold
    return False


def simulate_day(project: Project, settings: SettingsModel) -> SimResult:
    step = 1  # forced by your decision
    minute_total_w = [0.0] * MINUTES_PER_DAY
    dev_w = [ [0.0] * MINUTES_PER_DAY for _ in project.devices ]

    # Build per-device minute power
    for i, dev in enumerate(project.devices):
        qty = max(1, int(dev.quantity))
        base_power_w = max(0.0, float(dev.power_w)) * qty

        if dev.dtype == DeviceType.ALWAYS:
            for m in range(MINUTES_PER_DAY):
                dev_w[i][m] += base_power_w

        elif dev.dtype == DeviceType.SCHEDULED:
            for iv in dev.intervals:
                ivn = iv.normalized()
                for m in range(ivn.start_min, ivn.end_min):
                    dev_w[i][m] += base_power_w

        elif dev.dtype == DeviceType.EVENTS:
            # Each event either uses base_power_w for duration or fixed energy spread over its duration
            for ev in dev.events:
                evn = ev.normalized()
                if evn.energy_wh is None:
                    for m in range(evn.start_min, evn.start_min + evn.duration_min):
                        dev_w[i][m] += base_power_w
                else:
                    # Spread energy across duration -> power = Wh / hours
                    hours = evn.duration_min / 60.0
                    if hours <= 0:
                        continue
                    pw = (float(evn.energy_wh) / hours)  # W (since Wh / h = W)
                    pw = max(0.0, pw) * qty
                    for m in range(evn.start_min, evn.start_min + evn.duration_min):
                        dev_w[i][m] += pw

        # else: unknown -> do nothing

    # Sum to total
    for m in range(MINUTES_PER_DAY):
        minute_total_w[m] = sum(dev_w[i][m] for i in range(len(project.devices)))

    per_device_kwh = [0.0] * len(project.devices)
    per_device_cost = [0.0] * len(project.devices)
    per_device_on_minutes = [0] * len(project.devices)

    total_kwh = 0.0
    total_cost = 0.0

    # Compute cost minute by minute
    for m in range(MINUTES_PER_DAY):
        total_w = minute_total_w[m]
        total_kw = total_w / 1000.0
        kwh_this_min = total_kw * (1.0 / 60.0)  # 1 minute
        base_rate = tariff_rate_for_minute(settings, m)

        if is_free_this_minute(settings, m, total_kw):
            cost_this_min = 0.0
        else:
            cost_this_min = kwh_this_min * base_rate

        total_kwh += kwh_this_min
        total_cost += cost_this_min

        # device shares: apply same "free minute" treatment proportionally by energy
        # (simple & consistent with “instantaneous threshold” concept)
        if total_w > 0:
            free = is_free_this_minute(settings, m, total_kw)
            for i in range(len(project.devices)):
                w_i = dev_w[i][m]
                if w_i > 0:
                    per_device_on_minutes[i] += 1
                    kwh_i = (w_i / 1000.0) * (1.0 / 60.0)
                    per_device_kwh[i] += kwh_i
                    if not free:
                        per_device_cost[i] += kwh_i * base_rate

    return SimResult(
        total_kwh_day=total_kwh,
        total_cost_day=total_cost,
        per_device_kwh_day=per_device_kwh,
        per_device_cost_day=per_device_cost,
        per_device_on_minutes=per_device_on_minutes,
        minute_total_w=minute_total_w
    )


# =========================
# Settings dialog
# =========================

class SettingsDialog(QDialog):
    def __init__(self, parent=None, settings_model: Optional[SettingsModel] = None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.model = settings_model or SettingsModel()

        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.currency_symbol = QLineEdit(self.model.currency_symbol)
        self.currency_symbol.setMaxLength(4)
        form.addRow("Currency symbol", self.currency_symbol)

        # Tariff group
        tariff_group = QGroupBox("Electricity tariff")
        tariff_layout = QFormLayout(tariff_group)

        self.flat_rate = QDoubleSpinBox()
        self.flat_rate.setDecimals(4)
        self.flat_rate.setRange(0.0, 10.0)
        self.flat_rate.setSingleStep(0.01)
        self.flat_rate.setValue(self.model.base_rate_flat)
        tariff_layout.addRow("Price per kWh", self.flat_rate)

        self.use_tod = QCheckBox("Off peak rate")
        self.use_tod.setChecked(self.model.use_time_of_day)
        tariff_layout.addRow(self.use_tod)

        self.offpeak_rate = QDoubleSpinBox()
        self.offpeak_rate.setDecimals(4)
        self.offpeak_rate.setRange(0.0, 10.0)
        self.offpeak_rate.setSingleStep(0.01)
        self.offpeak_rate.setValue(self.model.offpeak_rate)
        tariff_layout.addRow("Off peak rate", self.offpeak_rate)

        self.offpeak_start = QSpinBox()
        self.offpeak_start.setRange(0, 23)
        self.offpeak_start.setValue(self.model.offpeak_start_min // 60)
        tariff_layout.addRow("Off peak start", self.offpeak_start)

        self.offpeak_end = QSpinBox()
        self.offpeak_end.setRange(1, 24)
        self.offpeak_end.setValue(max(1, self.model.offpeak_end_min // 60))
        tariff_layout.addRow("Off peak end", self.offpeak_end)

        # Free window (solar-ish)
        free_group = QGroupBox("Free window (solar simulation)")
        free_layout = QFormLayout(free_group)

        self.free_enabled = QCheckBox("Enable free window")
        self.free_enabled.setChecked(self.model.free_rule.enabled)
        free_layout.addRow(self.free_enabled)

        self.free_start = QSpinBox()
        self.free_start.setRange(0, 23)
        self.free_start.setValue(self.model.free_rule.start_min // 60)
        free_layout.addRow("Free start (hour)", self.free_start)

        self.free_end = QSpinBox()
        self.free_end.setRange(1, 24)
        self.free_end.setValue(max(1, self.model.free_rule.end_min // 60))
        free_layout.addRow("Free end (hour)", self.free_end)

        self.free_kw = QDoubleSpinBox()
        self.free_kw.setDecimals(2)
        self.free_kw.setRange(0.0, 100.0)
        self.free_kw.setSingleStep(0.25)
        self.free_kw.setValue(self.model.free_rule.free_kw_threshold)
        free_layout.addRow("Free if total ≤ (kW)", self.free_kw)

        layout.addLayout(form)
        layout.addWidget(tariff_group)
        layout.addWidget(free_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.use_tod.stateChanged.connect(self._update_enabled)
        self._update_enabled()

    def _update_enabled(self):
        tod = self.use_tod.isChecked()
        self.offpeak_rate.setEnabled(tod)
        self.offpeak_start.setEnabled(tod)
        self.offpeak_end.setEnabled(tod)
        self.flat_rate.setEnabled(True)

    def get_model(self) -> SettingsModel:
        m = SettingsModel.from_qsettings(QSettings())  # start from current persisted defaults
        m.currency_symbol = self.currency_symbol.text().strip() or "£"
        m.month_days = 30
        m.step_minutes = 1

        m.use_time_of_day = self.use_tod.isChecked()
        m.base_rate_flat = float(self.flat_rate.value())
        m.offpeak_rate = float(self.offpeak_rate.value())
        m.offpeak_start_min = int(self.offpeak_start.value()) * 60
        m.offpeak_end_min = int(self.offpeak_end.value()) * 60

        fr = FreeRule(
            enabled=self.free_enabled.isChecked(),
            start_min=int(self.free_start.value()) * 60,
            end_min=int(self.free_end.value()) * 60,
            free_kw_threshold=float(self.free_kw.value()),
        ).normalized()
        m.free_rule = fr
        return m


# =========================
# Timeline widget
# =========================

class HitKind:
    NONE = "none"
    INTERVAL_BODY = "interval_body"
    INTERVAL_LEFT = "interval_left"
    INTERVAL_RIGHT = "interval_right"
    EVENT_BODY = "event_body"


@dataclass
class HitTest:
    kind: str = HitKind.NONE
    device_index: int = -1
    item_index: int = -1
    # for intervals/events: original start/end for drag operations
    anchor_minute: int = 0


class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        self.project: Optional[Project] = None
        self.settings: Optional[SettingsModel] = None
        self.sim: Optional[SimResult] = None

        # layout constants
        self.left_label_w = 220
        self.right_info_w = 240
        self.top_tariff_h = 28
        self.row_h = 44
        self.row_gap = 6
        self.axis_h = 22

        # interaction state
        self.hit = HitTest()
        self.dragging = False
        self.drag_start_pos = QPoint()
        self.drag_start_min = 0
        self.drag_original = None  # store original data
        self.hover_hit = HitTest()

        # appearance
        self.font = QFont("Sans", 9)
        self.setMinimumHeight(400)

    def set_data(self, project: Project, settings: SettingsModel, sim: SimResult):
        self.project = project
        self.settings = settings
        self.sim = sim
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        if not self.project:
            return QSize(900, 500)
        h = self.top_tariff_h + self.axis_h + (len(self.project.devices) * (self.row_h + self.row_gap)) + 20
        return QSize(1100, max(500, h))

    def _timeline_rect(self) -> QRect:
        return QRect(self.left_label_w, self.top_tariff_h + self.axis_h,
                     self.width() - self.left_label_w - self.right_info_w - 12,
                     self.height() - self.top_tariff_h - self.axis_h - 12)

    def _minute_to_x(self, minute: int) -> int:
        tl = self._timeline_rect()
        frac = minute / MINUTES_PER_DAY
        return int(tl.left() + frac * tl.width())

    def _x_to_minute(self, x: int) -> int:
        tl = self._timeline_rect()
        frac = (x - tl.left()) / max(1, tl.width())
        return clamp(int(frac * MINUTES_PER_DAY), 0, MINUTES_PER_DAY - 1)

    def _row_rect(self, idx: int) -> QRect:
        tl = self._timeline_rect()
        y0 = tl.top() + idx * (self.row_h + self.row_gap)
        return QRect(tl.left(), y0, tl.width(), self.row_h)

    def _device_label_rect(self, idx: int) -> QRect:
        rr = self._row_rect(idx)
        return QRect(6, rr.top(), self.left_label_w - 12, rr.height())

    def _device_info_rect(self, idx: int) -> QRect:
        rr = self._row_rect(idx)
        x = rr.right() + 6
        return QRect(x, rr.top(), self.right_info_w - 12, rr.height())

    def _tariff_rect(self) -> QRect:
        return QRect(self.left_label_w, 6, self.width() - self.left_label_w - self.right_info_w - 12, self.top_tariff_h - 8)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), QColor(25, 25, 28))
        p.setFont(self.font)

        if not (self.project and self.settings and self.sim):
            p.setPen(QColor(220, 220, 220))
            p.drawText(self.rect(), Qt.AlignCenter, "No project loaded.")
            return

        # Draw tariff bar
        self._paint_tariff(p)

        # Draw time axis
        self._paint_axis(p)

        # Draw device rows
        for i, dev in enumerate(self.project.devices):
            self._paint_row(p, i, dev)

    def _paint_tariff(self, p: QPainter):
        tr = self._tariff_rect()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(40, 40, 46))
        p.drawRect(tr)

        # base rate visualization
        for hour in range(24):
            m0 = hour * 60
            m1 = (hour + 1) * 60
            r = tariff_rate_for_minute(self.settings, m0)
            # map rate to brightness (simple)
            # bigger rate -> brighter
            base = clamp(int(60 + 400 * r), 60, 220)
            col = QColor(base, base, base)
            x0 = self._minute_to_x(m0)
            x1 = self._minute_to_x(m1)
            p.setBrush(col)
            p.drawRect(QRect(x0, tr.top(), x1 - x0, tr.height()))

        # free overlay
        fr = self.settings.free_rule.normalized()
        if fr.enabled:
            x0 = self._minute_to_x(fr.start_min)
            x1 = self._minute_to_x(fr.end_min)
            overlay = QColor(80, 160, 80, 140)
            p.setBrush(overlay)
            p.drawRect(QRect(x0, tr.top(), x1 - x0, tr.height()))
            p.setPen(QColor(240, 240, 240))
            p.drawText(QRect(x0, tr.top(), x1 - x0, tr.height()), Qt.AlignCenter,
                       f"FREE ≤ {fr.free_kw_threshold:.1f} kW")

        # border + label
        p.setPen(QPen(QColor(120, 120, 130), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(tr)
        p.setPen(QColor(220, 220, 230))
        p.drawText(6, tr.bottom() + 16, "Tariff (visual only — edit in Settings)")

    def _paint_axis(self, p: QPainter):
        tl = self._timeline_rect()
        axis_y = self.top_tariff_h + 4
        p.setPen(QColor(160, 160, 170))
        # hour ticks
        for hour in range(25):
            m = hour * 60
            x = self._minute_to_x(m)
            p.drawLine(x, axis_y + 10, x, axis_y + 20)
            if hour < 24:
                p.drawText(x + 2, axis_y + 9, f"{hour:02d}:00")

        # axis baseline
        p.setPen(QColor(90, 90, 100))
        p.drawLine(tl.left(), axis_y + 20, tl.right(), axis_y + 20)

    def _paint_row(self, p: QPainter, idx: int, dev: Device):
        rr = self._row_rect(idx)
        # row background
        bg = QColor(34, 34, 38) if idx % 2 == 0 else QColor(30, 30, 34)
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRect(rr)

        # label left
        lr = self._device_label_rect(idx)
        p.setPen(QColor(220, 220, 230))
        p.drawText(lr, Qt.AlignVCenter | Qt.AlignLeft,
                   f"{dev.name}\n{dev.dtype} | {dev.power_w:g} W × {dev.quantity}")

        # draw items on timeline
        p.setPen(Qt.NoPen)

        if dev.dtype == DeviceType.ALWAYS:
            self._draw_block(p, rr, 0, MINUTES_PER_DAY, QColor(140, 140, 200, 180), hover=False)

        elif dev.dtype == DeviceType.SCHEDULED:
            for j, iv in enumerate(dev.intervals):
                ivn = iv.normalized()
                hover = (self.hover_hit.kind.startswith("interval") and self.hover_hit.device_index == idx and self.hover_hit.item_index == j)
                self._draw_block(p, rr, ivn.start_min, ivn.end_min, QColor(100, 170, 240, 190), hover=hover)

        elif dev.dtype == DeviceType.EVENTS:
            for j, ev in enumerate(dev.events):
                evn = ev.normalized()
                hover = (self.hover_hit.kind == HitKind.EVENT_BODY and self.hover_hit.device_index == idx and self.hover_hit.item_index == j)
                self._draw_event(p, rr, evn.start_min, evn.duration_min, QColor(240, 170, 90, 200), hover=hover)

        # row border
        p.setPen(QPen(QColor(55, 55, 62), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rr)

        # right info
        info = self._device_info_rect(idx)
        on_min = self.sim.per_device_on_minutes[idx] if self.sim else 0
        kwh = self.sim.per_device_kwh_day[idx] if self.sim else 0.0
        cost = self.sim.per_device_cost_day[idx] if self.sim else 0.0
        p.setPen(QColor(220, 220, 230))
        p.drawText(
            info, Qt.AlignVCenter | Qt.AlignLeft,
            f"On: {on_min/60:.2f} h/day\nEnergy: {kwh:.3f} kWh/day\nCost: {self.settings.currency_symbol}{cost:.2f}/day"
        )

    def _draw_block(self, p: QPainter, rr: QRect, start_min: int, end_min: int, color: QColor, hover: bool):
        x0 = self._minute_to_x(start_min)
        x1 = self._minute_to_x(end_min)
        rect = QRect(x0, rr.top() + 8, max(2, x1 - x0), rr.height() - 16)
        c = QColor(color)
        if hover:
            c.setAlpha(240)
        p.setBrush(c)
        p.drawRect(rect)

        # resize grips
        p.setBrush(QColor(0, 0, 0, 90))
        p.drawRect(QRect(rect.left(), rect.top(), 4, rect.height()))
        p.drawRect(QRect(rect.right() - 3, rect.top(), 4, rect.height()))

    def _draw_event(self, p: QPainter, rr: QRect, start_min: int, duration_min: int, color: QColor, hover: bool):
        x0 = self._minute_to_x(start_min)
        x1 = self._minute_to_x(start_min + duration_min)
        rect = QRect(x0, rr.top() + 10, max(3, x1 - x0), rr.height() - 20)
        c = QColor(color)
        if hover:
            c.setAlpha(240)
        p.setBrush(c)
        p.drawRect(rect)

    # -------------------------
    # Hit testing & interactions
    # -------------------------

    def mouseMoveEvent(self, e):
        if not (self.project and self.settings and self.sim):
            return

        pos = e.position().toPoint()
        if self.dragging:
            self._handle_drag(pos)
            return

        self.hover_hit = self._hit_test(pos)
        # cursor feedback
        if self.hover_hit.kind == HitKind.INTERVAL_LEFT or self.hover_hit.kind == HitKind.INTERVAL_RIGHT:
            self.setCursor(Qt.SizeHorCursor)
        elif self.hover_hit.kind in (HitKind.INTERVAL_BODY, HitKind.EVENT_BODY):
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def mousePressEvent(self, e):
        if not self.project:
            return
        pos = e.position().toPoint()

        if e.button() == Qt.RightButton:
            hit = self._hit_test(pos)
            if hit.kind == HitKind.EVENT_BODY:
                dev = self.project.devices[hit.device_index]
                if 0 <= hit.item_index < len(dev.events):
                    dev.events.pop(hit.item_index)
                    self._trigger_recompute()  # main window call
            elif hit.kind.startswith("interval"):
                dev = self.project.devices[hit.device_index]
                if 0 <= hit.item_index < len(dev.intervals):
                    dev.intervals.pop(hit.item_index)
                    self._trigger_recompute()
            return

        if e.button() != Qt.LeftButton:
            return

        hit = self._hit_test(pos)
        self.hit = hit

        # click empty space on a row to add:
        if hit.kind == HitKind.NONE:
            # identify row if clicked in a row
            di = self._device_index_from_pos(pos)
            if di != -1:
                dev = self.project.devices[di]
                m = self._x_to_minute(pos.x())
                if dev.dtype == DeviceType.SCHEDULED:
                    # add a 30 min block by default
                    dev.intervals.append(Interval(m, min(MINUTES_PER_DAY, m + 30)).normalized())
                    self._trigger_recompute()
                    return
                elif dev.dtype == DeviceType.EVENTS:
                    # add event with default duration 3 minutes, no fixed energy by default
                    dev.events.append(Event(m, 3, None).normalized())
                    self._trigger_recompute()
                    return
                # always-on: do nothing
            return

        # start drag
        self.dragging = True
        self.drag_start_pos = pos
        self.drag_start_min = self._x_to_minute(pos.x())
        self.drag_original = self._snapshot_item(hit)

        if hit.kind in (HitKind.INTERVAL_BODY, HitKind.EVENT_BODY):
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.drag_original = None
            self.hit = HitTest()
            self.setCursor(Qt.ArrowCursor)
            self._trigger_recompute()

    def _trigger_recompute(self):
        win = self.window()
        if hasattr(win, "recompute"):
            win.recompute()
            return
        parent = self.parent()
        if parent and hasattr(parent, "recompute"):
            parent.recompute()

    def _device_index_from_pos(self, pos: QPoint) -> int:
        tl = self._timeline_rect()
        if pos.y() < tl.top() or pos.y() > tl.bottom():
            return -1
        # check each row rect
        for i in range(len(self.project.devices)):
            rr = self._row_rect(i)
            if rr.contains(pos):
                return i
        return -1

    def _hit_test(self, pos: QPoint) -> HitTest:
        di = self._device_index_from_pos(pos)
        if di == -1:
            return HitTest()

        dev = self.project.devices[di]
        rr = self._row_rect(di)

        def near(x, target, px=5):
            return abs(x - target) <= px

        if dev.dtype == DeviceType.SCHEDULED:
            for j, iv in enumerate(dev.intervals):
                ivn = iv.normalized()
                x0 = self._minute_to_x(ivn.start_min)
                x1 = self._minute_to_x(ivn.end_min)
                rect = QRect(x0, rr.top() + 8, max(2, x1 - x0), rr.height() - 16)
                if rect.contains(pos):
                    # edges?
                    if near(pos.x(), rect.left()):
                        return HitTest(HitKind.INTERVAL_LEFT, di, j, self._x_to_minute(pos.x()))
                    if near(pos.x(), rect.right()):
                        return HitTest(HitKind.INTERVAL_RIGHT, di, j, self._x_to_minute(pos.x()))
                    return HitTest(HitKind.INTERVAL_BODY, di, j, self._x_to_minute(pos.x()))

        if dev.dtype == DeviceType.EVENTS:
            for j, ev in enumerate(dev.events):
                evn = ev.normalized()
                x0 = self._minute_to_x(evn.start_min)
                x1 = self._minute_to_x(evn.start_min + evn.duration_min)
                rect = QRect(x0, rr.top() + 10, max(3, x1 - x0), rr.height() - 20)
                if rect.contains(pos):
                    return HitTest(HitKind.EVENT_BODY, di, j, self._x_to_minute(pos.x()))

        if dev.dtype == DeviceType.ALWAYS:
            # optionally allow click to do nothing
            return HitTest()

        return HitTest()

    def _snapshot_item(self, hit: HitTest):
        dev = self.project.devices[hit.device_index]
        if hit.kind.startswith("interval"):
            iv = dev.intervals[hit.item_index].normalized()
            return ("interval", iv.start_min, iv.end_min)
        if hit.kind == HitKind.EVENT_BODY:
            ev = dev.events[hit.item_index].normalized()
            return ("event", ev.start_min, ev.duration_min, ev.energy_wh)
        return None

    def _handle_drag(self, pos: QPoint):
        if not self.project or not self.drag_original:
            return

        hit = self.hit
        dev = self.project.devices[hit.device_index]
        m_now = self._x_to_minute(pos.x())
        delta = m_now - self.drag_start_min

        if self.drag_original[0] == "interval":
            _, s0, e0 = self.drag_original
            if hit.kind == HitKind.INTERVAL_BODY:
                s = clamp(s0 + delta, 0, MINUTES_PER_DAY - 1)
                length = e0 - s0
                e = clamp(s + length, 1, MINUTES_PER_DAY)
                # if clamped at end, pull start back
                if e - s < length:
                    s = max(0, e - length)
                dev.intervals[hit.item_index] = Interval(s, e).normalized()

            elif hit.kind == HitKind.INTERVAL_LEFT:
                s = clamp(s0 + delta, 0, e0 - 1)
                dev.intervals[hit.item_index] = Interval(s, e0).normalized()

            elif hit.kind == HitKind.INTERVAL_RIGHT:
                e = clamp(e0 + delta, s0 + 1, MINUTES_PER_DAY)
                dev.intervals[hit.item_index] = Interval(s0, e).normalized()

        elif self.drag_original[0] == "event":
            _, s0, dur, ewh = self.drag_original
            if hit.kind == HitKind.EVENT_BODY:
                s = clamp(s0 + delta, 0, MINUTES_PER_DAY - 1)
                # enforce split at midnight by clamping duration
                if s + dur > MINUTES_PER_DAY:
                    s = MINUTES_PER_DAY - dur
                dev.events[hit.item_index] = Event(s, dur, ewh).normalized()

        self.update()


# =========================
# Main window UI
# =========================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Power & Cost Timeline Simulator")

        self.qs = QSettings("JumbleSaleOfStimuli", "PowerCostTimeline")
        self.settings_model = SettingsModel.from_qsettings(self.qs)

        # project state
        self.project = Project(devices=[
            Device(name="Router", dtype=DeviceType.ALWAYS, power_w=10.0, quantity=1),
            Device(name="Grow light", dtype=DeviceType.SCHEDULED, power_w=120.0, quantity=1,
                   intervals=[Interval(8*60, 18*60)]),
            Device(name="Kettle", dtype=DeviceType.EVENTS, power_w=2400.0, quantity=1,
                   events=[Event(7*60+15, 3, None), Event(18*60+40, 3, None)]),
        ])
        self.current_file: Optional[str] = None

        # central layout
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # left panel: device table + controls + custom periods
        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.device_table = QTableWidget(0, 4)
        self.device_table.setHorizontalHeaderLabels(["Name", "Type", "Power (W)", "Qty"])
        self.device_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.device_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.device_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.device_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.device_table.verticalHeader().setVisible(False)
        self.device_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.device_table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed | QTableWidget.SelectedClicked)
        left_layout.addWidget(QLabel("Devices"))
        left_layout.addWidget(self.device_table)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add device")
        self.btn_del = QPushButton("Remove device")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_del)
        left_layout.addLayout(btn_row)

        left_layout.addWidget(QLabel("Custom periods (days)"))

        self.period_list = QTableWidget(0, 2)
        self.period_list.setHorizontalHeaderLabels(["Name", "Days"])
        self.period_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.period_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.period_list.verticalHeader().setVisible(False)
        left_layout.addWidget(self.period_list)

        pr_btn = QHBoxLayout()
        self.btn_add_period = QPushButton("Add period")
        self.btn_del_period = QPushButton("Remove period")
        pr_btn.addWidget(self.btn_add_period)
        pr_btn.addWidget(self.btn_del_period)
        left_layout.addLayout(pr_btn)

        splitter.addWidget(left)

        # right panel: timeline + summary
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.timeline = TimelineWidget(parent=self)
        right_layout.addWidget(self.timeline, stretch=1)

        self.summary = QLabel("")
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary.setStyleSheet("QLabel { padding: 8px; background: #1e1e22; color: #e8e8ee; }")
        right_layout.addWidget(self.summary, stretch=0)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 900])

        # menu
        self._build_menu()

        # wire signals
        self.btn_add.clicked.connect(self.add_device)
        self.btn_del.clicked.connect(self.remove_device)
        self.btn_add_period.clicked.connect(self.add_period)
        self.btn_del_period.clicked.connect(self.remove_period)

        self.device_table.itemChanged.connect(self.on_device_table_changed)
        self.period_list.itemChanged.connect(self.on_periods_changed)

        # first run -> show settings
        if not SettingsModel.has_run_before(self.qs):
            self.open_settings(first_run=True)

        self.refresh_tables()
        self.recompute()

    def _build_menu(self):
        mb = self.menuBar()

        filem = mb.addMenu("&File")
        act_new = QAction("New", self)
        act_open = QAction("Open…", self)
        act_save = QAction("Save", self)
        act_save_as = QAction("Save As…", self)
        filem.addAction(act_new)
        filem.addAction(act_open)
        filem.addSeparator()
        filem.addAction(act_save)
        filem.addAction(act_save_as)

        act_new.triggered.connect(self.new_project)
        act_open.triggered.connect(self.open_project)
        act_save.triggered.connect(self.save_project)
        act_save_as.triggered.connect(self.save_project_as)

        editm = mb.addMenu("&Edit")
        act_settings = QAction("Settings…", self)
        editm.addAction(act_settings)
        act_settings.triggered.connect(self.open_settings)

        helpm = mb.addMenu("&Help")
        act_about = QAction("About", self)
        helpm.addAction(act_about)
        act_about.triggered.connect(self.about)

    def about(self):
        QMessageBox.information(
            self, "About",
            "Power & Cost Timeline Simulator (MVP)\n"
            "• 1-minute simulation step\n"
            "• 30-day months\n"
            "• Drag/resize schedule blocks, add/move/delete events\n"
            "• Tariff edited in Settings\n"
        )

    # ---------------------
    # Project I/O
    # ---------------------

    def new_project(self):
        self.project = Project(devices=[])
        self.current_file = None
        self.refresh_tables()
        self.recompute()

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.project = Project.from_dict(d)
            self.current_file = path
            self.refresh_tables()
            self.recompute()
        except Exception as ex:
            QMessageBox.critical(self, "Open failed", f"Could not open:\n{ex}")

    def save_project(self):
        if not self.current_file:
            return self.save_project_as()
        self._save_to_path(self.current_file)

    def save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save project as", "", "JSON (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        self.current_file = path
        self._save_to_path(path)

    def _save_to_path(self, path: str):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.project.to_dict(), f, indent=2)
        except Exception as ex:
            QMessageBox.critical(self, "Save failed", f"Could not save:\n{ex}")

    # ---------------------
    # Settings
    # ---------------------

    def open_settings(self, first_run: bool = False):
        dlg = SettingsDialog(self, self.settings_model)
        if dlg.exec() == QDialog.Accepted:
            self.settings_model = dlg.get_model()
            self.settings_model.to_qsettings(self.qs)
            self.recompute()
        elif first_run:
            # even if they cancel on first run, write defaults so it doesn't pop every time
            self.settings_model.to_qsettings(self.qs)

    # ---------------------
    # Tables <-> model sync
    # ---------------------

    def refresh_tables(self):
        # devices
        self.device_table.blockSignals(True)
        self.device_table.setRowCount(len(self.project.devices))
        for r, dev in enumerate(self.project.devices):
            it_name = QTableWidgetItem(dev.name)
            self.device_table.setItem(r, 0, it_name)

            cb = QComboBox()
            cb.addItems([DeviceType.ALWAYS, DeviceType.SCHEDULED, DeviceType.EVENTS])
            cb.setCurrentText(dev.dtype)
            cb.currentTextChanged.connect(lambda _txt, row=r: self._on_device_type_changed(row))
            self.device_table.setCellWidget(r, 1, cb)

            it_p = QTableWidgetItem(f"{dev.power_w:g}")
            self.device_table.setItem(r, 2, it_p)

            it_q = QTableWidgetItem(str(dev.quantity))
            self.device_table.setItem(r, 3, it_q)

        self.device_table.blockSignals(False)

        # periods
        self.period_list.blockSignals(True)
        self.period_list.setRowCount(len(self.project.custom_periods))
        for r, (name, days) in enumerate(self.project.custom_periods):
            self.period_list.setItem(r, 0, QTableWidgetItem(name))
            self.period_list.setItem(r, 1, QTableWidgetItem(f"{days:g}"))
        self.period_list.blockSignals(False)

    def _on_device_type_changed(self, row: int):
        if not (0 <= row < len(self.project.devices)):
            return
        cb: QComboBox = self.device_table.cellWidget(row, 1)
        dtype = cb.currentText()
        self.project.devices[row].dtype = dtype
        self.recompute()

    def on_device_table_changed(self, item: QTableWidgetItem):
        r, c = item.row(), item.column()
        if not (0 <= r < len(self.project.devices)):
            return
        dev = self.project.devices[r]
        try:
            if c == 0:
                dev.name = item.text().strip() or dev.name
            elif c == 2:
                dev.power_w = max(0.0, float(item.text()))
            elif c == 3:
                dev.quantity = max(1, int(float(item.text())))
        except Exception:
            pass
        self.recompute()

    def on_periods_changed(self, item: QTableWidgetItem):
        r, c = item.row(), item.column()
        if not (0 <= r < len(self.project.custom_periods)):
            return
        name, days = self.project.custom_periods[r]
        try:
            if c == 0:
                name = item.text().strip() or name
            elif c == 1:
                days = float(item.text())
        except Exception:
            pass
        self.project.custom_periods[r] = (name, days)
        self.recompute()

    # ---------------------
    # Buttons
    # ---------------------

    def add_device(self):
        self.project.devices.append(Device(name="Device", dtype=DeviceType.SCHEDULED, power_w=20.0, quantity=1))
        self.refresh_tables()
        self.recompute()

    def remove_device(self):
        rows = sorted({i.row() for i in self.device_table.selectedItems()}, reverse=True)
        if not rows:
            return
        for r in rows:
            if 0 <= r < len(self.project.devices):
                self.project.devices.pop(r)
        self.refresh_tables()
        self.recompute()

    def add_period(self):
        self.project.custom_periods.append(("Period", 7.0))
        self.refresh_tables()
        self.recompute()

    def remove_period(self):
        rows = sorted({i.row() for i in self.period_list.selectedItems()}, reverse=True)
        if not rows:
            return
        for r in rows:
            if 0 <= r < len(self.project.custom_periods):
                self.project.custom_periods.pop(r)
        self.refresh_tables()
        self.recompute()

    # ---------------------
    # Recompute & summary
    # ---------------------

    def recompute(self):
        sim = simulate_day(self.project, self.settings_model)
        self.timeline.set_data(self.project, self.settings_model, sim)
        self._update_summary(sim)

    def _update_summary(self, sim: SimResult):
        cs = self.settings_model.currency_symbol
        day_kwh = sim.total_kwh_day
        day_cost = sim.total_cost_day

        week_kwh = day_kwh * 7
        week_cost = day_cost * 7

        month_kwh = day_kwh * 30
        month_cost = day_cost * 30

        year_kwh = day_kwh * 365
        year_cost = day_cost * 365

        lines = []
        lines.append(f"TOTAL (from timeline + tariff)")
        lines.append(f"Day:   {day_kwh:.3f} kWh   |   {cs}{day_cost:.2f}")
        lines.append(f"Week:  {week_kwh:.3f} kWh  |   {cs}{week_cost:.2f}")
        lines.append(f"Month (30d): {month_kwh:.3f} kWh  |   {cs}{month_cost:.2f}")
        lines.append(f"Year (365d): {year_kwh:.3f} kWh  |   {cs}{year_cost:.2f}")
        lines.append("")
        lines.append("Custom periods:")
        for name, days in self.project.custom_periods:
            try:
                d = float(days)
            except Exception:
                d = 1.0
            lines.append(f"• {name} ({d:g} days): {day_kwh*d:.3f} kWh | {cs}{day_cost*d:.2f}")

        self.summary.setText("\n".join(lines))


# =========================
# App entry point
# =========================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.resize(1280, 760)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
