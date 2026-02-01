import copy
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict

from PySide6.QtCore import (
    Qt, QRect, QRectF, QPoint, QPointF, QSize, QSettings, QTimer
)
from PySide6.QtGui import (
    QAction, QBrush, QColor, QFont, QPainter, QPen, QCursor, QFontMetrics
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel,
    QFileDialog, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QDialog, QFormLayout, QDialogButtonBox, QGroupBox, QCheckBox,
    QLineEdit, QHeaderView, QListWidget, QListWidgetItem, QScrollArea
)

# =========================
# Models
# =========================

MINUTES_PER_DAY = 24 * 60


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def is_minute_in_window(minute: int, start_min: int, end_min: int) -> bool:
    if start_min == end_min:
        return False
    if start_min < end_min:
        return start_min <= minute < end_min
    return minute >= start_min or minute < end_min


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


def format_duration_minutes(m: int) -> str:
    m = clamp(int(m), 1, MINUTES_PER_DAY)
    hh = m // 60
    mm = m % 60
    return f"{hh:d}:{mm:02d}"


def parse_duration_text(text: str) -> Optional[int]:
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        if ":" in cleaned:
            parts = cleaned.split(":")
            if len(parts) != 2:
                return None
            hours = int(parts[0]) if parts[0] else 0
            minutes = int(parts[1]) if parts[1] else 0
            if minutes < 0:
                return None
            hours += minutes // 60
            minutes = minutes % 60
            total = hours * 60 + minutes
        else:
            hours = float(cleaned)
            if hours < 0:
                return None
            total = int(hours * 60)
        if total >= MINUTES_PER_DAY:
            return MINUTES_PER_DAY
        return max(1, total)
    except Exception:
        return None


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return cleaned.strip("-")


def derive_modified_path(path: str) -> str:
    if path.lower().endswith("_modified.json"):
        return path
    base, ext = os.path.splitext(path)
    if ext.lower() != ".json":
        ext = ".json"
    return f"{base}_modified{ext}"


def parse_hhmm(text: str) -> Optional[int]:
    text = text.strip()
    if not re.match(r"^\d{2}:\d{2}$", text):
        return None
    hh = int(text[:2])
    mm = int(text[3:])
    if hh < 0 or hh > 24 or mm < 0 or mm > 59:
        return None
    if hh == 24 and mm != 0:
        return None
    return hh * 60 + mm


def is_all_days(day_sets: List[dict]) -> bool:
    if not day_sets:
        return True
    all_days = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
    for day_set in day_sets:
        days = set(day_set.get("days", []))
        if all_days.issubset(days):
            return True
    return False


def is_all_day_rate(rate: dict) -> bool:
    schedule = rate.get("schedule") or {}
    day_sets = schedule.get("day_sets", [])
    if not is_all_days(day_sets):
        return False
    for time_range in schedule.get("time_ranges", []):
        start = parse_hhmm(time_range.get("start", ""))
        end = parse_hhmm(time_range.get("end", ""))
        if start == 0 and end == MINUTES_PER_DAY:
            return True
    return False


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
    enabled: bool = True
    default_duration_min: int = 30
    variable: bool = True

    intervals: List[Interval] = field(default_factory=list)  # for scheduled
    events: List[Event] = field(default_factory=list)        # for per-use

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "power_w": self.power_w,
            "enabled": self.enabled,
            "default_duration_min": self.default_duration_min,
            "variable": self.variable,
            "intervals": [asdict(i) for i in self.intervals],
            "events": [asdict(e) for e in self.events],
        }

    def apply_usage_settings(self):
        self.default_duration_min = clamp(int(self.default_duration_min), 1, MINUTES_PER_DAY)
        if self.default_duration_min >= MINUTES_PER_DAY:
            self.dtype = DeviceType.ALWAYS
            return
        self.dtype = DeviceType.SCHEDULED if self.variable else DeviceType.EVENTS

    @staticmethod
    def from_dict(d: dict) -> "Device":
        dev = Device(
            name=d.get("name", "Device"),
            dtype=d.get("dtype", DeviceType.SCHEDULED),
            power_w=float(d.get("power_w", 20.0)),
        )
        dev.enabled = parse_bool(d.get("enabled", True))
        dev.variable = parse_bool(d.get("variable", dev.variable))
        if "default_duration_min" in d:
            dev.default_duration_min = int(d.get("default_duration_min", dev.default_duration_min))
        else:
            if dev.dtype == DeviceType.ALWAYS:
                dev.default_duration_min = MINUTES_PER_DAY
                dev.variable = False
            elif dev.dtype == DeviceType.EVENTS:
                dev.default_duration_min = 3
                dev.variable = False
            else:
                dev.default_duration_min = 30
                dev.variable = True
        dev.intervals = [Interval(**i).normalized() for i in d.get("intervals", [])]
        dev.events = [Event(**e).normalized() for e in d.get("events", [])]
        dev.apply_usage_settings()
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
class CustomPeriod:
    name: str = "Period"
    duration: float = 7.0
    unit: str = "days"
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration": self.duration,
            "unit": self.unit,
            "enabled": self.enabled,
        }


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

    # Optional imported tariff schedule
    tariff_minute_rates: Optional[List[float]] = None
    tariff_label: Optional[str] = None

    def clone(self) -> "SettingsModel":
        return SettingsModel(
            currency_symbol=self.currency_symbol,
            month_days=self.month_days,
            step_minutes=self.step_minutes,
            base_rate_flat=self.base_rate_flat,
            use_time_of_day=self.use_time_of_day,
            offpeak_rate=self.offpeak_rate,
            offpeak_start_min=self.offpeak_start_min,
            offpeak_end_min=self.offpeak_end_min,
            free_rule=FreeRule(
                enabled=self.free_rule.enabled,
                start_min=self.free_rule.start_min,
                end_min=self.free_rule.end_min,
                free_kw_threshold=self.free_rule.free_kw_threshold,
            ),
            tariff_minute_rates=list(self.tariff_minute_rates) if self.tariff_minute_rates else None,
            tariff_label=self.tariff_label,
        )

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
    custom_periods: List["CustomPeriod"] = field(default_factory=lambda: [
        CustomPeriod(name="Lettuce", duration=45.0, unit="days", enabled=True)
    ])

    def to_dict(self) -> dict:
        return {
            "devices": [d.to_dict() for d in self.devices],
            "custom_periods": [cp.to_dict() for cp in self.custom_periods],
        }

    @staticmethod
    def from_dict(d: dict) -> "Project":
        p = Project()
        p.devices = [Device.from_dict(x) for x in d.get("devices", [])]
        cps: List[CustomPeriod] = []
        for item in d.get("custom_periods", []):
            try:
                if "days" in item and "duration" not in item:
                    cps.append(CustomPeriod(
                        name=item.get("name", "Period"),
                        duration=float(item.get("days", 1.0)),
                        unit="days",
                        enabled=parse_bool(item.get("enabled", True)),
                    ))
                else:
                    cps.append(CustomPeriod(
                        name=item.get("name", "Period"),
                        duration=float(item.get("duration", 1.0)),
                        unit=item.get("unit", "days"),
                        enabled=parse_bool(item.get("enabled", True)),
                    ))
            except Exception:
                pass
        if cps:
            p.custom_periods = cps
        return p


@dataclass
class TariffDocument:
    source_path: str
    save_path: str
    data: dict


@dataclass
class TariffEntry:
    document: TariffDocument
    supplier_index: int
    tariff_index: int

    def supplier(self) -> dict:
        return self.document.data["suppliers"][self.supplier_index]

    def tariff(self) -> dict:
        return self.supplier()["tariffs"][self.tariff_index]


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
    if settings.tariff_minute_rates:
        return settings.tariff_minute_rates[clamp(minute, 0, MINUTES_PER_DAY - 1)]
    if not settings.use_time_of_day:
        return settings.base_rate_flat
    if is_minute_in_window(minute, settings.offpeak_start_min, settings.offpeak_end_min):
        return settings.offpeak_rate
    return settings.base_rate_flat


def tariff_segments(settings: SettingsModel) -> List[Tuple[int, int, float]]:
    if settings.tariff_minute_rates:
        segments: List[Tuple[int, int, float]] = []
        current_rate = settings.tariff_minute_rates[0]
        start = 0
        for minute in range(1, MINUTES_PER_DAY + 1):
            next_rate = None if minute == MINUTES_PER_DAY else settings.tariff_minute_rates[minute]
            if minute == MINUTES_PER_DAY or next_rate != current_rate:
                segments.append((start, minute, current_rate))
                start = minute
                current_rate = next_rate
        return segments
    if not settings.use_time_of_day:
        return [(0, MINUTES_PER_DAY, settings.base_rate_flat)]
    segments: List[Tuple[int, int, float]] = []
    current_rate = tariff_rate_for_minute(settings, 0)
    start = 0
    for minute in range(1, MINUTES_PER_DAY + 1):
        if minute == MINUTES_PER_DAY:
            next_rate = None
        else:
            next_rate = tariff_rate_for_minute(settings, minute)
        if minute == MINUTES_PER_DAY or next_rate != current_rate:
            segments.append((start, minute, current_rate))
            start = minute
            current_rate = next_rate
    return segments


def build_tariff_minute_rates(tariff: dict) -> List[float]:
    minute_rates = [0.0] * MINUTES_PER_DAY
    minute_priority = [-10**9] * MINUTES_PER_DAY
    rates = tariff.get("rates") or []
    for rate in rates:
        rate_value = rate.get("rate_gbp_per_kwh")
        if rate_value is None:
            continue
        priority = int(rate.get("priority", 0))
        schedule = rate.get("schedule") or {}
        time_ranges = schedule.get("time_ranges") or []
        if not time_ranges:
            continue
        for time_range in time_ranges:
            start = parse_hhmm(time_range.get("start", ""))
            end = parse_hhmm(time_range.get("end", ""))
            if start is None or end is None or start == end:
                continue
            for minute in range(MINUTES_PER_DAY):
                if is_minute_in_window(minute, start, end):
                    if priority >= minute_priority[minute]:
                        minute_priority[minute] = priority
                        minute_rates[minute] = float(rate_value)
    return minute_rates


def tariff_headline(tariff: dict, currency_symbol: str) -> str:
    rates = tariff.get("rates") or []
    if tariff.get("complicated") or not rates:
        return "Complicated"
    if all(rate.get("rate_gbp_per_kwh") is None for rate in rates):
        return "Complicated"
    if tariff.get("pricing_model") == "flat_or_tou":
        for rate in rates:
            if rate.get("rate_gbp_per_kwh") is None:
                continue
            if is_all_day_rate(rate):
                return f"{currency_symbol}{float(rate['rate_gbp_per_kwh']):.4f}/kWh"
        return "TOU"
    return "Complicated"


def load_tariff_document(path: str) -> TariffDocument:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema_version") != "uk-tariffs-1.0":
        raise ValueError("Unsupported tariff schema version.")
    save_path = derive_modified_path(path)
    return TariffDocument(source_path=path, save_path=save_path, data=data)


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
        if not dev.enabled:
            continue
        base_power_w = max(0.0, float(dev.power_w))

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
                    pw = max(0.0, pw)
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


def custom_period_to_days(period: CustomPeriod, settings: SettingsModel) -> float:
    duration = float(period.duration)
    if period.unit == "mins":
        return duration / (60.0 * 24.0)
    if period.unit == "hours":
        return duration / 24.0
    if period.unit == "days":
        return duration
    if period.unit == "months":
        return duration * settings.month_days
    if period.unit == "years":
        return duration * 365.0
    return duration


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
        m = self.model.clone()
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
# Custom periods dialog
# =========================

CUSTOM_PERIOD_UNITS: List[Tuple[str, str]] = [
    ("mins", "Mins"),
    ("hours", "Hours"),
    ("days", "Days"),
    ("months", "Months"),
    ("years", "Years"),
]


class CustomPeriodsDialog(QDialog):
    def __init__(self, parent=None, periods: Optional[List[CustomPeriod]] = None):
        super().__init__(parent)
        self.setWindowTitle("Custom Periods")
        self.setModal(True)
        self._periods = [
            CustomPeriod(
                name=cp.name,
                duration=cp.duration,
                unit=cp.unit,
                enabled=cp.enabled,
            )
            for cp in (periods or [])
        ]

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Enabled", "Name", "Duration", "Unit"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed | QTableWidget.SelectedClicked)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.btn_add_period = QPushButton("Add period")
        self.btn_del_period = QPushButton("Delete selected")
        btn_row.addWidget(self.btn_add_period)
        btn_row.addWidget(self.btn_del_period)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.table.itemChanged.connect(self._on_item_changed)
        self.btn_add_period.clicked.connect(self._add_period)
        self.btn_del_period.clicked.connect(self._remove_selected)

        self._refresh_table()

    def _refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._periods))
        for row, period in enumerate(self._periods):
            enabled_item = QTableWidgetItem("")
            enabled_item.setFlags(enabled_item.flags() | Qt.ItemIsUserCheckable)
            enabled_item.setFlags(enabled_item.flags() & ~Qt.ItemIsEditable)
            enabled_item.setCheckState(Qt.Checked if period.enabled else Qt.Unchecked)
            self.table.setItem(row, 0, enabled_item)

            name_item = QTableWidgetItem(period.name)
            self.table.setItem(row, 1, name_item)

            duration_item = QTableWidgetItem(f"{period.duration:g}")
            self.table.setItem(row, 2, duration_item)

            unit_box = QComboBox()
            for value, label in CUSTOM_PERIOD_UNITS:
                unit_box.addItem(label, value)
            unit_index = next(
                (i for i, (value, _label) in enumerate(CUSTOM_PERIOD_UNITS) if value == period.unit),
                2,
            )
            unit_box.setCurrentIndex(unit_index)
            unit_box.currentIndexChanged.connect(lambda _idx, r=row: self._on_unit_changed(r))
            self.table.setCellWidget(row, 3, unit_box)
        self.table.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem):
        row, col = item.row(), item.column()
        if not (0 <= row < len(self._periods)):
            return
        period = self._periods[row]
        if col == 0:
            period.enabled = (item.checkState() == Qt.Checked)
        elif col == 1:
            period.name = item.text().strip() or period.name
        elif col == 2:
            try:
                period.duration = max(0.0, float(item.text()))
            except Exception:
                self.table.blockSignals(True)
                item.setText(f"{period.duration:g}")
                self.table.blockSignals(False)

    def _on_unit_changed(self, row: int):
        if not (0 <= row < len(self._periods)):
            return
        box: QComboBox = self.table.cellWidget(row, 3)
        self._periods[row].unit = box.currentData()

    def _add_period(self):
        self._periods.append(CustomPeriod(name="Period", duration=7.0, unit="days", enabled=True))
        self._refresh_table()
        self.table.selectRow(len(self._periods) - 1)

    def _remove_selected(self):
        rows = sorted({item.row() for item in self.table.selectedItems()}, reverse=True)
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self._periods):
                self._periods.pop(row)
        self._refresh_table()

    def get_periods(self) -> List[CustomPeriod]:
        return [
            CustomPeriod(
                name=cp.name,
                duration=cp.duration,
                unit=cp.unit,
                enabled=cp.enabled,
            )
            for cp in self._periods
        ]


# =========================
# Tariff dialogs
# =========================

class TariffImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Tariffs")
        self.setModal(True)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        browse_btn = QPushButton("Browse…")
        path_row.addWidget(self.path_edit, stretch=1)
        path_row.addWidget(browse_btn)
        form.addRow("Tariff JSON file", path_row)
        layout.addLayout(form)

        self.append_checkbox = QCheckBox("Append to current tariff list")
        layout.addWidget(self.append_checkbox)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        browse_btn.clicked.connect(self._browse)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import tariffs", "", "JSON (*.json)")
        if path:
            self.path_edit.setText(path)

    def get_result(self) -> Tuple[Optional[str], bool]:
        path = self.path_edit.text().strip()
        if not path:
            return None, False
        return path, self.append_checkbox.isChecked()


class TariffRateEditor(QGroupBox):
    def __init__(self, rate: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.setTitle("Rate")
        self._rate = copy.deepcopy(rate) if rate else {}
        self._extra_day_sets = []
        self._build_ui()
        self._load_rate()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.rate_name = QLineEdit()
        form.addRow("Rate name", self.rate_name)

        priority_row = QHBoxLayout()
        self.use_priority = QCheckBox("Use priority")
        self.priority_value = QSpinBox()
        self.priority_value.setRange(-1000, 1000)
        priority_row.addWidget(self.use_priority)
        priority_row.addWidget(self.priority_value)
        form.addRow("Priority", priority_row)

        rate_row = QHBoxLayout()
        self.has_rate = QCheckBox("Has rate")
        self.rate_value = QDoubleSpinBox()
        self.rate_value.setDecimals(4)
        self.rate_value.setRange(0.0, 10.0)
        self.rate_value.setSingleStep(0.01)
        rate_row.addWidget(self.has_rate)
        rate_row.addWidget(self.rate_value)
        form.addRow("Rate (GBP/kWh)", rate_row)

        layout.addLayout(form)

        day_group = QGroupBox("Days (first day set)")
        day_layout = QHBoxLayout(day_group)
        self.day_checks: Dict[str, QCheckBox] = {}
        for day in ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]:
            cb = QCheckBox(day.capitalize()[:3])
            self.day_checks[day] = cb
            day_layout.addWidget(cb)
        layout.addWidget(day_group)

        ranges_group = QGroupBox("Time ranges")
        ranges_layout = QVBoxLayout(ranges_group)
        self.ranges_table = QTableWidget(0, 2)
        self.ranges_table.setHorizontalHeaderLabels(["Start (HH:MM)", "End (HH:MM)"])
        self.ranges_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ranges_table.verticalHeader().setVisible(False)
        ranges_layout.addWidget(self.ranges_table)

        btn_row = QHBoxLayout()
        self.btn_add_range = QPushButton("Add time range")
        self.btn_del_range = QPushButton("Remove selected")
        btn_row.addWidget(self.btn_add_range)
        btn_row.addWidget(self.btn_del_range)
        ranges_layout.addLayout(btn_row)
        layout.addWidget(ranges_group)

        self.btn_add_range.clicked.connect(self._add_time_range)
        self.btn_del_range.clicked.connect(self._remove_time_ranges)
        self.has_rate.toggled.connect(self.rate_value.setEnabled)
        self.use_priority.toggled.connect(self.priority_value.setEnabled)

    def _load_rate(self):
        self.rate_name.setText(self._rate.get("rate_name", ""))
        priority = self._rate.get("priority")
        if priority is None:
            self.use_priority.setChecked(False)
            self.priority_value.setValue(0)
        else:
            self.use_priority.setChecked(True)
            self.priority_value.setValue(int(priority))
        rate_value = self._rate.get("rate_gbp_per_kwh")
        if rate_value is None:
            self.has_rate.setChecked(False)
            self.rate_value.setValue(0.0)
        else:
            self.has_rate.setChecked(True)
            self.rate_value.setValue(float(rate_value))

        schedule = self._rate.get("schedule") or {}
        day_sets = schedule.get("day_sets") or []
        first_days = []
        if day_sets:
            first_days = day_sets[0].get("days", [])
            self._extra_day_sets = day_sets[1:]
        for day, cb in self.day_checks.items():
            cb.setChecked(day in first_days if first_days else True)

        self.ranges_table.setRowCount(0)
        for time_range in schedule.get("time_ranges", []) or []:
            self._append_time_range(time_range.get("start", "00:00"), time_range.get("end", "01:00"))

        self.rate_value.setEnabled(self.has_rate.isChecked())
        self.priority_value.setEnabled(self.use_priority.isChecked())

    def _append_time_range(self, start: str, end: str):
        row = self.ranges_table.rowCount()
        self.ranges_table.insertRow(row)
        self.ranges_table.setItem(row, 0, QTableWidgetItem(start))
        self.ranges_table.setItem(row, 1, QTableWidgetItem(end))

    def _add_time_range(self):
        self._append_time_range("00:00", "01:00")

    def _remove_time_ranges(self):
        rows = sorted({item.row() for item in self.ranges_table.selectedItems()}, reverse=True)
        for row in rows:
            if 0 <= row < self.ranges_table.rowCount():
                self.ranges_table.removeRow(row)

    def validate(self) -> Optional[str]:
        if self.ranges_table.rowCount() == 0:
            return "Each rate needs at least one time range."
        for row in range(self.ranges_table.rowCount()):
            start_item = self.ranges_table.item(row, 0)
            end_item = self.ranges_table.item(row, 1)
            start_text = start_item.text().strip() if start_item else ""
            end_text = end_item.text().strip() if end_item else ""
            start = parse_hhmm(start_text)
            end = parse_hhmm(end_text)
            if start is None or end is None:
                return "Time ranges must be in HH:MM format."
            if start == end:
                return "Time range end must be different from start."
        return None

    def get_rate(self) -> dict:
        rate = copy.deepcopy(self._rate)
        rate["rate_name"] = self.rate_name.text().strip() or rate.get("rate_name", "Rate")

        if self.use_priority.isChecked():
            rate["priority"] = int(self.priority_value.value())
        else:
            rate.pop("priority", None)

        if self.has_rate.isChecked():
            rate["rate_gbp_per_kwh"] = float(self.rate_value.value())
        else:
            rate["rate_gbp_per_kwh"] = None

        days = [day for day, cb in self.day_checks.items() if cb.isChecked()]
        if not days:
            days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        day_sets = [{"days": days}] + self._extra_day_sets

        time_ranges = []
        for row in range(self.ranges_table.rowCount()):
            start = self.ranges_table.item(row, 0).text().strip()
            end = self.ranges_table.item(row, 1).text().strip()
            time_ranges.append({"start": start, "end": end})

        rate["schedule"] = {
            "type": "weekly",
            "day_sets": day_sets,
            "time_ranges": time_ranges,
        }
        return rate


class TariffEditorDialog(QDialog):
    def __init__(self, parent=None, supplier_name: str = "", tariff: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Tariff Editor")
        self.setModal(True)
        self._original_supplier_name = supplier_name
        self._original_tariff = copy.deepcopy(tariff) if tariff else {}
        self._rates: List[TariffRateEditor] = []

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.supplier_name = QLineEdit(supplier_name)
        form.addRow("Supplier name", self.supplier_name)

        self.tariff_name = QLineEdit(self._original_tariff.get("tariff_name", ""))
        form.addRow("Tariff name", self.tariff_name)

        self.complicated = QCheckBox("Complicated tariff")
        self.complicated.setChecked(bool(self._original_tariff.get("complicated", False)))
        form.addRow(self.complicated)

        self.complication_notes = QLineEdit(self._original_tariff.get("complication_notes") or "")
        form.addRow("Complication notes", self.complication_notes)

        standing_row = QHBoxLayout()
        self.has_standing = QCheckBox("Has standing charge")
        self.standing_value = QDoubleSpinBox()
        self.standing_value.setDecimals(4)
        self.standing_value.setRange(0.0, 10.0)
        self.standing_value.setSingleStep(0.01)
        standing_row.addWidget(self.has_standing)
        standing_row.addWidget(self.standing_value)
        form.addRow("Standing charge (GBP/day)", standing_row)

        layout.addLayout(form)

        rates_group = QGroupBox("Rates")
        rates_layout = QVBoxLayout(rates_group)
        self.rates_container = QWidget()
        self.rates_container_layout = QVBoxLayout(self.rates_container)
        self.rates_container_layout.setContentsMargins(0, 0, 0, 0)
        self.rates_container_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.rates_container)
        rates_layout.addWidget(scroll)

        btn_row = QHBoxLayout()
        self.btn_add_rate = QPushButton("Add rate")
        self.btn_del_rate = QPushButton("Remove last rate")
        btn_row.addWidget(self.btn_add_rate)
        btn_row.addWidget(self.btn_del_rate)
        rates_layout.addLayout(btn_row)
        layout.addWidget(rates_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.btn_add_rate.clicked.connect(self._add_rate)
        self.btn_del_rate.clicked.connect(self._remove_rate)
        self.has_standing.toggled.connect(self.standing_value.setEnabled)

        self._load_tariff()

    def _load_tariff(self):
        standing = self._original_tariff.get("standing_charge_gbp_per_day")
        if standing is None:
            self.has_standing.setChecked(False)
            self.standing_value.setValue(0.0)
        else:
            self.has_standing.setChecked(True)
            self.standing_value.setValue(float(standing))
        self.standing_value.setEnabled(self.has_standing.isChecked())

        for rate in self._original_tariff.get("rates", []) or []:
            self._append_rate(rate)

    def _append_rate(self, rate: Optional[dict] = None):
        widget = TariffRateEditor(rate)
        self._rates.append(widget)
        self.rates_container_layout.addWidget(widget)

    def _add_rate(self):
        self._append_rate({
            "rate_name": "",
            "rate_gbp_per_kwh": None,
            "schedule": {
                "type": "weekly",
                "day_sets": [{"days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]}],
                "time_ranges": [{"start": "00:00", "end": "01:00"}],
            },
        })

    def _remove_rate(self):
        if not self._rates:
            return
        widget = self._rates.pop()
        widget.setParent(None)
        widget.deleteLater()

    def _on_accept(self):
        error = self.validate()
        if error:
            QMessageBox.warning(self, "Invalid tariff", error)
            return
        self.accept()

    def validate(self) -> Optional[str]:
        if not self.supplier_name.text().strip():
            return "Supplier name is required."
        if not self.tariff_name.text().strip():
            return "Tariff name is required."
        if not self._rates:
            return "Add at least one rate."
        for rate_widget in self._rates:
            err = rate_widget.validate()
            if err:
                return err
        return None

    def get_result(self) -> Tuple[str, dict, bool]:
        supplier_name = self.supplier_name.text().strip()
        tariff = copy.deepcopy(self._original_tariff)
        tariff["tariff_name"] = self.tariff_name.text().strip()
        tariff["complicated"] = self.complicated.isChecked()
        notes = self.complication_notes.text().strip()
        tariff["complication_notes"] = notes if notes else None
        if self.has_standing.isChecked():
            tariff["standing_charge_gbp_per_day"] = float(self.standing_value.value())
        else:
            tariff["standing_charge_gbp_per_day"] = None

        tariff["rates"] = [rate_widget.get_rate() for rate_widget in self._rates]
        dirty = json.dumps(tariff, sort_keys=True) != json.dumps(self._original_tariff, sort_keys=True)
        supplier_dirty = supplier_name != self._original_supplier_name
        return supplier_name, tariff, (dirty or supplier_dirty)


class TariffSelectionDialog(QDialog):
    def __init__(self, parent=None, documents: Optional[List[TariffDocument]] = None,
                 current_doc: Optional[TariffDocument] = None, selected_entry: Optional[TariffEntry] = None,
                 currency_symbol: str = "£"):
        super().__init__(parent)
        self.setWindowTitle("Select Tariff")
        self.setModal(True)
        self._documents = documents or []
        self._current_doc = current_doc
        self._entries: List[Optional[TariffEntry]] = []
        self._selected_entry = selected_entry
        self._currency_symbol = currency_symbol

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Select", "Supplier", "Tariff", "Headline price"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add New…")
        btn_row.addWidget(self.btn_add)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        self.btn_add.clicked.connect(self._on_add_new)

        self._refresh_table()

    def _refresh_table(self):
        self.table.blockSignals(True)
        self._entries = []
        rows: List[Tuple[str, str, str]] = []
        rows.append(("Custom", "", "Use Settings"))
        self._entries.append(None)

        for doc in self._documents:
            for supplier_index, supplier in enumerate(doc.data.get("suppliers", [])):
                for tariff_index, tariff in enumerate(supplier.get("tariffs", [])):
                    headline = tariff_headline(tariff, self._currency_symbol)
                    rows.append((supplier.get("supplier_name", ""), tariff.get("tariff_name", ""), headline))
                    self._entries.append(TariffEntry(doc, supplier_index, tariff_index))

        self.table.setRowCount(len(rows))
        for row_index, (supplier_name, tariff_name, headline) in enumerate(rows):
            check_item = QTableWidgetItem("")
            check_item.setFlags(check_item.flags() | Qt.ItemIsUserCheckable)
            check_item.setFlags(check_item.flags() & ~Qt.ItemIsEditable)
            check_state = Qt.Unchecked
            entry = self._entries[row_index]
            if entry is None and self._selected_entry is None:
                check_state = Qt.Checked
            elif entry is not None and self._selected_entry is not None:
                if (entry.document == self._selected_entry.document
                        and entry.supplier_index == self._selected_entry.supplier_index
                        and entry.tariff_index == self._selected_entry.tariff_index):
                    check_state = Qt.Checked
            check_item.setCheckState(check_state)
            self.table.setItem(row_index, 0, check_item)
            self.table.setItem(row_index, 1, QTableWidgetItem(supplier_name))
            self.table.setItem(row_index, 2, QTableWidgetItem(tariff_name))
            self.table.setItem(row_index, 3, QTableWidgetItem(headline))

        self.table.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem):
        if item.column() != 0:
            return
        if item.checkState() != Qt.Checked:
            return
        row = item.row()
        self._select_row(row)

    def _select_row(self, row: int):
        self.table.blockSignals(True)
        for i in range(self.table.rowCount()):
            it = self.table.item(i, 0)
            if not it:
                continue
            it.setCheckState(Qt.Checked if i == row else Qt.Unchecked)
        self.table.blockSignals(False)

    def _on_double_click(self, row: int, _column: int):
        entry = self._entries[row]
        if entry is None:
            return
        supplier = entry.supplier()
        tariff = entry.tariff()
        dlg = TariffEditorDialog(self, supplier.get("supplier_name", ""), tariff)
        if dlg.exec() == QDialog.Accepted:
            supplier_name, updated_tariff, dirty = dlg.get_result()
            if dirty:
                supplier["supplier_name"] = supplier_name
                tariff.clear()
                tariff.update(updated_tariff)
                self._save_document(entry.document)
                self._refresh_table()

    def _on_add_new(self):
        if not self._current_doc:
            QMessageBox.information(self, "Add tariff", "Import a tariff file first.")
            return
        dlg = TariffEditorDialog(self, "", {
            "tariff_id": "",
            "tariff_name": "",
            "market_segment": "domestic",
            "availability": {
                "countries": ["GB"],
                "regions": ["all"],
                "new_customers_only": False,
                "smart_meter_required": False,
                "payment_methods": [],
                "exit_fees": False,
                "contract_length_months": 0,
            },
            "usp_tags": [],
            "complicated": False,
            "complication_notes": None,
            "pricing_model": "flat_or_tou",
            "standing_charge_gbp_per_day": None,
            "rates": [],
            "export": {
                "has_export_rate": False,
                "export_notes": None,
                "export_rates": [],
            },
            "effective": {
                "start_date": None,
                "end_date": None,
            },
            "source": {
                "retrieved_at": "",
                "source_urls": [],
                "evidence_notes": "",
            },
        })
        if dlg.exec() == QDialog.Accepted:
            supplier_name, updated_tariff, dirty = dlg.get_result()
            if not dirty:
                return
            self._add_new_tariff(self._current_doc, supplier_name, updated_tariff)
            self._refresh_table()

    def _add_new_tariff(self, doc: TariffDocument, supplier_name: str, tariff: dict):
        suppliers = doc.data.setdefault("suppliers", [])
        supplier = next((s for s in suppliers if s.get("supplier_name") == supplier_name), None)
        is_new_supplier = supplier is None
        if is_new_supplier:
            supplier = {
                "supplier_id": "",
                "supplier_name": supplier_name,
                "supplier_website": "",
                "notes": [],
                "tariffs": [],
            }
            suppliers.append(supplier)
        if is_new_supplier:
            supplier_id = supplier.get("supplier_id") or slugify(supplier_name) or "custom-supplier"
            existing_ids = {s.get("supplier_id") for s in suppliers if s is not supplier}
            if supplier_id in existing_ids:
                suffix = 2
                base = supplier_id
                while f"{base}-{suffix}" in existing_ids:
                    suffix += 1
                supplier_id = f"{base}-{suffix}"
            supplier["supplier_id"] = supplier_id

        if not tariff.get("tariff_id"):
            tariff_id = slugify(tariff.get("tariff_name", "")) or "custom-tariff"
            existing_tariff_ids = {t.get("tariff_id") for t in supplier.get("tariffs", [])}
            if tariff_id in existing_tariff_ids:
                suffix = 2
                base = tariff_id
                while f"{base}-{suffix}" in existing_tariff_ids:
                    suffix += 1
                tariff_id = f"{base}-{suffix}"
            tariff["tariff_id"] = tariff_id
        supplier.setdefault("tariffs", []).append(tariff)
        self._save_document(doc)

    def _save_document(self, doc: TariffDocument):
        try:
            with open(doc.save_path, "w", encoding="utf-8") as f:
                json.dump(doc.data, f, indent=2)
        except Exception as ex:
            QMessageBox.critical(self, "Save failed", f"Could not save:\n{ex}")

    def _on_accept(self):
        selected_row = None
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                selected_row = row
                break
        if selected_row is None:
            QMessageBox.warning(self, "Select tariff", "Select a tariff before continuing.")
            return
        self._selected_entry = self._entries[selected_row]
        self.accept()

    def get_selected_entry(self) -> Optional[TariffEntry]:
        return self._selected_entry


# =========================
# Devices dialog
# =========================

class DeviceListDialog(QDialog):
    def __init__(self, parent=None, project: Optional[Project] = None):
        super().__init__(parent)
        self.setWindowTitle("Device List")
        self.setModal(True)
        self.project = project

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Show",
            "Name",
            "Power (W)",
            "Usage duration (H:M)",
            "Variable",
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed | QTableWidget.SelectedClicked)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add device")
        self.btn_remove = QPushButton("Remove device")
        self.btn_duplicate = QPushButton("Duplicate selected")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_duplicate)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self.table.itemChanged.connect(self._on_item_changed)
        self.btn_add.clicked.connect(self._add_device)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_duplicate.clicked.connect(self._duplicate_selected)

        self.refresh_table()

    def refresh_table(self):
        if not self.project:
            return
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.project.devices))
        for row, dev in enumerate(self.project.devices):
            enabled_item = QTableWidgetItem("")
            enabled_item.setFlags(enabled_item.flags() | Qt.ItemIsUserCheckable)
            enabled_item.setFlags(enabled_item.flags() & ~Qt.ItemIsEditable)
            enabled_item.setCheckState(Qt.Checked if dev.enabled else Qt.Unchecked)
            self.table.setItem(row, 0, enabled_item)

            name_item = QTableWidgetItem(dev.name)
            self.table.setItem(row, 1, name_item)

            power_item = QTableWidgetItem(f"{dev.power_w:g}")
            self.table.setItem(row, 2, power_item)

            duration_item = QTableWidgetItem(format_duration_minutes(dev.default_duration_min))
            self.table.setItem(row, 3, duration_item)

            variable_item = QTableWidgetItem("")
            variable_item.setFlags(variable_item.flags() | Qt.ItemIsUserCheckable)
            variable_item.setFlags(variable_item.flags() & ~Qt.ItemIsEditable)
            variable_item.setCheckState(Qt.Checked if dev.variable else Qt.Unchecked)
            self.table.setItem(row, 4, variable_item)
        self.table.blockSignals(False)

    def _notify_parent(self):
        parent = self.parent()
        if parent and hasattr(parent, "refresh_tables"):
            parent.refresh_tables()
        if parent and hasattr(parent, "recompute"):
            parent.recompute()

    def _on_item_changed(self, item: QTableWidgetItem):
        if not self.project:
            return
        row, col = item.row(), item.column()
        if not (0 <= row < len(self.project.devices)):
            return
        dev = self.project.devices[row]
        updated = False
        if col == 0:
            dev.enabled = (item.checkState() == Qt.Checked)
            updated = True
        elif col == 1:
            dev.name = item.text().strip() or dev.name
            updated = True
        elif col == 2:
            try:
                dev.power_w = max(0.0, float(item.text()))
                updated = True
            except Exception:
                self.table.blockSignals(True)
                item.setText(f"{dev.power_w:g}")
                self.table.blockSignals(False)
        elif col == 3:
            parsed = parse_duration_text(item.text())
            if parsed is None:
                self.table.blockSignals(True)
                item.setText(format_duration_minutes(dev.default_duration_min))
                self.table.blockSignals(False)
            else:
                dev.default_duration_min = parsed
                updated = True
        elif col == 4:
            dev.variable = (item.checkState() == Qt.Checked)
            updated = True

        if updated:
            dev.apply_usage_settings()
            self.table.blockSignals(True)
            self.table.item(row, 3).setText(format_duration_minutes(dev.default_duration_min))
            self.table.blockSignals(False)
            self._notify_parent()

    def _add_device(self):
        if not self.project:
            return
        dev = Device(
            name="Device",
            dtype=DeviceType.SCHEDULED,
            power_w=20.0,
            enabled=True,
            default_duration_min=30,
            variable=True,
        )
        dev.apply_usage_settings()
        self.project.devices.append(dev)
        self.refresh_table()
        self.table.selectRow(len(self.project.devices) - 1)
        self._notify_parent()

    def _remove_selected(self):
        if not self.project:
            return
        rows = sorted({item.row() for item in self.table.selectedItems()}, reverse=True)
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self.project.devices):
                self.project.devices.pop(row)
        self.refresh_table()
        self._notify_parent()

    def _duplicate_selected(self):
        if not self.project:
            return
        rows = sorted({item.row() for item in self.table.selectedItems()})
        if not rows:
            return
        insert_at = rows[-1] + 1
        for row in rows:
            if 0 <= row < len(self.project.devices):
                clone = Device.from_dict(self.project.devices[row].to_dict())
                clone.name = f"{clone.name} Copy"
                self.project.devices.insert(insert_at, clone)
                insert_at += 1
        self.refresh_table()
        self.table.selectRow(insert_at - 1)
        self._notify_parent()

# =========================
# Scheduled block dialog
# =========================

class IntervalDialog(QDialog):
    def __init__(self, parent=None, start_min: int = 0, end_min: int = MINUTES_PER_DAY, always_on: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Schedule block")
        self._updating = False

        layout = QVBoxLayout(self)

        self.always_on = QCheckBox("Always On")
        self.always_on.setChecked(always_on)
        layout.addWidget(self.always_on)

        form = QFormLayout()

        self.start_hour = QSpinBox()
        self.start_hour.setRange(0, 23)
        self.start_minute = QSpinBox()
        self.start_minute.setRange(0, 59)
        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("Hour"))
        start_row.addWidget(self.start_hour)
        start_row.addWidget(QLabel("Min"))
        start_row.addWidget(self.start_minute)
        form.addRow("Start time", start_row)

        self.end_hour = QSpinBox()
        self.end_hour.setRange(0, 24)
        self.end_minute = QSpinBox()
        self.end_minute.setRange(0, 59)
        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("Hour"))
        end_row.addWidget(self.end_hour)
        end_row.addWidget(QLabel("Min"))
        end_row.addWidget(self.end_minute)
        form.addRow("End time", end_row)

        self.duration_hour = QSpinBox()
        self.duration_hour.setRange(0, 24)
        self.duration_minute = QSpinBox()
        self.duration_minute.setRange(0, 59)
        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Hour"))
        dur_row.addWidget(self.duration_hour)
        dur_row.addWidget(QLabel("Min"))
        dur_row.addWidget(self.duration_minute)
        form.addRow("Duration", dur_row)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._set_start_end(start_min, end_min)
        self._set_duration_from_times()

        self.always_on.stateChanged.connect(self._toggle_always_on)
        self.start_hour.valueChanged.connect(self._time_changed)
        self.start_minute.valueChanged.connect(self._time_changed)
        self.end_hour.valueChanged.connect(self._time_changed)
        self.end_minute.valueChanged.connect(self._time_changed)
        self.duration_hour.valueChanged.connect(self._duration_changed)
        self.duration_minute.valueChanged.connect(self._duration_changed)

        self._toggle_always_on()

    def _toggle_always_on(self):
        is_always = self.always_on.isChecked()
        for widget in (
            self.start_hour, self.start_minute,
            self.end_hour, self.end_minute,
            self.duration_hour, self.duration_minute,
        ):
            widget.setEnabled(not is_always)
        if is_always:
            self._set_start_end(0, MINUTES_PER_DAY)
            self._set_duration_from_times()

    def _set_start_end(self, start_min: int, end_min: int):
        self._updating = True
        start_min = clamp(int(start_min), 0, MINUTES_PER_DAY - 1)
        end_min = clamp(int(end_min), 1, MINUTES_PER_DAY)
        if end_min <= start_min:
            end_min = min(MINUTES_PER_DAY, start_min + 1)
        self.start_hour.setValue(start_min // 60)
        self.start_minute.setValue(start_min % 60)
        self.end_hour.setValue(end_min // 60)
        self.end_minute.setValue(end_min % 60 if end_min < MINUTES_PER_DAY else 0)
        self._sync_end_minute_state()
        self._updating = False

    def _sync_end_minute_state(self):
        if self.end_hour.value() == 24:
            self.end_minute.setValue(0)
            self.end_minute.setEnabled(False)
        else:
            self.end_minute.setEnabled(True)

    def _start_minutes(self) -> int:
        return self.start_hour.value() * 60 + self.start_minute.value()

    def _end_minutes(self) -> int:
        end_hour = self.end_hour.value()
        if end_hour == 24:
            return MINUTES_PER_DAY
        return end_hour * 60 + self.end_minute.value()

    def _duration_minutes(self) -> int:
        return self.duration_hour.value() * 60 + self.duration_minute.value()

    def _set_duration_from_times(self):
        self._updating = True
        duration = max(1, self._end_minutes() - self._start_minutes())
        self.duration_hour.setValue(duration // 60)
        self.duration_minute.setValue(duration % 60)
        self._updating = False

    def _time_changed(self):
        if self._updating or self.always_on.isChecked():
            return
        self._sync_end_minute_state()
        start_min = self._start_minutes()
        end_min = self._end_minutes()
        if end_min <= start_min:
            end_min = min(MINUTES_PER_DAY, start_min + 1)
            self._set_start_end(start_min, end_min)
        self._set_duration_from_times()

    def _duration_changed(self):
        if self._updating or self.always_on.isChecked():
            return
        duration = max(1, self._duration_minutes())
        start_min = self._start_minutes()
        max_duration = max(1, MINUTES_PER_DAY - start_min)
        if duration > max_duration:
            duration = max_duration
            self._updating = True
            self.duration_hour.setValue(duration // 60)
            self.duration_minute.setValue(duration % 60)
            self._updating = False
        end_min = start_min + duration
        self._set_start_end(start_min, end_min)

    def get_result(self) -> Tuple[bool, int, int]:
        if self.always_on.isChecked():
            return True, 0, MINUTES_PER_DAY
        start_min = self._start_minutes()
        end_min = self._end_minutes()
        if end_min <= start_min:
            end_min = min(MINUTES_PER_DAY, start_min + 1)
        return False, start_min, end_min


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
        self.visible_device_indices: List[int] = []

        # layout constants
        self.left_label_w = 220
        self.right_info_w = 240
        self.top_tariff_h = 56
        self.row_h = 44
        self.row_gap = 6
        self.axis_h = 32
        self.label_pad = 12
        self.axis_top_pad = 10
        self.axis_bottom_pad = 6
        self.axis_label_padding = 4
        self.axis_labels_rotated = False
        self.axis_label_rot_height = 0

        # interaction state
        self.hit = HitTest()
        self.dragging = False
        self.drag_start_pos = QPoint()
        self.drag_start_min = 0
        self.drag_original = None  # store original data
        self.hover_hit = HitTest()

        # appearance
        self.font = QFont("Sans", 9)
        self.tariff_font = QFont("Sans", 15)
        self.name_font = QFont("Sans", 10, QFont.DemiBold)
        self.power_font = QFont("Sans", 9)
        self.info_font = QFont("Sans", 11)
        self.setMinimumHeight(400)

    def set_data(self, project: Project, settings: SettingsModel, sim: SimResult):
        self.project = project
        self.settings = settings
        self.sim = sim
        self._update_visible_devices()
        self._update_left_label_width()
        self._update_right_info_width()
        self._update_axis_layout()
        self.updateGeometry()
        self.update()

    def _update_visible_devices(self):
        if not self.project:
            self.visible_device_indices = []
            return
        self.visible_device_indices = [
            idx for idx, dev in enumerate(self.project.devices) if dev.enabled
        ]

    def _update_left_label_width(self):
        if not self.project:
            self.left_label_w = 220
            return
        name_metrics = QFontMetrics(self.name_font)
        power_metrics = QFontMetrics(self.power_font)
        widest = 0
        for idx in self.visible_device_indices:
            dev = self.project.devices[idx]
            widest = max(
                widest,
                name_metrics.horizontalAdvance(dev.name),
                power_metrics.horizontalAdvance(f"{dev.power_w:g} W"),
            )
        self.left_label_w = max(160, widest + self.label_pad * 2)

    def _update_right_info_width(self):
        if not (self.project and self.settings and self.sim):
            self.right_info_w = 240
            return
        metrics = QFontMetrics(self.info_font)
        max_kwh = 0.0
        max_cost = 0.0
        for idx in self.visible_device_indices:
            max_kwh = max(max_kwh, self.sim.per_device_kwh_day[idx])
            max_cost = max(max_cost, self.sim.per_device_cost_day[idx])
        kwh_text = f"▲ {max_kwh:.2f} kWh"
        cost_text = f"▼ {self.settings.currency_symbol}{max_cost:.2f}"
        text_width = max(
            metrics.horizontalAdvance(kwh_text),
            metrics.horizontalAdvance(cost_text),
        )
        diameter = max(6, self.row_h - 16)
        info_inner_width = 4 + diameter + 10 + text_width + 4
        self.right_info_w = max(180, info_inner_width + 12)

    def _update_axis_layout(self):
        if not self.project:
            self.axis_labels_rotated = False
            self.axis_h = 32
            return
        tl = self._timeline_rect()
        tick_spacing = max(1, tl.width() / 24)
        metrics = QFontMetrics(self.font)
        label_width = metrics.horizontalAdvance("00:00")
        label_height = metrics.height()
        needs_rotation = label_width + self.axis_label_padding > tick_spacing
        if needs_rotation:
            angle = math.radians(45)
            rot_height = abs(label_width * math.sin(angle)) + abs(label_height * math.cos(angle))
            self.axis_label_rot_height = int(rot_height)
            self.axis_h = max(32, int(rot_height) + self.axis_top_pad + self.axis_bottom_pad)
        else:
            self.axis_label_rot_height = label_height
            self.axis_h = max(32, label_height + self.axis_top_pad + self.axis_bottom_pad)
        self.axis_labels_rotated = needs_rotation

    def sizeHint(self):
        if not self.project:
            return QSize(900, 500)
        h = self.top_tariff_h + self.axis_h + (len(self.visible_device_indices) * (self.row_h + self.row_gap)) + 20
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
        return QRect(self.label_pad, rr.top(), self.left_label_w - self.label_pad * 2, rr.height())

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
        for row_index, dev_index in enumerate(self.visible_device_indices):
            dev = self.project.devices[dev_index]
            self._paint_row(p, row_index, dev_index, dev)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_axis_layout()

    def _paint_tariff(self, p: QPainter):
        tr = self._tariff_rect()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(12, 20, 40))
        p.drawRect(tr)

        # base rate visualization
        for hour in range(24):
            m0 = hour * 60
            m1 = (hour + 1) * 60
            r = tariff_rate_for_minute(self.settings, m0)
            # map rate to brightness (simple)
            # bigger rate -> brighter
            base = clamp(int(40 + 180 * r), 40, 200)
            col = QColor(
                clamp(15 + int(40 * r), 15, 80),
                clamp(30 + int(60 * r), 30, 120),
                clamp(90 + base, 90, 220),
            )
            x0 = self._minute_to_x(m0)
            x1 = self._minute_to_x(m1)
            p.setBrush(col)
            p.drawRect(QRect(x0, tr.top(), x1 - x0, tr.height()))

        segment_label = f"{self.settings.currency_symbol}{{rate:.2f}}/kWh"
        for start_min, end_min, rate in tariff_segments(self.settings):
            x0 = self._minute_to_x(start_min)
            x1 = self._minute_to_x(end_min)
            if x1 - x0 < 40:
                continue
            label_rect = QRect(x0 + 4, tr.top(), x1 - x0 - 8, tr.height())
            p.setFont(self.tariff_font)
            p.setPen(QColor(230, 235, 245))
            p.drawText(label_rect, Qt.AlignCenter, segment_label.format(rate=rate))

        # free overlay
        fr = self.settings.free_rule.normalized()
        if fr.enabled:
            x0 = self._minute_to_x(fr.start_min)
            x1 = self._minute_to_x(fr.end_min)
            overlay = QColor(80, 160, 80, 140)
            p.setBrush(overlay)
            p.drawRect(QRect(x0, tr.top(), x1 - x0, tr.height()))
            p.setFont(self.tariff_font)
            p.setPen(QColor(230, 235, 245))
            p.drawText(QRect(x0, tr.top(), x1 - x0, tr.height()), Qt.AlignCenter,
                       f"FREE ≤ {fr.free_kw_threshold:.1f} kW")

        # border + label
        p.setPen(QPen(QColor(120, 120, 130), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(tr)
        p.setFont(self.tariff_font)
        p.setPen(QColor(230, 235, 245))
        label_width = self.left_label_w - self.label_pad * 2
        label_right = tr.left() - 4
        label_rect = QRect(label_right - label_width, tr.top(), label_width, tr.height())
        label = self.settings.tariff_label or "Tariff"
        p.drawText(label_rect, Qt.AlignVCenter | Qt.AlignRight, label)

    def _paint_axis(self, p: QPainter):
        tl = self._timeline_rect()
        axis_top = self.top_tariff_h
        axis_baseline = axis_top + self.axis_h - self.axis_bottom_pad + 2
        tick_top = axis_baseline - 10
        label_bottom = axis_baseline - 2
        metrics = QFontMetrics(self.font)
        p.setPen(QColor(160, 160, 170))
        # hour ticks
        for hour in range(25):
            m = hour * 60
            x = self._minute_to_x(m)
            p.drawLine(x, tick_top, x, axis_baseline)
            if hour < 24:
                label = f"{hour:02d}:00"
                if self.axis_labels_rotated:
                    p.save()
                    p.translate(x + 2, label_bottom - self.axis_label_rot_height)
                    p.rotate(-45)
                    p.drawText(0, metrics.ascent(), label)
                    p.restore()
                else:
                    p.drawText(x + 2, label_bottom - metrics.descent(), label)

        # axis baseline
        p.setPen(QColor(90, 90, 100))
        p.drawLine(tl.left(), axis_baseline, tl.right(), axis_baseline)

    def _paint_row(self, p: QPainter, row_index: int, dev_index: int, dev: Device):
        rr = self._row_rect(row_index)
        # row background
        bg = QColor(34, 34, 38) if row_index % 2 == 0 else QColor(30, 30, 34)
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRect(rr)

        # label left
        lr = self._device_label_rect(row_index)
        name_metrics = QFontMetrics(self.name_font)
        power_metrics = QFontMetrics(self.power_font)
        name_h = name_metrics.height()
        power_h = power_metrics.height()
        total_h = name_h + power_h
        start_y = lr.top() + (lr.height() - total_h) // 2
        name_rect = QRect(lr.left(), start_y, lr.width(), name_h)
        power_rect = QRect(lr.left(), start_y + name_h, lr.width(), power_h)

        p.setPen(QColor(220, 220, 230))
        p.setFont(self.name_font)
        p.drawText(name_rect, Qt.AlignCenter, dev.name)
        p.setFont(self.power_font)
        p.drawText(power_rect, Qt.AlignCenter, f"{dev.power_w:g} W")

        # draw items on timeline
        p.setPen(Qt.NoPen)

        if dev.dtype == DeviceType.ALWAYS:
            self._draw_block(p, rr, 0, MINUTES_PER_DAY, QColor(140, 140, 200, 180), hover=False)

        elif dev.dtype == DeviceType.SCHEDULED:
            for j, iv in enumerate(dev.intervals):
                ivn = iv.normalized()
                hover = (
                    self.hover_hit.kind.startswith("interval")
                    and self.hover_hit.device_index == dev_index
                    and self.hover_hit.item_index == j
                )
                self._draw_block(p, rr, ivn.start_min, ivn.end_min, QColor(100, 170, 240, 190), hover=hover)

        elif dev.dtype == DeviceType.EVENTS:
            for j, ev in enumerate(dev.events):
                evn = ev.normalized()
                hover = (
                    self.hover_hit.kind == HitKind.EVENT_BODY
                    and self.hover_hit.device_index == dev_index
                    and self.hover_hit.item_index == j
                )
                self._draw_event(p, rr, evn.start_min, evn.duration_min, QColor(240, 170, 90, 200), hover=hover)

        # row border
        p.setPen(QPen(QColor(55, 55, 62), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rr)

        # right info
        info = self._device_info_rect(row_index)
        on_min = self.sim.per_device_on_minutes[dev_index] if self.sim else 0
        kwh = self.sim.per_device_kwh_day[dev_index] if self.sim else 0.0
        cost = self.sim.per_device_cost_day[dev_index] if self.sim else 0.0
        bar_height = rr.height() - 16
        diameter = max(6, min(bar_height, info.height() - 8))
        circle_rect = QRect(
            info.left() + 4,
            info.top() + (info.height() - diameter) // 2,
            diameter,
            diameter,
        )
        on_ratio = clamp(on_min / MINUTES_PER_DAY, 0.0, 1.0)
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(200, 70, 70))
        p.drawEllipse(circle_rect)
        p.setBrush(QColor(80, 180, 90))
        on_span = int(on_ratio * 360 * 16)
        p.drawPie(circle_rect, 90 * 16, -on_span)
        p.restore()

        text_x = circle_rect.right() + 10
        text_rect = QRect(text_x, info.top(), info.right() - text_x, info.height())
        p.setPen(QColor(220, 220, 230))
        p.setFont(self.info_font)
        metrics = QFontMetrics(self.info_font)
        line_h = metrics.height()
        total_h = line_h * 2
        start_y = text_rect.top() + (text_rect.height() - total_h) // 2
        kwh_text = f"{kwh:.2f} kWh"
        cost_text = f"{self.settings.currency_symbol}{cost:.2f}"
        p.drawText(QRect(text_rect.left(), start_y, text_rect.width(), line_h),
                   Qt.AlignLeft | Qt.AlignVCenter, f"▲ {kwh_text}")
        p.drawText(QRect(text_rect.left(), start_y + line_h, text_rect.width(), line_h),
                   Qt.AlignLeft | Qt.AlignVCenter, f"▼ {cost_text}")

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
            _row_index, dev_index = self._device_index_from_pos(pos)
            if dev_index != -1:
                dev = self.project.devices[dev_index]
                m = self._x_to_minute(pos.x())
                if dev.dtype == DeviceType.SCHEDULED:
                    duration = max(1, dev.default_duration_min)
                    dev.intervals.append(Interval(m, min(MINUTES_PER_DAY, m + duration)).normalized())
                    self._trigger_recompute()
                    return
                elif dev.dtype == DeviceType.EVENTS:
                    duration = max(1, dev.default_duration_min)
                    dev.events.append(Event(m, duration, None).normalized())
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

    def mouseDoubleClickEvent(self, e):
        if not (self.project and self.settings and self.sim):
            return
        pos = e.position().toPoint()
        _row_index, dev_index = self._device_index_from_pos(pos)
        if dev_index == -1:
            return
        dev = self.project.devices[dev_index]
        if dev.dtype == DeviceType.SCHEDULED:
            hit = self._hit_test(pos)
            if not hit.kind.startswith("interval"):
                return
            interval = dev.intervals[hit.item_index].normalized()
            dlg = IntervalDialog(self, interval.start_min, interval.end_min, always_on=False)
            if dlg.exec() == QDialog.Accepted:
                always_on, start_min, end_min = dlg.get_result()
                if always_on:
                    dev.dtype = DeviceType.ALWAYS
                    dev.intervals = []
                    dev.default_duration_min = MINUTES_PER_DAY
                    dev.variable = False
                else:
                    dev.dtype = DeviceType.SCHEDULED
                    dev.intervals[hit.item_index] = Interval(start_min, end_min).normalized()
                    dev.default_duration_min = max(1, end_min - start_min)
                    dev.variable = True
                self._trigger_refresh()
        elif dev.dtype == DeviceType.ALWAYS:
            dlg = IntervalDialog(self, 0, MINUTES_PER_DAY, always_on=True)
            if dlg.exec() == QDialog.Accepted:
                always_on, start_min, end_min = dlg.get_result()
                if always_on:
                    dev.dtype = DeviceType.ALWAYS
                    dev.intervals = []
                    dev.default_duration_min = MINUTES_PER_DAY
                    dev.variable = False
                else:
                    dev.dtype = DeviceType.SCHEDULED
                    dev.intervals = [Interval(start_min, end_min).normalized()]
                    dev.default_duration_min = max(1, end_min - start_min)
                    dev.variable = True
                self._trigger_refresh()

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

    def _trigger_refresh(self):
        win = self.window()
        if hasattr(win, "refresh_tables"):
            win.refresh_tables()
        self._trigger_recompute()

    def _device_index_from_pos(self, pos: QPoint) -> Tuple[int, int]:
        tl = self._timeline_rect()
        if pos.y() < tl.top() or pos.y() > tl.bottom():
            return -1, -1
        # check each visible row rect
        for row_index, dev_index in enumerate(self.visible_device_indices):
            rr = self._row_rect(row_index)
            if rr.contains(pos):
                return row_index, dev_index
        return -1, -1

    def _hit_test(self, pos: QPoint) -> HitTest:
        row_index, dev_index = self._device_index_from_pos(pos)
        if dev_index == -1:
            return HitTest()

        dev = self.project.devices[dev_index]
        rr = self._row_rect(row_index)

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
                        return HitTest(HitKind.INTERVAL_LEFT, dev_index, j, self._x_to_minute(pos.x()))
                    if near(pos.x(), rect.right()):
                        return HitTest(HitKind.INTERVAL_RIGHT, dev_index, j, self._x_to_minute(pos.x()))
                    return HitTest(HitKind.INTERVAL_BODY, dev_index, j, self._x_to_minute(pos.x()))

        if dev.dtype == DeviceType.EVENTS:
            for j, ev in enumerate(dev.events):
                evn = ev.normalized()
                x0 = self._minute_to_x(evn.start_min)
                x1 = self._minute_to_x(evn.start_min + evn.duration_min)
                rect = QRect(x0, rr.top() + 10, max(3, x1 - x0), rr.height() - 20)
                if rect.contains(pos):
                    return HitTest(HitKind.EVENT_BODY, dev_index, j, self._x_to_minute(pos.x()))

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
        self.tariff_documents: List[TariffDocument] = []
        self.current_tariff_doc: Optional[TariffDocument] = None
        self.active_tariff_entry: Optional[TariffEntry] = None

        # project state
        self.project = Project(devices=[
            Device(
                name="Router",
                dtype=DeviceType.ALWAYS,
                power_w=10.0,
                enabled=True,
                default_duration_min=MINUTES_PER_DAY,
                variable=False,
            ),
            Device(
                name="Grow light",
                dtype=DeviceType.SCHEDULED,
                power_w=120.0,
                enabled=True,
                default_duration_min=10 * 60,
                variable=True,
                intervals=[Interval(8*60, 18*60)],
            ),
            Device(
                name="Kettle",
                dtype=DeviceType.EVENTS,
                power_w=2400.0,
                enabled=True,
                default_duration_min=3,
                variable=False,
                events=[Event(7*60+15, 3, None), Event(18*60+40, 3, None)],
            ),
        ])
        self.current_file: Optional[str] = None
        self.device_dialog: Optional["DeviceListDialog"] = None

        # central layout
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        self.timeline = TimelineWidget(parent=self)
        root_layout.addWidget(self.timeline, stretch=1)

        self.summary = QLabel("")
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary.setStyleSheet("QLabel { padding: 8px; background: #1e1e22; color: #e8e8ee; }")
        root_layout.addWidget(self.summary, stretch=0)

        # menu
        self._build_menu()

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

        devm = mb.addMenu("&Devices")
        act_device_list = QAction("Device List…", self)
        devm.addAction(act_device_list)
        act_device_list.triggered.connect(self.open_device_list)

        timingm = mb.addMenu("&Timing")
        act_custom_periods = QAction("Custom Periods…", self)
        timingm.addAction(act_custom_periods)
        act_custom_periods.triggered.connect(self.open_custom_periods)

        tariffm = mb.addMenu("&Tariff")
        act_import_tariff = QAction("Import…", self)
        act_select_tariff = QAction("Select Tariff…", self)
        tariffm.addAction(act_import_tariff)
        tariffm.addAction(act_select_tariff)
        act_import_tariff.triggered.connect(self.import_tariffs)
        act_select_tariff.triggered.connect(self.select_tariff)

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

    def open_custom_periods(self):
        dlg = CustomPeriodsDialog(self, self.project.custom_periods)
        if dlg.exec() == QDialog.Accepted:
            self.project.custom_periods = dlg.get_periods()
            self.recompute()

    def open_device_list(self):
        dlg = DeviceListDialog(self, self.project)
        self.device_dialog = dlg
        try:
            dlg.exec()
        finally:
            self.device_dialog = None
            self.refresh_tables()
            self.recompute()

    # ---------------------
    # Tariffs
    # ---------------------

    def import_tariffs(self):
        dlg = TariffImportDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        path, append = dlg.get_result()
        if not path:
            return
        try:
            doc = load_tariff_document(path)
        except Exception as ex:
            QMessageBox.critical(self, "Import failed", f"Could not import:\n{ex}")
            return
        if not append:
            self.tariff_documents = []
        self.tariff_documents.append(doc)
        self.current_tariff_doc = doc
        QMessageBox.information(self, "Import complete", f"Loaded tariffs from:\n{path}")

    def select_tariff(self):
        dlg = TariffSelectionDialog(
            self,
            documents=self.tariff_documents,
            current_doc=self.current_tariff_doc,
            selected_entry=self.active_tariff_entry,
            currency_symbol=self.settings_model.currency_symbol,
        )
        if dlg.exec() == QDialog.Accepted:
            entry = dlg.get_selected_entry()
            self.apply_tariff_selection(entry)

    def apply_tariff_selection(self, entry: Optional[TariffEntry]):
        self.active_tariff_entry = entry
        if entry is None:
            self.settings_model.tariff_minute_rates = None
            self.settings_model.tariff_label = None
        else:
            tariff = entry.tariff()
            supplier = entry.supplier()
            self.settings_model.tariff_minute_rates = build_tariff_minute_rates(tariff)
            supplier_name = supplier.get("supplier_name", "Supplier")
            tariff_name = tariff.get("tariff_name", "Tariff")
            self.settings_model.tariff_label = f"{supplier_name} — {tariff_name}"
        self.recompute()

    # ---------------------
    # Tables <-> model sync
    # ---------------------

    def refresh_tables(self):
        if self.device_dialog and self.device_dialog.isVisible():
            self.device_dialog.refresh_table()

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
        enabled_periods = [cp for cp in self.project.custom_periods if cp.enabled]
        if not enabled_periods:
            lines.append("• (none enabled)")
        for period in enabled_periods:
            try:
                d = custom_period_to_days(period, self.settings_model)
            except Exception:
                d = 1.0
            lines.append(
                f"• {period.name} ({period.duration:g} {period.unit}): "
                f"{day_kwh*d:.3f} kWh | {cs}{day_cost*d:.2f}"
            )

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
