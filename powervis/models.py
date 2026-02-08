import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

from PySide6.QtCore import QSettings

from .utils import MINUTES_PER_DAY, clamp, parse_duration_text


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


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
    category: str = "Custom"
    dtype: str = DeviceType.SCHEDULED
    power_w: float = 20.0
    enabled: bool = True
    default_duration_min: int = 30
    variable: bool = True

    intervals: List[Interval] = field(default_factory=list)  # for scheduled
    events: List[Event] = field(default_factory=list)        # for per-use
    day_intervals: Dict[str, List[Interval]] = field(default_factory=dict)
    day_events: Dict[str, List[Event]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "dtype": self.dtype,
            "power_w": self.power_w,
            "enabled": self.enabled,
            "default_duration_min": self.default_duration_min,
            "variable": self.variable,
            "intervals": [asdict(i) for i in self.intervals],
            "events": [asdict(e) for e in self.events],
            "day_intervals": {
                day: [asdict(i) for i in intervals]
                for day, intervals in self.day_intervals.items()
            },
            "day_events": {
                day: [asdict(e) for e in events]
                for day, events in self.day_events.items()
            },
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
            category=d.get("category", "Custom"),
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
        dev.day_intervals = {}
        for day, intervals in d.get("day_intervals", {}).items():
            if isinstance(intervals, list):
                dev.day_intervals[str(day)] = [Interval(**i).normalized() for i in intervals]
        dev.day_events = {}
        for day, events in d.get("day_events", {}).items():
            if isinstance(events, list):
                dev.day_events[str(day)] = [Event(**e).normalized() for e in events]
        dev.apply_usage_settings()
        return dev


def device_from_catalog_item(item: dict, category: str) -> Optional[Device]:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name", "Device")).strip() or "Device"
    try:
        power_w = float(item.get("power_w", 20.0))
    except Exception:
        power_w = 20.0
    variable = parse_bool(item.get("variable_time", True))
    duration_text = str(item.get("usage_duration", "")).strip()
    duration_min = parse_duration_text(duration_text)
    if duration_min is None:
        duration_min = 30
    dev = Device(
        name=name,
        category=category or "Custom",
        dtype=DeviceType.SCHEDULED,
        power_w=power_w,
        enabled=False,
        default_duration_min=duration_min,
        variable=variable,
    )
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
    mains_voltage: float = 240.0

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
    tariff_minute_rates: Optional[Dict[int, List[float]]] = None
    tariff_label: Optional[str] = None
    show_standing_charge: bool = False
    show_amps: bool = False
    show_total_power: bool = True
    show_timeline_totals: bool = True
    sort_by_category: bool = False
    show_day_night: bool = True
    simulation_length_key: str = "1_week"
    affect_every_day: bool = True
    location_label: str = "Custom"
    location_lat: float = 51.5074
    location_lon: float = -0.1278

    def clone(self) -> "SettingsModel":
        return SettingsModel(
            currency_symbol=self.currency_symbol,
            month_days=self.month_days,
            step_minutes=self.step_minutes,
            mains_voltage=self.mains_voltage,
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
            tariff_minute_rates=(
                {day: list(rates) for day, rates in self.tariff_minute_rates.items()}
                if self.tariff_minute_rates
                else None
            ),
            tariff_label=self.tariff_label,
            show_standing_charge=self.show_standing_charge,
            show_amps=self.show_amps,
            show_total_power=self.show_total_power,
            show_timeline_totals=self.show_timeline_totals,
            sort_by_category=self.sort_by_category,
            show_day_night=self.show_day_night,
            simulation_length_key=self.simulation_length_key,
            affect_every_day=self.affect_every_day,
            location_label=self.location_label,
            location_lat=self.location_lat,
            location_lon=self.location_lon,
        )

    def to_qsettings(self, qs: QSettings):
        qs.setValue("currency_symbol", self.currency_symbol)
        qs.setValue("month_days", self.month_days)
        qs.setValue("step_minutes", self.step_minutes)
        qs.setValue("mains_voltage", self.mains_voltage)

        qs.setValue("base_rate_flat", self.base_rate_flat)

        qs.setValue("use_time_of_day", self.use_time_of_day)
        qs.setValue("offpeak_rate", self.offpeak_rate)
        qs.setValue("offpeak_start_min", self.offpeak_start_min)
        qs.setValue("offpeak_end_min", self.offpeak_end_min)

        qs.setValue("free_enabled", self.free_rule.enabled)
        qs.setValue("free_start_min", self.free_rule.start_min)
        qs.setValue("free_end_min", self.free_rule.end_min)
        qs.setValue("free_kw_threshold", self.free_rule.free_kw_threshold)
        qs.setValue("show_standing_charge", self.show_standing_charge)
        qs.setValue("show_amps", self.show_amps)
        qs.setValue("show_total_power", self.show_total_power)
        qs.setValue("show_timeline_totals", self.show_timeline_totals)
        qs.setValue("sort_by_category", self.sort_by_category)
        qs.setValue("show_day_night", self.show_day_night)
        qs.setValue("simulation_length_key", self.simulation_length_key)
        qs.setValue("affect_every_day", self.affect_every_day)
        qs.setValue("location_label", self.location_label)
        qs.setValue("location_lat", self.location_lat)
        qs.setValue("location_lon", self.location_lon)

        qs.setValue("has_run_before", True)

    @staticmethod
    def from_qsettings(qs: QSettings) -> "SettingsModel":
        sm = SettingsModel()
        sm.currency_symbol = qs.value("currency_symbol", sm.currency_symbol)
        sm.month_days = int(qs.value("month_days", sm.month_days))
        sm.step_minutes = int(qs.value("step_minutes", sm.step_minutes))
        sm.mains_voltage = float(qs.value("mains_voltage", sm.mains_voltage))

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
        sm.show_standing_charge = (
            str(qs.value("show_standing_charge", sm.show_standing_charge)).lower() == "true"
        )
        sm.show_amps = (
            str(qs.value("show_amps", sm.show_amps)).lower() == "true"
        )
        sm.show_total_power = (
            str(qs.value("show_total_power", sm.show_total_power)).lower() == "true"
        )
        sm.show_timeline_totals = (
            str(qs.value("show_timeline_totals", sm.show_timeline_totals)).lower() == "true"
        )
        sm.sort_by_category = (
            str(qs.value("sort_by_category", sm.sort_by_category)).lower() == "true"
        )
        sm.show_day_night = (
            str(qs.value("show_day_night", sm.show_day_night)).lower() == "true"
        )
        sm.simulation_length_key = str(
            qs.value("simulation_length_key", sm.simulation_length_key)
        )
        sm.affect_every_day = (
            str(qs.value("affect_every_day", sm.affect_every_day)).lower() == "true"
        )
        sm.location_label = str(qs.value("location_label", sm.location_label))
        sm.location_lat = float(qs.value("location_lat", sm.location_lat))
        sm.location_lon = float(qs.value("location_lon", sm.location_lon))
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
