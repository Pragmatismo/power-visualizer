import datetime
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Iterable, Union

from .models import (
    SettingsModel,
    Project,
    DeviceType,
    CustomPeriod,
    TariffDocument,
    TariffEntry,
)
from .utils import MINUTES_PER_DAY, clamp, parse_hhmm, derive_modified_path, slugify

DAY_NAME_TO_INDEX = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


def is_minute_in_window(minute: int, start_min: int, end_min: int) -> bool:
    if start_min == end_min:
        return False
    if start_min < end_min:
        return start_min <= minute < end_min
    return minute >= start_min or minute < end_min


def is_all_days(day_sets: List[dict]) -> bool:
    if not day_sets:
        return True
    all_days = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
    for day_set in day_sets:
        days = set(day_set.get("days", []))
        if all_days.issubset(days):
            return True
    return False


def day_index_from_date(day: Optional[Union[datetime.date, int]]) -> int:
    if day is None:
        return 0
    if isinstance(day, datetime.date):
        return day.weekday()
    try:
        return int(day) % 7
    except (TypeError, ValueError):
        return 0


def days_from_day_sets(day_sets: List[dict]) -> Optional[Iterable[int]]:
    if not day_sets:
        return None
    day_indices: List[int] = []
    for day_set in day_sets:
        for day in day_set.get("days", []):
            index = DAY_NAME_TO_INDEX.get(str(day).lower())
            if index is not None and index not in day_indices:
                day_indices.append(index)
    return day_indices


def tariff_day_rates(
    settings: SettingsModel,
    day: Optional[Union[datetime.date, int]] = None,
) -> Optional[List[float]]:
    if not settings.tariff_minute_rates:
        return None
    day_index = day_index_from_date(day)
    return settings.tariff_minute_rates.get(day_index) or settings.tariff_minute_rates.get(0)


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
class SimResult:
    total_kwh_day: float
    total_cost_day: float
    per_device_kwh_day: List[float]
    per_device_cost_day: List[float]
    per_device_on_minutes: List[int]
    minute_total_w: List[float]  # length 1440


@dataclass
class SimRangeResult:
    days: List[datetime.date]
    day_results: List[SimResult]
    total_kwh: float
    total_cost: float


@dataclass
class SimTotals:
    total_kwh: float
    total_cost: float
    per_device_kwh: List[float]
    per_device_cost: List[float]
    per_device_on_minutes: List[float]


def tariff_rate_for_minute(
    settings: SettingsModel,
    minute: int,
    day: Optional[Union[datetime.date, int]] = None,
) -> float:
    """Return £/kWh rate for the minute (base schedule only)."""
    if settings.tariff_minute_rates:
        day_rates = tariff_day_rates(settings, day)
        if day_rates:
            return day_rates[clamp(minute, 0, MINUTES_PER_DAY - 1)]
    if not settings.use_time_of_day:
        return settings.base_rate_flat
    if is_minute_in_window(minute, settings.offpeak_start_min, settings.offpeak_end_min):
        return settings.offpeak_rate
    return settings.base_rate_flat


def tariff_segments(
    settings: SettingsModel,
    day: Optional[Union[datetime.date, int]] = None,
) -> List[Tuple[int, int, float]]:
    if settings.tariff_minute_rates:
        segments: List[Tuple[int, int, float]] = []
        day_rates = tariff_day_rates(settings, day)
        if not day_rates:
            return segments
        current_rate = day_rates[0]
        start = 0
        for minute in range(1, MINUTES_PER_DAY + 1):
            next_rate = None if minute == MINUTES_PER_DAY else day_rates[minute]
            if minute == MINUTES_PER_DAY or next_rate != current_rate:
                segments.append((start, minute, current_rate))
                start = minute
                current_rate = next_rate
        return segments
    if not settings.use_time_of_day:
        return [(0, MINUTES_PER_DAY, settings.base_rate_flat)]
    segments: List[Tuple[int, int, float]] = []
    current_rate = tariff_rate_for_minute(settings, 0, day)
    start = 0
    for minute in range(1, MINUTES_PER_DAY + 1):
        if minute == MINUTES_PER_DAY:
            next_rate = None
        else:
            next_rate = tariff_rate_for_minute(settings, minute, day)
        if minute == MINUTES_PER_DAY or next_rate != current_rate:
            segments.append((start, minute, current_rate))
            start = minute
            current_rate = next_rate
    return segments


def build_tariff_minute_rates(tariff: dict) -> Dict[int, List[float]]:
    minute_rates = {day: [0.0] * MINUTES_PER_DAY for day in range(7)}
    minute_priority = {day: [-10**9] * MINUTES_PER_DAY for day in range(7)}
    rates = tariff.get("rates") or []
    for rate in rates:
        rate_value = rate.get("rate_gbp_per_kwh")
        if rate_value is None:
            continue
        priority = int(rate.get("priority", 0))
        schedule = rate.get("schedule") or {}
        time_ranges = schedule.get("time_ranges") or []
        day_sets = schedule.get("day_sets") or []
        applicable_days = days_from_day_sets(day_sets)
        if applicable_days is None:
            applicable_days = range(7)
        elif not applicable_days:
            continue
        apply_all_day = False
        if not time_ranges:
            apply_all_day = True
        for time_range in time_ranges:
            start = parse_hhmm(time_range.get("start", ""))
            end = parse_hhmm(time_range.get("end", ""))
            if start is None or end is None:
                continue
            if start == end:
                apply_all_day = True
                continue
            for day_index in applicable_days:
                for minute in range(MINUTES_PER_DAY):
                    if is_minute_in_window(minute, start, end):
                        if priority >= minute_priority[day_index][minute]:
                            minute_priority[day_index][minute] = priority
                            minute_rates[day_index][minute] = float(rate_value)
        if apply_all_day:
            for day_index in applicable_days:
                for minute in range(MINUTES_PER_DAY):
                    if priority >= minute_priority[day_index][minute]:
                        minute_priority[day_index][minute] = priority
                        minute_rates[day_index][minute] = float(rate_value)
    return minute_rates


def format_tariff_rate(rate: float, currency_symbol: str) -> str:
    return f"{currency_symbol}{rate:.4f}/kWh"


def has_timed_offpeak_rate(rates: List[dict], offpeak_rate: float) -> bool:
    for rate in rates:
        rate_value = rate.get("rate_gbp_per_kwh")
        if rate_value is None or float(rate_value) != offpeak_rate:
            continue
        schedule = rate.get("schedule") or {}
        day_sets = schedule.get("day_sets", [])
        if not is_all_days(day_sets):
            continue
        time_ranges = schedule.get("time_ranges") or []
        if len(time_ranges) != 1:
            continue
        time_range = time_ranges[0]
        start = parse_hhmm(time_range.get("start", ""))
        end = parse_hhmm(time_range.get("end", ""))
        if start is None or end is None or start == end:
            continue
        if start == 0 and end == MINUTES_PER_DAY:
            continue
        return True
    return False


def tariff_price_summary(tariff: dict, currency_symbol: str) -> Tuple[str, str, str]:
    rates = tariff.get("rates") or []
    rate_values = [float(rate["rate_gbp_per_kwh"]) for rate in rates if rate.get("rate_gbp_per_kwh") is not None]
    if tariff.get("complicated") or not rate_values:
        return "Complicated", "", "False"
    main_rate = max(rate_values)
    main_display = format_tariff_rate(main_rate, currency_symbol)
    unique_rates = sorted(set(rate_values))
    if len(unique_rates) == 1:
        return main_display, "", "False"
    offpeak_rate = unique_rates[0]
    offpeak_display = format_tariff_rate(offpeak_rate, currency_symbol)
    peak_times = has_timed_offpeak_rate(rates, offpeak_rate)
    return main_display, offpeak_display, "True" if peak_times else "False"


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


def build_device_minute_power(
    project: Project,
    day: Optional[Union[datetime.date, int]] = None,
) -> Tuple[List[List[float]], List[float]]:
    day_key = None
    if isinstance(day, datetime.date):
        day_key = day.isoformat()
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
            intervals = dev.intervals
            if day_key and day_key in dev.day_intervals:
                intervals = dev.day_intervals[day_key]
            for iv in intervals:
                ivn = iv.normalized()
                for m in range(ivn.start_min, ivn.end_min):
                    dev_w[i][m] += base_power_w

        elif dev.dtype == DeviceType.EVENTS:
            # Each event either uses base_power_w for duration or fixed energy spread over its duration
            events = dev.events
            if day_key and day_key in dev.day_events:
                events = dev.day_events[day_key]
            for ev in events:
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

    return dev_w, minute_total_w


def simulate_single_day(
    project: Project,
    settings: SettingsModel,
    day: Optional[Union[datetime.date, int]] = None,
    device_profiles: Optional[Tuple[List[List[float]], List[float]]] = None,
) -> SimResult:
    if device_profiles is None:
        dev_w, minute_total_w = build_device_minute_power(project, day)
    else:
        dev_w, minute_total_w = device_profiles
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
        base_rate = tariff_rate_for_minute(settings, m, day)

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


def simulate_range(
    project: Project,
    settings: SettingsModel,
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
) -> SimRangeResult:
    if start_date is None or end_date is None:
        today = datetime.date.today()
        start_date = start_date or today
        end_date = end_date or today
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    days: List[datetime.date] = []
    day_results: List[SimResult] = []
    total_kwh = 0.0
    total_cost = 0.0
    day_count = (end_date - start_date).days + 1

    for offset in range(day_count):
        day = start_date + datetime.timedelta(days=offset)
        day_result = simulate_single_day(project, settings, day)
        days.append(day)
        day_results.append(day_result)
        total_kwh += day_result.total_kwh_day
        total_cost += day_result.total_cost_day

    return SimRangeResult(
        days=days,
        day_results=day_results,
        total_kwh=total_kwh,
        total_cost=total_cost,
    )


def simulate_single_day_window(
    project: Project,
    settings: SettingsModel,
    day: Optional[Union[datetime.date, int]],
    start_min: int,
    end_min: int,
) -> SimTotals:
    start_min = clamp(int(start_min), 0, MINUTES_PER_DAY)
    end_min = clamp(int(end_min), 0, MINUTES_PER_DAY)
    if end_min <= start_min:
        device_count = len(project.devices)
        return SimTotals(
            total_kwh=0.0,
            total_cost=0.0,
            per_device_kwh=[0.0] * device_count,
            per_device_cost=[0.0] * device_count,
            per_device_on_minutes=[0.0] * device_count,
        )

    dev_w, minute_total_w = build_device_minute_power(project, day)
    device_count = len(project.devices)
    per_device_kwh = [0.0] * device_count
    per_device_cost = [0.0] * device_count
    per_device_on_minutes = [0.0] * device_count
    total_kwh = 0.0
    total_cost = 0.0

    for m in range(start_min, end_min):
        total_w = minute_total_w[m]
        total_kw = total_w / 1000.0
        kwh_this_min = total_kw * (1.0 / 60.0)
        base_rate = tariff_rate_for_minute(settings, m, day)
        if is_free_this_minute(settings, m, total_kw):
            cost_this_min = 0.0
        else:
            cost_this_min = kwh_this_min * base_rate
        total_kwh += kwh_this_min
        total_cost += cost_this_min
        if total_w > 0:
            free = is_free_this_minute(settings, m, total_kw)
            for i in range(device_count):
                w_i = dev_w[i][m]
                if w_i > 0:
                    per_device_on_minutes[i] += 1
                    kwh_i = (w_i / 1000.0) * (1.0 / 60.0)
                    per_device_kwh[i] += kwh_i
                    if not free:
                        per_device_cost[i] += kwh_i * base_rate

    return SimTotals(
        total_kwh=total_kwh,
        total_cost=total_cost,
        per_device_kwh=per_device_kwh,
        per_device_cost=per_device_cost,
        per_device_on_minutes=per_device_on_minutes,
    )


def simulate_period(
    project: Project,
    settings: SettingsModel,
    start_date: Optional[datetime.date],
    total_minutes: int,
) -> SimTotals:
    if total_minutes <= 0:
        device_count = len(project.devices)
        return SimTotals(
            total_kwh=0.0,
            total_cost=0.0,
            per_device_kwh=[0.0] * device_count,
            per_device_cost=[0.0] * device_count,
            per_device_on_minutes=[0.0] * device_count,
        )
    if start_date is None:
        start_date = datetime.date.today()

    total_minutes = int(total_minutes)
    day_count = max(1, math.ceil(total_minutes / MINUTES_PER_DAY))
    device_count = len(project.devices)
    per_device_kwh = [0.0] * device_count
    per_device_cost = [0.0] * device_count
    per_device_on_minutes = [0.0] * device_count
    total_kwh = 0.0
    total_cost = 0.0

    for offset in range(day_count):
        minutes_remaining = total_minutes - offset * MINUTES_PER_DAY
        if minutes_remaining <= 0:
            break
        day_minutes = min(MINUTES_PER_DAY, minutes_remaining)
        day = start_date + datetime.timedelta(days=offset)
        if day_minutes == MINUTES_PER_DAY:
            day_result = simulate_single_day(project, settings, day)
            total_kwh += day_result.total_kwh_day
            total_cost += day_result.total_cost_day
            for i in range(device_count):
                per_device_kwh[i] += day_result.per_device_kwh_day[i]
                per_device_cost[i] += day_result.per_device_cost_day[i]
                per_device_on_minutes[i] += day_result.per_device_on_minutes[i]
        else:
            day_totals = simulate_single_day_window(project, settings, day, 0, day_minutes)
            total_kwh += day_totals.total_kwh
            total_cost += day_totals.total_cost
            for i in range(device_count):
                per_device_kwh[i] += day_totals.per_device_kwh[i]
                per_device_cost[i] += day_totals.per_device_cost[i]
                per_device_on_minutes[i] += day_totals.per_device_on_minutes[i]

    return SimTotals(
        total_kwh=total_kwh,
        total_cost=total_cost,
        per_device_kwh=per_device_kwh,
        per_device_cost=per_device_cost,
        per_device_on_minutes=per_device_on_minutes,
    )


def simulate_day(
    project: Project,
    settings: SettingsModel,
    day: Optional[datetime.date] = None,
) -> SimResult:
    if day is None:
        day = datetime.date.today()
    return simulate_range(project, settings, day, day).day_results[0]


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


SIMULATION_LENGTH_PRESETS: List[Tuple[str, str, float]] = [
    ("1_day", "1 day", 1.0),
    ("1_week", "1 week", 7.0),
    ("1_month", "1 month", 30.0),
]


def simulation_length_key_for_period(period: CustomPeriod) -> str:
    duration = f"{period.duration:g}"
    return f"custom:{slugify(period.name)}:{duration}:{period.unit}"


def simulation_length_options(
    settings: SettingsModel,
    custom_periods: Iterable[CustomPeriod],
) -> List[Tuple[str, str, int]]:
    options: List[Tuple[str, str, int]] = []
    for key, label, days in SIMULATION_LENGTH_PRESETS:
        minutes = max(1, int(days * MINUTES_PER_DAY))
        options.append((key, label, minutes))
    for period in custom_periods:
        days = custom_period_to_days(period, settings)
        minutes = max(1, int(round(days * MINUTES_PER_DAY)))
        label = f"{period.name} ({period.duration:g} {period.unit})"
        options.append((simulation_length_key_for_period(period), label, minutes))
    return options


def simulation_length_minutes(
    settings: SettingsModel,
    custom_periods: Iterable[CustomPeriod],
) -> int:
    options = simulation_length_options(settings, custom_periods)
    for key, _label, minutes in options:
        if key == settings.simulation_length_key:
            return minutes
    for key, _label, minutes in options:
        if key == "1_week":
            return minutes
    return MINUTES_PER_DAY
