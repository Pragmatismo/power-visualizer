import copy
import datetime
import json
from html import escape
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from astral import Observer
from astral.sun import sun
from PySide6.QtCore import (
    Qt, QRect, QRectF, QPoint, QPointF, QSize, QSettings, QTimer, QEvent, Signal, QDate
)
from PySide6.QtGui import (
    QAction, QBrush, QColor, QFont, QPainter, QPen, QCursor, QFontMetrics
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QToolButton,
    QFileDialog, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QDialog, QFormLayout, QDialogButtonBox, QGroupBox, QCheckBox,
    QLineEdit, QHeaderView, QListWidget, QListWidgetItem, QScrollArea,
    QInputDialog, QAbstractScrollArea, QCalendarWidget, QSizePolicy
)

from powervis.models import (
    Interval,
    Event,
    DeviceType,
    Device,
    TariffWindow,
    FreeRule,
    CustomPeriod,
    SettingsModel,
    Project,
    TariffDocument,
    TariffEntry,
    device_from_catalog_item,
)
from powervis.simulate import (
    SimResult,
    SimRangeResult,
    tariff_rate_for_minute,
    tariff_segments,
    build_tariff_minute_rates,
    format_tariff_rate,
    tariff_price_summary,
    load_tariff_document,
    simulate_single_day,
    simulate_range,
    simulate_day,
    custom_period_to_days,
    simulation_length_options,
)
from powervis.utils import (
    MINUTES_PER_DAY,
    clamp,
    parse_hhmm,
    minutes_to_time,
    format_duration_minutes,
    format_duration_hhmm,
    format_duration_export,
    parse_duration_text,
    slugify,
)

CITY_LOCATIONS: List[Tuple[str, Optional[float], Optional[float]]] = [
    ("Custom", None, None),
    ("London, UK", 51.5074, -0.1278),
    ("New York, USA", 40.7128, -74.0060),
    ("Baltimore, USA", 39.2904, -76.6122),
    ("Los Angeles, USA", 34.0522, -118.2437),
    ("Toronto, Canada", 43.6532, -79.3832),
    ("Sydney, Australia", -33.8688, 151.2093),
    ("Melbourne, Australia", -37.8136, 144.9631),
    ("Tokyo, Japan", 35.6762, 139.6503),
    ("Singapore", 1.3521, 103.8198),
    ("Paris, France", 48.8566, 2.3522),
    ("Berlin, Germany", 52.5200, 13.4050),
    ("Dubai, UAE", 25.2048, 55.2708),
    ("Mumbai, India", 19.0760, 72.8777),
    ("São Paulo, Brazil", -23.5505, -46.6333),
    ("Cape Town, South Africa", -33.9249, 18.4241),
]


class SettingsDialog(QDialog):
    def __init__(
        self,
        parent=None,
        settings_model: Optional[SettingsModel] = None,
        custom_periods: Optional[List[CustomPeriod]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.model = settings_model or SettingsModel()
        self.custom_periods = custom_periods or []

        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.currency_symbol = QLineEdit(self.model.currency_symbol)
        self.currency_symbol.setMaxLength(4)
        form.addRow("Currency symbol", self.currency_symbol)

        self.mains_voltage = QDoubleSpinBox()
        self.mains_voltage.setDecimals(1)
        self.mains_voltage.setRange(1.0, 1000.0)
        self.mains_voltage.setSingleStep(1.0)
        self.mains_voltage.setValue(self.model.mains_voltage)
        form.addRow("Mains voltage (V)", self.mains_voltage)

        self.simulation_length = QComboBox()
        self._simulation_options = simulation_length_options(
            self.model, self.custom_periods
        )
        for key, label, _minutes in self._simulation_options:
            self.simulation_length.addItem(label, key)
        default_index = next(
            (i for i, (key, _label, _minutes) in enumerate(self._simulation_options) if key == "1_week"),
            0,
        )
        current_index = next(
            (
                i
                for i, (key, _label, _minutes) in enumerate(self._simulation_options)
                if key == self.model.simulation_length_key
            ),
            default_index,
        )
        self.simulation_length.setCurrentIndex(current_index)
        form.addRow("Simulation length", self.simulation_length)

        location_group = QGroupBox("Location")
        location_layout = QFormLayout(location_group)

        self.location_combo = QComboBox()
        for name, lat, lon in CITY_LOCATIONS:
            self.location_combo.addItem(name, (lat, lon))
        location_layout.addRow("City", self.location_combo)

        self.location_lat = QDoubleSpinBox()
        self.location_lat.setDecimals(5)
        self.location_lat.setRange(-90.0, 90.0)
        self.location_lat.setSingleStep(0.01)
        self.location_lat.setValue(self.model.location_lat)
        location_layout.addRow("Latitude", self.location_lat)

        self.location_lon = QDoubleSpinBox()
        self.location_lon.setDecimals(5)
        self.location_lon.setRange(-180.0, 180.0)
        self.location_lon.setSingleStep(0.01)
        self.location_lon.setValue(self.model.location_lon)
        location_layout.addRow("Longitude", self.location_lon)

        layout.addWidget(location_group)

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
        self.location_combo.currentIndexChanged.connect(self._on_location_changed)
        self._set_location_from_model()
        self._update_enabled()

    def _set_location_from_model(self):
        index = 0
        for i, (name, lat, lon) in enumerate(CITY_LOCATIONS):
            if name == self.model.location_label and lat is not None and lon is not None:
                index = i
                self.location_lat.setValue(lat)
                self.location_lon.setValue(lon)
                break
        else:
            self.location_lat.setValue(self.model.location_lat)
            self.location_lon.setValue(self.model.location_lon)
        self.location_combo.setCurrentIndex(index)
        self._set_location_enabled()

    def _set_location_enabled(self):
        is_custom = self.location_combo.currentText() == "Custom"
        self.location_lat.setEnabled(is_custom)
        self.location_lon.setEnabled(is_custom)

    def _on_location_changed(self):
        lat, lon = self.location_combo.currentData()
        if lat is not None and lon is not None:
            self.location_lat.setValue(lat)
            self.location_lon.setValue(lon)
        self._set_location_enabled()

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
        m.mains_voltage = float(self.mains_voltage.value())

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
        current_label = self.location_combo.currentText()
        m.location_label = current_label or "Custom"
        m.location_lat = float(self.location_lat.value())
        m.location_lon = float(self.location_lon.value())
        m.simulation_length_key = str(self.simulation_length.currentData())
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

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Select", "Supplier", "Tariff", "Price", "Off peak", "Peak times"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
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
        rows: List[Tuple[str, str, str, str, str]] = []
        rows.append(("Custom", "", "Use Settings", "", ""))
        self._entries.append(None)

        for doc in self._documents:
            for supplier_index, supplier in enumerate(doc.data.get("suppliers", [])):
                for tariff_index, tariff in enumerate(supplier.get("tariffs", [])):
                    price, offpeak, peak_times = tariff_price_summary(tariff, self._currency_symbol)
                    rows.append((supplier.get("supplier_name", ""), tariff.get("tariff_name", ""), price, offpeak, peak_times))
                    self._entries.append(TariffEntry(doc, supplier_index, tariff_index))

        self.table.setRowCount(len(rows))
        for row_index, (supplier_name, tariff_name, price, offpeak, peak_times) in enumerate(rows):
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
            self.table.setItem(row_index, 3, QTableWidgetItem(price))
            self.table.setItem(row_index, 4, QTableWidgetItem(offpeak))
            self.table.setItem(row_index, 5, QTableWidgetItem(peak_times))

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

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Show",
            "Category",
            "Name",
            "Power (W)",
            "Usage duration (H:M)",
            "Variable",
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
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

            category_item = QTableWidgetItem(dev.category)
            self.table.setItem(row, 1, category_item)

            name_item = QTableWidgetItem(dev.name)
            self.table.setItem(row, 2, name_item)

            power_item = QTableWidgetItem(f"{dev.power_w:g}")
            self.table.setItem(row, 3, power_item)

            duration_item = QTableWidgetItem(format_duration_minutes(dev.default_duration_min))
            self.table.setItem(row, 4, duration_item)

            variable_item = QTableWidgetItem("")
            variable_item.setFlags(variable_item.flags() | Qt.ItemIsUserCheckable)
            variable_item.setFlags(variable_item.flags() & ~Qt.ItemIsEditable)
            variable_item.setCheckState(Qt.Checked if dev.variable else Qt.Unchecked)
            self.table.setItem(row, 5, variable_item)
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
            dev.category = item.text().strip() or dev.category
            updated = True
        elif col == 2:
            dev.name = item.text().strip() or dev.name
            updated = True
        elif col == 3:
            try:
                dev.power_w = max(0.0, float(item.text()))
                updated = True
            except Exception:
                self.table.blockSignals(True)
                item.setText(f"{dev.power_w:g}")
                self.table.blockSignals(False)
        elif col == 4:
            parsed = parse_duration_text(item.text())
            if parsed is None:
                self.table.blockSignals(True)
                item.setText(format_duration_minutes(dev.default_duration_min))
                self.table.blockSignals(False)
            else:
                dev.default_duration_min = parsed
                updated = True
        elif col == 5:
            dev.variable = (item.checkState() == Qt.Checked)
            updated = True

        if updated:
            dev.apply_usage_settings()
            self.table.blockSignals(True)
            self.table.item(row, 4).setText(format_duration_minutes(dev.default_duration_min))
            self.table.blockSignals(False)
            self._notify_parent()

    def _add_device(self):
        if not self.project:
            return
        dev = Device(
            name="Device",
            category="Custom",
            dtype=DeviceType.SCHEDULED,
            power_w=20.0,
            enabled=False,
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
# Device import dialog
# =========================

class DeviceImportDialog(QDialog):
    def __init__(self, parent=None, categories: Optional[List[str]] = None):
        super().__init__(parent)
        self.setWindowTitle("Import Devices")
        self.setModal(True)
        self._show_device_list = False
        self._checkboxes: Dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        list_container = QWidget()
        list_layout = QVBoxLayout(list_container)
        list_layout.setContentsMargins(4, 4, 4, 4)

        for category in categories or []:
            checkbox = QCheckBox(category)
            checkbox.setChecked(True)
            list_layout.addWidget(checkbox)
            self._checkboxes[category] = checkbox

        list_layout.addStretch(1)
        scroll.setWidget(list_container)
        layout.addWidget(scroll)

        self.clear_checkbox = QCheckBox("Clear current devices")
        layout.addWidget(self.clear_checkbox)

        buttons = QDialogButtonBox()
        ok_btn = buttons.addButton("Ok", QDialogButtonBox.AcceptRole)
        cancel_btn = buttons.addButton("Cancel", QDialogButtonBox.RejectRole)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self._accept_and_show_list)
        layout.addWidget(buttons)

    def _accept_and_show_list(self):
        self._show_device_list = True
        self.accept()

    def get_result(self) -> Tuple[List[str], bool, bool]:
        selected = [
            name for name, checkbox in self._checkboxes.items()
            if checkbox.isChecked()
        ]
        return selected, self.clear_checkbox.isChecked(), self._show_device_list

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
    day_offset: int = 0
    # for intervals/events: original start/end for drag operations
    anchor_minute: int = 0


class TimelineWidget(QAbstractScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.project: Optional[Project] = None
        self.settings: Optional[SettingsModel] = None
        self.sim: Optional[SimResult] = None
        self.reference_date: Optional[datetime.date] = None
        self.visible_device_indices: List[int] = []
        self.row_items: List[Dict[str, object]] = []

        # layout constants
        self.left_label_w = 220
        self.top_tariff_h = 56
        self.day_night_gap = 4
        self.row_h = 44
        self.row_gap = 6
        self.axis_h = 32
        self.label_pad = 12
        self.axis_top_pad = 10
        self.axis_bottom_pad = 6
        self.axis_label_padding = 4
        self.axis_labels_rotated = False
        self.axis_label_rot_height = 0
        self.axis_day_title_height = 0
        self.axis_day_title_gap = 4
        self.pixels_per_minute = 1.0
        self.scroll_offset_px = 0
        self.timeline_start_min = 0
        self.timeline_end_min = MINUTES_PER_DAY

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
        self.category_font = QFont("Sans", 9, QFont.Bold)
        self.tariff_company_font = QFont(self.name_font)
        self.tariff_company_font.setWeight(QFont.Bold)
        self.tariff_label_font = QFont(self.name_font)
        self.tariff_label_font.setWeight(QFont.Normal)
        self.day_night_font = QFont("Sans", 10)
        self.power_font = QFont("Sans", 9)
        self.info_font = QFont("Sans", 11)
        self.setMinimumHeight(400)
        self.horizontalScrollBar().valueChanged.connect(self._on_scroll)
        self.info_panel: Optional["TimelineTotalsPanel"] = None

    def set_info_panel(self, panel: Optional["TimelineTotalsPanel"]):
        self.info_panel = panel
        if self.info_panel:
            self.info_panel.sync_from_timeline()

    def set_data(
        self,
        project: Project,
        settings: SettingsModel,
        sim: SimResult,
        day: Optional[datetime.date] = None,
    ):
        self.project = project
        self.settings = settings
        self.sim = sim
        self.reference_date = day
        self._update_visible_devices()
        self._build_row_items()
        self._update_left_label_width()
        self._update_axis_layout()
        self._update_scrollbar()
        self.updateGeometry()
        self._refresh_viewport()
        if self.info_panel:
            self.info_panel.sync_from_timeline()

    def _update_visible_devices(self):
        if not self.project:
            self.visible_device_indices = []
            return
        self.visible_device_indices = [
            idx for idx, dev in enumerate(self.project.devices) if dev.enabled
        ]
        if self.settings and self.settings.sort_by_category:
            self.visible_device_indices.sort(
                key=lambda idx: (
                    (self.project.devices[idx].category or "").strip().lower(),
                    self.project.devices[idx].name.strip().lower(),
                    idx,
                )
            )

    def _build_row_items(self):
        self.row_items = []
        if not self.project:
            return
        if not (self.settings and self.settings.sort_by_category):
            for idx in self.visible_device_indices:
                self.row_items.append({"type": "device", "device_index": idx})
            return
        current_category = None
        for idx in self.visible_device_indices:
            dev = self.project.devices[idx]
            category = (dev.category or "").strip() or "Uncategorized"
            if category != current_category:
                self.row_items.append({"type": "category", "label": category})
                current_category = category
            self.row_items.append({"type": "device", "device_index": idx})

    def _update_left_label_width(self):
        if not self.project:
            self.left_label_w = 220
            return
        name_metrics = QFontMetrics(self.name_font)
        power_metrics = QFontMetrics(self.power_font)
        category_metrics = QFontMetrics(self.category_font)
        widest = 0
        for item in self.row_items:
            if item["type"] == "device":
                dev = self.project.devices[item["device_index"]]
                widest = max(
                    widest,
                    name_metrics.horizontalAdvance(dev.name),
                    power_metrics.horizontalAdvance(f"{dev.power_w:g} W"),
                )
            elif item["type"] == "category":
                widest = max(widest, category_metrics.horizontalAdvance(str(item["label"])))
        if self.settings and self.settings.show_total_power:
            widest = max(widest, name_metrics.horizontalAdvance("Total Power"))
        self.left_label_w = max(160, widest + self.label_pad * 2)

    def _update_axis_layout(self):
        if not self.project:
            self.axis_labels_rotated = False
            self.axis_h = 32
            return
        mode = self._axis_mode()
        tick_interval = self._axis_tick_interval()
        tick_spacing = max(1, int(self.pixels_per_minute * tick_interval))
        metrics = QFontMetrics(self.font)
        title_metrics = QFontMetrics(self.name_font)
        show_day_titles = self._day_count() > 1
        self.axis_day_title_height = title_metrics.height() if show_day_titles else 0
        label_height = metrics.height() if mode != "days_only" else 0
        label_width = metrics.horizontalAdvance("24") if mode != "days_only" else 0
        needs_rotation = mode != "days_only" and label_width + self.axis_label_padding > tick_spacing
        if needs_rotation:
            angle = math.radians(45)
            rot_height = abs(label_width * math.sin(angle)) + abs(label_height * math.cos(angle))
            self.axis_label_rot_height = int(rot_height)
            label_area = int(rot_height)
        else:
            self.axis_label_rot_height = label_height
            label_area = label_height
        title_gap = self.axis_day_title_gap if self.axis_day_title_height and label_area else 0
        self.axis_h = max(
            32,
            self.axis_day_title_height + title_gap + label_area + self.axis_top_pad + self.axis_bottom_pad,
        )
        self.axis_labels_rotated = needs_rotation

    def sizeHint(self):
        if not self.project:
            return QSize(900, 500)
        total_row_h = self._total_row_height()
        total_row_gap = self.row_gap if total_row_h > 0 else 0
        h = (
            self._axis_top()
            + self.axis_h
            + (len(self.row_items) * (self.row_h + self.row_gap))
            + total_row_h
            + total_row_gap
            + 20
        )
        return QSize(1100, max(500, h))

    def _day_night_bar_height(self) -> int:
        if not (self.settings and self.settings.show_day_night):
            return 0
        metrics = QFontMetrics(self.day_night_font)
        return metrics.height() + 6

    def _day_night_block_height(self) -> int:
        bar_height = self._day_night_bar_height()
        if bar_height <= 0:
            return 0
        return bar_height + self.day_night_gap * 2

    def _axis_top(self) -> int:
        return self.top_tariff_h + self._day_night_block_height()

    def _timeline_rect(self) -> QRect:
        vp = self.viewport().rect()
        return QRect(
            self.left_label_w,
            self._axis_top() + self.axis_h,
            vp.width() - self.left_label_w,
            vp.height() - self._axis_top() - self.axis_h - 12,
        )

    def timeline_draw_width(self) -> int:
        return self._timeline_rect().width()

    def _minute_to_x(self, minute: int) -> int:
        tl = self._timeline_rect()
        offset = self._scroll_offset()
        return int(
            tl.left() + (minute - self.timeline_start_min) * self.pixels_per_minute - offset
        )

    def _x_to_minute(self, x: int) -> int:
        tl = self._timeline_rect()
        offset = self._scroll_offset()
        minute = self.timeline_start_min + (x - tl.left() + offset) / max(self.pixels_per_minute, 0.01)
        return clamp(int(minute), self.timeline_start_min, self.timeline_end_min - 1)

    def _row_rect(self, idx: int) -> QRect:
        tl = self._timeline_rect()
        y0 = tl.top() + idx * (self.row_h + self.row_gap)
        return QRect(tl.left(), y0, tl.width(), self.row_h)

    def _total_row_height(self) -> int:
        if not (self.settings and self.settings.show_total_power):
            return 0
        return max(self.row_h, self.row_h * 2 - 16)

    def _total_row_rect(self) -> QRect:
        tl = self._timeline_rect()
        y0 = tl.top() + len(self.row_items) * (self.row_h + self.row_gap)
        return QRect(tl.left(), y0, tl.width(), self._total_row_height())

    def _device_label_rect(self, idx: int) -> QRect:
        rr = self._row_rect(idx)
        return QRect(self.label_pad, rr.top(), self.left_label_w - self.label_pad * 2, rr.height())

    def _tariff_rect(self) -> QRect:
        vp = self.viewport().rect()
        return QRect(
            self.left_label_w,
            6,
            vp.width() - self.left_label_w,
            self.top_tariff_h - 8,
        )

    def _day_night_rect(self) -> QRect:
        vp = self.viewport().rect()
        bar_height = self._day_night_bar_height()
        if bar_height <= 0:
            return QRect()
        return QRect(
            self.left_label_w,
            self.top_tariff_h + self.day_night_gap,
            vp.width() - self.left_label_w,
            bar_height,
        )

    def _scroll_offset(self) -> int:
        return self.horizontalScrollBar().value()

    def _content_width(self) -> int:
        return max(1, int((self.timeline_end_min - self.timeline_start_min) * self.pixels_per_minute))

    def _day_count(self) -> int:
        total_minutes = max(1, self.timeline_end_min - self.timeline_start_min)
        return max(1, math.ceil(total_minutes / MINUTES_PER_DAY))

    def _day_key(self, day_offset: int) -> Optional[str]:
        if not self.reference_date:
            return None
        day = self.reference_date + datetime.timedelta(days=day_offset)
        return day.isoformat()

    def _sunrise_sunset_minutes(self, day: datetime.date) -> Optional[Tuple[int, int]]:
        if not self.settings:
            return None
        observer = Observer(
            latitude=self.settings.location_lat,
            longitude=self.settings.location_lon,
            elevation=0.0,
        )
        tzinfo = datetime.datetime.now().astimezone().tzinfo
        try:
            times = sun(observer, date=day, tzinfo=tzinfo)
        except Exception as exc:
            message = str(exc).lower()
            if "always above" in message:
                return 0, MINUTES_PER_DAY
            if "always below" in message:
                return 0, 0
            return None
        sunrise = times.get("sunrise")
        sunset = times.get("sunset")
        if not (sunrise and sunset):
            return None
        sunrise_min = int(round(sunrise.hour * 60 + sunrise.minute + sunrise.second / 60))
        sunset_min = int(round(sunset.hour * 60 + sunset.minute + sunset.second / 60))
        sunrise_min = clamp(sunrise_min, 0, MINUTES_PER_DAY)
        sunset_min = clamp(sunset_min, 0, MINUTES_PER_DAY)
        if sunset_min < sunrise_min:
            sunrise_min, sunset_min = sunset_min, sunrise_min
        return sunrise_min, sunset_min

    def _intervals_for_day(self, dev: Device, day_offset: int) -> List[Interval]:
        key = self._day_key(day_offset)
        if key and key in dev.day_intervals:
            return dev.day_intervals[key]
        return dev.intervals

    def _events_for_day(self, dev: Device, day_offset: int) -> List[Event]:
        key = self._day_key(day_offset)
        if key and key in dev.day_events:
            return dev.day_events[key]
        return dev.events

    def _ensure_day_intervals(self, dev: Device, day_offset: int) -> Tuple[List[Interval], Optional[str]]:
        key = self._day_key(day_offset)
        if not key:
            return dev.intervals, None
        if key not in dev.day_intervals:
            dev.day_intervals[key] = [Interval(iv.start_min, iv.end_min).normalized() for iv in dev.intervals]
        return dev.day_intervals[key], key

    def _ensure_day_events(self, dev: Device, day_offset: int) -> Tuple[List[Event], Optional[str]]:
        key = self._day_key(day_offset)
        if not key:
            return dev.events, None
        if key not in dev.day_events:
            dev.day_events[key] = [Event(ev.start_min, ev.duration_min, ev.energy_wh).normalized() for ev in dev.events]
        return dev.day_events[key], key

    def _interval_signature(self, interval: Interval) -> Tuple[int, int]:
        ivn = interval.normalized()
        return ivn.start_min, ivn.end_min

    def _event_signature(self, event: Event) -> Tuple[int, int]:
        evn = event.normalized()
        return evn.start_min, evn.duration_min

    def _remove_matching_interval(self, dev: Device, signature: Tuple[int, int]) -> None:
        dev.intervals = [iv for iv in dev.intervals if self._interval_signature(iv) != signature]
        for day_key, intervals in list(dev.day_intervals.items()):
            dev.day_intervals[day_key] = [
                iv for iv in intervals if self._interval_signature(iv) != signature
            ]
            if not dev.day_intervals[day_key]:
                dev.day_intervals.pop(day_key, None)

    def _remove_matching_event(self, dev: Device, signature: Tuple[int, int]) -> None:
        dev.events = [ev for ev in dev.events if self._event_signature(ev) != signature]
        for day_key, events in list(dev.day_events.items()):
            dev.day_events[day_key] = [
                ev for ev in events if self._event_signature(ev) != signature
            ]
            if not dev.day_events[day_key]:
                dev.day_events.pop(day_key, None)

    def _append_interval_unique(self, intervals: List[Interval], interval: Interval) -> None:
        signature = self._interval_signature(interval)
        if any(self._interval_signature(iv) == signature for iv in intervals):
            return
        intervals.append(interval)

    def _append_event_unique(self, events: List[Event], event: Event) -> None:
        signature = self._event_signature(event)
        if any(self._event_signature(ev) == signature for ev in events):
            return
        events.append(event)

    def _day_bounds(self, day_offset: int) -> Tuple[int, int]:
        day_start = day_offset * MINUTES_PER_DAY
        day_end = min(self.timeline_end_min, (day_offset + 1) * MINUTES_PER_DAY)
        return day_start, day_end

    def _visible_minutes(self) -> int:
        total_minutes = max(1, self.timeline_end_min - self.timeline_start_min)
        draw_width = max(1, self.timeline_draw_width())
        visible_minutes = draw_width / max(self.pixels_per_minute, 0.01)
        return max(1, min(total_minutes, int(math.ceil(visible_minutes))))

    def _update_scrollbar(self):
        tl = self._timeline_rect()
        visible_width = max(1, tl.width())
        max_offset = max(0, self._content_width() - visible_width)
        bar = self.horizontalScrollBar()
        bar.setPageStep(visible_width)
        bar.setRange(0, max_offset)
        if bar.value() > max_offset:
            bar.setValue(max_offset)

    def _on_scroll(self, value: int):
        self.scroll_offset_px = value
        self._refresh_viewport()

    def _axis_mode(self) -> str:
        visible_minutes = self._visible_minutes()
        if visible_minutes > 14 * MINUTES_PER_DAY:
            return "days_only"
        if visible_minutes > MINUTES_PER_DAY:
            return "multi_day"
        return "single_day"

    def _axis_tick_interval(self) -> int:
        visible_minutes = self._visible_minutes()
        if visible_minutes > 14 * MINUTES_PER_DAY:
            return 24 * 60
        if visible_minutes > 2 * MINUTES_PER_DAY:
            return 12 * 60
        return 60

    def _axis_tick_label(self, minute: int, mode: str) -> Optional[str]:
        if mode == "days_only":
            return None
        if minute == self.timeline_end_min and minute % MINUTES_PER_DAY == 0 and mode == "multi_day":
            return "24"
        if minute >= self.timeline_end_min:
            return None
        hour = (minute // 60) % 24
        return f"{hour:d}"

    def set_pixels_per_minute(self, pixels_per_minute: float):
        self.pixels_per_minute = max(0.1, pixels_per_minute)
        self._update_axis_layout()
        self._update_scrollbar()
        self._refresh_viewport()
        if self.info_panel:
            self.info_panel.sync_from_timeline()

    def set_time_range(self, start_min: int, end_min: int):
        start = int(start_min)
        end = int(end_min)
        if end <= start:
            end = start + 1
        self.timeline_start_min = start
        self.timeline_end_min = end
        self._update_axis_layout()
        self._update_scrollbar()
        self._refresh_viewport()
        if self.info_panel:
            self.info_panel.sync_from_timeline()

    def _refresh_viewport(self):
        self.viewport().update()
        if self.info_panel:
            self.info_panel.update()

    def paintEvent(self, event):
        p = QPainter(self.viewport())
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.viewport().rect(), QColor(25, 25, 28))
        p.setFont(self.font)

        if not (self.project and self.settings and self.sim):
            p.setPen(QColor(220, 220, 220))
            p.drawText(self.viewport().rect(), Qt.AlignCenter, "No project loaded.")
            return

        # Draw tariff bar
        self._paint_tariff(p)

        # Draw day/night bar
        self._paint_day_night(p)

        # Draw time axis
        self._paint_axis(p)

        # Draw device rows
        for row_index, item in enumerate(self.row_items):
            if item["type"] == "category":
                self._paint_category_row(p, row_index, str(item["label"]))
            else:
                dev_index = int(item["device_index"])
                dev = self.project.devices[dev_index]
                self._paint_row(p, row_index, dev_index, dev)
        self._paint_total_row(p)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_axis_layout()
        self._update_scrollbar()
        if self.info_panel:
            self.info_panel.sync_from_timeline()

    def _paint_tariff(self, p: QPainter):
        tr = self._tariff_rect()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(12, 20, 40))
        p.drawRect(tr)

        day_count = self._day_count()
        segment_label = f"{self.settings.currency_symbol}{{rate:.2f}}/kWh"
        for day_offset in range(day_count):
            day_start = day_offset * MINUTES_PER_DAY
            day = None
            if self.reference_date:
                day = self.reference_date + datetime.timedelta(days=day_offset)

            # base rate visualization
            tl = self._timeline_rect()
            offset = self._scroll_offset()

            def minute_to_xf(minute: int) -> float:
                return (
                    tl.left()
                    + (minute - self.timeline_start_min) * self.pixels_per_minute
                    - offset
                )

            for hour in range(24):
                m0 = day_start + hour * 60
                m1 = day_start + (hour + 1) * 60
                r = tariff_rate_for_minute(self.settings, hour * 60, day)
                # map rate to brightness (simple)
                # bigger rate -> brighter
                base = clamp(int(40 + 180 * r), 40, 200)
                col = QColor(
                    clamp(15 + int(40 * r), 15, 80),
                    clamp(30 + int(60 * r), 30, 120),
                    clamp(90 + base, 90, 220),
                )
                x0 = minute_to_xf(m0)
                x1 = minute_to_xf(m1)
                p.fillRect(QRectF(x0, tr.top(), x1 - x0, tr.height()), col)

            for start_min, end_min, rate in tariff_segments(self.settings, day):
                x0 = self._minute_to_x(day_start + start_min)
                x1 = self._minute_to_x(day_start + end_min)
                if x1 - x0 < 40:
                    continue
                label_rect = QRect(x0 + 4, tr.top(), x1 - x0 - 8, tr.height())
                p.setFont(self.tariff_font)
                p.setPen(QColor(230, 235, 245))
                p.drawText(label_rect, Qt.AlignCenter, segment_label.format(rate=rate))

        # free overlay
        fr = self.settings.free_rule.normalized()
        if fr.enabled:
            for day_offset in range(day_count):
                day_start = day_offset * MINUTES_PER_DAY
                x0 = self._minute_to_x(day_start + fr.start_min)
                x1 = self._minute_to_x(day_start + fr.end_min)
                overlay = QColor(80, 160, 80, 140)
                p.setBrush(overlay)
                p.drawRect(QRect(x0, tr.top(), x1 - x0, tr.height()))
                p.setFont(self.tariff_font)
                p.setPen(QColor(230, 235, 245))
                p.drawText(
                    QRect(x0, tr.top(), x1 - x0, tr.height()),
                    Qt.AlignCenter,
                    f"FREE ≤ {fr.free_kw_threshold:.1f} kW",
                )

        # border + label
        p.setPen(QPen(QColor(120, 120, 130), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(tr)
        p.setPen(QColor(230, 235, 245))
        label_width = self.left_label_w - self.label_pad * 2
        label_right = tr.left() - 4
        label_rect = QRect(label_right - label_width, tr.top(), label_width, tr.height())
        label = self.settings.tariff_label or "Tariff"
        if " — " in label:
            company_name, tariff_name = label.split(" — ", 1)
        elif " - " in label:
            company_name, tariff_name = label.split(" - ", 1)
        else:
            company_name, tariff_name = label, ""
        company_metrics = QFontMetrics(self.tariff_company_font)
        label_metrics = QFontMetrics(self.tariff_label_font)
        line_gap = 2
        total_h = company_metrics.height() + (label_metrics.height() if tariff_name else 0)
        if tariff_name:
            total_h += line_gap
        start_y = label_rect.top() + (label_rect.height() - total_h) // 2
        company_rect = QRect(label_rect.left(), start_y, label_rect.width(), company_metrics.height())
        p.setFont(self.tariff_company_font)
        p.drawText(company_rect, Qt.AlignRight | Qt.AlignVCenter, company_name)
        if tariff_name:
            tariff_rect = QRect(
                label_rect.left(),
                company_rect.bottom() + 1 + line_gap,
                label_rect.width(),
                label_metrics.height(),
            )
            p.setFont(self.tariff_label_font)
            p.drawText(tariff_rect, Qt.AlignRight | Qt.AlignVCenter, tariff_name)

    def _paint_day_night(self, p: QPainter):
        bar_rect = self._day_night_rect()
        if bar_rect.width() <= 0 or bar_rect.height() <= 0:
            return
        if not self.settings:
            return
        day_color = QColor(230, 200, 85)
        night_color = QColor(20, 32, 70)
        day_text_color = QColor(40, 35, 20)
        night_text_color = QColor(225, 230, 245)
        base_date = self.reference_date or datetime.date.today()
        metrics = QFontMetrics(self.day_night_font)
        p.setFont(self.day_night_font)
        p.setPen(Qt.NoPen)
        day_count = self._day_count()
        for day_offset in range(day_count):
            day_start = day_offset * MINUTES_PER_DAY
            day_end = min((day_offset + 1) * MINUTES_PER_DAY, self.timeline_end_min)
            x0 = self._minute_to_x(day_start)
            x1 = self._minute_to_x(day_end)
            if x1 <= x0:
                continue
            p.setBrush(night_color)
            p.drawRect(QRectF(x0, bar_rect.top(), x1 - x0, bar_rect.height()))
            sunrise_sunset = self._sunrise_sunset_minutes(base_date + datetime.timedelta(days=day_offset))
            if sunrise_sunset is None:
                sunrise_min = 6 * 60
                sunset_min = 18 * 60
            else:
                sunrise_min, sunset_min = sunrise_sunset
            day_start_min = day_start + sunrise_min
            day_end_min = day_start + sunset_min
            day_x0 = self._minute_to_x(day_start_min)
            day_x1 = self._minute_to_x(day_end_min)
            if day_x1 > day_x0:
                p.setBrush(day_color)
                p.drawRect(QRectF(day_x0, bar_rect.top(), day_x1 - day_x0, bar_rect.height()))

            day_length = max(0, sunset_min - sunrise_min)
            night_length = max(0, MINUTES_PER_DAY - day_length)
            day_label = format_duration_hhmm(day_length)
            night_label = format_duration_hhmm(night_length)

            day_label_width = metrics.horizontalAdvance(day_label)
            if day_x1 - day_x0 >= day_label_width + 6:
                p.setPen(day_text_color)
                p.drawText(
                    QRectF(day_x0, bar_rect.top(), day_x1 - day_x0, bar_rect.height()),
                    Qt.AlignCenter | Qt.AlignVCenter,
                    day_label,
                )

            night1 = sunrise_min
            night2 = MINUTES_PER_DAY - sunset_min
            night_label_start = day_start
            night_label_end = day_start + sunrise_min
            if night2 > night1:
                night_label_start = day_start + sunset_min
                night_label_end = day_start + MINUTES_PER_DAY
            night_x0 = self._minute_to_x(night_label_start)
            night_x1 = self._minute_to_x(night_label_end)
            night_label_width = metrics.horizontalAdvance(night_label)
            if night_x1 - night_x0 >= night_label_width + 6:
                p.setPen(night_text_color)
                p.drawText(
                    QRectF(night_x0, bar_rect.top(), night_x1 - night_x0, bar_rect.height()),
                    Qt.AlignCenter | Qt.AlignVCenter,
                    night_label,
                )

        p.setPen(QPen(QColor(90, 90, 100), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(bar_rect)

    def _paint_axis(self, p: QPainter):
        tl = self._timeline_rect()
        axis_top = self._axis_top()
        axis_baseline = axis_top + self.axis_h - self.axis_bottom_pad + 2
        tick_top = axis_baseline - 10
        label_bottom = axis_baseline - 2
        metrics = QFontMetrics(self.font)
        p.setPen(QColor(160, 160, 170))
        mode = self._axis_mode()
        tick_interval = self._axis_tick_interval()
        # day titles
        if self._day_count() > 1:
            title_y = axis_top + self.axis_top_pad
            p.setFont(self.name_font)
            for day_offset in range(self._day_count()):
                day_start = day_offset * MINUTES_PER_DAY
                day_end = min((day_offset + 1) * MINUTES_PER_DAY, self.timeline_end_min)
                x0 = self._minute_to_x(day_start)
                x1 = self._minute_to_x(day_end)
                if x1 <= x0:
                    continue
                p.drawText(
                    QRect(x0, title_y, x1 - x0, self.axis_day_title_height),
                    Qt.AlignCenter | Qt.AlignVCenter,
                    f"Day {day_offset + 1}",
                )

        # axis ticks
        visible_start = self._x_to_minute(tl.left())
        visible_end = self._x_to_minute(tl.right())
        start_tick = (visible_start // tick_interval) * tick_interval
        if start_tick < visible_start:
            start_tick += tick_interval
        end_tick = min(self.timeline_end_min, visible_end + tick_interval)
        for m in range(start_tick, end_tick + 1, tick_interval):
            x = self._minute_to_x(m)
            p.drawLine(x, tick_top, x, axis_baseline)
            label = self._axis_tick_label(m, mode)
            if label:
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
            self._draw_block(
                p,
                rr,
                0,
                self.timeline_end_min,
                QColor(140, 140, 200, 180),
                hover=False,
            )

        elif dev.dtype == DeviceType.SCHEDULED:
            for day_offset in range(self._day_count()):
                intervals = self._intervals_for_day(dev, day_offset)
                for j, iv in enumerate(intervals):
                    ivn = iv.normalized()
                    hover = (
                        self.hover_hit.kind.startswith("interval")
                        and self.hover_hit.device_index == dev_index
                        and self.hover_hit.item_index == j
                        and self.hover_hit.day_offset == day_offset
                    )
                    duration_text = format_duration_hhmm(ivn.end_min - ivn.start_min)
                    start_min = day_offset * MINUTES_PER_DAY + ivn.start_min
                    end_min = day_offset * MINUTES_PER_DAY + ivn.end_min
                    if start_min >= self.timeline_end_min:
                        continue
                    self._draw_block(
                        p,
                        rr,
                        start_min,
                        min(end_min, self.timeline_end_min),
                        QColor(100, 170, 240, 190),
                        hover=hover,
                        label_text=duration_text,
                    )

        elif dev.dtype == DeviceType.EVENTS:
            for day_offset in range(self._day_count()):
                events = self._events_for_day(dev, day_offset)
                for j, ev in enumerate(events):
                    evn = ev.normalized()
                    hover = (
                        self.hover_hit.kind == HitKind.EVENT_BODY
                        and self.hover_hit.device_index == dev_index
                        and self.hover_hit.item_index == j
                        and self.hover_hit.day_offset == day_offset
                    )
                    start_min = day_offset * MINUTES_PER_DAY + evn.start_min
                    if start_min >= self.timeline_end_min:
                        continue
                    self._draw_event(
                        p,
                        rr,
                        start_min,
                        evn.duration_min,
                        QColor(240, 170, 90, 200),
                        hover=hover,
                    )

        # row border
        p.setPen(QPen(QColor(55, 55, 62), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rr)

    def _paint_category_row(self, p: QPainter, row_index: int, label: str):
        rr = self._row_rect(row_index)
        bg = QColor(22, 22, 26)
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRect(rr)

        lr = self._device_label_rect(row_index)
        p.setPen(QColor(190, 190, 205))
        p.setFont(self.category_font)
        p.drawText(lr, Qt.AlignLeft | Qt.AlignVCenter, label)

        p.setPen(QPen(QColor(55, 55, 62), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rr)

    def _paint_total_row(self, p: QPainter):
        if not (self.project and self.sim and self.settings and self.settings.show_total_power):
            return
        rr = self._total_row_rect()
        bg = QColor(28, 28, 32)
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRect(rr)

        # label left
        lr = QRect(self.label_pad, rr.top(), self.left_label_w - self.label_pad * 2, rr.height())
        p.setPen(QColor(220, 220, 230))
        p.setFont(self.name_font)
        p.drawText(lr, Qt.AlignCenter, "Total Power")

        # draw total power graph
        total_w = self.sim.minute_total_w
        peak = max(total_w) if total_w else 0.0
        graph_h = max(1, rr.height() - 16)
        base_y = rr.bottom() - 8
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(120, 200, 140, 200))
        for day_offset in range(self._day_count()):
            day_start = day_offset * MINUTES_PER_DAY
            for minute in range(MINUTES_PER_DAY):
                timeline_minute = day_start + minute
                if timeline_minute >= self.timeline_end_min:
                    break
                value = total_w[minute] if minute < len(total_w) else 0.0
                ratio = (value / peak) if peak > 0 else 0.0
                bar_h = int(ratio * graph_h)
                if bar_h <= 0:
                    continue
                x0 = self._minute_to_x(timeline_minute)
                x1 = self._minute_to_x(timeline_minute + 1)
                width = max(1, x1 - x0)
                p.drawRect(QRect(x0, base_y - bar_h, width, bar_h))

        p.setPen(QPen(QColor(55, 55, 62), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rr)

        return

    def _draw_block(
        self,
        p: QPainter,
        rr: QRect,
        start_min: int,
        end_min: int,
        color: QColor,
        hover: bool,
        label_text: Optional[str] = None,
    ):
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
        if label_text:
            metrics = QFontMetrics(self.font)
            text_width = metrics.horizontalAdvance(label_text)
            if text_width + 8 <= rect.width():
                p.save()
                p.setFont(self.font)
                p.setPen(QColor(240, 240, 245))
                text_rect = QRect(rect.left() + 4, rect.top(), rect.width() - 8, rect.height())
                p.drawText(text_rect, Qt.AlignCenter, label_text)
                p.restore()

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
        self._refresh_viewport()

    def mousePressEvent(self, e):
        if not self.project:
            return
        pos = e.position().toPoint()

        if e.button() == Qt.RightButton:
            hit = self._hit_test(pos)
            if hit.kind == HitKind.EVENT_BODY:
                dev = self.project.devices[hit.device_index]
                events = self._events_for_day(dev, hit.day_offset)
                if 0 <= hit.item_index < len(events):
                    signature = self._event_signature(events[hit.item_index])
                    if self.settings and self.settings.affect_every_day:
                        self._remove_matching_event(dev, signature)
                    else:
                        events, _ = self._ensure_day_events(dev, hit.day_offset)
                        if 0 <= hit.item_index < len(events):
                            events.pop(hit.item_index)
                    self._trigger_recompute()  # main window call
            elif hit.kind.startswith("interval"):
                dev = self.project.devices[hit.device_index]
                intervals = self._intervals_for_day(dev, hit.day_offset)
                if 0 <= hit.item_index < len(intervals):
                    signature = self._interval_signature(intervals[hit.item_index])
                    if self.settings and self.settings.affect_every_day:
                        self._remove_matching_interval(dev, signature)
                    else:
                        intervals, _ = self._ensure_day_intervals(dev, hit.day_offset)
                        if 0 <= hit.item_index < len(intervals):
                            intervals.pop(hit.item_index)
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
                day_offset = m // MINUTES_PER_DAY
                day_start, day_end = self._day_bounds(day_offset)
                minute_in_day = m - day_start
                if minute_in_day < 0 or m >= day_end:
                    return
                if dev.dtype == DeviceType.SCHEDULED:
                    duration = max(1, dev.default_duration_min)
                    end_in_day = min(minute_in_day + duration, MINUTES_PER_DAY, day_end - day_start)
                    if end_in_day <= minute_in_day:
                        return
                    new_interval = Interval(minute_in_day, end_in_day).normalized()
                    if self.settings and self.settings.affect_every_day:
                        self._append_interval_unique(dev.intervals, new_interval)
                        for intervals in dev.day_intervals.values():
                            self._append_interval_unique(intervals, new_interval)
                    else:
                        intervals, _ = self._ensure_day_intervals(dev, day_offset)
                        self._append_interval_unique(intervals, new_interval)
                    self._trigger_recompute()
                    return
                elif dev.dtype == DeviceType.EVENTS:
                    duration = max(1, dev.default_duration_min)
                    end_in_day = min(minute_in_day + duration, MINUTES_PER_DAY, day_end - day_start)
                    duration = max(1, end_in_day - minute_in_day)
                    new_event = Event(minute_in_day, duration, None).normalized()
                    if self.settings and self.settings.affect_every_day:
                        self._append_event_unique(dev.events, new_event)
                        for events in dev.day_events.values():
                            self._append_event_unique(events, new_event)
                    else:
                        events, _ = self._ensure_day_events(dev, day_offset)
                        self._append_event_unique(events, new_event)
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
        for row_index, item in enumerate(self.row_items):
            rr = self._row_rect(row_index)
            if rr.contains(pos):
                if item["type"] == "device":
                    return row_index, int(item["device_index"])
                return -1, -1
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
            for day_offset in range(self._day_count()):
                intervals = self._intervals_for_day(dev, day_offset)
                for j, iv in enumerate(intervals):
                    ivn = iv.normalized()
                    x0 = self._minute_to_x(day_offset * MINUTES_PER_DAY + ivn.start_min)
                    x1 = self._minute_to_x(day_offset * MINUTES_PER_DAY + ivn.end_min)
                    rect = QRect(x0, rr.top() + 8, max(2, x1 - x0), rr.height() - 16)
                    if rect.contains(pos):
                        # edges?
                        if near(pos.x(), rect.left()):
                            return HitTest(
                                HitKind.INTERVAL_LEFT,
                                dev_index,
                                j,
                                day_offset,
                                self._x_to_minute(pos.x()),
                            )
                        if near(pos.x(), rect.right()):
                            return HitTest(
                                HitKind.INTERVAL_RIGHT,
                                dev_index,
                                j,
                                day_offset,
                                self._x_to_minute(pos.x()),
                            )
                        return HitTest(
                            HitKind.INTERVAL_BODY,
                            dev_index,
                            j,
                            day_offset,
                            self._x_to_minute(pos.x()),
                        )

        if dev.dtype == DeviceType.EVENTS:
            for day_offset in range(self._day_count()):
                events = self._events_for_day(dev, day_offset)
                for j, ev in enumerate(events):
                    evn = ev.normalized()
                    x0 = self._minute_to_x(day_offset * MINUTES_PER_DAY + evn.start_min)
                    x1 = self._minute_to_x(day_offset * MINUTES_PER_DAY + evn.start_min + evn.duration_min)
                    rect = QRect(x0, rr.top() + 10, max(3, x1 - x0), rr.height() - 20)
                    if rect.contains(pos):
                        return HitTest(
                            HitKind.EVENT_BODY,
                            dev_index,
                            j,
                            day_offset,
                            self._x_to_minute(pos.x()),
                        )

        if dev.dtype == DeviceType.ALWAYS:
            # optionally allow click to do nothing
            return HitTest()

        return HitTest()

    def _snapshot_item(self, hit: HitTest):
        dev = self.project.devices[hit.device_index]
        if hit.kind.startswith("interval"):
            intervals = self._intervals_for_day(dev, hit.day_offset)
            iv = intervals[hit.item_index].normalized()
            return ("interval", hit.day_offset, iv.start_min, iv.end_min)
        if hit.kind == HitKind.EVENT_BODY:
            events = self._events_for_day(dev, hit.day_offset)
            ev = events[hit.item_index].normalized()
            return ("event", hit.day_offset, ev.start_min, ev.duration_min, ev.energy_wh)
        return None

    def _handle_drag(self, pos: QPoint):
        if not self.project or not self.drag_original:
            return

        hit = self.hit
        dev = self.project.devices[hit.device_index]
        m_now = self._x_to_minute(pos.x())
        delta = m_now - self.drag_start_min

        if self.drag_original[0] == "interval":
            _, day_offset, s0, e0 = self.drag_original
            intervals = self._intervals_for_day(dev, day_offset)
            if hit.kind == HitKind.INTERVAL_BODY:
                s = clamp(s0 + delta, 0, MINUTES_PER_DAY - 1)
                length = e0 - s0
                e = clamp(s + length, 1, MINUTES_PER_DAY)
                # if clamped at end, pull start back
                if e - s < length:
                    s = max(0, e - length)
                intervals[hit.item_index] = Interval(s, e).normalized()

            elif hit.kind == HitKind.INTERVAL_LEFT:
                s = clamp(s0 + delta, 0, e0 - 1)
                intervals[hit.item_index] = Interval(s, e0).normalized()

            elif hit.kind == HitKind.INTERVAL_RIGHT:
                e = clamp(e0 + delta, s0 + 1, MINUTES_PER_DAY)
                intervals[hit.item_index] = Interval(s0, e).normalized()

        elif self.drag_original[0] == "event":
            _, day_offset, s0, dur, ewh = self.drag_original
            events = self._events_for_day(dev, day_offset)
            if hit.kind == HitKind.EVENT_BODY:
                s = clamp(s0 + delta, 0, MINUTES_PER_DAY - 1)
                # enforce split at midnight by clamping duration
                if s + dur > MINUTES_PER_DAY:
                    s = MINUTES_PER_DAY - dur
                events[hit.item_index] = Event(s, dur, ewh).normalized()

        self._refresh_viewport()


class TimelineTotalsPanel(QWidget):
    def __init__(self, timeline: TimelineWidget, parent=None):
        super().__init__(parent)
        self.timeline = timeline
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setMinimumWidth(0)

    def _info_panel_visible(self) -> bool:
        return bool(self.timeline.settings and self.timeline.settings.show_timeline_totals)

    def sync_from_timeline(self):
        if not self._info_panel_visible():
            self.setVisible(False)
            self.setFixedWidth(0)
            return
        self.setVisible(True)
        self._update_panel_width()
        self.updateGeometry()
        self.update()

    def _update_panel_width(self):
        if not (self.timeline.project and self.timeline.settings and self.timeline.sim):
            self.setFixedWidth(0)
            return
        metrics = QFontMetrics(self.timeline.info_font)
        max_kwh = 0.0
        max_cost = 0.0
        for idx in self.timeline.visible_device_indices:
            max_kwh = max(max_kwh, self.timeline.sim.per_device_kwh_day[idx])
            max_cost = max(max_cost, self.timeline.sim.per_device_cost_day[idx])
        kwh_text = f"▲ {max_kwh:.2f} kWh"
        cost_text = f"▼ {self.timeline.settings.currency_symbol}{max_cost:.2f}"
        text_width = max(
            metrics.horizontalAdvance(kwh_text),
            metrics.horizontalAdvance(cost_text),
        )
        if self.timeline.settings.show_total_power:
            peak_value = max(self.timeline.sim.minute_total_w) if self.timeline.sim.minute_total_w else 0.0
            peak_text = f"Peak: {peak_value:.0f} W"
            text_width = max(text_width, metrics.horizontalAdvance(peak_text))
            if self.timeline.settings.show_amps:
                amps = peak_value / max(self.timeline.settings.mains_voltage, 0.1)
                amp_text = f"Peak: {amps:.1f} A"
                text_width = max(text_width, metrics.horizontalAdvance(amp_text))
        diameter = max(6, self.timeline.row_h - 16)
        info_inner_width = 4 + diameter + 10 + text_width + 4
        self.setFixedWidth(max(0, info_inner_width + 12))

    def _panel_rect(self) -> QRect:
        content_top = self.timeline._axis_top() + self.timeline.axis_h
        return QRect(0, content_top, self.width(), self.height() - content_top - 12)

    def _row_rect(self, idx: int) -> QRect:
        panel_rect = self._panel_rect()
        y0 = panel_rect.top() + idx * (self.timeline.row_h + self.timeline.row_gap)
        return QRect(panel_rect.left(), y0, panel_rect.width(), self.timeline.row_h)

    def _total_row_rect(self) -> QRect:
        panel_rect = self._panel_rect()
        y0 = panel_rect.top() + len(self.timeline.row_items) * (
            self.timeline.row_h + self.timeline.row_gap
        )
        return QRect(panel_rect.left(), y0, panel_rect.width(), self.timeline._total_row_height())

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), QColor(25, 25, 28))

        if not self._info_panel_visible():
            return
        if not (self.timeline.project and self.timeline.settings and self.timeline.sim):
            return

        panel_rect = self._panel_rect()
        if panel_rect.width() <= 0 or panel_rect.height() <= 0:
            return

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(26, 26, 30))
        p.drawRect(panel_rect)
        p.setPen(QPen(QColor(45, 45, 52), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(panel_rect)

        for row_index, item in enumerate(self.timeline.row_items):
            rr = self._row_rect(row_index)
            if rr.bottom() < panel_rect.top() or rr.top() > panel_rect.bottom():
                continue
            if item["type"] != "device":
                continue
            dev_index = int(item["device_index"])
            info = QRect(rr.left() + 6, rr.top(), rr.width() - 12, rr.height())
            on_min = self.timeline.sim.per_device_on_minutes[dev_index]
            kwh = self.timeline.sim.per_device_kwh_day[dev_index]
            cost = self.timeline.sim.per_device_cost_day[dev_index]
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
            p.setFont(self.timeline.info_font)
            metrics = QFontMetrics(self.timeline.info_font)
            line_h = metrics.height()
            total_h = line_h * 2
            start_y = text_rect.top() + (text_rect.height() - total_h) // 2
            kwh_text = f"{kwh:.2f} kWh"
            cost_text = f"{self.timeline.settings.currency_symbol}{cost:.2f}"
            p.drawText(
                QRect(text_rect.left(), start_y, text_rect.width(), line_h),
                Qt.AlignLeft | Qt.AlignVCenter,
                f"▲ {kwh_text}",
            )
            p.drawText(
                QRect(text_rect.left(), start_y + line_h, text_rect.width(), line_h),
                Qt.AlignLeft | Qt.AlignVCenter,
                f"▼ {cost_text}",
            )

        if not self.timeline.settings.show_total_power:
            return

        rr = self._total_row_rect()
        if rr.height() <= 0:
            return
        info = QRect(rr.left() + 6, rr.top(), rr.width() - 12, rr.height())
        peak = max(self.timeline.sim.minute_total_w) if self.timeline.sim.minute_total_w else 0.0
        p.setPen(QColor(220, 220, 230))
        p.setFont(self.timeline.info_font)
        metrics = QFontMetrics(self.timeline.info_font)
        line_h = metrics.height()
        text_rect = QRect(info.left(), info.top(), info.width(), info.height())
        if self.timeline.settings.show_amps:
            amps = peak / max(self.timeline.settings.mains_voltage, 0.1)
            total_h = line_h * 2
            start_y = text_rect.top() + (text_rect.height() - total_h) // 2
            peak_w_text = f"Peak: {peak:.0f} W"
            peak_a_text = f"Peak: {amps:.1f} A"
            p.drawText(
                QRect(text_rect.left(), start_y, text_rect.width(), line_h),
                Qt.AlignLeft | Qt.AlignVCenter,
                peak_w_text,
            )
            p.drawText(
                QRect(text_rect.left(), start_y + line_h, text_rect.width(), line_h),
                Qt.AlignLeft | Qt.AlignVCenter,
                peak_a_text,
            )
        else:
            peak_text = f"Peak: {peak:.0f} W"
            p.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, peak_text)


# =========================
# Main window UI
# =========================

class DateDisplayButton(QPushButton):
    clickedSingle = Signal()
    doubleClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._suppress_click = False

    def mouseDoubleClickEvent(self, event):
        self._suppress_click = True
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and not self._suppress_click:
            self.clickedSingle.emit()
        self._suppress_click = False
        super().mouseReleaseEvent(event)


class StartDateDialog(QDialog):
    def __init__(self, parent=None, current_date: Optional[datetime.date] = None):
        super().__init__(parent)
        self.setWindowTitle("Set start date")
        self.setModal(True)

        self.current_date = current_date or datetime.date.today()
        self._manual_date = self.current_date
        self.use_today_checkbox = QCheckBox("Use today")
        self.use_today_checkbox.setChecked(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.use_today_checkbox)

        controls = QWidget(self)
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        self.prev_month_btn = QToolButton()
        self.prev_month_btn.setText("⏮")
        self.prev_month_btn.setToolTip("Jump to first day of previous month")
        self.prev_month_btn.clicked.connect(self._jump_prev_month)

        self.prev_day_btn = QToolButton()
        self.prev_day_btn.setText("◀")
        self.prev_day_btn.setToolTip("Go back one day")
        self.prev_day_btn.clicked.connect(lambda: self._shift_day(-1))

        self.date_button = DateDisplayButton()
        self.date_button.setToolTip("Click to jump to today. Double-click to pick a date.")
        self.date_button.clickedSingle.connect(self._jump_today)
        self.date_button.doubleClicked.connect(self._pick_date)
        self.date_button.setMinimumWidth(160)
        self.date_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

        self.next_day_btn = QToolButton()
        self.next_day_btn.setText("▶")
        self.next_day_btn.setToolTip("Go forward one day")
        self.next_day_btn.clicked.connect(lambda: self._shift_day(1))

        self.next_month_btn = QToolButton()
        self.next_month_btn.setText("⏭")
        self.next_month_btn.setToolTip("Jump to first day of next month")
        self.next_month_btn.clicked.connect(self._jump_next_month)

        controls_layout.addWidget(self.prev_month_btn)
        controls_layout.addWidget(self.prev_day_btn)
        controls_layout.addWidget(self.date_button)
        controls_layout.addWidget(self.next_day_btn)
        controls_layout.addWidget(self.next_month_btn)
        layout.addWidget(controls)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.use_today_checkbox.toggled.connect(self._update_controls)
        self._update_date_display()
        self._update_controls(self.use_today_checkbox.isChecked())

    def _update_controls(self, use_today: bool):
        if use_today:
            self._manual_date = self.current_date
            self.current_date = datetime.date.today()
            self._update_date_display()
        else:
            self.current_date = self._manual_date
            self._update_date_display()
        for widget in (
            self.prev_month_btn,
            self.prev_day_btn,
            self.date_button,
            self.next_day_btn,
            self.next_month_btn,
        ):
            widget.setEnabled(not use_today)

    def _update_date_display(self):
        self.date_button.setText(self.current_date.strftime("%a %d %b %Y"))

    def _set_current_date(self, day: datetime.date):
        self.current_date = day
        if not self.use_today_checkbox.isChecked():
            self._manual_date = day
        self._update_date_display()

    def _shift_day(self, delta: int):
        self._set_current_date(self.current_date + datetime.timedelta(days=delta))

    def _jump_prev_month(self):
        first_of_current = self.current_date.replace(day=1)
        prev_month_last = first_of_current - datetime.timedelta(days=1)
        self._set_current_date(prev_month_last.replace(day=1))

    def _jump_next_month(self):
        year = self.current_date.year + (1 if self.current_date.month == 12 else 0)
        month = 1 if self.current_date.month == 12 else self.current_date.month + 1
        self._set_current_date(datetime.date(year, month, 1))

    def _jump_today(self):
        self._set_current_date(datetime.date.today())

    def _pick_date(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Select date")
        layout = QVBoxLayout(dlg)
        calendar = QCalendarWidget()
        calendar.setSelectedDate(
            QDate(self.current_date.year, self.current_date.month, self.current_date.day)
        )
        layout.addWidget(calendar)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() == QDialog.Accepted:
            qdate = calendar.selectedDate()
            self._set_current_date(datetime.date(qdate.year(), qdate.month(), qdate.day()))

    def get_result(self) -> Tuple[bool, datetime.date]:
        return self.use_today_checkbox.isChecked(), self.current_date


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Power & Cost Timeline Simulator")

        self.qs = QSettings("JumbleSaleOfStimuli", "PowerCostTimeline")
        self.settings_model = SettingsModel.from_qsettings(self.qs)
        self.tariff_documents: List[TariffDocument] = []
        self.current_tariff_doc: Optional[TariffDocument] = None
        self.active_tariff_entry: Optional[TariffEntry] = None
        self.current_date = datetime.date.today()
        self.zoom_levels = [
            6 * 60,
            12 * 60,
            24 * 60,
            2 * 24 * 60,
            5 * 24 * 60,
            7 * 24 * 60,
            14 * 24 * 60,
            30 * 24 * 60,
        ]
        self.zoom_index = 2

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

        timeline_row = QWidget()
        timeline_layout = QHBoxLayout(timeline_row)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(12)

        self.timeline = TimelineWidget(parent=self)
        self.timeline_totals = TimelineTotalsPanel(self.timeline, parent=self)
        self.timeline.set_info_panel(self.timeline_totals)

        timeline_layout.addWidget(self.timeline, stretch=1)
        timeline_layout.addWidget(self.timeline_totals, stretch=0)
        root_layout.addWidget(timeline_row, stretch=1)
        self._build_timeline_controls()

        self.summary = QLabel("")
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary.setStyleSheet(
            "QLabel { padding: 8px; background: #1e1e22; color: #e8e8ee; font-size: 14px; }"
        )
        root_layout.addWidget(self.summary, stretch=0)

        # menu
        self._build_menu()

        # first run -> show settings
        if not SettingsModel.has_run_before(self.qs):
            self.open_settings(first_run=True)

        self.refresh_tables()
        self.recompute()
        QTimer.singleShot(0, self._apply_zoom)

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
        act_import_devices = QAction("Import…", self)
        act_export_devices = QAction("Export…", self)
        devm.addAction(act_device_list)
        devm.addSeparator()
        devm.addAction(act_import_devices)
        devm.addAction(act_export_devices)
        act_device_list.triggered.connect(self.open_device_list)
        act_import_devices.triggered.connect(self.import_devices)
        act_export_devices.triggered.connect(self.export_devices)

        timingm = mb.addMenu("&Timing")
        act_custom_periods = QAction("Custom Periods…", self)
        act_set_start_date = QAction("Set Start Date…", self)
        timingm.addAction(act_custom_periods)
        timingm.addAction(act_set_start_date)
        act_custom_periods.triggered.connect(self.open_custom_periods)
        act_set_start_date.triggered.connect(self.open_start_date)

        tariffm = mb.addMenu("&Tariff")
        act_import_tariff = QAction("Import…", self)
        act_select_tariff = QAction("Select Tariff…", self)
        tariffm.addAction(act_import_tariff)
        tariffm.addAction(act_select_tariff)
        act_import_tariff.triggered.connect(self.import_tariffs)
        act_select_tariff.triggered.connect(self.select_tariff)

        viewm = mb.addMenu("&View")
        act_show_total_power = QAction("Show Total Power", self)
        act_show_total_power.setCheckable(True)
        act_show_total_power.setChecked(self.settings_model.show_total_power)
        viewm.addAction(act_show_total_power)

        act_show_amps = QAction("Show Amps", self)
        act_show_amps.setCheckable(True)
        act_show_amps.setChecked(self.settings_model.show_amps)
        viewm.addAction(act_show_amps)

        act_timeline_totals = QAction("Timeline Totals", self)
        act_timeline_totals.setCheckable(True)
        act_timeline_totals.setChecked(self.settings_model.show_timeline_totals)
        viewm.addAction(act_timeline_totals)

        act_day_night = QAction("Day - Night", self)
        act_day_night.setCheckable(True)
        act_day_night.setChecked(self.settings_model.show_day_night)
        viewm.addAction(act_day_night)

        viewm.addSeparator()
        act_sort_by_category = QAction("Sort by Category", self)
        act_sort_by_category.setCheckable(True)
        act_sort_by_category.setChecked(self.settings_model.sort_by_category)
        viewm.addAction(act_sort_by_category)

        viewm.addSeparator()
        act_show_standing = QAction("Show Standing Charge", self)
        act_show_standing.setCheckable(True)
        act_show_standing.setChecked(self.settings_model.show_standing_charge)
        viewm.addAction(act_show_standing)
        act_show_total_power.toggled.connect(self._toggle_total_power)
        act_show_amps.toggled.connect(self._toggle_show_amps)
        act_timeline_totals.toggled.connect(self._toggle_timeline_totals)
        act_day_night.toggled.connect(self._toggle_day_night)
        act_sort_by_category.toggled.connect(self._toggle_sort_by_category)
        act_show_standing.toggled.connect(self._toggle_standing_charge)

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
        dlg = SettingsDialog(self, self.settings_model, self.project.custom_periods)
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

    def open_start_date(self):
        dlg = StartDateDialog(self, self.current_date)
        if dlg.exec() == QDialog.Accepted:
            use_today, selected_date = dlg.get_result()
            if use_today:
                self._set_current_date(datetime.date.today())
            else:
                self._set_current_date(selected_date)

    def open_device_list(self):
        dlg = DeviceListDialog(self, self.project)
        self.device_dialog = dlg
        try:
            dlg.exec()
        finally:
            self.device_dialog = None
            self.refresh_tables()
            self.recompute()

    def import_devices(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import devices", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as ex:
            QMessageBox.critical(self, "Import failed", f"Could not import:\n{ex}")
            return
        if not isinstance(data, dict):
            QMessageBox.critical(self, "Import failed", "Device list must be a JSON object of categories.")
            return
        categories = list(data.keys())
        dlg = DeviceImportDialog(self, categories)
        if dlg.exec() != QDialog.Accepted:
            return
        selected_categories, clear_current, show_device_list = dlg.get_result()
        imported_devices: List[Device] = []
        for category in selected_categories:
            items = data.get(category, [])
            if not isinstance(items, list):
                continue
            for item in items:
                dev = device_from_catalog_item(item, category)
                if dev:
                    imported_devices.append(dev)
        if clear_current:
            self.project.devices = []
        self.project.devices.extend(imported_devices)
        self.refresh_tables()
        self.recompute()
        if show_device_list:
            self.open_device_list()

    def export_devices(self):
        name, ok = QInputDialog.getText(self, "Export devices", "File name")
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        default_name = name if name.lower().endswith(".json") else f"{name}.json"
        path, _ = QFileDialog.getSaveFileName(self, "Export devices", default_name, "JSON (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        payload: Dict[str, List[dict]] = {}
        for dev in self.project.devices:
            category = dev.category.strip() or "Custom"
            payload.setdefault(category, []).append({
                "name": dev.name,
                "power_w": dev.power_w,
                "usage_duration": format_duration_export(dev.default_duration_min),
                "variable_time": dev.variable,
            })
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as ex:
            QMessageBox.critical(self, "Export failed", f"Could not export:\n{ex}")

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
        sim = simulate_day(self.project, self.settings_model, self.current_date)
        self.timeline.set_data(self.project, self.settings_model, sim, day=self.current_date)
        self.timeline_totals.sync_from_timeline()
        self._apply_zoom()
        self._update_summary(sim)

    def _build_timeline_controls(self):
        self.timeline_controls = QWidget(self.timeline.viewport())
        self.timeline_controls.setObjectName("timelineControls")
        self.timeline_controls.setAttribute(Qt.WA_StyledBackground, True)
        self.timeline_controls.setStyleSheet(
            "#timelineControls { background: rgba(20, 20, 24, 210); border: 1px solid #2d2d33; "
            "border-radius: 6px; }"
        )

        layout = QHBoxLayout(self.timeline_controls)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        self.affect_every_day_checkbox = QCheckBox("Affect Every Day")
        self.affect_every_day_checkbox.setChecked(self.settings_model.affect_every_day)
        self.affect_every_day_checkbox.setStyleSheet("QCheckBox { color: #d6d6de; }")
        self.affect_every_day_checkbox.toggled.connect(self._toggle_affect_every_day)

        self.zoom_out_btn = QToolButton()
        self.zoom_out_btn.setText("−")
        self.zoom_out_btn.setToolTip("Zoom out")
        self.zoom_out_btn.clicked.connect(lambda: self._change_zoom(-1))

        self.zoom_in_btn = QToolButton()
        self.zoom_in_btn.setText("+")
        self.zoom_in_btn.setToolTip("Zoom in")
        self.zoom_in_btn.clicked.connect(lambda: self._change_zoom(1))

        self.zoom_label = QLabel("")
        self.zoom_label.setStyleSheet("QLabel { color: #d6d6de; }")

        layout.addWidget(self.affect_every_day_checkbox)
        layout.addSpacing(8)
        layout.addSpacing(12)
        layout.addWidget(self.zoom_label)
        layout.addWidget(self.zoom_out_btn)
        layout.addWidget(self.zoom_in_btn)

        self._update_zoom_display()
        self.timeline_controls.adjustSize()
        self.timeline_controls.raise_()
        self.timeline.viewport().installEventFilter(self)
        self._position_timeline_controls()

    def eventFilter(self, obj, event):
        if obj is self.timeline.viewport() and event.type() == QEvent.Resize:
            self._apply_zoom()
            self._position_timeline_controls()
        return super().eventFilter(obj, event)

    def _position_timeline_controls(self):
        if not hasattr(self, "timeline_controls"):
            return
        self.timeline_controls.adjustSize()
        vp = self.timeline.viewport().rect()
        margin = 8
        x = max(margin, vp.right() - self.timeline_controls.width() - margin)
        y = max(margin, vp.bottom() - self.timeline_controls.height() - margin)
        self.timeline_controls.move(x, y)

    def _format_date_display(self, day: datetime.date) -> str:
        return day.strftime("%a %d %b %Y")

    def _update_date_display(self):
        if hasattr(self, "date_button"):
            self.date_button.setText(self._format_date_display(self.current_date))

    def _set_current_date(self, day: datetime.date):
        self.current_date = day
        self._update_date_display()
        self.recompute()

    def _shift_day(self, delta: int):
        self._set_current_date(self.current_date + datetime.timedelta(days=delta))

    def _jump_prev_month(self):
        first_of_current = self.current_date.replace(day=1)
        prev_month_last = first_of_current - datetime.timedelta(days=1)
        self._set_current_date(prev_month_last.replace(day=1))

    def _jump_next_month(self):
        year = self.current_date.year + (1 if self.current_date.month == 12 else 0)
        month = 1 if self.current_date.month == 12 else self.current_date.month + 1
        self._set_current_date(datetime.date(year, month, 1))

    def _jump_today(self):
        self._set_current_date(datetime.date.today())

    def _pick_date(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Select date")
        layout = QVBoxLayout(dlg)
        calendar = QCalendarWidget()
        calendar.setSelectedDate(
            QDate(self.current_date.year, self.current_date.month, self.current_date.day)
        )
        layout.addWidget(calendar)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() == QDialog.Accepted:
            qdate = calendar.selectedDate()
            self._set_current_date(datetime.date(qdate.year(), qdate.month(), qdate.day()))

    def _change_zoom(self, delta: int):
        self.zoom_index = clamp(self.zoom_index + delta, 0, len(self.zoom_levels) - 1)
        self._apply_zoom()

    def _simulation_length_minutes(self) -> int:
        options = simulation_length_options(self.settings_model, self.project.custom_periods)
        for key, _label, minutes in options:
            if key == self.settings_model.simulation_length_key:
                return minutes
        for key, _label, minutes in options:
            if key == "1_week":
                self.settings_model.simulation_length_key = key
                self.settings_model.to_qsettings(self.qs)
                return minutes
        return MINUTES_PER_DAY

    def _apply_zoom(self):
        if not hasattr(self, "zoom_levels"):
            return
        zoom_minutes = self.zoom_levels[self.zoom_index]
        timeline_draw_width = max(1, self.timeline.timeline_draw_width())
        range_minutes = max(1, self._simulation_length_minutes())
        visible_minutes = min(zoom_minutes, range_minutes)
        pixels_per_minute = timeline_draw_width / max(1, visible_minutes)
        self.timeline.set_time_range(0, range_minutes)
        self.timeline.set_pixels_per_minute(pixels_per_minute)
        self._update_zoom_display()

    def _format_zoom_label(self, minutes: int) -> str:
        week_minutes = 7 * 24 * 60
        day_minutes = 24 * 60
        if minutes % week_minutes == 0:
            weeks = minutes // week_minutes
            return f"{weeks} week" if weeks == 1 else f"{weeks} weeks"
        if minutes % day_minutes == 0:
            days = minutes // day_minutes
            return f"{days} day" if days == 1 else f"{days} days"
        if minutes % 60 == 0:
            hours = minutes // 60
            return f"{hours} hour" if hours == 1 else f"{hours} hours"
        return f"{minutes} mins"

    def _update_zoom_display(self):
        if not hasattr(self, "zoom_label"):
            return
        zoom_minutes = self.zoom_levels[self.zoom_index]
        self.zoom_label.setText(self._format_zoom_label(zoom_minutes))
        self.zoom_out_btn.setEnabled(self.zoom_index > 0)
        self.zoom_in_btn.setEnabled(self.zoom_index < len(self.zoom_levels) - 1)

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

        show_standing = self.settings_model.show_standing_charge
        standing_charge = None
        if show_standing and self.active_tariff_entry is not None:
            tariff = self.active_tariff_entry.tariff()
            standing_charge = tariff.get("standing_charge_gbp_per_day")
            if standing_charge is not None:
                standing_charge = float(standing_charge)
        has_standing = standing_charge is not None and standing_charge > 0

        data_rows = []

        def add_data_row(label: str, kwh: str, cost: str, days: float) -> None:
            data_rows.append((label, kwh, cost, days))

        add_data_row("Day", f"{day_kwh:.3f} kWh", f"{cs}{day_cost:.2f}", 1.0)
        add_data_row("Week", f"{week_kwh:.3f} kWh", f"{cs}{week_cost:.2f}", 7.0)
        add_data_row("Month", f"{month_kwh:.3f} kWh", f"{cs}{month_cost:.2f}", 30.0)
        add_data_row("Year", f"{year_kwh:.3f} kWh", f"{cs}{year_cost:.2f}", 365.0)

        enabled_periods = [cp for cp in self.project.custom_periods if cp.enabled]
        for period in enabled_periods:
            try:
                d = custom_period_to_days(period, self.settings_model)
            except Exception:
                d = 1.0
            add_data_row(
                f"{period.name} ({period.duration:g} {period.unit})",
                f"{day_kwh*d:.3f} kWh",
                f"{cs}{day_cost*d:.2f}",
                d,
            )

        rows = []
        if show_standing:
            rows.append(
                "<tr>"
                "<th style='text-align: left; padding-right: 18px; font-weight: normal;'>Period</th>"
                "<th style='text-align: right; padding-right: 18px; font-weight: normal;'>Usage</th>"
                "<th style='text-align: right; padding-right: 18px; font-weight: normal;'>Cost</th>"
                "<th style='text-align: right; padding-right: 18px; font-weight: normal;'>Standing Charge</th>"
                "<th style='text-align: right; font-weight: normal;'>Total</th>"
                "</tr>"
            )
        else:
            rows.append(
                "<tr>"
                "<th style='text-align: left; padding-right: 18px; font-weight: normal;'>Period</th>"
                "<th style='text-align: right; padding-right: 18px; font-weight: normal;'>Usage</th>"
                "<th style='text-align: right; font-weight: normal;'>Cost</th>"
                "</tr>"
            )

        no_standing_rowspan = len(data_rows) if data_rows else 1
        for row_index, (label, kwh, cost, days) in enumerate(data_rows):
            row_html = (
                "<tr>"
                f"<td style='padding-right: 18px; white-space: nowrap;'>{escape(label)}</td>"
                f"<td style='padding-right: 18px; text-align: right; white-space: nowrap;'>{escape(kwh)}</td>"
                f"<td style='text-align: right; white-space: nowrap; padding-right: 18px;'>{escape(cost)}</td>"
            )
            if show_standing:
                if has_standing:
                    standing_cost = standing_charge * days
                    total_cost = (day_cost * days) + standing_cost
                    row_html += (
                        f"<td style='text-align: right; white-space: nowrap; padding-right: 18px;'>"
                        f"{cs}{standing_cost:.2f}</td>"
                        f"<td style='text-align: right; white-space: nowrap;'>{cs}{total_cost:.2f}</td>"
                    )
                elif row_index == 0:
                    row_html += (
                        f"<td colspan='2' rowspan='{no_standing_rowspan}' "
                        "style='text-align: center; vertical-align: middle; color: #b6b6bf;'>"
                        "No Standing Charge</td>"
                    )
            row_html += "</tr>"
            rows.append(row_html)

        rows.append(
            f"<tr><td colspan='{5 if show_standing else 3}' style='height: 8px;'></td></tr>"
        )

        if not enabled_periods:
            rows.append(
                "<tr>"
                f"<td colspan='{5 if show_standing else 3}' "
                "style='color: #b6b6bf; padding-left: 2px;'>"
                "No custom periods enabled"
                "</td>"
                "</tr>"
            )

        html = (
            "<table style='width: 100%; border-collapse: collapse;'>"
            + "".join(rows)
            + "</table>"
        )
        self.summary.setText(html)

    def _toggle_standing_charge(self, enabled: bool):
        self.settings_model.show_standing_charge = enabled
        self.settings_model.to_qsettings(self.qs)
        self.recompute()

    def _toggle_show_amps(self, enabled: bool):
        self.settings_model.show_amps = enabled
        self.settings_model.to_qsettings(self.qs)
        self.recompute()

    def _toggle_total_power(self, enabled: bool):
        self.settings_model.show_total_power = enabled
        self.settings_model.to_qsettings(self.qs)
        self.recompute()

    def _toggle_timeline_totals(self, enabled: bool):
        self.settings_model.show_timeline_totals = enabled
        self.settings_model.to_qsettings(self.qs)
        self.timeline_totals.sync_from_timeline()
        self.recompute()

    def _toggle_day_night(self, enabled: bool):
        self.settings_model.show_day_night = enabled
        self.settings_model.to_qsettings(self.qs)
        self.timeline.updateGeometry()
        self.timeline._refresh_viewport()

    def _toggle_affect_every_day(self, enabled: bool):
        self.settings_model.affect_every_day = enabled
        self.settings_model.to_qsettings(self.qs)
        self.timeline._refresh_viewport()

    def _toggle_sort_by_category(self, enabled: bool):
        self.settings_model.sort_by_category = enabled
        self.settings_model.to_qsettings(self.qs)
        self.recompute()


# =========================
