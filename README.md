# Power Visualizer

Power Visualizer is a desktop GUI tool for simulating daily electricity usage and cost across devices and tariff schedules.

## Requirements

See [REQUIREMENTS.md](REQUIREMENTS.md) for supported platforms and dependencies.

## Installation

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows (PowerShell):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```bash
   pip install PySide6 astral
   ```

## Run the app

From the repository root:

```bash
python -m powervis
```

You can also run the launcher script:

```bash
python powervis.py
```

## Import formats

The app supports importing device catalogs and tariff schedules from JSON files.

### Device import

Use **Devices → Import…** to load a device catalog. The file must be a JSON object where each key is a category name and the value is a list of devices.

Each device supports the following fields:

- `name` (string, required): Device label.
- `power_w` (number, required): Power draw in watts.
- `usage_duration` (string, required): Default duration. Supported formats:
  - `H:MM` (for example `1:30`)
  - decimal hours (for example `1.5`)
- `variable_time` (boolean, optional): `true` for scheduled devices, `false` for fixed-duration events.

Example (see [basic-devices.json](basic-devices.json)):

```json
{
  "Kitchen": [
    {
      "name": "Kettle",
      "power_w": 3000,
      "usage_duration": "0:03",
      "variable_time": false
    }
  ]
}
```

After importing, choose which categories to include. You can also clear the current device list before importing.

### Tariff import

Use **Tariff → Import…** to load a tariff schedule. The file must include the schema version `uk-tariffs-1.0`.

Minimum structure:

- `schema_version`: must be `uk-tariffs-1.0`.
- `suppliers`: list of suppliers, each with:
  - `supplier_name`
  - `tariffs`: list of tariffs, each with:
    - `tariff_name`
    - `rates`: list of rates. Each rate includes:
      - `rate_gbp_per_kwh` (number)
      - `priority` (number, optional)
      - `schedule` with:
        - `day_sets`: list of day sets (`mon`, `tue`, `wed`, `thu`, `fri`, `sat`, `sun`)
        - `time_ranges`: list of `start`/`end` in `HH:MM` 24-hour format

Example (see [UK-tariffs.json](UK-tariffs.json)):

```json
{
  "schema_version": "uk-tariffs-1.0",
  "suppliers": [
    {
      "supplier_name": "Example Energy",
      "tariffs": [
        {
          "tariff_name": "Example Flat",
          "rates": [
            {
              "rate_gbp_per_kwh": 0.28,
              "schedule": {
                "day_sets": [{"days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]}],
                "time_ranges": [{"start": "00:00", "end": "00:00"}]
              }
            }
          ]
        }
      ]
    }
  ]
}
```

After importing, choose **Tariff → Select Tariff…** to activate a specific supplier/tariff entry.
