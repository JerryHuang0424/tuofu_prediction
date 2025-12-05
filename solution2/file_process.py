from pathlib import Path
import pandas as pd
import numpy as np
import re

# Default reference date for difference calculations. Change this in one place if needed.
REF_DATE_DEFAULT = '2025-11-08'


def detect_time_columns(df: pd.DataFrame):
    # Prefer columns whose name contains 'time' or 'date' (case-insensitive)
    candidates = [c for c in df.columns if ('time' in c.lower()) or ('date' in c.lower())]
    # Also include any columns that are datetime dtype
    dtype_dt = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cols = list(dict.fromkeys(candidates + dtype_dt))
    return cols


def process_df(df: pd.DataFrame, title_col='title', ref_date=REF_DATE_DEFAULT):
    # Ensure potential time columns parsed as datetimes
    time_cols = detect_time_columns(df)
    # normalize reference date to Timestamp
    ref_ts = pd.to_datetime(ref_date)
    for c in time_cols:
        df[c] = pd.to_datetime(df[c], errors='coerce')

    # We'll keep non-time columns (features) and title
    keep_cols = [c for c in df.columns if c not in time_cols]
    # Remove columns that represent later-occurrence time fields like '第6次出现', '第7次出现' etc.
    # (matches patterns like '第6次', '第 6 次出现' etc.)
    occ_pattern = re.compile(r'第\s*\d+\s*次', flags=re.IGNORECASE)
    keep_cols = [c for c in keep_cols if not occ_pattern.search(str(c))]

    out_rows = []
    for _, row in df.iterrows():
        title = row.get(title_col)
        # collect non-null times
        times = [row[c] for c in time_cols if pd.notnull(row[c])]
        times = [pd.to_datetime(t) for t in times]

        if len(times) == 0:
            latest = pd.NaT
            earliest = pd.NaT
            count = 0
            var_days = np.nan
            mean_time = pd.NaT
            median_time = pd.NaT
            timespan_days = np.nan
        else:
            times_sorted = sorted(times)
            earliest = times_sorted[0]
            latest = times_sorted[-1]
            count = len(times)
            # Convert to days (float) since epoch for variance calculation
            days = np.array([(t - pd.Timestamp('1970-01-01')) / pd.Timedelta(days=1) for t in times_sorted], dtype=float)
            var_days = float(np.var(days))
            mean_time = pd.to_datetime((days.mean() * (24 * 3600 * 1e9)).astype('int64') + pd.Timestamp('1970-01-01').value, unit='ns') if len(days) > 0 else pd.NaT
            # median
            median_val = np.median(days)
            median_time = pd.to_datetime((median_val * (24 * 3600 * 1e9)).astype('int64') + pd.Timestamp('1970-01-01').value, unit='ns')
            timespan_days = float((latest - earliest) / pd.Timedelta(days=1))

        # difference with reference date for the latest time: ref_date - latest (days).
        # Positive if ref_date later than latest.
        if pd.isna(latest):
            days_diff_latest = np.nan
        else:
            days_diff_latest = float((ref_ts - latest) / pd.Timedelta(days=1))

        # differences between ref_date and ALL occurrence times (in days)
        if len(times) == 0:
            # no occurrence
            diffs_list = []
            diff_min = np.nan
            diff_mean = np.nan
            diff_max = np.nan
            diff_std = np.nan
            # diffs_str = ''
            diff1 = np.nan
            diff2 = np.nan
            diff3 = np.nan
        else:
            # days array already computed as `days` (days since epoch)
            ref_days = float((ref_ts - pd.Timestamp('1970-01-01')) / pd.Timedelta(days=1))
            diffs = ref_days - days  # array of differences (days) for all occurrences
            diffs_list = [float(x) for x in diffs]
            diff_min = float(diffs.min())
            diff_mean = float(diffs.mean())
            diff_max = float(diffs.max())
            diff_std = float(diffs.std())
            # string representation for CSV/Excel friendliness (all diffs)
            # diffs_str = ';'.join([str(round(x, 6)) for x in diffs_list])

            # select only the most recent 3 occurrences (chronologically latest)
            # days is an array of days since epoch in ascending order corresponding to times_sorted
            last_n = 3
            recent_days = days[-last_n:][::-1]  # most recent first
            # compute diffs for recent up to 3, pad with NaN when missing
            recent_diffs = [float(ref_days - d) for d in recent_days]
            while len(recent_diffs) < last_n:
                recent_diffs.append(np.nan)
            diff1, diff2, diff3 = recent_diffs[0], recent_diffs[1], recent_diffs[2]
            # string representation for CSV/Excel friendliness: only the three recent diffs
            def fmt(x):
                return str(round(x, 6)) if (not isinstance(x, float) or not np.isnan(x)) else ''
            # diffs_str = ';'.join([fmt(diff1), fmt(diff2), fmt(diff3)])

        # prepare base output dict: include title and requested features
        out = {
            'title': title,
            'latest_time': latest,
            'days_diff_from_ref_date': days_diff_latest,
            'appearance_count': int(count),
            'time_variance_days2': var_days,
            'earliest_time': earliest,
            'mean_time': mean_time,
            'median_time': median_time,
            'timespan_days': timespan_days,
            # 'missing_time_count': len(time_cols) - int(count) if time_cols else np.nan,
            # time-series differences to reference date
            # 'diffs_to_ref_days': diffs_str,
            'diff_min_days': diff_min,
            'diff_mean_days': diff_mean,
            'diff_max_days': diff_max,
            'diff_std_days': diff_std,
            # only keep first 3 diffs (earliest occurrences). NaN padded when fewer than 3.
            # 'diff1_days': diff1,
            'diff2_days': diff2,
            'diff3_days': diff3,
        }

        # attach other non-time features (feature1, feature2, etc.)
        for c in keep_cols:
            out[c] = row.get(c)

        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)
    # Ensure datetime columns preserved as datetimes
    for c in ['latest_time', 'earliest_time', 'mean_time', 'median_time']:
        if c in out_df.columns:
            out_df[c] = pd.to_datetime(out_df[c], errors='coerce')

    return out_df


def process_path(input_path: Path, output_path: Path, ref_date: pd.Timestamp, title_col='title'):
    if input_path.is_dir():
        all_files = list(input_path.glob('*.xlsx'))
    else:
        all_files = [input_path]

    dfs = []
    for f in all_files:
        df = pd.read_excel(f, engine='openpyxl')
        dfs.append(df)

    if not dfs:
        raise SystemExit('No input files found')

    big_df = pd.concat(dfs, ignore_index=True)
    out_df = process_df(big_df, title_col=title_col, ref_date=ref_date)

    # write output depending on extension
    if output_path.suffix.lower() in ['.xls', '.xlsx']:
        out_df.to_excel(output_path, index=False)
    else:
        out_df.to_csv(output_path, index=False)


def main():
    # Edit the following variables directly in code to set file paths and options.
    # Put your input file (or folder) path here. Example: Path('./reading.xlsx')
    input_path = Path("./reading.xlsx")

    # Output file. Use .csv or .xlsx suffix to choose format.
    output_path = Path("./processed_reading.xlsx")

    # Reference date for difference calculations. You can set a custom date string
    # (e.g. '2025-11-01') or leave as the module default `REF_DATE_DEFAULT`.
    ref_date = pd.to_datetime(REF_DATE_DEFAULT)

    # Title column name in your input file. Edit if your column has a different name.
    title_col = 'processed_reading'

    # Run processing
    process_path(input_path, output_path, ref_date, title_col=title_col)


if __name__ == '__main__':
    main()
