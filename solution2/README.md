# file_process.py

用途：将包含若干时间列（如 `time1, time2, ...`）的 xlsx 数据转换为按 `title` 聚合的时间特征表。

默认行为：
- 默认输入文件：当前目录下的 `reading.xlsx`
- 默认输出文件：`processed_reading.xlsx`
- 默认标题列名：`processed_reading`
- 默认参考日：`2025-11-08`（文件顶部 `REF_DATE_DEFAULT` 常量，编辑即改全局）

输出字段详细说明：
- **title**: 输入记录的标识/标题（字符串），对应输入表中指定的标题列；用于分组或作为行索引。
- **latest_time**: 最晚出现的时间（`Timestamp`）；若无任何时间值则为 `NaT`。用于表示最近一次事件发生时间。
- **days_diff_from_ref_date**: 参考日减去 `latest_time` 的差值（以天为单位，float）；若 `latest_time` 缺失则为 `NaN`。正值表示参考日在最晚时间之后。
- **appearance_count**: 出现次数（整数），即所有时间列中非空时间的个数。
- **time_variance_days2**: 出现时间的方差（以"天²"为单位，float）；无时间值时为 `NaN`。衡量事件时间分布的离散程度。
- **earliest_time**: 最早出现的时间（`Timestamp`）；若无时间值则为 `NaT`。与 `latest_time` 配合可得到时间跨度。
- **mean_time**: 所有出现时间的算术平均时间（`Timestamp`）；无时间值时为 `NaT`。
- **median_time**: 出现时间的中位数时间（`Timestamp`）；无时间值时为 `NaT`。
- **timespan_days**: `latest_time - earliest_time` 的差值，以天为单位（float）；衡量首次与末次出现间隔。
- **missing_time_count**: 识别出的时间列总数减去 `appearance_count`；表示该行在时间列中有多少空值。
- **diff_min_days**: 参考日与所有出现时间差值的最小值（以天为单位，float）；无时间则为 `NaN`。
- **diff_mean_days**: 参考日与所有出现时间差值的平均值（以天为单位，float）；聚合差值的中心趋势。
- **diff_max_days**: 参考日与所有出现时间差值的最大值（以天为单位，float）。
- **diff_std_days**: 参考日与所有出现时间差值的标准差（以天为单位，float）；衡量差值的离散程度。
- **diff1_days**: 与参考日的差值（以天为单位，float），对应**最近第一次**出现；若次数 < 1 则为 `NaN`。
- **diff2_days**: 与参考日的差值（以天为单位，float），对应**最近第二次**出现；若次数 < 2 则为 `NaN`。
- **diff3_days**: 与参考日的差值（以天为单位，float），对应**最近第三次**出现；若次数 < 3 则为 `NaN`。
- **其它非时间列**: 输入表中的其它列（如 `feature1`, `feature2` 等）会原样复制到输出表。

使用示例（PowerShell）:

```powershell
# 直接运行脚本（使用代码内设置的默认路径与参数）
python .\file_process.py
```

如何修改默认参数：
- 打开 `file_process.py`，在 `main()` 函数中修改以下变量：
  - `input_path`: 输入文件或目录路径（例如 `Path("./reading.xlsx")`）
  - `output_path`: 输出文件路径（`.csv` 或 `.xlsx` 后缀）
  - `ref_date`: 参考日期（默认使用 `REF_DATE_DEFAULT` 常量；也可改为 `pd.to_datetime('2025-11-01')`）
  - `title_col`: 输入表中标题列的列名

安装依赖：

```powershell
pip install -r requirements.txt
```

说明与特殊处理：
- 脚本会自动检测列名中包含 `time` 或 `date` 的列为时间列；也会识别 datetime dtype 的列。
- 为避免把原始表中像 `第6次出现`、`第7次出现` 这样的列带入输出，脚本自动过滤掉匹配 `第\d+次` 的列名。
- 所有时间差值单位均为"天"（float）；若需秒或小时，可修改 `process_df()` 中的 `Timedelta` 单位。
- `diff1/2/3` 是按**时间倒序**（最近的三次）选取；若需最早的三次，可联系修改。
- 若参考日需要每次运行时动态指定，可将代码改回使用命令行参数；现在是代码内常量模式。
- 推荐用于下游建模的特征组合：`appearance_count`、`timespan_days`、`time_variance_days2`、`diff1_days`、`diff_mean_days`。

示例数据输出格式（部分字段）：
| title | latest_time | days_diff_from_ref_date | appearance_count | diff1_days | diff2_days | diff3_days |
|-------|-------------|------------------------|------------------|-----------|-----------|-----------|
| item1 | 2025-11-05  | 3.0                    | 3                | 3.0       | 10.5      | 20.0      |
| item2 | 2025-10-15  | 24.0                   | 1                | 24.0      | NaN       | NaN       |
