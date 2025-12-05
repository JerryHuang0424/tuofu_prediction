# file_process.py

用途：将包含若干时间列（如 `time1, time2, ...`）的 xlsx 数据转换为按 `title` 聚合的时间特征表。

默认行为：
- 默认输入文件：当前目录下的 `reading.xlsx`（若未在命令行指定）
- 默认输出文件：`pred_reading.csv`
- 默认标题列名：`pred_reading`

主要输出字段（部分）：
- `title`: 原始标题列
- `latest_time`: 最晚出现时间
- `days_diff_from_ref_date`: 最晚出现时间与参考日之间的差值（以天为单位，参考日由 `--ref-date` 指定或使用脚本顶部的默认值）
- `ref_date`: 使用的参考日期（Timestamp）
- `appearance_count`: 出现次数（非空时间个数）
- `time_variance_days2`: 所有出现时间的方差（以天为单位）
- `earliest_time`, `mean_time`, `median_time`, `timespan_days`, `missing_time_count`
- `diff1_days`, `diff2_days`, `diff3_days`: 与参考日的差值（以天为单位），仅保留**最近的三次**出现（`diff1_days` 为最近一次）。出现次数不足时对应项为空/NaN。

使用示例（PowerShell）:

```powershell
# 使用默认文件名 reading.xlsx -> pred_reading.csv，标题列 pred_reading
python .\file_process.py

# 显式指定输入/输出/参考日/标题列
python .\file_process.py .\reading.xlsx -o .\pred_reading.csv --ref-date 2025-11-01 --title-col pred_reading
```

安装依赖：

```powershell
pip install -r requirements.txt
```

说明与拓展：
- 脚本会自动检测列名中包含 `time` 或 `date` 的列为时间列；也会识别 datetime dtype 的列。
- 为避免把原始表中像 `第6次出现`、`第7次出现` 这样的列原样带入输出，脚本会自动过滤掉匹配 `第\d+次` 的列名。
- 如果你希望改为保留那些列（或保留某些以 `time6` 命名的列），告诉我具体列名或命名规则，我可以调整 `detect_time_columns` 或过滤规则。
- 方差以“天”为单位计算（float），如需标准差、秒为单位或其它变体，请告知。
