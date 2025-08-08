from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Literal

import pandas as pd
import pandas_market_calendars as pmc
import pytz


def get_now_tz(tz_name: str = "America/New_York") -> datetime:
    tz = pytz.timezone(tz_name)
    return datetime.now(tz)


def is_market_open_now(tz_name: str = "America/New_York") -> bool:
    now = get_now_tz(tz_name)
    cal = pmc.get_calendar("XNYS")
    schedule = cal.schedule(start_date=now.date(), end_date=now.date())
    if schedule.empty:
        return False
    open_time = schedule.iloc[0]["market_open"].tz_convert(tz_name)
    close_time = schedule.iloc[0]["market_close"].tz_convert(tz_name)
    return open_time <= now <= close_time


def is_rebalance_day(date: pd.Timestamp, rule: Literal["last_business_day", "weekly", "daily"] = "last_business_day") -> bool:
    date = pd.Timestamp(date).normalize()
    if rule == "daily":
        return True
    if rule == "weekly":
        # Rebalance on Friday or last business day of the week
        return date.weekday() == 4 or (date + pd.offsets.BDay(1)).month != date.month
    if rule == "last_business_day":
        next_bday = date + pd.offsets.BDay(1)
        return next_bday.month != date.month
    return False


def month_end_bdays(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Last business day of month for each date on index
    last_bdays = pd.DatetimeIndex(sorted(set((idx + pd.offsets.BMonthEnd(0)).normalize() for idx in index)))
    return last_bdays.intersection(index.normalize().unique())


