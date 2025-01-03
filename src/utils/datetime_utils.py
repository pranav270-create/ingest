from datetime import datetime, timezone, timedelta
import dateutil.parser
import re
from typing import Union


def get_current_utc_datetime() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_iso_string(dt) -> str:
    return dt.isoformat()


def get_iso8601_string(date_string: str) -> str:
    try:
        # Try parsing with dateutil for flexible date strings
        dt = dateutil.parser.parse(date_string)
        # If the parsed datetime doesn't have a timezone, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # Convert to UTC and format as ISO 8601 with 'Z'
        return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
    except (ValueError, TypeError):
        # If parsing fails, raise an exception
        raise ValueError(f"Unable to parse date string: {date_string}")


def parse_datetime(date_string: Union[str, datetime]) -> str:
    if not date_string:
        return None

    # Handle datetime object directly
    if isinstance(date_string, datetime):
        if date_string.tzinfo is None:
            date_string = date_string.replace(tzinfo=timezone.utc)
        return date_string.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    try:
        # Try parsing as a Unix timestamp (with potential microsecond precision)
        try:
            timestamp = float(date_string)
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass  # Not a Unix timestamp, continue with other parsing methods

        # Handle PDF date format (D:YYYYMMDDHHmmSS[+/-]HH'mm')
        pdf_date_match = re.match(r'D:(\d{14})([-+]\d{2}\'(\d{2}\')?)?(Z)?', date_string)
        if pdf_date_match:
            date_part = pdf_date_match.group(1)
            tz_part = pdf_date_match.group(2)

            dt = datetime.strptime(date_part, "%Y%m%d%H%M%S")

            if tz_part:
                if tz_part == 'Z':
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    tz_hours, tz_minutes = int(tz_part[1:3]), int(tz_part[4:6])
                    tz_offset = timezone(timedelta(hours=tz_hours, minutes=tz_minutes))
                    dt = dt.replace(tzinfo=tz_offset)
            else:
                dt = dt.replace(tzinfo=timezone.utc)
            
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Try parsing with dateutil
        dt = dateutil.parser.parse(date_string)
        
        # If the parsed datetime doesn't have a timezone, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        # If it's not in UTC, convert it to UTC 
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        # If dateutil fails, try a few common formats
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",  # RSS feed format
            "%Y-%m-%dT%H:%M:%S%z",       # ISO 8601 with timezone
            "%Y-%m-%d %H:%M:%S",         # Common format without timezone
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_string, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        
        # If all parsing attempts fail, raise an exception
        raise ValueError(f"Unable to parse date string: {date_string}")


def parse_pdf_date(date_str: str) -> str:
    """Parse PDF date format to datetime string, handling various formats."""
    if not date_str:
        return None

    # Remove 'D:' prefix if present
    date_str = date_str.strip()
    if date_str.startswith('D:'):
        date_str = date_str[2:]

    # Try different date patterns
    patterns = [
        # Standard PDF format: YYYYMMDDHHmmSS
        r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})',
        # Format with timezone: YYYYMMDDHHmmSS+HH'mm'
        r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})[-+]\d{2}\'?\d{2}\'?',
        # Simple date format: YYYYMMDD
        r'(\d{4})(\d{2})(\d{2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            groups = match.groups()
            try:
                if len(groups) == 6:  # Full datetime
                    year, month, day, hour, minute, second = map(int, groups)
                    dt = datetime(year, month, day, hour, minute, second)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                elif len(groups) == 3:  # Date only
                    year, month, day = map(int, groups)
                    dt = datetime(year, month, day)
                    return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    return None
