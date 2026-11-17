//! The time the log is opened, in UTC, from the standard library alone

use std::time::{Duration, SystemTime};

/// The current time in RFC 3339, `2026-11-16T09:12:03.120Z`;
/// `None` when the clock of the system is set before 1970
pub(crate) fn utc_now() -> Option<String> {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(rfc3339)
}

pub(crate) fn rfc3339(since_epoch: Duration) -> String {
    let seconds = since_epoch.as_secs();
    let days = seconds / 86_400;
    let in_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year,
        month,
        day,
        in_day / 3600,
        in_day % 3600 / 60,
        in_day % 60,
        since_epoch.subsec_millis()
    )
}

/// The date of a number of days since 1970-01-01, in the proleptic Gregorian calendar
/// (Howard Hinnant, "chrono-Compatible Low-Level Date Algorithms")
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let day_of_era = z % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    (year, month, day)
}
