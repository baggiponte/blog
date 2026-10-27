// "may 23, 2024": the one date format of the site. Dates are calendar days, so read them in UTC,
// or a build in a timezone west of Greenwich would show the day before.
export const formatDate = (date: Date): string =>
  date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).toLowerCase();
