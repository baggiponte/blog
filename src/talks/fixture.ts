// The data files of a small catalog, for the catalog and views tests.
export const talk = {
  title: "polars & friends",
  description: "Dataframes, fast.",
  given: [
    { date: "2024-05-23", venue: "PyCon Italia 2024", slides: "/talks/pyconit-polars/" },
    { date: "2025-06-06", venue: "PyData London 2025", slides: "https://example.com/slides" },
  ],
};
export const venues = {
  "PyCon Italia": { logo: "pycon-italia.png" },
  "PyCon Italia 2024": { city: "Florence" },
  PyData: { logo: "pydata.png" },
  "PyData London": { city: "London" },
};
export const cities = { Florence: { at: [11.25, 43.77] }, London: { at: [-0.13, 51.51] } };
export const decks = ["pyconit-polars"];
export const files = { talks: [talk], venues, cities };
