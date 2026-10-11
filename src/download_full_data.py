from pathlib import Path
import requests
import pandas as pd


RESOURCE_ID = "dff4d804-5031-443a-8409-8344efd0e5c8"

METADATA_URL = "https://data.boston.gov/api/3/action/resource_show"

OUTPUT_PATH = Path("data/raw/boston_311_2024_full.csv")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Finding official 2024 dataset URL...")

    response = requests.get(
        METADATA_URL,
        params={"id": RESOURCE_ID},
        timeout=30
    )
    response.raise_for_status()

    metadata = response.json()

    if not metadata.get("success"):
        raise RuntimeError("Could not retrieve dataset metadata.")

    download_url = metadata["result"]["url"]

    print("Downloading complete 2024 dataset...")

    temp_path = OUTPUT_PATH.with_suffix(".part")

    with requests.get(
        download_url,
        stream=True,
        timeout=(30, 180)
    ) as response:

        response.raise_for_status()

        downloaded_bytes = 0

        with open(temp_path, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file.write(chunk)
                    downloaded_bytes += len(chunk)

    print(
        "Downloaded:",
        round(downloaded_bytes / (1024 * 1024), 2),
        "MB"
    )

    # Verify the downloaded file before using it
    preview = pd.read_csv(temp_path, nrows=5)

    required_columns = {
        "open_dt",
        "on_time",
        "case_status"
    }

    if not required_columns.issubset(preview.columns):
        raise ValueError("Downloaded file has unexpected columns.")

    temp_path.replace(OUTPUT_PATH)

    print("Saved to:", OUTPUT_PATH)

    # Inspect complete dataset coverage
    df = pd.read_csv(
        OUTPUT_PATH,
        usecols=["open_dt"]
    )

    dates = pd.to_datetime(
        df["open_dt"],
        format="mixed",
        errors="coerce"
    )

    print("\nTotal records:", len(df))
    print("Earliest request:", dates.min())
    print("Latest request:", dates.max())

    print("\nRequests per month:")
    print(
        dates.dt.to_period("M")
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()