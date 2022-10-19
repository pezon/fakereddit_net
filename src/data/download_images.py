# -*- coding: utf-8 -*-
import os
import logging
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--frac", default=0.02, type=float)
def main(
    input_filepath: Path,
    output_filepath: Path,
    frac: float = 0.02
):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    for file in Path(input_filepath).glob("*.tsv"):
        # sample and clean dataframe
        df = pd.read_csv(file, sep="\t").sample(frac=frac, random_state=42)
        df.replace(np.nan, "", regex=True)
        df.fillna("", inplace=True)

        # prepare output path
        output_filepath = Path(output_filepath)
        image_output_filepath = output_filepath / file.stem
        image_output_filepath.mkdir(exist_ok=True)

        # track ids that aren't found
        not_found_ids = []

        # add progress bar
        pbar = tqdm(total=len(df))

        for index, row in df.iterrows():
            # skip records without images
            if row["hasImage"] == False\
                    or row["image_url"] in ("", "nan"):
                not_found_ids.append(row["id"])
                pbar.write(f"skipping: no image_url")
                pbar.update(1)
                continue

            image_url = row["image_url"]
            image_dest = image_output_filepath / f"{row['id']}.jpg"

            # skip records that were already downloaded
            if image_dest.exists():
                pbar.write(f"skipping: {image_url}")
                pbar.update(1)
                continue

            try:
                # download images
                urlretrieve(image_url, str(image_dest))
                sleep(0.01)
            except (HTTPError, URLError) as err:
                # logger.info(f"could not find: {image_url}")
                # track files not found
                not_found_ids.append(row["id"])
                pbar.write(f"not found: {image_url}")
            pbar.update(1)

        # add column indicating whether image was downloaded and save
        df["image_downloaded"] = (~df["id"].isin(not_found_ids)).astype(int)
        df.to_csv(f"{output_filepath}/{file.stem}_resampled.csv", index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it"s found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
