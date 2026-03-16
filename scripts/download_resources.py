"""Download standard NLP resources (closed-track allowed)."""

import os
import sys
import urllib.request

RESOURCES_DIR = "resources"
os.makedirs(RESOURCES_DIR, exist_ok=True)

# URLs and expected columns (resource name -> (url, local_filename, expected_columns_note))
RESOURCES = [
    (
        "SUBTLEX-US (Brysbaert & New, 2009)",
        "https://raw.githubusercontent.com/LanguageAndLearningLab/SUBTLEX-US/master/SUBTLEX-US.csv",
        "subtlex_us.csv",
        "Word, FREQcount (and others). Use Word + frequency column.",
    ),
    (
        "Age of Acquisition (Kuperman et al., 2012)",
        "https://raw.githubusercontent.com/LanguageAndLearningLab/aoa/master/aoa.csv",
        "aoa.csv",
        "Word, AoA (or word, aoa).",
    ),
    (
        "Concreteness (Brysbaert et al., 2014)",
        "https://raw.githubusercontent.com/LanguageAndLearningLab/concreteness/master/concreteness.csv",
        "concreteness.csv",
        "Word, Conc.M (or word, concreteness).",
    ),
    (
        "MRC Psycholinguistic (imageability)",
        "https://raw.githubusercontent.com/LanguageAndLearningLab/mrc/master/mrc.csv",
        "mrc.csv",
        "Word, IMAGE (or word, imageability).",
    ),
]


def download_one(name: str, url: str, filename: str, expected: str) -> bool:
    path = os.path.join(RESOURCES_DIR, filename)
    print(f"Downloading {name}...")
    print(f"  URL: {url}")
    print(f"  Expected columns: {expected}")
    try:
        urllib.request.urlretrieve(url, path)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            print(f"  -> Saved to {path}")
            return True
        print(f"  -> Failed (empty or missing)")
        return False
    except Exception as e:
        print(f"  -> Error: {e}")
        print(f"  Manual: save file to {os.path.abspath(path)}")
        return False

def download_unihan() -> bool:
    """Download Unicode Unihan data for Chinese stroke counts."""
    import zipfile
    import io
    path = os.path.join(RESOURCES_DIR, "Unihan_DictionaryLikeData.txt")
    if os.path.isfile(path):
        print(f"Unihan already exists at {path}")
        return True
    url = "https://www.unicode.org/Public/UCD/latest/uax38/Unihan.zip"
    print(f"Downloading Unihan (Chinese stroke counts)...")
    print(f"  URL: {url}")
    print(f"  Expected: kTotalStrokes field for cn_stroke_complexity feature")
    try:
        resp = urllib.request.urlopen(url)
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        target = "Unihan_DictionaryLikeData.txt"
        if target in z.namelist():
            z.extract(target, RESOURCES_DIR)
            print(f"  -> Saved to {os.path.join(RESOURCES_DIR, target)}")
            return True
        else:
            z.extractall(RESOURCES_DIR)
            print(f"  -> Extracted all Unihan files to {RESOURCES_DIR}")
            return True
    except Exception as e:
        print(f"  -> Error: {e}")
        print(f"  Manual: download {url}, unzip, place Unihan_DictionaryLikeData.txt in {RESOURCES_DIR}/")
        return False
def main():
    success = 0
    for name, url, filename, expected in RESOURCES:
        if download_one(name, url, filename, expected):
            success += 1
    if download_unihan():
        success += 1
    total = len(RESOURCES) + 1  # +1 for Unihan
    print(f"\nDone. {success}/{total} resources downloaded.")
    if success < len(RESOURCES):
        print("For failed downloads, place the CSV in resources/ with the expected column names.")
        sys.exit(1)


if __name__ == "__main__":
    main()
