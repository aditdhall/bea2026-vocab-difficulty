"""
Download external NLP resources required by the feature pipeline.
All resources are permitted under the BEA 2026 closed track rules.

Run from the repo root:
    python scripts/download_resources.py

Resources downloaded:
    - SUBTLEX-US word frequency norms (Brysbaert & New, 2009)
    - Age of Acquisition norms (Kuperman et al., 2012)
    - Concreteness ratings (Brysbaert et al., 2014)
    - Oxford 5000 CEFR word list
    - Unicode Unihan database (stroke counts for Mandarin)
"""

import io
import os
import sys
import urllib.request
import zipfile

import pandas as pd

RESOURCES_DIR = "resources"
os.makedirs(RESOURCES_DIR, exist_ok=True)


def download_subtlex() -> bool:
    """SUBTLEX-US (Brysbaert & New, 2009) — word frequency norms."""
    out_path = os.path.join(RESOURCES_DIR, "SUBTLEXus74286wordstextversion.txt")
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 100_000:
        print(f"SUBTLEX-US already exists at {out_path}")
        return True

    print("Downloading SUBTLEX-US (Brysbaert & New, 2009)...")
    url = "https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip"
    try:
        resp = urllib.request.urlopen(url)
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        z.extractall(RESOURCES_DIR)
        print(f"  -> Extracted: {z.namelist()}")
        return any(os.path.isfile(os.path.join(RESOURCES_DIR, f)) for f in z.namelist())
    except Exception as e:
        print(f"  -> Error: {e}")
        print("  Manual: https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus")
        print("  Download subtlexus2.zip, extract SUBTLEXus74286wordstextversion.txt to resources/")
        return False


def download_aoa() -> bool:
    """Age of Acquisition norms (Kuperman et al., 2012)."""
    out_path = os.path.join(RESOURCES_DIR, "aoa_kuperman.xlsx")
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 100_000:
        print(f"AoA already exists at {out_path}")
        return True

    print("Downloading Age of Acquisition (Kuperman et al., 2012)...")
    url = "https://osf.io/bx7vm/download"
    try:
        urllib.request.urlretrieve(url, out_path)
        size = os.path.getsize(out_path)
        if size > 100_000:
            print(f"  -> Saved: {out_path} ({size:,} bytes)")
            df = pd.read_excel(out_path)
            if "AoA_Kup" in df.columns:
                print(f"  -> Verified: AoA_Kup column present ({len(df):,} words)")
            return True
        os.remove(out_path)
        print("  -> Failed (file too small)")
        return False
    except Exception as e:
        print(f"  -> Error: {e}")
        print("  Manual: https://osf.io/bx7vm -> Download")
        print("  Save as resources/aoa_kuperman.xlsx")
        return False


def download_concreteness() -> bool:
    """Concreteness ratings (Brysbaert et al., 2014)."""
    out_path = os.path.join(RESOURCES_DIR, "concreteness.txt")
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 100_000:
        print(f"Concreteness already exists at {out_path}")
        return True

    print("Downloading Concreteness ratings (Brysbaert et al., 2014)...")
    url = "https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
    try:
        urllib.request.urlretrieve(url, out_path)
        size = os.path.getsize(out_path)
        if size > 100_000:
            print(f"  -> Saved: {out_path} ({size:,} bytes)")
            df = pd.read_csv(out_path, sep="\t")
            if "Conc.M" in df.columns:
                print(f"  -> Verified: Conc.M column present ({len(df):,} words)")
            return True
        os.remove(out_path)
        print("  -> Failed (file too small)")
        return False
    except Exception as e:
        print(f"  -> Error: {e}")
        print("  Manual: search 'Brysbaert concreteness ratings 2014 BRM supplementary'")
        print("  Save tab-separated file as resources/concreteness.txt")
        return False


def download_cefr() -> bool:
    """Oxford 5000 CEFR word list."""
    out_path = os.path.join(RESOURCES_DIR, "cefr_wordlist.csv")
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 10_000:
        print(f"CEFR wordlist already exists at {out_path}")
        return True

    print("Downloading Oxford 5000 CEFR word list...")
    url = "https://raw.githubusercontent.com/winterdl/oxford-5000-vocabulary-audio-definition/main/data/oxford_5000.csv"
    raw_path = os.path.join(RESOURCES_DIR, "oxford_5000_raw.csv")
    try:
        urllib.request.urlretrieve(url, raw_path)
        df = pd.read_csv(raw_path)
        cefr = df[["word", "cefr"]].copy()
        cefr.columns = ["Word", "Level"]
        cefr = cefr.dropna()
        cefr.to_csv(out_path, index=False)
        print(f"  -> Saved: {out_path} ({len(cefr):,} words)")
        print(f"  -> CEFR levels: {sorted(cefr['Level'].unique().tolist())}")
        return True
    except Exception as e:
        print(f"  -> Error: {e}")
        print("  Manual: https://github.com/winterdl/oxford-5000-vocabulary-audio-definition")
        print("  Download oxford_5000.csv, extract word+cefr columns, save as resources/cefr_wordlist.csv")
        return False


def download_unihan() -> bool:
    """Unicode Unihan database -- stroke counts for Mandarin features.
    Note: kTotalStrokes entries are in Unihan_IRGSources.txt, not DictionaryLikeData.txt.
    """
    out_path = os.path.join(RESOURCES_DIR, "Unihan_IRGSources.txt")
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 1_000_000:
        print(f"Unihan already exists at {out_path}")
        return True

    print("Downloading Unicode Unihan database (stroke counts)...")
    url = "https://www.unicode.org/Public/16.0.0/ucd/Unihan.zip"
    try:
        resp = urllib.request.urlopen(url)
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        z.extractall(RESOURCES_DIR)
        print(f"  -> Extracted: {z.namelist()}")
        if os.path.isfile(out_path):
            count = sum(1 for line in open(out_path, encoding="utf-8") if "kTotalStrokes" in line)
            print(f"  -> Verified: {count:,} kTotalStrokes entries in Unihan_IRGSources.txt")
            return count > 0
        return False
    except Exception as e:
        print(f"  -> Error: {e}")
        print("  Manual: https://www.unicode.org/Public/16.0.0/ucd/Unihan.zip")
        print("  Extract Unihan_IRGSources.txt to resources/")
        return False


def main():
    print("=" * 60)
    print("  BEA 2026 Resource Downloader")
    print("=" * 60)

    results = {
        "SUBTLEX-US": download_subtlex(),
        "AoA (Kuperman)": download_aoa(),
        "Concreteness (Brysbaert)": download_concreteness(),
        "CEFR (Oxford 5000)": download_cefr(),
        "Unihan (stroke counts)": download_unihan(),
    }

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    success = 0
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {name}")
        if ok:
            success += 1

    total = len(results)
    print(f"\n{success}/{total} resources downloaded successfully.")

    if success < total:
        print("\nFor failed resources, follow the manual download instructions above.")
        print("Note: WordNet is downloaded automatically via NLTK at runtime.")
        sys.exit(1)
    else:
        print("\nAll resources ready. You can now run the feature pipeline.")


if __name__ == "__main__":
    main()
