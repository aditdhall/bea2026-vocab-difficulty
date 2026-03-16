"""Write frozen feature lists to frozen_features.json."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Freeze selected features to frozen_features.json")
    parser.add_argument("--shared", type=str, default="", help="Comma-separated shared feature names")
    parser.add_argument("--es_de", type=str, default="", help="Comma-separated es/de-specific feature names")
    parser.add_argument("--cn", type=str, default="", help="Comma-separated cn-specific feature names")
    parser.add_argument("-o", "--output", default="frozen_features.json", help="Output JSON path")
    args = parser.parse_args()

    shared = [s.strip() for s in args.shared.split(",") if s.strip()]
    es_de_specific = [s.strip() for s in args.es_de.split(",") if s.strip()]
    cn_specific = [s.strip() for s in args.cn.split(",") if s.strip()]

    data = {
        "shared": shared,
        "es_de_specific": es_de_specific,
        "cn_specific": cn_specific,
    }
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
