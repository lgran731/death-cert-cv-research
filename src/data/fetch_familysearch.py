import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Placeholder for Tier-3 FamilySearch API fetcher."
    )
    parser.add_argument("--state", default="md", help="State code")
    args = parser.parse_args()
    print(
        "Tier-3 fetcher stub. Next step: implement FamilySearch OAuth + collection "
        f"download flow for state '{args.state}'."
    )


if __name__ == "__main__":
    main()

