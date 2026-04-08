from pathlib import Path


def main() -> None:
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print("Populate this script with offline model acquisition steps for your local environment.")


if __name__ == "__main__":
    main()
