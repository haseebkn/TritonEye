"""Optional real-data tests may open only registered development measurements."""

from pathlib import Path

from agents.nl_benchmark import bounded_path, load_dataset


def measurements(root: Path) -> list[str]:
    manifests = [
        load_dataset(path.parent)[0]
        for path in sorted((root / "datasets/nl_benchmark").glob("*/release.json"))
    ]
    held = {
        s["acquisition_group"]
        for m in manifests
        for s in m["scenes"]
        if s["split"] == "test"
    }
    allowed = {
        source["path"]
        for m in manifests
        for s in m["scenes"]
        if s["split"] in {"train", "validation"} and s["acquisition_group"] not in held
        for source in s["sources"]
    }
    result = []
    for name in sorted(allowed):
        path = bounded_path(root, name)
        if (
            path.parent.name == "measurement"
            and path.suffix in {".tif", ".tiff"}
            and path.is_file()
        ):
            result.append(str(path))
    return result
