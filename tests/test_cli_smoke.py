from lsd_app.cli.main import main


def test_prepare_command_smoke() -> None:
    assert main(["prepare", "--dry-run"]) == 0


def test_weight_and_dataset_registry_commands(tmp_path, capsys) -> None:
    weight = tmp_path / "weight.pth"
    imagery = tmp_path / "source.tif"
    label = tmp_path / "label.tif"
    weight.write_bytes(b"checkpoint")
    imagery.write_bytes(b"raster")
    label.write_bytes(b"label")

    assert (
        main(
            [
                "weights",
                "add",
                "--registry-root",
                str(tmp_path / "outputs"),
                "--model-id",
                "lsdsam",
                "--stage",
                "TAFT",
                "--weight-id",
                "taft-smoke",
                "--version",
                "v1",
                "--path",
                str(weight),
            ]
        )
        == 0
    )
    assert main(["weights", "list", "--registry-root", str(tmp_path / "outputs")]) == 0
    assert (
        main(
            [
                "weights",
                "delete",
                "--registry-root",
                str(tmp_path / "outputs"),
                "--weight-id",
                "taft-smoke",
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "datasets",
                "add",
                "--registry-root",
                str(tmp_path / "outputs"),
                "--name",
                "Small TSFT",
                "--class",
                "TSFT",
                "--imagery",
                str(imagery),
                "--label",
                str(label),
                "--has-gt",
                "--rgb-bands",
                "RGB_bands: 1,2,3",
            ]
        )
        == 0
    )

    assert main(["datasets", "list", "--registry-root", str(tmp_path / "outputs")]) == 0
    stdout = capsys.readouterr().out
    assert '"weight_id": "taft-smoke"' in stdout
    assert '"stage": "TAFT"' in stdout
    assert '"dataset_class": "TSFT"' in stdout
