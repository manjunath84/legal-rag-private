import pytest

from raglab.ingest import load_dir, load_file


def test_load_txt_and_md(tmp_path):
    (tmp_path / "a.txt").write_text("Plain text clause.")
    (tmp_path / "b.md").write_text("# Heading\n\nMarkdown clause.")
    docs = load_dir(tmp_path)
    assert {d.source for d in docs} == {"a.txt", "b.md"}
    assert all(d.page is None for d in docs)


def test_unsupported_type_raises(tmp_path):
    p = tmp_path / "c.docx"
    p.write_text("not supported in stage 1")
    with pytest.raises(ValueError):
        load_file(p)


def test_empty_text_file_yields_nothing(tmp_path):
    (tmp_path / "blank.txt").write_text("   ")
    assert load_dir(tmp_path) == []
