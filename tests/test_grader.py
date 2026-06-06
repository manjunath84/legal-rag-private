from raglab.crag import parse_grade


def test_parse_grade_yes_variants():
    for reply in ["yes", "Yes", "YES", " yes ", "yes.", "Yes, relevant", '"yes"', "*yes*"]:
        assert parse_grade(reply) is True


def test_parse_grade_no_variants():
    for reply in ["no", "No", "NO", " no ", "no.", "not relevant", "", "irrelevant"]:
        assert parse_grade(reply) is False
