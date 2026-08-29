import pytest

from ipind2.nlp_interface import RuleBasedQueryParser, TargetParameters, parse_query


class TestParseQuery:
    def test_extracts_scaffold_and_tissue_persian(self):
        text = "یک نانوحامل لیپیدی برای هدف‌گیری تومور می‌خواهم"
        params = parse_query(text)
        assert params.scaffold_type == "lipid"
        assert params.target_tissue == "tumor"

    def test_extracts_size_range_persian_digits(self):
        text = "اندازه بین ۸۰ تا ۱۲۰ نانومتر باشد"
        params = parse_query(text)
        assert params.size_range_nm == (80.0, 120.0)

    def test_extracts_size_max_only(self):
        text = "اندازه زیر ۵۰ نانومتر"
        params = parse_query(text)
        assert params.size_range_nm == (0.0, 50.0)

    def test_extracts_toxicity_constraint(self):
        text = "سمیت زیر ۲۰ باشد"
        params = parse_query(text)
        assert params.max_toxicity_ic50 == pytest.approx(20.0)

    def test_extracts_loading_efficiency(self):
        text = "کارایی بارگذاری بالای ۷۰ درصد"
        params = parse_query(text)
        assert params.min_loading_efficiency == pytest.approx(70.0)

    def test_english_query(self):
        text = "polymer nanocarrier targeting liver, size between 90 to 110 nm"
        params = parse_query(text)
        assert params.scaffold_type == "polymer"
        assert params.target_tissue == "liver"
        assert params.size_range_nm == (90.0, 110.0)

    def test_metal_scaffold_keyword(self):
        params = parse_query("nanocarrier فلزی برای مغز")
        assert params.scaffold_type == "metal"
        assert params.target_tissue == "brain"

    def test_unresolved_terms_when_nothing_recognized(self):
        params = parse_query("یک پیام کاملا نامرتبط")
        assert "scaffold_type" in params.unresolved_terms
        assert "target_tissue" in params.unresolved_terms
        assert params.size_range_nm is None

    def test_is_complete(self):
        complete = TargetParameters(scaffold_type="lipid", target_tissue="liver")
        incomplete = TargetParameters(scaffold_type="lipid")
        assert complete.is_complete()
        assert not incomplete.is_complete()

    def test_to_dict_roundtrip_keys(self):
        params = parse_query("نانوحامل لیپیدی برای کبد")
        d = params.to_dict()
        assert d["scaffold_type"] == "lipid"
        assert d["target_tissue"] == "liver"
        assert "unresolved_terms" in d


def test_rule_based_query_parser_delegates():
    parser = RuleBasedQueryParser()
    result = parser.parse("نانوحامل پلیمری برای ریه")
    assert isinstance(result, TargetParameters)
    assert result.scaffold_type == "polymer"
    assert result.target_tissue == "lung"
