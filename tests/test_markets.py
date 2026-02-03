from src.markets import parse_clob_token_ids


def test_parse_clob_token_ids():
    s = '["123","456"]'
    ids = parse_clob_token_ids(s)
    assert ids == ["123", "456"]
