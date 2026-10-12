from interaction_gym.viewer.commentary import decode_pieces, units


def test_byte_level_pieces_become_text():
    toks = [[151706, "<|speak|>"], [108194, "æľĪäº®"], [18830, "æľī"], [3837, "ï¼Į"], [525, " are"], [151718, "<|chunk_eos|>"]]
    assert decode_pieces(toks) == [[151706, "<|speak|>"], [108194, "月亮"], [18830, "有"], [3837, "，"], [525, " are"], [151718, "<|chunk_eos|>"]]


def test_character_split_over_tokens_is_merged():
    # 中 = e4 b8 ad, split 1 + 2 bytes; the merged entry keeps every id
    assert decode_pieces([[1, "ä"], [2, "¸Ń"], [3, "x"]]) == [[[1, 2], "中"], [3, "x"]]


def test_broken_bytes_do_not_raise():
    out = decode_pieces([[1, "¸Ń"], [2, "<|listen|>"]])
    assert out[0][0] == 1 and "�" in out[0][1] and out[1] == [2, "<|listen|>"]


def test_units_text_is_readable():
    tr = {"unit_ms": 1000, "units": [{"unit_index": 0, "end_ms": 1000, "decision": "speak", "special_ids": [151706],
                                      "stages": [{"stage": "thinker", "output": [[151706, "<|speak|>"], [108194, "æľĪäº®"], [525, "Ġare"]]}]}]}
    assert units(tr)["units"][0]["text"] == "月亮 are"
