import pytest
from auditok import StreamTokenizer, StringDataSource, DataValidator


class AValidator(DataValidator):
    def is_valid(self, frame):
        return frame == "A"


@pytest.fixture
def validator():
    return AValidator()


def test_init_min_0_init_max_silence_0(validator):
    tokenizer = StreamTokenizer(
        validator,
        min_length=5,
        max_length=20,
        max_continuous_silence=4,
        init_min=0,
        init_max_silence=0,
        mode=0,
    )

    data_source = StringDataSource("aAaaaAaAaaAaAaaaaaaaAAAAAAAA")
    #                                ^              ^   ^      ^
    #                                2              16  20     27
    tokens = tokenizer.tokenize(data_source)

    assert (
        len(tokens) == 2
    ), f"wrong number of tokens, expected: 2, found: {len(tokens)}"
    tok1, tok2 = tokens[0], tokens[1]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AaaaAaAaaAaAaaaa"
    ), f"wrong data for token 1, expected: 'AaaaAaAaaAaAaaaa', found: {data}"
    assert (
        start == 1
    ), f"wrong start frame for token 1, expected: 1, found: {start}"
    assert end == 16, f"wrong end frame for token 1, expected: 16, found: {end}"

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAAAAAAA"
    ), f"wrong data for token 2, expected: 'AAAAAAAA', found: {data}"
    assert (
        start == 20
    ), f"wrong start frame for token 2, expected: 20, found: {start}"
    assert end == 27, f"wrong end frame for token 2, expected: 27, found: {end}"


def test_init_min_3_init_max_silence_0(validator):
    tokenizer = StreamTokenizer(
        validator,
        min_length=5,
        max_length=20,
        max_continuous_silence=4,
        init_min=3,
        init_max_silence=0,
        mode=0,
    )

    data_source = StringDataSource("aAaaaAaAaaAaAaaaaaAAAAAAAAAaaaaaaAAAAA")
    #                                                 ^           ^  ^   ^
    #                                                 18          30 33  37

    tokens = tokenizer.tokenize(data_source)

    assert (
        len(tokens) == 2
    ), f"wrong number of tokens, expected: 2, found: {len(tokens)}"
    tok1, tok2 = tokens[0], tokens[1]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AAAAAAAAAaaaa"
    ), f"wrong data for token 1, expected: 'AAAAAAAAAaaaa', found: '{data}'"
    assert (
        start == 18
    ), f"wrong start frame for token 1, expected: 18, found: {start}"
    assert end == 30, f"wrong end frame for token 1, expected: 30, found: {end}"

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAAAA"
    ), f"wrong data for token 2, expected: 'AAAAA', found: '{data}'"
    assert (
        start == 33
    ), f"wrong start frame for token 2, expected: 33, found: {start}"
    assert end == 37, f"wrong end frame for token 2, expected: 37, found: {end}"


def test_init_min_3_init_max_silence_2(validator):
    tokenizer = StreamTokenizer(
        validator,
        min_length=5,
        max_length=20,
        max_continuous_silence=4,
        init_min=3,
        init_max_silence=2,
        mode=0,
    )

    data_source = StringDataSource("aAaaaAaAaaAaAaaaaaaAAAAAAAAAaaaaaaaAAAAA")
    #                                    ^          ^  ^           ^   ^   ^
    #                                    5          16 19          31  35  39
    tokens = tokenizer.tokenize(data_source)

    assert (
        len(tokens) == 3
    ), f"wrong number of tokens, expected: 3, found: {len(tokens)}"
    tok1, tok2, tok3 = tokens[0], tokens[1], tokens[2]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AaAaaAaAaaaa"
    ), f"wrong data for token 1, expected: 'AaAaaAaA', found: '{data}'"
    assert (
        start == 5
    ), f"wrong start frame for token 1, expected: 5, found: {start}"
    assert end == 16, f"wrong end frame for token 1, expected: 16, found: {end}"

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAAAAAAAAaaaa"
    ), f"wrong data for token 2, expected: 'AAAAAAAAAaaaa', found: '{data}'"
    assert (
        start == 19
    ), f"wrong start frame for token 2, expected: 19, found: {start}"
    assert end == 31, f"wrong end frame for token 2, expected: 31, found: {end}"

    data = "".join(tok3[0])
    start = tok3[1]
    end = tok3[2]
    assert (
        data == "AAAAA"
    ), f"wrong data for token 3, expected: 'AAAAA', found: '{data}'"
    assert (
        start == 35
    ), f"wrong start frame for token 3, expected: 35, found: {start}"
    assert end == 39, f"wrong end frame for token 3, expected: 39, found: {end}"


@pytest.fixture
def tokenizer_min_max_length(validator):
    return StreamTokenizer(
        validator,
        min_length=6,
        max_length=20,
        max_continuous_silence=2,
        init_min=3,
        init_max_silence=3,
        mode=0,
    )


def test_min_length_6_init_max_length_20(tokenizer_min_max_length):
    data_source = StringDataSource("aAaaaAaAaaAaAaaaaaAAAAAAAAAaaaaaAAAAA")
    #                                ^            ^   ^         ^
    #                                1            14  18        28

    tokens = tokenizer_min_max_length.tokenize(data_source)

    assert (
        len(tokens) == 2
    ), f"wrong number of tokens, expected: 2, found: {len(tokens)}"
    tok1, tok2 = tokens[0], tokens[1]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AaaaAaAaaAaAaa"
    ), f"wrong data for token 1, expected: 'AaaaAaAaaAaAaa', found: '{data}'"
    assert (
        start == 1
    ), f"wrong start frame for token 1, expected: 1, found: {start}"
    assert end == 14, f"wrong end frame for token 1, expected: 14, found: {end}"

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAAAAAAAAaa"
    ), f"wrong data for token 2, expected: 'AAAAAAAAAaa', found: '{data}'"
    assert (
        start == 18
    ), f"wrong start frame for token 2, expected: 18, found: {start}"
    assert end == 28, f"wrong end frame for token 2, expected: 28, found: {end}"


@pytest.fixture
def tokenizer_min_max_length_1_1(validator):
    return StreamTokenizer(
        validator,
        min_length=1,
        max_length=1,
        max_continuous_silence=0,
        init_min=0,
        init_max_silence=0,
        mode=0,
    )


def test_min_length_1_init_max_length_1(tokenizer_min_max_length_1_1):
    data_source = StringDataSource("AAaaaAaaaAaAaaAaAaaaaaAAAAAAAAAaaaaaAAAAA")

    tokens = tokenizer_min_max_length_1_1.tokenize(data_source)

    assert (
        len(tokens) == 21
    ), f"wrong number of tokens, expected: 21, found: {len(tokens)}"


@pytest.fixture
def tokenizer_min_max_length_10_20(validator):
    return StreamTokenizer(
        validator,
        min_length=10,
        max_length=20,
        max_continuous_silence=4,
        init_min=3,
        init_max_silence=3,
        mode=0,
    )


def test_min_length_10_init_max_length_20(tokenizer_min_max_length_10_20):
    data_source = StringDataSource(
        "aAaaaAaAaaAaAaaaaaaAAAAAaaaaaaAAAAAaaAAaaAAA"
    )
    #     ^              ^             ^            ^
    #     1              16            30           45

    tokens = tokenizer_min_max_length_10_20.tokenize(data_source)

    assert (
        len(tokens) == 2
    ), f"wrong number of tokens, expected: 2, found: {len(tokens)}"
    tok1, tok2 = tokens[0], tokens[1]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AaaaAaAaaAaAaaaa"
    ), f"wrong data for token 1, expected: 'AaaaAaAaaAaAaaaa', found: '{data}'"
    assert (
        start == 1
    ), f"wrong start frame for token 1, expected: 1, found: {start}"
    assert end == 16, f"wrong end frame for token 1, expected: 16, found: {end}"

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAAAAaaAAaaAAA"
    ), f"wrong data for token 2, expected: 'AAAAAaaAAaaAAA', found: '{data}'"
    assert (
        start == 30
    ), f"wrong start frame for token 2, expected: 30, found: {start}"
    assert end == 43, f"wrong end frame for token 2, expected: 43, found: {end}"


@pytest.fixture
def tokenizer_min_max_length_4_5(validator):
    return StreamTokenizer(
        validator,
        min_length=4,
        max_length=5,
        max_continuous_silence=4,
        init_min=3,
        init_max_silence=3,
        mode=0,
    )


def test_min_length_4_init_max_length_5(tokenizer_min_max_length_4_5):
    data_source = StringDataSource(
        "aAaaaAaAaaAaAaaaaaAAAAAAAAaaaaaaAAAAAaaaaaAAaaAaa"
    )
    #                      ^   ^^   ^    ^   ^     ^   ^
    #                      18 2223  27   32  36    42  46

    tokens = tokenizer_min_max_length_4_5.tokenize(data_source)

    assert (
        len(tokens) == 4
    ), f"wrong number of tokens, expected: 4, found: {len(tokens)}"
    tok1, tok2, tok3, tok4 = tokens[0], tokens[1], tokens[2], tokens[3]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AAAAA"
    ), f"wrong data for token 1, expected: 'AAAAA', found: '{data}'"
    assert (
        start == 18
    ), f"wrong start frame for token 1, expected: 18, found: {start}"
    assert end == 22, f"wrong end frame for token 1, expected: 22, found: {end}"

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAAaa"
    ), f"wrong data for token 2, expected: 'AAAaa', found: '{data}'"
    assert (
        start == 23
    ), f"wrong start frame for token 2, expected: 23, found: {start}"
    assert end == 27, f"wrong end frame for token 2, expected: 27, found: {end}"

    data = "".join(tok3[0])
    start = tok3[1]
    end = tok3[2]
    assert (
        data == "AAAAA"
    ), f"wrong data for token 3, expected: 'AAAAA', found: '{data}'"
    assert (
        start == 32
    ), f"wrong start frame for token 3, expected: 32, found: {start}"
    assert end == 36, f"wrong end frame for token 3, expected: 36, found: {end}"

    data = "".join(tok4[0])
    start = tok4[1]
    end = tok4[2]
    assert (
        data == "AAaaA"
    ), f"wrong data for token 4, expected: 'AAaaA', found: '{data}'"
    assert (
        start == 42
    ), f"wrong start frame for token 4, expected: 42, found: {start}"
    assert end == 46, f"wrong end frame for token 4, expected: 46, found: {end}"


@pytest.fixture
def tokenizer_max_continuous_silence_0(validator):
    return StreamTokenizer(
        validator,
        min_length=5,
        max_length=10,
        max_continuous_silence=0,
        init_min=3,
        init_max_silence=3,
        mode=0,
    )


def test_min_5_max_10_max_continuous_silence_0(
    tokenizer_max_continuous_silence_0,
):
    data_source = StringDataSource("aaaAAAAAaAAAAAAaaAAAAAAAAAa")
    #                                  ^   ^ ^    ^  ^       ^
    #                                  3   7 9   14 17      25

    tokens = tokenizer_max_continuous_silence_0.tokenize(data_source)

    assert (
        len(tokens) == 3
    ), f"wrong number of tokens, expected: 3, found: {len(tokens)}"
    tok1, tok2, tok3 = tokens[0], tokens[1], tokens[2]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AAAAA"
    ), f"wrong data for token 1, expected: 'AAAAA', found: '{data}'"
    assert (
        start == 3
    ), f"wrong start frame for token 1, expected: 3, found: {start}"
    assert end == 7, f"wrong end frame for token 1, expected: 7, found: {end}"

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAAAAA"
    ), f"wrong data for token 2, expected: 'AAAAAA', found: '{data}'"
    assert (
        start == 9
    ), f"wrong start frame for token 2, expected: 9, found: {start}"
    assert end == 14, f"wrong end frame for token 2, expected: 14, found: {end}"

    data = "".join(tok3[0])
    start = tok3[1]
    end = tok3[2]
    assert (
        data == "AAAAAAAAA"
    ), f"wrong data for token 3, expected: 'AAAAAAAAA', found: '{data}'"
    assert (
        start == 17
    ), f"wrong start frame for token 3, expected: 17, found: {start}"
    assert end == 25, f"wrong end frame for token 3, expected: 25, found: {end}"


@pytest.fixture
def tokenizer_max_continuous_silence_1(validator):
    return StreamTokenizer(
        validator,
        min_length=5,
        max_length=10,
        max_continuous_silence=1,
        init_min=3,
        init_max_silence=3,
        mode=0,
    )


def test_min_5_max_10_max_continuous_silence_1(
    tokenizer_max_continuous_silence_1,
):
    data_source = StringDataSource("aaaAAAAAaAAAAAAaaAAAAAAAAAa")
    #                                  ^        ^^ ^ ^        ^
    #                                  3       12131517      26
    #                                         (12 13 15 17)

    tokens = tokenizer_max_continuous_silence_1.tokenize(data_source)

    assert (
        len(tokens) == 3
    ), f"wrong number of tokens, expected: 3, found: {len(tokens)}"
    tok1, tok2, tok3 = tokens[0], tokens[1], tokens[2]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AAAAAaAAAA"
    ), f"wrong data for token 1, expected: 'AAAAAaAAAA', found: '{data}'"
    assert (
        start == 3
    ), f"wrong start frame for token 1, expected: 3, found: {start}"
    assert end == 12, f"wrong end frame for token 1, expected: 12, found: {end}"

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAa"
    ), f"wrong data for token 2, expected: 'AAa', found: '{data}'"
    assert (
        start == 13
    ), f"wrong start frame for token 2, expected: 13, found: {start}"
    assert end == 15, f"wrong end frame for token 2, expected: 15, found: {end}"

    data = "".join(tok3[0])
    start = tok3[1]
    end = tok3[2]
    assert (
        data == "AAAAAAAAAa"
    ), f"wrong data for token 3, expected: 'AAAAAAAAAa', found: '{data}'"
    assert (
        start == 17
    ), f"wrong start frame for token 3, expected: 17, found: {start}"
    assert end == 26, f"wrong end frame for token 3, expected: 26, found: {end}"


@pytest.fixture
def tokenizer_strict_min_length(validator):
    return StreamTokenizer(
        validator,
        min_length=5,
        max_length=8,
        max_continuous_silence=3,
        init_min=3,
        init_max_silence=3,
        mode=StreamTokenizer.STRICT_MIN_LENGTH,
    )


def test_STRICT_MIN_LENGTH(tokenizer_strict_min_length):
    data_source = StringDataSource("aaAAAAAAAAAAAA")
    #                                 ^      ^
    #                                 2      9

    tokens = tokenizer_strict_min_length.tokenize(data_source)

    assert (
        len(tokens) == 1
    ), f"wrong number of tokens, expected: 1, found: {len(tokens)}"
    tok1 = tokens[0]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AAAAAAAA"
    ), f"wrong data for token 1, expected: 'AAAAAAAA', found: '{data}'"
    assert (
        start == 2
    ), f"wrong start frame for token 1, expected: 2, found: {start}"
    assert end == 9, f"wrong end frame for token 1, expected: 9, found: {end}"


@pytest.fixture
def tokenizer_drop_trailing_silence(validator):
    return StreamTokenizer(
        validator,
        min_length=5,
        max_length=10,
        max_continuous_silence=2,
        init_min=3,
        init_max_silence=3,
        mode=StreamTokenizer.DROP_TRAILING_SILENCE,
    )


def test_DROP_TAILING_SILENCE(tokenizer_drop_trailing_silence):
    data_source = StringDataSource("aaAAAAAaaaaa")
    #                                 ^   ^
    #                                 2   6

    tokens = tokenizer_drop_trailing_silence.tokenize(data_source)

    assert (
        len(tokens) == 1
    ), f"wrong number of tokens, expected: 1, found: {len(tokens)}"
    tok1 = tokens[0]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AAAAA"
    ), f"wrong data for token 1, expected: 'AAAAA', found: '{data}'"
    assert (
        start == 2
    ), f"wrong start frame for token 1, expected: 2, found: {start}"
    assert end == 6, f"wrong end frame for token 1, expected: 6, found: {end}"


@pytest.fixture
def tokenizer_strict_min_and_drop_trailing_silence(validator):
    return StreamTokenizer(
        validator,
        min_length=5,
        max_length=8,
        max_continuous_silence=3,
        init_min=3,
        init_max_silence=3,
        mode=StreamTokenizer.STRICT_MIN_LENGTH
        | StreamTokenizer.DROP_TRAILING_SILENCE,
    )


def test_STRICT_MIN_LENGTH_and_DROP_TAILING_SILENCE(
    tokenizer_strict_min_and_drop_trailing_silence,
):
    data_source = StringDataSource("aaAAAAAAAAAAAAaa")
    #                                 ^      ^
    #                                 2      8

    tokens = tokenizer_strict_min_and_drop_trailing_silence.tokenize(
        data_source
    )

    assert (
        len(tokens) == 1
    ), f"wrong number of tokens, expected: 1, found: {len(tokens)}"
    tok1 = tokens[0]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AAAAAAAA"
    ), f"wrong data for token 1, expected: 'AAAAAAAA', found: '{data}'"
    assert (
        start == 2
    ), f"wrong start frame for token 1, expected: 2, found: {start}"
    assert end == 9, f"wrong end frame for token 1, expected: 9, found: {end}"


@pytest.fixture
def tokenizer_callback(validator):
    return StreamTokenizer(
        validator,
        min_length=5,
        max_length=8,
        max_continuous_silence=3,
        init_min=3,
        init_max_silence=3,
        mode=0,
    )


def test_callback(tokenizer_callback):
    tokens = []

    def callback(data, start, end):
        tokens.append((data, start, end))

    data_source = StringDataSource("aaAAAAAAAAAAAAa")
    #                                 ^      ^^   ^
    #                                 2      910  14

    tokenizer_callback.tokenize(data_source, callback=callback)

    assert (
        len(tokens) == 2
    ), f"wrong number of tokens, expected: 2, found: {len(tokens)}"
