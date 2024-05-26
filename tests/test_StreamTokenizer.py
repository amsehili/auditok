import os

import pytest

from auditok import DataValidator, StreamTokenizer, StringDataSource


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
    ), "wrong number of tokens, expected: 2, found: {}".format(len(tokens))
    tok1, tok2 = tokens[0], tokens[1]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AaaaAaAaaAaAaaaa"
    ), "wrong data for token 1, expected: 'AaaaAaAaaAaAaaaa', found: {}".format(
        data
    )
    assert (
        start == 1
    ), "wrong start frame for token 1, expected: 1, found: {}".format(start)
    assert (
        end == 16
    ), "wrong end frame for token 1, expected: 16, found: {}".format(end)

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAAAAAAA"
    ), "wrong data for token 2, expected: 'AAAAAAAA', found: {}".format(data)
    assert (
        start == 20
    ), "wrong start frame for token 2, expected: 20, found: {}".format(start)
    assert (
        end == 27
    ), "wrong end frame for token 2, expected: 27, found: {}".format(end)


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
    ), "wrong number of tokens, expected: 2, found: {}".format(len(tokens))
    tok1, tok2 = tokens[0], tokens[1]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AAAAAAAAAaaaa"
    ), "wrong data for token 1, expected: 'AAAAAAAAAaaaa', found: {}".format(
        data
    )
    assert (
        start == 18
    ), "wrong start frame for token 1, expected: 18, found: {}".format(start)
    assert (
        end == 30
    ), "wrong end frame for token 1, expected: 30, found: {}".format(end)

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAAAA"
    ), "wrong data for token 2, expected: 'AAAAA', found: {}".format(data)
    assert (
        start == 33
    ), "wrong start frame for token 2, expected: 33, found: {}".format(start)
    assert (
        end == 37
    ), "wrong end frame for token 2, expected: 37, found: {}".format(end)


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
    ), "wrong number of tokens, expected: 3, found: {}".format(len(tokens))
    tok1, tok2, tok3 = tokens[0], tokens[1], tokens[2]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AaAaaAaAaaaa"
    ), "wrong data for token 1, expected: 'AaAaaAaA', found: {}".format(data)
    assert (
        start == 5
    ), "wrong start frame for token 1, expected: 5, found: {}".format(start)
    assert (
        end == 16
    ), "wrong end frame for token 1, expected: 16, found: {}".format(end)

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAAAAAAAAaaaa"
    ), "wrong data for token 2, expected: 'AAAAAAAAAaaaa', found: {}".format(
        data
    )
    assert (
        start == 19
    ), "wrong start frame for token 2, expected: 19, found: {}".format(start)
    assert (
        end == 31
    ), "wrong end frame for token 2, expected: 31, found: {}".format(end)

    data = "".join(tok3[0])
    start = tok3[1]
    end = tok3[2]
    assert (
        data == "AAAAA"
    ), "wrong data for token 3, expected: 'AAAAA', found: {}".format(data)
    assert (
        start == 35
    ), "wrong start frame for token 3, expected: 35, found: {}".format(start)
    assert (
        end == 39
    ), "wrong end frame for token 3, expected: 39, found: {}".format(end)


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
    ), "wrong number of tokens, expected: 2, found: {}".format(len(tokens))
    tok1, tok2 = tokens[0], tokens[1]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AaaaAaAaaAaAaa"
    ), "wrong data for token 1, expected: 'AaaaAaAaaAaAaa', found: {}".format(
        data
    )
    assert (
        start == 1
    ), "wrong start frame for token 1, expected: 1, found: {}".format(start)
    assert (
        end == 14
    ), "wrong end frame for token 1, expected: 14, found: {}".format(end)

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAAAAAAAAaa"
    ), "wrong data for token 2, expected: 'AAAAAAAAAaa', found: {}".format(data)
    assert (
        start == 18
    ), "wrong start frame for token 2, expected: 18, found: {}".format(start)
    assert (
        end == 28
    ), "wrong end frame for token 2, expected: 28, found: {}".format(end)


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
    ), "wrong number of tokens, expected: 21, found: {}".format(len(tokens))


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
    ), "wrong number of tokens, expected: 2, found: {}".format(len(tokens))
    tok1, tok2 = tokens[0], tokens[1]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AaaaAaAaaAaAaaaa"
    ), "wrong data for token 1, expected: 'AaaaAaAaaAaAaaaa', found: {}".format(
        data
    )
    assert (
        start == 1
    ), "wrong start frame for token 1, expected: 1, found: {}".format(start)
    assert (
        end == 16
    ), "wrong end frame for token 1, expected: 16, found: {}".format(end)

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAAAAaaAAaaAAA"
    ), "wrong data for token 2, expected: 'AAAAAaaAAaaAAA', found: {}".format(
        data
    )
    assert (
        start == 30
    ), "wrong start frame for token 2, expected: 30, found: {}".format(start)
    assert (
        end == 43
    ), "wrong end frame for token 2, expected: 43, found: {}".format(end)


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
    ), "wrong number of tokens, expected: 4, found: {}".format(len(tokens))
    tok1, tok2, tok3, tok4 = tokens[0], tokens[1], tokens[2], tokens[3]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AAAAA"
    ), "wrong data for token 1, expected: 'AAAAA', found: {}".format(data)
    assert (
        start == 18
    ), "wrong start frame for token 1, expected: 18, found: {}".format(start)
    assert (
        end == 22
    ), "wrong end frame for token 1, expected: 22, found: {}".format(end)

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAAaa"
    ), "wrong data for token 2, expected: 'AAAaa', found: {}".format(data)
    assert (
        start == 23
    ), "wrong start frame for token 2, expected: 23, found: {}".format(start)
    assert (
        end == 27
    ), "wrong end frame for token 2, expected: 27, found: {}".format(end)

    data = "".join(tok3[0])
    start = tok3[1]
    end = tok3[2]
    assert (
        data == "AAAAA"
    ), "wrong data for token 3, expected: 'AAAAA', found: {}".format(data)
    assert (
        start == 32
    ), "wrong start frame for token 3, expected: 32, found: {}".format(start)
    assert (
        end == 36
    ), "wrong end frame for token 3, expected: 36, found: {}".format(end)

    data = "".join(tok4[0])
    start = tok4[1]
    end = tok4[2]
    assert (
        data == "AAaaA"
    ), "wrong data for token 4, expected: 'AAaaA', found: {}".format(data)
    assert (
        start == 42
    ), "wrong start frame for token 4, expected: 42, found: {}".format(start)
    assert (
        end == 46
    ), "wrong end frame for token 4, expected: 46, found: {}".format(end)


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
    ), "wrong number of tokens, expected: 3, found: {}".format(len(tokens))
    tok1, tok2, tok3 = tokens[0], tokens[1], tokens[2]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AAAAA"
    ), "wrong data for token 1, expected: 'AAAAA', found: {}".format(data)
    assert (
        start == 3
    ), "wrong start frame for token 1, expected: 3, found: {}".format(start)
    assert (
        end == 7
    ), "wrong end frame for token 1, expected: 7, found: {}".format(end)

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAAAAA"
    ), "wrong data for token 2, expected: 'AAAAAA', found: {}".format(data)
    assert (
        start == 9
    ), "wrong start frame for token 2, expected: 9, found: {}".format(start)
    assert (
        end == 14
    ), "wrong end frame for token 2, expected: 14, found: {}".format(end)

    data = "".join(tok3[0])
    start = tok3[1]
    end = tok3[2]
    assert (
        data == "AAAAAAAAA"
    ), "wrong data for token 3, expected: 'AAAAAAAAA', found: {}".format(data)
    assert (
        start == 17
    ), "wrong start frame for token 3, expected: 17, found: {}".format(start)
    assert (
        end == 25
    ), "wrong end frame for token 3, expected: 25, found: {}".format(end)


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
    ), "wrong number of tokens, expected: 3, found: {}".format(len(tokens))
    tok1, tok2, tok3 = tokens[0], tokens[1], tokens[2]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AAAAAaAAAA"
    ), "wrong data for token 1, expected: 'AAAAAaAAAA', found: {}".format(data)
    assert (
        start == 3
    ), "wrong start frame for token 1, expected: 3, found: {}".format(start)
    assert (
        end == 12
    ), "wrong end frame for token 1, expected: 12, found: {}".format(end)

    data = "".join(tok2[0])
    start = tok2[1]
    end = tok2[2]
    assert (
        data == "AAa"
    ), "wrong data for token 2, expected: 'AAa', found: {}".format(data)
    assert (
        start == 13
    ), "wrong start frame for token 2, expected: 13, found: {}".format(start)
    assert (
        end == 15
    ), "wrong end frame for token 2, expected: 15, found: {}".format(end)

    data = "".join(tok3[0])
    start = tok3[1]
    end = tok3[2]
    assert (
        data == "AAAAAAAAAa"
    ), "wrong data for token 3, expected: 'AAAAAAAAAa', found: {}".format(data)
    assert (
        start == 17
    ), "wrong start frame for token 3, expected: 17, found: {}".format(start)
    assert (
        end == 26
    ), "wrong end frame for token 3, expected: 26, found: {}".format(end)


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
    ), "wrong number of tokens, expected: 1, found: {}".format(len(tokens))
    tok1 = tokens[0]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AAAAAAAA"
    ), "wrong data for token 1, expected: 'AAAAAAAA', found: {}".format(data)
    assert (
        start == 2
    ), "wrong start frame for token 1, expected: 2, found: {}".format(start)
    assert (
        end == 9
    ), "wrong end frame for token 1, expected: 9, found: {}".format(end)


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
    ), "wrong number of tokens, expected: 1, found: {}".format(len(tokens))
    tok1 = tokens[0]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AAAAA"
    ), "wrong data for token 1, expected: 'AAAAA', found: {}".format(data)
    assert (
        start == 2
    ), "wrong start frame for token 1, expected: 2, found: {}".format(start)
    assert (
        end == 6
    ), "wrong end frame for token 1, expected: 6, found: {}".format(end)


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
    ), "wrong number of tokens, expected: 1, found: {}".format(len(tokens))
    tok1 = tokens[0]

    data = "".join(tok1[0])
    start = tok1[1]
    end = tok1[2]
    assert (
        data == "AAAAAAAA"
    ), "wrong data for token 1, expected: 'AAAAAAAA', found: {}".format(data)
    assert (
        start == 2
    ), "wrong start frame for token 1, expected: 2, found: {}".format(start)
    assert (
        end == 9
    ), "wrong end frame for token 1, expected: 9, found: {}".format(end)


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
    ), "wrong number of tokens, expected: 2, found: {}".format(len(tokens))
