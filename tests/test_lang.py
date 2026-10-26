from bhasharag.lang import detect_language


def test_detects_scripts():
    assert detect_language("මහනුවර ශ්‍රී ලංකාවේ නගරයකි") == "si"
    assert detect_language("கண்டி இலங்கையின் நகரம்") == "ta"
    assert detect_language("Kandy is a city") == "en"
