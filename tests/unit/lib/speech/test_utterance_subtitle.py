"""A dialogue line has a spoken text and an optional on-screen subtitle that can differ.

The spoken text is what goes into the video prompt (and so what the provider voices); the subtitle is only
what is shown. Without a subtitle the spoken text is shown, as before.
"""

from __future__ import annotations

import pytest

from lib.script.script_models import Utterance
from lib.speech.speech_artifact_provenance import (
    CharacterVoiceEvidence,
    build_video_speech_basis,
    project_subtitle_utterances,
)
from lib.speech.speech_composition import (
    SpeechComposition,
    SpeechFieldLocation,
    SpeechMode,
    SpeechOwner,
    SpeechPreparation,
    SpeechProblemCode,
    SpeechUtterance,
    adapt_drama_scene,
)


def _scene(*lines: dict) -> dict:
    return {"scene_id": "E1S01", "utterances": [{"kind": "dialogue", "speaker": "Haru", **line} for line in lines]}


def _prepare(scene: dict) -> SpeechPreparation:
    return SpeechComposition.prepare(adapt_drama_scene(scene))


def test_the_model_accepts_a_subtitle_and_defaults_to_none() -> None:
    plain = Utterance(kind="dialogue", speaker="Haru", text="うわっ、遅刻する！")
    assert plain.subtitle is None
    with_subtitle = Utterance(
        kind="dialogue", speaker="Haru", text="うわっ、遅刻する！", subtitle="Ôi, trễ học mất rồi!"
    )
    assert with_subtitle.subtitle == "Ôi, trễ học mất rồi!"


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_a_blank_subtitle_means_no_subtitle(blank: str) -> None:
    assert Utterance(kind="dialogue", speaker="Haru", text="はい", subtitle=blank).subtitle is None


def test_the_model_still_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="caption"):
        Utterance(kind="dialogue", speaker="Haru", text="はい", caption="x")  # type: ignore[call-arg]


def test_the_subtitle_travels_next_to_the_spoken_text_without_replacing_it() -> None:
    preparation = _prepare(
        _scene(
            {"text": "うわっ、遅刻する！", "subtitle": "Ôi, trễ học mất rồi!"},
            {"text": "えっ…なんだこれ！？"},
        )
    )

    assert preparation.mode is SpeechMode.CHARACTER_SPEECH
    assert [(u.text, u.subtitle) for u in preparation.utterances] == [
        ("うわっ、遅刻する！", "Ôi, trễ học mất rồi!"),
        ("えっ…なんだこれ！？", None),
    ]


def test_a_non_string_subtitle_is_reported_at_its_own_location() -> None:
    preparation = _prepare(_scene({"text": "はい", "subtitle": 5}))

    assert preparation.mode is None
    [problem] = preparation.problems
    assert problem.code is SpeechProblemCode.PARSE_FAILED
    assert problem.locations == (SpeechFieldLocation(("utterances", 0, "subtitle")),)


def test_subtitles_are_projected_from_the_subtitle_when_set_and_from_the_text_otherwise() -> None:
    preparation = _prepare(
        _scene(
            {"text": "うわっ、遅刻する！", "subtitle": "Ôi, trễ học mất rồi!"},
            {"text": "えっ…なんだこれ！？"},
        )
    )

    shown = project_subtitle_utterances(preparation)

    # the first line is shown as its subtitle; the second has none, so its spoken text is shown (cut into
    # sentences at the ellipsis, as subtitles always are)
    assert all((u.owner, u.speaker) == (SpeechOwner.CHARACTER, "Haru") for u in shown)
    assert shown[0].text == "Ôi, trễ học mất rồi!"
    assert "".join(u.text for u in shown[1:]) == "えっ…なんだこれ！？"
    assert "うわっ" not in "".join(u.text for u in shown)


def test_a_changed_subtitle_changes_the_subtitle_evidence_but_not_the_video_speech_basis() -> None:
    before = _prepare(_scene({"text": "うわっ、遅刻する！", "subtitle": "Ôi, trễ học mất rồi!"}))
    after = _prepare(_scene({"text": "うわっ、遅刻する！", "subtitle": "Á, muộn giờ mất!"}))
    voices = (CharacterVoiceEvidence(speaker="Haru", voice_style="nam 17 tuổi"),)

    assert [u.basis_input() for u in project_subtitle_utterances(before)] != [
        u.basis_input() for u in project_subtitle_utterances(after)
    ]
    # the paid video only depends on what is spoken, so editing the subtitle must not make it stale
    assert build_video_speech_basis(before, voices=voices) == build_video_speech_basis(after, voices=voices)
    # ...whereas changing the spoken text does
    spoken_changed = _prepare(_scene({"text": "あっ、遅刻だ！", "subtitle": "Ôi, trễ học mất rồi!"}))
    assert build_video_speech_basis(before, voices=voices) != build_video_speech_basis(spoken_changed, voices=voices)


def test_speech_utterance_subtitle_is_optional_for_existing_callers() -> None:
    utterance = SpeechUtterance(
        owner=SpeechOwner.CHARACTER,
        text="はい",
        speaker="Haru",
        location=SpeechFieldLocation(("utterances", 0, "text")),
    )
    assert utterance.subtitle is None


def test_dumping_omits_an_empty_subtitle_so_stored_scripts_keep_their_shape() -> None:
    assert Utterance(kind="dialogue", speaker="Haru", text="はい").model_dump() == {
        "kind": "dialogue",
        "speaker": "Haru",
        "text": "はい",
    }
    assert Utterance(kind="dialogue", speaker="Haru", text="はい", subtitle="Vâng").model_dump()["subtitle"] == "Vâng"
