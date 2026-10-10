import random

from tinygpt.data_io import EOT, iter_stories
from tinygpt.sft_data import build_prompt, check#, extract_constraint


def test_iter_stories_from_txt(tmp_path):
    (tmp_path / "train.txt").write_text(f"Story one.\nSecond line.\n{EOT}\nStory two.\n{EOT}\n\n{EOT}\nlast one")
    got = list(iter_stories(str(tmp_path), "train"))
    assert got == ["Story one.\nSecond line.", "Story two.", "last one"]
    assert len(list(iter_stories(str(tmp_path), "train", max_n=2))) == 2


# def test_constraints_are_satisfiable_and_checkable():
#     story = 'Tom had a big red ball. "Look at my ball!" he said. The garden was beautiful and bright.'
#     c = extract_constraints(story, random.Random(0))
#     assert c is not None and c["dialogue"] is True
#     res = check(story, c)
#     assert res["success"] and res["all_words"] and res["dialogue_ok"]
#     prompt = build_prompt(c)
#     assert prompt.endswith("Story:\n") and "Dialogue: yes" in prompt


def test_check_detects_failures():
    c = {"words": ["garden", "ball"], "dialogue": False}
    assert check("The garden had a ball.", c)["success"]
    assert not check("The garden was empty.", c)["all_words"]
    assert not check('She said "hi" in the garden with a ball.', c)["dialogue_ok"]
    assert not check("The gardener played ball.", c)["all_words"]  # whole-word match only

def test_iter_stories_from_txt(tmp_path):
    (tmp_path / "train.txt").write_text(f"Story one.\nSecond line.\n{EOT}\nStory two.\n{EOT}\n\n{EOT}\nlast one")
    got = list(iter_stories(str(tmp_path), "train"))
    assert got == ["Story one.\nSecond line.", "Story two.", "last one"]   # no accumulation, no empty story
    assert len(list(iter_stories(str(tmp_path), "train", max_n=2))) == 2