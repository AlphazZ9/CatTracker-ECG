# -*- coding: utf-8 -*-
"""The subject number and sex read from a recording's file name."""
from __future__ import annotations

import pytest

from ecg.io.subject_name import subject_from_filename


@pytest.mark.parametrize("name, expected", [
    # The user's own names.
    ("Stim BNST - 1146 F.smr", ("1146", "F")),
    ("Baseline_STIM_BNST_206.mat", ("206", None)),
    ("Exp4_1146 F_2026-11-19.smr", ("1146", "F")),
    ("M_206.mat", ("206", "M")),
    ("1146F.mat", ("1146", "F")),
    ("F1146.smr", ("1146", "F")),
    # The number next to M / F wins, before it first.
    ("BNST_206_M_2026-11-19.mat", ("206", "M")),
    ("rec 12 F 3.mat", ("12", "F")),
    ("1146-F-20261119.mat", ("1146", "F")),
    ("206 (F).mat", ("206", "F")),
    ("2026-11-19 F 206.mat", ("206", "F")),       # the date before F is skipped
    # Letters of words are never a sex.
    ("Mouse12.mat", ("12", None)),
    ("FIELD 33.smr", ("33", None)),
    ("Femelle_206.mat", ("206", None)),
    ("MF_206.mat", ("206", None)),
    # Dates and times are not subjects.
    ("1600_segments_20261119_173903.pzfx", ("1600", None)),
    ("19112026_206.mat", ("206", None)),
    ("206_2026-11-19_17-39-03.mat", ("206", None)),
    ("206_17-39-03.mat", ("206", None)),
    ("206_19.11.2026.mat", ("206", None)),
    ("Rec_20261119.mat", (None, None)),
    # The last number of 2+ digits otherwise; single digits are not subjects.
    ("Exp4_cage 7_mouse 31_day 12.mat", ("12", None)),
    ("Exp4.mat", (None, None)),
    # No number at all.
    ("baseline.mat", (None, None)),
    ("F.mat", (None, "F")),
    ("", (None, None)),
    (None, (None, None)),
])
def test_examples(name, expected):
    assert subject_from_filename(name) == expected


def test_the_folder_and_the_extension_are_ignored():
    assert subject_from_filename(r"C:\data\exp 99\Stim 1146 F.smr") == ("1146", "F")
    assert subject_from_filename("/data/exp 99/Stim 1146 F.smr") == ("1146", "F")
    assert subject_from_filename("Stim 1146.v2.mat") == ("1146", None)   # 'v2' is one digit
    assert subject_from_filename("rec 206.s2rx") == ("206", None)


def test_lower_case_letters_count_only_when_separated():
    assert subject_from_filename("1146 f.mat") == ("1146", "F")
    assert subject_from_filename("m_206.mat") == ("206", "M")
    # '10m' is ten minutes, not a male.
    assert subject_from_filename("Stim 10m 1146.mat") == ("1146", None)
    assert subject_from_filename("206m.mat") == ("206", None)


def test_two_different_sexes_far_from_any_number_give_none():
    assert subject_from_filename("M vs F_stim_1146.mat") == ("1146", None)
    assert subject_from_filename("F_Stim_1146_M.mat") == ("1146", "M")   # M is next to it
    assert subject_from_filename("F_stim_F_1146x.mat") == ("1146", "F")


def test_leading_zeros_are_kept():
    assert subject_from_filename("M_007.mat") == ("007", "M")
    assert subject_from_filename("mouse_042.mat") == ("042", None)


@pytest.mark.parametrize("name, expected", [
    # A number glued to a word on its far side is a channel or an
    # experiment, not the animal next to the sex letter.
    ("STIM_BNST_206_Ch6_F.mat", ("206", "F")),
    ("Baseline_STIM_BNST_206_Ch6 F.mat", ("206", "F")),
    ("Exp4_F_2026-11-19 rec.mat", (None, "F")),
    ("Exp4_F_1146.mat", ("1146", "F")),
    ("Ch1 F 206.mat", ("206", "F")),
    ("206_Ch1_F.mat", ("206", "F")),
    ("M206F.mat", ("206", "M")),              # the far side is a sex letter
    # A single digit set apart from the letter is not the subject.
    ("206 F 3.mat", ("206", "F")),
    ("F 3_206.mat", ("206", "F")),
])
def test_numbers_glued_to_words_are_not_next_to_the_sex(name, expected):
    assert subject_from_filename(name) == expected


@pytest.mark.parametrize("name, expected", [
    # A lower case m after a number is minutes, with or without a space.
    ("206 10 m.mat", ("206", None)),
    ("206_5_m_post.mat", ("206", None)),
    ("206 post 30 m.mat", ("206", None)),
    ("206 30_m.mat", ("206", None)),
    # Upper case M stays a male.
    ("206 M.mat", ("206", "M")),
])
def test_lower_case_m_after_a_number_is_minutes(name, expected):
    assert subject_from_filename(name) == expected


@pytest.mark.parametrize("name, expected", [
    ("206 100Hz.mat", ("206", None)),
    ("Baseline 10mA 206.mat", ("206", None)),
    ("206 stim 20 Hz.mat", ("206", None)),
    ("206_stim_50ms.mat", ("206", None)),
    # A number with a unit only when it is all the name has.
    ("Mouse 12 s.mat", ("12", None)),
])
def test_numbers_with_a_unit_come_last(name, expected):
    assert subject_from_filename(name) == expected
