from pathlib import Path

from scripts.regression_suite import load_suite, summarize, quality_failures


def test_portable_fixture_has_every_case_and_preserves_clean_reference_identity():
    root = Path(__file__).resolve().parents[1]
    suite = load_suite(root/'tests/fixtures/regression118/manifest.json')
    cases = suite['cases']
    assert len(cases) == 118
    assert len({c['seed'] for c in cases}) == 118
    assert all(len(c['expected_input_geometry_sha256']) == 64 for c in cases)
    assert {c['bag'] for c in cases} == {'roundT_doubleT', 'roundT_pressureGate_roundT'}
    assert 'not whole-bag' in suite['note']
    baseline = summarize([c['baseline'] for c in cases])
    assert baseline['positive_cases'] == baseline['detected'] == 112
    assert baseline['negative_cases'] == 6
    assert baseline['post_confirmation_missed_frames'] == 0


def test_summary_does_not_count_negative_controls_as_detected_objects():
    positive = dict(negative_control=False, detected=True, eligible_missed_frames=2,
                    post_confirmation_missed_eligible_frames=0, other_alarm_frames=1)
    negative = dict(positive, negative_control=True, detected=True, other_alarm_frames=3)
    result = summarize([positive, negative])
    assert result['detected'] == result['positive_cases'] == 1
    assert result['extra_alarm_frames'] == 1 and result['negative_alarm_frames'] == 3


def test_quality_gate_catches_individual_regression_despite_aggregate_improvement():
    good=dict(negative_control=False,detected=True,post_confirmation_missed_eligible_frames=0,other_alarm_frames=0)
    cases=[dict(seed=1,baseline=dict(good,other_alarm_frames=5)),dict(seed=2,baseline=good)]
    runs=[good,dict(good,detected=False,other_alarm_frames=1)]
    assert quality_failures(runs,cases)==[dict(seed=2,reasons=['lost_target','other_alarm_frames'])]
