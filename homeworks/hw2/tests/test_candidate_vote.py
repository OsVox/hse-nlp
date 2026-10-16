import unittest

from homeworks.hw2.candidate_vote import select_indices


def ok(x):
    return {'status': 'ok', 'value': x}


class CandidateVoteTests(unittest.TestCase):
    def test_strict_majority_can_replace_valid_top(self):
        self.assertEqual(select_indices([ok(9), ok(3), ok(3), ok(3)]),
                         {'top_beam': 0, 'first_executable': 0, 'strict_majority': 1})

    def test_tie_keeps_first_executable(self):
        self.assertEqual(select_indices([ok(9), ok(3), ok(3), ok(9)])['strict_majority'], 0)

    def test_failures_count_against_majority(self):
        r = select_indices([{'status': 'timeout'}, ok(9), ok(3), ok(3)])
        self.assertEqual(r['first_executable'], 1)
        self.assertEqual(r['strict_majority'], 1)

    def test_all_invalid_keep_exact_top(self):
        self.assertEqual(select_indices([{'status': 'timeout'}, ok(float('nan')),
                                         ok(float('inf')), ok(True)])['strict_majority'], 0)

    def test_numeric_roundoff(self):
        self.assertEqual(select_indices([ok(7), ok(0.3), ok(0.1 + 0.2), ok(0.3)])
                         ['strict_majority'], 1)

    def test_nontransitive_tolerance_does_not_form_false_majority(self):
        self.assertEqual(select_indices([ok(9), ok(0), ok(0.9e-6), ok(1.8e-6)])
                         ['strict_majority'], 0)

    def test_empty_is_rejected(self):
        with self.assertRaises(ValueError):
            select_indices([])


if __name__ == '__main__':
    unittest.main()
