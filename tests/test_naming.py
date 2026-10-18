import unittest
from naming import apply_name_scores
from core import verification_labels


class NamingRules(unittest.TestCase):
    def test_strong_species_estimate_keeps_original_detector_score_separate(self):
        item = {"label": "bear", "model_label": "bear", "confidence": .94}
        scores = [{"label": "gorilla", "similarity": .326}, {"label": "chimpanzee", "similarity": .304}]
        out = apply_name_scores(item, scores)
        self.assertEqual(out["label"], "gorilla")
        self.assertEqual(out["model_label"], "bear")
        self.assertEqual(out["confidence"], .94)
        self.assertFalse(out["verified"])
        self.assertEqual(verification_labels([out])[0]["label"], "gorilla")

    def test_ambiguous_species_is_not_forced(self):
        out = apply_name_scores({"label": "bear", "confidence": .9}, [{"label": "gorilla", "similarity": .30}, {"label": "chimpanzee", "similarity": .295}])
        self.assertEqual(out["label"], "animal (unverified)")

    def test_matching_names_remain_estimates_with_close_alternatives_visible(self):
        out = apply_name_scores({"label": "gorilla", "confidence": .69}, [{"label": "gorilla", "similarity": .28}, {"label": "monkey", "similarity": .273}])
        self.assertEqual(out["label"], "gorilla")
        self.assertFalse(out["verified"])
        self.assertIn("close alternatives", out["naming_status"])

    def test_namer_does_not_silently_remove_a_possible_person(self):
        out = apply_name_scores({"label": "person", "confidence": .8}, [{"label": "gorilla", "similarity": .32}, {"label": "person", "similarity": .29}])
        self.assertEqual(out["label"], "person")
        self.assertTrue(out["naming_status"].startswith("Conflicting"))
