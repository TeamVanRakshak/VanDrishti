import unittest

from evaluation import eval_set
from evaluation.metrics import iou, score_image, summarize


def truth(category, box, species=None):
    return {"category": category, "species": species, "box": box}


def prediction(category, box, label=None, confidence=.9):
    return {"category": category, "label": label or category, "confidence": confidence, "box": box}


class Scoring(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)

    def test_match_miss_and_false_positive(self):
        result = score_image([truth("person", [0, 0, 10, 10]), truth("vehicle", [50, 50, 90, 90])],
                             [], [prediction("person", [1, 1, 10, 10]), prediction("person", [200, 200, 220, 220])])
        m = summarize([result])
        self.assertEqual((m["person"]["tp"], m["person"]["fp"], m["person"]["fn"]), (1, 1, 0))
        self.assertEqual((m["vehicle"]["recall"], m["vehicle"]["precision"]), (0, None))
        self.assertEqual(m["person"]["precision"], .5)

    def test_one_prediction_matches_one_object(self):
        result = score_image([truth("animal", [0, 0, 10, 10])], [],
                             [prediction("animal", [0, 0, 10, 10], confidence=.9), prediction("animal", [0, 0, 10, 9], confidence=.5)])
        m = summarize([result])["animal"]
        self.assertEqual((m["tp"], m["fp"]), (1, 1))

    def test_wrong_category_counts_as_confusion(self):
        result = score_image([truth("animal", [0, 0, 10, 10])], [], [prediction("person", [0, 0, 10, 10])])
        m = summarize([result])
        self.assertEqual((m["animal"]["fn"], m["person"]["fp"]), (1, 1))
        self.assertEqual(m["confusions"], {"animal predicted as person": 1})

    def test_ignore_regions_and_irrelevant_labels(self):
        result = score_image([], [[0, 0, 100, 100]], [prediction("vehicle", [10, 10, 20, 20]), prediction("other", [200, 200, 210, 210])])
        self.assertEqual(result["false_positives"], [])

    def test_species_accuracy_separates_abstentions(self):
        results = [score_image([truth("animal", [0, 0, 10, 10], "zebra"), truth("animal", [20, 20, 30, 30], "zebra"),
                                truth("animal", [40, 40, 50, 50], "giraffe"), truth("animal", [60, 60, 70, 70])], [],
                               [prediction("animal", [0, 0, 10, 10], "zebra"), prediction("animal", [20, 20, 30, 30], "animal (unverified)"),
                                prediction("animal", [40, 40, 50, 50], "horse"), prediction("animal", [60, 60, 70, 70], "bear")])]
        s = summarize(results)["species"]
        self.assertEqual((s["matched_with_known_species"], s["correct"], s["abstained"], s["wrong"]), (3, 1, 1, 1))
        self.assertEqual(s["accuracy_when_named"], .5)

    def test_recall_by_size(self):
        result = score_image([truth("person", [0, 0, 10, 10]), truth("person", [0, 0, 200, 200])], [], [prediction("person", [0, 0, 200, 200])])
        sizes = summarize([result])["person"]["recall_by_size"]
        self.assertEqual((sizes["small"], sizes["medium"], sizes["large"]), (0, None, 1))


class EvaluationSet(unittest.TestCase):
    def test_manifest_matches_quotas_and_boxes_are_valid(self):
        manifest = eval_set.load_manifest()
        for condition, sources in manifest["quotas"].items():
            for source, quota in sources.items():
                count = sum(1 for e in manifest["images"] if e["condition"] == condition and e["source"] == source)
                self.assertEqual(count, quota, f"{condition}/{source}")
        self.assertEqual(len({e["id"] for e in manifest["images"]}), len(manifest["images"]))
        for entry in manifest["images"]:
            self.assertTrue(entry["boxes"] or entry["ignore"], entry["id"])
            self.assertEqual(len(entry["sha256"]), 64)
            for b in entry["boxes"]:
                x1, y1, x2, y2 = b["box"]
                self.assertIn(b["category"], ("person", "vehicle", "animal"))
                self.assertTrue(0 <= x1 < x2 <= entry["width"] and 0 <= y1 < y2 <= entry["height"], entry["id"])
                if b["category"] != "animal":
                    self.assertIsNone(b["species"])

    def test_every_category_is_represented_in_every_condition(self):
        images = eval_set.load_manifest()["images"]
        for condition in ("day", "night", "aerial"):
            categories = {b["category"] for e in images if e["condition"] == condition for b in e["boxes"]}
            expected = {"person", "animal"} | ({"vehicle"} if condition != "night" else set())
            self.assertLessEqual(expected, categories, condition)


if __name__ == "__main__":
    unittest.main()
