import unittest

from blockchain_bridge import (
    label_to_verdict,
    normalize_confidence,
    PredictionPayload,
)


class BlockchainBridgeTests(unittest.TestCase):
    def test_weibo_label_mapping(self):
        self.assertFalse(label_to_verdict("weibo", 0))
        self.assertTrue(label_to_verdict("weibo", 1))

    def test_gossip_label_mapping(self):
        self.assertTrue(label_to_verdict("gossip", 0))
        self.assertFalse(label_to_verdict("gossip", 1))

    def test_confidence_is_clipped(self):
        self.assertEqual(normalize_confidence(-0.1), 0.0)
        self.assertEqual(normalize_confidence(1.8), 1.0)

    def test_prediction_payload_maps_to_api_shape(self):
        payload = PredictionPayload(
            dataset="gossip",
            image_path="/data/news.png",
            predicted_label=1,
            confidence=1.2,
            source="mmfn-eval",
        )

        self.assertEqual(
            payload.to_api_payload(),
            {
                "image_path": "/data/news.png",
                "verdict": False,
                "confidence": 1.0,
                "source": "mmfn-eval",
            },
        )
