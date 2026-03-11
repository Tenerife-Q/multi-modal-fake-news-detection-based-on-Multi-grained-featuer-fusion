import hashlib
import tempfile
import unittest

from blockchain_bridge import (
    build_integration_bundle,
    label_to_verdict,
    normalize_confidence,
    sha256_text,
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

    def test_bundle_contains_register_model_when_checkpoint_exists(self):
        with tempfile.NamedTemporaryFile("wb", delete=True) as handle:
            handle.write(b"fake checkpoint")
            handle.flush()

            bundle = build_integration_bundle(
                dataset="weibo",
                image_path="/data/news.png",
                predicted_label=1,
                confidence=0.97321,
                checkpoint_path=handle.name,
                description="MMFN unit-test checkpoint",
                external_knowledge="新华社报道摘要",
                activated_prompts=[1, 5],
            )

        self.assertIn("register_model", bundle)
        self.assertEqual(bundle["prove"]["verdict"], True)
        self.assertEqual(bundle["prove"]["confidence"], 0.97321)
        self.assertEqual(
            bundle["local_record"]["external_knowledge_hash"],
            sha256_text("新华社报道摘要"),
        )
        self.assertEqual(
            bundle["register_model"]["hash"],
            hashlib.sha256(b"fake checkpoint").hexdigest(),
        )
