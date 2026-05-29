from __future__ import annotations

import logging
from transformers import BertForSequenceClassification

logger = logging.getLogger(__name__)

# =============================================================================
# SECTION 1 — Model factory
# =============================================================================
def build_model(
    model_path: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int]
) -> BertForSequenceClassification:
    """
    Load BertForSequenceClassification from a local checkpoint.
    Sets id2label and label2id on the model config so the saved model
    is self-describing — no external label mapping file needed at inference.

    Parameters
    ----------
    model_path : str
        Local path to BERT weights  e.g. `models/base-models/bert-base-uncased`
        or a fine-tuned checkpoint  e.g. `models/bert-bbc-finetuned`.
    num_labels : int
        Number of output classes  e.g. `5`.
    id2label : dict[int, str]
        Mapping from label index to class name  e.g. `{0: "business", ...}`.
    label2id : dict[str, int]
        Mapping from class name to label index  e.g. `{"business": 0, ...}`.
    
    Returns
    -------
    BertForSequenceClassification
        Model ready for `.to(device)` and training or inference.
    """

    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        local_files_only=True,
    )

    model.config.id2label = id2label
    model.config.label2id = label2id

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Model loaded from %s", model_path)
    logger.info("Total params    : %d", total_params)
    logger.info("Trainable params: %d", trainable_params)

    return model



if __name__ == "__main__":

    import hydra
    import torch
    from omegaconf import DictConfig
    from pathlib import Path

    from news_topic_classifier.config import ID2LABEL, LABEL2ID

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    @hydra.main(
        config_path=str(PROJECT_ROOT / "conf"),
        config_name="config",
        version_base=None,
    )
    def main(cfg: DictConfig) -> None:

        print("\n" + "=" * 80)
        print("BertClassifier — smoke test")
        print("=" * 80)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        model_path = str(
            PROJECT_ROOT / "models" / "base-models" / cfg.model.bert_base_model
        )

        model = build_model(
            model_path=model_path,
            num_labels=cfg.model.num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        model = model.to(device)
        print(f"Model device: {next(model.parameters()).device}")

        # ── forward pass with random input ──────────────────────────────
        batch_size = 2
        seq_len    = cfg.model.max_seq_length

        input_ids      = torch.randint(0, 30522, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        print(f"Input shape  : {input_ids.shape}")
        print(f"Logits shape : {logits.shape}")

        assert logits.shape == (batch_size, cfg.model.num_labels), (
            f"Expected ({batch_size}, {cfg.model.num_labels}), got {logits.shape}"
        )
        print("Smoke test passed.")

    main()
