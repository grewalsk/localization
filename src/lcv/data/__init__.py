"""Datasets and gold-span→token mapping.

Synthetic NIAH (§10, exact gold spans by construction), HotpotQA distractor
(sentence-level supporting facts), LongBench subsets, TriviaQA easy-contrast.
Gold-span token mapping is a named bug surface; see gate 11.1e.

Two probing/detection sets feed the non-attribution signals and are held out from
the §10 QA evaluation sets: TruthfulQA labeled true/false statements for the ITI
truth probe (:mod:`lcv.data.iti`, §5.4) and LongMemEval long-context QA with
gold-document annotations for QRHead detection (:mod:`lcv.data.qrhead_qa`, §5.3).
"""
