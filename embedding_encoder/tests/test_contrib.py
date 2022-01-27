from sklearn.utils.estimator_checks import check_estimator

from embedding_encoder import EmbeddingEncoder


def test_contrib():
    ee = EmbeddingEncoder(task="classification")
    check_estimator(ee)
