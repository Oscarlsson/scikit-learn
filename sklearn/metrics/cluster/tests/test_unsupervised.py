import numpy as np
from scipy.sparse import csr_matrix

from .... import datasets
from ..unsupervised import silhouette_score, _stratified_sampling
from ... import pairwise_distances
from nose.tools import assert_false, assert_almost_equal, assert_equal


def test_silhouette():
    """Tests the Silhouette Coefficient. """
    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    D = pairwise_distances(X, metric='euclidean')
    # Given that the actual labels are used, we can assume that S would be
    # positive.
    silhouette = silhouette_score(D, y, metric='precomputed')
    assert(silhouette > 0)
    # Test without calculating D
    silhouette_metric = silhouette_score(X, y, metric='euclidean')
    assert_almost_equal(silhouette, silhouette_metric)
    # Test with sampling
    silhouette = silhouette_score(D, y, metric='precomputed',
                                  percentage=0.5,
                                  random_state=0)
    silhouette_metric = silhouette_score(X, y, metric='euclidean',
                                         percentage=0.5,
                                         random_state=0)
    assert(silhouette > 0)
    assert(silhouette_metric > 0)
    assert_almost_equal(silhouette_metric, silhouette)
    # Test with sparse X
    X_sparse = csr_matrix(X)
    D = pairwise_distances(X_sparse, metric='euclidean')
    silhouette = silhouette_score(D, y, metric='precomputed')
    assert(silhouette > 0)


def test_no_nan():
    """Assert Silhouette Coefficient != nan when there is 1 sample in a class.

        This tests for the condition that caused issue 960.
    """
    # Note that there is only one sample in cluster 0. This used to cause the
    # silhouette_score to return nan (see bug #960).
    labels = np.array([1, 0, 1, 1, 1])
    # The distance matrix doesn't actually matter.
    D = np.random.RandomState(0).rand(len(labels), len(labels))
    silhouette = silhouette_score(D, labels, metric='precomputed')
    assert_false(np.isnan(silhouette))


def test_stratif_size():
    """ Tests stratified sampling. Tests that the size is approximately correct
    """
    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    D = pairwise_distances(X, metric='euclidean')

    for perc in np.arange(0, 1.1, 0.1):
        indices = _stratified_sampling(D, y, perc, None)
        assert_almost_equal(len(indices), len(y)*perc)


def test_stratsampl_label():
    """ Tests stratified sampling. Tests such that all labels are covered in subsample 
    """
    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    D = pairwise_distances(X, metric='euclidean')

    for perc in np.arange(0.1, 1.1, 0.1):
        samplesize = perc
        indices = _stratified_sampling(X, y, samplesize, None)
        sampledlabels = np.unique(y[indices])
        assert_equal(len(sampledlabels), len(np.unique(y)))

        indices = _stratified_sampling(D, y, samplesize, None)
        sampledlabels = np.unique(y[indices])
        assert_equal(len(sampledlabels), len(np.unique(y)))
