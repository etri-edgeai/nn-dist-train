import torch

__all__ = ["gram_linear", "gram_rbf", "center_gram", "cka", "feature_space_linear_cka"]


def gram_linear(x):
    return torch.matmul(x, x.T)


def gram_rbf(x, threshold=1.0):
    dot_products = torch.matmul(x, x.T)
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = torch.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    if not torch.allclose(gram, gram.T):
        raise ValueError("Input must be a symmetric matrix.")
    gram = gram.detach()

    if unbiased:
        n = gram.shape[0]
        torch.fill_diagonal(gram, 0)
        means = torch.sum(gram, 0, dtype=torch.float64) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        torch.fill_diagonal(gram, 0)
    else:
        means = torch.mean(gram, 0, dtype=torch.float64)
        means -= torch.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = torch.matmul(torch.flatten(gram_x), torch.flatten(gram_y).T)

    normalization_x = torch.linalg.norm(gram_x)
    normalization_y = torch.linalg.norm(gram_y)

    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n
):
    return (
        xty
        - n / (n - 2.0) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2))
    )


def feature_space_linear_cka(features_x, features_y, debiased=False):
    features_x = features_x - torch.mean(features_x, 0, keepdims=True)
    features_y = features_y - torch.mean(features_y, 0, keepdims=True)

    dot_product_similarity = torch.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = torch.linalg.norm(features_x.T.dot(features_x))
    normalization_y = torch.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to torch.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = torch.einsum("ij,ij->i", features_x, features_x)
        sum_squared_rows_y = torch.einsum("ij,ij->i", features_y, features_y)
        squared_norm_x = torch.sum(sum_squared_rows_x)
        squared_norm_y = torch.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity,
            sum_squared_rows_x,
            sum_squared_rows_y,
            squared_norm_x,
            squared_norm_y,
            n,
        )
        normalization_x = torch.sqrt(
            _debiased_dot_product_similarity_helper(
                normalization_x ** 2,
                sum_squared_rows_x,
                sum_squared_rows_x,
                squared_norm_x,
                squared_norm_x,
                n,
            )
        )
        normalization_y = torch.sqrt(
            _debiased_dot_product_similarity_helper(
                normalization_y ** 2,
                sum_squared_rows_y,
                sum_squared_rows_y,
                squared_norm_y,
                squared_norm_y,
                n,
            )
        )

    return dot_product_similarity / (normalization_x * normalization_y)
