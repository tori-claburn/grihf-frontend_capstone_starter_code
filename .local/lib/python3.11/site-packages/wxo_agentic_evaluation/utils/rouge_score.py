def lcs_length(x, y):
    """Compute the length of the Longest Common Subsequence (LCS)."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def rouge_l_recall(prediction, reference):
    """Compute ROUGE-L recall. No stemming."""
    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    lcs = lcs_length(pred_tokens, ref_tokens)
    if len(pred_tokens) == 0:
        return 0.0

    return lcs / len(pred_tokens)
