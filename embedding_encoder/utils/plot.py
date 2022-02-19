import pandas as pd


def plot_embeddings(embedding_encoder, variable: str, model: str = "pca"):
    """Plot embeddings for a variable by passing a fitted EmbeddingEncoder and reducing to 2D.

    Parameters
    ----------
    embedding_encoder : EmbeddingEncoder
        Fitted transformer.
    variable :
        Variable to plot. Please note that scikit-learn's Pipeline might strip column names.
    model : str, optional
        Dimensionality reduction model. Either "tsne" or "pca". Default "pca".

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Seaborn scatterplot (Matplotlib axes)

    Raises
    ------
    ValueError
        If selected variable has less than 3 unique values.
    ValueError
        If selected model is not "tsne" or "pca".
    ImportError
        If seaborn is not installed.
    """
    if embedding_encoder._embeddings_mapping[variable].shape[0] < 3:
        raise ValueError("Nothing to plot when variable has less than 3 unique values.")
    dimensions = 2
    if model not in ["tsne", "pca"]:
        raise ValueError("model must be either 'tsne' or 'pca'.")
    try:
        import seaborn as sns
        sns.set(rc={"figure.figsize": (8, 6), "figure.dpi": 100})
        sns.set_palette("viridis")
    except ImportError:
        raise ImportError("Plotting requires seaborn.")
    if model == "tsne":
        from sklearn.manifold import TSNE

        model = TSNE(init="pca", n_components=dimensions, learning_rate="auto")
    else:
        from sklearn.decomposition import PCA

        model = PCA(n_components=dimensions)

    embeddings = embedding_encoder._embeddings_mapping[variable]
    variable_position = embedding_encoder._categorical_vars.index(variable)
    original_classes = embedding_encoder._ordinal_encoder.categories_[variable_position]
    original_index = ["OOV"] + list(original_classes)

    reduced = model.fit_transform(embeddings)
    reduced = pd.DataFrame(
        reduced,
        index=original_index,
        columns=[f"Component {i}" for i in range(dimensions)],
    ).rename_axis("Classes").reset_index()
    plot = sns.scatterplot(data=reduced, x="Component 0", y="Component 1", hue="Classes", s=100)
    plot.set_title(f"{model.__class__.__name__} embeddings projection for variable '{variable}'")
    return plot
