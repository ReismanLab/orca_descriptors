import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.base import clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, ShuffleSplit, cross_validate
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline


def remove_highly_correlated_features(X, threshold=0.95):
    """Remove features with absolute correlation above threshold."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X.drop(columns=to_drop), to_drop


def report_mlr_equation(model, X, y, target_col):
    """Fit model and print MLR equation / coefficients."""
    model = clone(model)
    model.fit(X, y)
    coefs = np.asarray(model.coef_).ravel()
    intercept = float(model.intercept_)
    feature_names = list(X.columns)

    terms = [f"({coef:.4f} * {name})" for coef, name in zip(coefs, feature_names)]
    equation = " + ".join(terms)

    print('\nMLR equation:')
    print(f"{target_col} = {intercept:.4f} + {equation}")
    print('\nMLR parameters:')
    for name, coef in zip(feature_names, coefs):
        print(f"  {name}: {coef:.6f}")
    print(f"intercept: {intercept:.6f}")
    print(f"R2 (full data fit): {model.score(X, y):.4f}")

    return {
        'model': model,
        'intercept': intercept,
        'coefs': coefs,
        'features': feature_names,
        'equation': f"{target_col} = {intercept:.4f} + {equation}",
        'r2': model.score(X, y),
    }


def build_cv_splits(
    X,
    y,
    method='cv',
    n_splits=5,
    test_size=0.2,
    random_state=None,
    extreme='high',
):
    """Build cross-validation split generator based on selection method."""
    n_samples = len(y)
    indices = np.arange(n_samples)

    if method == 'cv':
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if method == 'random':
        return ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    if method == 'y_equidistant':
        ordered = np.argsort(y.values)
        folds = [ordered[i::n_splits] for i in range(n_splits)]
        splits = []
        for i in range(n_splits):
            test_idx = folds[i]
            train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
            splits.append((train_idx, test_idx))
        return splits

    if method == 'kennard_stone':
        test_count = max(1, int(np.round(test_size * n_samples)))
        if test_count >= n_samples:
            raise ValueError("test_size too large for Kennard-Stone split")

        dist = pairwise_distances(X.values, metric='euclidean')
        i0, j0 = np.unravel_index(np.nanargmax(dist), dist.shape)
        selected = [i0, j0]
        available = set(indices) - set(selected)

        while len(selected) < test_count and available:
            nearest = {
                idx: np.min(dist[idx, selected])
                for idx in available
            }
            next_idx = max(nearest, key=nearest.get)
            selected.append(next_idx)
            available.remove(next_idx)

        test_idx = np.array(selected, dtype=int)
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
        return [(train_idx, test_idx)]

    if method == 'extreme':
        # 80/20 extreme split by target values. default test = 20% highest values
        sort_idx = np.argsort(y.values)
        n_test = max(1, int(np.round(test_size * n_samples)))
        if extreme == 'high':
            test_idx = sort_idx[-n_test:]
        elif extreme == 'low':
            test_idx = sort_idx[:n_test]
        else:
            raise ValueError("extreme must be 'high' or 'low'")
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
        return [(train_idx, test_idx)]

    raise ValueError(f"Unknown split method: {method}")


def auto_feature_selection(
    df,
    target_col,
    key='SMILES',
    n_features_to_select=None,
    direction='forward',
    selection_mode='sfs',
    split_method='cv',
    n_splits=5,
    test_size=0.2,
    random_state=None,
    extreme='high',
    corr_threshold=None,
    exclude_cols=None,
    scoring='r2',
    model=None,
    max_combinations=5000,
    n_jobs=-1,
    print_ml_equation=False,
):
    """Select features from a merged descriptor/target dataframe.

    This function supports:
      - sequential forward/backward selection ('sfs') for one specified feature count
      - exhaustive search of combinations over user-specified sizes ('exhaustive')
      - multiple data split schemes for validation
      - pre-filtering of highly correlated features
      - explicit exclusion of chosen columns

    Parameters:
      df (pd.DataFrame): merged data, including key column and target.
      target_col (str): target variable column name.
      key (str): identity column name to keep but exclude from feature matrix.
      n_features_to_select (int or list[int] or None): subset sizes. If None, defaults to [1,2,3,4,5].
      direction (str): 'forward' or 'backward' (only used in SFS mode).
      selection_mode (str): 'sfs' or 'exhaustive'.
      split_method (str): 'cv', 'random', 'y_equidistant', 'kennard_stone', 'extreme'.
      n_splits (int): number of folds for CV-style splits.
      test_size (float): proportion for holdout splits in random/extreme/kennard_stone.
      random_state (int|None): random state for reproducibility.
      extreme (str): 'high' or 'low' for extreme split strategy.
      corr_threshold (float|None): if set, drop highly correlated features above threshold.
      exclude_cols (list[str]|None): columns to always omit from feature matrix.
      scoring (str): metric name for cross-validation (default 'r2').
      model (estimator|None): model object (default LinearRegression()).
      max_combinations (int): maximum combos in exhaustive mode to protect runtime.
      n_jobs (int): parallel jobs for sklearn calls.

    Returns:
      dict with keys:
        selected_features, selector, X_numeric, y, history, best, dropped_corr, excluded_cols.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")
    if key not in df.columns:
        raise KeyError(f"Key column '{key}' not found in dataframe")

    X = df.drop(columns=[key, target_col])
    y = df[target_col]

    # numeric columns only for regression-based filtering
    X_numeric = X.select_dtypes(include=[np.number])

    if exclude_cols is not None:
        cols_to_exclude = [c for c in exclude_cols if c in X_numeric.columns]
        X_numeric = X_numeric.drop(columns=cols_to_exclude)
    else:
        cols_to_exclude = []

    if corr_threshold is not None:
        X_numeric, dropped_corr = remove_highly_correlated_features(X_numeric, threshold=corr_threshold)
    else:
        dropped_corr = []

    if X_numeric.shape[1] == 0:
        raise ValueError("No numeric columns available for feature selection")

    if model is None:
        model = LinearRegression()

    if selection_mode not in ('sfs', 'exhaustive'):
        raise ValueError("selection_mode must be 'sfs' or 'exhaustive'")

    if n_features_to_select is None:
        n_features_to_select = [1, 2, 3, 4, 5]
    elif isinstance(n_features_to_select, int):
        n_features_to_select = [n_features_to_select]
    else:
        n_features_to_select = list(sorted(set(n_features_to_select)))

    cv_splits = build_cv_splits(
        X_numeric,
        y,
        method=split_method,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
        extreme=extreme,
    )

    if selection_mode == 'sfs':
        k = n_features_to_select[0]
        sfs = SequentialFeatureSelector(
            model,
            n_features_to_select=k,
            direction=direction,
            cv=cv_splits,
            n_jobs=n_jobs,
        )
        sfs.fit(X_numeric, y)
        selected = X_numeric.columns[sfs.get_support()].tolist()

        cv_res = cross_validate(
            model,
            X_numeric[selected],
            y,
            cv=cv_splits,
            scoring=scoring,
            return_train_score=True,
            n_jobs=n_jobs,
        )
        mean_train = np.mean(cv_res['train_score'])
        mean_test = np.mean(cv_res['test_score'])

        best_output = {
            'n_features': len(selected),
            'features': selected,
            'train_score': mean_train,
            'val_score': mean_test,
        }

        mlr_info = None
        if print_ml_equation:
            mlr_info = report_mlr_equation(model, X_numeric[selected], y, target_col)

        return {
            'selected_features': selected,
            'selector': sfs,
            'X_numeric': X_numeric,
            'y': y,
            'history': None,
            'best': best_output,
            'dropped_corr': dropped_corr,
            'excluded_cols': cols_to_exclude,
            'mlr_info': mlr_info,
        }

    # exhaustive mode
    best_output = None
    history = []
    allowed_features = X_numeric.columns.tolist()

    for k in n_features_to_select:
        if k < 1 or k > len(allowed_features):
            continue

        combos = list(combinations(allowed_features, k))
        if len(combos) > max_combinations:
            raise ValueError(
                f"Too many combos ({len(combos)}). Reduce n_features_to_select or set max_combinations higher."
            )

        for combo in combos:
            selected_df = X_numeric[list(combo)]
            cv_res = cross_validate(
                model,
                selected_df,
                y,
                cv=cv_splits,
                scoring=scoring,
                return_train_score=True,
                n_jobs=n_jobs,
            )

            mean_train = np.mean(cv_res['train_score'])
            mean_test = np.mean(cv_res['test_score'])

            record = {
                'n_features': k,
                'features': list(combo),
                'train_score': mean_train,
                'val_score': mean_test,
            }
            history.append(record)

            if best_output is None or mean_test > best_output['val_score']:
                best_output = record

    mlr_info = None
    if print_ml_equation and best_output:
        mlr_info = report_mlr_equation(model, X_numeric[best_output['features']], y, target_col)

    return {
        'selected_features': best_output['features'] if best_output else [],
        'selector': None,
        'X_numeric': X_numeric,
        'y': y,
        'history': history,
        'best': best_output,
        'dropped_corr': dropped_corr,
        'excluded_cols': cols_to_exclude,
        'mlr_info': mlr_info,
    }


def merge_descriptor_target_and_select(
    desc_path,
    target_path,
    target_col,
    key='SMILES',
    **sfs_kwargs,
):
    """Convenience wrapper: merge + automatic feature selection."""
    from make_mlr_search import merge_descriptor_target

    merged, encoding = merge_descriptor_target(desc_path, target_path, key=key)
    selection = auto_feature_selection(merged, target_col=target_col, key=key, **sfs_kwargs)
    return merged, encoding, selection
