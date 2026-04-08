import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
from scipy.stats import pearsonr
from mlr_utils.feature_selection import build_cv_splits

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    _RDKit_AVAILABLE = True
except ImportError:
    _RDKit_AVAILABLE = False


def regression_metrics(y_true, y_pred):
    # supports pandas Series or arrays
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    if len(y_true) > 1:
        r, p = pearsonr(y_true, y_pred)
    else:
        r, p = np.nan, np.nan
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_r': r,
        'pearson_p': p,
    }


def parity_plot(y_true, y_pred, ax=None, title='Parity plot', color='tab:blue', target_name=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    spread = mx - mn
    buffer = spread * 0.1 if spread > 0 else max(1.0, abs(mx) * 0.1)
    lower = mn - buffer
    upper = mx + buffer

    ax.scatter(y_true, y_pred, alpha=0.6, color=color, edgecolor='k', linewidth=0.2)
    ax.plot([lower, upper], [lower, upper], 'k--', lw=1)
    xlabel = f"Actual ({target_name})" if target_name else 'Actual'
    ylabel = f"Predicted ({target_name})" if target_name else 'Predicted'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.grid(True, linestyle=':', alpha=0.5)

    metrics = regression_metrics(y_true, y_pred)
    text = (
        f"R2={metrics['r2']:.3f}\n"
        f"RMSE={metrics['rmse']:.3f}\n"
        f"MAE={metrics['mae']:.3f}\n"
        f"Pearson r={metrics['pearson_r']:.3f}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return fig, ax, metrics


def plot_result_parity(
    merged_df,
    target_col,
    features,
    model,
    cv=5,
    split_method='cv',
    test_size=0.2,
    random_state=None,
    extreme='high',
    ax=None,
    print_ml_equation=True,
    stats_location='right',  # 'left', 'right', or 'bottom'
):
    from sklearn.model_selection import cross_val_predict, cross_validate

    X = merged_df[features]
    y = merged_df[target_col]

    splits = build_cv_splits(X, y, method=split_method, n_splits=cv, test_size=test_size, random_state=random_state, extreme=extreme)

    y_pred = cross_val_predict(model, X, y, cv=splits, n_jobs=-1)
    cv_res = cross_validate(model, X, y, cv=splits, scoring='r2', return_train_score=True, n_jobs=-1)
    train_r2 = np.mean(cv_res['train_score'])
    val_r2 = np.mean(cv_res['test_score'])

    fig, ax, metrics = parity_plot(
        y,
        y_pred,
        ax=ax,
        title=f'Parity plot ({split_method})',
        color='tab:blue',
        target_name=target_col,
    )

    stats_text = (
        f"CV method: {split_method}\n"
        f"Train R2={train_r2:.3f}\n"
        f"Val R2={val_r2:.3f}\n"
        f"R2={metrics['r2']:.3f}\n"
        f"RMSE={metrics['rmse']:.3f}\n"
        f"MAE={metrics['mae']:.3f}\n"
        f"Pearson r={metrics['pearson_r']:.3f}"
    )
    if stats_location == 'right':
        text_x, text_y, va, ha = 1.02, 0.5, 'center', 'left'
    elif stats_location == 'bottom':
        text_x, text_y, va, ha = 0.5, -0.3, 'top', 'center'
    else:  # left
        text_x, text_y, va, ha = -0.25, 0.5, 'center', 'left'

    ax.text(
        text_x,
        text_y,
        stats_text,
        transform=ax.transAxes,
        va=va,
        ha=ha,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    if print_ml_equation:
        model = clone(model)
        model.fit(X, y)
        coefs = model.coef_
        intercept = model.intercept_
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

    return fig, ax, metrics, {'train_r2': train_r2, 'val_r2': val_r2}


def _inchi_to_smiles(df, inchi_col='INCHI', smiles_col='SMILES'):
    if smiles_col in df.columns:
        return df[smiles_col].astype(str).tolist()
    if inchi_col not in df.columns:
        raise KeyError(f"INCHI column '{inchi_col}' not found in dataframe")
    # attempt to resolve smiles via RDKit InChiToSmiles for each inchi
    if not _RDKit_AVAILABLE:
        return [''] * len(df)
    smiles = []
    for ich in df[inchi_col].astype(str):
        try:
            mol = Chem.MolFromInchi(ich)
            smi = Chem.MolToSmiles(mol) if mol is not None else ''
        except Exception:
            smi = ''
        smiles.append(smi)
    return smiles


def _smiles_to_data_uri(smiles):
    if not _RDKit_AVAILABLE or not smiles:
        return [''] * len(smiles)
    uri_list = []
    for smi in smiles:
        if not smi:
            uri_list.append('')
            continue
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                uri_list.append('')
                continue
            img = Draw.MolToImage(m, size=(200, 200))
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            uri_list.append(f"data:image/png;base64,{b64}")
        except Exception:
            uri_list.append('')
    return uri_list


def bokeh_parity_with_tooltips(merged_df, target_col, features, model, cv=5, split_method='cv', test_size=0.2, random_state=None, extreme='high', inchi_col='INCHI'):
    from sklearn.model_selection import cross_val_predict
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, HoverTool

    if inchi_col not in merged_df.columns:
        raise KeyError(f"INCHI column '{inchi_col}' not found in dataframe")

    X = merged_df[features]
    y = merged_df[target_col]

    splits = build_cv_splits(X, y, method=split_method, n_splits=cv, test_size=test_size, random_state=random_state, extreme=extreme)
    y_pred = cross_val_predict(model, X, y, cv=splits, n_jobs=-1)

    actual = np.asarray(y).ravel()
    pred = np.asarray(y_pred).ravel()

    mn = min(np.min(actual), np.min(pred))
    mx = max(np.max(actual), np.max(pred))
    spread = mx - mn
    buffer = spread * 0.1 if spread > 0 else max(1.0, abs(mx) * 0.1)
    lower = mn - buffer
    upper = mx + buffer

    smiles = _inchi_to_smiles(merged_df, inchi_col=inchi_col)
    mol_images = _smiles_to_data_uri(smiles)

    data = {
        'actual': actual,
        'predicted': pred,
        'inchi': merged_df[inchi_col].astype(str).tolist(),
        'smiles': smiles,
        'mol_img': mol_images,
    }
    for f in features:
        data[f] = merged_df[f].values

    source = ColumnDataSource(data=data)

    p = figure(
        title=f"Bokeh parity plot ({split_method})",
        width=750,
        height=750,
        x_axis_label=f"Actual ({target_col})",
        y_axis_label=f"Predicted ({target_col})",
        x_range=(lower, upper),
        y_range=(lower, upper),
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    p.line([lower, upper], [lower, upper], line_dash='dashed', line_width=1, color='black')
    p.circle('actual', 'predicted', source=source, size=8, alpha=0.75, color='navy')

    tooltips = [
        ("INCHI", "@inchi"),
        ("SMILES", "@smiles"),
        ("Actual", "@actual"),
        ("Predicted", "@predicted"),
    ] + [(f, f"@{f}") for f in features] + [
        ("Structure", "<img src='@mol_img' width='120' height='120'>")
    ]

    p.add_tools(HoverTool(tooltips=tooltips, mode='mouse'))

    show(p)
    return p

