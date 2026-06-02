import os
import pandas as pd
import numpy as np
from datetime import timedelta
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
import joblib
import pulp

CSV_PATH = "hybrid_manufacturing_categorical.csv"
OUT_DIR = "artifacts/v2"

def _ensure_column_aliases(df):
    """데이터프레임에 서로 다른 케이스/표기법으로 사용된 컬럼명이 혼재된 경우를 보정합니다.
    예: Scheduled_Start, Scheduled_start, scheduled_start 등 중 하나를 찾아서 나머지 표기법으로도 복제합니다.
    """
    if df is None:
        return df
    cols = list(df.columns)
    groups = {
        'scheduled_start': ['Scheduled_Start', 'Scheduled_start', 'scheduled_start'],
        'scheduled_end': ['Scheduled_End', 'Scheduled_end', 'scheduled_end'],
        'actual_start': ['Actual_Start', 'Actual_start', 'actual_start'],
        'actual_end': ['Actual_End', 'Actual_end', 'actual_end'],
        'processing_time': ['Processing_Time', 'Processing_time', 'processing_time'],
        'machine_availability': ['Machine_Availability', 'Machine_availability', 'machine_availability'],
        'job_status': ['Job_Status', 'Job_status', 'job_status'],
        'machine_id': ['Machine_ID', 'Machine_id', 'machine_id'],
        'job_id': ['Job_ID', 'Job_id', 'job_id']
    }
    for aliases in groups.values():
        found = None
        for a in aliases:
            for c in cols:
                if c.lower() == a.lower():
                    found = c
                    break
            if found:
                break
        if found:
            for a in aliases:
                if a not in df.columns:
                    df[a] = df[found]
    return df

def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def _to_datetime_series(series):
    return pd.to_datetime(series, errors='coerce')


def _span_minutes(start_series, end_series):
    start_series = _to_datetime_series(start_series)
    end_series = _to_datetime_series(end_series)
    valid = start_series.notna() & end_series.notna()
    if not valid.any():
        return np.nan
    return (end_series[valid].max() - start_series[valid].min()).total_seconds() / 60.0


def _safe_numeric(series, default=0.0):
    return pd.to_numeric(series, errors='coerce').fillna(default)


def _predict_abnormal_probabilities(df, model):
    frame = _ensure_column_aliases(df).copy()
    feature_frame = frame.copy()
    drop_like = [
        c for c in feature_frame.columns
        if c.lower().endswith("_id")
        or c.lower().startswith("job_")
        or c.lower().startswith("jobid")
        or c.lower() in {"is_abnormal", "job_status"}
    ]
    feature_frame = feature_frame.drop(columns=[c for c in drop_like if c in feature_frame.columns])

    try:
        proba = model.predict_proba(feature_frame)
        return np.asarray(proba)[:, 1]
    except Exception:
        preds = model.predict(feature_frame)
        return np.asarray(preds, dtype=float)


def _machine_relative_minutes(series, machine_start):
    values = _to_datetime_series(series)
    if pd.isna(machine_start):
        return pd.Series(np.nan, index=values.index)
    return (values - machine_start).dt.total_seconds() / 60.0


def build_baseline_schedule(df, model=None, out_path=None):
    """현재 계획을 release-time 우선 기준선으로 변환합니다."""
    ensure_out()
    df = _ensure_column_aliases(df).copy().reset_index(drop=True)
    probs = None
    if model is not None:
        probs = _predict_abnormal_probabilities(df, model)
    schedules = []

    for machine, group in df.groupby('Machine_ID'):
        grp = group.copy()
        grp['scheduled_start_dt'] = _to_datetime_series(grp.get('Scheduled_start'))
        grp['scheduled_end_dt'] = _to_datetime_series(grp.get('Scheduled_end'))
        grp = grp.sort_values(by=['scheduled_start_dt', 'scheduled_end_dt'], ascending=[True, True])

        machine_start = grp['scheduled_start_dt'].min()
        if pd.isna(machine_start):
            machine_start = pd.Timestamp.now()
        cur = machine_start

        for _, row in grp.iterrows():
            planned_start = pd.to_datetime(row.get('Scheduled_start'), errors='coerce')
            planned_end = pd.to_datetime(row.get('Scheduled_end'), errors='coerce')
            start = max(cur, planned_start) if pd.notnull(planned_start) else cur
            duration_min = float(row.get('Processing_Time', 0) or 0)
            end = start + pd.Timedelta(minutes=duration_min)
            schedules.append({
                'Job_ID': row.get('Job_ID'),
                'Machine_ID': machine,
                'Job_Status': row.get('Job_Status'),
                'Energy_Consumption': float(row.get('Energy_Consumption') or 0),
                'pred_abnormal_prob': float(probs[row.name]) if probs is not None else np.nan,
                'orig_Scheduled_start': planned_start,
                'orig_Scheduled_end': planned_end,
                'priority_score': np.nan,
                'optimized_start': start,
                'optimized_end': end,
                'Processing_Time': duration_min,
            })
            cur = end

    baseline_df = pd.DataFrame(schedules)
    if out_path is None:
        out_path = os.path.join(OUT_DIR, 'baseline_schedule.csv')
    if not baseline_df.empty:
        baseline_df['optimized_start'] = pd.to_datetime(baseline_df['optimized_start'])
        baseline_df['optimized_end'] = pd.to_datetime(baseline_df['optimized_end'])
    baseline_df.to_csv(out_path, index=False)
    print(f"기준선 스케줄 저장됨: {out_path}")
    return baseline_df


def plot_model_evaluation(y_true, y_prob, y_pred, save_path=None):
    if not HAS_MPL:
        print("matplotlib이 없어 모델 평가 그림은 생략합니다.")
        return None

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    fpr, tpr, _ = roc_curve(y_true, y_prob) if len(np.unique(y_true)) > 1 else ([], [], [])
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax0 = axes[0]
    im = ax0.imshow(cm, cmap="Blues")
    ax0.set_title("Confusion Matrix")
    ax0.set_xlabel("Predicted")
    ax0.set_ylabel("Actual")
    ax0.set_xticks([0, 1])
    ax0.set_yticks([0, 1])
    ax0.set_xticklabels(["Normal", "Abnormal"])
    ax0.set_yticklabels(["Normal", "Abnormal"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax0.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = axes[1]
    if len(fpr) > 0:
        ax1.plot(fpr, tpr, color="#2E7D32", linewidth=2.5, label=f"ROC AUC = {auc:.3f}")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right")
    ax1.text(
        0.05,
        0.05,
        f"Accuracy: {acc:.3f}\nF1: {f1:.3f}",
        transform=ax1.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"모델 평가 시각화 저장됨: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


def plot_schedule_effects(baseline_df, optimized_df, save_path=None, summary_csv=None):
    if not HAS_MPL:
        print("matplotlib이 없어 스케줄 비교 그림은 생략합니다.")
        return None

    if baseline_df is None or baseline_df.empty or optimized_df is None or optimized_df.empty:
        return None

    base = baseline_df.copy()
    opt = optimized_df.copy()

    for frame in (base, opt):
        frame["optimized_start"] = _to_datetime_series(frame.get("optimized_start"))
        frame["optimized_end"] = _to_datetime_series(frame.get("optimized_end"))
        frame["orig_Scheduled_start"] = _to_datetime_series(frame.get("orig_Scheduled_start"))
        frame["orig_Scheduled_end"] = _to_datetime_series(frame.get("orig_Scheduled_end"))
        if "pred_abnormal_prob" not in frame.columns:
            frame["pred_abnormal_prob"] = np.nan
        if "Energy_Consumption" not in frame.columns:
            frame["Energy_Consumption"] = np.nan
        frame["pred_abnormal_prob"] = _safe_numeric(frame["pred_abnormal_prob"], default=0.0)
        frame["Energy_Consumption"] = _safe_numeric(frame["Energy_Consumption"], default=0.0)

    rows = []
    machine_values = sorted(set(base["Machine_ID"].dropna().astype(str).tolist()) | set(opt["Machine_ID"].dropna().astype(str).tolist()))
    scopes = ["ALL"] + machine_values
    for scope in scopes:
        base_subset = base if scope == "ALL" else base[base["Machine_ID"].astype(str) == scope]
        opt_subset = opt if scope == "ALL" else opt[opt["Machine_ID"].astype(str) == scope]
        if base_subset.empty or opt_subset.empty:
            continue

        if base_subset.empty or opt_subset.empty:
            continue

        base_cycle = (
            pd.to_datetime(base_subset["optimized_end"], errors="coerce")
            - pd.to_datetime(base_subset["orig_Scheduled_start"], errors="coerce")
        ).dt.total_seconds().div(60.0).clip(lower=0)
        opt_cycle = (
            pd.to_datetime(opt_subset["optimized_end"], errors="coerce")
            - pd.to_datetime(opt_subset["orig_Scheduled_start"], errors="coerce")
        ).dt.total_seconds().div(60.0).clip(lower=0)

        if base_cycle.dropna().empty or opt_cycle.dropna().empty:
            continue

        base_energy_cost = float((base_subset["Energy_Consumption"] * base_cycle.fillna(0.0)).sum())
        opt_energy_cost = float((opt_subset["Energy_Consumption"] * opt_cycle.fillna(0.0)).sum())

        base_risk_exposure = float((base_subset["pred_abnormal_prob"].fillna(0.0) * base_cycle.fillna(0.0)).sum())
        opt_risk_exposure = float((opt_subset["pred_abnormal_prob"].fillna(0.0) * opt_cycle.fillna(0.0)).sum())

        base_leadtime = float(base_cycle.mean()) if len(base_cycle) else np.nan
        opt_leadtime = float(opt_cycle.mean()) if len(opt_cycle) else np.nan

        rows.append({
            "scope": scope,
            "baseline_risk_exposure": base_risk_exposure,
            "optimized_risk_exposure": opt_risk_exposure,
            "risk_reduction_pct": ((base_risk_exposure - opt_risk_exposure) / base_risk_exposure * 100.0) if base_risk_exposure > 0 else np.nan,
            "baseline_leadtime_min": base_leadtime,
            "optimized_leadtime_min": opt_leadtime,
            "leadtime_reduction_pct": ((base_leadtime - opt_leadtime) / base_leadtime * 100.0) if base_leadtime > 0 else np.nan,
            "baseline_energy_cost": base_energy_cost,
            "optimized_energy_cost": opt_energy_cost,
            "energy_reduction_pct": ((base_energy_cost - opt_energy_cost) / base_energy_cost * 100.0) if base_energy_cost > 0 else np.nan,
        })

    summary_df = pd.DataFrame(rows)
    if summary_csv:
        summary_df.to_csv(summary_csv, index=False)
        print(f"스케줄 효과 요약 저장됨: {summary_csv}")

    if summary_df.empty:
        return summary_df

    plot_df = summary_df[summary_df["scope"] != "ALL"].copy()
    if plot_df.empty:
        plot_df = summary_df.copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("risk", "Expected delay/failure exposure", "risk_reduction_pct", "#EF5350"),
        ("leadtime", "Average leadtime", "leadtime_reduction_pct", "#42A5F5"),
        ("energy", "Energy cost proxy", "energy_reduction_pct", "#66BB6A"),
    ]

    for ax, (_, title, col, color) in zip(axes, metrics):
        bars = ax.bar(plot_df["scope"], plot_df[col], color=color, edgecolor="black")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("Reduction (%)")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"스케줄 효과 시각화 저장됨: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return summary_df

def summarize(df):
    ensure_out()
    df.head(10).to_csv(os.path.join(OUT_DIR, "head.csv"), index=False)
    pd.Series(df.dtypes.astype(str)).to_csv(os.path.join(OUT_DIR, "dtypes.csv"))
    pd.Series(df.isnull().sum()).to_csv(os.path.join(OUT_DIR, "missing.csv"))
    pd.Series({c: int(df[c].nunique(dropna=True)) for c in df.columns}).to_csv(os.path.join(OUT_DIR, "unique_counts.csv"))

def prepare_features(df):
    # normalize column aliases (fix casing/notation mismatches)
    df = _ensure_column_aliases(df)
    # parse datetimes if present
    for col in ("Scheduled_Start","Scheduled_End","Actual_Start","Actual_End"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # ensure all alias variants for datetime columns are also datetime-typed
    datetime_groups = [
        ['Scheduled_Start','Scheduled_start','scheduled_start'],
        ['Scheduled_End','Scheduled_end','scheduled_end'],
        ['Actual_Start','Actual_start','actual_start'],
        ['Actual_End','Actual_end','actual_end']
    ]
    for aliases in datetime_groups:
        parsed = None
        for a in aliases:
            if a in df.columns and pd.api.types.is_datetime64_any_dtype(df[a]):
                parsed = a
                break
        if parsed:
            for a in aliases:
                df[a] = pd.to_datetime(df[parsed], errors='coerce')

    # engineered features: delays in minutes (can be NaN)
    if "Scheduled_Start" in df.columns and "Actual_start" in df.columns:
        df["start_delay_min"] = (df["Actual_start"] - df["Scheduled_start"]).dt.total_seconds() / 60.0
    else:
        df["start_delay_min"] = np.nan

    if "Scheduled_end" in df.columns and "Actual_end" in df.columns:
        df["end_delay_min"] = (df["Actual_end"] - df["Scheduled_end"]).dt.total_seconds() / 60.0
    else:
        df["end_delay_min"] = np.nan

    # make binary target: 정상(Completed)=0, 비정상(Delayed/Failed/...) =1
    if "Job_Status" in df.columns:
        df["is_abnormal"] = df["Job_Status"].fillna("").astype(str).str.lower().apply(lambda s: 0 if "completed" in s else 1)
    else:
        raise RuntimeError("타깃 컬럼 `Job_Status`가 필요합니다.")

    return df

def build_and_train(df, target="is_abnormal"):
    ensure_out()
    X = df.copy()
    y = X.pop(target)

    # drop identifier-like columns from features
    id_like = [c for c in X.columns if c.lower().endswith("_id") or c.lower().startswith("job_") or c.lower().startswith("jobid")]
    X = X.drop(columns=[c for c in id_like if c in X.columns])

    # select feature columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

    # ensure helpful numeric features present
    # keep Processing_Time, Energy_Consumption, Machine_Availability if exist
    # set up transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', cat_transformer, cat_cols)
    ], remainder='drop')

    clf = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1))
    ])

    # drop rows where y is null
    mask = pd.Series(y).notnull().values
    X = X.loc[mask]
    y = pd.Series(y).loc[mask].astype(int).values

    if len(np.unique(y)) < 2:
        raise RuntimeError("타깃에 단일 클래스만 있습니다. 레이블 확인 필요.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # cross-validation
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')
    print(f"CV F1: {scores.mean():.4f} ± {scores.std():.4f}")

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        test_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        test_auc = float('nan')

    print(f"Test accuracy: {test_score:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test ROC AUC: {test_auc:.4f}")

    plot_model_evaluation(
        y_test,
        y_prob,
        y_pred,
        save_path=os.path.join(OUT_DIR, "model_evaluation.png"),
    )

    joblib.dump(clf, os.path.join(OUT_DIR, "model.pkl"))
    with open(os.path.join(OUT_DIR, "model_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"train_shape: {X_train.shape}\n")
        f.write(f"test_shape: {X_test.shape}\n")
        f.write(f"cv_f1_mean: {scores.mean()}\n")
        f.write(f"test_accuracy: {test_score}\n")
        f.write(f"test_f1: {test_f1}\n")
        f.write(f"test_roc_auc: {test_auc}\n")
    return clf

def schedule_with_predictions(df, model, out_path=None):
    ensure_out()
    df = _ensure_column_aliases(df)
    X = df.copy()
    probs = _predict_abnormal_probabilities(X, model)

    df_out = df.copy().reset_index(drop=True)
    df_out['pred_abnormal_prob'] = probs
    df_out['Processing_Time_filled'] = _safe_numeric(df_out.get('Processing_Time', pd.Series(np.nan, index=df_out.index)), default=0.0)
    df_out['Energy_Consumption_filled'] = _safe_numeric(df_out.get('Energy_Consumption', pd.Series(np.nan, index=df_out.index)), default=0.0)
    df_out['Machine_Availability_filled'] = _safe_numeric(df_out.get('Machine_Availability', pd.Series(np.nan, index=df_out.index)), default=50.0)

    # release time / duration / energy를 함께 반영한 복합 우선순위
    risk_norm = (df_out['pred_abnormal_prob'] - df_out['pred_abnormal_prob'].min())
    risk_norm = risk_norm / (risk_norm.max() + 1e-9)
    proc_norm = (df_out['Processing_Time_filled'] - df_out['Processing_Time_filled'].min())
    proc_norm = proc_norm / (proc_norm.max() + 1e-9)
    energy_norm = (df_out['Energy_Consumption_filled'] - df_out['Energy_Consumption_filled'].min())
    energy_norm = energy_norm / (energy_norm.max() + 1e-9)
    avail_norm = (df_out['Machine_Availability_filled'] - df_out['Machine_Availability_filled'].min())
    avail_norm = avail_norm / (avail_norm.max() + 1e-9)

    df_out['priority_score'] = (
        0.45 * risk_norm
        + 0.25 * (1.0 - proc_norm)
        + 0.20 * (1.0 - energy_norm)
        + 0.10 * (1.0 - avail_norm)
    )

    # release-time aware ready-queue scheduler:
    # at each step, pick the highest-priority job among jobs that are already available.
    schedules = []
    for machine, group in df_out.groupby('Machine_ID'):
        grp = group.copy()
        grp['scheduled_start_dt'] = _to_datetime_series(grp.get('Scheduled_start'))
        grp['scheduled_end_dt'] = _to_datetime_series(grp.get('Scheduled_end'))
        pending = grp.sort_values(by=['scheduled_start_dt', 'scheduled_end_dt']).copy().reset_index(drop=True)

        if pending['scheduled_start_dt'].notna().any():
            cur = pending['scheduled_start_dt'].min()
        else:
            cur = pd.Timestamp.now()

        while not pending.empty:
            ready_mask = pending['scheduled_start_dt'].isna() | (pending['scheduled_start_dt'] <= cur)
            ready = pending[ready_mask].copy()

            if ready.empty:
                cur = pending['scheduled_start_dt'].min()
                continue

            ready = ready.sort_values(
                by=['priority_score', 'scheduled_start_dt', 'scheduled_end_dt'],
                ascending=[False, True, True],
            )
            row = ready.iloc[0]
            duration_min = float(row.get('Processing_Time', row.get('Processing_time', 0)) or 0)
            sched_start = pd.to_datetime(row.get('Scheduled_start'), errors='coerce')
            start = max(cur, sched_start) if pd.notnull(sched_start) else cur
            end = start + pd.Timedelta(minutes=duration_min)
            schedules.append({
                'Job_ID': row.get('Job_ID'),
                'Machine_ID': machine,
                'Job_Status': row.get('Job_Status'),
                'orig_Scheduled_start': row.get('Scheduled_start'),
                'orig_Scheduled_end': row.get('Scheduled_end'),
                'pred_abnormal_prob': row['pred_abnormal_prob'],
                'priority_score': row['priority_score'],
                'optimized_start': start,
                'optimized_end': end,
                'Processing_Time': duration_min,
                'Energy_Consumption': float(row.get('Energy_Consumption', 0) or 0)
            })
            cur = end  # next job starts after this
            pending = pending.drop(index=row.name).reset_index(drop=True)
    sched_df = pd.DataFrame(schedules)
    if out_path is None:
        out_path = os.path.join(OUT_DIR, "schedule.csv")
    # format datetimes
    if not sched_df.empty:
        sched_df['optimized_start'] = pd.to_datetime(sched_df['optimized_start'])
        sched_df['optimized_end'] = pd.to_datetime(sched_df['optimized_end'])
    sched_df.to_csv(out_path, index=False)
    print(f"스케줄 결과 저장됨: {out_path}")
    return sched_df


def schedule_with_mip(df, model, alpha=1.0, beta=10.0, gamma=0.0, time_limit=None, out_path=None):
    """MIP 기반 스케줄러 (pulp/CBC)
    - alpha: makespan 가중치
    - beta: 확률 기반 tardiness 비용 가중치
    - gamma: 예비 에너지 가중치(현재 미사용)
    """
    ensure_out()
    df = _ensure_column_aliases(df)
    X = df.copy()
    id_like = [c for c in X.columns if c.lower().endswith("_id") or c.lower().startswith("job_") or c.lower().startswith("jobid")]
    X_features = X.drop(columns=[c for c in id_like if c in X.columns])

    try:
        proba = model.predict_proba(X_features)
        probs = proba[:, 1]
    except Exception:
        preds = model.predict(X_features)
        probs = np.array(preds, dtype=float)

    jobs = list(range(len(df)))
    proc_times = [float(df.loc[i].get('Processing_Time') or 0) for i in jobs]

    # 시간 기준점 (minutes)
    if 'Scheduled_Start' in df.columns:
        sched_starts = pd.to_datetime(df['Scheduled_Start'], errors='coerce')
        min_time = sched_starts.min()
        if pd.isnull(min_time):
            min_time = pd.Timestamp.now()
    else:
        min_time = pd.Timestamp.now()

    def to_minutes(ts):
        if pd.isnull(ts):
            return None
        return (pd.to_datetime(ts) - min_time).total_seconds() / 60.0

    earliest = [to_minutes(df.loc[i].get('Scheduled_Start')) or 0.0 for i in jobs]
    scheduled_end = [to_minutes(df.loc[i].get('Scheduled_End')) or (earliest[i] + proc_times[i]) for i in jobs]

    prob_model = pulp.LpProblem('scheduling_mip', pulp.LpMinimize)

    # 변수
    s = {i: pulp.LpVariable(f'start_{i}', lowBound=0, cat='Continuous') for i in jobs}
    Cmax = pulp.LpVariable('Cmax', lowBound=0, cat='Continuous')
    T = {i: pulp.LpVariable(f'tardy_{i}', lowBound=0, cat='Continuous') for i in jobs}

    # 기계별 순서 이진변수
    y = {}
    for m, group in df.groupby('Machine_ID'):
        idx = group.index.tolist()
        for i in idx:
            for j in idx:
                if i >= j:
                    continue
                y[(i, j)] = pulp.LpVariable(f'y_{i}_{j}', cat='Binary')

    bigM = sum(proc_times) + 10000.0

    # 비중첩 제약
    for (i, j), yvar in y.items():
        p_i = proc_times[i]
        p_j = proc_times[j]
        prob_model += s[i] + p_i <= s[j] + bigM * (1 - yvar)
        prob_model += s[j] + p_j <= s[i] + bigM * yvar

    # 시작시간 하한, tardiness 정의, makespan
    for i in jobs:
        prob_model += s[i] >= earliest[i]
        prob_model += T[i] >= s[i] + proc_times[i] - scheduled_end[i]
        prob_model += T[i] >= 0
        prob_model += Cmax >= s[i] + proc_times[i]

    # 목적함수: makespan + 확률가중 tardiness
    prob_model += alpha * Cmax + beta * pulp.lpSum([float(probs[i]) * T[i] for i in jobs])

    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=False) if time_limit else pulp.PULP_CBC_CMD(msg=False)
    prob_model.solve(solver)

    schedules = []
    for i in jobs:
        start_min = pulp.value(s[i]) or 0.0
        end_min = start_min + proc_times[i]
        schedules.append({
            'Job_ID': df.loc[i].get('Job_ID'),
            'Machine_ID': df.loc[i].get('Machine_ID'),
            'optimized_start': (min_time + pd.Timedelta(minutes=float(start_min))).isoformat(),
            'optimized_end': (min_time + pd.Timedelta(minutes=float(end_min))).isoformat(),
            'pred_abnormal_prob': float(probs[i]),
            'tardiness_min': max(0.0, float(pulp.value(T[i]) or 0.0)),
            'Processing_Time': proc_times[i]
        })

    sched_df = pd.DataFrame(schedules)
    if out_path is None:
        out_path = os.path.join(OUT_DIR, 'schedule_mip.csv')
    sched_df.to_csv(out_path, index=False)
    print(f"MIP 스케줄 결과 저장됨: {out_path}")
    return sched_df

def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV 파일 없음: {CSV_PATH}")
        return
    df = pd.read_csv(CSV_PATH)
    print("데이터 로드:", df.shape)
    summarize(df)
    df_prepared = prepare_features(df)
    # train model to predict is_abnormal
    model = build_and_train(df_prepared, target="is_abnormal")
    # save label encoder hint (Job_Status mapping)
    # create schedule using model probabilities
    baseline_sched = build_baseline_schedule(df_prepared, model=model)
    sched = schedule_with_predictions(df_prepared, model)
    effect_summary = plot_schedule_effects(
        baseline_sched,
        sched,
        save_path=os.path.join(OUT_DIR, "kpi_comparison.png"),
        summary_csv=os.path.join(OUT_DIR, "quantitative_effect_summary.csv"),
    )
    if effect_summary is not None and not effect_summary.empty and "ALL" in effect_summary["scope"].values:
        overall = effect_summary.loc[effect_summary["scope"] == "ALL"].iloc[0]
        print(
            "정량 요약: "
            f"리스크 {overall['baseline_risk_exposure']:.1f} -> {overall['optimized_risk_exposure']:.1f}, "
            f"리드타임 {overall['baseline_leadtime_min']:.1f} -> {overall['optimized_leadtime_min']:.1f}, "
            f"에너지 {overall['baseline_energy_cost']:.1f} -> {overall['optimized_energy_cost']:.1f}"
        )
    print("완료. artifacts/ 디렉터리를 확인하세요.")

if __name__ == "__main__":
    main()
