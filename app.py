import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

# кеширование:
# st.cache_resource сохраняет один объект модели для повторного использования без загрузки с диска при каждом действии пользователя
# st.cache_data сохраняет таблицы и результаты вычислений возвращая их копии, используем его для тестовых данных и вероятностей

# без декоратора joblib.load заново читал бы модель с диска при каждом перезапуске скрипта, и при изменении порога тоже

# вероятности на тесте не зависят от порога, поэтому кешируются. Аргумент _model исключён из хеширования благодаря подчёркиванию.


st.set_page_config(page_title="Оценка вина", layout="wide")
APP_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model.joblib")


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / "test.csv")


@st.cache_data
def load_config():
    return json.loads((APP_DIR / "config.json").read_text(encoding="utf-8"))


model = load_model()
test_data = load_test()
config = load_config()

features = test_data.drop(columns="target")

st.title("Оценка качества вина")
st.caption("Хорошее вино - вино с экспертной оценкой не ниже 7.")

threshold = st.sidebar.slider(
    "Порог рекомендации",
    min_value=0.0,
    max_value=1.0,
    value=float(config["threshold"]),
    step=0.001,
    format="%.6f",
    key="threshold",
)

st.sidebar.caption(
    "Модель рекомендует вино, если его вероятность " "не ниже выбранного порога."
)

labels = {
    "fixed_acidity": "Фиксированная кислотность",
    "volatile_acidity": "Летучая кислотность",
    "citric_acid": "Лимонная кислота",
    "residual_sugar": "Остаточный сахар",
    "chlorides": "Хлориды",
    "free_sulfur_dioxide": "Свободный диоксид серы",
    "total_sulfur_dioxide": "Общий диоксид серы",
    "density": "Плотность",
    "pH": "pH",
    "sulphates": "Сульфаты",
    "alcohol": "Алкоголь",
}
tab_single, tab_metrics, tab_batch, tab_weights = st.tabs(
    [
        "Оценка одного вина",
        "Модельные метрики",
        "Пакетное предсказание",
        "Что выучила модель",
    ]
)

with tab_single:
    st.write("Введите характеристики вина.")

    values = {}
    columns = st.columns(3)

    numeric_features = features.columns.drop("is_red")

    for i, feature in enumerate(numeric_features):
        with columns[i % 3]:
            values[feature] = st.number_input(
                labels.get(feature, feature),
                min_value=float(features[feature].min()),
                max_value=float(features[feature].max()),
                value=float(features[feature].median()),
                format="%.5f",
                key=f"input_{feature}",
            )

    color = st.radio(
        "Тип вина",
        ["Белое", "Красное"],
        index=int(features["is_red"].median()),
        horizontal=True,
    )
    values["is_red"] = int(color == "Красное")
    input_data = pd.DataFrame([values])[features.columns]

    probability = model.predict_proba(input_data)[0, 1]

    st.metric("Вероятность хорошего вина по модели", f"{probability:.1%}")

    if probability >= threshold:
        st.success("Решение модели: рекомендовать вино.")
    else:
        st.info("Решение модели: не рекомендовать вино.")


@st.cache_data
def get_test_probabilities(_model, X):
    return _model.predict_proba(X)[:, 1]


with tab_metrics:
    st.header("Качество модели на тестовой выборке")

    X_eval = test_data.drop(columns="target")
    y_eval = test_data["target"]

    p_eval = get_test_probabilities(model, X_eval)
    pred_eval = (p_eval >= threshold).astype(int)

    precision = metrics.precision_score(y_eval, pred_eval, zero_division=0)
    recall = metrics.recall_score(y_eval, pred_eval, zero_division=0)
    f1 = metrics.f1_score(y_eval, pred_eval, zero_division=0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{precision:.2%}")
    col2.metric("Recall", f"{recall:.2%}")
    col3.metric("F1", f"{f1:.4f}")

    st.caption(f"Текущий порог: {threshold:.6f}")

    # матрица ошибок
    cm = metrics.confusion_matrix(y_eval, pred_eval, labels=[0, 1])

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))

    metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Остальные", "Хорошие"]
    ).plot(ax=ax_cm, cmap="Blues", values_format="d")

    ax_cm.set_xlabel("Предсказанный класс")
    ax_cm.set_ylabel("Настоящий класс")
    ax_cm.set_title("Матрица ошибок")

    st.pyplot(fig_cm)
    plt.close(fig_cm)

    fpr, tpr, _ = metrics.roc_curve(y_eval, p_eval)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_eval, p_eval)
    roc_auc = metrics.roc_auc_score(y_eval, p_eval)
    ap = metrics.average_precision_score(y_eval, p_eval)

    tn, fp, fn, tp = cm.ravel()
    current_fpr = fp / (fp + tn) if fp + tn > 0 else 0

    fig_curves, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].scatter(
        current_fpr, recall, color="red", s=70, zorder=5, label="Текущий порог"
    )
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("Recall")
    axes[0].set_title("ROC-кривая")
    axes[1].plot(pr_recall, pr_precision, label=f"AP = {ap:.4f}")
    axes[1].axhline(
        y_eval.mean(), linestyle="--", color="gray", label="Случайная модель"
    )
    axes[1].scatter(
        recall, precision, color="red", s=70, zorder=5, label="Текущий порог"
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR-кривая")

    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)

    fig_curves.tight_layout()
    st.pyplot(fig_curves)
    plt.close(fig_curves)

    st.write(
        "При снижении порога подборка становится шире: "
        "модель находит больше хороших вин, но может чаще "
        "ошибаться в рекомендациях. При повышении порога "
        "отбор становится строже и часть хороших вин пропускается."
    )

    if tp + fp == 0:
        st.info(
            "При этом пороге нет рекомендаций. "
            "Precision не определён и отображается как 0."
        )
with tab_batch:
    st.header("Прогноз для нескольких вин")
    st.write(
        "Загрузите CSV с 12 признаками вина без столбца target. "
        "Используется порог из боковой панели."
    )
    st.caption(f"Текущий порог: {threshold:.6f}")
    st.download_button(
        "Скачать пример CSV",
        data=features.head(5).to_csv(index=False).encode("utf-8"),
        file_name="wine_example.csv",
        mime="text/csv",
        key="download_example",
    )
    uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
        except (pd.errors.ParserError, ValueError, UnicodeError):
            st.error(
                "Не удалось прочитать CSV. " "Проверьте формат файла и кодировку UTF-8."
            )
        else:
            required_columns = features.columns.tolist()
            missing_columns = [
                column
                for column in required_columns
                if column not in uploaded_data.columns
            ]
            if missing_columns:
                st.error("В файле не хватает столбцов: " + ", ".join(missing_columns))
            elif uploaded_data.empty:
                st.error("В файле нет строк с данными.")
            else:
                batch_X = uploaded_data[required_columns].apply(
                    pd.to_numeric, errors="coerce"
                )

                if not np.isfinite(batch_X.to_numpy()).all():
                    st.error(
                        "В признаках есть пропуски, текст или бесконечные "
                        "значения. Заполните их корректными числами."
                    )

                elif not batch_X["is_red"].isin([0, 1]).all():
                    st.error(
                        "В столбце is_red допустимы только "
                        "0 — белое вино и 1 — красное."
                    )

                else:
                    batch_proba = model.predict_proba(batch_X)[:, 1]
                    batch_pred = (batch_proba >= threshold).astype(int)

                    result = uploaded_data.copy()
                    result["probability"] = batch_proba
                    result["prediction"] = batch_pred
                    result["recommendation"] = np.where(
                        batch_pred == 1, "Рекомендовать", "Не рекомендовать"
                    )

                    st.dataframe(result)

                    st.write(
                        f"Всего вин: {len(result)}. "
                        f"Рекомендовано: {int(batch_pred.sum())}."
                    )

                    st.download_button(
                        "Скачать прогнозы",
                        data=result.to_csv(index=False).encode("utf-8-sig"),
                        file_name="wine_predictions.csv",
                        mime="text/csv",
                        key="download_predictions",
                    )

with tab_weights:
    st.header("Что выучила модель")

    # Последний шаг пайплайна — логистическая регрессия.
    classifier = model.steps[-1][1]

    coef_table = pd.DataFrame(
        {"Признак": features.columns, "Коэффициент": classifier.coef_[0]}
    ).sort_values("Коэффициент")

    feature_labels = {**labels, "is_red": "Красное вино"}

    coef_table["Название"] = coef_table["Признак"].map(feature_labels)

    colors = np.where(coef_table["Коэффициент"] >= 0, "seagreen", "indianred")

    fig_weights, ax_weights = plt.subplots(figsize=(10, 6))

    ax_weights.barh(coef_table["Название"], coef_table["Коэффициент"], color=colors)

    ax_weights.axvline(0, color="black", linewidth=1)
    ax_weights.set_xlabel("Коэффициент после стандартизации")
    ax_weights.set_title("Коэффициенты логистической регрессии")
    ax_weights.grid(axis="x", alpha=0.3)

    fig_weights.tight_layout()
    st.pyplot(fig_weights)
    plt.close(fig_weights)

    positive = coef_table[coef_table["Коэффициент"] > 0].nlargest(3, "Коэффициент")

    negative = coef_table[coef_table["Коэффициент"] < 0].nsmallest(3, "Коэффициент")

    st.write(
        "**Сильнее всего в сторону хорошего вина:** "
        + ", ".join(positive["Название"])
        + "."
    )

    st.write(
        "**Сильнее всего в противоположную сторону:** "
        + ", ".join(negative["Название"])
        + "."
    )

    st.caption(
        "Коэффициенты относятся к стандартизированным признакам. "
        "Знак показывает направление изменения вероятности "
        "при увеличении признака и неизменных остальных признаках. "
        "Это зависимости, найденные моделью, а не доказательство "
        "причинного влияния."
    )
