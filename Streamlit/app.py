import io
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes, make_regression 
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
	page_title="Regression Studio",
	page_icon="📈",
	layout="wide",
)

st.title("📈 Regression Studio")
st.caption(
	"Train and evaluate regression models on your dataset, visualize metrics, and export the trained pipeline."
)


# -----------------------------
# Helpers
# -----------------------------
def get_sample_dataset() -> Tuple[pd.DataFrame, str]:
	"""Return a small numeric regression dataset suitable as a demo."""
	dataset = load_diabetes(as_frame=True)
	frame = dataset.frame.copy()
	frame["target"] = dataset.target
	return frame, "target"


def generate_synthetic_dataset(
	n_samples: int,
	n_features: int,
	noise: float,
) -> Tuple[pd.DataFrame, str]:
	X, y = make_regression(
		n_samples=n_samples,
		n_features=n_features,
		noise=noise,
		random_state=42,
	)
	X_df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])
	X_df["target"] = y
	return X_df, "target"


def split_features_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
	X = df.drop(columns=[target_column])
	y = df[target_column]
	return X, y


def find_project_csv_files() -> List[Path]:
	"""Search common project folders for CSV files to load inside the app.

	Looks into the app directory and common subfolders: data/, dataset/, datasets/, input/
	"""
	base_dir = Path(__file__).resolve().parent
	candidate_dirs = [
		base_dir,
		base_dir / "data",
		base_dir / "dataset",
		base_dir / "datasets",
		base_dir / "input",
	]
	files: List[Path] = []
	for d in candidate_dirs:
		if d.exists() and d.is_dir():
			files.extend(sorted(d.glob("*.csv")))
	return files


def infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
	numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
	categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
	return numeric_cols, categorical_cols


def build_preprocessor(
	numeric_columns: List[str],
	categorical_columns: List[str],
	use_scaler: bool,
) -> ColumnTransformer:
	numeric_steps: List[Tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
	if use_scaler:
		numeric_steps.append(("scaler", StandardScaler()))

	numeric_pipeline = Pipeline(steps=numeric_steps)
	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_pipeline, numeric_columns),
			("cat", categorical_pipeline, categorical_columns),
		]
	)
	return preprocessor


def build_model(model_name: str, params: Dict[str, object]):
	if model_name == "Linear Regression":
		return LinearRegression(**params)
	if model_name == "Random Forest":
		return RandomForestRegressor(**params)
	if model_name == "Gradient Boosting":
		return GradientBoostingRegressor(**params)
	raise ValueError(f"Unsupported model: {model_name}")


def build_pipeline(
	model_name: str,
	model_params: Dict[str, object],
	X: pd.DataFrame,
) -> Pipeline:
	# Scale only for linear models by default
	use_scaler = model_name == "Linear Regression"
	numeric_cols, categorical_cols = infer_feature_types(X)
	preprocessor = build_preprocessor(numeric_cols, categorical_cols, use_scaler)
	model = build_model(model_name, model_params)
	return Pipeline(
		steps=[
			("preprocess", preprocessor),
			("model", model),
		]
	)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	mae = mean_absolute_error(y_true, y_pred)
	mse = mean_squared_error(y_true, y_pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_true, y_pred)
	return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer, X: pd.DataFrame) -> List[str]:
	feature_names: List[str] = []
	for name, transformer, columns in preprocessor.transformers_:
		if name == "remainder":
			continue
		if name == "num":
			feature_names.extend(list(columns))
		elif name == "cat":
			ohe: OneHotEncoder = transformer.named_steps["onehot"]
			cat_features = ohe.get_feature_names_out(columns)
			feature_names.extend(list(cat_features))
	return feature_names


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray):
	fig, ax = plt.subplots(figsize=(6, 4))
	ax.scatter(y_true, y_pred, alpha=0.6)
	ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", linewidth=2)
	ax.set_xlabel("Actual")
	ax.set_ylabel("Predicted")
	ax.set_title("Actual vs Predicted")
	st.pyplot(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray):
	residuals = y_true - y_pred
	fig, ax = plt.subplots(figsize=(6, 4))
	sns.histplot(residuals, kde=True, ax=ax)
	ax.set_title("Residuals Distribution")
	st.pyplot(fig)


def plot_feature_importance(pipeline: Pipeline, X: pd.DataFrame):
	model = pipeline.named_steps["model"]
	preprocessor = pipeline.named_steps["preprocess"]
	feature_names = get_feature_names_from_preprocessor(preprocessor, X)

	values: np.ndarray
	if hasattr(model, "feature_importances_"):
		values = getattr(model, "feature_importances_")
	elif hasattr(model, "coef_"):
		coef = getattr(model, "coef_")
		values = np.abs(coef) if coef.ndim == 1 else np.abs(coef).mean(axis=0)
	else:
		st.info("This model does not expose feature importances/coefficients.")
		return

	order = np.argsort(values)[::-1]
	ordered_names = [feature_names[i] for i in order][:20]
	ordered_values = values[order][:20]

	fig, ax = plt.subplots(figsize=(7, 5))
	ax.barh(ordered_names[::-1], ordered_values[::-1])
	ax.set_title("Top Feature Importances")
	st.pyplot(fig)


# -----------------------------
# Sidebar - data and model selection
# -----------------------------
with st.sidebar:
	st.header("Setup")
	data_source = st.radio(
		"Choose data source",
		["Upload CSV", "Project CSV", "Sample (Diabetes)", "Generate synthetic"],
		index=0,
	)

	uploaded_file = None
	dataframe: pd.DataFrame
	default_target: str

	if data_source == "Upload CSV":
		uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"]) 
		if uploaded_file is not None:
			dataframe = pd.read_csv(uploaded_file)
			default_target = dataframe.columns[-1]
		else:
			dataframe = pd.DataFrame()
			default_target = ""
	elif data_source == "Project CSV":
		csv_files = find_project_csv_files()
		if not csv_files:
			st.info("No CSV files found in project folders (data/, datasets/, etc.).")
			dataframe = pd.DataFrame()
			default_target = ""
		else:
			labels = [str(p.relative_to(Path(__file__).resolve().parent)) for p in csv_files]
			choice = st.selectbox("Pick a CSV from project", options=labels, index=0)
			chosen_path = csv_files[labels.index(choice)]
			try:
				dataframe = pd.read_csv(chosen_path)
				default_target = dataframe.columns[-1]
			except Exception as exc:  # noqa: BLE001
				st.error(f"Failed to read {chosen_path}: {exc}")
				dataframe = pd.DataFrame()
				default_target = ""
	elif data_source == "Sample (Diabetes)":
		dataframe, default_target = get_sample_dataset()
	else:
		n_samples = st.slider("Samples", min_value=100, max_value=5000, value=1000, step=100)
		n_features = st.slider("Features", min_value=3, max_value=50, value=10, step=1)
		noise = st.slider("Noise", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
		dataframe, default_target = generate_synthetic_dataset(n_samples, n_features, noise)

	if not dataframe.empty:
		st.success(f"Loaded dataset with shape: {dataframe.shape}")

	if dataframe.empty:
		st.info("Upload or select a dataset to begin.")
		target_column = ""
	else:
		candidate_targets = [c for c in dataframe.columns if dataframe[c].dtype.kind in ("i", "u", "f")]
		target_column = st.selectbox(
			"Target column",
			options=candidate_targets if candidate_targets else list(dataframe.columns),
			index=(candidate_targets.index(default_target) if default_target in candidate_targets else 0),
		)

	test_size = st.slider("Test size (fraction)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
	random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

	st.header("Model")
	model_name = st.selectbox("Algorithm", ["Linear Regression", "Random Forest", "Gradient Boosting"], index=1)

	model_params: Dict[str, object] = {}
	if model_name == "Linear Regression":
		# No major hyperparameters; normalize handled by pipeline
		st.caption("Uses median-impute + standardize for numeric features and one-hot for categoricals.")
	elif model_name == "Random Forest":
		n_estimators = st.slider("n_estimators", 50, 1000, 300, 50)
		max_depth = st.slider("max_depth (None for unlimited)", 0, 30, 0, 1)
		model_params = {
			"n_estimators": n_estimators,
			"max_depth": None if max_depth == 0 else max_depth,
			"n_jobs": -1,
			"random_state": int(random_state),
		}
	elif model_name == "Gradient Boosting":
		n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
		learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1, 0.01)
		max_depth = st.slider("max_depth", 1, 8, 3, 1)
		model_params = {
			"n_estimators": n_estimators,
			"learning_rate": learning_rate,
			"max_depth": max_depth,
			"random_state": int(random_state),
		}

	train_button = st.button("Train model", type="primary", use_container_width=True)


# -----------------------------
# Main - training, evaluation, prediction
# -----------------------------
if "trained_pipeline" not in st.session_state:
	st.session_state.trained_pipeline = None
	st.session_state.feature_frame_columns = None


if train_button:
	if dataframe.empty or not target_column:
		st.warning("Please load a dataset and choose a target column.")
	else:
		with st.spinner("Training..."):
			try:
				X, y = split_features_target(dataframe, target_column)
				x_train, x_test, y_train, y_test = train_test_split(
					X, y, test_size=float(test_size), random_state=int(random_state)
				)
				pipeline = build_pipeline(model_name, model_params, x_train)
				pipeline.fit(x_train, y_train)
				y_pred = pipeline.predict(x_test)
				metrics = evaluate_regression(y_test, y_pred)

				st.session_state.trained_pipeline = pipeline
				st.session_state.feature_frame_columns = list(X.columns)

				st.success("Training complete!")

				col_a, col_b, col_c, col_d = st.columns(4)
				col_a.metric("MAE", f"{metrics['MAE']:.4f}")
				col_b.metric("MSE", f"{metrics['MSE']:.4f}")
				col_c.metric("RMSE", f"{metrics['RMSE']:.4f}")
				col_d.metric("R²", f"{metrics['R2']:.4f}")

				st.subheader("Diagnostics")
				col1, col2 = st.columns(2)
				with col1:
					plot_predictions(y_test.values if isinstance(y_test, pd.Series) else y_test, y_pred)
				with col2:
					plot_residuals(y_test.values if isinstance(y_test, pd.Series) else y_test, y_pred)

				st.subheader("Feature importance / coefficients")
				plot_feature_importance(pipeline, X)

			except Exception as exc:  # noqa: BLE001
				st.error(f"Training failed: {exc}")


st.divider()

st.subheader("Predict on new data")
if st.session_state.trained_pipeline is None:
	st.info("Train a model first to enable predictions.")
else:
	pipeline: Pipeline = st.session_state.trained_pipeline
	feature_columns = st.session_state.feature_frame_columns or []

	st.write(
		"Upload a CSV containing the same feature columns as used for training (without the target column)."
	)
	pred_file = st.file_uploader("Upload CSV for prediction", type=["csv"], key="pred_uploader")
	if pred_file is not None:
		try:
			pred_df = pd.read_csv(pred_file)
			missing_cols = [c for c in feature_columns if c not in pred_df.columns]
			extra_cols = [c for c in pred_df.columns if c not in feature_columns]

			if missing_cols:
				st.error(f"Missing required feature columns: {missing_cols}")
			else:
				pred_df_aligned = pred_df[feature_columns].copy()
				predictions = pipeline.predict(pred_df_aligned)
				result_df = pred_df.copy()
				result_df["prediction"] = predictions

				st.success("Predictions generated!")
				st.dataframe(result_df.head(50), use_container_width=True)

				csv_buffer = io.StringIO()
				result_df.to_csv(csv_buffer, index=False)
				st.download_button(
					label="Download predictions CSV",
					data=csv_buffer.getvalue(),
					file_name="predictions.csv",
					mime="text/csv",
				)

				if extra_cols:
					st.caption(f"Note: Extra columns ignored during prediction: {extra_cols}")
		except Exception as exc:  # noqa: BLE001
			st.error(f"Prediction failed: {exc}")


st.subheader("Export trained model")
if st.session_state.trained_pipeline is None:
	st.caption("Train a model to enable export.")
else:
	try:
		buffer = io.BytesIO()
		joblib.dump(st.session_state.trained_pipeline, buffer)
		buffer.seek(0)
		st.download_button(
			label="Download trained pipeline (.joblib)",
			data=buffer,
			file_name="trained_regression_pipeline.joblib",
			mime="application/octet-stream",
		)
	except Exception as exc:  # noqa: BLE001
		st.error(f"Failed to serialize model: {exc}")


with st.expander("How to use"):
	st.markdown(
		"""
		1. Select a data source in the sidebar and choose the target column.
		2. Pick an algorithm and adjust hyperparameters if desired.
		3. Click "Train model" to see metrics and diagnostics.
		4. Upload a CSV of new data (without the target) to get predictions.
		5. Download the predictions or export the trained pipeline for reuse.
		
		To run locally:
		- Install dependencies: `pip install -r requirements.txt`
		- Start the app: `streamlit run app.py`
		"""
	)


