from sklearn.ensemble import RandomForestRegressor
import xgboost

def perf_predictor_wrapper():
	predictor = RandomForestRegressor(n_estimators=100,
	                                  verbose=True,
	                                  n_jobs=8)
	return predictor

def xgboost_models():
	return xgboost.XGBRegressor(n_estimators=100, colsample_bynode=1.0, subsample=0.5)