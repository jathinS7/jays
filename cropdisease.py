import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 120)

possible_dirs = [
    '/kaggle/input/global-crop-disease-risk-prediction-dataset-2026-2100',
    '/kaggle/input/global-crop-disease-risk-prediction',
    '/mnt/data/crop_disease_grandmaster_v1',
    '.'
]

DATA_DIR = None
for d in possible_dirs:
    if os.path.exists(os.path.join(d, 'global_crop_disease_risk_prediction_2026_2100.csv')):
        DATA_DIR = d
        break

if DATA_DIR is None:
    raise FileNotFoundError('Dataset folder not found. Please update DATA_DIR path.')

main_file = os.path.join(DATA_DIR, 'global_crop_disease_risk_prediction_2026_2100.csv')
knowledge_file = os.path.join(DATA_DIR, 'disease_knowledge_base_symptoms_controls.csv')
solution_file = os.path.join(DATA_DIR, 'crop_disease_solution_library.csv')
forecast_file = os.path.join(DATA_DIR, 'future_disease_forecasts_2030_2050_2100.csv')
ranking_file = os.path.join(DATA_DIR, 'crop_vulnerability_rankings.csv')

print('Using dataset folder:', DATA_DIR)

df = pd.read_csv(main_file)
knowledge = pd.read_csv(knowledge_file)
solutions = pd.read_csv(solution_file)
forecast = pd.read_csv(forecast_file)
rankings = pd.read_csv(ranking_file)

print('Main dataset shape:', df.shape)


print(df.shape[0])
print(df.shape[1])
print(df['country'].nunique())
print(df['crop_type'].nunique())
print(df['disease_name'].nunique())
print(df['year'].min(), '-', df['year'].max())

summary = pd.DataFrame({
	'missing_values':df.isna().sum(),
	'missing_percent': (df.isna().mean() * 100),
	'data_type' : df.dtypes.astype(str)
})
print(summary.head(20))


print()

kpis = {
	'Average Disease Probability': df['disease_probability'].mean(),
	'Average Yield Loss%' : df['expected_yield_loss_percent'].mean(),
	'Average Food Security Impact' : df['food_security_impact_score'].mean(),
	'Average 90-Day Outbreak Risk ': df['outbreak_probability_next_90_days'].mean(),
	'Total Economic Loss / Hectare' : df['estimated_economic_loss_usd_per_hectare'].sum()
}
print(pd.DataFrame(kpis.items(), columns = ['Metric' , 'Value']))


print('severity_counts')

severity_counts = df['disease_severity'].value_counts().reindex(['Low', 'Moderate', 'High', 'Critical'])

plt.figure(figsize = (10,6))
plt.bar(severity_counts.index, severity_counts.values)
plt.title('dis severe distri')
plt.xlabel('dis severeness')
plt.ylabel('num of rec')
plt.xticks(rotation = 0)



print()

category_risk = df.groupby('disease_category', as_index = False).agg(
avg_disease_probability = ('disease_probability', 'mean'),
avg_yield_loss_percent = ('expected_yield_loss_percent', 'mean'),
avg_outbreak_90_days = ('outbreak_probability_next_90_days', 'mean'),
records = ('disease_name', 'count')
).sort_values('avg_disease_probability', ascending = False)


plt.figure(figsize = (10,6))
plt.bar(category_risk['disease_category'], category_risk['avg_disease_probability'])
plt.title( 'avg disease prob')
plt.xlabel('disease_category')
plt.ylabel('avg disease proba')
plt.xticks( rotation = 32 )
plt.show()
print(category_risk)



top_crops = df.groupby('crop_type', as_index = False).agg(
avg_disease_probability = ('disease_probability' , 'mean'),
avg_yield_loss_percent = ('expected_yield_loss_percent', 'mean'),
avg_food_security_impact_score = ('food_security_impact_score' , 'mean'),
avg_economic_loss = ('estimated_economic_loss_usd_per_hectare' , 'mean')
).sort_values('avg_food_security_impact_score' , ascending = False).head(12)

plt.figure(figsize = (20,12))
plt.bar(top_crops['crop_type'] , top_crops['avg_food_security_impact_score'])
plt.title('top_crops by food security imp score')
plt.xlabel('crop')
plt.ylabel('avg food sec imp')
plt.xticks(rotation = 45)
plt.show()

print(top_crops)


top_countries = df.groupby('country', as_index=False).agg(
    avg_disease_probability=('disease_probability','mean'),
    avg_yield_loss_percent=('expected_yield_loss_percent','mean'),
    avg_food_security_impact_score=('food_security_impact_score','mean'),
    avg_drought_index=('drought_index','mean'),
    avg_heatwave_days=('heatwave_days','mean')
).sort_values('avg_food_security_impact_score', ascending=False).head(15)

plt.figure(figsize=(12,6))
plt.bar(top_countries['country'], top_countries['avg_food_security_impact_score'])
plt.title('Top 15 Countries by Food Security Impact Score')
plt.xlabel('Country')
plt.ylabel('Average Food Security Impact Score')
plt.xticks(rotation=45, ha='right')
plt.show()

print(top_countries)


import seaborn as sns
climate_cols = [ 'avg_temperature_c', 'humidity_percent', 'rainfall_mm', 'soil_moisture', 'soil_ph', 'wind_speed_kmh', 'leaf_wetness_index', 'drought_index', 'flood_risk_score','heatwave_days', 'pest_pressure_score', 'disease_probability', 'expected_yield_loss_percent']

corr = df[climate_cols].corr(numeric_only = True)

plt.figure(figsize=(10,8))
plt.imshow(corr, aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title('Correlation Heatmap: Climate, Soil, Pest & Disease Variables')
plt.tight_layout()
plt.show()

corr[['disease_probability','expected_yield_loss_percent']].sort_values('disease_probability', ascending=False)

knowledge[['crop_type', 'disease_name', 'disease_category', 'symptoms', 'affected_plant_part', 'biological_control', 'chemical_control', 'prevention_method']].head(15)

print()
advisory = solutions[['crop_type','disease_name','symptoms','biological_control','chemical_control','prevention_method','recommended_action_priority'
]].sample(10, random_state=42)
print(advisory)

print()
forecast_trend = forecast.groupby('year', as_index = False).agg(
projected_disease_probability=('projected_disease_probability', 'mean'),
projected_yield_loss_percent =('projected_yield_loss_percent', 'mean'),
projected_outbreak_risk_90_days = ('projected_outbreak_risk_90_days', 'mean'),
projected_food_security_impact_score = ('projected_food_security_impact_score' , 'mean')
)

plt.figure(figsize=(10,8))	
plt.plot(forecast_trend['year'], forecast_trend['projected_disease_probability'] , marker = 'o', label = 'dis proba')
plt.plot(forecast_trend['year'], forecast_trend['projected_outbreak_risk_90_days'] , marker = 'o', label = '90 days outbreak risk')
plt.title('future pred of crop dis')
plt.xlabel('year')
plt.ylabel('avg risk')
plt.legend()
plt.grid(True, alpha = 0.3)
plt.show()
forecast_trend

print()


model_df = df.copy()

features = [
    'avg_temperature_c','max_temperature_c','min_temperature_c','humidity_percent','rainfall_mm','soil_moisture','soil_ph',
    'wind_speed_kmh','leaf_wetness_index','drought_index','flood_risk_score','heatwave_days','pest_pressure_score',
    'fungal_risk_score','bacterial_risk_score','viral_risk_score','disease_probability',
    'outbreak_probability_next_30_days','outbreak_probability_next_90_days'
]

target = 'disease_severity'

X = model_df[features]
y = model_df[target]

if len(X) > 500000:
	sample_idx = X.sample(50000 , random_state = 10).index
	X = X.loc[sample_idx]
	y = y.loc[sample_idx]

X_train ,X_test, y_train ,  y_test = train_test_split(X, y , test_size = 0.2 , random_state = 12, stratify = y)

clf = RandomForestClassifier(n_estimators =120 , random_state = 31, n_jobs = -1, class_weight = 'balanced')
clf.fit(X_train , y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test , y_pred), 4)
print(classification_report(y_test, y_pred))



print()
importance = pd.DataFrame({'feature': features, 'importance': clf.feature_importances_}
).sort_values('importance', ascending = False).head(10)

plt.figure(figsize=(10,6))
plt.barh(importance['feature'][::-1], importance['importance'][::-1])
plt.title('top feature imp')
plt.xlabel('imp')
plt.ylabel('feature')
plt.show()
print(importance)

print()

reg_features = ['avg_temperature_c', 'humidity_percent', 'rainfall_mm','soil_moisture', 'soil_ph', 'wind_speed_kmh',
	'leaf_wetness_index', 'drought_index', 'flood_risk_score', 'heatwave_days', 'pest_pressure_score',
	'fungal_risk_score', 'bacterial_risk_score' , 'viral_risk_score', 'disease_probability',
	'outbreak_probability_next_30_days', 'outbreak_probability_next_90_days'
]

Xr = df[reg_features]
yr = df['expected_yield_loss_percent']
if len(Xr) > 450000:
	sample_idx = Xr.sample(450000, random_state = 31).index
	Xr = Xr.loc[sample_idx]
	yr = yr.loc[sample_idx]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size = 0.3, random_state = 31)

reg =RandomForestRegressor(n_estimators = 120, random_state = 79, n_jobs =1)
reg.fit(Xr_train, yr_train)

yr_pred = reg.predict(Xr_test)

print(mean_absolute_error(yr_test, yr_pred), 3)
print(r2_score(yr_test, yr_pred),3)


print()

plt.figure(figsize = (12,9))
plt.scatter(yr_test, yr_pred, alpha = 0.25)
plt.title('actual vs pred  yield loss')
plt.xlabel('actual yield')
plt.ylabel('pred yield')
plt.grid(True , alpha = 0.4)
plt.show()

pd.DataFrame({'actual_yield_loss_percent': yr_test.values[:20],	'predicted_yield_loss_percent': yr_pred[:20].round(3)})

print()
early_warning = df[(df['disease_probability'] >= 0.75) & (df['outbreak_probability_next_90_days']>= 0.75)].copy()


summary= early_warning.groupby(['country', 'crop_type', 'disease_name'], as_index= False).agg(
avg_disease_probability = ('disease_probability', 'mean'),
avg_outbreak_90_days = ('outbreak_probability_next_90_days' , 'mean'),
avg_yield_loss_percent = ('expected_yield_loss_percent', 'mean'),
avg_food_security_impact_score = ('food_security_impact_score' , 'mean'),
records = ('year', 'count')
).sort_values(['avg_food_security_impact_score','avg_outbreak_90_days'] , ascending = False)

print(summary.head(20))




#<pandas.api.typing.DataFrameGroupBy object at 0x0000024F208D3620>

