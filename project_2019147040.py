# 필요한 library import

import pandas as pd
import matplotlib.pyplot as plt



### Part 1 : 전세계적으로 Life Satisfaction에 영향을 미치는 요인별 상관계수 도출 ###
# Part 1 가설 : 인간이 살아가는 데 기본이 되는 의식주/건강에 밀접한 요인들이 높은 상관계수를 가질 것
# ex. Dwellings without basic facilities, Life expectancy, Self-reported health

# CSV file 불러오기
file_path = 'C:/Users/gtpar/문서/연세대 문서/4학년 1학기/컴퓨터프로그래밍/Project/OECD,DF_BLI,+all.csv'
data = pd.read_csv(file_path)

# 전처리 : 비어있는/모두 동일한 값인 column 삭제 및 철자 수정
columns_to_remove_empty = ['Inequality', 'Observation Value', 'Observation Status', 'BASE_PER','Base reference period']
data_cleaned = data.drop(columns=columns_to_remove_empty)


columns_to_remove_redundant = [
    'STRUCTURE', 'STRUCTURE_ID', 'STRUCTURE_NAME', 'ACTION', 'LOCATION', 
    'INDICATOR', 'MEASURE','Measure', 'OBS_STATUS', 'UNIT_MEASURE', 'UNIT_MULT','Multiplier'
]
data_cleaned = data_cleaned.drop(columns=columns_to_remove_redundant)

data_cleaned['Country'] = data_cleaned['Country'].replace({'Türkiye': 'Turkey'})

# Observation Value 중 Mean을 선택하되 Mean값이 없는 Indicator들에 대하여 Total 값을 참고하도록 설정
mn_indicators = [
    'Feeling safe walking alone at night', 'Employment rate', 'Long-term unemployment rate',
    'Quality of support network', 'Educational attainment', 'Water quality',
    'Self-reported health', 'Employees working very long hours', 'Student skills',
    'Years in education', 'Life expectancy', 'Life satisfaction', 'Homicide rate',
    'Time devoted to leisure and personal care'
]

tot_indicators = [
    'Labour market insecurity', 'Dwellings without basic facilities', 'Housing expenditure',
    'Voter turnout', 'Stakeholder engagement for developing regulations',
    'Household net adjusted disposable income', 'Household net wealth',
    'Rooms per person', 'Personal earnings', 'Air pollution'
]

# INEQUALITY == 'MN' 추출
mn_data = data_cleaned[(data_cleaned['INEQUALITY'] == 'MN') & (data_cleaned['Indicator'].isin(mn_indicators))]

# INEQUALITY == 'TOT' 추출
tot_data = data_cleaned[(data_cleaned['INEQUALITY'] == 'TOT') & (data_cleaned['Indicator'].isin(tot_indicators))]

# 병합
data_cleaned = pd.concat([mn_data, tot_data], ignore_index=True)

# OECD - total data와 korea data 추출
oecd_total_data = data_cleaned[data_cleaned['Country'] == 'OECD - Total']
korea_data = data_cleaned[data_cleaned['Country'] == 'Korea']

data_cleaned = data_cleaned[data_cleaned['Country'] != 'OECD - Total']

numeric_columns = ['OBS_VALUE']
data_cleaned = data_cleaned.groupby(['Country', 'Indicator'], as_index=False)[numeric_columns].mean()
oecd_total_data = oecd_total_data.groupby(['Country', 'Indicator'], as_index=False)[numeric_columns].mean()
korea_data = korea_data.groupby(['Country', 'Indicator'], as_index=False)[numeric_columns].mean()

# Pivot table
data_pivoted = data_cleaned.pivot(index='Country', columns='Indicator', values='OBS_VALUE').reset_index()
oecd_total_data_pivoted = oecd_total_data.pivot(index='Country', columns='Indicator', values='OBS_VALUE').reset_index()
korea_data_pivoted = korea_data.pivot(index='Country', columns='Indicator', values='OBS_VALUE').reset_index()

data_pivoted = data_pivoted.drop(columns=['Country'])
oecd_total_data_pivoted = oecd_total_data_pivoted.drop(columns=['Country'])
korea_data_pivoted = korea_data_pivoted.drop(columns=['Country'])

# 이후 시각화를 위해 음의 상관성을 보이는 Indicator들에 대하여 사전에 부호 변환 진행
indicators_to_negate = ['Long-term unemployment rate', 'Employees working very long hours', 'Homicide rate', 'Labour market insecurity', 'Dwellings without basic facilities','Air pollution']
for indicator in indicators_to_negate:
    if indicator in data_pivoted.columns:
        data_pivoted[indicator] = -data_pivoted[indicator]

for indicator in indicators_to_negate:
    if indicator in oecd_total_data_pivoted.columns:
        oecd_total_data_pivoted[indicator] = -oecd_total_data_pivoted[indicator]

for indicator in indicators_to_negate:
    if indicator in korea_data_pivoted.columns:
        korea_data_pivoted[indicator] = -korea_data_pivoted[indicator]

life_satisfaction_column = 'Life satisfaction'

# 상관계수 계산
correlations = data_pivoted.corr()[life_satisfaction_column].drop(life_satisfaction_column)
correlations_sorted = correlations.sort_values(ascending=False)

print("Correlations with Life Satisfaction Around the World:")
print(correlations_sorted)
print()

plt.figure(figsize=(10, 6))
correlations_sorted.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Correlations with Life Satisfaction Around the World', fontsize=14)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.xlabel('Indicator', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



### Part 2 : 상관계수 상위 2개, 하위 2개 각각 Life Satisfaction과의 유사도 꺾은선 그래프 비교 ###

# 상위 2개, 하위 2개 Indicator 도출
top_2 = correlations_sorted.index[:2]
bottom_2 = correlations_sorted.index[-2:]

# Prepare data for plotting
data_cleaned_pivoted = data_cleaned.pivot(index='Country', columns='Indicator', values='OBS_VALUE').reset_index()

# 꺾은선 그래프에서의 비교를 위해 Min-Max Scaling 함수 구현
def scale_data_manual(df, columns):
    scaled_df = df.copy()
    for col in columns:
        if col in scaled_df.columns:
            col_min = scaled_df[col].min()
            col_max = scaled_df[col].max()
            scaled_df[col] = (scaled_df[col] - col_min) / (col_max - col_min)
    return scaled_df

# 상위 2개 상관계수 Indicator 시각화
indicators_to_plot_top = list(top_2) + [life_satisfaction_column]
scaled_data_top = scale_data_manual(data_cleaned_pivoted, indicators_to_plot_top)

plt.figure(figsize=(12, 8))
for indicator in indicators_to_plot_top:
    if indicator in scaled_data_top.columns:
        plt.plot(scaled_data_top['Country'], scaled_data_top[indicator], label=indicator, marker='o')

plt.title('Top 2 Correlated Indicators with Life Satisfaction', fontsize=16)
plt.ylabel('Scaled Value (0 to 1)', fontsize=12)
plt.xlabel('Country', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title="Indicators", fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 하위 2개 상관계수 Indicator 시각화
indicators_to_plot_bottom = list(bottom_2) + [life_satisfaction_column]
scaled_data_bottom = scale_data_manual(data_cleaned_pivoted, indicators_to_plot_bottom)

plt.figure(figsize=(12, 8))
for indicator in indicators_to_plot_bottom:
    if indicator in scaled_data_bottom.columns:
        plt.plot(scaled_data_bottom['Country'], scaled_data_bottom[indicator], label=indicator, marker='o')

plt.title('Bottom 2 Correlated Indicators with Life Satisfaction', fontsize=16)
plt.ylabel('Scaled Value (0 to 1)', fontsize=12)
plt.xlabel('Country', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title="Indicators", fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



### Part 3 : 대륙별 상관계수 분석 ###
# 전세계를 하나로 통합한 분석은 "일반화의 오류" 발생 가능성 존재 -> 유사한 문화를 향유하는 "대륙"별로 상관계수 계산 재시도 (이전보다 구체적인 Scope)

# 국가별 대륙 정의 by Dictionary
continent_mapping = {
    "South Africa": "Africa",
    "Korea": "Asia",
    "Japan": "Asia",
    "Israel": "Asia",
    "Turkey": "Asia",
    "Austria": "Europe",
    "Belgium": "Europe",
    "Czechia": "Europe",
    "Denmark": "Europe",
    "Estonia": "Europe",
    "Finland": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Greece": "Europe",
    "Hungary": "Europe",
    "Iceland": "Europe",
    "Ireland": "Europe",
    "Italy": "Europe",
    "Latvia": "Europe",
    "Lithuania": "Europe",
    "Luxembourg": "Europe",
    "Netherlands": "Europe",
    "Norway": "Europe",
    "Poland": "Europe",
    "Portugal": "Europe",
    "Slovak Republic": "Europe",
    "Slovenia": "Europe",
    "Spain": "Europe",
    "Sweden": "Europe",
    "Switzerland": "Europe",
    "United Kingdom": "Europe",
    "Russia": "Europe",
    "Canada": "North America",
    "Mexico": "North America",
    "United States": "North America",
    "Brazil": "South America",
    "Chile": "South America",
    "Colombia": "South America",
    "Costa Rica": "South America",
    "Australia": "Oceania",
    "New Zealand": "Oceania"
}

# 대륙 Mapping을 위해 기존 data에 다시 'Country' 추가
data_pivoted['Country'] = data_cleaned.pivot(index='Country', columns='Indicator', values='OBS_VALUE').reset_index()['Country']

# 'Continent' column 추가
data_pivoted['Continent'] = data_pivoted['Country'].map(lambda country: continent_mapping.get(country, 'Unknown'))

# 특정 대륙에 대한 상관계수 계산을 위한 함수 구현
def calculate_continent_correlations(data, continent_name):
    continent_data = data[data['Continent'] == continent_name]
    numeric_data = continent_data.drop(columns=['Continent', 'Country'], errors='ignore')
    numeric_data = numeric_data.select_dtypes(include=['float', 'int'])
    if numeric_data.empty:
        return None
    return numeric_data.corr()[life_satisfaction_column].drop(life_satisfaction_column, errors='ignore')

# 대륙별 상관계수 계산
continents = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
continent_correlations = {}

for continent in continents:
    correlations_cont = calculate_continent_correlations(data_pivoted, continent)
    if correlations_cont is not None:
        continent_correlations[continent] = correlations_cont.abs().sort_values(ascending=False)

# 결과 호출
for continent, correlations_cont in continent_correlations.items():
    print(f"Correlations with Life Satisfaction for {continent}:")
    print(correlations_cont)
    print("\n")



### Part 4-1 : Korea의 오차값과 Total 상관계수의 곱 계산을 통한 Korea의 급선무 도출 ###
# Part 4 가설 : 경쟁이 과열된 Korea에서는 심리적인 지표들이 오차 크게 발생해 급선무로 도출될 것
# ex. Labour market insecurity, Employees working very long hours, Time devoted to leisure and personal care

# Korea에서 측정되지 않은 값들은 무시하기 위해 OECD - toal과 Korea의 data size를 맞춰줌
common_columns = oecd_total_data_pivoted.columns.intersection(korea_data_pivoted.columns).drop('Life satisfaction')
oecd_total_data_pivoted = oecd_total_data_pivoted[common_columns]
korea_data_pivoted = korea_data_pivoted[common_columns]
indicators = oecd_total_data_pivoted.columns

# 0으로 나누는 Error 방지 : 0 을 매우 작은 값으로 대체
oecd_total_values = oecd_total_data_pivoted.values
oecd_total_values[oecd_total_values == 0] = 1e-10

# Indicator별 오차 계산
differences = ((oecd_total_data_pivoted.values - korea_data_pivoted.values) / oecd_total_values).flatten()
difference_df = pd.DataFrame({'Indicator': indicators, 'Difference': differences})

# 음의 상관성을 가지는 Indicator에 대하여 부호 변환
difference_df.loc[difference_df['Indicator'].isin(indicators_to_negate), 'Difference'] *= -1

# Correlations 계산을 위한 전처리
correlations = correlations.reset_index().rename(columns={'Life satisfaction': 'Correlation'})

# differences 와 correlations 병합
difference_df = difference_df.merge(correlations, how='left', on='Indicator')

# 존재하지 않는 값 처리
difference_df = difference_df.dropna()

# 우선순위 도출
difference_df['Priority'] = difference_df['Difference'] * abs(difference_df['Correlation'])

# 우선순위 Sort
priority_df = difference_df.sort_values(by='Priority', ascending=False).reset_index(drop=True)

# 결과 출력
print("Indicator Priorities for Improving Life Satisfaction in Korea according to World's Correlation:")
print(priority_df)
print()

# 시각화
plt.figure(figsize=(12, 6))
plt.bar(priority_df['Indicator'], priority_df['Priority'], color='skyblue', edgecolor='black')
plt.title("Indicator Priorities for Improving Life Satisfaction in Korea according to World's Correlation", fontsize=16)
plt.xlabel('Indicators', fontsize=12)
plt.ylabel('Priority Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



### Part 4-2 : Korea의 오차값과 Asia 상관계수의 곱 계산을 통한 Korea의 급선무 도출 ###
# Part 4 가설 : 경쟁이 과열된 Korea에서는 심리적인 지표들이 오차 크게 발생해 급선무로 도출될 것
# ex. Labour market insecurity, Employees working very long hours, Time devoted to leisure and personal care

# Asia 국가들의 Indicator 평균값 계산
asia_data = data_pivoted[data_pivoted['Continent'] == 'Asia']
asia_means = asia_data.drop(columns=['Country', 'Continent']).mean()

# Korea의 데이터 추출
korea_values = korea_data_pivoted.iloc[0]

# 공통된 Indicator만 남기기
common_indicators = asia_means.index.intersection(korea_values.index)
asia_means_filtered = asia_means[common_indicators]
korea_values_filtered = korea_values[common_indicators]

# 오차 계산 (Asia 평균값 - Korea 값)
differences_asia_korea = (asia_means_filtered - korea_values_filtered) / asia_means_filtered

# Asia 상관계수 추출
asia_correlations = continent_correlations['Asia'].reindex(common_indicators)

# 우선순위 도출
priority_scores = differences_asia_korea * asia_correlations

# Sort
priority_df = pd.DataFrame({
    'Indicator': differences_asia_korea.index,
    'Difference': differences_asia_korea.values,
    'Correlation': asia_correlations.values,
    'Priority': priority_scores.values
}).sort_values(by='Priority', ascending=False).reset_index(drop=True)

# 결과 출력
print("Indicator Priorities for Improving Life Satisfaction in Korea according to Asia's Correlation:")
print(priority_df)

# 시각화
plt.figure(figsize=(12, 6))
plt.bar(priority_df['Indicator'], priority_df['Priority'], color='skyblue', edgecolor='black')
plt.title("Indicator Priorities for Improving Life Satisfaction in Korea according to Asia's Correlation:", fontsize=16)
plt.xlabel('Indicators', fontsize=12)
plt.ylabel('Priority Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()