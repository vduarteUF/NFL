import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data_dir = "NFL Play by Play 2009-2018 (v5).csv"
nfl_df = pd.read_csv(data_dir, low_memory=False)

print("COLUMNS IN DATASET")
print(nfl_df.columns.values)

homescores = {}
awayscores = {}
homescores = nfl_df['total_home_score'].groupby(nfl_df['game_id']).max()
awayscores = nfl_df['total_away_score'].groupby(nfl_df['game_id']).max()
nfl_df['final_home'] = nfl_df['game_id'].map(homescores)
nfl_df['final_away'] = nfl_df['game_id'].map(awayscores)

def getScoreDifferential(nfl_df):
    homeScore = nfl_df['final_home']
    awayScore = nfl_df['final_away']
    scoreDifferential = homeScore - awayScore
    if scoreDifferential <= 0:
        return 0
    elif scoreDifferential > 0:
        return 1


def determineTeam(nfl_df):
    if (nfl_df['posteam'] == nfl_df['home_team']):
        return 1
    else:
        return 0


def isRedZone(nfl_df):
    if (nfl_df['yardline_100'] <= 20):
        return 1
    else:
        return 0


def is3Points(nfl_df):
    if (nfl_df['field_goal_result'] == "made"):
        return 1
    else:
        return 0

def is2Points(nfl_df):
    if(nfl_df['two_point_conv_result'] == "success"):
        return 1
    else:
        return 0

def is1Point(nfl_df):
    if(nfl_df['extra_point_result'] == 'good'):
        return 1
    else:
        return 0

def isPenalty(nfl_df):
    if (nfl_df['penalty_team'] == nfl_df['home_team']):
            return 1
    else:
        return 0

def isTD(nfl_df):
    if (nfl_df['posteam_score_post'] - nfl_df['posteam_score'] == 6):
        return 1
    elif (nfl_df['defteam_score_post'] - nfl_df['defteam_score'] == 6):
        return 1
    else:
        return 0


def isAFCW(val):
    if val == 'KC' or val == 'SD' or val == 'LAC' or val == 'OAK' or val == 'DEN':
        return 1
    else:
        return 0


def isAFCN(val):
    if val == 'PIT' or val == 'CLE' or val == 'CIN' or val == 'BAL':
        return 1
    else:
        return 0


def isAFCS(val):
    if val == 'JAC' or val == 'JAX' or val == 'HOU' or val == 'IND' or val == 'TEN':
        return 1
    else:
        return 0


def isAFCE(val):
    if val == 'MIA' or val == 'NE' or val == 'NYJ' or val == 'BUF':
        return 1
    else:
        return 0


def isNFCN(val):
    if val == 'MIN' or val == 'DET' or val == 'CHI' or val == 'GB':
        return 1
    else:
        return 0


def isNFCW(val):
    if val == 'SEA' or val == 'LA' or val == 'STL' or val == 'ARI' or val == 'SF':
        return 1
    else:
        return 0


def isNFCS(val):
    if val == 'CAR' or val == 'NO' or val == 'TB' or val == 'ATL':
        return 1
    else:
        return 0


def isNFCE(val):
    if val == 'NYG' or val == 'WAS' or val == 'DAL' or val == 'PHI':
        return 1
    else:
        return 0

# top active QB win percentages
def isTomBrady(val):
    if (val == 'T.Brady'):
        return 1
    else:
        return 0
def isDrewBrees(val):
    if (val == 'D.Brees'):
        return 1
    else:
        return 0
def isAaronRodgers(val):
    if (val == 'A.Rodgers'):
        return 1
    else:
        return 0
def isBenRoethlisberger(val):
    if (val == 'B.Roethlisberger'):
        return 1
    else:
        return 0
def isPeytonManning(val):
    if (val == 'P.Manning'):
        return 1
    else:
        return 0
def isRussellWilson(val):
    if (val == 'R.Wilson'):
        return 1
    else:
        return 0
def isAndrewLuck(val):
    if (val == 'A.Luck'):
        return 1
    else:
        return 0
# lowest active QB win percentages
def isRyanFitzpatrick(val):
    if (val == 'R.Fitzpatrick'):
        return 1
    else:
        return 0
def isMatthewStafford(val):
    if (val == 'M.Stafford'):
        return 1
    else:
        return 0

# engineered features
nfl_df['Brady'] = nfl_df['passer_player_name'].apply(isTomBrady)
nfl_df['Brees'] = nfl_df['passer_player_name'].apply(isDrewBrees)
nfl_df['Rodgers'] = nfl_df['passer_player_name'].apply(isAaronRodgers)
nfl_df['Roethlisberger'] = nfl_df['passer_player_name'].apply(isBenRoethlisberger)
nfl_df['Manning'] = nfl_df['passer_player_name'].apply(isPeytonManning)
nfl_df['Wilson'] = nfl_df['passer_player_name'].apply(isRussellWilson)
nfl_df['Luck'] = nfl_df['passer_player_name'].apply(isAndrewLuck)
nfl_df['Fitzpatrick'] = nfl_df['passer_player_name'].apply(isRyanFitzpatrick)
nfl_df['Stafford'] = nfl_df['passer_player_name'].apply(isMatthewStafford)

# Engineered Feature to determine if a possession team is also the home team
nfl_df['isHome'] = nfl_df.apply(determineTeam, axis=1)

# Engineered Feature to determine whether a team is inside the red zone
nfl_df['redZone'] = nfl_df.apply(isRedZone, axis=1)

# Engineered Feaure to determine whether a team made a field goal
nfl_df['made3Points'] = nfl_df.apply(is3Points, axis=1)

# Engineered Feaure to determine if a team is AFC NORTH
nfl_df['AFCNorth'] = nfl_df['home_team'].apply(isAFCN)

# Engineered Feaure to determine if a team is AFC
nfl_df['AFCWest'] = nfl_df['home_team'].apply(isAFCW)

# Engineered Feaure to determine if a team is AFC
nfl_df['AFCSouth'] = nfl_df['home_team'].apply(isAFCS)

# Engineered Feaure to determine if a team is AFC
nfl_df['AFCEast'] = nfl_df['home_team'].apply(isAFCE)

# Engineered Feaure to determine if a team is AFC
nfl_df['NFCNorth'] = nfl_df['home_team'].apply(isNFCN)

# Engineered Feaure to determine if a team is AFC
nfl_df['NFCWest'] = nfl_df['home_team'].apply(isNFCW)

# Engineered Feaure to determine if a team is AFC
nfl_df['NFCSouth'] = nfl_df['home_team'].apply(isNFCS)

# Engineered Feaure to determine if a team is AFC
nfl_df['NFCEast'] = nfl_df['home_team'].apply(isNFCE)

#Engineered Feaure to determine if team made 2 point conv
nfl_df['made2points'] = nfl_df.apply(is2Points, axis = 1)

#Engineered Feature to determine if TD occured
nfl_df['is6Pts'] = nfl_df.apply(isTD, axis = 1)

#Engineered Feature for extra point
nfl_df['made1point'] = nfl_df.apply(is1Point, axis = 1)

#Engineered Feautre to determine wheich team got a penalty
nfl_df['Penalty'] = nfl_df.apply(isPenalty, axis = 1)

nfl_df['posteam_score_pre'] = nfl_df['posteam_score'].shift()
nfl_df['defteam_score_pre'] = nfl_df['defteam_score'].shift()
nfl_df['score_differential_pre'] = nfl_df['score_differential'].shift()

nfl_df.drive = nfl_df.drive.astype(float)

replaceFeatures = [
    'interception',
    'fourth_down_converted',
    'fourth_down_failed',
    'fumble',
    'safety',
    'touchdown',
    'sack',
    'third_down_converted',
    'third_down_failed',
    'punt_attempt',
    'tackled_for_loss',
    'punt_blocked',
    'punt_attempt',
    'isHome',
    'no_huddle',
    'shotgun',
    'made3Points',
    'incomplete_pass'
]

elimFeatures = [
    'game_seconds_remaining',
    'field_goal_result_new',
    'two_point_conv_result_new',
    'extra_point_result_new',
    'ydstogo',
    'home_timeouts_remaining',
    'away_timeouts_remaining',
    'redZone',
    'NFCNorth',
    'NFCSouth',
    'NFCWest',
    'NFCEast',
    'AFCNorth',
    'AFCSouth',
    'AFCEast',
    'AFCWest',
    'Brady',
    'Brees',
    'Rodgers',
    'Roethlisberger',
    'Manning',
    'Wilson',
    'Luck',
    'Fitzpatrick',
    'Stafford',
    'posteam_score',
    'defteam_score',
    'score_differential',
    'yardline_100'
]

# Eliminates any data set with an entire row and column of null values
nfl_df.dropna(axis=1, how='all', inplace = True)
nfl_df.dropna(axis=0, how='all', inplace=True)

 # Impute 0 for all the missing NULL values
nfl_df[replaceFeatures] = nfl_df[replaceFeatures].fillna(0)
nfl_df['field_goal_result'] = nfl_df['field_goal_result'].fillna("N/A")
nfl_df['two_point_conv_result'] = nfl_df['two_point_conv_result'].fillna("N/A")
nfl_df['extra_point_result'] = nfl_df['extra_point_result'].fillna("N/A")
nfl_df['posteam_score'] = nfl_df['posteam_score'].fillna(nfl_df['posteam_score_pre'])
nfl_df['defteam_score'] = nfl_df['defteam_score'].fillna(nfl_df['defteam_score_pre'])
nfl_df['score_differential'] = nfl_df['score_differential'].fillna(nfl_df['score_differential_pre'])
nfl_df['score_differential'] = nfl_df['score_differential'].fillna(0)
nfl_df['score_differential_pre'] = nfl_df['score_differential_pre'].fillna(0)
nfl_df['posteam_score_pre'] = nfl_df['posteam_score_pre'].fillna(0)
nfl_df['posteam_score'] = nfl_df['posteam_score'].fillna(0)
nfl_df['defteam_score'] = nfl_df['defteam_score'].fillna(0)
nfl_df['defteam_score_pre'] = nfl_df['defteam_score_pre'].fillna(0)

nfl_df = nfl_df.dropna(axis=0, subset=['yardline_100', 'game_seconds_remaining'])

features = replaceFeatures + elimFeatures

# Label Encoder for field goal result
le = LabelEncoder()
nfl_df['field_goal_result_new'] = le.fit_transform(nfl_df['field_goal_result'])

# Label Encoder for 2 pt conversion
le1 = LabelEncoder()
nfl_df['two_point_conv_result_new'] = le1.fit_transform(nfl_df['two_point_conv_result'])

# Label Encoder for Extra Point
le2 = LabelEncoder()
nfl_df['extra_point_result_new'] = le2.fit_transform(nfl_df['extra_point_result'])

# Evaluating the data types and making sure the features in the array are floats
# print("DATA TYPES OF FEATURES")
# print(nfl_df[features].dtypes)

# Evaluate how many null values we have for each element in the 'features' array
missing_values_count = (nfl_df[features].isnull().sum())
print()
print("NULL VALUES LEFT IN FEATURES")
print(missing_values_count)

print()
print("CLEANED DATAFRAME")
print(nfl_df)

# determine features and y-label
X = nfl_df[features]
nfl_df['win'] = nfl_df.apply(getScoreDifferential, axis=1)
y = nfl_df['win']

# build the model
model = RandomForestClassifier(max_depth=15, n_estimators=125, min_samples_leaf=50)


# Use TRAIN_TEST_SPlIT for large datasets and when large datasets take too long to cross-validate

# Use sklearn to split df into train/test sets
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, test_size=0.2)
model.fit(train_X, train_y)
predictions_train = model.predict_proba(train_X)
predictions_test = model.predict_proba(test_X)

# Score the train/test splits
print("ROC AUC")
auc = roc_auc_score(train_y, predictions_train[:, 1])
print("train: " + str(auc))
auc = roc_auc_score(test_y, predictions_test[:, 1])
print("test: " + str(auc))

# Evaluate the feature importances of the training features
featureImportance = model.feature_importances_
featureImportance_df = pd.DataFrame(featureImportance, index=train_X.columns, columns=["rate of importance"])
print(featureImportance_df)


# Use CROSS_VAL_SCORE on smaller datasets or when the process doesn't add signifcant run time

# Use sklearn to run cross validation on (5) folds of the data
cross_validation = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print("ROC AUC CROSS VALIDATION")
print(cross_validation)
print("ROC AUC MEAN")
print(np.mean(cross_validation))

nfl_df['Probability'] = model.predict_proba(X)[:,1]
nfl_df['group'] = 0
tr_indices = [index for index, values in enumerate (train_y) if values == 1]
nfl_df['group'] = nfl_df.loc[nfl_df.index.isin(tr_indices),'group'] = 1
nfl_df.head()
nfl_df.to_csv('output.csv')
print("Test Results were successfully saved!")
