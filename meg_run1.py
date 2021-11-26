import pickle
import pandas as pd


DATA_PATH = 'data_test.csv'
FEATURES_PATH = 'features.csv'
MODEL_PATH = 'final_pipeline.pkl'


def run():
    
    data = pd.read_csv(DATA_PATH, sep=',').drop(columns=['Unnamed: 0'])
    features = pd.read_csv('features.csv',sep='\t').drop(columns=['Unnamed: 0'])
    
    data = data.sort_values(by=['buy_time'])
    features = features.sort_values(by=['buy_time'])
    
    data_test_merged = pd.merge_asof(data, features, on='buy_time', by='id', direction='nearest')
    res = data_test_merged[['buy_time', 'id', 'vas_id']]
    
        
    
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
        res['target'] = model.predict_proba(data_test_merged)[:,1]
        res.to_csv('answers_test.csv', index=False)

if __name__ == '__main__':
    run()