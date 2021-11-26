import pickle
import pandas as pd


# data_test_m.pickle - объединенные данные test и features,
# так как из-за размера файла features были сложности на локальном комьютере

DATA_PATH = 'data_test_m.pickle'
MODEL_PATH = 'final_pipeline.pkl'

def run():
    
    data = pd.read_pickle(DATA_PATH)
    
    res = data[['buy_time', 'id', 'vas_id']]
        
    
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
        res['target'] = model.predict_proba(data)[:,1]
        res.to_csv('answers_test.csv', index=False)

if __name__ == '__main__':
    run()