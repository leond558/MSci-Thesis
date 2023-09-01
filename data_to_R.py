import pickle as pkl

with open('pickles/linear_model.pkl', 'rb') as f:
    [data] = pkl.load(f)
f.close()

data[data.columns[:-3]] = data[data.columns[:-3]] + 3
data.to_csv(r'/Users/leondailani/Documents/Part III Project/d.csv',index=False)


