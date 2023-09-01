import pickle as pkl
import matplotlib.pyplot as plt

with open('pickles/metrics_and_atlases/MD_malp.pkl', 'rb') as f:
    [mdm] = pkl.load(f)
f.close()

with open('pickles/metrics_and_atlases/FA_malp.pkl', 'rb') as f:
    [fam] = pkl.load(f)
f.close()

with open('pickles/metrics_and_atlases/FA_tract.pkl', 'rb') as f:
    [fat] = pkl.load(f)
f.close()

with open('pickles/metrics_and_atlases/MD_tract.pkl', 'rb') as f:
    [mdt] = pkl.load(f)
f.close()

md_new = {}
# print(mdm)
# print(fam)
# print(mdt)
# print(fat)

data = [mdm,fam,fat,mdt]
datavalues = []
names = ['MD_malpem','FA_malpem','FA_Tractseg','MD_tractseg']
for point in data:
    d = list(point.values())
    datavalues.append(d)

dict_of_values = {}
for d,name in zip(datavalues,names):
    dict_of_values[name] = d[:8]

# fig,ax = plt.subplots(figsize=(10,10))
#
# for k,v in dict_of_values.items():
#     ax.plot(v, label=k)
#
# ax.set_title('Comparing the statistical significance in the measured changes between acute and longitudinal scans ')
# ax.set_xlabel('Ordinality of the ROI in the list of most signifcantly changed regions')
# ax.set_ylabel('p-value before correction for multiple comparisons')
# ax.legend()
# plt.tight_layout()
# plt.savefig('plots/atlas_and_metric_significance/1')
# plt.show()

names = names[1:2] + names[3:]
datavalues = datavalues[1:2] + datavalues[3:]
dict_of_values = {}
for d,name in zip(datavalues,names):
    dict_of_values[name] = d[:15]

fig,ax = plt.subplots(figsize=(10,10))

for k,v in dict_of_values.items():
    ax.plot(v, label=k)

ax.set_title('Comparing the statistical significance in the measured changes between acute and longitudinal scans ')
ax.set_xlabel('Ordinality of the ROI in the list of most signifcantly changed regions')
ax.set_ylabel('p-value before correction for multiple comparisons')
ax.legend()
plt.tight_layout()
plt.savefig('plots/atlas_and_metric_significance/2')
plt.show()


