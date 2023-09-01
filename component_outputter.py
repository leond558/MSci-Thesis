import pickle as pkl

with open('results.pkl', 'rb') as f:
    [control_results,patient_results] = pkl.load(f)
f.close()

patient_files = ['30292', '30293', '30294', '30295', '30297', '30302']
control_files = ['10000', '10058', '10179', '10459', '14136', '16242', '18770', '21091', '23108', '24972', '25002',
                 '25044', '27336', '27949', '28069', '28973']

for patient, result in zip(patient_files,patient_results):
    with open('results/' + patient + '.txt', mode='w') as file:
            file.write(str(result))

for control, result in zip(control_files,control_results):
    with open('results/' + control + '.txt', mode='w') as file:
            file.write(str(result))
