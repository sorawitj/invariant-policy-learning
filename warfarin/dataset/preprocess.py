# adapted from https://github.com/CausalML/continuous-policy-learning/blob/master/Warfarin-Copy1.ipynb
import numpy as np
import csv

f = open('warfrin.csv', 'rU')
csvr = csv.reader(f, dialect=csv.excel)
header = np.array(next(csvr))
data = list(csvr)
f.close()

# filter only to subjects that reached stable dose of warfarin
# and stable observed INR
data = [x for x in data if x[37] == '1' and x[38] != 'NA' and x[39] != 'NA']

agegroups = sorted(set(x[8].replace('NA', '') for x in data))
mean_height = np.mean([float(x[9]) for x in data if x[9] not in ('NA', '')])
mean_weight = np.mean([float(x[10]) for x in data if x[10] not in ('NA', '')])

xmap = lambda x: \
    [
        ('Site', int(x[2])),
        # BMI
        ('BMI', float(x[10]) * 100. * 100. / float(x[9]) / float(x[9]) if x[10] not in ('NA', '') and x[9] not in (
            'NA', '') else 0.),
        ('Gender Male?', x[3] == 'male'),
        ('Race White?', x[5] == 'White'),
        ('Race Asian?', x[5] == 'Asian'),
        ('Race Black?', x[5] == 'Black or African American'),
        ('Non-hispanic?', x[7] == 'not Hispanic or Latino'),
        ('Age group', agegroups.index(x[8]) if x[8] in agegroups else 0),
        ('No Age?', x[8] not in agegroups or agegroups.index(x[8]) == 0),
        ('Height', float(x[9]) if x[9] not in ('NA', '') else mean_height),
        ('Weight', float(x[10]) if x[10] not in ('NA', '') else mean_weight),
    ] + [
        ('Indication for Warfarin Treatment: ' + str(i), str(i) in x[11])
        for i in range(1, 9)
    ] + [
        ('Diabetes=1?', x[13] == '1'),
    ] + [
        ('Congestive Heart Failure and/or Cardiomyopathy=1?', x[14] == '1'),
    ] + [
        ('Valve Replacement=1?', x[15] == '1'),
    ] + [
        ('aspirin=1?', x[17] == '1'),
        ('Acetaminophen=1?', x[18] == '1'),
        ('Acetaminophen hi dose=1?', x[19] == '1'),
        ('Simvastatin=1?', x[20] == '1'),
        ('Atorvastatin=1?', x[21] == '1'),
        ('Fluvastatin=1?', x[22] == '1'),
        ('Lovastatin=1?', x[23] == '1'),
        ('Pravastatin=1?', x[24] == '1'),
        ('Rosuvastatin=1?', x[25] == '1'),
        ('Cerivastatin=1?', x[26] == '1'),
        ('Amiodarone=1?', x[27] == '1'),
        ('Enzyme inducer status', x[28] == '1' or x[29] == '1' or x[30] == '1'),
        ('Sulfonamide Antibiotics=1?', x[31] == '1'),
        ('Macrolide Antibiotics=1?', x[32] == '1'),
        ('Anti-fungal Azoles=1?', x[33] == '1'),
        ('Herbal Medications, Vitamins, Supplements=1?', x[34] == '1'),
    ] + [
        ('Smoker=1?', x[40] == '1'),
    ] + [
        ('CYP2C9 *1/*1', x[59] == '*1/*1'),
        ('CYP2C9 *1/*2', x[59] == '*1/*2'),
        ('CYP2C9 *1/*3', x[59] == '*1/*3'),
        ('CYP2C9 *2/*2', x[59] == '*2/*2'),
        ('CYP2C9 *2/*3', x[59] == '*2/*3'),
        ('CYP2C9 *3/*3', x[59] == '*3/*3'),
        ('CYP2C9 NA', x[59] == '' or x[59] == 'NA'),
        ('VKORC1 -1639 A/A', x[60] == 'A/A'),
        ('VKORC1 -1639 A/G', x[60] == 'A/G'),
        ('VKORC1 -1639 G/G', x[60] == 'G/G'),
        ('VKORC1 -1639 NA', x[60] == 'NA')
    ]

X = np.array([list(zip(*xmap(x)))[1] for x in data])
Xnames = np.array(list(zip(*xmap(data[0])))[0])
Race = np.array([x[5] for x in data])
# filter features with enough datapoints
goodidx = np.where((X.mean(0) >= .01) | (X.max(0) > 1))[0]
X = X[:, goodidx]
Xnames = Xnames[goodidx]

# filter out by where BMI is nonzero
goodbmi = np.where((X[:, 1] > 0.003))[0]
X = X[goodbmi, :]
Race = Race[goodbmi]

# get target
therapeut_dose = np.array([float(x[38]) if x[38] not in ('NA', '') else 0. for x in data])
therapeut_dose = therapeut_dose[goodbmi]

# save to numpy arrays
np.save('therapeut_dose.npy', therapeut_dose)
np.save('race_info.npy', Race)
np.save('xnames.npy', Xnames)
np.save('warfarin_relevant.npy', X)
