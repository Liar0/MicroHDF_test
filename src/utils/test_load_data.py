import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import itertools
np.set_printoptions(suppress=True)
# def test_load_data(score="S", dataid=9):
def test_load_data(dataid):
    if dataid == 1:
        meta_file = "DataMicrobiome/IBD/Metadata_IBD.csv"
        abundance_file = "DataMicrobiome/IBD/abundance_IBD.csv"
    elif dataid == 2:
        meta_file = "DataMicrobiome/Colorectal/Metadata_CRC.csv"
        abundance_file = "DataMicrobiome/Colorectal/abundance_CRC.csv"

    elif dataid == 3:
        meta_file = "DataMicrobiome/Obesity/Metadata_Obesity.csv"
        abundance_file = "DataMicrobiome/Obesity/abundance_Obesity.csv"

    elif dataid == 4:
        meta_file = "DataMicrobiome/ASD/Metada.csv"
        abundance_file = "DataMicrobiome/ASD/abundance.csv"
    elif dataid == 5:
        meta_file = "DataMicrobiome/T2D/Metadata_T2D.csv"
        abundance_file = "DataMicrobiome/T2D/abundance_T2D.csv"

    elif dataid == 6:
        meta_file = "DataMicrobiome/WT2D/Metadata_WT2D.csv"
        abundance_file = "DataMicrobiome/WT2D/abundance_WT2D.csv"

    elif dataid == 7:
        meta_file = "DataMicrobiome/Cirrhosis/Metadata_Cirrhosis.csv"
        abundance_file = "DataMicrobiome/Cirrhosis/Cirrhosis_phylum_abundance.csv"

    elif dataid == 8:
        meta_file = "DataMicrobiome/Synthetic_ibd/Metadata.csv"
        abundance_file = "DataMicrobiome/Synthetic_ibd/Synthetic.csv"

    elif dataid == 9:
        meta_file = "DataMicrobiome/IjazUZ/Metadata_IjazUz.csv"
        abundance_file = "DataMicrobiome/IjazUZ/abundance_IjazUz.csv"


    elif dataid == 10:
        meta_file = "DataMicrobiome/crossDisease/NielsenHB/Metadata_test.csv"
        abundance_file = "DataMicrobiome/crossDisease/NielsenHB/abundance_NielsenHB.csv"

    elif dataid == 11:
        meta_file = "DataMicrobiome/crossDisease/iCDf/Metadata.csv"
        abundance_file = "DataMicrobiome/crossDisease/iCDf/ICDf.csv"
    elif dataid == 12:
        meta_file = "DataMicrobiome/crossDisease/Arizo_ASD/metadata.csv"
        abundance_file = "DataMicrobiome/crossDisease/Arizo_ASD/abundance.csv"
    elif dataid == 13:
        meta_file = "DataMicrobiome/crossDisease/Dan_ASD/Metadata.csv"
        abundance_file = "DataMicrobiome/crossDisease/Dan_ASD/abundance.csv"
    elif dataid == 14:
        meta_file = "DataMicrobiome/crossDisease/Chen_ASD/metadata.csv"
        abundance_file = "DataMicrobiome/crossDisease/Chen_ASD/abundance.csv"

    else:
        print(f"Unknown dataid {dataid}")
        return None, None, None

    metadata = pd.read_csv(meta_file)
    abundance = pd.read_csv(abundance_file)
    metadata = encode_gender(metadata)
    label = pd.Categorical(metadata["disease"])
    # print("label ", label)
    metadata["disease"] = label.codes + 1
    # print("labels is:", metadata["disease"])
    n_sample = metadata.shape[0]
    n_feature = (abundance.shape[1] - 1) + (metadata.shape[1] - 2)
    print("sample is:", n_sample)
    print("features is:", n_feature)
    normalize_metadata = normalize_feature(metadata)
    # print(metadata.head())

    # feature integration
    label = pd.Categorical(metadata["disease"]).codes + 1
    metadata = metadata.iloc[:, -3:].copy()
    abundance = abundance.drop(abundance.columns[0], axis=1)
    # label = label[:, np.newaxis]
    label = label.reshape(-1, 1)
    data = np.concatenate((abundance, metadata, label), axis=1)
    # print(data.shape[0])
    if dataid == 3:
        metadata = metadata.iloc[:, :-1].copy()
        data = np.concatenate((abundance, metadata, label), axis=1)
    return data[:, 0:-1], data[:, -1], dataid,(abundance.shape[1] - 1)

def normalize_feature(metadata):
    # age, bmi
    age = pd.to_numeric(metadata['age'], errors='coerce').values
    if np.isnan(age).any():
        age[np.isnan(age)] = np.random.randint(18, 60, size=np.isnan(age).sum())
    age_scaler = MinMaxScaler()
    age_norm = age_scaler.fit_transform(age.reshape(-1, 1))
    metadata['age'] = age_norm.flatten()

    bmi = pd.to_numeric(metadata['bmi'], errors='coerce').values
    if np.isnan(bmi).any():
        bmi[np.isnan(bmi)] = np.random.uniform(16.0, 50.0, size=np.isnan(bmi).sum())
    bmi_scaler = MinMaxScaler()
    bmi_norm = bmi_scaler.fit_transform(bmi.reshape(-1, 1))
    metadata['bmi'] = bmi_norm.flatten()
    return metadata

def encode_gender(metadata):
    #  female：1， male：0
    gender = pd.to_numeric(metadata['gender'], errors='coerce')
    if gender.isnull().any():
        gender[gender.isnull()] = np.random.randint(0, 2, size=gender.isnull().sum())
    metadata['gender'] = gender.replace({1: 'female', 0: 'male'})
    metadata['gender'] = metadata['gender'].replace({'female': 1, 'male': 0})
    return metadata







