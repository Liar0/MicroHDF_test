"""
    The ConcatenatTestLoad.py is concatenating the feature matrices corresponding to
    each level of hierarchy (taxonomic lineage).
"""

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
        abundance_g_file = "DataMicrobiome/IBD/IBD_genus_abundance.csv"
        abundance_f_file = "DataMicrobiome/IBD/IBD_family_abundance.csv"
        abundance_c_file = "DataMicrobiome/IBD/IBD_class_abundance.csv"
        abundance_o_file = "DataMicrobiome/IBD/IBD_order_abundance.csv"
        abundance_p_file = "DataMicrobiome/IBD/IBD_phylum_abundance.csv"
        phylogenTree_file_l = 'DataMicrobiome/IBD/phylogenTree_l_IBD.csv'
        phylogenTree_file_p = 'DataMicrobiome/IBD/PhylogenTree_p_IBD.csv'

    elif dataid == 2:
        meta_file = "DataMicrobiome/Colorectal/Metadata_CRC.csv"
        abundance_file = "DataMicrobiome/Colorectal/abundance_CRC.csv"
        phylogenTree_file_l = 'DataMicrobiome/Colorectal/phylogenTree_l_CRC.csv'
        phylogenTree_file_p = 'DataMicrobiome/Colorectal/phylogenTree_p_CRC.csv'
    elif dataid == 3:
        meta_file = "DataMicrobiome/Obesity/Metadata_Obesity.csv"
        abundance_file = "DataMicrobiome/Obesity/abundance_Obesity.csv"
        phylogenTree_file_l = 'DataMicrobiome/Obesity/phylogenTree_l_Obesity.csv'
        phylogenTree_file_p = 'DataMicrobiome/Obesity/phylogenTree_p_Obesity.csv'
    elif dataid == 4:
        meta_file = "DataMicrobiome/ASD/Metadata_ASD.csv"
        abundance_file = "DataMicrobiome/ASD/abundance_ASD.csv"
    elif dataid == 5:
        meta_file = "DataMicrobiome/T2D/Metadata_T2D.csv"
        abundance_file = "DataMicrobiome/T2D/abundance_T2D.csv"
        phylogenTree_file_l = 'DataMicrobiome/T2D/phylogenTree_l_T2D.csv'
        phylogenTree_file_p = 'DataMicrobiome/T2D/phylogenTree_p_T2D.csv'
    elif dataid == 6:
        meta_file = "DataMicrobiome/WT2D/Metadata_WT2D.csv"
        abundance_file = "DataMicrobiome/WT2D/abundance_WT2D.csv"
        phylogenTree_file_l = "DataMicrobiome/WT2D/phylogenTree_l_WT2D.csv"
        phylogenTree_file_p = "DataMicrobiome/WT2D/phylogenTree_p_WT2D.csv"
    elif dataid == 7:
        meta_file = "DataMicrobiome/Cirrhosis/Metadata_Cirrhosis.csv"
        abundance_file = "DataMicrobiome/Cirrhosis/abundance_Cirrhosis.csv"
        phylogenTree_file_l = 'DataMicrobiome/Cirrhosis/phylogenTree_l_Cirrhosis.csv'
        phylogenTree_file_p = 'DataMicrobiome/Cirrhosis/phylogenTree_p_Cirrhosis.csv'
    elif dataid == 8:
        meta_file = "DataMicrobiome/Synthetic_ibd/Metadata.csv"
        abundance_file = "DataMicrobiome/Synthetic_ibd/Synthetic.csv"
        phylogenTree_file_l = "DataMicrobiome/Synthetic_ibd/phylogentic_l_synthetic.csv"
        phylogenTree_file_p = "DataMicrobiome/Synthetic_ibd/phylogentic_p_synthetic.csv"
    elif dataid == 9:
        meta_file = "DataMicrobiome/IjazUZ/Metadata_IjazUz.csv"
        abundance_file = "DataMicrobiome/IjazUZ/abundance_IjazUz.csv"
        phylogenTree_file_l = "DataMicrobiome/IjazUZ/phylogenTree_l_IjazUz.csv"
        phylogenTree_file_p = "DataMicrobiome/IjazUZ/phylogenTree_p_IjazUz.csv"

    elif dataid == 10:
        meta_file = "DataMicrobiome/test_ibd/Metadata_test.csv"
        abundance_file = "DataMicrobiome/test_ibd/test_ibd.csv"

    else:
        print(f"Unknown dataid {dataid}")
        return None, None, None

    metadata = pd.read_csv(meta_file)
    abundance = pd.read_csv(abundance_file)
    abundance_g = pd.read_csv(abundance_g_file)
    abundance_f = pd.read_csv(abundance_f_file)
    abundance_c = pd.read_csv(abundance_o_file)
    abundance_o = pd.read_csv(abundance_c_file)
    abundance_p = pd.read_csv(abundance_p_file)
    phylogen_l = pd.read_csv(phylogenTree_file_l)
    phylogen_p = pd.read_csv(phylogenTree_file_p)
    metadata = encode_gender(metadata)
    label = pd.Categorical(metadata["disease"])
    # print("label ", label)
    metadata["disease"] = label.codes + 1
    # print("labels is:", metadata["disease"])
    n_sample = metadata.shape[0]
    n_feature = (abundance.shape[1] - 1) + (metadata.shape[1] - 2) + (abundance_g.shape[1] - 1) + (abundance_g.shape[1] - 1) + (abundance_f.shape[1] - 1) + (abundance_c.shape[1] - 1) + (abundance_o.shape[1] - 1) + (abundance_p.shape[1] - 1)
    n_feature_2 = (phylogen_p.shape[1] - 1) + (phylogen_l.shape[1] - 1)
    # n_feature_2 = phylogen_p.shape[1] - 1
    print("sample is:", n_sample)
    print("features is:", n_feature)
    normalize_metadata = normalize_feature(metadata)
    # print(metadata.head())

    # feature integration
    label = pd.Categorical(metadata["disease"]).codes + 1
    metadata = metadata.iloc[:, -3:].copy()
    abundance = abundance.drop(abundance.columns[0], axis=1)
    phylogen_l = phylogen_l.drop(phylogen_l.columns[0], axis=1)
    phylogen_p = phylogen_p.drop(phylogen_p.columns[0], axis=1)
    # label = label[:, np.newaxis]
    label = label.reshape(-1, 1)
    data = np.concatenate((phylogen_l ,phylogen_p , abundance, metadata, label), axis=1)
    # data = np.concatenate((abundance, metadata, label), axis=1)
    # print(sample)
    if dataid == 3:
        metadata = metadata.iloc[:, :-1].copy()
        data = np.concatenate((phylogen_l,phylogen_p, abundance, metadata, label), axis=1)
    return data[:, 0:-1], data[:, -1], dataid, n_feature_2
    # return data[:, 0:-1], data[:, -1], dataid

def normalize_feature(metadata):
    # age, bmi nomalization
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
    gender = pd.to_numeric(metadata['gender'], errors='coerce')
    if gender.isnull().any():
        gender[gender.isnull()] = np.random.randint(0, 2, size=gender.isnull().sum())
    metadata['gender'] = gender.replace({1: 'female', 0: 'male'})
    metadata['gender'] = metadata['gender'].replace({'female': 1, 'male': 0})
    return metadata








