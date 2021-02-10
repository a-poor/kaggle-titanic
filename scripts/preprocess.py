
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from typing import Union


def load_df(data_path: Union[Path,str], return_pid:bool = False) -> pd.DataFrame:
    # Load the csv
    df = pd.read_csv(data_path)
    # Extract cabin letter
    df["CabinGroup"] = df.Cabin.apply(
        lambda c: np.nan if pd.isna(c) else c.split(" ")[0][0]
    )
    
    # Create feature flags for `is {Cabin,Age} NA?`
    df["IsCabinNA"] = df.Cabin.isna()
    df["IsAgeNA"] = df.Age.isna()
    
    # Impute Age column
    df = _impute_age(df)
    
    # Add ticket base
    df = _get_ticket_base(df)
    
    # Create dummy columns
    df = pd.get_dummies(
        df,
        columns=["Sex","Embarked","CabinGroup","TicketBase"],
        drop_first=True
    )
    # Confirm column groups
    for c in "BCDEFGT":
        colname = f"CabinGroup_{c}"
        if colname not in df.columns:
            print("Adding column:", colname)
            df[colname] = 0
    
    if return_pid:
        pid = df.PassengerId
    
    # Remove unnecessary columns
    df = df.drop(
        columns=["PassengerId","Name","Cabin","Ticket"]
    )
    colnames = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 
                'IsCabinNA', 'IsAgeNA', 'Sex_male', 'Embarked_Q', 
                'Embarked_S', 'CabinGroup_B', 'CabinGroup_C', 
                'CabinGroup_D', 'CabinGroup_E', 'CabinGroup_F', 
                'CabinGroup_G', 'CabinGroup_T', 'TicketBase_0', 
                'TicketBase_1', 'TicketBase_2', 'TicketBase_3', 
                'TicketBase_4']
    df = df[colnames]
    return (df, pid) if return_pid else df

def _impute_age(df: pd.DataFrame) -> pd.DataFrame:
    df.Age = df.Age.fillna(df.Age.median())
    return df

def _get_ticket_base(df: pd.DataFrame) -> pd.DataFrame:
    df["TicketBase"] = df.Ticket.str\
        .upper().str\
        .replace("[^A-Z]","",case=False,regex=True)
    # Group `TicketBase` into the top 5 values plus
    # a catch-all group for everything else.
    # According to nb: `01.eda.ipynb`, the top 5
    # groups are...
    top_5 = ["","PC","CA","A","STONO"]
    df.TicketBase = df.TicketBase.apply(
        lambda t: -1 if t not in top_5 else top_5.index(t)
    )
    return df
