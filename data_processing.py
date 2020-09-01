import numpy as np
import pandas as pd

def data_process1():
    data = pd.read_csv('compiledData3.csv')
    print(data.tail(15))
    data['Empty Beacon 2'].fillna(data['Empty Beacon 2'].mean(), inplace=True)
    data['Empty Beacon 4'].fillna(data['Empty Beacon 4'].mean(), inplace=True)
    data['Empty Beacon 3'].fillna(data['Empty Beacon 3'].mean(), inplace=True)
    data['Empty Beacon 1'].fillna(data['Empty Beacon 1'].mean(), inplace=True)

    data['1PersonPos Beacon 2'].fillna(data['1PersonPos Beacon 2'].mean(), inplace=True)
    data['1PersonPos Beacon 4'].fillna(data['1PersonPos Beacon 4'].mean(), inplace=True)
    data['1PersonPos Beacon 3'].fillna(data['1PersonPos Beacon 3'].mean(), inplace=True)
    data['1PersonPos Beacon 1'].fillna(data['1PersonPos Beacon 1'].mean(), inplace=True)

    data['1PersonMov Beacon 2'].fillna(data['1PersonMov Beacon 2'].mean(), inplace=True)
    data['1PersonMov Beacon 4'].fillna(data['1PersonMov Beacon 4'].mean(), inplace=True)
    data['1PersonMov Beacon 3'].fillna(data['1PersonMov Beacon 3'].mean(), inplace=True)
    data['1PersonMov Beacon 1'].fillna(data['1PersonMov Beacon 1'].mean(), inplace=True)

    data['2PersonPos Beacon 2'].fillna(data['2PersonPos Beacon 2'].mean(), inplace=True)
    data['2PersonPos Beacon 4'].fillna(data['2PersonPos Beacon 4'].mean(), inplace=True)
    data['2PersonPos Beacon 3'].fillna(data['2PersonPos Beacon 3'].mean(), inplace=True)
    data['2PersonPos Beacon 1'].fillna(data['2PersonPos Beacon 1'].mean(), inplace=True)

    data['2PersonMov Beacon 2'].fillna(data['2PersonMov Beacon 2'].mean(), inplace=True)
    data['2PersonMov Beacon 4'].fillna(data['2PersonMov Beacon 4'].mean(), inplace=True)
    data['2PersonMov Beacon 3'].fillna(data['2PersonMov Beacon 3'].mean(), inplace=True)
    data['2PersonMov Beacon 1'].fillna(data['2PersonMov Beacon 1'].mean(), inplace=True)

    print(data.tail(15))

    #data.to_excel("processedData2.xlsx")


def data_process2():
    data = pd.read_csv('compiledData3.csv')
    print(data.tail(15))
    #data['State3'][1202:] = data['State3'][1202:].replace(np.nan, 'Empty')
    #print(data['State3'][1195:])

    data['State2'][0:94] = data['State2'][0:94].replace(np.nan, 'Empty')
    data['State2'][136:214] = data['State2'][136:214].replace(np.nan, 'Empty')
    data['State2'][252:342] = data['State2'][252:342].replace(np.nan, 'Empty')
    data['State2'][386:424] = data['State2'][386:424].replace(np.nan, 'Empty')
    data['State2'][524:550] = data['State2'][524:550].replace(np.nan, 'Empty')
    data['State2'][596:612] = data['State2'][596:612].replace(np.nan, 'Empty')
    data['State2'][678:704] = data['State2'][678:704].replace(np.nan, 'Empty')
    data['State2'][758:796] = data['State2'][758:796].replace(np.nan, 'Empty')
    data['State2'][860:926] = data['State2'][860:926].replace(np.nan, 'Empty')
    data['State2'][982:1004] = data['State2'][982:1004].replace(np.nan, 'Empty')
    data['State2'][1042:] = data['State2'][1042:].replace(np.nan, 'Empty')

    data['State2'] = data['State2'].replace(np.nan, 'One person')

    print(data['State2'][596:612])
    print(data['State2'][612:650])

    data['State3'][0:48] = data['State3'][0:48].replace(np.nan, 'Empty')
    data['State3'][70:110] = data['State3'][70:110].replace(np.nan, 'Empty')
    data['State3'][134:172] = data['State3'][134:172].replace(np.nan, 'Empty')
    data['State3'][182:232] = data['State3'][182:232].replace(np.nan, 'Empty')
    data['State3'][264:328] = data['State3'][264:328].replace(np.nan, 'Empty')
    data['State3'][358:394] = data['State3'][358:394].replace(np.nan, 'Empty')
    data['State3'][404:524] = data['State3'][404:524].replace(np.nan, 'Empty')
    data['State3'][602:652] = data['State3'][602:652].replace(np.nan, 'Empty')
    data['State3'][672:712] = data['State3'][672:712].replace(np.nan, 'Empty')
    data['State3'][808:846] = data['State3'][808:846].replace(np.nan, 'Empty')
    data['State3'][894:910] = data['State3'][894:910].replace(np.nan, 'Empty')
    data['State3'][936:948] = data['State3'][936:948].replace(np.nan, 'Empty')
    data['State3'][956:1050] = data['State3'][956:1050].replace(np.nan, 'Empty')
    data['State3'][1076:] = data['State3'][1076:].replace(np.nan, 'Empty')

    data['State3'] = data['State3'].replace(np.nan, 'Two people')

    print(data['State3'][808:846])
    print(data['State3'][846:900])

    data.to_excel("processedData3.xlsx")

def data_plot1():
    data = pd.read_csv('data/dataset1.csv')

