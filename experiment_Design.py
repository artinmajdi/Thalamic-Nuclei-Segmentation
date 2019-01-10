
AllExperimentsList = {
    1: dict(nucleus_Index = [6] , GPU_Index = 5 , lossFunctionIx = 1),
    2: dict(nucleus_Index = [6] , GPU_Index = 6 , lossFunctionIx = 2),
    3: dict(nucleus_Index = [6] , GPU_Index = 7 , lossFunctionIx = 3),
    4: dict(nucleus_Index = [8] , GPU_Index = 5 , lossFunctionIx = 1),
    5: dict(nucleus_Index = [8] , GPU_Index = 6 , lossFunctionIx = 2),
    6: dict(nucleus_Index = [8] , GPU_Index = 7 , lossFunctionIx = 3),
}


def main(params):

    AllParamsList = {}
    for Keys in list(AllExperimentsList.keys()):

        AllParamsList[Keys] = params
        for entry in list(AllExperimentsList[Keys].keys()):
            AllParamsList[Keys].UserInfo[entry] = AllExperimentsList[Keys][entry]
            print(AllParamsList[Keys].UserInfo[entry])

    return AllParamsList
