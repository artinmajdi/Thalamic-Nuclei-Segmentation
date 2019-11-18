import json



DirSave = '/array/ssd/msmajdi/experiments/keras/exp7/models/sE12_Cascade_FM40_Res_Unet2_NL3_LS_MyLogDice_US1_CSFn2_Init_Main_wBiasCorrection_CV_a/MultiClass_24567891011121314/sd0'

with open(DirSave + '/UserInfo.json', "r") as j:
    data = json.load(j)



print(data)