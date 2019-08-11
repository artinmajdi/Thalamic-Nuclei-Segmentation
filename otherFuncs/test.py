class mergingHDValues:
    def __init__(self, Info):
        self.Info   = Info
        self.writer = pd.ExcelWriter(Info.Experiment.address + '/results/All_HD.xlsx', engine='xlsxwriter')
        
        def save_TagList(self):
            pd.DataFrame(data=self.Info.Experiment.TagsList).to_excel(self.writer, sheet_name='TagsList')
            class All_Subjs_Ns:
                def insertNuclei_Excel(self, sheetName):
                    A = pd.DataFrame()
                    A['Nuclei'] = All_Nuclei_Names # self.Info.Nuclei_Names[1:18]
                    A.to_excel(self.writer, sheet_name=sheetName)

                    class out:
                        pd = A
                        sheet_name = sheetName
                    return out()
                                    
                ET    = insertNuclei_Excel(self, 'AllHDs_ET')
                Main  = insertNuclei_Excel(self, 'AllHDs_Main')
                CSFn1 = insertNuclei_Excel(self, 'AllHDs_CSFn1')
                CSFn2 = insertNuclei_Excel(self, 'AllHDs_CSFn2')
                SRI   = insertNuclei_Excel(self, 'AllHDs_SRI')

            self.All_Subjs_Ns = All_Subjs_Ns()
        save_TagList(self)

        def func_Load_Subexperiment(self):
            
            # if self.plane.Flag:

            self.subExperiment.address = self.Info.Experiment.address + '/results/' + self.subExperiment.name +'/'+ self.plane.name
            
            def func_1subject_HDs(self):                    
                                                                     
                def func_Search_Over_Single_Class(HD_Single):
                    for name in All_Nuclei_Names:
                        Dir_subject = self.subExperiment.address + '/' + self.subject + '/HD_' + name +'.txt'
                        if os.path.isfile(Dir_subject): HD_Single[name] = math.ceil( np.loadtxt(Dir_subject)[1] *1e3)/1e3
                    return HD_Single
                
                def func_Search_Over_Multi_Class(HD_Single):

                    for HDTag in ['/HD_All' , '/HD_All_Groups' , '/HD_All_Medial' , '/HD_All_lateral' , '/HD_All_posterior']:
                        Dir_subject = self.subExperiment.address + '/' + self.subject + HDTag + '.txt'
                        if os.path.isfile(Dir_subject): 
                            A = np.loadtxt(Dir_subject)

                            if not isinstance(A[0],np.ndarray): 
                                HD_Single[smallFuncs.Nuclei_Class(index=A[0], method = 'HCascade').name] = math.ceil(  A[1]*1e3 )/1e3
                            else:
                                for id, nIx in enumerate(A[:,0]):
                                    HD_Single[smallFuncs.Nuclei_Class(index=nIx, method = 'HCascade').name] = math.ceil(  A[id,1]*1e3 )/1e3

                    return HD_Single

                HD_Single = {'subject':self.subject} 
                HD_Single = func_Search_Over_Single_Class(HD_Single)   
                HD_Single = func_Search_Over_Multi_Class(HD_Single)

                return HD_Single                                    
            sE_HDs = np.array([  func_1subject_HDs(self)  for self.subject in self.plane.subject_List  ])

            if len(sE_HDs) > 0:
                def save_HDs_subExp_In_ExcelFormat(self , sE_HDs):
                    pd_sE = pd.DataFrame()

                    pd_sE['subject'] = [s['subject'] for s in sE_HDs]
                    for nucleus in All_Nuclei_Names:

                        # try:
                        A = np.nan*np.ones(len(sE_HDs))
                        for ix, s in enumerate(sE_HDs):
                            if nucleus in s: A[ix] = s[nucleus]
                        
                        pd_sE[nucleus] = A

                        #     if nucleus in sE_HDs[0]: pd_sE[nucleus] = [s[nucleus] for s in sE_HDs] # .astype(np.float16)
                        # except Exception as e:
                            # print(e)

                    pd_sE.to_excel(  self.writer, sheet_name=self.plane.tagList[0] )    
                save_HDs_subExp_In_ExcelFormat(self , sE_HDs)
                
                def divideSubjects_BasedOnModality(self, sE_HDs):
                    class subjectHD:
                        ET   = []
                        Main = []
                        CSFn1 = []
                        CSFn2 = []
                        SRI  = []

                    # for sIx , subject in enumerate(sE_HDs[:,0]):
                    for s in sE_HDs:
                        if 'ET' in s['subject']:     subjectHD.ET.append(s)
                        elif 'CSFn1' in s['subject']: subjectHD.CSFn1.append(s)
                        elif 'CSFn2' in s['subject']: subjectHD.CSFn2.append(s)
                        elif 'SRI'  in s['subject']: subjectHD.SRI.append(s)
                        else:                        subjectHD.Main.append(s)

                    def func_Average(subjectHDList):

                        Average_HDs = np.nan*np.ones(len(All_Nuclei_Names))
                        for ix, nucleus in enumerate(All_Nuclei_Names):
                            A = np.nan*np.ones(len(subjectHDList))
                            for ct, s in enumerate(subjectHDList):
                                if nucleus in s: A[ct] = s[nucleus]

                            Average_HDs[ix] = np.round(1e3*np.nanmean(A, axis=0))/1e3
                                                        
                        return Average_HDs

                    tag = self.plane.direction +'-' + self.plane.tagIndex    
                    # for dataset in ['ET' , 'Main' , '1' , 'SRI']:
                    #     A = subjectHD.__getattribute__(dataset)
                    #     if len(A) > 0  : self.All_Subjs_Ns.__getattribute__(dataset).pd[tag] = func_Average(A)                   
                    #     # self.All_Subjs_Ns.__setattribute__(dataset) = A

                    if len(subjectHD.ET) > 0   : self.All_Subjs_Ns.ET.pd[  tag]  = func_Average(subjectHD.ET)   
                    if len(subjectHD.Main) > 0 : self.All_Subjs_Ns.Main.pd[tag]  = func_Average(subjectHD.Main) 
                    if len(subjectHD.CSFn1) > 0: self.All_Subjs_Ns.CSFn1.pd[tag] = func_Average(subjectHD.CSFn1) 
                    if len(subjectHD.CSFn2) > 0: self.All_Subjs_Ns.CSFn2.pd[tag] = func_Average(subjectHD.CSFn2) 
                    if len(subjectHD.SRI) > 0  : self.All_Subjs_Ns.SRI.pd[ tag]  = func_Average(subjectHD.SRI)  
                divideSubjects_BasedOnModality(self, sE_HDs)

        def loopOver_Subexperiments(self):
            class smallActions():                                                                                              
                def add_space(self):
                    self.All_Subjs_Ns.ET.pd[  self.subExperiment.Tag[0]]  = np.nan*np.ones(17)
                    self.All_Subjs_Ns.Main.pd[self.subExperiment.Tag[0]]  = np.nan*np.ones(17)
                    self.All_Subjs_Ns.CSFn1.pd[self.subExperiment.Tag[0]] = np.nan*np.ones(17)
                    self.All_Subjs_Ns.CSFn2.pd[self.subExperiment.Tag[0]] = np.nan*np.ones(17)
                    self.All_Subjs_Ns.SRI.pd[ self.subExperiment.Tag[0]]  = np.nan*np.ones(17)

                def save_All_HDs(PD):
                    PD.ET.pd.to_excel( self.writer , sheet_name=PD.ET.sheet_name  )
                    PD.Main.pd.to_excel(self.writer, sheet_name=PD.Main.sheet_name)
                    PD.CSFn1.pd.to_excel(self.writer, sheet_name=PD.CSFn1.sheet_name)
                    PD.CSFn2.pd.to_excel(self.writer, sheet_name=PD.CSFn2.sheet_name)
                    PD.SRI.pd.to_excel( self.writer, sheet_name=PD.SRI.sheet_name )

            A = zip(self.Info.Experiment.List_subExperiments , self.Info.Experiment.TagsList)
            for self.subExperiment, tag in tqdm( A , desc='HDs:'):
                # try: 
                self.subExperiment.Tag = tag
                # print(self.subExperiment.name)
                smallActions.add_space(self)
                for self.plane in self.subExperiment.multiPlanar:
                    if self.plane.Flag:
                        # print(self.subExperiment.name , self.plane.name)
                        # try: 
                        func_Load_Subexperiment(self)
                        # except: print('failed' ,self.subExperiment )                                            
                # except Exception as e:
                #     print(e)

            smallActions.save_All_HDs(self.All_Subjs_Ns)

            self.writer.close()
        loopOver_Subexperiments(self)

