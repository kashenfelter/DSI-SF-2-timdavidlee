import pandas as pd
import numpy as np
import datetime
import cPickle as pickle
import unidecode



# takes in list of text fields, manipulates them

class hardSkillParser(object):


    def __init__(self,list_of_text):
        #expecting a single value data frame
        self.text_dict = [{'text':self._fixText(x)} for x in list_of_text] 
        
    def load(self,list_of_text):
        self.text_dict = [{'text':self._fixText(x)} for x in list_of_text]
        
#===========================================================================        
#===========================================================================        
#===========================================================================        
#===========================================================================
    # some of hte HTML data that was pulled needs to be 
    # decoded from unicode and either falls in one of the
    # two combinations: decode UTF8 then UNICODE decode
    
    def _fixText(self, input_string):
        prevlower = False
        newText = ''
        for x in input_string:
            if x=='*':
                newText +='\n'
            elif x.isalpha()==False:
                prevlower=0
            elif x.islower():
                prevlower = True

            if (prevlower == True) & (x.isupper()):
                newText += '.\n' + x
                prevlower = False        
            else:
                newText += x
        return newText
            
            
    # similar to the previous function but 
    # designed for pandas MAP and APPLY functions
    
    def _cleanLI(self, yy):
        newlist = []
        for y in yy:
            try:
                newlist.append(unidecode.unidecode(y.decode('utf-8')).lower())
            except:
                try:
                    newlist.append(y.decode('utf-8').lower())
                except:
                    newlist.append(y.lower())
        return newlist       
   
    
    # =========================================================
    # pulls newline delinated lines, since these are typically
    # the bullet point descriptions
    # and have a higher percentage of relevant details
    # relating to job instead of disclaimers, or 
    # company background or descriptions
    
    def _parseLI(self,y):
        LI = []
        y  = y.replace('i.e.','-ie-').replace('e.g.','-eg-')
        for x in y.split('\n'):
            ct = len(x.strip().split('.'))
            if ct ==1:
                if len(x)>0:
                    LI.append(x)
            elif len(x.strip().split('.')[1])<=1:
                if len(x.strip().split('.')[0])>0:
                    LI.append(x.strip().split('.')[0])
        return LI

    # =========================================================    
    # this will provide a second level of detail
    # from the LI items pulled, this will go through and further
    # pull out more hard topics
    
    def _splitLI(self,y,phrase):
        try:
            if phrase in ('to','in'):
                return y.split(' '+phrase+' ')[1]
            else:
                return y.split(phrase)[1]
        except:
            return ''

#===========================================================================        
#===========================================================================        
#===========================================================================        
#===========================================================================
     
    def fit(self):
        start = datetime.datetime.now()
        terms = ['in','including','knowledge of','experience with', 'understanding of', 'to','develop ','design ','requirements']
        for x in self.text_dict:
            x['LI'] =self._cleanLI(self._parseLI(x['text']))
            hardskill = []
            for y in terms:
                x[y] = [self._splitLI(LI,y) for LI in x['LI']]
                x[y] = [z for z in x[y] if z!='']
                hardskill.extend(x[y])
            x['hardskill'] = hardskill
            x['LI_text'] = ' '.join(x['LI'])
            x['hardskill_text'] = ' '.join(x['hardskill'])
        print datetime.datetime.now()-start
        
        self.parsed_df = pd.DataFrame(self.text_dict)[['text','LI','LI_text','hardskill','hardskill_text']]
    
    def save(self,prefix='_'):
        suffix = str(int(time.mktime(datetime.datetime.now().timetuple())))[-6:]
        with open("008_"+prefix+"_parsed_text_dict_"+suffix+".p",'wb') as f:
            pickle.dump(self.text_dict,f)
        with open("008_"+prefix+"_parsed_df_"+suffix+".p",'wb') as f:
            pickle.dump(self.parsed_df,f)

