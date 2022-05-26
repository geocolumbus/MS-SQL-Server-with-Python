ALTER PROC [dbo].[sp_predictDRPRIMEAdmissions]
AS
BEGIN
	SET NOCOUNT ON
	DECLARE @PREDICTIONS TABLE(
	id int,
	race nvarchar(4000) ,
	gender nvarchar(4000) ,
	age nvarchar(4000) ,
	weight nvarchar(4000) ,
	admission_type_id nvarchar(4000) ,
	discharge_disposition_id nvarchar(4000) ,
	admission_source_id nvarchar(4000) ,
	time_in_hospital nvarchar(4000) ,
	payer_code nvarchar(4000) ,
	medical_specialty nvarchar(4000) ,
	num_lab_procedures nvarchar(4000) ,
	num_procedures nvarchar(4000) ,
	num_medications nvarchar(4000) ,
	number_outpatient nvarchar(4000) ,
	number_emergency nvarchar(4000) ,
	number_inpatient nvarchar(4000) ,
	diag_1 nvarchar(4000) ,
	diag_2 nvarchar(4000) ,
	diag_3 nvarchar(4000) ,
	number_diagnoses nvarchar(4000) ,
	max_glu_serum nvarchar(4000) ,
	A1Cresult nvarchar(4000) ,
	metformin nvarchar(4000) ,
	repaglinide nvarchar(4000) ,
	nateglinide nvarchar(4000) ,
	chlorpropamide nvarchar(4000) ,
	glimepiride nvarchar(4000) ,
	acetohexamide nvarchar(4000) ,
	glipizide nvarchar(4000) ,
	glyburide nvarchar(4000) ,
	tolbutamide nvarchar(4000) ,
	pioglitazone nvarchar(4000) ,
	rosiglitazone nvarchar(4000) ,
	acarbose nvarchar(4000) ,
	miglitol nvarchar(4000) ,
	troglitazone nvarchar(4000) ,
	tolazamide nvarchar(4000) ,
	examide nvarchar(4000) ,
	citoglipton nvarchar(4000) ,
	insulin nvarchar(4000) ,
	glyburidemetformin nvarchar(4000),
	glipizidemetformin nvarchar(4000),
	glimepiridepioglitazone nvarchar(4000) ,
	metforminrosiglitazone nvarchar(4000) ,
	metforminpioglitazone nvarchar(4000) ,
	change nvarchar(4000) ,
	diabetesMed nvarchar(4000) ,
	readmitted nvarchar(4000) ,
	diag_1_desc nvarchar(4000) ,
	diag_2_desc nvarchar(4000) ,
	diag_3_desc nvarchar(4000) ,
	prediction nvarchar(500),
	predictionLabel nvarchar(500) ,
	predictionValue nvarchar(500) ,
	predictionDate nvarchar(500),
	predictionThreshold nvarchar(max)
)

DECLARE @PSCRIPT NVARCHAR(MAX);
SET @PSCRIPT = N'
import pandas as pd
import datarobot as dr
from datetime import date
import time
from pandas import DataFrame
import urllib3
import requests
import json

# DR Prime specific
import calendar
from datetime import datetime
from collections import namedtuple
import re
import sys
import time
import os

import numpy as np
import pandas as pd

# 0. to test outside of MS SQL just initialize the InputDataSet
#InputDataSet = pd.read_csv("../SampleData/10k_diabetes.csv",nrows=10000)


# 1. DR Prime model definition
PY3 = sys.version_info[0] == 3
if PY3:
    string_types = str,
    text_type = str
    long_type = int
else:
    string_types = basestring,
    text_type = unicode
    long_type = long

def predict(row):
    admission_type_id = row[u"admission_type_id"]
    chlorpropamide = row[u"chlorpropamide"]
    weight = row[u"weight"]
    repaglinide = row[u"repaglinide"]
    payer_code = row[u"payer_code"]
    acarbose = row[u"acarbose"]
    round_num_medications = np.float32(row[u"num_medications"])
    rosiglitazone = row[u"rosiglitazone"]
    diag_2_desc_words = bag_of_words(row[u"diag_2_desc"])
    round_time_in_hospital = np.float32(row[u"time_in_hospital"])
    diag_3_desc_words = bag_of_words(row[u"diag_3_desc"])
    admission_source_id = row[u"admission_source_id"]
    nateglinide = row[u"nateglinide"]
    round_number_diagnoses = np.float32(row[u"number_diagnoses"])
    glyburide = row[u"glyburide"]
    metformin = row[u"metformin"]
    glyburidemetformin = row[u"glyburidemetformin"]
    diag_1_desc_words = bag_of_words(row[u"diag_1_desc"])
    pioglitazone = row[u"pioglitazone"]
    round_id = np.float32(row[u"id"])
    round_num_lab_procedures = np.float32(row[u"num_lab_procedures"])
    A1Cresult = row[u"A1Cresult"]
    glimepiride = row[u"glimepiride"]
    max_glu_serum = row[u"max_glu_serum"]
    diabetesMed = row[u"diabetesMed"]
    discharge_disposition_id = row[u"discharge_disposition_id"]
    change = row[u"change"]
    round_number_emergency = np.float32(row[u"number_emergency"])
    gender = row[u"gender"]
    age = row[u"age"]
    medical_specialty = row[u"medical_specialty"]
    race = row[u"race"]
    insulin = row[u"insulin"]
    round_number_outpatient = np.float32(row[u"number_outpatient"])
    round_number_inpatient = np.float32(row[u"number_inpatient"])
    diag_2 = row[u"diag_2"]
    diag_3 = row[u"diag_3"]
    round_num_procedures = np.float32(row[u"num_procedures"])
    diag_1 = row[u"diag_1"]
    return sum([
        -0.4384309,
           0.015860811490601561813 * (int(u"congestive" in diag_3_desc_words)),
         -0.0074950592092998483823 * (not age == u"[70-80)" and 
                                     not discharge_disposition_id == u"Expired" and 
                                     not insulin == u"Up" and 
                                     round_number_inpatient <= 2.5),
            0.11203456741532756558 * (admission_type_id == u"nan"),
            0.43781956478484751472 * (diag_2 == u"276"),
          -0.013039860025011142802 * (not age == u"[50-60)" and 
                                     glimepiride == u"No" and 
                                     not medical_specialty == u"nan" and 
                                     round_number_inpatient <= 0.5),
           -0.46324921739905555729 * (int(u"pneumonia" in diag_2_desc_words)),
           0.036186649452210943589 * (not age == u"[70-80)" and 
                                     change == u"Ch" and 
                                     not discharge_disposition_id == u"Expired" and 
                                     not gender == u"Female"),
            0.13421089929516885619 * (not age == u"[50-60)" and 
                                     round_number_inpatient > 0.5 and 
                                     round_number_diagnoses > 3.5),
            0.15361047621742099367 * (not A1Cresult == u"Norm" and 
                                     medical_specialty == u"nan" and 
                                     round_num_lab_procedures > 57.5 and 
                                     round_num_medications <= 32.5),
           0.022912352562252730898 * (int(u"congestive" in diag_1_desc_words)),
          -0.061157197586349978935 * (int(u"hypothyroidism" in diag_3_desc_words)),
            0.11534590647594542001 * (not age == u"[50-60)" and 
                                     not insulin == u"Steady" and 
                                     round_number_outpatient <= 1.5 and 
                                     round_number_inpatient > 0.5),
          -0.085592556879727291208 * (diag_1 == u"682"),
           -0.10629281325221746901 * (not age == u"[70-80)" and 
                                     not insulin == u"Up" and 
                                     not medical_specialty == u"Nephrology"),
          -0.004472809808511699764 * (not age == u"[70-80)" and 
                                     not medical_specialty == u"Emergency/Trauma" and 
                                     not medical_specialty == u"Orthopedics" and 
                                     not payer_code == u"CP"),
         0.00084762984285389564396 * (int(u"or" in diag_2_desc_words)),
          -0.065292915051856553754 * (age == u"[50-60)" and 
                                     not insulin == u"Down" and 
                                     not race == u"AfricanAmerican" and 
                                     round_num_lab_procedures <= 54.5),
           0.078947270876629616065 * (not age == u"[50-60)" and 
                                     round_num_medications > 11.5 and 
                                     round_number_emergency <= 1.5 and 
                                     round_number_inpatient > 0.5),
            0.12756390894803304459 * (not age == u"[50-60)" and 
                                     medical_specialty == u"nan" and 
                                     round_time_in_hospital <= 6.5),
           -0.09274007333481350257 * (age == u"[20-30)"),
            0.35124559231137042481 * (int(u"to" in diag_1_desc_words)),
           -0.01943705054099665458 * (not metformin == u"Up" and 
                                     rosiglitazone == u"No" and 
                                     round_number_inpatient <= 1.5),
           -0.39114063362093781651 * (8.5 < round_time_in_hospital <= 13.5 and 
                                     round_num_lab_procedures <= 67.5 and 
                                     round_num_medications > 10.5),
           -0.14018792700273688401 * (int(u"or" in diag_3_desc_words)),
            0.17883066794492250007 * (not A1Cresult == u"Norm" and 
                                     not discharge_disposition_id == u"small_count" and 
                                     not insulin == u"Steady" and 
                                     not payer_code == u"BC"),
          -0.015535404627278710799 * (not max_glu_serum == u">200" and 
                                     890.5 < round_id <= 975.0 and 
                                     round_time_in_hospital <= 13.5),
          -0.052892373007256653084 * (rosiglitazone == u"No"),
           -0.10630175513383373354 * (int(u"eye" in diag_3_desc_words)),
           -0.27733630113767365755 * (not admission_type_id == u"nan" and 
                                     not insulin == u"Down" and 
                                     round_num_lab_procedures > 35.5 and 
                                     round_number_inpatient <= 0.5),
          -0.054074301333397432889 * (not discharge_disposition_id == u"small_count" and 
                                     pioglitazone == u"No" and 
                                     round_num_lab_procedures <= 60.5 and 
                                     round_num_procedures <= 5.5),
            0.19599492214590946704 * (diag_2 == u"250.01"),
           -0.16558177993984379839 * (not admission_source_id == u"Clinic Referral" and 
                                     not age == u"[60-70)" and 
                                     not age == u"[70-80)" and 
                                     not medical_specialty == u"Cardiology"),
            -0.0143585564558746763 * (not A1Cresult == u"Norm" and 
                                     not discharge_disposition_id == u"Expired" and 
                                     rosiglitazone == u"No" and 
                                     round_number_inpatient <= 1.5),
           -0.20420975830138970997 * (admission_source_id == u"Clinic Referral"),
           0.010539491228712505322 * (int(u"anterolateral" in diag_1_desc_words)),
           -0.11727008541204825276 * (not admission_source_id == u"small_count" and 
                                     not admission_type_id == u"nan" and 
                                     weight == u"nan" and 
                                     round_num_medications > 15.5),
            0.13359044537763367644 * (not age == u"[50-60)" and 
                                     not payer_code == u"CP" and 
                                     not race == u"AfricanAmerican" and 
                                     round_num_medications > 8.5),
            0.06202105516863947593 * (not admission_source_id == u"Clinic Referral" and 
                                     not age == u"[30-40)" and 
                                     not discharge_disposition_id == u"small_count" and 
                                     round_num_lab_procedures > 57.5),
         0.00052441772327039258301 * (not insulin == u"No" and 
                                     not medical_specialty == u"Family/GeneralPractice" and 
                                     round_num_medications <= 15.5),
              0.119478851084431556 * (round_id > 434.5 and 
                                     round_number_diagnoses > 6.5),
           0.043286285673435044574 * (not A1Cresult == u"Norm" and 
                                     not admission_source_id == u"Clinic Referral" and 
                                     not discharge_disposition_id == u"small_count" and 
                                     medical_specialty == u"nan"),
          0.0051624535593392051683 * (not A1Cresult == u"Norm" and 
                                     not age == u"[60-70)" and 
                                     medical_specialty == u"nan" and 
                                     round_id <= 879.0),
            0.14764785981317715691 * (int(u"hyperosmolality" in diag_2_desc_words)),
            0.10591421538691690729 * (payer_code == u"CP"),
             0.1036948490794850769 * (A1Cresult == u"nan" and 
                                     not admission_source_id == u"Transfer from a Skilled Nursing Facility (SNF)" and 
                                     round_num_procedures <= 3.5 and 
                                     round_num_medications <= 28.5),
            -0.1243149070833187847 * (glimepiride == u"No" and 
                                     round_number_outpatient <= 0.5 and 
                                     round_number_inpatient <= 0.5),
           0.032276515162026037098 * (diag_3 == u"518"),
           -0.21929504295376217593 * (int(u"to" in diag_3_desc_words)),
           -0.02938095082765793814 * (rosiglitazone == u"No" and 
                                     round_num_procedures <= 1.5 and 
                                     round_num_medications <= 34.0 and 
                                     round_number_emergency <= 0.5),
          -0.089071143759373236359 * (not admission_type_id == u"nan" and 
                                     not medical_specialty == u"Surgery-Cardiovascular/Thoracic" and 
                                     not payer_code == u"CP" and 
                                     round_num_procedures <= 3.5),
            0.60192582743926870137 * (int(u"cerebrovascular" in diag_3_desc_words)),
        -0.00043850979572884513265 * (not A1Cresult == u"Norm" and 
                                     not age == u"[20-30)" and 
                                     rosiglitazone == u"No" and 
                                     round_number_inpatient <= 1.5),
            0.16391936135792720131 * (A1Cresult == u"nan" and 
                                     not admission_source_id == u"Clinic Referral" and 
                                     not discharge_disposition_id == u"small_count" and 
                                     not payer_code == u"BC"),
            0.04821928428086907914 * (diag_2 == u"403"),
          -0.010131710470799765861 * (chlorpropamide == u"No"),
         0.00089044644987321445138 * (age == u"[70-80)" and 
                                     not insulin == u"Up" and 
                                     not max_glu_serum == u">200" and 
                                     round_num_medications <= 39.0),
           0.053610932073379281848 * (not medical_specialty == u"Cardiology" and 
                                     27.5 < round_id <= 669.5 and 
                                     round_number_diagnoses > 3.5),
          -0.017494781767194836353 * (not age == u"[70-80)" and 
                                     round_number_outpatient <= 0.5 and 
                                     round_number_diagnoses <= 6.5),
           -0.11411844198340416467 * (int(u"postsurgical" in diag_3_desc_words)),
            0.30218236634299155963 * (int(u"juvenile" in diag_3_desc_words)),
            0.09851453068678385494 * (age == u"[70-80)" and 
                                     not insulin == u"Steady" and 
                                     not insulin == u"Up"),
            0.11110186020931404893 * (not discharge_disposition_id == u"nan" and 
                                     179.0 < round_id <= 837.0 and 
                                     round_number_inpatient <= 0.5),
           -0.10178204652355138382 * (not max_glu_serum == u">200" and 
                                     not medical_specialty == u"Cardiology" and 
                                     round_id > 716.5 and 
                                     round_num_medications <= 34.0),
           -0.52770508532764104359 * (int(u"hemorrhage" in diag_3_desc_words)),
            0.16534686626491154615 * (int(u"abscess" in diag_3_desc_words)),
         0.00082614798319302856864 * (diag_1 == u"410"),
           0.095961625481998977238 * (int(u"device" in diag_1_desc_words)),
            0.22675523624767496278 * (int(u"care" in diag_2_desc_words)),
          -0.016662558366532739113 * (round_time_in_hospital <= 13.5 and 
                                     35.5 < round_num_lab_procedures <= 55.5 and 
                                     round_number_outpatient <= 0.5),
          -0.012323420370292475362 * (not payer_code == u"BC" and 
                                     repaglinide == u"No" and 
                                     round_id <= 379.5 and 
                                     round_number_diagnoses > 7.5),
          -0.092994858878561834081 * (not age == u"[70-80)" and 
                                     not pioglitazone == u"Steady" and 
                                     rosiglitazone == u"No" and 
                                     round_number_inpatient <= 0.5),
           -0.03685281532476072236 * (not age == u"[70-80)" and 
                                     not medical_specialty == u"Orthopedics" and 
                                     not payer_code == u"CP" and 
                                     round_time_in_hospital <= 2.5),
          -0.016428756494780490105 * (glimepiride == u"No" and 
                                     not medical_specialty == u"nan" and 
                                     not payer_code == u"nan" and 
                                     round_number_diagnoses <= 6.5),
             0.1924615993928360802 * (pioglitazone == u"small_count"),
           0.096592000564816926644 * (not age == u"[50-60)" and 
                                     diabetesMed == u"Yes" and 
                                     not discharge_disposition_id == u"Expired" and 
                                     medical_specialty == u"nan"),
          0.0027871304166787978784 * (diag_1 == u"996"),
            0.06259874949379103104 * (age == u"[70-80)" and 
                                     not insulin == u"Up" and 
                                     not medical_specialty == u"Orthopedics" and 
                                     not payer_code == u"CP"),
            0.14686033916388571696 * (not age == u"[70-80)" and 
                                     change == u"Ch" and 
                                     not gender == u"Female"),
           -0.02668194080981332178 * (int(u"cerebral" in diag_1_desc_words)),
           -0.55741951008617385277 * (diag_2 == u"424"),
           -0.05608823764813081203 * (discharge_disposition_id == u"nan"),
          -0.055269981173537888197 * (diag_3 == u"small_count"),
           0.054621204089228034273 * (int(u"hypertension" in diag_3_desc_words)),
           0.033827586044708833624 * (A1Cresult == u">7"),
           -0.08385657326541706702 * (round_time_in_hospital > 3.5 and 
                                     35.5 < round_num_lab_procedures <= 55.5 and 
                                     round_number_outpatient <= 0.5),
            0.38236709622664533104 * (rosiglitazone == u"small_count"),
          -0.086673143230843519014 * (not age == u"[50-60)" and 
                                     not medical_specialty == u"Emergency/Trauma" and 
                                     not payer_code == u"CP" and 
                                     round_id <= 434.5),
            0.23524899109989821921 * (not age == u"[50-60)" and 
                                     discharge_disposition_id == u"Discharged to home" and 
                                     medical_specialty == u"nan" and 
                                     round_num_medications > 11.5),
            0.25316433444947339382 * (diag_2 == u"428"),
         -0.0080207776684559917157 * (not age == u"[50-60)" and 
                                     glimepiride == u"No" and 
                                     not medical_specialty == u"nan" and 
                                     round_number_emergency <= 0.5),
           -0.01761600370219699313 * (not pioglitazone == u"Steady" and 
                                     35.5 < round_num_lab_procedures <= 60.5 and 
                                     round_number_outpatient <= 0.5),
          0.0076592989069876491956 * (int(u"unspecified" in diag_3_desc_words)),
          0.0019936015561797387763 * (acarbose == u"No"),
           -0.58760054037418485429 * (discharge_disposition_id == u"Expired"),
            0.08289179867653227729 * (diag_2 == u"427"),
          -0.017216584622630315415 * (not admission_type_id == u"nan" and 
                                     not age == u"[70-80)" and 
                                     not discharge_disposition_id == u"Expired" and 
                                     round_number_inpatient <= 1.5),
          -0.016542513086419304014 * (round_id > 29.5 and 
                                     35.5 < round_num_lab_procedures <= 57.5 and 
                                     round_number_outpatient <= 0.5),
            0.16603323213556370197 * (diabetesMed == u"Yes" and 
                                     round_id > 175.0 and 
                                     round_num_medications <= 15.5 and 
                                     round_number_inpatient <= 0.5),
          -0.072950470753279178515 * (not age == u"[60-70)" and 
                                     round_time_in_hospital <= 13.5 and 
                                     round_num_lab_procedures <= 28.5),
           0.044417063354977685818 * (round_time_in_hospital <= 4.5 and 
                                     round_num_lab_procedures > 57.5 and 
                                     round_num_procedures > 0.5),
            0.32676164085525033487 * (glimepiride == u"small_count"),
           -0.32237995695584320544 * (int(u"current" in diag_3_desc_words)),
           0.019231168841134165665 * (age == u"[80-90)"),
          -0.030021737138377295462 * (gender == u"Female" and 
                                     glyburide == u"No" and 
                                     round_id > 716.5),
           0.049231805628774896744 * (int(u"tract" in diag_3_desc_words)),
           -0.08895977081972183953 * (int(u"alteration" in diag_2_desc_words)),
            0.12488534087046909704 * (diag_3 == u"424"),
           -0.15160974553416672883 * (glimepiride == u"No" and 
                                     not medical_specialty == u"nan" and 
                                     round_number_inpatient <= 0.5 and 
                                     round_number_diagnoses <= 5.5),
           -0.11527464473878239193 * (int(u"myocardial" in diag_3_desc_words)),
         -0.0078481576355197129463 * (diag_1 == u"434"),
           0.010086663980036329052 * (not age == u"[90-100)" and 
                                     not medical_specialty == u"Family/GeneralPractice" and 
                                     round_id > 166.5 and 
                                     round_num_lab_procedures <= 60.5),
            0.30775013946612972404 * (int(u"tachycardia" in diag_1_desc_words)),
            0.05614407346462505638 * (round_number_inpatient),
            0.14111004517570599481 * (not admission_source_id == u"Transfer from a Skilled Nursing Facility (SNF)" and 
                                     not age == u"[50-60)" and 
                                     round_num_medications > 11.5 and 
                                     round_number_inpatient > 0.5),
           -0.03261903026868674671 * (not medical_specialty == u"nan" and 
                                     round_number_inpatient <= 0.5 and 
                                     round_number_diagnoses <= 6.5),
            0.51351605114977361133 * (int(u"implant" in diag_2_desc_words)),
          -0.001954898413352930929 * (nateglinide == u"No"),
           -0.04719268028801278797 * (int(u"disorder" in diag_3_desc_words)),
           -0.23698803422604097779 * (not age == u"[50-60)" and 
                                     not discharge_disposition_id == u"Discharged/transferred to another  type of inpatient care institution" and 
                                     round_id <= 467.5 and 
                                     round_number_inpatient <= 0.5),
           -0.13503423356214463991 * (int(u"ii" in diag_3_desc_words)),
            0.13832940605283863822 * (round_num_lab_procedures <= 35.5 and 
                                     round_num_medications <= 10.5),
           0.024327743393161911645 * (diag_3 == u"414"),
           0.054758473840879430539 * (not age == u"[50-60)" and 
                                     not discharge_disposition_id == u"Expired" and 
                                     not medical_specialty == u"Gastroenterology" and 
                                     not race == u"AfricanAmerican"),
          0.0061456513802474622282 * (not admission_type_id == u"nan" and 
                                     not age == u"[50-60)" and 
                                     not max_glu_serum == u"Norm" and 
                                     round_number_diagnoses > 3.5),
           0.067784094149088841563 * (not discharge_disposition_id == u"nan" and 
                                     round_time_in_hospital <= 13.5 and 
                                     round_num_lab_procedures > 46.5),
           -0.17624518807997047176 * (int(u"face" in diag_1_desc_words)),
           -0.12519510887541721034 * (not glimepiride == u"small_count" and 
                                     pioglitazone == u"No" and 
                                     round_number_outpatient <= 0.5 and 
                                     round_number_inpatient <= 0.5),
         -0.0016096374300518878978 * (pioglitazone == u"No" and 
                                     round_number_emergency <= 0.5),
           0.015538883749286403618 * (int(u"malignant" in diag_3_desc_words)),
           0.017430900893576906086 * (round_time_in_hospital <= 13.5 and 
                                     round_num_lab_procedures <= 35.5 and 
                                     round_num_medications <= 10.5),
            0.36600857551394694323 * (int(u"infection" in diag_3_desc_words)),
           -0.14480141340608937428 * (int(u"postoperative" in diag_3_desc_words)),
          0.0055356451405178357106 * (int(u"ii" in diag_2_desc_words)),
           -0.13813690907456011026 * (int(u"in" in diag_2_desc_words)),
          -0.045619225093861549836 * (not A1Cresult == u">7" and 
                                     not age == u"[60-70)" and 
                                     medical_specialty == u"nan" and 
                                     round_num_medications <= 21.5),
          -0.084012042810474737986 * (35.5 < round_num_lab_procedures <= 47.5 and 
                                     round_num_medications <= 19.5 and 
                                     round_number_inpatient <= 1.5),
            0.42398594088457364215 * (weight == u"[75-100)"),
          -0.019776757825660552792 * (int(u"due" in diag_3_desc_words)),
          -0.012881216686764916657 * (metformin == u"Steady"),
            0.28935118811838261843 * (int(u"renal" in diag_3_desc_words)),
            0.03976935494177775976 * (not admission_type_id == u"nan" and 
                                     27.5 < round_id <= 837.0 and 
                                     round_time_in_hospital <= 9.5),
            0.10052374405025027437 * (medical_specialty == u"Nephrology"),
          -0.067337232901757929082 * (not age == u"[50-60)" and 
                                     pioglitazone == u"No" and 
                                     round_num_lab_procedures <= 60.5 and 
                                     round_number_outpatient <= 0.5),
          -0.025073303773566240488 * (pioglitazone == u"No" and 
                                     round_id <= 466.5 and 
                                     round_number_inpatient <= 0.5 and 
                                     round_number_diagnoses > 3.5),
            0.25227300093050947227 * (diag_2 == u"585"),
           -0.16551047932209525526 * (not max_glu_serum == u">200" and 
                                     not payer_code == u"MC" and 
                                     round_id > 716.5),
            -0.1505133969709279429 * (not admission_type_id == u"nan" and 
                                     not payer_code == u"CP" and 
                                     round_num_procedures <= 3.5),
            0.21581490932659475046 * (not admission_source_id == u"Transfer from a Skilled Nursing Facility (SNF)" and 
                                     not max_glu_serum == u"Norm" and 
                                     round_id > 27.5 and 
                                     round_number_diagnoses > 3.5),
           -0.05025791422047965612 * (int(u"without" in diag_3_desc_words)),
            -0.2659384378367185553 * (diag_2 == u"518"),
            0.11349193454779936407 * (int(u"myocardial" in diag_1_desc_words)),
          0.0088318573223481900564 * (not admission_source_id == u"Clinic Referral" and 
                                     age == u"[60-70)" and 
                                     round_time_in_hospital > 2.5 and 
                                     round_num_procedures <= 4.5),
          -0.021116403161830700486 * (not age == u"[70-80)" and 
                                     pioglitazone == u"No" and 
                                     round_num_lab_procedures > 35.5 and 
                                     round_number_diagnoses <= 6.5),
             0.1188287776482982866 * (not admission_type_id == u"Elective" and 
                                     not max_glu_serum == u"Norm" and 
                                     633.0 < round_id <= 716.5),
           -0.26128693183091239449 * (int(u"infection" in diag_2_desc_words)),
         -0.0048382574619572765728 * (glyburidemetformin == u"No"),
           -0.01673473684709969253 * (glimepiride == u"No" and 
                                     round_number_inpatient <= 0.5),
           -0.17921615391164513742 * (glimepiride == u"No" and 
                                     pioglitazone == u"No" and 
                                     round_num_lab_procedures > 35.5 and 
                                     round_number_inpatient <= 0.5),
           0.047926245862613264803 * (diabetesMed == u"Yes" and 
                                     not max_glu_serum == u">200" and 
                                     not medical_specialty == u"Family/GeneralPractice" and 
                                     round_id <= 890.5),
           0.010357054686368601798 * (not payer_code == u"BC" and 
                                     not payer_code == u"MC" and 
                                     379.5 < round_id <= 716.5)    ])

def get_type_conversion():
    return {}
INDICATOR_COLS = []

IMPUTE_VALUES = {
    u"number_emergency": 0.000000,
    u"time_in_hospital": 4.000000,
    u"number_diagnoses": 7.000000,
    u"num_lab_procedures": 44.000000,
    u"number_outpatient": 0.000000,
    u"num_medications": 14.000000,
    u"number_inpatient": 0.000000,
    u"num_procedures": 1.000000,
    u"id": 466.500000,}


def bag_of_words(text):
    
    if type(text) == float:
        return set()

    return set(word.lower() for word in
               re.findall(r"\w+", text, re.UNICODE | re.IGNORECASE))


def parse_date(x, date_format):
    
    try:
        # float values no longer pass isinstance(x, np.float64)
        if isinstance(x, (np.float64, float)):
            x = long_type(x)
        if "%f" in date_format and date_format.startswith("v2"):
            temp = str(x)
            if re.search("[\+-][0-9]+$", temp):
                temp = re.sub("[\+-][0-9]+$", "", temp)

            date_format = date_format[2:]
            dt = datetime.strptime(temp, date_format)
            sec = calendar.timegm(dt.timetuple())
            return sec * 1000 + dt.microsecond // 1000
        elif "%M" in date_format:
            temp = str(x)
            if re.search("[\+-][0-9]+$", temp):
                temp = re.sub("[\+-][0-9]+$", "", temp)

            return calendar.timegm(datetime.strptime(temp, date_format).timetuple())
        else:
            return datetime.strptime(str(x), date_format).toordinal()
    except:
        return float("nan")


def parse_percentage(s):
    
    if isinstance(s, float):
        return s
    if isinstance(s, int):
        return float(s)
    try:
        return float(s.replace("%", ""))
    except:
        return float("nan")

def parse_nonstandard_na(s):

    try:
        ret = float(s)
        if np.isinf(ret):
            return float("nan")
        return ret
    except:
        return float("nan")

def parse_length(s):
    
    try:
        if "\"" in s and "''" in s:
            sp = s.split("''")
            return float(sp[0]) * 12 + float(sp[1].replace("\"", ""))
        else:
            if "''" in s:
                return float(s.replace("''", "")) * 12
            else:
                return float(s.replace("\"", ""))
    except:
        return float("nan")

def parse_currency(s):
    
    if not isinstance(s, text_type):
        return float("nan")
    s = re.sub(u"[\$\u20AC\u00A3\uFFE1\u00A5\uFFE5]|(EUR)", "", s)
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return float("nan")


def parse_currency_replace_cents_period(val, currency_symbol):
    try:
        if np.isnan(val):
            return val
    except TypeError:
        pass
    if not isinstance(val, string_types):
        raise ValueError("Found wrong value for currency: {}".format(val))
    try:
        val = val.replace(currency_symbol, "", 1)
        val = val.replace(" ", "")
        val = val.replace(",", "")
        val = float(val)
    except ValueError:
        val = float("nan")
    return val


def parse_currency_replace_cents_comma(val, currency_symbol):
    try:
        if np.isnan(val):
            return val
    except TypeError:
        pass
    if not isinstance(val, string_types):
        raise ValueError("Found wrong value for currency: {}".format(val))
    try:
        val = val.replace(currency_symbol, "", 1)
        val = val.replace(" ", "")
        val = val.replace(".", "")
        val = val.replace(",", ".")
        val = float(val)
    except ValueError:
        val = float("nan")
    return val


def parse_currency_replace_no_cents(val, currency_symbol):
    try:
        if np.isnan(val):
            return val
    except TypeError:
        pass
    if not isinstance(val, string_types):
        raise ValueError("Found wrong value for currency: {}".format(val))
    try:
        val = val.replace(currency_symbol, "", 1)
        val = val.replace(" ", "")
        val = val.replace(",", "")
        val = val.replace(".", "")
        val = float(val)
    except ValueError:
        val = float("nan")
    return val

def parse_numeric_types(ds):
    TYPE_CONVERSION = get_type_conversion()
    for col in ds.columns:
        if col in TYPE_CONVERSION:
            convert_func = TYPE_CONVERSION[col]["convert_func"]
            convert_args = TYPE_CONVERSION[col]["convert_args"]
            ds[col] = ds[col].apply(convert_func, args=convert_args)
    return ds

def sanitize_name(name):
    safe = name.strip().replace("-", "_").replace("$", "_").replace(".", "_")
    safe = safe.replace("{", "_").replace("}", "_")
    safe = safe.replace("\"", "_")
    return safe


def rename_columns(ds):
    new_names = {}
    existing_names = set()
    disambiguation = {}
    blank_index = 0
    for old_col in ds.columns:
        col = sanitize_name(old_col)
        if col == "":
            col = "Unnamed: %d" % blank_index
            blank_index += 1
        if col in existing_names:
            suffix = "_%d" % disambiguation.setdefault(col, 1)
            disambiguation[col] += 1
            col = col + suffix
        existing_names.add(col)
        new_names[old_col] = col
    ds.rename(columns=new_names, inplace=True)
    return ds

def add_missing_indicators(ds):
    for col in INDICATOR_COLS:
        ds[col + "-mi"] = ds[col].isnull().astype(int)
    return ds

def impute_values(ds):
    for col in ds:
        if col in IMPUTE_VALUES:
            ds.loc[ds[col].isnull(), col] = IMPUTE_VALUES[col]
    return ds

BIG_LEVELS = {
    u"chlorpropamide": [
        u"No",
    ],
    u"weight": [
        u"[50-75)",
        u"[75-100)",
    ],
    u"repaglinide": [
        u"No",
    ],
    u"payer_code": [
        u"BC",
        u"CP",
        u"HM",
        u"MC",
        u"MD",
        u"SP",
        u"UN",
    ],
    u"acarbose": [
        u"No",
    ],
    u"rosiglitazone": [
        u"No",
        u"Steady",
    ],
    u"admission_source_id": [
        u"Clinic Referral",
        u"Emergency Room",
        u"Physician Referral",
        u"Transfer from a Skilled Nursing Facility (SNF)",
        u"Transfer from a hospital",
        u"Transfer from another health care facility",
    ],
    u"medical_specialty": [
        u"Cardiology",
        u"Emergency/Trauma",
        u"Family/GeneralPractice",
        u"Gastroenterology",
        u"InternalMedicine",
        u"Nephrology",
        u"Orthopedics",
        u"Orthopedics-Reconstructive",
        u"PhysicalMedicineandRehabilitation",
        u"Psychiatry",
        u"Surgery-Cardiovascular/Thoracic",
        u"Surgery-General",
    ],
    u"metformin": [
        u"No",
        u"Steady",
        u"Up",
    ],
    u"glyburidemetformin": [
        u"No",
    ],
    u"pioglitazone": [
        u"No",
        u"Steady",
    ],
    u"A1Cresult": [
        u">7",
        u">8",
        u"Norm",
    ],
    u"glimepiride": [
        u"No",
        u"Steady",
    ],
    u"max_glu_serum": [
        u">200",
        u"Norm",
    ],
    u"diabetesMed": [
        u"No",
        u"Yes",
    ],
    u"discharge_disposition_id": [
        u"Discharged to home",
        u"Discharged/transferred to SNF",
        u"Discharged/transferred to another  type of inpatient care institution",
        u"Discharged/transferred to another rehab fac including rehab units of a hospital.",
        u"Discharged/transferred to home with home health service",
        u"Expired",
        u"Not Mapped",
    ],
    u"glyburide": [
        u"No",
        u"Steady",
        u"Up",
    ],
    u"change": [
        u"Ch",
        u"No",
    ],
    u"gender": [
        u"Female",
        u"Male",
    ],
    u"age": [
        u"[20-30)",
        u"[30-40)",
        u"[40-50)",
        u"[50-60)",
        u"[60-70)",
        u"[70-80)",
        u"[80-90)",
        u"[90-100)",
    ],
    u"nateglinide": [
        u"No",
    ],
    u"race": [
        u"AfricanAmerican",
        u"Caucasian",
        u"Hispanic",
        u"Other",
    ],
    u"insulin": [
        u"Down",
        u"No",
        u"Steady",
        u"Up",
    ],
    u"diag_1": [
        u"276",
        u"278",
        u"38",
        u"410",
        u"414",
        u"427",
        u"428",
        u"433",
        u"434",
        u"486",
        u"491",
        u"574",
        u"577",
        u"584",
        u"682",
        u"715",
        u"722",
        u"780",
        u"786",
        u"820",
        u"996",
        u"V57",
    ],
    u"diag_2": [
        u"250",
        u"250.01",
        u"250.02",
        u"276",
        u"285",
        u"305",
        u"401",
        u"403",
        u"411",
        u"414",
        u"424",
        u"425",
        u"427",
        u"428",
        u"486",
        u"491",
        u"496",
        u"518",
        u"584",
        u"585",
        u"599",
        u"707",
        u"733",
        u"780",
    ],
    u"diag_3": [
        u"250",
        u"250.01",
        u"250.02",
        u"250.6",
        u"272",
        u"276",
        u"278",
        u"285",
        u"401",
        u"403",
        u"41",
        u"414",
        u"424",
        u"427",
        u"428",
        u"496",
        u"518",
        u"585",
        u"599",
        u"682",
        u"707",
        u"780",
        u"V45",
    ],
    u"admission_type_id": [
        u"Elective",
        u"Emergency",
        u"Not Available",
        u"Urgent",
    ],
}


SMALL_NULLS = {
    u"gender": 1, 
    u"age": 1, 
    u"metformin": 1, 
    u"repaglinide": 1, 
    u"nateglinide": 1, 
    u"chlorpropamide": 1, 
    u"glimepiride": 1, 
    u"glyburide": 1, 
    u"pioglitazone": 1, 
    u"rosiglitazone": 1, 
    u"acarbose": 1, 
    u"insulin": 1, 
    u"glyburidemetformin": 1, 
    u"change": 1, 
    u"diabetesMed": 1, 
    u"diag_1": 1, 
    u"diag_2": 1, 
}


VAR_TYPES = {
    u"admission_type_id": "C",
    u"diabetesMed": "C",
    u"chlorpropamide": "C",
    u"weight": "C",
    u"repaglinide": "C",
    u"payer_code": "C",
    u"number_diagnoses": "N",
    u"diag_3_desc": "T",
    u"diag_1_desc": "T",
    u"num_medications": "N",
    u"rosiglitazone": "C",
    u"id": "N",
    u"admission_source_id": "C",
    u"nateglinide": "C",
    u"num_lab_procedures": "N",
    u"glyburide": "C",
    u"metformin": "C",
    u"glyburidemetformin": "C",
    u"pioglitazone": "C",
    u"glimepiride": "C",
    u"A1Cresult": "C",
    u"time_in_hospital": "N",
    u"max_glu_serum": "C",
    u"acarbose": "C",
    u"number_inpatient": "N",
    u"discharge_disposition_id": "C",
    u"change": "C",
    u"gender": "C",
    u"age": "C",
    u"medical_specialty": "C",
    u"diag_2_desc": "T",
    u"race": "C",
    u"number_outpatient": "N",
    u"insulin": "C",
    u"number_emergency": "N",
    u"num_procedures": "N",
    u"diag_2": "C",
    u"diag_3": "C",
    u"diag_1": "C",
}


def combine_small_levels(ds):
    for col in ds:
        if BIG_LEVELS.get(col, None) is not None:
            mask = np.logical_and(~ds[col].isin(BIG_LEVELS[col]), ds[col].notnull())
            if np.any(mask):
                ds.loc[mask, col] = "small_count"
        if SMALL_NULLS.get(col):
            mask = ds[col].isnull()
            if np.any(mask):
                ds.loc[mask, col] = "small_count"
        if VAR_TYPES.get(col) == "C" or VAR_TYPES.get(col) == "T":
            mask = ds[col].isnull()
            if np.any(mask):
                if ds[col].dtype == float:
                    ds[col] = ds[col].astype(object)
                ds.loc[mask, col] = "nan"
    return ds

# N/A strings in addition to the ones used by Pandas read_csv()
NA_VALUES = ["null", "na", "n/a", "#N/A", "N/A", "?", ".", "", "Inf", "INF", "inf", "-inf", "-Inf", "-INF", " ", "None", "NaN", "-nan", "NULL", "NA", "-1.#IND", "1.#IND", "-1.#QNAN", "1.#QNAN", "#NA", "#N/A N/A", "-NaN", "nan"]

# True/False strings in addition to the ones used by Pandas read_csv()
TRUE_VALUES = ["TRUE", "True", "true"]
FALSE_VALUES = ["FALSE", "False", "false"]

DEFAULT_ENCODING = "utf8"

REQUIRED_COLUMNS = [u"diabetesMed",u"chlorpropamide",u"weight",u"repaglinide",u"payer_code",u"number_diagnoses",u"diag_3_desc",u"diag_1_desc",u"num_medications",u"rosiglitazone",u"id",u"admission_source_id",u"nateglinide",u"num_lab_procedures",u"glyburide",u"metformin",u"glyburidemetformin",u"diag_1",u"pioglitazone",u"glimepiride",u"A1Cresult",u"time_in_hospital",u"max_glu_serum",u"acarbose",u"number_inpatient",u"discharge_disposition_id",u"change",u"gender",u"age",u"medical_specialty",u"diag_2_desc",u"race",u"number_outpatient",u"insulin",u"number_emergency",u"num_procedures",u"diag_2",u"diag_3",u"admission_type_id"]


def validate_columns(column_list):
    if set(REQUIRED_COLUMNS) <= set(column_list):
        return True
    else :
        raise ValueError("Required columns missing: %s" %
                         (set(REQUIRED_COLUMNS) - set(column_list)))

def convert_bool(ds):
    TYPE_CONVERSION = get_type_conversion()
    for col in ds.columns:
        if VAR_TYPES.get(col) == "C" and ds[col].dtype in (int, float):
            mask = ds[col].notnull()
            ds[col] = ds[col].astype(object)
            ds.loc[mask, col] = ds.loc[mask, col].astype(text_type)
        elif VAR_TYPES.get(col) == "N" and ds[col].dtype == bool:
            ds[col] = ds[col].astype(float)
        elif ds[col].dtype == bool:
            ds[col] = ds[col].astype(text_type)
        elif ds[col].dtype == object:
            if VAR_TYPES.get(col) == "N" and col not in TYPE_CONVERSION:
                mask = ds[col].apply(lambda x: x in TRUE_VALUES)
                if np.any(mask):
                    ds.loc[mask, col] = 1
                mask = ds[col].apply(lambda x: x in FALSE_VALUES)
                if np.any(mask):
                    ds.loc[mask, col] = 0
                ds[col] = ds[col].astype(float)
            elif TYPE_CONVERSION.get(col) is None:
                mask = ds[col].notnull()
                ds.loc[mask, col] = ds.loc[mask, col].astype(text_type)
    return ds

def get_dtypes():
    return {a: object for a, b in VAR_TYPES.items() if b == "C"}

def predict_dataframe(ds):
    return ds.apply(predict, axis=1)

def run_dataframe(ds):
    ds = rename_columns(ds)
    ds = convert_bool(ds)
    validate_columns(ds.columns)
    ds = parse_numeric_types(ds)
    ds = add_missing_indicators(ds)
    ds = impute_values(ds)
    ds = combine_small_levels(ds)
    prediction = 1/(1 + np.exp(-predict_dataframe(ds)))
    return prediction


def run(dataset_path, output_path, encoding=None):
    if encoding is None:
        encoding = DEFAULT_ENCODING

    ds = pd.read_csv(dataset_path, na_values=NA_VALUES, low_memory=False,
                     dtype=get_dtypes(), encoding=encoding)

    prediction = run_dataframe(ds)
    prediction_file = output_path
    prediction.name = "Prediction"
    prediction.to_csv(prediction_file, header=True, index_label="Index")


def _construct_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Make offline predictions with DataRobot Prime")

    parser.add_argument(
        "--encoding",
        type=str,
        help=("the encoding of the dataset you are going to make predictions with."
              "for possible alternative entries."),
        metavar="<encoding>"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help=("a .csv file (your dataset); columns must correspond to the "
              "feature set used to generate the DataRobot Prime model."),
        metavar="<data_file>"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="the filename where DataRobot writes the results.",
        metavar="<output_file>"
    )

    return parser


def _parse_command(args):
    parser = _construct_parser()
    parsed_args = parser.parse_args(args[1:])

    if parsed_args.encoding is None:
        sys.stderr.write("Warning: For input data encodings other than UTF-8, "
                         "search Prime examples in the DataRobot Users Guide at https://app.datarobot.com/docs/users-guide/index.html")
        parsed_args.encoding = DEFAULT_ENCODING

    return parsed_args
   


# 2. get the data and do any data preparation / enrichment as required
#    in this tutorial no data preparation is required, we will just remove columns that are not required for the model
idata = InputDataSet
try:
    idata = idata.drop(["predictionValue","predictionLabel", "prediction","predictionDate","predictionThreshold"], axis=1)
except:
    # for testing outside of SQL server
    idata = idata
    idata["glyburidemetformin"]=""
    idata["id"] = "1"


# 3. make predictions
print("make predictions with DRPrime model")
predictions = run_dataframe(idata)

#merge results with source data                      
sourcedata = idata.reset_index()
sourcedata["rowId"] = sourcedata.index

df = predictions.to_frame().reset_index()
df = df.rename(columns= {0: "prediction" })
df.index.name = "index"

output = pd.merge(sourcedata,df,left_on = ["rowId"], right_on = ["index"],how="left")
output["predictionValue"] = ""
output["predictionLabel"] =""
output["predictionDate"] = str(date.today())
output["predictionThreshold"] = ""
output = output.drop(["rowId","index_x","index_y"], axis=1)
print(list(output))
OutputDataSet = output
';


DECLARE @SQL NVARCHAR(MAX)
IF object_id('dbo.#admissions') IS NOT NULL
BEGIN
	SET @SQL = N' SELECT * FROM dbo.#admissions;'
END
ELSE
BEGIN
	SET @SQL = N' SELECT *  FROM dbo.admissions;'
END
DELETE FROM @PREDICTIONS
INSERT INTO @PREDICTIONS
EXEC sp_execute_external_script 
  @language=N'Python'
, @script = @PSCRIPT
, @input_data_1 = @SQL

UPDATE a
SET prediction = p.prediction,
	predictionDate = p.predictionDate
FROM admissions a
JOIN @PREDICTIONS p
ON a.id = p.id
END
