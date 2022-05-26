GO
ALTER PROC [dbo].[sp_predictAdmissions] (@RETRAIN int = 1)
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
	predictionThreshold nvarchar(max),
	predictionLabel nvarchar(500),
	predictionValue nvarchar(500),
	DEPLOYMENT_ID nvarchar(500),
	predictionDate nvarchar(500)	
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
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import json

def wait_for_async_resolution(client, status_url):
    status = False
    while status == False:
        resp = client.get(status_url)
        try:
            r = json.loads(resp.content.decode("utf-8"))
            if resp.status_code == 200 and r["status"].upper() == "ACTIVE":
                status = True
                return resp
        except:
            print("error")
        time.sleep(10)   # Delays for 10 seconds.

def wait_for_result(client, response):
    assert response.status_code in (200, 201, 202), response.content

    if response.status_code == 200:
        data = response.json()

    elif response.status_code == 201:
        status_url = response.headers["Location"]
        resp = client.get(status_url)
        assert resp.status_code == 200, resp.content
        data = resp.json()

    elif response.status_code == 202:
        status_url = response.headers["Location"]
        resp = wait_for_async_resolution(client, status_url)
        data = resp.json()
    return data

# 1. Get the DR client settings and authenticate
API_TOKEN = InputDataSet["API_KEY"].iloc[0]
USERNAME = InputDataSet["USERNAME"].iloc[0]
WORKERCOUNT = InputDataSet["WORKERCOUNT"].iloc[0]
PROJECTNAME = InputDataSet["PROJECTNAME"].iloc[0]
TARGET = InputDataSet["TARGET"].iloc[0]
DR_KEY = InputDataSet["DR_KEY"].iloc[0]
INSTANCE_ID = InputDataSet["INSTANCE_ID"].iloc[0]
MODELMANAGEMENTENDPOINT = InputDataSet["MODELMANAGEMENTENDPOINT"].iloc[0]
PREDICTIONSENDPOINT = InputDataSet["PREDICTIONSENDPOINT"].iloc[0]
DEPLOYMENT_ID = InputDataSet["DEPLOYMENT_ID"].iloc[0]
RETRAIN = InputDataSet["RETRAIN"].iloc[0]

MODELMANAGEMENTHEADERS = {"Content-Type": "application/json", "Authorization": "token %s" % API_TOKEN}
PREDICTIONSHEADERS = {"Content-Type": "application/json; charset=UTF-8","datarobot-key": "%s" % DR_KEY}

drclient = dr.Client(endpoint=MODELMANAGEMENTENDPOINT, token=API_TOKEN, ssl_verify=False)


# 2. get the data and do any data preparation / enrichment as required
#    in this tutorial no data preparation is required, we will just remove columns that are not required for the model
idata = InputDataSet
idata = idata.drop(["API_KEY","USERNAME","DEPLOYMENT_ID","WORKERCOUNT","RETRAIN","TARGET","PROJECTNAME","DR_KEY","INSTANCE_ID","MODELMANAGEMENTENDPOINT","PREDICTIONSENDPOINT","predictionValue","predictionLabel", "prediction","predictionDate","predictionThreshold"], axis=1)

# 3. Check if deployment already exists, if not create one, else use the existing one
if str(DEPLOYMENT_ID) == "None" or str(DEPLOYMENT_ID) == "":
	# create new project & deployment
	print("creating new project with autopilot")
	newProject = dr.Project.start(sourcedata=idata, project_name =PROJECTNAME, target = TARGET, autopilot_on=True)
	newProject.set_worker_count(int(WORKERCOUNT))
	newProject.wait_for_autopilot()
	print("waiting for autopilot")
	recommendation_type = dr.enums.RECOMMENDED_MODEL_TYPE.RECOMMENDED_FOR_DEPLOYMENT
	recommendation = dr.models.ModelRecommendation.get(newProject.id, recommendation_type)
	bestModelId = recommendation.model_id
	crossvalidation = requests.post("%s/projects/%s/models/%s/crossValidation" % (MODELMANAGEMENTENDPOINT, newProject.id, bestModelId), headers=MODELMANAGEMENTHEADERS )
	print(crossvalidation)
	payload = {
        "projectId": str(newProject.id),
        "modelId": str(bestModelId),
        "instanceId": str(INSTANCE_ID),
        "label": "DataRobot ML with MS SQL"+ str(date.today()),
        "description": "DataRobot ML with MS SQL"+ str(date.today()),
        "status": "active",
        "deploymentType": "dedicated",
        "trainingDataSubset": "eda"
    }
	deploymentresponse = drclient.post(
			"/modelDeployments/asyncCreate",
			data=payload,
			headers={"Content-Type": "application/json"}
		)
	print(str(deploymentresponse))
	deployment_response = wait_for_result(drclient, deploymentresponse)
	DEPLOYMENT_ID = str(deployment_response["id"])
	print(str(DEPLOYMENT_ID))

else:
	print("use existing deployment: " + DEPLOYMENT_ID)
	
# 4. check if model retraining is required
if RETRAIN == "1":
    print("retrain model first before making predictions")
    # a. get current model of deployment
    deployment = requests.get("%s/modelDeployments/%s" % (MODELMANAGEMENTENDPOINT,DEPLOYMENT_ID), headers=MODELMANAGEMENTHEADERS)
    model = dr.Model.get(deployment.json()["project"]["id"],deployment.json()["model"]["id"])    
    # b. create new project with new training data
    retrainProject = dr.Project.start(sourcedata=idata, project_name=PROJECTNAME + str(date.today()) , target=TARGET, autopilot_on=False)
        
    # c. train new model with training data
    modelJobId = retrainProject.train(model.blueprint_id)
    newModel = dr.models.modeljob.wait_for_async_model_creation(project_id=retrainProject.id, model_job_id=modelJobId)
    fi = newModel.get_or_request_feature_impact(600)
        
    # d. update deployment with new model
    model_Update = requests.patch("%s/modelDeployments/%s/model" % (MODELMANAGEMENTENDPOINT,DEPLOYMENT_ID), headers=MODELMANAGEMENTHEADERS, data="{\"modelId\":\"%s\"}" % (newModel.id))
   
# 5. make predictions
print("make predictions with existing model")
predictions = pd.DataFrame([])
predictionResults = requests.post("%s/deployments/%s/predictions" % (PREDICTIONSENDPOINT,DEPLOYMENT_ID), data=idata.to_json(orient="records"),auth=(USERNAME, API_TOKEN), headers=PREDICTIONSHEADERS)

#map results to a dataframe and fail gracefully if no predictions got returned
if "message" in predictionResults:
                print ("error during game result prediction: " + str(predictionResults["message"]))
if predictionResults.status_code == 200:
    items = json.loads(predictionResults.text)["data"]
    for item in items:
        rowId = item["rowId"]
        prediction = item["prediction"]
        predictionThreshold = item["predictionThreshold"]
        predictionLabel =  item["predictionValues"][0]["label"]
        predictionValue =  item["predictionValues"][0]["value"]
        
        predictions = predictions.append({"rowId": rowId, "prediction": prediction, "predictionThreshold": predictionThreshold, "predictionLabel": predictionLabel, "predictionValue": predictionValue }, ignore_index=True)

#merge results with source                      
sourcedata = idata.reset_index()
sourcedata["rowId"] = sourcedata.index
output = pd.merge(sourcedata,predictions,left_on = ["rowId"], right_on = ["rowId"],how="left")
output["DEPLOYMENT_ID_NEW"] = str(DEPLOYMENT_ID)
output["predictionDate"] = str(date.today())
output = output.drop(["rowId","index"], axis=1)
print(list(output))
OutputDataSet = output';


DECLARE @SQL NVARCHAR(MAX)
IF object_id('dbo.#admissions') IS NOT NULL
BEGIN
	SET @SQL = N' SELECT *, ' + CONVERT(nvarchar(max), @RETRAIN) + ' as RETRAIN FROM dbo.#admissions JOIN dbo.DRconfiguration ON 1 = 1;'
END
ELSE
BEGIN
	SET @SQL = N' SELECT *, ' + CONVERT(nvarchar(max), @RETRAIN) + ' as RETRAIN FROM dbo.admissions JOIN dbo.DRconfiguration ON 1 = 1;'
END
DELETE FROM @PREDICTIONS
INSERT INTO @PREDICTIONS
EXEC sp_execute_external_script 
  @language=N'Python'
, @script = @PSCRIPT
, @input_data_1 = @SQL

UPDATE a
SET prediction = p.prediction,
    predictionValue = p.predictionValue,
	predictionLabel = p.predictionLabel,
	predictionDate = p.predictionDate,
	predictionThreshold = p.predictionThreshold
FROM admissions a
JOIN @PREDICTIONS p
ON a.id = p.id

DECLARE @DEPLOYMENT_ID nvarchar(500)
SELECT TOP 1 @DEPLOYMENT_ID = DEPLOYMENT_ID FROM @PREDICTIONS
IF @DEPLOYMENT_ID IS NOT NULL
BEGIN
	UPDATE DRconfiguration
		SET DEPLOYMENT_ID =  CONVERT(nvarchar(500),@DEPLOYMENT_ID)
END

END
