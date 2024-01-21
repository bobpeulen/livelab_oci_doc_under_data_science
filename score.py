
##################################################################################
################################################################################## Imports
##################################################################################
os.system('pip install opencv-python-headless torch')
os.system('pip install oci --upgrade')

from PIL import Image
import torch
import numpy as np
import pandas as pd
import os
import io
import shutil
import sys
import glob
import ads
import urllib
import base64
import uuid
import json
import oci

##################################################################################
################################################################################## Config file
##################################################################################

config = oci.config.from_file("./config", 'DEFAULT') #in model deployment
#config = oci.config.from_file("./model_artifacts/config", 'DEFAULT')  #in the notebook

##################################################################################
################################################################################## Load a Dummy model, used when storing and deploying the model
##################################################################################

def load_model():
    class DummyModel:
        def __init__(self):
            pass
    return DummyModel()

##################################################################################
################################################################################## OCI Document Understanding, Functions and APIs
##################################################################################

def parse_results_visualise(result_dict_kv, top_category):
        
    # Create Empty DataFrame
    results_df = pd.DataFrame([], columns = ["page", "label", "value", "confidence", "receipt"])

    # Iterate over Pages
    for page in result_dict_kv['pages']:

        # Iterate over Key Value Pairs and Respective Bounding Boxes
        for kv_pair in page['documentFields']:

            # get values
            name = kv_pair['fieldLabel']['name']
            value = kv_pair['fieldValue']['value']
            confidence = round(kv_pair['fieldLabel']['confidence']*100, 2)

            # key values
            results_list = [page['pageNumber'], name, value, confidence, top_category]
            a_series = pd.Series(results_list, index = results_df.columns)
            results_df = results_df._append(a_series, ignore_index=True)
    
    return results_df

##################################################################################
################################################################################## OCI Document Understanding, Functions and APIs
##################################################################################

def receipt_classification(config, receipt_encoded):

    aiservicedocument_client = oci.ai_document.AIServiceDocumentClient(config=config)
    
    # Start Document Classification, using custom model OCID
    key_value_extraction_feature = oci.ai_document.models.DocumentClassificationFeature(model_id = model_1_classification)
    
    # input receipt and output location
    input_loc = oci.ai_document.models.InlineDocumentContent(data=receipt_encoded)
    output_location = oci.ai_document.models.OutputLocation()
    output_location.namespace_name = namespace
    output_location.bucket_name = bucket_name
    output_location.prefix = output_name_prefix

    # Define the Features
    processor_config = oci.ai_document.models.GeneralProcessorConfig(features=[key_value_extraction_feature])
    
    # Start Key value extraction
    kv_job_details = oci.ai_document.models.CreateProcessorJobDetails(
                        display_name=str(uuid.uuid4()),
                        compartment_id=compartment_ocid,
                        input_location=input_loc,
                        output_location=output_location,
                        processor_config=processor_config)

    processor_response = aiservicedocument_client.create_processor_job(create_processor_job_details=kv_job_details)
    
    return processor_response, output_location

##################################################################################
################################################################################## OCI Document Understanding, Functions and APIs
##################################################################################

def kv_extraction(config, receipt_encoded, selected_model_ocid):
    
    aiservicedocument_client = oci.ai_document.AIServiceDocumentClient(config=config)
    key_value_extraction_feature = oci.ai_document.models.DocumentKeyValueExtractionFeature(model_id=selected_model_ocid)
    input_loc = oci.ai_document.models.InlineDocumentContent(data=receipt_encoded)
    output_location = oci.ai_document.models.OutputLocation()
    output_location.namespace_name = namespace
    output_location.bucket_name = bucket_name
    output_location.prefix = output_name_prefix
    processor_config = oci.ai_document.models.GeneralProcessorConfig(features=[key_value_extraction_feature])
    kv_job_details = oci.ai_document.models.CreateProcessorJobDetails(
                        display_name=str(uuid.uuid4()),
                        compartment_id=compartment_ocid,
                        input_location=input_loc,
                        output_location=output_location,
                        processor_config=processor_config)

    processor_response = aiservicedocument_client.create_processor_job(create_processor_job_details=kv_job_details)
    output_location_kv = output_location
    processor_response_kv = processor_response
    
    return processor_response_kv, output_location_kv

##################################################################################
################################################################################## OCI Document Understanding, Functions and APIs
##################################################################################

def retrieve_results(config, processor_response, output_location):
    
    processor_job_id = processor_response.data.id
    object_storage_client = oci.object_storage.ObjectStorageClient(config=config)
    object_response = object_storage_client.get_object(namespace_name=output_location.namespace_name,
                                                           bucket_name=output_location.bucket_name,
                                                           object_name="{}/{}/_/results/defaultObject.json".format(
                                                               output_location.prefix, processor_job_id))
    
    result = str(object_response.data.content.decode())
    result_dict = json.loads(result)
    
    return result_dict

##################################################################################
################################################################################## OCI Document Understanding, Functions and APIs
##################################################################################

def predict(data, model=load_model()):
    
    #Get the Base64 pdf file
    document_encoded = data['data']
    
    
    ###################################################################################################################
    #### Model 1 - Classification of the receipt
    ###################################################################################################################
    
    print("--"*100)
    print("**Start model 1 - Receipt Classification**")
    print("--"*100)
    
    # Receipt classification functions
    processor_response, output_location = receipt_classification(config, receipt_encoded)
    
    # Retrieve Results from Object Storage
    result_dict = retrieve_results(config, processor_response, output_location)
    
    #define top category
    top_category = result_dict['pages'][0]['detectedDocumentTypes'][0]['documentType']
    
    print("--"*50)
    print("Detected type of documents")
    print(result_dict['pages'][0]['detectedDocumentTypes'])
    print("--"*50)
    print("Selected document type is ")
    print(top_category)
    print("--"*50)
    
    #define which key value extraction model to use    
    mapping = {"wholefoods":model_2_wholefoods_kv, 'walgreens':model_3_walgreens_kv}
    print("This is the mapping used between top category and key value extraction model chosen:")
    print(mapping)
    
    #select the correct Model OCID from the mapping
    selected_model_ocid = mapping[top_category]
    
    print("--"*50)
    print("Model OCID and top category used in key value extraction are ")
    print(top_category)
    print(selected_model_ocid)
    print("--"*50)
   
    ###################################################################################################################
    #### Key value extraction - Model 2 or Model 3
    ###################################################################################################################
    
    print("--"*100)
    print(f"**Start model 2 or model 3 - Key value extraction for {top_category}**")
    print("--"*100)
    
    #use key value extraction function and fetch results from bucket
    processor_response_kv, output_location_kv = kv_extraction(config, receipt_encoded, selected_model_ocid)
    result_dict_kv = retrieve_results(config, processor_response_kv, output_location_kv)
        
    ###################################################################################################################
    #### Final response
    ###################################################################################################################
    
    #parse response to dataframe and convert to json
    results_df = parse_results_visualise(result_dict_kv, top_category)
    results_df_json = results_df.to_json(orient="records")

    print("Process completed")
    pages_completed = result_dict_kv['documentMetadata']['pageCount']
    print(f"Number of pages in PDF analyzed: {pages_completed}")
    
    return results_df_json
