import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions

URL = ""
apiKey = ""

authenticator = IAMAuthenticator(apiKey)
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2022-04-07',
    authenticator=authenticator)

natural_language_understanding.set_service_url( URL)

def sentiment_analysis(text):

    response = natural_language_understanding.analyze(
        text=text,
        features=Features(
            entities=EntitiesOptions(emotion=True, sentiment=True, limit=2),
            keywords=KeywordsOptions(emotion=True, sentiment=True,
                                    limit=2))).get_result()
    return {'score' : response.get('keywords')[0].get('sentiment').get('score'), 
            'label' : response.get('keywords')[0].get('sentiment').get('label')}

#print(json.dumps(sentiment_analysis('awesome service!!!'), indent=2))
