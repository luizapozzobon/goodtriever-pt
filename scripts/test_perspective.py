import os
import json

from googleapiclient import discovery

# prior to executing this script run: export PERSPECTIVE_API_KEY="key"
PERSPECTIVE_API_KEY = os.environ['PERSPECTIVE_API_KEY']

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=PERSPECTIVE_API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

analyze_request = {
  'comment': { 'text': 'friendly greetings from python' },
  'requestedAttributes': {'TOXICITY': {}}
}

response = client.comments().analyze(body=analyze_request).execute()
print(json.dumps(response, indent=2))
