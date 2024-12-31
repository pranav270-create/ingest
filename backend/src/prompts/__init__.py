from src.prompts.registry import PromptRegistry
from src.prompts.online.client_prompts import ClientQAPrompt, ClientTrendsPrompt, SummaryEmailPrompt
from src.prompts.offline.etl_featurization import SummarizeEntryPrompt, SummarizeIngestionPrompt
from src.prompts.offline.evaluation_prompts import SyntheticQAPairPrompt