from loguru import logger

from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from langflow.helpers import docs_to_data
from langflow.inputs import DictInput, FloatInput
from langflow.io import (
    BoolInput,
    DataInput,
    DropdownInput,
    HandleInput,
    IntInput,
    MultilineInput,
    SecretStrInput,
    StrInput,
)
from langflow.schema import Data


class HCDVectorStoreComponent(LCVectorStoreComponent):
    display_name: str = "Hyper-Converged Database"
    description: str = "Implementation of Vector Store using Hyper-Converged Database (HCD) with search capabilities"
    documentation: str = "https://python.langchain.com/docs/integrations/vectorstores/astradb"
    name = "HCD"
    icon: str = "HCD"

    inputs = [
        StrInput(
            name="collection_name",
            display_name="Collection Name",
            info="The name of the collection within HCD where the vectors will be stored.",
            required=True,
        ),
        StrInput(
            name="username",
            display_name="HCD Username",
            info="Authentication username for accessing HCD.",
            value="hcd-superuser",
            required=True,
        ),
        SecretStrInput(
            name="password",
            display_name="HCD Password",
            info="Authentication password for accessing HCD.",
            value="HCD_PASSWORD",
            required=True,
        ),
        SecretStrInput(
            name="api_endpoint",
            display_name="HCD API Endpoint",
            info="API endpoint URL for the HCD service.",
            value="HCD_API_ENDPOINT",
            required=True,
        ),
        MultilineInput(
            name="search_input",
            display_name="Search Input",
        ),
        DataInput(
            name="ingest_data",
            display_name="Ingest Data",
            is_list=True,
        ),
        StrInput(
            name="namespace",
            display_name="Namespace",
            info="Optional namespace within HCD to use for the collection.",
            value="default_namespace",
            advanced=True,
        ),
        MultilineInput(
            name="ca_certificate",
            display_name="CA Certificate",
            info="Optional CA certificate for TLS connections to HCD.",
            advanced=True,
        ),
        DropdownInput(
            name="metric",
            display_name="Metric",
            info="Optional distance metric for vector comparisons in the vector store.",
            options=["cosine", "dot_product", "euclidean"],
            advanced=True,
        ),
        IntInput(
            name="batch_size",
            display_name="Batch Size",
            info="Optional number of data to process in a single batch.",
            advanced=True,
        ),
        IntInput(
            name="bulk_insert_batch_concurrency",
            display_name="Bulk Insert Batch Concurrency",
            info="Optional concurrency level for bulk insert operations.",
            advanced=True,
        ),
        IntInput(
            name="bulk_insert_overwrite_concurrency",
            display_name="Bulk Insert Overwrite Concurrency",
            info="Optional concurrency level for bulk insert operations that overwrite existing data.",
            advanced=True,
        ),
        IntInput(
            name="bulk_delete_concurrency",
            display_name="Bulk Delete Concurrency",
            info="Optional concurrency level for bulk delete operations.",
            advanced=True,
        ),
        DropdownInput(
            name="setup_mode",
            display_name="Setup Mode",
            info="Configuration mode for setting up the vector store, with options like 'Sync', 'Async', or 'Off'.",
            options=["Sync", "Async", "Off"],
            advanced=True,
            value="Sync",
        ),
        BoolInput(
            name="pre_delete_collection",
            display_name="Pre Delete Collection",
            info="Boolean flag to determine whether to delete the collection before creating a new one.",
            advanced=True,
        ),
        StrInput(
            name="metadata_indexing_include",
            display_name="Metadata Indexing Include",
            info="Optional list of metadata fields to include in the indexing.",
            advanced=True,
        ),
        HandleInput(
            name="embedding",
            display_name="Embedding or Astra Vectorize",
            input_types=["Embeddings", "dict"],
            # TODO: This should be optional, but need to refactor langchain-astradb first.
            info="Allows either an embedding model or an Astra Vectorize configuration.",
        ),
        StrInput(
            name="metadata_indexing_exclude",
            display_name="Metadata Indexing Exclude",
            info="Optional list of metadata fields to exclude from the indexing.",
            advanced=True,
        ),
        StrInput(
            name="collection_indexing_policy",
            display_name="Collection Indexing Policy",
            info="Optional dictionary defining the indexing policy for the collection.",
            advanced=True,
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            advanced=True,
            value=4,
        ),
        DropdownInput(
            name="search_type",
            display_name="Search Type",
            info="Search type to use",
            options=["Similarity", "Similarity with score threshold", "MMR (Max Marginal Relevance)"],
            value="Similarity",
            advanced=True,
        ),
        FloatInput(
            name="search_score_threshold",
            display_name="Search Score Threshold",
            info="Minimum similarity score threshold for search results. "
            "(when using 'Similarity with score threshold')",
            value=0,
            advanced=True,
        ),
        DictInput(
            name="search_filter",
            display_name="Search Metadata Filter",
            info="Optional dictionary of filters to apply to the search query.",
            advanced=True,
            is_list=True,
        ),
    ]

    @check_cached_vector_store
    def build_vector_store(self):
        try:
            from langchain_astradb import AstraDBVectorStore
            from langchain_astradb.utils.astradb import SetupMode
        except ImportError as e:
            raise ImportError(
                "Could not import langchain Astra DB integration package. "
                "Please install it with `pip install langchain-astradb`."
            ) from e

        try:
            from astrapy.authentication import UsernamePasswordTokenProvider
            from astrapy.constants import Environment
        except ImportError as e:
            raise ImportError(
                "Could not import astrapy integration package. Please install it with `pip install astrapy`."
            ) from e

        if not self.setup_mode:
            self.setup_mode = self._inputs["setup_mode"].options[0]

        setup_mode_value = SetupMode[self.setup_mode.upper()]

        embedding_dict = (
            {"embedding": self.embedding}
            if not isinstance(self.embedding, dict)
            else {
                "collection_vector_service_options": CollectionVectorServiceOptions.from_dict(
                    {
                        **self.embedding.get("collection_vector_service_options", {}),
                        "authentication": {
                            k: v
                            for k, v in self.embedding.get("collection_vector_service_options", {})
                            .get("authentication", {})
                            .items()
                            if k and v
                        },
                        "parameters": {
                            k: v
                            for k, v in self.embedding.get("collection_vector_service_options", {})
                            .get("parameters", {})
                            .items()
                            if k and v
                        },
                    }
                ),
                **(
                    {"collection_embedding_api_key": self.embedding.get("collection_embedding_api_key")}
                    if "collection_embedding_api_key" in self.embedding
                    else {}
                ),
            }
        )

        token_provider = UsernamePasswordTokenProvider(self.username, self.password)
        vector_store_kwargs = {
            **embedding_dict,
            "collection_name": self.collection_name,
            "token": token_provider,
            "api_endpoint": self.api_endpoint,
            "namespace": self.namespace,
            "metric": self.metric,
            "batch_size": self.batch_size,
            "bulk_insert_batch_concurrency": self.bulk_insert_batch_concurrency,
            "bulk_insert_overwrite_concurrency": self.bulk_insert_overwrite_concurrency,
            "bulk_delete_concurrency": self.bulk_delete_concurrency,
            "setup_mode": setup_mode_value,
            "pre_delete_collection": self.pre_delete_collection or False,
            "environment": Environment.HCD,
            **({"metadata_indexing_include": self.metadata_indexing_include} if self.metadata_indexing_include else {}),
            **({"metadata_indexing_exclude": self.metadata_indexing_exclude} if self.metadata_indexing_exclude else {}),
            **(
                {"collection_indexing_policy": self.collection_indexing_policy}
                if self.collection_indexing_policy
                else {}
            ),
        }

        return AstraDBVectorStore(**vector_store_kwargs)

    def _add_documents_to_vector_store(self, vector_store) -> None:
        documents = []
        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                msg = "Vector Store Inputs must be Data objects."
                raise TypeError(msg)

        if documents:
            logger.debug(f"Adding {len(documents)} documents to the Vector Store.")
            try:
                vector_store.add_documents(documents)
            except Exception as e:
                msg = f"Error adding documents to AstraDBVectorStore: {e}"
                raise ValueError(msg) from e
        else:
            logger.debug("No documents to add to the Vector Store.")

    def _map_search_type(self) -> str:
        return {
            "Similarity with score threshold": "similarity_score_threshold",
            "MMR (Max Marginal Relevance)": "mmr",
        }.get(self.search_type, "similarity")

    def _build_search_args(self):
        args = {
            "k": self.number_of_results,
            "score_threshold": self.search_score_threshold,
        }
        if self.search_filter:
            clean_filter = {k: v for k, v in self.search_filter.items() if k and v}
            if clean_filter:
                args["filter"] = clean_filter
        return args

    def search_documents(self) -> list[Data]:
        vector_store = self.build_vector_store()

        if self.search_input and isinstance(self.search_input, str) and self.search_input.strip():
            try:
                docs = vector_store.search(
                    query=self.search_input, search_type=self._map_search_type(), **self._build_search_args()
                )
            except Exception as e:
                raise ValueError(f"Error performing search in AstraDBVectorStore: {e}") from e

            return docs_to_data(docs)
        return []

    def get_retriever_kwargs(self):
        search_args = self._build_search_args()
        return {
            "search_type": self._map_search_type(),
            "search_kwargs": search_args,
        }
