/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.grpc;

import static org.opensearch.neuralsearch.util.TestUtils.TEST_DIMENSION;
import static org.opensearch.neuralsearch.util.TestUtils.createRandomVector;

import java.util.concurrent.TimeUnit;

import org.junit.After;
import org.junit.Before;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.neuralsearch.BaseNeuralSearchIT;
import org.opensearch.protobufs.BoolQuery;
import org.opensearch.protobufs.FieldValue;
import org.opensearch.protobufs.HybridQuery;
import org.opensearch.protobufs.MatchAllQuery;
import org.opensearch.protobufs.MatchQuery;
import org.opensearch.protobufs.QueryContainer;
import org.opensearch.protobufs.SearchRequest;
import org.opensearch.protobufs.SearchRequestBody;
import org.opensearch.protobufs.SearchResponse;
import org.opensearch.protobufs.TermQuery;
import org.opensearch.protobufs.services.SearchServiceGrpc;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import lombok.SneakyThrows;

/**
 * Integration test for gRPC Hybrid Query functionality.
 *
 * This test verifies that HybridQuery can be:
 * 1. Created in Protocol Buffer format
 * 2. Sent via gRPC to OpenSearch
 * 3. Executed correctly and return expected results
 *
 * Prerequisites:
 * - OpenSearch cluster must have gRPC transport plugin enabled
 * - The cluster should be running and accessible via gRPC (typically on port 9300)
 *
 * Note: This test creates protobuf queries directly since HybridQueryBuilderProtoConverter
 * only implements fromProto (protobuf -> QueryBuilder), not toProto (QueryBuilder -> protobuf).
 *
 * See GRPC_INTEGRATION_TEST_GUIDE.md for detailed documentation.
 */
public class HybridQueryGrpcIT extends BaseNeuralSearchIT {

    private static final String TEST_INDEX_NAME = "test-grpc-hybrid-index";
    private static final String TEST_TEXT_FIELD_NAME = "text";
    private static final String TEST_TITLE_FIELD_NAME = "title";
    private static final String TEST_STATUS_FIELD_NAME = "status";
    private static final String TEST_CATEGORY_FIELD_NAME = "category";
    private static final String TEST_KNN_VECTOR_FIELD_NAME = "vector";

    private ManagedChannel grpcChannel;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        updateClusterSettings();

        // Set up test index with diverse test data
        initializeIndexIfNotExist(TEST_INDEX_NAME);

        // Create gRPC channel for tests
        grpcChannel = createGrpcChannel();
    }

    @After
    public void tearDownGrpc() throws Exception {
        if (grpcChannel != null && !grpcChannel.isShutdown()) {
            grpcChannel.shutdown();
            grpcChannel.awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    /**
     * Gets the gRPC server address from cluster configuration.
     * gRPC port is typically HTTP port + 200 (e.g., 9200 -> 9400) or configured via aux.transport.transport-grpc.port
     */
    protected String getGrpcHost() {
        // Default to localhost for integration tests
        return System.getProperty("tests.grpc.host", "localhost");
    }

    /**
     * Gets the gRPC port. Defaults to HTTP port + 200 (e.g., 9200 -> 9400).
     */
    protected int getGrpcPort() {
        // gRPC port is typically HTTP port + 200 (default range 9400-9500)
        // Can be overridden via system property
        return Integer.parseInt(System.getProperty("tests.grpc.port", "9400"));
    }

    /**
     * Creates a gRPC channel connected to the cluster.
     */
    protected ManagedChannel createGrpcChannel() {
        return ManagedChannelBuilder.forAddress(getGrpcHost(), getGrpcPort())
            .usePlaintext() // Use TLS in production if needed
            .build();
    }

    // ===========================================================================================
    // BASIC FUNCTIONALITY TESTS - Verify end-to-end gRPC query execution
    // ===========================================================================================

    /**
     * Test that a basic hybrid query executes via gRPC and returns results.
     * This validates the complete round-trip: client -> gRPC -> OpenSearch -> response
     */
    @SneakyThrows
    public void testBasicHybridQueryReturnsResults() {
        // Query for documents containing "search" - should match doc1 and doc2
        HybridQuery hybridQuery = HybridQuery.newBuilder()
            .addQueries(createMatchQueryContainer(TEST_TEXT_FIELD_NAME, "search"))
            .addQueries(createMatchAllQueryContainer())
            .build();

        SearchResponse response = executeHybridSearch(hybridQuery);

        assertNotNull("Search response should not be null", response);
        assertTrue("Response should have hits", response.getHits().getHitsCount() > 0);
        // All 4 documents should match due to match_all sub-query
        assertEquals("Should return all documents due to match_all", 4, response.getHits().getHitsCount());
    }

    /**
     * Test hybrid query with multiple term queries combines results correctly.
     * Validates that sub-queries are executed and results are merged.
     */
    @SneakyThrows
    public void testHybridQueryCombinesMultipleSubQueries() {
        // Query for "machine" OR "vector" - should find doc1 (machine learning) and doc3 (vector database)
        HybridQuery hybridQuery = HybridQuery.newBuilder()
            .addQueries(createTermQueryContainer(TEST_TEXT_FIELD_NAME, "machine"))
            .addQueries(createTermQueryContainer(TEST_TEXT_FIELD_NAME, "vector"))
            .build();

        SearchResponse response = executeHybridSearch(hybridQuery);

        assertNotNull("Search response should not be null", response);
        assertTrue("Should find documents matching either sub-query", response.getHits().getHitsCount() >= 2);
    }

    // ===========================================================================================
    // FILTER TESTS - Verify filter restricts results across all sub-queries
    // ===========================================================================================

    /**
     * Test that filter restricts hybrid query results.
     * Filter should apply to all sub-queries, not just one.
     */
    @SneakyThrows
    public void testHybridQueryWithFilterRestrictsResults() {
        // Match all documents but filter to only "active" status
        // doc1 and doc2 are active, doc3 and doc4 are inactive
        HybridQuery hybridQuery = HybridQuery.newBuilder()
            .addQueries(createMatchAllQueryContainer())
            .setFilter(createTermQueryContainer(TEST_STATUS_FIELD_NAME, "active"))
            .build();

        SearchResponse response = executeHybridSearch(hybridQuery);

        assertNotNull("Search response should not be null", response);
        assertEquals("Should return only active documents", 2, response.getHits().getHitsCount());
    }

    /**
     * Test filter with no matching documents returns empty results.
     */
    @SneakyThrows
    public void testHybridQueryWithFilterNoMatches() {
        // Match all documents but filter to non-existent status
        HybridQuery hybridQuery = HybridQuery.newBuilder()
            .addQueries(createMatchAllQueryContainer())
            .setFilter(createTermQueryContainer(TEST_STATUS_FIELD_NAME, "nonexistent"))
            .build();

        SearchResponse response = executeHybridSearch(hybridQuery);

        assertNotNull("Search response should not be null", response);
        assertEquals("Should return no documents with non-matching filter", 0, response.getHits().getHitsCount());
    }

    /**
     * Test hybrid query with complex bool filter.
     */
    @SneakyThrows
    public void testHybridQueryWithBoolFilter() {
        // Match all but filter to: status=active AND category=tech
        BoolQuery boolFilter = BoolQuery.newBuilder()
            .addMust(createTermQueryContainer(TEST_STATUS_FIELD_NAME, "active"))
            .addMust(createTermQueryContainer(TEST_CATEGORY_FIELD_NAME, "tech"))
            .build();

        HybridQuery hybridQuery = HybridQuery.newBuilder()
            .addQueries(createMatchAllQueryContainer())
            .setFilter(QueryContainer.newBuilder().setBool(boolFilter).build())
            .build();

        SearchResponse response = executeHybridSearch(hybridQuery);

        assertNotNull("Search response should not be null", response);
        // Only doc1 is active AND in tech category
        assertEquals("Should return only docs matching all filter conditions", 1, response.getHits().getHitsCount());
    }

    // ===========================================================================================
    // OPTIONAL FIELDS TESTS - Verify paginationDepth, queryName, boost handling
    // ===========================================================================================

    /**
     * Test hybrid query with pagination depth setting.
     */
    @SneakyThrows
    public void testHybridQueryWithPaginationDepth() {
        HybridQuery hybridQuery = HybridQuery.newBuilder().addQueries(createMatchAllQueryContainer()).setPaginationDepth(100).build();

        SearchResponse response = executeHybridSearch(hybridQuery);

        assertNotNull("Search response should not be null", response);
        assertTrue("Query with pagination depth should execute successfully", response.getHits().getHitsCount() > 0);
    }

    /**
     * Test hybrid query with query name for debugging/profiling.
     */
    @SneakyThrows
    public void testHybridQueryWithQueryName() {
        HybridQuery hybridQuery = HybridQuery.newBuilder()
            .addQueries(createMatchAllQueryContainer())
            .setXName("my-hybrid-query-name")
            .build();

        SearchResponse response = executeHybridSearch(hybridQuery);

        assertNotNull("Search response should not be null", response);
        assertTrue("Query with name should execute successfully", response.getHits().getHitsCount() > 0);
    }

    /**
     * Test hybrid query with all optional fields combined.
     */
    @SneakyThrows
    public void testHybridQueryWithAllOptionalFields() {
        HybridQuery hybridQuery = HybridQuery.newBuilder()
            .addQueries(createMatchQueryContainer(TEST_TEXT_FIELD_NAME, "search"))
            .addQueries(createMatchQueryContainer(TEST_TEXT_FIELD_NAME, "learning"))
            .setFilter(createTermQueryContainer(TEST_STATUS_FIELD_NAME, "active"))
            .setPaginationDepth(50)
            .setXName("complete-hybrid-query")
            .build();

        SearchResponse response = executeHybridSearch(hybridQuery);

        assertNotNull("Search response should not be null", response);
        // Should find active docs matching "search" or "learning"
        assertTrue("Should return matching active documents", response.getHits().getHitsCount() > 0);
    }

    // ===========================================================================================
    // MIXED QUERY TYPES - Verify different sub-query types work together
    // ===========================================================================================

    /**
     * Test hybrid query mixing Term + Match + MatchAll queries.
     */
    @SneakyThrows
    public void testHybridQueryWithMixedQueryTypes() {
        HybridQuery hybridQuery = HybridQuery.newBuilder()
            .addQueries(createTermQueryContainer(TEST_CATEGORY_FIELD_NAME, "tech"))
            .addQueries(createMatchQueryContainer(TEST_TEXT_FIELD_NAME, "neural search"))
            .addQueries(createMatchAllQueryContainer())
            .build();

        SearchResponse response = executeHybridSearch(hybridQuery);

        assertNotNull("Search response should not be null", response);
        // All documents should match due to match_all
        assertEquals("Should return all documents", 4, response.getHits().getHitsCount());
    }

    /**
     * Test hybrid query with maximum allowed sub-queries (5).
     */
    @SneakyThrows
    public void testHybridQueryWithMaxSubQueries() {
        HybridQuery hybridQuery = HybridQuery.newBuilder()
            .addQueries(createTermQueryContainer(TEST_TEXT_FIELD_NAME, "search"))
            .addQueries(createTermQueryContainer(TEST_TEXT_FIELD_NAME, "learning"))
            .addQueries(createTermQueryContainer(TEST_TEXT_FIELD_NAME, "neural"))
            .addQueries(createTermQueryContainer(TEST_TEXT_FIELD_NAME, "vector"))
            .addQueries(createMatchAllQueryContainer())
            .build();

        SearchResponse response = executeHybridSearch(hybridQuery);

        assertNotNull("Search response should not be null", response);
        assertTrue("Should execute with max sub-queries", response.getHits().getHitsCount() > 0);
    }

    // ===========================================================================================
    // ERROR HANDLING TESTS - Verify proper gRPC error codes for invalid requests
    // ===========================================================================================

    /**
     * Test that exceeding max sub-queries returns proper gRPC error.
     */
    @SneakyThrows
    public void testErrorExceedsMaxSubQueries() {
        HybridQuery.Builder builder = HybridQuery.newBuilder();
        // Add 6 sub-queries (max is 5)
        for (int i = 0; i < 6; i++) {
            builder.addQueries(createTermQueryContainer(TEST_TEXT_FIELD_NAME, "term" + i));
        }

        StatusRuntimeException exception = expectThrows(StatusRuntimeException.class, () -> executeHybridSearch(builder.build()));

        assertTrue(
            "Error should indicate max sub-queries exceeded",
            exception.getMessage().contains("Number of sub-queries exceeds maximum")
                || exception.getStatus().getCode().name().equals("INVALID_ARGUMENT")
        );
    }

    /**
     * Test that empty queries returns proper gRPC error.
     */
    @SneakyThrows
    public void testErrorEmptyQueries() {
        HybridQuery hybridQuery = HybridQuery.newBuilder().build();

        StatusRuntimeException exception = expectThrows(StatusRuntimeException.class, () -> executeHybridSearch(hybridQuery));

        assertTrue(
            "Error should indicate queries required",
            exception.getMessage().contains("requires 'queries' field") || exception.getStatus().getCode().name().equals("INVALID_ARGUMENT")
        );
    }

    /**
     * Test that non-default boost returns proper gRPC error.
     */
    @SneakyThrows
    public void testErrorNonDefaultBoost() {
        HybridQuery hybridQuery = HybridQuery.newBuilder().addQueries(createMatchAllQueryContainer()).setBoost(2.0f).build();

        StatusRuntimeException exception = expectThrows(StatusRuntimeException.class, () -> executeHybridSearch(hybridQuery));

        assertTrue(
            "Error should indicate boost not supported",
            exception.getMessage().contains("does not support") || exception.getStatus().getCode().name().equals("INVALID_ARGUMENT")
        );
    }

    /**
     * Test search on non-existent index returns proper gRPC error.
     */
    @SneakyThrows
    public void testErrorNonExistentIndex() {
        HybridQuery hybridQuery = HybridQuery.newBuilder().addQueries(createMatchAllQueryContainer()).build();

        QueryContainer queryContainer = QueryContainer.newBuilder().setHybrid(hybridQuery).build();
        SearchRequest request = buildSearchRequest("non-existent-index-12345", queryContainer);

        SearchServiceGrpc.SearchServiceBlockingStub stub = SearchServiceGrpc.newBlockingStub(grpcChannel);

        StatusRuntimeException exception = expectThrows(StatusRuntimeException.class, () -> stub.search(request));

        assertNotNull("Should throw exception for non-existent index", exception);
    }

    // ===========================================================================================
    // HELPER METHODS
    // ===========================================================================================

    /**
     * Execute a hybrid search using the shared gRPC channel.
     */
    private SearchResponse executeHybridSearch(HybridQuery hybridQuery) {
        QueryContainer queryContainer = QueryContainer.newBuilder().setHybrid(hybridQuery).build();
        SearchRequest request = buildSearchRequest(TEST_INDEX_NAME, queryContainer);
        SearchServiceGrpc.SearchServiceBlockingStub stub = SearchServiceGrpc.newBlockingStub(grpcChannel);
        return stub.search(request);
    }

    /**
     * Execute a hybrid search with custom size.
     */
    private SearchResponse executeHybridSearch(HybridQuery hybridQuery, int size) {
        QueryContainer queryContainer = QueryContainer.newBuilder().setHybrid(hybridQuery).build();
        SearchRequest request = buildSearchRequest(TEST_INDEX_NAME, queryContainer, size);
        SearchServiceGrpc.SearchServiceBlockingStub stub = SearchServiceGrpc.newBlockingStub(grpcChannel);
        return stub.search(request);
    }

    // ===========================================================================================
    // INDEX AND DATA SETUP
    // ===========================================================================================

    /**
     * Initialize test index with mappings and diverse test documents.
     *
     * Test Documents:
     * - doc1: text="neural search for machine learning", title="ML Guide", status=active, category=tech
     * - doc2: text="semantic search with transformers", title="NLP Tutorial", status=active, category=science
     * - doc3: text="vector database for embeddings", title="DB Overview", status=inactive, category=tech
     * - doc4: text="traditional keyword search methods", title="Search Basics", status=inactive, category=education
     */
    @SneakyThrows
    private void initializeIndexIfNotExist(String indexName) {
        if (checkIndexExists(indexName)) {
            return;
        }

        String indexConfiguration = """
            {
              "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
              },
              "mappings": {
                "properties": {
                  "%s": {
                    "type": "text",
                    "analyzer": "standard"
                  },
                  "%s": {
                    "type": "text"
                  },
                  "%s": {
                    "type": "keyword"
                  },
                  "%s": {
                    "type": "keyword"
                  },
                  "%s": {
                    "type": "knn_vector",
                    "dimension": %d
                  }
                }
              }
            }
            """.formatted(
            TEST_TEXT_FIELD_NAME,
            TEST_TITLE_FIELD_NAME,
            TEST_STATUS_FIELD_NAME,
            TEST_CATEGORY_FIELD_NAME,
            TEST_KNN_VECTOR_FIELD_NAME,
            TEST_DIMENSION
        );

        createIndex(indexName, indexConfiguration);
        indexTestDocuments(indexName);
    }

    /**
     * Check if index exists
     */
    @SneakyThrows
    private boolean checkIndexExists(String indexName) {
        try {
            Response response = client().performRequest(new Request("HEAD", "/" + indexName));
            return response.getStatusLine().getStatusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Index diverse test documents for comprehensive testing.
     */
    @SneakyThrows
    private void indexTestDocuments(String indexName) {
        // Document 1: Active tech document about ML
        String doc1 = """
            {
              "%s": "neural search for machine learning applications",
              "%s": "ML Guide",
              "%s": "active",
              "%s": "tech",
              "%s": %s
            }
            """.formatted(
            TEST_TEXT_FIELD_NAME,
            TEST_TITLE_FIELD_NAME,
            TEST_STATUS_FIELD_NAME,
            TEST_CATEGORY_FIELD_NAME,
            TEST_KNN_VECTOR_FIELD_NAME,
            formatVectorAsJsonArray(createRandomVector(TEST_DIMENSION))
        );

        // Document 2: Active science document about NLP
        String doc2 = """
            {
              "%s": "semantic search with transformers and embeddings",
              "%s": "NLP Tutorial",
              "%s": "active",
              "%s": "science",
              "%s": %s
            }
            """.formatted(
            TEST_TEXT_FIELD_NAME,
            TEST_TITLE_FIELD_NAME,
            TEST_STATUS_FIELD_NAME,
            TEST_CATEGORY_FIELD_NAME,
            TEST_KNN_VECTOR_FIELD_NAME,
            formatVectorAsJsonArray(createRandomVector(TEST_DIMENSION))
        );

        // Document 3: Inactive tech document about vector DB
        String doc3 = """
            {
              "%s": "vector database for storing embeddings efficiently",
              "%s": "DB Overview",
              "%s": "inactive",
              "%s": "tech",
              "%s": %s
            }
            """.formatted(
            TEST_TEXT_FIELD_NAME,
            TEST_TITLE_FIELD_NAME,
            TEST_STATUS_FIELD_NAME,
            TEST_CATEGORY_FIELD_NAME,
            TEST_KNN_VECTOR_FIELD_NAME,
            formatVectorAsJsonArray(createRandomVector(TEST_DIMENSION))
        );

        // Document 4: Inactive education document about traditional search
        String doc4 = """
            {
              "%s": "traditional keyword search methods and techniques",
              "%s": "Search Basics",
              "%s": "inactive",
              "%s": "education",
              "%s": %s
            }
            """.formatted(
            TEST_TEXT_FIELD_NAME,
            TEST_TITLE_FIELD_NAME,
            TEST_STATUS_FIELD_NAME,
            TEST_CATEGORY_FIELD_NAME,
            TEST_KNN_VECTOR_FIELD_NAME,
            formatVectorAsJsonArray(createRandomVector(TEST_DIMENSION))
        );

        ingestDocument(indexName, doc1, "1");
        ingestDocument(indexName, doc2, "2");
        ingestDocument(indexName, doc3, "3");
        ingestDocument(indexName, doc4, "4");

        // Refresh to make documents searchable
        client().performRequest(new Request("POST", "/" + indexName + "/_refresh"));
    }

    // ===========================================================================================
    // SEARCH REQUEST BUILDERS
    // ===========================================================================================

    /**
     * Build a SearchRequest from index name and query container with default size.
     */
    private SearchRequest buildSearchRequest(String index, QueryContainer query) {
        return buildSearchRequest(index, query, 10);
    }

    /**
     * Build a SearchRequest from index name, query container, and size.
     */
    private SearchRequest buildSearchRequest(String index, QueryContainer query, int size) {
        return SearchRequest.newBuilder()
            .addIndex(index)
            .setSearchRequestBody(SearchRequestBody.newBuilder().setQuery(query).setSize(size).build())
            .build();
    }

    // ===========================================================================================
    // QUERY CONTAINER BUILDERS
    // ===========================================================================================

    /**
     * Create a TermQuery QueryContainer (exact match on keyword fields).
     */
    private QueryContainer createTermQueryContainer(String field, String value) {
        TermQuery termQuery = TermQuery.newBuilder().setField(field).setValue(FieldValue.newBuilder().setString(value).build()).build();
        return QueryContainer.newBuilder().setTerm(termQuery).build();
    }

    /**
     * Create a MatchQuery QueryContainer (analyzed text search).
     */
    private QueryContainer createMatchQueryContainer(String field, String query) {
        MatchQuery matchQuery = MatchQuery.newBuilder().setField(field).setQuery(FieldValue.newBuilder().setString(query).build()).build();
        return QueryContainer.newBuilder().setMatch(matchQuery).build();
    }

    /**
     * Create a MatchAll QueryContainer.
     */
    private QueryContainer createMatchAllQueryContainer() {
        return QueryContainer.newBuilder().setMatchAll(MatchAllQuery.newBuilder().build()).build();
    }

    /**
     * Format vector array as JSON array string.
     */
    private String formatVectorAsJsonArray(float[] vector) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < vector.length; i++) {
            if (i > 0) {
                sb.append(",");
            }
            sb.append(vector[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    protected boolean preserveClusterUponCompletion() {
        return true;
    }
}
