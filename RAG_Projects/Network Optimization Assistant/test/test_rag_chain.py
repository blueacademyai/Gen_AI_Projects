import pytest
from unittest.mock import patch
from src.rag_chain import RAGChain

@pytest.fixture
def rag_chain():
    """
    Fixture to initialize the RAGChain with a template name and model.
    """
    return RAGChain(template_name="test_template", model="gpt-4o")

def test_rag_chain_init(rag_chain):
    """
    Test that the RAGChain initializes correctly with the template, prompt, LLM, and chain.
    If the __init__ method is not implemented, this test will fail.
    """
    assert rag_chain.template_name == "test_template", "Template name should be set correctly"
    assert rag_chain.template is not None, "Template should be defined"
    assert "Technical_context" in rag_chain.template, "Template should contain Technical_context placeholder"
    assert "Incident_records" in rag_chain.template, "Template should contain Incident_records placeholder"
    assert "query" in rag_chain.template, "Template should contain query placeholder"
    
    assert hasattr(rag_chain, 'prompt'), "Prompt should be defined"
    assert hasattr(rag_chain, 'llm'), "Language model should be defined"
    assert hasattr(rag_chain, 'output_parser'), "Output parser should be defined"
    
    assert hasattr(rag_chain, 'chain'), "Chain should be defined"

@patch('langchain_openai.ChatOpenAI.invoke')  
def test_rag_chain_run_valid_query(mock_llm_invoke, rag_chain):
    """
    Test that the run method processes a valid query with relevant documents and returns a response.
    If the run method is not implemented, this test will fail.
    """
    mock_llm_invoke.return_value = "Solution guide:\n1. Step 1\n2. Step 2\nSource: Technical Context"

    query = "How to fix network connectivity?"
    tech_results = ["Technical step 1", "Technical step 2"]
    incident_results = ["Incident record 1"]

    response = rag_chain.run(query, tech_results, incident_results)

    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"
    assert "No Relevant Documentation Found" not in response, "Should not return fallback response for valid query with documents"

def test_rag_chain_run_invalid_query_type(rag_chain):
    """
    Test that the run method raises a ValueError for an invalid query type.
    If the run method is not implemented or doesn't handle input validation, this test will fail.
    """
    query = 123
    tech_results = ["Technical step 1"]
    incident_results = ["Incident record 1"]

    with pytest.raises(ValueError, match="Query must be a string"):
        rag_chain.run(query, tech_results, incident_results)

def test_rag_chain_run_empty_query(rag_chain):
    """
    Test that the run method raises a ValueError for an empty query.
    If the run method is not implemented or doesn't handle input validation, this test will fail.
    """
    query = "   "
    tech_results = ["Technical step 1"]
    incident_results = ["Incident record 1"]

    with pytest.raises(ValueError, match="Query cannot be empty"):
        rag_chain.run(query, tech_results, incident_results)

@patch('langchain_openai.ChatOpenAI.invoke')  
def test_rag_chain_run_no_relevant_documents(mock_llm_invoke, rag_chain):
    """
    Test that the run method returns a fallback response when no relevant documents are found.
    If the run method or _generate_fallback_response is not implemented, this test will fail.
    """
    mock_llm_invoke.return_value = "No specific documentation found for your query. Please consult the product manual."

    query = "How to fix network connectivity?"
    tech_results = "No related document found related to the product"
    incident_results = "No related document found related to the product"

    response = rag_chain.run(query, tech_results, incident_results)

    assert isinstance(response, str), "Response should be a string"
    assert "No Specific Documentation Found" in response, "Should return fallback response when no documents are found"

@patch('langchain_openai.ChatOpenAI.invoke')  
def test_generate_fallback_response(mock_llm_invoke, rag_chain):
    """
    Test that the _generate_fallback_response method returns a fallback message.
    If the _generate_fallback_response method is not implemented, this test will fail.
    """
    mock_llm_invoke.return_value = "No specific documentation found for your query. Please consult the product manual."

    query = "How to fix network connectivity?"
    response = rag_chain._generate_fallback_response(query)

    assert isinstance(response, str), "Response should be a string"
    assert "No Specific Documentation Found" in response, "Fallback response should indicate no documentation found"