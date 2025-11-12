"""
Test script to verify all packages are installed correctly and compatible.
Run this AFTER installing requirements.txt
"""

import sys

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name} - FAILED: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {package_name or module_name} - WARNING: {e}")
        return True

def main():
    print("=" * 60)
    print("TESTING PACKAGE COMPATIBILITY")
    print("=" * 60)
    print(f"\nPython Version: {sys.version}")
    print("-" * 60)
    
    all_passed = True
    
    # Test Core Packages
    print("\n📦 CORE PACKAGES:")
    all_passed &= test_import("streamlit", "Streamlit")
    all_passed &= test_import("dotenv", "python-dotenv")
    
    # Test LangChain
    print("\n🔗 LANGCHAIN PACKAGES:")
    all_passed &= test_import("langchain", "LangChain")
    all_passed &= test_import("langchain_core", "LangChain Core")
    all_passed &= test_import("langchain_community", "LangChain Community")
    
    # Test LangChain specific imports
    print("\n🔍 LANGCHAIN COMPONENTS:")
    try:
        from langchain_core.prompts import ChatPromptTemplate
        print("✅ ChatPromptTemplate - OK")
    except ImportError as e:
        print(f"❌ ChatPromptTemplate - FAILED: {e}")
        all_passed = False
    
    try:
        from langchain_core.output_parsers import StrOutputParser
        print("✅ StrOutputParser - OK")
    except ImportError as e:
        print(f"❌ StrOutputParser - FAILED: {e}")
        all_passed = False
    
    # Test LLM Providers
    print("\n🤖 LLM PROVIDERS:")
    all_passed &= test_import("langchain_openai", "LangChain OpenAI")
    all_passed &= test_import("langchain_groq", "LangChain Groq")
    all_passed &= test_import("langchain_google_genai", "LangChain Google GenAI")
    
    # Test Vector DB
    print("\n💾 VECTOR DATABASE:")
    all_passed &= test_import("chromadb", "ChromaDB")
    
    # Test Embeddings
    print("\n🧮 EMBEDDING MODELS:")
    all_passed &= test_import("sentence_transformers", "Sentence Transformers")
    
    # Test Core Dependencies
    print("\n🔧 CORE DEPENDENCIES:")
    all_passed &= test_import("pydantic", "Pydantic")
    all_passed &= test_import("tiktoken", "Tiktoken")
    all_passed &= test_import("numpy", "NumPy")
    all_passed &= test_import("requests", "Requests")
    
    # Test Optional Dependencies
    print("\n⚙️  OPTIONAL DEPENDENCIES:")
    test_import("torch", "PyTorch")
    test_import("transformers", "Transformers")
    
    # Final Result
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CRITICAL PACKAGES INSTALLED SUCCESSFULLY!")
        print("=" * 60)
        print("\n🚀 You can now run: streamlit run streamlit_app.py")
    else:
        print("❌ SOME PACKAGES FAILED TO IMPORT")
        print("=" * 60)
        print("\n⚠️  Please reinstall failed packages or check Python version")
        print("   Recommended: Python 3.10 or 3.11")
    print()

if __name__ == "__main__":
    main()