# Agentic-AI-RAG-Langchain
End-to-end Agentic AI project demonstrating LLM agents, ReAct, LangChain, RAG, FAISS vector search, Hugging Face embeddings, Gemini, Tavily web search, fact checking, tool calling, fallback mechanisms, and API retry handling.

# 🤖 Agentic AI Frameworks and Basics

A hands-on **Agentic AI and Large Language Model (LLM) project** demonstrating the fundamental concepts behind AI agents, tool calling, ReAct architecture, Retrieval-Augmented Generation (RAG), vector databases, embeddings, web search, fact checking, fallback mechanisms, API retry handling, and Model Context Protocol (MCP).

This project is implemented primarily using **Python, Google Gemini, LangChain, LangGraph, FAISS, Hugging Face Embeddings, and Tavily**.

The project is designed as a practical learning repository for understanding how modern LLM-powered agents work internally and how different components can be combined to build intelligent applications.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Technologies Used](#-technologies-used)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [API Key Configuration](#-api-key-configuration)
- [Running the Project](#-running-the-project)
- [1. Google Gemini Setup](#1-google-gemini-setup)
- [2. Understanding LLM Temperature](#2-understanding-llm-temperature)
- [3. ReAct Agent](#3-react-agent)
- [4. Custom Tools](#4-custom-tools)
- [5. Vector Embeddings](#5-vector-embeddings)
- [6. FAISS Vector Database](#6-faiss-vector-database)
- [7. RAG Agent](#7-rag-agent)
- [8. Fact Checking Application](#8-fact-checking-application)
- [9. Product Review Analysis](#9-product-review-analysis)
- [10. RAG with Internet Search Fallback](#10-rag-with-internet-search-fallback)
- [11. Text Chunking](#11-text-chunking)
- [12. API Retry Mechanism](#12-api-retry-mechanism)
- [13. Model Context Protocol](#13-model-context-protocol)
- [Example Workflow](#-example-workflow)
- [Important Concepts](#-important-concepts)
- [Known Limitations](#-known-limitations)
- [Security](#-security)
- [Future Improvements](#-future-improvements)
- [Learning Outcomes](#-learning-outcomes)
- [Author](#-author)

---

# 📖 Project Overview

This project explores the building blocks of **Agentic AI systems**.

Traditional LLM applications generally follow a simple pattern:

```text
User
 ↓
LLM
 ↓
Response
