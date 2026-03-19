#ifndef NETWORK_H
#define NETWORK_H

#include <string>
#include <vector>
#include <map>

class NetworkTools {
public:
  NetworkTools(const std::string& searxng_url = "http://127.0.0.1:8888");

  std::string web_search(const std::string& query);
  std::string fetch_and_clean_html(const std::string& url);

  // Fetch multiple URLs (files, HTML, PDFs) and return as vector of results
  std::vector<std::map<std::string, std::string>> fetch_urls(const std::vector<std::string>& urls);

  // Reset web search state after failures
  void reset_search();

  // Cleanly shut down ALL background processes (SearxNG & Docling)
  static void cleanup_services();

  // PDF processing utilities - made public for filesystem.cc access
  std::string process_pdf_with_docling(const std::string& pdf_binary);
  std::string start_and_wait_for_docling();

  // Start Docling service if not already running
  static void start_docling_if_needed();

  // Static helper to process local PDF files (used by filesystem.cc)
  static std::string process_local_pdf(const std::string& pdf_binary);

  // Context size limiting utility - used for both HTML and PDF content
  static std::string limit_context_size(const std::string& text, size_t max_chars = 80000);

private:
  std::string base_url;

  // Process management
  static void start_searxng_if_needed(const std::string& base_url);
};

extern std::string HOME;

#endif // NETWORK_H
