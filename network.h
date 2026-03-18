#ifndef NETWORK_H
#define NETWORK_H

#include <string>

class NetworkTools {
public:
  NetworkTools(const std::string& searxng_url = "http://127.0.0.1:8888");

  std::string web_search(const std::string& query);
  std::string fetch_and_clean_html(const std::string& url);

  // Reset web search state after failures
  void reset_search();

  // Cleanly shut down ALL background processes (SearxNG & Docling)
  static void cleanup_services();

private:
  std::string base_url;

  // Process management
  static void start_searxng_if_needed(const std::string& base_url);
  static void start_docling_if_needed();

  // PDF processor
  std::string process_pdf_with_docling(const std::string& pdf_binary);
};

extern std::string HOME;

#endif // NETWORK_H
