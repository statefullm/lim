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

  // Start Docling service if not already running
  static void start_docling_if_needed();

  // Static helper to process local PDF files (used by filesystem.cc)
  static std::string process_local_pdf(const std::string& pdf_binary);

  // Context size limiting utility - defaults to ~100k chars per file (approx 25-30 pages)
  static std::string limit_context_size(const std::string& text, size_t per_file_max = 100000);

  // Strip base64 images from markdown to prevent cache corruption
  static std::string strip_base64_images(const std::string& text);

  // Initialize SSL certificate support (downloads CA bundle if needed)
  static void init_ssl_certificates();

  // Track context usage across an agentic session
  static void reset_context_usage();
  static size_t get_context_usage();

private:
  std::string base_url;

  // Process management
  static void start_searxng_if_needed(const std::string& base_url);

  // Stateful Context Trackers
  static size_t g_cumulative_context_chars;
  static const size_t SESSION_MAX_CHARS;
};

extern std::string HOME;
extern std::string LIM_CONFIG_DIR;
extern std::string LIM_LOG_DIR;
extern std::string LIM_CACHE_DIR;
extern std::string LIM_SAVE_DIR;

#endif // NETWORK_H
