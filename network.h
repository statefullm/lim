#ifndef NETWORK_H
#define NETWORK_H

#include <string>

class NetworkTools {
public:
  NetworkTools(const std::string& searxng_url = "http://127.0.0.1:8888");

  std::string web_search(const std::string& query);

  // Cleanly shut down the background process when llm exits
  static void cleanup_searxng();

private:
  std::string base_url;

  // Lazy-initializer that forks the process on first search
  static void start_searxng_if_needed(const std::string& base_url);
};

extern std::string HOME;

#endif // NETWORK_H
