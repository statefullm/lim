#include "network.h"
#include "filesystem.h"
#include "output.h"
#include "taskset.h"
#include "session_utils.h"
#include <curl/curl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>
#include <sys/wait.h>
#include <chrono>
#include <thread>
#include <functional>
#include <algorithm>
#include "json.hpp"
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <cstring>  // For strlen
#include <ctime>    // For time()

using json = nlohmann::json;
using namespace std;

// Helper: read an env var with a fallback default value.
static string getenvOrDefault(const char* name, const string& fallback) {
  const char* env = getenv(name);
  return (env && env[0]) ? string(env) : fallback;
}

// --- Helper to strip trailing whitespace from each line ---
static std::string strip_trailing_whitespace(const std::string& text) {
  std::string result;
  std::istringstream iss(text);
  std::string line;
  while (std::getline(iss, line)) {
    // Find last non-whitespace character
    size_t end = line.find_last_not_of(" \t\r\n");
    if (end != std::string::npos) {
      result += line.substr(0, end + 1) + "\n";
    } else {
      // Line is all whitespace or empty - keep as-is for blank lines
      result += "\n";
    }
  }
  return result;
}

// --- Global Config & State ---
string SEARXNG_LOG_PATH;
string DOCLING_LOG_PATH;
string HOME;
string LIM_CONFIG_DIR;
string LIM_LOG_DIR;
string LIM_CACHE_DIR;
string LIM_SAVE_DIR;
static struct HomeInit { HomeInit() {
  const char* h = getenv("HOME");
  HOME = h ? h : "";
  const char* c = getenv("LIM_CONFIG_DIR");
  if (c && c[0]) {
    LIM_CONFIG_DIR = c;
  } else {
    LIM_CONFIG_DIR = HOME + "/.config/lim";
  }
  const char* l = getenv("LIM_LOG_DIR");
  if (l && l[0]) {
    LIM_LOG_DIR = l;
  } else {
    LIM_LOG_DIR = "log";
  }
  const char* k = getenv("LIM_CACHE_DIR");
  if (k && k[0]) {
    LIM_CACHE_DIR = k;
  } else {
    LIM_CACHE_DIR = ".cache";
  }
  const char* s = getenv("LIM_SAVE_DIR");
  if (s && s[0]) {
    LIM_SAVE_DIR = s;
  } else {
    LIM_SAVE_DIR = ".";
  }
  SEARXNG_LOG_PATH = LIM_LOG_DIR + "/searxng.log";
  DOCLING_LOG_PATH = LIM_LOG_DIR + "/docling.log";
} } g_homeInit;

extern bool is_debug;
extern volatile sig_atomic_t stop_generation;  // Forward declaration for interrupt checking

static pid_t g_searxng_pid = -1;
static pid_t g_docling_pid = -1;
static bool g_searxng_disabled = false;
static int g_consecutive_empty_searches = 0;

static chrono::steady_clock::time_point g_last_network_request = chrono::steady_clock::now() - chrono::seconds(3);

// --- Configurable web/network limits (all overridable via LIM_* env vars) ---
static int g_search_cooldown_seconds = []() -> int {
  const char* env = getenv("LIM_SEARCH_COOLDOWN");
  if (env && strlen(env) > 0) { int v = atoi(env); if (v > 0) return v; }
  return 3;
}();

static size_t g_web_html_max_bytes = []() -> size_t {
  const char* env = getenv("LIM_WEB_HTML_MAX");
  if (env && strlen(env) > 0) { long v = atol(env); if (v > 0) return static_cast<size_t>(v); }
  return 500000;
}();

static size_t g_web_pdf_max_bytes = []() -> size_t {
  const char* env = getenv("LIM_WEB_PDF_MAX");
  if (env && strlen(env) > 0) { long v = atol(env); if (v > 0) return static_cast<size_t>(v); }
  return 50000000;
}();

static long g_web_timeout_seconds = []() -> long {
  const char* env = getenv("LIM_WEB_TIMEOUT");
  if (env && strlen(env) > 0) { long v = atol(env); if (v > 0) return v; }
  return 600;
}();

static string g_brave_api_key = []() -> string {
  const char* env = getenv("LIM_BRAVE_API_KEY");
  return (env && env[0]) ? string(env) : "";
}();

// Session web-content budget: LIM_CTX * LIM_WEB_CONTEXT_FRACTION * 4 bytes/token.
static size_t compute_session_max_chars() {
  int n_ctx = LIM_DEFAULT_CTX;
  const char* ctx_env = getenv("LIM_CTX");
  if (ctx_env && strlen(ctx_env) > 0) {
    int val = atoi(ctx_env);
    if (val > 0) n_ctx = val;
  }
  double fraction = 0.75;
  const char* frac_env = getenv("LIM_WEB_CONTEXT_FRACTION");
  if (frac_env && strlen(frac_env) > 0) {
    double val = atof(frac_env);
    if (val > 0.0 && val <= 1.0) fraction = val;
  }
  return static_cast<size_t>(n_ctx * fraction * 4);
}

static size_t g_session_max_chars = compute_session_max_chars();

size_t NetworkTools::g_cumulative_context_chars = 0;
const size_t NetworkTools::SESSION_MAX_CHARS = g_session_max_chars;

static size_t g_web_file_max_chars = []() -> size_t {
  const char* env = getenv("LIM_WEB_FILE_MAX");
  if (env && strlen(env) > 0) { long v = atol(env); if (v > 0) return static_cast<size_t>(v); }
  // Default: 25% of the session budget, so ~3-4 files fill the budget.
  return g_session_max_chars / 4;
}();

void NetworkTools::reset_context_usage() {
  g_cumulative_context_chars = 0;
}

// --- Interrupt-aware curl callback to check for SIGINT during long operations ---
static int interrupt_check_callback(void *clientp, double dltotal, double dlnow, double ultotal, double ulnow) {
  // Return non-zero to abort the transfer if stop_generation is set
  return stop_generation ? 1 : 0;
}

// --- RAG Fetcher Callbacks & State ---
struct FetchState {
  string buffer;
  bool is_text;
  bool is_pdf;
  bool exceeded_limit;  // Track if content exceeded size limit (without causing curl error 23)
  FetchState() : is_text(true), is_pdf(false), exceeded_limit(false) {}
};

// Helper to detect PDF by magic bytes (first 5 bytes should be "%PDF-")
static bool is_pdf_by_magic(const string& buffer) {
  if (buffer.size() >= 5 && buffer.substr(0, 5) == "%PDF-") {
    return true;
  }
  return false;
}

static size_t HeaderCallback(char *buffer, size_t size, size_t nitems, void *userdata) {
  size_t numbytes = size * nitems;
  string header(buffer, numbytes);
  FetchState* state = (FetchState*)userdata;

  string lower_header = header;
  transform(lower_header.begin(), lower_header.end(), lower_header.begin(), [](unsigned char c){ return std::tolower(c); });

  // Detect HTTP status line (e.g., "HTTP/1.1 301" or "HTTP/2 200")
  // Reset state on redirects to handle intermediate responses correctly
  if (lower_header.find("http/") == 0) {
    size_t space_pos = lower_header.find(' ');
    if (space_pos != string::npos) {
      string status_code_str = lower_header.substr(space_pos + 1);
      // Check if this is a redirect response (3xx)
      if (status_code_str.size() >= 3) {
        char* endp = nullptr;
        long status_code = strtol(status_code_str.c_str(), &endp, 10);
        if (*endp == '\0' && status_code >= 300 && status_code < 400) {
          // This is a redirect - reset state for the final response
          state->is_text = false;
          state->is_pdf = false;
        }
      }
    }
  }

  if (lower_header.find("content-type:") == 0) {
    if (lower_header.find("application/pdf") != string::npos) {
      state->is_pdf = true;
      state->is_text = false;
    } else if (lower_header.find("text/html") != string::npos ||
               lower_header.find("text/plain") != string::npos) {
      // Explicitly mark as text content
      state->is_text = true;
    }
    // If Content-Type is present but not text/pdf, leave is_text=false (skip binary)
  }
  return numbytes;
}

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
  FetchState* state = (FetchState*)userp;

  // If neither flag is set yet (headers not fully processed), default to text mode
  if (!state->is_text && !state->is_pdf) {
    state->is_text = true;  // Default assumption for unknown content
  }

  size_t total_size = size * nmemb;
  size_t max_size = state->is_pdf ? g_web_pdf_max_bytes : g_web_html_max_bytes;

  // CRITICAL: If we've already exceeded the limit, skip buffering entirely
  if (state->exceeded_limit) {
    return total_size;  // Acknowledge receipt but don't buffer
  }

  if (state->buffer.size() + total_size > max_size) {
    // Buffer would exceed limit - set flag but DON'T return 0
    // Returning 0 causes curl error 23 "Failed writing output"
    state->exceeded_limit = true;
    return total_size;  // Continue accepting data but stop buffering
  }

  state->buffer.append((char*)contents, total_size);
  return total_size;
}

static size_t StringWriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
  ((string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

static size_t DummyWriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
  return size * nmemb;
}

void NetworkTools::cleanup_services() {
  if (g_searxng_pid > 0) {
    kill(-g_searxng_pid, SIGKILL);
    waitpid(g_searxng_pid, NULL, 0); // Reap the zombie
    g_searxng_pid = -1;
  }

  if (g_docling_pid > 0) {
    kill(-g_docling_pid, SIGKILL);
    waitpid(g_docling_pid, NULL, 0); // Reap the zombie
    g_docling_pid = -1;
  }
}

// --- Shared startup logic for local background services (SearxNG, Docling) ---
//   1. Probe probe_url; if something is already responding, mark *pid = -2 and return.
//   2. Otherwise fork, redirect logs to log_path, and exec cmd in a new process group.
//   3. Wait up to 40x500ms for probe_url to respond; on timeout kill the child
//      and reset *pid = -1 so the next request can retry.
static void start_service(const string& name, const string& probe_url,
                          const string& log_path, const string& cmd, pid_t& pid) {
  // Check if it's already running externally
  CURL *curl = curl_easy_init();
  if (curl) {
    curl_easy_setopt(curl, CURLOPT_URL, probe_url.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 500L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, DummyWriteCallback);
    if (curl_easy_perform(curl) == CURLE_OK) {
      pid = -2;
      curl_easy_cleanup(curl);
      return;
    }
    curl_easy_cleanup(curl);
  }

  std::cerr << "\033[0mSpinning up local " + name + " instance..." << std::endl;

  pid_t child = fork();
  if (child == 0) {
    setpgid(0, 0);
    freopen(log_path.c_str(), "w", stdout);
    freopen(log_path.c_str(), "w", stderr);

    execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)NULL);
    exit(1);
  } else if (child > 0) {
    pid = child;

    // --- Wait for the service to actually wake up! ---
    std::cerr << "\033[0mWaiting for " + name + " to become ready..." << std::endl;
    CURL *wait_curl = curl_easy_init();
    if (wait_curl) {
      curl_easy_setopt(wait_curl, CURLOPT_URL, probe_url.c_str());
      curl_easy_setopt(wait_curl, CURLOPT_WRITEFUNCTION, DummyWriteCallback);
      curl_easy_setopt(wait_curl, CURLOPT_TIMEOUT_MS, 500L);

      int retries = 0;
      while (retries < 40) {
        if (stop_generation) {
          std::cerr << "\033[0m" + name + " startup interrupted by user" << std::endl;
          break;
        }
        if (curl_easy_perform(wait_curl) == CURLE_OK) {
          std::cerr << "\033[0m" + name + " is ready and responding!" << std::endl;
          break;
        }
        this_thread::sleep_for(chrono::milliseconds(500));
        retries++;
      }
      curl_easy_cleanup(wait_curl);

      // If the service never became ready after max retries, reset PID so it
      // can be retried on the next request (e.g., after a dependency is fixed).
      if (retries >= 40) {
        std::cerr << "\033[0m" + name + " failed to start after waiting. Will retry on next request." << std::endl;
        kill(-child, SIGKILL);
        waitpid(child, NULL, 0);
        pid = -1;
      }
    }
  }
}

void NetworkTools::start_searxng_if_needed(const string& base_url) {
  if (g_searxng_pid != -1) return;
  string cmd = Taskset::e_core_taskset()+"cd "+HOME+"/searxng && exec python -m searx.webapp";
  start_service("SearxNG", base_url, SEARXNG_LOG_PATH, cmd, g_searxng_pid);
}

void NetworkTools::start_docling_if_needed() {
  if (g_docling_pid != -1) return;
  string docling_cmd = getenvOrDefault("LIM_DOCLING_CMD", HOME+"/venv/bin/docling-serve run --enable-ui");
  string cmd = "UVICORN_LOG_LEVEL=error CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=8 exec "+Taskset::p_core_taskset()+docling_cmd;
  start_service("Docling", "http://127.0.0.1:5001/docs", DOCLING_LOG_PATH, cmd, g_docling_pid);
}

// --- Strict Base64 Helper (No Newlines) ---
static string base64_encode(const string &in) {
  static const char lookup[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  string out;
  int val = 0, valb = -6;
  for (unsigned char c : in) {
    val = (val << 8) + c;
    valb += 8;
    while (valb >= 0) {
      out.push_back(lookup[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6) out.push_back(lookup[((val << 8) >> (valb + 8)) & 0x3F]);
  while (out.size() % 4) out.push_back('=');
  return out;
}

// --- SSL Certificate Initialization with Cache ---
void NetworkTools::init_ssl_certificates() {
  static bool initialized = false;
  if (initialized) return;

  struct stat st;

  // Use LIM_CONFIG_DIR for certificate storage
  string cached_ca = LIM_CONFIG_DIR + "/combined-ca.crt";

  const int MAX_AGE_SECONDS = 30 * 24 * 60 * 60; // 30 days in seconds

  // Check if cached CA exists and is not older than 30 days
  bool needs_update = false;
  if (stat(cached_ca.c_str(), &st) != 0 || st.st_size == 0) {
    needs_update = true;
  } else {
    time_t now = time(NULL);
    if (now - st.st_mtime > MAX_AGE_SECONDS) {
      needs_update = true;
    }
  }

  if (needs_update) {
    std::cerr << "\033[0mUpdating SSL certificate cache..." << std::endl;

    // Get system CA bundle path
    string system_ca;
    const char* ca_paths[] = {
      "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
      "/etc/ssl/certs/ca-certificates.crt",
      "/etc/pki/tls/certs/ca-bundle.crt",
      NULL
    };

    for (const char* path : ca_paths) {
      if (stat(path, &st) == 0 && st.st_size > 0) {
        system_ca = path;
        break;
      }
    }

    // Download Cloudflare certificate chain to config directory
    string cloudflare_cache = LIM_CONFIG_DIR + "/cloudflare-chain.pem";
    string cloudflare_cmd = "openssl s_client -connect example.com:443 -showcerts 2>/dev/null | "
      "awk '/BEGIN CERTIFICATE/,/END CERTIFICATE/{if(/BEGIN CERTIFICATE/)p=1; if(p)print; if(/END CERTIFICATE/)p=0}' > " + cloudflare_cache;
    system(cloudflare_cmd.c_str());

    // Combine CA bundles into cache directory
    string combine_cmd;
    if (!system_ca.empty()) {
      combine_cmd = "cat " + system_ca + " " + cloudflare_cache + " > " + cached_ca;
    } else {
      // Download standard CA bundle and add Cloudflare certs to cache
      string temp_bundle = LIM_CONFIG_DIR + "/ca-bundle-temp.crt";
      combine_cmd = "curl -s https://curl.se/ca/cacert.pem > " + temp_bundle + " && "
        "cat " + temp_bundle + " " + cloudflare_cache + " > " + cached_ca;
    }

    int result = system(combine_cmd.c_str());
    if (result == 0 && stat(cached_ca.c_str(), &st) == 0 && st.st_size > 0) {
      std::cerr << "\033[0mCreated combined CA bundle in cache: " + cached_ca << std::endl;
      setenv("CURL_CA_BUNDLE", cached_ca.c_str(), 1);
    } else {
      std::cerr << "\033[0mFailed to create combined CA bundle - using curl defaults" << std::endl;
    }
  } else {
    cout << "\033[0m" << "Using cached SSL certificate from: " << cached_ca << endl;
    consoleMarkNewline(true);
    cout.flush();
    setenv("CURL_CA_BUNDLE", cached_ca.c_str(), 1);
  }

  initialized = true;
}

// --- Smart Context Truncation (Stateful) ---
string NetworkTools::limit_context_size(const string& text, size_t per_file_max) {
  // Use env-configured default if caller passed 0.
  if (per_file_max == 0) per_file_max = g_web_file_max_chars;

  // Strip base64 images to prevent corrupting tags
  string cleaned_text = NetworkTools::strip_base64_images(text);

  // Calculate how much budget is left in the overall session
  size_t remaining_budget = 0;
  if (g_cumulative_context_chars < SESSION_MAX_CHARS) {
    remaining_budget = SESSION_MAX_CHARS - g_cumulative_context_chars;
  }

  // If memory budget is exhausted, forcefully stop the LLM from loading more
  if (remaining_budget < 5000) {
    return "[SYSTEM NOTIFICATION: Context memory budget is full. Cannot load more documents. Please rely on existing memory or ask the user to type 'clear' to reset the chat.]";
  }

  // The limit for this specific file is whichever is smaller
  size_t active_limit = std::min(per_file_max, remaining_budget);

  // If it fits within the active limit, update the tracker and return it whole
  if (cleaned_text.length() <= active_limit) {
    g_cumulative_context_chars += cleaned_text.length();
    return cleaned_text;
  }

  // Otherwise, apply the middle-drop heuristic
  size_t head_size = (active_limit * 6) / 10;
  size_t tail_size = (active_limit * 4) / 10;

  size_t head_end = cleaned_text.rfind(' ', head_size);
  if (head_end == string::npos) head_end = head_size;

  size_t tail_start = cleaned_text.length() - tail_size;
  size_t actual_tail_start = cleaned_text.find(' ', tail_start);
  if (actual_tail_start == string::npos) actual_tail_start = tail_start;

  string truncated = cleaned_text.substr(0, head_end);
  truncated += "\n\n... [MASSIVE CONTENT OMITTED DUE TO CONTEXT LIMITS] ...\n\n";
  truncated += cleaned_text.substr(actual_tail_start);

  // Update global usage
  g_cumulative_context_chars += truncated.length();

  return truncated;
}

// --- Helper to configure SSL certificate options for curl ---
static void configure_curl_ssl(CURL* curl, const string& base_url) {
  bool is_https = base_url.substr(0, 8) == "https://";
  if (is_https) {
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

    // Use the cached CA bundle from LIM_CONFIG_DIR
    string cached_ca = LIM_CONFIG_DIR + "/combined-ca.crt";
    struct stat st;
    if (stat(cached_ca.c_str(), &st) == 0 && st.st_size > 0) {
      curl_easy_setopt(curl, CURLOPT_CAINFO, cached_ca.c_str());
      cerr << "\033[0mUsing cached CA bundle: " + cached_ca << endl;
    } else {
      // Fall back to system CA bundle paths
      static const char* ca_paths[] = {
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
        "/etc/pki/tls/certs/ca-bundle.crt",
        NULL
      };
      bool found = false;
      for (const char* path : ca_paths) {
        if (stat(path, &st) == 0 && st.st_size > 0) {
          curl_easy_setopt(curl, CURLOPT_CAINFO, path);
          cerr << "\033[0mUsing system CA bundle: " + string(path) << endl;
          found = true;
          break;
        }
      }
      if (!found) {
        static const char* default_ca = "";
        curl_easy_setopt(curl, CURLOPT_CAINFO, default_ca);
        cerr << "\033[0mUsing curl's built-in system certificate store" << endl;
      }
    }
  } else {
    // HTTP requests (localhost, http:// URLs) - disable SSL verification options
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    cerr << "\033[0mHTTP request detected - SSL verification disabled" << endl;
  }
}

// --- Helper to configure common curl options for fetch operations ---
// Returns the header list that caller must free with curl_slist_free_all()
static struct curl_slist* configure_curl_fetch(CURL* curl, const string& url) {
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, HeaderCallback);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, nullptr);  // Must be overridden by caller before perform
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, nullptr);    // Must be overridden by caller before perform
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, g_web_timeout_seconds);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0");
  curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "gzip, deflate");

  // Add common headers to avoid 403 from servers that check for browser-like requests
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Accept: */*");
  headers = curl_slist_append(headers, "Accept-Language: en-US,en;q=0.5");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

  configure_curl_ssl(curl, url);

  // Enable interrupt checking during transfer
  curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, interrupt_check_callback);
  curl_easy_setopt(curl, CURLOPT_XFERINFODATA, nullptr);

  return headers;  // Caller must free with curl_slist_free_all()
}

// --- Buffered fetch shared by fetch_and_clean_html() and the PDF branch of
//     fetch_urls() ---
// Runs a size-limited, content-type-aware curl transfer into 'state'.
// Returns {curl_res, http_code}.  On curl_easy_init failure returns
// {CURLE_FAILED_INIT, 0} with state untouched.
static pair<CURLcode, long> curl_fetch_buffer(const string& url, FetchState& state) {
  CURL *curl = curl_easy_init();
  if (!curl) return {CURLE_FAILED_INIT, 0};

  struct curl_slist* headers = configure_curl_fetch(curl, url);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, &state);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &state);

  CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  return {res, http_code};
}

// --- Strip Base64 Images from Text ---
// Removes data:image/*;base64,... patterns to prevent cache corruption
string NetworkTools::strip_base64_images(const string& text) {
  string result = text;

  // Pattern: ![Image](data:image/...;base64,.......)
  size_t pos = 0;
  while ((pos = result.find("![Image](data:image/", pos)) != string::npos) {
    size_t end_pos = result.find(")", pos);
    if (end_pos != string::npos) {
      // Replace with placeholder instead of removing entirely to preserve structure
      result.replace(pos, end_pos - pos + 1, "[IMAGE OMITTED]");
      pos += strlen("[IMAGE OMITTED]");
    } else {
      break;
    }
  }

  return result;
}

string NetworkTools::process_pdf_with_docling(const string& pdf_binary) {
  start_docling_if_needed();

  cerr << "\033[0mUploading PDF to Docling (Strict JSON Schema)..." << endl;

  CURL *curl = curl_easy_init();
  if (!curl) return "[Curl Init Failed]";

  string readBuffer;
  string docling_url = "http://127.0.0.1:5001/v1/convert/source";

  // Build the JSON payload EXACTLY as defined in your schema
  json payload;

  // We only send the minimum required options to avoid Pydantic validation traps
  payload["options"] = {
    {"to_formats", {"md"}},
    // Using false here helps prevent crashes on malformed arXiv PDFs
    {"abort_on_error", false}
  };

  json source_obj;
  source_obj["kind"] = "file";
  source_obj["filename"] = "input.pdf";
  source_obj["base64_string"] = base64_encode(pdf_binary);

  payload["sources"] = json::array({source_obj});
  payload["target"] = {{"kind", "inbody"}};

  string json_str = payload.dump();

  struct curl_slist *headers = NULL;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  // CRITICAL FIX: Disable curl's automatic 100-continue for large payloads
  headers = curl_slist_append(headers, "Expect:");

  curl_easy_setopt(curl, CURLOPT_URL, docling_url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str.c_str());
  // Explicitly declare size so FastAPI allocates memory correctly
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_str.length());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, StringWriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, g_web_timeout_seconds);
  // Enable interrupt checking during transfer
  curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, interrupt_check_callback);
  curl_easy_setopt(curl, CURLOPT_XFERINFODATA, NULL);

  long http_code = 0;
  CURLcode res = curl_easy_perform(curl);

  // Check for interrupt after completion
  if (stop_generation) {
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return "[PDF conversion interrupted by user]";
  }
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (http_code == 200) {
    try {
      auto j = json::parse(readBuffer);
      string full_md = j["document"]["md_content"].get<string>();
      return NetworkTools::limit_context_size(full_md); // Apply smart truncation
    } catch (...) { return "[Docling JSON Parse Error]"; }
  }

  cerr << "\033[0mDocling Final Failure: " + to_string(http_code) + " | " + readBuffer << endl;
  return "[Docling Error " + to_string(http_code) + "]";
}

// --- Recursive visible-text extraction from an XML node tree ---
// Skips script, style, noscript, template, svg, canvas, and hidden elements.
static void extract_visible_text(xmlNodePtr node, string& out) {
  if (!node) return;

  if (node->type == XML_ELEMENT_NODE && node->name) {
    string tag((const char*)node->name);
    if (tag == "script" || tag == "style" || tag == "noscript" ||
        tag == "template" || tag == "svg" || tag == "canvas" ||
        tag == "iframe" || tag == "object") {
      return;
    }

    // Check for hidden attribute
    xmlChar* hidden_attr = xmlGetProp(node, (const xmlChar*)"hidden");
    if (hidden_attr) { xmlFree(hidden_attr); return; }

    // Check for display:none / visibility:hidden in style attribute
    xmlChar* style_attr = xmlGetProp(node, (const xmlChar*)"style");
    if (style_attr) {
      string s((const char*)style_attr);
      transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
      if (s.find("display:none") != string::npos ||
          s.find("display: none") != string::npos ||
          s.find("visibility:hidden") != string::npos ||
          s.find("visibility: hidden") != string::npos) {
        xmlFree(style_attr);
        return;
      }
      xmlFree(style_attr);
    }
  }

  if (node->type == XML_TEXT_NODE || node->type == XML_CDATA_SECTION_NODE) {
    if (node->content) out += (const char*)node->content;
  }

  for (xmlNodePtr child = node->children; child; child = child->next) {
    extract_visible_text(child, out);
  }
}

// --- Extract <title> and meta description from document head ---
static pair<string, string> extract_page_metadata(xmlDocPtr doc) {
  string title, description;
  if (!doc || !doc->children) return {title, description};

  for (xmlNodePtr cur = doc->children; cur; cur = cur->next) {
    if (cur->type != XML_ELEMENT_NODE || !cur->name) continue;
    string tag((const char*)cur->name);
    if (tag != "head" && tag != "title") {
      // Also handle case where <title> is a direct child of root (malformed HTML)
      if (tag == "title") {
        xmlChar* content = xmlNodeGetContent(cur);
        if (content) { title = (const char*)content; xmlFree(content); }
        break;
      }
      continue;
    }

    // Walk head children (or just process title directly)
    xmlNodePtr start = (tag == "title") ? cur : cur->children;
    for (xmlNodePtr hchild = start; hchild; hchild = (tag == "title") ? nullptr : hchild->next) {
      if (hchild->type != XML_ELEMENT_NODE || !hchild->name) continue;
      string htag((const char*)hchild->name);

      if (htag == "title") {
        xmlChar* content = xmlNodeGetContent(hchild);
        if (content) { title = (const char*)content; xmlFree(content); }
      } else if (htag == "meta") {
        xmlChar* name = xmlGetProp(hchild, (const xmlChar*)"name");
        xmlChar* prop = xmlGetProp(hchild, (const xmlChar*)"property");
        xmlChar* content = xmlGetProp(hchild, (const xmlChar*)"content");
        if (content) {
          string key = name ? (const char*)name : (prop ? (const char*)prop : "");
          if (key == "description" || key == "og:description" || key == "twitter:description") {
            description = (const char*)content;
          }
        }
        if (name) xmlFree(name);
        if (prop) xmlFree(prop);
        if (content) xmlFree(content);
      }
    }
    break; // Only process first <head> or <title> found
  }
  return {title, description};
}

// --- Find <body> element in document tree ---
static xmlNodePtr find_body_element(xmlDocPtr doc) {
  if (!doc || !doc->children) return nullptr;

  function<xmlNodePtr(xmlNodePtr)> dfs = [&](xmlNodePtr node) -> xmlNodePtr {
    if (!node) return nullptr;
    if (node->type == XML_ELEMENT_NODE && node->name) {
      string tag((const char*)node->name);
      if (tag == "body") return node;
    }
    for (xmlNodePtr child = node->children; child; child = child->next) {
      xmlNodePtr found = dfs(child);
      if (found) return found;
    }
    return nullptr;
  };

  return dfs(doc->children);
}

// --- Return API endpoint hints for known SPA sites ---
static string api_hint_for_url(const string& url) {
  if (url.find("github.com/") != string::npos) {
    if (url.find("/actions/") != string::npos)
      return "Tip: Use the GitHub API via exec_shell: curl -s https://api.github.com/repos/<owner>/<repo>/actions/runs/<run_id>";
    if (url.find("/pulls/") != string::npos || url.find("/issues/") != string::npos)
      return "Tip: Use the GitHub API via exec_shell: curl -s https://api.github.com/repos/<owner>/<repo>/pulls/<num>";
    return "Tip: Use the GitHub API via exec_shell: curl -s https://api.github.com/repos/<owner>/<repo>";
  }
  if (url.find("stackoverflow.com/questions/") != string::npos)
    return "Tip: Stack Overflow has an API: curl -s 'https://api.stackexchange.com/2.3/questions/<id>?site=stackoverflow&filter=withbody'";
  if (url.find("gitlab.com/") != string::npos)
    return "Tip: Use the GitLab API via exec_shell: curl -s https://gitlab.com/api/v4/projects/<project_id>";
  return "";
}

// --- GitHub API auto-fetch ---
// When a GitHub URL is detected, try to fetch content via the API before
// attempting HTML scrape (which would fail on SPA pages).
// Returns formatted text on success, or "" on failure (caller falls through to HTML).
static string github_api_fetch(const string& url) {
  // Parse: https://github.com/{owner}/{repo}/{section}/...
  const string prefix = "github.com/";
  size_t github_pos = url.find(prefix);
  if (github_pos == string::npos) return "";

  string path = url.substr(github_pos + prefix.length());

  // Remove query params and fragments
  size_t qmark = path.find('?');
  if (qmark != string::npos) path = path.substr(0, qmark);
  size_t frag = path.find('#');
  if (frag != string::npos) path = path.substr(0, frag);

  // Split path into segments
  vector<string> segs;
  size_t start = 0;
  while (start < path.length()) {
    size_t slash = path.find('/', start);
    if (slash == string::npos) { segs.push_back(path.substr(start)); break; }
    segs.push_back(path.substr(start, slash - start));
    start = slash + 1;
  }

  if (segs.size() < 3) return ""; // Need at least owner/repo/section
  string owner = segs[0];
  string repo = segs[1];
  string section = segs[2];

  // Build API base
  string api_base = "https://api.github.com/repos/" + owner + "/" + repo;

  // Get optional token for higher rate limits
  const char* token = getenv("GH_TOKEN");
  string auth_header = "";
  if (token && token[0]) {
    auth_header = string("Authorization: Bearer ") + token;
  }

  string api_url;
  bool is_graphql = false;
  string graphql_query;

  if (section == "actions" && segs.size() >= 5 && segs[3] == "runs") {
    // /actions/runs/{id}
    api_url = api_base + "/actions/runs/" + segs[4];
  } else if ((section == "pulls" || section == "issues") && segs.size() >= 4) {
    // /pulls/{number} or /issues/{number} -> use issues API (PRs are issues)
    api_url = api_base + "/issues/" + segs[3];
  } else if (section == "discussions" && segs.size() >= 4) {
    // /discussions/{number} -> GraphQL only
    is_graphql = true;
    graphql_query = "{\"query\":\"{ repository(owner: \\\"" + owner + "\\\", name: \\\"" + repo + "\\\") { discussion(number: " + segs[3] + ") { title bodyText comments(first: 10) { nodes { bodyText author { login } } } } } }\"}";
    api_url = "https://api.github.com/graphql";
  } else if (section == "blob" && segs.size() >= 5) {
    // /blob/{branch}/{path...} -> contents API
    string branch = segs[3];
    string file_path;
    for (size_t i = 4; i < segs.size(); i++) {
      if (!file_path.empty()) file_path += "/";
      file_path += segs[i];
    }
    api_url = api_base + "/contents/" + file_path + "?ref=" + branch;
  } else {
    return ""; // Unsupported GitHub URL pattern
  }

  cerr << "\033[0mGitHub API fetch: " << (is_graphql ? "GraphQL" : api_url) << endl;

  NetworkTools::init_ssl_certificates();

  CURL *curl = curl_easy_init();
  if (!curl) return "";

  string readBuffer;
  struct curl_slist *headers = NULL;
  headers = curl_slist_append(headers, "User-Agent: LIM/1.0");
  headers = curl_slist_append(headers, "Accept: application/vnd.github+json");
  headers = curl_slist_append(headers, "X-GitHub-Api-Version: 2026-03-10");
  if (!auth_header.empty()) {
    headers = curl_slist_append(headers, auth_header.c_str());
  }

  curl_easy_setopt(curl, CURLOPT_URL, api_url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, StringWriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
  configure_curl_ssl(curl, api_url);

  if (is_graphql) {
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, graphql_query.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)graphql_query.length());
  }

  CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (stop_generation) return "";
  if (res != CURLE_OK) {
    cerr << "\033[0mGitHub API curl error: " << curl_easy_strerror(res) << endl;
    return "";
  }
  if (http_code == 403 || http_code == 429) {
    cerr << "\033[0mGitHub API rate limited (HTTP " << http_code << ")" << endl;
    return "";
  }
  if (http_code != 200) {
    cerr << "\033[0mGitHub API returned HTTP " << http_code << endl;
    return "";
  }

  try {
    auto j = json::parse(readBuffer);

    // Check for GraphQL errors
    if (is_graphql && j.contains("errors") && j["errors"].is_array() && !j["errors"].empty()) {
      cerr << "\033[0mGitHub GraphQL error: " << j["errors"][0].value("message", "?") << endl;
      return "";
    }

    string result = "";

    if (section == "actions" && segs.size() >= 5 && segs[3] == "runs") {
      // Actions run
      result = "[GitHub Actions Run]\n";
      if (j.contains("name")) result += "Workflow: " + j["name"].get<string>() + "\n";
      if (j.contains("display_title")) result += "Title: " + j["display_title"].get<string>() + "\n";
      if (j.contains("head_branch")) result += "Branch: " + j["head_branch"].get<string>() + "\n";
      if (j.contains("head_sha")) result += "Commit: " + j["head_sha"].get<string>().substr(0, 7) + "\n";
      if (j.contains("status")) result += "Status: " + j["status"].get<string>() + "\n";
      if (j.contains("conclusion") && !j["conclusion"].is_null())
        result += "Conclusion: " + j["conclusion"].get<string>() + "\n";
      if (j.contains("created_at")) result += "Created: " + j["created_at"].get<string>() + "\n";
      if (j.contains("updated_at")) result += "Updated: " + j["updated_at"].get<string>() + "\n";
      if (j.contains("run_number")) result += "Run #" + to_string(j["run_number"].get<int>()) + "\n";

    } else if ((section == "pulls" || section == "issues") && segs.size() >= 4) {
      // Issue or PR
      result = string("[GitHub ") + (section == "pulls" ? "Pull Request" : "Issue") + "]\n";
      if (j.contains("title")) result += "Title: " + j["title"].get<string>() + "\n";
      if (j.contains("state")) result += "State: " + j["state"].get<string>() + "\n";
      if (j.contains("user") && j["user"].contains("login"))
        result += "Author: " + j["user"]["login"].get<string>() + "\n";
      if (j.contains("created_at")) result += "Created: " + j["created_at"].get<string>() + "\n";
      if (j.contains("body") && !j["body"].is_null())
        result += "\nBody:\n" + j["body"].get<string>() + "\n";

      // Fetch first few comments
      string comments_url = api_base + "/issues/" + segs[3] + "/comments?per_page=5";
      CURL *c2 = curl_easy_init();
      if (c2) {
        string comments_buf;
        struct curl_slist *h2 = NULL;
        h2 = curl_slist_append(h2, "User-Agent: LIM/1.0");
        h2 = curl_slist_append(h2, "Accept: application/vnd.github+json");
        if (!auth_header.empty()) h2 = curl_slist_append(h2, auth_header.c_str());
        curl_easy_setopt(c2, CURLOPT_URL, comments_url.c_str());
        curl_easy_setopt(c2, CURLOPT_HTTPHEADER, h2);
        curl_easy_setopt(c2, CURLOPT_WRITEFUNCTION, StringWriteCallback);
        curl_easy_setopt(c2, CURLOPT_WRITEDATA, &comments_buf);
        curl_easy_setopt(c2, CURLOPT_TIMEOUT, 30L);
        configure_curl_ssl(c2, comments_url);
        curl_easy_perform(c2);
        long c2_code = 0;
        curl_easy_getinfo(c2, CURLINFO_RESPONSE_CODE, &c2_code);
        curl_slist_free_all(h2);
        curl_easy_cleanup(c2);

        if (c2_code == 200) {
          try {
            auto comments = json::parse(comments_buf);
            if (comments.is_array() && !comments.empty()) {
              result += "\nComments (" + to_string(comments.size()) + " shown):\n";
              for (size_t i = 0; i < comments.size() && i < 5; i++) {
                string author = comments[i].value("user", json::object()).value("login", "?");
                string body = comments[i].value("body", "");
                result += "\n--- " + author + " ---\n" + body + "\n";
              }
            }
          } catch (...) {}
        }
      }

    } else if (section == "discussions" && segs.size() >= 4) {
      // Discussion (GraphQL response)
      auto data = j.value("data", json::object());
      auto repo_j = data.value("repository", json::object());
      auto disc = repo_j.value("discussion", json::object());

      result = "[GitHub Discussion]\n";
      if (!disc.value("title", "").empty()) result += "Title: " + disc["title"].get<string>() + "\n";
      if (!disc.value("bodyText", "").empty()) result += "\nBody:\n" + disc["bodyText"].get<string>() + "\n";

      auto comments = disc.value("comments", json::object());
      auto nodes = comments.value("nodes", json::array());
      if (!nodes.empty()) {
        result += "\nComments (" + to_string(nodes.size()) + " shown):\n";
        for (auto& node : nodes) {
          string author = node.value("author", json::object()).value("login", "?");
          string body = node.value("bodyText", "");
          result += "\n--- " + author + " ---\n" + body + "\n";
        }
      }

    } else if (section == "blob" && segs.size() >= 5) {
      // File contents (base64 encoded)
      if (j.contains("content") && !j["content"].is_null()) {
        string b64 = j["content"].get<string>();
        // Remove whitespace from base64
        b64.erase(remove_if(b64.begin(), b64.end(), ::isspace), b64.end());
        // Base64 decode
        auto b64_val = [](unsigned char c) -> int {
          if (c >= 'A' && c <= 'Z') return c - 'A';
          if (c >= 'a' && c <= 'z') return c - 'a' + 26;
          if (c >= '0' && c <= '9') return c - '0' + 52;
          if (c == '+') return 62;
          if (c == '/') return 63;
          return -1;
        };
        string decoded;
        int val = 0, valb = -6;
        for (unsigned char c : b64) {
          if (c == '=') break;
          int d = b64_val(c);
          if (d < 0) continue;
          val = (val << 6) + d;
          valb += 6;
          if (valb >= 0) {
            decoded.push_back((val >> valb) & 0xFF);
            valb -= 8;
          }
        }
        result = "[File: " + owner + "/" + repo + "]\n" + decoded;
      } else if (j.contains("message")) {
        cerr << "\033[0mGitHub contents API: " << j["message"].get<string>() << endl;
        return "";
      }
    }

    return result;
  } catch (const exception& e) {
    cerr << "\033[0mGitHub API JSON parse error: " << e.what() << endl;
    return "";
  }
}

// --- HTML Fetcher ---
string NetworkTools::fetch_and_clean_html(const string& url) {
  // Ensure SSL certificates are initialized before any HTTPS fetch
  init_ssl_certificates();

  if (url.length() > 4) {
    string ext = url.substr(url.length() - 4);
    transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
    if (ext == ".zip" || ext == ".exe" || ext == ".tar") {
      return "[Binary file skipped]";
    }
  }

  // --- GitHub URLs: try API first (avoids SPA problem entirely) ---
  if (url.find("github.com/") != string::npos) {
    string api_result = github_api_fetch(url);
    if (!api_result.empty()) {
      cerr << "\033[0mGitHub API fetch succeeded (" << api_result.length() << " chars)" << endl;
      return NetworkTools::limit_context_size(api_result);
    }
    // API failed (rate limit, unsupported pattern, etc.) - fall through to HTML
  }

  FetchState state;
  CURLcode res = CURLE_OK;
  long http_code = 0;
  {
    auto fetched = curl_fetch_buffer(url, state);
    res = fetched.first;
    http_code = fetched.second;
  }

  // Check for interrupt after completion
  if (stop_generation) {
    return "[Fetch interrupted by user]";
  }

  // Check for network errors first
  if (res != CURLE_OK) {
    return "[Failed to fetch page content: " + string(curl_easy_strerror(res)) + "]";
  }

  // Check for HTTP error codes (4xx, 5xx)
  if (http_code >= 400) {
    return "[Failed to fetch page content: HTTP " + to_string(http_code) + "]";
  }

  // Skip non-text, non-PDF content
  if (!state.is_text && !state.is_pdf) {
    return "[Skipped non-text content early to save bandwidth.]";
  }

  // Check if we got any content at all (buffer is empty)
  if (state.buffer.empty()) {
    return "[Failed to fetch page content - empty response]";
  }

  // INTERCEPT PDFS
  if (state.is_pdf) {
    return process_pdf_with_docling(state.buffer);
  }

  string& readBuffer = state.buffer;

  // --- Parse HTML with libxml2 and extract visible text from <body> ---
  string final_text = "";
  string page_title, page_desc;
  bool parse_succeeded = false;

  try {
    xmlDocPtr doc = xmlReadMemory(
      readBuffer.c_str(),
      static_cast<int>(readBuffer.length()),
      nullptr,  // URL (not needed for memory)
      nullptr,  // encoding (auto-detect)
      XML_PARSE_NOERROR | XML_PARSE_NOWARNING | XML_PARSE_NONET | XML_PARSE_RECOVER
      );

    if (doc && doc->children) {
      parse_succeeded = true;

      // Extract metadata (title, description) from <head>
      auto metadata = extract_page_metadata(doc);
      page_title = metadata.first;
      page_desc = metadata.second;

      // Find <body> and extract visible text recursively
      xmlNodePtr body = find_body_element(doc);
      if (body) {
        extract_visible_text(body, final_text);
      } else {
        // No <body> found (fragment or malformed) - extract from root, skip <head>
        for (xmlNodePtr child = doc->children->children; child; child = child->next) {
          if (child->type == XML_ELEMENT_NODE && child->name &&
              string((const char*)child->name) == "head") continue;
          extract_visible_text(child, final_text);
        }
      }

      xmlFreeDoc(doc);
    } else {
      if (doc) xmlFreeDoc(doc);
    }
  } catch (...) {
    cerr << "\033[0mXML parsing exception - using raw buffer fallback" << endl;
  }

  // --- Fallback: if libxml2 failed, use naive tag stripping on raw HTML ---
  if (!parse_succeeded) {
    string decoded_html = readBuffer;

    // Strip <script>...</script> and <style>...</style> blocks
    string lower_buf = decoded_html;
    transform(lower_buf.begin(), lower_buf.end(), lower_buf.begin(), [](unsigned char c){ return std::tolower(c); });

    size_t pos = 0;
    while (true) {
      size_t script_start = lower_buf.find("<script", pos);
      size_t style_start = lower_buf.find("<style", pos);
      size_t next_tag = min(script_start, style_start);
      if (next_tag == string::npos) break;

      string end_tag = (next_tag == script_start) ? "</script>" : "</style>";
      size_t tag_end = lower_buf.find(end_tag, next_tag);

      if (tag_end != string::npos) {
        tag_end += end_tag.length();
        decoded_html.erase(next_tag, tag_end - next_tag);
        lower_buf.erase(next_tag, tag_end - next_tag);
      } else {
        decoded_html.erase(next_tag);
        lower_buf.erase(next_tag);
        break;
      }
    }

    // Strip all remaining tags
    string clean_text = "";
    clean_text.reserve(decoded_html.size());
    bool in_tag = false;
    for (char c : decoded_html) {
      if (c == '<') in_tag = true;
      else if (c == '>') { in_tag = false; clean_text += " "; }
      else if (!in_tag) clean_text += c;
    }

    // Clean whitespace
    final_text.reserve(clean_text.size());
    bool last_was_space = false;
    for (char c : clean_text) {
      if (isspace((unsigned char)c)) {
        if (!last_was_space) { final_text += ' '; last_was_space = true; }
      } else {
        final_text += c; last_was_space = false;
      }
    }
  }

  // Clean whitespace on the extracted text (collapses runs of spaces/newlines)
  {
    string cleaned = "";
    cleaned.reserve(final_text.size());
    bool last_was_space = false;
    for (char c : final_text) {
      if (isspace((unsigned char)c)) {
        if (!last_was_space) { cleaned += ' '; last_was_space = true; }
      } else {
        cleaned += c; last_was_space = false;
      }
    }
    final_text = strip_trailing_whitespace(cleaned);
  }

  // --- SPA / JS-rendered page detection ---
  // If the raw HTML is large but we extracted very little visible text,
  // the page is almost certainly client-side rendered.
  if (final_text.length() < 200 && readBuffer.length() > 50000) {
    cerr << "\033[0mSPA detected: " << final_text.length() << " chars text from "
         << readBuffer.length() << " bytes HTML (" << url << ")" << endl;

    string diag = "[This page appears to be JavaScript-rendered (SPA). The static HTML contains no useful body content.";
    if (!page_title.empty()) diag += "\nPage title: " + page_title;
    if (!page_desc.empty()) diag += "\nDescription: " + page_desc;

    string hint = api_hint_for_url(url);
    if (!hint.empty()) diag += "\n" + hint;
    else diag += "\nTry: use exec_shell with curl to access an API endpoint, or use web_search for alternative sources.";
    diag += "]";
    return diag;
  }

  // --- Prepend page title as context header (if available and content is non-trivial) ---
  if (!page_title.empty() && final_text.length() >= 200) {
    final_text = "[Page: " + page_title + "]\n" + final_text;
  }

  // Ensure we have something to return
  if (final_text.empty()) {
    if (!page_title.empty()) {
      return "[No body content extracted. Page title: " + page_title + "]";
    }
    return "[No content extracted from page]";
  }

  return NetworkTools::limit_context_size(final_text);
}

// --- Fetch Multiple URLs (Files, HTML, PDFs) ---
vector<map<string, string>> NetworkTools::fetch_urls(const vector<string>& urls) {
  vector<map<string, string>> results;

  for (const auto& url : urls) {
    map<string, string> result;
    result["path"] = url;
    result["content"] = "";
    result["error"] = "";

    cerr << "\033[0mfetch_url(" + url + ")" << endl;

    // Check file extension to determine type
    string ext = url;
    size_t last_dot = url.rfind('.');
    if (last_dot != string::npos) {
      ext = url.substr(last_dot);
      transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
    }

    bool is_pdf = (ext == ".pdf");

    if (is_pdf) {
      // Fetch PDF binary and process with Docling
      FetchState state;
      CURLcode res = CURLE_OK;
      long http_code = 0;
      {
        auto fetched = curl_fetch_buffer(url, state);
        res = fetched.first;
        http_code = fetched.second;
      }

      // Check for interrupt after completion
      if (stop_generation) {
        result["error"] = "[PDF fetch interrupted by user]";
      } else if (res == CURLE_FAILED_INIT) {
        result["error"] = "[Curl Init Failed]";
      } else if (res != CURLE_OK || http_code >= 400 || (!state.is_pdf && !is_pdf_by_magic(state.buffer)) || state.exceeded_limit) {
        cerr << "\033[0mPDF fetch failed for: " + url + " - HTTP " << http_code << endl;

        if (state.exceeded_limit) {
          result["error"] = "[Failed to fetch PDF: file too large (exceeds 50MB)]";
        } else if (res != CURLE_OK) {
          result["error"] = "[Failed to fetch PDF: curl error " + to_string(res) + "]";
        } else if (http_code >= 400) {
          result["error"] = "[Failed to fetch PDF: HTTP " + to_string(http_code) + "]";
        } else {
          result["error"] = "[Failed to fetch PDF: content not recognized as PDF]";
        }
      } else {
        string pdf_content = process_pdf_with_docling(state.buffer);
        if (pdf_content.find("[Docling Error") != string::npos ||
            pdf_content.find("[Failed to") != string::npos) {
          result["error"] = pdf_content;
        } else {
          result["content"] = NetworkTools::limit_context_size(pdf_content);
        }
      }
    } else {
      // Fetch HTML/text content
      string content = fetch_and_clean_html(url);

      if (content.find("[Failed to") != string::npos ||
          content.find("[Skipped") != string::npos ||
          content.find("[Binary file skipped]") != string::npos) {
        result["error"] = content;
      } else {
        result["content"] = content;
      }
    }

    results.push_back(result);
  }

  return results;
}

// --- Search result quality gate & formatting (shared by SearXNG and Brave) ---

// Returns true if a fetched page's text is an error/diagnostic message rather
// than real content (fetch failure, SPA detection, empty extraction, etc.).
static bool is_fetch_diagnostic(const string& text) {
  return text.find("[Failed to fetch") != string::npos ||
         text.find("[Skipped") != string::npos ||
         text.find("[Failed to process") != string::npos ||
         text.find("[This page appears to be JavaScript-rendered") != string::npos ||
         text.find("[No content extracted") != string::npos ||
         text.find("[No body content extracted") != string::npos;
}

// Append up to 3 results from a search engine's JSON array to llm_result.
// Each result is expected to have "title" and "url" plus an optional snippet
// field (SearXNG: "content", Brave: "description").  For each result with a
// URL, fetches the full page text and applies the quality gate: substantial
// non-diagnostic text becomes "Page Content:", otherwise the snippet is used,
// or the diagnostic message itself when there is no snippet.
// Returns the number of results processed (capped at 3).
static int format_top_results(const json& results_array, const string& snippet_key, string& llm_result) {
  int count = 0;
  for (const auto& result : results_array) {
    if (count++ >= 3) break;

    string title, result_url, snippet;
    bool has_snippet = false;
    if (result.contains("title") && !result["title"].is_null())
      title = result["title"].get<string>();
    if (result.contains("url") && !result["url"].is_null())
      result_url = result["url"].get<string>();
    if (result.contains(snippet_key) && !result[snippet_key].is_null()) {
      snippet = result[snippet_key].get<string>();
      has_snippet = true;
    }

    if (!title.empty()) llm_result += "Title: " + title + "\n";
    if (!result_url.empty()) llm_result += "URL: " + result_url + "\n";

    if (!result_url.empty()) {
      cerr << "\033[0mFetching text from: " + result_url << endl;
      NetworkTools net;
      string full_text = net.fetch_and_clean_html(result_url);

      // Quality gate: use full text only if it's substantial and not an error/diagnostic.
      bool is_error_or_spa = is_fetch_diagnostic(full_text);

      if (full_text.length() > 200 && !is_error_or_spa) {
        cerr << "\033[0mSuccessfully fetched & parsed text from: " + result_url << endl;
        llm_result += "Page Content: " + full_text + "\n\n";
      } else if (has_snippet) {
        cerr << "\033[0mSkipped full fetch, using snippet for: " + result_url << endl;
        llm_result += "Snippet: " + snippet + "\n\n";
      } else if (is_error_or_spa && !full_text.empty()) {
        // Include the diagnostic (e.g., SPA hint with API suggestion) so the LLM can act on it
        cerr << "\033[0mUsing diagnostic message for: " + result_url << endl;
        llm_result += full_text + "\n\n";
      }
    }
  }
  return count;
}

// --- Brave Search API Fallback ---
// Direct REST call to api.search.brave.com when SearXNG fails or returns empty.
// Returns formatted results string (same shape as SearXNG output), or "" on failure.
static string brave_api_search(const string& query) {
  if (g_brave_api_key.empty()) return "";

  cerr << "\033[0mFalling back to Brave Search API..." << endl;

  NetworkTools::init_ssl_certificates();

  CURL *curl = curl_easy_init();
  if (!curl) return "";

  char *encoded_query = curl_easy_escape(curl, query.c_str(), query.length());
  string url = string("https://api.search.brave.com/res/v1/web/search?q=") + encoded_query + "&count=5";
  curl_free(encoded_query);

  struct curl_slist *headers = NULL;
  headers = curl_slist_append(headers, "Accept: application/json");
  headers = curl_slist_append(headers, "Accept-Encoding: gzip");
  headers = curl_slist_append(headers, ("X-Subscription-Token: " + g_brave_api_key).c_str());

  string readBuffer;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, StringWriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  configure_curl_ssl(curl, url);
  curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, interrupt_check_callback);
  curl_easy_setopt(curl, CURLOPT_XFERINFODATA, NULL);

  CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (stop_generation) return "";
  if (res != CURLE_OK) {
    cerr << "\033[0mBrave API curl error: " << curl_easy_strerror(res) << endl;
    return "";
  }
  if (http_code == 429) {
    cerr << "\033[0mBrave API rate limited (429)" << endl;
    return "";
  }
  if (http_code != 200) {
    cerr << "\033[0mBrave API returned HTTP " << http_code << ": " << readBuffer.substr(0, 200) << endl;
    return "";
  }

  try {
    auto j = json::parse(readBuffer);
    string llm_result = "Search Results for: " + query + " [via Brave API]\n\n";

    int count = 0;
    if (j.contains("web") && j["web"].contains("results") && j["web"]["results"].is_array()) {
      count = format_top_results(j["web"]["results"], "description", llm_result);
    }

    if (count == 0) return "";
    return llm_result;
  } catch (const exception& e) {
    cerr << "\033[0mBrave API JSON parse error: " << e.what() << endl;
    return "";
  }
}

// --- Main Search Interface ---
NetworkTools::NetworkTools(const string& searxng_url) : base_url(searxng_url) {}

string NetworkTools::web_search(const string& query) {
  if (g_searxng_disabled) {
    return "System Error: Web search is currently disabled for this session.";
  }

  // Initialize SSL certificates on first web search
  init_ssl_certificates();

  cerr << "\033[0mweb_search(\"" + query + "\")" << endl;

  string cache = LIM_CONFIG_DIR + "/searchCache";
  mkdir(cache.c_str(), 0777);
  size_t query_hash = hash<string>{}(query);
  string cache_filepath = cache + "/" + to_string(query_hash) + ".txt";

  ifstream cache_file(cache_filepath);
  if (cache_file.is_open()) {
    string cached_content((istreambuf_iterator<char>(cache_file)), istreambuf_iterator<char>());
    cerr << "\033[0mLocal file cache hit. Bypassing network & cooldown." << endl;
    return cached_content;
  }

  // Ensure SearxNG is running. Docling will be started on demand if a PDF is encountered.
  start_searxng_if_needed(base_url);

  auto now = chrono::steady_clock::now();
  auto elapsed = chrono::duration_cast<chrono::seconds>(now - g_last_network_request).count();
  if (elapsed < g_search_cooldown_seconds) {
    int sleep_time = g_search_cooldown_seconds - elapsed;
    cerr << "\033[0mPacing network requests. Sleeping " + to_string(sleep_time) + " seconds..." << endl;
    this_thread::sleep_for(chrono::seconds(sleep_time));
  }

  CURL *curl = curl_easy_init();
  CURLcode res;
  string readBuffer;
  long http_code = 0;

  if(curl) {
    char *encoded_query = curl_easy_escape(curl, query.c_str(), query.length());
    string url = base_url + "/search?q=" + string(encoded_query) + "&format=json";
    curl_free(encoded_query);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "X-Forwarded-For: 127.0.0.1");
    headers = curl_slist_append(headers, "X-Real-IP: 127.0.0.1");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, StringWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, g_web_timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    configure_curl_ssl(curl, url);

    // Enable interrupt checking during transfer
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, interrupt_check_callback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, NULL);

    res = curl_easy_perform(curl);

    // Check for interrupt after completion
    if (stop_generation) {
      g_searxng_disabled = true;
      curl_easy_cleanup(curl);
      return "Error: Search interrupted by user.";
    }
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    g_last_network_request = chrono::steady_clock::now();

    if (res != CURLE_OK) {
      cerr << "\033[0mSearxNG connection failed: " << curl_easy_strerror(res) << endl;
      string brave_result = brave_api_search(query);
      if (!brave_result.empty()) return brave_result;
      g_searxng_disabled = true;
      return "Error: SearxNG connection failed. (" + string(curl_easy_strerror(res)) + ")";
    }
  } else {
    string brave_result = brave_api_search(query);
    if (!brave_result.empty()) return brave_result;
    return "Error: Could not initialize libcurl.";
  }

  if (http_code == 429) {
    cerr << "\033[0mSearxNG rate limited, trying Brave API..." << endl;
    string brave_result = brave_api_search(query);
    if (!brave_result.empty()) return brave_result;
    g_searxng_disabled = true;
    return "Error: SearxNG rate limit exceeded (HTTP 429).";
  } else if (http_code != 200) {
    cerr << "\033[0mSearxNG returned HTTP " << http_code << ", trying Brave API..." << endl;
    string brave_result = brave_api_search(query);
    if (!brave_result.empty()) return brave_result;
    return "Error: SearxNG returned HTTP " + to_string(http_code) + ".\nRaw Response: " + readBuffer;
  }

  if (readBuffer.empty()) {
    string brave_result = brave_api_search(query);
    if (!brave_result.empty()) return brave_result;
    return "Error: Received empty response from search engine.";
  }

  try {
    auto j = json::parse(readBuffer);
    string llm_result = "Search Results for: " + query + "\n\n";

    int count = 0;
    if (j.contains("results") && j["results"].is_array()) {
      count = format_top_results(j["results"], "content", llm_result);
    }

    if (count == 0) {
      // SearXNG returned no results - try Brave API as fallback
      cerr << "\033[0mSearxNG returned no results, trying Brave API..." << endl;
      string brave_result = brave_api_search(query);
      if (!brave_result.empty()) return brave_result;

      g_consecutive_empty_searches++;
      if (g_consecutive_empty_searches >= 3) {
        g_searxng_disabled = true;
        return "System Error: Multiple empty searches. Search disabled to prevent loop.";
      }
      return "No results found for query: " + query;
    }

    g_consecutive_empty_searches = 0;

    ofstream new_cache_file(cache_filepath);
    if (new_cache_file.is_open()) {
      new_cache_file << llm_result;
      new_cache_file.close();
    }

    if(is_debug) {
      ofstream log_file(SEARXNG_LOG_PATH, ios_base::app);
      if (log_file.is_open()) {
        log_file << "\n[LLM INGESTED RESULTS] =====================\n";
        log_file << "QUERY: " << query << "\n";
        log_file << llm_result;
        log_file << "==========================================\n\n";
        log_file.close();
      }
    }

    return llm_result;

  } catch (const exception& e) {
    return "Error: Failed to parse JSON. " + string(e.what()) + "\nRaw: " + readBuffer;
  }
}

void NetworkTools::reset_search() {
  g_searxng_disabled = false;
  g_consecutive_empty_searches = 0;
}
