#include "network.h"
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
const string SEARXNG_LOG_PATH = "log/searxng.log";
const string DOCLING_LOG_PATH = "log/docling.log";
string HOME;
static struct HomeInit { HomeInit() { const char* h = getenv("HOME"); HOME = h ? h : ""; } } g_homeInit;

extern bool is_debug;
extern volatile sig_atomic_t stop_generation;  // Forward declaration for interrupt checking

static pid_t g_searxng_pid = -1;
static pid_t g_docling_pid = -1;
static bool g_searxng_disabled = false;
static int g_consecutive_empty_searches = 0;

static chrono::steady_clock::time_point g_last_network_request = chrono::steady_clock::now() - chrono::seconds(3);
const int SEARCH_COOLDOWN_SECONDS = 3;

// Stateful Context Budget for Agentic Sessions
size_t NetworkTools::g_cumulative_context_chars = 0;
const size_t NetworkTools::SESSION_MAX_CHARS = 800000; // Total safe limit for RTX 5090

void NetworkTools::reset_context_usage() {
    g_cumulative_context_chars = 0;
}

size_t NetworkTools::get_context_usage() {
    return g_cumulative_context_chars;
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
                int status_code = atoi(status_code_str.c_str());
                if (status_code >= 300 && status_code < 400) {
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
    size_t max_size = state->is_pdf ? 50000000 : 500000; // 50MB limit for PDFs, 500KB for HTML

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

void NetworkTools::start_searxng_if_needed(const string& base_url) {
    if (g_searxng_pid != -1) return;

    // Check if it's already running externally
    CURL *curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, base_url.c_str());
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 500L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, DummyWriteCallback);
        if (curl_easy_perform(curl) == CURLE_OK) {
            g_searxng_pid = -2;
            curl_easy_cleanup(curl);
            return;
        }
        curl_easy_cleanup(curl);
    }

    std::cerr << "Spinning up local SearxNG instance..." << std::endl;

    pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0);
        freopen(SEARXNG_LOG_PATH.c_str(), "w", stdout);
        freopen(SEARXNG_LOG_PATH.c_str(), "w", stderr);

        string cmd = "exec taskset -c 16-23 /usr/bin/python " + HOME + "/searxng/searx/webapp.py";
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)NULL);
        exit(1);
    } else if (pid > 0) {
        g_searxng_pid = pid;

        // --- Wait for SearxNG to actually wake up! ---
        std::cerr << "Waiting for SearxNG to become ready..." << std::endl;
        CURL *wait_curl = curl_easy_init();
        if (wait_curl) {
            curl_easy_setopt(wait_curl, CURLOPT_URL, base_url.c_str());
            curl_easy_setopt(wait_curl, CURLOPT_WRITEFUNCTION, DummyWriteCallback);
            curl_easy_setopt(wait_curl, CURLOPT_TIMEOUT_MS, 500L);

            int retries = 0;
            while (retries < 40) {
                if (stop_generation) {
                    std::cerr << "SearxNG startup interrupted by user" << std::endl;
                    break;
                }
                if (curl_easy_perform(wait_curl) == CURLE_OK) {
                    std::cerr << "SearxNG is ready and responding!" << std::endl;
                    break;
                }
                this_thread::sleep_for(chrono::milliseconds(500));
                retries++;
            }
            curl_easy_cleanup(wait_curl);
        }
    }
}

void NetworkTools::start_docling_if_needed() {
    if (g_docling_pid != -1) return;

    // Check if Docling is already running on port 5001
    CURL *curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:5001/docs");
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 500L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, DummyWriteCallback);
        if (curl_easy_perform(curl) == CURLE_OK) {
            g_docling_pid = -2;
            curl_easy_cleanup(curl);
            return;
        }
        curl_easy_cleanup(curl);
    }

    std::cerr << "Spinning up local Docling instance..." << std::endl;

    pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0);
        freopen(DOCLING_LOG_PATH.c_str(), "w", stdout);
        freopen(DOCLING_LOG_PATH.c_str(), "w", stderr);

        string cmd = "UVICORN_LOG_LEVEL=error CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=8 exec taskset -c 0-15 "+HOME+"/venv/bin/docling-serve run --enable-ui";
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)NULL);
        exit(1);
    } else if (pid > 0) {
        g_docling_pid = pid;

        // --- Wait for Docling to actually wake up! ---
        std::cerr << "Waiting for Docling ML models to load into RAM..." << std::endl;
        CURL *wait_curl = curl_easy_init();
        if (wait_curl) {
            curl_easy_setopt(wait_curl, CURLOPT_URL, "http://127.0.0.1:5001/docs");
            curl_easy_setopt(wait_curl, CURLOPT_WRITEFUNCTION, DummyWriteCallback);
            curl_easy_setopt(wait_curl, CURLOPT_TIMEOUT_MS,  500L);

            int retries = 0;
            while (retries < 40) {
                if (stop_generation) {
                    std::cerr << "Docling startup interrupted by user" << std::endl;
                    break;
                }
                if (curl_easy_perform(wait_curl) == CURLE_OK) {
                    break;
                }
                this_thread::sleep_for(chrono::milliseconds(500));
                retries++;
            }
            curl_easy_cleanup(wait_curl);
        }
    }
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

    // Use $HOME/.cache for certificate storage
    string cache_dir = HOME + "/.cache";
    mkdir(cache_dir.c_str(), 0755);
    string cached_ca = cache_dir + "/combined-ca.crt";

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
        std::cerr << "Updating SSL certificate cache..." << std::endl;

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

        // Download Cloudflare certificate chain to cache directory
        string cloudflare_cache = cache_dir + "/cloudflare-chain.pem";
        string cloudflare_cmd = "openssl s_client -connect example.com:443 -showcerts 2>/dev/null | "
                               "awk '/BEGIN CERTIFICATE/,/END CERTIFICATE/{if(/BEGIN CERTIFICATE/)p=1; if(p)print; if(/END CERTIFICATE/)p=0}' > " + cloudflare_cache;
        system(cloudflare_cmd.c_str());

        // Combine CA bundles into cache directory
        string combine_cmd;
        if (!system_ca.empty()) {
            combine_cmd = "cat " + system_ca + " " + cloudflare_cache + " > " + cached_ca;
        } else {
            // Download standard CA bundle and add Cloudflare certs to cache
            string temp_bundle = cache_dir + "/ca-bundle-temp.crt";
            combine_cmd = "curl -s https://curl.se/ca/cacert.pem > " + temp_bundle + " && "
                         "cat " + temp_bundle + " " + cloudflare_cache + " > " + cached_ca;
        }

        int result = system(combine_cmd.c_str());
        if (result == 0 && stat(cached_ca.c_str(), &st) == 0 && st.st_size > 0) {
            std::cerr << "Created combined CA bundle in cache: " + cached_ca << std::endl;
            setenv("CURL_CA_BUNDLE", cached_ca.c_str(), 1);
        } else {
            std::cerr << "Failed to create combined CA bundle - using curl defaults" << std::endl;
        }
    } else {
        cout << "Using cached SSL certificate from: " << cached_ca << endl;
        cout.flush();
        setenv("CURL_CA_BUNDLE", cached_ca.c_str(), 1);
    }

    initialized = true;
}

// --- Smart Context Truncation (Stateful) ---
string NetworkTools::limit_context_size(const string& text, size_t per_file_max) {
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

        // Use the cached CA bundle from $HOME/.cache
        string cached_ca = HOME + "/.cache/combined-ca.crt";
        struct stat st;
        if (stat(cached_ca.c_str(), &st) == 0 && st.st_size > 0) {
            curl_easy_setopt(curl, CURLOPT_CAINFO, cached_ca.c_str());
            cerr << "Using cached CA bundle: " + cached_ca << endl;
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
                    cerr << "Using system CA bundle: " + string(path) << endl;
                    found = true;
                    break;
                }
            }
            if (!found) {
                static const char* default_ca = "";
                curl_easy_setopt(curl, CURLOPT_CAINFO, default_ca);
                cerr << "Using curl's built-in system certificate store" << endl;
            }
        }
    } else {
        // HTTP requests (localhost, http:// URLs) - disable SSL verification options
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
        cerr << "HTTP request detected - SSL verification disabled" << endl;
    }
}

// --- Helper to configure common curl options for fetch operations ---
// Returns the header list that caller must free with curl_slist_free_all()
static struct curl_slist* configure_curl_fetch(CURL* curl, const string& url) {
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, HeaderCallback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, nullptr);  // Will be set by caller if needed
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, nullptr);    // Will be set by caller if needed
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 600L);
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

std::string NetworkTools::process_local_pdf(const std::string& pdf_binary) {
    // Ensure Docling is running
    start_docling_if_needed();

    // Create instance to call member function
    NetworkTools net;
    return net.process_pdf_with_docling(pdf_binary);
}

string NetworkTools::process_pdf_with_docling(const string& pdf_binary) {
    start_docling_if_needed();

    cerr << "Uploading PDF to Docling (Strict JSON Schema)..." << endl;

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
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 600L);
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

    cerr << "Docling Final Failure: " + to_string(http_code) + " | " + readBuffer << endl;
    return "[Docling Error " + to_string(http_code) + "]";
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

    CURL *curl = curl_easy_init();
    FetchState state;

    if (curl) {
        struct curl_slist* headers = configure_curl_fetch(curl, url);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &state);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &state);

        CURLcode res = curl_easy_perform(curl);

        // Check for interrupt after completion
        if (stop_generation) {
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            return "[Fetch interrupted by user]";
        }

        // Check HTTP status code BEFORE cleanup
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

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
    }

    // INTERCEPT PDFS
    if (state.is_pdf) {
        return process_pdf_with_docling(state.buffer);
    }

    string& readBuffer = state.buffer;

    // DECODE HTML ENTITIES using libxml2 FIRST (before stripping tags)
    string decoded_html = "";

    try {
        xmlDocPtr doc = xmlReadMemory(
            readBuffer.c_str(),
            static_cast<int>(readBuffer.length()),
            nullptr,  // URL (not needed for memory)
            nullptr,  // encoding (auto-detect)
            XML_PARSE_NOERROR | XML_PARSE_NOWARNING | XML_PARSE_NONET | XML_PARSE_RECOVER
        );

        if (doc && doc->children) {
            xmlChar *content = xmlNodeGetContent(doc->children);
            if (content) {
                decoded_html = reinterpret_cast<char*>(content);
                xmlFree(content);
            }
            xmlFreeDoc(doc);
        } else {
            // If parsing failed or no children, fall back to raw buffer
            decoded_html = readBuffer;
            cerr << "XML parsing fallback - using raw buffer" << endl;
        }
    } catch (...) {
        // On any exception, use the raw buffer as fallback
        decoded_html = readBuffer;
        cerr << "XML parsing exception - using raw buffer" << endl;
    }

    // Ensure we have content to process
    if (decoded_html.empty()) {
        return "[No content extracted from page]";
    }

    // SCRIPT & STYLE STRIPPING (on decoded content)
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

    // STRIP TAGS (from decoded content)
    string clean_text = "";
    clean_text.reserve(decoded_html.size());
    bool in_tag = false;
    for (char c : decoded_html) {
        if (c == '<') in_tag = true;
        else if (c == '>') { in_tag = false; clean_text += " "; }
        else if (!in_tag) clean_text += c;
    }

    // CLEAN WHITESPACE
    string final_text = "";
    final_text.reserve(clean_text.size());
    bool last_was_space = false;
    for (char c : clean_text) {
        if (isspace((unsigned char)c)) {
            if (!last_was_space) { final_text += ' '; last_was_space = true; }
        } else {
            final_text += c; last_was_space = false;
        }
    }

    // Strip trailing whitespace before returning
    final_text = strip_trailing_whitespace(final_text);

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

    cerr << "fetch_url(" + url + ")" << endl;

    // Check file extension to determine type
    string ext = url;
    size_t last_dot = url.rfind('.');
    if (last_dot != string::npos) {
      ext = url.substr(last_dot);
      transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
    }

    bool is_pdf = (ext == ".pdf" || ext == ".PDF");

    if (is_pdf) {
      // Fetch PDF binary and process with Docling
      CURL *curl = curl_easy_init();
      FetchState state;

      if (curl) {
        struct curl_slist* headers = configure_curl_fetch(curl, url);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &state);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &state);

        CURLcode res = curl_easy_perform(curl);

        // Check HTTP response code
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        // Check for interrupt after completion
        if (stop_generation) {
          result["error"] = "[PDF fetch interrupted by user]";
          curl_easy_cleanup(curl);
        } else if (res != CURLE_OK || http_code >= 400 || (!state.is_pdf && !is_pdf_by_magic(state.buffer)) || state.exceeded_limit) {
          cerr << "PDF fetch failed for: " + url + " - HTTP " << http_code << endl;

          if (state.exceeded_limit) {
            result["error"] = "[Failed to fetch PDF: file too large (exceeds 50MB)]";
          } else if (res != CURLE_OK) {
            result["error"] = "[Failed to fetch PDF: curl error " + to_string(res) + "]";
          } else if (http_code >= 400) {
            result["error"] = "[Failed to fetch PDF: HTTP " + to_string(http_code) + "]";
          } else {
            result["error"] = "[Failed to fetch PDF: content not recognized as PDF]";
          }
          curl_easy_cleanup(curl);
        } else {
          string pdf_content = process_pdf_with_docling(state.buffer);
          if (pdf_content.find("[Docling Error") != string::npos ||
              pdf_content.find("[Failed to") != string::npos) {
            result["error"] = pdf_content;
          } else {
            result["content"] = NetworkTools::limit_context_size(pdf_content);
          }
        }

        curl_slist_free_all(headers);  // Clean up header list (local to this block)
        curl_easy_cleanup(curl);
      } else {
        result["error"] = "[Curl Init Failed]";
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

// --- Main Search Interface ---
NetworkTools::NetworkTools(const string& searxng_url) : base_url(searxng_url) {}

string NetworkTools::web_search(const string& query) {
    if (g_searxng_disabled) {
        return "System Error: Web search is currently disabled for this session.";
    }

    // Initialize SSL certificates on first web search
    init_ssl_certificates();

    string query_str = "\"" + (query.length() > 80 ? query.substr(0, 77) + "..." : query) + "\"";
    cerr << "web_search(" + query_str + ")" << endl;

    string cache = HOME + "/.search_cache";
    mkdir(cache.c_str(), 0777);
    size_t query_hash = hash<string>{}(query);
    string cache_filepath = cache + "/" + to_string(query_hash) + ".txt";

    ifstream cache_file(cache_filepath);
    if (cache_file.is_open()) {
        string cached_content((istreambuf_iterator<char>(cache_file)), istreambuf_iterator<char>());
        cerr << "Local file cache hit. Bypassing network & cooldown." << endl;
        return cached_content;
    }

    // Ensure both services are running
    start_searxng_if_needed(base_url);
    start_docling_if_needed();

    auto now = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::seconds>(now - g_last_network_request).count();
    if (elapsed < SEARCH_COOLDOWN_SECONDS) {
        int sleep_time = SEARCH_COOLDOWN_SECONDS - elapsed;
        cerr << "Pacing network requests. Sleeping " + to_string(sleep_time) + " seconds..." << endl;
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
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 600L);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

        // SSL Certificate Support: Configure based on URL protocol
        bool is_https = url.substr(0, 8) == "https://";
        if (is_https) {
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

            // Use the cached CA bundle from $HOME/.cache
            string cached_ca = HOME + "/.cache/combined-ca.crt";
            struct stat st;
            if (stat(cached_ca.c_str(), &st) == 0 && st.st_size > 0) {
                curl_easy_setopt(curl, CURLOPT_CAINFO, cached_ca.c_str());
                cerr << "Using cached CA bundle: " + cached_ca << endl;
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
                        cerr << "Using system CA bundle: " + string(path) << endl;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    static const char* default_ca = "";
                    curl_easy_setopt(curl, CURLOPT_CAINFO, default_ca);
                    cerr << "Using curl's built-in system certificate store" << endl;
                }
            }
        } else {
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
            cerr << "HTTP request detected - SSL verification disabled" << endl;
        }
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
            g_searxng_disabled = true;
            return "Error: SearxNG connection failed. (" + string(curl_easy_strerror(res)) + ")";
        }
    } else {
        return "Error: Could not initialize libcurl.";
    }

    if (http_code == 429) {
        g_searxng_disabled = true;
        return "Error: SearxNG rate limit exceeded (HTTP 429).";
    } else if (http_code != 200) {
        return "Error: SearxNG returned HTTP " + to_string(http_code) + ".\nRaw Response: " + readBuffer;
    }

    if (readBuffer.empty()) return "Error: Received empty response from search engine.";

    try {
        auto j = json::parse(readBuffer);
        string llm_result = "Search Results for: " + query + "\n\n";

        int count = 0;
        if (j.contains("results") && j["results"].is_array()) {
            for (const auto& result : j["results"]) {
                if (count++ >= 3) break;

                string result_url = "";
                if (result.contains("title") && !result["title"].is_null())
                    llm_result += "Title: " + result["title"].get<string>() + "\n";
                if (result.contains("url") && !result["url"].is_null()) {
                    result_url = result["url"].get<string>();
                    llm_result += "URL: " + result_url + "\n";
                }

                if (!result_url.empty()) {
                    cerr << "Fetching full text/PDF from: " + result_url << endl;
                    string full_text = fetch_and_clean_html(result_url);

                    // Added check for "[Failed to process" to ensure Docling errors trigger snippet fallback
                    if (full_text.length() > 50 &&
                        full_text.find("[Failed to fetch") == string::npos &&
                        full_text.find("[Skipped") == string::npos &&
                        full_text.find("[Failed to process") == string::npos) {

                        cerr << "Successfully fetched & parsed text from: " + result_url << endl;
                        llm_result += "Page Content: " + full_text + "\n\n";
                    } else if (result.contains("content") && !result["content"].is_null()) {
                        cerr << "Skipped full fetch, using SearXNG snippet for: " + result_url << endl;
                        llm_result += "Snippet: " + result["content"].get<string>() + "\n\n";
                    }
                }
            }
        }

        if (count == 0) {
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
    if (g_searxng_disabled)
      cerr << "Web search re-enabled." << endl;
}
