#include "network.h"
#include "filesystem.h"  // For log_diagnostic
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

using json = nlohmann::json;
using namespace std;

// --- Global Config & State ---
const string SEARXNG_LOG_PATH = "log/searxng.log";
const string DOCLING_LOG_PATH = "log/docling.log";
string HOME;

extern bool is_debug;

static pid_t g_searxng_pid = -1;
static pid_t g_docling_pid = -1;
static bool g_searxng_disabled = false;
static int g_consecutive_empty_searches = 0;

static chrono::steady_clock::time_point g_last_network_request = chrono::steady_clock::now() - chrono::seconds(3);
const int SEARCH_COOLDOWN_SECONDS = 3;

// --- RAG Fetcher Callbacks & State ---
struct FetchState {
    string buffer;
    bool is_text;
    bool is_pdf;
    FetchState() : is_text(true), is_pdf(false) {}
};

static size_t HeaderCallback(char *buffer, size_t size, size_t nitems, void *userdata) {
    size_t numbytes = size * nitems;
    string header(buffer, numbytes);
    FetchState* state = (FetchState*)userdata;

    string lower_header = header;
    transform(lower_header.begin(), lower_header.end(), lower_header.begin(), [](unsigned char c){ return std::tolower(c); });

    if (lower_header.find("content-type:") == 0) {
        if (lower_header.find("application/pdf") != string::npos) {
            state->is_pdf = true;
            state->is_text = false;
        } else if (lower_header.find("text/html") == string::npos &&
                   lower_header.find("text/plain") == string::npos) {
            state->is_text = false; // Flag as other binary (skip)
        }
    }
    return numbytes;
}

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    FetchState* state = (FetchState*)userp;

    if (!state->is_text && !state->is_pdf) return 0; // Abort bad binaries

    size_t total_size = size * nmemb;
    size_t max_size = state->is_pdf ? 10000000 : 500000; // 10MB limit for PDFs, 500KB for HTML

    if (state->buffer.size() + total_size > max_size) return 0;

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
        kill(g_searxng_pid, SIGTERM);
        waitpid(g_searxng_pid, NULL, WNOHANG);
        g_searxng_pid = -1;
    }
    if (g_docling_pid > 0) {
        kill(g_docling_pid, SIGTERM);
        waitpid(g_docling_pid, NULL, WNOHANG);
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

    log_diagnostic("Spinning up local SearxNG instance on E-cores (16-23)...");

    pid_t pid = fork();
    if (pid == 0) {
        freopen(SEARXNG_LOG_PATH.c_str(), "w", stdout);
        freopen(SEARXNG_LOG_PATH.c_str(), "w", stderr);

        string cmd = "exec taskset -c 16-23 /usr/bin/python " + HOME + "/searxng/searx/webapp.py";
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)NULL);
        exit(1);
    } else if (pid > 0) {
        g_searxng_pid = pid;
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

    log_diagnostic("Spinning up local Docling instance on E-cores (16-23)...");

    pid_t pid = fork();
    if (pid == 0) {
        freopen(DOCLING_LOG_PATH.c_str(), "w", stdout);
        freopen(DOCLING_LOG_PATH.c_str(), "w", stderr);

        // Disable CUDA and pin to E-cores
        string cmd = "UVICORN_LOG_LEVEL=error CUDA_VISIBLE_DEVICES=\"\" exec taskset -c 16-23 "+HOME+"/venv/bin/docling-serve run --enable-ui";
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)NULL);
        exit(1);
    } else if (pid > 0) {
        g_docling_pid = pid;

        // --- Wait for Docling to actually wake up! ---
        log_diagnostic("Waiting for Docling ML models to load into RAM...");
        CURL *wait_curl = curl_easy_init();
        if (wait_curl) {
            curl_easy_setopt(wait_curl, CURLOPT_URL, "http://127.0.0.1:5001/docs");
            curl_easy_setopt(wait_curl, CURLOPT_WRITEFUNCTION, DummyWriteCallback);
            curl_easy_setopt(wait_curl, CURLOPT_TIMEOUT_MS, 1000L);

            int retries = 0;
            while (retries < 60) { // Give it up to 60 seconds to boot
                if (curl_easy_perform(wait_curl) == CURLE_OK) {
                    log_diagnostic("Docling server is online and ready!");
                    break;
                }
                this_thread::sleep_for(chrono::seconds(1));
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

// --- Smart Context Truncation ---
// Only truncates if the text exceeds max_chars (80,000 chars is ~20k tokens)
static string limit_context_size(const string& text, size_t max_chars = 80000) {
    if (text.length() <= max_chars) return text; // Returns untouched if small enough

    size_t head_size = (max_chars * 6) / 10; // First 60% (Abstract, Intro)
    size_t tail_size = (max_chars * 4) / 10; // Last 40% (Conclusion, Summary)

    string truncated = text.substr(0, head_size);
    truncated += "\n\n... [MASSIVE CONTENT OMITTED TO PRESERVE LLM CONTEXT MEMORY] ...\n\n";
    truncated += text.substr(text.length() - tail_size);

    return truncated;
}

string NetworkTools::process_pdf_with_docling(const string& pdf_binary) {
    log_diagnostic("Uploading PDF to Docling (Strict JSON Schema)...");

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
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 600L); // Generous timeout for E-cores

    long http_code = 0;
    CURLcode res = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (http_code == 200) {
        try {
            auto j = json::parse(readBuffer);
            string full_md = j["document"]["md_content"].get<string>();
            return limit_context_size(full_md); // Apply smart truncation
        } catch (...) { return "[Docling JSON Parse Error]"; }
    }

    log_diagnostic("Docling Final Failure: " + to_string(http_code) + " | " + readBuffer);
    return "[Docling Error " + to_string(http_code) + "]";
}

// --- HTML Fetcher ---
string NetworkTools::fetch_and_clean_html(const string& url) {
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
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, HeaderCallback);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &state);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &state);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        // ADDED: Transparent bot User-Agent
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "LocalResearchBot/1.0 (contact@example.com)");
        curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");

        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        if (!state.is_text && !state.is_pdf) {
            return "[Skipped non-text content early to save bandwidth.]";
        }
        if (res != CURLE_OK && res != CURLE_WRITE_ERROR) {
            return "[Failed to fetch page content]";
        }
    }

    // INTERCEPT PDFS
    if (state.is_pdf) {
        return process_pdf_with_docling(state.buffer);
    }

    string& readBuffer = state.buffer;

    // SCRIPT & STYLE STRIPPING
    string lower_buf = readBuffer;
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
            readBuffer.erase(next_tag, tag_end - next_tag);
            lower_buf.erase(next_tag, tag_end - next_tag);
        } else {
            readBuffer.erase(next_tag);
            lower_buf.erase(next_tag);
            break;
        }
    }

    // STRIP TAGS
    string clean_text = "";
    clean_text.reserve(readBuffer.size());
    bool in_tag = false;
    for (char c : readBuffer) {
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

    return limit_context_size(final_text);
}

// --- Main Search Interface ---
NetworkTools::NetworkTools(const string& searxng_url) : base_url(searxng_url) {}

string NetworkTools::web_search(const string& query) {
    if (g_searxng_disabled) {
        return "System Error: Web search is currently disabled for this session.";
    }

    string query_str = "\"" + (query.length() > 80 ? query.substr(0, 77) + "..." : query) + "\"";
    log_diagnostic("web_search(" + query_str + ")");

    string cache = HOME + "/.search_cache";
    mkdir(cache.c_str(), 0777);
    size_t query_hash = hash<string>{}(query);
    string cache_filepath = cache + "/" + to_string(query_hash) + ".txt";

    ifstream cache_file(cache_filepath);
    if (cache_file.is_open()) {
        string cached_content((istreambuf_iterator<char>(cache_file)), istreambuf_iterator<char>());
        log_diagnostic("Local file cache hit. Bypassing network & cooldown.");
        return cached_content;
    }

    // Ensure both services are running
    start_searxng_if_needed(base_url);
    start_docling_if_needed();

    auto now = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::seconds>(now - g_last_network_request).count();
    if (elapsed < SEARCH_COOLDOWN_SECONDS) {
        int sleep_time = SEARCH_COOLDOWN_SECONDS - elapsed;
        log_diagnostic("Pacing network requests. Sleeping " + to_string(sleep_time) + " seconds...");
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
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

        res = curl_easy_perform(curl);
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
                    log_diagnostic("Fetching full text/PDF from: " + result_url);
                    string full_text = fetch_and_clean_html(result_url);

                    // Added check for "[Failed to process" to ensure Docling errors trigger snippet fallback
                    if (full_text.length() > 50 &&
                        full_text.find("[Failed to fetch") == string::npos &&
                        full_text.find("[Skipped") == string::npos &&
                        full_text.find("[Failed to process") == string::npos) {

                        log_diagnostic("Successfully fetched & parsed text from: " + result_url);
                        llm_result += "Page Content: " + full_text + "\n\n";
                    } else if (result.contains("content") && !result["content"].is_null()) {
                        log_diagnostic("Skipped full fetch, using SearXNG snippet for: " + result_url);
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
    log_diagnostic("Web search re-enabled.");
}
