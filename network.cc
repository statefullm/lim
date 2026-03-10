#include "network.h"
#include <curl/curl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>  // For mkdir
#include <signal.h>
#include <sys/wait.h>
#include <chrono>
#include <thread>
#include <functional>  // For std::hash
#include "json.hpp"

using json = nlohmann::json;
using namespace std;

string HOME;

static pid_t g_searxng_pid = -1;
static bool g_searxng_disabled = false;
static int g_consecutive_empty_searches = 0;

// Simplified network pacing (we only ever hit this if we miss the local file cache)
static chrono::steady_clock::time_point g_last_network_request = chrono::steady_clock::now() - chrono::seconds(3);
const int SEARCH_COOLDOWN_SECONDS = 3;

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

static size_t DummyWriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    return size * nmemb;
}

void NetworkTools::cleanup_searxng() {
    if (g_searxng_pid > 0) {
        kill(g_searxng_pid, SIGTERM);
        waitpid(g_searxng_pid, NULL, WNOHANG);
        g_searxng_pid = -1;
    }
}

void NetworkTools::start_searxng_if_needed(const string& base_url) {
    if (g_searxng_pid != -1) return;

    CURL *curl = curl_easy_init();
    if (curl) {
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "X-Forwarded-For: 127.0.0.1");
        headers = curl_slist_append(headers, "X-Real-IP: 127.0.0.1");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        curl_easy_setopt(curl, CURLOPT_URL, base_url.c_str());
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 500L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, DummyWriteCallback);
        CURLcode res = curl_easy_perform(curl);

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (res == CURLE_OK) {
            g_searxng_pid = -2;
            return;
        }
    }

    printf("\n\033[90m[System: Spinning up local SearxNG instance...]\033[0m\n");
    fflush(stdout);

    pid_t pid = fork();
    if (pid == 0) {
        freopen("searxng.log", "w", stdout);
        freopen("searxng.log", "w", stderr);
        string python="/usr/bin/python";
        string searxng=HOME+"/searxng/searx/webapp.py";

        execl(python.c_str(),
              python.c_str(),
              searxng.c_str(),
              (char*)NULL);

        perror("execl failed");
        exit(1);
    } else if (pid > 0) {
        g_searxng_pid = pid;

        bool server_ready = false;
        for (int i = 0; i < 20; ++i) {
            usleep(500000);
            curl = curl_easy_init();
            if (curl) {
                struct curl_slist *headers = NULL;
                headers = curl_slist_append(headers, "X-Forwarded-For: 127.0.0.1");
                headers = curl_slist_append(headers, "X-Real-IP: 127.0.0.1");
                curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

                curl_easy_setopt(curl, CURLOPT_URL, base_url.c_str());
                curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 500L);
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, DummyWriteCallback);
                CURLcode res = curl_easy_perform(curl);

                curl_slist_free_all(headers);
                curl_easy_cleanup(curl);

                if (res == CURLE_OK) {
                    server_ready = true;
                    usleep(1000000);
                    break;
                }
            }
        }

        if (!server_ready) {
            printf("\n\033[1;31m[System: SearxNG failed to start or bind to port! Check searxng.log]\033[0m\n");
            fflush(stdout);
        }
    }
}

NetworkTools::NetworkTools(const string& searxng_url) : base_url(searxng_url) {}

string NetworkTools::web_search(const string& query) {
    if (g_searxng_disabled) {
        return "System Error: Web search is currently disabled for this session.";
    }

    // --- 1. LOCAL FILE CACHE CHECK ---
    string cache=HOME+"/.search_cache";
    mkdir(cache.c_str(), 0777); // Ensure directory exists
    size_t query_hash = hash<string>{}(query);
    string cache_filepath = cache + "/" + to_string(query_hash) + ".txt";

    ifstream cache_file(cache_filepath);
    if (cache_file.is_open()) {
        string cached_content((istreambuf_iterator<char>(cache_file)), istreambuf_iterator<char>());
        printf("\n\033[36m[System: Local file cache hit. Bypassing network & cooldown.]\033[0m\n");
        fflush(stdout);
        return cached_content;
    }

    // --- 2. LIVE NETWORK FETCH ---
    start_searxng_if_needed(base_url);

    auto now = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::seconds>(now - g_last_network_request).count();
    if (elapsed < SEARCH_COOLDOWN_SECONDS) {
        int sleep_time = SEARCH_COOLDOWN_SECONDS - elapsed;
        printf("\n\033[90m[System: Pacing network requests. Sleeping %d seconds to prevent IP ban...]\033[0m\n", sleep_time);
        fflush(stdout);
        this_thread::sleep_for(chrono::seconds(sleep_time));
    }

    CURL *curl;
    CURLcode res;
    string readBuffer;
    long http_code = 0;

    curl = curl_easy_init();
    if(curl) {
        char *encoded_query = curl_easy_escape(curl, query.c_str(), query.length());
        string url = base_url + "/search?q=" + string(encoded_query) + "&format=json";
        curl_free(encoded_query);

        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "X-Forwarded-For: 127.0.0.1");
        headers = curl_slist_append(headers, "X-Real-IP: 127.0.0.1");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

        res = curl_easy_perform(curl);
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        // Update network pacing timer only on live requests
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
                if (count++ >= 5) break;
                if (result.contains("title") && !result["title"].is_null())
                    llm_result += "Title: " + result["title"].get<string>() + "\n";
                if (result.contains("url") && !result["url"].is_null())
                    llm_result += "URL: " + result["url"].get<string>() + "\n";
                if (result.contains("content") && !result["content"].is_null())
                    llm_result += "Snippet: " + result["content"].get<string>() + "\n\n";
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

        // --- 3. SAVE TO LOCAL CACHE ---
        ofstream new_cache_file(cache_filepath);
        if (new_cache_file.is_open()) {
            new_cache_file << llm_result;
            new_cache_file.close();
        }

        // Append LLM-readable search text into the global log
        ofstream log_file("searxng.log", ios_base::app);
        if (log_file.is_open()) {
            log_file << "\n[LLM INGESTED RESULTS] =====================\n";
            log_file << "QUERY: " << query << "\n";
            log_file << "------------------------------------------\n";
            log_file << llm_result;
            log_file << "==========================================\n\n";
            log_file.close();
        }

        return llm_result;

    } catch (const exception& e) {
        return "Error: Failed to parse JSON. " + string(e.what()) + "\nRaw: " + readBuffer;
    }
}
