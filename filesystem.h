#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <string>
#include <vector>
#include <map>

class FileSystemTools {
public:
  FileSystemTools();

  std::string exec_shell(const std::string& command);

  std::vector<std::map<std::string, std::string>> read_files(const std::vector<std::string>& paths);

  std::map<std::string, std::string> write_file(const std::string& path, const std::string& content);

  std::map<std::string, std::string> edit_file(const std::string& path, const std::string& old_str, const std::string& new_str);

  std::map<std::string, std::string> chmod_file(const std::string& path, int mode);

  // Updated to return a block of context instead of line numbers
  std::string search_file(const std::string& path, const std::string& text);

private:
  std::string _get_fullpath(const std::string& path);
  static const std::string HOME;
};

#endif // FILESYSTEM_H
