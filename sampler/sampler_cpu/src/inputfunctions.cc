/*
 *
 *    Copyright (c) 2014
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */

#include "include/inputfunctions.h"

#include <sstream>
#include <vector>

namespace Smash {

std::vector<Line> line_parser(const std::string &input) {
  std::istringstream input_stream(input);
  std::vector<Line> lines;
  lines.reserve(50);

  std::string line;
  int line_number = 0;
  while (std::getline(input_stream, line)) {
    ++line_number;
    const auto hash_pos = line.find('#');
    if (hash_pos != std::string::npos) {
      // found a comment, remove it from the line and look further
      line = line.substr(0, hash_pos);
    }
    if (line.find_first_not_of(" \t") == std::string::npos) {
      // only whitespace (or nothing) on this line. Next, please.
      continue;
    }
    lines.emplace_back(line_number, std::move(line));
    line = std::string();
  }
  return std::move(lines);
}

}  // namespace Smash
