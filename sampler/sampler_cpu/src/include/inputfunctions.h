/*
 *
 *    Copyright (c) 2014
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */

#ifndef SRC_INCLUDE_INPUTFUNCTIONS_H_
#define SRC_INCLUDE_INPUTFUNCTIONS_H_

#include <iostream>
#include <string>

#include "forwarddeclarations.h"

namespace Smash {

/// Line consists of a line number and the contents of that line
struct Line {/*{{{*/
  /// initialize line with empty string and number
  Line() = default;
  /// initialize a line with line number \p n and text \p t
  Line(int n, std::string &&t) : number(n), text(std::move(t)) {
  }
  /// line number
  int number;
  /// line content.
  std::string text;
};/*}}}*/

/** builds a meaningful error message
 *
 * Takes the message and quotes the Line where the error occurs
 *
 * \param[in] message Error message
 * \param[in] line Line object containing line number and line content.
 */
inline std::string build_error_string(std::string message, const Line &line) {
  return message + " (on line " + std::to_string(line.number) + ": \"" +
         line.text + "\")";
}

/**
 * Helper function for parsing particles.txt and decaymodes.txt.
 *
 * This function goes through an input stream line by line and removes
 * comments and empty lines. The remaining lines will be returned as a vector
 * of strings and linenumber pairs (Line).
 *
 * \param input an lvalue reference to an input stream
 */
build_vector_<Line> line_parser(const std::string &input);

/**
 * Utility function to read a complete input stream (e.g. file) into one string.
 *
 * \param input The input stream. Since it reads until EOF und thus "uses up the
 * whole input stream" the function takes an rvalue reference to the stream
 * object (just pass a temporary).
 *
 * \note There's no slicing here: the actual istream object is a temporary that
 * is not destroyed until read_all returns.
 */
inline std::string read_all(std::istream &&input) {
  return {std::istreambuf_iterator<char>{input},
          std::istreambuf_iterator<char>{}};
}

}  // namespace Smash

#endif  // SRC_INCLUDE_INPUTFUNCTIONS_H_
