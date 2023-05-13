#include "string_utils.h"

std::string repr(const std::string& text) {
  int singleQuoteCount = 0;
  int doubleQuoteCount = 0;
  for (const auto& c : text) {
    singleQuoteCount += (c == '\'');
    doubleQuoteCount += (c == '"');
  }

  const char quoteChar = (singleQuoteCount < doubleQuoteCount ? '\'' : '"');

  std::string r = "";
  r += quoteChar;
  for (const auto& c : text) {
    if (c == '"' || c == '\\') {
      r += "\\";
    }
    r += c;
  }
  r += quoteChar;
  return r;
}

void ltrim(std::string *text) {
  text->erase(text->begin(), std::find_if(text->begin(), text->end(), [](unsigned char ch) {
      return !std::isspace(ch);
  }));
}

void rtrim(std::string *text) {
  text->erase(std::find_if(text->rbegin(), text->rend(), [](unsigned char ch) {
      return !std::isspace(ch);
  }).base(), text->end());
}

void remove_excess_whitespace(std::string *text) {
  ltrim(text);
  rtrim(text);
  size_t j = 1;
  for (size_t i = 1; i < text->size(); ++i) {
    if (std::isspace((*text)[i - 1]) && std::isspace((*text)[i])) {
      continue;
    }
    (*text)[j++] = (*text)[i];
  }
  text->erase(j);
}