//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tr_table.cpp
//! \brief Implementation of Table class
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tr_table.hpp"
#include "tr_utils.hpp"

using namespace TableReader; // NOLINT

Table::Table() : data(nullptr), ndim(0), npoints(0), mem_size(0), initialized(false) {
}

Table::~Table() {
  if (initialized) {
    delete[] data;
  }
}

ReadResult Table::ReadTable(const std::string fname) {
  ReadResult result;

  if (initialized) delete[] data;
  data = nullptr;
  ndim = 0;
  npoints = 0;
  mem_size = 0;
  initialized = false;
  metadata.clear();
  point_info.clear();
  field_names.clear();
  scalars.clear();
  fields.clear();

  std::ifstream file;
  try {
    file.open(fname.c_str(), std::ifstream::in);
  } catch (std::ifstream::failure& e) {
    result.error = ReadResult::BAD_FILENAME;
    std::stringstream ss;
    ss << "Could not read '" << fname << "'\n"
       << "open() returned the following error:\n"
       << e.what();
    result.message = ss.str();
    return result;
  }

  // Something bizarre happened while reading the file.
  if (!file.is_open()) {
    result.error = ReadResult::BAD_FILENAME;
    std::stringstream ss;
    ss << "No exception occurred, but ReadTable() failed to open '" << fname << "'\n";
    result.message = ss.str();
    return result;
  }

  // HEADER PARSING

  // Read in the metadata
  std::vector<std::string> block_lines;

  result = ExtractBlock(file, "metadata", block_lines);
  if (result.error != ReadResult::SUCCESS) {
    file.close();
    return result;
  }
  result = ParseBlock("metadata", block_lines,
  [&](const std::string& k, const std::string& v) {
    metadata[k] = v;
  });
  if (result.error != ReadResult::SUCCESS) {
    file.close();
    return result;
  }
  block_lines.clear();

  // Read in the scalars
  result = ExtractBlock(file, "scalars", block_lines);
  if (result.error != ReadResult::SUCCESS) {
    file.close();
    return result;
  }
  result = ParseBlock("scalars", block_lines,
  [&](const std::string& k, const std::string& v) {
    std::size_t used = 0;
    double value = std::stod(v, &used);
    if (used != v.size()) throw std::invalid_argument("scalar is not numeric");
    scalars[k] = value;
  });
  if (result.error != ReadResult::SUCCESS) {
    file.close();
    return result;
  }
  block_lines.clear();

  // Read in the points
  result = ExtractBlock(file, "points", block_lines);
  if (result.error != ReadResult::SUCCESS) {
    file.close();
    return result;
  }
  result = ParseBlock("points", block_lines,
  [&](const std::string& k, const std::string& v) {
    std::size_t used = 0;
    std::int64_t count = std::stoll(v, &used);
    if (used != v.size() || count <= 0) {
      throw std::invalid_argument("point count must be a positive integer");
    }
    for (const auto &point : point_info) {
      if (point.first == k) throw std::invalid_argument("duplicate point name");
    }
    point_info.push_back({k, static_cast<std::size_t>(count)});
  });
  if (result.error != ReadResult::SUCCESS) {
    file.close();
    return result;
  }
  block_lines.clear();
  ndim = point_info.size();

  // Read in the fields
  result = ExtractBlock(file, "fields", block_lines);
  if (result.error != ReadResult::SUCCESS) {
    file.close();
    return result;
  }
  for (auto line : block_lines) {
    TrimWhiteSpace(line);
    if (line.empty()) continue;
    if (std::find(field_names.begin(), field_names.end(), line) != field_names.end()) {
      result.error = ReadResult::BAD_HEADER;
      result.message = "Duplicate field name '" + line + "'.\n";
      file.close();
      return result;
    }
    for (const auto &point : point_info) {
      if (point.first == line) {
        result.error = ReadResult::BAD_HEADER;
        result.message = "Field name '" + line + "' duplicates a point name.\n";
        file.close();
        return result;
      }
    }
    field_names.push_back(line);
  }

  std::streampos header_position = file.tellg();
  if (header_position < 0) {
    result.error = ReadResult::BAD_HEADER;
    result.message = "Could not determine the table header size.\n";
    file.close();
    return result;
  }
  std::size_t header_size = static_cast<std::size_t>(header_position);
  file.close();

  // Compute the payload size with checked arithmetic before allocating memory.
  const std::size_t max_size = std::numeric_limits<std::size_t>::max();
  npoints = 1;
  mem_size = 0;
  for (const auto &p : point_info) {
    if (npoints > max_size/p.second || mem_size > max_size-p.second) {
      result.error = ReadResult::BAD_HEADER;
      result.message = "Table dimensions overflow the addressable payload size.\n";
      return result;
    }
    npoints *= p.second;
    mem_size += p.second;
  }
  if (!field_names.empty() && npoints > max_size/field_names.size()) {
    result.error = ReadResult::BAD_HEADER;
    result.message = "Table field dimensions overflow the addressable payload size.\n";
    return result;
  }
  std::size_t field_values = npoints*field_names.size();
  if (mem_size > max_size-field_values ||
      mem_size+field_values > max_size/sizeof(double)) {
    result.error = ReadResult::BAD_HEADER;
    result.message = "Table payload size overflows the addressable byte count.\n";
    return result;
  }
  mem_size += field_values;
  if (mem_size == 0) {
    result.error = ReadResult::BAD_HEADER;
    result.message = "Table payload contains no axes or fields.\n";
    return result;
  }
  std::size_t payload_bytes = mem_size*sizeof(double);
  if (payload_bytes > static_cast<std::size_t>(
          std::numeric_limits<std::streamsize>::max())) {
    result.error = ReadResult::BAD_HEADER;
    result.message = "Table payload is too large for a binary read.\n";
    return result;
  }

  auto endianness = metadata.find("endianness");
  if (endianness != metadata.end() && endianness->second != "little" &&
      endianness->second != "big") {
    result.error = ReadResult::BAD_HEADER;
    result.message = "Table endianness must be 'little' or 'big'.\n";
    return result;
  }

  // Reject a truncated payload before allocating its storage.
  file.clear();
  file.open(fname.c_str(), std::ifstream::in | std::ifstream::binary |
                           std::ifstream::ate);
  if (!file.is_open()) {
    result.error = ReadResult::BAD_FILENAME;
    result.message = "Could not reopen '" + fname + "' as a binary file.\n";
    return result;
  }
  std::streampos file_position = file.tellg();
  if (file_position < 0 || static_cast<std::size_t>(file_position) < header_size ||
      payload_bytes > static_cast<std::size_t>(file_position)-header_size) {
    result.error = ReadResult::BAD_HEADER;
    result.message = "Binary table payload is truncated.\n";
    file.close();
    return result;
  }

  try {
    data = new double[mem_size];
  } catch (const std::bad_alloc &) {
    result.error = ReadResult::BAD_HEADER;
    result.message = "Could not allocate memory for the table payload.\n";
    file.close();
    return result;
  }

  // Set the memory offsets for all the fields.
  size_t offset = 0;
  for (auto &p : point_info) {
    fields[p.first] = &data[offset];
    offset += p.second;
  }
  for (auto &s : field_names) {
    fields[s] = &data[offset];
    offset += npoints;
  }

  initialized = true;

  // Because we've already read the header, we skip ahead to the binary section.
  file.seekg(static_cast<std::streamoff>(header_size), std::ifstream::beg);
  char *memblock = reinterpret_cast<char*>(data);
  file.read(memblock, static_cast<std::streamsize>(payload_bytes));
  if (file.gcount() != static_cast<std::streamsize>(payload_bytes)) {
    result.error = ReadResult::BAD_HEADER;
    result.message = "Failed to read the complete binary table payload.\n";
    file.close();
    return result;
  }

  // Now we need to check for endianness.
  if (endianness != metadata.end() &&
      ((endianness->second == "little" && !IsLittleEndian()) ||
       (endianness->second == "big" && IsLittleEndian()))) {
    for (size_t i = 0; i < mem_size; i++) {
      data[i] = SwapEndianness(data[i]);
    }
    result.message = "Swapped endianness of data.\n";
  }

  file.close();

  result.error = ReadResult::SUCCESS;

  return result;
}

ReadResult Table::ExtractBlock(std::ifstream& file, const std::string name,
                               std::vector<std::string>& lines) {
  ReadResult result;
  std::stringstream ss;
  ss << "<" << name << "begin" << ">";
  std::string line;
  if (!std::getline(file, line)) {
    result.error = ReadResult::BAD_HEADER;
    result.message = "Unexpected end of file before the '" + name + "' block.\n";
    return result;
  }
  if (!line.empty() && line.back() == '\r') line.pop_back();
  if (line != ss.str()) {
    ss.str("");
    result.error = ReadResult::BAD_HEADER;
    ss << "Header is either missing '" << name << "' or is in the wrong order.\n";
    result.message = ss.str();
    return result;
  }

  ss.str("");
  ss << "<" << name << "end" << ">";
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    // Check if we're at the end of this block.
    if (!line.empty() && line[0] == '<') {
      if (line.compare(ss.str()) != 0) {
        ss.str("");
        result.error = ReadResult::BAD_HEADER;
        ss << "Unexpected new block before reaching end of '" << name << "' in header.\n";
        result.message = ss.str();
        return result;
      } else {
        result.error = ReadResult::SUCCESS;
        return result;
      }
    } else {
      lines.push_back(line);
    }
  }

  ss.str("");
  result.error = ReadResult::BAD_HEADER;
  ss << "Unexpected end of file while reading '" << name << "' in header.\n";
  result.message = ss.str();
  return result;
}

bool Table::SplitToken(const std::string& in, std::string& key, std::string& value) {
  if (in.empty()) return false;
  size_t pos = in.find('=');
  // The equals sign does not exist or is in the wrong location.
  if (pos == std::string::npos) {
    return false;
  } else if (in.back() == '=' || in.front() == '=') {
    return false;
  }

  key = in.substr(0, pos);
  value = in.substr(pos+1, in.size());

  TrimWhiteSpace(key);
  TrimWhiteSpace(value);

  return !key.empty() && !value.empty();
}

void Table::TrimWhiteSpace(std::string& str) {
  const std::string whitespace = " \t\r\n";
  std::size_t first = str.find_first_not_of(whitespace);
  if (first == std::string::npos) {
    str.clear();
    return;
  }
  std::size_t last = str.find_last_not_of(whitespace);
  str = str.substr(first, last-first+1);
}
