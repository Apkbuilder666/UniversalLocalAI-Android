// Copyright 2005-2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the 'License');
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an 'AS IS' BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include <fst/extensions/far/far.h>

#include <cstdint>
#include <ios>
#include <memory>
#include <sstream>
#include <string>

#include <fst/log.h>
#include <string_view>
#include <fst/extensions/far/stlist.h>
#include <fst/extensions/far/sttable.h>
#include <fst/arc.h>
#include <fstream>
#include <fst/fst.h>
#include <fst/util.h>

namespace fst {

bool IsFst(std::string_view source) {
  std::ifstream strm(
      // In portable mode, `std::string_view` is not supported.
      // NOLINTNEXTLINE(google3-readability-redundant-string-conversions)
      std::string(source), std::ios_base::in | std::ios_base::binary);
  if (!strm) return false;
  int32_t magic_number = 0;
  ReadType(strm, &magic_number);
  bool match = magic_number == kFstMagicNumber;
  return match;
}

bool FarHeader::Read(const std::string& source) {
  FstHeader fsthdr;
  arctype_ = "unknown";
  // This function assumes that opening `source` multiple times will
  // produce the same sequence of bytes.  This is definitely not the case
  // for `/dev/stdin`.  Of course, there are many other ways to spell this,
  // and we can't detect them all, but we make a tiny effort to provide a
  // reasonable warning.  Maybe we should be checking for a regular file
  // instead?
  if (source.empty() || source == "/dev/stdin") {
    // Header reading unsupported on stdin. Assumes STList and StdArc.
    LOG(WARNING) << "Cannot reopen stdin; assuming far_type=stlist and "
                    "arc_type=standard; either may be wrong.";
    fartype_ = FarType::STLIST;
    arctype_ = "standard";
    return true;
  } else if (IsSTTable(source)) {  // Checks if STTable.
    fartype_ = FarType::STTABLE;
    if (!ReadSTTableHeader(source, &fsthdr)) return false;
    arctype_ = fsthdr.ArcType().empty() ? ErrorArc::Type() : fsthdr.ArcType();
    return true;
  } else if (IsSTList(source)) {  // Checks if STList.
    fartype_ = FarType::STLIST;
    if (!ReadSTListHeader(source, &fsthdr)) return false;
    arctype_ = fsthdr.ArcType().empty() ? ErrorArc::Type() : fsthdr.ArcType();
    return true;
  } else if (IsFst(source)) {  // Checks if FST.
    fartype_ = FarType::FST;
    std::ifstream istrm(source, std::ios_base::in | std::ios_base::binary);
    if (!fsthdr.Read(istrm, source)) return false;
    arctype_ = fsthdr.ArcType().empty() ? ErrorArc::Type() : fsthdr.ArcType();
    return true;
  }
  return false;
}

}  // namespace fst
