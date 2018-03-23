//----------------------------------------------------------------------------
/** @file GoGtpCommandUtil.h
    Utils for Go GTP commands */
//----------------------------------------------------------------------------

#ifndef GO_GTPCOMMANDUTIL_H
#define GO_GTPCOMMANDUTIL_H

#include <cstddef>
#include "GoBoard.h"
#include "GtpEngine.h"
#include "GoGlobal.h"

using std::istringstream;

//----------------------------------------------------------------------------

namespace GoGtpCommandUtil{

    GoColor ColorArg(const GtpCommand& cmd, std::size_t number);

    GoPoint PointArg(const GtpCommand& cmd, std::size_t number);

    
}


#endif // GO_GTPCOMMANDUTIL_H

