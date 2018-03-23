//----------------------------------------------------------------------------
/** @file GoGtpCommandUtil.cpp */
//----------------------------------------------------------------------------

#include "SgSystem.h"
#include "GoGtpCommandUtil.h"

#include <limits>
#include "GoBoard.h"
#include "GtpEngine.h"
#include "SgDebug.h"

using std::string;
using GoPointUtil::Pt;



GoColor GoGtpCommandUtil::ColorArg(const GtpCommand& cmd,
                                             std::size_t number)
{
    string value = cmd.ArgToLower(number);
    if (value == "b" || value == "black")
        return GO_BLACK;
    if (value == "w" || value == "white")
        return GO_WHITE;
    throw GtpFailure() << "argument " << (number + 1)
                       << " must be black or white";
}



GoPoint GoGtpCommandUtil::PointArg(const GtpCommand& cmd, std::size_t number)
{
    string s = cmd.Arg(number);

    SG_ASSERT(s.length() > 0);

    if (s == "PASS" || s == "pass")
    {
        return GO_PASS;
    }
    
    if (s == "RESIGN" || s == "resign")
    {
        return  GO_RESIGN;
    }

    char c = s[0];
    if (c >= 'A' && c <= 'Z')
        c = char(c - 'A' + 'a');
    else if (c < 'a' || c > 'z'){
        throw GtpFailure() << "invalid point " << s;
    }
    int col = c - 'a' + 1;
    if (c >= 'j')
        --col;

    istringstream sin(s.substr(1));
    int row;
    sin >> row;

    if (! sin || ! GoPointUtil::InRange(col)
        || ! GoPointUtil::InRange(row)){
        throw GtpFailure() << "invalid point " << s;
    }

    GoPoint m_point = GoPointUtil::Pt(row, col);

    return m_point;
    
}




//----------------------------------------------------------------------------

