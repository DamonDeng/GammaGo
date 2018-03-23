//----------------------------------------------------------------------------
/** @file GoGtpEngine.h
    GtpEngine with GoBoard, GoPlayer and GoGame. */
//----------------------------------------------------------------------------

#ifndef GO_GTPENGINE_H
#define GO_GTPENGINE_H

#include <sstream>
#include <boost/filesystem/path.hpp>
#include <boost/scoped_ptr.hpp>
#include "GoBoard.h"
#include "GtpEngine.h"
#include "GoBitBoard.h"
#include "GoGlobal.h"
#include "SgGtpCommands.h"
#include "GoGtpCommandUtil.h"
#include "SgTimer.h"


//----------------------------------------------------------------------------

/** GtpEngine with GoBoard, GoPlayer and GoGame.
    @see gogtpenginecommands */
class GoGtpEngine
    : public GtpEngine
{
public:
    /** @page gogtpenginecommands GoGtpEngine Commands
        - @link CmdAllLegal() @c all_legal @endlink
        - @link CmdAllMoveValues() @c all_move_values @endlink
        - @link CmdBoardSize() @c boardsize @endlink
        - @link CmdClearBoard() @c clear_board @endlink
        - @link CmdFinalScore() @c final_score @endlink
        - @link CmdFixedHandicap() @c fixed_handicap @endlink
        - @link CmdGenMove() @c genmove @endlink
        - @link CmdGetKomi() @c get_komi @endlink
        - @link CmdGGUndo() @c gg-undo @endlink
        - @link CmdBoard() @c go_board @endlink
        - @link CmdClock() @c go_clock @endlink
        - @link CmdParam() @c go_param @endlink
        - @link CmdParamRules() @c go_param_rules @endlink
        - @link CmdParamTimecontrol() @c go_param_timecontrol @endlink
        - @link CmdPlayerBoard() @c go_player_board @endlink
        - @link CmdPointNumbers() @c go_point_numbers @endlink
        - @link CmdPointInfo() @c go_point_info @endlink
        - @link CmdRules() @c go_rules @endlink
        - @link CmdSentinelFile() @c go_sentinel_file @endlink
        - @link CmdSetInfo() @c go_set_info @endlink
        - @link CmdAnalyzeCommands() @c gogui-analyze_commands @endlink
        - @link CmdInterrupt() @c gogui-interrupt @endlink
        - @link CmdPlaySequence() @c gogui-play_sequence @endlink
        - @link CmdSetup() @c gogui-setup @endlink
        - @link CmdSetupPlayer() @c gogui-setup_player @endlink
        - @link CmdIsLegal() @c is_legal @endlink
        - @link CmdGameOver() @c cgos-gameover @endlink
        - @link CmdGenMoveCleanup() @c kgs-genmove_cleanup @endlink
        - @link CmdKgsTimeSettings() @c kgs-time_settings @endlink
        - @link CmdKomi() @c komi @endlink
        - @link CmdListStones() @c list_stones @endlink
        - @link CmdLoadSgf() @c loadsgf @endlink
        - @link CmdName() @c name @endlink
        - @link CmdPlaceFreeHandicap() @c place_free_handicap @endlink
        - @link CmdPlay() @c play @endlink
        - @link CmdRegGenMove() @c reg_genmove @endlink
        - @link CmdRegGenMoveToPlay() @c reg_genmove_toplay @endlink
        - @link CmdSaveSgf() @c savesgf @endlink
        - @link CmdSetFreeHandicap() @c set_free_handicap @endlink
        - @link CmdShowBoard() @c showboard @endlink
        - @link CmdStaticScore() @c static_score @endlink
        - @link CmdTimeLastMove() @c time_last_move @endlink
        - @link CmdTimeLeft() @c time_left @endlink
        - @link CmdTimeSettings() @c time_settings @endlink
        - @link CmdUndo() @c undo @endlink */
    /** @name Command Callbacks */
    // @{
    // The callback functions are documented in the cpp file
    virtual void CmdAllLegal(GtpCommand&);
    virtual void CmdAllMoveValues(GtpCommand&);
    virtual void CmdAnalyzeCommands(GtpCommand&);
    virtual void CmdBoard(GtpCommand&);
    virtual void CmdBoardSize(GtpCommand&);
    virtual void CmdClearBoard(GtpCommand&);
    virtual void CmdClock(GtpCommand&);
    virtual void CmdDistance(GtpCommand& cmd);
    virtual void CmdFinalScore(GtpCommand&);
    virtual void CmdFixedHandicap(GtpCommand&);
    virtual void CmdGameOver(GtpCommand&);
    virtual void CmdGenMove(GtpCommand&);
    virtual void CmdGenMoveCleanup(GtpCommand&);
    virtual void CmdGetKomi(GtpCommand&);
    virtual void CmdGGUndo(GtpCommand&);
    virtual void CmdInterrupt(GtpCommand&);
    virtual void CmdIsLegal(GtpCommand&);
    virtual void CmdKgsTimeSettings(GtpCommand& cmd);
    virtual void CmdKomi(GtpCommand&);
    virtual void CmdListStones(GtpCommand&);
    virtual void CmdLoadSgf(GtpCommand&);
    virtual void CmdName(GtpCommand&);
    virtual void CmdParam(GtpCommand&);
    virtual void CmdParamRules(GtpCommand&);
    virtual void CmdParamTimecontrol(GtpCommand&);
    virtual void CmdPlaceFreeHandicap(GtpCommand&);
    virtual void CmdPlay(GtpCommand&);
    virtual void CmdPlayerBoard(GtpCommand&);
    virtual void CmdPlaySequence(GtpCommand&);
    virtual void CmdPointNumbers(GtpCommand&);
    virtual void CmdPointInfo(GtpCommand&);
    virtual void CmdQuit(GtpCommand& cmd);
    virtual void CmdRegGenMove(GtpCommand&);
    virtual void CmdRegGenMoveToPlay(GtpCommand&);
    virtual void CmdRules(GtpCommand&);
    virtual void CmdSaveSgf(GtpCommand&);
    virtual void CmdSentinelFile(GtpCommand&);
    virtual void CmdSetFreeHandicap(GtpCommand&);
    virtual void CmdSetInfo(GtpCommand&);
    virtual void CmdSetup(GtpCommand&);
    virtual void CmdSetupPlayer(GtpCommand&);
    virtual void CmdShowBoard(GtpCommand&);
    virtual void CmdStaticScore(GtpCommand&);
    virtual void CmdTimeLastMove(GtpCommand&);
    virtual void CmdTimeLeft(GtpCommand&);
    virtual void CmdTimeSettings(GtpCommand&);
    virtual void CmdUndo(GtpCommand&);
    // @} // @name

    GoGtpEngine(int fixedBoardSize = 0, const char* programPath = 0,
                bool noPlayer = false, bool noHandicap = false);

    ~GoGtpEngine();

    

protected:

    void Init();
    GoPoint GenMove(GoColor color);
    void Play(GoColor color, GoPoint p);


    

private:
    SgGtpCommands m_sgCommands;

    GoBitBoard m_goBitBoard;

};


//----------------------------------------------------------------------------


//----------------------------------------------------------------------------

#endif // GO_GTPENGINE_H
