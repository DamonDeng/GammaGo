//----------------------------------------------------------------------------
/** @file GoGtpEngine.cpp
    See GoGtpEngine.h */
//----------------------------------------------------------------------------

#include "SgSystem.h"
#include "GoGtpEngine.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <time.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem.hpp>
#include "SgDebug.h"


using boost::filesystem::exists;
using boost::filesystem::path;
using boost::filesystem::remove;
using std::flush;
using std::string;

//----------------------------------------------------------------------------


GoGtpEngine::GoGtpEngine(int fixedBoardSize, const char* programPath,
                         bool noPlayer, bool noHandicap):
                         m_sgCommands(*this, programPath)
{
    Init();

    Register("all_legal", &GoGtpEngine::CmdAllLegal, this);
    Register("boardsize", &GoGtpEngine::CmdBoardSize, this);
    Register("clear_board", &GoGtpEngine::CmdClearBoard, this);
    Register("cgos-gameover", &GoGtpEngine::CmdGameOver, this);
    Register("get_komi", &GoGtpEngine::CmdGetKomi, this);
    Register("gg-undo", &GoGtpEngine::CmdGGUndo, this);
    Register("go_board", &GoGtpEngine::CmdBoard, this);
    Register("go_distance", &GoGtpEngine::CmdDistance, this);
    Register("go_param", &GoGtpEngine::CmdParam, this);
    Register("go_param_rules", &GoGtpEngine::CmdParamRules, this);
    Register("go_player_board", &GoGtpEngine::CmdPlayerBoard, this);
    Register("go_point_info", &GoGtpEngine::CmdPointInfo, this);
    Register("go_point_numbers", &GoGtpEngine::CmdPointNumbers, this);
    Register("go_rules", &GoGtpEngine::CmdRules, this);
    Register("go_sentinel_file", &GoGtpEngine::CmdSentinelFile, this);
    Register("go_set_info", &GoGtpEngine::CmdSetInfo, this);
    Register("gogui-analyze_commands", &GoGtpEngine::CmdAnalyzeCommands,
             this);
    Register("gogui-interrupt", &GoGtpEngine::CmdInterrupt, this);
    Register("gogui-play_sequence", &GoGtpEngine::CmdPlaySequence, this);
    Register("gogui-setup", &GoGtpEngine::CmdSetup, this);
    Register("gogui-setup_player", &GoGtpEngine::CmdSetupPlayer, this);
    Register("is_legal", &GoGtpEngine::CmdIsLegal, this);
    Register("kgs-genmove_cleanup", &GoGtpEngine::CmdGenMoveCleanup, this);
    Register("kgs-time_settings", &GoGtpEngine::CmdKgsTimeSettings, this);
    Register("komi", &GoGtpEngine::CmdKomi, this);
    Register("list_stones", &GoGtpEngine::CmdListStones, this);
    Register("loadsgf", &GoGtpEngine::CmdLoadSgf, this);
    Register("play", &GoGtpEngine::CmdPlay, this);
    Register("savesgf", &GoGtpEngine::CmdSaveSgf, this);
    Register("showboard", &GoGtpEngine::CmdShowBoard, this);
    Register("static_score", &GoGtpEngine::CmdStaticScore, this);
    Register("time_left", &GoGtpEngine::CmdTimeLeft, this);
    Register("time_settings", &GoGtpEngine::CmdTimeSettings, this);
    Register("undo", &GoGtpEngine::CmdUndo, this);
    m_sgCommands.Register(*this);

    // if (! noPlayer)
    // {
    //     Register("all_move_values", &GoGtpEngine::CmdAllMoveValues, this);
    //     Register("final_score", &GoGtpEngine::CmdFinalScore, this);
    //     Register("genmove", &GoGtpEngine::CmdGenMove, this);
    //     Register("go_clock", &GoGtpEngine::CmdClock, this);
    //     Register("go_param_timecontrol", &GoGtpEngine::CmdParamTimecontrol,
    //              this);
    //     Register("reg_genmove", &GoGtpEngine::CmdRegGenMove, this);
    //     Register("reg_genmove_toplay", &GoGtpEngine::CmdRegGenMoveToPlay,
    //              this);
    //     Register("time_lastmove", &GoGtpEngine::CmdTimeLastMove, this);
    //     m_bookCommands.Register(*this);
    // }
    // if (! noHandicap)
    // {
    //     Register("fixed_handicap", &GoGtpEngine::CmdFixedHandicap, this);
    //     Register("place_free_handicap", &GoGtpEngine::CmdPlaceFreeHandicap,
    //              this);
    //     Register("set_free_handicap", &GoGtpEngine::CmdSetFreeHandicap, this);
    // }
}

GoGtpEngine::~GoGtpEngine()
{
    // delete m_player;
}

/** Return all legal move points.
    Compatible with GNU Go's all_legal command.<br>
    Arguments: color */
void GoGtpEngine::CmdAllLegal(GtpCommand& cmd)
{
    cmd << "?";
}

/** Like GNU Go's all_move_values */
void GoGtpEngine::CmdAllMoveValues(GtpCommand& cmd)
{
    cmd << "?";
}

/** Return configuration for GoGui analyze commands.
    See the GoGui documentation http://gogui.sf.net */
void GoGtpEngine::CmdAnalyzeCommands(GtpCommand& cmd)
{
    cmd << "?";
}

/** Print some information about game board.
    See WriteBoardInfo for optional arguments. */
void GoGtpEngine::CmdBoard(GtpCommand& cmd)
{
    cmd << "?";
}

/** Init new game with given board size. */
void GoGtpEngine::CmdBoardSize(GtpCommand& cmd)
{
    cmd << "?";
}

/** Init new game.
    @see SetMaxGames() */
void GoGtpEngine::CmdClearBoard(GtpCommand& cmd)
{
    cmd << "?";
}

/** Show clock info from GoGame::Time() */
void GoGtpEngine::CmdClock(GtpCommand& cmd)
{
    cmd << "?";
}




void GoGtpEngine::CmdDistance(GtpCommand& cmd)
{
    cmd << "?";
}


/** Compute final score.
    Computes score only if GoRules::CaptureDead() == true.
    Otherwise it returns an error. Override this function for players that
    have enough knowledge to do a better scoring. */
void GoGtpEngine::CmdFinalScore(GtpCommand& cmd)
{
    cmd << "?";
}

/** Standard GTP command fixed_handicap. */
void GoGtpEngine::CmdFixedHandicap(GtpCommand& cmd)
{
    cmd << "?";
}

/** Implementation of cgos-gameover as used by the CGOS Python client.
    See http://cgos.sourceforge.net/client-python/doc/index.html
    Stores the game result in the root node of internal SGF tree and sets
    a flag that prevents the engine from pondering. */
void GoGtpEngine::CmdGameOver(GtpCommand& cmd)
{
    cmd << "?";
}

/** Generate and play a move. */
void GoGtpEngine::CmdGenMove(GtpCommand& cmd)
{
    cmd << "?";
}

/** Generate cleanup move.
    As defined in the kgsGtp interface to the KGS Go server.
    Should not return pass, before all enemy dead stones are captured.<br>
    Arguments: color
    @bug This does not work, if the opponent passes, before he captured all
    of our dead stones, because then we could also pass without capturing all
    of his dead stones (if it is a win according to Tromp-Taylor counting),
    but KGS will not use Tromp-Taylor counting in the cleanup phase, but
    send another @c final_status_list dead command. See also
    http://sourceforge.net/apps/trac/fuego/ticket/15 */
void GoGtpEngine::CmdGenMoveCleanup(GtpCommand& cmd)
{
    cmd << "?";
}

/** Get the komi.
    Compatible to GNU Go's get_komi. */
void GoGtpEngine::CmdGetKomi(GtpCommand& cmd)
{
    cmd << "?";
}

/** Undo multiple moves.
    Extension command introduced by GNU Go to undo multiple moves.<br>
    Arguments: optional int<br>
    Fails: if move history is too short<br>
    Returns: nothing */
void GoGtpEngine::CmdGGUndo(GtpCommand& cmd)
{
    cmd << "?";
}

/** Sets time settings on kgs.
    Handles the four different kinds of time control on kgs, "none",
    "absolute", "byoyomi" (which is not currently fully supported),
    and "canadian". */
void GoGtpEngine::CmdKgsTimeSettings(GtpCommand& cmd)
{
    cmd << "?";
}

/** This command indicates that commands can be interrupted using the GoGui
    convention.
    The command does nothing but indicate the ability to handle the
    special comment line <tt># interrupt</tt> used by GoGui.
    It is registered as a handler for @c gogui-interrupt. */
void GoGtpEngine::CmdInterrupt(GtpCommand& cmd)
{
    cmd << "?";
}

/** Check if move is legal.
    Compatible with GNU Go's is_legal.
    Arguments: color move<br>
    Returns: 0/1 */
void GoGtpEngine::CmdIsLegal(GtpCommand& cmd)
{
    cmd << "?";
}

/** Set the komi.
    GTP standard command. */
void GoGtpEngine::CmdKomi(GtpCommand& cmd)
{
   cmd << "?";
}

/** List stones on board.
    Mainly useful for regression tests to verify the board position.
    For compatibility with GNU Go's list_stones command, the points are
    returned in a single line in the same order that is used by GNU Go 3.6
    (A19, B19, ..., A18, B18, ...)

    Arguments: color<br>
    Returns: List of stones<br> */
void GoGtpEngine::CmdListStones(GtpCommand& cmd)
{
   cmd << "?";
}

/** Load a position from a SGF file. */
void GoGtpEngine::CmdLoadSgf(GtpCommand& cmd)
{
    cmd << "?";
}

/** Return name of player, if set, GtpEngine::Name otherwise. */
void GoGtpEngine::CmdName(GtpCommand& cmd)
{
    cmd << "?";
}

/** Get and set GoGtpEngine parameters.
    Parameters:
    @arg @c auto_save See SetAutoSave()
    @arg @c accept_illegal Accept illegal ko or suicide moves in CmdPlay()
    @arg @c debug_to_comment See SetDebugToComment()
    @arg @c overhead See SgTimeRecord::SetOverhead()
    @arg @c statistics_file See SetStatisticsFile()
    @arg @c timelimit See TimeLimit() */
void GoGtpEngine::CmdParam(GtpCommand& cmd)
{
    cmd << "?";
}

/** Get and set detailed rule parameters.
    Changes the rules in the current game as well as the default rule.

    Parameters:
    @arg @c allow_suicde See GoRules:AllowSuicide()
    @arg @c capture_dead See GoRules:CaptureDead()
    @arg @c extra_handicap_komi See GoRules:ExtraHandicapKomi()
    @arg @c japanese_scoring See GoRules:JapaneseScoring()
    @arg @c two_passes_end_game See GoRules:TwoPassesEndGame()
    @arg @c ko_rule (simple, superko, pos_superko) See GoRules:KoRule() */
void GoGtpEngine::CmdParamRules(GtpCommand& cmd)
{
    cmd << "?";
}

/** Get and set time control parameters.
    Fails if the current player is not a SgObjectWithDefaultTimeControl
    or the time control is not a GoTimeControl.

    Parameters:
    @arg @c fast_open_factor See SgDefaultTimeControl::FastOpenFactor()
    @arg @c fast_open_moves See SgDefaultTimeControl::FastOpenMoves()
    @arg @c final_space See GoTimeControl::FinalSpace()
    @arg @c remaining_constant See SgDefaultTimeControl::RemainingConstant() */
void GoGtpEngine::CmdParamTimecontrol(GtpCommand& cmd)
{
    cmd << "?";
}

/** Standard GTP command place_free_handicap.
    The current implementation uses the same locations as for fixed_handicap,
    if defined, and generates additional handicap locations by making the
    player play moves.

    Arguments: number of handicap stones <br>
    Effect: Places handicap stones at chosen locations <br>
    Returns: Handicap stone locations <br> */
void GoGtpEngine::CmdPlaceFreeHandicap(GtpCommand& cmd)
{
    cmd << "?";
}

/** Play a move. */
void GoGtpEngine::CmdPlay(GtpCommand& cmd)
{
    // SgDebug() << "trying to play. \n";

    GoColor color = GoGtpCommandUtil::ColorArg(cmd, 0);
    GoPoint point = GoGtpCommandUtil::PointArg(cmd, 1);

    SgTimer timer;
    double startTime = timer.GetTime();
    
    // for(int i=0; i<100000; i++){
    //     m_goBitBoard.Play(point, color);
    //     m_goBitBoard.Undo();
    // }
        
    m_goBitBoard.Play(point, color);

    double endTime = timer.GetTime();


    // SgDebug() << (100000/400)/(endTime - startTime) << ".\n";

    SgDebug() << m_goBitBoard << ". \n";
    // cmd << "=";
}

/** Play a sequence of moves.
    Extension to standard play command used by GoGui.
    This command is registered with the command name @c gogui-play_sequence
    as used in newer versions of GoGui and for a transition period also
    with @c play_sequence as used by older versions of GoGui */
void GoGtpEngine::CmdPlaySequence(GtpCommand& cmd)
{
    cmd << "?";
}

/** Print some information about point. */
void GoGtpEngine::CmdPointInfo(GtpCommand& cmd)
{
    cmd << "?";
}

/** Print some information about player board.
    See WriteBoardInfo for optional arguments. */
void GoGtpEngine::CmdPlayerBoard(GtpCommand& cmd)
{
    cmd << "?";
}

/** Show point numbers used in GoBoard. */
void GoGtpEngine::CmdPointNumbers(GtpCommand& cmd)
{
    cmd << "?";
}

void GoGtpEngine::CmdQuit(GtpCommand& cmd)
{
   cmd << "?";
}

/** Generate a move, but do not play it.
    Like in GNU Go, if there was a random seed set, it is initialized before
    each reg_genmove to avoid a dependency of the random numbers on previous
    move generations. */
void GoGtpEngine::CmdRegGenMove(GtpCommand& cmd)
{
    cmd << "?";
}

/** Version of CmdRegGenMove() without color argument.
    This is a non-standard version of reg_genmove without color argument.
    It generates a move for the color to play.
 */
void GoGtpEngine::CmdRegGenMoveToPlay(GtpCommand& cmd)
{
    cmd << "?";
}

/** Set named rules.
    @see GoGtpEngine::SetNamedRules() */
void GoGtpEngine::CmdRules(GtpCommand& cmd)
{
    cmd << "?";
}

/** Save current game to file.
    Saves the complete game tree, including any trees from searches
    if storing searches is enabled with global flags.<br>
    Argument: filename */
void GoGtpEngine::CmdSaveSgf(GtpCommand& cmd)
{
    cmd << "?";
}

/** Define a file that makes future clear_board commands fail.
    Defining a sentinel file can be used, for example, to abort playing on
    KGS, because <a href="http://www.gokgs.com/download.xhtml">kgsGtp.jar</a>
    quits, if a clear_board command fails. This command will remove the
    sentinel file, if it currently exists. Future invocations of clear_board
    will fail, if the sentinel file exists at that time. <br>
    Argument: filename */
void GoGtpEngine::CmdSentinelFile(GtpCommand& cmd)
{
    cmd << "?";
}

/** Standard GTP command for explicit placement of handicap stones.
    Arguments: list of points */
void GoGtpEngine::CmdSetFreeHandicap(GtpCommand& cmd)
{
    cmd << "?";
}

/** Set game info property in root node of internal SGF tree.
    Arguments: info value (value is remaining line after gameinfo) <br>
    Supported infos:
    - game_name
    - player_black
    - player_white
    - result */
void GoGtpEngine::CmdSetInfo(GtpCommand& cmd)
{
    cmd << "?";
}

/** Place setup stones.
    GTP extension command used by GoGui.<br>
    Argument: color point [color point ...] <br>
    With color: b, black, w, white */
void GoGtpEngine::CmdSetup(GtpCommand& cmd)
{
    cmd << "?";
}

/** Set color to play.
    GTP extension command used by GoGui.<br>
    Argument: color <br> */
void GoGtpEngine::CmdSetupPlayer(GtpCommand& cmd)
{
    cmd << "?";
}

/** Show current position. */
void GoGtpEngine::CmdShowBoard(GtpCommand& cmd)
{
    cmd << "?";
}

/** Compute static (no search) final score. */
void GoGtpEngine::CmdStaticScore(GtpCommand& cmd)
{
    cmd << "?";
}

/** Time of last genmove command. */
void GoGtpEngine::CmdTimeLastMove(GtpCommand& cmd)
{
   cmd << "?";
}

/** Standard GTP command. */
void GoGtpEngine::CmdTimeLeft(GtpCommand& cmd)
{
    cmd << "?";
}

/** Standard GTP command. */
void GoGtpEngine::CmdTimeSettings(GtpCommand& cmd)
{
    cmd << "?";
}

/** Undo a move. */
void GoGtpEngine::CmdUndo(GtpCommand& cmd)
{
    cmd << "?";
}



GoPoint GoGtpEngine::GenMove(GoColor color)
{
    return 0;
}

void GoGtpEngine::Init()
{
    
}


void GoGtpEngine::Play(GoColor color, GoPoint p)
{
    
}


