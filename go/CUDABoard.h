#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>

#include "GoGlobal.h"

#define GO_EMPTY 0
#define GO_BLACK 1
#define GO_WHITE 2
#define GO_BORDER 3

using namespace std;
 
const int boardSize = 21; 
const int totalSize = boardSize * boardSize;


struct BoardPoint{
  int color;
  int groupID;
  int libertyNumber;
  bool isBlackLegal;
  bool isWhiteLegal;
};

struct DebugFlag{
  int counter;
  int changeFlag;
  int targetGroupID[4];
  int libertyCount;
};

class CUDABoard{
public:

    CUDABoard();
    ~CUDABoard();

    friend ostream& operator<<(ostream& out, const CUDABoard& cudaBoard);
 
    void Play(int row, int col, GoColor color);

    void Play(GoPoint p, GoColor color);

    void Play(GoPoint p);

  void RestoreData();

private:

  BoardPoint boardHost[totalSize];
  BoardPoint *boardDevice;
  DebugFlag debugFlagHost[totalSize];
  DebugFlag *debugFlagDevice;

  const static int valueSizeDevice = totalSize*sizeof(BoardPoint);
  const static int debugFlagSize = totalSize*sizeof(DebugFlag);



};


