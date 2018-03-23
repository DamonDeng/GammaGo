#ifndef OMEGAGO_GOBITBLOCK
#define OMEGAGO_GOBITBLOCK

#include <iostream>
#include <bitset>

#include "GoGlobal.h"
#include "SgSystem.h"
#include "SgDebug.h"

using std::bitset;

class GoBitBlock{
public:
    GoBitBlock(GoColor inputColor);

    void AddStone(GoPoint point);
    void MergeStones(bitset<TOTAL_SIZE> inputStones);
    void RemoveStone(GoPoint point);

    void AddLiberty(GoPoint point);
    void MergeLiberties(bitset<TOTAL_SIZE> inputLiberties);
    void RemoveLiberty(GoPoint point);

    void MergeBlockFrom(GoBitBlock& sourceBlock);

    int GetStoneNumber();

    int GetLibertyNumber();
    
    ~GoBitBlock();



private:
    GoColor color;

    int stoneNumber;
    int libertyNumber;

    bitset<TOTAL_SIZE> stones;
    bitset<TOTAL_SIZE> liberties;

};
#endif
