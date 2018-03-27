#ifndef OMEGAGO_GODATATYPE
#define OMEGAGO_GODATATYPE

const int GO_EMPTY = 0;
const int GO_BLACK = 1;
const int GO_WHITE = 2;
const int GO_BORDER = 3;

const int GO_ILLEGAL_MOVE = -10;

const int BOARD_SIZE = 19;
const int PLAN_SIZE = BOARD_SIZE + 2;
const int TOTAL_SIZE = PLAN_SIZE * PLAN_SIZE;

//  using an point outside the plan to indentify PASS move. 100 is subjectively selected
const int GO_PASS = TOTAL_SIZE + 100; 
const int GO_RESIGN = GO_PASS + 1;

/** West-East  : offset of horizontal neighbors */
const int GO_WE = 1;

/** North-South: offset of vertical neighbors. */
const int GO_NS = PLAN_SIZE;

const int GO_UP = PLAN_SIZE;
const int GO_DOWN = -PLAN_SIZE;
const int GO_LEFT = -1;
const int GO_RIGHT = 1;


typedef int GoColor;

typedef int GoPoint;

typedef int GoDirection;

struct GoPointCombo{
  GoPoint point;
  int col;
  int row;
};


namespace GoUtil{

}


namespace GoPointUtil{

    inline GoPoint Pt(int row, int col){
        return GO_NS * row + col;
    }

    inline bool InRange(int number){
        return number > 0 && number <= BOARD_SIZE;
    }
}

#endif
