#include "GoBitBoard.h"

GoBitBoard::GoBitBoard(){

    emptyBoard.set();

    for (size_t i=0; i<PLAN_SIZE; i++){
        for (size_t j=0; j<PLAN_SIZE; j++){
            if ( i == 0 || i == (PLAN_SIZE-1) || j == 0 || j == (PLAN_SIZE-1) ){
                GoPoint targetPoint = GoPointUtil::Pt(i, j);
                emptyBoard.reset(targetPoint);
            }
        }
    }

    for (size_t i=0; i<TOTAL_SIZE; i++){
        allBlocks[i] = NULL;
    }

}

GoBitBoard::~GoBitBoard(){

}


void GoBitBoard::Play(int col, int row, GoColor color){    
    GoPoint targetPoint = GoPointUtil::Pt(col, row);
    Play(targetPoint, color);

}

void GoBitBoard::Play(GoPoint p){

}

void GoBitBoard::Play(GoPoint p, GoColor color){

    emptyBoardLast = emptyBoard;
    blackBoardLast = blackBoard;
    whiteBoardLast = whiteBoard;

    if(color == GO_BLACK){
        blackBoard.set(p);
        emptyBoard.reset(p);

        // UpdateBoard(GO_WHITE);

        UpdateBlock(GO_BLACK);

    }else if(color == GO_WHITE){
        whiteBoard.set(p);
        emptyBoard.reset(p);

        // UpdateBoard(GO_BLACK);

        UpdateBlock(GO_WHITE);
    }

    

    // SgDebug() << *this << "\n";

}

void GoBitBoard::Undo(){
    emptyBoard = emptyBoardLast;
    blackBoard = blackBoardLast;
    whiteBoard = whiteBoardLast;
}

// void GoBitBoard::BoardMoveUp(bitset<TOTAL_SIZE> &inputBoard){
//     BoardMove(inputBoard, GO_UP);
// }

// void GoBitBoard::BoardMoveDown(bitset<TOTAL_SIZE> &inputBoard){
//     BoardMove(inputBoard, GO_DOWN);
// }

// void GoBitBoard::BoardMoveLeft(bitset<TOTAL_SIZE> &inputBoard){
//     BoardMove(inputBoard, GO_LEFT);
// }

// void GoBitBoard::BoardMoveRight(bitset<TOTAL_SIZE> &inputBoard){
//     BoardMove(inputBoard, GO_RIGHT);
// }


void GoBitBoard::UpdateBlock(GoColor updateColor){
    if (updateColor == GO_BLACK){
        tempOriginalBoard = blackBoard;
    } else if (updateColor == GO_WHITE){
        tempOriginalBoard = whiteBoard;
    } else {
        return;
    }


}


void GoBitBoard::UpdateBoard(GoColor updateColor){

    // copy the board value from original board and empty board.
    if (updateColor == GO_BLACK){
        tempOriginalBoard = blackBoard;
        tempTargetBoard = blackBoard;
        
        tempUpBoard = emptyBoard;
        tempDownBoard = emptyBoard;
        tempLeftBoard = emptyBoard;
        tempRightBoard = emptyBoard;
    } else if (updateColor == GO_WHITE){
        tempOriginalBoard = whiteBoard;
        tempTargetBoard = whiteBoard;

        tempUpBoard = emptyBoard;
        tempDownBoard = emptyBoard;
        tempLeftBoard = emptyBoard;
        tempRightBoard = emptyBoard;

    } else {
        return ;
    }

    //really move the empty board.
    BoardMove(tempUpBoard, GO_UP);
    BoardMove(tempDownBoard, GO_DOWN);
    BoardMove(tempLeftBoard, GO_LEFT);
    BoardMove(tempRightBoard, GO_RIGHT);

    //flip the moved empty board.
    tempUpBoard.flip();
    tempDownBoard.flip();
    tempLeftBoard.flip();
    tempRightBoard.flip();

    int targetCount = tempTargetBoard.count();
    int newCount;

    int count = 0;
    while (tempTargetBoard.count() > 0){
        count++;

        tempTargetBoard &= tempUpBoard;
        tempTargetBoard &= tempDownBoard;
        tempTargetBoard &= tempLeftBoard;
        tempTargetBoard &= tempRightBoard;
        
        tempChangeBoard = tempOriginalBoard ^ tempTargetBoard;

        tempChangeBoard.flip();

        tempChangeUpBoard = tempChangeBoard;
        tempChangeDownBoard = tempChangeBoard;
        tempChangeLeftBoard = tempChangeBoard;
        tempChangeRightBoard = tempChangeBoard;

        BoardMove(tempChangeUpBoard, GO_UP);
        BoardMove(tempChangeDownBoard, GO_DOWN);
        BoardMove(tempChangeLeftBoard, GO_LEFT);
        BoardMove(tempChangeRightBoard, GO_RIGHT);

        // tempChangeUpBoard.flip();
        // tempChangeDownBoard.flip();
        // tempChangeLeftBoard.flip();
        // tempChangeRightBoard.flip();
        
        tempUpBoard &= tempChangeUpBoard;
        tempDownBoard &= tempChangeDownBoard;
        tempLeftBoard &= tempChangeLeftBoard;
        tempRightBoard &= tempChangeRightBoard;

        newCount = tempTargetBoard.count();

        if (targetCount == newCount){
            break;
        }else{
            targetCount = newCount;
        }

    }

    SgDebug() << count << ". \n";

    if (targetCount > 0){
        if (updateColor == GO_BLACK){
            emptyBoard |= tempTargetBoard;
            tempTargetBoard.flip();
            blackBoard &= tempTargetBoard;

        } else if (updateColor == GO_WHITE){
            emptyBoard |= tempTargetBoard;
            tempTargetBoard.flip();
            whiteBoard &= tempTargetBoard;
        }
    }
    // BoardDebug(tempTargetBoard);
    


}

void GoBitBoard::BoardDebug(bitset<TOTAL_SIZE>& inputBoard){

    SgDebug() << "Debug Board:  .\n";

    for (int row=(PLAN_SIZE-1); row >= 0; row--){
        for (int col=0; col<PLAN_SIZE; col++){
            
                GoPoint targetPoint = GoPointUtil::Pt(row, col);
                SgDebug() << inputBoard[targetPoint];
            
        }
        SgDebug() << ". \n";
    }
}

void GoBitBoard::Testing(){

    BoardMove(blackBoard, GO_UP);
    BoardMove(blackBoard, GO_LEFT);

    for (int row=(PLAN_SIZE-1); row >= 0; row--){
        for (int col=0; col<PLAN_SIZE; col++){
            
                GoPoint targetPoint = GoPointUtil::Pt(row, col);
                SgDebug() << blackBoard[targetPoint];
            
        }
        SgDebug() << ". \n";
    }

    


}



ostream& operator<<(ostream& out, const GoBitBoard& goBitBoard){

    out << "Whole board:" << endl;

    for (int row=PLAN_SIZE-1; row>=0; row--){
        for (int col=0; col<PLAN_SIZE; col++){
            
            GoPoint targetPoint = GoPointUtil::Pt(row, col);
            if (goBitBoard.emptyBoard[targetPoint]){
                out << ".";
            }else{
                if (goBitBoard.blackBoard[targetPoint]){
                    out << "x";
                } else if (goBitBoard.whiteBoard[targetPoint]){
                    out << "o";
                } else {
                    out << "H";
                }
            }       
        }
        out << endl;
    }


    // out << "Empty board:" << endl;

    // for (int row=PLAN_SIZE-1; row>=0; row--){
    //     for (int col=0; col<PLAN_SIZE; col++){
            
    //             GoPoint targetPoint = GoPointUtil::Pt(row, col);
    //             out << goBitBoard.emptyBoard[targetPoint];
            
        
    //     }
    //     out << endl;
    // }

    // out << "Black board:" << endl;
    
    // for (int row=PLAN_SIZE-1; row>=0; row--){
    //     for (int col=0; col<PLAN_SIZE; col++){
            
    //             GoPoint targetPoint = GoPointUtil::Pt(row, col);
    //             out << goBitBoard.blackBoard[targetPoint];
            
        
    //     }
    //     out << endl;
    // }

    // out << "White board:" << endl;
    
    // for (int row=PLAN_SIZE-1; row>=0; row--){
    //     for (int col=0; col<PLAN_SIZE; col++){
            
    //             GoPoint targetPoint = GoPointUtil::Pt(row, col);
    //             out << goBitBoard.whiteBoard[targetPoint];
            
        
    //     }
    //     out << endl;
    // }

    return out;
}




