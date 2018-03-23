#include <iostream>
#include "GoBitBoard.h"

using namespace std;

int main(int argc, char** argv){

    cout << "GoBitBoard testing 0.1" << endl;

    GoBitBoard testingBoard;

    testingBoard.Play(1,1,GO_BLACK);

    testingBoard.Play(18,1,GO_BLACK);
    
    testingBoard.Play(17,3,GO_BLACK);
    
    // testingBoard.Play(19,19,GO_BLACK);
    

    cout << testingBoard << endl;    

    testingBoard.Testing();

}