#ifndef OMEGAGO_GOBITBOARD
#define OMEGAGO_GOBITBOARD

#include <bitset>
#include <iostream>

#include "GoGlobal.h"
#include "SgDebug.h"
#include "GoBitBlock.h"

using std::bitset;
using std::ostream;
using std::endl;

/**Board description:
 * Using bitset to present a board.
 * Sample content in bitset:  00110101
 * Value in bitset: bitset[0][1][2]...[7] = 1 0 1 0 1 1 0 0 
 * 

  |  420| 421 422 423 424 245 246 427 428 429 430 431 432 433 434 435 436 437 438 439  |440
-----------------------------------------------------------------------------------------
19|  399| 400 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318  |319
18|  378| 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397  |398 
17|  357| 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376  |377
16|  336| 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355  |356
15|  315| 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334  |335 
14|  294| 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313  |314
13|  273| 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292  |293
12|  252| 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271  |272
11|  231| 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250  |251
10|  210| 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229  |230
 9|  189| 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208  |209
 8|  168| 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187  |188
 7|  147| 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166  |167
 6|  126| 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145  |146
 5|  105| 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124  |125 
 4|   84|  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101 102 103  |104 
 3|   63|  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  |83
 2|   42|  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  |62
 1|   21|  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  |41
 -----------------------------------------------------------------------------------------
  |    0|   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  |20
 -----------------------------------------------------------------------------------------
  |     |  [A] [B] [C] [D] [E] [F] [G] [H] [J] [K] [L] [M] [N] [O] [P] [Q] [R] [S] [T] |

          UP
          |
  LEFT <--|--> RIGHT
          |
         DOWN

    row
    |
    |
    ----> col

**/

class GoBitBoard{
public:
    static const int boardSize = 19;
    static const int totalSize = boardSize + 2;

    GoBitBoard();
    ~GoBitBoard();

    friend ostream& operator<<(ostream& out, const GoBitBoard& goBitBoard);
 
    void Play(int row, int col, GoColor color);

    void Play(GoPoint p, GoColor color);

    void Play(GoPoint p);

    void UpdateBlock(GoColor color);

    void Testing();

    void BoardDebug(bitset<TOTAL_SIZE>& inputBoard);

    void Undo();

 
private:

    GoBitBlock *allBlocks[TOTAL_SIZE];

    bitset<TOTAL_SIZE> emptyBoard;
    bitset<TOTAL_SIZE> blackBoard;
    bitset<TOTAL_SIZE> whiteBoard;

    bitset<TOTAL_SIZE> emptyBoardLast;
    bitset<TOTAL_SIZE> blackBoardLast;
    bitset<TOTAL_SIZE> whiteBoardLast;


    bitset<TOTAL_SIZE> tempOriginalBoard;
    bitset<TOTAL_SIZE> tempTargetBoard;

    bitset<TOTAL_SIZE> tempChangeBoard;
    
    bitset<TOTAL_SIZE> tempChangeUpBoard;
    bitset<TOTAL_SIZE> tempChangeDownBoard;
    bitset<TOTAL_SIZE> tempChangeLeftBoard;
    bitset<TOTAL_SIZE> tempChangeRightBoard;
    
    bitset<TOTAL_SIZE> tempUpBoard;
    bitset<TOTAL_SIZE> tempDownBoard;
    bitset<TOTAL_SIZE> tempLeftBoard;
    bitset<TOTAL_SIZE> tempRightBoard;

    // void BoardMoveUp(bitset<TOTAL_SIZE> &inputBoard);
    // void BoardMoveDown(bitset<TOTAL_SIZE> &inputBoard);
    // void BoardMoveLeft(bitset<TOTAL_SIZE> &inputBoard);
    // void BoardMoveRight(bitset<TOTAL_SIZE> &inputBoard);



    inline void BoardMove(bitset<TOTAL_SIZE> &inputBoard, GoDirection direction){

        if (direction == GO_UP){
            inputBoard <<= PLAN_SIZE;
        }else if (direction == GO_DOWN){
            inputBoard >>= PLAN_SIZE;
        }else if(direction == GO_LEFT){
            inputBoard >>= 1;
        }else if(direction == GO_RIGHT){
            inputBoard <<= 1;
        }

    }
    
    void UpdateBoard(GoColor updateColor);

    //void UpdateBlock(GoColor updateColor, GoPoint point);

};

#endif
