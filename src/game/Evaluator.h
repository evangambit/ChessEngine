#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <cstdint>

#include <algorithm>
#include <fstream>
#include <string>

#include "geometry.h"
#include "utils.h"
#include "Threats.h"
#include "Position.h"
#include "movegen.h"
#include "piece_maps.h"

namespace ChessEngine {

float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

enum EF {
  OUR_PAWNS,
  OUR_KNIGHTS,
  OUR_BISHOPS,
  OUR_ROOKS,
  OUR_QUEENS,
  THEIR_PAWNS,

  THEIR_KNIGHTS,
  THEIR_BISHOPS,
  THEIR_ROOKS,
  THEIR_QUEENS,
  IN_CHECK,
  KING_ON_BACK_RANK,

  KING_ON_CENTER_FILE,
  KING_ACTIVE,
  THREATS_NEAR_KING_2,
  THREATS_NEAR_KING_3,
  OUR_PASSED_PAWNS,
  THEIR_PASSED_PAWNS,

  ISOLATED_PAWNS,
  DOUBLED_PAWNS,
  DOUBLE_ISOLATED_PAWNS,
  PAWNS_CENTER_16,
  PAWNS_CENTER_4,
  ADVANCED_PASSED_PAWNS_2,

  ADVANCED_PASSED_PAWNS_3,
  ADVANCED_PASSED_PAWNS_4,
  PAWN_MINOR_CAPTURES,
  PAWN_MAJOR_CAPTURES,
  PROTECTED_PAWNS,
  PROTECTED_PASSED_PAWNS,

  BISHOPS_DEVELOPED,
  BISHOP_PAIR,
  BLOCKADED_BISHOPS,
  SCARY_BISHOPS,
  SCARIER_BISHOPS,
  BLOCKADED_ROOKS,

  SCARY_ROOKS,
  INFILTRATING_ROOKS,
  KNIGHTS_DEVELOPED,
  KNIGHT_MAJOR_CAPTURES,
  KNIGHTS_CENTER_16,
  KNIGHTS_CENTER_4,

  KNIGHT_ON_ENEMY_SIDE,
  OUR_HANGING_PAWNS,
  OUR_HANGING_KNIGHTS,
  OUR_HANGING_BISHOPS,
  OUR_HANGING_ROOKS,
  OUR_HANGING_QUEENS,

  THEIR_HANGING_PAWNS,
  THEIR_HANGING_KNIGHTS,
  THEIR_HANGING_BISHOPS,
  THEIR_HANGING_ROOKS,
  THEIR_HANGING_QUEENS,
  LONELY_KING_IN_CENTER,

  LONELY_KING_AWAY_FROM_ENEMY_KING,
  NUM_TARGET_SQUARES,
  TIME,
  KPVK_OPPOSITION,
  KPVK_IN_FRONT_OF_PAWN,
  KPVK_OFFENSIVE_KEY_SQUARES,

  KPVK_DEFENSIVE_KEY_SQUARES,
  SQUARE_RULE,
  ADVANCED_PAWNS_1,
  ADVANCED_PAWNS_2,
  OPEN_ROOKS,
  ROOKS_ON_THEIR_SIDE,

  KING_CASTLED,
  CASTLING_RIGHTS,
  KING_IN_FRONT_OF_PASSED_PAWN,
  KING_IN_FRONT_OF_PASSED_PAWN2,
  PAWN_V_LONELY_KING,
  KNIGHTS_V_LONELY_KING,

  BISHOPS_V_LONELY_KING,
  ROOK_V_LONELY_KING,
  QUEEN_V_LONELY_KING,
  OUR_MATERIAL_THREATS,
  THEIR_MATERIAL_THREATS,
  LONELY_KING_ON_EDGE,

  OUTPOSTED_KNIGHTS,
  OUTPOSTED_BISHOPS,
  PAWN_MOVES,
  KNIGHT_MOVES,
  BISHOP_MOVES,
  ROOK_MOVES,

  QUEEN_MOVES,
  PAWN_MOVES_ON_THEIR_SIDE,
  KNIGHT_MOVES_ON_THEIR_SIDE,
  BISHOP_MOVES_ON_THEIR_SIDE,
  ROOK_MOVES_ON_THEIR_SIDE,
  QUEEN_MOVES_ON_THEIR_SIDE,

  KING_HOME_QUALITY,
  BISHOPS_BLOCKING_KNIGHTS,
  OUR_HANGING_PAWNS_2,
  OUR_HANGING_KNIGHTS_2,
  OUR_HANGING_BISHOPS_2,
  OUR_HANGING_ROOKS_2,

  OUR_HANGING_QUEENS_2,
  THEIR_HANGING_PAWNS_2,
  THEIR_HANGING_KNIGHTS_2,
  THEIR_HANGING_BISHOPS_2,
  THEIR_HANGING_ROOKS_2,
  THEIR_HANGING_QUEENS_2,

  QUEEN_THREATS_NEAR_KING,
  MISSING_FIANCHETTO_BISHOP,
  BISHOP_PAWN_DISAGREEMENT,
  CLOSED_1,
  CLOSED_2,
  CLOSED_3,

  NUM_BAD_SQUARES_FOR_PAWNS,
  NUM_BAD_SQUARES_FOR_MINORS,
  NUM_BAD_SQUARES_FOR_ROOKS,
  NUM_BAD_SQUARES_FOR_QUEENS,

  NUM_EVAL_FEATURES,
};

constexpr Bitboard kWhiteKingCorner = bb(Square::H1) | bb(Square::H2) | bb(Square::G1) | bb(Square::G2) | bb(Square::F1);
constexpr Bitboard kWhiteQueenCorner = bb(Square::A1) | bb(Square::A2) | bb(Square::B1) | bb(Square::B2) | bb(Square::C1);
constexpr Bitboard kBlackKingCorner = bb(Square::H8) | bb(Square::H7) | bb(Square::G8) | bb(Square::G7) | bb(Square::F8);
constexpr Bitboard kBlackQueenCorner = bb(Square::A8) | bb(Square::A7) | bb(Square::B8) | bb(Square::B7) | bb(Square::C8);

std::string EFSTR[] = {
  "OUR_PAWNS",
  "OUR_KNIGHTS",
  "OUR_BISHOPS",
  "OUR_ROOKS",
  "OUR_QUEENS",
  "THEIR_PAWNS",
  "THEIR_KNIGHTS",
  "THEIR_BISHOPS",
  "THEIR_ROOKS",
  "THEIR_QUEENS",
  "IN_CHECK",
  "KING_ON_BACK_RANK",
  "KING_ON_CENTER_FILE",
  "KING_ACTIVE",
  "THREATS_NEAR_KING_2",
  "THREATS_NEAR_KING_3",
  "OUR_PASSED_PAWNS",
  "THEIR_PASSED_PAWNS",
  "ISOLATED_PAWNS",
  "DOUBLED_PAWNS",
  "DOUBLE_ISOLATED_PAWNS",
  "PAWNS_CENTER_16",
  "PAWNS_CENTER_4",
  "ADVANCED_PASSED_PAWNS_2",
  "ADVANCED_PASSED_PAWNS_3",
  "ADVANCED_PASSED_PAWNS_4",
  "PAWN_MINOR_CAPTURES",
  "PAWN_MAJOR_CAPTURES",
  "PROTECTED_PAWNS",
  "PROTECTED_PASSED_PAWNS",
  "BISHOPS_DEVELOPED",
  "BISHOP_PAIR",
  "BLOCKADED_BISHOPS",
  "SCARY_BISHOPS",
  "SCARIER_BISHOPS",
  "BLOCKADED_ROOKS",
  "SCARY_ROOKS",
  "INFILTRATING_ROOKS",
  "KNIGHTS_DEVELOPED",
  "KNIGHT_MAJOR_CAPTURES",
  "KNIGHTS_CENTER_16",
  "KNIGHTS_CENTER_4",
  "KNIGHT_ON_ENEMY_SIDE",
  "OUR_HANGING_PAWNS",
  "OUR_HANGING_KNIGHTS",
  "OUR_HANGING_BISHOPS",
  "OUR_HANGING_ROOKS",
  "OUR_HANGING_QUEENS",
  "THEIR_HANGING_PAWNS",
  "THEIR_HANGING_KNIGHTS",
  "THEIR_HANGING_BISHOPS",
  "THEIR_HANGING_ROOKS",
  "THEIR_HANGING_QUEENS",
  "LONELY_KING_IN_CENTER",
  "LONELY_KING_AWAY_FROM_ENEMY_KING",
  "NUM_TARGET_SQUARES",
  "TIME",
  "KPVK_OPPOSITION",
  "KPVK_IN_FRONT_OF_PAWN",
  "KPVK_OFFENSIVE_KEY_SQUARES",
  "KPVK_DEFENSIVE_KEY_SQUARES",
  "SQUARE_RULE",
  "ADVANCED_PAWNS_1",
  "ADVANCED_PAWNS_2",
  "OPEN_ROOKS",
  "ROOKS_ON_THEIR_SIDE",
  "KING_CASTLED",
  "CASTLING_RIGHTS",
  "KING_IN_FRONT_OF_PASSED_PAWN",
  "KING_IN_FRONT_OF_PASSED_PAWN2",
  "PAWN_V_LONELY_KING",
  "KNIGHTS_V_LONELY_KING",
  "BISHOPS_V_LONELY_KING",
  "ROOK_V_LONELY_KING",
  "QUEEN_V_LONELY_KING",
  "OUR_MATERIAL_THREATS",
  "THEIR_MATERIAL_THREATS",
  "LONELY_KING_ON_EDGE",
  "OUTPOSTED_KNIGHTS",
  "OUTPOSTED_BISHOPS",
  "PAWN_MOVES",
  "KNIGHT_MOVES",
  "BISHOP_MOVES",
  "ROOK_MOVES",
  "QUEEN_MOVES",
  "PAWN_MOVES_ON_THEIR_SIDE",
  "KNIGHT_MOVES_ON_THEIR_SIDE",
  "BISHOP_MOVES_ON_THEIR_SIDE",
  "ROOK_MOVES_ON_THEIR_SIDE",
  "QUEEN_MOVES_ON_THEIR_SIDE",
  "KING_HOME_QUALITY",
  "BISHOPS_BLOCKING_KNIGHTS",
  "OUR_HANGING_PAWNS_2",
  "OUR_HANGING_KNIGHTS_2",
  "OUR_HANGING_BISHOPS_2",
  "OUR_HANGING_ROOKS_2",
  "OUR_HANGING_QUEENS_2",
  "THEIR_HANGING_PAWNS_2",
  "THEIR_HANGING_KNIGHTS_2",
  "THEIR_HANGING_BISHOPS_2",
  "THEIR_HANGING_ROOKS_2",
  "THEIR_HANGING_QUEENS_2",
  "QUEEN_THREATS_NEAR_KING",
  "MISSING_FIANCHETTO_BISHOP",
  "BISHOP_PAWN_DISAGREEMENT",
  "CLOSED_1",
  "CLOSED_2",
  "CLOSED_3",
  "NUM_BAD_SQUARES_FOR_PAWNS",
  "NUM_BAD_SQUARES_FOR_MINORS",
  "NUM_BAD_SQUARES_FOR_ROOKS",
  "NUM_BAD_SQUARES_FOR_QUEENS",
};

// captures = difference in values divided by 2

struct Evaluator {
  Evaluator() {
    earlyB = 0;
    std::fill_n(earlyW, EF::NUM_EVAL_FEATURES, 0);
    lateB = 0;
    std::fill_n(lateW, EF::NUM_EVAL_FEATURES, 0);
    clippedB = 0;
    std::fill_n(clippedW, EF::NUM_EVAL_FEATURES, 0);
lonelyKingB = 0;
    std::fill_n(lonelyKingW, EF::NUM_EVAL_FEATURES, 0);
  }

  int32_t earlyB;
  int32_t earlyW[EF::NUM_EVAL_FEATURES];
  int32_t lateB;
  int32_t lateW[EF::NUM_EVAL_FEATURES];
  int32_t clippedB;
  int32_t clippedW[EF::NUM_EVAL_FEATURES];
  int32_t lonelyKingB;
  int32_t lonelyKingW[EF::NUM_EVAL_FEATURES];

  void save_weights_to_file(const std::string& filename) {
    std::ofstream myfile;
    myfile.open(filename);
    if (!myfile.is_open()) {
      std::cout << "Error opening file \"" << filename << "\"" << std::endl;
      exit(0);
    }

    myfile << earlyB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      myfile << " " << earlyW[i];
    }
    myfile << std::endl;

    myfile << lateB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      myfile << " " << lateW[i];
    }
    myfile << std::endl;

    myfile << clippedB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      myfile << " " << clippedW[i];
    }
    myfile << std::endl;

    myfile << lonelyKingB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      myfile << " " << lonelyKingW[i];
    }
    myfile << std::endl;

    myfile.close();
  }

  void load_weights_from_file(const std::string& filename) {
    std::ifstream myfile;
    myfile.open(filename);
    if (!myfile.is_open()) {
      std::cout << "Error opening file \"" << filename << "\"" << std::endl;
      exit(0);
    }

    std::string line;
    std::vector<std::string> params;

    getline(myfile, line);
    params = split(line, ' ');
    if (params.size() != EF::NUM_EVAL_FEATURES + 1) {
      throw std::runtime_error("Invalid weights file");
    }
    earlyB = stoi(params[0]);
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      earlyW[i] = stoi(params[i + 1]);
    }

    getline(myfile, line);
    params = split(line, ' ');
    if (params.size() != EF::NUM_EVAL_FEATURES + 1) {
      throw std::runtime_error("Invalid weights file");
    }
    lateB = stoi(params[0]);
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      lateW[i] = stoi(params[i + 1]);
    }

    getline(myfile, line);
    params = split(line, ' ');
    if (params.size() != EF::NUM_EVAL_FEATURES + 1) {
      throw std::runtime_error("Invalid weights file");
    }
    clippedB = stoi(params[0]);
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      clippedW[i] = stoi(params[i + 1]);
    }

    getline(myfile, line);
    params = split(line, ' ');
    if (params.size() != EF::NUM_EVAL_FEATURES + 1) {
      throw std::runtime_error("Invalid weights file");
    }
    lonelyKingB = stoi(params[0]);
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      lonelyKingW[i] = stoi(params[i + 1]);
    }


    myfile.close();
  }

  template<Color US>
  Evaluation score(const Position& pos) {
    assert(pos.pieceBitboards_[ColoredPiece::WHITE_KING] > 0);
    assert(pos.pieceBitboards_[ColoredPiece::BLACK_KING] > 0);
    Threats<US> threats(pos);
    return this->score<US>(pos, threats);
  }

  template<Color US>
  Evaluation score(const Position& pos, const Threats<US>& threats) {
    constexpr Color THEM = opposite_color<US>();

    assert(pos.pieceBitboards_[ColoredPiece::WHITE_KING] > 0);
    assert(pos.pieceBitboards_[ColoredPiece::BLACK_KING] > 0);

    const Square ourKingSq = lsb(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
    const Square theirKingSq = lsb(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]);

    const Bitboard ourPawns = pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()];
    const Bitboard ourKnights = pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()];
    const Bitboard ourBishops = pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()];
    const Bitboard ourRooks = pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()];
    const Bitboard ourQueens = pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()];
    const Bitboard ourKings = pos.pieceBitboards_[coloredPiece<US, Piece::KING>()];

    const Bitboard theirPawns = pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()];
    const Bitboard theirKnights = pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()];
    const Bitboard theirBishops = pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()];
    const Bitboard theirRooks = pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()];
    const Bitboard theirQueens = pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()];
    const Bitboard theirKings = pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()];

    const Bitboard ourMen = pos.colorBitboards_[US];
    const Bitboard theirMen = pos.colorBitboards_[THEM];
    const Bitboard everyone = ourMen | theirMen;
    const bool isThreeManEndgame = std::popcount(ourMen | theirMen) == 3;
    bool isDraw = false;
    isDraw |= (ourMen == ourKings) && (theirMen == theirKings);
    isDraw |= ((ourMen | theirMen) == (ourKings | ourKnights | theirKings | theirKnights)) && isThreeManEndgame;
    isDraw |= ((ourMen | theirMen) == (ourKings | ourBishops | theirKings | theirBishops)) && isThreeManEndgame;
    if (isDraw) {
      return 0;
    }

    // TODO: penalty for double attacks near king

    const Bitboard ourRoyalty = ourQueens | ourKings;
    const Bitboard theirRoyalty = theirQueens | theirKings;
    const Bitboard ourMajors = ourRooks | ourRoyalty;
    const Bitboard theirMajors = theirRooks | theirRoyalty;
    const Bitboard ourMinors = ourKnights | ourBishops;
    const Bitboard theirMinors = theirKnights | theirBishops;
    const Bitboard ourPieces = ourMajors | ourMinors;
    const Bitboard theirPieces = theirMajors | theirMinors;

    const Bitboard ourRooklikePieces = ourRooks | ourQueens;
    const Bitboard theirRooklikePieces = theirRooks | theirQueens;
    const Bitboard ourBishoplikePieces = ourBishops | ourQueens;
    const Bitboard theirBishoplikePieces = theirBishops | theirQueens;

    constexpr Bitboard kOurSide = (US == Color::WHITE ? kWhiteSide : kBlackSide);
    constexpr Bitboard kTheirSide = (US == Color::WHITE ? kBlackSide : kWhiteSide);
    constexpr Direction kForward = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
    constexpr Direction kForward2 = (US == Color::WHITE ? Direction::NORTHx2 : Direction::SOUTHx2);
    constexpr Direction kBackward = opposite_dir(kForward);
    constexpr Bitboard kOurBackRanks = (US == Color::WHITE ? kRanks[6] | kRanks[7] : kRanks[1] | kRanks[0]);
    constexpr Bitboard kTheirBackRanks = (US == Color::WHITE ? kRanks[1] | kRanks[0] : kRanks[6] | kRanks[7]);
    constexpr Bitboard kHappyKingSquares = bb(62) | bb(58) | bb(57) | bb(6) | bb(1) | bb(2);

    features[EF::OUR_PAWNS] = std::popcount(ourPawns);
    features[EF::OUR_KNIGHTS] = std::popcount(ourKnights);
    features[EF::OUR_BISHOPS] = std::popcount(ourBishops);
    features[EF::OUR_ROOKS] = std::popcount(ourRooks);
    features[EF::OUR_QUEENS] = std::popcount(ourQueens);

    features[EF::THEIR_PAWNS] = std::popcount(theirPawns);
    features[EF::THEIR_KNIGHTS] = std::popcount(theirKnights);
    features[EF::THEIR_BISHOPS] = std::popcount(theirBishops);
    features[EF::THEIR_ROOKS] = std::popcount(theirRooks);
    features[EF::THEIR_QUEENS] = std::popcount(theirQueens);

    // Note that TURN (the side to move) often gets larger bonuses since they can take advantage of threats better.

    bool isUsInCheck = can_enemy_attack<US>(pos, ourKingSq);
    bool isThemInCheck = can_enemy_attack<THEM>(pos, theirKingSq);
    // Note: in a rigorous engine isThemInCheck would always be false on your move. For this engine,
    // it means they have moved into check (e.g. moving a pinned piece). Unfortunately we cannot
    // immediately return kMaxEval since it's possible they had had legal moves (and were in stalemate).
    features[EF::IN_CHECK] = isUsInCheck - isThemInCheck;
    if (US == Color::WHITE) {
      features[EF::KING_ON_BACK_RANK] = (ourKingSq / 8 == 7) - (theirKingSq / 8 == 0);
      features[EF::KING_ACTIVE] = (ourKingSq / 8 < 5) - (theirKingSq / 8 > 2);
    } else {
      features[EF::KING_ON_BACK_RANK] = (ourKingSq / 8 == 0) - (theirKingSq / 8 == 7);
      features[EF::KING_ACTIVE] = (ourKingSq / 8 > 2) - (theirKingSq / 8 < 5);
    }
    features[EF::KING_ON_CENTER_FILE] = (ourKingSq % 8 == 3 || ourKingSq % 8 == 4) - (theirKingSq % 8 == 3 || theirKingSq % 8 == 4);
    features[EF::THREATS_NEAR_KING_2] = std::popcount(kNearby[2][ourKingSq] & threats.theirTargets & ~threats.ourTargets) - std::popcount(kNearby[2][theirKingSq] & threats.ourTargets & ~threats.theirTargets);
    features[EF::THREATS_NEAR_KING_3] = std::popcount(kNearby[3][ourKingSq] & threats.theirTargets & ~threats.ourTargets) - std::popcount(kNearby[3][theirKingSq] & threats.ourTargets & ~threats.theirTargets);
    features[EF::QUEEN_THREATS_NEAR_KING] = std::popcount(kNearby[1][ourKingSq] & threats.theirDoubleTargets & ~threats.ourDoubleTargets & threats.theirQueenTargets) - std::popcount(kNearby[1][theirKingSq] & threats.ourDoubleTargets & ~threats.theirDoubleTargets & threats.ourQueenTargets);

    {  // Add penalty if the king is in a fianchettoed corner and his bishop is not on the main diagonal.
      // Note: the "color" of a corner is the color of its fianchettoed bishop.
      constexpr Bitboard kOurWhiteCorner = (US == Color::WHITE ? kWhiteKingCorner : kBlackQueenCorner);
      constexpr Bitboard kOurBlackCorner = (US == Color::BLACK ? kWhiteQueenCorner : kBlackKingCorner);
      constexpr Bitboard kTheirWhiteCorner = (US != Color::WHITE ? kWhiteKingCorner : kBlackQueenCorner);
      constexpr Bitboard kTheirBlackCorner = (US != Color::BLACK ? kWhiteQueenCorner : kBlackKingCorner);
      constexpr Bitboard ourWhiteFianchettoPawn = bb(US == Color::WHITE ? Square::G2 : Square::B7);
      constexpr Bitboard ourBlackFianchettoPawn = bb(US == Color::WHITE ? Square::B2 : Square::G7);
      constexpr Bitboard theirWhiteFianchettoPawn = bb(US != Color::WHITE ? Square::G2 : Square::B7);
      constexpr Bitboard theirBlackFianchettoPawn = bb(US != Color::WHITE ? Square::B2 : Square::G7);
      features[EF::MISSING_FIANCHETTO_BISHOP] = 0;
      features[EF::MISSING_FIANCHETTO_BISHOP] += ((ourKings & kOurWhiteCorner) > 0) && ((kMainWhiteDiagonal & ourBishops) == 0) && ((kWhiteSquares & theirBishops) > 0) && ((ourPawns & ourWhiteFianchettoPawn) == 0);
      features[EF::MISSING_FIANCHETTO_BISHOP] += ((ourKings & kOurBlackCorner) > 0) && ((kMainBlackDiagonal & ourBishops) == 0) && ((kBlackSquares & theirBishops) > 0) && ((ourPawns & ourBlackFianchettoPawn) == 0);
      features[EF::MISSING_FIANCHETTO_BISHOP] -= ((theirKings & kOurWhiteCorner) > 0) && ((kMainWhiteDiagonal & theirBishops) == 0) && ((kWhiteSquares & ourBishops) > 0) && ((theirPawns & theirWhiteFianchettoPawn) == 0);
      features[EF::MISSING_FIANCHETTO_BISHOP] -= ((theirKings & kOurBlackCorner) > 0) && ((kMainBlackDiagonal & theirBishops) == 0) && ((kBlackSquares & ourBishops) > 0) && ((theirPawns & theirBlackFianchettoPawn) == 0);
    }

    // Pawns
    const Bitboard ourBlockadedPawns = shift<kBackward>(theirPawns) & ourPawns;
    const Bitboard theirBlockadedPawns = shift<kForward>(ourPawns) & theirPawns;
    const Bitboard ourProtectedPawns = ourPawns & threats.ourPawnTargets;
    const Bitboard theirProtectedPawns = theirPawns & threats.theirPawnTargets;
    Bitboard ourPassedPawns, theirPassedPawns;
    Bitboard filesWithoutOurPawns, filesWithoutTheirPawns;
    Bitboard possibleOutpostsForUs, possibleOutpostsForThem;
    {
      Bitboard aheadOfOurPawns, aheadOfTheirPawns;
      Bitboard filesWithOurPawns, filesWithTheirPawns;
      if (US == Color::WHITE) {
        aheadOfOurPawns = northFill(ourPawns);
        aheadOfTheirPawns = southFill(theirPawns);
        filesWithOurPawns = southFill(aheadOfOurPawns);
        filesWithTheirPawns = northFill(aheadOfTheirPawns);
      } else {
        aheadOfOurPawns = southFill(ourPawns);
        aheadOfTheirPawns = northFill(theirPawns);
        filesWithOurPawns = northFill(aheadOfOurPawns);
        filesWithTheirPawns = southFill(aheadOfTheirPawns);
      }
      const Bitboard aheadOfOurPawnsFat = fatten(aheadOfOurPawns);
      const Bitboard aheadOfTheirPawnsFat = fatten(aheadOfTheirPawns);
      filesWithoutOurPawns = ~filesWithOurPawns;
      filesWithoutTheirPawns = ~filesWithTheirPawns;
      ourPassedPawns = ourPawns & ~shift<kBackward>(fatten(aheadOfTheirPawns));
      theirPassedPawns = theirPawns & ~shift<kForward>(fatten(aheadOfOurPawns));
      const Bitboard ourIsolatedPawns = ourPawns & ~shift<Direction::WEST>(filesWithOurPawns) & ~shift<Direction::EAST>(filesWithOurPawns);
      const Bitboard theirIsolatedPawns = theirPawns & ~shift<Direction::WEST>(filesWithTheirPawns) & ~shift<Direction::EAST>(filesWithTheirPawns);
      const Bitboard ourDoubledPawns = ourPawns & shift<kForward>(aheadOfOurPawns);
      const Bitboard theirDoubledPawns = theirPawns & shift<kBackward>(aheadOfTheirPawns);

      possibleOutpostsForUs = ~(
        shift<Direction::EAST>(shift<kBackward>(aheadOfTheirPawns))
        | shift<Direction::WEST>(shift<kBackward>(aheadOfTheirPawns))
      ) & ~kTheirBackRanks;
      possibleOutpostsForThem = ~(
        shift<Direction::EAST>(shift<kForward>(aheadOfOurPawns))
        | shift<Direction::WEST>(shift<kForward>(aheadOfOurPawns))
      ) & ~kOurBackRanks;

      constexpr Bitboard kRookPawns = kFiles[0] | kFiles[7];

      features[EF::PAWNS_CENTER_16] = std::popcount(ourPawns & kCenter16) - std::popcount(theirPawns & kCenter16);
      features[EF::PAWNS_CENTER_16] = std::popcount(ourPawns & kCenter16) - std::popcount(theirPawns & kCenter16);
      features[EF::PAWNS_CENTER_4] = std::popcount(ourPawns & kCenter4) - std::popcount(theirPawns & kCenter4);
      features[EF::OUR_PASSED_PAWNS] = std::popcount(ourPassedPawns);
      features[EF::THEIR_PASSED_PAWNS] = std::popcount(theirPassedPawns);
      features[EF::ISOLATED_PAWNS] = std::popcount(ourIsolatedPawns) - std::popcount(theirIsolatedPawns);
      features[EF::DOUBLED_PAWNS] = std::popcount(ourDoubledPawns) - std::popcount(theirDoubledPawns);
      features[EF::DOUBLE_ISOLATED_PAWNS] = std::popcount(ourDoubledPawns & ourIsolatedPawns) - std::popcount(theirDoubledPawns & theirIsolatedPawns);
      features[EF::ADVANCED_PAWNS_1] = std::popcount(ourPawns & kTheirBackRanks) - std::popcount(theirPawns & kOurBackRanks);
      features[EF::ADVANCED_PAWNS_2] = std::popcount(ourPawns & shift<kBackward>(kTheirBackRanks)) - std::popcount(theirPawns & shift<kForward>(kOurBackRanks));

      if (US == Color::WHITE) {
        features[EF::ADVANCED_PASSED_PAWNS_2] = std::popcount(ourPassedPawns & kRanks[1]) * 2 - std::popcount(theirPassedPawns & kRanks[6]);
        features[EF::ADVANCED_PASSED_PAWNS_3] = std::popcount(ourPassedPawns & kRanks[2]) * 2 - std::popcount(theirPassedPawns & kRanks[5]);
        features[EF::ADVANCED_PASSED_PAWNS_4] = std::popcount(ourPassedPawns & kRanks[3]) * 2 - std::popcount(theirPassedPawns & kRanks[4]);
      } else {
        features[EF::ADVANCED_PASSED_PAWNS_2] = std::popcount(ourPassedPawns & kRanks[6]) * 2 - std::popcount(theirPassedPawns & kRanks[1]);
        features[EF::ADVANCED_PASSED_PAWNS_3] = std::popcount(ourPassedPawns & kRanks[5]) * 2 - std::popcount(theirPassedPawns & kRanks[2]);
        features[EF::ADVANCED_PASSED_PAWNS_4] = std::popcount(ourPassedPawns & kRanks[4]) * 2 - std::popcount(theirPassedPawns & kRanks[3]);
      }

      features[EF::PAWN_MINOR_CAPTURES] = std::popcount(threats.ourPawnTargets & theirMinors) - std::popcount(threats.theirPawnTargets & ourMinors);
      features[EF::PAWN_MAJOR_CAPTURES] = std::popcount(threats.ourPawnTargets & theirMajors) - std::popcount(threats.theirPawnTargets & ourMajors);
      features[EF::PROTECTED_PAWNS] = std::popcount(ourPawns & threats.ourPawnTargets) - std::popcount(theirPawns & threats.theirPawnTargets);
      features[EF::PROTECTED_PASSED_PAWNS] = std::popcount(ourPassedPawns & threats.ourPawnTargets) - std::popcount(theirPassedPawns & threats.theirPawnTargets);
    }

    const Bitboard ourBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets<US>(pos, ourBishops, ourBlockadedPawns);
    const Bitboard theirBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets<THEM>(pos, theirBishops, theirBlockadedPawns);
    {  // Bishops
      if (US == Color::WHITE) {
        features[EF::BISHOPS_DEVELOPED] = std::popcount(theirBishops & (bb( 2) | bb( 5))) - std::popcount(ourBishops & (bb(58) | bb(61)));
      } else {
        features[EF::BISHOPS_DEVELOPED] = std::popcount(theirBishops & (bb(58) | bb(61))) - std::popcount(ourBishops & (bb( 2) | bb( 5)));
      }
      features[EF::BISHOP_PAIR] = (std::popcount(ourBishops) >= 2) - (std::popcount(theirBishops) >= 2);
      features[EF::BLOCKADED_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & (ourBlockadedPawns | theirProtectedPawns)) - std::popcount(theirBishopTargetsIgnoringNonBlockades & (theirBlockadedPawns | ourProtectedPawns));
      features[EF::SCARY_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & theirMajors) - std::popcount(theirBishopTargetsIgnoringNonBlockades & ourMajors);
      features[EF::SCARIER_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & theirRoyalty) - std::popcount(theirBishopTargetsIgnoringNonBlockades & ourRoyalty);
      features[EF::OUTPOSTED_BISHOPS] = std::popcount(ourBishops & possibleOutpostsForUs) - std::popcount(theirBishops & possibleOutpostsForThem);

      features[EF::BISHOP_PAWN_DISAGREEMENT] = std::popcount(ourBishops & ourPawns & kWhiteSquares) + std::popcount(ourBishops & ourPawns & kBlackSquares);
      features[EF::BISHOP_PAWN_DISAGREEMENT] -= std::popcount(theirBishops & theirPawns & kWhiteSquares) + std::popcount(theirBishops & theirPawns & kBlackSquares);

      features[EF::CLOSED_1] = std::popcount(ourBlockadedPawns | theirBlockadedPawns);
      features[EF::CLOSED_2] = std::popcount((ourBlockadedPawns | theirBlockadedPawns) & kCenter16);
      features[EF::CLOSED_3] = std::popcount(
        ((shift<Direction::NORTH_EAST>(ourBlockadedPawns) | shift<Direction::NORTH_WEST>(ourBlockadedPawns)) & ourBlockadedPawns)
        |
        ((shift<Direction::NORTH_EAST>(theirBlockadedPawns) | shift<Direction::NORTH_WEST>(theirBlockadedPawns)) & theirBlockadedPawns)
      );
    }

    {  // Rooks
      const Bitboard openFiles = filesWithoutOurPawns & filesWithoutTheirPawns;
      features[EF::BLOCKADED_ROOKS] = std::popcount(ourRooks & filesWithoutOurPawns) - std::popcount(theirRooks & filesWithoutTheirPawns);
      features[EF::SCARY_ROOKS] = std::popcount(threats.ourRookTargets & theirRoyalty) - std::popcount(threats.theirRookTargets & ourRoyalty);
      features[EF::INFILTRATING_ROOKS] = std::popcount(ourRooks & kTheirBackRanks) - std::popcount(theirRooks & kOurBackRanks);
      features[EF::OPEN_ROOKS] = std::popcount(ourRooks & openFiles) - std::popcount(theirRooks & openFiles);
      features[EF::ROOKS_ON_THEIR_SIDE] = std::popcount(ourRooks & kTheirSide) - std::popcount(theirRooks & kOurSide);
    }

    {  // Knights
      if (US == Color::WHITE) {
        features[EF::KNIGHTS_DEVELOPED] = std::popcount(theirKnights & (bb( 1) | bb( 6))) - std::popcount(ourKnights & (bb(57) | bb(62)));
      } else {
        features[EF::KNIGHTS_DEVELOPED] = std::popcount(theirKnights & (bb(57) | bb(62))) - std::popcount(ourKnights & (bb( 1) | bb( 6)));
      }
      features[EF::KNIGHT_MAJOR_CAPTURES] = std::popcount(threats.ourKnightTargets & theirMajors) - std::popcount(threats.theirKnightTargets & ourMajors);
      features[EF::KNIGHTS_CENTER_16] = std::popcount(ourKnights & kCenter16) - std::popcount(theirKnights & kCenter16);
      features[EF::KNIGHTS_CENTER_4] = std::popcount(ourKnights & kCenter4) - std::popcount(theirKnights & kCenter4);
      features[EF::KNIGHT_ON_ENEMY_SIDE] = std::popcount(ourKnights & kTheirSide) - std::popcount(theirKnights & kOurSide);
      features[EF::OUTPOSTED_KNIGHTS] = std::popcount(ourKnights & possibleOutpostsForUs) - std::popcount(theirKnights & possibleOutpostsForThem);

      features[EF::BISHOPS_BLOCKING_KNIGHTS] = std::popcount(shift<kForward>(shift<kForward>(shift<kForward>(theirKnights))) & ourBishops) - std::popcount(shift<kForward>(shift<kForward>(shift<kForward>(ourKnights))) & theirBishops);
    }

    const bool isOurKingLonely = (ourMen == ourKings);
    const bool isTheirKingLonely = (theirMen == theirKings);

    const CastlingRights cr = pos.currentState_.castlingRights;
    {
      // Hanging pieces are more valuable if it is your turn since they're literally free material,
      // as opposed to threats. Also is a very useful heuristic so that leaf nodes don't sack a rook
      // for a pawn.
      const Bitboard usHanging = threats.theirTargets & ~threats.ourTargets & pos.colorBitboards_[US];
      const Bitboard themHanging = threats.ourTargets & ~threats.theirTargets & pos.colorBitboards_[THEM];
      features[EF::OUR_HANGING_PAWNS] = std::popcount(ourPawns & usHanging);
      features[EF::THEIR_HANGING_PAWNS] = std::popcount(theirPawns & themHanging);
      features[EF::OUR_HANGING_KNIGHTS] = std::popcount(ourKnights & usHanging);
      features[EF::THEIR_HANGING_KNIGHTS] = std::popcount(theirKnights & themHanging);
      features[EF::OUR_HANGING_BISHOPS] = std::popcount(ourBishops & usHanging);
      features[EF::THEIR_HANGING_BISHOPS] = std::popcount(theirBishops & themHanging);
      features[EF::OUR_HANGING_ROOKS] = std::popcount(ourRooks & usHanging);
      features[EF::THEIR_HANGING_ROOKS] = std::popcount(theirRooks & themHanging);
      features[EF::OUR_HANGING_QUEENS] = std::popcount(ourQueens & usHanging);
      features[EF::THEIR_HANGING_QUEENS] = std::popcount(theirQueens & themHanging);

      const int wx = ourKingSq % 8;
      const int wy = ourKingSq / 8;
      const int bx = theirKingSq % 8;
      const int by = theirKingSq / 8;
      const int kingsDist = std::max(std::abs(wx - bx), std::abs(wy - by));
      features[EF::LONELY_KING_IN_CENTER] = value_or_zero(std::popcount(pos.colorBitboards_[THEM]) == 1, 3 - kDistToCorner[theirKingSq]);
      features[EF::LONELY_KING_IN_CENTER] -= value_or_zero(std::popcount(pos.colorBitboards_[US]) == 1, 3 - kDistToCorner[ourKingSq]);

      features[EF::LONELY_KING_AWAY_FROM_ENEMY_KING] = value_or_zero(isTheirKingLonely, 8 - kingsDist);
      features[EF::LONELY_KING_AWAY_FROM_ENEMY_KING] -= value_or_zero(isOurKingLonely, 8 - kingsDist);

      features[EF::CASTLING_RIGHTS] = ((cr & kCastlingRights_WhiteKing) > 0);
      features[EF::CASTLING_RIGHTS] += ((cr & kCastlingRights_WhiteQueen) > 0);
      features[EF::CASTLING_RIGHTS] -= ((cr & kCastlingRights_BlackKing) > 0);
      features[EF::CASTLING_RIGHTS] -= ((cr & kCastlingRights_BlackQueen) > 0);
      features[EF::KING_CASTLED] = std::popcount(ourKings & kHappyKingSquares) - std::popcount(theirKings & kHappyKingSquares);

      features[EF::NUM_TARGET_SQUARES] = std::popcount(threats.ourTargets) - std::popcount(threats.theirTargets);
    }

    {
      const Bitboard usHanging = threats.theirDoubleTargets & ~threats.ourDoubleTargets & pos.colorBitboards_[US];
      const Bitboard themHanging = threats.ourDoubleTargets & ~threats.theirDoubleTargets & pos.colorBitboards_[THEM];
      features[EF::OUR_HANGING_PAWNS_2] = std::popcount(ourPawns & usHanging);
      features[EF::THEIR_HANGING_PAWNS_2] = std::popcount(theirPawns & themHanging);
      features[EF::OUR_HANGING_KNIGHTS_2] = std::popcount(ourKnights & usHanging);
      features[EF::THEIR_HANGING_KNIGHTS_2] = std::popcount(theirKnights & themHanging);
      features[EF::OUR_HANGING_BISHOPS_2] = std::popcount(ourBishops & usHanging);
      features[EF::THEIR_HANGING_BISHOPS_2] = std::popcount(theirBishops & themHanging);
      features[EF::OUR_HANGING_ROOKS_2] = std::popcount(ourRooks & usHanging);
      features[EF::THEIR_HANGING_ROOKS_2] = std::popcount(theirRooks & themHanging);
      features[EF::OUR_HANGING_QUEENS_2] = std::popcount(ourQueens & usHanging);
      features[EF::THEIR_HANGING_QUEENS_2] = std::popcount(theirQueens & themHanging);
    }

    const bool isKingPawnEndgame = (ourKings == ourPieces) && (theirKings == theirPieces);
    // TODO: move this negative into the corresponding weights.
    features[EF::KPVK_OPPOSITION] = value_or_zero(isKingPawnEndgame, -((shift<kForward>(shift<kForward>(ourKings)) & theirKings) > 0));
    features[EF::KPVK_IN_FRONT_OF_PAWN] = 0;
    features[EF::KPVK_OFFENSIVE_KEY_SQUARES] = 0;
    features[EF::KPVK_DEFENSIVE_KEY_SQUARES] = 0;

    features[EF::SQUARE_RULE] = 0;
    features[EF::SQUARE_RULE] += value_or_zero(isKingPawnEndgame, (kSquareRuleTheirTurn[US][theirKingSq] & ourPassedPawns) > 0);
    features[EF::SQUARE_RULE] -= value_or_zero(isKingPawnEndgame, (kSquareRuleYourTurn[THEM][ourKingSq] & theirPassedPawns) > 0);

    {  // If we have the pawn in a KPVK engame.
      bool isKPVK = isKingPawnEndgame && (std::popcount(ourPawns) == 1) && (theirPawns == 0);

      const Bitboard inFrontOfPawn = shift<kForward>(ourPawns);
      const Bitboard keySquares = fatten(shift<kForward>(inFrontOfPawn & ~kRookFiles));
      const Bitboard inFront = (US == Color::WHITE ? (ourPawns - 1) : ~(ourPawns - 1));

      features[EF::KPVK_IN_FRONT_OF_PAWN] += value_or_zero(isKPVK, (ourKings & inFront) > 0);
      features[EF::KPVK_OFFENSIVE_KEY_SQUARES] += value_or_zero(isKPVK, (ourKings & keySquares) > 0);
      features[EF::KPVK_DEFENSIVE_KEY_SQUARES] += value_or_zero(isKPVK, (theirKings & inFrontOfPawn) > 0);
    }
    {  // If they have the pawn in a KPVK engame. Note we add a '+1' penalty for square rule
      bool isKPVK = isKingPawnEndgame && (std::popcount(theirPawns) == 1) && (ourPawns == 0);

      const Bitboard inFrontOfPawn = shift<kBackward>(theirPawns);
      const Bitboard keySquares = fatten(shift<kBackward>(inFrontOfPawn & ~kRookFiles));
      const Bitboard inFront = (US == Color::WHITE ? ~(theirPawns - 1) : (theirPawns - 1));

      features[EF::KPVK_IN_FRONT_OF_PAWN] -= value_or_zero(isKPVK, (theirKings & inFront) > 0);
      features[EF::KPVK_OFFENSIVE_KEY_SQUARES] -= value_or_zero(isKPVK, (theirKings & keySquares) > 0);
      features[EF::KPVK_DEFENSIVE_KEY_SQUARES] -= value_or_zero(isKPVK, (ourKings & inFrontOfPawn) > 0);
    }

    Bitboard aheadOfOurPassedPawnsFat, aheadOfTheirPassedPawnsFat;
    if (US == Color::WHITE) {
      aheadOfOurPassedPawnsFat = fatten(northFill(ourPassedPawns));
      aheadOfTheirPassedPawnsFat = fatten(southFill(theirPassedPawns));
    } else {
      aheadOfOurPassedPawnsFat = fatten(southFill(ourPassedPawns));
      aheadOfTheirPassedPawnsFat = fatten(northFill(theirPassedPawns));
    }

    // We split these into two features, the idea being that being ahead of your pawns while your opponent's
    // queen is on the board is dangerous, but being ahead of your opponent's passed pawns is not.
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN] = ((ourKings & aheadOfOurPassedPawnsFat) > 0 && theirQueens == 0);
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN] -= ((theirKings & aheadOfTheirPassedPawnsFat) > 0 && ourQueens == 0);
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN2] = (ourKings & aheadOfTheirPassedPawnsFat) > 0;
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN2] -= (theirKings & aheadOfOurPassedPawnsFat) > 0;

    // Bonus vs lonely king.
    features[EF::PAWN_V_LONELY_KING] = value_or_zero(isTheirKingLonely, std::popcount(ourPawns));
    features[EF::PAWN_V_LONELY_KING] -= value_or_zero(isOurKingLonely, std::popcount(theirPawns));
    features[EF::KNIGHTS_V_LONELY_KING] = value_or_zero(isTheirKingLonely, std::popcount(ourKnights));
    features[EF::KNIGHTS_V_LONELY_KING] -= value_or_zero(isOurKingLonely, std::popcount(theirKnights));
    features[EF::BISHOPS_V_LONELY_KING] = value_or_zero(isTheirKingLonely, std::popcount(ourBishops));
    features[EF::BISHOPS_V_LONELY_KING] -= value_or_zero(isOurKingLonely, std::popcount(theirBishops));
    features[EF::ROOK_V_LONELY_KING] = value_or_zero(isTheirKingLonely, std::popcount(ourRooks));
    features[EF::ROOK_V_LONELY_KING] -= value_or_zero(isOurKingLonely, std::popcount(theirRooks));
    features[EF::QUEEN_V_LONELY_KING] = value_or_zero(isTheirKingLonely, std::popcount(ourQueens));
    features[EF::QUEEN_V_LONELY_KING] -= value_or_zero(isOurKingLonely, std::popcount(theirQueens));

    // loser in center: 0
    // they're lonely on edge: 3
    //   we're lonely on edge: -3
    features[EF::LONELY_KING_ON_EDGE] = value_or_zero(isTheirKingLonely, 3 - kDistToEdge[theirKingSq]);
    features[EF::LONELY_KING_ON_EDGE] -= value_or_zero(isOurKingLonely, 3 - kDistToEdge[ourKingSq]);

    {
      Bitboard ourMaterialThreats = 0;
      ourMaterialThreats |= threats.ourPawnTargets & theirPieces;
      ourMaterialThreats |= threats.ourKnightTargets & theirMajors;
      ourMaterialThreats |= threats.ourBishopTargets & theirMajors;
      ourMaterialThreats |= threats.ourRookTargets & theirRoyalty;
      ourMaterialThreats |= threats.ourQueenTargets & theirKings;

      Bitboard theirMaterialThreats = 0;
      theirMaterialThreats |= threats.theirPawnTargets & ourPieces;
      theirMaterialThreats |= threats.theirKnightTargets & ourMajors;
      theirMaterialThreats |= threats.theirBishopTargets & ourMajors;
      theirMaterialThreats |= threats.theirRookTargets & ourRoyalty;
      theirMaterialThreats |= threats.theirQueenTargets & ourKings;

      features[EF::OUR_MATERIAL_THREATS] = std::popcount(ourMaterialThreats);
      features[EF::THEIR_MATERIAL_THREATS] = std::popcount(theirMaterialThreats);
    }

    {
      const Bitboard ourGoodKnightTargets = threats.ourKnightTargets & ~threats.theirPawnTargets;
      const Bitboard theirGoodKnightTargets = threats.ourKnightTargets & ~threats.theirPawnTargets;
      const Bitboard ourGoodBishopTargets = ourBishopTargetsIgnoringNonBlockades & ~threats.theirPawnTargets;
      const Bitboard theirGoodBishopTargets = theirBishopTargetsIgnoringNonBlockades & ~threats.ourPawnTargets;
      const Bitboard ourGoodRookTargets = threats.ourRookTargets & ~threats.theirPawnTargets;
      const Bitboard theirGoodRookTargets = threats.theirRookTargets & ~threats.ourPawnTargets;
      const Bitboard ourGoodQueenTargets = threats.ourQueenTargets & ~(threats.theirPawnTargets | threats.theirKnightTargets | threats.theirBishopTargets | threats.theirRookTargets);
      const Bitboard theirGoodQueenTargets = threats.theirQueenTargets & ~(threats.ourPawnTargets | threats.ourKnightTargets | threats.ourBishopTargets | threats.ourRookTargets);
      features[EF::PAWN_MOVES] = std::popcount(threats.ourPawnTargets) - std::popcount(threats.theirPawnTargets);
      features[EF::KNIGHT_MOVES] = std::popcount(ourGoodKnightTargets) - std::popcount(theirGoodKnightTargets);
      features[EF::BISHOP_MOVES] = std::popcount(ourGoodBishopTargets) - std::popcount(theirGoodBishopTargets);
      features[EF::ROOK_MOVES] = std::popcount(ourGoodRookTargets) - std::popcount(theirGoodRookTargets);
      features[EF::QUEEN_MOVES] = std::popcount(ourGoodQueenTargets) - std::popcount(theirGoodQueenTargets);
      features[EF::PAWN_MOVES_ON_THEIR_SIDE] = std::popcount(threats.ourPawnTargets & kTheirSide) - std::popcount(threats.theirPawnTargets & kOurSide);
      features[EF::KNIGHT_MOVES_ON_THEIR_SIDE] = std::popcount(ourGoodKnightTargets & kTheirSide) - std::popcount(theirGoodKnightTargets & kOurSide);
      features[EF::BISHOP_MOVES_ON_THEIR_SIDE] = std::popcount(ourGoodBishopTargets & kTheirSide) - std::popcount(theirGoodBishopTargets & kOurSide);
      features[EF::ROOK_MOVES_ON_THEIR_SIDE] = std::popcount(ourGoodRookTargets & kTheirSide) - std::popcount(theirGoodRookTargets & kOurSide);
      features[EF::QUEEN_MOVES_ON_THEIR_SIDE] = std::popcount(ourGoodQueenTargets & kTheirSide) - std::popcount(theirGoodQueenTargets & kOurSide);
    }

    // Bonus for king having pawns in front of him, or having pawns in front of him once he castles.
    features[EF::KING_HOME_QUALITY] = std::popcount(kKingHome[ourKingSq] & ourPawns) - std::popcount(kKingHome[theirKingSq] & theirPawns);
    const Evaluation whitePotentialHome = std::max(
      value_or_zero((cr & kCastlingRights_WhiteKing) > 0, std::popcount(kKingHome[Square::G1] & ourPawns)),
      value_or_zero((cr & kCastlingRights_WhiteQueen) > 0, std::popcount(kKingHome[Square::B1] & ourPawns))
    );
    const Evaluation blackPotentialHome = std::max(
      value_or_zero((cr & kCastlingRights_BlackKing) > 0, std::popcount(kKingHome[Square::G8] & ourPawns)),
      value_or_zero((cr & kCastlingRights_BlackQueen) > 0, std::popcount(kKingHome[Square::B8] & ourPawns))
    );
    if (US == Color::WHITE) {
      features[EF::KING_HOME_QUALITY] += whitePotentialHome - blackPotentialHome;
    } else {
      features[EF::KING_HOME_QUALITY] += blackPotentialHome - whitePotentialHome;
    }

    features[EF::NUM_BAD_SQUARES_FOR_PAWNS] = std::popcount(threats.badForOur[Piece::PAWN] & kCenter16) - std::popcount(threats.badForTheir[Piece::PAWN] & kCenter16);
    features[EF::NUM_BAD_SQUARES_FOR_MINORS] = std::popcount(threats.badForOur[Piece::KNIGHT] & kCenter16) - std::popcount(threats.badForTheir[Piece::KNIGHT] & kCenter16);
    features[EF::NUM_BAD_SQUARES_FOR_ROOKS] = std::popcount(threats.badForOur[Piece::ROOK] & kCenter16) - std::popcount(threats.badForTheir[Piece::ROOK] & kCenter16);
    features[EF::NUM_BAD_SQUARES_FOR_QUEENS] = std::popcount(threats.badForOur[Piece::QUEEN] & kCenter16) - std::popcount(threats.badForTheir[Piece::QUEEN] & kCenter16);

    const int16_t ourPiecesRemaining = std::popcount(pos.colorBitboards_[US] & ~ourPawns) + std::popcount(ourQueens) * 2 - 1;
    const int16_t theirPiecesRemaining = std::popcount(pos.colorBitboards_[THEM] & ~theirPawns) + std::popcount(theirQueens) * 2 - 1;
    const int32_t time = 18 - (ourPiecesRemaining + theirPiecesRemaining);

    features[EF::TIME] = time;

    // Use larger integer to make arithmetic safe.
    const int32_t early = this->early<US>(pos);
    const int32_t late = this->late<US>(pos);
    const int32_t clipped = this->clipped<US>(pos);
    const int32_t lonely_king = this->lonely_king<US>(pos);

    // 0.043 ± 0.019
    int32_t pieceMap = (pos.earlyPieceMapScore_ * (18 - time) + pos.latePieceMapScore_ * time) / 18;
    if (US == Color::BLACK) {
      pieceMap *= -1;
    }

    int32_t eval = (early * (18 - time) + late * time) / 18 + clipped + lonely_king + pieceMap;


    const int wx = ourKingSq % 8;
    const int wy = ourKingSq / 8;
    const int bx = theirKingSq % 8;
    const int by = theirKingSq / 8;

    {
      // KPVK games are winning if square rule is true.
      const bool isOurKPPVK = isKingPawnEndgame && (std::popcount(ourPawns) >= 1) && (theirPawns == 0);
      const bool isTheirKPPVK = isKingPawnEndgame && (std::popcount(theirPawns) >= 1) && (ourPawns == 0);
      constexpr Bitboard kPromoRanks = kRanks[0] | kRanks[7];
      eval -= value_or_zero(isOurKPPVK && features[EF::SQUARE_RULE] < 0, 500);
      eval += value_or_zero(isTheirKPPVK && features[EF::SQUARE_RULE] > 0, 500);

      if (isOurKPPVK && std::popcount(ourPawns) == 1) {
        int result;
        if (US == Color::WHITE) {
          result = this->known_kpvk_result(ourKingSq, theirKingSq, lsb(ourPawns), true);
        } else {
          result = this->known_kpvk_result(Square(63 - ourKingSq), Square(63 - theirKingSq), Square(63 - lsb(ourPawns)), true);
        }
        if (result == 0) {
          return 0;
        }
        if (result == 2) {
          eval += 1000;
        }
      }
      if (isTheirKPPVK && std::popcount(theirPawns) == 1) {
        int result;
        if (US == Color::BLACK) {
          result = this->known_kpvk_result(theirKingSq, ourKingSq, lsb(theirPawns), false);
        } else {
          result = this->known_kpvk_result(Square(63 - theirKingSq), Square(63 - ourKingSq), Square(63 - lsb(theirPawns)), false);
        }
        if (result == 0) {
          return 0;
        }
        if (result == 2) {
          eval -= 1000;
        }
      }
    }
    {  // KRvK, KQvK, KBBvK, KBNvK
      const bool theyHaveLonelyKing = (theirMen == theirKings) && (features[EF::OUR_KNIGHTS] > 2 || features[EF::OUR_BISHOPS] > 1 || features[EF::OUR_ROOKS] > 0 || features[EF::OUR_QUEENS] > 0);
      const bool weHaveLonelyKing = (ourMen == ourKings) && (features[EF::THEIR_KNIGHTS] > 2 || features[EF::THEIR_BISHOPS] > 1 || features[EF::THEIR_ROOKS] > 0 || features[EF::THEIR_QUEENS] > 0);

      eval += value_or_zero(theyHaveLonelyKing, (3 - kDistToEdge[theirKingSq]) * 50);
      eval -= value_or_zero(  weHaveLonelyKing, (3 - kDistToEdge[ourKingSq]) * 50);
      eval += value_or_zero(theyHaveLonelyKing, (3 - kDistToCorner[theirKingSq]) * 50);
      eval -= value_or_zero(  weHaveLonelyKing, (3 - kDistToCorner[ourKingSq]) * 50);

      int dx = std::abs(wx - bx);
      int dy = std::abs(wy - by);
      const bool opposition = (dx == 2 && dy == 0) || (dx == 0 && dy == 2);

      // We don't want it to be our turn!
      eval -= value_or_zero(weHaveLonelyKing && opposition, 75);
      // We can achieve opposition from here.
      eval += value_or_zero(theyHaveLonelyKing && (
        (dx == 3 && dy <= 1)
        || (dy == 3 && dx <= 1)
      ), 50);

      // And put our king next to the enemy king.
      eval += value_or_zero(theyHaveLonelyKing, (8 - std::max(dx, dy)) * 50);
      eval -= value_or_zero(  weHaveLonelyKing, (8 - std::max(dx, dy)) * 50);
      eval += value_or_zero(theyHaveLonelyKing, (8 - std::min(dx, dy)) * 25);
      eval -= value_or_zero(  weHaveLonelyKing, (8 - std::min(dx, dy)) * 25);
    }

    return std::min(int32_t(-kLongestForcedMate), std::max(int32_t(kLongestForcedMate), eval));
  }

  template<Color US>
  Evaluation early(const Position& pos) const {
    int32_t r = earlyB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * earlyW[i];
    }
    return r;
  }

  template<Color US>
  Evaluation late(const Position& pos) const {
    int32_t r = lateB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * lateW[i];
    }
    return r;
  }

  template<Color US>
  Evaluation clipped(const Position& pos) const {
    int32_t r = clippedB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * clippedW[i];
    }
    return std::max(-100, std::min(100, r));
  }

  template<Color US>
  Evaluation lonely_king(const Position& pos) const {
    int32_t r = lonelyKingB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * lonelyKingW[i];
    }

    const int32_t ourPieces = features[EF::OUR_KNIGHTS] + features[EF::OUR_BISHOPS] + features[EF::OUR_ROOKS] + features[EF::OUR_QUEENS];
    const int32_t theirPieces = features[EF::THEIR_KNIGHTS] + features[EF::THEIR_BISHOPS] + features[EF::THEIR_ROOKS] + features[EF::THEIR_QUEENS];

    // todo: value_or_zero
    return r * (1 - (ourPieces != 0) * (theirPieces != 0));
  }

  // Assumes white has the pawn.
  // Returns 2 if white wins
  // Returns 0 if white draws
  // Returns 1 if unknown
  int known_kpvk_result(Square yourKing, Square theirKing, Square yourPawn, bool yourMove) {
    const int wx = yourKing % 8;
    const int wy = yourKing / 8;
    const int bx = theirKing % 8;
    const int by = theirKing / 8;
    const int px = yourPawn % 8;
    const int py = yourPawn / 8;

    const int wdist = std::max(std::abs(wx - px), std::abs(wy - py));
    const int bdist = std::max(std::abs(bx - px), std::abs(by - py));


    bool isWinning = false;
    {  // Square rule
      const int theirDistFromPromoSquare = std::max(std::abs(bx - px), by) - !yourMove;
      isWinning |= theirDistFromPromoSquare > py;
    }

    isWinning |= (wy == py - 2 && std::abs(wx - px) == 1 && bdist + yourMove > 1);

    // Horizontally symmetric is a win for white.
    isWinning |= (wx - px == px - bx && wy == by);

    if (isWinning) {
      return 2;
    }

    // if (wx == bx && wy >= py - 1 && by == wy - 2 && by != 0 && yourMove) {
    //   return 0;
    // }

    bool isDrawn = false;

    // Black king in front of pawn.
    isDrawn |= (bx == px && by == py - 1);

    // Black king two in front of pawn and not on back rank.
    isDrawn |= (by == py - 2 && by != 0);

    // Distance Rule:
    //   1) Compute the distance between your king and your pawn
    //   2) Compute the distance between the enemy king and your pawn
    //   3) Subtract 1 from your distance if it's your turn
    //   4) Add 1 to your enemy's distance if they're in front of your pawn and on a diagonal with it.
    // If your distance is greater than your opponent's, then it's a draw.
    isDrawn |= (wdist - yourMove > bdist + ((bx + by == wx + wy) || (bx - by == wx - wy)));

    // No-zones when you're behind your pawn.
    isDrawn |= (wy > py && py > by) && (std::abs(px - bx) - !yourMove <= wy - py);

    return !isDrawn;
  }

  Evaluation features[NUM_EVAL_FEATURES];
};

}  // namespace ChessEngine

#endif  // EVALUATOR_H