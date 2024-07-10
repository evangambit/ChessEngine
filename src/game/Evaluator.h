#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <cstdint>

#include <algorithm>
#include <fstream>
#include <string>

#include "geometry.h"
#include "kpvk.h"
#include "utils.h"
#include "Threats.h"
#include "Position.h"
#include "movegen.h"
#include "piece_maps.h"
#include "PawnAnalysis.h"

namespace ChessEngine {

// Arbitrary constant that our "special_boosts" will never accidentally return.
static const Evaluation kKnownDraw = kMinEval + 10;

/**
 * Engine is bad at
 * 1) knowing when trapped pieces (e.g. knights in corners) aren't long for this world
 * 2) knowing when very extended/isolated pawns will probably be captured in endgames
 *    (bc one king is close to it and the other is far)
 * 3) more generally, understanding what squares the king wants to get to, and rewarding
 *    the king for being close to them (namely promotion squares and isolated pawns).
 * 4) pawn storms
 */

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

  KING_ON_CENTER_FILE,  // +0.2578 ± 0.0532
  KING_ACTIVE,
  THREATS_NEAR_KING_2,
  THREATS_NEAR_KING_3,
  PASSED_PAWNS,

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
  TIME,
  KPVK_OPPOSITION,
  SQUARE_RULE,
  ADVANCED_PAWNS_1,
  ADVANCED_PAWNS_2,
  OPEN_ROOKS,
  ROOKS_ON_THEIR_SIDE,

  KING_IN_FRONT_OF_PASSED_PAWN,
  KING_IN_FRONT_OF_PASSED_PAWN2,
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

  NUM_BAD_SQUARES_FOR_PAWNS,
  NUM_BAD_SQUARES_FOR_MINORS,
  NUM_BAD_SQUARES_FOR_ROOKS,
  NUM_BAD_SQUARES_FOR_QUEENS,
  IN_TRIVIAL_CHECK,
  IN_DOUBLE_CHECK,

  THREATS_NEAR_OUR_KING,
  THREATS_NEAR_THEIR_KING,
  NUM_PIECES_HARRASSABLE_BY_PAWNS,  // 0.0322 ± 0.0174 (96.8%)

  PAWN_CHECKS,
  KNIGHT_CHECKS,
  BISHOP_CHECKS,
  ROOK_CHECKS,
  QUEEN_CHECKS,

  BACK_RANK_MATE_THREAT_AGAINST_US,
  BACK_RANK_MATE_THREAT_AGAINST_THEM,

  OUR_KING_HAS_0_ESCAPE_SQUARES,
  THEIR_KING_HAS_0_ESCAPE_SQUARES,
  OUR_KING_HAS_1_ESCAPE_SQUARES,
  THEIR_KING_HAS_1_ESCAPE_SQUARES,
  OUR_KING_HAS_2_ESCAPE_SQUARES,
  THEIR_KING_HAS_2_ESCAPE_SQUARES,

  OPPOSITE_SIDE_KINGS_PAWN_STORM,
  IN_CHECK_AND_OUR_HANGING_QUEENS,
  PROMOTABLE_PAWN,
  PINNED_PIECES,

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
  "PASSED_PAWNS",
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
  "TIME",
  "KPVK_OPPOSITION",
  "SQUARE_RULE",
  "ADVANCED_PAWNS_1",
  "ADVANCED_PAWNS_2",
  "OPEN_ROOKS",
  "ROOKS_ON_THEIR_SIDE",
  "KING_IN_FRONT_OF_PASSED_PAWN",
  "KING_IN_FRONT_OF_PASSED_PAWN2",
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
  "NUM_BAD_SQUARES_FOR_PAWNS",
  "NUM_BAD_SQUARES_FOR_MINORS",
  "NUM_BAD_SQUARES_FOR_ROOKS",
  "NUM_BAD_SQUARES_FOR_QUEENS",
  "IN_TRIVIAL_CHECK",
  "IN_DOUBLE_CHECK",
  "THREATS_NEAR_OUR_KING",
  "THREATS_NEAR_THEIR_KING",
  "NUM_PIECES_HARRASSABLE_BY_PAWNS",
  "PAWN_CHECKS",
  "KNIGHT_CHECKS",
  "BISHOP_CHECKS",
  "ROOK_CHECKS",
  "QUEEN_CHECKS",
  "BACK_RANK_MATE_THREAT_AGAINST_US",
  "BACK_RANK_MATE_THREAT_AGAINST_THEM",
  "OUR_KING_HAS_0_ESCAPE_SQUARES",
  "THEIR_KING_HAS_0_ESCAPE_SQUARES",
  "OUR_KING_HAS_1_ESCAPE_SQUARES",
  "THEIR_KING_HAS_1_ESCAPE_SQUARES",
  "OUR_KING_HAS_2_ESCAPE_SQUARES",
  "THEIR_KING_HAS_2_ESCAPE_SQUARES",
  "OPPOSITE_SIDE_KINGS_PAWN_STORM",
  "IN_CHECK_AND_OUR_HANING_QUEENS",
  "PROMOTABLE_PAWN",
  "PINNED_PIECES",
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
  }

  int32_t earlyB;
  int32_t earlyW[EF::NUM_EVAL_FEATURES];
  int32_t lateB;
  int32_t lateW[EF::NUM_EVAL_FEATURES];
  int32_t clippedB;
  int32_t clippedW[EF::NUM_EVAL_FEATURES];

  void save_weights_to_file(std::ofstream& myfile) {
    myfile << lpad(earlyB) << "  // early bias" << std::endl;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      myfile << lpad(earlyW[i]) << "  // early " << EFSTR[i] << std::endl;
    }

    myfile << lpad(lateB) << "  // late bias" << std::endl;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      myfile << lpad(lateW[i]) << "  // late " << EFSTR[i] << std::endl;
    }

    myfile << lpad(clippedB) << "  // clipped bias" << std::endl;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      myfile << lpad(clippedW[i]) << "  // clipped " << EFSTR[i] << std::endl;
    }
  }

  void load_weights_from_file(std::ifstream &myfile) {
    std::string line;
    std::vector<std::string> params;


    getline(myfile, line);
    try {
      earlyB = stoi(process_with_file_line(line));
    } catch (std::invalid_argument& err) {
      std::cout << "error loading earlyB" << std::endl;
      throw err;
    }
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      getline(myfile, line);
      try {
        earlyW[i] = stoi(process_with_file_line(line));
      } catch (std::invalid_argument& err) {
        std::cout << "error loading earlyW[i]" << std::endl;
        throw err;
      }
    }

    getline(myfile, line);
    try {
      lateB = stoi(process_with_file_line(line));
    } catch (std::invalid_argument& err) {
      std::cout << "error loading lateB" << std::endl;
      throw err;
    }
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      getline(myfile, line);
      try {
        lateW[i] = stoi(process_with_file_line(line));
      } catch (std::invalid_argument& err) {
        std::cout << "error loading lateW[i]" << std::endl;
        throw err;
      }
    }

    getline(myfile, line);
    try {
      clippedB = stoi(process_with_file_line(line));
    } catch (std::invalid_argument& err) {
      std::cout << "error loading clippedB" << std::endl;
      throw err;
    }
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      getline(myfile, line);
      try {
        clippedW[i] = stoi(process_with_file_line(line));
      } catch (std::invalid_argument& err) {
        std::cout << "error loading clippedW[" << i << "]" << std::endl;
        throw err;
      }
    }
  }

  Evaluation pawnValue() const {
    return this->earlyW[EF::OUR_PAWNS] + this->clippedW[EF::OUR_PAWNS];
  }

  template<Color US>
  Evaluation score(const Position& pos) {
    assert(pos.pieceBitboards_[ColoredPiece::WHITE_KING] > 0);
    assert(pos.pieceBitboards_[ColoredPiece::BLACK_KING] > 0);
    Threats<US> threats(pos);
    return this->score<US>(pos, threats);
  }

  bool is_material_draw(const Position& pos) const {
    const Bitboard everyone = pos.colorBitboards_[Color::WHITE] | pos.colorBitboards_[Color::BLACK];
    const Bitboard everyoneButKings = everyone & ~(pos.pieceBitboards_[ColoredPiece::WHITE_KING] | pos.pieceBitboards_[ColoredPiece::BLACK_KING]);
    const bool isThreeManEndgame = std::popcount(everyone) == 3;
    bool isDraw = false;
    isDraw |= (everyoneButKings == 0);
    isDraw |= (everyoneButKings == (pos.pieceBitboards_[ColoredPiece::WHITE_KNIGHT] | pos.pieceBitboards_[ColoredPiece::BLACK_KNIGHT])) && isThreeManEndgame;
    isDraw |= (everyoneButKings == (pos.pieceBitboards_[ColoredPiece::WHITE_BISHOP] | pos.pieceBitboards_[ColoredPiece::BLACK_BISHOP])) && isThreeManEndgame;
    return isDraw;
  }

  template<Color US>
  Evaluation score(const Position& pos, const Threats<US>& threats) {
#ifndef NO_NNUE_EVAL
    float score = pos.network->fastforward();
    if (US == Color::BLACK) {
      score *= -1.0;
    }
    return Evaluation(
      std::min(
        -float(kQLongestForcedMate),
        std::max(
          float(kQLongestForcedMate),
          std::round(score * 500)
        )
      )
    );
#else
    constexpr Color THEM = opposite_color<US>();

    assert(pos.pieceBitboards_[ColoredPiece::WHITE_KING] > 0);
    assert(pos.pieceBitboards_[ColoredPiece::BLACK_KING] > 0);

    if (this->is_material_draw(pos)) {
      return 0;
    }

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

    const Square whiteKingSq = (US == Color::WHITE ? ourKingSq : theirKingSq);
    const Square blackKingSq = (US == Color::BLACK ? ourKingSq : theirKingSq);

    const Bitboard ourMen = pos.colorBitboards_[US];
    const Bitboard theirMen = pos.colorBitboards_[THEM];
    const Bitboard everyone = ourMen | theirMen;

    // TODO: penalty for double attacks near king

    const Bitboard ourRoyalty = ourQueens | ourKings;
    const Bitboard theirRoyalty = theirQueens | theirKings;
    const Bitboard ourMajors = ourRooks | ourRoyalty;
    const Bitboard theirMajors = theirRooks | theirRoyalty;
    const Bitboard ourMinors = ourKnights | ourBishops;
    const Bitboard theirMinors = theirKnights | theirBishops;
    const Bitboard ourPieces = ourMajors | ourMinors;
    const Bitboard theirPieces = theirMajors | theirMinors;

    constexpr Bitboard kOurSide = (US == Color::WHITE ? kWhiteSide : kBlackSide);
    constexpr Bitboard kTheirSide = (US == Color::WHITE ? kBlackSide : kWhiteSide);
    constexpr Direction kForward = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
    constexpr Direction kBackward = opposite_dir<kForward>();
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

    const int16_t ourPiecesRemaining = std::popcount(pos.colorBitboards_[US] & ~ourPawns) + std::popcount(ourQueens) * 2 - 1;
    const int16_t theirPiecesRemaining = std::popcount(pos.colorBitboards_[THEM] & ~theirPawns) + std::popcount(theirQueens) * 2 - 1;
    const int32_t time = std::max(0, std::min(18, 18 - (ourPiecesRemaining + theirPiecesRemaining)));

    #ifndef SquareControl

    // Note that TURN (the side to move) often gets larger bonuses since they can take advantage of threats better.
    {
      Bitboard attackingOurKing = compute_attackers<THEM>(pos, ourKingSq);
      // Note: in a rigorous engine isThemInCheck would always be false on your move. For this engine,
      // it means they have moved into check (e.g. moving a pinned piece). Unfortunately we cannot
      // immediately return kMaxEval since it's possible they had had legal moves (and were in stalemate).
      features[EF::IN_CHECK] = (attackingOurKing > 0);
      features[EF::IN_DOUBLE_CHECK] = std::popcount(attackingOurKing) > 1;

      attackingOurKing &= ~(theirQueens & threats.badForTheir[Piece::QUEEN]);
      attackingOurKing &= ~(theirRooks & threats.badForTheir[Piece::ROOK]);
      attackingOurKing &= ~(theirBishops & threats.badForTheir[Piece::BISHOP]);
      attackingOurKing &= ~(theirKnights & threats.badForTheir[Piece::KNIGHT]);
      attackingOurKing &= ~(theirPawns & threats.badForTheir[Piece::PAWN]);

      // The piece checking our king can simply be captured.
      features[EF::IN_TRIVIAL_CHECK] = features[EF::IN_CHECK] && (attackingOurKing == 0) && !features[EF::IN_DOUBLE_CHECK];
    }

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
    features[EF::THREATS_NEAR_OUR_KING] = std::popcount(kNearby[1][ourKingSq] & threats.theirDoubleTargets & ~threats.ourDoubleTargets);
    features[EF::THREATS_NEAR_THEIR_KING] = std::popcount(kNearby[1][theirKingSq] & threats.ourDoubleTargets & ~threats.theirDoubleTargets);

    PinMasks ourPinnedMask = compute_pin_masks<US>(pos, ourKingSq);
    PinMasks theirPinnedMask = compute_pin_masks<THEM>(pos, theirKingSq);
    ourPinnedMask |= compute_pin_masks<US>(pos, lsb_or(ourQueens, ourKingSq));
    theirPinnedMask |= compute_pin_masks<THEM>(pos, lsb_or(theirQueens, theirKingSq));

    //   0: 21.8462 ± 0.0106847
    // -10: 21.826 ± 0.0106827
    // -20: 21.819  ± 0.0106812
    // -40: 21.8631 ± 0.0106809

    // -20: 21.8045 ± 0.0106802
    // -40: 21.7966 ± 0.0106796
    // -60: 21.7961 ± 0.0106792

    // Bishops being pinned by a bishop or rooks being pinned by a rook aren't
    // a big deal. We're concerned with pieces being pinned and not being able
    // to capture their pinner.
     // 0.021 ± 0.015
    features[EF::PINNED_PIECES] = 
      std::popcount((ourPinnedMask.horizontal | ourPinnedMask.vertical) & (ourPieces & ~ourRooks))
      + std::popcount((ourPinnedMask.northeast | ourPinnedMask.northwest) & (ourPieces & ~ourBishops))
      - std::popcount((theirPinnedMask.horizontal | theirPinnedMask.vertical) & (theirPieces & ~theirRooks))
      - std::popcount((theirPinnedMask.northeast | theirPinnedMask.northwest) & (theirPieces & ~theirBishops));

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
      features[EF::MISSING_FIANCHETTO_BISHOP] -= ((theirKings & kTheirWhiteCorner) > 0) && ((kMainWhiteDiagonal & theirBishops) == 0) && ((kWhiteSquares & ourBishops) > 0) && ((theirPawns & theirWhiteFianchettoPawn) == 0);
      features[EF::MISSING_FIANCHETTO_BISHOP] -= ((theirKings & kTheirBlackCorner) > 0) && ((kMainBlackDiagonal & theirBishops) == 0) && ((kBlackSquares & ourBishops) > 0) && ((theirPawns & theirBlackFianchettoPawn) == 0);
    }

    // Pawns
    PawnAnalysis<US> pawnAnalysis(pos, threats);
    {
      features[EF::PAWNS_CENTER_16] = std::popcount(ourPawns & kCenter16) - std::popcount(theirPawns & kCenter16);
      features[EF::PAWNS_CENTER_16] = std::popcount(ourPawns & kCenter16) - std::popcount(theirPawns & kCenter16);
      features[EF::PAWNS_CENTER_4] = std::popcount(ourPawns & kCenter4) - std::popcount(theirPawns & kCenter4);
      features[EF::PASSED_PAWNS] = std::popcount(pawnAnalysis.ourPassedPawns) - std::popcount(pawnAnalysis.theirPassedPawns);
      features[EF::ISOLATED_PAWNS] = std::popcount(pawnAnalysis.ourIsolatedPawns) - std::popcount(pawnAnalysis.theirIsolatedPawns);
      features[EF::DOUBLED_PAWNS] = std::popcount(pawnAnalysis.ourDoubledPawns) - std::popcount(pawnAnalysis.theirDoubledPawns);
      features[EF::DOUBLE_ISOLATED_PAWNS] = std::popcount(pawnAnalysis.ourDoubledPawns & pawnAnalysis.ourIsolatedPawns) - std::popcount(pawnAnalysis.theirDoubledPawns & pawnAnalysis.theirIsolatedPawns);
      features[EF::ADVANCED_PAWNS_1] = std::popcount(ourPawns & kTheirBackRanks) - std::popcount(theirPawns & kOurBackRanks);
      features[EF::ADVANCED_PAWNS_2] = std::popcount(ourPawns & shift<kBackward>(kTheirBackRanks)) - std::popcount(theirPawns & shift<kForward>(kOurBackRanks));

      if (US == Color::WHITE) {
        features[EF::ADVANCED_PASSED_PAWNS_2] = std::popcount(pawnAnalysis.ourPassedPawns & kRanks[1]) * 2 - std::popcount(pawnAnalysis.theirPassedPawns & kRanks[6]);
        features[EF::ADVANCED_PASSED_PAWNS_3] = std::popcount(pawnAnalysis.ourPassedPawns & kRanks[2]) * 2 - std::popcount(pawnAnalysis.theirPassedPawns & kRanks[5]);
        features[EF::ADVANCED_PASSED_PAWNS_4] = std::popcount(pawnAnalysis.ourPassedPawns & kRanks[3]) * 2 - std::popcount(pawnAnalysis.theirPassedPawns & kRanks[4]);
      } else {
        features[EF::ADVANCED_PASSED_PAWNS_2] = std::popcount(pawnAnalysis.ourPassedPawns & kRanks[6]) * 2 - std::popcount(pawnAnalysis.theirPassedPawns & kRanks[1]);
        features[EF::ADVANCED_PASSED_PAWNS_3] = std::popcount(pawnAnalysis.ourPassedPawns & kRanks[5]) * 2 - std::popcount(pawnAnalysis.theirPassedPawns & kRanks[2]);
        features[EF::ADVANCED_PASSED_PAWNS_4] = std::popcount(pawnAnalysis.ourPassedPawns & kRanks[4]) * 2 - std::popcount(pawnAnalysis.theirPassedPawns & kRanks[3]);
      }

      features[EF::PAWN_MINOR_CAPTURES] = std::popcount(threats.ourPawnTargets & theirMinors) - std::popcount(threats.theirPawnTargets & ourMinors);
      features[EF::PAWN_MAJOR_CAPTURES] = std::popcount(threats.ourPawnTargets & theirMajors) - std::popcount(threats.theirPawnTargets & ourMajors);
      features[EF::PROTECTED_PAWNS] = std::popcount(ourPawns & threats.ourPawnTargets) - std::popcount(theirPawns & threats.theirPawnTargets);
      features[EF::PROTECTED_PASSED_PAWNS] = std::popcount(pawnAnalysis.ourPassedPawns & threats.ourPawnTargets) - std::popcount(pawnAnalysis.theirPassedPawns & threats.theirPawnTargets);

      features[EF::NUM_PIECES_HARRASSABLE_BY_PAWNS] = std::popcount(pawnAnalysis.piecesOurPawnsCanThreaten) - std::popcount(pawnAnalysis.piecesTheirPawnsCanThreaten);

      Bitboard whitePromoSquares = shift<Direction::NORTH>(pos.pieceBitboards_[ColoredPiece::WHITE_PAWN] & kRanks[1]);
      Bitboard blackPromoSquares = shift<Direction::SOUTH>(pos.pieceBitboards_[ColoredPiece::BLACK_PAWN] & kRanks[6]);
      if (US == Color::WHITE) {
        features[EF::PROMOTABLE_PAWN] =
        std::popcount(whitePromoSquares & ~threats.badForOur[Piece::QUEEN] & ~everyone)
        - std::popcount(blackPromoSquares & ~threats.badForTheir[Piece::QUEEN] & ~everyone);
      } else {
        features[EF::PROMOTABLE_PAWN] =
        std::popcount(blackPromoSquares & ~threats.badForOur[Piece::QUEEN & ~everyone])
        - std::popcount(whitePromoSquares & ~threats.badForTheir[Piece::QUEEN] & ~everyone);
      }
    }

    const Bitboard ourBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets(ourBishops, pawnAnalysis.ourBlockadedPawns);
    const Bitboard theirBishopTargetsIgnoringNonBlockades = compute_bishoplike_targets(theirBishops, pawnAnalysis.theirBlockadedPawns);
    {  // Bishops
      if (US == Color::WHITE) {
        features[EF::BISHOPS_DEVELOPED] = std::popcount(theirBishops & (bb( 2) | bb( 5))) - std::popcount(ourBishops & (bb(58) | bb(61)));
      } else {
        features[EF::BISHOPS_DEVELOPED] = std::popcount(theirBishops & (bb(58) | bb(61))) - std::popcount(ourBishops & (bb( 2) | bb( 5)));
      }
      features[EF::BISHOP_PAIR] = (std::popcount(ourBishops) >= 2) - (std::popcount(theirBishops) >= 2);
      features[EF::BLOCKADED_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & (pawnAnalysis.ourBlockadedPawns | pawnAnalysis.theirProtectedPawns)) - std::popcount(theirBishopTargetsIgnoringNonBlockades & (pawnAnalysis.theirBlockadedPawns | pawnAnalysis.ourProtectedPawns));
      features[EF::SCARY_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & theirMajors) - std::popcount(theirBishopTargetsIgnoringNonBlockades & ourMajors);
      features[EF::SCARIER_BISHOPS] = std::popcount(ourBishopTargetsIgnoringNonBlockades & theirRoyalty) - std::popcount(theirBishopTargetsIgnoringNonBlockades & ourRoyalty);
      features[EF::OUTPOSTED_BISHOPS] = std::popcount(ourBishops & pawnAnalysis.possibleOutpostsForUs) - std::popcount(theirBishops & pawnAnalysis.possibleOutpostsForThem);
    }

    {  // Rooks
      const Bitboard openFiles = pawnAnalysis.filesWithoutOurPawns & pawnAnalysis.filesWithoutTheirPawns;
      features[EF::BLOCKADED_ROOKS] = std::popcount(ourRooks & pawnAnalysis.filesWithoutOurPawns) - std::popcount(theirRooks & pawnAnalysis.filesWithoutTheirPawns);
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
      features[EF::OUTPOSTED_KNIGHTS] = std::popcount(ourKnights & pawnAnalysis.possibleOutpostsForUs & kTheirSide) - std::popcount(theirKnights & pawnAnalysis.possibleOutpostsForThem & kOurSide);

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

      bool anyOfUsHanging = false;
      bool anyOfThemHanging = false;

      features[EF::THEIR_HANGING_QUEENS] = std::popcount(theirQueens & themHanging);
      features[EF::OUR_HANGING_QUEENS] = std::popcount(ourQueens & usHanging);
      anyOfThemHanging |= features[EF::THEIR_HANGING_QUEENS];
      anyOfUsHanging |= features[EF::OUR_HANGING_QUEENS];

      features[EF::THEIR_HANGING_ROOKS] = std::popcount(theirRooks & themHanging) && !anyOfThemHanging;
      features[EF::OUR_HANGING_ROOKS] = std::popcount(ourRooks & usHanging) && !anyOfUsHanging;
      anyOfThemHanging |= features[EF::THEIR_HANGING_ROOKS];
      anyOfUsHanging |= features[EF::OUR_HANGING_ROOKS];

      features[EF::THEIR_HANGING_BISHOPS] = std::popcount(theirBishops & themHanging) && !anyOfThemHanging;
      features[EF::OUR_HANGING_BISHOPS] = std::popcount(ourBishops & usHanging) && !anyOfUsHanging;
      anyOfThemHanging |= features[EF::THEIR_HANGING_BISHOPS];
      anyOfUsHanging |= features[EF::OUR_HANGING_BISHOPS];

      features[EF::THEIR_HANGING_KNIGHTS] = std::popcount(theirKnights & themHanging) && !anyOfThemHanging;
      features[EF::OUR_HANGING_KNIGHTS] = std::popcount(ourKnights & usHanging) && !anyOfUsHanging;
      anyOfThemHanging |= features[EF::THEIR_HANGING_KNIGHTS];
      anyOfUsHanging |= features[EF::OUR_HANGING_KNIGHTS];

      features[EF::THEIR_HANGING_PAWNS] = std::popcount(theirPawns & themHanging) && !anyOfThemHanging;
      features[EF::OUR_HANGING_PAWNS] = std::popcount(ourPawns & usHanging) && !anyOfUsHanging;

      features[EF::IN_CHECK_AND_OUR_HANGING_QUEENS] = features[EF::OUR_HANGING_QUEENS] * features[EF::IN_CHECK];

      const int wx = ourKingSq % 8;
      const int wy = ourKingSq / 8;
      const int bx = theirKingSq % 8;
      const int by = theirKingSq / 8;
      const int kingsDist = std::max(std::abs(wx - bx), std::abs(wy - by));
      features[EF::LONELY_KING_IN_CENTER] = value_or_zero(std::popcount(pos.colorBitboards_[THEM]) == 1, 3 - kDistToCorner[theirKingSq]);
      features[EF::LONELY_KING_IN_CENTER] -= value_or_zero(std::popcount(pos.colorBitboards_[US]) == 1, 3 - kDistToCorner[ourKingSq]);

      features[EF::LONELY_KING_AWAY_FROM_ENEMY_KING] = value_or_zero(isTheirKingLonely, 8 - kingsDist);
      features[EF::LONELY_KING_AWAY_FROM_ENEMY_KING] -= value_or_zero(isOurKingLonely, 8 - kingsDist);
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

    features[EF::SQUARE_RULE] = 0;
    features[EF::SQUARE_RULE] += value_or_zero(isKingPawnEndgame, (kSquareRuleTheirTurn[US][theirKingSq] & pawnAnalysis.ourPassedPawns) > 0);
    features[EF::SQUARE_RULE] -= value_or_zero(isKingPawnEndgame, (kSquareRuleYourTurn[THEM][ourKingSq] & pawnAnalysis.theirPassedPawns) > 0);

    Bitboard aheadOfOurPassedPawnsFat, aheadOfTheirPassedPawnsFat;
    if (US == Color::WHITE) {
      aheadOfOurPassedPawnsFat = fatten(northFill(pawnAnalysis.ourPassedPawns));
      aheadOfTheirPassedPawnsFat = fatten(southFill(pawnAnalysis.theirPassedPawns));
    } else {
      aheadOfOurPassedPawnsFat = fatten(southFill(pawnAnalysis.ourPassedPawns));
      aheadOfTheirPassedPawnsFat = fatten(northFill(pawnAnalysis.theirPassedPawns));
    }

    // We split these into two features, the idea being that being ahead of your pawns while your opponent's
    // queen is on the board is dangerous, but being ahead of your opponent's passed pawns is not.
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN] = ((ourKings & aheadOfOurPassedPawnsFat) > 0 && theirQueens == 0);
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN] -= ((theirKings & aheadOfTheirPassedPawnsFat) > 0 && ourQueens == 0);
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN2] = (ourKings & aheadOfTheirPassedPawnsFat) > 0;
    features[EF::KING_IN_FRONT_OF_PASSED_PAWN2] -= (theirKings & aheadOfOurPassedPawnsFat) > 0;

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
      const Bitboard ourGoodKnightTargets = threats.ourKnightTargets & ~threats.badForOur[Piece::KNIGHT];
      const Bitboard theirGoodKnightTargets = threats.theirKnightTargets & ~threats.badForTheir[Piece::KNIGHT];
      const Bitboard ourGoodBishopTargets = ourBishopTargetsIgnoringNonBlockades & ~threats.badForOur[Piece::BISHOP];
      const Bitboard theirGoodBishopTargets = theirBishopTargetsIgnoringNonBlockades & ~threats.badForTheir[Piece::BISHOP];
      const Bitboard ourGoodRookTargets = threats.ourRookTargets & ~threats.badForOur[Piece::ROOK];
      const Bitboard theirGoodRookTargets = threats.theirRookTargets & ~threats.badForTheir[Piece::ROOK];
      const Bitboard ourGoodQueenTargets = threats.ourQueenTargets & ~threats.badForOur[Piece::QUEEN];
      const Bitboard theirGoodQueenTargets = threats.theirQueenTargets & ~threats.badForTheir[Piece::QUEEN];
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
      value_or_zero((cr & kCastlingRights_BlackKing) > 0, std::popcount(kKingHome[Square::G8] & theirPawns)),
      value_or_zero((cr & kCastlingRights_BlackQueen) > 0, std::popcount(kKingHome[Square::B8] & theirPawns))
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

    // Checks
    // TODO: check map for our king too.
    const CheckMap checkMap = compute_potential_attackers<US>(pos, theirKingSq);
    {
      const Bitboard forwardPawns = shift<kForward>(ourPawns) & ~everyone;
      const Bitboard thirdRank = (US == Color::WHITE ? kRanks[5] : kRanks[2]);
      const Bitboard pawnCaptureChecks = checkMap.data[Piece::PAWN] & threats.ourPawnTargets & theirMen;
      Bitboard pawnPushChecks = checkMap.data[Piece::PAWN] & (forwardPawns | (shift<kForward>(forwardPawns & thirdRank) & ~everyone));
      const Bitboard knightChecks = checkMap.data[Piece::KNIGHT] & threats.ourKnightTargets;
      const Bitboard bishopChecks = checkMap.data[Piece::BISHOP] & threats.ourBishopTargets;
      const Bitboard rookChecks = checkMap.data[Piece::ROOK] & threats.ourRookTargets;
      const Bitboard queenChecks = checkMap.data[Piece::QUEEN] & threats.ourQueenTargets;

      // NOTE: we can't use badForOur[Piece::BISHOP] (or any piece) since the piece delivering the check
      // stops protecting the square when it moves.
      Bitboard safe = ~threats.theirTargets;
      safe |= (threats.ourTargets & ~threats.theirTargets) | (threats.ourDoubleTargets & ~threats.theirDoubleTargets);

      // NOTE: we *can* use threats.badForOur[Piece::PAWN], since it doesn't count the pawn's forward
      // push when determining square safety.

      features[EF::PAWN_CHECKS] = std::popcount((pawnPushChecks & ~threats.badForOur[Piece::PAWN]) | (pawnCaptureChecks & safe));
      features[EF::KNIGHT_CHECKS] = std::popcount(knightChecks & safe);
      features[EF::BISHOP_CHECKS] = std::popcount(bishopChecks & safe);
      features[EF::ROOK_CHECKS] = std::popcount(rookChecks & safe);
      features[EF::QUEEN_CHECKS] = std::popcount(queenChecks & safe);
    }

    {
      const Bitboard whitePawns = (US == Color::WHITE ? ourPawns : theirPawns);
      const Bitboard blackPawns = (US == Color::BLACK ? ourPawns : theirPawns);
      const Bitboard whiteKingCannotMoveTo = threats.template badFor<ColoredPiece::WHITE_KING>();
      const Bitboard blackKingCannotMoveTo = threats.template badFor<ColoredPiece::BLACK_KING>();
      const Bitboard whiteKingEscapes = compute_king_targets<Color::WHITE>(pos, whiteKingSq) & ~(whiteKingCannotMoveTo | whitePawns);
      const Bitboard blackKingEscapes = compute_king_targets<Color::BLACK>(pos, blackKingSq) & ~(blackKingCannotMoveTo | blackPawns);

      const bool backRankMateThreatAgainstWhite = (whiteKingSq >= 56 && ((whiteKingEscapes & kRanks[7]) == whiteKingEscapes) && ((threats.template targets<ColoredPiece::BLACK_ROOK>() & kRanks[7]) > 0));
      const bool backRankMateThreatAgainstBlack = (blackKingSq <=  7 && ((blackKingEscapes & kRanks[0]) == blackKingEscapes) && ((threats.template targets<ColoredPiece::WHITE_ROOK>() & kRanks[0]) > 0));

      if (US == Color::WHITE) {
        features[EF::BACK_RANK_MATE_THREAT_AGAINST_US] = backRankMateThreatAgainstWhite;
        features[EF::BACK_RANK_MATE_THREAT_AGAINST_THEM] = backRankMateThreatAgainstBlack;
      } else {
        features[EF::BACK_RANK_MATE_THREAT_AGAINST_US] = backRankMateThreatAgainstBlack;
        features[EF::BACK_RANK_MATE_THREAT_AGAINST_THEM] = backRankMateThreatAgainstWhite;
      }

      const Bitboard whitePieces = (US == Color::WHITE ? ourPieces : theirPieces);
      const Bitboard blackPieces = (US == Color::BLACK ? ourPieces : theirPieces);
      const int ourKingEscapes = std::popcount(US == Color::WHITE ? (whiteKingEscapes & ~whitePieces) : (blackKingEscapes & ~blackPieces));
      const int theirKingEscapes = std::popcount(US == Color::BLACK ? (whiteKingEscapes & ~whitePieces) : (blackKingEscapes & ~blackPieces));

      // TODO: probably should remove this
      features[EF::OUR_KING_HAS_0_ESCAPE_SQUARES] = (ourKingEscapes == 0);
      features[EF::THEIR_KING_HAS_0_ESCAPE_SQUARES] = (theirKingEscapes == 0);
      features[EF::OUR_KING_HAS_1_ESCAPE_SQUARES] = (ourKingEscapes == 1);
      features[EF::THEIR_KING_HAS_1_ESCAPE_SQUARES] = (theirKingEscapes == 1);
      features[EF::OUR_KING_HAS_2_ESCAPE_SQUARES] = (ourKingEscapes == 2);
      features[EF::THEIR_KING_HAS_2_ESCAPE_SQUARES] = (theirKingEscapes == 2);
    }

    const int kingsOnOppositesSideOfBoard = std::min(std::abs(whiteKingSq % 8 - blackKingSq % 8), 6);
    Bitboard aheadOfOurKing = fatten(ourKings);
    Bitboard aheadOfTheirKing = fatten(theirKings);
    if (US == Color::WHITE) {
      aheadOfOurKing >>= 8;
      aheadOfOurKing |= aheadOfOurKing >> 8;
      aheadOfOurKing |= aheadOfOurKing >> 16;
      aheadOfTheirKing <<= 8;
      aheadOfTheirKing |= aheadOfTheirKing << 8;
      aheadOfTheirKing |= aheadOfTheirKing << 16;
    } else {
      aheadOfOurKing <<= 8;
      aheadOfOurKing |= aheadOfOurKing << 8;
      aheadOfOurKing |= aheadOfOurKing << 16;
      aheadOfTheirKing >>= 8;
      aheadOfTheirKing |= aheadOfTheirKing >> 8;
      aheadOfTheirKing |= aheadOfTheirKing >> 16;
    }
    features[EF::OPPOSITE_SIDE_KINGS_PAWN_STORM] = kingsOnOppositesSideOfBoard * (std::popcount(aheadOfTheirKing & ourPawns) - std::popcount(aheadOfOurKing & theirPawns));

    features[EF::TIME] = time;

    // Use larger integer to make arithmetic safe.
    const int32_t early = this->early<US>(pos);
    const int32_t late = this->late<US>(pos);
    const int32_t clipped = this->clipped<US>(pos);

    // 0.043 ± 0.019
    // TODO: we'd like to learn piece maps based on king positions (king-side vs queen-side) in addition to time.
    // 4 maps total:
    // 1) early, white king side
    // 2) early, black king side
    // 1)  late, white king side
    // 2)  late, black king side
    int32_t pieceMap = (pos.pieceMapScores[PieceMapType::PieceMapTypeEarly] * (18 - time) + pos.pieceMapScores[PieceMapType::PieceMapTypeLate] * time) / 18;
    if (US == Color::BLACK) {
      pieceMap *= -1;
    }

    int32_t eval = (early * (18 - time) + late * time) / 18 + clipped + pieceMap;

    // Special end-game boosts.

    Evaluation bonus = 0;

    const int wx = ourKingSq % 8;
    const int wy = ourKingSq / 8;
    const int bx = theirKingSq % 8;
    const int by = theirKingSq / 8;

    {
      // KPVK games are winning if square rule is true.
      const bool isOurKPPVK = isKingPawnEndgame && (std::popcount(ourPawns) >= 1) && (theirPawns == 0);
      const bool isTheirKPPVK = isKingPawnEndgame && (std::popcount(theirPawns) >= 1) && (ourPawns == 0);
      bonus -= value_or_zero(isOurKPPVK && features[EF::SQUARE_RULE] < 0, 500);
      bonus += value_or_zero(isTheirKPPVK && features[EF::SQUARE_RULE] > 0, 500);

      if (isOurKPPVK && std::popcount(ourPawns) >= 1) {
        int result;
        if (US == Color::WHITE) {
          result = known_kpvk_result(ourKingSq, theirKingSq, lsb(ourPawns), true);
        } else {
          result = known_kpvk_result(Square(63 - ourKingSq), Square(63 - theirKingSq), Square(63 - lsb(ourPawns)), true);
        }
        if (result == 0) {
          return 0;
        }
        if (result == 2) {
          bonus += 1000;
        }
      }
      if (isTheirKPPVK && std::popcount(theirPawns) <= 1) {
        int result;
        if (US == Color::BLACK) {
          result = known_kpvk_result(theirKingSq, ourKingSq, lsb(theirPawns), false);
        } else {
          result = known_kpvk_result(Square(63 - theirKingSq), Square(63 - ourKingSq), Square(63 - lsb(theirPawns)), false);
        }
        if (result == 0) {
          return 0;
        }
        if (result == 2) {
          bonus -= 1000;
        }
      }
    }
    {  // KRvK, KQvK, KBBvK, KBNvK
      const bool theyHaveLonelyKing = (theirMen == theirKings) && (std::popcount(ourKnights) > 2 || std::popcount(ourBishops) > 1 || std::popcount(ourRooks) > 0 || std::popcount(ourQueens) > 0);
      const bool weHaveLonelyKing = (ourMen == ourKings) && (std::popcount(theirKnights) > 2 || std::popcount(theirBishops) > 1 || std::popcount(theirRooks) > 0 || std::popcount(theirQueens) > 0);

      bonus += value_or_zero(theyHaveLonelyKing, (3 - kDistToEdge[theirKingSq]) * 50);
      bonus -= value_or_zero(  weHaveLonelyKing, (3 - kDistToEdge[ourKingSq]) * 50);
      bonus += value_or_zero(theyHaveLonelyKing, (3 - kDistToCorner[theirKingSq]) * 50);
      bonus -= value_or_zero(  weHaveLonelyKing, (3 - kDistToCorner[ourKingSq]) * 50);

      int dx = std::abs(wx - bx);
      int dy = std::abs(wy - by);
      const bool opposition = (dx == 2 && dy == 0) || (dx == 0 && dy == 2);

      // We don't want it to be our turn!
      bonus -= value_or_zero(weHaveLonelyKing && opposition, 75);
      // We can achieve opposition from here.
      bonus += value_or_zero(theyHaveLonelyKing && (
        (dx == 3 && dy <= 1)
        || (dy == 3 && dx <= 1)
      ), 50);

      // And put our king next to the enemy king.
      bonus += value_or_zero(theyHaveLonelyKing, (8 - std::max(dx, dy)) * 50);
      bonus -= value_or_zero(  weHaveLonelyKing, (8 - std::max(dx, dy)) * 50);
      bonus += value_or_zero(theyHaveLonelyKing, (8 - std::min(dx, dy)) * 25);
      bonus -= value_or_zero(  weHaveLonelyKing, (8 - std::min(dx, dy)) * 25);
    }

    // TODO: decompose "bonus" into separate features.

    // TODO: bonus for controlling squares ahead of your own pawns

    eval += bonus;

    eval = std::min(int32_t(-kQLongestForcedMate), std::max(int32_t(kQLongestForcedMate), eval));

    #ifdef PRINT_LEAVES
      if (rand() % 10000 == 0) {
        std::string t = "";
        t += pos.fen() + "\n";
        t += std::to_string(features[0]);
        for (size_t i = 1; i < EF::NUM_EVAL_FEATURES; ++i) {
          t += " " + std::to_string(features[i]);
        }
        std::cout << t << std::endl;
      }
    #endif

    return eval;

    #else  // #ifndef SquareControl

    int32_t pieceMap = (pos.pieceMapScores[PieceMapType::PieceMapTypeEarly] * (18 - time) + pos.pieceMapScores[PieceMapType::PieceMapTypeLate] * time) / 18;
    if (US == Color::BLACK) {
      pieceMap *= -1;
    }

    int centerControl = 0;
    centerControl += std::popcount(threats.badForTheir[Piece::PAWN] & kCenter16) - std::popcount(threats.badForOur[Piece::PAWN] & kCenter16);
    centerControl += std::popcount((threats.badForTheir[Piece::BISHOP] | threats.badForTheir[Piece::KNIGHT]) & kCenter16)
    - std::popcount((threats.badForOur[Piece::KNIGHT] | threats.badForOur[Piece::BISHOP]) & kCenter16);
    centerControl += std::popcount(threats.badForTheir[Piece::ROOK] & kCenter16) - std::popcount(threats.badForOur[Piece::ROOK] & kCenter16);
    centerControl += std::popcount(threats.badForTheir[Piece::QUEEN] & kCenter16) - std::popcount(threats.badForOur[Piece::QUEEN] & kCenter16);

    int domination = 0;
    domination += std::popcount(threats.badForTheir[Piece::PAWN] & kTheirSide) - std::popcount(threats.badForOur[Piece::PAWN] & kOurSide);
    domination += std::popcount((threats.badForTheir[Piece::BISHOP] | threats.badForTheir[Piece::KNIGHT]) & kTheirSide)
    - std::popcount((threats.badForOur[Piece::KNIGHT] | threats.badForOur[Piece::BISHOP]) & kOurSide);
    domination += std::popcount(threats.badForTheir[Piece::ROOK] & kTheirSide) - std::popcount(threats.badForOur[Piece::ROOK] & kOurSide);
    domination += std::popcount(threats.badForTheir[Piece::QUEEN] & kTheirSide) - std::popcount(threats.badForOur[Piece::QUEEN] & kOurSide);

    int backRankControl = 0;
    backRankControl += std::popcount(threats.badForTheir[Piece::PAWN] & kTheirBackRanks) - std::popcount(threats.badForOur[Piece::PAWN] & kOurBackRanks);
    backRankControl += std::popcount((threats.badForTheir[Piece::BISHOP] | threats.badForTheir[Piece::KNIGHT]) & kTheirBackRanks)
    - std::popcount((threats.badForOur[Piece::KNIGHT] | threats.badForOur[Piece::BISHOP]) & kOurBackRanks);
    backRankControl += std::popcount(threats.badForTheir[Piece::ROOK] & kTheirBackRanks) - std::popcount(threats.badForOur[Piece::ROOK] & kOurBackRanks);
    backRankControl += std::popcount(threats.badForTheir[Piece::QUEEN] & kTheirBackRanks) - std::popcount(threats.badForOur[Piece::QUEEN] & kOurBackRanks);

    int freedom = 0;
    freedom += std::popcount(threats.ourPawnTargets & ~threats.badForOur[Piece::PAWN]) - std::popcount(threats.theirPawnTargets & ~threats.badForTheir[Piece::PAWN]);
    freedom += std::popcount(threats.ourKnightTargets & ~threats.badForOur[Piece::KNIGHT]) - std::popcount(threats.theirKnightTargets & ~threats.badForTheir[Piece::KNIGHT]);
    freedom += std::popcount(threats.ourBishopTargets & ~threats.badForOur[Piece::BISHOP]) - std::popcount(threats.theirBishopTargets & ~threats.badForTheir[Piece::BISHOP]);
    freedom += std::popcount(threats.ourRookTargets & ~threats.badForOur[Piece::ROOK]) - std::popcount(threats.theirRookTargets & ~threats.badForTheir[Piece::ROOK]);
    freedom += std::popcount(threats.ourQueenTargets & ~threats.badForOur[Piece::QUEEN]) - std::popcount(threats.theirQueenTargets & ~threats.badForTheir[Piece::QUEEN]);

    int homeQuality;
    {
      // Bonus for king having pawns in front of him, or having pawns in front of him once he castles.
      const CastlingRights cr = pos.currentState_.castlingRights;
      homeQuality = std::popcount(kKingHome[ourKingSq] & ourPawns) - std::popcount(kKingHome[theirKingSq] & theirPawns);
      const Evaluation whitePotentialHome = std::max<Evaluation>(
        value_or_zero((cr & kCastlingRights_WhiteKing) > 0, std::popcount(kKingHome[Square::G1] & pos.pieceBitboards_[ColoredPiece::WHITE_PAWN])),
        value_or_zero((cr & kCastlingRights_WhiteQueen) > 0, std::popcount(kKingHome[Square::B1] & pos.pieceBitboards_[ColoredPiece::WHITE_PAWN]))
      );
      const Evaluation blackPotentialHome = std::max<Evaluation>(
        value_or_zero((cr & kCastlingRights_BlackKing) > 0, std::popcount(kKingHome[Square::G8] & pos.pieceBitboards_[ColoredPiece::BLACK_PAWN])),
        value_or_zero((cr & kCastlingRights_BlackQueen) > 0, std::popcount(kKingHome[Square::B8] & pos.pieceBitboards_[ColoredPiece::BLACK_PAWN]))
      );
      if (US == Color::WHITE) {
        homeQuality += whitePotentialHome - blackPotentialHome;
      } else {
        homeQuality += blackPotentialHome - whitePotentialHome;
      }
    }

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

    // Our pawns, which are not defended by our other pawns.
    const Bitboard aheadOfOurPawns = (US == Color::WHITE ? northFill(shift<Direction::NORTH>(ourPawns)) : southFill(shift<Direction::SOUTH>(ourPawns)));
    const Bitboard aheadOfTheirPawns = (US == Color::BLACK ? northFill(shift<Direction::NORTH>(theirPawns)) : southFill(shift<Direction::SOUTH>(theirPawns)));
    const Bitboard ourPassedPawns = ourPawns & ~fatten(aheadOfTheirPawns);
    const Bitboard theirPassedPawns = theirPawns & ~fatten(aheadOfOurPawns);

    const Bitboard filesWithoutOurPawns = ~(US == Color::WHITE ? southFill(aheadOfOurPawns) : northFill(aheadOfOurPawns));
    const Bitboard filesWithoutTheirPawns = ~(US == Color::BLACK ? northFill(aheadOfTheirPawns) : southFill(aheadOfTheirPawns));

    const Bitboard ourIsolatedPawns = ourPawns & (shift<Direction::WEST>(filesWithoutOurPawns) & shift<Direction::EAST>(filesWithoutOurPawns));
    const Bitboard theirIsolatedPawns = theirPawns & (shift<Direction::WEST>(filesWithoutTheirPawns) & shift<Direction::EAST>(filesWithoutTheirPawns));

    // 0: no passed pawn
    // 6: passed pawn on the 7th rank
    Square ourClosestPassedPawnSq = lsb_or(ourPassedPawns, Square::NO_SQUARE);
    Square theirClosestPassedPawnSq = lsb_or(theirPassedPawns, Square::NO_SQUARE);
    const int progressOfOurClosestPassedPawn = value_or_zero(ourClosestPassedPawnSq != Square::NO_SQUARE, (US == Color::WHITE ? 63 - ourClosestPassedPawnSq : ourClosestPassedPawnSq) / 8);
    const int progressOfTheirClosestPassedPawn = value_or_zero(theirClosestPassedPawnSq != Square::NO_SQUARE, (US == Color::BLACK ? 63 - theirClosestPassedPawnSq : theirClosestPassedPawnSq) / 8);

    const Square aheadOfClosestPassedPawnSq = select(
      progressOfOurClosestPassedPawn > progressOfTheirClosestPassedPawn,
      Square(US == Color::WHITE ? ourClosestPassedPawnSq - 8 : ourClosestPassedPawnSq + 8),
      Square(US == Color::BLACK ? theirClosestPassedPawnSq - 8 : theirClosestPassedPawnSq + 8)
    );
    const int ourDistToClosestPassedPawn = select<int>(
      aheadOfClosestPassedPawnSq != Square::NO_SQUARE,
      king_dist(aheadOfClosestPassedPawnSq, ourKingSq),
      7
    );
    const int theirDistToClosestPassedPawn = select<int>(
      aheadOfClosestPassedPawnSq != Square::NO_SQUARE,
      king_dist(aheadOfClosestPassedPawnSq, theirKingSq),
      7
    );

    /*
    Let's create a new evaluator with a new philosophy: square control. The idea is that
    piece maps are largely thinking about square-control, but that square control is a
    much more reliable feature (e.g. you don't want to fianchetto your bishop onto a
    blocked diagonal).

    Early game:
      1) in the center
      2) on your opponent's side (with pref. to back 2 ranks)
      3) by opponent's king
    Late game:
      1) promotion squares
      2) squares in front of pawns, in general
    All:
      1) squares directly in front of pawns

    We'd also like to account for long-term control (i.e. "weaknesses"). "Losing your fianchettoed bishop
    is bad" is a special case of the general idea of "this is an important square, and I'll always struggle
    to control it".

    Other factors:
      1) identify weak pawns
      2) identify harrassable pieces
      3) pins
      4) forks
      5) skewers
      6) discovered attacks
      7) overworked
    */

    int32_t early = 0;
    early += centerControl * 3;
    early += domination * 3;
    early += backRankControl * 3;
    early += homeQuality * 3;

    int32_t base = 0;
    base += (features[EF::OUR_PAWNS] - features[EF::THEIR_PAWNS]) * 90;
    base += (features[EF::OUR_KNIGHTS] - features[EF::THEIR_KNIGHTS]) * 300;
    base += (features[EF::OUR_BISHOPS] - features[EF::THEIR_BISHOPS]) * 300;
    base += (features[EF::OUR_ROOKS] - features[EF::THEIR_ROOKS]) * 400;
    base += (features[EF::OUR_QUEENS] - features[EF::THEIR_QUEENS]) * 900;
    base += (features[EF::THEIR_HANGING_QUEENS] > 0) * 450;
    base += (features[EF::THEIR_HANGING_QUEENS] == 0 && features[EF::THEIR_HANGING_ROOKS] > 0) * 250;
    base += (features[EF::THEIR_HANGING_QUEENS] == 0 && features[EF::THEIR_HANGING_ROOKS] == 0 && features[EF::THEIR_HANGING_KNIGHTS] + features[EF::THEIR_HANGING_BISHOPS] > 0) * 150;
    base += freedom * 3;

    int32_t late = 0;
    late += (progressOfOurClosestPassedPawn - progressOfTheirClosestPassedPawn) * 20;
    late += (theirDistToClosestPassedPawn - ourDistToClosestPassedPawn) * 25;

    early = (early * (18 - time)) / 18;
    late = (late * time) / 18;

    Evaluation special = this->special_boosts<US>(pos);
    if (special == kKnownDraw) {
      return 0;
    }

    return base + pieceMap + early + late + special;
    #endif  // SquareControl
#endif  // NO_NNUE_EVAL
  }

  template<Color US>
  Evaluation special_boosts(const Position& pos) {
    constexpr Color THEM = opposite_color<US>();
    constexpr Evaluation kKnownWinBonus = 1000;

    const Square ourKingSq = lsb(pos.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
    const Square theirKingSq = lsb(pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]);

    const Bitboard ourPawns = pos.pieceBitboards_[coloredPiece<US, Piece::PAWN>()];
    const Bitboard ourKings = pos.pieceBitboards_[coloredPiece<US, Piece::KING>()];

    const Bitboard theirPawns = pos.pieceBitboards_[coloredPiece<THEM, Piece::PAWN>()];
    const Bitboard theirKings = pos.pieceBitboards_[coloredPiece<THEM, Piece::KING>()];

    const Square whiteKingSq = (US == Color::WHITE ? ourKingSq : theirKingSq);
    const Square blackKingSq = (US == Color::BLACK ? ourKingSq : theirKingSq);

    const Bitboard ourMen = pos.colorBitboards_[US];
    const Bitboard theirMen = pos.colorBitboards_[THEM];
    const Bitboard everyone = ourMen | theirMen;

    const bool doWeOnlyHavePawnsLeft = (ourMen & ~(ourPawns | ourKings)) == 0;
    const bool doTheyOnlyHavePawnsLeft = (theirMen & ~(theirPawns | theirKings)) == 0;
    const bool isKingPawnEndgame = doWeOnlyHavePawnsLeft && doTheyOnlyHavePawnsLeft;

    if (isKingPawnEndgame && (std::popcount(ourPawns) == 1) && (theirPawns == 0)) {
      int result;
      if (US == Color::WHITE) {
        result = known_kpvk_result(ourKingSq, theirKingSq, lsb(ourPawns), true);
      } else {
        result = known_kpvk_result(Square(63 - ourKingSq), Square(63 - theirKingSq), Square(63 - lsb(ourPawns)), true);
      }
      if (result == 0) {
        return kKnownDraw;
      }
      if (result == 2) {
        return 1000;
      }
    }
    else if (isKingPawnEndgame && (std::popcount(theirPawns) == 1) && (ourPawns == 0)) {
      int result;
      if (US == Color::BLACK) {
        result = known_kpvk_result(theirKingSq, ourKingSq, lsb(theirPawns), false);
      } else {
        result = known_kpvk_result(Square(63 - theirKingSq), Square(63 - ourKingSq), Square(63 - lsb(theirPawns)), false);
      }
      if (result == 0) {
        return kKnownDraw;
      }
      if (result == 2) {
        return -1000;
      }
    }

    const int wx = ourKingSq % 8;
    const int wy = ourKingSq / 8;
    const int bx = theirKingSq % 8;
    const int by = theirKingSq / 8;

    Evaluation r = 0;

    {
      const bool theyHaveLonelyKing = (theirMen == theirKings);
      const bool weHaveLonelyKing = (ourMen == ourKings);
      const bool weHaveMaterialToMate = (
        std::popcount(pos.pieceBitboards_[coloredPiece<US, Piece::KNIGHT>()]) > 2
        || std::popcount(pos.pieceBitboards_[coloredPiece<US, Piece::BISHOP>()]) > 1
        || std::popcount(pos.pieceBitboards_[coloredPiece<US, Piece::ROOK>()]) > 0
        || std::popcount(pos.pieceBitboards_[coloredPiece<US, Piece::QUEEN>()]) > 0
      );
      const bool theyHaveMaterialToMate = (
        std::popcount(pos.pieceBitboards_[coloredPiece<THEM, Piece::KNIGHT>()]) > 2
        || std::popcount(pos.pieceBitboards_[coloredPiece<THEM, Piece::BISHOP>()]) > 1
        || std::popcount(pos.pieceBitboards_[coloredPiece<THEM, Piece::ROOK>()]) > 0
        || std::popcount(pos.pieceBitboards_[coloredPiece<THEM, Piece::QUEEN>()]) > 0
      );

      r += value_or_zero(theyHaveLonelyKing && weHaveMaterialToMate, kKnownWinBonus);
      r -= value_or_zero(weHaveLonelyKing && theyHaveMaterialToMate, kKnownWinBonus);

      r += value_or_zero(theyHaveLonelyKing, (3 - kDistToEdge[theirKingSq]) * 50);
      r -= value_or_zero(  weHaveLonelyKing, (3 - kDistToEdge[ourKingSq]) * 50);
      r += value_or_zero(theyHaveLonelyKing, (3 - kDistToCorner[theirKingSq]) * 50);
      r -= value_or_zero(  weHaveLonelyKing, (3 - kDistToCorner[ourKingSq]) * 50);

      int dx = std::abs(wx - bx);
      int dy = std::abs(wy - by);
      const bool opposition = (dx == 2 && dy == 0) || (dx == 0 && dy == 2);

      // We don't want it to be our turn if they have the opposition.
      r -= value_or_zero(weHaveLonelyKing && opposition, 75);

      // And put our king next to the enemy king.
      r += value_or_zero(theyHaveLonelyKing, (8 - std::max(dx, dy)) * 50);
      r -= value_or_zero(  weHaveLonelyKing, (8 - std::max(dx, dy)) * 50);
      r += value_or_zero(theyHaveLonelyKing, (8 - std::min(dx, dy)) * 25);
      r -= value_or_zero(  weHaveLonelyKing, (8 - std::min(dx, dy)) * 25);
    }

    return r;
  }

  template<Color US>
  int32_t early(const Position& pos) const {
    int32_t r = earlyB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * earlyW[i];
    }
    return r;
  }

  template<Color US>
  int32_t late(const Position& pos) const {
    int32_t r = lateB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * lateW[i];
    }

    r += features[EF::SQUARE_RULE] * 2000;

    return r;
  }

  template<Color US>
  int32_t clipped(const Position& pos) const {
    int32_t r = clippedB;
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      r += features[i] * clippedW[i];
    }
    return std::max(-100, std::min(100, r));
  }

  void zero_() {
    for (size_t i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
      earlyW[i] = 0;
      lateW[i] = 0;
      clippedW[i] = 0;
    }
    earlyB = 0;
    lateB = 0;
    clippedB = 0;
  }

  Evaluation features[NUM_EVAL_FEATURES];
};

}  // namespace ChessEngine

#endif  // EVALUATOR_H
