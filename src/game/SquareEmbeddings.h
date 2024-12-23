#ifndef SQUARE_EMBEDDINGS_H
#define SQUARE_EMBEDDINGS_H

#include "utils.h"
#include "geometry.h"
#include "Position.h"
#include "Evaluator.h"

#include "movegen/bishops.h"
#include "movegen/rooks.h"
#include "movegen/knights.h"
#include "movegen/kings.h"

namespace ChessEngine {

enum SquareEmbeddingFeatures {
  SEF_globalOUR_PAWNS,
  SEF_globalOUR_KNIGHTS,
  SEF_globalOUR_BISHOPS,
  SEF_globalOUR_ROOKS,
  SEF_globalOUR_QUEENS,
  SEF_globalTHEIR_PAWNS,
  SEF_globalTHEIR_KNIGHTS,
  SEF_globalTHEIR_BISHOPS,
  SEF_globalTHEIR_ROOKS,
  SEF_globalTHEIR_QUEENS,
  SEF_globalIN_CHECK,
  SEF_globalKING_ON_BACK_RANK,
  SEF_globalKING_ON_CENTER_FILE,
  SEF_globalKING_ACTIVE,
  SEF_globalTHREATS_NEAR_KING_2,
  SEF_globalTHREATS_NEAR_KING_3,
  SEF_globalPASSED_PAWNS,
  SEF_globalISOLATED_PAWNS,
  SEF_globalDOUBLED_PAWNS,
  SEF_globalDOUBLE_ISOLATED_PAWNS,
  SEF_globalPAWNS_CENTER_16,
  SEF_globalPAWNS_CENTER_4,
  SEF_globalADVANCED_PASSED_PAWNS_2,
  SEF_globalADVANCED_PASSED_PAWNS_3,
  SEF_globalADVANCED_PASSED_PAWNS_4,
  SEF_globalPAWN_MINOR_CAPTURES,
  SEF_globalPAWN_MAJOR_CAPTURES,
  SEF_globalPROTECTED_PAWNS,
  SEF_globalPROTECTED_PASSED_PAWNS,
  SEF_globalBISHOPS_DEVELOPED,
  SEF_globalBISHOP_PAIR,
  SEF_globalBLOCKADED_BISHOPS,
  SEF_globalSCARY_BISHOPS,
  SEF_globalSCARIER_BISHOPS,
  SEF_globalBLOCKADED_ROOKS,
  SEF_globalSCARY_ROOKS,
  SEF_globalINFILTRATING_ROOKS,
  SEF_globalKNIGHTS_DEVELOPED,
  SEF_globalKNIGHT_MAJOR_CAPTURES,
  SEF_globalKNIGHTS_CENTER_16,
  SEF_globalKNIGHTS_CENTER_4,
  SEF_globalKNIGHT_ON_ENEMY_SIDE,
  SEF_globalOUR_HANGING_PAWNS,
  SEF_globalOUR_HANGING_KNIGHTS,
  SEF_globalOUR_HANGING_BISHOPS,
  SEF_globalOUR_HANGING_ROOKS,
  SEF_globalOUR_HANGING_QUEENS,
  SEF_globalTHEIR_HANGING_PAWNS,
  SEF_globalTHEIR_HANGING_KNIGHTS,
  SEF_globalTHEIR_HANGING_BISHOPS,
  SEF_globalTHEIR_HANGING_ROOKS,
  SEF_globalTHEIR_HANGING_QUEENS,
  SEF_globalLONELY_KING_IN_CENTER,
  SEF_globalLONELY_KING_AWAY_FROM_ENEMY_KING,
  SEF_globalTIME,
  SEF_globalKPVK_OPPOSITION,
  SEF_globalSQUARE_RULE,
  SEF_globalADVANCED_PAWNS_1,
  SEF_globalADVANCED_PAWNS_2,
  SEF_globalOPEN_ROOKS,
  SEF_globalROOKS_ON_THEIR_SIDE,
  SEF_globalKING_IN_FRONT_OF_PASSED_PAWN,
  SEF_globalKING_IN_FRONT_OF_PASSED_PAWN2,
  SEF_globalOUR_MATERIAL_THREATS,
  SEF_globalTHEIR_MATERIAL_THREATS,
  SEF_globalLONELY_KING_ON_EDGE,
  SEF_globalOUTPOSTED_KNIGHTS,
  SEF_globalOUTPOSTED_BISHOPS,
  SEF_globalPAWN_MOVES,
  SEF_globalKNIGHT_MOVES,
  SEF_globalBISHOP_MOVES,
  SEF_globalROOK_MOVES,
  SEF_globalQUEEN_MOVES,
  SEF_globalPAWN_MOVES_ON_THEIR_SIDE,
  SEF_globalKNIGHT_MOVES_ON_THEIR_SIDE,
  SEF_globalBISHOP_MOVES_ON_THEIR_SIDE,
  SEF_globalROOK_MOVES_ON_THEIR_SIDE,
  SEF_globalQUEEN_MOVES_ON_THEIR_SIDE,
  SEF_globalKING_HOME_QUALITY,
  SEF_globalBISHOPS_BLOCKING_KNIGHTS,
  SEF_globalOUR_HANGING_PAWNS_2,
  SEF_globalOUR_HANGING_KNIGHTS_2,
  SEF_globalOUR_HANGING_BISHOPS_2,
  SEF_globalOUR_HANGING_ROOKS_2,
  SEF_globalOUR_HANGING_QUEENS_2,
  SEF_globalTHEIR_HANGING_PAWNS_2,
  SEF_globalTHEIR_HANGING_KNIGHTS_2,
  SEF_globalTHEIR_HANGING_BISHOPS_2,
  SEF_globalTHEIR_HANGING_ROOKS_2,
  SEF_globalTHEIR_HANGING_QUEENS_2,
  SEF_globalQUEEN_THREATS_NEAR_KING,
  SEF_globalMISSING_FIANCHETTO_BISHOP,
  SEF_globalNUM_BAD_SQUARES_FOR_PAWNS,
  SEF_globalNUM_BAD_SQUARES_FOR_MINORS,
  SEF_globalNUM_BAD_SQUARES_FOR_ROOKS,
  SEF_globalNUM_BAD_SQUARES_FOR_QUEENS,
  SEF_globalIN_TRIVIAL_CHECK,
  SEF_globalIN_DOUBLE_CHECK,
  SEF_globalTHREATS_NEAR_OUR_KING,
  SEF_globalTHREATS_NEAR_THEIR_KING,
  SEF_globalNUM_PIECES_HARRASSABLE_BY_PAWNS,
  SEF_globalPAWN_CHECKS,
  SEF_globalKNIGHT_CHECKS,
  SEF_globalBISHOP_CHECKS,
  SEF_globalROOK_CHECKS,
  SEF_globalQUEEN_CHECKS,
  SEF_globalBACK_RANK_MATE_THREAT_AGAINST_US,
  SEF_globalBACK_RANK_MATE_THREAT_AGAINST_THEM,
  SEF_globalOUR_KING_HAS_0_ESCAPE_SQUARES,
  SEF_globalTHEIR_KING_HAS_0_ESCAPE_SQUARES,
  SEF_globalOUR_KING_HAS_1_ESCAPE_SQUARES,
  SEF_globalTHEIR_KING_HAS_1_ESCAPE_SQUARES,
  SEF_globalOUR_KING_HAS_2_ESCAPE_SQUARES,
  SEF_globalTHEIR_KING_HAS_2_ESCAPE_SQUARES,
  SEF_globalOPPOSITE_SIDE_KINGS_PAWN_STORM,
  SEF_globalIN_CHECK_AND_OUR_HANGING_QUEENS,
  SEF_globalPROMOTABLE_PAWN,

	SEF_isOurPawn,
	SEF_isOurKnight,
	SEF_isOurBishop,
	SEF_isOurRook,
	SEF_isOurQueen,
	SEF_isOurKing,
	SEF_isTheirPawn,
	SEF_isTheirKnight,
	SEF_isTheirBishop,
	SEF_isTheirRook,
	SEF_isTheirQueen,
	SEF_isTheirKing,

	SEF_isOurPassedPawn,
	SEF_isTheirPassedPawn,
	SEF_isOurProtectedPawn,
	SEF_isTheirProtectedPawn,
	SEF_aheadOfOurPawns,
	SEF_aheadOfTheirPawns,
	SEF_behindOurPawns,
	SEF_behindTheirPawns,
	SEF_aheadOfOurPassedPawn,
	SEF_aheadOfTheirPassedPawn,

	SEF_isRank0,
	SEF_isRank1,
	SEF_isRank2,
	SEF_isRank3,
	SEF_isRank4,
	SEF_isRank5,
	SEF_isRank6,
	SEF_isRank7,

	SEF_isFile0,
	SEF_isFile1,
	SEF_isFile2,
	SEF_isFile3,
	SEF_isFile4,
	SEF_isFile5,
	SEF_isFile6,
	SEF_isFile7,
	
	SEF_badForTheirPawns,
	SEF_badForTheirKnights,
	SEF_badForTheirBishops,
	SEF_badForTheirRooks,
	SEF_badForTheirQueens,
	SEF_badForTheirKing,

	SEF_badForMyPawns,
	SEF_badForMyKnights,
	SEF_badForMyBishops,
	SEF_badForMyRooks,
	SEF_badForMyQueens,
	SEF_badForMyKing,

	SEF_ourPawnTargets,
	SEF_ourKnightTargets,
	SEF_ourBishopTargets,
	SEF_ourRookTargets,
	SEF_ourQueenTargets,
	SEF_ourKingTargets,

	SEF_theirPawnTargets,
	SEF_theirKnightTargets,
	SEF_theirBishopTargets,
	SEF_theirRookTargets,
	SEF_theirQueenTargets,
	SEF_theirKingTargets,

	SEF_distToMyKing,
	SEF_distToTheirKing,
	SEF_vertDistToMyKing,
	SEF_vertDistToTheirKing,

	SEF_COUNT,
};

struct SquareEmbeddings {
	int16_t features[kNumSquares][SEF_COUNT];
};

std::ostream& operator<<(std::ostream& stream, const SquareEmbeddings& emb) {
	for (Square sq = Square(0); sq < kNumSquares; sq = Square(sq + 1)) {
		for (int i = 0; i < SEF_COUNT; ++i) {
			stream << " " << emb.features[sq][i];
		}
	}
	return stream;
}

template<Color US>
SquareEmbeddings compute_embeddings(const Position& position) {
	Evaluator evaluator;

	constexpr Color THEM = opposite_color<US>();
	constexpr Direction kForward = (US == Color::WHITE ? Direction::NORTH : Direction::SOUTH);
  constexpr Direction kBackward = opposite_dir<kForward>();

	SquareEmbeddings r;
	std::fill_n(&r.features[0][0], kNumSquares * SEF_COUNT, 0);

	const Threats<US> threats(position);
	const PawnAnalysis<US> pawnAnalysis(position, threats);
	const SafeSquare ourKingSq = safe_lsb(position.pieceBitboards_[coloredPiece<US, Piece::KING>()]);
  const SafeSquare theirKingSq = lsb(position.pieceBitboards_[coloredPiece<THEM, Piece::KING>()]);

  std::fill_n(evaluator.features, EF::NUM_EVAL_FEATURES, 0);
	evaluator.score<US>(position, threats);    

  int16_t globals[SEF_COUNT];
  std::fill_n(globals, SEF_COUNT, 0);
	// for (Piece piece = Piece::PAWN; piece < Piece::KING; piece = Piece(piece + 1)) {
	// 	globals[(piece - Piece::PAWN) + SEF_globalOurPawnsCount] = std::popcount(position.pieceBitboards_[coloredPiece<US>(piece)]);
	// 	globals[(piece - Piece::PAWN) + SEF_globalTheirPawnsCount] = std::popcount(position.pieceBitboards_[coloredPiece<THEM>(piece)]);
	// }
	for (int i = 0; i < EF::NUM_EVAL_FEATURES; ++i) {
		if (evaluator.features[i] < -100 || evaluator.features[i] > 100) {
			std::cout << EFSTR[i] << "  " << evaluator.features[i] << "  " << position.fen() << std::endl;
		}
		globals[i] = evaluator.features[i];
	}

	Bitboard aheadOfOurPawns, aheadOfTheirPawns;
	Bitboard behindOurPawns, behindTheirPawns;
	Bitboard aheadOfOurPassedPawns, aheadOfTheirPassedPawns;
	if (US == Color::WHITE) {
	  aheadOfOurPawns = northFill(position.pieceBitboards_[coloredPiece<US>(Piece::PAWN)]);
	  aheadOfTheirPawns = southFill(position.pieceBitboards_[coloredPiece<THEM>(Piece::PAWN)]);
	  behindOurPawns = southFill(position.pieceBitboards_[coloredPiece<US>(Piece::PAWN)]);
	  behindTheirPawns = northFill(position.pieceBitboards_[coloredPiece<THEM>(Piece::PAWN)]);
    aheadOfOurPassedPawns = northFill(pawnAnalysis.ourPassedPawns);
    aheadOfTheirPassedPawns = southFill(pawnAnalysis.theirPassedPawns);
  } else {
	  aheadOfOurPawns = southFill(position.pieceBitboards_[coloredPiece<US>(Piece::PAWN)]);
	  aheadOfTheirPawns = northFill(position.pieceBitboards_[coloredPiece<THEM>(Piece::PAWN)]);
	  behindOurPawns = northFill(position.pieceBitboards_[coloredPiece<US>(Piece::PAWN)]);
	  behindTheirPawns = southFill(position.pieceBitboards_[coloredPiece<THEM>(Piece::PAWN)]);
    aheadOfOurPassedPawns = southFill(pawnAnalysis.ourPassedPawns);
    aheadOfTheirPassedPawns = northFill(pawnAnalysis.theirPassedPawns);
  }

	for (SafeSquare sq = Square(0); sq < kNumSquares; sq = SafeSquare(sq + 1)) {
		const Location loc = square2location(sq);
		int16_t *f = r.features[sq];

		std::memcpy(f, globals, sizeof(globals));

		for (Piece piece = Piece::PAWN; piece <= Piece::KING; piece = Piece(piece + 1)) {
			f[(piece - Piece::PAWN) + SEF_isOurPawn] = (position.tiles_[sq] == coloredPiece<US>(piece));
			f[(piece - Piece::PAWN) + SEF_isTheirPawn] = (position.tiles_[sq] == coloredPiece<THEM>(piece));
		}

		f[SEF_isOurPassedPawn] = (pawnAnalysis.ourPassedPawns & loc) > 0;
		f[SEF_isTheirPassedPawn] = (pawnAnalysis.theirPassedPawns & loc) > 0;
		f[SEF_isOurProtectedPawn] = (pawnAnalysis.ourProtectedPawns & loc) > 0;
		f[SEF_isTheirProtectedPawn] = (pawnAnalysis.theirProtectedPawns & loc) > 0;
		f[SEF_aheadOfOurPawns] = (aheadOfOurPawns & loc) > 0;
		f[SEF_aheadOfTheirPawns] = (aheadOfTheirPawns & loc) > 0;
		f[SEF_behindOurPawns] = (behindOurPawns & loc) > 0;
		f[SEF_behindTheirPawns] = (behindTheirPawns & loc) > 0;
		f[SEF_aheadOfOurPassedPawn] = (aheadOfOurPassedPawns & loc) > 0;
		f[SEF_aheadOfTheirPassedPawn] = (aheadOfTheirPassedPawns & loc) > 0;


		if (US == Color::WHITE) {
			f[SEF_isRank0 + sq / 8] = 1;
		} else {
			f[SEF_isRank0 + (7 - sq / 8)] = 1;
		}
		f[SEF_isFile0 + sq % 8] = 1;

		for (Piece piece = Piece::PAWN; piece <= Piece::KING; piece = Piece(piece + 1)) {
			f[SEF_badForTheirPawns + (piece - Piece::PAWN)] = (threats.badForTheir[piece] & loc) > 0;
			f[SEF_badForMyPawns + (piece - Piece::PAWN)] = (threats.badForOur[piece] & loc) > 0;
		}

		f[SEF_distToMyKing] = king_dist(sq, ourKingSq);
		f[SEF_distToTheirKing] = king_dist(sq, theirKingSq);
		f[SEF_vertDistToMyKing] = std::abs(sq / 8 - ourKingSq / 8);
		f[SEF_vertDistToTheirKing] = std::abs(sq / 8 - theirKingSq / 8);

		f[SEF_ourPawnTargets] = (threats.ourPawnTargets & loc) > 0;
		f[SEF_ourKnightTargets] = (threats.ourKnightTargets & loc) > 0;
		f[SEF_ourBishopTargets] = (threats.ourBishopTargets & loc) > 0;
		f[SEF_ourRookTargets] = (threats.ourRookTargets & loc) > 0;
		f[SEF_ourQueenTargets] = (threats.ourQueenTargets & loc) > 0;
		f[SEF_ourKingTargets] = (threats.ourKingTargets & loc) > 0;

		f[SEF_theirPawnTargets] = (threats.theirPawnTargets & loc) > 0;
		f[SEF_theirKnightTargets] = (threats.theirKnightTargets & loc) > 0;
		f[SEF_theirBishopTargets] = (threats.theirBishopTargets & loc) > 0;
		f[SEF_theirRookTargets] = (threats.theirRookTargets & loc) > 0;
		f[SEF_theirQueenTargets] = (threats.theirQueenTargets & loc) > 0;
		f[SEF_theirKingTargets] = (threats.theirKingTargets & loc) > 0;
	}

	return r;
}

}  // namespace ChessEngine

#endif  // SQUARE_EMBEDDINGS_H