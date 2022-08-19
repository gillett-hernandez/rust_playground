#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PieceType {
    Pawn,
    Knight,
    Rook,
    Bishop,
    Queen,
    King,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Color {
    Black,
    White,
}

#[derive(Copy, Clone, Debug)]
struct Piece {
    piece_type: PieceType,
    color: Color,
}

type Move = (bool, (usize, usize));

impl Piece {
    // the bool represents whether the move captures or not
    // pub fn iter_possible_moves(
    //     &self,
    //     pos: (usize, usize),
    //     board: &Board,
    //     last_move: Move,
    // ) -> impl Iterator<Item = Move> {
    //     match self.piece_type {
    //         PieceType::Pawn => {
    //             let flip = if self.color == Color::Black { 1 } else { -1 };
    //             for offset in [(0, 1), (1, 1), (-1, 1)] {}
    //             // ((pos.1 == 1 && self.color == Color::Black)
    //             //     || (pos.1 == 6 && self.color == Color::White))
    //         }
    //         PieceType::Knight => todo!(),
    //         PieceType::Rook => todo!(),
    //         PieceType::Bishop => todo!(),
    //         PieceType::Queen => todo!(),
    //         PieceType::King => todo!(),
    //     }
    // }
}

#[derive(Copy, Clone, Debug)]
struct Board([[Option<Piece>; 8]; 8]);

impl Board {
    pub fn new() -> Self {
        // king and queen positions might need to be switched.
        Board([
            [
                Some(Piece {
                    color: Color::Black,
                    piece_type: PieceType::Rook,
                }),
                Some(Piece {
                    color: Color::Black,
                    piece_type: PieceType::Knight,
                }),
                Some(Piece {
                    color: Color::Black,
                    piece_type: PieceType::Bishop,
                }),
                Some(Piece {
                    color: Color::Black,
                    piece_type: PieceType::Queen,
                }),
                Some(Piece {
                    color: Color::Black,
                    piece_type: PieceType::King,
                }),
                Some(Piece {
                    color: Color::Black,
                    piece_type: PieceType::Bishop,
                }),
                Some(Piece {
                    color: Color::Black,
                    piece_type: PieceType::Knight,
                }),
                Some(Piece {
                    color: Color::Black,
                    piece_type: PieceType::Rook,
                }),
            ],
            [Some(Piece {
                color: Color::Black,
                piece_type: PieceType::Pawn,
            }); 8],
            [None; 8],
            [None; 8],
            [None; 8],
            [None; 8],
            [Some(Piece {
                color: Color::White,
                piece_type: PieceType::Pawn,
            }); 8],
            [
                Some(Piece {
                    color: Color::White,
                    piece_type: PieceType::Rook,
                }),
                Some(Piece {
                    color: Color::White,
                    piece_type: PieceType::Knight,
                }),
                Some(Piece {
                    color: Color::White,
                    piece_type: PieceType::Bishop,
                }),
                Some(Piece {
                    color: Color::White,
                    piece_type: PieceType::Queen,
                }),
                Some(Piece {
                    color: Color::White,
                    piece_type: PieceType::King,
                }),
                Some(Piece {
                    color: Color::White,
                    piece_type: PieceType::Bishop,
                }),
                Some(Piece {
                    color: Color::White,
                    piece_type: PieceType::Knight,
                }),
                Some(Piece {
                    color: Color::White,
                    piece_type: PieceType::Rook,
                }),
            ],
        ])
    }

    pub fn from_configuration(pieces: Vec<(Piece, (u8, u8))>) -> Self {
        // later appearances of pieces overwrite earlier appearances in the `pieces` Vec
        let mut inner_board = [[None; 8]; 8];
        for (piece, (x, y)) in &pieces {
            inner_board[*y as usize][*x as usize] = Some(*piece);
        }
        Board(inner_board)
    }
}

fn main() {
    // TODO: basic piece movements and move validity checking
    // TODO: piece capture movements and validity checking (pawn capture)
    // TODO: promotion
    // TODO: advanced piece movements and captures (en passant, castling), + validity checking
    // TODO: checkmate detection, check detection and subsequent move validity
    // TODO: rudimentary lookahead AI
}
