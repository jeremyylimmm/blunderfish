use std::collections::HashMap;

#[derive(Copy, Clone)]
pub enum Side {
    White,
    Black,
}

impl Side {
    pub fn other(&self) -> Side {
        match self {
            Side::White => Side::Black,
            Side::Black => Side::White,
        }
    }
}

#[derive(Copy, Clone, Hash, PartialEq)]
pub enum Piece {
    Pawn,
    Rook,
    Knight,
    Bishop,
    Queen,
    King,
}

impl Piece {
    fn alg(&self) -> &'static str {
        match self {
            Self::Pawn => "",
            Self::Rook => "R",
            Self::Knight => "N",
            Self::Bishop => "B",
            Self::Queen => "Q",
            Self::King => "K",
        }
    }

    fn iter() -> [Piece; 6] {
        [
            Piece::Pawn,
            Piece::Rook,
            Piece::Knight,
            Piece::Bishop,
            Piece::Queen,
            Piece::King,
        ]
    }
}

impl std::cmp::Eq for Piece {}

#[derive(Copy, Clone, Hash, PartialEq)]
pub struct Position {
    pub index: u8,
}

impl std::cmp::Eq for Position {}

impl Position {
    fn bb(&self) -> u64 {
        1u64 << self.index
    }

    fn rank(&self) -> usize {
        (self.index as usize / 8) + 1
    }

    fn file(&self) -> char {
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'][(self.index % 8) as usize]
    }
}

pub struct Board {
    pub sets: [[u64; 6]; 2],
    pub en_passant_mask: [u64; 2],
}

trait Bitboard {
    fn remove_piece(&mut self, p: Position);
    fn add_piece(&mut self, p: Position);
}

impl Bitboard for u64 {
    fn remove_piece(&mut self, p: Position) {
        assert!((*self & p.bb()) != 0);
        *self &= !p.bb();
    }

    fn add_piece(&mut self, p: Position) {
        assert!((*self & p.bb()) == 0);
        *self |= p.bb();
    }
}

trait Set {
    fn occupied(&self) -> u64;
}

impl Set for [u64; 6] {
    fn occupied(&self) -> u64 {
        self.iter().copied().fold(0, |acc, x| acc | x)
    }
}

impl std::ops::Index<Piece> for [u64; 6] {
    type Output = u64;

    fn index(&self, i: Piece) -> &Self::Output {
        &self[i as usize]
    }
}

impl std::ops::IndexMut<Piece> for [u64; 6] {
    fn index_mut(&mut self, i: Piece) -> &mut u64 {
        &mut self[i as usize]
    }
}

impl std::ops::Index<Side> for [[u64; 6]; 2] {
    type Output = [u64; 6];

    fn index(&self, i: Side) -> &Self::Output {
        &self[i as usize]
    }
}

impl std::ops::IndexMut<Side> for [[u64; 6]; 2] {
    fn index_mut(&mut self, i: Side) -> &mut [u64; 6] {
        &mut self[i as usize]
    }
}

pub const FILE_A: u64 = 0x0101010101010101;
pub const FILE_H: u64 = 0x8080808080808080;
pub const FILE_GH: u64 = 0xc0c0c0c0c0c0c0c0;
pub const FILE_AB: u64 = 0x0303030303030303;
pub const RANK_3: u64 = 0x0000000000ff0000;
pub const RANK_6: u64 = 0x0000ff0000000000;
pub const RANK_1: u64 = 0x00000000000000ff;
pub const RANK_8: u64 = 0xff00000000000000;

use std::sync::LazyLock;

pub static ROOK_TABLE: LazyLock<MagicTable> = LazyLock::new(|| MagicTable::generate(gen_mask_rook, gen_moves_rook_sliding));
pub static BISHOP_TABLE: LazyLock<MagicTable> = LazyLock::new(|| MagicTable::generate(gen_mask_bishop, gen_moves_bishop_sliding));

pub fn generate_tables() {
    LazyLock::force(&ROOK_TABLE);
    LazyLock::force(&BISHOP_TABLE);
}

#[derive(Copy, Clone)]
pub enum MoveKind {
    Normal,
    DoublePush,
    EnPassant,
    Promotion(Piece),
}

#[derive(Copy, Clone)]
pub struct Move {
    from: Position,
    to: Position,
    piece: Piece,
    kind: MoveKind,
}

fn iter_bb<F: FnMut(Position)>(mut bb: u64, mut f: F) {
    while bb > 0 {
        let i = Position {
            index: bb.trailing_zeros() as _,
        };
        f(i);
        bb &= bb - 1;
    }
}

impl Board {
    pub fn new() -> Self {
        Self {
            sets: [
                [
                    0x000000000000ff00,
                    0x0000000000000081,
                    0x0000000000000042,
                    0x0000000000000024,
                    0x0000000000000008,
                    0x0000000000000010,
                ],
                [
                    0x00ff000000000000,
                    0x8100000000000000,
                    0x4200000000000000,
                    0x2400000000000000,
                    0x0800000000000000,
                    0x1000000000000000,
                ],
            ],
            en_passant_mask: [0; 2],
        }
    }

    pub fn clear(&mut self) {
        self.sets = [[0; _]; _];
    }

    fn fill_chars(&self, chars: &mut [char; 64], bb: u64, which: char) {
        for i in 0..64 {
            if ((bb >> i) & 1) != 0 {
                chars[i] = which;
            }
        }
    }

    fn piece_at(&self, pos: Position) -> Option<Piece> {
        for set in &self.sets {
            for p in Piece::iter() {
                if (set[p] & pos.bb()) != 0 {
                    return Some(p);
                }
            }
        }

        None
    }

    pub fn check_integrity(&self) -> Result<(), &'static str> {
        let mut seen = 0u64;

        let mut check = |x: u64| {
            if (seen & x) != 0 {
                Err("bit boards clash")
            } else {
                seen |= x;
                Ok(())
            }
        };

        for set in self.sets {
            for bb in set {
                check(bb)?;
            }
        }

        Ok(())
    }

    pub fn dump(&self) {
        let mut chars = ['-'; 64];

        self.check_integrity().unwrap();

        self.fill_chars(&mut chars, self.sets[Side::White][Piece::Pawn], '♟');
        self.fill_chars(&mut chars, self.sets[Side::White][Piece::Rook], '♜');
        self.fill_chars(&mut chars, self.sets[Side::White][Piece::Knight], '♞');
        self.fill_chars(&mut chars, self.sets[Side::White][Piece::Bishop], '♝');
        self.fill_chars(&mut chars, self.sets[Side::White][Piece::Queen], '♛');
        self.fill_chars(&mut chars, self.sets[Side::White][Piece::King], '♚');
        self.fill_chars(&mut chars, self.sets[Side::Black][Piece::Pawn], '♙');
        self.fill_chars(&mut chars, self.sets[Side::Black][Piece::Rook], '♖');
        self.fill_chars(&mut chars, self.sets[Side::Black][Piece::Knight], '♘');
        self.fill_chars(&mut chars, self.sets[Side::Black][Piece::Bishop], '♗');
        self.fill_chars(&mut chars, self.sets[Side::Black][Piece::Queen], '♕');
        self.fill_chars(&mut chars, self.sets[Side::Black][Piece::King], '♔');

        for rank in (0..8).rev() {
            print!("{} ", rank + 1);

            for file in 0..8 {
                print!("{} ", chars[rank * 8 + file]);
            }

            print!("\n");
        }

        print!("  ");

        let file_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

        for file in 0..8 {
            print!("{} ", file_names[file]);
        }

        print!("\n");
    }

    pub fn occupied(&self) -> u64 {
        self.sets.iter().fold(0, |acc, x| acc | x.occupied())
    }

    pub fn empty(&self) -> u64 {
        !self.occupied()
    }

    fn white_pawn_single_moves(&self, from: Position) -> u64 {
        let bb = from.bb();

        let push = (bb << 8) & self.empty();
        let left_capture = (bb << 7) & self.sets[Side::Black].occupied() & !FILE_H;
        let right_capture = (bb << 9) & self.sets[Side::Black].occupied() & !FILE_A;

        push | left_capture | right_capture
    }

    fn black_pawn_single_moves(&self, from: Position) -> u64 {
        let bb = from.bb();

        let push = (bb >> 8) & self.empty();
        let left_capture = (bb >> 9) & self.sets[Side::White].occupied() & !FILE_H;
        let right_capture = (bb >> 7) & self.sets[Side::White].occupied() & !FILE_A;

        push | left_capture | right_capture
    }

    fn white_en_passant(&self, from: Position) -> u64 {
        let bb = from.bb();

        let left_capture = (bb << 7) & self.en_passant_mask[Side::White as usize] & !FILE_H;
        let right_capture = (bb << 9) & self.en_passant_mask[Side::White as usize] & !FILE_A;

        left_capture | right_capture
    }

    fn black_en_passant(&self, from: Position) -> u64 {
        let bb = from.bb();

        let left_capture = (bb >> 9) & self.en_passant_mask[Side::Black as usize] & !FILE_H;
        let right_capture = (bb >> 7) & self.en_passant_mask[Side::Black as usize] & !FILE_A;

        left_capture | right_capture
    }

    fn white_pawn_double_push(&self, from: Position) -> u64 {
        let bb = from.bb();
        let single = (bb << 8) & self.empty() & RANK_3;
        (single << 8) & self.empty()
    }

    fn black_pawn_double_push(&self, from: Position) -> u64 {
        let bb = from.bb();
        let single = (bb >> 8) & self.empty() & RANK_6;
        (single >> 8) & self.empty()
    }

    pub fn king_moves(&self, side: Side, from: Position) -> u64 {
        let bb = from.bb();

        let right = (bb << 1) & !FILE_A;
        let left = (bb >> 1) & !FILE_H;
        let up = bb << 8;
        let down = bb >> 8;
        let left_up = (bb << 7) & !FILE_H;
        let right_up = (bb << 9) & !FILE_A;
        let left_down = (bb >> 9) & !FILE_H;
        let right_down = (bb >> 7) & !FILE_A;

        let all = right | left | up | down | left_up | right_up | left_down | right_down;

        all & !self.sets[side].occupied()
    }

    pub fn knight_moves(&self, side: Side, from: Position) -> u64 {
        let bb = from.bb();

        let m1 = (bb << 6) & !FILE_GH;
        let m2 = (bb << 15) & !FILE_H;
        let m3 = (bb << 17) & !FILE_A;
        let m4 = (bb << 10) & !FILE_AB;
        let m5 = (bb >> 6) & !FILE_AB;
        let m6 = (bb >> 15) & !FILE_A;
        let m7 = (bb >> 17) & !FILE_H;
        let m8 = (bb >> 10) & !FILE_GH;

        let all = m1 | m2 | m3 | m4 | m5 | m6 | m7 | m8;

        all & !self.sets[side].occupied()
    }

    pub fn rook_moves(&self, side: Side, from: Position) -> u64 {
        let moves = ROOK_TABLE.lookup(from, self.occupied());
        moves & !self.sets[side].occupied()
    }

    pub fn bishop_moves(&self, side: Side, from: Position) -> u64 {
        let moves = BISHOP_TABLE.lookup(from, self.occupied());
        moves & !self.sets[side].occupied()
    }

    pub fn queen_moves(&self, side: Side, from: Position) -> u64 {
        self.bishop_moves(side, from) | self.rook_moves(side, from)
    }

    pub fn generate_moves(&self, side: Side, moves: &mut Vec<Move>) {
        moves.clear();

        iter_bb(self.sets[side][Piece::Pawn], |from| {
            let (single, double, en_passant, rank_mask) = match side {
                Side::White => (
                    self.white_pawn_single_moves(from),
                    self.white_pawn_double_push(from),
                    self.white_en_passant(from),
                    RANK_8,
                ),
                Side::Black => (
                    self.black_pawn_single_moves(from),
                    self.black_pawn_double_push(from),
                    self.black_en_passant(from),
                    RANK_1,
                ),
            };

            let non_promotion = single & (!rank_mask);

            iter_bb(non_promotion, |to| {
                moves.push(Move {
                    from,
                    to,
                    piece: Piece::Pawn,
                    kind: MoveKind::Normal,
                });
            });

            let promotion = single & rank_mask;

            iter_bb(promotion, |to| {
                moves.push(Move {
                    from,
                    to,
                    piece: Piece::Pawn,
                    kind: MoveKind::Promotion(Piece::Bishop),
                });
                moves.push(Move {
                    from,
                    to,
                    piece: Piece::Pawn,
                    kind: MoveKind::Promotion(Piece::Rook),
                });
                moves.push(Move {
                    from,
                    to,
                    piece: Piece::Pawn,
                    kind: MoveKind::Promotion(Piece::Queen),
                });
                moves.push(Move {
                    from,
                    to,
                    piece: Piece::Pawn,
                    kind: MoveKind::Promotion(Piece::Knight),
                });
            });

            iter_bb(double, |to| {
                moves.push(Move {
                    from,
                    to,
                    piece: Piece::Pawn,
                    kind: MoveKind::DoublePush,
                });
            });

            iter_bb(en_passant, |to| {
                moves.push(Move {
                    from,
                    to,
                    piece: Piece::Pawn,
                    kind: MoveKind::EnPassant,
                });
            });
        });

        iter_bb(self.sets[side][Piece::King], |from| {
            // realistically there should only be one king but whatever
            iter_bb(self.king_moves(side, from), |to| {
                moves.push(Move {
                    from,
                    to,
                    piece: Piece::King,
                    kind: MoveKind::Normal,
                });
            });
        });

        iter_bb(self.sets[side][Piece::Knight], |from| {
            iter_bb(self.knight_moves(side, from), |to| {
                moves.push(Move {
                    from,
                    to,
                    piece: Piece::Knight,
                    kind: MoveKind::Normal,
                });
            });
        });

        iter_bb(self.sets[side][Piece::Rook], |from| {
            iter_bb(self.rook_moves(side, from), |to| {
                moves.push(Move {
                    from,
                    to,
                    piece: Piece::Rook,
                    kind: MoveKind::Normal,
                });
            });
        });

        iter_bb(self.sets[side][Piece::Bishop], |from| {
            iter_bb(self.bishop_moves(side, from), |to| {
                moves.push(Move {
                    from,
                    to,
                    piece: Piece::Bishop,
                    kind: MoveKind::Normal,
                });
            });
        });

        iter_bb(self.sets[side][Piece::Queen], |from| {
            iter_bb(self.queen_moves(side, from), |to| {
                moves.push(Move {
                    from,
                    to,
                    piece: Piece::Queen,
                    kind: MoveKind::Normal,
                });
            });
        });
    }

    pub fn execute(&self, side: Side, m: &Move) -> Board {
        let mut b = Board {
            sets: self.sets,
            en_passant_mask: [0; 2],
        };

        b.sets[side][m.piece].remove_piece(m.from);

        let capture_loc = if matches!(m.kind, MoveKind::EnPassant) {
            match side {
                Side::White => Position {
                    index: m.to.index - 8,
                },
                Side::Black => Position {
                    index: m.to.index + 8,
                },
            }
        } else {
            m.to
        };

        if let Some(captured_piece) = b.piece_at(capture_loc) {
            b.sets[side.other()][captured_piece].remove_piece(capture_loc);
        }

        if let MoveKind::Promotion(prom) = m.kind {
            b.sets[side][prom].add_piece(m.to);
        } else {
            b.sets[side][m.piece].add_piece(m.to);
        }

        if matches!(m.kind, MoveKind::DoublePush) {
            b.en_passant_mask[side.other() as usize] = match side {
                Side::White => m.to.bb() >> 8,
                Side::Black => m.to.bb() << 8,
            }
        }

        b
    }
}

pub fn name_moves<'a>(board: &Board, all_moves: &'a Vec<Move>) -> HashMap<String, &'a Move> {
    let mut positions = HashMap::<Position, HashMap<Piece, Vec<&Move>>>::new();

    for m in all_moves {
        positions
            .entry(m.to)
            .or_default()
            .entry(m.piece)
            .or_default()
            .push(m);
    }

    let mut result = HashMap::new();

    for (pos, pieces) in positions {
        let is_capture = board.piece_at(pos).is_some();

        for (piece, moves) in pieces {
            match piece {
                Piece::Pawn => {
                    // pawns are special cases
                    for m in moves {
                        let capture = if is_capture || matches!(m.kind, MoveKind::EnPassant) {
                            format!("{}x", m.from.file())
                        } else {
                            "".to_string()
                        };

                        let suffix = match m.kind {
                            MoveKind::Promotion(prom) => {
                                format!("={}", prom.alg())
                            }
                            _ => "".to_string(),
                        };

                        let name = format!("{}{}{}{}", capture, m.to.file(), m.to.rank(), suffix);
                        result.insert(name, m);
                    }
                }

                _ => {
                    let capture = if is_capture { "x" } else { "" };

                    for (i, m) in moves.iter().enumerate() {
                        let others = moves.iter().enumerate().filter(|(j, _)| *j != i);

                        let need_file = others.clone().any(|(_, n)| m.from.file() != n.from.file());

                        let need_rank = others.clone().any(|(_, n)| {
                            m.from.file() == n.from.file() && m.from.rank() != n.from.rank()
                        });

                        let disambig_file = if need_file {
                            format!("{}", m.from.file())
                        } else {
                            "".to_string()
                        };
                        let disambig_rank = if need_rank {
                            format!("{}", m.from.rank())
                        } else {
                            "".to_string()
                        };

                        let name = format!(
                            "{}{}{}{}{}{}",
                            piece.alg(),
                            disambig_file,
                            disambig_rank,
                            capture,
                            pos.file(),
                            pos.rank()
                        );

                        if result.insert(name, m).is_some() {
                            panic!("disambiguation does NOT work!!");
                        }
                    }
                }
            }
        }
    }

    result
}

pub struct MagicTable {
    pub magic: [u64; 64],
    pub shift: [u32; 64],
    pub mask: [u64; 64],
    pub tables: [Vec<u64>; 64],
}

struct RNG {
    state: u64,
    inc: u64,
}

impl RNG {
    fn rand32(&mut self) -> u32 {
        let oldstate = self.state;
        // Advance internal state
        self.state = oldstate.overflowing_mul(6364136223846793005u64).0 + (self.inc | 1);
        // Calculate output function (XSH RR), uses old state for max ILP
        let xorshifted = (((oldstate >> 18u32) ^ oldstate) >> 27u32) as u32;
        let rot = (oldstate >> 59u32) as u32;
        (xorshifted >> rot) | (xorshifted << ((-(rot as i32)) & 31))
    }

    fn rand64(&mut self) -> u64 {
        let a = self.rand32() as u64;
        let b = self.rand32() as u64;

        a << 32 | b
    }
}

fn slide<T: Fn(u64)->u64>(pos: usize, blockers: u64, transform: T) -> u64 {
    let mut result = 0;
    let mut bb = 1u64 << pos;

    while bb > 0 {
        bb = transform(bb);

        result |= bb;
        
        if (bb & blockers) != 0 {
            break;
        }
    }

    result
}

pub fn gen_moves_bishop_sliding(pos: usize, blockers: u64) -> u64 {
    let mut result = 0;

    result |= slide(pos, blockers, |bb| (bb<<7) & !FILE_H);
    result |= slide(pos, blockers, |bb| (bb<<9) & !FILE_A);
    result |= slide(pos, blockers, |bb| (bb>>7) & !FILE_A);
    result |= slide(pos, blockers, |bb| (bb>>9) & !FILE_H);

    result
}

pub fn gen_mask_bishop(sq: usize) -> u64 {
    let mut result = 0;

    result |= slide(sq, 0, |bb| (bb<<7) & !FILE_H) & !(RANK_8|FILE_A);
    result |= slide(sq, 0, |bb| (bb<<9) & !FILE_A) & !(RANK_8|FILE_H);
    result |= slide(sq, 0, |bb| (bb>>7) & !FILE_A) & !(RANK_1|FILE_H);
    result |= slide(sq, 0, |bb| (bb>>9) & !FILE_H) & !(RANK_1|FILE_A);

    assert!((result & (1u64 << sq)) == 0);
    result
}

pub fn gen_moves_rook_sliding(pos: usize, blockers: u64) -> u64 {
    let mut result = 0;

    let rank = pos/8;

    result |= slide(pos, blockers, |bb|(bb >> 1) & rank_mask(rank));
    result |= slide(pos, blockers, |bb|(bb << 1) & rank_mask(rank));
    result |= slide(pos, blockers, |bb|bb << 8);
    result |= slide(pos, blockers, |bb|bb >> 8);

    result
}

fn rank_mask(rank: usize) -> u64 {
    0xffu64 << (rank*8)
}

fn file_mask(file: usize) -> u64 {
    let col = 1u64 << file;
    col | col << 8 | col << 16 | col << 24 | col << 32 | col << 40 | col << 48 | col << 56
}

pub fn gen_mask_rook(sq: usize) -> u64 {
    let rank = sq / 8;
    let file = sq % 8;

    let r = rank_mask(rank) & !(FILE_A | FILE_H);
    let f = file_mask(file) & !(RANK_1 | RANK_8);

    (r | f) & !(1u64 << sq)
}

impl MagicTable {
    #[allow(unused)]
    fn generate<GenMaskFn: Fn(usize) -> u64, GenMovesFn: Fn(usize, u64) -> u64>(
        gen_mask: GenMaskFn,
        gen_moves: GenMovesFn,
    ) -> Self {
        let mut magic = [0u64; _];
        let mut shift = [0; _];
        let mut mask = [0u64; _];
        let mut tables = std::array::from_fn(|_| vec![]);

        for sq in 0..64 {
            mask[sq] = gen_mask(sq);
            let relevant_bits = mask[sq].count_ones();

            if relevant_bits == 0 {
                shift[sq] = 0;
                magic[sq] = 0;
                tables[sq] = vec![gen_moves(sq, 0)];
                continue;
            }

            shift[sq] = 64 - relevant_bits;
            magic[sq] = Self::find_magic_number(mask[sq], relevant_bits);

            let mut table = vec![0u64; 1 << relevant_bits];

            let mut permutation = mask[sq];

            loop {
                let index = ((permutation.wrapping_mul(magic[sq])) >> shift[sq]) as usize;

                table[index] = gen_moves(sq, permutation);

                if permutation == 0 {
                    break;
                }

                permutation = (permutation - 1) & mask[sq];
            }

            tables[sq] = table;
        }

        Self {
            magic,
            shift,
            mask,
            tables,
        }
    }

    fn lookup(&self, from: Position, occupied: u64) -> u64 {
        let sq = from.index as usize;
        let blockers = occupied & self.mask[sq];
        let index = (blockers.wrapping_mul(self.magic[sq])) >> self.shift[sq];
        self.tables[sq][index as usize]
    }

    fn find_magic_number(mask: u64, relevant_bits: u32) -> u64 {
        let mut rng = RNG { state: 67, inc: 1 };
        let shift = 64 - relevant_bits;

        loop {
            let mut table = vec![false; 1 << relevant_bits];
            let magic = rng.rand64() & rng.rand64() & rng.rand64();

            let mut subset = mask;

            let mut ok = true;

            loop {
                let index = ((subset.wrapping_mul(magic)) >> shift) as usize;

                if table[index] {
                    ok = false;
                    break;
                }

                table[index] = true;

                if subset == 0 {
                    break;
                }

                subset = (subset - 1) & mask;
            }

            if ok {
                return magic;
            }
        }
    }
}
