mod board;

use board::*;

fn main() {
    let mut b = Board::new();
    b.sets[Side::Black][Piece::Queen] = 0;

    let mut side = Side::White;
    let mut moves = vec![];

    loop {
        println!("{} to move.", match side { Side::White => "White", Side::Black => "Black" });

        b.dump();
        b.generate_moves(side, &mut moves);

        let lookup = name_moves(&b, &moves);

        for (i, (k, _)) in lookup.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{}", k);
        }

        print!("\n");

        let m = loop {
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();

            if let Some(m) = lookup.get(input.trim()).cloned() {
                break m;
            }

            println!("Invalid move entered.");
        };

        b = b.execute(side, m);

        side = side.other();
    }
}
