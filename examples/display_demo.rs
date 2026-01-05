//! Demo of the color-coded octonion display.
//!
//! Run with: cargo run --example display_demo

use octonion::Octonion;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Octonion Display: Fano Plane Color Coding            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Dimensional folding demos
    println!("── Dimensional Folding ──────────────────────────────────────────");
    println!();

    println!("  Zero:       {}", Octonion::ZERO);
    println!("  Real:       {}", Octonion::from(3.14159));
    println!(
        "  Complex:    {}",
        Octonion::new(2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    );
    println!("  Pure i:     {}", Octonion::E1);
    println!(
        "  Quaternion: {}",
        Octonion::new(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
    );
    println!();

    // Basis elements with Fano coloring
    println!("── Basis Elements (Fano Plane Coloring) ─────────────────────────");
    println!();
    println!("  Quaternion subalgebra (RGB primaries):");
    println!(
        "    i = {}    j = {}    k = {}",
        Octonion::E1,
        Octonion::E2,
        Octonion::E3
    );
    println!();
    println!("  Octonion extensions (CMY secondaries):");
    println!(
        "    e₄ = {}    e₅ = {}    e₆ = {}    e₇ = {}",
        Octonion::E4,
        Octonion::E5,
        Octonion::E6,
        Octonion::E7
    );
    println!();

    // Full octonion
    println!("── Full Octonion ────────────────────────────────────────────────");
    println!();
    let x = Octonion::new(1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0);
    println!("  Display: {x}");
    println!("  Debug:   {x:?}");
    println!();

    // Products showing non-commutativity
    println!("── Non-Commutativity ────────────────────────────────────────────");
    println!();
    let e1 = Octonion::E1;
    let e2 = Octonion::E2;
    println!("  e₁ × e₂ = {}", e1 * e2);
    println!("  e₂ × e₁ = {}", e2 * e1);
    println!();

    // Color-free output
    println!("── Plain Output (no ANSI) ───────────────────────────────────────");
    println!();
    println!("  Display: {x:#}");
    println!("  Debug:   {x:#?}");
    println!();

    // The Fano plane legend
    println!("── Fano Plane Legend ────────────────────────────────────────────");
    println!();
    println!(
        "  \x1b[97m●\x1b[0m Real (white)     \x1b[91m●\x1b[0m e₁ (red)      \x1b[92m●\x1b[0m e₂ (green)    \x1b[94m●\x1b[0m e₃ (blue)"
    );
    println!(
        "  \x1b[93m●\x1b[0m e₄ (yellow)      \x1b[96m●\x1b[0m e₅ (cyan)     \x1b[95m●\x1b[0m e₆ (magenta)  \x1b[35m●\x1b[0m e₇ (violet)"
    );
    println!();
    println!("  The colors follow the Cayley-Dickson construction:");
    println!("    • RGB primaries (e₁,e₂,e₃) form the quaternion subalgebra");
    println!("    • CMY secondaries (e₄,e₅,e₆,e₇) are the 'new' octonion directions");
}
