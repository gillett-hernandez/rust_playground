use std::f32::consts::PI;

fn length_from_(
    spool_width: f32,
    r_i: f32,
    r_o: f32,
    packing_factor: f32,
    wire_diameter: f32,
) -> f32 {
    4.0 * spool_width * packing_factor * (r_o.powi(2) - r_i.powi(2)) / wire_diameter.powi(2)
}

fn main() {
    let spool_widths: Vec<f32> = vec![0.25, 0.5];
    let outer_radii: Vec<f32> = vec![1.5, 2.0];
    let inner_radii: Vec<f32> = vec![0.75f32 / 2.0, 0.5f32];
    let wire_diameter: f32 = 125.0 / 1000000.0 * 3.28 * 12.0;
    let hexagonal_packing_factor = PI * 3.0f32.sqrt() / 6.0;
    let hexagonal_vertical_packing_factor = 3.0f32.sqrt() / 2.0;

    println!("wire diameter is 125 microns == 0.125mm == 0.000125m");

    for inner_radius in inner_radii.iter() {
        for spool_width in spool_widths.iter() {
            for r_o in outer_radii.iter() {
                println!(
                    "{}\t{}\t{}\t{}",
                    *inner_radius,
                    *r_o,
                    *spool_width,
                    length_from_(
                        *spool_width,
                        *inner_radius,
                        *r_o,
                        hexagonal_packing_factor,
                        wire_diameter
                    ) / 12.0
                        / 5280.0
                );
                println!(
                    "{}\t{}\t{}\t{}\n",
                    *inner_radius,
                    *r_o,
                    *spool_width,
                    length_from_(*spool_width, *inner_radius, *r_o, PI / 4.0, wire_diameter)
                        / 12.0
                        / 5280.0
                );
            }
        }
    }
}
