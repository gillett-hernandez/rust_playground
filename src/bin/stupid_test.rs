struct Input {}

impl Input {
    pub fn is_pressed(&self, action: &str) -> bool {
        true
    }
}

struct PlayerInput {
    pub movement: [f32; 2],
    pub jump_pressed: bool,
    pub dash_pressed: bool,
}

impl PlayerInput {
    pub fn refresh(&mut self, input: &Input) {
        let left_right_dir = if input.is_pressed(&"right") { 1 } else { 0 }
            - if input.is_pressed(&"left") { 1 } else { 0 };
        let up_down_dir = if input.is_pressed(&"down") { 1 } else { 0 }
            - if input.is_pressed(&"up") { 1 } else { 0 };
        self.movement = [left_right_dir as f32, up_down_dir as f32];
        if input.is_pressed(&"dash") {
            self.dash_pressed = true;
        }
    }
}

fn main() {}
