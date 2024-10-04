import Mathlib

namespace triangle_area_l589_589170

noncomputable def line_eq := λ x : ℝ, 9 - 3 * x

theorem triangle_area :
    let x_intercept := (3 : ℝ)
    let y_intercept := (9 : ℝ)
    let triangle_base := x_intercept
    let triangle_height := y_intercept
    let area := (1 / 2) * triangle_base * triangle_height
    area = 13.5 :=
by
    sorry

end triangle_area_l589_589170


namespace find_z_l589_589023

-- Definitions from conditions
def rectangle_area (length width : ℕ) : ℕ := length * width
def square_area (side : ℝ) : ℝ := side^2

-- Problem statement
theorem find_z : 
  ∀ (length width : ℕ) (side z : ℝ),
  rectangle_area length width = 147 →
  square_area side = 147 →
  z = (1/3) * side →
  z = 7 * Real.sqrt 3 / 3 :=
by
  intros length width side z h1 h2 h3
  rw [rectangle_area, square_area] at *
  have side_eq : side = Real.sqrt 147 := by 
    -- proof omitted
    sorry
  rw [side_eq, mul_div_cancel_left (Real.sqrt 147) (by norm_num : (3 : ℝ) ≠ 0)] at h3
  exact h3

end find_z_l589_589023


namespace O_l589_589065

theorem O'Hara_triple_49_16_y : 
  (∃ y : ℕ, (49 : ℕ).sqrt + (16 : ℕ).sqrt = y) → y = 11 :=
by
  sorry

end O_l589_589065


namespace triangle_area_l589_589146

theorem triangle_area (x y : ℝ) (h : 3 * x + y = 9) : 
  (1 / 2) * 3 * 9 = 13.5 :=
sorry

end triangle_area_l589_589146


namespace sector_angle_l589_589281

theorem sector_angle (r l θ : ℝ) (h : 2 * r + l = π * r) : θ = π - 2 :=
sorry

end sector_angle_l589_589281


namespace selection_schemes_count_l589_589010

theorem selection_schemes_count :
  let candidates := {A, B, C, D, E},
      tasks := {taskA, taskB, taskC, taskD},
      restriction : (A ∈ candidates ∧ B ∈ candidates) ∧ (∀ t ∈ tasks, t = taskA ∨ t = taskB → A ∈ t ∨ B ∈ t)
  in ∃ selection_schemes,
     (selection_schemes.count = 18) :=
by 
  let candidates := {A, B, C, D, E},
  let tasks := {taskA, taskB, taskC, taskD},
  let restriction := (A ∈ candidates ∧ B ∈ candidates) ∧ (∀ t ∈ tasks, t = taskA ∨ t = taskB → A ∈ t ∨ B ∈ t),
  have : ∃ selection_schemes, (selection_schemes.count = 18),
  from sorry,
  exact this

end selection_schemes_count_l589_589010


namespace athlete_heartbeats_during_race_l589_589829

theorem athlete_heartbeats_during_race :
  ∀ (heartbeats_per_minute pace_minutes_per_mile race_miles : ℕ ),
    heartbeats_per_minute = 160 →
    pace_minutes_per_mile = 6 →
    race_miles = 30 →
    heartbeats_per_minute * pace_minutes_per_mile * race_miles = 28800 :=
by
  intros heartbeats_per_minute pace_minutes_per_mile race_miles
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end athlete_heartbeats_during_race_l589_589829


namespace circle_passing_three_points_l589_589414

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589414


namespace sin_cos_identity_l589_589386

variable {α β : ℝ} -- Declaring the variables as real numbers

theorem sin_cos_identity (h : sin α ≠ 0) :
  (sin (2 * α + β) / sin α - 2 * cos (α + β)) = (sin β / sin α) :=
by
  sorry

end sin_cos_identity_l589_589386


namespace determine_h_l589_589390

open Polynomial

noncomputable def f (x : ℚ) : ℚ := x^2

theorem determine_h (h : ℚ → ℚ) : 
  (∀ x, f (h x) = 9 * x^2 + 6 * x + 1) ↔ 
  (∀ x, h x = 3 * x + 1 ∨ h x = - (3 * x + 1)) :=
by
  sorry

end determine_h_l589_589390


namespace triangle_area_l589_589144

theorem triangle_area (x y : ℝ) (h : 3 * x + y = 9) : 
  (1 / 2) * 3 * 9 = 13.5 :=
sorry

end triangle_area_l589_589144


namespace equation_of_circle_passing_through_points_l589_589668

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589668


namespace train_distance_proof_l589_589069

-- Definitions
def speed_train1 : ℕ := 40
def speed_train2 : ℕ := 48
def time_hours : ℕ := 8
def initial_distance : ℕ := 892

-- Function to calculate distance after given time
def distance (speed time : ℕ) : ℕ := speed * time

-- Increased/Decreased distance after time
def distance_diff : ℕ := distance speed_train2 time_hours - distance speed_train1 time_hours

-- Final distances
def final_distance_same_direction : ℕ := initial_distance + distance_diff
def final_distance_opposite_direction : ℕ := initial_distance - distance_diff

-- Proof statement
theorem train_distance_proof :
  final_distance_same_direction = 956 ∧ final_distance_opposite_direction = 828 :=
by
  -- The proof is omitted here
  sorry

end train_distance_proof_l589_589069


namespace range_m_area_ADEB_parallel_line_eq_l589_589033

-- Definition of the line equation and conditions
def line_eq (m : ℝ) (x y : ℝ) : Prop := 5 * y + (2 * m - 4) * x - 10 * m = 0

-- Vertices of the rectangle
def O := (0 : ℝ, 0 : ℝ)
def A := (0 : ℝ, 6 : ℝ)
def B := (10 : ℝ, 6 : ℝ)
def C := (10 : ℝ, 0 : ℝ)

-- Intersection on OA (x = 0) and BC (x = 10)
def D (m : ℝ) := (0 : ℝ, 2 * m)
def E (m : ℝ) := (10 : ℝ, 8 - 2 * m)

-- Proving the conditions
theorem range_m (m : ℝ) : 
  (0 ≤ 2 * m ∧ 2 * m ≤ 6) ∧ (0 ≤ 8 - 2 * m ∧ 8 - 2 * m ≤ 6) → 1 ≤ m ∧ m ≤ 3 :=
sorry

-- Area of quadrilateral ADEB is 1/3 of rectangle OABC
theorem area_ADEB (m : ℝ) : 
  let A := (0 : ℝ, 6 : ℝ), D := (0 : ℝ, 2 * m), E := (10 : ℝ, 8 - 2 * m), B := (10 : ℝ, 6 : ℝ) in
  (abs (10 * ((6 - 2 * m) + (2 * m - 2)) / 2) = 20) →
  abs (10 * 6) = 60 →
  (1 / 3) * 60 = 20 :=
sorry

-- Equation of the line parallel to L for equal area quadrilaterals
theorem parallel_line_eq (m : ℝ) : ∃ k b : ℝ, y = k * x + b ∧ 
  k = (4 - 2 * m) / 5 ∧
  b = (2 * m - 2) :=
sorry

end range_m_area_ADEB_parallel_line_eq_l589_589033


namespace equation_of_circle_through_three_points_l589_589579

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589579


namespace circle_equation_l589_589621

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589621


namespace circle_through_points_l589_589583

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589583


namespace total_heartbeats_during_race_l589_589798

-- Definitions for the conditions
def heart_beats_per_minute : ℕ := 160
def pace_in_minutes_per_mile : ℕ := 6
def total_distance_in_miles : ℕ := 30

-- Main theorem statement
theorem total_heartbeats_during_race : 
  heart_beats_per_minute * pace_in_minutes_per_mile * total_distance_in_miles = 28800 :=
by
  -- Place the proof here
  sorry

end total_heartbeats_during_race_l589_589798


namespace no_tiling_possible_l589_589374

open Function

theorem no_tiling_possible (R : Fin 10 → Fin 10) 
  (hRinj : Injective R) : ∃ (B W : Fin 100 → Bool), (∑ i, cond (B i) 1 0) ≠ (∑ i, cond (W i) 1 0) := 
by
  sorry

end no_tiling_possible_l589_589374


namespace circle_through_points_l589_589461

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589461


namespace circle_passing_through_points_eqn_l589_589475

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589475


namespace locus_of_M_is_sphere_l589_589039

noncomputable def sum_squares_distances (M : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := M in
  x^2 + (1 - x)^2 + y^2 + (1 - y)^2 + z^2 + (1 - z)^2

theorem locus_of_M_is_sphere (M : ℝ × ℝ × ℝ) (C : ℝ) :
  sum_squares_distances M = C →
  (∃ k : ℝ, k ≠ 0 ∧ (∃ x y z ∈ Icc (0:ℝ) 1, M = (x, y, z))) →
  ∃ r : ℝ, ∃ c : ℝ × ℝ × ℝ, c = (0.5, 0.5, 0.5) ∧ r = sqrt ((C - 3) / 2) ∧ sum_squares_distances M = C :=
begin
  sorry
end

end locus_of_M_is_sphere_l589_589039


namespace find_x_when_y_is_6750_l589_589704

-- Definitions for the conditions of the problem
def is_positive (a : ℝ) := a > 0

def inversely_proportional (f g : ℝ → ℝ) := ∃ k, ∀ x, f x * g x = k

-- Main statement
theorem find_x_when_y_is_6750 :
  ∀ (x y : ℝ), 
    is_positive x → 
    is_positive y → 
    (∃ k, ∀ x, 3 * x^2 * y = k) → 
    y = 15 → x = 3 →
    y = 6750 →
    x = sqrt 2 / 10 :=
by
  sorry

end find_x_when_y_is_6750_l589_589704


namespace number_of_sides_with_measure_8_l589_589857

variable {PQRSTU : Hexagon}

-- Definitions
def distinct_side_lengths (PQRSTU : Hexagon) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ ∀ s ∈ (PQRSTU.sides), s = a ∨ s = b

def side_PQ (PQRSTU : Hexagon) := (PQRSTU.side 'PQ' = 7)
def side_QR (PQRSTU : Hexagon) := (PQRSTU.side 'QR' = 8)
def perimeter_PQRSTU (PQRSTU : Hexagon) := (PQRSTU.perimeter = 44)

-- Theorem
theorem number_of_sides_with_measure_8 (PQRSTU : Hexagon) :
  distinct_side_lengths PQRSTU →
  side_PQ PQRSTU →
  side_QR PQRSTU →
  perimeter_PQRSTU PQRSTU →
  (PQRSTU.sides.count (8) = 2) :=
by
  sorry

end number_of_sides_with_measure_8_l589_589857


namespace length_of_equal_pieces_l589_589744

theorem length_of_equal_pieces (total_length : ℕ) (num_pieces : ℕ) (num_unequal_pieces : ℕ) (unequal_piece_length : ℕ)
    (equal_pieces : ℕ) (equal_piece_length : ℕ) :
    total_length = 11650 ∧ num_pieces = 154 ∧ num_unequal_pieces = 4 ∧ unequal_piece_length = 100 ∧ equal_pieces = 150 →
    equal_piece_length = 75 :=
by
  sorry

end length_of_equal_pieces_l589_589744


namespace chord_length_AB_tangent_circle_symmetry_l589_589262

-- Define the circle and the line through generic parameters
def circle (a : ℝ) : ℝ → ℝ → Prop :=
λ x y, x^2 + y^2 + 4*x - 4*a*y + 4*a^2 + 1 = 0

def line (a : ℝ) : ℝ → ℝ → Prop :=
λ x y, a*x + y + 2*a = 0

-- Part 1: Length of chord AB when a = 3/2
theorem chord_length_AB (a : ℝ) (h : a = 3/2) :
  ∃ AB : ℝ, AB = 2 * real.sqrt(3 - (6 * real.sqrt(13) / 13)^2) ∧
  ∃ x y,
    circle a x y ∧ line a x y :=
sorry

-- Part 2: Equation of the circle C' when the line l is tangent to the circle C and a > 0
theorem tangent_circle_symmetry (a : ℝ) (h : a > 0) :
  (a ≠ real.sqrt(3)) →
  ∃ x y, circle a x y →
  let C' : ℝ → ℝ → Prop := λ x' y', (x' + 5)^2 + (y' - real.sqrt(3))^2 = 3 in
    ∃ x' y', C' x' y' :=
sorry

end chord_length_AB_tangent_circle_symmetry_l589_589262


namespace trig_identity_simplify_l589_589018

theorem trig_identity_simplify (x y : ℝ) :
  cos (x + y) * sin x - sin (x + y) * cos x = -sin x :=
by
  sorry

end trig_identity_simplify_l589_589018


namespace circle_passing_three_points_l589_589416

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589416


namespace sequence_eventually_congruent_mod_l589_589241

theorem sequence_eventually_congruent_mod (n : ℕ) (hn : n ≥ 1) : 
  ∃ N, ∀ m ≥ N, ∃ k, m = k * n + N ∧ (2^N.succ - 2^k) % n = 0 :=
by
  sorry

end sequence_eventually_congruent_mod_l589_589241


namespace equation_of_circle_through_three_points_l589_589567

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589567


namespace circle_equation_correct_l589_589542

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589542


namespace max_s_value_l589_589359

theorem max_s_value (p q r s : ℝ) (h1 : p + q + r + s = 10) (h2 : (p * q) + (p * r) + (p * s) + (q * r) + (q * s) + (r * s) = 20) :
  s ≤ (5 * (1 + Real.sqrt 21)) / 2 :=
sorry

end max_s_value_l589_589359


namespace heartbeats_during_race_l589_589801

theorem heartbeats_during_race
  (heart_rate : ℕ) -- beats per minute
  (pace : ℕ) -- minutes per mile
  (distance : ℕ) -- miles
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) : 
  heart_rate * (pace * distance) = 28800 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end heartbeats_during_race_l589_589801


namespace circle_through_points_l589_589593

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589593


namespace circle_through_points_l589_589587

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589587


namespace calculate_k_l589_589845

theorem calculate_k (β : ℝ) (hβ : (Real.tan β + 1 / Real.tan β) ^ 2 = k + 1) : k = 1 := by
  sorry

end calculate_k_l589_589845


namespace average_rate_of_change_l589_589892

def f (x : ℝ) : ℝ := 3 * x^2 + 2

theorem average_rate_of_change :
  let x₀ := 2
  let Δx := 0.1
  (f (x₀ + Δx) - f x₀) / Δx = 12.3 :=
by
  let x₀ := 2
  let Δx := 0.1
  have h₁ : f (x₀ + Δx) - f x₀ = 6 * x₀ * Δx + 3 * Δx^2 := sorry
  have h₂ : (6 * x₀ * Δx + 3 * Δx^2) / Δx = 6 * x₀ + 3 * Δx := sorry
  have h₃ : 6 * x₀ + 3 * Δx = 12.3 := sorry
  exact Eq.trans (Eq.trans (Eq.symm (h₁)) (h₂)) (h₃)

end average_rate_of_change_l589_589892


namespace new_volume_correct_l589_589046

-- Define the conditions.
def initial_radius (r : ℝ) : Prop := true
def initial_height (h : ℝ) : Prop := true
def initial_volume (V : ℝ) : Prop := V = 15
def new_radius (r : ℝ) : ℝ := 3 * r
def new_height (h : ℝ) : ℝ := h / 2

-- Define the volume of a cylinder.
def volume_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- State the proof problem.
theorem new_volume_correct (r h : ℝ) (V : ℝ)
  (h1 : initial_radius r)
  (h2 : initial_height h)
  (h3 : initial_volume V) :
  volume_cylinder (new_radius r) (new_height h) = 67.5 :=
by
  sorry

end new_volume_correct_l589_589046


namespace wire_ratio_theorem_l589_589843

theorem wire_ratio_theorem
  {pieces_bonnie : ℕ} {length_piece_bonnie : ℕ} {volume_bonnie : ℕ}
  {pieces_roark : ℕ} {length_piece_roark : ℕ} {volume_roark : ℕ}
  (h_length_bonnie : pieces_bonnie = 12)
  (h_piece_length_bonnie : length_piece_bonnie = 6)
  (h_volume_bonnie : volume_bonnie = 6^3)
  (h_pieces_roark : pieces_roark = volume_bonnie)
  (h_piece_length_roark : length_piece_roark = 12)
  (h_volume_roark : volume_roark = 1) :
  (pieces_bonnie * length_piece_bonnie : ℚ) / (pieces_roark * length_piece_roark : ℚ) = 1 / 36 := 
sorry

end wire_ratio_theorem_l589_589843


namespace principal_amount_l589_589721

theorem principal_amount (r t : ℝ) (H1 : r = 0.12) (H2 : t = 3) : 
  ∃ P : ℝ, (P - (P * r * t) = P - 5888) → P = 9200 :=
by
  -- use the conditions to derive the proof structure
  intro P H
  have : 0.64 * P = 5888 := by
    rw [H1, H2, mul_assoc, mul_comm P r, mul_assoc, sub_eq_iff_eq_add] at H
    exact H
  use 5888 / 0.64
  exact sorry

end principal_amount_l589_589721


namespace percentage_of_second_solution_is_16point67_l589_589305

open Real

def percentage_second_solution (x : ℝ) : Prop :=
  let v₁ := 50
  let c₁ := 0.10
  let c₂ := x / 100
  let v₂ := 200 - v₁
  let c_final := 0.15
  let v_final := 200
  (c₁ * v₁) + (c₂ * v₂) = (c_final * v_final)

theorem percentage_of_second_solution_is_16point67 :
  ∃ x, percentage_second_solution x ∧ x = (50/3) :=
sorry

end percentage_of_second_solution_is_16point67_l589_589305


namespace option_D_functions_same_l589_589089

theorem option_D_functions_same (x : ℝ) : (x^2) = (x^6)^(1/3) :=
by 
  sorry

end option_D_functions_same_l589_589089


namespace circle_passing_three_points_l589_589413

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589413


namespace greatest_difference_units_digit_l589_589699

theorem greatest_difference_units_digit :
  ∃ units_digit : ℕ, (units_digit = 0 ∨ units_digit = 3 ∨ units_digit = 6 ∨ units_digit = 9) ∧
  (∀ d1 d2 : ℕ, (d1 = 0 ∨ d1 = 3 ∨ d1 = 6 ∨ d1 = 9) ∧ (d2 = 0 ∨ d2 = 3 ∨ d2 = 6 ∨ d2 = 9) → abs (d1 - d2) ≤ 9) :=
begin
  use 9,
  split,
  { left, refl },
  { intros d1 d2 h1 h2,
    by_cases h : d1 = 9 ∧ d2 = 0,
    { rw [h.1, h.2], norm_num },
    { rw [abs_sub_le_iff], split,
      norm_num,
      norm_num } }
end

end greatest_difference_units_digit_l589_589699


namespace circle_passing_three_points_l589_589417

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589417


namespace arithmetic_sequence_general_term_l589_589293

theorem arithmetic_sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n - a (n - 1) = 2) : ∀ n, a n = 2 * n - 1 := by
  sorry

end arithmetic_sequence_general_term_l589_589293


namespace equation_of_circle_ABC_l589_589441

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589441


namespace diophantine_solution_unique_l589_589888

open Nat

noncomputable def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
noncomputable def isSolution (p q r s : ℕ) : Prop := |p^r - q^s| = 1 ∧ isPrime p ∧ isPrime q ∧ r > 1 ∧ s > 1
noncomputable def isExpectedSolution (p q r s : ℕ) : Prop := (p = 3 ∧ q = 2 ∧ r = 2 ∧ s = 3) ∨ (p = 2 ∧ q = 3 ∧ r = 3 ∧ s = 2)

theorem diophantine_solution_unique :
  ∀ (p q r s : ℕ), isSolution p q r s → isExpectedSolution p q r s := 
begin
  intros p q r s h,
  sorry
end

end diophantine_solution_unique_l589_589888


namespace seq_b_arithmetic_sum_seq_a_formula_l589_589265

open Classical

-- Definitions of the sequences according to the given conditions
def seq_a : ℕ → ℕ
| 0       := 0
| 1       := 2
| (n + 1) := 2 * (seq_a n) + 3 * 2^(n + 1)

def seq_b (n : ℕ) : ℕ := seq_a n / 2^n

-- Proving that seq_b is arithmetic
theorem seq_b_arithmetic : ∀ (n : ℕ), n > 0 → seq_b (n + 1) - seq_b n = 3 := sorry

-- Defining the sum of the first n terms of seq_a
def sum_seq_a (n : ℕ) : ℕ := (Finset.range n).sum (λ i, seq_a (i + 1))

-- Proving the sum of the first n terms of seq_a
theorem sum_seq_a_formula : ∀ (n : ℕ), sum_seq_a n = (3 * n - 5) * 2^(n + 1) + 10 := sorry

end seq_b_arithmetic_sum_seq_a_formula_l589_589265


namespace triangle_area_bound_by_line_l589_589154

noncomputable def area_of_triangle : ℝ :=
let x_intercept := 3 in
let y_intercept := 9 in
1 / 2 * x_intercept * y_intercept

theorem triangle_area_bound_by_line (x y : ℝ) (h : 3 * x + y = 9) :
  area_of_triangle = 13.5 :=
sorry

end triangle_area_bound_by_line_l589_589154


namespace triangle_area_l589_589175

-- Define the line equation 3x + y = 9
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the triangle bounded by the coordinate axes and the line
def bounded_triangle (x y : ℝ) : Prop :=
  (x = 0 ∧ 0 ≤ y ∧ y ≤ 9) ∨  -- y-axis segment
  (y = 0 ∧ 0 ≤ x ∧ x ≤ 3) ∨  -- x-axis segment
  (3 * x + y = 9 ∧ 0 ≤ x ∧ 0 ≤ y)  -- line segment

-- The area of the triangle formed by the coordinate axes and the line 3x + y = 9
theorem triangle_area : 
  let area := 1/2 * 3 * 9 in -- The calculation of the area
  area = 13.5 := 
by
  sorry

end triangle_area_l589_589175


namespace repeating_decimal_addition_l589_589883

theorem repeating_decimal_addition :
  (let x := (0.444444444... : ℚ) in let y := (0.262626262... : ℚ) in x + y = (70/99 : ℚ)) :=
by sorry

end repeating_decimal_addition_l589_589883


namespace bacterial_colony_growth_l589_589056

theorem bacterial_colony_growth :
  ∀ (n : ℕ), (∀ k : ℕ, k ≥ 40 → n = 1 → n / (1.5 ^ (k - 40)) > 0) →
  ∃ d : ℕ, d ≤ 40 ∧ 10 / 100 ≤ 1 / (1.5 ^ (40 - d)) ∧ 1 / (1.5 ^ (40 - d + 1)) < 10 / 100 :=
begin
  sorry
end

end bacterial_colony_growth_l589_589056


namespace mass_percentage_Na_in_bleach_l589_589897

def molar_mass_NaClO : ℝ := 74.44
def molar_mass_Na : ℝ := 22.99
def percentage_NaClO_in_bleach : ℝ := 5 / 100

theorem mass_percentage_Na_in_bleach : 
  (molar_mass_Na / molar_mass_NaClO) * 100 * percentage_NaClO_in_bleach = 1.5445 :=
by
  sorry

end mass_percentage_Na_in_bleach_l589_589897


namespace cylinder_new_volume_proof_l589_589051

noncomputable def cylinder_new_volume (V : ℝ) (r h : ℝ) : ℝ := 
  let new_r := 3 * r
  let new_h := h / 2
  π * (new_r^2) * new_h

theorem cylinder_new_volume_proof (r h : ℝ) (π : ℝ) 
  (h_volume : π * r^2 * h = 15) : 
  cylinder_new_volume (π * r^2 * h) r h = 67.5 := 
by 
  unfold cylinder_new_volume
  rw [←h_volume]
  sorry

end cylinder_new_volume_proof_l589_589051


namespace frequency_of_sample_data_l589_589916

def sample_data : List ℕ := [125, 120, 122, 105, 130, 114, 116, 95, 120, 134]

def in_range (x : ℕ) : Bool := (114.5 ≤ Float.ofNat x) ∧ (Float.ofNat x < 124.5)

def count_in_range (data : List ℕ) : ℕ :=
  data.countp (λ x => in_range x)

theorem frequency_of_sample_data : 
  (count_in_range sample_data).toFloat / sample_data.length = 0.4 := 
by 
  -- Proof goes here
  sorry

end frequency_of_sample_data_l589_589916


namespace max_smoothie_servings_l589_589132

def servings (bananas yogurt strawberries : ℕ) : ℕ :=
  min (bananas * 4 / 3) (min (yogurt * 4 / 2) (strawberries * 4 / 1))

theorem max_smoothie_servings :
  servings 9 10 3 = 12 :=
by
  -- Proof steps would be inserted here
  sorry

end max_smoothie_servings_l589_589132


namespace triangle_area_l589_589162

theorem triangle_area (x y : ℝ) (h1 : 3 * x + y = 9) (h2 : x = 0 ∨ y = 0) : 
  let base := 3
  let height := 9
  base * height / 2 = 13.5 :=
by
  let base := 3
  let height := 9
  have h_base : base = 3 := rfl
  have h_height : height = 9 := rfl
  calc
    base * height / 2 = 3 * 9 / 2 : by rw [h_base, h_height]
                    ... = 27 / 2   : by norm_num
                    ... = 13.5     : by norm_num

end triangle_area_l589_589162


namespace true_propositions_l589_589784

-- Define the conditions
def proposition1 (m : ℝ) : Prop :=
  m < -2 → ¬(∃ C : ℝ, C = (x^2 / (2-m) + y^2 / (m^2-4)) ∧ (C represents an ellipse))

def proposition2 : Prop :=
  ∃ (P : ℝ × ℝ), (45 * P.1^2 + 20 * P.2^2 = 1) ∧ (right_angled_triangle F1 P F2) → (P has at most 8 intersections)

def proposition3 (m : ℝ) : Prop :=
  (m < 6) → (focal_distance (x^2 / (10-m) + y^2 / (6-m))) = 
  ((5 < m < 9) → (focal_distance (x^2 / (5-m) + y^2 / (9-m))))

def proposition4 (a b : ℝ) : Prop :=
  (a > 0 ∧ b > 0) → (asymptote (y = ± (b / a) * x) → equation_of_hyperbola (x^2 / a^2 - y^2 / b^2 = 1))

def proposition5 (a : ℝ) : Prop :=
  (focus (y = ax^2) = (0, 1/(4*a)))

-- Prove the correct propositions
theorem true_propositions : 
  proposition3 ∧ proposition5 :=
by sorry

end true_propositions_l589_589784


namespace circle_equation_l589_589633

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589633


namespace circle_equation_through_points_l589_589643

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589643


namespace necessary_but_not_sufficient_for_inequality_l589_589736

theorem necessary_but_not_sufficient_for_inequality : 
  ∀ x : ℝ, (-2 < x ∧ x < 4) → (x < 5) ∧ (¬(x < 5) → (-2 < x ∧ x < 4) ) :=
by 
  sorry

end necessary_but_not_sufficient_for_inequality_l589_589736


namespace triangle_area_l589_589173

-- Define the line equation 3x + y = 9
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the triangle bounded by the coordinate axes and the line
def bounded_triangle (x y : ℝ) : Prop :=
  (x = 0 ∧ 0 ≤ y ∧ y ≤ 9) ∨  -- y-axis segment
  (y = 0 ∧ 0 ≤ x ∧ x ≤ 3) ∨  -- x-axis segment
  (3 * x + y = 9 ∧ 0 ≤ x ∧ 0 ≤ y)  -- line segment

-- The area of the triangle formed by the coordinate axes and the line 3x + y = 9
theorem triangle_area : 
  let area := 1/2 * 3 * 9 in -- The calculation of the area
  area = 13.5 := 
by
  sorry

end triangle_area_l589_589173


namespace find_angle_between_vectors_l589_589280

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem find_angle_between_vectors
  (a b : ℝ × ℝ)
  (h_a : magnitude a = 3)
  (h_b : magnitude b = 2)
  (h_sum : magnitude (2 * a.1 + b.1, 2 * a.2 + b.2) = 2 * real.sqrt 13) :
  ∃ θ : ℝ, θ = real.pi / 3 := sorry

end find_angle_between_vectors_l589_589280


namespace hyperbola_line_intersection_unique_points_l589_589296

-- Define the set P
def P : set ℤ := {x | 1 ≤ x ∧ x ≤ 8}

-- Statement of the theorem
theorem hyperbola_line_intersection_unique_points : 
    (∃ m n ∈ P, (mx^2 - ny^2 = 1 ∧ line y = 2x + 1 has exactly one common point) = 3) :=
sorry

end hyperbola_line_intersection_unique_points_l589_589296


namespace equation_of_circle_passing_through_points_l589_589669

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589669


namespace Bulgarian_Selection_2007_l589_589338

-- Definitions of points and geometric properties
variables {Point : Type}
variables (A B C D E F : Point)

-- Definitions of geometric properties and relations
def IsoscelesTriangle (A B C : Point) : Prop := (dist A B = dist A C)
def Midpoint (E : Point) (A C : Point) : Prop := dist A E = dist E C
def SegmentRatio (D : Point) (B C : Point) : Prop := 2 * dist D C = dist B D
def Perpendicular (F D BE: Point): Prop := PerpAt F D BE DF

-- The theorem statement
theorem Bulgarian_Selection_2007 (h1 : IsoscelesTriangle A B C)
	(h2 : Midpoint E A C)
	(h3 : SegmentRatio D B C) 
	(h4 : Perpendicular D F (LineSegment B E)) :
  ∠E F C = ∠A B C := by
sorry

end Bulgarian_Selection_2007_l589_589338


namespace angle_between_a_b_l589_589976

open Real

variables (a b : ℝ^3) (angle : ℝ)

constants (norm_a : ∥a∥ = 1)
          (norm_b : ∥b∥ = 2)
          (perp : dot a (a - b) = 0)

def angle_between_vecs : Prop :=
  angle = π / 3

theorem angle_between_a_b : angle_between_vecs a b angle := by
  sorry

end angle_between_a_b_l589_589976


namespace cost_of_steak_l589_589841

theorem cost_of_steak (paid : ℕ) (received_back : ℕ) (total_pounds : ℕ) (cost_per_pound : ℕ) :
  (paid = 20) → (received_back = 6) → (total_pounds = 2) → (paid - received_back = total_pounds * cost_per_pound) → (cost_per_pound = 7) :=
by
  intros h1 h2 h3 h4
  have h5 : 20 - 6 = 2 * cost_per_pound := by rw [h1, h2, h3, h4]
  sorry

end cost_of_steak_l589_589841


namespace four_thirds_of_nine_halves_l589_589891

theorem four_thirds_of_nine_halves :
  (4 / 3) * (9 / 2) = 6 := 
sorry

end four_thirds_of_nine_halves_l589_589891


namespace circle_equation_correct_l589_589528

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589528


namespace sum_of_powers_of_i_l589_589255

theorem sum_of_powers_of_i :
  (finset.range 101).sum (λ n => complex.I ^ n) = 1 :=
by
  sorry

end sum_of_powers_of_i_l589_589255


namespace common_chord_length_l589_589711

theorem common_chord_length 
  (r : ℝ) (h : r = 12) 
  (d : ℝ) (d_eq_r : d = r) : 
  ∃ chord_length : ℝ, chord_length = 12 * real.sqrt 3 :=
by
  use 12 * real.sqrt 3
  sorry

end common_chord_length_l589_589711


namespace contracting_schemes_count_l589_589853

theorem contracting_schemes_count :
  (nat.choose 6 3) * (nat.choose 3 2) * (nat.choose 1 1) = 60 :=
by
  sorry

end contracting_schemes_count_l589_589853


namespace cos_diff_l589_589735

theorem cos_diff : 
  (cos (Real.pi / 5) = (1 + Real.sqrt 5) / 4) → 
  (cos (2 * Real.pi / 5) = (Real.sqrt 5 - 1) / 4) → 
  cos (Real.pi / 5) - cos (2 * Real.pi / 5) = 1 / 2 :=
by
  intros h1 h2
  rw [h1, h2]
  -- additional steps are omitted
  sorry

end cos_diff_l589_589735


namespace find_m_l589_589849

theorem find_m (C D m : ℤ) (h1 : C = D + m) (h2 : C - 1 = 6 * (D - 1)) (h3 : C = D^3) : m = 0 :=
by sorry

end find_m_l589_589849


namespace quadratic_discriminant_l589_589893

theorem quadratic_discriminant :
  let a := 5
  let b := 5 + (1 / 5)
  let c := - (2 / 5)
  let Δ := b^2 - 4 * a * c
  Δ = 876 / 25 := by
  have : b = 26 / 5 := by norm_num
  have : b^2 = (26 / 5)^2 := by norm_num
  have : 4 * a * c = -8 := by norm_num
  have : b^2 - 4 * a * c = 876 / 25 := by norm_num
  exact this

end quadratic_discriminant_l589_589893


namespace find_number_l589_589113

theorem find_number (x : ℕ) (h : 695 - 329 = x - 254) : x = 620 :=
sorry

end find_number_l589_589113


namespace unique_solution_set_l589_589970

theorem unique_solution_set :
  {a : ℝ | ∃! x : ℝ, (x^2 - 4) / (x + a) = 1} = { -17 / 4, -2, 2 } :=
by sorry

end unique_solution_set_l589_589970


namespace max_intersections_of_line_and_circles_l589_589909

/-
Given: Four coplanar circles.
Prove: There exists a line that touches exactly 8 points on these four circles.
-/

noncomputable def max_points_on_line (C1 C2 C3 C4 : circle) : ℕ :=
  2 * 4

theorem max_intersections_of_line_and_circles (C1 C2 C3 C4 : circle) (h1 : coplanar {C1, C2, C3, C4}) :
  ∃ l : ℝ → ℝ, (∀ t : ℕ, t ∈ {1, 2, 3, 4} → (l /* intersects C_t with exactly two points */)) ∧
  max_points_on_line C1 C2 C3 C4 = 8 :=
begin
  sorry
end

end max_intersections_of_line_and_circles_l589_589909


namespace part_a_part_b_l589_589302

open scoped Classical

-- Define the graph structure and conditions
def prime_graph (p : ℕ) [Fact (Nat.Prime p)] : SimpleGraph (Fin p × Fin p) :=
{ adj := λ ⟨x, y⟩ ⟨x', y'⟩, (x.val * x'.val + y.val * y'.val) % p = 1,
  symm := by
    rintro ⟨x, y⟩ ⟨x', y'⟩ h
    rw [add_comm, mul_comm, mul_comm] at h
    exact h,
  loopless := by
    rintro ⟨x, y⟩ h
    simp only [prod.mk.inj_iff, eq_self_iff_true, and_self] at h
    cases Fact.mk ((Nat.Prime.pred_one _) h) }

-- Part (a) statement
theorem part_a (p : ℕ) [Fact (Nat.Prime p)] : 
  ¬∃ (c : Cycle (Fin p × Fin p)), c.card = 4 ∧ prime_graph p.adj c.toFinset :=
sorry

-- Part (b) statement
theorem part_b : ∃ᶠ (n : ℕ) in atTop, ∃ (G : SimpleGraph (Fin n)), 
  e(G) ≥ n^(3/2) / 2 - n ∧ ¬∃ (c : Cycle (Fin n)), c.card = 4 ∧ G.adj c.toFinset :=
sorry

end part_a_part_b_l589_589302


namespace triangle_area_bound_by_line_l589_589153

noncomputable def area_of_triangle : ℝ :=
let x_intercept := 3 in
let y_intercept := 9 in
1 / 2 * x_intercept * y_intercept

theorem triangle_area_bound_by_line (x y : ℝ) (h : 3 * x + y = 9) :
  area_of_triangle = 13.5 :=
sorry

end triangle_area_bound_by_line_l589_589153


namespace common_three_digit_count_608_reverse_digit_numbers_l589_589987

noncomputable def is_three_digit_base_9 (n : ℕ) : Prop :=
  81 ≤ n ∧ n ≤ 728

noncomputable def is_three_digit_base_11 (n : ℕ) : Prop :=
  121 ≤ n ∧ n ≤ 1330

noncomputable def common_three_digit_numbers : set ℕ :=
  {n | is_three_digit_base_9 n ∧ is_three_digit_base_11 n}

noncomputable def reverse_digits_base_9_to_base_11 (n : ℕ) : Prop :=
  let (a, b, c) := 
      if h : n ≤ 728 then let ⟨a, b, c⟩ := (n / 81, (n % 81) / 9, n % 9) in (a, b, c) else (0, 0, 0);
  let m := c * 121 + b * 11 + a;
  is_three_digit_base_11 m ∧ n = m ∧ m <= 728

theorem common_three_digit_count_608 : 
  finset.card (common_three_digit_numbers.to_finset) = 608 :=
sorry

theorem reverse_digit_numbers :
  ({245, 490} ⊆ {n | reverse_digits_base_9_to_base_11 n}) :=
sorry

end common_three_digit_count_608_reverse_digit_numbers_l589_589987


namespace circle_passes_through_points_l589_589515

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589515


namespace circle_passing_three_points_l589_589407

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589407


namespace circle_passing_three_points_l589_589418

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589418


namespace min_sum_squared_value_l589_589355

noncomputable def min_group_sum_squared (s : Finset ℤ) (h : s = Finset.mk [-8, -6, -4, -1, 1, 3, 5, 14]) : ℤ :=
  let l := s.val.sort (≤)
  match l with
  | [p, q, r, s, t, u, v, w] :=
    let y := p + q + r + s
    let z := t + u + v + w
    if y + z = 4 then y^2 + z^2 else 0
  | _ := 0

theorem min_sum_squared_value :
  ∀ (p q r s t u v w : ℤ), 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
  r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
  s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
  t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
  u ≠ v ∧ u ≠ w ∧
  v ≠ w ∧
  {p, q, r, s, t, u, v, w} = {-8, -6, -4, -1, 1, 3, 5, 14} :=
  (p+q+r+s)^2 + (t+u+v+w)^2 = 8 :=
by
  sorry

end min_sum_squared_value_l589_589355


namespace circle_passing_through_points_l589_589498

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589498


namespace grey_triangle_smallest_angle_l589_589026

-- Defining the basic properties and setup of the problem
section GreyTriangles

variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]

-- Each white triangle is isosceles and right-angled with equal sides of length 1
def white_triangle (T : Type) [metric_space T] (a b c : T) : Prop :=
  (dist a b = 1) ∧ (dist b c = 1) ∧ (dist a c = sqrt 2)

-- The grey region consists of a square with these white triangles placed on it
def grey_square_with_triangles (Sq : Type) [metric_space Sq] :=
  ∃ (V0 V1 V2 V3 : Sq), dist V0 V1 = 2 ∧ dist V1 V2 = 2 ∧ dist V2 V3 = 2 ∧ dist V3 V0 = 2

-- The total area of grey regions is equal to the total area of the white triangles
def areas_equal (white_area grey_area : ℝ) : Prop :=
  white_area = grey_area

-- The smallest angle in each of the identical grey triangles is 15 degrees
def smallest_angle_grey_triangle (angle : ℝ) : Prop :=
  angle = 15

-- The main theorem to prove
theorem grey_triangle_smallest_angle (white_area grey_area : ℝ) (angle : ℝ) :
  (∃ Sq : Type, [metric_space Sq] ∧ grey_square_with_triangles Sq)
  → (white_area = grey_area)
  → (smallest_angle_grey_triangle angle) :=
begin
  intro h1,
  intro h2,
  sorry -- proof goes here
end

end GreyTriangles

end grey_triangle_smallest_angle_l589_589026


namespace librarians_all_work_together_l589_589378

/-- Peter works every 5 days -/
def Peter_days := 5

/-- Quinn works every 8 days -/
def Quinn_days := 8

/-- Rachel works every 10 days -/
def Rachel_days := 10

/-- Sam works every 14 days -/
def Sam_days := 14

/-- Least common multiple of the intervals at which Peter, Quinn, Rachel, and Sam work -/
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem librarians_all_work_together : LCM (LCM (LCM Peter_days Quinn_days) Rachel_days) Sam_days = 280 :=
  by
  sorry

end librarians_all_work_together_l589_589378


namespace total_heartbeats_correct_l589_589818

-- Define the given conditions
def heartbeats_per_minute : ℕ := 160
def pace_per_mile : ℕ := 6
def race_distance : ℕ := 30

-- Define the total heartbeats during the race
def total_heartbeats_during_race : ℕ :=
  pace_per_mile * race_distance * heartbeats_per_minute

-- Theorem stating the mathematically equivalent proof problem
theorem total_heartbeats_correct :
  total_heartbeats_during_race = 28800 :=
by
  -- Proof is omitted
  sorry

end total_heartbeats_correct_l589_589818


namespace circle_equation_through_points_l589_589647

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589647


namespace coneSectorAngle_correct_l589_589117

noncomputable def coneSectorAngle 
  (radius_original : ℝ) (radius_base : ℝ) (volume : ℝ) : Real :=
  let h := (3 * volume) / (π * radius_base^2)
  let l := Real.sqrt (radius_base^2 + h^2)
  let arc_length := 2 * π * radius_base
  let circumference_original := 2 * π * radius_original
  let theta := (arc_length / circumference_original) * 360
  theta

theorem coneSectorAngle_correct :
  coneSectorAngle 18 8 (128 * Real.pi) ≈ 53 :=
by sorry

end coneSectorAngle_correct_l589_589117


namespace measurement_units_correct_l589_589232

structure Measurement (A : Type) where
  value : A
  unit : String

def height_of_desk : Measurement ℕ := ⟨70, "centimeters"⟩
def weight_of_apple : Measurement ℕ := ⟨240, "grams"⟩
def duration_of_soccer_game : Measurement ℕ := ⟨90, "minutes"⟩
def dad_daily_work_duration : Measurement ℕ := ⟨8, "hours"⟩

theorem measurement_units_correct :
  height_of_desk.unit = "centimeters" ∧
  weight_of_apple.unit = "grams" ∧
  duration_of_soccer_game.unit = "minutes" ∧
  dad_daily_work_duration.unit = "hours" :=
by
  sorry

end measurement_units_correct_l589_589232


namespace problem_statement_l589_589947

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (-2^x + b) / (2^(x+1) + a)

theorem problem_statement :
  (∀ (x : ℝ), f (x) 2 1 = -f (-x) 2 1) ∧
  (∀ (t : ℝ), f (t^2 - 2*t) 2 1 + f (2*t^2 - k) 2 1 < 0 → k < -1/3) :=
by
  sorry

end problem_statement_l589_589947


namespace derivative_of_function_l589_589399

theorem derivative_of_function : ∀ x : ℝ, deriv (λ x, 2^x + Real.log 2) x = 2^x * Real.log 2 :=
by
  sorry

end derivative_of_function_l589_589399


namespace increase_in_sold_items_l589_589120

variable (P N M : ℝ)
variable (discounted_price := 0.9 * P)
variable (increased_total_income := 1.17 * P * N)

theorem increase_in_sold_items (h: 0.9 * P * M = increased_total_income):
  M = 1.3 * N :=
  by sorry

end increase_in_sold_items_l589_589120


namespace rostov_survey_min_players_l589_589140

theorem rostov_survey_min_players :
  ∃ m : ℕ, (∀ n : ℕ, n < m → (95 + n * 1) % 100 ≠ 0) ∧ m = 11 :=
sorry

end rostov_survey_min_players_l589_589140


namespace circle_through_points_l589_589595

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589595


namespace elizabeth_stickers_l589_589873

def total_stickers (initial_bottles lost_bottles stolen_bottles stickers_per_bottle : ℕ) : ℕ :=
  let remaining_bottles := initial_bottles - lost_bottles - stolen_bottles
  remaining_bottles * stickers_per_bottle

theorem elizabeth_stickers :
  total_stickers 10 2 1 3 = 21 :=
by
  sorry

end elizabeth_stickers_l589_589873


namespace equal_pieces_length_l589_589747

theorem equal_pieces_length (total_length_cm : ℕ) (num_pieces : ℕ) (num_equal_pieces : ℕ) (length_remaining_piece_mm : ℕ) :
  total_length_cm = 1165 ∧ num_pieces = 154 ∧ num_equal_pieces = 150 ∧ length_remaining_piece_mm = 100 →
  (total_length_cm * 10 - (num_pieces - num_equal_pieces) * length_remaining_piece_mm) / num_equal_pieces = 75 :=
by
  sorry

end equal_pieces_length_l589_589747


namespace circle_passing_through_points_eqn_l589_589488

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589488


namespace largest_two_digit_number_l589_589093

theorem largest_two_digit_number : 
  ∃ x y : ℕ, (x ∈ {1, 2, 4, 6}) ∧ (y ∈ {1, 2, 4, 6}) ∧ (x ≠ y) ∧ 
  (∀ a b : ℕ, (a ∈ {1, 2, 4, 6}) ∧ (b ∈ {1, 2, 4, 6}) ∧ (a ≠ b) → 10 * x + y ≥ 10 * a + b) ∧ 
  (10 * x + y = 64) :=
sorry

end largest_two_digit_number_l589_589093


namespace equation_of_circle_ABC_l589_589449

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589449


namespace circle_passing_through_points_eqn_l589_589480

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589480


namespace heartbeats_during_race_l589_589802

theorem heartbeats_during_race
  (heart_rate : ℕ) -- beats per minute
  (pace : ℕ) -- minutes per mile
  (distance : ℕ) -- miles
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) : 
  heart_rate * (pace * distance) = 28800 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end heartbeats_during_race_l589_589802


namespace cumulus_to_cumulonimbus_ratio_l589_589057

theorem cumulus_to_cumulonimbus_ratio (cirrus cumulonimbus cumulus : ℕ) (x : ℕ)
  (h1 : cirrus = 4 * cumulus)
  (h2 : cumulus = x * cumulonimbus)
  (h3 : cumulonimbus = 3)
  (h4 : cirrus = 144) :
  x = 12 := by
  sorry

end cumulus_to_cumulonimbus_ratio_l589_589057


namespace min_children_l589_589867

theorem min_children (x : ℕ) : 
  (4 * x + 28 - 5 * (x - 1) < 5) ∧ (4 * x + 28 - 5 * (x - 1) ≥ 2) → (x = 29) :=
by
  sorry

end min_children_l589_589867


namespace num_eq_7_times_sum_of_digits_l589_589989

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem num_eq_7_times_sum_of_digits : ∃! n < 1000, n = 7 * sum_of_digits n :=
sorry

end num_eq_7_times_sum_of_digits_l589_589989


namespace solve_for_a_l589_589739

theorem solve_for_a (a x : ℝ) (h : x = 1) (eq : 2 * x + 5 = x + a) : a = 6 :=
by  
  -- Apply the given condition x = 1
  have h₁ : 2 * 1 + 5 = 1 + a, from eq.subst h,
  -- Simplify the equation
  have h₂ : 7 = 1 + a, by linarith,
  -- Solve for a
  linarith

end solve_for_a_l589_589739


namespace least_negative_values_l589_589861

theorem least_negative_values (a b c d : ℤ) (h : 2^a + 2^b = 5^c + 5^d) : 
  ∃ (nonnegative_count : ℕ), nonnegative_count = 4 ∧ 
  (∀ (i : ℤ), i ∈ [a, b, c, d] → i ≥ 0) :=
by
  sorry

end least_negative_values_l589_589861


namespace hollow_cylinder_surface_area_l589_589125

-- Define the given conditions in Lean
def outer_radius : ℝ := 5
def inner_radius : ℝ := 2
def height : ℝ := 12

-- Define the areas involved in Lean
def area_outer_ends := 2 * π * outer_radius ^ 2
def area_inner_ends := 2 * π * inner_radius ^ 2
def net_area_ends := area_outer_ends - area_inner_ends

def outer_lateral_surface_area := 2 * π * outer_radius * height
def inner_lateral_surface_area := 2 * π * inner_radius * height
def total_lateral_surface_area := outer_lateral_surface_area - inner_lateral_surface_area

def total_surface_area := net_area_ends + total_lateral_surface_area

-- The theorem to be proved
theorem hollow_cylinder_surface_area : total_surface_area = 114 * π :=
by
  sorry

end hollow_cylinder_surface_area_l589_589125


namespace construct_triangle_l589_589216

open Real

-- Definitions for the given conditions
structure RightTriangle (A B C : Type) :=
(hypotenuse : ℝ)
(leg_ratio : ℝ)
(right_angle_at : Prop)

def ConstructRightTriangle (c : ℝ) (m n : ℝ) : Type :=
{ T: RightTriangle 
  | T.hypotenuse = c 
  ∧ T.leg_ratio = m / n 
  ∧ T.right_angle_at = true }

-- Lean statement for the problem
theorem construct_triangle (c m n: ℝ) : ∃ (A B C : Type), ConstructRightTriangle c m n :=
sorry

end construct_triangle_l589_589216


namespace circle_equation_correct_l589_589543

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589543


namespace circle_passes_through_points_l589_589525

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589525


namespace tan_α_value_expression_value_l589_589246

noncomputable def alpha : ℝ := sorry  -- α is an unknown angle

axiom α_pos : 0 < alpha
axiom α_lt_half_pi : alpha < real.pi / 2
axiom sin_α_eq_4_5 : real.sin alpha = 4 / 5

theorem tan_α_value : real.tan alpha = 4 / 3 :=
by sorry

theorem expression_value : 
  (real.sin (alpha + real.pi) - 2 * real.cos (real.pi / 2 + alpha)) / 
  (-real.sin (-alpha) + real.cos (real.pi + alpha)) = 4 :=
by sorry

end tan_α_value_expression_value_l589_589246


namespace hyperbola_properties_l589_589960

variable {a b : ℝ}
variable (a_pos : a > 0) (b_pos : b > 0)

def hyperbola_equation (x y : ℝ) := x^2 / a^2 - y^2 / b^2 = 1
def line_equation (x y : ℝ) := x + y - 2 = 0
def F1 := (-2 : ℝ, 0)
def F2 := (2 : ℝ, 0)
def intersects_at (P : (ℝ × ℝ)) := hyperbola_equation a b P.1 P.2 ∧ line_equation P.1 P.2

theorem hyperbola_properties :
  (∃ a b, a > 0 ∧ b > 0 ∧ hyperbola_equation a b = (x^2 - y^2 = 2)) →
  (∃ P, intersects_at P → 
    (∃ k, k = 3 ∧ (3 * x - y - 4 = 0))) :=
begin
  sorry
end

end hyperbola_properties_l589_589960


namespace circle_equation_l589_589627

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589627


namespace circle_passing_through_points_eqn_l589_589481

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589481


namespace circle_passes_through_points_l589_589514

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589514


namespace parabola_and_line_l589_589966

noncomputable def parabola_equation : Prop :=
  ∃ p m : ℝ, p > 0 ∧ m > 1 ∧ (2 * p * m = 4) ∧ (m + p / 2 = 5 / 2) ∧ (y^2 = 2 * x)

noncomputable def line_equation (x3 y3 x4 y4 : ℝ) : Prop :=
  ∀ (y : ℝ), x3 = 2 - 2 * sqrt (3) * y ∨
             x3 = 2 + 2 * sqrt (3) * y

theorem parabola_and_line :
  ∀ (p : ℝ) (m : ℝ) (M N : ℝ × ℝ) (area : ℝ),
  p > 0 ∧ m > 1 ∧
  M = (-2, 0) ∧ N = (2, 0) ∧
  |1 / 2 * 4 * |y1 - y2|| = 16 → 
  parabola_equation ∧ 
  line_equation x3 y3 x4 y4 :=
by
  sorry

end parabola_and_line_l589_589966


namespace elizabeth_stickers_l589_589874

def total_stickers (initial_bottles lost_bottles stolen_bottles stickers_per_bottle : ℕ) : ℕ :=
  let remaining_bottles := initial_bottles - lost_bottles - stolen_bottles
  remaining_bottles * stickers_per_bottle

theorem elizabeth_stickers :
  total_stickers 10 2 1 3 = 21 :=
by
  sorry

end elizabeth_stickers_l589_589874


namespace smallest_45_45_90_triangle_leg_length_l589_589136

theorem smallest_45_45_90_triangle_leg_length : 
  ∀ (h : ℝ), 
  (∀ (h : ℝ), h = 16 → 
  ∃ (l : ℝ), l = (8:ℝ) * Real.sqrt 3 ∧ leg = (8:ℝ) * Real.sqrt 3 / Real.sqrt 2 -> leg = 4 * Real.sqrt 6) := 
by 
  intro h h_cond;
  use (8:ℝ) * Real.sqrt 3;
  use (8:ℝ) * Real.sqrt 3 / Real.sqrt 2;
  sorry

end smallest_45_45_90_triangle_leg_length_l589_589136


namespace proof_problem_l589_589269

noncomputable def p : Prop := ∃ x : ℝ, Real.sin x > 1
noncomputable def q : Prop := ∀ x : ℝ, Real.exp (-x) < 0

theorem proof_problem : ¬ (p ∨ q) :=
by sorry

end proof_problem_l589_589269


namespace cubical_tank_fraction_filled_l589_589097

theorem cubical_tank_fraction_filled (a : ℝ) (h1 : ∀ a:ℝ, (a * a * 1 = 16) )
  : (1 / 4) = (16 / (a^3)) :=
by
  sorry

end cubical_tank_fraction_filled_l589_589097


namespace nonagon_digit_assignment_l589_589003

/--
A regular nonagon $ABCDEFGHI$ with center $J$ and additional point $K$ defined as the midpoint 
of line segment $AJ$. Assign each of the vertices, the center $J$, and the point $K$ 
a unique digit from 1 through 11, such that the sums of the numbers on each of the lines
$AKJ, BKJ, \ldots, IJ$ are all equal. Prove that this can be done in $10321920$ distinct ways.
-/
theorem nonagon_digit_assignment : 
  let digits := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
  ∃ (J K : ℕ) (assign : Fin 9 → ℕ),
  (J ∈ digits) ∧ (K ∈ digits) ∧ (∀ i, assign i ∈ digits) ∧ 
  (∀ i j, i ≠ j → assign i ≠ assign j) ∧ 
  (∀ i, 1 ≤ assign i ∧ assign i ≤ 11) ∧ 
  J ≠ K ∧ 
  (∑ i in (Finset.range 9), assign i) + J + K = 66 ∧ 
  ∀ p q, ∑ d in [K, assign p, J], d = ∑ d in [K, assign q, J] → 
  (∑ d in [K, assign p, J], d = 12) ∧ (∑ d in [K, assign q, J], d = 12) ∧  
  ∃ count : ℕ, count = 10321920
:= 
sorry

end nonagon_digit_assignment_l589_589003


namespace circle_passing_through_points_l589_589607

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589607


namespace triangle_area_l589_589163

theorem triangle_area (x y : ℝ) (h1 : 3 * x + y = 9) (h2 : x = 0 ∨ y = 0) : 
  let base := 3
  let height := 9
  base * height / 2 = 13.5 :=
by
  let base := 3
  let height := 9
  have h_base : base = 3 := rfl
  have h_height : height = 9 := rfl
  calc
    base * height / 2 = 3 * 9 / 2 : by rw [h_base, h_height]
                    ... = 27 / 2   : by norm_num
                    ... = 13.5     : by norm_num

end triangle_area_l589_589163


namespace total_heartbeats_correct_l589_589820

-- Define the given conditions
def heartbeats_per_minute : ℕ := 160
def pace_per_mile : ℕ := 6
def race_distance : ℕ := 30

-- Define the total heartbeats during the race
def total_heartbeats_during_race : ℕ :=
  pace_per_mile * race_distance * heartbeats_per_minute

-- Theorem stating the mathematically equivalent proof problem
theorem total_heartbeats_correct :
  total_heartbeats_during_race = 28800 :=
by
  -- Proof is omitted
  sorry

end total_heartbeats_correct_l589_589820


namespace compare_neg_third_and_neg_point_three_l589_589207

/-- Compare two numbers -1/3 and -0.3 -/
theorem compare_neg_third_and_neg_point_three : (-1 / 3 : ℝ) < -0.3 :=
sorry

end compare_neg_third_and_neg_point_three_l589_589207


namespace probability_interval_l589_589690

-- Define the probability distribution and conditions
def P (xi : ℕ) (c : ℚ) : ℚ := c / (xi * (xi + 1))

-- Given conditions
variables (c : ℚ)
axiom condition : P 1 c + P 2 c + P 3 c + P 4 c = 1

-- Define the interval probability
def interval_prob (c : ℚ) : ℚ := P 1 c + P 2 c

-- Prove that the computed probability matches the expected value
theorem probability_interval : interval_prob (5 / 4) = 5 / 6 :=
by
  -- skip proof
  sorry

end probability_interval_l589_589690


namespace circle_equation_l589_589625

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589625


namespace number_properties_l589_589740

-- Definition of the components
def billions (n : ℕ) : ℕ := n * 10^9
def ten_millions (n : ℕ) : ℕ := n * 10^7
def ten_thousands (n : ℕ) : ℕ := n * 10^4
def thousands (n : ℕ) : ℕ := n * 10^3

-- Calculate the number based on the components
def calculate_number : ℕ :=
  billions 8 + ten_millions 9 + ten_thousands 3 + thousands 2

-- Properties to prove
theorem number_properties :
  calculate_number = 890032000 ∧ 
  calculate_number / 10000 = 89003.2 ∧
  calculate_number / 10^9 ≈ 9 :=
by
  sorry

end number_properties_l589_589740


namespace circle_through_points_l589_589470

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589470


namespace circle_through_points_l589_589436

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589436


namespace problem_l589_589307

theorem problem 
  (x y : ℝ) : 
  (x, y) ∈ {(1, 1), (-1, -1), (1, -1), (-1, 1)} → 
  (x ^ 4 + 1) * (y ^ 4 + 1) = 4 * x ^ 2 * y ^ 2 := 
by
  sorry

end problem_l589_589307


namespace max_intersections_of_line_and_circles_l589_589911

/-
Given: Four coplanar circles.
Prove: There exists a line that touches exactly 8 points on these four circles.
-/

noncomputable def max_points_on_line (C1 C2 C3 C4 : circle) : ℕ :=
  2 * 4

theorem max_intersections_of_line_and_circles (C1 C2 C3 C4 : circle) (h1 : coplanar {C1, C2, C3, C4}) :
  ∃ l : ℝ → ℝ, (∀ t : ℕ, t ∈ {1, 2, 3, 4} → (l /* intersects C_t with exactly two points */)) ∧
  max_points_on_line C1 C2 C3 C4 = 8 :=
begin
  sorry
end

end max_intersections_of_line_and_circles_l589_589911


namespace max_intersections_of_line_with_four_coplanar_circles_l589_589907

theorem max_intersections_of_line_with_four_coplanar_circles
  (C1 C2 C3 C4 : ℝ) -- Four coplanar circles
  (h1 : is_circle C1)
  (h2 : is_circle C2)
  (h3 : is_circle C3)
  (h4 : is_circle C4)
  : ∃ l : ℝ → ℝ, ∀ C ∈ {C1, C2, C3, C4}, (number_of_intersections l C ≤ 2) → 
    (number_of_intersections l C1 + number_of_intersections l C2 + number_of_intersections l C3 + number_of_intersections l C4) ≤ 8 := 
sorry

end max_intersections_of_line_with_four_coplanar_circles_l589_589907


namespace circle_equation_l589_589629

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589629


namespace circle_passing_through_points_l589_589610

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589610


namespace circle_equation_correct_l589_589535

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589535


namespace prob_three_friends_same_group_l589_589840

theorem prob_three_friends_same_group :
  let students := 800
  let groups := 4
  let group_size := students / groups
  let p_same_group := 1 / groups
  p_same_group * p_same_group = 1 / 16 := 
by
  sorry

end prob_three_friends_same_group_l589_589840


namespace part1_part2_l589_589952

theorem part1 (a : ℝ) (h : 48 * a^2 = 75) (ha : a > 0) : a = 5 / 4 :=
sorry

theorem part2 (θ : ℝ) 
  (h₁ : 10 * (Real.sin θ) ^ 2 = 5) 
  (h₀ : 0 < θ ∧ θ < Real.pi / 2) 
  : θ = Real.pi / 4 :=
sorry

end part1_part2_l589_589952


namespace max_intersections_convex_polygons_l589_589866

theorem max_intersections_convex_polygons (P1 P2 : Type) [convex_polygon P1] [convex_polygon P2]
  (n1 n2 : ℕ) (h1 : polygon_sides P1 = n1) (h2 : polygon_sides P2 = n2) (h : n1 ≤ n2)
  (no_overlap : ∀ (e1 : edge P1) (e2 : edge P2), ¬ (e1.segment.overlaps e2.segment)) :
  max_intersection_points P1 P2 = 2 * n1 :=
sorry

end max_intersections_convex_polygons_l589_589866


namespace circle_equation_through_points_l589_589644

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589644


namespace limit_S_n_div_n_sq_l589_589361

def A (n : ℕ) : Set ℝ := { x | ∃ i : ℕ, 0 ≤ i ∧ i < n ∧ x = 1 / 2^i }

def S (n : ℕ) : ℝ := 
  ∑ (i j k : ℕ) in finset.univ.powersetLen 3, 
  (1 / 2^i + 1 / 2^j + 1 / 2^k : ℝ)

theorem limit_S_n_div_n_sq :
  tendsto (fun n => S n / n^2) atTop (𝓝 1) :=
sorry

end limit_S_n_div_n_sq_l589_589361


namespace positive_difference_eq_six_l589_589060

theorem positive_difference_eq_six (x y : ℝ) (h1 : x + y = 8) (h2 : x ^ 2 - y ^ 2 = 48) : |x - y| = 6 := by
  sorry

end positive_difference_eq_six_l589_589060


namespace triangle_area_range_of_a_l589_589289

-- Part (1)
def f (a x : ℝ) : ℝ := a * real.exp (x - 1) - real.log x + real.log a

theorem triangle_area (x : ℝ) (a : ℝ) (ha : a = real.exp 1) : 
  let f_x : ℝ := f a x,
      tangent_line := fun y => y = (real.exp 1 - 1) * (x - 1) + (real.exp 1 + 1)  in
  (∃ x_inter y_inter : ℝ, tangent_line 0 = y_inter ∧ tangent_line x_inter = 0 ∧ 
  1/2 * real.abs x_inter * real.abs y_inter = 2 / (real.exp 1 - 1)) := 
sorry

-- Part (2)
def g (x : ℝ) (a : ℝ) : ℝ := a * real.exp (x - 1) - real.log x + real.log a

theorem range_of_a (x a : ℝ) (h : g x a ≥ 1) : 1 ≤ a := 
sorry

end triangle_area_range_of_a_l589_589289


namespace count_numbers_seven_times_sum_of_digits_l589_589991

open Nat

-- Function to calculate sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  ((n.digits 10).sum)

theorem count_numbers_seven_times_sum_of_digits :
  { n : ℕ // n > 0 ∧ n < 1000 ∧ (n = 7 * sum_of_digits n) }.card = 4 :=
by
  -- Proof would go here
  sorry

end count_numbers_seven_times_sum_of_digits_l589_589991


namespace greatest_fraction_l589_589933

theorem greatest_fraction 
  (w x y z : ℕ)
  (hw : w > 0)
  (h_ordering : w < x ∧ x < y ∧ y < z) :
  (x + y + z) / (w + x + y) > (w + x + y) / (x + y + z) ∧
  (x + y + z) / (w + x + y) > (w + y + z) / (x + w + z) ∧
  (x + y + z) / (w + x + y) > (x + w + z) / (w + y + z) ∧
  (x + y + z) / (w + x + y) > (y + z + w) / (x + y + z) :=
sorry

end greatest_fraction_l589_589933


namespace compare_abc_l589_589252

noncomputable def a := 1.7^(-2.5)
noncomputable def b := 2.5^(1.7)
noncomputable def c := log 2 (2/3)

theorem compare_abc : c < a ∧ a < b := by
  sorry

end compare_abc_l589_589252


namespace heartbeats_during_race_l589_589800

theorem heartbeats_during_race
  (heart_rate : ℕ) -- beats per minute
  (pace : ℕ) -- minutes per mile
  (distance : ℕ) -- miles
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) : 
  heart_rate * (pace * distance) = 28800 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end heartbeats_during_race_l589_589800


namespace elizabeth_stickers_l589_589878

def initial_bottles := 10
def lost_at_school := 2
def lost_at_practice := 1
def stickers_per_bottle := 3

def total_remaining_bottles := initial_bottles - lost_at_school - lost_at_practice
def total_stickers := total_remaining_bottles * stickers_per_bottle

theorem elizabeth_stickers : total_stickers = 21 :=
  by
  unfold total_stickers total_remaining_bottles initial_bottles lost_at_school lost_at_practice stickers_per_bottle
  simp
  sorry

end elizabeth_stickers_l589_589878


namespace circle_equation_correct_l589_589539

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589539


namespace proof_problem_l589_589282

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^x - 1) / (a^x + 1)
def g (x : ℝ) : ℝ := 3^x

theorem proof_problem :
  ∃ a : ℝ, a > 1 ∧ g (a + 2) = 81 ∧
  (∃ f_odd : ∀ x : ℝ, f a (-x) = -f a x) ∧
  (∃ f_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2) ∧
  (∃ f_range : ∀ y : ℝ, (f a y ∈ set.Ioo (-1) 1)) :=
begin
  sorry
end

end proof_problem_l589_589282


namespace circle_passing_through_points_l589_589612

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589612


namespace circle_equation_l589_589622

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589622


namespace parallelogram_altitude_DF_l589_589334

theorem parallelogram_altitude_DF
  (ABCD : parallelogram)
  (DC EB DE DF : ℝ)
  (hDC : DC = 15)
  (hEB : EB = 5)
  (hDE : DE = 9)
  (hAB_DC : ABCD.oppositeSidesEqual)
  : DF = 9 :=
sorry

end parallelogram_altitude_DF_l589_589334


namespace find_f_f_2_l589_589928

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^3 else 1 - x

theorem find_f_f_2 : f (f 2) = -1 :=
by
  sorry

end find_f_f_2_l589_589928


namespace sum_of_distances_eq_2023_l589_589270

/-- Points on the parabola y^2 = 4x with given x-coordinates -/
variables {P : ℕ → ℝ × ℝ} (hP : ∀ i, (P i).fst = 4 * (P i).snd^2)

/-- Focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- Given points P_1, P_2, ..., P_2013 on the parabola y^2 = 4x -/
noncomputable def points : fin 2013 → ℝ × ℝ := λ i, (classical.some (classical.arbitrary ℝ), 2 * (i : ℝ))

/-- Sum of x-coordinates -/
variable (hx_sum : finset.sum finset.univ (λ i : fin 2013, (points i).fst) = 10)

/-- Distance between two points in ℝ² -/
noncomputable def distance (a b : ℝ × ℝ) : ℝ := real.sqrt ((a.fst - b.fst)^2 + (a.snd - b.snd)^2)

/-- Proving the sum of the distances of the points to the focus equals 2023 -/
theorem sum_of_distances_eq_2023 :
  finset.sum finset.univ (λ i : fin 2013, distance (points i) focus) = 2023 :=
sorry

end sum_of_distances_eq_2023_l589_589270


namespace equation_of_circle_passing_through_points_l589_589662

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589662


namespace tangent_and_minimum_area_l589_589196

-- Definitions used in the problem
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
def focus (p : ℝ) := (p / 2, 0)
def directrix (p : ℝ) := -p / 2
def diameter {A B : ℝ × ℝ} := (A.1 + B.1) / 2

-- Given proof statement in Lean 4
theorem tangent_and_minimum_area (p : ℝ) (x y : ℝ) (A B : ℝ × ℝ) (Q : ℝ × ℝ) :
  0 < p →
  parabola p A.1 A.2 →
  parabola p B.1 B.2 →
  (let mid := diameter A B in
   mid = directrix p) →
  ∃ P : ℝ × ℝ, 
    tangent (circle_with_diameter A B) (line_through P) directrix p ∧
    min_area_triangle A B Q = -11 / 5 :=
sorry

end tangent_and_minimum_area_l589_589196


namespace octagon_area_l589_589712

noncomputable def area_of_octagon_concentric_squares : ℚ :=
  let m := 1
  let n := 8
  (m + n)

theorem octagon_area (O : ℝ × ℝ) (side_small side_large : ℚ) (AB : ℚ) 
  (h1 : side_small = 2) (h2 : side_large = 3) (h3 : AB = 1/4) : 
  area_of_octagon_concentric_squares = 9 := 
  by
  have h_area : 1/8 = 1/8 := rfl
  sorry

end octagon_area_l589_589712


namespace circle_through_points_l589_589586

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589586


namespace find_number_l589_589365

def sum : ℕ := 2468 + 1375
def diff : ℕ := 2468 - 1375
def first_quotient : ℕ := 3 * diff
def second_quotient : ℕ := 5 * diff
def remainder : ℕ := 150

theorem find_number (N : ℕ) (h1 : sum = 3843) (h2 : diff = 1093) 
                    (h3 : first_quotient = 3279) (h4 : second_quotient = 5465)
                    (h5 : remainder = 150) (h6 : N = sum * first_quotient + remainder)
                    (h7 : N = sum * second_quotient + remainder) :
  N = 12609027 := 
by 
  sorry

end find_number_l589_589365


namespace circle_equation_through_points_l589_589639

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589639


namespace simplify_expr1_simplify_expr2_l589_589020

-- Define the variables a and b
variables (a b : ℝ)

-- First problem: simplify 2a^2 - 3a^3 + 5a + 2a^3 - a^2 to a^2 - a^3 + 5a
theorem simplify_expr1 : 2*a^2 - 3*a^3 + 5*a + 2*a^3 - a^2 = a^2 - a^3 + 5*a :=
  by sorry

-- Second problem: simplify (2 / 3) (2 * a - b) + 2 (b - 2 * a) - 3 (2 * a - b) - (4 / 3) (b - 2 * a) to -6 * a + 3 * b
theorem simplify_expr2 : 
  (2 / 3) * (2 * a - b) + 2 * (b - 2 * a) - 3 * (2 * a - b) - (4 / 3) * (b - 2 * a) = -6 * a + 3 * b :=
  by sorry

end simplify_expr1_simplify_expr2_l589_589020


namespace circle_through_points_l589_589433

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589433


namespace mod_arithmetic_l589_589899

theorem mod_arithmetic :
  (172 * 15 - 13 * 8 + 6) % 12 = 10 := 
by
  -- Expanding the numbers according to the given conditions
  have h1 : 172 % 12 = 4 := sorry,
  have h2 : 13 % 12 = 1 := sorry,
  -- Result obtained from the combination
  sorry

end mod_arithmetic_l589_589899


namespace circle_with_diameter_l589_589027

variable (A : ℝ × ℝ) (B : ℝ × ℝ)
-- Define the points A and B as (1, 4) and (3, -2) respectively
def pointA : ℝ × ℝ := (1, 4)
def pointB : ℝ × ℝ := (3, -2)

-- Define the midpoint of A and B
def midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the equation of a circle given the center and radius
def circle_equation (center : ℝ × ℝ) (radius : ℝ) : ℝ → ℝ → Prop :=
  λ x y, (x - center.1)^2 + (y - center.2)^2 = radius^2

-- State the proof problem in Lean
theorem circle_with_diameter (A B : ℝ × ℝ)
  (hA : A = (1, 4)) (hB : B = (3, -2)) :
  ∃ eqn, eqn = circle_equation (midpoint A B) (distance (midpoint A B) A) 
  ∧ eqn = circle_equation (2, 1) (Real.sqrt 10) := by
  sorry

end circle_with_diameter_l589_589027


namespace circle_passing_through_points_l589_589500

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589500


namespace triangle_area_l589_589158

theorem triangle_area (x y : ℝ) (h1 : 3 * x + y = 9) (h2 : x = 0 ∨ y = 0) : 
  let base := 3
  let height := 9
  base * height / 2 = 13.5 :=
by
  let base := 3
  let height := 9
  have h_base : base = 3 := rfl
  have h_height : height = 9 := rfl
  calc
    base * height / 2 = 3 * 9 / 2 : by rw [h_base, h_height]
                    ... = 27 / 2   : by norm_num
                    ... = 13.5     : by norm_num

end triangle_area_l589_589158


namespace circle_passing_three_points_l589_589419

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589419


namespace largest_four_digit_sum_19_l589_589078

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digits_sum_to (n : ℕ) (sum : ℕ) : Prop := (n.digits 10).sum = sum

theorem largest_four_digit_sum_19 : ∃ n : ℕ, is_four_digit n ∧ digits_sum_to n 19 ∧ ∀ m : ℕ, is_four_digit m ∧ digits_sum_to m 19 → n ≥ m :=
by
  use 9730
  split
  · exact sorry
  · split
    · exact sorry
    · intros m hm
      exact sorry

end largest_four_digit_sum_19_l589_589078


namespace eccentricity_hyperbola_proof_l589_589961

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : (2,0) is the focus of the hyperbola) (h₄ : ∀ P, P is an intersection point and | P - (2,0) | = 4) :=
  (∃ (e : ℝ), e = sqrt 2 + 1)

theorem eccentricity_hyperbola_proof :
  ∀ (a b : ℝ), a > 0 → b > 0 → 
  (∃ (P : ℝ × ℝ), P is intersection point of hyperbola and parabola and |PF| = 4) →
  (∃ e : ℝ, e = sqrt 2 + 1) :=
by
  intros a b ha hb hint
  use (sqrt 2 + 1)
  sorry

end eccentricity_hyperbola_proof_l589_589961


namespace circle_through_points_l589_589425

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589425


namespace circle_equation_through_points_l589_589653

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589653


namespace magnitude_product_l589_589229

theorem magnitude_product (z1 z2 : ℂ) (h1 : z1 = 3 * Real.sqrt 5 - 6 * Complex.I) (h2 : z2 = 2 * Real.sqrt 2 + 4 * Complex.I) :
  Complex.abs (z1 * z2) = 18 * Real.sqrt 6 :=
by
  rw [Complex.abs_mul, h1, h2]
  sorry

end magnitude_product_l589_589229


namespace new_volume_is_correct_l589_589043

noncomputable def original_volume : ℝ := 15
def original_radius (r : ℝ) : ℝ := r
def original_height (h : ℝ) : ℝ := h
def new_radius (r : ℝ) : ℝ := 3 * r
def new_height (h : ℝ) : ℝ := h / 2

theorem new_volume_is_correct (r h : ℝ) (V := π * r^2 * h) (V' := π * (3 * r)^2 * (h / 2)) :
  V = original_volume →
  V' = (9 / 2) * V →
  V' = 67.5 :=
by
  intros hVr hV'_eq
  have hV : V = 15 := hVr
  have hV_15 : V' = (9 / 2) * 15 := calc
    V' = (9 / 2) * V     := hV'_eq
    ... = (9 / 2) * 15 = 67.5
  exact eq.trans hV_15 rfl

end new_volume_is_correct_l589_589043


namespace x_in_M_sufficient_condition_for_x_in_N_l589_589366

def M := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x < 0}
def N := {y : ℝ | ∃ x : ℝ, y = Real.sqrt ((1 - x) / x)}

theorem x_in_M_sufficient_condition_for_x_in_N :
  (∀ x, x ∈ M → x ∈ N) ∧ ¬ (∀ x, x ∈ N → x ∈ M) :=
by sorry

end x_in_M_sufficient_condition_for_x_in_N_l589_589366


namespace find_circle_equation_l589_589548

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589548


namespace gcd_217_155_l589_589678

theorem gcd_217_155 : Nat.gcd 217 155 = 1 := by
  sorry

end gcd_217_155_l589_589678


namespace man_time_to_complete_round_l589_589692

noncomputable def time_to_complete_round (L B : ℝ) (area : ℝ) (speed : ℝ) : ℝ :=
  if h : B = 2 * L ∧ L * B = area then
    let perimeter := 2 * L + 2 * B
    let perimeter_km := perimeter / 1000
    let time_hr := perimeter_km / speed
    time_hr * 60  -- convert time to minutes
  else 0

theorem man_time_to_complete_round : 
  ∀ (L B : ℝ), (B = 2 * L) → (L * B = 20000) → (time_to_complete_round L B 20000 6 = 6) :=
by 
  intros L B h_ratio h_area
  unfold time_to_complete_round
  split_ifs
  apply if_pos
  exact ⟨h_ratio, h_area⟩
  sorry

end man_time_to_complete_round_l589_589692


namespace B_squared_value_l589_589886

-- Definition of the function g
def g (x : ℝ) := real.sqrt 45 + 105 / x

-- Definition of the quadratic equation whose roots we need
def quadratic_eq (x : ℝ) := x^2 - x * real.sqrt 45 - 105 = 0

-- Definition of the sum of the absolute values of the roots
def sum_abs_roots := (| (real.sqrt 45 + real.sqrt 465) / 2 |) + (| (real.sqrt 45 - real.sqrt 465) / 2 |)

-- The main statement verifying the value of B^2
theorem B_squared_value : (sum_abs_roots)^2 = 465 :=
by
  sorry

end B_squared_value_l589_589886


namespace duration_of_Bs_money_use_l589_589727

theorem duration_of_Bs_money_use (C : ℝ) : 
    (∀ A_contrib_time B_profit_fraction : ℝ,
     A_contrib_time = 15 -> 
     A_contrib_time > 0 -> -- this ensures positive time period
     B_profit_fraction = 2/3 -> 
     B_profit_fraction > 0 -> -- this ensures positive profit fraction
     ∃ B_contrib_time : ℝ, B_contrib_time = 7.5) := 
by {
  intros A_contrib_time B_profit_fraction hA_contrib_time hA_gt_zero hB_profit_fraction hB_gt_zero,
  use 7.5,
  sorry
}

end duration_of_Bs_money_use_l589_589727


namespace triangle_area_l589_589148

theorem triangle_area (x y : ℝ) (h : 3 * x + y = 9) : 
  (1 / 2) * 3 * 9 = 13.5 :=
sorry

end triangle_area_l589_589148


namespace find_circle_equation_l589_589553

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589553


namespace angle_C_sides_sum_l589_589353

-- Definitions of vectors m and n
def vec_m (C : ℝ) := (Real.cos (C / 2), Real.sin (C / 2))
def vec_n (C : ℝ) := (Real.cos (C / 2), - Real.sin (C / 2))

-- Definition for the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The magnitude of angle C is π/3
theorem angle_C :
  ∀ C : ℝ, dot_product (vec_m C) (vec_n C) = Real.cos C → C = Real.pi / 3 :=
by
  sorry

-- The value of a + b is 11/2 given the conditions
theorem sides_sum (a b c S : ℝ) (h_c : c = 7 / 2) (h_S : S = 3 * Real.sqrt 3 / 2)
  (h_area : S = 1 / 2 * a * b * Real.sin (Real.pi / 3)) :
  a + b = 11 / 2 :=
by
  sorry

end angle_C_sides_sum_l589_589353


namespace circle_through_points_l589_589427

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589427


namespace baskets_weight_l589_589869

theorem baskets_weight 
  (weight_per_basket : ℕ)
  (num_baskets : ℕ)
  (total_weight : ℕ) 
  (h1 : weight_per_basket = 30)
  (h2 : num_baskets = 8)
  (h3 : total_weight = weight_per_basket * num_baskets) :
  total_weight = 240 := 
by
  sorry

end baskets_weight_l589_589869


namespace length_of_goods_train_l589_589768

/-- The length of the goods train is 320 meters given the conditions:
  - The man's train travels at 55 kmph.
  - The goods train travels in the opposite direction at 60.2 kmph.
  - The goods train takes 10 seconds to pass the man. --/
theorem length_of_goods_train :
  ∀ (speed_man_train speed_goods_train : ℝ) (time_pass : ℝ),
  speed_man_train = 55 →
  speed_goods_train = 60.2 →
  time_pass = 10 →
  let relative_speed := (speed_man_train + speed_goods_train) * (1000 / 3600) in
  let length_goods_train := relative_speed * time_pass in
  length_goods_train = 320 :=
by
  intros speed_man_train speed_goods_train time_pass h1 h2 h3
  let relative_speed := (speed_man_train + speed_goods_train) * (1000 / 3600)
  let length_goods_train := relative_speed * time_pass
  sorry

end length_of_goods_train_l589_589768


namespace average_speed_l589_589191

def total_distance : ℝ := 200
def total_time : ℝ := 40

theorem average_speed (d t : ℝ) (h₁: d = total_distance) (h₂: t = total_time) : d / t = 5 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end average_speed_l589_589191


namespace equal_piece_length_l589_589742

/-- A 1165 cm long rope is cut into 154 pieces, 150 of which are equally sized, and the remaining pieces are 100mm each.
    This theorem proves that the length of each equally sized piece is 75mm. -/
theorem equal_piece_length (total_length_cm : ℕ) (total_pieces : ℕ) (equal_pieces : ℕ) (remaining_piece_length_mm : ℕ) 
  (total_length_mm : ℕ) (remaining_pieces : ℕ) (equal_length_mm : ℕ) : 
  total_length_cm = 1165 ∧ 
  total_pieces = 154 ∧  
  equal_pieces = 150 ∧
  remaining_piece_length_mm = 100 ∧
  total_length_mm = total_length_cm * 10 ∧
  remaining_pieces = total_pieces - equal_pieces ∧ 
  equal_length_mm = (total_length_mm - remaining_pieces * remaining_piece_length_mm) / equal_pieces →
  equal_length_mm = 75 :=
by
  sorry

end equal_piece_length_l589_589742


namespace find_circle_equation_l589_589550

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589550


namespace rational_solves_abs_eq_l589_589905

theorem rational_solves_abs_eq (x : ℚ) : |6 + x| = |6| + |x| → 0 ≤ x := 
sorry

end rational_solves_abs_eq_l589_589905


namespace part_b_part_c_l589_589728

-- Definitions for the problem conditions
def board4x4 := Fin 4 × Fin 4
def board8x8 := Fin 8 × Fin 8

-- 4x4 board problem (part b)
theorem part_b:
  ∃ (choices: Finset board4x4), 
    (∀ i : Fin 4, (choices.filter (λ pos, pos.1 = i)).card = 3) ∧ 
    (∀ j : Fin 4, (choices.filter (λ pos, pos.2 = j)).card = 3) →
    (Finset.univ.card * 3 = 24) :=
sorry

-- 8x8 board problem (part c)
theorem part_c:
  ∃ (black_squares white_squares: Finset board8x8),
    (black_squares.filter (λ pos, (pos.1 + pos.2) % 2 = 0) ∧ 
    pos ∉ white_squares = Finset.univ.filter (λ pos, (pos.1 + pos.2) % 2 = 0)).card = 32 ∧
    (white_squares.filter (λ pos, (pos.1 + pos.2) % 2 ≠ 0)).card = 24 ∧
    (∀ i : Fin 8, (white_squares.filter (λ pos, pos.1 = i)).card = 3) ∧
    (∀ j : Fin 8, (white_squares.filter (λ pos, pos.2 = j)).card = 3) →
    ((Finset.univ.card // 2)^2 = 576) :=
sorry

end part_b_part_c_l589_589728


namespace planes_parallel_l589_589972

variables (α β : Plane) (l m : Line)

-- All conditions from the problem
axiom l_subset_α : l ⊆ α
axiom l_parallel_β : l ∥ β
axiom m_subset_β : m ⊆ β
axiom m_parallel_α : m ∥ α
axiom skew_lines : skew l m

-- The main theorem statement
theorem planes_parallel (α β : Plane) (l m : Line) 
  (h1 : l ⊆ α) 
  (h2 : l ∥ β) 
  (h3 : m ⊆ β) 
  (h4 : m ∥ α) 
  (h5 : skew l m) : 
  α ∥ β :=
by
  sorry

end planes_parallel_l589_589972


namespace correct_option_l589_589723

-- Definitions based on the conditions of the problem
def exprA (a : ℝ) : Prop := 7 * a + a = 7 * a^2
def exprB (x y : ℝ) : Prop := 3 * x^2 * y - 2 * x^2 * y = x^2 * y
def exprC (y : ℝ) : Prop := 5 * y - 3 * y = 2
def exprD (a b : ℝ) : Prop := 3 * a + 2 * b = 5 * a * b

-- Proof problem statement verifying the correctness of the given expressions
theorem correct_option (x y : ℝ) : exprB x y :=
by
  -- (No proof is required, the statement is sufficient)
  sorry

end correct_option_l589_589723


namespace find_z_l589_589295

def is_intersection (M N : Set ℂ) (x : ℂ) : Prop := x ∈ M ∧ x ∈ N

def M : Set ℂ := {1, 2, (4:ℂ)*Complex.i}
def N : Set ℂ := {3, 4}
def i := Complex.i

theorem find_z : ∃ z : ℂ, z = 4 * i ∧ is_intersection M N (4 : ℂ) :=
by
  sorry

end find_z_l589_589295


namespace volume_of_cone_divided_by_pi_l589_589119

-- Define the conditions
def cone_from_sector (radius : ℝ) (sector_angle : ℝ) : Prop :=
  sector_angle = 270 ∧ radius = 20

-- Mathematical proof problem statement
theorem volume_of_cone_divided_by_pi (radius : ℝ) (sector_angle : ℝ) 
  (h : cone_from_sector radius sector_angle) :
  (∃ (V : ℝ), V / real.pi = 1125 * real.sqrt 7) :=
sorry

end volume_of_cone_divided_by_pi_l589_589119


namespace circle_passing_through_points_eqn_l589_589490

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589490


namespace colton_stickers_left_l589_589852

theorem colton_stickers_left :
  let C := 72
  let F := 4 * 3 -- stickers given to three friends
  let M := F + 2 -- stickers given to Mandy
  let J := M - 10 -- stickers given to Justin
  let T := F + M + J -- total stickers given away
  C - T = 42 := by
  sorry

end colton_stickers_left_l589_589852


namespace circle_passing_three_points_l589_589405

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589405


namespace max_at_pi_six_l589_589674

theorem max_at_pi_six : ∃ (x : ℝ), (0 ≤ x ∧ x ≤ π / 2) ∧ (∀ y, (0 ≤ y ∧ y ≤ π / 2) → (x + 2 * Real.cos x) ≥ (y + 2 * Real.cos y)) ∧ x = π / 6 := sorry

end max_at_pi_six_l589_589674


namespace dilation_point_movement_l589_589121

theorem dilation_point_movement :
  let B := (4, -3)
  let r_orig := 4
  let B' := (-2, 9)
  let r_dil := 6
  let P := (1, 1)
  let center_dil := (-32, 39)
  let k := (r_dil : ℝ) / r_orig
  let d0 := real.sqrt ((-32 - P.1)^2 + (39 - P.2)^2)
  let d1 := k * d0
in d1 - d0 = 0.5 * real.sqrt 2533 :=
by {
  sorry
}

end dilation_point_movement_l589_589121


namespace equal_pieces_length_l589_589748

theorem equal_pieces_length (total_length_cm : ℕ) (num_pieces : ℕ) (num_equal_pieces : ℕ) (length_remaining_piece_mm : ℕ) :
  total_length_cm = 1165 ∧ num_pieces = 154 ∧ num_equal_pieces = 150 ∧ length_remaining_piece_mm = 100 →
  (total_length_cm * 10 - (num_pieces - num_equal_pieces) * length_remaining_piece_mm) / num_equal_pieces = 75 :=
by
  sorry

end equal_pieces_length_l589_589748


namespace circle_passing_through_points_l589_589606

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589606


namespace circle_through_points_l589_589463

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589463


namespace circle_through_points_l589_589472

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589472


namespace triangle_area_bound_by_line_l589_589152

noncomputable def area_of_triangle : ℝ :=
let x_intercept := 3 in
let y_intercept := 9 in
1 / 2 * x_intercept * y_intercept

theorem triangle_area_bound_by_line (x y : ℝ) (h : 3 * x + y = 9) :
  area_of_triangle = 13.5 :=
sorry

end triangle_area_bound_by_line_l589_589152


namespace monotonic_decreasing_interval_l589_589037

def f (x : ℝ) : ℝ := log (x^2)

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, x < 0 → ∀ y : ℝ, y < 0 → x < y → f x > f y :=
by sorry

end monotonic_decreasing_interval_l589_589037


namespace largest_four_digit_sum_19_l589_589082

theorem largest_four_digit_sum_19 : ∃ n : ℕ, n < 10000 ∧ 1000 ≤ n ∧ (∑ i in (n.digits 10), id i = 19) ∧
  (∀ m : ℕ, m < 10000 ∧ 1000 ≤ m ∧ (∑ i in (m.digits 10), id i = 19) → n ≥ m) ∧ n = 9910 :=
by 
  sorry

end largest_four_digit_sum_19_l589_589082


namespace cricket_target_runs_l589_589337

theorem cricket_target_runs (rr_10_overs : ℝ) (rr_40_overs : ℝ) (overs_10 : ℝ) (overs_40 : ℝ) : rr_10_overs = 3.4 → rr_40_overs = 6.2 → overs_10 = 10 → overs_40 = 40 → 
  let runs_10_overs := rr_10_overs * overs_10 in
  let total_runs_50_overs := rr_40_overs * (overs_10 + overs_40) in
  total_runs_50_overs = 310 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  let runs_10_overs := 3.4 * 10
  let total_runs_50_overs := 6.2 * (10 + 40)
  sorry

end cricket_target_runs_l589_589337


namespace johns_distance_l589_589345

noncomputable def distance_to_conference_center
  (v1 v2 : ℕ) (t : ℕ) (distance time_late time_early : ℝ) : ℝ :=
  if t = 1
  then (v1 * (distance / v1 + time_late) = distance)
       ∧ ((distance - v1 * t) / v2 = distance / v1 - t - time_late + time_early)
  then distance
  else 0

theorem johns_distance
  (v1 v2 : ℕ) (t : ℕ) (distance time_late time_early : ℝ)
  (hv1 : v1 = 45)
  (hv2 : v2 = 65)
  (ht : t = 1)
  (hdist : distance = 191.25)
  (hlate : time_late = 0.75)
  (hearly : time_early = 0.25) :
  distance_to_conference_center v1 v2 t distance time_late time_early = 191.25 := 
sorry

end johns_distance_l589_589345


namespace expression_evaluation_l589_589213

theorem expression_evaluation :
  2^3 + 4 * 5 - Real.sqrt 9 + (3^2 * 2) / 3 = 31 := sorry

end expression_evaluation_l589_589213


namespace variable_is_eleven_l589_589313

theorem variable_is_eleven (x : ℕ) (h : (1/2)^22 * (1/81)^x = 1/(18^22)) : x = 11 :=
by
  sorry

end variable_is_eleven_l589_589313


namespace sum_sequence_terms_l589_589950

theorem sum_sequence_terms (S : ℕ → ℕ) (a : ℕ → ℕ) (b : ℕ → ℚ) (c : ℕ → ℚ) (T : ℕ → ℚ) (n : ℕ) :
  (∀ n, S n = 2 * n^2) →
  (∀ n, a n = S n - S (n - 1) ∧ (a 1 = 2)) →
  (∀ n, b n = 2 / (4^(n - 1))) →
  (∀ n, c n = a n / b n) →
  (∀ n, T n = (1 / 9 : ℚ) * ((6 * n - 5) * 4^n + 5)) :=
by
  intros
  have a_n_form : ∀ n, a n = 4 * n - 2, from
    sorry -- Proof for the general term of a_n
  have b_n_form : ∀ n, b n = 2 / (4^(n - 1)), from
    sorry -- Proof for the general term of b_n
  have c_n_form : ∀ n, c n = (2 * n - 1) * 4^(n - 1), from
    sorry -- Proof for the general term of c_n

  -- Prove the sum of the first n terms of the sequence c_n
  have T_n_form : ∀ n, T n = (1 / 9 : ℚ) * ((6 * n - 5) * 4^n + 5), from
    sorry -- Proof for the sum of sequence c_n

  sorry -- Combine everything together

end sum_sequence_terms_l589_589950


namespace largest_four_digit_sum_19_l589_589074

theorem largest_four_digit_sum_19 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (nat.digits 10 n).sum = 19 ∧ 
           ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ (nat.digits 10 m).sum = 19 → m ≤ 8920 :=
begin
  sorry
end

end largest_four_digit_sum_19_l589_589074


namespace find_circle_equation_l589_589559

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589559


namespace probability_no_brown_balls_l589_589110

theorem probability_no_brown_balls 
  (total_balls : ℕ) (blue_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) (brown_balls : ℕ)
  (total_balls = 32) (blue_balls = 6) (red_balls = 8) (yellow_balls = 4) (brown_balls = 14)
  : (∀ n : ℕ, n = 3 → (n.choose 18 * 17 * 16) / (n.choose 32 * 31 * 30)) = 51 / 310 := sorry

end probability_no_brown_balls_l589_589110


namespace max_s_value_l589_589358

theorem max_s_value (p q r s : ℝ) (h1 : p + q + r + s = 10) (h2 : (p * q) + (p * r) + (p * s) + (q * r) + (q * s) + (r * s) = 20) :
  s ≤ (5 * (1 + Real.sqrt 21)) / 2 :=
sorry

end max_s_value_l589_589358


namespace equation_of_circle_passing_through_points_l589_589671

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589671


namespace equation_of_circle_through_three_points_l589_589569

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589569


namespace good_permutations_count_l589_589263

-- Define the main problem and the conditions
theorem good_permutations_count (n : ℕ) (hn : n > 0) : 
  ∃ P : ℕ → ℕ, 
  (P n = (1 / Real.sqrt 5) * (((1 + Real.sqrt 5) / 2) ^ (n + 1) - ((1 - Real.sqrt 5) / 2) ^ (n + 1))) := 
sorry

end good_permutations_count_l589_589263


namespace add_complex_numbers_l589_589942

-- Definitions based on the conditions
def i : ℂ := complex.I
def z1 : ℂ := 2 + i
def z2 : ℂ := 1 - 2 * i

-- The proof problem statement
theorem add_complex_numbers : 
  z1 + z2 = 3 - i :=
begin
  sorry
end

end add_complex_numbers_l589_589942


namespace intersection_x_value_l589_589702

theorem intersection_x_value :
  ∀ x y: ℝ,
    (y = 3 * x - 15) ∧ (3 * x + y = 120) → x = 22.5 := by
  sorry

end intersection_x_value_l589_589702


namespace second_player_prevents_equal_l589_589332

def player_turn (a : List ℕ) (i : ℕ) (first_player_turn : Bool) : List ℕ :=
if first_player_turn then
  -- First player adds 1 to two adjacent numbers
  if i + 1 < a.length then
    (a.take i) ++ [a[i] + 1, a[i+1] + 1] ++ (a.drop (i + 2))
  else a
else
  -- Second player swaps two adjacent numbers
  if i + 1 < a.length then
    (a.take i) ++ [a[i+1], a[i]] ++ (a.drop (i + 2))
  else a

theorem second_player_prevents_equal (a : List ℕ) (h : ∀ i, a[i] % 2 ≠ a[i+1] % 2) :
  ∃ f, ∀ (n : ℕ), player_turn (List.iterate f n a) 0 true ≠ List.replicate a.length (a[0] + n) :=
sorry

end second_player_prevents_equal_l589_589332


namespace circle_passing_through_points_l589_589603

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589603


namespace example_problem_l589_589385

noncomputable def exists_constant_C (n : ℕ) (a : ℕ → ℝ) : Prop :=
  ∃ C : ℝ, ∀ n : ℕ, ∀ a : ℕ → ℝ, 
    (∀ j, 1 ≤ j ∧ j ≤ n → a j ∈ (set.univ : set ℝ)) → 
    (∃ f : ℝ → ℝ, f = λ x, ∏ j in finset.range n, |x - a j| ∧ 
      (∃ S : ℝ, S = (finset.sup finset.Icc 0 1 (λ x, |f x|))) →
        (∀ x ∈ (set.Icc 0 2 : set ℝ), |f x| ≤ C^n * S))

theorem example_problem (n : ℕ) (a : ℕ → ℝ) (h1 : 0 < n) :
  exists_constant_C n a :=
by
  sorry

end example_problem_l589_589385


namespace area_of_wall_photo_l589_589185

theorem area_of_wall_photo (width_frame : ℕ) (width_paper : ℕ) (length_paper : ℕ) 
  (h_width_frame : width_frame = 2) (h_width_paper : width_paper = 8) (h_length_paper : length_paper = 12) :
  (width_paper + 2 * width_frame) * (length_paper + 2 * width_frame) = 192 :=
by
  sorry

end area_of_wall_photo_l589_589185


namespace circle_equation_correct_l589_589537

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589537


namespace count_five_digit_with_three_l589_589304

-- Define the digit placement and 5-digit number constraints.
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def has_ten_thousands_digit_three (n : ℕ) : Prop :=
  ∃ d₁ d₂ d₃ d₄ : ℕ, n = 30000 + d₁ * 1000 + d₂ * 100 + d₃ * 10 + d₄ ∧ 
  d₁ < 10 ∧ d₂ < 10 ∧ d₃ < 10 ∧ d₄ < 10

-- The proof statement formally.
theorem count_five_digit_with_three : 
  {n : ℕ | is_five_digit n ∧ has_ten_thousands_digit_three n}.card = 10000 :=
sorry

end count_five_digit_with_three_l589_589304


namespace expansion_terms_l589_589202

theorem expansion_terms {x y z w p q r s t : Type} :
  let set1 := {x, y, z, w}
  let set2 := {p, q, r, s, t}
  card set1 = 4 → card set2 = 5 → 
  card (set1 × set2) = 20 :=
by intros 
   intro h1 
   intro h2 
   sorry

end expansion_terms_l589_589202


namespace equation_of_circle_passing_through_points_l589_589660

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589660


namespace F_expression_l589_589923

noncomputable def alpha := (3 - Real.sqrt 5) / 2
noncomputable def f (n : ℕ) : ℕ := Int.floor (alpha * n)
noncomputable def F (k : ℕ) : ℕ := Nat.find (λ n, (Nat.iterate f k n) > 0)

theorem F_expression (k : ℕ) :
  F k = (1 / Real.sqrt 5 : ℝ) * ((3 + Real.sqrt 5) / 2)^(k + 1) - 
        (1 / Real.sqrt 5 : ℝ) * ((3 - Real.sqrt 5) / 2)^(k + 1) :=
sorry

end F_expression_l589_589923


namespace find_circle_equation_l589_589551

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589551


namespace circle_equation_through_points_l589_589638

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589638


namespace determine_number_of_pipes_l589_589221

theorem determine_number_of_pipes :
  let d_small : ℝ := 2  -- diameter of the smaller pipe in inches
  let l_small : ℝ := 2 * 39.37  -- length of the smaller pipe in inches (2 meters converted to inches)
  let d_large : ℝ := 8  -- diameter of the larger pipe in inches
  let l_large : ℝ := 39.37  -- length of the larger pipe in inches (1 meter converted to inches)
  let radius_small : ℝ := d_small / 2  -- radius of the smaller pipe
  let radius_large : ℝ := d_large / 2  -- radius of the larger pipe
  let area_small : ℝ := real.pi * radius_small ^ 2  -- cross-sectional area of the smaller pipe
  let area_large : ℝ := real.pi * radius_large ^ 2  -- cross-sectional area of the larger pipe
  let volume_small : ℝ := area_small * l_small  -- volume of the smaller pipe
  let volume_large : ℝ := area_large * l_large  -- volume of the larger pipe
  volume_large / volume_small = 8 :=
by
  sorry

end determine_number_of_pipes_l589_589221


namespace triangle_area_l589_589172

-- Define the line equation 3x + y = 9
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the triangle bounded by the coordinate axes and the line
def bounded_triangle (x y : ℝ) : Prop :=
  (x = 0 ∧ 0 ≤ y ∧ y ≤ 9) ∨  -- y-axis segment
  (y = 0 ∧ 0 ≤ x ∧ x ≤ 3) ∨  -- x-axis segment
  (3 * x + y = 9 ∧ 0 ≤ x ∧ 0 ≤ y)  -- line segment

-- The area of the triangle formed by the coordinate axes and the line 3x + y = 9
theorem triangle_area : 
  let area := 1/2 * 3 * 9 in -- The calculation of the area
  area = 13.5 := 
by
  sorry

end triangle_area_l589_589172


namespace athlete_heartbeats_calculation_l589_589786

theorem athlete_heartbeats_calculation :
  ∀ (heart_rate : ℕ) (distance : ℕ) (pace : ℕ),
  heart_rate = 160 →
  distance = 30 →
  pace = 6 →
  (pace * distance * heart_rate = 28800)
:= by
  intros heart_rate distance pace hr_eq dis_eq pace_eq
  rw [hr_eq, dis_eq, pace_eq]
  norm_num
  sorry

end athlete_heartbeats_calculation_l589_589786


namespace geometric_sequence_eleventh_term_l589_589854

theorem geometric_sequence_eleventh_term (a₁ : ℚ) (r : ℚ) (n : ℕ) (hₐ : a₁ = 5) (hᵣ : r = 2 / 3) (hₙ : n = 11) :
  (a₁ * r^(n - 1) = 5120 / 59049) :=
by
  -- conditions of the problem
  rw [hₐ, hᵣ, hₙ]
  sorry

end geometric_sequence_eleventh_term_l589_589854


namespace circle_equation_correct_l589_589544

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589544


namespace who_is_werewolf_choose_companion_l589_589095

-- Define inhabitants with their respective statements
inductive Inhabitant
| A | B | C

-- Assume each inhabitant can be either a knight (truth-teller) or a liar
def is_knight (i : Inhabitant) : Prop := sorry

-- Define statements made by each inhabitant
def A_statement : Prop := ∃ werewolf : Inhabitant, werewolf = Inhabitant.C
def B_statement : Prop := ¬(∃ werewolf : Inhabitant, werewolf = Inhabitant.B)
def C_statement : Prop := ∃ liar1 liar2 : Inhabitant, liar1 ≠ liar2 ∧ liar1 ≠ Inhabitant.C ∧ liar2 ≠ Inhabitant.C

-- Define who is the werewolf (liar)
def is_werewolf (i : Inhabitant) : Prop := ¬is_knight i

-- The given conditions from statements
axiom A_is_knight : is_knight Inhabitant.A ↔ A_statement
axiom B_is_knight : is_knight Inhabitant.B ↔ B_statement
axiom C_is_knight : is_knight Inhabitant.C ↔ C_statement

-- The conclusion: C is the werewolf and thus a liar.
theorem who_is_werewolf : is_werewolf Inhabitant.C :=
by sorry

-- Choosing a companion: 
-- If C is a werewolf, we prefer to pick A as a companion over B or C.
theorem choose_companion (worry_about_werewolf : Bool) : Inhabitant :=
if worry_about_werewolf then Inhabitant.A else sorry

end who_is_werewolf_choose_companion_l589_589095


namespace circle_through_points_l589_589599

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589599


namespace mean_less_median_l589_589730

-- Define the set q
def q : List ℝ := [-4, 1, 7, 18, 20, 26, 29, 33, 42.5, 50.3]

-- Calculate the mean of the set q
def mean (data : List ℝ) : ℝ := (data.sum / data.length.toReal)

-- Calculate the median of the set q (assuming it is already sorted)
def median (data : List ℝ) : ℝ :=
  if data.length % 2 = 0 then
    let mid := data.length / 2
    (data[mid - 1] + data[mid]) / 2
  else
    data[data.length / 2]

-- Define the theorem to be proved
theorem mean_less_median : 
  (median q - mean q) = 0.72 := 
by
  sorry

end mean_less_median_l589_589730


namespace total_heartbeats_during_race_l589_589799

-- Definitions for the conditions
def heart_beats_per_minute : ℕ := 160
def pace_in_minutes_per_mile : ℕ := 6
def total_distance_in_miles : ℕ := 30

-- Main theorem statement
theorem total_heartbeats_during_race : 
  heart_beats_per_minute * pace_in_minutes_per_mile * total_distance_in_miles = 28800 :=
by
  -- Place the proof here
  sorry

end total_heartbeats_during_race_l589_589799


namespace find_scalars_l589_589705

open Real

-- Definitions for the given vectors
def a : ℝ × ℝ × ℝ := (2, 2, 2)
def b : ℝ × ℝ × ℝ := (3, -4, 1)
def c : ℝ × ℝ × ℝ := (5, 1, -6)
def x : ℝ × ℝ × ℝ := (-8, 14, 6)

-- Dot product definition
def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Norm squared definition
def norm_squared (v : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v v

-- Orthogonality condition (a ⊥ b, a ⊥ c, b ⊥ c)
def orthogonal (v w : ℝ × ℝ × ℝ) : Prop :=
  dot_product v w = 0

-- Main statement
theorem find_scalars :
  orthogonal a b ∧ orthogonal a c ∧ orthogonal b c
  ∧ (∃ (p q r : ℝ), x = (p * a.1 + q * b.1 + r * c.1,
                         p * a.2 + q * b.2 + r * c.2,
                         p * a.3 + q * b.3 + r * c.3))
  ∧ ∃ (p q r : ℝ), p = 2 ∧ q = -37 / 13 ∧ r = -1 :=
by
  sorry

end find_scalars_l589_589705


namespace least_area_triangle_DEF_l589_589696

noncomputable def complex_solutions (z : ℂ) := (z - 4)^6 = 64

theorem least_area_triangle_DEF : 
  let S := {z : ℂ | complex_solutions z} in
  let vertices := { w : ℂ | ∃ k : ℤ, 0 ≤ k ∧ k < 6 ∧ w = 4 + (2 * complex.exp (complex.I * (2 * real.pi * k / 6))) } in
  let D := 6 
  let E := 4 + 1 + complex.I * real.sqrt 3
  let F := 4 - 1 + complex.I * real.sqrt 3
  let base := complex.abs (E - F)
  let height := real.sqrt 3
  (0.5 * base * height) = real.sqrt 3 :=
by 
  sorry

end least_area_triangle_DEF_l589_589696


namespace min_distance_from_circle_to_line_l589_589944

open_locale real

def line_l (t : ℝ) : ℝ × ℝ := (1 - (real.sqrt 2 / 2) * t, 2 + (real.sqrt 2 / 2) * t)
def circle_C (θ : ℝ) : ℝ × ℝ := (2 * real.cos θ * real.cos θ, 2 * real.cos θ * real.sin θ)

theorem min_distance_from_circle_to_line :
  let l : ℝ → ℝ × ℝ := λ t, (1 - (real.sqrt 2 / 2) * t, 2 + (real.sqrt 2 / 2) * t) in
  let C : ℝ → ℝ × ℝ := λ θ, (1 + real.cos θ, real.sin θ) in
  let d := λ p q : ℝ × ℝ, real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) in
  ∀ x y θ, 
  x = 1 + real.cos θ →
  y = real.sin θ →
  ∃ t, let p := l t in d (x, y) p = sqrt 2 - 1 :=
sorry

end min_distance_from_circle_to_line_l589_589944


namespace cos_B_geometric_sequence_l589_589340

theorem cos_B_geometric_sequence
  (a b c : ℝ)
  (h_geom : b = a * real.sqrt 2 ∧ c = 2 * a)
  (h_cos : ∀ (B : ℝ), c^2 = a^2 + b^2 - 2 * a * b * real.cos B) :
  real.cos B = -real.sqrt 2 / 4 :=
by
  sorry

end cos_B_geometric_sequence_l589_589340


namespace find_n_for_sine_equality_l589_589896

theorem find_n_for_sine_equality : 
  ∃ (n: ℤ), -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.sin (670 * Real.pi / 180) ∧ n = -50 := by
  sorry

end find_n_for_sine_equality_l589_589896


namespace circle_equation_l589_589635

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589635


namespace linda_pick_probability_l589_589369

open BigOperators

-- Definitions based on given conditions
def total_oranges : ℕ := 10
def sweet_oranges : ℕ := 6
def sour_oranges : ℕ := 4
def pick_count : ℕ := 3

-- Calculate binomial coefficients
def binom (n k : ℕ) : ℕ := nat.choose n k

noncomputable def total_ways := binom total_oranges pick_count
noncomputable def favorable_ways := binom sweet_oranges pick_count

-- Correct answer based on probabilities
noncomputable def probability_sweet_pick := (favorable_ways : ℚ) / (total_ways : ℚ)

-- The proof statement
theorem linda_pick_probability : probability_sweet_pick = (1 : ℚ) / 6 := 
by
  sorry

end linda_pick_probability_l589_589369


namespace rearranged_rows_preserve_order_l589_589762

def rearrange_condition (front_row back_row : List ℝ) : Prop :=
  ∀ i, i < front_row.length → front_row.nthLe i sorry < back_row.nthLe i sorry

theorem rearranged_rows_preserve_order (front_row back_row : List ℝ) (h_len : front_row.length = back_row.length)
  (h_condition : rearrange_condition front_row back_row) :
  rearrange_condition (front_row.qsort (· < ·)) (back_row.qsort (· < ·)) := 
sorry

end rearranged_rows_preserve_order_l589_589762


namespace SameFunction_l589_589091

noncomputable def f : ℝ → ℝ := λ x, x^2
noncomputable def g : ℝ → ℝ := λ x, (x^6)^(1/3)

theorem SameFunction : ∀ x : ℝ, f x = g x :=
by
  intro x
  sorry

end SameFunction_l589_589091


namespace trapezoid_longer_side_length_l589_589138

theorem trapezoid_longer_side_length (s : ℝ) : 
  let side := 2 in
  let mid_len := side / 2 in
  let triv_area_eq := s^2 = (1 + s) / 2 in
  ∃ s, (2*s^2 - s - 1 = 0) → s = 1 :=
by
  sorry

end trapezoid_longer_side_length_l589_589138


namespace length_of_equal_pieces_l589_589745

theorem length_of_equal_pieces (total_length : ℕ) (num_pieces : ℕ) (num_unequal_pieces : ℕ) (unequal_piece_length : ℕ)
    (equal_pieces : ℕ) (equal_piece_length : ℕ) :
    total_length = 11650 ∧ num_pieces = 154 ∧ num_unequal_pieces = 4 ∧ unequal_piece_length = 100 ∧ equal_pieces = 150 →
    equal_piece_length = 75 :=
by
  sorry

end length_of_equal_pieces_l589_589745


namespace exists_k_for_f_eq_m_unique_k_for_f_eq_m_l589_589242

-- Define the function f
def f (k : ℕ) : ℕ := 
  -- Placeholder definition: this needs to be correctly implemented
  sorry

-- Theorem 1
theorem exists_k_for_f_eq_m (m : ℕ) (hm : m > 0) : ∃ k : ℕ, f(k) = m :=
by
  -- Proof Required
  sorry

-- Theorem 2
theorem unique_k_for_f_eq_m (m : ℕ) (hm : m > 0) : ∃! k : ℕ, f(k) = m :=
by
  -- Proof Required
  sorry

end exists_k_for_f_eq_m_unique_k_for_f_eq_m_l589_589242


namespace find_number_l589_589317

theorem find_number :
  ∃ x : Int, x - (28 - (37 - (15 - 20))) = 59 ∧ x = 45 :=
by
  sorry

end find_number_l589_589317


namespace Murtha_pebbles_l589_589373

-- Definition of the geometric series sum formula
noncomputable def sum_geometric_series (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

-- Constants for the problem
def a : ℕ := 1
def r : ℕ := 2
def n : ℕ := 10

-- The theorem to be proven
theorem Murtha_pebbles : sum_geometric_series a r n = 1023 :=
by
  -- Our condition setup implies the formula
  sorry

end Murtha_pebbles_l589_589373


namespace find_circle_equation_l589_589562

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589562


namespace circle_passing_through_points_l589_589499

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589499


namespace log_expression_simplified_l589_589212

noncomputable def compute_log_expression : Real :=
  sqrt (log 12 / log 4 - log 12 / log 5)

theorem log_expression_simplified :
  compute_log_expression = sqrt ((log 12 * log 1.25) / (log 4 * log 5)) :=
by
  sorry

end log_expression_simplified_l589_589212


namespace train_passing_time_correct_l589_589996

-- Definitions based on conditions
def length_of_train : ℝ := 500  -- in meters
def speed_in_kmph : ℝ := 90     -- in km/hr
def kmph_to_mps_conversion_factor : ℝ := 1000 / 3600 -- conversion factor from km/hr to m/s
def speed_in_mps : ℝ := speed_in_kmph * (kmph_to_mps_conversion_factor) -- convert 90 km/hr to m/s
def expected_time : ℝ := 20 -- expected time in seconds

-- Theorem statement to be proved
theorem train_passing_time_correct : (length_of_train / speed_in_mps) = expected_time :=
by
  sorry

end train_passing_time_correct_l589_589996


namespace circle_equation_correct_l589_589536

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589536


namespace imaginary_part_of_i_l589_589254

variable (z : ℂ)

theorem imaginary_part_of_i : z = complex.I → complex.im z = 1 :=
by
  sorry

end imaginary_part_of_i_l589_589254


namespace circle_equation_correct_l589_589531

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589531


namespace athlete_heartbeats_l589_589807

theorem athlete_heartbeats
  (heart_beats_per_minute : ℕ)
  (pace_minutes_per_mile : ℕ)
  (total_miles : ℕ)
  (heart_beats_per_minute_eq : heart_beats_per_minute = 160)
  (pace_minutes_per_mile_eq : pace_minutes_per_mile = 6)
  (total_miles_eq : total_miles = 30) :
  heart_beats_per_minute * pace_minutes_per_mile * total_miles = 28800 :=
by {
  have h1 : heart_beats_per_minute = 160 := heart_beats_per_minute_eq,
  have h2 : pace_minutes_per_mile = 6 := pace_minutes_per_mile_eq,
  have h3 : total_miles = 30 := total_miles_eq,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end athlete_heartbeats_l589_589807


namespace circle_passing_three_points_l589_589415

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589415


namespace sum_of_vectors_zero_l589_589938

-- Define the plane and the points on the plane
variables {Point : Type} [AddCommGroup Point]

-- Given conditions
variables (points : set Point)
variables (A B : Point)
variables (vectors : set (Point × Point))
variables (origin : Point)

-- Each point has an equal number of vectors originating from it and ending at it
def condition (p : Point) : Prop :=
  vectors.countp (λ v, v.1 = p) = vectors.countp (λ v, v.2 = p)

-- Define the vector \(\vec{AB}\)
noncomputable def vector (p q : Point) : Point := q - p

-- Prove that the sum of all vectors is equal to the zero vector
theorem sum_of_vectors_zero (h : ∀ p ∈ points, condition p) : 
  (∑ v in vectors, vector v.1 v.2) = 0 :=
sorry

end sum_of_vectors_zero_l589_589938


namespace athlete_heartbeats_l589_589811

theorem athlete_heartbeats
  (heart_beats_per_minute : ℕ)
  (pace_minutes_per_mile : ℕ)
  (total_miles : ℕ)
  (heart_beats_per_minute_eq : heart_beats_per_minute = 160)
  (pace_minutes_per_mile_eq : pace_minutes_per_mile = 6)
  (total_miles_eq : total_miles = 30) :
  heart_beats_per_minute * pace_minutes_per_mile * total_miles = 28800 :=
by {
  have h1 : heart_beats_per_minute = 160 := heart_beats_per_minute_eq,
  have h2 : pace_minutes_per_mile = 6 := pace_minutes_per_mile_eq,
  have h3 : total_miles = 30 := total_miles_eq,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end athlete_heartbeats_l589_589811


namespace peanuts_in_box_l589_589320

theorem peanuts_in_box (initial_peanuts : ℕ) (added_peanuts : ℕ) (h1 : initial_peanuts = 4) (h2 : added_peanuts = 2) : initial_peanuts + added_peanuts = 6 := by
  sorry

end peanuts_in_box_l589_589320


namespace smallest_natural_greater_than_12_l589_589720

def smallest_greater_than (n : ℕ) : ℕ := n + 1

theorem smallest_natural_greater_than_12 : smallest_greater_than 12 = 13 :=
by
  sorry

end smallest_natural_greater_than_12_l589_589720


namespace gcd_sequence_1995_1996_l589_589215

def seq : ℕ → ℕ
| 0       := 19
| 1       := 95
| (n + 2) := Nat.lcm (seq (n + 1)) (seq n) + seq n

theorem gcd_sequence_1995_1996 : gcd (seq 1995) (seq 1996) = 19 :=
by
  sorry

end gcd_sequence_1995_1996_l589_589215


namespace equation_of_circle_ABC_l589_589440

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589440


namespace least_number_to_add_l589_589733

theorem least_number_to_add (n : ℕ) (k : ℕ) : n = 1101 → (n + k) % 24 = 0 → k = 3 :=
by
  assume h1 h2,
  sorry

end least_number_to_add_l589_589733


namespace card_game_equiv_tictactoe_l589_589714

/-- Two-player game where players pick cards numbered from 1 to 9.
The goal is for a player to have three cards that sum to 15.
Prove: this game is equivalent to playing tic-tac-toe on a predefined 3 × 3 grid. -/
theorem card_game_equiv_tictactoe : 
  let cards := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∃ grid : ℕ × ℕ → ℕ, 
    (∀ i j, 
      1 ≤ i ∧ i ≤ 3 ∧ 1 ≤ j ∧ j ≤ 3 → 
      (∃ (triplet : Finset ℕ), triplet.card = 3 ∧ triplet.sum = 15 ∧ 
        (∃ (r : Finset (ℕ × ℕ)), r = {(i, j) | ∃ x ∈ triplet, grid x = r} ∧ 
          (r = {(2, 9, 4), (7, 5, 3), (6, 1, 8), (2, 7, 6), (9, 5, 1), (4, 3, 8),
          (2, 5, 8), (6, 5, 4)}))) := sorry

end card_game_equiv_tictactoe_l589_589714


namespace circle_passing_through_points_l589_589608

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589608


namespace rearranged_rows_preserve_order_l589_589763

theorem rearranged_rows_preserve_order {n : ℕ} (a b : Fin n → ℝ)
    (h : ∀ i, a i > b i) :
    let a' := Finset.univ.val.sort (· ≤ ·)
        b' := Finset.univ.val.sort (λ i j, b i ≤ b j)
    in ∀ i, a' i > b' i :=
sorry

end rearranged_rows_preserve_order_l589_589763


namespace circle_passes_through_points_l589_589518

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589518


namespace athlete_heartbeats_during_race_l589_589830

theorem athlete_heartbeats_during_race :
  ∀ (heartbeats_per_minute pace_minutes_per_mile race_miles : ℕ ),
    heartbeats_per_minute = 160 →
    pace_minutes_per_mile = 6 →
    race_miles = 30 →
    heartbeats_per_minute * pace_minutes_per_mile * race_miles = 28800 :=
by
  intros heartbeats_per_minute pace_minutes_per_mile race_miles
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end athlete_heartbeats_during_race_l589_589830


namespace final_number_after_trebled_l589_589769

theorem final_number_after_trebled (initial_number : ℕ) (y z w : ℕ) 
  (h_initial : initial_number = 8)
  (h_doubled : y = 2 * initial_number)
  (h_added : z = y + 9)
  (h_trebled : w = 3 * z) :
  w = 75 :=
by
  rw [h_initial, h_doubled, h_added, h_trebled]
  sorry

end final_number_after_trebled_l589_589769


namespace line_intersects_circle_probability_l589_589943

noncomputable def probability_intersection (a b : ℝ) : ℝ :=
  if a ∈ set.Ioo (-1 : ℝ) 1 ∧ b ∈ set.Ioo 0 1 then (5 / 16) else 0

theorem line_intersects_circle_probability :
  ∀ (a b : ℝ), a ∈ set.Ioo (-1 : ℝ) 1 → b ∈ set.Ioo 0 1 →
  probability_intersection a b = 5 / 16 :=
by
  intros a b ha hb
  rw probability_intersection
  split_ifs
  · exact rfl
  · exfalso
    exact h ⟨ha, hb⟩

end line_intersects_circle_probability_l589_589943


namespace eric_has_correct_green_marbles_l589_589880

def total_marbles : ℕ := 20
def white_marbles : ℕ := 12
def blue_marbles : ℕ := 6
def green_marbles : ℕ := total_marbles - (white_marbles + blue_marbles)

theorem eric_has_correct_green_marbles : green_marbles = 2 :=
by
  sorry

end eric_has_correct_green_marbles_l589_589880


namespace elizabeth_stickers_l589_589877

def initial_bottles := 10
def lost_at_school := 2
def lost_at_practice := 1
def stickers_per_bottle := 3

def total_remaining_bottles := initial_bottles - lost_at_school - lost_at_practice
def total_stickers := total_remaining_bottles * stickers_per_bottle

theorem elizabeth_stickers : total_stickers = 21 :=
  by
  unfold total_stickers total_remaining_bottles initial_bottles lost_at_school lost_at_practice stickers_per_bottle
  simp
  sorry

end elizabeth_stickers_l589_589877


namespace inverse_proportional_to_x_iff_h_l589_589188

def f (x : ℝ) : ℝ := x / 3
def g (x : ℝ) : ℝ := 3 / (x + 1)
def h (x : ℝ) : ℝ := 3 / x
def i (x : ℝ) : ℝ := 3 * x

theorem inverse_proportional_to_x_iff_h :
  ∀ (y : ℝ → ℝ), (∀ x : ℝ, x ≠ 0 → x * y(x) = 3) ↔ y = h :=
sorry

end inverse_proportional_to_x_iff_h_l589_589188


namespace equation_of_circle_ABC_l589_589443

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589443


namespace reciprocal_neg_2023_l589_589693

theorem reciprocal_neg_2023 : (1 / (-2023: ℤ)) = - (1 / 2023) :=
by
  -- proof goes here
  sorry

end reciprocal_neg_2023_l589_589693


namespace circle_through_points_l589_589421

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589421


namespace new_volume_correct_l589_589047

-- Define the conditions.
def initial_radius (r : ℝ) : Prop := true
def initial_height (h : ℝ) : Prop := true
def initial_volume (V : ℝ) : Prop := V = 15
def new_radius (r : ℝ) : ℝ := 3 * r
def new_height (h : ℝ) : ℝ := h / 2

-- Define the volume of a cylinder.
def volume_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- State the proof problem.
theorem new_volume_correct (r h : ℝ) (V : ℝ)
  (h1 : initial_radius r)
  (h2 : initial_height h)
  (h3 : initial_volume V) :
  volume_cylinder (new_radius r) (new_height h) = 67.5 :=
by
  sorry

end new_volume_correct_l589_589047


namespace minimize_sum_of_cubes_l589_589865

theorem minimize_sum_of_cubes (x y : ℝ) (h : x + y = 8) : 
  (3 * x^2 - 3 * (8 - x)^2 = 0) → (x = 4) ∧ (y = 4) :=
by
  sorry

end minimize_sum_of_cubes_l589_589865


namespace regular_price_one_pound_is_20_l589_589774

variable (y : ℝ)
variable (discounted_price_quarter_pound : ℝ)

-- Conditions
axiom h1 : 0.6 * (y / 4) + 2 = discounted_price_quarter_pound
axiom h2 : discounted_price_quarter_pound = 2
axiom h3 : 0.1 * y = 2

-- Question: What is the regular price for one pound of cake?
theorem regular_price_one_pound_is_20 : y = 20 := 
  sorry

end regular_price_one_pound_is_20_l589_589774


namespace no_common_elements_in_sequences_l589_589068

theorem no_common_elements_in_sequences :
  ∀ (k : ℕ), (∃ n : ℕ, k = n^2 - 1) ∧ (∃ m : ℕ, k = m^2 + 1) → False :=
by sorry

end no_common_elements_in_sequences_l589_589068


namespace range_of_a_l589_589291

variable (a : ℝ)
def f (x : ℝ) : ℝ := a * x^2 - 2 * a * x - 4

theorem range_of_a :
  (∀ x : ℝ, f a x < 0) → (-4 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l589_589291


namespace marble_distribution_correct_l589_589195

def num_ways_to_distribute_marbles : ℕ :=
  -- Given:
  -- Evan divides 100 marbles among three volunteers with each getting at least one marble
  -- Lewis selects a positive integer n > 1 and for each volunteer, steals exactly 1/n of marbles if possible.
  -- Prove that the number of ways to distribute the marbles such that Lewis cannot steal from all volunteers
  3540

theorem marble_distribution_correct :
  num_ways_to_distribute_marbles = 3540 :=
sorry

end marble_distribution_correct_l589_589195


namespace smallest_number_with_divisibility_condition_l589_589084

theorem smallest_number_with_divisibility_condition :
  ∃ x : ℕ, (x + 7) % 24 = 0 ∧ (x + 7) % 36 = 0 ∧ (x + 7) % 50 = 0 ∧ (x + 7) % 56 = 0 ∧ (x + 7) % 81 = 0 ∧ x = 113393 :=
by {
  -- sorry is used to skip the proof.
  sorry
}

end smallest_number_with_divisibility_condition_l589_589084


namespace unit_prices_l589_589025

theorem unit_prices (x y : ℕ) (h1 : 5 * x + 4 * y = 139) (h2 : 4 * x + 5 * y = 140) :
  x = 15 ∧ y = 16 :=
by
  -- Proof will go here
  sorry

end unit_prices_l589_589025


namespace circle_passing_through_points_l589_589503

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589503


namespace largest_four_digit_sum_19_l589_589077

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digits_sum_to (n : ℕ) (sum : ℕ) : Prop := (n.digits 10).sum = sum

theorem largest_four_digit_sum_19 : ∃ n : ℕ, is_four_digit n ∧ digits_sum_to n 19 ∧ ∀ m : ℕ, is_four_digit m ∧ digits_sum_to m 19 → n ≥ m :=
by
  use 9730
  split
  · exact sorry
  · split
    · exact sorry
    · intros m hm
      exact sorry

end largest_four_digit_sum_19_l589_589077


namespace intersect_lines_XZ_YW_at_AC_BD_l589_589926

-- Defining the quadrilateral ABCD with an inscribed circle
variables {A B C D X Y Z W : Point}
variables {ℓAB ℓBC ℓCD ℓDA : Line}
variables (circleC : Circle)

-- Assumptions
def inscribed_circle_in_quadrilateral :=
  inscribed circleC A B C D ∧ 
  tangent_point circleC ℓAB X ∧
  tangent_point circleC ℓBC Y ∧
  tangent_point circleC ℓCD Z ∧
  tangent_point circleC ℓDA W

-- Proving the intersection of lines
theorem intersect_lines_XZ_YW_at_AC_BD (h : inscribed_circle_in_quadrilateral circleC) :
  ∃ O, intersect (line_through_points A C) (line_through_points B D) O ∧
       intersect (line_through_points X Z) (line_through_points Y W) O :=
sorry

end intersect_lines_XZ_YW_at_AC_BD_l589_589926


namespace evaluate_e_pow_T_l589_589362

noncomputable def integrand : ℝ → ℝ :=
  λ x, (2 * Real.exp (3 * x) + Real.exp (2 * x) - 1) / (Real.exp (3 * x) + Real.exp (2 * x) - Real.exp x + 1)

noncomputable def T : ℝ :=
  ∫ x in 0..Real.ln 2, integrand x

theorem evaluate_e_pow_T : Real.exp T = 11 / 4 :=
by
  sorry

end evaluate_e_pow_T_l589_589362


namespace equation_of_circle_passing_through_points_l589_589661

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589661


namespace max_intersections_of_line_and_circles_l589_589910

/-
Given: Four coplanar circles.
Prove: There exists a line that touches exactly 8 points on these four circles.
-/

noncomputable def max_points_on_line (C1 C2 C3 C4 : circle) : ℕ :=
  2 * 4

theorem max_intersections_of_line_and_circles (C1 C2 C3 C4 : circle) (h1 : coplanar {C1, C2, C3, C4}) :
  ∃ l : ℝ → ℝ, (∀ t : ℕ, t ∈ {1, 2, 3, 4} → (l /* intersects C_t with exactly two points */)) ∧
  max_points_on_line C1 C2 C3 C4 = 8 :=
begin
  sorry
end

end max_intersections_of_line_and_circles_l589_589910


namespace standard_eq_circle_l589_589276

open Real

def line_l (x y : ℝ) : Prop := 3 * x - 4 * y - 15 = 0

def circle_C (x y r : ℝ) : Prop := x^2 + y^2 - 2 * x - 4 * y + 5 - r^2 = 0

def distance_center_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / sqrt (a^2 + b^2)

def points_A_B_distance (A B : ℝ) := A = 6

theorem standard_eq_circle :
  (∃ (x y r : ℝ), line_l x y ∧ circle_C x y r ∧ 0 < r ∧ points_A_B_distance 6 6) →
  ∃ (r : ℝ), (x - 1)^2 + (y - 2)^2 = 25 :=
by
  sorry

end standard_eq_circle_l589_589276


namespace mechanic_total_cost_l589_589127

theorem mechanic_total_cost :
  ∀ (hourly_rate hours_per_day days parts_cost : ℕ),
  hourly_rate = 60 →
  hours_per_day = 8 →
  days = 14 →
  parts_cost = 2500 →
  let total_hours := hours_per_day * days in
  let labor_cost := total_hours * hourly_rate in
  let total_cost := labor_cost + parts_cost in
  total_cost = 9220 :=
by
  intros hourly_rate hours_per_day days parts_cost
  assume h1 h2 h3 h4
  let total_hours := hours_per_day * days
  let labor_cost := total_hours * hourly_rate
  let total_cost := labor_cost + parts_cost
  show total_cost = 9220
  sorry

end mechanic_total_cost_l589_589127


namespace zero_of_function_l589_589061

theorem zero_of_function : ∃ x : ℝ, (x + 1)^2 = 0 :=
by
  use -1
  sorry

end zero_of_function_l589_589061


namespace smallest_class_size_l589_589326

theorem smallest_class_size :
  ∃ x : ℕ, (5 * x + 2 > 30) ∧ (5 * x + 2 = 32) :=
by
  use 6
  simp
  split
  · linarith
  · refl

end smallest_class_size_l589_589326


namespace cubic_sum_roots_l589_589292

theorem cubic_sum_roots 
  (r s : ℝ)
  (h1 : r^2 - 5*r + 3 = 0)
  (h2 : s^2 - 5*s + 3 = 0)
  (h3 : r ≠ s) : r^3 + s^3 = 80 := 
by
  -- Roots by Vieta's formulas
  have h4 : r + s = 5 := 
    by sorry,
  have h5 : r * s = 3 := 
    by sorry,
  -- Sum of squares
  have h6 : r^2 + s^2 = (r + s)^2 - 2 * (r * s) :=
    by sorry,
  have h7 : (r + s)^2 - 2 * (r * s) = 19 := 
    by sorry,
  -- Using the cubic identity
  have h8 : r^3 + s^3 = (r + s) * (r^2 + s^2 - r * s) :=
    by sorry,
  have h9 : (r + s) * (19 - 3) = 5 * 16 :=
    by sorry,
  have h10 : 5 * 16 = 80 :=
    by sorry,
  exact 80

end cubic_sum_roots_l589_589292


namespace equation_of_circle_passing_through_points_l589_589658

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589658


namespace laura_mowing_time_correct_l589_589005

noncomputable def laura_mowing_time : ℝ := 
  let combined_time := 1.71428571429
  let sammy_time := 3
  let combined_rate := 1 / combined_time
  let sammy_rate := 1 / sammy_time
  let laura_rate := combined_rate - sammy_rate
  1 / laura_rate

theorem laura_mowing_time_correct : laura_mowing_time = 4.2 := 
  by
    sorry

end laura_mowing_time_correct_l589_589005


namespace solve_system1_solve_system2_l589_589021

-- Definition for System (1)
theorem solve_system1 (x y : ℤ) (h1 : x - 2 * y = 0) (h2 : 3 * x - y = 5) : x = 2 ∧ y = 1 := 
by
  sorry

-- Definition for System (2)
theorem solve_system2 (x y : ℤ) 
  (h1 : 3 * (x - 1) - 4 * (y + 1) = -1) 
  (h2 : (x / 2) + (y / 3) = -2) : x = -2 ∧ y = -3 := 
by
  sorry

end solve_system1_solve_system2_l589_589021


namespace gcd_polynomial_l589_589941

theorem gcd_polynomial {b : ℕ} (h : 570 ∣ b) : Nat.gcd (4*b^3 + 2*b^2 + 5*b + 95) b = 95 := 
sorry

end gcd_polynomial_l589_589941


namespace circle_equation_l589_589624

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589624


namespace circle_passing_through_points_eqn_l589_589478

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589478


namespace circle_through_points_l589_589585

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589585


namespace circle_passing_through_points_l589_589495

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589495


namespace hundredth_odd_integer_is_199_sum_of_first_100_odd_integers_is_10000_l589_589716

noncomputable def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

noncomputable def sum_first_n_odd_positive_integers (n : ℕ) : ℕ :=
  n * n

theorem hundredth_odd_integer_is_199 : nth_odd_positive_integer 100 = 199 :=
  by
  sorry

theorem sum_of_first_100_odd_integers_is_10000 : sum_first_n_odd_positive_integers 100 = 10000 :=
  by
  sorry

end hundredth_odd_integer_is_199_sum_of_first_100_odd_integers_is_10000_l589_589716


namespace problem_statement_l589_589348

theorem problem_statement (a b m : ℕ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_m : m > 0) :
  (∃ n : ℕ, n > 0 ∧ m ∣ (a^n - 1) * b) ↔ (Nat.gcd (a * b) m = Nat.gcd b m) :=
by
  sorry

end problem_statement_l589_589348


namespace elizabeth_stickers_count_l589_589872

theorem elizabeth_stickers_count :
  let initial_bottles := 10
  let lost_at_school := 2
  let stolen_at_dance := 1
  let stickers_per_bottle := 3
  let remaining_bottles := initial_bottles - lost_at_school - stolen_at_dance
  remaining_bottles * stickers_per_bottle = 21 := by sorry

end elizabeth_stickers_count_l589_589872


namespace circle_passes_through_points_l589_589517

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589517


namespace circle_equation_l589_589632

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589632


namespace f_div_36_l589_589342

open Nat

def f (n : ℕ) : ℕ :=
  (2 * n + 7) * 3^n + 9

theorem f_div_36 (n : ℕ) : (f n) % 36 = 0 := 
  sorry

end f_div_36_l589_589342


namespace investment_calculation_l589_589013

theorem investment_calculation :
  ∃ (x : ℝ), x * (1.04 ^ 14) = 1000 := by
  use 571.75
  sorry

end investment_calculation_l589_589013


namespace equation_of_circle_through_three_points_l589_589581

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589581


namespace circle_through_points_l589_589465

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589465


namespace total_heartbeats_during_race_l589_589794

-- Definitions for the conditions
def heart_beats_per_minute : ℕ := 160
def pace_in_minutes_per_mile : ℕ := 6
def total_distance_in_miles : ℕ := 30

-- Main theorem statement
theorem total_heartbeats_during_race : 
  heart_beats_per_minute * pace_in_minutes_per_mile * total_distance_in_miles = 28800 :=
by
  -- Place the proof here
  sorry

end total_heartbeats_during_race_l589_589794


namespace exists_increasing_sequence_finite_primes_l589_589014

theorem exists_increasing_sequence_finite_primes:
  ∃ (a : ℕ → ℕ), (∀ n, a (n + 1) > a n) ∧
  (∀ k ≥ 2, ∃ N, ∀ n ≥ N, ¬ Prime (k + a n)) ∧
  (∀ n ≥ 2, ¬ Prime ((n!)^3)) ∧
  (∀ n ≥ 2, ¬ Prime ((n!)^3 + 1)) :=
by
  let a := λ n, (n!)^3
  use a
  split
  { intros n, exact (Nat.factorial_succ n)^3.gt.trans (Nat.zero_lt_succ _) }
  split
  { intros k hk
    use k
    intros n hn
    have : k ∣ a n := Nat.dvd_pow_self k (Nat.factorial_pos n)
    exact (Nat.dvd_add_left this).mpr (not_prime_ne_one (k + a n)) (k.ne_zero hk.ne') }
  split
  { intro hn
    exact Nat.not_prime_factorial_pow hn }
  { intro hn
    obtain ⟨b, hab⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.factorial_pos n).ne'
    use b
    rw [hab, pow_succ, mul_assoc, mul_add_one, ←h_eq_succ hab]
    apply Nat.not_prime_of_dvd
    { exact (Nat.factorial_pos n).ne' }
    rw [add_comm, mul_assoc, pow_succ']
    exact ⟨_, b, Nat.dvd_factorial hn.le⟩ }

end exists_increasing_sequence_finite_primes_l589_589014


namespace common_ratio_q_l589_589058

noncomputable def Sn (n : ℕ) (a1 q : ℝ) := a1 * (1 - q^n) / (1 - q)

theorem common_ratio_q (a1 : ℝ) (q : ℝ) (h : q ≠ 1) (h1 : 6 * Sn 4 a1 q = Sn 5 a1 q + 5 * Sn 6 a1 q) : q = -6/5 := by
  sorry

end common_ratio_q_l589_589058


namespace equation_of_circle_ABC_l589_589447

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589447


namespace average_runs_next_10_matches_l589_589396

theorem average_runs_next_10_matches (avg_first_10 : ℕ) (avg_all_20 : ℕ) (n_matches : ℕ) (avg_next_10 : ℕ) :
  avg_first_10 = 40 ∧ avg_all_20 = 35 ∧ n_matches = 10 → avg_next_10 = 30 :=
by
  intros h
  sorry

end average_runs_next_10_matches_l589_589396


namespace paper_folding_creases_l589_589782

theorem paper_folding_creases {rect : Type} [IsRectangular rect] : 
  (creasing rect twice).relation = Relationship.parallel_or_perpendicular :=
sorry

end paper_folding_creases_l589_589782


namespace circle_through_points_l589_589467

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589467


namespace equation_of_circle_passing_through_points_l589_589664

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589664


namespace T_is_centroid_l589_589379

/-- Given a triangle ABE with points C and D on side BE such that BC = CD = DE,
    and let X, Y, Z, T be the circumcenters of triangles ABE, ABC, ADE, and ACD respectively,
    prove that T is the centroid of triangle XYZ. -/
theorem T_is_centroid 
  (A B C D E X Y Z T : Point)
  (h1 : Collinear BE C)
  (h2 : Collinear BE D)
  (h3 : distance B C = distance C D)
  (h4 : distance C D = distance D E)
  (hX : Circumcenter X A B E)
  (hY : Circumcenter Y A B C)
  (hZ : Circumcenter Z A D E)
  (hT : Circumcenter T A C D) :
  Centroid T X Y Z := 
by
  sorry

end T_is_centroid_l589_589379


namespace factorial_division_l589_589308

theorem factorial_division : (50! / 48!) = 2450 := by
  sorry

end factorial_division_l589_589308


namespace construct_point_X_l589_589264

theorem construct_point_X (A B O : Point) (l : Line) (hO : O ∈ l) (hAB : A ≠ B) :
  ∃ X : Point, O ∈ l ∧ OX = segment_length AB :=
sorry

end construct_point_X_l589_589264


namespace probability_red_king_top_card_l589_589775

/-- A standard deck of 52 cards is randomly arranged. -/
def total_cards := 52
def red_king_cards := 2

/-- The probability that the top card is a King of a red suit is 2/52. -/
theorem probability_red_king_top_card : 
  (red_king_cards / total_cards : ℚ) = 1 / 26 :=
begin
  -- Placeholder for the proof
  sorry
end

end probability_red_king_top_card_l589_589775


namespace circle_equation_l589_589628

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589628


namespace possible_values_AD_l589_589375

theorem possible_values_AD (A B C D : ℝ) (h1 : abs (B - A) = 1) (h2 : abs (C - B) = 2) (h3 : abs (D - C) = 4) :
  ∃ d : ℝ, d ∈ {1, 3, 5, 7} ∧ abs (D - A) = d :=
by
  sorry

end possible_values_AD_l589_589375


namespace problem1_l589_589239

theorem problem1 :
  let a := (0.25 : ℝ) ^ (-0.5)
  let b := (1 / 27 : ℝ) ^ (-1 / 3)
  let c := (625 : ℝ) ^ 0.25
  a + b - c = 0 :=
by {
  sorry,
}

end problem1_l589_589239


namespace find_t_l589_589219

theorem find_t (t a b : ℝ) :
  (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (5 * x^2 + t * x + 12) = 15 * x^4 - 47 * x^3 + a * x^2 + b * x + 60) →
  t = -9 :=
by
  intros h
  -- We'll skip the proof part
  sorry

end find_t_l589_589219


namespace triangle_area_l589_589166

noncomputable def line_eq := λ x : ℝ, 9 - 3 * x

theorem triangle_area :
    let x_intercept := (3 : ℝ)
    let y_intercept := (9 : ℝ)
    let triangle_base := x_intercept
    let triangle_height := y_intercept
    let area := (1 / 2) * triangle_base * triangle_height
    area = 13.5 :=
by
    sorry

end triangle_area_l589_589166


namespace triangle_area_range_of_a_l589_589288

-- Part (1)
def f (a x : ℝ) : ℝ := a * real.exp (x - 1) - real.log x + real.log a

theorem triangle_area (x : ℝ) (a : ℝ) (ha : a = real.exp 1) : 
  let f_x : ℝ := f a x,
      tangent_line := fun y => y = (real.exp 1 - 1) * (x - 1) + (real.exp 1 + 1)  in
  (∃ x_inter y_inter : ℝ, tangent_line 0 = y_inter ∧ tangent_line x_inter = 0 ∧ 
  1/2 * real.abs x_inter * real.abs y_inter = 2 / (real.exp 1 - 1)) := 
sorry

-- Part (2)
def g (x : ℝ) (a : ℝ) : ℝ := a * real.exp (x - 1) - real.log x + real.log a

theorem range_of_a (x a : ℝ) (h : g x a ≥ 1) : 1 ≤ a := 
sorry

end triangle_area_range_of_a_l589_589288


namespace simplify_expression_l589_589019

variable (a b : ℝ)

theorem simplify_expression :
  (-2 * a^2 * b^3) * (-a * b^2)^2 + (- (1 / 2) * a^2 * b^3)^2 * 4 * b = -a^4 * b^7 := 
by 
  sorry

end simplify_expression_l589_589019


namespace triangle_BN_NC_relation_l589_589339

theorem triangle_BN_NC_relation (A B C D M N : Type) 
    (hAB : AC = 7) 
    (hAC : AB = 12) 
    (hAngleBisector : ∃ D : Type, angleBisector A B C ⟶ (BC intersects at D)) 
    (hMidpoint : ∃ M : Type, midpoint(A, D) = M)
    (hCircleTangent : ∃ N : Type, circle_tangent(A, BC, D) ⟶ (AC intersects at N)) :
    let p := 19 in
    let q := 91 in
    p + q = 110 :=
by
  sorry

end triangle_BN_NC_relation_l589_589339


namespace athlete_heartbeats_during_race_l589_589826

theorem athlete_heartbeats_during_race
  (heart_rate : ℕ)
  (pace : ℕ)
  (distance : ℕ)
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) :
  heart_rate * pace * distance = 28800 :=
by
  rw [H1, H2, H3]
  norm_num

end athlete_heartbeats_during_race_l589_589826


namespace blocks_needed_l589_589750

-- Define the given constants and theorems
def volume_block : ℝ := 48
def radius_cylinder : ℝ := 2.5
def height_cylinder : ℝ := 10
def volume_cylinder : ℝ := 62.5 * Real.pi

-- Prove the number of blocks needed
theorem blocks_needed (V_block : ℝ) (V_cylinder : ℝ) (N : ℕ) : 
  V_block = 48 → 
  V_cylinder = 62.5 * Real.pi → 
  N = Nat.ceil (V_cylinder / V_block) → 
  N = 5 := 
by 
  intros
  sorry

end blocks_needed_l589_589750


namespace angle_triple_relation_l589_589105

variable (A B C D P M : Type)
variable [EuclideanGeometry P]

-- Conditions
variable (isParallelogram : parallelogram A B C D)
variable (twiceLength : ¬(vect A B ∥ vect A D) ∧ (dist B C = 2 * dist C D))
variable (isProjection : projection C onto A B = P )
variable (isMidpoint : midpoint D A = M )

-- Goal
theorem angle_triple_relation (α : Real) (h₁ : ∠ A P M = α) (h₂ : ∠ D M P = 3 * α) : 
  ∠ D M P = 3 * ∠ A P M := by
  sorry


end angle_triple_relation_l589_589105


namespace musical_puzzle_solution_l589_589129

def musical_puzzle_symbols := ['♭♭', 'A', 'C', 'B']
def translate_to_german_notation (symb : Char) : Char :=
  match symb with
  | 'B' => 'H'
  | _ => symb

theorem musical_puzzle_solution : 
  list.map translate_to_german_notation musical_puzzle_symbols = ['B', 'A', 'C', 'H'] :=
by
  sorry

end musical_puzzle_solution_l589_589129


namespace min_value_of_omega_l589_589250

theorem min_value_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ k : ℤ, ω = -10 * k + 5 / 2) → ω = 5 / 2 :=
begin
  sorry
end

end min_value_of_omega_l589_589250


namespace circle_through_points_l589_589435

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589435


namespace circle_equation_correct_l589_589533

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589533


namespace new_volume_correct_l589_589049

-- Define the conditions.
def initial_radius (r : ℝ) : Prop := true
def initial_height (h : ℝ) : Prop := true
def initial_volume (V : ℝ) : Prop := V = 15
def new_radius (r : ℝ) : ℝ := 3 * r
def new_height (h : ℝ) : ℝ := h / 2

-- Define the volume of a cylinder.
def volume_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- State the proof problem.
theorem new_volume_correct (r h : ℝ) (V : ℝ)
  (h1 : initial_radius r)
  (h2 : initial_height h)
  (h3 : initial_volume V) :
  volume_cylinder (new_radius r) (new_height h) = 67.5 :=
by
  sorry

end new_volume_correct_l589_589049


namespace circle_passes_through_points_l589_589522

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589522


namespace circle_through_points_l589_589598

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589598


namespace triangle_area_bound_by_line_l589_589150

noncomputable def area_of_triangle : ℝ :=
let x_intercept := 3 in
let y_intercept := 9 in
1 / 2 * x_intercept * y_intercept

theorem triangle_area_bound_by_line (x y : ℝ) (h : 3 * x + y = 9) :
  area_of_triangle = 13.5 :=
sorry

end triangle_area_bound_by_line_l589_589150


namespace factorization_x12_minus_729_l589_589210

theorem factorization_x12_minus_729 (x : ℝ) : 
  x^12 - 729 = (x^2 + 3) * (x^4 - 3 * x^2 + 9) * (x^3 - 3) * (x^3 + 3) :=
by sorry

end factorization_x12_minus_729_l589_589210


namespace investment_interest_rate_l589_589316

theorem investment_interest_rate 
(principal future_value : ℝ) (n : ℕ) 
(h₁ : principal = 12500)
(h₂ : future_value = 20000) 
(h₃ : n = 12) :
  ∃ r : ℝ, future_value = principal * (1 + r)^n ∧ r ≈ 0.0414 :=
by
  sorry

end investment_interest_rate_l589_589316


namespace saving_plan_at_year_13_l589_589771

noncomputable def interest_rate : ℝ := 0.02
noncomputable def initial_deposit : ℕ → ℝ
| y => if y < 7 then 20000 else 10000
noncomputable def compound_interest (n : ℕ) : ℝ := (1 + interest_rate) ^ n

noncomputable def sum_at_year_13 : ℝ :=
  let first_6_years := ∑ i in finset.range 6, initial_deposit (i+1) * compound_interest (12 - i)
  let next_6_years := ∑ i in finset.range 6, initial_deposit (i+7) * compound_interest (6 - i)
  (first_6_years + next_6_years) * compound_interest 6

theorem saving_plan_at_year_13 :
  abs (sum_at_year_13 / 10000 - 20.91) < 0.1 := 
sorry

end saving_plan_at_year_13_l589_589771


namespace equation_of_circle_through_three_points_l589_589565

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589565


namespace number_of_5_letter_words_with_at_least_two_vowels_l589_589303

theorem number_of_5_letter_words_with_at_least_two_vowels :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'];
  let vowels := ['A', 'E'];
  let consonants := ['B', 'C', 'D', 'F'];
  let total_combinations := 6^5;
  let combinations_with_one_vowel := 5 * 2 * 4^4;
  let combinations_with_no_vowel := 4^5;
  let at_least_two_vowels := total_combinations - (combinations_with_one_vowel + combinations_with_no_vowel);
  at_least_two_vowels = 4192 :=
by {
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'];
  let vowels := ['A', 'E'];
  let consonants := ['B', 'C', 'D', 'F'];
  let total_combinations := 6^5;
  let combinations_with_one_vowel := 5 * 2 * 4^4;
  let combinations_with_no_vowel := 4^5;
  let at_least_two_vowels := total_combinations - (combinations_with_one_vowel + combinations_with_no_vowel);
  show at_least_two_vowels = 4192, from sorry
}

end number_of_5_letter_words_with_at_least_two_vowels_l589_589303


namespace digit_in_thousandths_place_l589_589073

theorem digit_in_thousandths_place : 
  (Nat.digits 10 (7 * 10^3 / 25))^.get? 2 = some 0 := by
sorry

end digit_in_thousandths_place_l589_589073


namespace circle_through_points_l589_589466

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589466


namespace nearest_integer_to_sqrt_pqr_is_43_l589_589028

theorem nearest_integer_to_sqrt_pqr_is_43
  (r_a r_b r_c : ℝ)
  (h_ra : r_a = 10.5)
  (h_rb : r_b = 12)
  (h_rc : r_c = 14)
  (p q r : ℤ)
  (h_roots : ∃ (a b c : ℤ), (x^3 - p * x^2 + q * x - r = 0) ∧
                              (a, b, c are the roots)) :
  (Real.sqrt (p + q + r)).to_int = 43 := 
sorry

end nearest_integer_to_sqrt_pqr_is_43_l589_589028


namespace quadrilateral_pyramid_properties_l589_589036

variables {Point Segment Line Plane : Type}

structure Pyramid :=
(base_vertices : List Point)
(apex : Point)
(base_sides_midpoints : List Point)
(opposite_face_medians_intersections : List Point)

def intersect_and_divided (p : Pyramid) : Prop :=
  let segments := zip p.base_sides_midpoints p.opposite_face_medians_intersections
  ∃ (O : Point), ∀ s ∈ segments, ratio (s.1, O, s.2) = 3 / 2

def parallelogram_midpoints (p : Pyramid) : Prop :=
  let midpoints := segments_midpoints (zip p.base_sides_midpoints p.opposite_face_medians_intersections)
  is_parallelogram midpoints 

def area_ratio (p : Pyramid) : Prop :=
  let parallelogram_area := calculate_area(parallelogram_midpoints (zip p.base_sides_midpoints p.opposite_face_medians_intersections))
  let base_area := calculate_base_area p.base_vertices
  parallelogram_area / base_area = 1 / 72

theorem quadrilateral_pyramid_properties (p : Pyramid):
  intersect_and_divided p ∧ parallelogram_midpoints p ∧ area_ratio p :=
  sorry

end quadrilateral_pyramid_properties_l589_589036


namespace circle_through_points_l589_589430

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589430


namespace find_circle_equation_l589_589546

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589546


namespace termites_ate_black_squares_l589_589394

-- Define the overall dimensions of the chessboard
def chessboard_width : ℕ := 8
def chessboard_height : ℕ := 8

-- Defining the top-left corner color and the alternating color pattern
def top_left_color : string := "black"

-- Represent the eaten portion as a list of positions (row, col) (1-based indexing)
def eaten_portion : list (ℕ × ℕ) := [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), 
                                       (2,3), (2,4), (3,1), (3,2), (3,3), (3,4),
                                       (4,1), (4,2), (4,3), (4,4), (5,1), (5,2),
                                       (5,3), (5,4), (6,1), (6,2), (6,3), (6,4)]

-- Function to determine the color of a cell given its position
def cell_color (r c : ℕ) : string :=
  if (r + c) % 2 = 0 then "black" else "white"

-- Filter the eaten portion to count only the black cells
def count_black_cells (eaten : list (ℕ × ℕ)) : ℕ :=
  eaten.filter (λ rc, cell_color rc.1 rc.2 = "black").length

-- Main statement proving the problem
theorem termites_ate_black_squares :
  count_black_cells eaten_portion = 12 :=
by
  -- Placeholder for the actual proof
  sorry

end termites_ate_black_squares_l589_589394


namespace area_of_triangle_PQR_l589_589718

theorem area_of_triangle_PQR :
  let P := (-3, 2) in
  let Q := (1, 7) in
  let R := (4, 1) in
  let area := (P : ℝ × ℝ) * (Q : ℝ × ℝ) * (R : ℝ × ℝ) / 2) in
  area = 14.5 := sorry

end area_of_triangle_PQR_l589_589718


namespace train_length_l589_589779

theorem train_length :
  ∀ (speed_kmh : ℕ) (time_s : ℕ),
  speed_kmh = 72 →
  time_s = 12 →
  let speed_ms := (speed_kmh * 1000) / 3600 in
  let length_m := speed_ms * time_s in
  length_m = 240 :=
by
  intros speed_kmh time_s hk ht
  rw [hk, ht]
  let speed_ms := 72 * 1000 / 3600
  let length_m := speed_ms * 12
  have hs : speed_ms = 20 := by norm_num
  rw hs
  norm_num
  sorry

end train_length_l589_589779


namespace probability_sample_variance_le1_l589_589064

-- Let x1, x2, and x3 be the three distinct numbers drawn from {1, 2, ..., 10}
variables {x1 x2 x3 : ℕ}

-- Define the set
def S : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Conditions that x1, x2, and x3 are distinct elements from the set
axiom h_distinct : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3
axiom h_in_set_1 : x1 ∈ S
axiom h_in_set_2 : x2 ∈ S
axiom h_in_set_3 : x3 ∈ S

-- Define the sample mean
def sample_mean : ℚ := (x1 + x2 + x3 : ℚ) / 3

-- Define the sample variance
def sample_variance : ℚ :=
  ((x1 - sample_mean)^2 + (x2 - sample_mean)^2 + (x3 - sample_mean)^2) / 2

-- Define the theorem
theorem probability_sample_variance_le1 : 
  (∃ x1 x2 x3 ∈ S, h_distinct ∧ sample_variance ≤ 1) →
  (prob_in_S := (8 : ℚ / 120)) := sorry

end probability_sample_variance_le1_l589_589064


namespace cone_apex_angle_l589_589070

theorem cone_apex_angle 
  (l R r : ℝ) 
  (h_surface_area : 6 * r^2 = R * l)
  (h_similar_triangles : ∀ l R r : ℝ, l + R = R * (l^2 - R^2).sqrt / r) :
  ∃ (φ₁ φ₂ : ℝ),
    (φ₁ = 30 ∨ φ₁ ≡ (30 : ℝ)) ∧ (φ₂ ≡ 19 + 28/60 + 16.5/3600 ∨ φ₂ ≈ 19 + 28/60 + 16.5/3600/),
    2 * φ₁ = 60 ∧ 2 * φ₂ ≈ 38 + 56/60 + 33/3600 :=
by
  sorry

end cone_apex_angle_l589_589070


namespace rectangle_pentagon_ratio_l589_589134

theorem rectangle_pentagon_ratio
  (l w p : ℝ)
  (h1 : l = 2 * w)
  (h2 : 2 * (l + w) = 30)
  (h3 : 5 * p = 30) :
  l / p = 5 / 3 :=
by
  sorry

end rectangle_pentagon_ratio_l589_589134


namespace triangle_area_l589_589145

theorem triangle_area (x y : ℝ) (h : 3 * x + y = 9) : 
  (1 / 2) * 3 * 9 = 13.5 :=
sorry

end triangle_area_l589_589145


namespace circle_equation_through_points_l589_589648

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589648


namespace circle_through_points_l589_589597

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589597


namespace circle_passing_through_points_l589_589615

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589615


namespace length_of_each_stone_l589_589124

-- Define the dimensions of the hall in decimeters
def hall_length_dm : ℕ := 36 * 10
def hall_breadth_dm : ℕ := 15 * 10

-- Define the width of each stone in decimeters
def stone_width_dm : ℕ := 5

-- Define the number of stones
def number_of_stones : ℕ := 1350

-- Define the total area of the hall
def hall_area : ℕ := hall_length_dm * hall_breadth_dm

-- Define the area of one stone
def stone_area : ℕ := hall_area / number_of_stones

-- Define the length of each stone and state the theorem
theorem length_of_each_stone : (stone_area / stone_width_dm) = 8 :=
by
  sorry

end length_of_each_stone_l589_589124


namespace athlete_heartbeats_during_race_l589_589831

theorem athlete_heartbeats_during_race :
  ∀ (heartbeats_per_minute pace_minutes_per_mile race_miles : ℕ ),
    heartbeats_per_minute = 160 →
    pace_minutes_per_mile = 6 →
    race_miles = 30 →
    heartbeats_per_minute * pace_minutes_per_mile * race_miles = 28800 :=
by
  intros heartbeats_per_minute pace_minutes_per_mile race_miles
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end athlete_heartbeats_during_race_l589_589831


namespace lcm_trumpet_flute_piano_l589_589204

theorem lcm_trumpet_flute_piano :
  let trumpet_days : Nat := 11
  let flute_days : Nat := 3
  let piano_days : Nat := 7
  Nat.lcm trumpet_days (Nat.lcm flute_days piano_days) = 231 :=
begin
  -- Variable definitions
  let trumpet_days := 11
  let flute_days := 3
  let piano_days := 7,

  -- LCM calculation and result assertion
  have h_lcm := Nat.lcm_assoc trumpet_days flute_days piano_days,
  rw [h_lcm, Nat.lcm_comm flute_days piano_days],
  simp [Nat.lcm, trumpet_days, flute_days, piano_days],
  sorry
end

end lcm_trumpet_flute_piano_l589_589204


namespace max_value_of_s_l589_589357

-- Define the conditions
variables (p q r s : ℝ)

-- Add assumptions
axiom h1 : p + q + r + s = 10
axiom h2 : p * q + p * r + p * s + q * r + q * s + r * s = 20

-- State the theorem
theorem max_value_of_s : s ≤ (5 + real.sqrt 105) / 2 :=
sorry

end max_value_of_s_l589_589357


namespace circle_equation_l589_589623

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589623


namespace smallest_number_to_multiply_l589_589729

noncomputable def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem smallest_number_to_multiply (k : ℕ) (h : k = 1152) (h1 : 1152 = 2^7 * 3^1) :
  ∃ (x : ℕ), x = 6 ∧ isPerfectSquare (k * x) :=
by
  have k_factorization : k = 2^7 * 3^1 := by rw [h, h1]
  use 6
  split
  . rfl
  . sorry

end smallest_number_to_multiply_l589_589729


namespace integer_solutions_l589_589220

-- Define the polynomial equation as a predicate
def polynomial (n : ℤ) : Prop := n^5 - 2 * n^4 - 7 * n^2 - 7 * n + 3 = 0

-- The theorem statement
theorem integer_solutions :
  {n : ℤ | polynomial n} = {-1, 3} :=
by 
  sorry

end integer_solutions_l589_589220


namespace circle_passing_through_points_eqn_l589_589486

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589486


namespace price_of_72_cans_is_18_36_l589_589694

def regular_price_per_can : ℝ := 0.30
def discount_percent : ℝ := 0.15
def number_of_cans : ℝ := 72

def discounted_price_per_can : ℝ := regular_price_per_can - (discount_percent * regular_price_per_can)
def total_price (num_cans : ℝ) : ℝ := num_cans * discounted_price_per_can

theorem price_of_72_cans_is_18_36 :
  total_price number_of_cans = 18.36 :=
by
  /- Proof details omitted -/
  sorry

end price_of_72_cans_is_18_36_l589_589694


namespace num_of_nickels_is_two_l589_589766

theorem num_of_nickels_is_two (d n : ℕ) 
    (h1 : 10 * d + 5 * n = 70) 
    (h2 : d + n = 8) : 
    n = 2 := 
by 
    sorry

end num_of_nickels_is_two_l589_589766


namespace minimum_circle_area_l589_589290

open Real

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the points of tangency
def x1 (a : ℝ) : ℝ := a - sqrt (a^2 + 1)
def x2 (a : ℝ) : ℝ := a + sqrt (a^2 + 1)

-- Define point P
def P (a : ℝ) : ℝ × ℝ := (a, -1)

-- Define the circle's radius formula
def radius (a : ℝ) : ℝ := 2 * a^2 + 2 / sqrt (4 * a^2 + 1)

-- Define the area of the circle
def area (a : ℝ) : ℝ := π * radius a ^ 2

-- Theorem: Find points of tangency and the minimum area of the circle
theorem minimum_circle_area : ∀ (a : ℝ), 
  let x1 := x1 a,
      x2 := x2 a in
  x1 = a - sqrt (a^2 + 1) ∧ x2 = a + sqrt (a^2 + 1) ∧
  (∀ (a : ℝ), (a = sqrt 2 / 2 ∨ a = -sqrt 2 / 2) → area a = 3 * π) :=
by
  intros
  sorry

end minimum_circle_area_l589_589290


namespace equation_of_circle_ABC_l589_589445

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589445


namespace perfect_square_sum_remainder_l589_589360

-- Define the condition that n is a positive integer and n^2 + 12n - 507 is a perfect square
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n^2 + 12*n - 507 = m^2

-- Prove that the sum of all such n modulo 1000 is 358
theorem perfect_square_sum_remainder :
  (∑ n in Finset.filter (λ n, isPerfectSquare n) (Finset.Icc 1 1000), 
     n) % 1000 = 358 :=
  sorry

end perfect_square_sum_remainder_l589_589360


namespace gcd_45_75_eq_15_l589_589895

theorem gcd_45_75_eq_15 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_eq_15_l589_589895


namespace number_of_odd_m_equals_phi_l589_589350

/-- Definition of the function f: ℕ → ℕ --/
def f : ℕ → ℕ
| 1       := 1
| (2 * n) := f n
| (2 * n + 1) := f n + f (n + 1)

/-- Euler's totient function φ(n) which counts the number of positive integers up to n that are coprime with n. --/
def phi (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (nat.coprime n).card

theorem number_of_odd_m_equals_phi (n : ℕ) : 
  (Finset.range (n + 1)).filter (λ m, m % 2 = 1 ∧ f m = n).card = phi n := 
sorry

end number_of_odd_m_equals_phi_l589_589350


namespace negation_of_existence_l589_589380

theorem negation_of_existence (m : ℝ) : ¬ (∃ m : ℝ, ∃ x : ℂ, x^2 + m * x + 1 = 0) ↔ ∀ m : ℝ, ∀ x : ℂ, x^2 + m * x + 1 ≠ 0 :=
begin
  sorry
end

end negation_of_existence_l589_589380


namespace area_of_circle_tangent_to_square_side_tangent_and_through_vertices_l589_589683

-- Definitions of the square and circle properties
def side_length : ℝ := 4
def radius : ℝ := 5 / 2

-- Area of the circle to be proved
def circle_area : ℝ := π * radius * radius

theorem area_of_circle_tangent_to_square_side_tangent_and_through_vertices :
  circle_area = 25 * π / 4 :=
by
  simp [circle_area, radius]
  sorry

end area_of_circle_tangent_to_square_side_tangent_and_through_vertices_l589_589683


namespace roger_earned_54_dollars_l589_589108

-- Definitions based on problem conditions
def lawns_had : ℕ := 14
def lawns_forgot : ℕ := 8
def earn_per_lawn : ℕ := 9

-- The number of lawns actually mowed
def lawns_mowed : ℕ := lawns_had - lawns_forgot

-- The amount of money earned
def money_earned : ℕ := lawns_mowed * earn_per_lawn

-- Proof statement: Roger actually earned 54 dollars
theorem roger_earned_54_dollars : money_earned = 54 := sorry

end roger_earned_54_dollars_l589_589108


namespace find_polynomial_h_l589_589391

theorem find_polynomial_h (f h : ℝ → ℝ) (hf : ∀ x, f x = x^2) (hh : ∀ x, f (h x) = 9 * x^2 + 6 * x + 1) : 
  (∀ x, h x = 3 * x + 1) ∨ (∀ x, h x = -3 * x - 1) :=
by
  sorry

end find_polynomial_h_l589_589391


namespace number_of_pairs_l589_589190

theorem number_of_pairs (S : Set ℕ) (hS : S = {n | 1 ≤ n ∧ n ≤ 100}) :
  ∃ (n : ℕ), n = 49 ∧ ∀ (a b ∈ S), a + b = 100 → a ≠ b :=
begin
  sorry
end

end number_of_pairs_l589_589190


namespace circle_equation_l589_589398

theorem circle_equation (m : ℝ) (h1 : m > 0) (h2 : m = 5) : 
  ∃ (r : ℝ), r = 3 ∧ (∀ x y : ℝ, ((x - m)^2 + y^2 = r^2) ↔ ((x - 5) ^ 2 + y ^ 2 = 9)) := 
by
  -- Definitions and conditions
  let r := 3
  have hp : m = 5 := h2
  use r
  split
  -- Prove radius is 3
  exact rfl
  -- Prove circle equation
  intro x y
  split
  {
    intro h
    rw [←hp]
    exact h
  }
  {
    intro h
    rw [←hp] at h
    exact h
  }

end circle_equation_l589_589398


namespace find_circle_equation_l589_589561

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589561


namespace equation_of_circle_ABC_l589_589454

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589454


namespace monotonicity_of_f_range_of_a_l589_589284

-- Definitions of given functions
def f (a x : ℝ) := 1 / 2 * x^2 - (a + 2) * x + 2 * a * Real.log x
def g (a x : ℝ) := - (a + 2) * x

-- Statements of the proof problems

-- Part (1): Monotonicity of f(x)
theorem monotonicity_of_f (a : ℝ) (x : ℝ) (h : a > 0) : 
  (a = 2 ∧ ∀ x > 0, f a x > 0) ∨
  (0 < a ∧ a < 2 ∧ ∀ x, (x ∈ (0, a) ∪ (2, +∞)) → (f a x > 0) ∧ (x ∈ (a, 2)) → (f a x < 0)) ∨
  (a > 2 ∧ ∀ x, (x ∈ (0, 2) ∪ (a, +∞)) → (f a x > 0) ∧ (x ∈ (2, a)) → (f a x < 0)) :=
sorry

-- Part (2): Range of a for f(x_0) > g(x_0)
theorem range_of_a (a : ℝ) (x₀ : ℝ) (h₁ : x₀ ∈ Set.Icc (Real.exp 1) 4) (h₂ : f a x₀ > g a x₀) :
  a > -2 / Real.log 2 :=
sorry

end monotonicity_of_f_range_of_a_l589_589284


namespace negation_proposition_equivalence_l589_589038

theorem negation_proposition_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 2 * x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) :=
by
  sorry

end negation_proposition_equivalence_l589_589038


namespace earrings_ratio_l589_589126

noncomputable def initial_necklaces : ℕ := 10
noncomputable def initial_earrings : ℕ := 15
noncomputable def bought_necklaces : ℕ := 10
noncomputable def bought_earrings : ℕ := (2 / 3 : ℝ) * initial_earrings
noncomputable def total_pieces_after_buying : ℕ := (initial_necklaces + bought_necklaces) + (initial_earrings + bought_earrings)
noncomputable def total_pieces_after_mother : ℕ := 57
noncomputable def earrings_from_mother : ℕ := total_pieces_after_mother - total_pieces_after_buying

theorem earrings_ratio
  (hn_initial : initial_necklaces = 10)
  (he_initial : initial_earrings = 15)
  (hn_bought : bought_necklaces = 10)
  (he_bought : bought_earrings = (2 / 3) * 15)
  (htotal_after_buying : total_pieces_after_buying = 45)
  (htotal_after_mother : total_pieces_after_mother = 57)
  (mother_given_earrings : earrings_from_mother = 12) :
  (earrings_from_mother / bought_earrings : ℝ) = 6 / 5 :=
by sorry

end earrings_ratio_l589_589126


namespace eric_green_marbles_l589_589881

theorem eric_green_marbles (total_marbles white_marbles blue_marbles : ℕ) (h_total : total_marbles = 20)
  (h_white : white_marbles = 12) (h_blue : blue_marbles = 6) :
  total_marbles - (white_marbles + blue_marbles) = 2 := 
by
  sorry

end eric_green_marbles_l589_589881


namespace circle_through_points_l589_589420

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589420


namespace factorization_of_polynomial_l589_589209

theorem factorization_of_polynomial (x : ℂ) :
  x^12 - 729 = (x^2 + 3) * (x^4 - 3 * x^2 + 9) * (x^2 - 3) * (x^4 + 3 * x^2 + 9) :=
by sorry

end factorization_of_polynomial_l589_589209


namespace jason_initial_cards_l589_589343

/-- Jason initially had some Pokemon cards, Alyssa bought him 224 more, 
and now Jason has 900 Pokemon cards in total.
Prove that initially Jason had 676 Pokemon cards. -/
theorem jason_initial_cards (a b c : ℕ) (h_a : a = 224) (h_b : b = 900) (h_cond : b = a + 676) : 676 = c :=
by 
  sorry

end jason_initial_cards_l589_589343


namespace circle_equation_through_points_l589_589646

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589646


namespace triangle_area_l589_589149

theorem triangle_area (x y : ℝ) (h : 3 * x + y = 9) : 
  (1 / 2) * 3 * 9 = 13.5 :=
sorry

end triangle_area_l589_589149


namespace circle_through_points_l589_589469

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589469


namespace eq_eval_expression_l589_589726

theorem eq_eval_expression (x : ℝ) 
  (h1 : x ≥ 1/8) 
  (h2 : x ≠ 1/4) : 
  (4 * x - 1) * ((1 / (8 * x)) * (((sqrt (8 * x - 1) + 4 * x)⁻¹ - (sqrt (8 * x - 1) - 4 * x)⁻¹)))^(1/2) = 
  if x ∈ Icc (1/8 : ℝ) (1/4 : ℝ) then -1 else 1 := 
sorry

end eq_eval_expression_l589_589726


namespace athlete_heartbeats_during_race_l589_589828

theorem athlete_heartbeats_during_race :
  ∀ (heartbeats_per_minute pace_minutes_per_mile race_miles : ℕ ),
    heartbeats_per_minute = 160 →
    pace_minutes_per_mile = 6 →
    race_miles = 30 →
    heartbeats_per_minute * pace_minutes_per_mile * race_miles = 28800 :=
by
  intros heartbeats_per_minute pace_minutes_per_mile race_miles
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end athlete_heartbeats_during_race_l589_589828


namespace capulet_juliet_probability_l589_589336

noncomputable def probability_juliet_in_capulet : ℚ :=
let
  montague_population := 0.30,
  capulet_population := 0.40,
  escalus_population := 0.20,
  verona_population := 0.10,
  juliet_support_montague := 0.20 * montague_population,
  juliet_support_capulet := 0.40 * capulet_population,
  juliet_support_escalus := 0.30 * escalus_population,
  juliet_support_verona := 0.50 * verona_population,
  total_juliet_support := juliet_support_montague + juliet_support_capulet + juliet_support_escalus + juliet_support_verona
in
  juliet_support_capulet / total_juliet_support * 100

theorem capulet_juliet_probability :
  probability_juliet_in_capulet = 48 := by
    sorry

end capulet_juliet_probability_l589_589336


namespace convex_polygons_4017_l589_589066

theorem convex_polygons_4017 (points : Finset ℕ) (h_points : points.card = 12) :
  (∑ k in Finset.range 13, if 3 ≤ k then Nat.choose 12 k else 0) = 4017 :=
by
  -- The proof is omitted here.
  sorry

end convex_polygons_4017_l589_589066


namespace greatest_difference_units_digit_l589_589698

theorem greatest_difference_units_digit :
  ∃ units_digit : ℕ, (units_digit = 0 ∨ units_digit = 3 ∨ units_digit = 6 ∨ units_digit = 9) ∧
  (∀ d1 d2 : ℕ, (d1 = 0 ∨ d1 = 3 ∨ d1 = 6 ∨ d1 = 9) ∧ (d2 = 0 ∨ d2 = 3 ∨ d2 = 6 ∨ d2 = 9) → abs (d1 - d2) ≤ 9) :=
begin
  use 9,
  split,
  { left, refl },
  { intros d1 d2 h1 h2,
    by_cases h : d1 = 9 ∧ d2 = 0,
    { rw [h.1, h.2], norm_num },
    { rw [abs_sub_le_iff], split,
      norm_num,
      norm_num } }
end

end greatest_difference_units_digit_l589_589698


namespace calculate_polygon_sides_l589_589135

-- Let n be the number of sides of the regular polygon with each exterior angle of 18 degrees
def regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : Prop :=
  exterior_angle = 18 ∧ n * exterior_angle = 360

theorem calculate_polygon_sides (n : ℕ) (exterior_angle : ℝ) :
  regular_polygon_sides n exterior_angle → n = 20 :=
by
  intro h
  sorry

end calculate_polygon_sides_l589_589135


namespace no_common_points_of_parallel_line_and_plane_l589_589275

open Set

variables (Point Line Plane : Type) [inhabited Point]
variables (α : Plane) (a b : Line) (p : Point)

-- c : Line parallel to Plane α
def parallel_to_plane (l : Line) (α : Plane) : Prop :=
  ∀ p1 p2 : Point, (p1 ∉ α ∨ p2 ∉ α) ∨ (p1 ≠ p2 ∧ ∃ q ∈ α, q = (p1 + p2) / 2)

-- c : b lies in Plane α
def lies_in_plane (l : Line) (α : Plane) : Prop :=
  ∀ p : Point, p ∈ l → p ∈ α

-- The Lean proof problem statement
theorem no_common_points_of_parallel_line_and_plane (h₁ : parallel_to_plane a α) (h₂ : lies_in_plane b α) : ∀ p : Point, p ∉ a ∧ p ∈ b :=
begin
  sorry
end

end no_common_points_of_parallel_line_and_plane_l589_589275


namespace circle_through_points_l589_589429

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589429


namespace coefficient_x3_product_l589_589072

def f (x : ℚ) := x^5 - 4*x^3 + 6*x^2 - 8*x + 2
def g (x : ℚ) := 3*x^4 - 2*x^3 + x^2 + 5*x + 9

theorem coefficient_x3_product : 
  (f * g).coeff 3 = 18 :=
by
  sorry

end coefficient_x3_product_l589_589072


namespace cylinder_new_volume_proof_l589_589052

noncomputable def cylinder_new_volume (V : ℝ) (r h : ℝ) : ℝ := 
  let new_r := 3 * r
  let new_h := h / 2
  π * (new_r^2) * new_h

theorem cylinder_new_volume_proof (r h : ℝ) (π : ℝ) 
  (h_volume : π * r^2 * h = 15) : 
  cylinder_new_volume (π * r^2 * h) r h = 67.5 := 
by 
  unfold cylinder_new_volume
  rw [←h_volume]
  sorry

end cylinder_new_volume_proof_l589_589052


namespace circle_equation_l589_589634

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589634


namespace acute_triangle_probability_l589_589948

open Finset

noncomputable def isAcuteTriangleProb (n : ℕ) : Prop :=
  ∃ k : ℕ, (n = 2 * k ∧ (3 * (k - 2)) / (2 * (2 * k - 1)) = 93 / 125) ∨ (n = 2 * k + 1 ∧ (3 * (k - 1)) / (2 * (2 * k - 1)) = 93 / 125)

theorem acute_triangle_probability (n : ℕ) : isAcuteTriangleProb n → n = 376 ∨ n = 127 :=
by
  sorry

end acute_triangle_probability_l589_589948


namespace find_fourth_vertex_l589_589697

structure Point where
  x : ℝ
  y : ℝ

def vertices : List Point :=
  [ {x := 1, y := 1}, {x := 2, y := 2}, {x := 3, y := -1} ]

def isParallelogram (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x) = (p4.x - p3.x) ∧ (p2.y - p1.y) = (p4.y - p3.y) ∧ 
  (p3.x - p1.x) = (p4.x - p2.x) ∧ (p3.y - p1.y) = (p4.y - p2.y)

def fourth_vertex_1 : Point := {x := 2, y := -2}
def fourth_vertex_2 : Point := {x := 4, y := 0}

theorem find_fourth_vertex :
  ∃ (p4 : Point), (p4 = fourth_vertex_1 ∨ p4 = fourth_vertex_2) ∧ isParallelogram (vertices.nth 0) (vertices.nth 1) (vertices.nth 2) p4 :=
sorry

end find_fourth_vertex_l589_589697


namespace circle_equation_correct_l589_589538

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589538


namespace equation_of_circle_ABC_l589_589442

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589442


namespace circle_passing_through_points_l589_589504

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589504


namespace real_solutions_count_l589_589863

theorem real_solutions_count :
  ∃! (sols : Finset (ℝ × ℝ × ℝ × ℝ)), (∃ f : ℝ × ℝ × ℝ × ℝ → (ℝ × ℝ × ℝ × ℝ), 
  (∀ x y z w, (f (x, y, z, w) = (x, y, z, w) ↔ 
    (x = w + z + y*z*x) ∧ 
    (y = z + x + z*x*y) ∧ 
    (z = x + w + x*z*w) ∧ 
    (w = y + z + y*z*w))) ∧ 
  sols = Finset.filter (λ sol, f sol = sol) (Finset.univ)) ∧ sols.card = 9 :=
by
  sorry

end real_solutions_count_l589_589863


namespace equation_of_circle_passing_through_points_l589_589667

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589667


namespace five_digit_numbers_count_l589_589107

theorem five_digit_numbers_count : 
  let number_of_digits := 5
  let number_of_configurations := 3
  let a_choices := 9  -- 1 through 9
  let b_choices := 9  -- 0 through 9, but excluding a
  let c_choices := 8  -- 0 through 9, but excluding a and b
  let total_combinations_per_configuration := a_choices * b_choices * c_choices
 in 
 total_combinations_per_configuration * number_of_configurations = 1944 :=
by sorry

end five_digit_numbers_count_l589_589107


namespace circle_passing_through_points_l589_589611

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589611


namespace problem1_problem2_l589_589258

noncomputable def z : ℂ := 1 + Complex.I

def question1 (w : ℂ) : Prop :=
  w = z^2 + 3 * Complex.conj z - 4

def answer1 : ℂ :=
  -1 - Complex.I

theorem problem1 : ∃ w : ℂ, question1 w ∧ w = answer1 :=
by
  use z^2 + 3 * Complex.conj z - 4
  split
  · rfl
  · rfl
  sorry

def question2 (a b : ℝ) : Prop :=
  (z^2 + a * z + b) / (z^2 - z + 1) = 1 - Complex.I

def answer2 : ℝ × ℝ :=
  (-1, 2)

theorem problem2 : ∃ a b : ℝ, question2 a b ∧ (a, b) = answer2 :=
by
  use -1, 2
  split
  · dsimp [question2]
    rw [← Complex.of_real_inj, Complex.of_real_div]
    sorry
  · rfl

end problem1_problem2_l589_589258


namespace athlete_heartbeats_l589_589810

theorem athlete_heartbeats
  (heart_beats_per_minute : ℕ)
  (pace_minutes_per_mile : ℕ)
  (total_miles : ℕ)
  (heart_beats_per_minute_eq : heart_beats_per_minute = 160)
  (pace_minutes_per_mile_eq : pace_minutes_per_mile = 6)
  (total_miles_eq : total_miles = 30) :
  heart_beats_per_minute * pace_minutes_per_mile * total_miles = 28800 :=
by {
  have h1 : heart_beats_per_minute = 160 := heart_beats_per_minute_eq,
  have h2 : pace_minutes_per_mile = 6 := pace_minutes_per_mile_eq,
  have h3 : total_miles = 30 := total_miles_eq,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end athlete_heartbeats_l589_589810


namespace find_center_radius_l589_589397

def center_radius_of_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 3)^2 = 16

theorem find_center_radius : 
  ∃ (h k r : ℝ), center_radius_of_circle h k ∧ h = -2 ∧ k = 3 ∧ r = 4 :=
by
  use [-2, 3, 4]
  split
  -- Proof goes here
  sorry

end find_center_radius_l589_589397


namespace area_of_triangle_ABC_l589_589274

noncomputable def area_of_triangle (a b c A : ℝ) : ℝ :=
  0.5 * b * c * Real.sin A

theorem area_of_triangle_ABC :
  ∀ (a b c A: ℝ), b = 3 → (a - c = 2) → A = (2 * Real.pi) / 3 →
  a = c + 2 →
  a = 7 → c = 5 →
  area_of_triangle a b c A = 15 * Real.sqrt(3) / 4 :=
by
  intros a b c A hb h_ac hA ha hc 
  rw [hb, h_ac, hA, ha, hc]
  sorry

end area_of_triangle_ABC_l589_589274


namespace circle_passing_through_points_eqn_l589_589489

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589489


namespace max_value_of_s_l589_589356

-- Define the conditions
variables (p q r s : ℝ)

-- Add assumptions
axiom h1 : p + q + r + s = 10
axiom h2 : p * q + p * r + p * s + q * r + q * s + r * s = 20

-- State the theorem
theorem max_value_of_s : s ≤ (5 + real.sqrt 105) / 2 :=
sorry

end max_value_of_s_l589_589356


namespace range_of_t_l589_589312

noncomputable def f : ℝ → ℝ := sorry
lemma f_decreasing : ∀ x y : ℝ, x < y → f y < f x := sorry
lemma f_zero : f 0 = 3 := sorry
lemma f_three : f 3 = -1 := sorry
def P (t : ℝ) : set ℝ := {x : ℝ | abs (f (x + t) - 1) < 2}
def Q : set ℝ := {x : ℝ | f x < -1}

theorem range_of_t :
  ∀ t : ℝ, (∀ x, x ∈ P t → x ∈ Q) ∧ (∃ x, x ∉ Q ∧ x ∈ P t) ↔ t ≤ -3 :=
sorry

end range_of_t_l589_589312


namespace circle_passing_through_points_l589_589502

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589502


namespace circle_line_intersect_points_l589_589040

-- Define the center and radius of the circle
def circle_center : ℝ × ℝ := (3, 3)
def circle_radius : ℝ := 2

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 16 = 0

-- Distance point (a, b) to line 3x + 4y - 16 = 0
def point_line_distance (a b : ℝ) : ℝ :=
  |3 * a + 4 * b - 16| / 5

-- The number of points on the circle (x-3)² + (y-3)² = 4 
-- that are at a distance of 1 from the line 3x + 4y - 16 = 0
theorem circle_line_intersect_points : 
  ∃ p : ℝ × ℝ, (p.1 - 3) ^ 2 + (p.2 - 3) ^ 2 = 4 ∧ 
                 point_line_distance p.1 p.2 = 1 :=
sorry

end circle_line_intersect_points_l589_589040


namespace triangle_area_l589_589171

-- Define the line equation 3x + y = 9
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the triangle bounded by the coordinate axes and the line
def bounded_triangle (x y : ℝ) : Prop :=
  (x = 0 ∧ 0 ≤ y ∧ y ≤ 9) ∨  -- y-axis segment
  (y = 0 ∧ 0 ≤ x ∧ x ≤ 3) ∨  -- x-axis segment
  (3 * x + y = 9 ∧ 0 ≤ x ∧ 0 ≤ y)  -- line segment

-- The area of the triangle formed by the coordinate axes and the line 3x + y = 9
theorem triangle_area : 
  let area := 1/2 * 3 * 9 in -- The calculation of the area
  area = 13.5 := 
by
  sorry

end triangle_area_l589_589171


namespace circle_equation_l589_589619

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589619


namespace number_of_zeros_in_interval_l589_589272

def f (x : ℝ) : ℝ := 
  if 1 ≤ x ∧ x < 2 then 1 - abs (2 * x - 3) 
  else if x ≥ 2 then 1 / 2 * f (x / 2) 
  else 0

theorem number_of_zeros_in_interval :
  let g (x : ℝ) := 2 * x * f x - 3 in
  ∃ n, n = 11 ∧ ∀ y ∈ set.Ioo 1 2015, g y = 0 → (n = 11) :=
sorry

end number_of_zeros_in_interval_l589_589272


namespace consecutive_composites_l589_589915

theorem consecutive_composites 
  (a t d r : ℕ) (h_a_comp : ∃ p q, p > 1 ∧ q > 1 ∧ a = p * q)
  (h_t_comp : ∃ p q, p > 1 ∧ q > 1 ∧ t = p * q)
  (h_d_comp : ∃ p q, p > 1 ∧ q > 1 ∧ d = p * q)
  (h_r_pos : r > 0) :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k < r → ∃ m : ℕ, m > 1 ∧ m ∣ (a * t^(n + k) + d) :=
  sorry

end consecutive_composites_l589_589915


namespace remainder_sum_59_l589_589722

theorem remainder_sum_59 (x y z : ℕ) (h1 : x % 59 = 30) (h2 : y % 59 = 27) (h3 : z % 59 = 4) :
  (x + y + z) % 59 = 2 := 
sorry

end remainder_sum_59_l589_589722


namespace part1_part2_l589_589368

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, sin x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (cos x, sin x)

def magnitude_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

theorem part1 (x : ℝ) (h : x ∈ set.Icc 0 (π / 2))
  (h_mag : magnitude_squared (vector_a x) = magnitude_squared (vector_b x)) : 
  x = π / 6 := by
  sorry

noncomputable def f (x : ℝ) : ℝ := 
  let a := vector_a x
  let b := vector_b x
  a.1 * b.1 + a.2 * b.2

theorem part2 (k : ℤ) :
  ∃ x ∈ [0, π / 2], 
  f x = 3 / 2 ∧ 
  ∀ x ∈ [k * π - π / 6, k * π + π / 3], 
    ∀ y ∈ [k * π - π / 6, k * π + π / 3], 
    x < y → f x < f y :=
by
  sorry

end part1_part2_l589_589368


namespace circle_equation_l589_589631

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589631


namespace wrapping_paper_l589_589007

theorem wrapping_paper (total_used_per_roll : ℚ) (number_of_presents : ℕ) (fraction_used : ℚ) (fraction_left : ℚ) 
  (h1 : total_used_per_roll = 2 / 5) 
  (h2 : number_of_presents = 5) 
  (h3 : fraction_used = total_used_per_roll / number_of_presents) 
  (h4 : fraction_left = 1 - total_used_per_roll) : 
  fraction_used = 2 / 25 ∧ fraction_left = 3 / 5 := 
by 
  sorry

end wrapping_paper_l589_589007


namespace problem1_problem2_l589_589299

-- Given conditions
variables {A B C E I O : Point}
variables {R : ℝ}
variables (circumcircle : Circle O R)

-- Given angles
variables (angle_B : ∠ B = 60)
variables (angle_A_less_than_angle_C : ∠ A < ∠ C)
variables (external_angle_bisector : SeparatedAt A E circumcircle)

-- The questions: proving the statements
theorem problem1 : IO = AE := 
sorry

theorem problem2 : 2 * R < IO + IA + IC ∧ IO + IA + IC < (1 + sqrt 2) * R :=
sorry

end problem1_problem2_l589_589299


namespace triangle_area_l589_589167

noncomputable def line_eq := λ x : ℝ, 9 - 3 * x

theorem triangle_area :
    let x_intercept := (3 : ℝ)
    let y_intercept := (9 : ℝ)
    let triangle_base := x_intercept
    let triangle_height := y_intercept
    let area := (1 / 2) * triangle_base * triangle_height
    area = 13.5 :=
by
    sorry

end triangle_area_l589_589167


namespace area_of_triangle_l589_589180

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the points of intercepts with axes
def intercepts (x0 y0 x1 y1 : ℝ) : Prop :=
  line_eq 0 y0 ∧ y0 = 9 ∧
  line_eq x1 0 ∧ x1 = 3

-- Define the area of the triangle
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem area_of_triangle : ∃ (x0 y0 x1 y1 : ℝ), 
  intercepts x0 y0 x1 y1 ∧ triangle_area x1 y0 = 13.5 :=
by
  sorry

end area_of_triangle_l589_589180


namespace circle_passing_through_points_l589_589492

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589492


namespace athlete_heartbeats_during_race_l589_589822

theorem athlete_heartbeats_during_race
  (heart_rate : ℕ)
  (pace : ℕ)
  (distance : ℕ)
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) :
  heart_rate * pace * distance = 28800 :=
by
  rw [H1, H2, H3]
  norm_num

end athlete_heartbeats_during_race_l589_589822


namespace eta_expectation_and_variance_l589_589968

open Probability

def xi : PMF ℕ := PMF.binomial 5 0.5

def η : ℕ → ℝ := λ x => 5 * x

theorem eta_expectation_and_variance :
  (E (η <$> xi) = 25 / 2) ∧ (variance (η <$> xi) = 125 / 4) :=
by
  sorry

end eta_expectation_and_variance_l589_589968


namespace jerryEarningsPastFourNights_l589_589344

-- Define conditions as variables and the final assertion to be proved as the goal.
def jerryEarnings (earnings: ℕ → ℕ) (n: ℕ): Prop :=
  n = 4 ∧ earnings(5) = 115 ∧ (5 * 50 = 250) →
  ∑ i in (finset.range 5).filter (≠ 4), earnings i = 135

-- Prove that given the conditions, Jerry's earnings for the past four nights were 135
theorem jerryEarningsPastFourNights (earnings: ℕ → ℕ):
  jerryEarnings earnings 4 :=
sorry

end jerryEarningsPastFourNights_l589_589344


namespace parabola_vertex_l589_589240

theorem parabola_vertex (x y : ℝ) (h : y = -2 * x^2 - 20 * x - 50) : (m n : ℝ) (m = -5 ∧ n = 0) :=
sorry

end parabola_vertex_l589_589240


namespace total_games_played_l589_589732

open Nat

def number_of_teams : ℕ := 12

def combination (n k : ℕ) : ℕ := n.fact / (k.fact * (n - k).fact)

def games_played (n : ℕ) : ℕ := combination n 2

theorem total_games_played : games_played number_of_teams = 66 := by
  sorry

end total_games_played_l589_589732


namespace equation_of_circle_ABC_l589_589446

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589446


namespace num_eq_7_times_sum_of_digits_l589_589990

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem num_eq_7_times_sum_of_digits : ∃! n < 1000, n = 7 * sum_of_digits n :=
sorry

end num_eq_7_times_sum_of_digits_l589_589990


namespace max_intersections_of_line_with_four_coplanar_circles_l589_589906

theorem max_intersections_of_line_with_four_coplanar_circles
  (C1 C2 C3 C4 : ℝ) -- Four coplanar circles
  (h1 : is_circle C1)
  (h2 : is_circle C2)
  (h3 : is_circle C3)
  (h4 : is_circle C4)
  : ∃ l : ℝ → ℝ, ∀ C ∈ {C1, C2, C3, C4}, (number_of_intersections l C ≤ 2) → 
    (number_of_intersections l C1 + number_of_intersections l C2 + number_of_intersections l C3 + number_of_intersections l C4) ≤ 8 := 
sorry

end max_intersections_of_line_with_four_coplanar_circles_l589_589906


namespace equation_of_circle_through_three_points_l589_589571

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589571


namespace count_5_digit_numbers_l589_589917

-- Define a 5-digit number under specific ordering conditions
def num_5_digit_count (s : Finset ℕ): ℕ := s.choose 2 * (s.erase (s.max' sorry)).choose 2

theorem count_5_digit_numbers : num_5_digit_count (Finset.range 6.erase 0) 1 = 15 := by
  sorry

end count_5_digit_numbers_l589_589917


namespace depth_of_melted_sauce_l589_589137

theorem depth_of_melted_sauce
  (r_sphere : ℝ) (r_cylinder : ℝ) (h_cylinder : ℝ) (volume_conserved : Bool) :
  r_sphere = 3 ∧ r_cylinder = 10 ∧ volume_conserved → h_cylinder = 9/25 :=
by
  -- Explanation of the condition: 
  -- r_sphere is the radius of the original spherical globe (3 inches)
  -- r_cylinder is the radius of the cylindrical puddle (10 inches)
  -- h_cylinder is the depth we need to prove is 9/25 inches
  -- volume_conserved indicates that the volume is conserved
  sorry

end depth_of_melted_sauce_l589_589137


namespace magic_star_exists_l589_589194

/-- 
Given the numbers from 1 to 11, prove that there exists an arrangement of these numbers in circles 
such that the sum of the three numbers on each of the ten segments is the same.
-/
theorem magic_star_exists : 
  ∃ (f : Fin 11 → Fin 11), 
  -- Ensure 'f' is a valid permutation of numbers 1 to 11
  Permutation (Fin.enum 11) (f.ge),
  -- Define S as the required segment sum that we need to prove to be the same across ten segments
  ∃ S : ℕ, 
    -- Condition for the sum of segments
    ∀ n : Fin 10, segment_sum f n = S := 
sorry

end magic_star_exists_l589_589194


namespace circle_passing_through_points_l589_589609

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589609


namespace melinda_probability_l589_589371

def probability_of_valid_numbers_between_21_and_31 : ℚ :=
  let outcomes := [(1,2), (2,1), (2,2), (2,3), (3,2), (3,1)]  -- valid outcomes
  in outcomes.length / 36

theorem melinda_probability : probability_of_valid_numbers_between_21_and_31 = 1 / 9 := 
  by sorry

end melinda_probability_l589_589371


namespace sin_equation_l589_589253

theorem sin_equation (α : ℝ) (h1 : cos (α + π / 6) = 1 / 3) (h2 : 0 < α ∧ α < π / 2) :
  sin (π / 6 - 2 * α) = -7 / 9 :=
sorry

end sin_equation_l589_589253


namespace circle_passing_three_points_l589_589406

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589406


namespace sin_ratio_in_triangle_l589_589321

theorem sin_ratio_in_triangle (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_A : A = 2 * π / 3) 
  (h_c : c = 5)
  (h_a : a = 7)
  (h_cos_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) :
  b = 3 → 
  Real.sin B / Real.sin C = 3 / 5 :=
begin
  sorry
end

end sin_ratio_in_triangle_l589_589321


namespace find_circle_equation_l589_589557

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589557


namespace part1_part2_l589_589956

noncomputable def f (x k: ℝ) : ℝ := Real.log x - k * x + 1

theorem part1 (k : ℝ) :
  (∀ x > 0, f x k ≤ 0) → k ≥ 1 := 
sorry

theorem part2 (n : ℕ) (h : n > 1) :
  (∑ i in Finset.range (n-1).succ, Real.log (i + 2) / (i + 2)^2 - i) +
  (1 + 1/n : ℝ)^n < (n^2 + n + 10) / 4 :=
sorry

end part1_part2_l589_589956


namespace circle_equation_through_points_l589_589650

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589650


namespace gasoline_percentage_increase_l589_589233

-- Definitions for the initial and final prices
def P_initial : ℝ := 29.90
def P_final : ℝ := 149.70

-- Definition for the percentage increase function
def percentage_increase (P_initial P_final : ℝ) : ℝ :=
  ((P_final - P_initial) / P_initial) * 100

-- Now we state the theorem
theorem gasoline_percentage_increase : percentage_increase P_initial P_final = 400 := by
  sorry

end gasoline_percentage_increase_l589_589233


namespace circle_passing_through_points_eqn_l589_589485

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589485


namespace area_of_circle_l589_589031

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def polar_circle_eq (θ : ℝ) : ℝ :=
  3 * Real.cos θ - 4 * Real.sin θ

theorem area_of_circle :
  let r : ℝ := polar_circle_eq
  let x y : ℝ := polar_to_cartesian r θ
  (x - 3 / 2)^2 + (y + 2)^2 = 25 / 4 →
  π * (5 / 2)^2 = 25 / 4 * π :=
sorry

end area_of_circle_l589_589031


namespace smallest_value_of_x_l589_589085

theorem smallest_value_of_x :
  ∃ x, (12 * x^2 - 58 * x + 70 = 0) ∧ x = 7 / 3 :=
by
  sorry

end smallest_value_of_x_l589_589085


namespace tetrahedron_is_regular_regular_tetrahedron_has_five_spheres_l589_589266

-- Define the structure for a tetrahedron
structure Tetrahedron (V : Type*) [RealVectors V] :=
(vertices : list V)
(h_len : vertices.length = 4)

-- Define a regular tetrahedron in Lean
def is_regular_tetrahedron (T : Tetrahedron) : Prop :=
  ∃ r : ℝ, ∀ (i j : ℕ) (hi : i < 4) (hj : j < 4), i ≠ j → 
    dist T.vertices[i] T.vertices[j] = r

-- Define the existence of specific spheres
def five_spheres_exist (T : Tetrahedron) : Prop :=
  ∃ (spheres : list (V × ℝ)), spheres.length = 5 ∧
    ∀ (sphere : V × ℝ), sphere ∈ spheres →
      ∃ (edges : list (V × V)), edges.length = 3 ∧
        ∀ (edge : V × V), edge ∈ edges →
          ∃ t : ℝ, tangent_sphere_edge sphere edge t

variables (T : Tetrahedron)

-- Part (1): Proof that the tetrahedron is a regular tetrahedron
theorem tetrahedron_is_regular (h : five_spheres_exist T) : is_regular_tetrahedron T :=
sorry

-- Part (2): Proof that every regular tetrahedron has five such spheres
theorem regular_tetrahedron_has_five_spheres (h : is_regular_tetrahedron T) : five_spheres_exist T :=
sorry

end tetrahedron_is_regular_regular_tetrahedron_has_five_spheres_l589_589266


namespace equation_of_circle_passing_through_points_l589_589666

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589666


namespace fraction_finding_l589_589071

theorem fraction_finding (x : ℝ) (h : (3 / 4) * x * (2 / 3) = 0.4) : x = 0.8 :=
sorry

end fraction_finding_l589_589071


namespace smallest_enclosing_circle_radius_lemma_l589_589243

theorem smallest_enclosing_circle_radius_lemma (n : ℕ) (h : n ≥ 3) 
(points : Fin n → ℝ × ℝ)
(r : (Fin n → ℝ × ℝ) → ℝ) :
  ∃ i j k : Fin n, r points = r (λ x, if x = 0 then points i 
                                     else if x = 1 then points j 
                                     else points k) :=
sorry

end smallest_enclosing_circle_radius_lemma_l589_589243


namespace colored_n_gon_l589_589333

noncomputable def g (n k : ℕ) : ℕ :=
  if k = 0 then 1
  else g (n-3) (k-1) + g (n-1) k

theorem colored_n_gon (n k : ℕ) (h1 : 0 < k) (h2 : k < n) :
  g n k = Nat.choose (n - k - 1) (k - 1) + Nat.choose (n - k) k := by
  sorry

end colored_n_gon_l589_589333


namespace projection_ratio_l589_589214

open Matrix
open Rat

def P : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![9 / 41, -20 / 41], ![-20 / 41, 32 / 41]]

variable {u v : ℚ}

theorem projection_ratio (h : P ⬝ ![u, v] = ![u, v]) : v / u = -20 / 9 :=
  sorry

end projection_ratio_l589_589214


namespace percentage_freshmen_l589_589199

variables 
  (T : ℕ) -- Total number of students
  (F : ℝ) -- Percentage of freshmen (in decimal form)
  (h1 : 0.4 * F * T = 0.048 * T) -- 40% of freshmen are in liberal arts, and 4.8% of total students are freshmen psychology majors in liberal arts
  (h2 : T ≠ 0) -- To avoid division by zero

theorem percentage_freshmen : F = 0.6 := by
  have h3 : 0.08 * F * T = 0.048 * T,
  { rw mul_assoc at h1,
    exact h1 },
  have h4 : 0.08 * F = 0.048,
  { field_simp [h2] at h3,
    exact h3 },
  linarith

end percentage_freshmen_l589_589199


namespace heartbeats_during_race_l589_589803

theorem heartbeats_during_race
  (heart_rate : ℕ) -- beats per minute
  (pace : ℕ) -- minutes per mile
  (distance : ℕ) -- miles
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) : 
  heart_rate * (pace * distance) = 28800 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end heartbeats_during_race_l589_589803


namespace probability_of_harmonious_sets_l589_589315

def harmonious_set (A : set ℝ) : Prop :=
  ∀ x ∈ A, x ≠ 0 → (1 / x) ∈ A

def M : set ℝ := {-1, 0, 1/3, 1/2, 1, 2, 3, 4}

theorem probability_of_harmonious_sets : 
  (| (set.filter harmonious_set (set.powerset M)) | : ℕ) / (255 : ℕ) = 1/17 := 
sorry

end probability_of_harmonious_sets_l589_589315


namespace max_performances_l589_589710

theorem max_performances (n : ℕ) (students : Finset (Fin 12)) (performances : Finset (Finset (Fin 12))) 
  (H1 : ∀ P ∈ performances, P.card = 6) 
  (H2 : ∀ P1 P2 ∈ performances, P1 ≠ P2 → (P1 ∩ P2).card ≤ 2) :
  ∃ n_max : ℕ, n_max = 4 ∧ ∀ m : ℕ, (m ≤ n_max) :=
by
  sorry

end max_performances_l589_589710


namespace ways_to_distribute_balls_l589_589995

theorem ways_to_distribute_balls :
  let balls := 5
  let boxes := 4
  ( ∑ i in Finset.range (boxes + 1), if i < balls then 4^i * Nat.stirling2 balls i else 0 ) = 1024 :=
by
  let balls := 5
  let boxes := 4
  have h : ( ∑ i in Finset.range (boxes + 1), if i < balls then 4^i * Nat.stirling2 balls i else 0 ) = 1024 := sorry
  exact h

end ways_to_distribute_balls_l589_589995


namespace minimum_value_of_x_plus_2y_l589_589925

-- Definitions for the problem conditions
def isPositive (z : ℝ) : Prop := z > 0

def condition (x y : ℝ) : Prop := 
  isPositive x ∧ isPositive y ∧ (x + 2*y + 2*x*y = 8) 

-- Statement of the problem
theorem minimum_value_of_x_plus_2y (x y : ℝ) (h : condition x y) : x + 2 * y ≥ 4 :=
sorry

end minimum_value_of_x_plus_2y_l589_589925


namespace trapezoid_area_l589_589846

structure Point where
  x : ℝ
  y : ℝ

structure Trapezoid where
  E : Point
  F : Point
  G : Point
  H : Point

def distance (A B : Point) : ℝ :=
  real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

noncomputable def base_length_EF (T : Trapezoid) : ℝ :=
  distance T.E T.F

noncomputable def base_length_GH (T : Trapezoid) : ℝ :=
  distance T.G T.H

noncomputable def height (T : Trapezoid) : ℝ :=
  abs (T.E.x - T.G.x)

noncomputable def area (T : Trapezoid) : ℝ :=
  0.5 * (base_length_EF T + base_length_GH T) * height T

theorem trapezoid_area {T : Trapezoid} (hT : T = { E := ⟨0, 0⟩, F := ⟨0, -3⟩, G := ⟨5, 0⟩, H := ⟨5, 8⟩ }) :
  area T = 27.5 :=
by
  subst hT
  sorry

end trapezoid_area_l589_589846


namespace vec_expr_eq_l589_589975

variable (a b : ℝ × ℝ)
variable ha : a = (-3, 1)
variable hb : b = (-1, 2)

theorem vec_expr_eq : 3 • a - 2 • b = (-7, -1) := by
  sorry

end vec_expr_eq_l589_589975


namespace max_n_m_sum_l589_589703

-- Definition of the function f
def f (x : ℝ) : ℝ := -x^2 + 4 * x

-- Statement of the problem
theorem max_n_m_sum {m n : ℝ} (h : n > m) (h_range : ∀ x, m ≤ x ∧ x ≤ n → -5 ≤ f x ∧ f x ≤ 4) : n + m = 7 :=
sorry

end max_n_m_sum_l589_589703


namespace circle_through_points_l589_589458

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589458


namespace part1_solution_part2_solution_l589_589354

section Part1
variable (a x : ℝ)

-- Condition definitions for part (1)
def p_condition (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q_condition (x : ℝ) : Prop := x^2 + 6 * x + 8 ≤ 0

-- Assertion for part (1)
theorem part1_solution (a : ℝ) (h : a = -3) : 
  (p_condition a x ∧ q_condition x) → -4 ≤ x ∧ x < -3 := 
sorry
end Part1

section Part2
variable (a x : ℝ)

-- Condition definitions for part (2)
def p_condition (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q_condition (x : ℝ) : Prop := x^2 + 6 * x + 8 ≤ 0

-- Assertion for part (2)
theorem part2_solution : 
  (∀ x, q_condition x → p_condition a x) → -2 < a ∧ a < -4 / 3 := 
sorry
end Part2

end part1_solution_part2_solution_l589_589354


namespace triangle_area_l589_589168

noncomputable def line_eq := λ x : ℝ, 9 - 3 * x

theorem triangle_area :
    let x_intercept := (3 : ℝ)
    let y_intercept := (9 : ℝ)
    let triangle_base := x_intercept
    let triangle_height := y_intercept
    let area := (1 / 2) * triangle_base * triangle_height
    area = 13.5 :=
by
    sorry

end triangle_area_l589_589168


namespace lcd_of_rational_expressions_l589_589236

theorem lcd_of_rational_expressions (a : ℤ) :
  let d1 := a^2 - 2 * a + 1 in
  let d2 := a^2 - 1 in
  let d3 := a^2 + 2 * a + 1 in
  ∃ lcd : ℤ, lcd = (a - 1)^2 * (a + 1)^2 ∧ lcd = a^4 - 2 * a^2 + 1 :=
by
  let d1 := a^2 - 2 * a + 1
  let d2 := a^2 - 1
  let d3 := a^2 + 2 * a + 1
  let lcd := (a - 1)^2 * (a + 1)^2
  use lcd
  split
  . refl
  . refl

end lcd_of_rational_expressions_l589_589236


namespace determine_constants_l589_589904

theorem determine_constants :
  ∃ (a b c p : ℝ), (a = -1) ∧ (b = -1) ∧ (c = -1) ∧ (p = 3) ∧
  (∀ x : ℝ, x^3 + p*x^2 + 3*x - 10 = 0 ↔ (x = a ∨ x = b ∨ x = c)) ∧ 
  c - b = b - a ∧ c - b > 0 :=
by
  sorry

end determine_constants_l589_589904


namespace greatest_number_of_bouquets_l589_589006

/--
Sara has 42 red flowers, 63 yellow flowers, and 54 blue flowers.
She wants to make bouquets with the same number of each color flower in each bouquet.
Prove that the greatest number of bouquets she can make is 21.
-/
theorem greatest_number_of_bouquets (red yellow blue : ℕ) (h_red : red = 42) (h_yellow : yellow = 63) (h_blue : blue = 54) :
  Nat.gcd (Nat.gcd red yellow) blue = 21 :=
by
  rw [h_red, h_yellow, h_blue]
  sorry

end greatest_number_of_bouquets_l589_589006


namespace minimum_value_l589_589937

theorem minimum_value (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 1) :
  (2 / (x + 3 * y) + 1 / (x - y)) = (3 + 2 * Real.sqrt 2) / 2 := sorry

end minimum_value_l589_589937


namespace equation_of_circle_through_three_points_l589_589575

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589575


namespace smoking_related_to_lung_disease_l589_589760

noncomputable def K2 : ℝ := 5.231

axiom P_ge_3841 : P (X^2 ≥ 3.841) = 0.05
axiom P_ge_6635 : P (X^2 ≥ 6.635) = 0.01

theorem smoking_related_to_lung_disease : K2 >= 3.841 ∧ K2 < 6.635 → 
  ∃ (conf_level : ℝ), conf_level > 0.95 ∧ smoking_related_to_lung := sorry

end smoking_related_to_lung_disease_l589_589760


namespace circle_passing_through_points_l589_589507

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589507


namespace equation_of_circle_ABC_l589_589439

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589439


namespace hyperbola_eccentricity_range_l589_589311

theorem hyperbola_eccentricity_range (a : ℝ) (h : a > 2) :
  let e := sqrt(2 + 2 / a + 1 / a^2)
  in (e > sqrt(2)) ∧ (e < sqrt(13) / 2) :=
by
  let e := sqrt(2 + 2 / a + 1 / a^2)
  sorry

end hyperbola_eccentricity_range_l589_589311


namespace escalator_rate_l589_589836

theorem escalator_rate (length escalator_length feet : ℝ)
  (person_walk_rate feet_per_sec : ℝ)
  (time_to_cover length time_sec : ℝ)
  (effective_rate : length = (escalator_length + person_walk_rate) * time_sec) :
  escalator_length = 12 :=
by
  have h1 : 196 = (escalator_length + 2) * 14 := sorry,
  have h2 : escalator_length = 12 := sorry,
  exact h2

end escalator_rate_l589_589836


namespace equation_solution_unique_l589_589890

noncomputable def verifyEquation (x : ℝ) : Prop :=
  sqrt x + 2 * sqrt (x^2 + 9 * x) + sqrt (x + 9) = 45 - 2 * x

theorem equation_solution_unique :
  ∀ x : ℝ, verifyEquation x ↔ x = 729 / 144 :=
by
  intros x
  unfold verifyEquation
  sorry

end equation_solution_unique_l589_589890


namespace arithmetic_sequence_sum_l589_589932

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1))
  (h_condition : a 3 + a 13 - a 8 = 2) : ∑ k in finset.range 15, a k = 30 :=
by sorry

end arithmetic_sequence_sum_l589_589932


namespace area_of_triangle_l589_589287

noncomputable def f (x : ℝ) : ℝ := e^(x-1) - log x + 1

theorem area_of_triangle (x : ℝ) (e : ℝ) (h0 : e = Real.exp 1) (h1 : x = 1) :
  let y := f x;
    let slope := e^(x-1) - (1/x);
    let tangent_line (x: ℝ) := slope * (x - 1) + y;
    let x_intercept := - tangent_line(0) / slope;
    let y_intercept := tangent_line 0;
    let area := (1/2) * (abs x_intercept) * (abs y_intercept)
  in area = 2 / (e-1) :=
sorry

end area_of_triangle_l589_589287


namespace circle_equation_through_points_l589_589640

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589640


namespace max_n_for_neg_sum_correct_l589_589267

noncomputable def max_n_for_neg_sum (S : ℕ → ℤ) : ℕ :=
  if h₁ : S 19 > 0 then
    if h₂ : S 20 < 0 then
      11
    else 0  -- default value
  else 0  -- default value

theorem max_n_for_neg_sum_correct (S : ℕ → ℤ) (h₁ : S 19 > 0) (h₂ : S 20 < 0) : max_n_for_neg_sum S = 11 :=
by
  sorry

end max_n_for_neg_sum_correct_l589_589267


namespace graveling_cost_l589_589099

theorem graveling_cost
  (length_lawn : ℝ) (width_lawn : ℝ)
  (width_road : ℝ)
  (cost_per_sq_m : ℝ)
  (h1: length_lawn = 80) (h2: width_lawn = 40) (h3: width_road = 10) (h4: cost_per_sq_m = 3) :
  (length_lawn * width_road + width_lawn * width_road - width_road * width_road) * cost_per_sq_m = 3900 := 
by
  sorry

end graveling_cost_l589_589099


namespace ratio_of_cost_to_marked_price_l589_589128

theorem ratio_of_cost_to_marked_price (x : ℝ) (hx : x ≠ 0) :
  let selling_price := (3 / 4) * x in
  let cost_price := (2 / 3) * selling_price in
  cost_price / x = 1 / 2 :=
by
  let selling_price := (3 / 4) * x
  let cost_price := (2 / 3) * selling_price
  show cost_price / x = 1 / 2
  sorry

end ratio_of_cost_to_marked_price_l589_589128


namespace tautology_a_tautology_b_tautology_c_tautology_d_l589_589193

variable (p q : Prop)

theorem tautology_a : p ∨ ¬ p := by
  sorry

theorem tautology_b : ¬ ¬ p ↔ p := by
  sorry

theorem tautology_c : ((p → q) → p) → p := by
  sorry

theorem tautology_d : ¬ (p ∧ ¬ p) := by
  sorry

end tautology_a_tautology_b_tautology_c_tautology_d_l589_589193


namespace charlie_more_apples_than_bella_l589_589205

variable (D : ℝ) 

theorem charlie_more_apples_than_bella 
    (hC : C = 1.75 * D)
    (hB : B = 1.50 * D) :
    (C - B) / B = 0.1667 := 
by
  sorry

end charlie_more_apples_than_bella_l589_589205


namespace triangle_area_bound_by_line_l589_589156

noncomputable def area_of_triangle : ℝ :=
let x_intercept := 3 in
let y_intercept := 9 in
1 / 2 * x_intercept * y_intercept

theorem triangle_area_bound_by_line (x y : ℝ) (h : 3 * x + y = 9) :
  area_of_triangle = 13.5 :=
sorry

end triangle_area_bound_by_line_l589_589156


namespace athlete_heartbeats_calculation_l589_589791

theorem athlete_heartbeats_calculation :
  ∀ (heart_rate : ℕ) (distance : ℕ) (pace : ℕ),
  heart_rate = 160 →
  distance = 30 →
  pace = 6 →
  (pace * distance * heart_rate = 28800)
:= by
  intros heart_rate distance pace hr_eq dis_eq pace_eq
  rw [hr_eq, dis_eq, pace_eq]
  norm_num
  sorry

end athlete_heartbeats_calculation_l589_589791


namespace maximize_product_of_two_five_digit_numbers_l589_589715

theorem maximize_product_of_two_five_digit_numbers :
  ∃ (x y : ℕ), x * y = 86420 ∧
  (∀ (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.to_finset), digit_included x d ∨ digit_included y d) ∧
  length_of_number x = 5 ∧
  length_of_number y = 5 :=
sorry

-- Helper definitions for the theorem statement
def digit_included (n d : ℕ) : Prop :=
  d ∈ n.digits 10

def length_of_number (n : ℕ) : ℕ :=
  n.digits 10).length

end maximize_product_of_two_five_digit_numbers_l589_589715


namespace trajectory_point_M_l589_589261

theorem trajectory_point_M (x y : ℝ) : 
  (∃ (m n : ℝ), x^2 + y^2 = 9 ∧ (m = x) ∧ (n = 3 * y)) → 
  (x^2 / 9 + y^2 = 1) :=
by
  sorry

end trajectory_point_M_l589_589261


namespace find_white_bread_loaves_l589_589848

def cost_of_white_bread_per_loaf := 3.50
def cost_of_baguette := 1.50
def cost_of_sourdough_per_loaf := 4.50
def cost_of_almond_croissant := 2.00
def total_spent_over_4_weeks := 78.00
def weeks := 4

def total_spent_per_week := total_spent_over_4_weeks / weeks
def cost_of_other_items_per_week := cost_of_baguette + 2 * cost_of_sourdough_per_loaf + cost_of_almond_croissant

theorem find_white_bread_loaves (W : ℝ) :
  cost_of_other_items_per_week + W * cost_of_white_bread_per_loaf = total_spent_per_week →
  W = 2 := by
  sorry

end find_white_bread_loaves_l589_589848


namespace total_heartbeats_during_race_l589_589793

-- Definitions for the conditions
def heart_beats_per_minute : ℕ := 160
def pace_in_minutes_per_mile : ℕ := 6
def total_distance_in_miles : ℕ := 30

-- Main theorem statement
theorem total_heartbeats_during_race : 
  heart_beats_per_minute * pace_in_minutes_per_mile * total_distance_in_miles = 28800 :=
by
  -- Place the proof here
  sorry

end total_heartbeats_during_race_l589_589793


namespace circle_passing_through_points_eqn_l589_589483

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589483


namespace amyl_alcohol_required_l589_589985

def R : Type := ℝ

noncomputable def reaction : R → R → R := λ ch3cch2OH HCl, (ch3cch2OH, HCl, (ch3cch2OH = HCl ∧ (ch3cch2OH = (3 : R))))

theorem amyl_alcohol_required (ch3cch2OH HCl: R) (h1: ch3cch2OH = 3) (h2: HCl = 3) :
  reaction ch3cch2OH HCl = (3, 3, (ch3cch2OH = HCl ∧ ch3cch2OH = 3)) :=
by
  sorry

end amyl_alcohol_required_l589_589985


namespace sufficient_but_not_necessary_condition_l589_589271

noncomputable def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → (a - 1) * (a ^ x) < (a - 1) * (a ^ y) → a > 1) ∧
  (¬ (∀ c : ℝ, is_increasing_function (λ x => (c - 1) * (c ^ x)) → c > 1)) :=
sorry

end sufficient_but_not_necessary_condition_l589_589271


namespace eccentricity_of_ellipse_l589_589953

theorem eccentricity_of_ellipse 
  (a b : ℝ) (hp : a > b) (hb : b > 0)
  (m n : ℝ) (c : ℝ) 
  (ellipse_eq : ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1) → 
                      (P ≠ (0,0))) 
  (A : (-a, 0)) 
  (P : (m, n))
  (Q : (-m, -n))
  (F : (-c, 0))
  (H : (- (a + m) / 2, - (n / 2))) 
  (collinear : (n / (m + c) = (n / 2) / (-c + (a + m) / 2))) :
  |c / a| = 1 / 3 := 
sorry

end eccentricity_of_ellipse_l589_589953


namespace intersection_of_A_and_B_l589_589297

open Set

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | x < 0}

theorem intersection_of_A_and_B : A ∩ B = {-1} :=
  sorry

end intersection_of_A_and_B_l589_589297


namespace count_positive_integers_less_than_500k_powers_of_2_not_divisible_by_5_l589_589993

theorem count_positive_integers_less_than_500k_powers_of_2_not_divisible_by_5 : 
  ∃ n : ℕ, n = 19 ∧ (∀ k : ℕ, k ≤ 18 → ¬ (2^k % 5 = 0) ∧ 2^k < 500000) :=
begin
  sorry
end

end count_positive_integers_less_than_500k_powers_of_2_not_divisible_by_5_l589_589993


namespace area_of_cos_l589_589395

-- Define the function cos
def f (x : ℝ) := Real.cos x

-- Define the interval bounds
def a : ℝ := 0
def b : ℝ := (3 * Real.pi) / 2

-- Define the definite integral of f from a to b
def area := ∫ x in a..b, f x

-- Prove the area is 3
theorem area_of_cos : area = 3 :=
by
  sorry

end area_of_cos_l589_589395


namespace inequality_proof_l589_589924

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + a^2) / (1 + a * b) + (1 + b^2) / (1 + b * c) + (1 + c^2) / (1 + c * a) ≥ 3 :=
begin
  sorry
end

end inequality_proof_l589_589924


namespace repeated_root_value_of_m_l589_589387

theorem repeated_root_value_of_m (m : ℝ) :
  (∃ x : ℝ, (x - 6) / (x - 5) + 1 = m / (x - 5) ∧ (deriv (λ x : ℝ, (x - 6) / (x - 5) + 1 - m / (x - 5)) x = 0)) → m = -1 := 
by 
  sorry

end repeated_root_value_of_m_l589_589387


namespace minimumPerimeterTriangleDEF_l589_589106

-- Definitions for the given problem conditions
structure RegularTriangularPyramid where
  A A1 B1 C1 B C : Point
  AB2 : dist A B = 2
  A1A2root3 : dist A1 A = 2 * Real.sqrt 3
  midD : D = midpoint A B
  midF : F = midpoint A A1
  perpA1A_AB : ∀ {x}, x ∈ line A B → ∀ {y}, y ∈ line A1 A → dot_product x y = 0
  perpA1A_AC : ∀ {x}, x ∈ line A C → ∀ {y}, y ∈ line A1 A → dot_product x y = 0

noncomputable def minimumPerimeter (E : Point) (pyramid : RegularTriangularPyramid) : ℝ :=
  dist pyramid.D E + dist E pyramid.F + dist pyramid.F pyramid.D

theorem minimumPerimeterTriangleDEF (pyramid : RegularTriangularPyramid) (E : Point) :
  ∃ E : Point, minimumPerimeter E pyramid = Real.sqrt 7 + 2 := by
  sorry

end minimumPerimeterTriangleDEF_l589_589106


namespace find_circle_equation_l589_589555

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589555


namespace circle_equation_through_points_l589_589641

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589641


namespace possible_values_P_i_l589_589781

theorem possible_values_P_i (P : ℤ → ℂ) (h1 : ∀ n : ℤ, P n ∈ ℤ) (i : ℂ) (hi : i^2 = -1) :
  ∃ a b : ℚ, (P i = a + b * i ∧
  (∀ (p : ℕ), nat.prime p → (p % 4 = 1) → ¬(p ∣ a.denom) ∧ ¬(p ∣ b.denom))) :=
sorry

end possible_values_P_i_l589_589781


namespace value_of_a_ab_b_l589_589677

-- Define conditions
variables {a b : ℝ} (h1 : a * b = 1) (h2 : b = a + 2)

-- The proof problem
theorem value_of_a_ab_b : a - a * b - b = -3 :=
by
  sorry

end value_of_a_ab_b_l589_589677


namespace regression_analysis_and_independence_test_l589_589785

noncomputable def isIncorrectA : Prop :=
  ¬(there is no difference between regression analysis and independence test)

noncomputable def isIncorrectB : Prop :=
  ¬(Regression analysis is the analysis of the precise relationship between two variables,
    while independence test analyzes the uncertain relationship between two variables)

noncomputable def isCorrectC : Prop :=
  Regression analysis studies the correlation between two variables, and independence test is a test on whether two variables have some kind of relationship

noncomputable def isIncorrectD : Prop :=
  ¬(Independence test can determine with 100% certainty whether there is some kind of relationship between two variables)

theorem regression_analysis_and_independence_test :
  isIncorrectA ∧ isIncorrectB ∧ isCorrectC ∧ isIncorrectD → isCorrectC :=
by
  intros
  exact sorry

end regression_analysis_and_independence_test_l589_589785


namespace exists_set_S_l589_589382

open Set

theorem exists_set_S (n : ℕ) (h_n : n ≥ 4) :
  ∃ S : Set ℕ, 
    (∀ x ∈ S, x < 2^(n-1)) ∧ 
    (S.card = n) ∧ 
    (∀ A B : Finset ℕ, A ⊆ S → B ⊆ S → A ≠ B → A.nonempty → B.nonempty → 
      A.sum id ≠ B.sum id) :=
sorry

end exists_set_S_l589_589382


namespace households_with_car_correct_l589_589325

noncomputable def households_with_car : ℕ := 44

theorem households_with_car_correct :
  ∃ C B : ℕ, C + B - 14 = 79 ∧ B = 35 + 14 ∧ C = 44 :=
by
  let C := 44
  let B := 49
  use C, B
  constructor
  { 
    show C + B - 14 = 79
    { rw B, exact 44 + 49 - 14 = 79 }
  }
  constructor
  { 
    show B = 35 + 14
    { exact rfl }
  }
  { 
    show C = 44
    { exact rfl }
  }
  sorry

end households_with_car_correct_l589_589325


namespace largest_four_digit_sum_19_l589_589081

theorem largest_four_digit_sum_19 : ∃ n : ℕ, n < 10000 ∧ 1000 ≤ n ∧ (∑ i in (n.digits 10), id i = 19) ∧
  (∀ m : ℕ, m < 10000 ∧ 1000 ≤ m ∧ (∑ i in (m.digits 10), id i = 19) → n ≥ m) ∧ n = 9910 :=
by 
  sorry

end largest_four_digit_sum_19_l589_589081


namespace circle_passes_through_points_l589_589527

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589527


namespace total_distance_l589_589754

variable (v1 v2 : ℝ) (t1 t2 : ℝ)

def D1 := v1 * t1
def D2 := v2 * t2
def D := D1 + D2

theorem total_distance :
  D v1 v2 t1 t2 = (v1 * t1) + (v2 * t2) :=
by
  unfold D D1 D2
  sorry

end total_distance_l589_589754


namespace new_volume_correct_l589_589048

-- Define the conditions.
def initial_radius (r : ℝ) : Prop := true
def initial_height (h : ℝ) : Prop := true
def initial_volume (V : ℝ) : Prop := V = 15
def new_radius (r : ℝ) : ℝ := 3 * r
def new_height (h : ℝ) : ℝ := h / 2

-- Define the volume of a cylinder.
def volume_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- State the proof problem.
theorem new_volume_correct (r h : ℝ) (V : ℝ)
  (h1 : initial_radius r)
  (h2 : initial_height h)
  (h3 : initial_volume V) :
  volume_cylinder (new_radius r) (new_height h) = 67.5 :=
by
  sorry

end new_volume_correct_l589_589048


namespace eric_green_marbles_l589_589882

theorem eric_green_marbles (total_marbles white_marbles blue_marbles : ℕ) (h_total : total_marbles = 20)
  (h_white : white_marbles = 12) (h_blue : blue_marbles = 6) :
  total_marbles - (white_marbles + blue_marbles) = 2 := 
by
  sorry

end eric_green_marbles_l589_589882


namespace problem_inequality_l589_589922

theorem problem_inequality 
  {n : ℕ} (t : Fin n → ℝ) 
  (h_sorted : ∀ i j, i ≤ j → t i ≤ t j)
  (h_bounded : ∀ i, 0 < t i ∧ t i < 1) : 
  (1 - t n.succ.pred) ^ 2 * (Finset.sum (Finset.range n.succ) (λ k, t (Fin.ofNat k) ^ k * (1 - t (Fin.ofNat k) ^ (k + 1)) ^ (-2))) < 1 :=
by
  sorry

end problem_inequality_l589_589922


namespace find_c_value_l589_589367

noncomputable def c_value (ξ : ℝ → ℝ) (P : ℝ → ℝ) : ℝ :=
  let μ := 2
  let σ := 3 -- since variance is 9, σ = 3
  classical.some (classical.some_spec (classical.some_spec (
    sorry -- This needs the axioms and properties related to normal distribution
  )))

theorem find_c_value :
  ∀ (ξ : ℝ → ℝ) (P : ℝ → ℝ),
  (∀ x, ξ x = (1 / (σ * sqrt (2 * π))) * exp (- ((x - μ) ^ 2) / (2 * σ ^ 2))) →
  (P (c + 3) = P (c - 1)) →
  c = 1 :=
by
  sorry -- Placeholder for the actual proof

end find_c_value_l589_589367


namespace symmetric_points_l589_589935

variable (a b : ℝ)

def condition_1 := a - 1 = 2
def condition_2 := 5 = -(b - 1)

theorem symmetric_points (h1 : condition_1 a) (h2 : condition_2 b) :
  (a + b) ^ 2023 = -1 := 
by
  sorry

end symmetric_points_l589_589935


namespace largest_four_digit_sum_19_l589_589076

theorem largest_four_digit_sum_19 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (nat.digits 10 n).sum = 19 ∧ 
           ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ (nat.digits 10 m).sum = 19 → m ≤ 8920 :=
begin
  sorry
end

end largest_four_digit_sum_19_l589_589076


namespace circle_equation_through_points_l589_589637

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589637


namespace average_study_diff_l589_589331

theorem average_study_diff (diff : List ℤ) (h_diff : diff = [15, -5, 25, -10, 5, 20, -15]) :
  (List.sum diff) / (List.length diff) = 5 := by
  sorry

end average_study_diff_l589_589331


namespace six_digit_number_prime_factors_l589_589001

theorem six_digit_number_prime_factors (a b : ℕ) (h_a : 0 ≤ a ∧ a ≤ 9) (h_b : 0 ≤ b ∧ b ≤ 9) :
  ∀ p, prime p → p ∣ (101010 * a + 10101 * b) → p < 100 :=
by sorry

end six_digit_number_prime_factors_l589_589001


namespace athlete_heartbeats_during_race_l589_589832

theorem athlete_heartbeats_during_race :
  ∀ (heartbeats_per_minute pace_minutes_per_mile race_miles : ℕ ),
    heartbeats_per_minute = 160 →
    pace_minutes_per_mile = 6 →
    race_miles = 30 →
    heartbeats_per_minute * pace_minutes_per_mile * race_miles = 28800 :=
by
  intros heartbeats_per_minute pace_minutes_per_mile race_miles
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end athlete_heartbeats_during_race_l589_589832


namespace inequality_solution_set_l589_589055

theorem inequality_solution_set :
  { x : ℝ | -3 < x ∧ x < 2 } = { x : ℝ | abs (x - 1) + abs (x + 2) < 5 } :=
by
  sorry

end inequality_solution_set_l589_589055


namespace circle_equation_l589_589626

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589626


namespace triangle_area_l589_589143

theorem triangle_area (x y : ℝ) (h : 3 * x + y = 9) : 
  (1 / 2) * 3 * 9 = 13.5 :=
sorry

end triangle_area_l589_589143


namespace percentage_increase_in_yield_after_every_harvest_is_20_l589_589327

theorem percentage_increase_in_yield_after_every_harvest_is_20
  (P : ℝ)
  (h1 : ∀ n : ℕ, n = 1 → 20 * n = 20)
  (h2 : 20 + 20 * (1 + P / 100) = 44) :
  P = 20 := 
sorry

end percentage_increase_in_yield_after_every_harvest_is_20_l589_589327


namespace circle_equation_correct_l589_589540

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589540


namespace circle_passing_three_points_l589_589402

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589402


namespace work_efficiency_ratio_l589_589098

variables (A_eff B_eff : ℚ) (a b : Type)

theorem work_efficiency_ratio (h1 : B_eff = 1 / 33)
  (h2 : A_eff + B_eff = 1 / 11) :
  A_eff / B_eff = 2 :=
by 
  sorry

end work_efficiency_ratio_l589_589098


namespace intersection_A_B_l589_589298

noncomputable def A : Set ℝ := { y | ∃ x, y = -x^2 - 2x }

noncomputable def B : Set ℝ := { y | ∃ x, y = x + 1 }

theorem intersection_A_B :
  (A ∩ B) = Set.Iic 1 := sorry

end intersection_A_B_l589_589298


namespace cutting_time_is_2_minutes_l589_589979

variables (total_strands : ℕ) (hannah_rate : ℕ) (son_rate : ℕ)

def combined_rate (hannah_rate : ℕ) (son_rate : ℕ) : ℕ :=
  hannah_rate + son_rate

def time_to_cut_all_strands (total_strands : ℕ) (combined_rate : ℕ) : ℕ :=
  total_strands / combined_rate

theorem cutting_time_is_2_minutes (htotal_strands : total_strands = 22) 
                                   (hhannah_rate : hannah_rate = 8) 
                                   (hson_rate : son_rate = 3) : 
  time_to_cut_all_strands total_strands (combined_rate hannah_rate son_rate) = 2 :=
by
  rw [htotal_strands, hhannah_rate, hson_rate]
  show time_to_cut_all_strands 22 (combined_rate 8 3) = 2
  rw combined_rate -- resolves to 11
  show time_to_cut_all_strands 22 11 = 2
  rw time_to_cut_all_strands -- resolves to 22 / 11
  exact rfl

end cutting_time_is_2_minutes_l589_589979


namespace smoking_related_to_lung_disease_l589_589115

theorem smoking_related_to_lung_disease
  (N : ℕ)
  (K_squared : ℝ)
  (P_3_841 : ℝ)
  (P_6_635 : ℝ)
  (hN : N = 11000)
  (hK_squared : K_squared = 5.231)
  (hP_3_841 : P_3_841 = 0.05)
  (hP_6_635 : P_6_635 = 0.01)
  : P_3_841 ≥ 0.05 ∧ P_6_635 ≤ 0.01 → (K_squared ≥ 3.841 ∧ K_squared < 6.635) → "smoking is related to lung disease" :=
sorry

end smoking_related_to_lung_disease_l589_589115


namespace range_of_m_l589_589919

-- Definitions for conditions p and q
def condition_p (m : ℝ) : Prop :=
  let Δ := 8 - 4 * m in Δ > 0

def condition_q (m : ℝ) : Prop :=
  let Δ := 16 * (m - 2) ^ 2 - 16 in Δ < 0

-- Main proof goal
theorem range_of_m (m : ℝ) : 
  (condition_p m ∨ condition_q m) ∧ ¬ (condition_p m ∧ condition_q m) ↔ (m ≤ 1 ∨ (2 ≤ m ∧ m < 3)) := 
sorry

end range_of_m_l589_589919


namespace athlete_heartbeats_during_race_l589_589821

theorem athlete_heartbeats_during_race
  (heart_rate : ℕ)
  (pace : ℕ)
  (distance : ℕ)
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) :
  heart_rate * pace * distance = 28800 :=
by
  rw [H1, H2, H3]
  norm_num

end athlete_heartbeats_during_race_l589_589821


namespace circle_passes_through_points_l589_589521

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589521


namespace complex_expression_result_l589_589844

theorem complex_expression_result : (1 + complex.i) ^ 3 / (1 - complex.i) ^ 2 = -1 - complex.i :=
by sorry

end complex_expression_result_l589_589844


namespace equation_of_circle_through_three_points_l589_589572

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589572


namespace parabola_focus_distance_l589_589930

open Real

noncomputable def parabola (P : ℝ × ℝ) : Prop := (P.2)^2 = 4 * P.1
def line_eq (P : ℝ × ℝ) : Prop := abs (P.1 + 2) = 6

theorem parabola_focus_distance (P : ℝ × ℝ) 
  (hp : parabola P) 
  (hl : line_eq P) : 
  dist P (1 / 4, 0) = 5 :=
sorry

end parabola_focus_distance_l589_589930


namespace factorization_x12_minus_729_l589_589211

theorem factorization_x12_minus_729 (x : ℝ) : 
  x^12 - 729 = (x^2 + 3) * (x^4 - 3 * x^2 + 9) * (x^3 - 3) * (x^3 + 3) :=
by sorry

end factorization_x12_minus_729_l589_589211


namespace sin_pow_sin_lt_cos_pow_cos_l589_589381

theorem sin_pow_sin_lt_cos_pow_cos (x : ℝ) (hx : 0 < x ∧ x < π / 4) : 
  (sin x)^sin x < (cos x)^cos x := 
by sorry

end sin_pow_sin_lt_cos_pow_cos_l589_589381


namespace hare_probability_l589_589231

-- Definitions for events
def P (x : Type) [ProbabilityMeasure x] := measure x
def event_A : Prop := -- The creature is a rabbit
def event_B : Prop := -- The creature declared it is not a rabbit
def event_C : Prop := -- The creature declared it is not a hare

-- Given conditions
variable [ProbabilityMeasure (event_A : ℙ)] (h1 : P[event_A] = 1/2)
variable (h2 : P[event_B | event_A] = 3/4) -- Probability declared not rabbit given is rabbit (75% correct)
variable (h3 : P[event_C | event_A] = 1/4) -- Probability declared not hare given is rabbit (25% wrong)
variable (h4 : P[¬event_A | event_B] = 2/3) -- Probability declared not rabbit given is hare (67% wrong)
variable (h5 : P[¬event_A | event_C] = 1/3) -- Probability declared not hare given is hare (33% wrong)

-- Goal: Prove the conditional probability
theorem hare_probability :
  P[event_A | event_B ∧ event_C] = 27 / 59 := by
  sorry

end hare_probability_l589_589231


namespace equation_of_circle_through_three_points_l589_589568

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589568


namespace solution_set_l589_589946

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

-- Conditions
axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom symmetric_f : ∀ x : ℝ, f (2 - x) = f x
axiom derivative_f : ∀ x : ℝ, deriv f x = f' x
axiom inequality_condition : ∀ x : ℝ, x < 1 → (x-1) * (f x + (x-1) * f' x) > 0

-- The theorem
theorem solution_set : { x : ℝ | x * f (x + 1) > f 2 } = {x : ℝ | x < -1 ∨ x > 1} :=
sorry

end solution_set_l589_589946


namespace circle_passing_through_points_l589_589600

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589600


namespace problem_part1_problem_part2_l589_589949
noncomputable theory

-- Definitions based on given conditions
def S (a : ℕ → ℕ) (n : ℕ) := ∑ i in finset.range n, a i
axiom a1 : ℕ → ℕ
axiom a1_base : a1 0 = 1
axiom a1_recursion : ∀ n : ℕ, S a1 n = (1/3 : ℚ) * a1 (n + 1)

-- Formal statement of the required proof
theorem problem_part1 : ∀ n : ℕ, a1 n = 
  if n = 0 then 1 else 3 * 4^(n - 1) :=
sorry

def b (n : ℕ) : ℕ := a1 n + (Int.log 4 (S a1 n)).toNat
def T (n : ℕ) := ∑ i in finset.range n, b i

-- Sum of first n terms of b_n
theorem problem_part2 : ∀ n : ℕ, T n = 4^(n-1) + (n^2 - n) / 2 :=
sorry

end problem_part1_problem_part2_l589_589949


namespace sum_y_coeffs_eq_38_l589_589319

noncomputable def sum_of_y_coefficients (p q : Polynomial ℤ) : ℤ :=
  (p * q).support.sum (λ n, if n ≥ 1 then (p * q).coeff n else 0)

theorem sum_y_coeffs_eq_38 :
  let p := Polynomial.C 2 + Polynomial.C 3 * Polynomial.Y + Polynomial.X in
  let q := Polynomial.C 3 + Polynomial.C 2 * Polynomial.Y + Polynomial.X in
  sum_of_y_coefficients p q = 38 :=
by
  -- Setup polynomial p and q
  let p := Polynomial.C 2 + Polynomial.C 3 * Polynomial.Y + Polynomial.X
  let q := Polynomial.C 3 + Polynomial.C 2 * Polynomial.Y + Polynomial.X
  -- Sorry to skip the proof for now
  sorry

end sum_y_coeffs_eq_38_l589_589319


namespace cylinder_new_volume_proof_l589_589050

noncomputable def cylinder_new_volume (V : ℝ) (r h : ℝ) : ℝ := 
  let new_r := 3 * r
  let new_h := h / 2
  π * (new_r^2) * new_h

theorem cylinder_new_volume_proof (r h : ℝ) (π : ℝ) 
  (h_volume : π * r^2 * h = 15) : 
  cylinder_new_volume (π * r^2 * h) r h = 67.5 := 
by 
  unfold cylinder_new_volume
  rw [←h_volume]
  sorry

end cylinder_new_volume_proof_l589_589050


namespace bridge_length_is_correct_l589_589141

-- Definitions based on conditions
def train_length : ℝ := 100  -- Length of the train in meters
def crossing_time : ℝ := 15  -- Time to cross the bridge in seconds
def train_speed_kmh : ℝ := 96 -- Speed of the train in km/h

-- Conversion function from km/h to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

-- Prove that the length of the bridge is 300.05 meters given the conditions
theorem bridge_length_is_correct :
  let train_speed_ms := kmh_to_ms train_speed_kmh in
  let total_distance := train_speed_ms * crossing_time in
  let bridge_length := total_distance - train_length in
  bridge_length = 300.05 :=
by
  -- Define the intermediate values
  let train_speed_ms := kmh_to_ms train_speed_kmh
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  -- Prove the final statement
  sorry

end bridge_length_is_correct_l589_589141


namespace dot_product_is_4_l589_589974

-- Define vectors a and b
def a (x : ℝ) : ℝ × ℝ := (2, x)
def b : ℝ × ℝ := (1, -1)

-- Define the condition that a is parallel to (a + b)
def is_parallel (u v : ℝ × ℝ) : Prop := 
  (u.1 * v.2 - u.2 * v.1) = 0

theorem dot_product_is_4 (x : ℝ) (h_parallel : is_parallel (a x) (a x + b)) : 
  (a x).1 * b.1 + (a x).2 * b.2 = 4 :=
sorry

end dot_product_is_4_l589_589974


namespace function_behavior_l589_589030

noncomputable def f (x : ℝ) : ℝ := abs (2^x - 2)

theorem function_behavior :
  (∀ x y : ℝ, x < y ∧ y ≤ 1 → f x ≥ f y) ∧ (∀ x y : ℝ, x < y ∧ x ≥ 1 → f x ≤ f y) :=
by
  sorry

end function_behavior_l589_589030


namespace highest_selling_price_day_weekly_profit_promotional_method_cost_cost_effectiveness_l589_589755

theorem highest_selling_price_day 
  (prices : List Int) 
  (days : List String) :
  prices = [1, -2, 3, -1, 2, 5, -4] →
  days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] →
  List.nth prices 5 = 5 :=
by
  sorry

theorem weekly_profit 
  (prices_relative : List Int) 
  (quantities_sold : List Int) 
  (cost_price : Int) 
  (standard_selling_price : Int) :
  prices_relative = [1, -2, 3, -1, 2, 5, -4] →
  quantities_sold = [20, 35, 10, 30, 15, 5, 50] →
  cost_price = 8 →
  standard_selling_price = 10 →
  let base_profit := 2 * (20 + 35 + 10 + 30 + 15 + 5 + 50),
  let adjustments := 1 * 20 - 2 * 35 + 3 * 10 - 1 * 30 + 2 * 15 + 5 * 5 - 4 * 50,
  base_profit + adjustments = 135 :=
by
  sorry

theorem promotional_method_cost
  (a : Int) (H : a > 5) :
  let method_one_cost := 9.6 * a + 12,
  let method_two_cost := 10 * a,
  method_one_cost = 9.6 * a + 12 ∧
  method_two_cost = 10 * a :=
by
  sorry

theorem cost_effectiveness 
  (a : Int) :
  a = 35 →
  let method_one_cost := (35 - 5) * 12 * 0.8 + 12 * 5,
  let method_two_cost := 35 * 10,
  method_one_cost = 348 ∧
  method_two_cost = 350 ∧
  method_one_cost < method_two_cost :=
by
  sorry

end highest_selling_price_day_weekly_profit_promotional_method_cost_cost_effectiveness_l589_589755


namespace equation_of_circle_ABC_l589_589455

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589455


namespace savings_after_increase_l589_589767

-- Conditions
def salary : ℕ := 5000
def initial_savings_ratio : ℚ := 0.20
def expense_increase_ratio : ℚ := 1.20

-- Derived initial values
def initial_savings : ℚ := initial_savings_ratio * salary
def initial_expenses : ℚ := ((1 : ℚ) - initial_savings_ratio) * salary

-- New expenses after increase
def new_expenses : ℚ := expense_increase_ratio * initial_expenses

-- Savings after expense increase
def final_savings : ℚ := salary - new_expenses

theorem savings_after_increase : final_savings = 200 := by
  sorry

end savings_after_increase_l589_589767


namespace circle_through_points_l589_589473

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589473


namespace find_percentage_second_alloy_l589_589837

open Real

def percentage_copper_second_alloy (percentage_alloy1: ℝ) (ounces_alloy1: ℝ) (percentage_desired_alloy: ℝ) (total_ounces: ℝ) (percentage_second_alloy: ℝ) : Prop :=
  let copper_ounces_alloy1 := percentage_alloy1 * ounces_alloy1 / 100
  let desired_copper_ounces := percentage_desired_alloy * total_ounces / 100
  let needed_copper_ounces := desired_copper_ounces - copper_ounces_alloy1
  let ounces_alloy2 := total_ounces - ounces_alloy1
  (needed_copper_ounces / ounces_alloy2) * 100 = percentage_second_alloy

theorem find_percentage_second_alloy :
  percentage_copper_second_alloy 18 45 19.75 108 21 :=
by
  sorry

end find_percentage_second_alloy_l589_589837


namespace circle_equation_correct_l589_589530

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589530


namespace replacement_paint_intensity_l589_589388

def original_intensity : ℝ := 0.50
def new_intensity : ℝ := 0.40
def fraction_replaced : ℝ := 0.40

theorem replacement_paint_intensity : ∃ I : ℝ, (1 - fraction_replaced) * original_intensity + fraction_replaced * I = new_intensity ∧ I = 0.25 :=
by
  exists 0.25
  split
  · sorry -- proving the calculations which are essentially the steps in the solution
  · rfl

end replacement_paint_intensity_l589_589388


namespace general_term_formula_l589_589969

noncomputable def sequence (n : ℕ) : ℕ → ℚ 
| 0     := 1
| (n+1) := (2^(n+1) - 1) * sequence n / (sequence n + 2^n - 1)

theorem general_term_formula (n : ℕ) : sequence (n) = (2^n - 1) / n := by
  sorry

end general_term_formula_l589_589969


namespace scientific_notation_of_1653_billion_l589_589780

theorem scientific_notation_of_1653_billion :
  (1653 * (10 ^ 9) = 1.6553 * (10 ^ 12)) :=
sorry

end scientific_notation_of_1653_billion_l589_589780


namespace circle_equation_correct_l589_589545

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589545


namespace rationalize_denominator_and_sum_l589_589002

theorem rationalize_denominator_and_sum (a b : ℝ) (h₁ : a = real.cbrt 7)
  (h₂ : b = real.cbrt 5) : 
  (let A := 49 in
   let B := 35 in
   let C := 25 in
   let D := 2 in
     (1 / (a - b) = (real.cbrt A + real.cbrt B + real.cbrt C) / D) ∧ 
     A + B + C + D = 111) :=
by 
  sorry

end rationalize_denominator_and_sum_l589_589002


namespace greatest_difference_units_digit_l589_589700

theorem greatest_difference_units_digit :
  let a_values := {a | 9 + a % 3 = 0 ∧ 0 ≤ a ∧ a ≤ 9} in
  ∃ x y ∈ a_values, ∀ z ∈ a_values, x - y = 9 :=
by
  sorry

end greatest_difference_units_digit_l589_589700


namespace parabola_directrix_l589_589939

theorem parabola_directrix
  (p : ℝ) (hp : p > 0)
  (O : ℝ × ℝ := (0,0))
  (Focus_F : ℝ × ℝ := (p / 2, 0))
  (Point_P : ℝ × ℝ)
  (Point_Q : ℝ × ℝ)
  (H1 : Point_P.1 = p / 2 ∧ Point_P.2^2 = 2 * p * Point_P.1)
  (H2 : Point_P.1 = Point_P.1) -- This comes out of the perpendicularity of PF to x-axis
  (H3 : Point_Q.2 = 0)
  (H4 : ∃ k_OP slope_OP, slope_OP = 2 ∧ ∃ k_PQ slope_PQ, slope_PQ = -1 / 2 ∧ k_OP * k_PQ = -1)
  (H5 : abs (Point_Q.1 - Focus_F.1) = 6) :
  x = -3 / 2 := 
sorry

end parabola_directrix_l589_589939


namespace circle_equation_l589_589630

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589630


namespace circle_passing_through_points_l589_589605

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589605


namespace circle_passes_through_points_l589_589523

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589523


namespace cricket_bat_selling_price_l589_589758

theorem cricket_bat_selling_price (profit : ℝ) (profit_percentage : ℝ) (C : ℝ) (selling_price : ℝ) 
  (h1 : profit = 150) 
  (h2 : profit_percentage = 20) 
  (h3 : profit = (profit_percentage / 100) * C) 
  (h4 : selling_price = C + profit) : 
  selling_price = 900 := 
sorry

end cricket_bat_selling_price_l589_589758


namespace determine_h_l589_589389

open Polynomial

noncomputable def f (x : ℚ) : ℚ := x^2

theorem determine_h (h : ℚ → ℚ) : 
  (∀ x, f (h x) = 9 * x^2 + 6 * x + 1) ↔ 
  (∀ x, h x = 3 * x + 1 ∨ h x = - (3 * x + 1)) :=
by
  sorry

end determine_h_l589_589389


namespace equation_of_circle_through_three_points_l589_589577

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589577


namespace circle_passes_through_points_l589_589526

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589526


namespace circle_passing_through_points_l589_589613

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589613


namespace sin_1200_eq_sqrt3_div_2_l589_589222

theorem sin_1200_eq_sqrt3_div_2 : Real.sin (1200 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_1200_eq_sqrt3_div_2_l589_589222


namespace equation_of_circle_through_three_points_l589_589566

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589566


namespace circle_passing_through_points_eqn_l589_589482

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589482


namespace area_of_triangle_l589_589181

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the points of intercepts with axes
def intercepts (x0 y0 x1 y1 : ℝ) : Prop :=
  line_eq 0 y0 ∧ y0 = 9 ∧
  line_eq x1 0 ∧ x1 = 3

-- Define the area of the triangle
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem area_of_triangle : ∃ (x0 y0 x1 y1 : ℝ), 
  intercepts x0 y0 x1 y1 ∧ triangle_area x1 y0 = 13.5 :=
by
  sorry

end area_of_triangle_l589_589181


namespace find_solution_l589_589235

theorem find_solution (n k m : ℕ) (h : 5^n - 3^k = m^2) : 
  (n = 0 ∧ k = 0 ∧ m = 0) ∨ (n = 2 ∧ k = 2 ∧ m = 4) :=
begin
  sorry
end

end find_solution_l589_589235


namespace triangle_area_l589_589177

-- Define the line equation 3x + y = 9
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the triangle bounded by the coordinate axes and the line
def bounded_triangle (x y : ℝ) : Prop :=
  (x = 0 ∧ 0 ≤ y ∧ y ≤ 9) ∨  -- y-axis segment
  (y = 0 ∧ 0 ≤ x ∧ x ≤ 3) ∨  -- x-axis segment
  (3 * x + y = 9 ∧ 0 ≤ x ∧ 0 ≤ y)  -- line segment

-- The area of the triangle formed by the coordinate axes and the line 3x + y = 9
theorem triangle_area : 
  let area := 1/2 * 3 * 9 in -- The calculation of the area
  area = 13.5 := 
by
  sorry

end triangle_area_l589_589177


namespace circle_through_points_l589_589422

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589422


namespace equation_of_circle_through_three_points_l589_589574

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589574


namespace interval_contains_p_l589_589393

noncomputable def P : Type := ℚ

variable (A B : P)

axiom P_A : A = 5 / 6
axiom P_B : B = 3 / 4
axiom P_A_given_B : A / B = 2 / 3

theorem interval_contains_p : ∃ p : P, p = (2 / 3) * (3 / 4) ∧ p ∈ set.Icc (1 / 2 : P) (3 / 4 : P) :=
by {
  let p := (2 / 3) * (3 / 4),
  use p,
  split,
  { exact rfl },
  { split,
    { exact le_refl (1 / 2) },
    exact le_of_lt (by norm_num) }
}

end interval_contains_p_l589_589393


namespace rhombus_area_l589_589400

/-- The area of a rhombus with diagonals of 13 cm and 20 cm is 130 square centimeters. -/
theorem rhombus_area :
  ∀ (d1 d2 : ℝ), d1 = 13 → d2 = 20 → (d1 * d2) / 2 = 130 :=
by
  intros d1 d2 h1 h2
  rw [h1, h2]
  calc
    (13 * 20) / 2 = 260 / 2 : by norm_num -- Simplify the multiplication
                ... = 130    : by norm_num -- Divide to get the final result

end rhombus_area_l589_589400


namespace equation_of_circle_ABC_l589_589438

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589438


namespace problem_solution_l589_589927

open Real

variable {f : ℝ → ℝ}
variable {a : ℝ}

-- function and derivative conditions
def symmetric_about_2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = f(4 - x)

def derivative_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 2 → (x-2) * (deriv f x) > 0

-- Hypotheses
variables (h_symm : symmetric_about_2 f) (h_deriv : derivative_condition f) (h_a : 2 < a ∧ a < 4)

theorem problem_solution : f(log 2 a) < f 3 ∧ f 3 < f (2 ^ a) :=
by
  sorry

end problem_solution_l589_589927


namespace max_points_on_line_through_circles_l589_589914

theorem max_points_on_line_through_circles (C₁ C₂ C₃ C₄ : Circle) (h_coplanar : coplanar {C₁, C₂, C₃, C₄}) : 
  ∃ max_points : ℕ, max_points = 8 := 
sorry

end max_points_on_line_through_circles_l589_589914


namespace bus_speed_and_interval_l589_589094

-- Definitions and conditions
variables (a b c : ℝ) (h : c > b)

-- Definitions of the unknowns
def x : ℝ := a * (c + b) / (c - b)
def t : ℝ := 2 * b * c / (b + c)

theorem bus_speed_and_interval :
  (a + x) * b = t * x ∧ (x - a) * c = t * x :=
by
  unfold x t
  have h₁ : (a * (c + b) / (c - b)) = x, unfold x
  rw [h₁]
  split
  · calc
      (a + a * (c + b) / (c - b)) * b
      = (a * (c - b) / (c - b) + a * (c + b) / (c - b)) * b : by rw [mul_div_cancel' _ (sub_ne_zero.mpr h.ne.symm)]
  sorry
  · calc
      (a * (c + b) / (c - b) - a) * c
      = (a * (c + b) / (c - b) - a * (c - b) / (c - b)) * c : by rw [mul_div_cancel' _ (sub_ne_zero.mpr h.ne.symm)]
  sorry

end bus_speed_and_interval_l589_589094


namespace circle_passing_through_points_l589_589506

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589506


namespace dacid_physics_marks_l589_589217

theorem dacid_physics_marks 
  (english : ℕ := 73)
  (math : ℕ := 69)
  (chem : ℕ := 64)
  (bio : ℕ := 82)
  (avg_marks : ℕ := 76)
  (num_subjects : ℕ := 5)
  : ∃ physics : ℕ, physics = 92 :=
by
  let total_marks := avg_marks * num_subjects
  let known_marks := english + math + chem + bio
  have physics := total_marks - known_marks
  use physics
  sorry

end dacid_physics_marks_l589_589217


namespace exponent_equivalence_l589_589247

theorem exponent_equivalence (a b : ℕ) (m n : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) (h1 : 9 ^ m = a) (h2 : 3 ^ n = b) : 
  3 ^ (2 * m + 4 * n) = a * b ^ 4 := 
by 
  sorry

end exponent_equivalence_l589_589247


namespace circle_passing_through_points_eqn_l589_589491

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589491


namespace first_term_to_common_difference_ratio_l589_589087

theorem first_term_to_common_difference_ratio (a d : ℝ) 
  (h : (14 / 2) * (2 * a + 13 * d) = 3 * (7 / 2) * (2 * a + 6 * d)) :
  a / d = 4 :=
by
  sorry

end first_term_to_common_difference_ratio_l589_589087


namespace circle_through_points_l589_589431

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589431


namespace change_four_numbers_to_break_magic_square_l589_589902

def magic_square := matrix (fin 3) (fin 3) ℕ

def is_magic_square (m : magic_square) : Prop :=
  let sum := (∑ i, m 0 i) in
  (∀ i, (∑ j, m i j) = sum) ∧
  (∀ j, (∑ i, m i j) = sum) ∧
  (∑ i, m i i) = sum ∧ 
  (∑ i, m i (fin.mk (2 - i) sorry)) = sum

def given_magic_square : magic_square :=
  ![[4, 9, 2],
    [8, 1, 6],
    [3, 5, 7]]

theorem change_four_numbers_to_break_magic_square :
  ∃ (m' : magic_square), (∑ i, m' 0 i ≠ ∑ i, m' 1 i) ∧ (∑ i, m' 1 i ≠ ∑ i, m' 2 i) ∧ (∃ (a b c d: (fin 3) × (fin 3)), m' a ≠ given_magic_square a ∧ m' b ≠ given_magic_square b ∧ m' c ≠ given_magic_square c ∧ m' d ≠ given_magic_square d) :=
sorry

end change_four_numbers_to_break_magic_square_l589_589902


namespace university_chess_tournament_number_of_students_l589_589067

theorem university_chess_tournament_number_of_students (x : ℕ) (p : ℚ) :
  (∀ i j : ℕ, i ≠ j → 0 ≤ p ∧ p ≤ 1) →
  ((2*p + x*p = x*(x - 1)/2) ∧ (2*p + x*p = 6.5 + x*p) →
  x = 11 := 
begin
  sorry
end

end university_chess_tournament_number_of_students_l589_589067


namespace circle_passing_through_points_l589_589493

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589493


namespace distinct_real_roots_iff_m_lt_13_over_4_equal_real_roots_root_eq_3_over_2_l589_589967

variable (m : ℝ)

-- Part 1: Prove that if the quadratic equation has two distinct real roots, then m < 13/4.
theorem distinct_real_roots_iff_m_lt_13_over_4 (h : (3 * 3 - 4 * (m - 1)) > 0) : m < 13 / 4 := 
by
  sorry

-- Part 2: Prove that if the quadratic equation has two equal real roots, then the root is 3/2.
theorem equal_real_roots_root_eq_3_over_2 (h : (3 * 3 - 4 * (m - 1)) = 0) : m = 13 / 4 ∧ ∀ x, (x^2 + 3 * x + (13/4 - 1) = 0) → x = 3 / 2 :=
by
  sorry

end distinct_real_roots_iff_m_lt_13_over_4_equal_real_roots_root_eq_3_over_2_l589_589967


namespace circle_through_points_l589_589426

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589426


namespace inverse_function_correct_l589_589681

noncomputable def inverse_of_exp_shifted (y : ℝ) := ln (y - 1) + 1

theorem inverse_function_correct :
  ∀ (x : ℝ), 
    y = Real.exp (x - 1) + 1 -> (inverse_of_exp_shifted y = x) :=
by
  sorry

end inverse_function_correct_l589_589681


namespace calculate_sum_of_money_l589_589102

theorem calculate_sum_of_money
    (P : ℝ)
    (r : ℝ := 0.06)
    (t : ℕ := 5)
    (diff : ℝ := 250) :
    let SI := P * r * t
    let CI := P * (1 + r / 2)^(2 * t) - P
    CI - SI = diff →
    P ≈ 5692.47 :=
by
  intro h
  sorry

end calculate_sum_of_money_l589_589102


namespace circle_through_points_l589_589424

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589424


namespace intersection_points_with_x_axis_l589_589676

theorem intersection_points_with_x_axis (a : ℝ) :
    (∃ x : ℝ, a * x^2 - a * x + 3 * x + 1 = 0 ∧ 
              ∀ x' : ℝ, (x' ≠ x → a * x'^2 - a * x' + 3 * x' + 1 ≠ 0)) ↔ 
    (a = 0 ∨ a = 1 ∨ a = 9) := by 
  sorry

end intersection_points_with_x_axis_l589_589676


namespace equation_of_circle_ABC_l589_589444

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589444


namespace equation_of_circle_passing_through_points_l589_589657

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589657


namespace greatest_difference_units_digit_l589_589701

theorem greatest_difference_units_digit :
  let a_values := {a | 9 + a % 3 = 0 ∧ 0 ≤ a ∧ a ≤ 9} in
  ∃ x y ∈ a_values, ∀ z ∈ a_values, x - y = 9 :=
by
  sorry

end greatest_difference_units_digit_l589_589701


namespace triangle_area_l589_589176

-- Define the line equation 3x + y = 9
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the triangle bounded by the coordinate axes and the line
def bounded_triangle (x y : ℝ) : Prop :=
  (x = 0 ∧ 0 ≤ y ∧ y ≤ 9) ∨  -- y-axis segment
  (y = 0 ∧ 0 ≤ x ∧ x ≤ 3) ∨  -- x-axis segment
  (3 * x + y = 9 ∧ 0 ≤ x ∧ 0 ≤ y)  -- line segment

-- The area of the triangle formed by the coordinate axes and the line 3x + y = 9
theorem triangle_area : 
  let area := 1/2 * 3 * 9 in -- The calculation of the area
  area = 13.5 := 
by
  sorry

end triangle_area_l589_589176


namespace marbles_in_larger_container_l589_589757

-- Defining the conditions
def volume1 := 24 -- in cm³
def marbles1 := 30 -- number of marbles in the first container
def volume2 := 72 -- in cm³

-- Statement of the theorem
theorem marbles_in_larger_container : (marbles1 / volume1 : ℚ) * volume2 = 90 := by
  sorry

end marbles_in_larger_container_l589_589757


namespace Alyssa_total_spent_l589_589187

-- Declare the costs of grapes and cherries.
def costOfGrapes : ℝ := 12.08
def costOfCherries : ℝ := 9.85

-- Total amount spent by Alyssa.
def totalSpent : ℝ := 21.93

-- Statement to prove that the sum of the costs is equal to the total spent.
theorem Alyssa_total_spent (g : ℝ) (c : ℝ) (t : ℝ) 
  (hg : g = costOfGrapes) 
  (hc : c = costOfCherries) 
  (ht : t = totalSpent) :
  g + c = t := by
  sorry

end Alyssa_total_spent_l589_589187


namespace equation_of_circle_passing_through_points_l589_589655

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589655


namespace circle_passing_three_points_l589_589409

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589409


namespace circle_through_points_l589_589584

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589584


namespace circle_passes_through_points_l589_589524

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589524


namespace circle_passing_three_points_l589_589410

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589410


namespace maria_potatoes_l589_589370

theorem maria_potatoes : ∃ P : ℕ, (let C := 6 * P in let O := 2 * C in let G := (1 / 3 : ℚ) * O in G = 8) ↔ P = 2 :=
by
  sorry

end maria_potatoes_l589_589370


namespace parallelogram_sides_l589_589687

theorem parallelogram_sides (h₁ : parallelogram ABCD)
    (h₂ : ABCD.perimeter = 26)
    (h₃ : ∠ABC = 120)
    (h₄ : circle_inscribed BCD r)
    (h₅ : r = √3)
    (h₆ : side AD > side AB) :
    exists (a b : ℝ), (AB = a ∧ AD = b) ∧ (a = 5) ∧ (b = 8) :=
by
  sorry

end parallelogram_sides_l589_589687


namespace equation_of_circle_through_three_points_l589_589578

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589578


namespace age_difference_l589_589024

theorem age_difference (E Y : ℕ) (h1 : E = 32) (h2 : Y = 12) : E - Y = 20 :=
by
  rw [h1, h2]
  norm_num

end age_difference_l589_589024


namespace total_heartbeats_during_race_l589_589797

-- Definitions for the conditions
def heart_beats_per_minute : ℕ := 160
def pace_in_minutes_per_mile : ℕ := 6
def total_distance_in_miles : ℕ := 30

-- Main theorem statement
theorem total_heartbeats_during_race : 
  heart_beats_per_minute * pace_in_minutes_per_mile * total_distance_in_miles = 28800 :=
by
  -- Place the proof here
  sorry

end total_heartbeats_during_race_l589_589797


namespace monotonic_interval_of_trig_function_l589_589032

theorem monotonic_interval_of_trig_function :
  ∃ (k : ℤ), ∀ x, (x ∈ set.Icc (-5 + 12 * k) (1 + 12 * k)) → 
    ∃ ω φ, 
      (f : ℝ → ℝ) (f = λ x, sin (ω * x + φ) + sqrt 3 * cos (ω * x + φ)) ∧
      (f 1 = 2) ∧
      ∃ x1 x2, f x1 = 0 ∧ f x2 = 0 ∧ abs (x1 - x2) = 6 :=
  sorry

end monotonic_interval_of_trig_function_l589_589032


namespace expected_angle_ABC_l589_589709

noncomputable def expected_angle_ABC_uniform_unit_square : ℝ := 60

theorem expected_angle_ABC :
    ∀ (A B C : ℝ × ℝ),
    A ∈ set.Icc (0 : ℝ) 1 × set.Icc (0 : ℝ) 1 →
    B ∈ set.Icc (0 : ℝ) 1 × set.Icc (0 : ℝ) 1 →
    C ∈ set.Icc (0 : ℝ) 1 × set.Icc (0 : ℝ) 1 →
    (random_points : (ℝ × ℝ) → (ℝ × ℝ)) →
    (uniform_dist : measure_theory.measure (ℝ × ℝ)) →
    uniform_dist = measure_theory.volume →
    measure_theory.integrable (λ (p : ℝ × ℝ), (p.fst : ℝ) * (p.snd : ℝ)) uniform_dist →
    (∃ θ : ℝ → ℝ, θ = expected_angle_ABC_uniform_unit_square) :=
by sorry

end expected_angle_ABC_l589_589709


namespace perpendicular_lines_l589_589223

theorem perpendicular_lines (a : ℝ) 
  (h1 : (3 : ℝ) * y + (2 : ℝ) * x - 6 = 0) 
  (h2 : (4 : ℝ) * y + a * x - 5 = 0) : 
  a = -6 :=
sorry

end perpendicular_lines_l589_589223


namespace exists_root_interval_1_exists_root_interval_2_l589_589341

noncomputable def f (x : ℝ) : ℝ :=
  x - log x - 2

theorem exists_root_interval_1 : ∃ x, 0 < x ∧ x < 1 ∧ f x = 0 :=
  sorry

theorem exists_root_interval_2 : ∃ x, 3 < x ∧ x < 4 ∧ f x = 0 :=
  sorry

end exists_root_interval_1_exists_root_interval_2_l589_589341


namespace rectangle_area_ratio_l589_589100

theorem rectangle_area_ratio (s x y : ℝ) (h_square : s > 0)
    (h_side_ae : x > 0) (h_side_ag : y > 0)
    (h_ratio_area : x * y = (1 / 4) * s^2) :
    ∃ (r : ℝ), r > 0 ∧ r = x / y := 
sorry

end rectangle_area_ratio_l589_589100


namespace determine_criminal_l589_589868

def is_criminal (P : Type) (A B C D : P) (criminal : P) : Prop :=
  (criminal = B ∨ criminal = C ∨ criminal = D) →
  (criminal ≠ B ∧ criminal = C) →
  (criminal = A ∨ criminal = B) →
  (criminal = C) →
  (criminal = B)

/-- Assume there exists only one criminal among A, B, C, and D,
    and from the conditions given, determine the criminal -/
theorem determine_criminal (P : Type) (A B C D : P) (criminal : P)
  (h1 : criminal ∈ {B, C, D})
  (h2 : B ≠ criminal ∧ criminal = C)
  (h3 : criminal ∈ {A, B})
  (h4 : criminal = C)
  (h5 : (¬(criminal ≠ B ∧ criminal = C)) ∧ (criminal = C)) :
  criminal = B :=
by
  sorry

end determine_criminal_l589_589868


namespace converges_in_probability_and_same_distribution_l589_589363

theorem converges_in_probability_and_same_distribution
  (ξ η : ℕ → ℝ)
  (ξ_lim η_lim : ℝ)
  (h_same_finite_distributions: ∀ n : ℕ, finite_dist_equiv (ξ n) (η n))
  (h_converges_in_probability: converges_in_probability ξ ξ_lim) :
  ∃ η_lim, converges_in_probability η η_lim ∧ same_distribution ξ_lim η_lim :=
by
  sorry

end converges_in_probability_and_same_distribution_l589_589363


namespace stratified_sampling_correct_l589_589773

def stratified_sampling (total_teachers senior intermediate junior sample_size : ℕ) : 
  ℕ × ℕ × ℕ :=
  let total_ = senior + intermediate + junior
  let senior_sample := senior * sample_size / total_
  let intermediate_sample := intermediate * sample_size / total_
  let junior_sample := sample_size - senior_sample - intermediate_sample
  (senior_sample, intermediate_sample, junior_sample)

theorem stratified_sampling_correct : 
  stratified_sampling 300 90 150 60 40 = (12, 20, 8) :=
by {
  -- computation calculations
  sorry
}

end stratified_sampling_correct_l589_589773


namespace carnations_percentage_l589_589101

-- Definitions based on given conditions
def Carnations := ℕ
def Violets (C : Carnations) := (1 / 3 : ℚ) * C
def Tulips (C : Carnations) := (1 / 12 : ℚ) * C
def Roses (C : Carnations) := Tulips C

-- Theorem stating that the percentage of carnations is 66.67%
theorem carnations_percentage (C : Carnations) :
  let total_flowers := C + Violets C + Tulips C + Roses C in
  ((C : ℚ) / total_flowers) * 100 = (200 / 3 : ℚ) :=
by
  sorry

end carnations_percentage_l589_589101


namespace find_circle_equation_l589_589554

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589554


namespace circle_through_points_l589_589456

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589456


namespace circle_through_points_l589_589592

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589592


namespace tan_double_angle_l589_589921

open Real

theorem tan_double_angle (x : ℝ) (h1 : x ∈ Ioo(-π/2) 0) (h2 : cos x = 4/5) : tan (2 * x) = -24/7 :=
by
  sorry

end tan_double_angle_l589_589921


namespace factorization_of_polynomial_l589_589208

theorem factorization_of_polynomial (x : ℂ) :
  x^12 - 729 = (x^2 + 3) * (x^4 - 3 * x^2 + 9) * (x^2 - 3) * (x^4 + 3 * x^2 + 9) :=
by sorry

end factorization_of_polynomial_l589_589208


namespace mass_of_man_l589_589751

def length_of_boat : ℝ := 7 -- length in meters
def breadth_of_boat : ℝ := 2 -- breadth in meters
def sinking_height : ℝ := 0.01 -- height in meters (1 cm = 0.01 m)
def density_of_water : ℝ := 1000 -- density of water in kg/m³

theorem mass_of_man :
  let volume_displaced := length_of_boat * breadth_of_boat * sinking_height in
  let mass := density_of_water * volume_displaced in
  mass = 140 :=
by
  -- proof goes here
  sorry

end mass_of_man_l589_589751


namespace ab_value_l589_589310

theorem ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : let L := 4*a*x + 4*b*y - 48 in ∀ x y, (x = 0 → y ≥ 0 → L = 0) ∧ (y = 0 → x ≥ 0 → L = 0) ∧ by calculus.area.triangle (4*a*x + 4*b*y = 48) = 48) : 
  a * b = 3/2 :=
sorry

end ab_value_l589_589310


namespace sufficient_not_necessary_condition_l589_589273

variables (A B C D : Prop)

theorem sufficient_not_necessary_condition :
  (A → B) → (B ↔ C) → (C → D) → (A → D) :=
begin
  intros h1 h2 h3,
  exact h3 (h2.1 (h1 _)),
end

end sufficient_not_necessary_condition_l589_589273


namespace num_right_triangles_with_leg_sqrt_1001_l589_589983

theorem num_right_triangles_with_leg_sqrt_1001 : 
  ∃ (a b : ℕ), a^2 + 1001 = b^2 → 
  (∃ x y : ℕ, x^2 + 1001 = y^2 ∧ x ≠ a ∧ y ≠ b) → 
  4 := 
sorry

end num_right_triangles_with_leg_sqrt_1001_l589_589983


namespace slope_AB_l589_589695

noncomputable def A := (0 : ℝ, -1 : ℝ)
noncomputable def B := (2 : ℝ, 4 : ℝ)

theorem slope_AB : (B.snd - A.snd) / (B.fst - A.fst) = 5 / 2 := by 
  sorry

end slope_AB_l589_589695


namespace equation_of_circle_passing_through_points_l589_589670

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589670


namespace part1_part2_l589_589283

open Set Real

def f (x : ℝ) := sqrt (6 / (x + 1) - 1)

def A := { x : ℝ | -1 < x ∧ x ≤ 5 }

def B (m : ℝ) := { x : ℝ | -x^2 + m * x + 4 > 0 }

variable (m : ℝ)

theorem part1 (hm : m = 3) : A ∩ (B m)ᶜ = Icc 4 5 :=
by {
  rw hm,
  sorry
}

theorem part2 : ¬ ∃ m, B m ⊆ A :=
by {
  sorry
}

end part1_part2_l589_589283


namespace parallel_lines_m_value_l589_589971

/-- Given two lines l_1: (3 + m) * x + 4 * y = 5 - 3 * m, and l_2: 2 * x + (5 + m) * y = 8,
the value of m for which l_1 is parallel to l_2 is -7. -/
theorem parallel_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m) →
  (∀ x y : ℝ, 2 * x + (5 + m) * y = 8) →
  m = -7 :=
sorry

end parallel_lines_m_value_l589_589971


namespace sum_of_angles_l589_589248

theorem sum_of_angles (α β : ℝ) (h1 : α ∈ Ioo 0 π) (h2 : β ∈ Ioo 0 π)
  (h3 : Real.cos α = (Real.sqrt 10) / 10) (h4 : Real.cos β = (Real.sqrt 5) / 5) :
  α + β = 3 * π / 4 :=
sorry

end sum_of_angles_l589_589248


namespace sum_of_sequence_l589_589675

noncomputable def a (n : ℕ) : ℝ := n * Real.cos (n * Real.pi / 2) + 1

noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)

theorem sum_of_sequence : S 2014 = 1006 :=
by 
-- The proof is omitted as requested
sorry

end sum_of_sequence_l589_589675


namespace number_of_solutions_sin_eq_exp_decay_l589_589898

theorem number_of_solutions_sin_eq_exp_decay :
  ∃ n : ℕ, n = 75 ∧
  (∀ x ∈ Icc (0 : ℝ) (150 * π), sin x = (1 / 3) ^ x → x ∈ ⋃ n : ℕ, Ioc (2 * π * n) (2 * π * n + π)) :=
sorry

end number_of_solutions_sin_eq_exp_decay_l589_589898


namespace picnic_total_slices_l589_589859

def dannys_slices := 3 * 10
def sisters_slices := 1 * 15
def cousins_watermelon_slices := 2 * 8
def cousins_apple_slices := 5 * 4
def aunts_watermelon_slices := 4 * 12
def aunts_orange_slices := 7 * 6
def grandfathers_watermelon_slices := 1 * 6
def grandfathers_pineapple_slices := 3 * 10

def total_watermelon_slices := dannys_slices + sisters_slices + cousins_watermelon_slices + aunts_watermelon_slices + grandfathers_watermelon_slices
def total_fruit_slices := cousins_apple_slices + aunts_orange_slices + grandfathers_pineapple_slices
def total_slices := total_watermelon_slices + total_fruit_slices

theorem picnic_total_slices :
  total_slices = 207 := by
  unfold total_slices
  unfold total_watermelon_slices
  unfold total_fruit_slices
  unfold dannys_slices
  unfold sisters_slices
  unfold cousins_watermelon_slices
  unfold aunts_watermelon_slices
  unfold grandfathers_watermelon_slices
  unfold cousins_apple_slices
  unfold aunts_orange_slices
  unfold grandfathers_pineapple_slices
  simp
  sorry

end picnic_total_slices_l589_589859


namespace find_incorrect_statement_l589_589256

variables {m n : Line} {α β γ : Plane}

-- Conditions for each statement.
axiom statement_A_conditions : m ∥ α ∧ m ∈ β ∧ (α ∩ β = n)
axiom statement_B_conditions : m ∥ n ∧ m ∥ α
axiom statement_C_conditions : (α ∩ β = n) ∧ α ⟂ γ ∧ β ⟂ γ
axiom statement_D_conditions : m ⟂ α ∧ m ⟂ β ∧ α ∥ γ

-- Statements
def statement_A : Prop := m ∥ α ∧ m ∈ β ∧ (α ∩ β = n) → m ∥ n
def statement_B : Prop := m ∥ n ∧ m ∥ α → n ∥ α
def statement_C : Prop := (α ∩ β = n) ∧ α ⟂ γ ∧ β ⟂ γ → n ⟂ γ
def statement_D : Prop := m ⟂ α ∧ m ⟂ β ∧ α ∥ γ → β ∥ γ

theorem find_incorrect_statement : ¬statement_B :=
by
    -- Sorry is used here to skip the actual proof
    sorry

end find_incorrect_statement_l589_589256


namespace auntie_em_can_park_l589_589770

noncomputable def parking_probability : ℚ := 
  let total_ways := (nat.choose 20 15) in
  let no_adjacent_empty_spaces := (nat.choose 16 5) in
  1 - no_adjacent_empty_spaces / total_ways

theorem auntie_em_can_park : parking_probability = 232 / 323 :=
  sorry

end auntie_em_can_park_l589_589770


namespace total_students_multiple_of_8_l589_589685

theorem total_students_multiple_of_8 (B G T : ℕ) (h : G = 7 * B) (ht : T = B + G) : T % 8 = 0 :=
by
  sorry

end total_students_multiple_of_8_l589_589685


namespace Ken_bought_2_pounds_of_steak_l589_589198

theorem Ken_bought_2_pounds_of_steak (pound_cost total_paid change: ℝ) 
    (h1 : pound_cost = 7) 
    (h2 : total_paid = 20) 
    (h3 : change = 6) : 
    (total_paid - change) / pound_cost = 2 :=
by
  sorry

end Ken_bought_2_pounds_of_steak_l589_589198


namespace expression_positive_for_all_integers_l589_589000

theorem expression_positive_for_all_integers (n : ℤ) : 6 * n^2 - 7 * n + 2 > 0 :=
by
  sorry

end expression_positive_for_all_integers_l589_589000


namespace max_f_l589_589965

def star (a b : ℝ) : ℝ := if a ≤ b then a else b

def f (x : ℝ) : ℝ := star 1 (2 ^ x)

theorem max_f : ∀ x, f x ≤ 1 := 
by 
  sorry

end max_f_l589_589965


namespace solve_cubic_equation_l589_589238

theorem solve_cubic_equation :
  ∀ (z : ℂ), z^3 = 27 ↔ (z = 3 ∨ z = -3/2 + (3 * complex.I * real.sqrt 3) / 2 ∨ z = -3/2 - (3 * complex.I * real.sqrt 3) / 2) :=
by
  sorry

end solve_cubic_equation_l589_589238


namespace melted_mixture_weight_l589_589103

theorem melted_mixture_weight :
  ∀ (zinc copper : ℕ) (weight_of_zinc : ℕ), 
    zinc = 9 ∧ copper = 11 ∧ weight_of_zinc = 27 → 
    ∃ (total_weight : ℕ), total_weight = 60 :=
by
  intros zinc copper weight_of_zinc h
  rcases h with ⟨hzinc, hcopper, hweight_of_zinc⟩
  have p1 : weight_of_zinc / zinc = 3 := by
    rw [hweight_of_zinc, hzinc]
    norm_num
  have p2 : weight_of_zinc / zinc * copper = 33 := by
    rw [p1, hcopper]
    norm_num
  use weight_of_zinc + weight_of_zinc / zinc * copper
  rw [hweight_of_zinc, p2]
  norm_num

end melted_mixture_weight_l589_589103


namespace smallest_non_factor_product_of_factors_of_72_l589_589713

theorem smallest_non_factor_product_of_factors_of_72 : 
  ∃ x y : ℕ, x ≠ y ∧ x * y ∣ 72 ∧ ¬ (x * y ∣ 72) ∧ x * y = 32 := 
by
  sorry

end smallest_non_factor_product_of_factors_of_72_l589_589713


namespace range_of_m_l589_589920

-- Define the polynomial p(x)
def p (x : ℝ) (m : ℝ) := x^2 + 2*x - m

-- Given conditions: p(1) is false and p(2) is true
theorem range_of_m (m : ℝ) : 
  (p 1 m ≤ 0) ∧ (p 2 m > 0) → (3 ≤ m ∧ m < 8) :=
by
  sorry

end range_of_m_l589_589920


namespace fifth_term_arithmetic_sequence_l589_589029

theorem fifth_term_arithmetic_sequence :
  ∃ (a_1 a_9 a_5 d : ℚ),
    a_1 = 3/5 ∧
    a_9 = 2/3 ∧
    d = (2/3 - 3/5) / 8 ∧
    a_5 = a_1 + 4 * d ∧
    a_5 = 19/30 :=
begin
  existsi 3/5, existsi 2/3, existsi 19/30, existsi (2/3 - 3/5) / 8,
  split, refl,
  split, refl,
  split, norm_num,
  split, norm_num,
  sorry,
end

end fifth_term_arithmetic_sequence_l589_589029


namespace circle_through_points_l589_589582

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589582


namespace tan_inequality_l589_589958

open Real

theorem tan_inequality {x1 x2 : ℝ} 
  (h1 : 0 < x1 ∧ x1 < π / 2) 
  (h2 : 0 < x2 ∧ x2 < π / 2) 
  (h3 : x1 ≠ x2) : 
  (1 / 2 * (tan x1 + tan x2) > tan ((x1 + x2) / 2)) :=
sorry

end tan_inequality_l589_589958


namespace sequence_general_term_l589_589259

noncomputable def a_n (n : ℕ) : ℝ :=
  sorry

-- The main statement
theorem sequence_general_term (p q : ℝ) (hp : 0 < p) (hq : 0 < q)
  (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ m n : ℕ, |a (m + n) - a m - a n| ≤ 1 / (p * m + q * n)) :
  ∀ n : ℕ, a n = n :=
by
  sorry

end sequence_general_term_l589_589259


namespace circle_passing_through_points_l589_589614

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589614


namespace heartbeats_during_race_l589_589804

theorem heartbeats_during_race
  (heart_rate : ℕ) -- beats per minute
  (pace : ℕ) -- minutes per mile
  (distance : ℕ) -- miles
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) : 
  heart_rate * (pace * distance) = 28800 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end heartbeats_during_race_l589_589804


namespace circle_passing_through_points_l589_589496

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589496


namespace athlete_heartbeats_l589_589812

theorem athlete_heartbeats
  (heart_beats_per_minute : ℕ)
  (pace_minutes_per_mile : ℕ)
  (total_miles : ℕ)
  (heart_beats_per_minute_eq : heart_beats_per_minute = 160)
  (pace_minutes_per_mile_eq : pace_minutes_per_mile = 6)
  (total_miles_eq : total_miles = 30) :
  heart_beats_per_minute * pace_minutes_per_mile * total_miles = 28800 :=
by {
  have h1 : heart_beats_per_minute = 160 := heart_beats_per_minute_eq,
  have h2 : pace_minutes_per_mile = 6 := pace_minutes_per_mile_eq,
  have h3 : total_miles = 30 := total_miles_eq,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end athlete_heartbeats_l589_589812


namespace travel_from_A_to_D_count_l589_589206

theorem travel_from_A_to_D_count :
  let roads_AB := 3 in
  let roads_AD := 3 in
  let roads_BC := 2 in
  let roads_AC := 1 in
  let roads_CD := 1 in
  roads_AB * roads_BC * roads_CD + roads_AC * roads_BC * roads_AD = 20 :=
by
  let roads_AB := 3
  let roads_AD := 3
  let roads_BC := 2
  let roads_AC := 1
  let roads_CD := 1
  sorry

end travel_from_A_to_D_count_l589_589206


namespace numberOfKevins_l589_589323

-- Definitions based on conditions
def Barrys := 24
def Julies := 80
def Joes := 50
def totalNicePeople := 99

def niceBarrys := Barrys -- All Barrys are nice
def niceJulies := (3 / 4) * Julies
def niceJoes := (10 / 100) * Joes

def nicePeopleWithoutKevins := niceBarrys + niceJulies + niceJoes

-- Theorem to prove the number of Kevins
theorem numberOfKevins : ∃ (Kevins : ℕ), 2 * (totalNicePeople - nicePeopleWithoutKevins) = Kevins := by
  let niceKevins := totalNicePeople - nicePeopleWithoutKevins
  use 2 * niceKevins
  sorry

end numberOfKevins_l589_589323


namespace right_triangle_hypotenuse_equals_area_l589_589887

/-- Given a right triangle where the hypotenuse is equal to the area, 
    show that the scaling factor x satisfies the equation. -/
theorem right_triangle_hypotenuse_equals_area 
  (m n x : ℝ) (h_hyp: (m^2 + n^2) * x = mn * (m^2 - n^2) * x^2) :
  x = (m^2 + n^2) / (mn * (m^2 - n^2)) := 
by
  sorry

end right_triangle_hypotenuse_equals_area_l589_589887


namespace count_multiples_of_10_between_9_and_101_l589_589986

theorem count_multiples_of_10_between_9_and_101 : 
  let multiples := { n : ℤ | 9 < n ∧ n < 101 ∧ ∃ k : ℤ, n = 10 * k } in
  ∃ N : ℕ, N = 10 ∧ N = (multiples.to_finset.card : ℕ) :=
begin
  sorry
end

end count_multiples_of_10_between_9_and_101_l589_589986


namespace find_polynomial_h_l589_589392

theorem find_polynomial_h (f h : ℝ → ℝ) (hf : ∀ x, f x = x^2) (hh : ∀ x, f (h x) = 9 * x^2 + 6 * x + 1) : 
  (∀ x, h x = 3 * x + 1) ∨ (∀ x, h x = -3 * x - 1) :=
by
  sorry

end find_polynomial_h_l589_589392


namespace athlete_heartbeats_during_race_l589_589825

theorem athlete_heartbeats_during_race
  (heart_rate : ℕ)
  (pace : ℕ)
  (distance : ℕ)
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) :
  heart_rate * pace * distance = 28800 :=
by
  rw [H1, H2, H3]
  norm_num

end athlete_heartbeats_during_race_l589_589825


namespace percentage_increase_in_overtime_rate_l589_589114

def regular_rate : ℝ := 16
def regular_hours : ℝ := 40
def total_compensation : ℝ := 976
def total_hours_worked : ℝ := 52

theorem percentage_increase_in_overtime_rate :
  ((total_compensation - (regular_rate * regular_hours)) / (total_hours_worked - regular_hours) - regular_rate) / regular_rate * 100 = 75 :=
by
  sorry

end percentage_increase_in_overtime_rate_l589_589114


namespace no_zero_digit_product_l589_589842

noncomputable def no_zero_digits (n : ℕ) : Prop :=
  ¬ (∃ k, n = k * 10 + 0 ∨ n = k * 100 + 0 ∨ n = k * 1000 + 0)

theorem no_zero_digit_product :
  ∃ (a b : ℕ), a * b = 1000000000 ∧ no_zero_digits a ∧ no_zero_digits b :=
by
  use 512, 1953125
  exact ⟨ by norm_num, by decidable.norm_not_has_zero, by decidable.norm_not_has_zero ⟩

end no_zero_digit_product_l589_589842


namespace find_circle_equation_l589_589563

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589563


namespace circle_passing_through_points_l589_589601

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589601


namespace proof_problem_l589_589062

-- Define propositions P1, P2, P3, P4
def P1 := ∀ (x y : ℝ), (x + y = 0) → (x = -y)
def converse_P1 := ∀ (x y : ℝ), (x = -y) → (x + y = 0)

def T1 := "The areas of congruent triangles are equal."
def P2 := "If two triangles are not congruent, then their areas are not equal."

def P3 := ∀ (q : ℝ), (q ≤ 1) → ∃ (x : ℝ), (x ^ 2 + 2 * x + q = 0)
def converse_P3 := ∀ (q : ℝ), ∃ (x : ℝ), (x ^ 2 + 2 * x + q = 0) → (q ≤ 1)

def P4 := "The three interior angles of a scalene triangle are equal."
def contrapositive_P4 := "If the three interior angles of a triangle are equal, then it is not scalene."

-- The Lean statement asserting which propositions are true and which are false.
theorem proof_problem : 
  (converse_P1) ∧ ¬(P2) ∧ (converse_P3) ∧ ¬(contrapositive_P4) :=
by
  sorry

end proof_problem_l589_589062


namespace circle_equation_through_points_l589_589645

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589645


namespace B_contribution_to_capital_l589_589776

theorem B_contribution_to_capital (A_capital : ℝ) (A_months : ℝ) (B_months : ℝ) (profit_ratio_A : ℝ) (profit_ratio_B : ℝ) (B_contribution : ℝ) :
  A_capital = 4500 →
  A_months = 12 →
  B_months = 5 →
  profit_ratio_A = 2 →
  profit_ratio_B = 3 →
  B_contribution = (4500 * 12 * 3) / (5 * 2) → 
  B_contribution = 16200 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end B_contribution_to_capital_l589_589776


namespace correct_equation_l589_589322

-- Define the total number of students
def total_students : ℕ := 49

-- let the number of boys be x
variable (x : ℕ)

-- When there is one less boy, the number of boys is half the number of girls
def boys_less_one = x - 1
def girls := 2 * boys_less_one

-- Formulate the equation according to the condition
theorem correct_equation : 2 * (x - 1) + x = 49 :=
by
  sorry

end correct_equation_l589_589322


namespace projection_vector_l589_589131

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let denom := (u.1 * u.1 + u.2 * u.2)
  let scalar := (v.1 * u.1 + v.2 * u.2) / denom
  (scalar * u.1, scalar * u.2)

def vector_3_neg3 := (3 : ℝ, -3 : ℝ)
def vector_75_26_neg15_26 := (75 / 26 : ℝ, -15 / 26 : ℝ)
def vector_5_7 := (5 : ℝ, 7 : ℝ)
def vector_neg3_neg4 := (-3 : ℝ, -4 : ℝ)
def vector_2_3 := (2 : ℝ, 3 : ℝ)
def vector_35_26_neg7_26 := (35 / 26 : ℝ, -7 / 26 : ℝ)
def vector_5_neg1 := (5 : ℝ, -1 : ℝ)

theorem projection_vector:
  (proj vector_5_neg1 vector_3_neg3 = vector_75_26_neg15_26) →
  (proj vector_5_neg1 vector_2_3 = vector_35_26_neg7_26) :=
by
  sorry

end projection_vector_l589_589131


namespace triangle_area_bound_by_line_l589_589151

noncomputable def area_of_triangle : ℝ :=
let x_intercept := 3 in
let y_intercept := 9 in
1 / 2 * x_intercept * y_intercept

theorem triangle_area_bound_by_line (x y : ℝ) (h : 3 * x + y = 9) :
  area_of_triangle = 13.5 :=
sorry

end triangle_area_bound_by_line_l589_589151


namespace largest_four_digit_sum_19_l589_589075

theorem largest_four_digit_sum_19 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (nat.digits 10 n).sum = 19 ∧ 
           ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ (nat.digits 10 m).sum = 19 → m ≤ 8920 :=
begin
  sorry
end

end largest_four_digit_sum_19_l589_589075


namespace circle_passing_through_points_eqn_l589_589487

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589487


namespace circle_through_points_l589_589460

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589460


namespace circle_passing_through_points_eqn_l589_589474

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589474


namespace circle_through_points_l589_589464

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589464


namespace probability_of_red_light_l589_589200

-- Definitions based on the conditions
def red_duration : ℕ := 30
def yellow_duration : ℕ := 5
def green_duration : ℕ := 40

def total_cycle_time : ℕ := red_duration + yellow_duration + green_duration

-- Statement of the problem to prove the probability of seeing red light
theorem probability_of_red_light : (red_duration : ℚ) / total_cycle_time = 2 / 5 := 
by sorry

end probability_of_red_light_l589_589200


namespace line_intersects_circle_find_slope_of_line_l589_589260

-- Definitions for the circle and line
def circle (x y : ℝ) := x^2 + (y - 1)^2 = 5
def line (m x y : ℝ) := mx - y + 1 - m = 0

-- Part 1: Prove that the line intersects the circle at two distinct points for all m ∈ ℝ
theorem line_intersects_circle (m : ℝ) : ∀ x y : ℝ, 
  (circle x y) → (line m x y) → (∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ line m p1.1 p1.2 ∧ line m p2.1 p2.2) :=
sorry

-- Part 2: Given the distance |AB| = √17, find the slope of the line
theorem find_slope_of_line (m : ℝ) : ∀ x1 y1 x2 y2 : ℝ, 
  (circle x1 y1) ∧ (circle x2 y2) ∧ (line m x1 y1) ∧ (line m x2 y2) ∧ (dist (x1, y1) (x2, y2) = √17) → 
  (m = ± √3) :=
sorry

end line_intersects_circle_find_slope_of_line_l589_589260


namespace circle_equation_through_points_l589_589642

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589642


namespace equation_of_circle_through_three_points_l589_589570

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589570


namespace circle_through_points_l589_589459

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589459


namespace heartbeats_during_race_l589_589806

theorem heartbeats_during_race
  (heart_rate : ℕ) -- beats per minute
  (pace : ℕ) -- minutes per mile
  (distance : ℕ) -- miles
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) : 
  heart_rate * (pace * distance) = 28800 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end heartbeats_during_race_l589_589806


namespace find_polynomial_l589_589783

def p (x : ℕ) := x^3 + 2 * x^2 + 4 * x + 8

theorem find_polynomial (p : ℕ → ℕ) (h : ∀ i, (i < 10) → ∃ n, p(n)=i) (h₁: p 10 = 1248) : p = λ x, x^3 + 2 * x^2 + 4 * x + 8 := 
sorry

end find_polynomial_l589_589783


namespace smallest_number_of_coins_l589_589851

theorem smallest_number_of_coins (X Y n : ℕ) (h1 : Y > 1) (h2 : Y < n) (h3: ∀ d, d ∣ n → d > 1 → d < n → d = Y) (h4: (n.proper_divisors).card = 13) : n = 144 :=
by
  sorry

end smallest_number_of_coins_l589_589851


namespace circle_passes_through_points_l589_589513

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589513


namespace circle_through_points_l589_589457

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589457


namespace height_from_A_l589_589756

theorem height_from_A {A B C : Type} [EuclideanGeometry A B C]
  (T : Triangle A B C)
  (radius : ℝ)
  (hcircle : radius = 2)
  (angle_A : ℕ)
  (angle_B : ℕ)
  (hangle_A : angle_A = 30)
  (hangle_B : angle_B = 45) :
  height_from_vertex T A = 2 + 2 * real.sqrt 3 :=
sorry

end height_from_A_l589_589756


namespace equal_pieces_length_l589_589749

theorem equal_pieces_length (total_length_cm : ℕ) (num_pieces : ℕ) (num_equal_pieces : ℕ) (length_remaining_piece_mm : ℕ) :
  total_length_cm = 1165 ∧ num_pieces = 154 ∧ num_equal_pieces = 150 ∧ length_remaining_piece_mm = 100 →
  (total_length_cm * 10 - (num_pieces - num_equal_pieces) * length_remaining_piece_mm) / num_equal_pieces = 75 :=
by
  sorry

end equal_pieces_length_l589_589749


namespace equation_of_circle_passing_through_points_l589_589665

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589665


namespace triangle_area_l589_589169

noncomputable def line_eq := λ x : ℝ, 9 - 3 * x

theorem triangle_area :
    let x_intercept := (3 : ℝ)
    let y_intercept := (9 : ℝ)
    let triangle_base := x_intercept
    let triangle_height := y_intercept
    let area := (1 / 2) * triangle_base * triangle_height
    area = 13.5 :=
by
    sorry

end triangle_area_l589_589169


namespace total_payment_correct_l589_589192

def cost (n : ℕ) : ℕ :=
  if n <= 10 then n * 25
  else 10 * 25 + (n - 10) * (4 * 25 / 5)

def final_cost_with_discount (n : ℕ) : ℕ :=
  let initial_cost := cost n
  if n > 20 then initial_cost - initial_cost / 10
  else initial_cost

def orders_X := 60 * 20 / 100
def orders_Y := 60 * 25 / 100
def orders_Z := 60 * 55 / 100

def cost_X := final_cost_with_discount orders_X
def cost_Y := final_cost_with_discount orders_Y
def cost_Z := final_cost_with_discount orders_Z

theorem total_payment_correct : cost_X + cost_Y + cost_Z = 1279 := by
  sorry

end total_payment_correct_l589_589192


namespace circle_passing_through_points_eqn_l589_589477

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589477


namespace triangle_perimeter_l589_589279

-- Define the conditions of the problem
def a := 4
def b := 8
def quadratic_eq (x : ℝ) : Prop := x^2 - 14 * x + 40 = 0

-- Define the perimeter calculation, ensuring triangle inequality and correct side length
def valid_triangle (x : ℝ) : Prop :=
  x ≠ a ∧ x ≠ b ∧ quadratic_eq x ∧ (a + b > x) ∧ (a + x > b) ∧ (b + x > a)

-- Define the problem statement as a theorem
theorem triangle_perimeter : ∃ x : ℝ, valid_triangle x ∧ (a + b + x = 22) :=
by {
  -- Placeholder for the proof
  sorry
}

end triangle_perimeter_l589_589279


namespace sqrt_x_y_squared_opposite_sign_l589_589999

theorem sqrt_x_y_squared_opposite_sign
  (x y : ℝ) 
  (h1 : sqrt (x - 1) ≥ 0) 
  (h2 : | 2 * x + y - 6 | ≥ 0)
  (h3 : (sqrt (x - 1) = 0 ∨ | 2 * x + y - 6| = 0))
  (h4 : 0 ≤ sqrt ((x + y) ^ 2))
  (h5 : 0 ≤ (x + y) ^ 2):
  (sqrt ((x + y) ^ 2) = 5 ∨ sqrt ((x + y) ^ 2) = -5) := sorry

end sqrt_x_y_squared_opposite_sign_l589_589999


namespace prime_divides_power_l589_589383

theorem prime_divides_power (a b p : ℕ) (hprim : prime p) (p_odd : p % 2 = 1) (hpositive_a : 0 < a) (hpositive_b : 0 < b) (hp_divides : p ∣ a^b - 1) :
  let d := Nat.gcd b (p-1)
  in Nat.v_p (a^b - 1) p = Nat.v_p (a^d - 1) p + Nat.v_p b p :=
by
  sorry

end prime_divides_power_l589_589383


namespace ratio_surface_area_cylinder_to_sphere_l589_589318

noncomputable def surface_area_cylinder (r : ℝ) : ℝ := 2 * π * r * r + 2 * π * r * 2 * r
noncomputable def surface_area_sphere (r : ℝ) : ℝ := 4 * π * r * r

theorem ratio_surface_area_cylinder_to_sphere
  (r : ℝ)
  (h_cylinder : ∃ h : ℝ, h = 2 * r) :
  surface_area_cylinder r / surface_area_sphere r = 3 / 2 := by
  let r := 1
  have surface_area_cylinder_val : surface_area_cylinder r = 6 * π := by sorry
  have surface_area_sphere_val : surface_area_sphere r = 4 * π := by sorry
  rw [surface_area_cylinder_val, surface_area_sphere_val]
  norm_num
  sorry

end ratio_surface_area_cylinder_to_sphere_l589_589318


namespace total_heartbeats_during_race_l589_589796

-- Definitions for the conditions
def heart_beats_per_minute : ℕ := 160
def pace_in_minutes_per_mile : ℕ := 6
def total_distance_in_miles : ℕ := 30

-- Main theorem statement
theorem total_heartbeats_during_race : 
  heart_beats_per_minute * pace_in_minutes_per_mile * total_distance_in_miles = 28800 :=
by
  -- Place the proof here
  sorry

end total_heartbeats_during_race_l589_589796


namespace circle_through_points_l589_589591

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589591


namespace cube_planes_diff_l589_589224

theorem cube_planes_diff (Q : Type) [cube Q] (p : ℕ → Type) [plane_intersects_cube p Q] :
  (max_k : ℕ) - (min_k : ℕ) = 18 :=
by
  -- The proof is omitted here
  sorry

end cube_planes_diff_l589_589224


namespace square_area_of_chords_l589_589116

noncomputable def area_of_square (R : ℝ) : ℝ :=
  R^2 * (2 + Real.sqrt 2)

theorem square_area_of_chords (R : ℝ) :
  ∀ (A B C D E F G H : geometric_point),
    -- Define the circle of radius R
    circle_of_radius R A B C D E F G H →
    -- Chords AF, BE, CH, and DG form a square
    chords_form_square A F B E C H D G →
    -- Prove the area of the square
    area_of_square R = R^2 * (2 + Real.sqrt 2) :=
by
  intros
  sorry

end square_area_of_chords_l589_589116


namespace circle_through_points_l589_589589

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589589


namespace factorize_expression_l589_589884

theorem factorize_expression (a : ℝ) : 
  (a + 1) * (a + 2) + 1 / 4 = (a + 3 / 2)^2 := 
by 
  sorry

end factorize_expression_l589_589884


namespace value_of_m_solve_system_relationship_x_y_l589_589738

-- Part 1: Prove the value of m is 1
theorem value_of_m (x : ℝ) (m : ℝ) (h1 : 2 - x = x + 4) (h2 : m * (1 - x) = x + 3) : m = 1 := sorry

-- Part 2: Solve the system of equations given m = 1
theorem solve_system (x y : ℝ) (h1 : 3 * x + 2 * 1 = - y) (h2 : 2 * x + 2 * y = 1 - 1) : x = -1 ∧ y = 1 := sorry

-- Part 3: Relationship between x and y regardless of m
theorem relationship_x_y (x y m : ℝ) (h1 : 3 * x + y = -2 * m) (h2 : 2 * x + 2 * y = m - 1) : 7 * x + 5 * y = -2 := sorry

end value_of_m_solve_system_relationship_x_y_l589_589738


namespace circle_through_points_l589_589423

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589423


namespace circle_through_points_l589_589594

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589594


namespace triangle_area_l589_589157

theorem triangle_area (x y : ℝ) (h1 : 3 * x + y = 9) (h2 : x = 0 ∨ y = 0) : 
  let base := 3
  let height := 9
  base * height / 2 = 13.5 :=
by
  let base := 3
  let height := 9
  have h_base : base = 3 := rfl
  have h_height : height = 9 := rfl
  calc
    base * height / 2 = 3 * 9 / 2 : by rw [h_base, h_height]
                    ... = 27 / 2   : by norm_num
                    ... = 13.5     : by norm_num

end triangle_area_l589_589157


namespace triangle_area_l589_589164

noncomputable def line_eq := λ x : ℝ, 9 - 3 * x

theorem triangle_area :
    let x_intercept := (3 : ℝ)
    let y_intercept := (9 : ℝ)
    let triangle_base := x_intercept
    let triangle_height := y_intercept
    let area := (1 / 2) * triangle_base * triangle_height
    area = 13.5 :=
by
    sorry

end triangle_area_l589_589164


namespace count_numbers_seven_times_sum_of_digits_l589_589992

open Nat

-- Function to calculate sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  ((n.digits 10).sum)

theorem count_numbers_seven_times_sum_of_digits :
  { n : ℕ // n > 0 ∧ n < 1000 ∧ (n = 7 * sum_of_digits n) }.card = 4 :=
by
  -- Proof would go here
  sorry

end count_numbers_seven_times_sum_of_digits_l589_589992


namespace circle_equation_l589_589620

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589620


namespace circle_through_points_l589_589596

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589596


namespace rearranged_rows_preserve_order_l589_589764

theorem rearranged_rows_preserve_order {n : ℕ} (a b : Fin n → ℝ)
    (h : ∀ i, a i > b i) :
    let a' := Finset.univ.val.sort (· ≤ ·)
        b' := Finset.univ.val.sort (λ i j, b i ≤ b j)
    in ∀ i, a' i > b' i :=
sorry

end rearranged_rows_preserve_order_l589_589764


namespace length_of_crease_l589_589772

theorem length_of_crease (AF FD θ : ℝ) (h₀ : AF + FD = 16) (h₁ : AF / FD = 1 / 2) : 
  let L := (16 / 3) * Real.csc θ in 
  L = (16 / 3) * Real.csc θ :=
by
  sorry

end length_of_crease_l589_589772


namespace elizabeth_stickers_l589_589875

def total_stickers (initial_bottles lost_bottles stolen_bottles stickers_per_bottle : ℕ) : ℕ :=
  let remaining_bottles := initial_bottles - lost_bottles - stolen_bottles
  remaining_bottles * stickers_per_bottle

theorem elizabeth_stickers :
  total_stickers 10 2 1 3 = 21 :=
by
  sorry

end elizabeth_stickers_l589_589875


namespace power_of_five_modulo_eighteen_l589_589112

theorem power_of_five_modulo_eighteen (x : ℕ) : (5^4) % 18 = 13 :=
by 
  have h : 5^4 = 625, by norm_num,
  rw h,
  calc 
    625 % 18 = 625 - 34 * 18 : by norm_num
    ...        = 13 : by norm_num

end power_of_five_modulo_eighteen_l589_589112


namespace area_of_triangle_l589_589183

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the points of intercepts with axes
def intercepts (x0 y0 x1 y1 : ℝ) : Prop :=
  line_eq 0 y0 ∧ y0 = 9 ∧
  line_eq x1 0 ∧ x1 = 3

-- Define the area of the triangle
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem area_of_triangle : ∃ (x0 y0 x1 y1 : ℝ), 
  intercepts x0 y0 x1 y1 ∧ triangle_area x1 y0 = 13.5 :=
by
  sorry

end area_of_triangle_l589_589183


namespace vector_magnitude_l589_589973

variable {V : Type*} [InnerProductSpace ℝ V]

theorem vector_magnitude (a b : V) (ha : ∥a + 2 • b∥ = 2 * Real.sqrt 3)
  (hb : ∥b∥ = 1) (θ : real.angle) (hθ : θ = real.pi / 3) :
  ∥a∥ = 2 := by
  sorry

end vector_magnitude_l589_589973


namespace area_of_triangle_l589_589178

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the points of intercepts with axes
def intercepts (x0 y0 x1 y1 : ℝ) : Prop :=
  line_eq 0 y0 ∧ y0 = 9 ∧
  line_eq x1 0 ∧ x1 = 3

-- Define the area of the triangle
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem area_of_triangle : ∃ (x0 y0 x1 y1 : ℝ), 
  intercepts x0 y0 x1 y1 ∧ triangle_area x1 y0 = 13.5 :=
by
  sorry

end area_of_triangle_l589_589178


namespace circle_passing_through_points_l589_589501

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589501


namespace circle_passing_through_points_l589_589604

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589604


namespace stock_return_to_original_l589_589777

theorem stock_return_to_original (x : ℝ) : 
  let price_2006 := x
  let price_end_2006 := 1.30 * price_2006
  let price_end_2007 := 1.20 * price_end_2006
  let p := (0.56 * 100 / 1.56)
  1.56 * (1 - p/100) = 1 :=
by
  intro x
  let price_2006 := x
  let price_end_2006 := 1.30 * price_2006
  let price_end_2007 := 1.20 * price_end_2006
  let p := (0.56 * 100 / 1.56)
  show 1.56 * (1 - p / 100) = 1
  sorry

end stock_return_to_original_l589_589777


namespace find_circle_equation_l589_589547

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589547


namespace equation_of_circle_passing_through_points_l589_589663

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589663


namespace original_rectangle_perimeter_l589_589133

theorem original_rectangle_perimeter (l w : ℝ) (h1 : w = l / 2)
  (h2 : 2 * (w + l / 3) = 40) : 2 * l + 2 * w = 72 :=
by
  sorry

end original_rectangle_perimeter_l589_589133


namespace RandomEvent_Proof_l589_589725

-- Define the events and conditions
def EventA : Prop := ∀ (θ₁ θ₂ θ₃ : ℝ), θ₁ + θ₂ + θ₃ = 360 → False
def EventB : Prop := ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 6) → n < 7
def EventC : Prop := ∃ (factors : ℕ → ℝ), (∃ (uncertainty : ℕ → ℝ), True)
def EventD : Prop := ∀ (balls : ℕ), (balls = 0 ∨ balls ≠ 0) → False

-- The theorem represents the proof problem
theorem RandomEvent_Proof : EventC :=
by
  sorry

end RandomEvent_Proof_l589_589725


namespace circle_equation_correct_l589_589532

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589532


namespace max_points_on_line_through_circles_l589_589913

theorem max_points_on_line_through_circles (C₁ C₂ C₃ C₄ : Circle) (h_coplanar : coplanar {C₁, C₂, C₃, C₄}) : 
  ∃ max_points : ℕ, max_points = 8 := 
sorry

end max_points_on_line_through_circles_l589_589913


namespace ratio_of_segments_l589_589328

theorem ratio_of_segments (a b c r s : ℝ) (h : a / b = 1 / 4)
  (h₁ : c ^ 2 = a ^ 2 + b ^ 2)
  (h₂ : r = a ^ 2 / c)
  (h₃ : s = b ^ 2 / c) :
  r / s = 1 / 16 :=
by
  sorry

end ratio_of_segments_l589_589328


namespace smallest_circle_equation_l589_589894

theorem smallest_circle_equation :
  ∃ (a : ℝ), a > 0 ∧ (y = 3/a) ∧ (r = (|3 * a + (12 / a) + 3|) / sqrt (3^2 + 4^2)) ∧
  (3 * x + 4 * y + 3 = 0) →
    (x - 2)^2 + (y - 3 / 2)^2 = 9 :=
by
  have h := sorry,
  exact h

end smallest_circle_equation_l589_589894


namespace proof_of_true_propositions_count_l589_589988

noncomputable def count_true_propositions : ℕ := 
  let prop1 := ∀ (m n : ℝ), (|m| > |n|) → (m^2 > n^2)
  let prop2 := ∀ (m n : ℝ), (m^2 > n^2) → (|m| > |n|)
  let prop3 := ∀ (m n : ℝ), (|m| ≤ |n|) → (m^2 ≤ n^2)
  let prop4 := ∀ (m n : ℝ), (m^2 ≤ n^2) → (|m| ≤ |n|)
  if prop1 ∧ prop2 ∧ prop3 ∧ prop4 then 4 else 0

theorem proof_of_true_propositions_count : count_true_propositions = 4 := 
by sorry

end proof_of_true_propositions_count_l589_589988


namespace cos_seventh_expansion_coefficients_squared_sum_l589_589063

theorem cos_seventh_expansion_coefficients_squared_sum :
  ∃ (b_1 b_2 b_3 b_4 b_5 b_6 b_7 : ℝ),
  (∀ (θ : ℝ), 
    cos θ ^ 7 = b_1 * cos θ + b_2 * cos (2 * θ) + b_3 * cos (3 * θ) + b_4 * cos (4 * θ) +
                 b_5 * cos (5 * θ) + b_6 * cos (6 * θ) + b_7 * cos (7 * θ)) ∧
  b_1^2 + b_2^2 + b_3^2 + b_4^2 + b_5^2 + b_6^2 + b_7^2 = 1716 / 4096 :=
sorry

end cos_seventh_expansion_coefficients_squared_sum_l589_589063


namespace circle_passing_through_points_l589_589508

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589508


namespace train_time_pole_l589_589142

theorem train_time_pole (v T : ℝ) :
  let L := 120 in
  (L + 120 = v * 22) →
  (L = v * T) →
  T = 11 :=
by
  intros L h1 h2
  sorry

end train_time_pole_l589_589142


namespace circle_through_points_l589_589588

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589588


namespace evaluate_expression_l589_589230

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 :=
by
  sorry

end evaluate_expression_l589_589230


namespace find_circle_equation_l589_589556

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589556


namespace area_in_terms_of_diagonal_l589_589054

variables (l w d : ℝ)

-- Given conditions
def length_to_width_ratio := l / w = 5 / 2
def diagonal_relation := d^2 = l^2 + w^2

-- Proving the area is kd^2 with k = 10 / 29
theorem area_in_terms_of_diagonal 
    (ratio : length_to_width_ratio l w)
    (diag_rel : diagonal_relation l w d) :
  ∃ k, k = 10 / 29 ∧ (l * w = k * d^2) :=
sorry

end area_in_terms_of_diagonal_l589_589054


namespace hannah_cuts_enough_strands_l589_589981

theorem hannah_cuts_enough_strands (
    strands : ℕ,
    rate_hannah : ℕ,
    rate_son : ℕ,
    total_strands : ℕ,
    combined_rate : ℕ,
    cutting_time : ℕ
  )
  (hannah_rate : rate_hannah = 8)
  (son_rate : rate_son = 3)
  (total_strands_condition : total_strands = 22)
  (combined_rate_condition : combined_rate = rate_hannah + rate_son)
  (time_condition : cutting_time = total_strands / combined_rate) :
  cutting_time = 2 :=
by sorry

end hannah_cuts_enough_strands_l589_589981


namespace equation_of_circle_through_three_points_l589_589573

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589573


namespace distance_travelled_l589_589753

noncomputable def velocity (t : ℝ) : ℝ := t^2 - t + 6

theorem distance_travelled :
  (∫ t in 1..4, velocity t) = 31.5 :=
by sorry

end distance_travelled_l589_589753


namespace cube_assembly_possible_l589_589858

def side_ratios (a b c : ℕ) := a = 8 ∧ b = 8 ∧ c = 27

theorem cube_assembly_possible (x : ℕ) :
  ∃ (parts : list (ℕ × ℕ × ℕ)),
    side_ratios 8 8 27 →
    (∀ p ∈ parts, p.1 * p.2 * p.3 = 8 * 8 * 27 * x^3 / 4) ∧
    (∃ y, ∀ p ∈ parts, p = (y, y, y)) :=
sorry

end cube_assembly_possible_l589_589858


namespace ratio_problem_l589_589314

open Real

theorem ratio_problem 
  (A B C : ℚ)
  (h1 : A / B = 5 / 29)
  (h2 : C / A = 11 / 20) :
  A / B / C = 10 / 29 / 6 := 
begin
  sorry
end

end ratio_problem_l589_589314


namespace quadratic_inequality_solution_l589_589964

theorem quadratic_inequality_solution (
  a b c : ℝ,
  h_condition : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0 → (x ≤ -2 ∨ x ≥ 3)
) :
  (a > 0) ∧ 
  (∀ x : ℝ, (b * x + c > 0) → x < -6) ∧
  (a + b + c < 0) :=
by
  -- start
  sorry

end quadratic_inequality_solution_l589_589964


namespace circle_equation_through_points_l589_589652

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589652


namespace complex_matrix_entry_l589_589347

variables {a b c d : ℂ}

def N : Matrix (Fin 4) (Fin 4) ℂ :=
  ![![a, b, c, d],
    ![b, c, d, a],
    ![c, d, a, b],
    ![d, a, b, c]]

theorem complex_matrix_entry (h1 : N * N = 1) (h2 : a * b * c * d = 1) : (a ^ 4 + b ^ 4 + c ^ 4 + d ^ 4 = 2) := sorry

end complex_matrix_entry_l589_589347


namespace max_value_fn_l589_589034

theorem max_value_fn : ∀ x : ℝ, y = 1 / (|x| + 2) → 
  ∃ y : ℝ, y = 1 / 2 ∧ ∀ x : ℝ, 1 / (|x| + 2) ≤ y :=
sorry

end max_value_fn_l589_589034


namespace circle_equation_correct_l589_589529

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589529


namespace range_of_m_chord_length_l589_589963

-- Given conditions
def eq_line (x y : ℝ) : Prop := √3 * x - y + 1 = 0
def eq_circle (x y m : ℝ) : Prop := x^2 + y^2 - 2*m*x - 2*y + m + 3 = 0

-- Problem (I)
theorem range_of_m (m : ℝ) : (∃ (x y : ℝ), eq_circle x y m) → m < -1 ∨ m > 2 :=
sorry

-- Problem (II)
theorem chord_length (x y : ℝ) (m : ℝ) (d : ℝ) :
  m = -2 ∧ eq_circle x y m → 
  (∃ z, eq_line (-2 : ℝ) z ∧ eq_line 1 z) ∧ 
  d = | -2 * √3 - 1 + 1 | / √(3 + 1) ∧ d < 2 ∧ 2 √(4 - d^2) = 2 := 
sorry

end range_of_m_chord_length_l589_589963


namespace RandomEvent_Proof_l589_589724

-- Define the events and conditions
def EventA : Prop := ∀ (θ₁ θ₂ θ₃ : ℝ), θ₁ + θ₂ + θ₃ = 360 → False
def EventB : Prop := ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 6) → n < 7
def EventC : Prop := ∃ (factors : ℕ → ℝ), (∃ (uncertainty : ℕ → ℝ), True)
def EventD : Prop := ∀ (balls : ℕ), (balls = 0 ∨ balls ≠ 0) → False

-- The theorem represents the proof problem
theorem RandomEvent_Proof : EventC :=
by
  sorry

end RandomEvent_Proof_l589_589724


namespace number_of_zero_points_l589_589686

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(x-1) + x else -1 + Real.log x

theorem number_of_zero_points : ∃ x1 x2 : ℝ, (f x1 = 0 ∧ f x2 = 0) ∧ x1 ≠ x2 :=
by {
  sorry
}

end number_of_zero_points_l589_589686


namespace circle_passing_three_points_l589_589404

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589404


namespace original_price_is_975_l589_589731

variable (x : ℝ)
variable (discounted_price : ℝ := 780)
variable (discount : ℝ := 0.20)

-- The condition that Smith bought the shirt for Rs. 780 after a 20% discount
def original_price_calculation (x : ℝ) (discounted_price : ℝ) (discount : ℝ) : Prop :=
  (1 - discount) * x = discounted_price

theorem original_price_is_975 : ∃ x : ℝ, original_price_calculation x 780 0.20 ∧ x = 975 := 
by
  -- Proof will be provided here
  sorry

end original_price_is_975_l589_589731


namespace limit_at_2_of_fraction_is_7_over_4_l589_589901

theorem limit_at_2_of_fraction_is_7_over_4 :
  tendsto (λ x : ℝ, (x^3 - 1) / (x^2 - 4)) (𝓝 2) (𝓝 (7 / 4)) :=
sorry

end limit_at_2_of_fraction_is_7_over_4_l589_589901


namespace circle_passing_through_points_eqn_l589_589476

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589476


namespace polynomial_condition_l589_589889

variable {R : Type*} [CommRing R] [IsDomain R]
variable (a b : R) (P : R → R)

theorem polynomial_condition (P : R → R)
  (h : ∀ x, x ≠ 0 → 
    P x + P (1 / x) = (P (x + 1 / x) + P (x - 1 / x)) / 2) :
  (∃ a b : R, ∀ x, P x = a * x^4 + b * x^2 + 6 * a) :=
sorry

end polynomial_condition_l589_589889


namespace limit_log_expression_tends_to_neg_infty_l589_589862

open Real

noncomputable def limit_log_expression : Prop :=
  let f (x : ℝ) := log (8*x - 10) / log 4 - log (3*x^2 + x + 2) / log 4 in
  tendsto f at_top (𝓝 (-∞))

theorem limit_log_expression_tends_to_neg_infty : limit_log_expression :=
sorry

end limit_log_expression_tends_to_neg_infty_l589_589862


namespace heartbeats_during_race_l589_589805

theorem heartbeats_during_race
  (heart_rate : ℕ) -- beats per minute
  (pace : ℕ) -- minutes per mile
  (distance : ℕ) -- miles
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) : 
  heart_rate * (pace * distance) = 28800 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end heartbeats_during_race_l589_589805


namespace tan_theta_neq_2sqrt2_l589_589998

theorem tan_theta_neq_2sqrt2 (θ : ℝ) (h₀ : 0 < θ ∧ θ < Real.pi) (h₁ : Real.sin θ + Real.cos θ = (2 * Real.sqrt 2 - 1) / 3) : Real.tan θ = -2 * Real.sqrt 2 := by
  sorry

end tan_theta_neq_2sqrt2_l589_589998


namespace triangle_area_l589_589161

theorem triangle_area (x y : ℝ) (h1 : 3 * x + y = 9) (h2 : x = 0 ∨ y = 0) : 
  let base := 3
  let height := 9
  base * height / 2 = 13.5 :=
by
  let base := 3
  let height := 9
  have h_base : base = 3 := rfl
  have h_height : height = 9 := rfl
  calc
    base * height / 2 = 3 * 9 / 2 : by rw [h_base, h_height]
                    ... = 27 / 2   : by norm_num
                    ... = 13.5     : by norm_num

end triangle_area_l589_589161


namespace radius_range_condition_l589_589951

theorem radius_range_condition {r : ℝ} :
  (∃ x y : ℝ, (x - 3)^2 + (y + 5)^2 = r^2 ∧ abs((4 * x - 3 * y - 17) / sqrt(4^2 + (-3)^2)) = 1) 
  → 1 < r ∧ r < 3 :=
by
  sorry

end radius_range_condition_l589_589951


namespace trig_identity_simplify_l589_589017

theorem trig_identity_simplify (x y : ℝ) :
  cos (x + y) * sin x - sin (x + y) * cos x = -sin x :=
by
  sorry

end trig_identity_simplify_l589_589017


namespace rectangle_width_l589_589706

theorem rectangle_width (L W : ℕ)
  (h1 : W = L + 3)
  (h2 : 2 * L + 2 * W = 54) :
  W = 15 :=
by
  sorry

end rectangle_width_l589_589706


namespace length_of_equal_pieces_l589_589746

theorem length_of_equal_pieces (total_length : ℕ) (num_pieces : ℕ) (num_unequal_pieces : ℕ) (unequal_piece_length : ℕ)
    (equal_pieces : ℕ) (equal_piece_length : ℕ) :
    total_length = 11650 ∧ num_pieces = 154 ∧ num_unequal_pieces = 4 ∧ unequal_piece_length = 100 ∧ equal_pieces = 150 →
    equal_piece_length = 75 :=
by
  sorry

end length_of_equal_pieces_l589_589746


namespace find_circle_equation_l589_589558

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589558


namespace find_circle_equation_l589_589552

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589552


namespace cutting_time_is_2_minutes_l589_589980

variables (total_strands : ℕ) (hannah_rate : ℕ) (son_rate : ℕ)

def combined_rate (hannah_rate : ℕ) (son_rate : ℕ) : ℕ :=
  hannah_rate + son_rate

def time_to_cut_all_strands (total_strands : ℕ) (combined_rate : ℕ) : ℕ :=
  total_strands / combined_rate

theorem cutting_time_is_2_minutes (htotal_strands : total_strands = 22) 
                                   (hhannah_rate : hannah_rate = 8) 
                                   (hson_rate : son_rate = 3) : 
  time_to_cut_all_strands total_strands (combined_rate hannah_rate son_rate) = 2 :=
by
  rw [htotal_strands, hhannah_rate, hson_rate]
  show time_to_cut_all_strands 22 (combined_rate 8 3) = 2
  rw combined_rate -- resolves to 11
  show time_to_cut_all_strands 22 11 = 2
  rw time_to_cut_all_strands -- resolves to 22 / 11
  exact rfl

end cutting_time_is_2_minutes_l589_589980


namespace triangle_area_l589_589165

noncomputable def line_eq := λ x : ℝ, 9 - 3 * x

theorem triangle_area :
    let x_intercept := (3 : ℝ)
    let y_intercept := (9 : ℝ)
    let triangle_base := x_intercept
    let triangle_height := y_intercept
    let area := (1 / 2) * triangle_base * triangle_height
    area = 13.5 :=
by
    sorry

end triangle_area_l589_589165


namespace problem_1_problem_2_problem_3_l589_589009

def students : Type := {a : String // a ∈ ["A", "B", "C", "D", "E", "F", "G", "H", "I"]}

def male_students : set students := {⟨"A", sorry⟩, ⟨"B", sorry⟩, ⟨"C", sorry⟩, ⟨"D", sorry⟩}
def female_students : set students := {⟨"E", sorry⟩, ⟨"F", sorry⟩, ⟨"G", sorry⟩, ⟨"H", sorry⟩, ⟨"I", sorry⟩}

-- 1: Prove number of selection methods for 2 males and 2 females is 1440
theorem problem_1 : 
  let male_combinations := nat.choose 4 2,
      female_combinations := nat.choose 5 2,
      total_selections := male_combinations * female_combinations * nat.factorial 4
  in total_selections = 1440 := sorry

-- 2: Prove number of selection methods with at least 1 male and 1 female is 2880
theorem problem_2 : 
  let one_male_three_females := nat.choose 4 1 * nat.choose 5 3,
      two_males_two_females := nat.choose 4 2 * nat.choose 5 2,
      three_males_one_female := nat.choose 4 3 * nat.choose 5 1,
      total_selections := (one_male_three_females + two_males_two_females + three_males_one_female) * nat.factorial 4
  in total_selections = 2880 := sorry

-- 3: Prove number of selection methods where A and B cannot be selected together is 2376
theorem problem_3 :
  let undesired_combinations := nat.choose 3 2 + nat.choose 4 1 * nat.choose 3 1 + nat.choose 4 2,
      total_combinations := 2880 - undesired_combinations * nat.factorial 4
  in total_combinations = 2376 := sorry

end problem_1_problem_2_problem_3_l589_589009


namespace athlete_heartbeats_calculation_l589_589790

theorem athlete_heartbeats_calculation :
  ∀ (heart_rate : ℕ) (distance : ℕ) (pace : ℕ),
  heart_rate = 160 →
  distance = 30 →
  pace = 6 →
  (pace * distance * heart_rate = 28800)
:= by
  intros heart_rate distance pace hr_eq dis_eq pace_eq
  rw [hr_eq, dis_eq, pace_eq]
  norm_num
  sorry

end athlete_heartbeats_calculation_l589_589790


namespace vector_projection_correct_l589_589977

open scoped Real  -- To work with real numbers

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, 4)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
def projection (u v : ℝ × ℝ) : ℝ := dot_product u v / magnitude v

theorem vector_projection_correct :
  projection a b = Real.sqrt 5 :=
by
  sorry

end vector_projection_correct_l589_589977


namespace surface_area_of_given_cube_l589_589759

-- Define the cube with its volume
def volume_of_cube : ℝ := 4913

-- Define the side length of the cube
def side_of_cube : ℝ := volume_of_cube^(1/3)

-- Define the surface area of the cube
def surface_area_of_cube (side : ℝ) : ℝ := 6 * (side^2)

-- Statement of the theorem
theorem surface_area_of_given_cube : 
  surface_area_of_cube side_of_cube = 1734 := 
by
  -- Proof goes here
  sorry

end surface_area_of_given_cube_l589_589759


namespace part_1_part_2_part_3_l589_589285

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (2 * Real.exp x) / (Real.exp x + 1) + k

theorem part_1 (k : ℝ) :
  (∀ x, f x k = -f (-x) k) → k = -1 :=
sorry

theorem part_2 (m : ℝ) :
  (∀ x > 0, (2 * Real.exp x - 1) / (Real.exp x + 1) ≤ m * (Real.exp x - 1) / (Real.exp x + 1)) → 2 ≤ m :=
sorry

noncomputable def g (x : ℝ) : ℝ := (f x (-1) + 1) / (1 - f x (-1))

theorem part_3 (n : ℝ) :
  (∀ a b c : ℝ, 0 < a ∧ a ≤ n → 0 < b ∧ b ≤ n → 0 < c ∧ c ≤ n → (a + b > c ∧ b + c > a ∧ c + a > b) →
   (g a + g b > g c ∧ g b + g c > g a ∧ g c + g a > g b)) → n = 2 * Real.log 2 :=
sorry

end part_1_part_2_part_3_l589_589285


namespace athlete_heartbeats_calculation_l589_589787

theorem athlete_heartbeats_calculation :
  ∀ (heart_rate : ℕ) (distance : ℕ) (pace : ℕ),
  heart_rate = 160 →
  distance = 30 →
  pace = 6 →
  (pace * distance * heart_rate = 28800)
:= by
  intros heart_rate distance pace hr_eq dis_eq pace_eq
  rw [hr_eq, dis_eq, pace_eq]
  norm_num
  sorry

end athlete_heartbeats_calculation_l589_589787


namespace rearranged_rows_preserve_order_l589_589761

def rearrange_condition (front_row back_row : List ℝ) : Prop :=
  ∀ i, i < front_row.length → front_row.nthLe i sorry < back_row.nthLe i sorry

theorem rearranged_rows_preserve_order (front_row back_row : List ℝ) (h_len : front_row.length = back_row.length)
  (h_condition : rearrange_condition front_row back_row) :
  rearrange_condition (front_row.qsort (· < ·)) (back_row.qsort (· < ·)) := 
sorry

end rearranged_rows_preserve_order_l589_589761


namespace time_between_ticks_at_6_l589_589197

def intervals_12 := 11
def ticks_12 := 12
def seconds_12 := 77
def intervals_6 := 5
def ticks_6 := 6

theorem time_between_ticks_at_6 :
  let interval_time := seconds_12 / intervals_12
  let total_time_6 := intervals_6 * interval_time
  total_time_6 = 35 := sorry

end time_between_ticks_at_6_l589_589197


namespace circle_equation_correct_l589_589541

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589541


namespace total_number_of_subjects_l589_589839

-- Definitions from conditions
def average_marks_5_subjects (total_marks : ℕ) : Prop :=
  74 * 5 = total_marks

def marks_in_last_subject (marks : ℕ) : Prop :=
  marks = 74

def total_average_marks (n : ℕ) (total_marks : ℕ) : Prop :=
  74 * n = total_marks

-- Lean 4 statement
theorem total_number_of_subjects (n total_marks total_marks_5 last_subject_marks : ℕ)
  (h1 : total_average_marks n total_marks)
  (h2 : average_marks_5_subjects total_marks_5)
  (h3 : marks_in_last_subject last_subject_marks)
  (h4 : total_marks = total_marks_5 + last_subject_marks) :
  n = 6 :=
sorry

end total_number_of_subjects_l589_589839


namespace parabola_focus_distance_is_two_l589_589945

noncomputable def parabola_focus_distance : ℝ :=
  let M := (1 : ℝ, 2 : ℝ)
  let p := 2
  let F := (1 : ℝ, 0 : ℝ)
  in real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2)

theorem parabola_focus_distance_is_two {p : ℝ} (hp : 0 < p) (hM : (1 : ℝ, 2 : ℝ) ∈ { p // ∃ x y, y^2 = 2 * p * x }) :
  parabola_focus_distance = 2 :=
sorry

end parabola_focus_distance_is_two_l589_589945


namespace athlete_heartbeats_calculation_l589_589788

theorem athlete_heartbeats_calculation :
  ∀ (heart_rate : ℕ) (distance : ℕ) (pace : ℕ),
  heart_rate = 160 →
  distance = 30 →
  pace = 6 →
  (pace * distance * heart_rate = 28800)
:= by
  intros heart_rate distance pace hr_eq dis_eq pace_eq
  rw [hr_eq, dis_eq, pace_eq]
  norm_num
  sorry

end athlete_heartbeats_calculation_l589_589788


namespace athlete_heartbeats_during_race_l589_589834

theorem athlete_heartbeats_during_race :
  ∀ (heartbeats_per_minute pace_minutes_per_mile race_miles : ℕ ),
    heartbeats_per_minute = 160 →
    pace_minutes_per_mile = 6 →
    race_miles = 30 →
    heartbeats_per_minute * pace_minutes_per_mile * race_miles = 28800 :=
by
  intros heartbeats_per_minute pace_minutes_per_mile race_miles
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end athlete_heartbeats_during_race_l589_589834


namespace total_heartbeats_correct_l589_589817

-- Define the given conditions
def heartbeats_per_minute : ℕ := 160
def pace_per_mile : ℕ := 6
def race_distance : ℕ := 30

-- Define the total heartbeats during the race
def total_heartbeats_during_race : ℕ :=
  pace_per_mile * race_distance * heartbeats_per_minute

-- Theorem stating the mathematically equivalent proof problem
theorem total_heartbeats_correct :
  total_heartbeats_during_race = 28800 :=
by
  -- Proof is omitted
  sorry

end total_heartbeats_correct_l589_589817


namespace probability_a_not_lose_eq_two_thirds_l589_589109

-- Define the gestures
inductive Gesture
| rock
| paper
| scissors

open Gesture

-- Define the result of a game from player A's perspective
inductive Result
| win
| draw
| lose

open Result

-- Define the outcome function for player A against player B
def outcome (a b : Gesture) : Result :=
  match a, b with
  | rock, scissors => win
  | scissors, paper => win
  | paper, rock => win
  | rock, rock => draw
  | scissors, scissors => draw
  | paper, paper => draw
  | rock, paper => lose
  | scissors, rock => lose
  | paper, scissors => lose

-- Define all possible outcomes for two players
def all_outcomes : List (Gesture × Gesture) :=
  [ (rock, rock), (rock, scissors), (rock, paper)
  , (scissors, rock), (scissors, scissors), (scissors, paper)
  , (paper, rock), (paper, scissors), (paper, paper) ]

-- Compute the probability player A does not lose
def prob_a_not_lose : ℚ :=
  let count_not_lose : ℕ := all_outcomes.filter (λ (p : Gesture × Gesture), 
                                                let (a, b) := p in 
                                                outcome a b ≠ lose).length
  let total_possible : ℕ := all_outcomes.length
  count_not_lose / total_possible

-- The theorem statement
theorem probability_a_not_lose_eq_two_thirds : prob_a_not_lose = 2 / 3 := by
  sorry

end probability_a_not_lose_eq_two_thirds_l589_589109


namespace athlete_heartbeats_l589_589808

theorem athlete_heartbeats
  (heart_beats_per_minute : ℕ)
  (pace_minutes_per_mile : ℕ)
  (total_miles : ℕ)
  (heart_beats_per_minute_eq : heart_beats_per_minute = 160)
  (pace_minutes_per_mile_eq : pace_minutes_per_mile = 6)
  (total_miles_eq : total_miles = 30) :
  heart_beats_per_minute * pace_minutes_per_mile * total_miles = 28800 :=
by {
  have h1 : heart_beats_per_minute = 160 := heart_beats_per_minute_eq,
  have h2 : pace_minutes_per_mile = 6 := pace_minutes_per_mile_eq,
  have h3 : total_miles = 30 := total_miles_eq,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end athlete_heartbeats_l589_589808


namespace equation_of_circle_ABC_l589_589451

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589451


namespace circle_passing_through_points_l589_589505

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589505


namespace circle_passing_through_points_eqn_l589_589484

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589484


namespace circle_through_points_l589_589428

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589428


namespace largest_four_digit_sum_19_l589_589079

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digits_sum_to (n : ℕ) (sum : ℕ) : Prop := (n.digits 10).sum = sum

theorem largest_four_digit_sum_19 : ∃ n : ℕ, is_four_digit n ∧ digits_sum_to n 19 ∧ ∀ m : ℕ, is_four_digit m ∧ digits_sum_to m 19 → n ≥ m :=
by
  use 9730
  split
  · exact sorry
  · split
    · exact sorry
    · intros m hm
      exact sorry

end largest_four_digit_sum_19_l589_589079


namespace convert_base_8_to_7_l589_589856

def convert_base_8_to_10 (n : Nat) : Nat :=
  let d2 := n / 100 % 10
  let d1 := n / 10 % 10
  let d0 := n % 10
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

def convert_base_10_to_7 (n : Nat) : List Nat :=
  if n = 0 then [0]
  else 
    let rec helper (n : Nat) (acc : List Nat) : List Nat :=
      if n = 0 then acc
      else helper (n / 7) ((n % 7) :: acc)
    helper n []

def represent_in_base_7 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem convert_base_8_to_7 :
  represent_in_base_7 (convert_base_10_to_7 (convert_base_8_to_10 653)) = 1150 :=
by
  sorry

end convert_base_8_to_7_l589_589856


namespace final_calculation_l589_589688

-- Definitions
def v : ℕ := 1
def w : ℕ := 2
def x : ℕ := 3
def y : ℕ := 5
def z : ℕ := 4

-- The given condition
lemma hw_condition : v + 1 / (w + 1 / (x + 1 / (y + 1 / z))) = 222 / 155 := 
by
  have h : 1 + 1 / (2 + 1 / (3 + 1 / (5 + 1 / 4:ℝ))) = 222 / 155 := by norm_num
  exact h

-- The statement to prove
theorem final_calculation : 10^4 * v + 10^3 * w + 10^2 * x + 10 * y + z = 12354 :=
by
  calc
    10^4 * 1 + 10^3 * 2 + 10^2 * 3 + 10 * 5 + 4 = 10000 + 2000 + 300 + 50 + 4 : by norm_num
    ... = 12354 : by norm_num

end final_calculation_l589_589688


namespace four_element_subset_sum_even_l589_589994

theorem four_element_subset_sum_even :
  let s := {91, 96, 101, 134, 167, 172}
  let subsets := { c | c ⊆ s ∧ ∃ a b c d, c = {a, b, c, d} ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ (91 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 172) }
  let even_sum_subsets := { c ∈ subsets | (c.sum % 2 = 0) }
  even_sum_subsets.card = 9 :=
begin
  sorry
end

end four_element_subset_sum_even_l589_589994


namespace athlete_heartbeats_calculation_l589_589789

theorem athlete_heartbeats_calculation :
  ∀ (heart_rate : ℕ) (distance : ℕ) (pace : ℕ),
  heart_rate = 160 →
  distance = 30 →
  pace = 6 →
  (pace * distance * heart_rate = 28800)
:= by
  intros heart_rate distance pace hr_eq dis_eq pace_eq
  rw [hr_eq, dis_eq, pace_eq]
  norm_num
  sorry

end athlete_heartbeats_calculation_l589_589789


namespace area_of_triangle_l589_589286

noncomputable def f (x : ℝ) : ℝ := e^(x-1) - log x + 1

theorem area_of_triangle (x : ℝ) (e : ℝ) (h0 : e = Real.exp 1) (h1 : x = 1) :
  let y := f x;
    let slope := e^(x-1) - (1/x);
    let tangent_line (x: ℝ) := slope * (x - 1) + y;
    let x_intercept := - tangent_line(0) / slope;
    let y_intercept := tangent_line 0;
    let area := (1/2) * (abs x_intercept) * (abs y_intercept)
  in area = 2 / (e-1) :=
sorry

end area_of_triangle_l589_589286


namespace triangle_area_l589_589160

theorem triangle_area (x y : ℝ) (h1 : 3 * x + y = 9) (h2 : x = 0 ∨ y = 0) : 
  let base := 3
  let height := 9
  base * height / 2 = 13.5 :=
by
  let base := 3
  let height := 9
  have h_base : base = 3 := rfl
  have h_height : height = 9 := rfl
  calc
    base * height / 2 = 3 * 9 / 2 : by rw [h_base, h_height]
                    ... = 27 / 2   : by norm_num
                    ... = 13.5     : by norm_num

end triangle_area_l589_589160


namespace exam_scores_l589_589245

theorem exam_scores (A B C D : ℤ) 
  (h1 : A + B = C + D + 17) 
  (h2 : A = B - 4) 
  (h3 : C = D + 5) :
  ∃ highest lowest, (highest - lowest = 13) ∧ 
                   (highest = A ∨ highest = B ∨ highest = C ∨ highest = D) ∧ 
                   (lowest = A ∨ lowest = B ∨ lowest = C ∨ lowest = D) :=
by
  sorry

end exam_scores_l589_589245


namespace athlete_heartbeats_calculation_l589_589792

theorem athlete_heartbeats_calculation :
  ∀ (heart_rate : ℕ) (distance : ℕ) (pace : ℕ),
  heart_rate = 160 →
  distance = 30 →
  pace = 6 →
  (pace * distance * heart_rate = 28800)
:= by
  intros heart_rate distance pace hr_eq dis_eq pace_eq
  rw [hr_eq, dis_eq, pace_eq]
  norm_num
  sorry

end athlete_heartbeats_calculation_l589_589792


namespace probability_f_ge_0_l589_589959

noncomputable def f (x : ℝ) := log x / log 2

def interval := {x : ℝ | x ∈ Icc (1 / 2) 2}

theorem probability_f_ge_0 : 
  let I := interval
  let prob := (| {x : I | f x ≥ 0} | : ℤ) / (|I| : ℤ)
  in prob = 2 / 3 := by
sorry

end probability_f_ge_0_l589_589959


namespace circle_passes_through_points_l589_589511

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589511


namespace number_of_subsets_A_star_B_is_four_l589_589860

def A : Set Nat := {1, 2, 3, 4, 5}
def B : Set Nat := {2, 4, 5}
def A_star_B : Set Nat := {x ∈ A | ¬ (x ∈ B)}

theorem number_of_subsets_A_star_B_is_four :
  {s : Set (Set Nat) | ∀ x ∈ s, x ⊆ A_star_B}.card = 4 :=
by
  sorry

end number_of_subsets_A_star_B_is_four_l589_589860


namespace compare_abc_l589_589940

noncomputable def a : ℝ := 3^0.2
noncomputable def b : ℝ := 0.3^2
noncomputable def c : ℝ := Real.log 2 / Real.log 0.3

theorem compare_abc : c < b ∧ b < a :=
by
  have h_a : a = 3^0.2 := rfl
  have h_b : b = 0.3^2 := rfl
  have h_c : c = Real.log 2 / Real.log 0.3 := rfl
  sorry

end compare_abc_l589_589940


namespace circle_equation_through_points_l589_589636

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589636


namespace circle_passing_through_points_l589_589616

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589616


namespace find_angles_l589_589234

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  2 * y = x + z

theorem find_angles (a : ℝ) (h1 : 0 < a) (h2 : a < 360)
  (h3 : is_arithmetic_sequence (Real.sin a) (Real.sin (2 * a)) (Real.sin (3 * a))) :
  a = 90 ∨ a = 270 := by
  sorry

end find_angles_l589_589234


namespace cylinder_new_volume_proof_l589_589053

noncomputable def cylinder_new_volume (V : ℝ) (r h : ℝ) : ℝ := 
  let new_r := 3 * r
  let new_h := h / 2
  π * (new_r^2) * new_h

theorem cylinder_new_volume_proof (r h : ℝ) (π : ℝ) 
  (h_volume : π * r^2 * h = 15) : 
  cylinder_new_volume (π * r^2 * h) r h = 67.5 := 
by 
  unfold cylinder_new_volume
  rw [←h_volume]
  sorry

end cylinder_new_volume_proof_l589_589053


namespace eccentricity_is_one_fifth_l589_589401

-- Define the basic structure and conditions of the ellipse
variables (a b c : ℝ)
variables (h : a > b) (k : b > 0)
variables (h_eq : ∀ x y: ℝ, (x/a)^2 + (y/b)^2 = 1)

-- Define vertices, foci, and point D
def vertex_A := (a, 0)
def vertex_D := (0, b)
def focus_1 := (-c, 0)
def focus_2 := (c, 0)

-- Define the vector equation condition
def vector_condition : Prop :=
  (3 * (c, b) = (a, -b) + 2 * (-c, b))

-- Define eccentricity and state the theorem
def eccentricity := c / a

theorem eccentricity_is_one_fifth (hvc: vector_condition) : eccentricity a c = 1 / 5 := 
by { sorry }

end eccentricity_is_one_fifth_l589_589401


namespace houston_to_dallas_bus_passes_Dallas_bound_buses_l589_589201

theorem houston_to_dallas_bus_passes_Dallas_bound_buses :
  ∀ (time_interval_dallas_to_houston : ℕ) (time_interval_houston_to_dallas : ℕ) (trip_duration : ℕ),
  time_interval_dallas_to_houston = 40 ∧ time_interval_houston_to_dallas = 60 ∧ trip_duration = 6 → 
  count_dallas_bound_buses_passed time_interval_dallas_to_houston time_interval_houston_to_dallas trip_duration = 10 :=
by
  intros time_interval_dallas_to_houston time_interval_houston_to_dallas trip_duration h
  sorry

end houston_to_dallas_bus_passes_Dallas_bound_buses_l589_589201


namespace smallest_nat_with_12_divisors_largest_prime_101_ends_in_zero_l589_589237

theorem smallest_nat_with_12_divisors_largest_prime_101_ends_in_zero :
  ∃ n : ℕ, (∀ k : ℕ, (k < n ∧ (∀ d : ℕ, d ∈ (List.range k).filter (λ x => k % x = 0) → d ∣ k → list.length (List.range k).filter (λ x => k % x = 0) = 12 ∧
  ∃ p : ℕ, p.prime ∧ p ≤ 101 ∧ (∀ q : ℕ, q.prime ∧ q ≤ 101 → q ∣ k) ∧ (k % 10 = 0))) → k = n) ∧
  n = 2020 :=
by
  sorry

end smallest_nat_with_12_divisors_largest_prime_101_ends_in_zero_l589_589237


namespace bela_always_wins_l589_589324

noncomputable def game_winner (n : ℕ) (h : n > 10) : Prop :=
∀ strategy_jenn, ∃ strategy_bela, bela_wins (strategy_bela strategy_jenn)

theorem bela_always_wins (n : ℕ) (h : n > 10) : game_winner n h :=
by
  sorry

end bela_always_wins_l589_589324


namespace elizabeth_stickers_l589_589876

def initial_bottles := 10
def lost_at_school := 2
def lost_at_practice := 1
def stickers_per_bottle := 3

def total_remaining_bottles := initial_bottles - lost_at_school - lost_at_practice
def total_stickers := total_remaining_bottles * stickers_per_bottle

theorem elizabeth_stickers : total_stickers = 21 :=
  by
  unfold total_stickers total_remaining_bottles initial_bottles lost_at_school lost_at_practice stickers_per_bottle
  simp
  sorry

end elizabeth_stickers_l589_589876


namespace circle_through_points_l589_589462

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589462


namespace cistern_leak_empty_time_l589_589096

theorem cistern_leak_empty_time :
  (∀ (fill_rate : ℝ) (leak_rate : ℝ) (combined_rate : ℝ),
   fill_rate = 1 / 14 →
   combined_rate = 1 / 16 →
   (combined_rate = fill_rate - leak_rate) →
   leak_rate = 1 / 112) →
  (∀ (leak_rate : ℝ), leak_rate = 1 / 112 → 112 = 1 / leak_rate) :=
begin
  intros fill_rate leak_rate combined_rate fill_rate_eq combined_rate_eq combined_eq,
  rw fill_rate_eq at combined_eq,
  rw combined_rate_eq at combined_eq,
  linarith,
  sorry
end

end cistern_leak_empty_time_l589_589096


namespace circle_equation_l589_589618

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l589_589618


namespace total_heartbeats_correct_l589_589815

-- Define the given conditions
def heartbeats_per_minute : ℕ := 160
def pace_per_mile : ℕ := 6
def race_distance : ℕ := 30

-- Define the total heartbeats during the race
def total_heartbeats_during_race : ℕ :=
  pace_per_mile * race_distance * heartbeats_per_minute

-- Theorem stating the mathematically equivalent proof problem
theorem total_heartbeats_correct :
  total_heartbeats_during_race = 28800 :=
by
  -- Proof is omitted
  sorry

end total_heartbeats_correct_l589_589815


namespace sum_b_a1_a2_a3_a4_eq_60_l589_589277

def a_n (n : ℕ) : ℕ := n + 2
def b_n (n : ℕ) : ℕ := 2^(n-1)

theorem sum_b_a1_a2_a3_a4_eq_60 :
  b_n (a_n 1) + b_n (a_n 2) + b_n (a_n 3) + b_n (a_n 4) = 60 :=
by
  sorry

end sum_b_a1_a2_a3_a4_eq_60_l589_589277


namespace circle_equation_through_points_l589_589649

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589649


namespace circle_passing_through_points_l589_589509

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589509


namespace part1_part2_l589_589954

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := Real.log x - a * x^2 - b * x

noncomputable def g (x : ℝ) (a b : ℝ) : ℝ := f x a b + a * x^2 

theorem part1 (x : ℝ) (h1 : a = 3) (h2 : b = -5) :
  (0 < x ∧ x < 1 → deriv (λ x, f x a b) x > 0) ∧ (1 < x → deriv (λ x, f x a b) x < 0) :=
sorry

theorem part2 (x1 x2 : ℝ) (h1 : x1 ≠ x2) (hx1 : g x1 a b = 0) (hx2 : g x2 a b = 0) :
  Real.log x1 + Real.log x2 > 2 :=
sorry

end part1_part2_l589_589954


namespace triangle_area_l589_589147

theorem triangle_area (x y : ℝ) (h : 3 * x + y = 9) : 
  (1 / 2) * 3 * 9 = 13.5 :=
sorry

end triangle_area_l589_589147


namespace circle_passing_through_points_l589_589494

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589494


namespace new_volume_is_correct_l589_589044

noncomputable def original_volume : ℝ := 15
def original_radius (r : ℝ) : ℝ := r
def original_height (h : ℝ) : ℝ := h
def new_radius (r : ℝ) : ℝ := 3 * r
def new_height (h : ℝ) : ℝ := h / 2

theorem new_volume_is_correct (r h : ℝ) (V := π * r^2 * h) (V' := π * (3 * r)^2 * (h / 2)) :
  V = original_volume →
  V' = (9 / 2) * V →
  V' = 67.5 :=
by
  intros hVr hV'_eq
  have hV : V = 15 := hVr
  have hV_15 : V' = (9 / 2) * 15 := calc
    V' = (9 / 2) * V     := hV'_eq
    ... = (9 / 2) * 15 = 67.5
  exact eq.trans hV_15 rfl

end new_volume_is_correct_l589_589044


namespace total_cost_is_correct_l589_589978

def gravel_cost_per_cubic_foot : ℝ := 8
def discount_rate : ℝ := 0.10
def volume_in_cubic_yards : ℝ := 8
def conversion_factor : ℝ := 27

-- The initial cost for the given volume of gravel in cubic feet
noncomputable def initial_cost : ℝ := gravel_cost_per_cubic_foot * (volume_in_cubic_yards * conversion_factor)

-- The discount amount
noncomputable def discount_amount : ℝ := initial_cost * discount_rate

-- Total cost after applying discount
noncomputable def total_cost_after_discount : ℝ := initial_cost - discount_amount

theorem total_cost_is_correct : total_cost_after_discount = 1555.20 :=
sorry

end total_cost_is_correct_l589_589978


namespace athlete_heartbeats_l589_589813

theorem athlete_heartbeats
  (heart_beats_per_minute : ℕ)
  (pace_minutes_per_mile : ℕ)
  (total_miles : ℕ)
  (heart_beats_per_minute_eq : heart_beats_per_minute = 160)
  (pace_minutes_per_mile_eq : pace_minutes_per_mile = 6)
  (total_miles_eq : total_miles = 30) :
  heart_beats_per_minute * pace_minutes_per_mile * total_miles = 28800 :=
by {
  have h1 : heart_beats_per_minute = 160 := heart_beats_per_minute_eq,
  have h2 : pace_minutes_per_mile = 6 := pace_minutes_per_mile_eq,
  have h3 : total_miles = 30 := total_miles_eq,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end athlete_heartbeats_l589_589813


namespace frog_probability_vertical_side_l589_589123

-- Definition of initial frog position and grid dimensions
def frog_initial_position := (2, 3)
def grid_bottom_left := (0, 0)
def grid_top_left := (0, 5)
def grid_top_right := (6, 5)
def grid_bottom_right := (6, 0)

-- Definition of grid boundaries (vertical sides)
def is_on_vertical_side (x y : ℕ) : Prop :=
  x = 0 ∨ x = 6

-- Probability that frog ends on vertical side given initial position and grid restrictions
def P (x y : ℕ) : ℚ := sorry

theorem frog_probability_vertical_side :
  P 2 3 = 2 / 3 := sorry

end frog_probability_vertical_side_l589_589123


namespace sum_of_odd_numbers_l589_589847

theorem sum_of_odd_numbers (n : ℕ) : 
  (∑ k in Finset.range n, 2 * k + 1) = n ^ 2 := 
by
  sorry

end sum_of_odd_numbers_l589_589847


namespace circle_through_points_l589_589590

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l589_589590


namespace new_volume_is_correct_l589_589042

noncomputable def original_volume : ℝ := 15
def original_radius (r : ℝ) : ℝ := r
def original_height (h : ℝ) : ℝ := h
def new_radius (r : ℝ) : ℝ := 3 * r
def new_height (h : ℝ) : ℝ := h / 2

theorem new_volume_is_correct (r h : ℝ) (V := π * r^2 * h) (V' := π * (3 * r)^2 * (h / 2)) :
  V = original_volume →
  V' = (9 / 2) * V →
  V' = 67.5 :=
by
  intros hVr hV'_eq
  have hV : V = 15 := hVr
  have hV_15 : V' = (9 / 2) * 15 := calc
    V' = (9 / 2) * V     := hV'_eq
    ... = (9 / 2) * 15 = 67.5
  exact eq.trans hV_15 rfl

end new_volume_is_correct_l589_589042


namespace symmetric_points_l589_589934

variable (a b : ℝ)

def condition_1 := a - 1 = 2
def condition_2 := 5 = -(b - 1)

theorem symmetric_points (h1 : condition_1 a) (h2 : condition_2 b) :
  (a + b) ^ 2023 = -1 := 
by
  sorry

end symmetric_points_l589_589934


namespace daily_harvest_sacks_l589_589672

theorem daily_harvest_sacks (sacks_per_section : ℕ) (num_sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 65 → num_sections = 12 → total_sacks = sacks_per_section * num_sections → total_sacks = 780 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end daily_harvest_sacks_l589_589672


namespace minimum_n_constant_term_l589_589309

open Real

noncomputable def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, ∃ (m : ℝ) (hn : n = 5 * r / 3), 
    (x^((1 : ℚ)/2 * n - (5 : ℚ)/6 * r) = x^0)

theorem minimum_n_constant_term :
  (∃ n : ℕ, has_constant_term n) -> (∃ n : ℕ, has_constant_term n ∧ n = 5) :=
by
  sorry

end minimum_n_constant_term_l589_589309


namespace total_heartbeats_correct_l589_589816

-- Define the given conditions
def heartbeats_per_minute : ℕ := 160
def pace_per_mile : ℕ := 6
def race_distance : ℕ := 30

-- Define the total heartbeats during the race
def total_heartbeats_during_race : ℕ :=
  pace_per_mile * race_distance * heartbeats_per_minute

-- Theorem stating the mathematically equivalent proof problem
theorem total_heartbeats_correct :
  total_heartbeats_during_race = 28800 :=
by
  -- Proof is omitted
  sorry

end total_heartbeats_correct_l589_589816


namespace cards_valid_arrangements_l589_589012

theorem cards_valid_arrangements : 
  (∀ (cards : Finset ℕ), 
    (cards = {1, 2, 3, 4, 5, 6, 7}) →
    (∀ (pos7 : Fin 7 → Bool), 
      (∀ (remove_card : Fin 7),
        ∃ (remaining_cards : Finset ℕ), 
          (remaining_cards ⊆ cards \ {7}) ∧ 
          ((remaining_cards.card = 5 ∧ 
           (remaining_cards = {a, b, c, d, e} ∧ 
             a < b ∧ b < c ∧ c < d ∧ d < e) ∨ 
           (remaining_cards = {e, d, c, b, a} ∧ 
             a < b ∧ b < c ∧ c < d ∧ d < e))) ∧ 
        cardSym (pos7) remove_card = 2880)))

end cards_valid_arrangements_l589_589012


namespace decrement_from_observation_l589_589035

theorem decrement_from_observation 
  (n : ℕ) (mean_original mean_updated : ℚ)
  (h1 : n = 50)
  (h2 : mean_original = 200)
  (h3 : mean_updated = 194)
  : (mean_original - mean_updated) = 6 :=
by
  sorry

end decrement_from_observation_l589_589035


namespace triangle_ends_on_right_face_l589_589139

/-- Prove that after rolling a square with a marked solid triangle (initially on the left face)
around a regular octagon, the triangle ends up on the right face after four complete sides of rolling. -/
theorem triangle_ends_on_right_face :
  ∀ square : Type, ∀ octagon : Type, initial_triangle_pos : ℝ,
  travels_around_octagon : ℝ, 
  initial_triangle_pos = -1 → (* -1 means left face *)
  travels_around_octagon = 4 →
  final_triangle_pos : ℝ :=
by
  -- Assuming initially the triangle is on the left face; the final position is the right face after rolling
  sorry

end triangle_ends_on_right_face_l589_589139


namespace find_angle_EFG_l589_589918

variable (x y : ℝ)

-- Given: Parallel lines AD || FG and the angle measures
axiom AD_parallel_FG : ∃ (AD FG : ℝ), AD = FG
axiom angle_CEA : ∃ (x y : ℝ), ∠CEA = x + 3y
axiom angle_CFG : ∃ (x : ℝ), ∠CFG = 2x
axiom angle_BAE : ∃ (y : ℝ), ∠BAE = y
axiom angle_EFG : ∃ (x : ℝ), ∠EFG = 2x  -- This is reaffirming the condition trickly

theorem find_angle_EFG : (AD_parallel_FG → angle_CEA x y → angle_CFG x → angle_BAE y → angle_EFG x) → 
  ∠EFG = 90 :=
by
  sorry

end find_angle_EFG_l589_589918


namespace price_reduction_achieves_profit_l589_589118

theorem price_reduction_achieves_profit :
  ∃ x : ℝ, (40 - x) * (20 + 2 * (x / 4) * 8) = 1200 ∧ x = 20 :=
by
  sorry

end price_reduction_achieves_profit_l589_589118


namespace integers_between_sqrt3_and_sqrt14_l589_589189

theorem integers_between_sqrt3_and_sqrt14 (x : ℤ) (hx1: x > real.sqrt 3) (hx2: x < real.sqrt 14) : x = 2 ∨ x = 3 :=
sorry

end integers_between_sqrt3_and_sqrt14_l589_589189


namespace power_function_solution_l589_589689

noncomputable def f (x : ℝ) : ℝ := x^(2/3)

theorem power_function_solution :
  (∀ x : ℝ, f (-x) = f x) ∧ (f (-1) < f 2) ∧ (f 2 < 2) :=
by
  -- Conditions
  let f_neg_x_eq_f_x := ∀ x : ℝ, f (-x) = f x
  let f_neg_1_lt_f_2 := f (-1) < f 2
  let f_2_lt_2 := f 2 < 2
  -- Goals to prove
  exact ⟨f_neg_x_eq_f_x, f_neg_1_lt_f_2, f_2_lt_2⟩
  sorry

end power_function_solution_l589_589689


namespace total_weight_of_balls_l589_589004

theorem total_weight_of_balls :
  let weight_blue := 6
  let weight_brown := 3.12
  weight_blue + weight_brown = 9.12 :=
by
  sorry

end total_weight_of_balls_l589_589004


namespace g_values_sum_l589_589364

def g (x : ℝ) : ℝ :=
  if x > 6 then x^2 + 2
  else if x > -6 then 3 * x - 4
  else 4

theorem g_values_sum : g (-7) + g 0 + g 8 = 66 := by
  sorry

end g_values_sum_l589_589364


namespace pirate_captain_age_l589_589130

theorem pirate_captain_age
  (N C : ℤ)
  (H13_digits : N = 1 * 10^12 + (49 * 10^10) + 1271153044)
  (H_control : C = 67)
  (H_mod : (N + C) % 97 = 0)
  (year : ℕ)
  (current_year : year = 2011)
  : year - ((10^12 + 49 * 10^10 + 1271153044) / 10^10 * 73 % 97) = 65 :=
by sorry

-- Definitions based on given conditions
def first_thirteen_digits := 1 * 10^12 + 49 * 10^10 + 1271153044
def control_digits := 67

-- Proof of the given conditions directly as definitions:
-- Proving that the correct interpretation results in the age being 65
assertion : pirate_captain_age first_thirteen_digits control_digits 2011 = 65 :=
by sorry

end pirate_captain_age_l589_589130


namespace sedrach_divides_each_pie_l589_589008

theorem sedrach_divides_each_pie (P : ℕ) :
  (13 * P * 5 = 130) → P = 2 :=
by
  sorry

end sedrach_divides_each_pie_l589_589008


namespace ordering_of_integers_l589_589225

theorem ordering_of_integers (a b c d : ℕ) (h_a : a > 3) (h_b : b > 3) (h_c : c > 3) (h_d : d > 3) 
  (h : 1 / (a - 2) = 1 / (b + 2) ∧ 1 / (b + 2) = 1 / (c + 1) ∧ 1 / (c + 1) = 1 / (d - 3)) :
  b < c ∧ c < a ∧ a < d :=
begin
  sorry
end

end ordering_of_integers_l589_589225


namespace circle_equation_correct_l589_589534

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l589_589534


namespace quadrant_of_negative_angle_l589_589737

def angle (θ : ℝ) : ℕ :=
if 0 ≤ θ ∧ θ < 90 then 1
else if 90 ≤ θ ∧ θ < 180 then 2
else if 180 ≤ θ ∧ θ < 270 then 3
else if 270 ≤ θ ∧ θ < 360 then 4
else sorry -- We define how to handle out of range cases later if necessary

theorem quadrant_of_negative_angle :
  angle ((-510 : ℝ) % 360 + if (-510 : ℝ) % 360 < 0 then 360 else 0) = 3 := by
sorry

end quadrant_of_negative_angle_l589_589737


namespace equation_of_circle_ABC_l589_589450

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589450


namespace evaluate_complex_magnitudes_l589_589227

theorem evaluate_complex_magnitudes :
  3 * Complex.abs (1 - 3 * Complex.i) + 2 * Complex.abs (1 + 3 * Complex.i) = 5 * Real.sqrt 10 :=
by
  sorry

end evaluate_complex_magnitudes_l589_589227


namespace equation_of_circle_through_three_points_l589_589580

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589580


namespace math_problem_l589_589203

-- Define the main variables a and b
def a : ℕ := 312
def b : ℕ := 288

-- State the main theorem to be proved
theorem math_problem : (a^2 - b^2) / 24 + 50 = 650 := 
by 
  sorry

end math_problem_l589_589203


namespace option_D_functions_same_l589_589088

theorem option_D_functions_same (x : ℝ) : (x^2) = (x^6)^(1/3) :=
by 
  sorry

end option_D_functions_same_l589_589088


namespace hannah_cuts_enough_strands_l589_589982

theorem hannah_cuts_enough_strands (
    strands : ℕ,
    rate_hannah : ℕ,
    rate_son : ℕ,
    total_strands : ℕ,
    combined_rate : ℕ,
    cutting_time : ℕ
  )
  (hannah_rate : rate_hannah = 8)
  (son_rate : rate_son = 3)
  (total_strands_condition : total_strands = 22)
  (combined_rate_condition : combined_rate = rate_hannah + rate_son)
  (time_condition : cutting_time = total_strands / combined_rate) :
  cutting_time = 2 :=
by sorry

end hannah_cuts_enough_strands_l589_589982


namespace new_volume_is_correct_l589_589045

noncomputable def original_volume : ℝ := 15
def original_radius (r : ℝ) : ℝ := r
def original_height (h : ℝ) : ℝ := h
def new_radius (r : ℝ) : ℝ := 3 * r
def new_height (h : ℝ) : ℝ := h / 2

theorem new_volume_is_correct (r h : ℝ) (V := π * r^2 * h) (V' := π * (3 * r)^2 * (h / 2)) :
  V = original_volume →
  V' = (9 / 2) * V →
  V' = 67.5 :=
by
  intros hVr hV'_eq
  have hV : V = 15 := hVr
  have hV_15 : V' = (9 / 2) * 15 := calc
    V' = (9 / 2) * V     := hV'_eq
    ... = (9 / 2) * 15 = 67.5
  exact eq.trans hV_15 rfl

end new_volume_is_correct_l589_589045


namespace dist_to_other_focus_l589_589376

-- Definition of the hyperbola
def is_hyperbola (x y : ℝ) : Prop := (x^2 / 1^2 - y^2 / (2*sqrt 2)^2 = 1)

-- Distance to one focus is given as 3
def dist_to_focus_3 (P : ℝ × ℝ) (F1 : ℝ × ℝ) (F2 : ℝ × ℝ) : Prop := 
  let d1 := real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) in
  d1 = 3

-- Determine the distance to the other focus
theorem dist_to_other_focus (P F1 F2: ℝ × ℝ) (hP : is_hyperbola P.1 P.2) (hF : dist_to_focus_3 P F1 F2) :
  let d2 := real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) in
  d2 = 5 := by
    sorry

end dist_to_other_focus_l589_589376


namespace area_of_triangle_l589_589179

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the points of intercepts with axes
def intercepts (x0 y0 x1 y1 : ℝ) : Prop :=
  line_eq 0 y0 ∧ y0 = 9 ∧
  line_eq x1 0 ∧ x1 = 3

-- Define the area of the triangle
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem area_of_triangle : ∃ (x0 y0 x1 y1 : ℝ), 
  intercepts x0 y0 x1 y1 ∧ triangle_area x1 y0 = 13.5 :=
by
  sorry

end area_of_triangle_l589_589179


namespace monica_total_savings_l589_589372

noncomputable def weekly_savings (week: ℕ) : ℕ :=
  if week < 6 then 15 + 5 * week
  else if week < 11 then 40 - 5 * (week - 5)
  else weekly_savings (week % 10)

theorem monica_total_savings : 
  let cycle_savings := (15 + 20 + 25 + 30 + 35 + 40) + (40 + 35 + 30 + 25 + 20 + 15) - 40 
  let total_savings := 5 * cycle_savings
  total_savings = 1450 := by
  sorry

end monica_total_savings_l589_589372


namespace number_of_squares_is_16_l589_589931

def point_in_H (x y : ℤ) : Prop := 2 ≤ |x| ∧ |x| ≤ 8 ∧ 2 ≤ |y| ∧ |y| ≤ 8

def is_square_of_side_length_7 (x1 y1 x2 y2 : ℤ) : Prop :=
  let side_length := (x2 - x1)^2 + (y2 - y1)^2
  side_length = 7^2

-- The main theorem statement proving the number of such squares is exactly 16
theorem number_of_squares_is_16 : 
  ∑ x1 in {i : ℤ | 2 ≤ |i| ∧ |i| ≤ 8},
  ∑ y1 in {i : ℤ | 2 ≤ |i| ∧ |i| ≤ 8},
  ∑ x2 in {i : ℤ | 2 ≤ |i| ∧ |i| ≤ 8},
  ∑ y2 in {i : ℤ | 2 ≤ |i| ∧ |i| ≤ 8},
  (if point_in_H x1 y1 ∧ point_in_H x2 y2 ∧ is_square_of_side_length_7 x1 y1 x2 y2 then 1 else 0) = 16 :=
sorry

end number_of_squares_is_16_l589_589931


namespace touches_excircle_isosceles_right_triangle_l589_589384

open EuclideanGeometry Real

noncomputable def isRightTriangle (A B C : Point) : Prop :=
  ∃ (O : Point),
    isConvex A B C ∧ rightAngle ∡ A B C ∧ dist A B = dist B C

noncomputable def isMidpointCircle (K : Point) (r : ℝ) (A B C : Point) : Circle :=
  { center := K, radius := r }

noncomputable def isExcircle (r : ℝ) (O : Point) : Circle :=
  { center := O, radius := 2 * r }

theorem touches_excircle_isosceles_right_triangle
  (A B C K O : Point) (r : ℝ) [isRightTriangle A B C] :
  touches (isMidpointCircle K r A B C) (isExcircle r O) :=
sorry

end touches_excircle_isosceles_right_triangle_l589_589384


namespace total_fish_correct_l589_589708

-- Define the number of pufferfish
def num_pufferfish : ℕ := 15

-- Define the number of swordfish as 5 times the number of pufferfish
def num_swordfish : ℕ := 5 * num_pufferfish

-- Define the total number of fish as the sum of pufferfish and swordfish
def total_num_fish : ℕ := num_pufferfish + num_swordfish

-- Theorem stating the total number of fish
theorem total_fish_correct : total_num_fish = 90 := by
  -- Proof is omitted
  sorry

end total_fish_correct_l589_589708


namespace total_heartbeats_correct_l589_589819

-- Define the given conditions
def heartbeats_per_minute : ℕ := 160
def pace_per_mile : ℕ := 6
def race_distance : ℕ := 30

-- Define the total heartbeats during the race
def total_heartbeats_during_race : ℕ :=
  pace_per_mile * race_distance * heartbeats_per_minute

-- Theorem stating the mathematically equivalent proof problem
theorem total_heartbeats_correct :
  total_heartbeats_during_race = 28800 :=
by
  -- Proof is omitted
  sorry

end total_heartbeats_correct_l589_589819


namespace prob_13_know_news_prob_14_know_news_expected_know_news_l589_589226

variables (num_scientists knowers : ℕ) (know_news : ℕ → Prop)

-- conditions
def conditions : Prop := num_scientists = 18 ∧ knowers = 10 ∧ (∀ i < num_scientists, know_news i) ∧ num_scientists % 2 = 0

-- Question 1: Probability of exactly 13 scientists knowing the news after the coffee break is 0
theorem prob_13_know_news (h : conditions num_scientists knowers know_news) : 
  probability (λ s, s.countp know_news = 13) = 0 := 
sorry

-- Question 2: Probability of exactly 14 scientists knowing the news after the coffee break is 1120/2431
theorem prob_14_know_news (h : conditions num_scientists knowers know_news) : 
  probability (λ s, s.countp know_news = 14) = 1120 / 2431 := 
sorry

-- Question 3: Expected number of scientists who know the news after the coffee break is approximately 14.7
theorem expected_know_news (h : conditions num_scientists knowers know_news) : 
  E (λ s, s.countp know_news) ≈ 14.7 :=
sorry

end prob_13_know_news_prob_14_know_news_expected_know_news_l589_589226


namespace circle_passing_through_points_eqn_l589_589479

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l589_589479


namespace circle_passing_three_points_l589_589411

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589411


namespace trig_identity_simplify_l589_589016

theorem trig_identity_simplify (x y : ℝ) :
  cos (x + y) * sin x - sin (x + y) * cos x = -sin x :=
by
  sorry

end trig_identity_simplify_l589_589016


namespace train_length_is_450_l589_589778

noncomputable def train_length (L V : ℝ) : Prop :=
  (L = V * 18) ∧ (L + 525 = V * 39)

theorem train_length_is_450 (L V : ℝ) (h : train_length L V) : L = 450 :=
by
  cases h with h1 h2
  sorry

end train_length_is_450_l589_589778


namespace equation_of_circle_ABC_l589_589453

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589453


namespace minimum_value_of_fraction_l589_589936

-- Define the real condition on x and y
def is_point_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the minimum value condition of the given expression
def target_minimum_value (k : ℝ) : Prop :=
  k = 4 / 3

-- The main theorem
theorem minimum_value_of_fraction {x y : ℝ} (h : is_point_on_circle x y) :
  ∃ k : ℝ, target_minimum_value k ∧ k = (y - 4) / (x - 2) :=
begin
  sorry
end

end minimum_value_of_fraction_l589_589936


namespace sum_inferiors_2026_l589_589301

noncomputable def a_n (n : ℕ) : ℝ := Real.log (n + 2) / Real.log (n + 1)

def is_inferior (n : ℕ) : Prop :=
  (∏ i in Finset.range (n + 1), a_n i).isInt

def sum_inferiors (m : ℕ) : ℕ :=
  (Finset.range m).filter is_inferior |>.sum id

theorem sum_inferiors_2026 : sum_inferiors 2016 = 2026 :=
by
  sorry

end sum_inferiors_2026_l589_589301


namespace root_in_interval_l589_589680

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 3

theorem root_in_interval : ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 :=
  sorry

end root_in_interval_l589_589680


namespace athlete_heartbeats_during_race_l589_589827

theorem athlete_heartbeats_during_race
  (heart_rate : ℕ)
  (pace : ℕ)
  (distance : ℕ)
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) :
  heart_rate * pace * distance = 28800 :=
by
  rw [H1, H2, H3]
  norm_num

end athlete_heartbeats_during_race_l589_589827


namespace triangle_area_l589_589174

-- Define the line equation 3x + y = 9
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the triangle bounded by the coordinate axes and the line
def bounded_triangle (x y : ℝ) : Prop :=
  (x = 0 ∧ 0 ≤ y ∧ y ≤ 9) ∨  -- y-axis segment
  (y = 0 ∧ 0 ≤ x ∧ x ≤ 3) ∨  -- x-axis segment
  (3 * x + y = 9 ∧ 0 ≤ x ∧ 0 ≤ y)  -- line segment

-- The area of the triangle formed by the coordinate axes and the line 3x + y = 9
theorem triangle_area : 
  let area := 1/2 * 3 * 9 in -- The calculation of the area
  area = 13.5 := 
by
  sorry

end triangle_area_l589_589174


namespace maximize_integral_l589_589349

noncomputable def integral_sq_condition (a b : ℝ) : Prop :=
  ∫ x in 0..1, (a * x + b)^2 = 1

noncomputable def integral_to_maximize (a b : ℝ) : ℝ :=
  ∫ x in 0..1, 3 * x * (a * x + b)

theorem maximize_integral :
  ∃ a b : ℝ, integral_sq_condition a b ∧
  (∀ a' b' : ℝ, integral_sq_condition a' b' → integral_to_maximize a b ≥ integral_to_maximize a' b')
   ∧ a = real.sqrt 3 ∧ b = 0 :=
sorry

end maximize_integral_l589_589349


namespace circle_passing_three_points_l589_589412

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589412


namespace circle_through_points_l589_589471

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589471


namespace range_of_values_l589_589962

theorem range_of_values (a b : ℝ) : (∀ x : ℝ, x < 1 → ax + b > 2 * (x + 1)) → b > 4 := 
by
  sorry

end range_of_values_l589_589962


namespace slide_all_papers_off_table_l589_589850

-- Define the type for the table.
structure Table where
  is_rectangular : Prop

-- Define the type for a piece of paper.
structure Paper where
  is_convex_polygon : Prop

-- Define the type for the configuration on the table.
structure Configuration where
  table : Table
  papers : Finset Paper
  non_overlapping_interiors : Prop

-- The theorem stating that Chim Tu can slide all pieces of paper off the table in finitely many steps.
theorem slide_all_papers_off_table (cfg : Configuration)
  (finitely_many_papers : cfg.papers.finite)
  (rectangular_table : cfg.table.is_rectangular)
  (convex_polygons : ∀ p ∈ cfg.papers, p.is_convex_polygon)
  (non_overlapping : cfg.non_overlapping_interiors) :
  ∃ n : ℕ, ∀ t : ℕ, t ≥ n → ∀ p ∈ cfg.papers, p.is_off_table_at_time t :=
by
  sorry

end slide_all_papers_off_table_l589_589850


namespace circle_passing_three_points_l589_589408

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589408


namespace concurrency_in_inscribed_triangles_l589_589838

open EuclideanGeometry

theorem concurrency_in_inscribed_triangles 
  {O : Point} {A B C A1 B1 C1 A2 B2 C2 : Point} 
  (h_inscribed: Incircle O A B C)
  (h_tangency: Tangent O A1 (line B C) ∧ Tangent O B1 (line C A) ∧ Tangent O C1 (line A B))
  (h_intersections: Line O A ∩ Circle O = A2 ∧ Line O B ∩ Circle O = B2 ∧ Line O C ∩ Circle O = C2) :
  Concurrent (line A1 A2) (line B1 B2) (line C1 C2) := sorry

end concurrency_in_inscribed_triangles_l589_589838


namespace athlete_heartbeats_during_race_l589_589833

theorem athlete_heartbeats_during_race :
  ∀ (heartbeats_per_minute pace_minutes_per_mile race_miles : ℕ ),
    heartbeats_per_minute = 160 →
    pace_minutes_per_mile = 6 →
    race_miles = 30 →
    heartbeats_per_minute * pace_minutes_per_mile * race_miles = 28800 :=
by
  intros heartbeats_per_minute pace_minutes_per_mile race_miles
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end athlete_heartbeats_during_race_l589_589833


namespace circle_through_points_l589_589468

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l589_589468


namespace regular_polyhedron_equal_properties_regular_polyhedron_rotation_symmetry_regular_polyhedron_topologically_regular_l589_589734

-- Definitions based on the problem statement
structure RegularPolyhedron (P : Type) :=
  (faces : list (set P)) 
  (vertices : set P)
  (edges : list (set P))
  (is_convex : ∀ (f : set P), f ∈ faces → convex f)
  (congruent_faces : ∀ (f1 f2 : set P), f1 ∈ faces → f2 ∈ faces → congruent f1 f2)
  (vertex_transitive : ∀ (v1 v2 : P), v1 ∈ vertices → v2 ∈ vertices → ∃ ϕ : P → P, (isometry ϕ) ∧ (ϕ v1 = v2))
  (edge_transitive : ∀ (e1 e2 : set P), e1 ∈ edges → e2 ∈ edges → ∃ ψ : set P → set P, (isometry ψ) ∧ (ψ e1 = e2))

-- Part (a)
theorem regular_polyhedron_equal_properties
  (T : Type) [RegularPolyhedron T] :
  ∀ (f1 f2 : set T) (e1 e2 : set T) (a1 a2 : angle T) (d1 d2 : angle T) (p1 p2 : angle T),
    (f1 ∈ RegularPolyhedron.faces T) → (f2 ∈ RegularPolyhedron.faces T) → 
    (e1 ∈ RegularPolyhedron.edges T) → (e2 ∈ RegularPolyhedron.edges T) →
    (a1 ∈ planar_angles(T)) → (a2 ∈ planar_angles(T)) →
    (d1 ∈ dihedral_angles(T)) → (d2 ∈ dihedral_angles(T)) →
    (p1 ∈ polyhedral_angles(T)) → (p2 ∈ polyhedral_angles(T)) →
    (f1 = f2) ∧ (e1 = e2) ∧ (a1 = a2) ∧ (d1 = d2) ∧ (p1 = p2) :=
sorry

-- Part (b)
theorem regular_polyhedron_rotation_symmetry
  (T : Type) [RegularPolyhedron T] :
  (∀ (μ1 μ2 : set T) (m1 m2 : set T), μ1 ∈ RegularPolyhedron.faces T → μ2 ∈ RegularPolyhedron.faces T → 
    m1 ∈ RegularPolyhedron.edges T → m2 ∈ RegularPolyhedron.edges T → 
    ∃ ϕ : T → T, (isometry ϕ) ∧ (ϕ μ1 = μ2) ∧ (ϕ m1 = m2)) ↔
  (∀ (μ1 μ2 : set T) (m1 m2 : set T), isometry (RegularPolyhedron.edge_transitive T m1 m2)) :=
sorry

-- Part (c)
theorem regular_polyhedron_topologically_regular
  (T : Type) [RegularPolyhedron T] :
  topological_regular(T) :=
sorry

end regular_polyhedron_equal_properties_regular_polyhedron_rotation_symmetry_regular_polyhedron_topologically_regular_l589_589734


namespace equation_of_circle_passing_through_points_l589_589656

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589656


namespace digit_125_in_decimal_of_4_over_7_l589_589717

theorem digit_125_in_decimal_of_4_over_7 : 
  (decimal_of_fraction_digit_at_pos 125 4 7 = 2) :=
  sorry

def decimal_of_fraction_digit_at_pos (n : ℕ) (a : ℕ) (b : ℕ) : ℕ :=
  let repeating_seq := [5, 7, 1, 4, 2, 8]
  let seq_len := repeating_seq.length
  let pos_in_seq := (n - 1) % seq_len
  repeating_seq.get ⟨pos_in_seq, by simp [seq_len, pos_in_seq] mentioned_length⟩

end digit_125_in_decimal_of_4_over_7_l589_589717


namespace A_is_all_positive_integers_l589_589352

theorem A_is_all_positive_integers (A : Set ℕ) (hA1: ∀ a ∈ A, ∀ d ∈ (Nat.divisors a), d ∈ A) 
(hA2: ∀ a b ∈ A, 1 < a -> a < b -> (1 + a * b) ∈ A) :
  (3 ≤ A.card) → (∀ n : ℕ, 0 < n → n ∈ A) :=
by
  -- Proof omitted
  sorry

end A_is_all_positive_integers_l589_589352


namespace customer_seating_l589_589111

theorem customer_seating (h1 : ∀ (n : ℕ), n = 7 * 3 + 1 * 2)
  (h2 : ∀ (c : ℕ), c ≤ 16)
  (h3 : ∀ (e : bool), e = tt ∨ e = ff) :
  ∃ (seating : Fin 23 → option (bool × ℕ))
    (valid : ∀ (i j : Fin 23), i ≠ j → seating i ≠ seating j)
    (singles : ∀ (i : Fin 23), seating i = some (ff, _) → ((i.val - 3) % 3 = 1))
    (pairs : ∀ (i : Fin 22), (seating i).isSome → (seating (Fin.succ i)).isSome → (seating i).get = (seating (Fin.succ i)).get) :
  true :=
by
  sorry

end customer_seating_l589_589111


namespace simplify_trig_expression_l589_589015

theorem simplify_trig_expression (x : ℝ) :
  (1 + Real.sin x + Real.cos x) / (1 + Real.sin x - Real.cos x) = Real.cot (x / 2) := by
  sorry

end simplify_trig_expression_l589_589015


namespace inradius_semiperimeter_circumradius_relation_l589_589300

theorem inradius_semiperimeter_circumradius_relation
  (A B C A1 B1 C1: Point)
  (ex_bisector_A1C: Line)
  (ex_bisector_B1A: Line)
  (ex_bisector_C1B: Line)
  (proj_A_on_A1C: Point)
  (proj_B_on_B1A: Point)
  (proj_C_on_C1B: Point)
  (d r p: ℝ)
  (circumscribed_circle_A1B1C1: Circle)
  (inradius_triangle_ABC: ℝ)
  (semiperimeter_triangle_ABC: ℝ) :
  (r = inradius_triangle_ABC) →
  (p = semiperimeter_triangle_ABC) →
  (diameter circumscribed_circle_A1B1C1 = d) →
  (r^2 + p^2 = d^2) := 
sorry

end inradius_semiperimeter_circumradius_relation_l589_589300


namespace same_sign_abc_l589_589251
open Classical

theorem same_sign_abc (a b c : ℝ) (h1 : (b / a) * (c / a) > 1) (h2 : (b / a) + (c / a) ≥ -2) : 
  (a > 0 ∧ b > 0 ∧ c > 0) ∨ (a < 0 ∧ b < 0 ∧ c < 0) :=
sorry

end same_sign_abc_l589_589251


namespace complex_sum_of_z_is_neg5_l589_589249

-- Define the complex number z and its condition
def complex_sum_condition (z : ℂ) : Prop :=
  (conj z) / (1 + 2 * complex.i) = 2 + complex.i

-- Prove that the sum of the real part and imaginary part of z is -5 given the condition
theorem complex_sum_of_z_is_neg5 (z : ℂ) (hz : complex_sum_condition z) : z.re + z.im = -5 :=
  sorry

end complex_sum_of_z_is_neg5_l589_589249


namespace find_a2017_l589_589294

variable (a : ℕ → ℤ)
variable h1 : a 1 = 2
variable h2 : ∀ n, a (n + 1) + a n = 2 * n - 1

theorem find_a2017 : a 2017 = 2018 :=
by
  sorry

end find_a2017_l589_589294


namespace equation_of_circle_through_three_points_l589_589576

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589576


namespace equal_piece_length_l589_589741

/-- A 1165 cm long rope is cut into 154 pieces, 150 of which are equally sized, and the remaining pieces are 100mm each.
    This theorem proves that the length of each equally sized piece is 75mm. -/
theorem equal_piece_length (total_length_cm : ℕ) (total_pieces : ℕ) (equal_pieces : ℕ) (remaining_piece_length_mm : ℕ) 
  (total_length_mm : ℕ) (remaining_pieces : ℕ) (equal_length_mm : ℕ) : 
  total_length_cm = 1165 ∧ 
  total_pieces = 154 ∧  
  equal_pieces = 150 ∧
  remaining_piece_length_mm = 100 ∧
  total_length_mm = total_length_cm * 10 ∧
  remaining_pieces = total_pieces - equal_pieces ∧ 
  equal_length_mm = (total_length_mm - remaining_pieces * remaining_piece_length_mm) / equal_pieces →
  equal_length_mm = 75 :=
by
  sorry

end equal_piece_length_l589_589741


namespace aria_spent_on_cookies_l589_589903

def aria_spent : ℕ := 2356

theorem aria_spent_on_cookies :
  (let cookies_per_day := 4
  let cost_per_cookie := 19
  let days_in_march := 31
  let total_cookies := days_in_march * cookies_per_day
  let total_cost := total_cookies * cost_per_cookie
  total_cost = aria_spent) :=
  sorry

end aria_spent_on_cookies_l589_589903


namespace purchase_price_of_article_l589_589691

theorem purchase_price_of_article (P : ℝ) (h : 45 = 0.20 * P + 12) : P = 165 :=
by
  sorry

end purchase_price_of_article_l589_589691


namespace equation_of_circle_passing_through_points_l589_589654

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589654


namespace max_points_on_line_through_circles_l589_589912

theorem max_points_on_line_through_circles (C₁ C₂ C₃ C₄ : Circle) (h_coplanar : coplanar {C₁, C₂, C₃, C₄}) : 
  ∃ max_points : ℕ, max_points = 8 := 
sorry

end max_points_on_line_through_circles_l589_589912


namespace smallest_row_sum_greater_than_50_l589_589900

noncomputable def sum_interior_pascal (n : ℕ) : ℕ :=
  2^(n-1) - 2

theorem smallest_row_sum_greater_than_50 : ∃ n, sum_interior_pascal n > 50 ∧ (∀ m, m < n → sum_interior_pascal m ≤ 50) ∧ sum_interior_pascal 7 = 62 ∧ (sum_interior_pascal 7) % 2 = 0 :=
by
  sorry

end smallest_row_sum_greater_than_50_l589_589900


namespace circle_equation_through_points_l589_589651

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l589_589651


namespace complete_the_square_l589_589086

theorem complete_the_square :
  ∀ (x : ℝ), (x^2 + 14 * x + 24 = 0) → (∃ c d : ℝ, (x + c)^2 = d ∧ d = 25) :=
by
  intro x h
  sorry

end complete_the_square_l589_589086


namespace equation_of_circle_through_three_points_l589_589564

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l589_589564


namespace capacity_of_gunny_bag_l589_589377

/-- Converting weights and calculating the capacity of the gunny bag in tons. -/
theorem capacity_of_gunny_bag :
  ∀ (pounds_per_ton ounces_per_pound packets weight_per_packet_pounds weight_per_packet_ounces : ℕ), 
  pounds_per_ton = 2600 →
  ounces_per_pound = 16 →
  packets = 2080 →
  weight_per_packet_pounds = 16 →
  weight_per_packet_ounces = 4 →
  let weight_per_packet := weight_per_packet_pounds + weight_per_packet_ounces / ounces_per_pound in
  let total_weight_pounds := packets * weight_per_packet in
  total_weight_pounds / pounds_per_ton = 13 :=
by
  intros pounds_per_ton ounces_per_pound packets weight_per_packet_pounds weight_per_packet_ounces
  sorry

end capacity_of_gunny_bag_l589_589377


namespace expected_value_boy_girl_adjacent_pairs_l589_589022

/-- Considering 10 boys and 15 girls lined up in a row, we need to show that
    the expected number of adjacent positions where a boy and a girl stand next to each other is 12. -/
theorem expected_value_boy_girl_adjacent_pairs :
  let boys := 10
  let girls := 15
  let total_people := boys + girls
  let total_adjacent_pairs := total_people - 1
  let p_boy_then_girl := (boys / total_people) * (girls / (total_people - 1))
  let p_girl_then_boy := (girls / total_people) * (boys / (total_people - 1))
  let expected_T := total_adjacent_pairs * (p_boy_then_girl + p_girl_then_boy)
  expected_T = 12 :=
by
  sorry

end expected_value_boy_girl_adjacent_pairs_l589_589022


namespace book_arrangement_l589_589306

theorem book_arrangement : 
  let blocks_permutations := 2!
  let math_books_permutations := 4!
  let english_books_permutations := 4!
  blocks_permutations * math_books_permutations * english_books_permutations = 1152 :=
by
  let blocks_permutations := fact 2
  let math_books_permutations := fact 4
  let english_books_permutations := fact 4
  calc
    (blocks_permutations * math_books_permutations * english_books_permutations)
      = (2 * 24 * 24) : by repeat { rw fact }
    ... = 1152 : by norm_num
  done

end book_arrangement_l589_589306


namespace increasing_interval_l589_589679

open Real

noncomputable def f (x : ℝ) : ℝ := log (1/2) (-x^2 - 2 * x + 3)

theorem increasing_interval : (∀ x ∈ Ioo (-1 : ℝ) (1 : ℝ), -x^2 - 2 * x + 3 > 0) →
  ∃ (a b : ℝ), f(x) increasing_on Ico a b :=
by
  sorry

end increasing_interval_l589_589679


namespace avg_ratio_eq_two_l589_589186

def rectangular_array (a : ℕ → ℕ → ℕ) :=
  (∀ i j, 1 ≤ a i j ∧ a i j ≤ 3) ∧ (∀ i, 1 ≤ i ∧ i ≤ 50) ∧ (∀ j, 1 ≤ j ∧ j ≤ 100)

def row_sum (a : ℕ → ℕ → ℕ) (i : ℕ) : ℕ := ∑ j in finset.range 100, a i j
def col_sum (a : ℕ → ℕ → ℕ) (j : ℕ) : ℕ := ∑ i in finset.range 50, a i j

def avg_row_sum (a : ℕ → ℕ → ℕ) : ℝ := (∑ i in finset.range 50, (row_sum a i : ℝ)) / 50
def avg_col_sum (a : ℕ → ℕ → ℕ) : ℝ := (∑ j in finset.range 100, (col_sum a j : ℝ)) / 100

theorem avg_ratio_eq_two (a : ℕ → ℕ → ℕ) (h : rectangular_array a) :
  avg_row_sum a / avg_col_sum a = 2 :=
by
  sorry

end avg_ratio_eq_two_l589_589186


namespace current_population_correct_l589_589330

def initial_population : ℕ := 4079
def percentage_died : ℕ := 5
def percentage_left : ℕ := 15

def calculate_current_population (initial_population : ℕ) (percentage_died : ℕ) (percentage_left : ℕ) : ℕ :=
  let died := (initial_population * percentage_died) / 100
  let remaining_after_bombardment := initial_population - died
  let left := (remaining_after_bombardment * percentage_left) / 100
  remaining_after_bombardment - left

theorem current_population_correct : calculate_current_population initial_population percentage_died percentage_left = 3295 :=
  by
  unfold calculate_current_population
  sorry

end current_population_correct_l589_589330


namespace triangle_area_bound_by_line_l589_589155

noncomputable def area_of_triangle : ℝ :=
let x_intercept := 3 in
let y_intercept := 9 in
1 / 2 * x_intercept * y_intercept

theorem triangle_area_bound_by_line (x y : ℝ) (h : 3 * x + y = 9) :
  area_of_triangle = 13.5 :=
sorry

end triangle_area_bound_by_line_l589_589155


namespace range_of_m_l589_589997

theorem range_of_m (m : ℝ) (H : ∀ x, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) : m < -1 / 2 :=
sorry

end range_of_m_l589_589997


namespace find_circle_equation_l589_589549

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589549


namespace equation_of_circle_ABC_l589_589452

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589452


namespace count_integers_200_to_250_with_increasing_digits_l589_589984

theorem count_integers_200_to_250_with_increasing_digits :
  let count_valid_integers := 
    (λ (n : ℕ), (200 ≤ n) ∧ (n ≤ 250) ∧ ((∀ (d1 d2 d3 : ℕ), 
      d1 < d2 ∧ d2 < d3 ∧ n = 200*d1 + 10*d2 + d3 ∧ d1 = 2))) in
  (nat.card (set_of count_valid_integers)) = 11 :=
by 
  sorry

end count_integers_200_to_250_with_increasing_digits_l589_589984


namespace elizabeth_stickers_count_l589_589871

theorem elizabeth_stickers_count :
  let initial_bottles := 10
  let lost_at_school := 2
  let stolen_at_dance := 1
  let stickers_per_bottle := 3
  let remaining_bottles := initial_bottles - lost_at_school - stolen_at_dance
  remaining_bottles * stickers_per_bottle = 21 := by sorry

end elizabeth_stickers_count_l589_589871


namespace complex_number_in_first_quadrant_l589_589335

def complex_num := (2 + Complex.i) / 3

theorem complex_number_in_first_quadrant (z : ℂ) (h : z = complex_num) :
    0 < z.re ∧ 0 < z.im :=
by
  sorry

end complex_number_in_first_quadrant_l589_589335


namespace mod_3_power_87_plus_5_l589_589083

theorem mod_3_power_87_plus_5 :
  (3 ^ 87 + 5) % 11 = 3 := 
by
  sorry

end mod_3_power_87_plus_5_l589_589083


namespace bases_with_final_digit_one_540_l589_589244

theorem bases_with_final_digit_one_540 (b : ℕ) : ∃ n, n = 2 ∧ (2 ≤ b ∧ b ≤ 9) ∧ 539 % b = 0 :=
by
  have h1 : ∀ b, b ∈ [2, 3, 4, 5, 6, 7, 8, 9] → 539 % b = 0 ↔ b ∈ [7, 11],
    sorry
  use 2
  split
  - refl
  - split
    -- The base 7 and 11 between the interval [2, 9]
    sorry

end bases_with_final_digit_one_540_l589_589244


namespace triangle_area_l589_589159

theorem triangle_area (x y : ℝ) (h1 : 3 * x + y = 9) (h2 : x = 0 ∨ y = 0) : 
  let base := 3
  let height := 9
  base * height / 2 = 13.5 :=
by
  let base := 3
  let height := 9
  have h_base : base = 3 := rfl
  have h_height : height = 9 := rfl
  calc
    base * height / 2 = 3 * 9 / 2 : by rw [h_base, h_height]
                    ... = 27 / 2   : by norm_num
                    ... = 13.5     : by norm_num

end triangle_area_l589_589159


namespace circle_passes_through_points_l589_589516

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589516


namespace find_circle_equation_l589_589560

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l589_589560


namespace area_of_triangle_l589_589184

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the points of intercepts with axes
def intercepts (x0 y0 x1 y1 : ℝ) : Prop :=
  line_eq 0 y0 ∧ y0 = 9 ∧
  line_eq x1 0 ∧ x1 = 3

-- Define the area of the triangle
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem area_of_triangle : ∃ (x0 y0 x1 y1 : ℝ), 
  intercepts x0 y0 x1 y1 ∧ triangle_area x1 y0 = 13.5 :=
by
  sorry

end area_of_triangle_l589_589184


namespace circle_passing_three_points_l589_589403

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l589_589403


namespace athlete_heartbeats_l589_589809

theorem athlete_heartbeats
  (heart_beats_per_minute : ℕ)
  (pace_minutes_per_mile : ℕ)
  (total_miles : ℕ)
  (heart_beats_per_minute_eq : heart_beats_per_minute = 160)
  (pace_minutes_per_mile_eq : pace_minutes_per_mile = 6)
  (total_miles_eq : total_miles = 30) :
  heart_beats_per_minute * pace_minutes_per_mile * total_miles = 28800 :=
by {
  have h1 : heart_beats_per_minute = 160 := heart_beats_per_minute_eq,
  have h2 : pace_minutes_per_mile = 6 := pace_minutes_per_mile_eq,
  have h3 : total_miles = 30 := total_miles_eq,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end athlete_heartbeats_l589_589809


namespace area_of_triangle_l589_589182

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the points of intercepts with axes
def intercepts (x0 y0 x1 y1 : ℝ) : Prop :=
  line_eq 0 y0 ∧ y0 = 9 ∧
  line_eq x1 0 ∧ x1 = 3

-- Define the area of the triangle
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem area_of_triangle : ∃ (x0 y0 x1 y1 : ℝ), 
  intercepts x0 y0 x1 y1 ∧ triangle_area x1 y0 = 13.5 :=
by
  sorry

end area_of_triangle_l589_589182


namespace max_intersections_of_line_with_four_coplanar_circles_l589_589908

theorem max_intersections_of_line_with_four_coplanar_circles
  (C1 C2 C3 C4 : ℝ) -- Four coplanar circles
  (h1 : is_circle C1)
  (h2 : is_circle C2)
  (h3 : is_circle C3)
  (h4 : is_circle C4)
  : ∃ l : ℝ → ℝ, ∀ C ∈ {C1, C2, C3, C4}, (number_of_intersections l C ≤ 2) → 
    (number_of_intersections l C1 + number_of_intersections l C2 + number_of_intersections l C3 + number_of_intersections l C4) ≤ 8 := 
sorry

end max_intersections_of_line_with_four_coplanar_circles_l589_589908


namespace largest_four_digit_sum_19_l589_589080

theorem largest_four_digit_sum_19 : ∃ n : ℕ, n < 10000 ∧ 1000 ≤ n ∧ (∑ i in (n.digits 10), id i = 19) ∧
  (∀ m : ℕ, m < 10000 ∧ 1000 ≤ m ∧ (∑ i in (m.digits 10), id i = 19) → n ≥ m) ∧ n = 9910 :=
by 
  sorry

end largest_four_digit_sum_19_l589_589080


namespace length_of_PQ_is_correct_l589_589929

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define the angle with y-axis
def angle_with_y_axis : ℝ := Real.pi / 6

-- Define the parabola
def parabola (y : ℝ) : ℝ := y^2 / 4

-- Define the slope of the line
def slope : ℝ := Real.sqrt 3

-- Define the line equation passing through A
def line (x : ℝ) : ℝ := slope * (x - 1)

-- Define the intersection points (operations omitted for brevity)
def intersection_points : set (ℝ × ℝ) := 
  {p | ∃ x y, p = (x, y) ∧ y = line x ∧ x = parabola y}

-- Define the length of |PQ|
noncomputable def length_PQ : ℝ := sorry

-- The proof statement
theorem length_of_PQ_is_correct : length_PQ = 16 / 3 := sorry

end length_of_PQ_is_correct_l589_589929


namespace circle_passes_through_points_l589_589520

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589520


namespace greater_number_l589_589059

theorem greater_number (x y : ℕ) (h_sum : x + y = 50) (h_diff : x - y = 16) : x = 33 :=
by
  sorry

end greater_number_l589_589059


namespace circle_passing_through_points_l589_589602

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589602


namespace probability_of_two_passes_is_half_l589_589122

def simulate_pass_or_fail (n : Nat) : Bool :=
  -- Simulating pass (true) or fail (false) based on given mapping
  n = 0 ∨ (n ≥ 5 ∧ n ≤ 9)

def is_exactly_two_passes (a b c : Nat) : Bool :=
  [simulate_pass_or_fail a, simulate_pass_or_fail b, simulate_pass_or_fail c].count true = 2

def test_results : List (Nat × Nat × Nat) :=
  [(9, 1, 7), (9, 6, 6), (8, 9, 1), (9, 2, 5), (2, 7, 1), (9, 3, 2), (8, 7, 2),
   (4, 5, 8), (5, 6, 9), (6, 8, 3), (4, 3, 1), (2, 5, 7), (3, 9, 3), (0, 2, 7),
   (5, 5, 6), (4, 8, 8), (7, 3, 0), (1, 1, 3), (5, 0, 7), (9, 8, 9)]

def count_exactly_two_passes (results : List (Nat × Nat × Nat)) : Nat :=
  results.count (λ r, is_exactly_two_passes r.1 r.2 r.3)

def total_probability_of_two_passes : Real :=
  (count_exactly_two_passes test_results : Real) / (test_results.length : Real)

theorem probability_of_two_passes_is_half : total_probability_of_two_passes = 0.5 := by
  sorry

end probability_of_two_passes_is_half_l589_589122


namespace circle_passes_through_points_l589_589510

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589510


namespace evaluate_expression_l589_589228

theorem evaluate_expression : (Real.sqrt (Real.sqrt 5 ^ 4))^3 = 125 := by
  sorry

end evaluate_expression_l589_589228


namespace circle_through_points_l589_589434

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589434


namespace trig_identity_l589_589104

theorem trig_identity :
  sin (25 * real.pi / 180) * sin (35 * real.pi / 180) * sin (60 * real.pi / 180) * sin (85 * real.pi / 180) =
  sin (20 * real.pi / 180) * sin (40 * real.pi / 180) * sin (75 * real.pi / 180) * sin (80 * real.pi / 180) :=
sorry

end trig_identity_l589_589104


namespace equation_of_circle_passing_through_points_l589_589659

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l589_589659


namespace equilateral_triangle_perimeters_sum_l589_589835

noncomputable def sum_of_perimeters (initial_side_length : ℝ) (iterations : ℕ) : ℝ :=
  let rec triangle_perimeter (side_length : ℝ) (iters : ℕ) : ℝ :=
    if iters = 0 then 0
    else 3 * side_length + triangle_perimeter (side_length / 3) (iters - 1)
  in triangle_perimeter initial_side_length iterations

theorem equilateral_triangle_perimeters_sum :
  sum_of_perimeters 18 5 = 80 + 2 / 3 :=
by
  sorry

end equilateral_triangle_perimeters_sum_l589_589835


namespace range_of_m_l589_589257

open Real Set

variable (x m : ℝ)

def p (x : ℝ) := (x + 1) * (x - 1) ≤ 0
def q (x m : ℝ) := (x + 1) * (x - (3 * m - 1)) ≤ 0 ∧ m > 0

theorem range_of_m (hpsuffq : ∀ x, p x → q x m) (hqnotsuffp : ∃ x, q x m ∧ ¬ p x) : m > 2 / 3 := by
  sorry

end range_of_m_l589_589257


namespace toy_car_arrangement_l589_589707

-- Given conditions
def number_of_brands : ℕ := 4
def cars_per_brand : ℕ := 2
def number_of_garages : ℕ := 4
def cars_per_garage : ℕ := 2

-- Problem statement
theorem toy_car_arrangement :
  ∃ (ways : ℕ), ways = 72 ∧ 
    (∃ (choose_brands : ℕ), choose_brands = Nat.choose number_of_brands 2) ∧
    (∃ (place_cars : ℕ), place_cars = (Nat.perm number_of_garages number_of_garages) / 2) ∧
    (ways = (choose_brands * place_cars)) := 
  sorry

end toy_car_arrangement_l589_589707


namespace max_at_minus_two_l589_589673

noncomputable def f (x m : ℝ) : ℝ := x * (x - m)^2

theorem max_at_minus_two (m : ℝ) : (∀ x, f'(-2) = (x - m)^2 + 2x(x - m)) → f'(-2) = 0 → f''(-2) < 0 → m = -2 := by
  sorry

-- Definitions for f' and f'' can be added if needed.

end max_at_minus_two_l589_589673


namespace circle_through_points_l589_589437

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589437


namespace elizabeth_stickers_count_l589_589870

theorem elizabeth_stickers_count :
  let initial_bottles := 10
  let lost_at_school := 2
  let stolen_at_dance := 1
  let stickers_per_bottle := 3
  let remaining_bottles := initial_bottles - lost_at_school - stolen_at_dance
  remaining_bottles * stickers_per_bottle = 21 := by sorry

end elizabeth_stickers_count_l589_589870


namespace largest_prime_factor_l589_589092

theorem largest_prime_factor (a b : ℕ) (ha : a = 17) (hb : b = 16) :
  (17^4 + 2 * 17^2 + 1 - 16^4).prime_factors.max = 17 := 
sorry

end largest_prime_factor_l589_589092


namespace largest_power_of_5_l589_589351

def sequence (n : ℕ) : ℕ :=
  Nat.recOn n 2 (λ n x_n, 2 * x_n^3 + x_n)

theorem largest_power_of_5 (n : ℕ) :
  n > 0 → ∃ (k : ℕ), (5^n ∣ (sequence n)^2 + 1) ∧ ¬ (5^(n+1) ∣ (sequence n)^2 + 1) :=
by
  intro h
  sorry

end largest_power_of_5_l589_589351


namespace equation_of_circle_ABC_l589_589448

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l589_589448


namespace total_heartbeats_correct_l589_589814

-- Define the given conditions
def heartbeats_per_minute : ℕ := 160
def pace_per_mile : ℕ := 6
def race_distance : ℕ := 30

-- Define the total heartbeats during the race
def total_heartbeats_during_race : ℕ :=
  pace_per_mile * race_distance * heartbeats_per_minute

-- Theorem stating the mathematically equivalent proof problem
theorem total_heartbeats_correct :
  total_heartbeats_during_race = 28800 :=
by
  -- Proof is omitted
  sorry

end total_heartbeats_correct_l589_589814


namespace solution_set_inequality_range_of_t_l589_589957

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem solution_set_inequality :
  {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 2} :=
sorry

theorem range_of_t (t : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f (x - t) ≤ x - 2) ↔ 3 ≤ t ∧ t ≤ 3 + Real.sqrt 2 :=
sorry

end solution_set_inequality_range_of_t_l589_589957


namespace eric_has_correct_green_marbles_l589_589879

def total_marbles : ℕ := 20
def white_marbles : ℕ := 12
def blue_marbles : ℕ := 6
def green_marbles : ℕ := total_marbles - (white_marbles + blue_marbles)

theorem eric_has_correct_green_marbles : green_marbles = 2 :=
by
  sorry

end eric_has_correct_green_marbles_l589_589879


namespace area_of_fourth_rectangle_l589_589765

theorem area_of_fourth_rectangle
    (x y z w : ℝ)
    (h1 : x * y = 24)
    (h2 : z * y = 15)
    (h3 : z * w = 9) :
    y * w = 15 := 
sorry

end area_of_fourth_rectangle_l589_589765


namespace trajectory_of_point_P_l589_589684

theorem trajectory_of_point_P (x y : ℝ) :
  (∃ A B P : ℝ × ℝ, 
    let A := (x1, y1),
    let B := (x2, y2),
    let P := ((2 * x2 + x1) / 3, (2 * y2 + y1) / 3),
    (y1 = x1) ∧ (y2 = x2) ∧ (x1^2 + y1^4 = 1) ∧ (x2^2 + y2^4 = 1) ∧ (y = x + P.2 - P.1)) →
  148 * x^2 + 13 * y^2 + 64 * x * y - 20 = 0 :=
by 
  sorry

end trajectory_of_point_P_l589_589684


namespace inverse_ratio_l589_589682

theorem inverse_ratio (a b c d : ℝ) :
  (∀ x, x ≠ -6 → (3 * x - 2) / (x + 6) = (a * x + b) / (c * x + d)) →
  a/c = -6 :=
by
  sorry

end inverse_ratio_l589_589682


namespace group_total_cost_l589_589218

noncomputable def total_cost
  (num_people : Nat) 
  (cost_per_person : Nat) : Nat :=
  num_people * cost_per_person

theorem group_total_cost (num_people := 15) (cost_per_person := 900) :
  total_cost num_people cost_per_person = 13500 :=
by
  sorry

end group_total_cost_l589_589218


namespace range_of_m_minimum_value_ab_l589_589011

-- Define the given condition as a predicate on the real numbers
def domain_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0

-- Define the first part of the proof problem: range of m
theorem range_of_m :
  (∀ m : ℝ, domain_condition m) → ∀ m : ℝ, m ≤ 6 :=
sorry

-- Define the second part of the proof problem: minimum value of 4a + 7b
theorem minimum_value_ab (n : ℝ) (a b : ℝ) (h : n = 6) :
  (∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (4 / (a + 5 * b) + 1 / (3 * a + 2 * b) = n)) → 
  ∃ (a b : ℝ), 4 * a + 7 * b = 3 / 2 :=
sorry

end range_of_m_minimum_value_ab_l589_589011


namespace julia_paid_for_puppy_l589_589346

theorem julia_paid_for_puppy :
  let dog_food := 20
  let treat := 2.5
  let treats := 2 * treat
  let toys := 15
  let crate := 20
  let bed := 20
  let collar_leash := 15
  let discount_rate := 0.20
  let total_before_discount := dog_food + treats + toys + crate + bed + collar_leash
  let discount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount
  let total_spent := 96
  total_spent - total_after_discount = 20 := 
by 
  sorry

end julia_paid_for_puppy_l589_589346


namespace equal_piece_length_l589_589743

/-- A 1165 cm long rope is cut into 154 pieces, 150 of which are equally sized, and the remaining pieces are 100mm each.
    This theorem proves that the length of each equally sized piece is 75mm. -/
theorem equal_piece_length (total_length_cm : ℕ) (total_pieces : ℕ) (equal_pieces : ℕ) (remaining_piece_length_mm : ℕ) 
  (total_length_mm : ℕ) (remaining_pieces : ℕ) (equal_length_mm : ℕ) : 
  total_length_cm = 1165 ∧ 
  total_pieces = 154 ∧  
  equal_pieces = 150 ∧
  remaining_piece_length_mm = 100 ∧
  total_length_mm = total_length_cm * 10 ∧
  remaining_pieces = total_pieces - equal_pieces ∧ 
  equal_length_mm = (total_length_mm - remaining_pieces * remaining_piece_length_mm) / equal_pieces →
  equal_length_mm = 75 :=
by
  sorry

end equal_piece_length_l589_589743


namespace minimize_total_time_l589_589329

def exercise_time (s : ℕ → ℕ) : Prop :=
  ∀ i, s i < 45

def total_exercises (a : ℕ → ℕ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 25

def minimize_time (a : ℕ → ℕ) (s : ℕ → ℕ) : Prop :=
  ∃ (j : ℕ), (1 ≤ j ∧ j ≤ 7 ∧ 
  (∀ i, 1 ≤ i ∧ i ≤ 7 → if i = j then a i = 25 else a i = 0) ∧
  ∀ i, 1 ≤ i ∧ i ≤ 7 → s i ≥ s j)

theorem minimize_total_time
  (a : ℕ → ℕ) (s : ℕ → ℕ) 
  (h_exercise_time : exercise_time s)
  (h_total_exercises : total_exercises a) :
  minimize_time a s := by
  sorry

end minimize_total_time_l589_589329


namespace distribution_methods_l589_589864

open Finset

def volunteers : Finset (Fin 5) := univ

def groupings : Finset (Finset (Finset (Fin 5))) :=
  (Finset.powerset volunteers).filter (λ s, s.card = 2)

def count_groupings : ℕ :=
  groupings.card * (groupings.erase univ).card * 1 / (2 * 6)

def venues : Finset (Fin 3) := univ

def permutations : Finset (Finset (Fin 3)) :=
  powersetLen 3 venues

def count_permutations : ℕ := permutations.card

theorem distribution_methods :
  count_groupings * count_permutations = 90 :=
  by
    sorry

end distribution_methods_l589_589864


namespace athlete_heartbeats_during_race_l589_589824

theorem athlete_heartbeats_during_race
  (heart_rate : ℕ)
  (pace : ℕ)
  (distance : ℕ)
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) :
  heart_rate * pace * distance = 28800 :=
by
  rw [H1, H2, H3]
  norm_num

end athlete_heartbeats_during_race_l589_589824


namespace find_ff_one_ninth_l589_589955

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.logBase 3 x else 2 ^ x

theorem find_ff_one_ninth : f(f(1 / 9)) = 1 / 4 :=
  by
  sorry

end find_ff_one_ninth_l589_589955


namespace boat_speed_l589_589752

-- Definitions based on the conditions
def distance_1 : ℝ := 96 -- Distance traveled against the current
def time_1 : ℝ := 8 -- Time taken against the current
def distance_2 : ℝ := 96 -- Distance traveled with the current
def time_2 : ℝ := 5 -- Time taken with the current
def x : ℝ -- Speed of the current

-- The speed of the boat in still water
def boat_speed_in_still_water (b : ℝ) : Prop := 
  (b - x) * time_1 = distance_1 ∧ (b + x) * time_2 = distance_2

-- Prove that the speed of the boat in still water is 15.6 under the given conditions
theorem boat_speed (b : ℝ) : boat_speed_in_still_water b → b = 15.6 := by
  sorry

end boat_speed_l589_589752


namespace common_tangents_l589_589041

noncomputable def radius1 := 8
noncomputable def radius2 := 6
noncomputable def distance := 2

theorem common_tangents (r1 r2 d : ℕ) 
  (h1 : r1 = radius1) 
  (h2 : r2 = radius2) 
  (h3 : d = distance) :
  (d = r1 - r2) → 1 = 1 := by 
  sorry

end common_tangents_l589_589041


namespace distance_segment_l589_589719

def point1 : ℝ × ℝ × ℝ := (3, 4, 0)
def point2 : ℝ × ℝ × ℝ := (8, 8, 6)

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

theorem distance_segment : distance point1 point2 = real.sqrt 77 :=
by
  sorry

end distance_segment_l589_589719


namespace grid_filling_ways_l589_589885

theorem grid_filling_ways :
  ∃! (fill_grid : Matrix (Fin 3) (Fin 3) (Fin 3)), 
    (∀ (i : Fin 3), ∃! (v : Fin 3 → Fin 3), ∀ (j : Fin 3), fill_grid i j = v j) ∧
    (∀ (j : Fin 3), ∃! (v : Fin 3 → Fin 3), ∀ (i : Fin 3), fill_grid i j = v j) ∧
    (∀ (n : Fin 3), ∃ (count : Fin 3 → Fin 3), count 1 + count 2 + count 3 = 3) ∧
    (count_solutions fill_grid = 12) :=
sorry

noncomputable def count_solutions {α β γ : Type*} [DecidableEq α] [DecidableEq β] (fill_grid : α → β → γ) : ℕ := 12

end grid_filling_ways_l589_589885


namespace circle_through_points_l589_589432

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l589_589432


namespace range_of_m_min_value_of_7a_4b_l589_589278

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x - 1| + |x + 1| - m ≥ 0) → m ≤ 2 :=
sorry

theorem min_value_of_7a_4b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
    (h_eq : 2 / (3 * a + b) + 1 / (a + 2 * b) = 2) : 7 * a + 4 * b ≥ 9 / 2 :=
sorry

end range_of_m_min_value_of_7a_4b_l589_589278


namespace ellipse_l589_589268

noncomputable def ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > c) : Prop :=
  ∀ x y : ℝ, (x^2)/(a^2) + (y^2)/(b^2) = 1

noncomputable def find_ellipse (h : ℝ) (m : ℝ) : Prop :=
  let a : ℝ := real.sqrt 2 in
  let c : ℝ := 1 in
  let b : ℝ := real.sqrt (a^2 - c^2) in
  (h = (x^2)/2 + y^2 = 1) ∧ ((x - y + m = 0) ∧ (2 = 2)) ∧ (m = ± real.sqrt(3)/2)

theorem ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : b = 1) (e : ℝ) (hfocal : (2 * e / a = 2))
  (hc :  e = real.sqrt 2 / 2) :
  (∃ (h : ℝ), ellipse_equation a b e h1 h2 ∧ find_ellipse h (real.sqrt 2 / 2)) := 
sorry

end ellipse_l589_589268


namespace athlete_heartbeats_during_race_l589_589823

theorem athlete_heartbeats_during_race
  (heart_rate : ℕ)
  (pace : ℕ)
  (distance : ℕ)
  (H1 : heart_rate = 160)
  (H2 : pace = 6)
  (H3 : distance = 30) :
  heart_rate * pace * distance = 28800 :=
by
  rw [H1, H2, H3]
  norm_num

end athlete_heartbeats_during_race_l589_589823


namespace circle_passing_through_points_l589_589617

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l589_589617


namespace SameFunction_l589_589090

noncomputable def f : ℝ → ℝ := λ x, x^2
noncomputable def g : ℝ → ℝ := λ x, (x^6)^(1/3)

theorem SameFunction : ∀ x : ℝ, f x = g x :=
by
  intro x
  sorry

end SameFunction_l589_589090


namespace circle_passes_through_points_l589_589512

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589512


namespace total_heartbeats_during_race_l589_589795

-- Definitions for the conditions
def heart_beats_per_minute : ℕ := 160
def pace_in_minutes_per_mile : ℕ := 6
def total_distance_in_miles : ℕ := 30

-- Main theorem statement
theorem total_heartbeats_during_race : 
  heart_beats_per_minute * pace_in_minutes_per_mile * total_distance_in_miles = 28800 :=
by
  -- Place the proof here
  sorry

end total_heartbeats_during_race_l589_589795


namespace circle_passes_through_points_l589_589519

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l589_589519


namespace finite_prime_sequence_l589_589855

theorem finite_prime_sequence (p : ℕ → ℕ) (h : ∀ i ≥ 2, prime (p i) ∧ (p i = 2 * p (i - 1) - 1 ∨ p i = 2 * p (i - 1) + 1)) : ∃ n, ∀ m > n, p m = 0 :=
sorry

end finite_prime_sequence_l589_589855


namespace circle_passing_through_points_l589_589497

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l589_589497
