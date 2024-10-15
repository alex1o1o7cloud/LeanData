import Mathlib

namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l273_27379

-- Define what it means to be a root of the equation x^2 - 5x + 6 = 0
def is_root (x : ℝ) : Prop := x^2 - 5 * x + 6 = 0

-- Define the perimeter based on given conditions
theorem isosceles_triangle_perimeter (x : ℝ) (base : ℝ) (h_base : base = 4) (h_root : is_root x) :
    2 * x + base = 10 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l273_27379


namespace NUMINAMATH_GPT_ratio_of_width_to_length_l273_27326

-- Definitions of length, width, perimeter
def l : ℕ := 10
def P : ℕ := 30

-- Define the condition for the width
def width_from_perimeter (l P : ℕ) : ℕ :=
  (P - 2 * l) / 2

-- Calculate the width using the given length and perimeter
def w : ℕ := width_from_perimeter l P

-- Theorem stating the ratio of width to length
theorem ratio_of_width_to_length : (w : ℚ) / l = 1 / 2 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_ratio_of_width_to_length_l273_27326


namespace NUMINAMATH_GPT_compare_a_b_c_l273_27350

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end NUMINAMATH_GPT_compare_a_b_c_l273_27350


namespace NUMINAMATH_GPT_remaining_soup_feeds_adults_l273_27372

theorem remaining_soup_feeds_adults :
  (∀ (cans : ℕ), cans ≥ 8 ∧ cans / 6 ≥ 24) → (∃ (adults : ℕ), adults = 16) :=
by
  sorry

end NUMINAMATH_GPT_remaining_soup_feeds_adults_l273_27372


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_m_l273_27334

def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

theorem part1_solution_set (x : ℝ) : (f x 3 >= 6) ↔ (x ≤ -2 ∨ x ≥ 4) :=
by sorry

theorem part2_range_of_m (m : ℝ) (x : ℝ) : 
 (∀ x : ℝ, f x m ≥ 8) ↔ (m ≤ -9 ∨ m ≥ 7) :=
by sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_m_l273_27334


namespace NUMINAMATH_GPT_proof_solution_l273_27302

variable (U : Set ℝ) (A : Set ℝ) (C_U_A : Set ℝ)
variables (a b : ℝ)

noncomputable def proof_problem : Prop :=
  (U = Set.univ) →
  (A = {x | a ≤ x ∧ x ≤ b}) →
  (C_U_A = {x | x > 4 ∨ x < 3}) →
  A = {x | 3 ≤ x ∧ x ≤ 4} ∧ a = 3 ∧ b = 4

theorem proof_solution : proof_problem U A C_U_A a b :=
by
  intro hU hA hCUA
  have hA_eq : A = {x | 3 ≤ x ∧ x ≤ 4} :=
    by { sorry }
  have ha : a = 3 :=
    by { sorry }
  have hb : b = 4 :=
    by { sorry }
  exact ⟨hA_eq, ha, hb⟩

end NUMINAMATH_GPT_proof_solution_l273_27302


namespace NUMINAMATH_GPT_total_hangers_l273_27371

def pink_hangers : ℕ := 7
def green_hangers : ℕ := 4
def blue_hangers : ℕ := green_hangers - 1
def yellow_hangers : ℕ := blue_hangers - 1

theorem total_hangers :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers = 16 := by
  sorry

end NUMINAMATH_GPT_total_hangers_l273_27371


namespace NUMINAMATH_GPT_work_done_resistive_force_l273_27341

noncomputable def mass : ℝ := 0.01  -- 10 grams converted to kilograms
noncomputable def v1 : ℝ := 400.0  -- initial speed in m/s
noncomputable def v2 : ℝ := 100.0  -- final speed in m/s

noncomputable def kinetic_energy (m v : ℝ) : ℝ := 0.5 * m * v^2

theorem work_done_resistive_force :
  let KE1 := kinetic_energy mass v1
  let KE2 := kinetic_energy mass v2
  KE1 - KE2 = 750 :=
by
  sorry

end NUMINAMATH_GPT_work_done_resistive_force_l273_27341


namespace NUMINAMATH_GPT_product_price_reduction_l273_27332

theorem product_price_reduction (z : ℝ) (x : ℝ) (hp1 : z > 0) (hp2 : 0.85 * 0.85 * z = z * (1 - x / 100)) : x = 27.75 := by
  sorry

end NUMINAMATH_GPT_product_price_reduction_l273_27332


namespace NUMINAMATH_GPT_find_a_plus_b_l273_27303

noncomputable def real_part (z : ℂ) : ℝ := z.re
noncomputable def imag_part (z : ℂ) : ℝ := z.im

theorem find_a_plus_b (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (1 + i) * (2 + i) = a + b * i) : a + b = 4 :=
by sorry

end NUMINAMATH_GPT_find_a_plus_b_l273_27303


namespace NUMINAMATH_GPT_couscous_dishes_l273_27304

def dishes (a b c d : ℕ) : ℕ := (a + b + c) / d

theorem couscous_dishes :
  dishes 7 13 45 5 = 13 :=
by
  unfold dishes
  sorry

end NUMINAMATH_GPT_couscous_dishes_l273_27304


namespace NUMINAMATH_GPT_monotonically_increasing_intervals_sin_value_l273_27393

noncomputable def f (x : Real) : Real := 2 * Real.cos (x - Real.pi / 3) * Real.cos x + 1

theorem monotonically_increasing_intervals :
  ∀ (k : Int), ∃ (a b : Real), a = k * Real.pi - Real.pi / 3 ∧ b = k * Real.pi + Real.pi / 6 ∧
                 ∀ (x y : Real), a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y :=
sorry

theorem sin_value 
  (α : Real) (hα : 0 < α ∧ α < Real.pi / 2) 
  (h : f (α + Real.pi / 12) = 7 / 6) : 
  Real.sin (7 * Real.pi / 6 - 2 * α) = 2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_GPT_monotonically_increasing_intervals_sin_value_l273_27393


namespace NUMINAMATH_GPT_find_x_l273_27357

theorem find_x (x : ℝ) (h : 45 * x = 0.60 * 900) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l273_27357


namespace NUMINAMATH_GPT_divisible_by_five_l273_27381

theorem divisible_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
sorry

end NUMINAMATH_GPT_divisible_by_five_l273_27381


namespace NUMINAMATH_GPT_zombies_count_decrease_l273_27399

theorem zombies_count_decrease (z : ℕ) (d : ℕ) : z = 480 → (∀ n, d = 2^n * z) → ∃ t, d / t < 50 :=
by
  intros hz hdz
  let initial_count := 480
  have := 480 / (2 ^ 4)
  sorry

end NUMINAMATH_GPT_zombies_count_decrease_l273_27399


namespace NUMINAMATH_GPT_find_smaller_number_l273_27313

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 := by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l273_27313


namespace NUMINAMATH_GPT_proof_probability_and_expectations_l273_27380

/-- Number of white balls drawn from two boxes --/
def X : ℕ := 1

/-- Number of red balls drawn from two boxes --/
def Y : ℕ := 1

/-- Given the conditions, the probability of drawing one white ball is 1/2, and
the expected value of white balls drawn is greater than the expected value of red balls drawn --/
theorem proof_probability_and_expectations :
  (∃ (P_X : ℚ), P_X = 1 / 2) ∧ (∃ (E_X E_Y : ℚ), E_X > E_Y) :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_probability_and_expectations_l273_27380


namespace NUMINAMATH_GPT_sin_cos_relation_l273_27358

theorem sin_cos_relation (α : ℝ) (h : Real.tan (π / 4 + α) = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_relation_l273_27358


namespace NUMINAMATH_GPT_function_maximum_at_1_l273_27390

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x ^ 2

theorem function_maximum_at_1 :
  ∀ x > 0, (f x ≤ f 1) :=
by
  intro x hx
  have hx_pos : 0 < x := hx
  sorry

end NUMINAMATH_GPT_function_maximum_at_1_l273_27390


namespace NUMINAMATH_GPT_contradiction_example_l273_27384

theorem contradiction_example (x y : ℝ) (h1 : x + y > 2) (h2 : x ≤ 1) (h3 : y ≤ 1) : False :=
by
  sorry

end NUMINAMATH_GPT_contradiction_example_l273_27384


namespace NUMINAMATH_GPT_solve_for_q_l273_27352

noncomputable def is_arithmetic_SUM_seq (a₁ q: ℝ) (n: ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem solve_for_q (a₁ q S3 S6 S9: ℝ) (hq: q ≠ 1) (hS3: S3 = is_arithmetic_SUM_seq a₁ q 3) 
(hS6: S6 = is_arithmetic_SUM_seq a₁ q 6) (hS9: S9 = is_arithmetic_SUM_seq a₁ q 9) 
(h_arith: 2 * S9 = S3 + S6) : q^3 = 3 / 2 :=
sorry

end NUMINAMATH_GPT_solve_for_q_l273_27352


namespace NUMINAMATH_GPT_find_xyz_l273_27322

open Complex

-- Definitions of the variables and conditions
variables {a b c x y z : ℂ} (h_a_ne_zero : a ≠ 0) (h_b_ne_zero : b ≠ 0) (h_c_ne_zero : c ≠ 0)
  (h_x_ne_zero : x ≠ 0) (h_y_ne_zero : y ≠ 0) (h_z_ne_zero : z ≠ 0)
  (h1 : a = (b - c) * (x + 2))
  (h2 : b = (a - c) * (y + 2))
  (h3 : c = (a - b) * (z + 2))
  (h4 : x * y + x * z + y * z = 12)
  (h5 : x + y + z = 6)

-- Statement of the theorem
theorem find_xyz : x * y * z = 7 := 
by
  -- Proof steps to be filled in
  sorry

end NUMINAMATH_GPT_find_xyz_l273_27322


namespace NUMINAMATH_GPT_simplified_expression_at_3_l273_27342

noncomputable def simplify_and_evaluate (x : ℝ) : ℝ :=
  (3 * x ^ 2 + 8 * x - 6) - (2 * x ^ 2 + 4 * x - 15)

theorem simplified_expression_at_3 : simplify_and_evaluate 3 = 30 :=
by
  sorry

end NUMINAMATH_GPT_simplified_expression_at_3_l273_27342


namespace NUMINAMATH_GPT_candy_pack_cost_l273_27364

theorem candy_pack_cost (c : ℝ) (h1 : 20 + 78 = 98) (h2 : 2 * c = 98) : c = 49 :=
by {
  sorry
}

end NUMINAMATH_GPT_candy_pack_cost_l273_27364


namespace NUMINAMATH_GPT_chord_length_of_circle_l273_27389

theorem chord_length_of_circle (x y : ℝ) (h1 : (x - 0)^2 + (y - 2)^2 = 4) (h2 : y = x) : 
  length_of_chord_intercepted_by_line_eq_2sqrt2 :=
sorry

end NUMINAMATH_GPT_chord_length_of_circle_l273_27389


namespace NUMINAMATH_GPT_number_of_bad_cards_l273_27361

-- Define the initial conditions
def janessa_initial_cards : ℕ := 4
def father_given_cards : ℕ := 13
def ordered_cards : ℕ := 36
def cards_given_to_dexter : ℕ := 29
def cards_kept_for_herself : ℕ := 20

-- Define the total cards and cards in bad shape calculation
theorem number_of_bad_cards : 
  let total_initial_cards := janessa_initial_cards + father_given_cards;
  let total_cards := total_initial_cards + ordered_cards;
  let total_distributed_cards := cards_given_to_dexter + cards_kept_for_herself;
  total_cards - total_distributed_cards = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_bad_cards_l273_27361


namespace NUMINAMATH_GPT_pow_gt_of_gt_l273_27387

variable {a x1 x2 : ℝ}

theorem pow_gt_of_gt (ha : a > 1) (hx : x1 > x2) : a^x1 > a^x2 :=
by sorry

end NUMINAMATH_GPT_pow_gt_of_gt_l273_27387


namespace NUMINAMATH_GPT_golden_section_BC_length_l273_27363

-- Definition of a golden section point
def is_golden_section_point (A B C : ℝ) : Prop :=
  ∃ (φ : ℝ), φ = (1 + Real.sqrt 5) / 2 ∧ B = φ * C

-- The given problem translated to Lean
theorem golden_section_BC_length (A B C : ℝ) (h1 : is_golden_section_point A B C) (h2 : B - A = 6) : 
  C - B = 3 * Real.sqrt 5 - 3 ∨ C - B = 9 - 3 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_golden_section_BC_length_l273_27363


namespace NUMINAMATH_GPT_billy_trays_l273_27377

def trays_needed (total_ice_cubes : ℕ) (ice_cubes_per_tray : ℕ) : ℕ :=
  total_ice_cubes / ice_cubes_per_tray

theorem billy_trays (total_ice_cubes ice_cubes_per_tray : ℕ) (h1 : total_ice_cubes = 72) (h2 : ice_cubes_per_tray = 9) :
  trays_needed total_ice_cubes ice_cubes_per_tray = 8 :=
by
  sorry

end NUMINAMATH_GPT_billy_trays_l273_27377


namespace NUMINAMATH_GPT_coefficient_square_sum_l273_27347

theorem coefficient_square_sum (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1728 * x ^ 3 + 64 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_square_sum_l273_27347


namespace NUMINAMATH_GPT_number_of_cooks_l273_27376

variable (C W : ℕ)

-- Conditions
def initial_ratio := 3 * W = 8 * C
def new_ratio := 4 * C = W + 12

theorem number_of_cooks (h1 : initial_ratio W C) (h2 : new_ratio W C) : C = 9 := by
  sorry

end NUMINAMATH_GPT_number_of_cooks_l273_27376


namespace NUMINAMATH_GPT_altitude_division_l273_27337

variables {A B C D E : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E]

theorem altitude_division 
  (AD DC CE EB y : ℝ)
  (hAD : AD = 6)
  (hDC : DC = 4)
  (hCE : CE = 3)
  (hEB : EB = y)
  (h_similarity : CE / DC = (AD + DC) / (y + CE)) : 
  y = 31 / 3 :=
by
  sorry

end NUMINAMATH_GPT_altitude_division_l273_27337


namespace NUMINAMATH_GPT_nicholas_bottle_caps_l273_27386

theorem nicholas_bottle_caps (initial : ℕ) (additional : ℕ) (final : ℕ) (h1 : initial = 8) (h2 : additional = 85) :
  final = 93 :=
by
  sorry

end NUMINAMATH_GPT_nicholas_bottle_caps_l273_27386


namespace NUMINAMATH_GPT_surface_area_of_solid_l273_27315

theorem surface_area_of_solid (num_unit_cubes : ℕ) (top_layer_cubes : ℕ) 
(bottom_layer_cubes : ℕ) (side_layer_cubes : ℕ) 
(front_and_back_cubes : ℕ) (left_and_right_cubes : ℕ) :
  num_unit_cubes = 15 →
  top_layer_cubes = 5 →
  bottom_layer_cubes = 5 →
  side_layer_cubes = 3 →
  front_and_back_cubes = 5 →
  left_and_right_cubes = 3 →
  let top_and_bottom_surface := top_layer_cubes + bottom_layer_cubes
  let front_and_back_surface := 2 * front_and_back_cubes
  let left_and_right_surface := 2 * left_and_right_cubes
  let total_surface := top_and_bottom_surface + front_and_back_surface + left_and_right_surface
  total_surface = 26 :=
by
  intros h_n h_t h_b h_s h_f h_lr
  let top_and_bottom_surface := top_layer_cubes + bottom_layer_cubes
  let front_and_back_surface := 2 * front_and_back_cubes
  let left_and_right_surface := 2 * left_and_right_cubes
  let total_surface := top_and_bottom_surface + front_and_back_surface + left_and_right_surface
  sorry

end NUMINAMATH_GPT_surface_area_of_solid_l273_27315


namespace NUMINAMATH_GPT_valid_y_values_for_triangle_l273_27328

-- Define the triangle inequality conditions for sides 8, 11, and y^2
theorem valid_y_values_for_triangle (y : ℕ) (h_pos : y > 0) :
  (8 + 11 > y^2) ∧ (8 + y^2 > 11) ∧ (11 + y^2 > 8) ↔ (y = 2 ∨ y = 3 ∨ y = 4) :=
by
  sorry

end NUMINAMATH_GPT_valid_y_values_for_triangle_l273_27328


namespace NUMINAMATH_GPT_hare_race_l273_27391

theorem hare_race :
  ∃ (total_jumps: ℕ) (final_jump_leg: String), total_jumps = 548 ∧ final_jump_leg = "right leg" :=
by
  sorry

end NUMINAMATH_GPT_hare_race_l273_27391


namespace NUMINAMATH_GPT_initial_cards_l273_27300

variable (x : ℕ)
variable (h1 : x - 3 = 2)

theorem initial_cards (x : ℕ) (h1 : x - 3 = 2) : x = 5 := by
  sorry

end NUMINAMATH_GPT_initial_cards_l273_27300


namespace NUMINAMATH_GPT_ramu_repair_cost_l273_27346

theorem ramu_repair_cost
  (initial_cost : ℝ)
  (selling_price : ℝ)
  (profit_percent : ℝ)
  (repair_cost : ℝ)
  (h1 : initial_cost = 42000)
  (h2 : selling_price = 64900)
  (h3 : profit_percent = 13.859649122807017 / 100)
  (h4 : selling_price = initial_cost + repair_cost + profit_percent * (initial_cost + repair_cost)) :
  repair_cost = 15000 :=
by
  sorry

end NUMINAMATH_GPT_ramu_repair_cost_l273_27346


namespace NUMINAMATH_GPT_cross_product_scaled_v_and_w_l273_27367

-- Assume the vectors and their scalar multiple
def v : ℝ × ℝ × ℝ := (3, 1, 4)
def w : ℝ × ℝ × ℝ := (-2, 2, -3)
def v_scaled : ℝ × ℝ × ℝ := (6, 2, 8)

-- Define the cross product function
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.1 * b.2.2 - a.2.2 * b.1,
   a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_scaled_v_and_w :
  cross_product v_scaled w = (-22, -2, 16) :=
by
  sorry

end NUMINAMATH_GPT_cross_product_scaled_v_and_w_l273_27367


namespace NUMINAMATH_GPT_trains_meet_in_32_seconds_l273_27333

noncomputable def train_meeting_time
  (length_train1 : ℕ)
  (length_train2 : ℕ)
  (initial_distance : ℕ)
  (speed_train1_kmph : ℕ)
  (speed_train2_kmph : ℕ)
  : ℕ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600
  let relative_speed := speed_train1_mps + speed_train2_mps
  let total_distance := length_train1 + length_train2 + initial_distance
  total_distance / relative_speed

theorem trains_meet_in_32_seconds :
  train_meeting_time 400 200 200 54 36 = 32 := 
by
  sorry

end NUMINAMATH_GPT_trains_meet_in_32_seconds_l273_27333


namespace NUMINAMATH_GPT_top_card_is_queen_probability_l273_27365

theorem top_card_is_queen_probability :
  let num_queens := 4
  let total_cards := 52
  let prob := num_queens / total_cards
  prob = 1 / 13 :=
by 
  sorry

end NUMINAMATH_GPT_top_card_is_queen_probability_l273_27365


namespace NUMINAMATH_GPT_parking_lot_problem_l273_27309

variable (M S : Nat)

theorem parking_lot_problem (h1 : M + S = 30) (h2 : 15 * M + 8 * S = 324) :
  M = 12 ∧ S = 18 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_parking_lot_problem_l273_27309


namespace NUMINAMATH_GPT_a_n_sequence_term2015_l273_27338

theorem a_n_sequence_term2015 :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ a 2 = 1/2 ∧ (∀ n ≥ 2, a n * (a (n-1) + a (n+1)) = 2 * a (n+1) * a (n-1)) ∧ a 2015 = 1/2015 :=
sorry

end NUMINAMATH_GPT_a_n_sequence_term2015_l273_27338


namespace NUMINAMATH_GPT_remainder_of_polynomial_l273_27336

noncomputable def P (x : ℝ) := 3 * x^5 - 2 * x^3 + 5 * x^2 - 8
noncomputable def D (x : ℝ) := x^2 + 3 * x + 2
noncomputable def R (x : ℝ) := 64 * x + 60

theorem remainder_of_polynomial :
  ∀ x : ℝ, P x % D x = R x :=
sorry

end NUMINAMATH_GPT_remainder_of_polynomial_l273_27336


namespace NUMINAMATH_GPT_sum_of_a_for_unique_solution_l273_27353

theorem sum_of_a_for_unique_solution (a : ℝ) (x : ℝ) :
  (∃ (a : ℝ), 3 * x ^ 2 + a * x + 6 * x + 7 = 0 ∧ (a + 6) ^ 2 - 4 * 3 * 7 = 0) →
  (-6 + 2 * Real.sqrt 21 + -6 - 2 * Real.sqrt 21 = -12) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_a_for_unique_solution_l273_27353


namespace NUMINAMATH_GPT_circle_through_points_and_intercepts_l273_27369

noncomputable def circle_eq (x y D E F : ℝ) : ℝ := x^2 + y^2 + D * x + E * y + F

theorem circle_through_points_and_intercepts :
  ∃ (D E F : ℝ), 
    circle_eq 4 2 D E F = 0 ∧
    circle_eq (-1) 3 D E F = 0 ∧ 
    D + E = -2 ∧
    circle_eq x y (-2) 0 (-12) = 0 :=
by
  unfold circle_eq
  sorry

end NUMINAMATH_GPT_circle_through_points_and_intercepts_l273_27369


namespace NUMINAMATH_GPT_eval_expression_l273_27351

def f (x : ℤ) : ℤ := 3 * x^2 - 6 * x + 10

theorem eval_expression : 3 * f 2 + 2 * f (-2) = 98 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l273_27351


namespace NUMINAMATH_GPT_coefficient_of_x_neg_2_in_binomial_expansion_l273_27311

theorem coefficient_of_x_neg_2_in_binomial_expansion :
  let x := (x : ℚ)
  let term := (x^3 - (2 / x))^6
  (coeff_of_term : Int) ->
  (coeff_of_term = -192) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_coefficient_of_x_neg_2_in_binomial_expansion_l273_27311


namespace NUMINAMATH_GPT_average_lecture_minutes_l273_27348

theorem average_lecture_minutes
  (lecture_duration : ℕ)
  (total_audience : ℕ)
  (percent_entire : ℝ)
  (percent_missed : ℝ)
  (percent_half : ℝ)
  (average_minutes : ℝ) :
  lecture_duration = 90 →
  total_audience = 200 →
  percent_entire = 0.30 →
  percent_missed = 0.20 →
  percent_half = 0.40 →
  average_minutes = 56.25 :=
by
  sorry

end NUMINAMATH_GPT_average_lecture_minutes_l273_27348


namespace NUMINAMATH_GPT_line_parabola_intersection_l273_27385

noncomputable def intersection_range (m : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + m * x - 1 = 2 * x - 2 * m → -1 ≤ x ∧ x ≤ 3

theorem line_parabola_intersection (m : ℝ) :
  intersection_range m ↔ -3 / 5 < m ∧ m < 5 :=
by
  sorry

end NUMINAMATH_GPT_line_parabola_intersection_l273_27385


namespace NUMINAMATH_GPT_inequality_solution_set_l273_27301

theorem inequality_solution_set :
  {x : ℝ | (x^2 - x - 6) / (x - 1) > 0} = {x : ℝ | (-2 < x ∧ x < 1) ∨ (3 < x)} := by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l273_27301


namespace NUMINAMATH_GPT_find_multiple_l273_27327

theorem find_multiple:
  let number := 220025
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := number / sum
  let remainder := number % sum
  (remainder = 25) → (quotient = 220) → (quotient / diff = 2) :=
by
  intros number sum diff quotient remainder h1 h2
  sorry

end NUMINAMATH_GPT_find_multiple_l273_27327


namespace NUMINAMATH_GPT_find_line_equation_l273_27319

noncomputable def y_line (m b x : ℝ) : ℝ := m * x + b
noncomputable def quadratic_y (x : ℝ) : ℝ := x ^ 2 + 8 * x + 7

noncomputable def equation_of_the_line : Prop :=
  ∃ (m b k : ℝ),
    (quadratic_y k = y_line m b k + 6 ∨ quadratic_y k = y_line m b k - 6) ∧
    (y_line m b 2 = 7) ∧ 
    b ≠ 0 ∧
    y_line 19.5 (-32) = y_line m b

theorem find_line_equation : equation_of_the_line :=
sorry

end NUMINAMATH_GPT_find_line_equation_l273_27319


namespace NUMINAMATH_GPT_inequality_proof_l273_27310

theorem inequality_proof 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 > 0) 
  (h2 : a2 > 0) 
  (h3 : a3 > 0)
  (h4 : a4 > 0):
  (a1 + a3) / (a1 + a2) + 
  (a2 + a4) / (a2 + a3) + 
  (a3 + a1) / (a3 + a4) + 
  (a4 + a2) / (a4 + a1) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l273_27310


namespace NUMINAMATH_GPT_cube_volume_l273_27370

theorem cube_volume (a : ℝ) (h : (a - 1) * (a - 1) * (a + 1) = a^3 - 7) : a^3 = 8 :=
  sorry

end NUMINAMATH_GPT_cube_volume_l273_27370


namespace NUMINAMATH_GPT_total_cost_train_and_bus_l273_27378

noncomputable def trainFare := 3.75 + 2.35
noncomputable def busFare := 3.75
noncomputable def totalFare := trainFare + busFare

theorem total_cost_train_and_bus : totalFare = 9.85 :=
by
  -- We'll need a proof here if required.
  sorry

end NUMINAMATH_GPT_total_cost_train_and_bus_l273_27378


namespace NUMINAMATH_GPT_inscribed_circle_radius_l273_27320

theorem inscribed_circle_radius 
  (A : ℝ) -- Area of the triangle
  (p : ℝ) -- Perimeter of the triangle
  (r : ℝ) -- Radius of the inscribed circle
  (s : ℝ) -- Semiperimeter of the triangle
  (h1 : A = 2 * p) -- Condition: Area is numerically equal to twice the perimeter
  (h2 : p = 2 * s) -- Perimeter is twice the semiperimeter
  (h3 : A = r * s) -- Formula: Area in terms of inradius and semiperimeter
  (h4 : s ≠ 0) -- Semiperimeter is non-zero
  : r = 4 := 
sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l273_27320


namespace NUMINAMATH_GPT_problem1_problem2_l273_27359

-- Problem 1
theorem problem1 (b : ℝ) :
  4 * b^2 * (b^3 - 1) - 3 * (1 - 2 * b^2) > 4 * (b^5 - 1) :=
by
  sorry

-- Problem 2
theorem problem2 (a : ℝ) :
  a - a * abs (-a^2 - 1) < 1 - a^2 * (a - 1) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l273_27359


namespace NUMINAMATH_GPT_min_value_eq_l273_27325

open Real
open Classical

noncomputable def min_value (x y : ℝ) : ℝ := x + 4 * y

theorem min_value_eq :
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (1 / x + 1 / (2 * y) = 1) → (min_value x y) = 3 + 2 * sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_eq_l273_27325


namespace NUMINAMATH_GPT_abs_sum_condition_l273_27366

theorem abs_sum_condition (a b : ℝ) (h1 : |a| = 7) (h2 : |b| = 3) (h3 : a * b > 0) : a + b = 10 ∨ a + b = -10 :=
by { sorry }

end NUMINAMATH_GPT_abs_sum_condition_l273_27366


namespace NUMINAMATH_GPT_jon_found_marbles_l273_27344

-- Definitions based on the conditions
variables (M J B : ℕ)

-- Prove that Jon found 110 marbles
theorem jon_found_marbles
  (h1 : M + J = 66)
  (h2 : M = 2 * J)
  (h3 : J + B = 3 * M) :
  B = 110 :=
by
  sorry -- proof to be completed

end NUMINAMATH_GPT_jon_found_marbles_l273_27344


namespace NUMINAMATH_GPT_find_f_1789_l273_27307

def f : ℕ → ℕ := sorry

axiom f_1 : f 1 = 5
axiom f_f_n : ∀ n, f (f n) = 4 * n + 9
axiom f_2n : ∀ n, f (2 * n) = (2 * n) + 1 + 3

theorem find_f_1789 : f 1789 = 3581 :=
by
  sorry

end NUMINAMATH_GPT_find_f_1789_l273_27307


namespace NUMINAMATH_GPT_speed_in_still_water_l273_27374

/-- Conditions -/
def upstream_speed : ℝ := 30
def downstream_speed : ℝ := 40

/-- Theorem: The speed of the man in still water is 35 kmph. -/
theorem speed_in_still_water : 
  (upstream_speed + downstream_speed) / 2 = 35 := 
by 
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l273_27374


namespace NUMINAMATH_GPT_polynomial_value_at_minus_2_l273_27306

variable (a b : ℝ)

def polynomial (x : ℝ) : ℝ := a * x^3 + b * x - 3

theorem polynomial_value_at_minus_2 :
  (polynomial a b (-2) = -21) :=
  sorry

end NUMINAMATH_GPT_polynomial_value_at_minus_2_l273_27306


namespace NUMINAMATH_GPT_coin_count_l273_27368

theorem coin_count (x : ℝ) (h₁ : x + 0.50 * x + 0.25 * x = 35) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_coin_count_l273_27368


namespace NUMINAMATH_GPT_det_matrix_A_l273_27354

noncomputable def matrix_A (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, y, z], ![z, x, y], ![y, z, x]]

theorem det_matrix_A (x y z : ℝ) : 
  Matrix.det (matrix_A x y z) = x^3 + y^3 + z^3 - 3*x*y*z := by
  sorry

end NUMINAMATH_GPT_det_matrix_A_l273_27354


namespace NUMINAMATH_GPT_monday_rainfall_l273_27318

theorem monday_rainfall (tuesday_rainfall monday_rainfall: ℝ) 
(less_rain: ℝ) (h1: tuesday_rainfall = 0.2) 
(h2: less_rain = 0.7) 
(h3: tuesday_rainfall = monday_rainfall - less_rain): 
monday_rainfall = 0.9 :=
by sorry

end NUMINAMATH_GPT_monday_rainfall_l273_27318


namespace NUMINAMATH_GPT_part1_part2_l273_27349

noncomputable def f (x : ℝ) := Real.exp x

theorem part1 (x : ℝ) (h : x ≥ 0) (m : ℝ) : 
  (x - 1) * f x ≥ m * x^2 - 1 ↔ m ≤ 1 / 2 :=
sorry

theorem part2 (x : ℝ) (h : x > 0) : 
  f x > 4 * Real.log x + 8 - 8 * Real.log 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l273_27349


namespace NUMINAMATH_GPT_chocolate_cost_first_store_l273_27331

def cost_first_store (x : ℕ) : ℕ := x
def chocolate_promotion_store : ℕ := 2
def savings_in_three_weeks : ℕ := 6
def number_of_chocolates (weeks : ℕ) : ℕ := 2 * weeks

theorem chocolate_cost_first_store :
  ∀ (weeks : ℕ) (x : ℕ), 
    number_of_chocolates weeks = 6 →
    chocolate_promotion_store * number_of_chocolates weeks + savings_in_three_weeks = cost_first_store x * number_of_chocolates weeks →
    cost_first_store x = 3 :=
by
  intros weeks x h1 h2
  sorry

end NUMINAMATH_GPT_chocolate_cost_first_store_l273_27331


namespace NUMINAMATH_GPT_christopher_sword_length_l273_27398

variable (C J U : ℤ)

def jameson_sword (C : ℤ) : ℤ := 2 * C + 3
def june_sword (J : ℤ) : ℤ := J + 5
def june_sword_christopher (C : ℤ) : ℤ := C + 23

theorem christopher_sword_length (h1 : J = jameson_sword C)
                                (h2 : U = june_sword J)
                                (h3 : U = june_sword_christopher C) :
                                C = 15 :=
by
  sorry

end NUMINAMATH_GPT_christopher_sword_length_l273_27398


namespace NUMINAMATH_GPT_find_ab_l273_27373

theorem find_ab (a b : ℝ) 
  (H_period : (1 : ℝ) * (π / b) = π / 2)
  (H_point : a * Real.tan (b * (π / 8)) = 4) :
  a * b = 8 :=
sorry

end NUMINAMATH_GPT_find_ab_l273_27373


namespace NUMINAMATH_GPT_multiple_of_son_age_last_year_l273_27356

theorem multiple_of_son_age_last_year
  (G : ℕ) (S : ℕ) (M : ℕ)
  (h1 : G = 42 - 1)
  (h2 : S = 16 - 1)
  (h3 : G = M * S - 4) :
  M = 3 := by
  sorry

end NUMINAMATH_GPT_multiple_of_son_age_last_year_l273_27356


namespace NUMINAMATH_GPT_no_symmetric_a_l273_27382

noncomputable def f (a x : ℝ) : ℝ := Real.log (((x + 1) / (x - 1)) * (x - 1) * (a - x))

theorem no_symmetric_a (a : ℝ) (h_a : 1 < a) : ¬ ∃ c : ℝ, ∀ d : ℝ, 1 < c - d ∧ c - d < a ∧ 1 < c + d ∧ c + d < a → f a (c - d) = f a (c + d) :=
sorry

end NUMINAMATH_GPT_no_symmetric_a_l273_27382


namespace NUMINAMATH_GPT_postcard_cost_l273_27345

theorem postcard_cost (x : ℕ) (h₁ : 9 * x < 1000) (h₂ : 10 * x > 1100) : x = 111 :=
by
  sorry

end NUMINAMATH_GPT_postcard_cost_l273_27345


namespace NUMINAMATH_GPT_possible_orange_cells_l273_27396

theorem possible_orange_cells :
  ∃ (n : ℕ), n = 2021 * 2020 ∨ n = 2022 * 2020 := 
sorry

end NUMINAMATH_GPT_possible_orange_cells_l273_27396


namespace NUMINAMATH_GPT_parallel_lines_m_value_l273_27329

/-- Given two lines x + m * y + 6 = 0 and (m - 2) * x + 3 * y + 2 * m = 0 are parallel,
    prove that the value of the real number m that makes the lines parallel is -1. -/
theorem parallel_lines_m_value (m : ℝ) : 
  (x + m * y + 6 = 0 ∧ (m - 2) * x + 3 * y + 2 * m = 0 → 
  (m = -1)) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_m_value_l273_27329


namespace NUMINAMATH_GPT_summer_discount_percentage_l273_27335

/--
Given:
1. The original cost of the jeans (original_price) is $49.
2. On Wednesdays, there is an additional $10.00 off on all jeans after the summer discount is applied.
3. Before the sales tax applies, the cost of a pair of jeans (final_price) is $14.50.

Prove:
The summer discount percentage (D) is 50%.
-/
theorem summer_discount_percentage (original_price final_price : ℝ) (D : ℝ) :
  original_price = 49 → 
  final_price = 14.50 → 
  (original_price - (original_price * D / 100) - 10 = final_price) → 
  D = 50 :=
by intros h_original h_final h_discount; sorry

end NUMINAMATH_GPT_summer_discount_percentage_l273_27335


namespace NUMINAMATH_GPT_caleb_trip_duration_l273_27375

-- Define the times when the clock hands meet
def startTime := 7 * 60 + 38 -- 7:38 a.m. in minutes from midnight
def endTime := 13 * 60 + 5 -- 1:05 p.m. in minutes from midnight

def duration := endTime - startTime

theorem caleb_trip_duration :
  duration = 5 * 60 + 27 := by
sorry

end NUMINAMATH_GPT_caleb_trip_duration_l273_27375


namespace NUMINAMATH_GPT_convert_speed_kmph_to_mps_l273_27343

def kilometers_to_meters := 1000
def hours_to_seconds := 3600
def speed_kmph := 18
def expected_speed_mps := 5

theorem convert_speed_kmph_to_mps :
  speed_kmph * (kilometers_to_meters / hours_to_seconds) = expected_speed_mps :=
by
  sorry

end NUMINAMATH_GPT_convert_speed_kmph_to_mps_l273_27343


namespace NUMINAMATH_GPT_domain_of_myFunction_l273_27392

-- Define the function
def myFunction (x : ℝ) : ℝ := (x + 2) ^ (1 / 2) - (x + 1) ^ 0

-- State the domain constraints as a theorem
theorem domain_of_myFunction (x : ℝ) : 
  (x ≥ -2 ∧ x ≠ -1) →
  ∃ y : ℝ, y = myFunction x := 
sorry

end NUMINAMATH_GPT_domain_of_myFunction_l273_27392


namespace NUMINAMATH_GPT_melanie_plums_count_l273_27339

theorem melanie_plums_count (dan_plums sally_plums total_plums melanie_plums : ℕ)
    (h1 : dan_plums = 9)
    (h2 : sally_plums = 3)
    (h3 : total_plums = 16)
    (h4 : melanie_plums = total_plums - (dan_plums + sally_plums)) :
    melanie_plums = 4 := by
  -- Proof will be filled here
  sorry

end NUMINAMATH_GPT_melanie_plums_count_l273_27339


namespace NUMINAMATH_GPT_gas_usage_l273_27360

def distance_dermatologist : ℕ := 30
def distance_gynecologist : ℕ := 50
def car_efficiency : ℕ := 20

theorem gas_usage (d_1 d_2 e : ℕ) (H1 : d_1 = distance_dermatologist) (H2 : d_2 = distance_gynecologist) (H3 : e = car_efficiency) :
  (2 * d_1 + 2 * d_2) / e = 8 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end NUMINAMATH_GPT_gas_usage_l273_27360


namespace NUMINAMATH_GPT_equality_am_bn_l273_27362

theorem equality_am_bn (m n : ℝ) (x : ℝ) (a b : ℝ) (hmn : m ≠ n) (hm : m ≠ 0) (hn : n ≠ 0) :
  ((x + m) ^ 2 - (x + n) ^ 2 = (m - n) ^ 2) → (x = am + bn) → (a = 0 ∧ b = -1) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_equality_am_bn_l273_27362


namespace NUMINAMATH_GPT_total_time_before_playing_game_l273_27394

theorem total_time_before_playing_game : 
  ∀ (d i t_t t : ℕ), 
  d = 10 → 
  i = d / 2 → 
  t_t = 3 * (d + i) → 
  t = d + i + t_t → 
  t = 60 := 
by
  intros d i t_t t h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_total_time_before_playing_game_l273_27394


namespace NUMINAMATH_GPT_solution_of_inequality_l273_27312

theorem solution_of_inequality : 
  {x : ℝ | x^2 - x - 2 > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 2 < x} :=
by
  sorry

end NUMINAMATH_GPT_solution_of_inequality_l273_27312


namespace NUMINAMATH_GPT_range_of_x_l273_27340

-- Define the necessary properties and functions.
variable (f : ℝ → ℝ)
variable (hf_even : ∀ x : ℝ, f (-x) = f x)
variable (hf_monotonic : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y)

-- Define the statement to be proved.
theorem range_of_x (f : ℝ → ℝ) (hf_even : ∀ x, f (-x) = f x) (hf_monotonic : ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  { x : ℝ | f (2 * x - 1) ≤ f 3 } = { x | -1 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l273_27340


namespace NUMINAMATH_GPT_not_axiom_l273_27330

theorem not_axiom (P Q R S : Prop)
  (B : P -> Q -> R -> S)
  (C : P -> Q)
  (D : P -> R)
  : ¬ (P -> Q -> S) :=
sorry

end NUMINAMATH_GPT_not_axiom_l273_27330


namespace NUMINAMATH_GPT_problem_statement_l273_27383

def f (x : ℕ) : ℝ := sorry

theorem problem_statement (h_cond : ∀ k : ℕ, f k ≤ (k : ℝ) ^ 2 → f (k + 1) ≤ (k + 1 : ℝ) ^ 2)
    (h_f7 : f 7 = 50) : ∀ k : ℕ, k ≤ 7 → f k > (k : ℝ) ^ 2 :=
sorry

end NUMINAMATH_GPT_problem_statement_l273_27383


namespace NUMINAMATH_GPT_parallel_lines_constant_l273_27316

theorem parallel_lines_constant (a : ℝ) : 
  (∀ x y : ℝ, (a - 1) * x + 2 * y + 3 = 0 → x + a * y + 3 = 0) → a = -1 :=
by sorry

end NUMINAMATH_GPT_parallel_lines_constant_l273_27316


namespace NUMINAMATH_GPT_CarriesJellybeanCount_l273_27395

-- Definitions based on conditions in part a)
def BertBoxJellybeans : ℕ := 150
def BertBoxVolume : ℕ := 6
def CarriesBoxVolume : ℕ := 3 * 2 * 4 * BertBoxVolume -- (3 * height, 2 * width, 4 * length)

-- Theorem statement in Lean based on part c)
theorem CarriesJellybeanCount : (CarriesBoxVolume / BertBoxVolume) * BertBoxJellybeans = 3600 := by 
  sorry

end NUMINAMATH_GPT_CarriesJellybeanCount_l273_27395


namespace NUMINAMATH_GPT_largest_a_mul_b_l273_27314

-- Given conditions and proof statement
theorem largest_a_mul_b {m k q a b : ℕ} (hm : m = 720 * k + 83)
  (ha : m = a * q + b) (h_b_lt_a: b < a): a * b = 5112 :=
sorry

end NUMINAMATH_GPT_largest_a_mul_b_l273_27314


namespace NUMINAMATH_GPT_geometric_seq_property_l273_27388

noncomputable def a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

theorem geometric_seq_property (n : ℕ) (h_arith : S (n + 1) + S (n + 1) = 2 * S (n)) (h_condition : a 2 = -2) :
  a 7 = 64 := 
by sorry

end NUMINAMATH_GPT_geometric_seq_property_l273_27388


namespace NUMINAMATH_GPT_average_income_of_other_40_customers_l273_27305

/-
Given:
1. The average income of 50 customers is $45,000.
2. The average income of the wealthiest 10 customers is $55,000.

Prove:
1. The average income of the other 40 customers is $42,500.
-/

theorem average_income_of_other_40_customers 
  (avg_income_50 : ℝ)
  (wealthiest_10_avg : ℝ) 
  (total_customers : ℕ)
  (wealthiest_customers : ℕ)
  (remaining_customers : ℕ)
  (h1 : avg_income_50 = 45000)
  (h2 : wealthiest_10_avg = 55000)
  (h3 : total_customers = 50)
  (h4 : wealthiest_customers = 10)
  (h5 : remaining_customers = 40) :
  let total_income_50 := total_customers * avg_income_50
  let total_income_wealthiest_10 := wealthiest_customers * wealthiest_10_avg
  let income_remaining_customers := total_income_50 - total_income_wealthiest_10
  let avg_income_remaining := income_remaining_customers / remaining_customers
  avg_income_remaining = 42500 := 
sorry

end NUMINAMATH_GPT_average_income_of_other_40_customers_l273_27305


namespace NUMINAMATH_GPT_simplify_expression_l273_27324

theorem simplify_expression (x : ℝ) : (3 * x)^4 + 3 * x * x^3 + 2 * x^5 = 84 * x^4 + 2 * x^5 := by
    sorry

end NUMINAMATH_GPT_simplify_expression_l273_27324


namespace NUMINAMATH_GPT_minimum_dwarfs_to_prevent_empty_chair_sitting_l273_27308

theorem minimum_dwarfs_to_prevent_empty_chair_sitting :
  ∀ (C : Fin 30 → Bool), (∀ i, C i ∨ C ((i + 1) % 30) ∨ C ((i + 2) % 30)) ↔ (∃ n, n = 10) :=
by
  sorry

end NUMINAMATH_GPT_minimum_dwarfs_to_prevent_empty_chair_sitting_l273_27308


namespace NUMINAMATH_GPT_all_solutions_of_diophantine_eq_l273_27323

theorem all_solutions_of_diophantine_eq
  (a b c x0 y0 : ℤ) (h_gcd : Int.gcd a b = 1)
  (h_sol : a * x0 + b * y0 = c) :
  ∀ x y : ℤ, (a * x + b * y = c) →
  ∃ t : ℤ, x = x0 + b * t ∧ y = y0 - a * t :=
by
  sorry

end NUMINAMATH_GPT_all_solutions_of_diophantine_eq_l273_27323


namespace NUMINAMATH_GPT_payments_option1_option2_option1_more_effective_combined_option_cost_l273_27317

variable {x : ℕ}

-- Condition 1: Prices and discount options
def badminton_rackets_price : ℕ := 40
def shuttlecocks_price : ℕ := 10
def discount_option1_free_shuttlecocks (pairs : ℕ): ℕ := pairs
def discount_option2_price (price : ℕ) : ℕ := price * 9 / 10

-- Condition 2: Buying requirements
def pairs_needed : ℕ := 10
def shuttlecocks_needed (n : ℕ) : ℕ := n
axiom x_gt_10 : x > 10

-- Proof Problem 1: Payment calculations
theorem payments_option1_option2 (x : ℕ) (h : x > 10) :
  (shuttlecocks_price * (shuttlecocks_needed x - discount_option1_free_shuttlecocks pairs_needed) + badminton_rackets_price * pairs_needed =
    10 * x + 300) ∧
  (discount_option2_price (shuttlecocks_price * shuttlecocks_needed x + badminton_rackets_price * pairs_needed) =
    9 * x + 360) :=
sorry

-- Proof Problem 2: More cost-effective option when x=30
theorem option1_more_effective (x : ℕ) (h : x = 30) :
  (10 * x + 300 < 9 * x + 360) :=
sorry

-- Proof Problem 3: Another cost-effective method when x=30
theorem combined_option_cost (x : ℕ) (h : x = 30) :
  (badminton_rackets_price * pairs_needed + discount_option2_price (shuttlecocks_price * (shuttlecocks_needed x - 10)) = 580) :=
sorry

end NUMINAMATH_GPT_payments_option1_option2_option1_more_effective_combined_option_cost_l273_27317


namespace NUMINAMATH_GPT_basketball_team_selection_l273_27397

theorem basketball_team_selection :
  (Nat.choose 4 2) * (Nat.choose 14 6) = 18018 := 
by
  -- number of ways to choose 2 out of 4 quadruplets
  -- number of ways to choose 6 out of the remaining 14 players
  -- the product of these combinations equals the required number of ways
  sorry

end NUMINAMATH_GPT_basketball_team_selection_l273_27397


namespace NUMINAMATH_GPT_frankie_pets_total_l273_27321

noncomputable def total_pets (c : ℕ) : ℕ :=
  let dogs := 2
  let cats := c
  let snakes := c + 5
  let parrots := c - 1
  dogs + cats + snakes + parrots

theorem frankie_pets_total (c : ℕ) (hc : 2 + 4 + (c + 1) + (c - 1) = 19) : total_pets c = 19 := by
  sorry

end NUMINAMATH_GPT_frankie_pets_total_l273_27321


namespace NUMINAMATH_GPT_proof_problem_l273_27355

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom condition1 : ∀ x : ℝ, f x + x * g x = x ^ 2 - 1
axiom condition2 : f 1 = 1

theorem proof_problem : deriv f 1 + deriv g 1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l273_27355
