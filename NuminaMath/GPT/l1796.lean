import Mathlib

namespace NUMINAMATH_GPT_max_value_of_b_minus_a_l1796_179652

theorem max_value_of_b_minus_a (a b : ℝ) (h1 : a < 0) (h2 : ∀ x : ℝ, a < x ∧ x < b → (3 * x^2 + a) * (2 * x + b) ≥ 0) : b - a ≤ 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_b_minus_a_l1796_179652


namespace NUMINAMATH_GPT_turns_in_two_hours_l1796_179670

theorem turns_in_two_hours (turns_per_30_sec : ℕ) (minutes_in_hour : ℕ) (hours : ℕ) : 
  turns_per_30_sec = 6 → 
  minutes_in_hour = 60 → 
  hours = 2 → 
  (12 * (minutes_in_hour * hours)) = 1440 := 
by
  sorry

end NUMINAMATH_GPT_turns_in_two_hours_l1796_179670


namespace NUMINAMATH_GPT_smallest_positive_integer_x_l1796_179638

theorem smallest_positive_integer_x :
  ∃ (x : ℕ), 0 < x ∧ (45 * x + 13) % 17 = 5 % 17 ∧ ∀ y : ℕ, 0 < y ∧ (45 * y + 13) % 17 = 5 % 17 → y ≥ x := 
sorry

end NUMINAMATH_GPT_smallest_positive_integer_x_l1796_179638


namespace NUMINAMATH_GPT_coats_leftover_l1796_179671

theorem coats_leftover :
  ∀ (total_coats : ℝ) (num_boxes : ℝ),
  total_coats = 385.5 →
  num_boxes = 7.5 →
  ∃ extra_coats : ℕ, extra_coats = 3 :=
by
  intros total_coats num_boxes h1 h2
  sorry

end NUMINAMATH_GPT_coats_leftover_l1796_179671


namespace NUMINAMATH_GPT_max_pieces_with_3_cuts_l1796_179644

theorem max_pieces_with_3_cuts (cake : Type) : 
  (∀ (cuts : ℕ), cuts = 3 → (∃ (max_pieces : ℕ), max_pieces = 8)) := by
  sorry

end NUMINAMATH_GPT_max_pieces_with_3_cuts_l1796_179644


namespace NUMINAMATH_GPT_possible_box_dimensions_l1796_179665

-- Define the initial conditions
def edge_length_original_box := 4
def edge_length_dice := 1
def total_cubes := (edge_length_original_box * edge_length_original_box * edge_length_original_box)

-- Prove that these are the possible dimensions of boxes with square bases that fit all the dice
theorem possible_box_dimensions :
  ∃ (len1 len2 len3 : ℕ), 
  total_cubes = (len1 * len2 * len3) ∧ 
  (len1 = len2) ∧ 
  ((len1, len2, len3) = (1, 1, 64) ∨ (len1, len2, len3) = (2, 2, 16) ∨ (len1, len2, len3) = (4, 4, 4) ∨ (len1, len2, len3) = (8, 8, 1)) :=
by {
  sorry -- The proof would be placed here
}

end NUMINAMATH_GPT_possible_box_dimensions_l1796_179665


namespace NUMINAMATH_GPT_distance_to_store_l1796_179641

noncomputable def D : ℝ := 4

theorem distance_to_store :
  (1/3) * (D/2 + D/10 + D/10) = 56/60 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_store_l1796_179641


namespace NUMINAMATH_GPT_plum_balances_pear_l1796_179646

variable (A G S : ℕ)

-- Definitions as per the problem conditions
axiom condition1 : 3 * A + G = 10 * S
axiom condition2 : A + 6 * S = G

-- The goal is to prove the following statement
theorem plum_balances_pear : G = 7 * S :=
by
  -- Skipping the proof as only statement is needed
  sorry

end NUMINAMATH_GPT_plum_balances_pear_l1796_179646


namespace NUMINAMATH_GPT_reduce_to_one_piece_l1796_179642

-- Definitions representing the conditions:
def plane_divided_into_unit_triangles : Prop := sorry
def initial_configuration (n : ℕ) : Prop := sorry
def possible_moves : Prop := sorry

-- Main theorem statement:
theorem reduce_to_one_piece (n : ℕ) 
  (H1 : plane_divided_into_unit_triangles) 
  (H2 : initial_configuration n) 
  (H3 : possible_moves) : 
  ∃ k : ℕ, k * 3 = n :=
sorry

end NUMINAMATH_GPT_reduce_to_one_piece_l1796_179642


namespace NUMINAMATH_GPT_rectangle_area_l1796_179695

theorem rectangle_area (L W P : ℝ) (hL : L = 13) (hP : P = 50) (hP_eq : P = 2 * L + 2 * W) :
  L * W = 156 :=
by
  have hL_val : L = 13 := hL
  have hP_val : P = 50 := hP
  have h_perimeter : P = 2 * L + 2 * W := hP_eq
  sorry

end NUMINAMATH_GPT_rectangle_area_l1796_179695


namespace NUMINAMATH_GPT_sqrt_product_simplified_l1796_179672

theorem sqrt_product_simplified (x : ℝ) (hx : 0 ≤ x) :
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 84 * x * Real.sqrt (2 * x) :=
by 
  sorry

end NUMINAMATH_GPT_sqrt_product_simplified_l1796_179672


namespace NUMINAMATH_GPT_part1_part2_l1796_179658

def A (x : ℝ) : Prop := x^2 + 2*x - 3 < 0
def B (x : ℝ) (a : ℝ) : Prop := abs (x + a) < 1

theorem part1 (a : ℝ) (h : a = 3) : (∃ x : ℝ, (A x ∨ B x a)) ↔ (∃ x : ℝ, -4 < x ∧ x < 1) :=
by {
  sorry
}

theorem part2 : (∀ x : ℝ, B x a → A x) ∧ (¬ ∀ x : ℝ, A x → B x a) ↔ 0 ≤ a ∧ a ≤ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_part1_part2_l1796_179658


namespace NUMINAMATH_GPT_students_in_both_clubs_l1796_179633

theorem students_in_both_clubs :
  ∀ (total_students drama_club science_club either_club both_club : ℕ),
  total_students = 300 →
  drama_club = 100 →
  science_club = 140 →
  either_club = 220 →
  (drama_club + science_club - both_club = either_club) →
  both_club = 20 :=
by
  intros total_students drama_club science_club either_club both_club
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_students_in_both_clubs_l1796_179633


namespace NUMINAMATH_GPT_total_distance_covered_l1796_179623

theorem total_distance_covered :
  let speed1 := 40 -- miles per hour
  let speed2 := 50 -- miles per hour
  let speed3 := 30 -- miles per hour
  let time1 := 1.5 -- hours
  let time2 := 1 -- hour
  let time3 := 2.25 -- hours
  let distance1 := speed1 * time1 -- distance covered in the first part of the trip
  let distance2 := speed2 * time2 -- distance covered in the second part of the trip
  let distance3 := speed3 * time3 -- distance covered in the third part of the trip
  distance1 + distance2 + distance3 = 177.5 := 
by
  sorry

end NUMINAMATH_GPT_total_distance_covered_l1796_179623


namespace NUMINAMATH_GPT_coordinates_of_P_with_respect_to_origin_l1796_179679

def point (x y : ℝ) : Prop := True

theorem coordinates_of_P_with_respect_to_origin :
  point 2 (-3) ↔ point 2 (-3) := by
  sorry

end NUMINAMATH_GPT_coordinates_of_P_with_respect_to_origin_l1796_179679


namespace NUMINAMATH_GPT_modular_arithmetic_proof_l1796_179632

open Nat

theorem modular_arithmetic_proof (m : ℕ) (h0 : 0 ≤ m ∧ m < 37) (h1 : 4 * m ≡ 1 [MOD 37]) :
  (3^m)^4 ≡ 27 + 3 [MOD 37] :=
by
  -- Although some parts like modular inverse calculation or finding specific m are skipped,
  -- the conclusion directly should reflect (3^m)^4 ≡ 27 + 3 [MOD 37]
  -- Considering (3^m)^4 - 3 ≡ 24 [MOD 37] translates to the above statement
  sorry

end NUMINAMATH_GPT_modular_arithmetic_proof_l1796_179632


namespace NUMINAMATH_GPT_star_number_of_intersections_2018_25_l1796_179645

-- Definitions for the conditions
def rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def star_intersections (n k : ℕ) : ℕ := 
  n * (k - 1)

-- The main theorem
theorem star_number_of_intersections_2018_25 :
  2018 ≥ 5 ∧ 25 < 2018 / 2 ∧ rel_prime 2018 25 → 
  star_intersections 2018 25 = 48432 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_star_number_of_intersections_2018_25_l1796_179645


namespace NUMINAMATH_GPT_train_distance_difference_l1796_179669

theorem train_distance_difference 
  (speed1 speed2 : ℕ) (distance : ℕ) (meet_time : ℕ)
  (h_speed1 : speed1 = 16)
  (h_speed2 : speed2 = 21)
  (h_distance : distance = 444)
  (h_meet_time : meet_time = distance / (speed1 + speed2)) :
  (speed2 * meet_time) - (speed1 * meet_time) = 60 :=
by
  sorry

end NUMINAMATH_GPT_train_distance_difference_l1796_179669


namespace NUMINAMATH_GPT_part1_part2_l1796_179627

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) - abs (x + 2)

theorem part1 {x : ℝ} : f x > 0 ↔ (x < -1 / 3 ∨ x > 3) := sorry

theorem part2 {m : ℝ} (h : ∃ x₀ : ℝ, f x₀ + 2 * m^2 < 4 * m) : -1 / 2 < m ∧ m < 5 / 2 := sorry

end NUMINAMATH_GPT_part1_part2_l1796_179627


namespace NUMINAMATH_GPT_pure_imaginary_b_eq_two_l1796_179655

theorem pure_imaginary_b_eq_two (b : ℝ) : (∃ (im_part : ℝ), (1 + b * Complex.I) / (2 - Complex.I) = im_part * Complex.I) ↔ b = 2 :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_b_eq_two_l1796_179655


namespace NUMINAMATH_GPT_minimum_omega_l1796_179631

theorem minimum_omega (ω : ℝ) (k : ℤ) (hω : ω > 0) 
  (h_symmetry : ∃ k : ℤ, ω * (π / 12) + π / 6 = k * π + π / 2) : ω = 4 :=
sorry

end NUMINAMATH_GPT_minimum_omega_l1796_179631


namespace NUMINAMATH_GPT_determine_solution_set_inequality_l1796_179615

-- Definitions based on given conditions
def quadratic_inequality_solution (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c > 0
def new_quadratic_inequality_solution (c b a : ℝ) (x : ℝ) := c * x^2 + b * x + a < 0

-- The proof statement
theorem determine_solution_set_inequality (a b c : ℝ):
  (∀ x : ℝ, -1/3 < x ∧ x < 2 → quadratic_inequality_solution a b c x) →
  (∀ x : ℝ, -3 < x ∧ x < 1/2 ↔ new_quadratic_inequality_solution c b a x) := sorry

end NUMINAMATH_GPT_determine_solution_set_inequality_l1796_179615


namespace NUMINAMATH_GPT_nth_monomial_correct_l1796_179678

-- Definitions of the sequence of monomials

def coeff (n : ℕ) : ℕ := 3 * n + 2
def exponent (n : ℕ) : ℕ := n

def nth_monomial (n : ℕ) (a : ℕ) : ℕ := (coeff n) * (a ^ (exponent n))

-- Theorem statement
theorem nth_monomial_correct (n : ℕ) (a : ℕ) : nth_monomial n a = (3 * n + 2) * (a ^ n) :=
by
  sorry

end NUMINAMATH_GPT_nth_monomial_correct_l1796_179678


namespace NUMINAMATH_GPT_max_value_fx_when_a_neg1_find_a_when_max_fx_is_neg3_inequality_gx_if_a_pos_l1796_179685

noncomputable def f (a x : ℝ) := a * x + Real.log x
noncomputable def g (a x : ℝ) := x * f a x
noncomputable def e := Real.exp 1

-- Statement for part (1)
theorem max_value_fx_when_a_neg1 : 
  ∀ x : ℝ, 0 < x → (f (-1) x ≤ f (-1) 1) :=
sorry

-- Statement for part (2)
theorem find_a_when_max_fx_is_neg3 : 
  (∀ x : ℝ, 0 < x ∧ x ≤ e → (f (-e^2) x ≤ -3)) →
  (∃ a : ℝ, a = -e^2) :=
sorry

-- Statement for part (3)
theorem inequality_gx_if_a_pos (a : ℝ) (hapos : 0 < a) 
  (x1 x2 : ℝ) (hxpos1 : 0 < x1) (hxpos2 : 0 < x2) (hx12 : x1 ≠ x2) :
  2 * g a ((x1 + x2) / 2) < g a x1 + g a x2 :=
sorry

end NUMINAMATH_GPT_max_value_fx_when_a_neg1_find_a_when_max_fx_is_neg3_inequality_gx_if_a_pos_l1796_179685


namespace NUMINAMATH_GPT_find_circle_center_l1796_179607

-- The statement to prove that the center of the given circle equation is (1, -2)
theorem find_circle_center : 
  ∃ (h k : ℝ), 3 * x^2 - 6 * x + 3 * y^2 + 12 * y - 75 = 0 → (h, k) = (1, -2) := 
by
  sorry

end NUMINAMATH_GPT_find_circle_center_l1796_179607


namespace NUMINAMATH_GPT_parts_of_alloys_l1796_179608

def ratio_of_metals_in_alloy (a1 a2 a3 b1 b2 : ℚ) (x y : ℚ) : Prop :=
  let first_metal := (1 / a3) * x + (a1 / b2) * y
  let second_metal := (2 / a3) * x + (b1 / b2) * y
  (first_metal / second_metal) = (17 / 27)

theorem parts_of_alloys
  (x y : ℚ)
  (a1 a2 a3 b1 b2 : ℚ)
  (h1 : a1 = 1)
  (h2 : a2 = 2)
  (h3 : a3 = 3)
  (h4 : b1 = 2)
  (h5 : b2 = 5)
  (h6 : ratio_of_metals_in_alloy a1 a2 a3 b1 b2 x y) :
  x = 9 ∧ y = 35 :=
sorry

end NUMINAMATH_GPT_parts_of_alloys_l1796_179608


namespace NUMINAMATH_GPT_sequence_an_formula_l1796_179635

theorem sequence_an_formula (a : ℕ → ℝ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, a (n + 1) = a n^2 - n * a n + 1) :
  ∀ n : ℕ, a n = n + 1 :=
sorry

end NUMINAMATH_GPT_sequence_an_formula_l1796_179635


namespace NUMINAMATH_GPT_geometric_sequence_arithmetic_Sn_l1796_179639

theorem geometric_sequence_arithmetic_Sn (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (a1 : ℝ) (n : ℕ) :
  (∀ n, a n = a1 * q ^ (n - 1)) →
  (∀ n, S n = a1 * (1 - q ^ n) / (1 - q)) →
  (∀ n, S (n + 1) - S n = S n - S (n - 1)) →
  q = 1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_arithmetic_Sn_l1796_179639


namespace NUMINAMATH_GPT_trajectory_of_P_l1796_179677

open Real

-- Definitions of points F1 and F2
def F1 : (ℝ × ℝ) := (-4, 0)
def F2 : (ℝ × ℝ) := (4, 0)

-- Definition of the condition on moving point P
def satisfies_condition (P : (ℝ × ℝ)) : Prop :=
  abs (dist P F2 - dist P F1) = 4

-- Definition of the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = 1 ∧ x ≤ -2

-- Theorem statement
theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, satisfies_condition P → ∃ x y : ℝ, P = (x, y) ∧ hyperbola_equation x y :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_P_l1796_179677


namespace NUMINAMATH_GPT_probability_of_passing_l1796_179657

theorem probability_of_passing (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
  sorry

end NUMINAMATH_GPT_probability_of_passing_l1796_179657


namespace NUMINAMATH_GPT_symmetric_circle_eq_l1796_179656

theorem symmetric_circle_eq {x y : ℝ} :
  (∃ x y : ℝ, (x+2)^2 + (y-1)^2 = 5) →
  (x - 1)^2 + (y + 2)^2 = 5 :=
sorry

end NUMINAMATH_GPT_symmetric_circle_eq_l1796_179656


namespace NUMINAMATH_GPT_find_a_and_b_l1796_179614

theorem find_a_and_b (a b : ℤ) (h1 : 3 * (b + a^2) = 99) (h2 : 3 * a * b^2 = 162) : a = 6 ∧ b = -3 :=
sorry

end NUMINAMATH_GPT_find_a_and_b_l1796_179614


namespace NUMINAMATH_GPT_evaluate_expression_l1796_179690

theorem evaluate_expression (k : ℤ): 
  2^(-(3*k+1)) - 2^(-(3*k-2)) + 2^(-(3*k)) - 2^(-(3*k+3)) = -((21:ℚ)/(8:ℚ)) * 2^(-(3*k)) := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1796_179690


namespace NUMINAMATH_GPT_y_increase_by_41_8_units_l1796_179688

theorem y_increase_by_41_8_units :
  ∀ (x y : ℝ),
    (∀ k : ℝ, y = 2 + k * 11 / 5 → x = 1 + k * 5) →
    x = 20 → y = 41.8 :=
by
  sorry

end NUMINAMATH_GPT_y_increase_by_41_8_units_l1796_179688


namespace NUMINAMATH_GPT_no_such_decreasing_h_exists_l1796_179605

-- Define the interval [0, ∞)
def nonneg_reals := {x : ℝ // 0 ≤ x}

-- Define a decreasing function h on [0, ∞)
def is_decreasing (h : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → h x ≥ h y

-- Define the function f based on h
def f (h : ℝ → ℝ) (x : ℝ) : ℝ := (x^2 - x + 1) * h x

-- Define the increasing property for f on [0, ∞)
def is_increasing_on_nonneg_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x ≤ f y

theorem no_such_decreasing_h_exists :
  ¬ ∃ h : ℝ → ℝ, is_decreasing h ∧ is_increasing_on_nonneg_reals (f h) :=
by sorry

end NUMINAMATH_GPT_no_such_decreasing_h_exists_l1796_179605


namespace NUMINAMATH_GPT_safe_travel_exists_l1796_179683

def total_travel_time : ℕ := 16
def first_crater_cycle : ℕ := 18
def first_crater_duration : ℕ := 1
def second_crater_cycle : ℕ := 10
def second_crater_duration : ℕ := 1

theorem safe_travel_exists : 
  ∃ t : ℕ, t ∈ { t | (∀ k : ℕ, t % first_crater_cycle ≠ k ∨ t % first_crater_cycle ≥ first_crater_duration) 
  ∧ (∀ k : ℕ, t % second_crater_cycle ≠ k ∨ t % second_crater_cycle ≥ second_crater_duration) 
  ∧ (∀ k : ℕ, (t + total_travel_time) % first_crater_cycle ≠ k ∨ (t + total_travel_time) % first_crater_cycle ≥ first_crater_duration) 
  ∧ (∀ k : ℕ, (t + total_travel_time) % second_crater_cycle ≠ k ∨ (t + total_travel_time) % second_crater_cycle ≥ second_crater_duration) } :=
sorry

end NUMINAMATH_GPT_safe_travel_exists_l1796_179683


namespace NUMINAMATH_GPT_a_pow_a_b_pow_b_c_pow_c_ge_one_l1796_179697

theorem a_pow_a_b_pow_b_c_pow_c_ge_one
    (a b c : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h : a + b + c = Real.rpow a (1/7) + Real.rpow b (1/7) + Real.rpow c (1/7)) :
    a^a * b^b * c^c ≥ 1 := 
by
  sorry

end NUMINAMATH_GPT_a_pow_a_b_pow_b_c_pow_c_ge_one_l1796_179697


namespace NUMINAMATH_GPT_older_brother_has_17_stamps_l1796_179643

def stamps_problem (y : ℕ) : Prop := y + (2 * y + 1) = 25

theorem older_brother_has_17_stamps (y : ℕ) (h : stamps_problem y) : 2 * y + 1 = 17 :=
by
  sorry

end NUMINAMATH_GPT_older_brother_has_17_stamps_l1796_179643


namespace NUMINAMATH_GPT_biology_exam_students_l1796_179654

theorem biology_exam_students :
  let students := 200
  let score_A := (1 / 4) * students
  let remaining_students := students - score_A
  let score_B := (1 / 5) * remaining_students
  let score_C := (1 / 3) * remaining_students
  let score_D := (5 / 12) * remaining_students
  let score_F := students - (score_A + score_B + score_C + score_D)
  let re_assessed_C := (3 / 5) * score_C
  let final_score_B := score_B + re_assessed_C
  let final_score_C := score_C - re_assessed_C
  score_A = 50 ∧ 
  final_score_B = 60 ∧ 
  final_score_C = 20 ∧ 
  score_D = 62 ∧ 
  score_F = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_biology_exam_students_l1796_179654


namespace NUMINAMATH_GPT_second_lady_distance_l1796_179620

theorem second_lady_distance (x : ℕ) 
  (h1 : ∃ y, y = 2 * x) 
  (h2 : x + 2 * x = 12) : x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_second_lady_distance_l1796_179620


namespace NUMINAMATH_GPT_max_k_l1796_179674

def seq (a : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = k * (a n) ^ 2 + 1

def bounded (a : ℕ → ℝ) (c : ℝ) : Prop :=
∀ n : ℕ, a n < c

theorem max_k (k : ℝ) (c : ℝ) (a : ℕ → ℝ) :
  a 1 = 1 →
  seq a k →
  bounded a c →
  0 < k ∧ k ≤ 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_k_l1796_179674


namespace NUMINAMATH_GPT_inequality_solution_l1796_179651

def solutionSetInequality (x : ℝ) : Prop :=
  (x > 1 ∨ x < -2)

theorem inequality_solution (x : ℝ) : 
  (x+2)/(x-1) > 0 ↔ solutionSetInequality x := 
  sorry

end NUMINAMATH_GPT_inequality_solution_l1796_179651


namespace NUMINAMATH_GPT_binomial_inequality_l1796_179660

theorem binomial_inequality (n : ℤ) (x : ℝ) (hn : n ≥ 2) (hx : |x| < 1) : 
  2^n > (1 - x)^n + (1 + x)^n := 
sorry

end NUMINAMATH_GPT_binomial_inequality_l1796_179660


namespace NUMINAMATH_GPT_multiply_658217_99999_l1796_179680

theorem multiply_658217_99999 : 658217 * 99999 = 65821034183 := 
by
  sorry

end NUMINAMATH_GPT_multiply_658217_99999_l1796_179680


namespace NUMINAMATH_GPT_mother_daughter_age_equality_l1796_179609

theorem mother_daughter_age_equality :
  ∀ (x : ℕ), (24 * 12 + 3) + x = 12 * ((-5 : ℤ) + x) → x = 32 := 
by
  intros x h
  sorry

end NUMINAMATH_GPT_mother_daughter_age_equality_l1796_179609


namespace NUMINAMATH_GPT_student_tickets_sold_l1796_179648

theorem student_tickets_sold (S NS : ℕ) (h1 : 9 * S + 11 * NS = 20960) (h2 : S + NS = 2000) : S = 520 :=
by
  sorry

end NUMINAMATH_GPT_student_tickets_sold_l1796_179648


namespace NUMINAMATH_GPT_tan_value_l1796_179629

variable (a : ℕ → ℝ) (b : ℕ → ℝ)
variable (a_geom : ∀ m n : ℕ, a m / a n = a (m - n))
variable (b_arith : ∃ c d : ℝ, ∀ n : ℕ, b n = c + n * d)
variable (ha : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
variable (hb : b 1 + b 6 + b 11 = 7 * Real.pi)

theorem tan_value : Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_value_l1796_179629


namespace NUMINAMATH_GPT_remaining_distance_l1796_179661

theorem remaining_distance (total_depth distance_traveled remaining_distance : ℕ) (h_total_depth : total_depth = 1218) 
  (h_distance_traveled : distance_traveled = 849) : remaining_distance = total_depth - distance_traveled := 
by
  sorry

end NUMINAMATH_GPT_remaining_distance_l1796_179661


namespace NUMINAMATH_GPT_shaded_region_area_l1796_179625

def area_of_square (side : ℕ) : ℕ := side * side

def area_of_triangle (base height : ℕ) : ℕ := (base * height) / 2

def combined_area_of_triangles (base height : ℕ) : ℕ := 2 * area_of_triangle base height

def shaded_area (square_side : ℕ) (triangle_base triangle_height : ℕ) : ℕ :=
  area_of_square square_side - combined_area_of_triangles triangle_base triangle_height

theorem shaded_region_area (h₁ : area_of_square 40 = 1600)
                          (h₂ : area_of_triangle 30 30 = 450)
                          (h₃ : combined_area_of_triangles 30 30 = 900) :
  shaded_area 40 30 30 = 700 :=
by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l1796_179625


namespace NUMINAMATH_GPT_total_students_l1796_179689

theorem total_students (h1 : 15 * 70 = 1050) 
                       (h2 : 10 * 95 = 950) 
                       (h3 : 1050 + 950 = 2000)
                       (h4 : 80 * N = 2000) :
  N = 25 :=
by sorry

end NUMINAMATH_GPT_total_students_l1796_179689


namespace NUMINAMATH_GPT_eval_expression_l1796_179698

theorem eval_expression (x y z : ℝ) (h1 : y > z) (h2 : z > 0) (h3 : x = y + z) : 
  ( (y+z+y)^z + (y+z+z)^y ) / (y^z + z^y) = 2^y + 2^z :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1796_179698


namespace NUMINAMATH_GPT_bottles_last_days_l1796_179675

theorem bottles_last_days :
  let total_bottles := 8066
  let bottles_per_day := 109
  total_bottles / bottles_per_day = 74 :=
by
  sorry

end NUMINAMATH_GPT_bottles_last_days_l1796_179675


namespace NUMINAMATH_GPT_value_of_f_at_2_l1796_179637

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem value_of_f_at_2 : f 2 = 62 :=
by
  -- The proof will be inserted here, it follows Horner's method steps shown in the solution
  sorry

end NUMINAMATH_GPT_value_of_f_at_2_l1796_179637


namespace NUMINAMATH_GPT_inequality_proof_l1796_179604

variable (a b c : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : c > 0)
variable (h4 : a + b + c = 1)

theorem inequality_proof : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1796_179604


namespace NUMINAMATH_GPT_david_english_marks_l1796_179628

def david_marks (math physics chemistry biology avg : ℕ) : ℕ :=
  avg * 5 - (math + physics + chemistry + biology)

theorem david_english_marks :
  let math := 95
  let physics := 82
  let chemistry := 97
  let biology := 95
  let avg := 93
  david_marks math physics chemistry biology avg = 96 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_david_english_marks_l1796_179628


namespace NUMINAMATH_GPT_find_y_intercept_l1796_179663

theorem find_y_intercept (m : ℝ) (x_intercept : ℝ × ℝ) (hx : x_intercept = (4, 0)) (hm : m = -3) : ∃ y_intercept : ℝ × ℝ, y_intercept = (0, 12) := 
by
  sorry

end NUMINAMATH_GPT_find_y_intercept_l1796_179663


namespace NUMINAMATH_GPT_brother_birth_year_1990_l1796_179668

variable (current_year : ℕ) -- Assuming the current year is implicit for the problem, it should be 2010 if Karina is 40 years old.
variable (karina_birth_year : ℕ)
variable (karina_current_age : ℕ)
variable (brother_current_age : ℕ)
variable (karina_twice_of_brother : Prop)

def karinas_brother_birth_year (karina_birth_year karina_current_age brother_current_age : ℕ) : ℕ :=
  karina_birth_year + brother_current_age

theorem brother_birth_year_1990 
  (h1 : karina_birth_year = 1970) 
  (h2 : karina_current_age = 40) 
  (h3 : karina_twice_of_brother) : 
  karinas_brother_birth_year 1970 40 20 = 1990 := 
by
  sorry

end NUMINAMATH_GPT_brother_birth_year_1990_l1796_179668


namespace NUMINAMATH_GPT_three_digit_number_is_11_times_sum_of_digits_l1796_179699

theorem three_digit_number_is_11_times_sum_of_digits :
    ∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ 
        (100 * a + 10 * b + c = 11 * (a + b + c)) ↔ 
        (100 * 1 + 10 * 9 + 8 = 11 * (1 + 9 + 8)) := 
by
    sorry

end NUMINAMATH_GPT_three_digit_number_is_11_times_sum_of_digits_l1796_179699


namespace NUMINAMATH_GPT_find_number_l1796_179626

theorem find_number (x : ℝ) (h : 140 = 3.5 * x) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1796_179626


namespace NUMINAMATH_GPT_odd_not_div_by_3_l1796_179691

theorem odd_not_div_by_3 (n : ℤ) (h1 : Odd n) (h2 : ¬ ∃ k : ℤ, n = 3 * k) : 6 ∣ (n^2 + 5) :=
  sorry

end NUMINAMATH_GPT_odd_not_div_by_3_l1796_179691


namespace NUMINAMATH_GPT_roots_of_quadratic_l1796_179610

theorem roots_of_quadratic (a b : ℝ) (h : ab ≠ 0) : 
  (a + b = -2 * b) ∧ (a * b = a) → (a = -3 ∧ b = 1) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_l1796_179610


namespace NUMINAMATH_GPT_exponent_division_l1796_179634

theorem exponent_division : (23 ^ 11) / (23 ^ 8) = 12167 := 
by {
  sorry
}

end NUMINAMATH_GPT_exponent_division_l1796_179634


namespace NUMINAMATH_GPT_tina_days_to_use_pink_pens_tina_total_pens_l1796_179636

-- Definitions based on the problem conditions.
def pink_pens : ℕ := 15
def green_pens : ℕ := pink_pens - 9
def blue_pens : ℕ := green_pens + 3
def total_pink_green := pink_pens + green_pens
def yellow_pens : ℕ := total_pink_green - 5
def pink_pens_per_day := 4

-- Prove the two statements based on the definitions.
theorem tina_days_to_use_pink_pens 
  (h1 : pink_pens = 15)
  (h2 : pink_pens_per_day = 4) :
  4 = 4 :=
by sorry

theorem tina_total_pens 
  (h1 : pink_pens = 15)
  (h2 : green_pens = pink_pens - 9)
  (h3 : blue_pens = green_pens + 3)
  (h4 : yellow_pens = total_pink_green - 5) :
  pink_pens + green_pens + blue_pens + yellow_pens = 46 :=
by sorry

end NUMINAMATH_GPT_tina_days_to_use_pink_pens_tina_total_pens_l1796_179636


namespace NUMINAMATH_GPT_original_dining_bill_l1796_179616

theorem original_dining_bill (B : ℝ) (h1 : B * 1.15 / 5 = 48.53) : B = 211 := 
sorry

end NUMINAMATH_GPT_original_dining_bill_l1796_179616


namespace NUMINAMATH_GPT_area_of_fifteen_sided_figure_l1796_179696

def point : Type := ℕ × ℕ

def vertices : List point :=
  [(1,1), (1,3), (3,5), (4,5), (5,4), (5,3), (6,3), (6,2), (5,1), (4,1), (3,2), (2,2), (1,1)]

def graph_paper_area (vs : List point) : ℚ :=
  -- Placeholder for actual area calculation logic
  -- The area for the provided vertices is found to be 11 cm^2.
  11

theorem area_of_fifteen_sided_figure : graph_paper_area vertices = 11 :=
by
  -- The actual proof would involve detailed steps to show that the area is indeed 11 cm^2
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_area_of_fifteen_sided_figure_l1796_179696


namespace NUMINAMATH_GPT_sum_geometric_sequence_l1796_179622

theorem sum_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (a1 : ℝ)
  (h1 : a 5 = -2) (h2 : a 8 = 16)
  (hq : q^3 = a 8 / a 5) (ha1 : a 1 = a1)
  (hS : S n = a1 * (1 - q^n) / (1 - q))
  : S 6 = 21 / 8 :=
sorry

end NUMINAMATH_GPT_sum_geometric_sequence_l1796_179622


namespace NUMINAMATH_GPT_correct_calculation_given_conditions_l1796_179676

variable (number : ℤ)

theorem correct_calculation_given_conditions 
  (h : number + 16 = 64) : number - 16 = 32 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_given_conditions_l1796_179676


namespace NUMINAMATH_GPT_distinct_digit_S_problem_l1796_179621

theorem distinct_digit_S_problem :
  ∃! (S : ℕ), S < 10 ∧ 
  ∃ (P Q R : ℕ), P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ S ∧ R ≠ S ∧ 
  P < 10 ∧ Q < 10 ∧ R < 10 ∧
  ((P + Q = S) ∨ (P + Q = S + 10)) ∧
  (R = 0) :=
sorry

end NUMINAMATH_GPT_distinct_digit_S_problem_l1796_179621


namespace NUMINAMATH_GPT_price_of_other_stamp_l1796_179650

-- Define the conditions
def total_stamps : ℕ := 75
def total_value_cents : ℕ := 480
def known_stamp_price : ℕ := 8
def known_stamp_count : ℕ := 40
def unknown_stamp_count : ℕ := total_stamps - known_stamp_count

-- The problem to solve
theorem price_of_other_stamp (x : ℕ) :
  (known_stamp_count * known_stamp_price) + (unknown_stamp_count * x) = total_value_cents → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_price_of_other_stamp_l1796_179650


namespace NUMINAMATH_GPT_bunny_burrows_l1796_179601

theorem bunny_burrows (x : ℕ) (h1 : 20 * x * 600 = 36000) : x = 3 :=
by
  -- Skipping proof using sorry
  sorry

end NUMINAMATH_GPT_bunny_burrows_l1796_179601


namespace NUMINAMATH_GPT_eq_of_divides_l1796_179653

theorem eq_of_divides (a b : ℕ) (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b :=
sorry

end NUMINAMATH_GPT_eq_of_divides_l1796_179653


namespace NUMINAMATH_GPT_complete_the_square_result_l1796_179612

-- Define the equation
def initial_eq (x : ℝ) : Prop := x^2 + 4 * x + 3 = 0

-- State the theorem based on the condition and required to prove the question equals the answer
theorem complete_the_square_result (x : ℝ) : initial_eq x → (x + 2) ^ 2 = 1 := 
by
  intro h
  -- Proof is to be skipped
  sorry

end NUMINAMATH_GPT_complete_the_square_result_l1796_179612


namespace NUMINAMATH_GPT_rational_functional_equation_l1796_179686

theorem rational_functional_equation (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + f y) = f x + y) :
  (f = λ x => x) ∨ (f = λ x => -x) :=
by
  sorry

end NUMINAMATH_GPT_rational_functional_equation_l1796_179686


namespace NUMINAMATH_GPT_max_sum_of_xj4_minus_xj5_l1796_179682

theorem max_sum_of_xj4_minus_xj5 (n : ℕ) (x : Fin n → ℝ) 
  (hx : ∀ i, 0 ≤ x i) 
  (h_sum : (Finset.univ.sum x) = 1) : 
  (Finset.univ.sum (λ j => (x j)^4 - (x j)^5)) ≤ 1 / 12 :=
sorry

end NUMINAMATH_GPT_max_sum_of_xj4_minus_xj5_l1796_179682


namespace NUMINAMATH_GPT_proof_problem_l1796_179613

theorem proof_problem (x : ℝ) (a : ℝ) :
  (0 < x) → 
  (x + 1 / x ≥ 2) →
  (x + 4 / x^2 ≥ 3) →
  (x + 27 / x^3 ≥ 4) →
  a = 4^4 → 
  x + a / x^4 ≥ 5 :=
  sorry

end NUMINAMATH_GPT_proof_problem_l1796_179613


namespace NUMINAMATH_GPT_min_value_of_expression_l1796_179681

theorem min_value_of_expression (x y z : ℝ) : ∃ a : ℝ, (∀ x y z : ℝ, x^2 + x * y + y^2 + y * z + z^2 ≥ a) ∧ (a = 0) :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1796_179681


namespace NUMINAMATH_GPT_quadratic_inequality_l1796_179666

theorem quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x + 4 ≥ 0) ↔ 0 ≤ k ∧ k ≤ 16 :=
by sorry

end NUMINAMATH_GPT_quadratic_inequality_l1796_179666


namespace NUMINAMATH_GPT_promotional_codes_one_tenth_l1796_179687

open Nat

def promotional_chars : List Char := ['C', 'A', 'T', '3', '1', '1', '9']

def count_promotional_codes (chars : List Char) (len : Nat) : Nat := sorry

theorem promotional_codes_one_tenth : count_promotional_codes promotional_chars 5 / 10 = 60 :=
by 
  sorry

end NUMINAMATH_GPT_promotional_codes_one_tenth_l1796_179687


namespace NUMINAMATH_GPT_least_multiple_of_11_not_lucky_l1796_179647

-- Define what it means for a number to be a lucky integer
def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

-- Define what it means for a number to be a multiple of 11
def is_multiple_of_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- State the problem: the least positive multiple of 11 that is not a lucky integer is 132
theorem least_multiple_of_11_not_lucky :
  ∃ n : ℕ, is_multiple_of_11 n ∧ ¬ is_lucky n ∧ n = 132 :=
sorry

end NUMINAMATH_GPT_least_multiple_of_11_not_lucky_l1796_179647


namespace NUMINAMATH_GPT_man_l1796_179606

-- Constants and conditions
def V_down : ℝ := 18  -- downstream speed in km/hr
def V_c : ℝ := 3.4    -- speed of the current in km/hr

-- Main statement to prove
theorem man's_speed_against_the_current : (V_down - V_c - V_c) = 11.2 := by
  sorry

end NUMINAMATH_GPT_man_l1796_179606


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l1796_179662

-- Problem 1
theorem system1_solution (x y : ℝ) (h1 : 3 * x - 2 * y = 6) (h2 : 2 * x + 3 * y = 17) : 
  x = 4 ∧ y = 3 :=
by {
  sorry
}

-- Problem 2
theorem system2_solution (x y : ℝ) (h1 : x + 4 * y = 14) 
  (h2 : (x - 3) / 4 - (y - 3) / 3 = 1 / 12) : 
  x = 3 ∧ y = 11 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_system1_solution_system2_solution_l1796_179662


namespace NUMINAMATH_GPT_kaleb_initial_cherries_l1796_179673

/-- Kaleb's initial number of cherries -/
def initial_cherries : ℕ := 67

/-- Cherries that Kaleb ate -/
def eaten_cherries : ℕ := 25

/-- Cherries left after eating -/
def left_cherries : ℕ := 42

/-- Prove that the initial number of cherries is 67 given the conditions. -/
theorem kaleb_initial_cherries :
  eaten_cherries + left_cherries = initial_cherries :=
by
  sorry

end NUMINAMATH_GPT_kaleb_initial_cherries_l1796_179673


namespace NUMINAMATH_GPT_geom_seq_common_ratio_l1796_179659

theorem geom_seq_common_ratio (a1 : ℤ) (S3 : ℚ) (q : ℚ) (hq : -2 * (1 + q + q^2) = - (7 / 2)) : 
  q = 1 / 2 ∨ q = -3 / 2 :=
sorry

end NUMINAMATH_GPT_geom_seq_common_ratio_l1796_179659


namespace NUMINAMATH_GPT_johns_number_l1796_179602

theorem johns_number (n : ℕ) (h1 : ∃ k₁ : ℤ, n = 125 * k₁) (h2 : ∃ k₂ : ℤ, n = 180 * k₂) (h3 : 1000 < n) (h4 : n < 3000) : n = 1800 :=
sorry

end NUMINAMATH_GPT_johns_number_l1796_179602


namespace NUMINAMATH_GPT_value_of_fraction_l1796_179649

theorem value_of_fraction (y : ℝ) (h : 4 - 9 / y + 9 / (y^2) = 0) : 3 / y = 2 :=
sorry

end NUMINAMATH_GPT_value_of_fraction_l1796_179649


namespace NUMINAMATH_GPT_two_circles_with_tangents_l1796_179640

theorem two_circles_with_tangents
  (a b : ℝ)                -- radii of the circles
  (length_PQ length_AB : ℝ) -- lengths of the tangents PQ and AB
  (h1 : length_PQ = 14)     -- condition: length of PQ is 14
  (h2 : length_AB = 16)     -- condition: length of AB is 16
  (h3 : length_AB^2 + (a - b)^2 = length_PQ^2 + (a + b)^2) -- from the Pythagorean theorem
  : a * b = 15 := 
sorry

end NUMINAMATH_GPT_two_circles_with_tangents_l1796_179640


namespace NUMINAMATH_GPT_negation_of_existence_l1796_179667

theorem negation_of_existence :
  ¬(∃ x : ℝ, x^2 + 2 * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_l1796_179667


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1796_179617

theorem sufficient_not_necessary (a : ℝ) :
  a > 1 → (a^2 > 1) ∧ (∀ a : ℝ, a^2 > 1 → a = -1 ∨ a > 1 → false) :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_not_necessary_l1796_179617


namespace NUMINAMATH_GPT_total_pages_l1796_179603

theorem total_pages (x : ℕ) (h : 9 + 180 + 3 * (x - 99) = 1392) : x = 500 :=
by
  sorry

end NUMINAMATH_GPT_total_pages_l1796_179603


namespace NUMINAMATH_GPT_deposit_correct_l1796_179624

-- Define the conditions
def monthly_income : ℝ := 10000
def deposit_percentage : ℝ := 0.25

-- Define the deposit calculation based on the conditions
def deposit_amount (income : ℝ) (percentage : ℝ) : ℝ :=
  percentage * income

-- Theorem: Prove that the deposit amount is Rs. 2500
theorem deposit_correct :
    deposit_amount monthly_income deposit_percentage = 2500 :=
  sorry

end NUMINAMATH_GPT_deposit_correct_l1796_179624


namespace NUMINAMATH_GPT_probability_exactly_one_red_ball_l1796_179664

-- Define the given conditions
def total_balls : ℕ := 10
def red_balls : ℕ := 3
def children : ℕ := 10

-- Define the question and calculate the probability
theorem probability_exactly_one_red_ball : 
  (3 * (3 / 10) * ((7 / 10) * (7 / 10))) = 0.441 := 
by 
  sorry

end NUMINAMATH_GPT_probability_exactly_one_red_ball_l1796_179664


namespace NUMINAMATH_GPT_triangle_acd_area_l1796_179694

noncomputable def area_of_triangle : ℝ := sorry

theorem triangle_acd_area (AB CD : ℝ) (h : CD = 3 * AB) (area_trapezoid: ℝ) (h1: area_trapezoid = 20) :
  area_of_triangle = 15 := 
sorry

end NUMINAMATH_GPT_triangle_acd_area_l1796_179694


namespace NUMINAMATH_GPT_perfect_square_of_division_l1796_179611

theorem perfect_square_of_division (a b : ℤ) (ha : 0 < a) (hb : 0 < b) 
  (h : (a * b + 1) ∣ (a^2 + b^2)) : ∃ k : ℤ, 0 < k ∧ k^2 = (a^2 + b^2) / (a * b + 1) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_of_division_l1796_179611


namespace NUMINAMATH_GPT_total_cases_l1796_179630

-- Define the number of boys' high schools and girls' high schools
def boys_high_schools : Nat := 4
def girls_high_schools : Nat := 3

-- Theorem to be proven
theorem total_cases (B G : Nat) (hB : B = boys_high_schools) (hG : G = girls_high_schools) : 
  B + G = 7 :=
by
  rw [hB, hG]
  exact rfl

end NUMINAMATH_GPT_total_cases_l1796_179630


namespace NUMINAMATH_GPT_other_team_scored_l1796_179684

open Nat

def points_liz_scored (free_throws three_pointers jump_shots : Nat) : Nat :=
  free_throws * 1 + three_pointers * 3 + jump_shots * 2

def points_deficit := 20
def points_liz_deficit := points_liz_scored 5 3 4 - points_deficit
def final_loss_margin := 8
def other_team_score := points_liz_scored 5 3 4 + final_loss_margin

theorem other_team_scored
  (points_liz : Nat := points_liz_scored 5 3 4)
  (final_deficit : Nat := points_deficit)
  (final_margin : Nat := final_loss_margin)
  (other_team_points : Nat := other_team_score) :
  other_team_points = 30 := 
sorry

end NUMINAMATH_GPT_other_team_scored_l1796_179684


namespace NUMINAMATH_GPT_new_selling_price_l1796_179618

theorem new_selling_price (C : ℝ) (h1 : 1.10 * C = 88) :
  1.15 * C = 92 :=
sorry

end NUMINAMATH_GPT_new_selling_price_l1796_179618


namespace NUMINAMATH_GPT_sum_of_first_15_terms_of_arithmetic_sequence_l1796_179619

theorem sum_of_first_15_terms_of_arithmetic_sequence 
  (a d : ℕ) 
  (h1 : (5 * (2 * a + 4 * d)) / 2 = 10) 
  (h2 : (10 * (2 * a + 9 * d)) / 2 = 50) :
  (15 * (2 * a + 14 * d)) / 2 = 120 :=
sorry

end NUMINAMATH_GPT_sum_of_first_15_terms_of_arithmetic_sequence_l1796_179619


namespace NUMINAMATH_GPT_salesperson_commission_l1796_179692

noncomputable def commission (sale_price : ℕ) (rate : ℚ) : ℚ :=
  rate * sale_price

noncomputable def total_commission (machines_sold : ℕ) (first_rate : ℚ) (second_rate : ℚ) (sale_price : ℕ) : ℚ :=
  let first_commission := commission sale_price first_rate * 100
  let second_commission := commission sale_price second_rate * (machines_sold - 100)
  first_commission + second_commission

theorem salesperson_commission :
  total_commission 130 0.03 0.04 10000 = 42000 := by
  sorry

end NUMINAMATH_GPT_salesperson_commission_l1796_179692


namespace NUMINAMATH_GPT_short_trees_after_planting_l1796_179693

-- Defining the conditions as Lean definitions
def current_short_trees : Nat := 3
def newly_planted_short_trees : Nat := 9

-- Defining the question (assertion to prove) with the expected answer
theorem short_trees_after_planting : current_short_trees + newly_planted_short_trees = 12 := by
  sorry

end NUMINAMATH_GPT_short_trees_after_planting_l1796_179693


namespace NUMINAMATH_GPT_building_height_l1796_179600

theorem building_height (h : ℕ) 
  (flagpole_height flagpole_shadow building_shadow : ℕ)
  (h_flagpole : flagpole_height = 18)
  (s_flagpole : flagpole_shadow = 45)
  (s_building : building_shadow = 70) 
  (condition : flagpole_height / flagpole_shadow = h / building_shadow) :
  h = 28 := by
  sorry

end NUMINAMATH_GPT_building_height_l1796_179600
