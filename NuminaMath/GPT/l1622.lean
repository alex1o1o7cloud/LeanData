import Mathlib

namespace NUMINAMATH_GPT_maximize_S_n_l1622_162207

-- Define the general term of the sequence and the sum of the first n terms.
def a_n (n : ℕ) : ℤ := -2 * n + 25

def S_n (n : ℕ) : ℤ := 24 * n - n^2

-- The main statement to prove
theorem maximize_S_n : ∃ (n : ℕ), n = 11 ∧ ∀ m, S_n m ≤ S_n 11 :=
  sorry

end NUMINAMATH_GPT_maximize_S_n_l1622_162207


namespace NUMINAMATH_GPT_range_of_a_l1622_162202

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, 2 * a * (x : ℝ)^2 - 4 * (x : ℝ) < a * (x : ℝ) - 2 → ∃! x₀ : ℤ, x₀ = x) → 1 ≤ a ∧ a < 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1622_162202


namespace NUMINAMATH_GPT_find_oranges_to_put_back_l1622_162281

theorem find_oranges_to_put_back (A O x : ℕ) (h₁ : A + O = 15) (h₂ : 40 * A + 60 * O = 720) (h₃ : (360 + 360 - 60 * x) / (15 - x) = 45) : x = 3 := by
  sorry

end NUMINAMATH_GPT_find_oranges_to_put_back_l1622_162281


namespace NUMINAMATH_GPT_sqrt8_same_type_as_sqrt2_l1622_162265

theorem sqrt8_same_type_as_sqrt2 :
  (∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 8) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 4) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 6) ∧
  ¬(∃ a : ℚ, a * Real.sqrt 2 = Real.sqrt 10) :=
by
  sorry

end NUMINAMATH_GPT_sqrt8_same_type_as_sqrt2_l1622_162265


namespace NUMINAMATH_GPT_value_of_a_l1622_162233

theorem value_of_a (a : ℝ) (A B : ℝ × ℝ) (hA : A = (a - 2, 2 * a + 7)) (hB : B = (1, 5)) (h_parallel : (A.1 = B.1)) : a = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_a_l1622_162233


namespace NUMINAMATH_GPT_sequence_value_x_l1622_162247

theorem sequence_value_x (a1 a2 a3 a4 a5 a6 : ℕ) 
  (h1 : a1 = 2) 
  (h2 : a2 = 5) 
  (h3 : a3 = 11) 
  (h4 : a4 = 20) 
  (h5 : a6 = 47)
  (h6 : a2 - a1 = 3) 
  (h7 : a3 - a2 = 6) 
  (h8 : a4 - a3 = 9) 
  (h9 : a6 - a5 = 15) : 
  a5 = 32 :=
sorry

end NUMINAMATH_GPT_sequence_value_x_l1622_162247


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1622_162273

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 - 6 * x + k = 0) ↔ k < 9 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1622_162273


namespace NUMINAMATH_GPT_q_true_or_false_l1622_162269

variable (p q : Prop)

theorem q_true_or_false (h1 : ¬ (p ∧ q)) (h2 : ¬ p) : q ∨ ¬ q :=
by
  sorry

end NUMINAMATH_GPT_q_true_or_false_l1622_162269


namespace NUMINAMATH_GPT_initial_roses_in_vase_l1622_162226

theorem initial_roses_in_vase (current_roses : ℕ) (added_roses : ℕ) (total_garden_roses : ℕ) (initial_roses : ℕ) :
  current_roses = 20 → added_roses = 13 → total_garden_roses = 59 → initial_roses = current_roses - added_roses → 
  initial_roses = 7 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  sorry

end NUMINAMATH_GPT_initial_roses_in_vase_l1622_162226


namespace NUMINAMATH_GPT_no_solution_exists_l1622_162216

theorem no_solution_exists (f : ℝ → ℝ) :
  ¬ (∀ x y : ℝ, f (f x + 2 * y) = 3 * x + f (f (f y) - x)) :=
sorry

end NUMINAMATH_GPT_no_solution_exists_l1622_162216


namespace NUMINAMATH_GPT_sport_tournament_attendance_l1622_162298

theorem sport_tournament_attendance :
  let total_attendance := 500
  let team_A_supporters := 0.35 * total_attendance
  let team_B_supporters := 0.25 * total_attendance
  let team_C_supporters := 0.20 * total_attendance
  let team_D_supporters := 0.15 * total_attendance
  let AB_overlap := 0.10 * team_A_supporters
  let BC_overlap := 0.05 * team_B_supporters
  let CD_overlap := 0.07 * team_C_supporters
  let atmosphere_attendees := 30
  let total_supporters := team_A_supporters + team_B_supporters + team_C_supporters + team_D_supporters
                         - (AB_overlap + BC_overlap + CD_overlap)
  let unsupported_people := total_attendance - total_supporters - atmosphere_attendees
  unsupported_people = 26 :=
by
  sorry

end NUMINAMATH_GPT_sport_tournament_attendance_l1622_162298


namespace NUMINAMATH_GPT_math_proof_l1622_162278

noncomputable def math_problem (x : ℝ) : ℝ :=
  (3 / (2 * x) * (1 / 2) * (2 / 5) * 5020) - ((2 ^ 3) * (1 / (3 * x + 2)) * 250) + Real.sqrt (900 / x)

theorem math_proof :
  math_problem 4 = 60.393 :=
by
  sorry

end NUMINAMATH_GPT_math_proof_l1622_162278


namespace NUMINAMATH_GPT_find_interest_rate_of_first_investment_l1622_162268

noncomputable def total_interest : ℚ := 73
noncomputable def interest_rate_7_percent : ℚ := 0.07
noncomputable def invested_400 : ℚ := 400
noncomputable def interest_7_percent := invested_400 * interest_rate_7_percent
noncomputable def interest_first_investment := total_interest - interest_7_percent
noncomputable def invested_first : ℚ := invested_400 - 100
noncomputable def interest_first : ℚ := 45  -- calculated as total_interest - interest_7_percent

theorem find_interest_rate_of_first_investment (r : ℚ) :
  interest_first = invested_first * r * 1 → 
  r = 0.15 :=
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_of_first_investment_l1622_162268


namespace NUMINAMATH_GPT_no_such_coins_l1622_162280

theorem no_such_coins (p1 p2 : ℝ) (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1)
  (cond1 : (1 - p1) * (1 - p2) = p1 * p2)
  (cond2 : p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :
  false :=
  sorry

end NUMINAMATH_GPT_no_such_coins_l1622_162280


namespace NUMINAMATH_GPT_gwen_total_books_l1622_162206

def mystery_shelves : Nat := 6
def mystery_books_per_shelf : Nat := 7

def picture_shelves : Nat := 4
def picture_books_per_shelf : Nat := 5

def biography_shelves : Nat := 3
def biography_books_per_shelf : Nat := 3

def scifi_shelves : Nat := 2
def scifi_books_per_shelf : Nat := 9

theorem gwen_total_books :
    (mystery_books_per_shelf * mystery_shelves) +
    (picture_books_per_shelf * picture_shelves) +
    (biography_books_per_shelf * biography_shelves) +
    (scifi_books_per_shelf * scifi_shelves) = 89 := 
by 
    sorry

end NUMINAMATH_GPT_gwen_total_books_l1622_162206


namespace NUMINAMATH_GPT_find_ratio_of_d1_and_d2_l1622_162257

theorem find_ratio_of_d1_and_d2
  (x y d1 d2 : ℝ)
  (h1 : x + 4 * d1 = y)
  (h2 : x + 5 * d2 = y)
  (h3 : d1 ≠ 0)
  (h4 : d2 ≠ 0) :
  d1 / d2 = 5 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_ratio_of_d1_and_d2_l1622_162257


namespace NUMINAMATH_GPT_intersection_complement_l1622_162254

open Set

variable {α : Type*}
noncomputable def A : Set ℝ := {x | x^2 ≥ 1}
noncomputable def B : Set ℝ := {x | (x - 2) / x ≤ 0}

theorem intersection_complement :
  A ∩ (compl B) = (Iic (-1)) ∪ (Ioi 2) := by
sorry

end NUMINAMATH_GPT_intersection_complement_l1622_162254


namespace NUMINAMATH_GPT_height_relationship_l1622_162299

theorem height_relationship
  (r1 h1 r2 h2 : ℝ)
  (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relation : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
sorry

end NUMINAMATH_GPT_height_relationship_l1622_162299


namespace NUMINAMATH_GPT_remaining_numbers_l1622_162260

-- Define the problem statement in Lean 4
theorem remaining_numbers (S S5 S3 : ℝ) (A3 : ℝ) 
  (h1 : S / 8 = 20) 
  (h2 : S5 / 5 = 12) 
  (h3 : S3 = S - S5) 
  (h4 : A3 = 100 / 3) : 
  S3 / A3 = 3 :=
sorry

end NUMINAMATH_GPT_remaining_numbers_l1622_162260


namespace NUMINAMATH_GPT_students_receiving_B_lee_l1622_162218

def num_students_receiving_B (students_kipling: ℕ) (B_kipling: ℕ) (students_lee: ℕ) : ℕ :=
  let ratio := (B_kipling * students_lee) / students_kipling
  ratio

theorem students_receiving_B_lee (students_kipling B_kipling students_lee : ℕ) 
  (h : B_kipling = 8 ∧ students_kipling = 12 ∧ students_lee = 30) :
  num_students_receiving_B students_kipling B_kipling students_lee = 20 :=
by
  sorry

end NUMINAMATH_GPT_students_receiving_B_lee_l1622_162218


namespace NUMINAMATH_GPT_range_of_a_l1622_162261

theorem range_of_a (a : ℝ) :
  (∀ (x y : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ y ∧ y ≤ 4 → 2 * x^2 - 2 * a * x * y + y^2 ≥ 0) →
  a ≤ Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1622_162261


namespace NUMINAMATH_GPT_lcm_48_180_l1622_162212

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  -- Here follows the proof, which is omitted
  sorry

end NUMINAMATH_GPT_lcm_48_180_l1622_162212


namespace NUMINAMATH_GPT_pier_to_village_trip_l1622_162256

theorem pier_to_village_trip :
  ∃ (x t : ℝ), 
  (x / 10 + x / 8 = t + 1 / 60) ∧
  (5 * t / 2 + 4 * t / 2 = x) ∧
  (x = 6) ∧
  (t = 4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_pier_to_village_trip_l1622_162256


namespace NUMINAMATH_GPT_narrow_black_stripes_count_l1622_162266

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end NUMINAMATH_GPT_narrow_black_stripes_count_l1622_162266


namespace NUMINAMATH_GPT_opposite_of_neg_five_l1622_162210

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_five_l1622_162210


namespace NUMINAMATH_GPT_vector_expression_l1622_162282

variables (a b c : ℝ × ℝ)
variables (m n : ℝ)

noncomputable def vec_a : ℝ × ℝ := (1, 1)
noncomputable def vec_b : ℝ × ℝ := (1, -1)
noncomputable def vec_c : ℝ × ℝ := (-1, 2)

/-- Prove that vector c can be expressed in terms of vectors a and b --/
theorem vector_expression : 
  vec_c = m • vec_a + n • vec_b → (m = 1/2 ∧ n = -3/2) :=
sorry

end NUMINAMATH_GPT_vector_expression_l1622_162282


namespace NUMINAMATH_GPT_total_students_in_class_l1622_162267

theorem total_students_in_class
  (S : ℕ)
  (H1 : 5/8 * S = S - 60)
  (H2 : 60 = 3/8 * S) :
  S = 160 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_class_l1622_162267


namespace NUMINAMATH_GPT_diameter_of_circle_l1622_162223

theorem diameter_of_circle (A : ℝ) (h : A = 100 * Real.pi) : ∃ d : ℝ, d = 20 :=
by
  sorry

end NUMINAMATH_GPT_diameter_of_circle_l1622_162223


namespace NUMINAMATH_GPT_algebra_expression_value_l1622_162228

variable (x : ℝ)

theorem algebra_expression_value (h : x^2 - 3 * x - 12 = 0) : 3 * x^2 - 9 * x + 5 = 41 := 
sorry

end NUMINAMATH_GPT_algebra_expression_value_l1622_162228


namespace NUMINAMATH_GPT_perimeter_of_square_from_quadratic_roots_l1622_162230

theorem perimeter_of_square_from_quadratic_roots :
  let r1 := 1
  let r2 := 10
  let larger_root := if r1 > r2 then r1 else r2
  let area := larger_root * larger_root
  let side_length := Real.sqrt area
  4 * side_length = 40 := by
  let r1 := 1
  let r2 := 10
  let larger_root := if r1 > r2 then r1 else r2
  let area := larger_root * larger_root
  let side_length := Real.sqrt area
  sorry

end NUMINAMATH_GPT_perimeter_of_square_from_quadratic_roots_l1622_162230


namespace NUMINAMATH_GPT_rectangular_block_height_l1622_162283

theorem rectangular_block_height (l w h : ℕ) 
  (volume_eq : l * w * h = 42) 
  (perimeter_eq : 2 * l + 2 * w = 18) : 
  h = 3 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_block_height_l1622_162283


namespace NUMINAMATH_GPT_cutting_stick_ways_l1622_162245

theorem cutting_stick_ways :
  ∃ (s : Finset (ℕ × ℕ)), 
  (∀ a ∈ s, 2 * a.1 + 3 * a.2 = 14) ∧
  s.card = 2 := 
by
  sorry

end NUMINAMATH_GPT_cutting_stick_ways_l1622_162245


namespace NUMINAMATH_GPT_find_k_of_inverse_proportion_l1622_162276

theorem find_k_of_inverse_proportion (k x y : ℝ) (h : y = k / x) (hx : x = 2) (hy : y = 6) : k = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_k_of_inverse_proportion_l1622_162276


namespace NUMINAMATH_GPT_sin_30_is_half_l1622_162259

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_30_is_half_l1622_162259


namespace NUMINAMATH_GPT_nth_equation_l1622_162294

theorem nth_equation (n : ℕ) (hn : n > 0) : 9 * n + (n - 1) = 10 * n - 1 :=
sorry

end NUMINAMATH_GPT_nth_equation_l1622_162294


namespace NUMINAMATH_GPT_number_of_extreme_points_l1622_162290

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + 3 * x^2 + 4 * x - a

theorem number_of_extreme_points (a : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + 6 * x + 4) > 0) →
  0 = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_extreme_points_l1622_162290


namespace NUMINAMATH_GPT_average_monthly_growth_rate_correct_l1622_162264

theorem average_monthly_growth_rate_correct:
  (∃ x : ℝ, 30000 * (1 + x)^2 = 36300) ↔ 3 * (1 + x)^2 = 3.63 := 
by {
  sorry -- proof placeholder
}

end NUMINAMATH_GPT_average_monthly_growth_rate_correct_l1622_162264


namespace NUMINAMATH_GPT_pages_per_day_l1622_162211

variable (P : ℕ) (D : ℕ)

theorem pages_per_day (hP : P = 66) (hD : D = 6) : P / D = 11 :=
by
  sorry

end NUMINAMATH_GPT_pages_per_day_l1622_162211


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1622_162249

def p (x : ℝ) : Prop := |x - 4| > 2
def q (x : ℝ) : Prop := x > 1

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 6 → x > 1) ∧ ¬(∀ x, x > 1 → 2 ≤ x ∧ x ≤ 6) :=
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1622_162249


namespace NUMINAMATH_GPT_largest_integral_solution_l1622_162205

theorem largest_integral_solution (x : ℤ) : (1 / 4 : ℝ) < (x / 7 : ℝ) ∧ (x / 7 : ℝ) < (3 / 5 : ℝ) → x = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_integral_solution_l1622_162205


namespace NUMINAMATH_GPT_cos_90_eq_zero_l1622_162239

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end NUMINAMATH_GPT_cos_90_eq_zero_l1622_162239


namespace NUMINAMATH_GPT_abs_eq_implies_y_eq_half_l1622_162279

theorem abs_eq_implies_y_eq_half (y : ℝ) (h : |y - 3| = |y + 2|) : y = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_abs_eq_implies_y_eq_half_l1622_162279


namespace NUMINAMATH_GPT_union_complement_eq_l1622_162284

/-- The universal set U and sets A and B as given in the problem. -/
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

/-- The lean statement of our proof problem. -/
theorem union_complement_eq : A ∪ (U \ B) = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_GPT_union_complement_eq_l1622_162284


namespace NUMINAMATH_GPT_toy_swords_count_l1622_162296

variable (s : ℕ)

def cost_lego := 250
def cost_toy_sword := 120
def cost_play_dough := 35

def total_cost (s : ℕ) :=
  3 * cost_lego + s * cost_toy_sword + 10 * cost_play_dough

theorem toy_swords_count : total_cost s = 1940 → s = 7 := by
  sorry

end NUMINAMATH_GPT_toy_swords_count_l1622_162296


namespace NUMINAMATH_GPT_common_area_of_rectangle_and_circle_l1622_162229

theorem common_area_of_rectangle_and_circle (r : ℝ) (a b : ℝ) (h_center : r = 5) (h_dim : a = 10 ∧ b = 4) :
  let sector_area := (25 * Real.pi) / 2 
  let triangle_area := 4 * Real.sqrt 21 
  let result := sector_area + triangle_area 
  result = (25 * Real.pi) / 2 + 4 * Real.sqrt 21 := 
by
  sorry

end NUMINAMATH_GPT_common_area_of_rectangle_and_circle_l1622_162229


namespace NUMINAMATH_GPT_no_non_trivial_power_ending_222_l1622_162203

theorem no_non_trivial_power_ending_222 (x y : ℕ) (hx : x > 1) (hy : y > 1) : ¬ (∃ n : ℕ, n % 1000 = 222 ∧ n = x^y) :=
by
  sorry

end NUMINAMATH_GPT_no_non_trivial_power_ending_222_l1622_162203


namespace NUMINAMATH_GPT_equivalent_problem_l1622_162262

-- Definitions that correspond to conditions
def valid_n (n : ℕ) : Prop := n < 13 ∧ (4 * n) % 13 = 1

-- The equivalent proof problem
theorem equivalent_problem (n : ℕ) (h : valid_n n) : ((3 ^ n) ^ 4 - 3) % 13 = 6 := by
  sorry

end NUMINAMATH_GPT_equivalent_problem_l1622_162262


namespace NUMINAMATH_GPT_fraction_identity_l1622_162286

theorem fraction_identity (a b : ℚ) (h : a / b = 3 / 4) : (b - a) / b = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l1622_162286


namespace NUMINAMATH_GPT_larger_number_is_34_l1622_162242

theorem larger_number_is_34 (x y : ℕ) (h1 : x + y = 56) (h2 : y = x + 12) : y = 34 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_34_l1622_162242


namespace NUMINAMATH_GPT_circuit_length_is_365_l1622_162240

-- Definitions based on given conditions
def runs_morning := 7
def runs_afternoon := 3
def total_distance_week := 25550
def total_runs_day := runs_morning + runs_afternoon
def total_runs_week := total_runs_day * 7

-- Statement of the problem to be proved
theorem circuit_length_is_365 :
  total_distance_week / total_runs_week = 365 :=
sorry

end NUMINAMATH_GPT_circuit_length_is_365_l1622_162240


namespace NUMINAMATH_GPT_win_sector_area_l1622_162272

noncomputable def radius : ℝ := 8
noncomputable def probability : ℝ := 1 / 4
noncomputable def total_area : ℝ := Real.pi * radius^2

theorem win_sector_area :
  ∃ (W : ℝ), W = probability * total_area ∧ W = 16 * Real.pi :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_win_sector_area_l1622_162272


namespace NUMINAMATH_GPT_returns_to_start_point_after_fourth_passenger_distance_after_last_passenger_total_earnings_l1622_162285

noncomputable def driving_distances : List ℤ := [-5, 3, 6, -4, 7, -2]

def fare (distance : ℕ) : ℕ :=
  if distance ≤ 3 then 8 else 8 + 2 * (distance - 3)

theorem returns_to_start_point_after_fourth_passenger :
  List.sum (driving_distances.take 4) = 0 :=
by
  sorry

theorem distance_after_last_passenger :
  List.sum driving_distances = 5 :=
by
  sorry

theorem total_earnings :
  (fare 5 + fare 3 + fare 6 + fare 4 + fare 7 + fare 2) = 68 :=
by
  sorry

end NUMINAMATH_GPT_returns_to_start_point_after_fourth_passenger_distance_after_last_passenger_total_earnings_l1622_162285


namespace NUMINAMATH_GPT_prime_p_is_2_l1622_162271

theorem prime_p_is_2 (p q r : ℕ) 
  (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (h_sum : p + q = r) (h_lt : p < q) : 
  p = 2 :=
sorry

end NUMINAMATH_GPT_prime_p_is_2_l1622_162271


namespace NUMINAMATH_GPT_sqrt_49_times_sqrt_25_l1622_162224

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_49_times_sqrt_25_l1622_162224


namespace NUMINAMATH_GPT_total_gold_cost_l1622_162208

-- Given conditions
def gary_grams : ℕ := 30
def gary_cost_per_gram : ℕ := 15
def anna_grams : ℕ := 50
def anna_cost_per_gram : ℕ := 20

-- Theorem statement to prove
theorem total_gold_cost :
  (gary_grams * gary_cost_per_gram + anna_grams * anna_cost_per_gram) = 1450 := 
by
  sorry

end NUMINAMATH_GPT_total_gold_cost_l1622_162208


namespace NUMINAMATH_GPT_solve_equation_l1622_162297

theorem solve_equation :
  (∀ x : ℝ, x ≠ 2/3 → (6 * x + 2) / (3 * x^2 + 6 * x - 4) = (3 * x) / (3 * x - 2)) →
  (∀ x : ℝ, x = 1 / Real.sqrt 3 ∨ x = -1 / Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1622_162297


namespace NUMINAMATH_GPT_sum_of_squares_219_l1622_162231

theorem sum_of_squares_219 :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a^2 + b^2 + c^2 = 219 ∧ a + b + c = 21 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_219_l1622_162231


namespace NUMINAMATH_GPT_triangle_side_length_l1622_162295

theorem triangle_side_length (a b c : ℝ) (A : ℝ) 
  (h_a : a = 2) (h_c : c = 2) (h_A : A = 30) :
  b = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l1622_162295


namespace NUMINAMATH_GPT_flowers_given_to_mother_l1622_162252

-- Definitions based on conditions:
def Alissa_flowers : Nat := 16
def Melissa_flowers : Nat := 16
def flowers_left : Nat := 14

-- The proof problem statement:
theorem flowers_given_to_mother :
  Alissa_flowers + Melissa_flowers - flowers_left = 18 := by
  sorry

end NUMINAMATH_GPT_flowers_given_to_mother_l1622_162252


namespace NUMINAMATH_GPT_sum_of_consecutive_even_integers_divisible_by_three_l1622_162250

theorem sum_of_consecutive_even_integers_divisible_by_three (n : ℤ) : 
  ∃ p : ℤ, Prime p ∧ p = 3 ∧ p ∣ (n + (n + 2) + (n + 4)) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_even_integers_divisible_by_three_l1622_162250


namespace NUMINAMATH_GPT_length_of_second_parallel_side_l1622_162246

-- Define the given conditions
def parallel_side1 : ℝ := 20
def distance : ℝ := 14
def area : ℝ := 266

-- Define the theorem to prove the length of the second parallel side
theorem length_of_second_parallel_side (x : ℝ) 
  (h : area = (1 / 2) * (parallel_side1 + x) * distance) : 
  x = 18 :=
sorry

end NUMINAMATH_GPT_length_of_second_parallel_side_l1622_162246


namespace NUMINAMATH_GPT_prime_numbers_solution_l1622_162270

theorem prime_numbers_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h1 : Nat.Prime (p + q)) (h2 : Nat.Prime (p^2 + q^2 - q)) : p = 3 ∧ q = 2 :=
by
  sorry

end NUMINAMATH_GPT_prime_numbers_solution_l1622_162270


namespace NUMINAMATH_GPT_problem1_problem2_l1622_162237

-- Define points A, B, C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 1, y := -2}
def B : Point := {x := 2, y := 1}
def C : Point := {x := 3, y := 2}

-- Function to compute vector difference
def vector_sub (p1 p2 : Point) : Point :=
  {x := p1.x - p2.x, y := p1.y - p2.y}

-- Function to compute vector scalar multiplication
def scalar_mul (k : ℝ) (p : Point) : Point :=
  {x := k * p.x, y := k * p.y}

-- Function to add two vectors
def vec_add (p1 p2 : Point) : Point :=
  {x := p1.x + p2.x, y := p1.y + p2.y}

-- Problem 1
def result_vector : Point :=
  let AB := vector_sub B A
  let AC := vector_sub C A
  let BC := vector_sub C B
  vec_add (scalar_mul 3 AB) (vec_add (scalar_mul (-2) AC) BC)

-- Prove the coordinates are (0, 2)
theorem problem1 : result_vector = {x := 0, y := 2} := by
  sorry

-- Problem 2
def D : Point :=
  let BC := vector_sub C B
  {x := 1 + BC.x, y := (-2) + BC.y}

-- Prove the coordinates are (2, -1)
theorem problem2 : D = {x := 2, y := -1} := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1622_162237


namespace NUMINAMATH_GPT_balloons_initial_count_l1622_162234

theorem balloons_initial_count (x : ℕ) (h : x + 13 = 60) : x = 47 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_balloons_initial_count_l1622_162234


namespace NUMINAMATH_GPT_tenth_number_drawn_eq_195_l1622_162288

noncomputable def total_students : Nat := 1000
noncomputable def sample_size : Nat := 50
noncomputable def first_selected_number : Nat := 15  -- Note: 0015 is 15 in natural number

theorem tenth_number_drawn_eq_195 
  (h1 : total_students = 1000)
  (h2 : sample_size = 50)
  (h3 : first_selected_number = 15) :
  15 + (20 * 9) = 195 := 
by
  sorry

end NUMINAMATH_GPT_tenth_number_drawn_eq_195_l1622_162288


namespace NUMINAMATH_GPT_values_of_x_minus_y_l1622_162238

theorem values_of_x_minus_y (x y : ℤ) (h1 : |x| = 5) (h2 : |y| = 3) (h3 : y > x) : x - y = -2 ∨ x - y = -8 :=
  sorry

end NUMINAMATH_GPT_values_of_x_minus_y_l1622_162238


namespace NUMINAMATH_GPT_number_of_chickens_l1622_162255

variable (C P : ℕ) (legs_total : ℕ := 48) (legs_pig : ℕ := 4) (legs_chicken : ℕ := 2) (number_pigs : ℕ := 9)

theorem number_of_chickens (h1 : P = number_pigs)
                           (h2 : legs_pig * P + legs_chicken * C = legs_total) :
                           C = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_chickens_l1622_162255


namespace NUMINAMATH_GPT_white_balls_in_bag_l1622_162235

   theorem white_balls_in_bag (m : ℕ) (h : m ≤ 7) :
     (2 * (m * (m - 1) / 2) / (7 * 6 / 2)) + ((m * (7 - m)) / (7 * 6 / 2)) = 6 / 7 → m = 3 :=
   by
     intros h_eq
     sorry
   
end NUMINAMATH_GPT_white_balls_in_bag_l1622_162235


namespace NUMINAMATH_GPT_darry_small_ladder_climbs_l1622_162251

-- Define the constants based on the conditions
def full_ladder_steps := 11
def full_ladder_climbs := 10
def small_ladder_steps := 6
def total_steps := 152

-- Darry's total steps climbed via full ladder
def full_ladder_total_steps := full_ladder_steps * full_ladder_climbs

-- Define x as the number of times Darry climbed the smaller ladder
variable (x : ℕ)

-- Prove that x = 7 given the conditions
theorem darry_small_ladder_climbs (h : full_ladder_total_steps + small_ladder_steps * x = total_steps) : x = 7 :=
by 
  sorry

end NUMINAMATH_GPT_darry_small_ladder_climbs_l1622_162251


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l1622_162253

theorem arithmetic_expression_evaluation : 1997 * (2000 / 2000) - 2000 * (1997 / 1997) = -3 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l1622_162253


namespace NUMINAMATH_GPT_option_C_l1622_162258

theorem option_C (a b c : ℝ) (h₀ : a > b) (h₁ : b > c) (h₂ : c > 0) :
  (b + c) / (a + c) > b / a :=
sorry

end NUMINAMATH_GPT_option_C_l1622_162258


namespace NUMINAMATH_GPT_number_of_cows_l1622_162217

theorem number_of_cows (n : ℝ) (h1 : n / 2 + n / 4 + n / 5 + 7 = n) : n = 140 := 
sorry

end NUMINAMATH_GPT_number_of_cows_l1622_162217


namespace NUMINAMATH_GPT_solution_to_absolute_value_equation_l1622_162220

theorem solution_to_absolute_value_equation (x : ℝ) : 
    abs x - 2 - abs (-1) = 2 ↔ x = 5 ∨ x = -5 :=
by
  sorry

end NUMINAMATH_GPT_solution_to_absolute_value_equation_l1622_162220


namespace NUMINAMATH_GPT_evaporation_amount_l1622_162293

noncomputable def water_evaporated_per_day (total_water: ℝ) (percentage_evaporated: ℝ) (days: ℕ) : ℝ :=
  (percentage_evaporated / 100) * total_water / days

theorem evaporation_amount :
  water_evaporated_per_day 10 7 50 = 0.014 :=
by
  sorry

end NUMINAMATH_GPT_evaporation_amount_l1622_162293


namespace NUMINAMATH_GPT_cole_drive_time_l1622_162232

noncomputable def T_work (D : ℝ) : ℝ := D / 75
noncomputable def T_home (D : ℝ) : ℝ := D / 105

theorem cole_drive_time (v1 v2 T : ℝ) (D : ℝ) 
  (h_v1 : v1 = 75) (h_v2 : v2 = 105) (h_T : T = 4)
  (h_round_trip : T_work D + T_home D = T) : 
  T_work D = 140 / 60 :=
sorry

end NUMINAMATH_GPT_cole_drive_time_l1622_162232


namespace NUMINAMATH_GPT_fran_ate_15_green_macaroons_l1622_162241

variable (total_red total_green initial_remaining green_macaroons_eaten : ℕ)

-- Conditions as definitions
def initial_red_macaroons := 50
def initial_green_macaroons := 40
def total_macaroons := 90
def remaining_macaroons := 45

-- Total eaten macaroons
def total_eaten_macaroons (G : ℕ) := G + 2 * G

-- The proof statement
theorem fran_ate_15_green_macaroons
  (h1 : total_red = initial_red_macaroons)
  (h2 : total_green = initial_green_macaroons)
  (h3 : initial_remaining = remaining_macaroons)
  (h4 : total_macaroons = initial_red_macaroons + initial_green_macaroons)
  (h5 : initial_remaining = total_macaroons - total_eaten_macaroons green_macaroons_eaten):
  green_macaroons_eaten = 15 :=
  by
  sorry

end NUMINAMATH_GPT_fran_ate_15_green_macaroons_l1622_162241


namespace NUMINAMATH_GPT_trigonometric_product_eq_l1622_162215

open Real

theorem trigonometric_product_eq :
  3.420 * (sin (10 * pi / 180)) * (sin (20 * pi / 180)) * (sin (30 * pi / 180)) *
  (sin (40 * pi / 180)) * (sin (50 * pi / 180)) * (sin (60 * pi / 180)) *
  (sin (70 * pi / 180)) * (sin (80 * pi / 180)) = 3 / 256 := 
sorry

end NUMINAMATH_GPT_trigonometric_product_eq_l1622_162215


namespace NUMINAMATH_GPT_expression_evaluation_l1622_162227

theorem expression_evaluation :
  100 + (120 / 15) + (18 * 20) - 250 - (360 / 12) = 188 := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1622_162227


namespace NUMINAMATH_GPT_simplify_expression_l1622_162292

theorem simplify_expression : ( (3 + 4 + 5 + 6) / 3 ) + ( (3 * 6 + 9) / 4 ) = 12.75 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1622_162292


namespace NUMINAMATH_GPT_unique_non_zero_b_for_unique_x_solution_l1622_162209

theorem unique_non_zero_b_for_unique_x_solution (c : ℝ) (hc : c ≠ 0) :
  c = 3 / 2 ↔ ∃! b : ℝ, b ≠ 0 ∧ ∃ x : ℝ, (x^2 + (b + 3 / b) * x + c = 0) ∧ 
  ∀ x1 x2 : ℝ, (x1^2 + (b + 3 / b) * x1 + c = 0) ∧ (x2^2 + (b + 3 / b) * x2 + c = 0) → x1 = x2 :=
sorry

end NUMINAMATH_GPT_unique_non_zero_b_for_unique_x_solution_l1622_162209


namespace NUMINAMATH_GPT_bob_questions_three_hours_l1622_162200

theorem bob_questions_three_hours : 
  let first_hour := 13
  let second_hour := first_hour * 2
  let third_hour := second_hour * 2
  first_hour + second_hour + third_hour = 91 :=
by
  sorry

end NUMINAMATH_GPT_bob_questions_three_hours_l1622_162200


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1622_162204

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ) (d : ℤ),
    (∀ n, a (n + 1) = a n + d) →
    (a 1 + a 4 + a 7 = 45) →
    (a 2 + a_5 + a_8 = 39) →
    (a 3 + a_6 + a_9 = 33) :=
by 
  intros a d h_arith_seq h_cond1 h_cond2
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1622_162204


namespace NUMINAMATH_GPT_corrected_sum_l1622_162275

theorem corrected_sum : 37541 + 43839 ≠ 80280 → 37541 + 43839 = 81380 :=
by
  sorry

end NUMINAMATH_GPT_corrected_sum_l1622_162275


namespace NUMINAMATH_GPT_combined_degrees_l1622_162214

-- Definitions based on conditions
def summer_degrees : ℕ := 150
def jolly_degrees (summer_degrees : ℕ) : ℕ := summer_degrees - 5

-- Theorem stating the combined degrees
theorem combined_degrees : summer_degrees + jolly_degrees summer_degrees = 295 :=
by
  sorry

end NUMINAMATH_GPT_combined_degrees_l1622_162214


namespace NUMINAMATH_GPT_basketball_team_free_throws_l1622_162201

theorem basketball_team_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a)
  (h2 : x = 2 * a - 1)
  (h3 : 2 * a + 3 * b + x = 89) : 
  x = 29 :=
by
  sorry

end NUMINAMATH_GPT_basketball_team_free_throws_l1622_162201


namespace NUMINAMATH_GPT_minimize_average_cost_l1622_162263

noncomputable def average_comprehensive_cost (x : ℝ) : ℝ :=
  560 + 48 * x + 2160 * 10^6 / (2000 * x)

theorem minimize_average_cost : 
  ∃ x_min : ℝ, x_min ≥ 10 ∧ 
  ∀ x ≥ 10, average_comprehensive_cost x ≥ average_comprehensive_cost x_min :=
sorry

end NUMINAMATH_GPT_minimize_average_cost_l1622_162263


namespace NUMINAMATH_GPT_path_length_of_dot_l1622_162213

-- Define the dimensions of the rectangular prism
def prism_width := 1 -- cm
def prism_height := 1 -- cm
def prism_length := 2 -- cm

-- Define the condition that the dot is marked at the center of the top face
def dot_position := (0.5, 1)

-- Define the condition that the prism starts with the 1 cm by 2 cm face on the table
def initial_face_on_table := (prism_length, prism_height)

-- Define the statement to prove the length of the path followed by the dot
theorem path_length_of_dot: 
  ∃ length_of_path : ℝ, length_of_path = 2 * Real.pi :=
sorry

end NUMINAMATH_GPT_path_length_of_dot_l1622_162213


namespace NUMINAMATH_GPT_cube_painted_faces_l1622_162277

noncomputable def painted_faces_count (side_length painted_cubes_edge middle_cubes_edge : ℕ) : ℕ :=
  let total_corners := 8
  let total_edges := 12
  total_corners + total_edges * middle_cubes_edge

theorem cube_painted_faces :
  ∀ side_length : ℕ, side_length = 4 →
  ∀ painted_cubes_edge middle_cubes_edge total_cubes : ℕ,
  total_cubes = side_length * side_length * side_length →
  painted_cubes_edge = 3 →
  middle_cubes_edge = 2 →
  painted_faces_count side_length painted_cubes_edge middle_cubes_edge = 32 := sorry

end NUMINAMATH_GPT_cube_painted_faces_l1622_162277


namespace NUMINAMATH_GPT_value_calculation_l1622_162243

-- Definition of constants used in the problem
def a : ℝ := 1.3333
def b : ℝ := 3.615
def expected_value : ℝ := 4.81998845

-- The proposition to be proven
theorem value_calculation : a * b = expected_value :=
by sorry

end NUMINAMATH_GPT_value_calculation_l1622_162243


namespace NUMINAMATH_GPT_mean_temperature_is_0_5_l1622_162236

def temperatures : List ℝ := [-3.5, -2.25, 0, 3.75, 4.5]

theorem mean_temperature_is_0_5 :
  (temperatures.sum / temperatures.length) = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_mean_temperature_is_0_5_l1622_162236


namespace NUMINAMATH_GPT_distance_of_hyperbola_vertices_l1622_162291

-- Define the hyperbola equation condition
def hyperbola : Prop := ∃ (y x : ℝ), (y^2 / 16) - (x^2 / 9) = 1

-- Define a variable for the distance between the vertices
def distance_between_vertices (a : ℝ) : ℝ := 2 * a

-- The main statement to be proved
theorem distance_of_hyperbola_vertices :
  hyperbola → distance_between_vertices 4 = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_distance_of_hyperbola_vertices_l1622_162291


namespace NUMINAMATH_GPT_time_to_pass_pole_l1622_162274

def length_of_train : ℝ := 240
def length_of_platform : ℝ := 650
def time_to_pass_platform : ℝ := 89

theorem time_to_pass_pole (length_of_train length_of_platform time_to_pass_platform : ℝ) 
  (h_train : length_of_train = 240)
  (h_platform : length_of_platform = 650)
  (h_time : time_to_pass_platform = 89)
  : (length_of_train / ((length_of_train + length_of_platform) / time_to_pass_platform)) = 24 := by
  -- Let the speed of the train be v, hence
  -- v = (length_of_train + length_of_platform) / time_to_pass_platform
  -- What we need to prove is  
  -- length_of_train / v = 24
  sorry

end NUMINAMATH_GPT_time_to_pass_pole_l1622_162274


namespace NUMINAMATH_GPT_min_even_integers_six_l1622_162219

theorem min_even_integers_six (x y a b m n : ℤ) 
  (h1 : x + y = 30) 
  (h2 : x + y + a + b = 50) 
  (h3 : x + y + a + b + m + n = 70) 
  (hm_even : Even m) 
  (hn_even: Even n) : 
  ∃ k, (0 ≤ k ∧ k ≤ 6) ∧ (∀ e, (e = m ∨ e = n) → ∃ j, (j = 2)) :=
by
  sorry

end NUMINAMATH_GPT_min_even_integers_six_l1622_162219


namespace NUMINAMATH_GPT_particle_max_height_and_time_l1622_162244

theorem particle_max_height_and_time (t : ℝ) (s : ℝ) 
  (height_eq : s = 180 * t - 18 * t^2) :
  ∃ t₁ : ℝ, ∃ s₁ : ℝ, s₁ = 450 ∧ t₁ = 5 ∧ s = 180 * t₁ - 18 * t₁^2 :=
sorry

end NUMINAMATH_GPT_particle_max_height_and_time_l1622_162244


namespace NUMINAMATH_GPT_runway_show_time_l1622_162222

/-
Problem: Prove that it will take 60 minutes to complete all of the runway trips during the show, 
given the following conditions:
- There are 6 models in the show.
- Each model will wear two sets of bathing suits and three sets of evening wear clothes during the runway portion of the show.
- It takes a model 2 minutes to walk out to the end of the runway and back, and models take turns, one at a time.
-/

theorem runway_show_time 
    (num_models : ℕ) 
    (sets_bathing_suits_per_model : ℕ) 
    (sets_evening_wear_per_model : ℕ) 
    (time_per_trip : ℕ) 
    (total_time : ℕ) :
    num_models = 6 →
    sets_bathing_suits_per_model = 2 →
    sets_evening_wear_per_model = 3 →
    time_per_trip = 2 →
    total_time = num_models * (sets_bathing_suits_per_model + sets_evening_wear_per_model) * time_per_trip →
    total_time = 60 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5


end NUMINAMATH_GPT_runway_show_time_l1622_162222


namespace NUMINAMATH_GPT_initial_candies_l1622_162225

theorem initial_candies (x : ℕ) (h1 : x % 4 = 0) (h2 : x / 4 * 3 / 3 * 2 / 2 - 24 ≥ 6) (h3 : x / 4 * 3 / 3 * 2 / 2 - 24 ≤ 9) :
  x = 64 :=
sorry

end NUMINAMATH_GPT_initial_candies_l1622_162225


namespace NUMINAMATH_GPT_average_income_l1622_162289

/-- The daily incomes of the cab driver over 5 days. --/
def incomes : List ℕ := [400, 250, 650, 400, 500]

/-- Prove that the average income of the cab driver over these 5 days is $440. --/
theorem average_income : (incomes.sum / incomes.length) = 440 := by
  sorry

end NUMINAMATH_GPT_average_income_l1622_162289


namespace NUMINAMATH_GPT_area_of_square_field_l1622_162248

def side_length : ℕ := 7
def expected_area : ℕ := 49

theorem area_of_square_field : (side_length * side_length) = expected_area := 
by
  -- The proof steps will be filled here
  sorry

end NUMINAMATH_GPT_area_of_square_field_l1622_162248


namespace NUMINAMATH_GPT_correct_quotient_division_l1622_162221

variable (k : Nat) -- the unknown original number

def mistaken_division := k = 7 * 12 + 4

theorem correct_quotient_division (h : mistaken_division k) : 
  (k / 3) = 29 :=
by
  sorry

end NUMINAMATH_GPT_correct_quotient_division_l1622_162221


namespace NUMINAMATH_GPT_evaluate_fraction_sqrt_l1622_162287

theorem evaluate_fraction_sqrt :
  (Real.sqrt ((1 / 8) + (1 / 18)) = (Real.sqrt 26) / 12) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_sqrt_l1622_162287
