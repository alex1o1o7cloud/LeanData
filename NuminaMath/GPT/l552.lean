import Mathlib

namespace NUMINAMATH_GPT_second_crew_tractors_l552_55278

theorem second_crew_tractors
    (total_acres : ℕ)
    (days : ℕ)
    (first_crew_days : ℕ)
    (first_crew_tractors : ℕ)
    (acres_per_tractor_per_day : ℕ)
    (remaining_days : ℕ)
    (remaining_acres_after_first_crew : ℕ)
    (second_crew_acres_per_tractor : ℕ) :
    total_acres = 1700 → days = 5 → first_crew_days = 2 → first_crew_tractors = 2 → 
    acres_per_tractor_per_day = 68 → remaining_days = 3 → 
    remaining_acres_after_first_crew = total_acres - (first_crew_tractors * acres_per_tractor_per_day * first_crew_days) → 
    second_crew_acres_per_tractor = acres_per_tractor_per_day * remaining_days → 
    (remaining_acres_after_first_crew / second_crew_acres_per_tractor = 7) := 
by
  sorry

end NUMINAMATH_GPT_second_crew_tractors_l552_55278


namespace NUMINAMATH_GPT_flour_already_put_in_l552_55233

def total_flour : ℕ := 8
def additional_flour_needed : ℕ := 6

theorem flour_already_put_in : total_flour - additional_flour_needed = 2 := by
  sorry

end NUMINAMATH_GPT_flour_already_put_in_l552_55233


namespace NUMINAMATH_GPT_time_to_save_for_downpayment_l552_55280

def annual_salary : ℝ := 120000
def savings_percentage : ℝ := 0.15
def house_cost : ℝ := 550000
def downpayment_percentage : ℝ := 0.25

def annual_savings : ℝ := savings_percentage * annual_salary
def downpayment_needed : ℝ := downpayment_percentage * house_cost

theorem time_to_save_for_downpayment :
  (downpayment_needed / annual_savings) = 7.64 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_time_to_save_for_downpayment_l552_55280


namespace NUMINAMATH_GPT_arithmetic_series_sum_base6_l552_55279

-- Define the terms in the arithmetic series in base 6
def a₁ := 1
def a₄₅ := 45
def n := a₄₅

-- Sum of arithmetic series in base 6
def sum_arithmetic_series := (n * (a₁ + a₄₅)) / 2

-- Expected result for the arithmetic series sum
def expected_result := 2003

theorem arithmetic_series_sum_base6 :
  sum_arithmetic_series = expected_result := by
  sorry

end NUMINAMATH_GPT_arithmetic_series_sum_base6_l552_55279


namespace NUMINAMATH_GPT_optimal_fence_area_l552_55214

variables {l w : ℝ}

theorem optimal_fence_area
  (h1 : 2 * l + 2 * w = 400) -- Tiffany must use exactly 400 feet of fencing.
  (h2 : l ≥ 100) -- The length must be at least 100 feet.
  (h3 : w ≥ 50) -- The width must be at least 50 feet.
  : l * w ≤ 10000 :=      -- We need to prove that the area is at most 10000 square feet.
by
  sorry

end NUMINAMATH_GPT_optimal_fence_area_l552_55214


namespace NUMINAMATH_GPT_solve_equation_l552_55272

theorem solve_equation (x : ℝ) (hx : x ≠ 0) 
  (h : 1 / 4 + 8 / x = 13 / x + 1 / 8) : 
  x = 40 :=
sorry

end NUMINAMATH_GPT_solve_equation_l552_55272


namespace NUMINAMATH_GPT_part_i_part_ii_l552_55220

-- Define the operations for the weird calculator.
def Dsharp (n : ℕ) : ℕ := 2 * n + 1
def Dflat (n : ℕ) : ℕ := 2 * n - 1

-- Define the initial starting point.
def initial_display : ℕ := 1

-- Define a function to execute a sequence of button presses.
def execute_sequence (seq : List (ℕ → ℕ)) (initial : ℕ) : ℕ :=
  seq.foldl (fun x f => f x) initial

-- Problem (i): Prove there is a sequence that results in 313 starting from 1 after eight presses.
theorem part_i : ∃ seq : List (ℕ → ℕ), seq.length = 8 ∧ execute_sequence seq 1 = 313 :=
by sorry

-- Problem (ii): Describe all numbers that can be achieved from exactly eight button presses starting from 1.
theorem part_ii : 
  ∀ n : ℕ, n % 2 = 1 ∧ n < 2^9 →
  ∃ seq : List (ℕ → ℕ), seq.length = 8 ∧ execute_sequence seq 1 = n :=
by sorry

end NUMINAMATH_GPT_part_i_part_ii_l552_55220


namespace NUMINAMATH_GPT_motorboat_max_distance_l552_55204

/-- Given a motorboat which, when fully fueled, can travel exactly 40 km against the current 
    or 60 km with the current, proves that the maximum distance it can travel up the river and 
    return to the starting point with the available fuel is 24 km. -/
theorem motorboat_max_distance (upstream_dist : ℕ) (downstream_dist : ℕ) : 
  upstream_dist = 40 → downstream_dist = 60 → 
  ∃ max_round_trip_dist : ℕ, max_round_trip_dist = 24 :=
by
  intros h1 h2
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_motorboat_max_distance_l552_55204


namespace NUMINAMATH_GPT_probability_of_first_good_product_on_third_try_l552_55290

-- Define the problem parameters
def pass_rate : ℚ := 3 / 4
def failure_rate : ℚ := 1 / 4
def epsilon := 3

-- The target probability statement
theorem probability_of_first_good_product_on_third_try :
  (failure_rate * failure_rate * pass_rate) = ((1 / 4) ^ 2 * (3 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_first_good_product_on_third_try_l552_55290


namespace NUMINAMATH_GPT_range_of_x8_l552_55252

theorem range_of_x8 (x : ℕ → ℝ) (h1 : 0 ≤ x 1 ∧ x 1 ≤ x 2)
  (h_recurrence : ∀ n ≥ 1, x (n+2) = x (n+1) + x n)
  (h_x7 : 1 ≤ x 7 ∧ x 7 ≤ 2) : 
  (21/13 : ℝ) ≤ x 8 ∧ x 8 ≤ (13/4) :=
sorry

end NUMINAMATH_GPT_range_of_x8_l552_55252


namespace NUMINAMATH_GPT_price_of_first_oil_is_54_l552_55230

/-- Let x be the price per litre of the first oil.
Given that 10 litres of the first oil are mixed with 5 litres of second oil priced at Rs. 66 per litre,
resulting in a 15-litre mixture costing Rs. 58 per litre, prove that x = 54. -/
theorem price_of_first_oil_is_54 :
  (∃ x : ℝ, x = 54) ↔
  (10 * x + 5 * 66 = 15 * 58) :=
by
  sorry

end NUMINAMATH_GPT_price_of_first_oil_is_54_l552_55230


namespace NUMINAMATH_GPT_medicine_dosage_per_kg_l552_55265

theorem medicine_dosage_per_kg :
  ∀ (child_weight parts dose_per_part total_dose dose_per_kg : ℕ),
    (child_weight = 30) →
    (parts = 3) →
    (dose_per_part = 50) →
    (total_dose = parts * dose_per_part) →
    (dose_per_kg = total_dose / child_weight) →
    dose_per_kg = 5 :=
by
  intros child_weight parts dose_per_part total_dose dose_per_kg
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_medicine_dosage_per_kg_l552_55265


namespace NUMINAMATH_GPT_bug_final_position_after_2023_jumps_l552_55200

open Nat

def bug_jump (pos : Nat) : Nat :=
  if pos % 2 = 1 then (pos + 2) % 6 else (pos + 1) % 6

noncomputable def final_position (n : Nat) : Nat :=
  (iterate bug_jump n 6) % 6

theorem bug_final_position_after_2023_jumps : final_position 2023 = 1 := by
  sorry

end NUMINAMATH_GPT_bug_final_position_after_2023_jumps_l552_55200


namespace NUMINAMATH_GPT_value_of_a3_l552_55251

variable {a_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}

-- Given conditions
def S (n : ℕ) : ℤ := 2 * (n ^ 2) - 1
def a (n : ℕ) : ℤ := S n - S (n - 1)

-- The theorem to prove
theorem value_of_a3 : a 3 = 10 := by
  sorry

end NUMINAMATH_GPT_value_of_a3_l552_55251


namespace NUMINAMATH_GPT_ratio_part_to_whole_number_l552_55249

theorem ratio_part_to_whole_number (P N : ℚ) 
  (h1 : (1 / 4) * (1 / 3) * P = 25) 
  (h2 : 0.40 * N = 300) : P / N = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_part_to_whole_number_l552_55249


namespace NUMINAMATH_GPT_acute_angles_45_degrees_l552_55276

-- Assuming quadrilaterals ABCD and A'B'C'D' such that sides of each lie on 
-- the perpendicular bisectors of the sides of the other. We want to prove that
-- the acute angles of A'B'C'D' are 45 degrees.

def convex_quadrilateral (Q : Type) := 
  ∃ (A B C D : Q), True -- Placeholder for a more detailed convex quadrilateral structure

def perpendicular_bisector (S1 S2 T1 T2: Type) := 
  ∃ (M : Type), True -- Placeholder for a more detailed perpendicular bisector structure

theorem acute_angles_45_degrees
  (Q1 Q2 : Type)
  (h1 : convex_quadrilateral Q1)
  (h2 : convex_quadrilateral Q2)
  (perp1 : perpendicular_bisector Q1 Q1 Q2 Q2)
  (perp2 : perpendicular_bisector Q2 Q2 Q1 Q1) :
  ∀ (θ : ℝ), θ = 45 := 
by
  sorry

end NUMINAMATH_GPT_acute_angles_45_degrees_l552_55276


namespace NUMINAMATH_GPT_main_theorem_l552_55232

-- Definitions based on conditions
variables (A P H M E C : ℕ) 
-- Thickness of an algebra book
def x := 1
-- Thickness of a history book (twice that of algebra)
def history_thickness := 2 * x
-- Length of shelf filled by books
def z := A * x

-- Condition equations based on shelf length equivalences
def equation1 := A = P
def equation2 := 2 * H * x = M * x
def equation3 := E * x + C * history_thickness = z

-- Prove the relationship
theorem main_theorem : C = (M * (A - E)) / (2 * A * H) :=
by
  sorry

end NUMINAMATH_GPT_main_theorem_l552_55232


namespace NUMINAMATH_GPT_number_of_truthful_warriors_l552_55213

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end NUMINAMATH_GPT_number_of_truthful_warriors_l552_55213


namespace NUMINAMATH_GPT_number_of_mixed_vegetable_plates_l552_55274

def cost_of_chapati := 6
def cost_of_rice := 45
def cost_of_mixed_vegetable := 70
def chapatis_ordered := 16
def rice_ordered := 5
def ice_cream_cups := 6 -- though not used, included for completeness
def total_amount_paid := 1111

def total_cost_of_known_items := (chapatis_ordered * cost_of_chapati) + (rice_ordered * cost_of_rice)
def amount_spent_on_mixed_vegetable := total_amount_paid - total_cost_of_known_items

theorem number_of_mixed_vegetable_plates : 
  amount_spent_on_mixed_vegetable / cost_of_mixed_vegetable = 11 := 
by sorry

end NUMINAMATH_GPT_number_of_mixed_vegetable_plates_l552_55274


namespace NUMINAMATH_GPT_runs_scored_by_c_l552_55270

-- Definitions
variables (A B C : ℕ)

-- Conditions as hypotheses
theorem runs_scored_by_c (h1 : B = 3 * A) (h2 : C = 5 * B) (h3 : A + B + C = 95) : C = 75 :=
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_runs_scored_by_c_l552_55270


namespace NUMINAMATH_GPT_subtract_add_example_l552_55223

theorem subtract_add_example : (3005 - 3000) + 10 = 15 :=
by
  sorry

end NUMINAMATH_GPT_subtract_add_example_l552_55223


namespace NUMINAMATH_GPT_no_nat_number_satisfies_l552_55268

theorem no_nat_number_satisfies (n : ℕ) : ¬ ((n^2 + 6 * n + 2019) % 100 = 0) :=
sorry

end NUMINAMATH_GPT_no_nat_number_satisfies_l552_55268


namespace NUMINAMATH_GPT_swimming_pool_min_cost_l552_55231

theorem swimming_pool_min_cost (a : ℝ) (x : ℝ) (y : ℝ) :
  (∀ (x : ℝ), x > 0 → y = 2400 * a + 6 * (x + 1600 / x) * a) →
  (∃ (x : ℝ), x > 0 ∧ y = 2880 * a) :=
by
  sorry

end NUMINAMATH_GPT_swimming_pool_min_cost_l552_55231


namespace NUMINAMATH_GPT_applicants_majored_in_political_science_l552_55259

theorem applicants_majored_in_political_science
  (total_applicants : ℕ)
  (gpa_above_3 : ℕ)
  (non_political_science_and_gpa_leq_3 : ℕ)
  (political_science_and_gpa_above_3 : ℕ) :
  total_applicants = 40 →
  gpa_above_3 = 20 →
  non_political_science_and_gpa_leq_3 = 10 →
  political_science_and_gpa_above_3 = 5 →
  ∃ P : ℕ, P = 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_applicants_majored_in_political_science_l552_55259


namespace NUMINAMATH_GPT_find_common_difference_l552_55224

variable {a : ℕ → ℝ}
variable {p q : ℕ}
variable {d : ℝ}

-- Definitions based on the conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition1 (a : ℕ → ℝ) (p : ℕ) := a p = 4
def condition2 (a : ℕ → ℝ) (q : ℕ) := a q = 2
def condition3 (p q : ℕ) := p = 4 + q

-- The goal statement
theorem find_common_difference
  (a_seq : arithmetic_sequence a d)
  (h1 : condition1 a p)
  (h2 : condition2 a q)
  (h3 : condition3 p q) :
  d = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_common_difference_l552_55224


namespace NUMINAMATH_GPT_find_x_plus_y_l552_55216

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2010) (h2 : x + 2010 * Real.sin y = 2009) (h3 : Real.pi / 2 ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2011 + Real.pi :=
sorry

end NUMINAMATH_GPT_find_x_plus_y_l552_55216


namespace NUMINAMATH_GPT_find_a_l552_55275

def polys_are_integers (a b : Int) : Prop :=
  a ≠ 0 ∧ b ≠ 0

def factor_condition (a b : Int) : Prop :=
  ∀ x : ℝ, (x^2 - x - 1 = 0) → (a * x^19 + b * x^18 + 1 = 0)

theorem find_a (a b : Int) (h : polys_are_integers a b) (h_fac : factor_condition a b) : a = 1597 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l552_55275


namespace NUMINAMATH_GPT_sum_m_n_is_55_l552_55282

theorem sum_m_n_is_55 (a b c : ℝ) (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1)
  (h1 : 5 / a = b + c) (h2 : 10 / b = c + a) (h3 : 13 / c = a + b) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : (a + b + c) = m / n) : m + n = 55 :=
  sorry

end NUMINAMATH_GPT_sum_m_n_is_55_l552_55282


namespace NUMINAMATH_GPT_number_of_days_l552_55219

theorem number_of_days (m1 d1 m2 d2 : ℕ) (h1 : m1 * d1 = m2 * d2) (k : ℕ) 
(h2 : m1 = 10) (h3 : d1 = 6) (h4 : m2 = 15) (h5 : k = 60) : 
d2 = 4 :=
by
  have : 10 * 6 = 60 := by sorry
  have : 15 * d2 = 60 := by sorry
  exact sorry

end NUMINAMATH_GPT_number_of_days_l552_55219


namespace NUMINAMATH_GPT_target_hit_probability_l552_55218

-- Define the probabilities given in the problem
def prob_A_hits : ℚ := 9 / 10
def prob_B_hits : ℚ := 8 / 9

-- The required probability that at least one hits the target
def prob_target_hit : ℚ := 89 / 90

-- Theorem stating that the probability calculated matches the expected outcome
theorem target_hit_probability :
  1 - ((1 - prob_A_hits) * (1 - prob_B_hits)) = prob_target_hit :=
by
  sorry

end NUMINAMATH_GPT_target_hit_probability_l552_55218


namespace NUMINAMATH_GPT_correct_calculated_value_l552_55277

theorem correct_calculated_value (N : ℕ) (h : N ≠ 0) :
  N * 16 = 2048 * (N / 128) := by 
  sorry

end NUMINAMATH_GPT_correct_calculated_value_l552_55277


namespace NUMINAMATH_GPT_solve_equation_l552_55261

theorem solve_equation :
  ∀ x : ℝ, 18 / (x^2 - 9) - 3 / (x - 3) = 2 ↔ (x = 4.5 ∨ x = -3) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l552_55261


namespace NUMINAMATH_GPT_find_b_l552_55221

theorem find_b (b : ℤ) (h_quad : ∃ m : ℤ, (x + m)^2 + 20 = x^2 + b * x + 56) (h_pos : b > 0) : b = 12 :=
sorry

end NUMINAMATH_GPT_find_b_l552_55221


namespace NUMINAMATH_GPT_percentage_of_rotten_bananas_l552_55254

-- Define the initial conditions and the question as a Lean theorem statement
theorem percentage_of_rotten_bananas (oranges bananas : ℕ) (perc_rot_oranges perc_good_fruits : ℝ) 
  (total_fruits good_fruits good_oranges good_bananas rotten_bananas perc_rot_bananas : ℝ) :
  oranges = 600 →
  bananas = 400 →
  perc_rot_oranges = 0.15 →
  perc_good_fruits = 0.886 →
  total_fruits = (oranges + bananas) →
  good_fruits = (perc_good_fruits * total_fruits) →
  good_oranges = ((1 - perc_rot_oranges) * oranges) →
  good_bananas = (good_fruits - good_oranges) →
  rotten_bananas = (bananas - good_bananas) →
  perc_rot_bananas = ((rotten_bananas / bananas) * 100) →
  perc_rot_bananas = 6 :=
by
  intros; sorry

end NUMINAMATH_GPT_percentage_of_rotten_bananas_l552_55254


namespace NUMINAMATH_GPT_statement_b_statement_c_l552_55206
-- Import all of Mathlib to include necessary mathematical functions and properties

-- First, the Lean statement for Statement B
theorem statement_b (a b : ℝ) (h : a > |b|) : a^2 > b^2 := 
sorry

-- Second, the Lean statement for Statement C
theorem statement_c (a b : ℝ) (h : a > b) : a^3 > b^3 := 
sorry

end NUMINAMATH_GPT_statement_b_statement_c_l552_55206


namespace NUMINAMATH_GPT_intersections_of_absolute_value_functions_l552_55255

theorem intersections_of_absolute_value_functions : 
  (∃ x y : ℝ, y = |3 * x + 4| ∧ y = -|4 * x + 3|) → ∃ (x y : ℝ), (x = -1 ∧ y = 1) ∧ ¬(∃ (x' y' : ℝ), y' = |3 * x' + 4| ∧ y' = -|4 * x' + 3| ∧ (x' ≠ -1 ∨ y' ≠ 1)) :=
by
  sorry

end NUMINAMATH_GPT_intersections_of_absolute_value_functions_l552_55255


namespace NUMINAMATH_GPT_a_is_5_if_extreme_at_neg3_l552_55211

-- Define the function f with parameter a
def f (a x : ℝ) : ℝ := x^3 + a * x^2 + 3 * x - 9

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 3

-- Define the given condition that f reaches an extreme value at x = -3
def reaches_extreme_at (a : ℝ) : Prop := f_prime a (-3) = 0

-- Prove that a = 5 if f reaches an extreme value at x = -3
theorem a_is_5_if_extreme_at_neg3 : ∀ a : ℝ, reaches_extreme_at a → a = 5 :=
by
  intros a h
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_a_is_5_if_extreme_at_neg3_l552_55211


namespace NUMINAMATH_GPT_time_to_be_100_miles_apart_l552_55248

noncomputable def distance_apart (x : ℝ) : ℝ :=
  Real.sqrt ((12 * x) ^ 2 + (16 * x) ^ 2)

theorem time_to_be_100_miles_apart : ∃ x : ℝ, distance_apart x = 100 ↔ x = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_time_to_be_100_miles_apart_l552_55248


namespace NUMINAMATH_GPT_johns_brother_age_l552_55293

variable (B : ℕ)
variable (J : ℕ)

-- Conditions given in the problem
def condition1 : Prop := J = 6 * B - 4
def condition2 : Prop := J + B = 10

-- The statement we want to prove, which is the answer to the problem:
theorem johns_brother_age (h1 : condition1 B J) (h2 : condition2 B J) : B = 2 := 
by 
  sorry

end NUMINAMATH_GPT_johns_brother_age_l552_55293


namespace NUMINAMATH_GPT_circumcircle_radius_min_cosA_l552_55295

noncomputable def circumcircle_radius (a b c : ℝ) (A B C : ℝ) :=
  a / (2 * (Real.sin A))

theorem circumcircle_radius_min_cosA
  (a b c A B C : ℝ)
  (h1 : a = 2)
  (h2 : Real.sin C + Real.sin B = 4 * Real.sin A)
  (h3 : a^2 + b^2 - 2 * a * b * (Real.cos A) = c^2)
  (h4 : a^2 + c^2 - 2 * a * c * (Real.cos B) = b^2)
  (h5 : b^2 + c^2 - 2 * b * c * (Real.cos C) = a^2) :
  circumcircle_radius a b c A B C = 8 * Real.sqrt 15 / 15 :=
sorry

end NUMINAMATH_GPT_circumcircle_radius_min_cosA_l552_55295


namespace NUMINAMATH_GPT_range_x_when_p_and_q_m_eq_1_range_m_for_not_p_necessary_not_sufficient_q_l552_55225

-- Define the propositions p and q in terms of x and m
def p (x m : ℝ) : Prop := |2 * x - m| ≥ 1
def q (x : ℝ) : Prop := (1 - 3 * x) / (x + 2) > 0

-- The range of x for p ∧ q when m = 1
theorem range_x_when_p_and_q_m_eq_1 : {x : ℝ | p x 1 ∧ q x} = {x : ℝ | -2 < x ∧ x ≤ 0} :=
by sorry

-- The range of m where ¬p is a necessary but not sufficient condition for q
theorem range_m_for_not_p_necessary_not_sufficient_q : {m : ℝ | ∀ x, ¬p x m → q x} ∩ {m : ℝ | ∃ x, ¬p x m ∧ q x} = {m : ℝ | -3 ≤ m ∧ m ≤ -1/3} :=
by sorry

end NUMINAMATH_GPT_range_x_when_p_and_q_m_eq_1_range_m_for_not_p_necessary_not_sufficient_q_l552_55225


namespace NUMINAMATH_GPT_how_many_tuna_l552_55238

-- Definitions for conditions
variables (customers : ℕ) (weightPerTuna : ℕ) (weightPerCustomer : ℕ)
variables (unsatisfiedCustomers : ℕ)

-- Hypotheses based on the problem conditions
def conditions :=
  customers = 100 ∧
  weightPerTuna = 200 ∧
  weightPerCustomer = 25 ∧
  unsatisfiedCustomers = 20

-- Statement to prove how many tuna Mr. Ray needs
theorem how_many_tuna (h : conditions customers weightPerTuna weightPerCustomer unsatisfiedCustomers) : 
  ∃ n, n = 10 :=
by
  sorry

end NUMINAMATH_GPT_how_many_tuna_l552_55238


namespace NUMINAMATH_GPT_missing_number_l552_55263

theorem missing_number (mean : ℝ) (numbers : List ℝ) (x : ℝ) (h_mean : mean = 14.2) (h_numbers : numbers = [13.0, 8.0, 13.0, 21.0, 23.0]) :
  (numbers.sum + x) / (numbers.length + 1) = mean → x = 7.2 :=
by
  -- states the hypothesis about the mean calculation into the theorem structure
  intro h
  sorry

end NUMINAMATH_GPT_missing_number_l552_55263


namespace NUMINAMATH_GPT_abs_neg_eight_l552_55217

theorem abs_neg_eight : abs (-8) = 8 := by
  sorry

end NUMINAMATH_GPT_abs_neg_eight_l552_55217


namespace NUMINAMATH_GPT_combined_perimeter_of_squares_l552_55269

theorem combined_perimeter_of_squares (p1 p2 : ℝ) (s1 s2 : ℝ) :
  p1 = 40 → p2 = 100 → 4 * s1 = p1 → 4 * s2 = p2 →
  (p1 + p2 - 2 * s1) = 120 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_combined_perimeter_of_squares_l552_55269


namespace NUMINAMATH_GPT_probability_of_both_selected_l552_55271

theorem probability_of_both_selected (pX pY : ℚ) (hX : pX = 1/7) (hY : pY = 2/5) : 
  pX * pY = 2 / 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_of_both_selected_l552_55271


namespace NUMINAMATH_GPT_smallest_M_conditions_l552_55297

theorem smallest_M_conditions :
  ∃ M : ℕ, M > 0 ∧
  ((∃ k₁, M = 8 * k₁) ∨ (∃ k₂, M + 2 = 8 * k₂) ∨ (∃ k₃, M + 4 = 8 * k₃)) ∧
  ((∃ k₄, M = 9 * k₄) ∨ (∃ k₅, M + 2 = 9 * k₅) ∨ (∃ k₆, M + 4 = 9 * k₆)) ∧
  ((∃ k₇, M = 25 * k₇) ∨ (∃ k₈, M + 2 = 25 * k₈) ∨ (∃ k₉, M + 4 = 25 * k₉)) ∧
  M = 100 :=
sorry

end NUMINAMATH_GPT_smallest_M_conditions_l552_55297


namespace NUMINAMATH_GPT_chocolate_bars_per_box_l552_55241

theorem chocolate_bars_per_box (total_chocolate_bars num_small_boxes : ℕ) (h1 : total_chocolate_bars = 300) (h2 : num_small_boxes = 15) : 
  total_chocolate_bars / num_small_boxes = 20 :=
by 
  sorry

end NUMINAMATH_GPT_chocolate_bars_per_box_l552_55241


namespace NUMINAMATH_GPT_current_year_2021_l552_55250

variables (Y : ℤ)

def parents_moved_to_America := 1982
def Aziz_age := 36
def years_before_born := 3

theorem current_year_2021
  (h1 : parents_moved_to_America = 1982)
  (h2 : Aziz_age = 36)
  (h3 : years_before_born = 3)
  (h4 : Y - (Aziz_age) - (years_before_born) = 1982) : 
  Y = 2021 :=
by {
  sorry
}

end NUMINAMATH_GPT_current_year_2021_l552_55250


namespace NUMINAMATH_GPT_gcd_problem_l552_55242

theorem gcd_problem (x : ℤ) (h : ∃ k, x = 2 * 2027 * k) :
  Int.gcd (3 * x ^ 2 + 47 * x + 101) (x + 23) = 1 :=
sorry

end NUMINAMATH_GPT_gcd_problem_l552_55242


namespace NUMINAMATH_GPT_triangle_equilateral_l552_55283

noncomputable def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = b ∧ b = c ∧ A = B ∧ B = C

theorem triangle_equilateral 
  (a b c A B C : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * a * b * c) 
  (h2 : Real.sin A = 2 * Real.sin B * Real.cos C) : 
  is_equilateral_triangle a b c A B C :=
sorry

end NUMINAMATH_GPT_triangle_equilateral_l552_55283


namespace NUMINAMATH_GPT_find_third_number_l552_55273
open BigOperators

def LCM (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

def LCM_of_three (a b c : ℕ) : ℕ := LCM (LCM a b) c

theorem find_third_number (n : ℕ) (h₁ : LCM 15 25 = 75) (h₂ : LCM_of_three 15 25 n = 525) : n = 7 :=
by 
  sorry

end NUMINAMATH_GPT_find_third_number_l552_55273


namespace NUMINAMATH_GPT_sin_neg_225_eq_sqrt2_div2_l552_55215

theorem sin_neg_225_eq_sqrt2_div2 :
  Real.sin (-225 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_neg_225_eq_sqrt2_div2_l552_55215


namespace NUMINAMATH_GPT_age_of_cat_l552_55299

variables (cat_age rabbit_age dog_age : ℕ)

-- Conditions
def condition1 : Prop := rabbit_age = cat_age / 2
def condition2 : Prop := dog_age = 3 * rabbit_age
def condition3 : Prop := dog_age = 12

-- Question
def question (cat_age : ℕ) : Prop := cat_age = 8

theorem age_of_cat (h1 : condition1 cat_age rabbit_age) (h2 : condition2 rabbit_age dog_age) (h3 : condition3 dog_age) : question cat_age :=
by
  sorry

end NUMINAMATH_GPT_age_of_cat_l552_55299


namespace NUMINAMATH_GPT_algebraic_expression_value_l552_55267

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 2 * x - 1 = 0) : x^3 - x^2 - 3 * x + 2 = 3 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l552_55267


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l552_55226

-- Define the first equation
def equation1 (x : ℝ) : Prop :=
  2 * x^2 = 3 * (2 * x + 1)

-- Define the solution set for the first equation
def solution1 (x : ℝ) : Prop :=
  x = (3 + Real.sqrt 15) / 2 ∨ x = (3 - Real.sqrt 15) / 2

-- Prove that the solutions for the first equation are correct
theorem solve_equation1 (x : ℝ) : equation1 x ↔ solution1 x :=
by
  sorry

-- Define the second equation
def equation2 (x : ℝ) : Prop :=
  3 * x * (x + 2) = 4 * x + 8

-- Define the solution set for the second equation
def solution2 (x : ℝ) : Prop :=
  x = -2 ∨ x = 4 / 3

-- Prove that the solutions for the second equation are correct
theorem solve_equation2 (x : ℝ) : equation2 x ↔ solution2 x :=
by
  sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l552_55226


namespace NUMINAMATH_GPT_find_number_of_As_l552_55205

variables (M L S : ℕ)

def number_of_As (M L S : ℕ) : Prop :=
  M + L = 23 ∧ S + M = 18 ∧ S + L = 15

theorem find_number_of_As (M L S : ℕ) (h : number_of_As M L S) :
  M = 13 ∧ L = 10 ∧ S = 5 := by
  sorry

end NUMINAMATH_GPT_find_number_of_As_l552_55205


namespace NUMINAMATH_GPT_min_area_is_fifteen_l552_55291

variable (L W : ℕ)

def minimum_possible_area (L W : ℕ) : ℕ :=
  if L = 3 ∧ W = 5 then 3 * 5 else 0

theorem min_area_is_fifteen (hL : 3 ≤ L ∧ L ≤ 5) (hW : 5 ≤ W ∧ W ≤ 7) : 
  minimum_possible_area 3 5 = 15 := 
by
  sorry

end NUMINAMATH_GPT_min_area_is_fifteen_l552_55291


namespace NUMINAMATH_GPT_pyramid_volume_l552_55284

noncomputable def volume_of_pyramid (a b c : ℝ) : ℝ :=
  (1 / 3) * a * b * c * Real.sqrt 2

theorem pyramid_volume (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (A1 : ∃ x y, 1 / 2 * x * y = a^2) 
  (A2 : ∃ y z, 1 / 2 * y * z = b^2) 
  (A3 : ∃ x z, 1 / 2 * x * z = c^2)
  (h_perpendicular : True) :
  volume_of_pyramid a b c = (1 / 3) * a * b * c * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_pyramid_volume_l552_55284


namespace NUMINAMATH_GPT_problem_solution_l552_55212

theorem problem_solution :
  0.45 * 0.65 + 0.1 * 0.2 = 0.3125 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l552_55212


namespace NUMINAMATH_GPT_scientific_notation_123000_l552_55289

theorem scientific_notation_123000 : (123000 : ℝ) = 1.23 * 10^5 := by
  sorry

end NUMINAMATH_GPT_scientific_notation_123000_l552_55289


namespace NUMINAMATH_GPT_box_dimensions_l552_55244

-- Given conditions
variables (a b c : ℕ)
axiom h1 : a + c = 17
axiom h2 : a + b = 13
axiom h3 : b + c = 20

theorem box_dimensions : a = 5 ∧ b = 8 ∧ c = 12 :=
by {
  -- These parts will contain the actual proof, which we omit for now
  sorry
}

end NUMINAMATH_GPT_box_dimensions_l552_55244


namespace NUMINAMATH_GPT_sum_of_integers_l552_55236

theorem sum_of_integers (m n : ℕ) (h1 : m * n = 2 * (m + n)) (h2 : m * n = 6 * (m - n)) :
  m + n = 9 := by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l552_55236


namespace NUMINAMATH_GPT_find_polynomial_P_l552_55239

noncomputable def P (x : ℝ) : ℝ :=
  - (5/8) * x^3 + (5/2) * x^2 + (1/8) * x - 1

theorem find_polynomial_P 
  (α β γ : ℝ)
  (h_roots : ∀ {x: ℝ}, x^3 - 4 * x^2 + 6 * x + 8 = 0 → x = α ∨ x = β ∨ x = γ)
  (h1 : P α = β + γ)
  (h2 : P β = α + γ)
  (h3 : P γ = α + β)
  (h4 : P (α + β + γ) = -20) :
  P x = - (5/8) * x^3 + (5/2) * x^2 + (1/8) * x - 1 :=
by sorry

end NUMINAMATH_GPT_find_polynomial_P_l552_55239


namespace NUMINAMATH_GPT_value_of_a_l552_55247

theorem value_of_a (a b : ℝ) (h1 : b = 2 * a) (h2 : b = 15 - 4 * a) : a = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l552_55247


namespace NUMINAMATH_GPT_exists_pairs_of_stops_l552_55296

def problem := ∃ (A1 B1 A2 B2 : Fin 6) (h1 : A1 < B1) (h2 : A2 < B2),
  (A1 ≠ A2 ∧ A1 ≠ B2 ∧ B1 ≠ A2 ∧ B1 ≠ B2) ∧
  ¬(∃ (a b : Fin 6), A1 = a ∧ B1 = b ∧ A2 = a ∧ B2 = b) -- such that no passenger boards at A1 and alights at B1
                                                              -- and no passenger boards at A2 and alights at B2.

theorem exists_pairs_of_stops (n : ℕ) (stops : Fin n) (max_passengers : ℕ) 
  (h : n = 6 ∧ max_passengers = 5 ∧ 
  ∀ (a b : Fin n), a < b → a < stops ∧ b < stops) : problem :=
sorry

end NUMINAMATH_GPT_exists_pairs_of_stops_l552_55296


namespace NUMINAMATH_GPT_total_books_bought_l552_55210

-- Let x be the number of math books and y be the number of history books
variables (x y : ℕ)

-- Conditions
def math_book_cost := 4
def history_book_cost := 5
def total_price := 368
def num_math_books := 32

-- The total number of books bought is the sum of the number of math books and history books, which should result in 80
theorem total_books_bought : 
  y * history_book_cost + num_math_books * math_book_cost = total_price → 
  x = num_math_books → 
  x + y = 80 :=
by
  sorry

end NUMINAMATH_GPT_total_books_bought_l552_55210


namespace NUMINAMATH_GPT_minimum_wins_l552_55234

theorem minimum_wins (x y : ℕ) (h_score : 3 * x + y = 10) (h_games : x + y ≤ 7) (h_bounds : 0 < x ∧ x < 4) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_wins_l552_55234


namespace NUMINAMATH_GPT_min_value_of_expression_l552_55201

theorem min_value_of_expression (x y : ℝ) (hposx : x > 0) (hposy : y > 0) (heq : 2 / x + 1 / y = 1) : 
  x + 2 * y ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l552_55201


namespace NUMINAMATH_GPT_algebra_ineq_a2_b2_geq_2_l552_55286

theorem algebra_ineq_a2_b2_geq_2
  (a b : ℝ)
  (h1 : a^3 - b^3 = 2)
  (h2 : a^5 - b^5 ≥ 4) :
  a^2 + b^2 ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_algebra_ineq_a2_b2_geq_2_l552_55286


namespace NUMINAMATH_GPT_kho_kho_only_l552_55245

variable (K H B : ℕ)

theorem kho_kho_only :
  (K + B = 10) ∧ (H + 5 = H + B) ∧ (B = 5) ∧ (K + H + B = 45) → H = 35 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_kho_kho_only_l552_55245


namespace NUMINAMATH_GPT_derivative_at_1_l552_55258

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.log x

theorem derivative_at_1 : deriv f 1 = 1 + Real.cos 1 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_1_l552_55258


namespace NUMINAMATH_GPT_total_cost_l552_55253

def daily_rental_cost : ℝ := 25
def cost_per_mile : ℝ := 0.20
def duration_days : ℕ := 4
def distance_miles : ℕ := 400

theorem total_cost 
: (daily_rental_cost * duration_days + cost_per_mile * distance_miles) = 180 := 
by
  sorry

end NUMINAMATH_GPT_total_cost_l552_55253


namespace NUMINAMATH_GPT_product_ends_in_36_l552_55294

theorem product_ends_in_36 (a b : ℕ) (ha : a < 10) (hb : b < 10) :
  ((10 * a + 6) * (10 * b + 6)) % 100 = 36 ↔ (a + b = 0 ∨ a + b = 5 ∨ a + b = 10 ∨ a + b = 15) :=
by
  sorry

end NUMINAMATH_GPT_product_ends_in_36_l552_55294


namespace NUMINAMATH_GPT_investment_difference_l552_55257

noncomputable def compound_yearly (P : ℕ) (r : ℚ) (t : ℕ) : ℚ :=
  P * (1 + r)^t

noncomputable def compound_monthly (P : ℕ) (r : ℚ) (months : ℕ) : ℚ :=
  P * (1 + r)^(months)

theorem investment_difference :
  let P := 70000
  let r := 0.05
  let t := 3
  let monthly_r := r / 12
  let months := t * 12
  compound_monthly P monthly_r months - compound_yearly P r t = 263.71 :=
by
  sorry

end NUMINAMATH_GPT_investment_difference_l552_55257


namespace NUMINAMATH_GPT_correct_polynomials_are_l552_55209

noncomputable def polynomial_solution (p : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, p.eval (x^2) = (p.eval x) * (p.eval (x - 1))

theorem correct_polynomials_are (p : Polynomial ℝ) :
  polynomial_solution p ↔ ∃ n : ℕ, p = (Polynomial.C (1 : ℝ) * Polynomial.X ^ 2 + Polynomial.C (1 : ℝ) * Polynomial.X + Polynomial.C (1 : ℝ)) ^ n :=
by
  sorry

end NUMINAMATH_GPT_correct_polynomials_are_l552_55209


namespace NUMINAMATH_GPT_rachel_picked_apples_l552_55281

-- Define relevant variables based on problem conditions
variable (trees : ℕ) (apples_per_tree : ℕ) (remaining_apples : ℕ)
variable (total_apples_picked : ℕ)

-- Assume the given conditions
axiom num_trees : trees = 4
axiom apples_each_tree : apples_per_tree = 7
axiom apples_left : remaining_apples = 29

-- Define the number of apples picked
def total_apples_picked_def := trees * apples_per_tree

-- State the theorem to prove the total apples picked
theorem rachel_picked_apples :
  total_apples_picked_def trees apples_per_tree = 28 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_rachel_picked_apples_l552_55281


namespace NUMINAMATH_GPT_hungarian_1905_l552_55288

open Nat

theorem hungarian_1905 (n p : ℕ) : (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + p * y = n ∧ x + y = p^z) ↔ 
  (p > 1 ∧ (n - 1) % (p - 1) = 0 ∧ ¬ ∃ k : ℕ, n = p^k) :=
by
  sorry

end NUMINAMATH_GPT_hungarian_1905_l552_55288


namespace NUMINAMATH_GPT_oliver_cycling_distance_l552_55208

/-- Oliver has a training loop for his weekend cycling. He starts by cycling due north for 3 miles. 
  Then he cycles northeast, making a 30° angle with the north for 2 miles, followed by cycling 
  southeast, making a 60° angle with the south for 2 miles. He completes his loop by cycling 
  directly back to the starting point. Prove that the distance of this final segment of his ride 
  is √(11 + 6√3) miles. -/
theorem oliver_cycling_distance :
  let north_displacement : ℝ := 3
  let northeast_displacement : ℝ := 2
  let northeast_angle : ℝ := 30
  let southeast_displacement : ℝ := 2
  let southeast_angle : ℝ := 60
  let north_northeast : ℝ := northeast_displacement * Real.cos (northeast_angle * Real.pi / 180)
  let east_northeast : ℝ := northeast_displacement * Real.sin (northeast_angle * Real.pi / 180)
  let south_southeast : ℝ := southeast_displacement * Real.cos (southeast_angle * Real.pi / 180)
  let east_southeast : ℝ := southeast_displacement * Real.sin (southeast_angle * Real.pi / 180)
  let total_north : ℝ := north_displacement + north_northeast - south_southeast
  let total_east : ℝ := east_northeast + east_southeast
  total_north = 2 + Real.sqrt 3 ∧ total_east = 1 + Real.sqrt 3
  → Real.sqrt (total_north^2 + total_east^2) = Real.sqrt (11 + 6 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_oliver_cycling_distance_l552_55208


namespace NUMINAMATH_GPT_savings_calculation_l552_55222

-- Define the conditions as given in the problem
def income_expenditure_ratio (income expenditure : ℝ) : Prop :=
  ∃ x : ℝ, income = 10 * x ∧ expenditure = 4 * x

def income_value : ℝ := 19000

-- The final statement for the savings, where we will prove the above question == answer
theorem savings_calculation (income expenditure savings : ℝ)
  (h_ratio : income_expenditure_ratio income expenditure)
  (h_income : income = income_value) : savings = 11400 :=
by
  sorry

end NUMINAMATH_GPT_savings_calculation_l552_55222


namespace NUMINAMATH_GPT_second_smallest_relative_prime_210_l552_55285

theorem second_smallest_relative_prime_210 (x : ℕ) (h1 : x > 1) (h2 : Nat.gcd x 210 = 1) : x = 13 :=
sorry

end NUMINAMATH_GPT_second_smallest_relative_prime_210_l552_55285


namespace NUMINAMATH_GPT_minimum_m_value_l552_55256

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * Real.log x + 1

theorem minimum_m_value :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Ici (3 : ℝ) → x2 ∈ Set.Ici (3 : ℝ) → x1 ≠ x2 →
     ∃ a : ℝ, a ∈ Set.Icc (1 : ℝ) (2 : ℝ) ∧
     (f x1 a - f x2 a) / (x2 - x1) < m) →
  m ≥ -20 / 3 := sorry

end NUMINAMATH_GPT_minimum_m_value_l552_55256


namespace NUMINAMATH_GPT_find_m_value_l552_55227

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V)
variables (m : ℝ)
variables (A B C D : V)

-- Assuming vectors a and b are non-collinear
axiom non_collinear (ha : a ≠ 0) (hb : b ≠ 0) : ¬ (∃ (k : ℝ), a = k • b)

-- Given vectors
axiom hAB : B - A = 9 • a + m • b
axiom hBC : C - B = -2 • a - 1 • b
axiom hDC : C - D = a - 2 • b

-- Collinearity condition for A, B, and D
axiom collinear (k : ℝ) : B - A = k • (B - D)

theorem find_m_value : m = -3 :=
by sorry

end NUMINAMATH_GPT_find_m_value_l552_55227


namespace NUMINAMATH_GPT_circles_intersect_and_common_chord_l552_55229

open Real

def circle1 (x y : ℝ) := x^2 + y^2 - 6 * x - 6 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4 * y - 6 = 0

theorem circles_intersect_and_common_chord :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y) ∧ (∀ x y : ℝ, circle1 x y → circle2 x y → 3 * x - 2 * y = 0) :=
by
  sorry

end NUMINAMATH_GPT_circles_intersect_and_common_chord_l552_55229


namespace NUMINAMATH_GPT_grocery_cost_l552_55266

def rent : ℕ := 1100
def utilities : ℕ := 114
def roommate_payment : ℕ := 757

theorem grocery_cost (total_payment : ℕ) (half_rent_utilities : ℕ) (half_groceries : ℕ) (total_groceries : ℕ) :
  total_payment = 757 →
  half_rent_utilities = (rent + utilities) / 2 →
  half_groceries = total_payment - half_rent_utilities →
  total_groceries = half_groceries * 2 →
  total_groceries = 300 :=
by
  intros
  sorry

end NUMINAMATH_GPT_grocery_cost_l552_55266


namespace NUMINAMATH_GPT_cost_prices_three_watches_l552_55292

theorem cost_prices_three_watches :
  ∃ (C1 C2 C3 : ℝ), 
    (0.9 * C1 + 210 = 1.04 * C1) ∧ 
    (0.85 * C2 + 180 = 1.03 * C2) ∧ 
    (0.95 * C3 + 250 = 1.06 * C3) ∧ 
    C1 = 1500 ∧ 
    C2 = 1000 ∧ 
    C3 = (25000 / 11) :=
by 
  sorry

end NUMINAMATH_GPT_cost_prices_three_watches_l552_55292


namespace NUMINAMATH_GPT_find_a_l552_55287

theorem find_a (a : ℝ) (h : ((2 * a + 16) + (3 * a - 8)) / 2 = 89) : a = 34 :=
sorry

end NUMINAMATH_GPT_find_a_l552_55287


namespace NUMINAMATH_GPT_largest_A_smallest_A_l552_55235

noncomputable def is_coprime_with_12 (n : Nat) : Prop :=
  Nat.gcd n 12 = 1

noncomputable def rotated_number (n : Nat) : Option Nat :=
  if n < 10^7 then none else
  let b := n % 10
  let k := n / 10
  some (b * 10^7 + k)

noncomputable def satisfies_conditions (B : Nat) : Prop :=
  B > 44444444 ∧ is_coprime_with_12 B

theorem largest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 99999998 :=
sorry

theorem smallest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 14444446 :=
sorry

end NUMINAMATH_GPT_largest_A_smallest_A_l552_55235


namespace NUMINAMATH_GPT_differences_occur_10_times_l552_55207

variable (a : Fin 45 → Nat)

theorem differences_occur_10_times 
    (h : ∀ i j : Fin 44, i < j → a i < a j)
    (h_lt_125 : ∀ i : Fin 44, a i < 125) :
    ∃ i : Fin 43, ∃ j : Fin 43, i ≠ j ∧ (a (i + 1) - a i) = (a (j + 1) - a j) ∧ 
    (∃ k : Nat, k ≥ 10 ∧ (a (j + 1) - a j) = (a (k + 1) - a k)) :=
sorry

end NUMINAMATH_GPT_differences_occur_10_times_l552_55207


namespace NUMINAMATH_GPT_basketball_weight_l552_55240

variable (b c : ℝ)

theorem basketball_weight (h1 : 9 * b = 5 * c) (h2 : 3 * c = 75) : b = 125 / 9 :=
by
  sorry

end NUMINAMATH_GPT_basketball_weight_l552_55240


namespace NUMINAMATH_GPT_total_fruits_proof_l552_55298

-- Definitions of the quantities involved in the problem.
def apples_basket1_to_3 := 9
def oranges_basket1_to_3 := 15
def bananas_basket1_to_3 := 14
def apples_basket4 := apples_basket1_to_3 - 2
def oranges_basket4 := oranges_basket1_to_3 - 2
def bananas_basket4 := bananas_basket1_to_3 - 2

-- Total fruits in first three baskets
def total_fruits_baskets1_to_3 := 3 * (apples_basket1_to_3 + oranges_basket1_to_3 + bananas_basket1_to_3)

-- Total fruits in fourth basket
def total_fruits_basket4 := apples_basket4 + oranges_basket4 + bananas_basket4

-- Total fruits in all four baskets
def total_fruits_all_baskets := total_fruits_baskets1_to_3 + total_fruits_basket4

-- Theorem statement
theorem total_fruits_proof : total_fruits_all_baskets = 146 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_total_fruits_proof_l552_55298


namespace NUMINAMATH_GPT_train_speed_l552_55202

theorem train_speed (distance : ℝ) (time_minutes : ℝ) (speed : ℝ) (h_distance : distance = 7.5) (h_time : time_minutes = 5) :
  speed = 90 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l552_55202


namespace NUMINAMATH_GPT_no_such_base_exists_l552_55246

theorem no_such_base_exists : ¬ ∃ b : ℕ, (b^3 ≤ 630 ∧ 630 < b^4) ∧ (630 % b) % 2 = 1 := by
  sorry

end NUMINAMATH_GPT_no_such_base_exists_l552_55246


namespace NUMINAMATH_GPT_condition1_condition2_l552_55243

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (m + 1, 2 * m - 4)

-- Define the point A
def A : ℝ × ℝ := (-5, 2)

-- Condition 1: P lies on the x-axis
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

-- Condition 2: AP is parallel to the y-axis
def parallel_y_axis (a p : ℝ × ℝ) : Prop := a.1 = p.1

-- Prove the conditions
theorem condition1 (m : ℝ) (h : on_x_axis (P m)) : P m = (3, 0) :=
by
  sorry

theorem condition2 (m : ℝ) (h : parallel_y_axis A (P m)) : P m = (-5, -16) :=
by
  sorry

end NUMINAMATH_GPT_condition1_condition2_l552_55243


namespace NUMINAMATH_GPT_max_children_l552_55264

theorem max_children (x : ℕ) (h1 : x * (x - 2) + 2 * 5 = 58) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_max_children_l552_55264


namespace NUMINAMATH_GPT_no_real_roots_l552_55262

theorem no_real_roots 
    (h : ∀ x : ℝ, (3 * x^2 / (x - 2)) - (3 * x + 8) / 2 + (5 - 9 * x) / (x - 2) + 2 = 0) 
    : False := by
  sorry

end NUMINAMATH_GPT_no_real_roots_l552_55262


namespace NUMINAMATH_GPT_product_is_correct_l552_55260

-- Define the numbers a and b
def a : ℕ := 72519
def b : ℕ := 9999

-- Theorem statement that proves the correctness of the product
theorem product_is_correct : a * b = 725117481 :=
by
  sorry

end NUMINAMATH_GPT_product_is_correct_l552_55260


namespace NUMINAMATH_GPT_sum_and_product_of_roots_l552_55237

-- Define the polynomial equation and the conditions on the roots
def cubic_eqn (x : ℝ) : Prop := 3 * x ^ 3 - 18 * x ^ 2 + 27 * x - 6 = 0

-- The Lean statement for the given problem
theorem sum_and_product_of_roots (p q r : ℝ) :
  cubic_eqn p ∧ cubic_eqn q ∧ cubic_eqn r →
  (p + q + r = 6) ∧ (p * q * r = 2) :=
by
  sorry

end NUMINAMATH_GPT_sum_and_product_of_roots_l552_55237


namespace NUMINAMATH_GPT_cube_volume_l552_55228

theorem cube_volume (SA : ℕ) (h : SA = 294) : 
  ∃ V : ℕ, V = 343 := 
by
  sorry

end NUMINAMATH_GPT_cube_volume_l552_55228


namespace NUMINAMATH_GPT_delivery_boxes_l552_55203

-- Define the conditions
def stops : ℕ := 3
def boxes_per_stop : ℕ := 9

-- Define the total number of boxes
def total_boxes : ℕ := stops * boxes_per_stop

-- State the theorem
theorem delivery_boxes : total_boxes = 27 := by
  sorry

end NUMINAMATH_GPT_delivery_boxes_l552_55203
