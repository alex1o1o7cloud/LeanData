import Mathlib

namespace NUMINAMATH_GPT_find_sum_of_squares_l1669_166910

theorem find_sum_of_squares (x y : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x * y + x + y = 35) (h4 : x^2 * y + x * y^2 = 210) : x^2 + y^2 = 154 :=
sorry

end NUMINAMATH_GPT_find_sum_of_squares_l1669_166910


namespace NUMINAMATH_GPT_like_term_l1669_166993

theorem like_term (a : ℝ) : ∃ (a : ℝ), a * x ^ 5 * y ^ 3 = a * x ^ 5 * y ^ 3 :=
by sorry

end NUMINAMATH_GPT_like_term_l1669_166993


namespace NUMINAMATH_GPT_dasha_strip_dimensions_l1669_166907

theorem dasha_strip_dimensions (a b c : ℕ) (h1 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43) : 
  (a = 1 ∧ (b + c = 22)) ∨ (a = 22 ∧ (b + c = 1)) :=
by sorry

end NUMINAMATH_GPT_dasha_strip_dimensions_l1669_166907


namespace NUMINAMATH_GPT_volleyball_team_arrangements_l1669_166978

theorem volleyball_team_arrangements (n : ℕ) (n_pos : 0 < n) :
  ∃ arrangements : ℕ, arrangements = 2^n * (Nat.factorial n)^2 :=
sorry

end NUMINAMATH_GPT_volleyball_team_arrangements_l1669_166978


namespace NUMINAMATH_GPT_fair_collection_l1669_166940

theorem fair_collection 
  (children : ℕ) (fee_child : ℝ) (adults : ℕ) (fee_adult : ℝ) 
  (total_people : ℕ) (count_children : ℕ) (count_adults : ℕ)
  (total_collected: ℝ) :
  children = 700 →
  fee_child = 1.5 →
  adults = 1500 →
  fee_adult = 4.0 →
  total_people = children + adults →
  count_children = 700 →
  count_adults = 1500 →
  total_collected = (count_children * fee_child) + (count_adults * fee_adult) →
  total_collected = 7050 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fair_collection_l1669_166940


namespace NUMINAMATH_GPT_solve_eq_l1669_166914

theorem solve_eq :
  { x : ℝ | (14 * x - x^2) / (x + 2) * (x + (14 - x) / (x + 2)) = 48 } =
  {4, (1 + Real.sqrt 193) / 2, (1 - Real.sqrt 193) / 2} :=
by
  sorry

end NUMINAMATH_GPT_solve_eq_l1669_166914


namespace NUMINAMATH_GPT_exam_standard_deviation_l1669_166938

-- Define the mean score
def mean_score : ℝ := 74

-- Define the standard deviation and conditions
def standard_deviation (σ : ℝ) : Prop :=
  mean_score - 2 * σ = 58

-- Define the condition to prove
def standard_deviation_above_mean (σ : ℝ) : Prop :=
  (98 - mean_score) / σ = 3

theorem exam_standard_deviation {σ : ℝ} (h1 : standard_deviation σ) : standard_deviation_above_mean σ :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_exam_standard_deviation_l1669_166938


namespace NUMINAMATH_GPT_odd_f_neg1_l1669_166920

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 
  if 0 ≤ x 
  then 2^x + 2 * x + b 
  else - (2^(-x) + 2 * (-x) + b)

theorem odd_f_neg1 (b : ℝ) (h : f 0 b = 0) : f (-1) b = -3 :=
by
  sorry

end NUMINAMATH_GPT_odd_f_neg1_l1669_166920


namespace NUMINAMATH_GPT_amoeba_count_after_two_weeks_l1669_166985

theorem amoeba_count_after_two_weeks :
  let initial_day_count := 1
  let days_double_split := 7
  let days_triple_split := 7
  let end_of_first_phase := initial_day_count * 2 ^ days_double_split
  let final_amoeba_count := end_of_first_phase * 3 ^ days_triple_split
  final_amoeba_count = 279936 :=
by
  sorry

end NUMINAMATH_GPT_amoeba_count_after_two_weeks_l1669_166985


namespace NUMINAMATH_GPT_wholesale_price_of_pen_l1669_166947

-- Definitions and conditions
def wholesale_price (P : ℝ) : Prop :=
  (5 - P = 10 - 3 * P)

-- Statement of the proof problem
theorem wholesale_price_of_pen : ∃ P : ℝ, wholesale_price P ∧ P = 2.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_wholesale_price_of_pen_l1669_166947


namespace NUMINAMATH_GPT_largest_number_of_gold_coins_l1669_166955

theorem largest_number_of_gold_coins (n : ℕ) (h1 : n % 15 = 4) (h2 : n < 150) : n ≤ 139 :=
by {
  -- This is where the proof would go.
  sorry
}

end NUMINAMATH_GPT_largest_number_of_gold_coins_l1669_166955


namespace NUMINAMATH_GPT_range_of_x_l1669_166990

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) :
  (2 * Real.cos x ≤ abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ∧
   abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ≤ Real.sqrt 2)
  ↔ (Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1669_166990


namespace NUMINAMATH_GPT_integer_root_abs_sum_l1669_166924

noncomputable def solve_abs_sum (p q r : ℤ) : ℤ := |p| + |q| + |r|

theorem integer_root_abs_sum (p q r m : ℤ) 
  (h1 : p + q + r = 0)
  (h2 : p * q + q * r + r * p = -2024)
  (h3 : ∃ m, ∀ x, x^3 - 2024 * x + m = (x - p) * (x - q) * (x - r)) :
  solve_abs_sum p q r = 104 :=
by sorry

end NUMINAMATH_GPT_integer_root_abs_sum_l1669_166924


namespace NUMINAMATH_GPT_shape_with_congruent_views_is_sphere_l1669_166992

def is_congruent_views (shape : Type) : Prop :=
  ∀ (front_view left_view top_view : shape), 
  (front_view = left_view) ∧ (left_view = top_view) ∧ (front_view = top_view)

noncomputable def is_sphere (shape : Type) : Prop := 
  ∀ (s : shape), true -- Placeholder definition for a sphere, as recognizing a sphere is outside Lean's scope

theorem shape_with_congruent_views_is_sphere (shape : Type) :
  is_congruent_views shape → is_sphere shape :=
by
  intro h
  sorry

end NUMINAMATH_GPT_shape_with_congruent_views_is_sphere_l1669_166992


namespace NUMINAMATH_GPT_oranges_given_to_friend_l1669_166942

theorem oranges_given_to_friend (initial_oranges : ℕ) 
  (given_to_brother : ℕ)
  (given_to_friend : ℕ)
  (h1 : initial_oranges = 60)
  (h2 : given_to_brother = (1 / 3 : ℚ) * initial_oranges)
  (h3 : given_to_friend = (1 / 4 : ℚ) * (initial_oranges - given_to_brother)) : 
  given_to_friend = 10 := 
by 
  sorry

end NUMINAMATH_GPT_oranges_given_to_friend_l1669_166942


namespace NUMINAMATH_GPT_minimum_number_is_correct_l1669_166922

-- Define the operations and conditions on the digits
def transform (n : ℕ) : ℕ :=
if 2 ≤ n then n - 2 + 1 else n

noncomputable def minimum_transformed_number (l : List ℕ) : List ℕ :=
l.map transform

def initial_number : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def expected_number : List ℕ := [1, 0, 1, 0, 1, 0, 1, 0, 1]

theorem minimum_number_is_correct :
  minimum_transformed_number initial_number = expected_number := 
by
  -- sorry is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_minimum_number_is_correct_l1669_166922


namespace NUMINAMATH_GPT_sequence_a5_l1669_166972

/-- In the sequence {a_n}, with a_1 = 1, a_2 = 2, and a_(n+2) = 2 * a_(n+1) + a_n, prove that a_5 = 29. -/
theorem sequence_a5 (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 2) (h_rec : ∀ n, a (n + 2) = 2 * a (n + 1) + a n) :
  a 5 = 29 :=
sorry

end NUMINAMATH_GPT_sequence_a5_l1669_166972


namespace NUMINAMATH_GPT_prove_lesser_fraction_l1669_166977

noncomputable def lesser_fraction (x y : ℚ) : Prop :=
  x + y = 8/9 ∧ x * y = 1/8 ∧ min x y = 7/40

theorem prove_lesser_fraction :
  ∃ x y : ℚ, lesser_fraction x y :=
sorry

end NUMINAMATH_GPT_prove_lesser_fraction_l1669_166977


namespace NUMINAMATH_GPT_fraction_simplification_l1669_166929

theorem fraction_simplification : 
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = (1 / 3) := 
sorry

end NUMINAMATH_GPT_fraction_simplification_l1669_166929


namespace NUMINAMATH_GPT_relationship_y1_y2_y3_l1669_166932

variables {m y_1 y_2 y_3 : ℝ}

theorem relationship_y1_y2_y3 :
  (∃ (m : ℝ), (y_1 = (-1)^2 - 2*(-1) + m) ∧ (y_2 = 2^2 - 2*2 + m) ∧ (y_3 = 3^2 - 2*3 + m)) →
  y_2 < y_1 ∧ y_1 = y_3 :=
by
  sorry

end NUMINAMATH_GPT_relationship_y1_y2_y3_l1669_166932


namespace NUMINAMATH_GPT_problem1_problem2_l1669_166948

theorem problem1 (a b : ℝ) : (-(2 : ℝ) * a ^ 2 * b) ^ 3 / (-(2 * a * b)) * (1 / 3 * a ^ 2 * b ^ 3) = (4 / 3) * a ^ 7 * b ^ 5 :=
  by
  sorry

theorem problem2 (x : ℝ) : (27 * x ^ 3 + 18 * x ^ 2 - 3 * x) / -3 * x = -9 * x ^ 2 - 6 * x + 1 :=
  by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1669_166948


namespace NUMINAMATH_GPT_number_multiplied_by_9_l1669_166900

theorem number_multiplied_by_9 (x : ℕ) (h : 50 = x + 26) : 9 * x = 216 := by
  sorry

end NUMINAMATH_GPT_number_multiplied_by_9_l1669_166900


namespace NUMINAMATH_GPT_equivalent_single_discount_l1669_166928

theorem equivalent_single_discount (p : ℝ) : 
  let discount1 := 0.15
  let discount2 := 0.25
  let price_after_first_discount := (1 - discount1) * p
  let price_after_second_discount := (1 - discount2) * price_after_first_discount
  let equivalent_single_discount := 1 - price_after_second_discount / p
  equivalent_single_discount = 0.3625 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_single_discount_l1669_166928


namespace NUMINAMATH_GPT_min_value_exists_l1669_166981

noncomputable def point_on_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 9 ∧ y ≥ 2

theorem min_value_exists : ∃ x y : ℝ, point_on_circle x y ∧ x + Real.sqrt 3 * y = 2 * Real.sqrt 3 - 2 := 
sorry

end NUMINAMATH_GPT_min_value_exists_l1669_166981


namespace NUMINAMATH_GPT_BD_value_l1669_166950

def quadrilateral_ABCD_sides (AB BC CD DA : ℕ) (BD : ℕ) : Prop :=
  AB = 5 ∧ BC = 17 ∧ CD = 5 ∧ DA = 9 ∧ 12 < BD ∧ BD < 14 ∧ BD = 13

theorem BD_value (AB BC CD DA : ℕ) (BD : ℕ) : 
  quadrilateral_ABCD_sides AB BC CD DA BD → BD = 13 :=
by
  sorry

end NUMINAMATH_GPT_BD_value_l1669_166950


namespace NUMINAMATH_GPT_total_gifts_l1669_166918

theorem total_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end NUMINAMATH_GPT_total_gifts_l1669_166918


namespace NUMINAMATH_GPT_man_salary_problem_l1669_166903

-- Define the problem in Lean 4
theorem man_salary_problem (S : ℝ) :
  (1/3 * S) + (1/4 * S) + (1/5 * S) + 1760 = S → 
  S = 8123.08 :=
sorry

end NUMINAMATH_GPT_man_salary_problem_l1669_166903


namespace NUMINAMATH_GPT_students_did_not_eat_2_l1669_166902

-- Define the given conditions
def total_students : ℕ := 20
def total_crackers_eaten : ℕ := 180
def crackers_per_pack : ℕ := 10

-- Calculate the number of packs eaten
def packs_eaten : ℕ := total_crackers_eaten / crackers_per_pack

-- Calculate the number of students who did not eat their animal crackers
def students_who_did_not_eat : ℕ := total_students - packs_eaten

-- Prove that the number of students who did not eat their animal crackers is 2
theorem students_did_not_eat_2 :
  students_who_did_not_eat = 2 :=
  by
    sorry

end NUMINAMATH_GPT_students_did_not_eat_2_l1669_166902


namespace NUMINAMATH_GPT_probability_of_detecting_non_conforming_l1669_166949

noncomputable def prob_detecting_non_conforming (total_cans non_conforming_cans selected_cans : ℕ) : ℚ :=
  let total_outcomes := Nat.choose total_cans selected_cans
  let outcomes_with_one_non_conforming := Nat.choose non_conforming_cans 1 * Nat.choose (total_cans - non_conforming_cans) (selected_cans - 1)
  let outcomes_with_two_non_conforming := Nat.choose non_conforming_cans 2
  (outcomes_with_one_non_conforming + outcomes_with_two_non_conforming) / total_outcomes

theorem probability_of_detecting_non_conforming :
  prob_detecting_non_conforming 5 2 2 = 7 / 10 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_probability_of_detecting_non_conforming_l1669_166949


namespace NUMINAMATH_GPT_max_chips_with_constraints_l1669_166911

theorem max_chips_with_constraints (n : ℕ) (h1 : n > 0) 
  (h2 : ∀ i j : ℕ, (i < n) → (j = i + 10 ∨ j = i + 15) → ((i % 25) = 0 ∨ (j % 25) = 0)) :
  n ≤ 25 := 
sorry

end NUMINAMATH_GPT_max_chips_with_constraints_l1669_166911


namespace NUMINAMATH_GPT_relay_race_time_l1669_166901

theorem relay_race_time (R S D : ℕ) (h1 : S = R + 2) (h2 : D = R - 3) (h3 : R + S + D = 71) : R = 24 :=
by
  sorry

end NUMINAMATH_GPT_relay_race_time_l1669_166901


namespace NUMINAMATH_GPT_general_term_of_arithmetic_sequence_l1669_166991

theorem general_term_of_arithmetic_sequence
  (a : ℕ → ℤ)
  (h_arith : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a3 : a 3 = -2)
  (h_a7 : a 7 = -10) :
  ∀ n : ℕ, a n = 4 - 2 * n :=
sorry

end NUMINAMATH_GPT_general_term_of_arithmetic_sequence_l1669_166991


namespace NUMINAMATH_GPT_total_gum_l1669_166930

-- Define the conditions
def original_gum : ℕ := 38
def additional_gum : ℕ := 16

-- Define the statement to be proved
theorem total_gum : original_gum + additional_gum = 54 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_gum_l1669_166930


namespace NUMINAMATH_GPT_smallest_positive_value_l1669_166916

noncomputable def exprA := 30 - 4 * Real.sqrt 14
noncomputable def exprB := 4 * Real.sqrt 14 - 30
noncomputable def exprC := 25 - 6 * Real.sqrt 15
noncomputable def exprD := 75 - 15 * Real.sqrt 30
noncomputable def exprE := 15 * Real.sqrt 30 - 75

theorem smallest_positive_value :
  exprC = 25 - 6 * Real.sqrt 15 ∧
  exprC < exprA ∧
  exprC < exprB ∧
  exprC < exprD ∧
  exprC < exprE ∧
  exprC > 0 :=
by sorry

end NUMINAMATH_GPT_smallest_positive_value_l1669_166916


namespace NUMINAMATH_GPT_jason_grass_cutting_time_l1669_166970

def total_minutes (hours : ℕ) : ℕ := hours * 60
def minutes_per_yard : ℕ := 30
def total_yards_per_weekend : ℕ := 8 * 2
def total_minutes_per_weekend : ℕ := minutes_per_yard * total_yards_per_weekend
def convert_minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

theorem jason_grass_cutting_time : 
  convert_minutes_to_hours total_minutes_per_weekend = 8 := by
  sorry

end NUMINAMATH_GPT_jason_grass_cutting_time_l1669_166970


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_ge_four_l1669_166908

theorem sum_arithmetic_sequence_ge_four
  (a_n : ℕ → ℚ) -- arithmetic sequence
  (S : ℕ → ℚ) -- sum of the first n terms of the sequence
  (h_arith_seq : ∀ n, S n = (n * a_n 1) + (n * (n - 1) / 2) * (a_n 2 - a_n 1))
  (p q : ℕ)
  (hpq_ne : p ≠ q)
  (h_sp : S p = p / q)
  (h_sq : S q = q / p) :
  S (p + q) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_ge_four_l1669_166908


namespace NUMINAMATH_GPT_solution_set_product_positive_l1669_166961

variable {R : Type*} [LinearOrderedField R]

def is_odd (f : R → R) : Prop := ∀ x : R, f (-x) = -f (x)

variable (f g : R → R)

noncomputable def solution_set_positive_f : Set R := { x | 4 < x ∧ x < 10 }
noncomputable def solution_set_positive_g : Set R := { x | 2 < x ∧ x < 5 }

theorem solution_set_product_positive :
  is_odd f →
  is_odd g →
  (∀ x, f x > 0 ↔ x ∈ solution_set_positive_f) →
  (∀ x, g x > 0 ↔ x ∈ solution_set_positive_g) →
  { x | f x * g x > 0 } = { x | (4 < x ∧ x < 5) ∨ (-5 < x ∧ x < -4) } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_product_positive_l1669_166961


namespace NUMINAMATH_GPT_complex_trajectory_is_ellipse_l1669_166946

open Complex

theorem complex_trajectory_is_ellipse (z : ℂ) (h : abs (z - i) + abs (z + i) = 3) : 
  true := 
sorry

end NUMINAMATH_GPT_complex_trajectory_is_ellipse_l1669_166946


namespace NUMINAMATH_GPT_evaluate_at_3_l1669_166989

def f (x : ℝ) : ℝ := 9 * x^4 + 7 * x^3 - 5 * x^2 + 3 * x - 6

theorem evaluate_at_3 : f 3 = 876 := by
  sorry

end NUMINAMATH_GPT_evaluate_at_3_l1669_166989


namespace NUMINAMATH_GPT_whale_length_l1669_166904

theorem whale_length
  (velocity_fast : ℕ)
  (velocity_slow : ℕ)
  (time : ℕ)
  (h1 : velocity_fast = 18)
  (h2 : velocity_slow = 15)
  (h3 : time = 15) :
  (velocity_fast - velocity_slow) * time = 45 := 
by
  sorry

end NUMINAMATH_GPT_whale_length_l1669_166904


namespace NUMINAMATH_GPT_a7_arithmetic_sequence_l1669_166915

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def a1 : ℝ := 2
def a4 : ℝ := 5

theorem a7_arithmetic_sequence : ∃ d : ℝ, is_arithmetic_sequence a d ∧ a 1 = a1 ∧ a 4 = a4 → a 7 = 8 :=
by
  sorry

end NUMINAMATH_GPT_a7_arithmetic_sequence_l1669_166915


namespace NUMINAMATH_GPT_complement_intersection_l1669_166957

open Set

-- Define the universal set U
def U : Set ℕ := {x | 0 < x ∧ x < 7}

-- Define Set A
def A : Set ℕ := {2, 3, 5}

-- Define Set B
def B : Set ℕ := {1, 4}

-- Define the complement of A in U
def CU_A : Set ℕ := U \ A

-- Define the complement of B in U
def CU_B : Set ℕ := U \ B

-- Define the intersection of CU_A and CU_B
def intersection_CU_A_CU_B : Set ℕ := CU_A ∩ CU_B

-- The theorem statement
theorem complement_intersection :
  intersection_CU_A_CU_B = {6} := by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1669_166957


namespace NUMINAMATH_GPT_correct_reaction_equation_l1669_166944

noncomputable def reaction_equation (vA vB vC : ℝ) : Prop :=
  vB = 3 * vA ∧ 3 * vC = 2 * vB

theorem correct_reaction_equation (vA vB vC : ℝ) (h : reaction_equation vA vB vC) :
  ∃ (α β γ : ℕ), α = 1 ∧ β = 3 ∧ γ = 2 :=
sorry

end NUMINAMATH_GPT_correct_reaction_equation_l1669_166944


namespace NUMINAMATH_GPT_find_m_n_diff_l1669_166951

theorem find_m_n_diff (a : ℝ) (n m: ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1)
  (h_pass : a^(2 * m - 6) + n = 2) :
  m - n = 2 :=
sorry

end NUMINAMATH_GPT_find_m_n_diff_l1669_166951


namespace NUMINAMATH_GPT_g_nine_l1669_166969

variable (g : ℝ → ℝ)

theorem g_nine : (∀ x y : ℝ, g (x + y) = g x * g y) → g 3 = 4 → g 9 = 64 :=
by intros h1 h2; sorry

end NUMINAMATH_GPT_g_nine_l1669_166969


namespace NUMINAMATH_GPT_cylinder_radius_l1669_166966

theorem cylinder_radius
  (diameter_c : ℝ) (altitude_c : ℝ) (height_relation : ℝ → ℝ)
  (same_axis : Bool) (radius_cylinder : ℝ → ℝ)
  (h1 : diameter_c = 14)
  (h2 : altitude_c = 20)
  (h3 : ∀ r, height_relation r = 3 * r)
  (h4 : same_axis = true)
  (h5 : ∀ r, radius_cylinder r = r) :
  ∃ r, r = 140 / 41 :=
by {
  sorry
}

end NUMINAMATH_GPT_cylinder_radius_l1669_166966


namespace NUMINAMATH_GPT_intersection_point_l1669_166995

noncomputable def f (x : ℝ) := (x^2 - 8 * x + 7) / (2 * x - 6)

noncomputable def g (a b c : ℝ) (x : ℝ) := (a * x^2 + b * x + c) / (x - 3)

theorem intersection_point (a b c : ℝ) :
  (∀ x, 2 * x - 6 = 0 <-> x ≠ 3) →
  ∃ (k : ℝ), (g a b c x = -2 * x - 4 + k / (x - 3)) →
  (f x = g a b c x) ∧ x ≠ -3 → x = 1 ∧ f 1 = 0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_intersection_point_l1669_166995


namespace NUMINAMATH_GPT_solve_equation_l1669_166926

theorem solve_equation (x : ℝ) : x * (x + 5)^3 * (5 - x) = 0 ↔ x = 0 ∨ x = -5 ∨ x = 5 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1669_166926


namespace NUMINAMATH_GPT_linear_coefficient_is_one_l1669_166996

-- Define the given equation and the coefficient of the linear term
variables {x m : ℝ}
def equation := (m - 3) * x + 4 * m^2 - 2 * m - 1 - m * x + 6

-- State the main theorem: the coefficient of the linear term in the equation is 1 given the conditions
theorem linear_coefficient_is_one (m : ℝ) (hm_neq_3 : m ≠ 3) :
  (m - 3) - m = 1 :=
by sorry

end NUMINAMATH_GPT_linear_coefficient_is_one_l1669_166996


namespace NUMINAMATH_GPT_nickel_ate_4_chocolates_l1669_166973

theorem nickel_ate_4_chocolates (R N : ℕ) (h1 : R = 13) (h2 : R = N + 9) : N = 4 :=
by
  sorry

end NUMINAMATH_GPT_nickel_ate_4_chocolates_l1669_166973


namespace NUMINAMATH_GPT_bigger_part_is_45_l1669_166959

variable (x y : ℕ)

theorem bigger_part_is_45
  (h1 : x + y = 60)
  (h2 : 10 * x + 22 * y = 780) :
  max x y = 45 := by
  sorry

end NUMINAMATH_GPT_bigger_part_is_45_l1669_166959


namespace NUMINAMATH_GPT_intersection_complement_eq_l1669_166933

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 4}

theorem intersection_complement_eq : A ∩ (U \ B) = {0, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l1669_166933


namespace NUMINAMATH_GPT_cost_of_each_top_l1669_166987

theorem cost_of_each_top
  (total_spent : ℝ)
  (num_shorts : ℕ)
  (price_per_short : ℝ)
  (num_shoes : ℕ)
  (price_per_shoe : ℝ)
  (num_tops : ℕ)
  (total_cost_shorts : ℝ)
  (total_cost_shoes : ℝ)
  (amount_spent_on_tops : ℝ)
  (cost_per_top : ℝ) :
  total_spent = 75 →
  num_shorts = 5 →
  price_per_short = 7 →
  num_shoes = 2 →
  price_per_shoe = 10 →
  num_tops = 4 →
  total_cost_shorts = num_shorts * price_per_short →
  total_cost_shoes = num_shoes * price_per_shoe →
  amount_spent_on_tops = total_spent - (total_cost_shorts + total_cost_shoes) →
  cost_per_top = amount_spent_on_tops / num_tops →
  cost_per_top = 5 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_top_l1669_166987


namespace NUMINAMATH_GPT_gold_beads_cannot_be_determined_without_cost_per_bead_l1669_166935

-- Carly's bead conditions
def purple_rows : ℕ := 50
def purple_beads_per_row : ℕ := 20
def blue_rows : ℕ := 40
def blue_beads_per_row : ℕ := 18
def total_cost : ℝ := 180

-- The calculation of total purple and blue beads
def purple_beads : ℕ := purple_rows * purple_beads_per_row
def blue_beads : ℕ := blue_rows * blue_beads_per_row
def total_beads_without_gold : ℕ := purple_beads + blue_beads

-- Given the lack of cost per bead, the number of gold beads cannot be determined
theorem gold_beads_cannot_be_determined_without_cost_per_bead :
  ¬ (∃ cost_per_bead : ℝ, ∃ gold_beads : ℕ, (purple_beads + blue_beads + gold_beads) * cost_per_bead = total_cost) :=
sorry

end NUMINAMATH_GPT_gold_beads_cannot_be_determined_without_cost_per_bead_l1669_166935


namespace NUMINAMATH_GPT_correct_calculation_A_incorrect_calculation_B_incorrect_calculation_C_incorrect_calculation_D_correct_answer_is_A_l1669_166927

theorem correct_calculation_A : (Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6) :=
by { sorry }

theorem incorrect_calculation_B : (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) :=
by { sorry }

theorem incorrect_calculation_C : ((Real.sqrt 2)^2 ≠ 2 * Real.sqrt 2) :=
by { sorry }

theorem incorrect_calculation_D : (2 + Real.sqrt 2 ≠ 2 * Real.sqrt 2) :=
by { sorry }

theorem correct_answer_is_A :
  (Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6) ∧
  (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) ∧
  ((Real.sqrt 2)^2 ≠ 2 * Real.sqrt 2) ∧
  (2 + Real.sqrt 2 ≠ 2 * Real.sqrt 2) :=
by {
  exact ⟨correct_calculation_A, incorrect_calculation_B, incorrect_calculation_C, incorrect_calculation_D⟩
}

end NUMINAMATH_GPT_correct_calculation_A_incorrect_calculation_B_incorrect_calculation_C_incorrect_calculation_D_correct_answer_is_A_l1669_166927


namespace NUMINAMATH_GPT_min_a_for_inequality_l1669_166954

theorem min_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x ≤ 1/2) → (x^2 + a*x + 1 ≥ 0)) ↔ a ≥ -5/2 :=
by
  sorry

end NUMINAMATH_GPT_min_a_for_inequality_l1669_166954


namespace NUMINAMATH_GPT_net_gain_mr_A_l1669_166988

def home_worth : ℝ := 12000
def sale1 : ℝ := home_worth * 1.2
def sale2 : ℝ := sale1 * 0.85
def sale3 : ℝ := sale2 * 1.1

theorem net_gain_mr_A : sale1 - sale2 + sale3 = 3384 := by
  sorry -- Proof will be provided here

end NUMINAMATH_GPT_net_gain_mr_A_l1669_166988


namespace NUMINAMATH_GPT_correct_optionD_l1669_166963

def operationA (a : ℝ) : Prop := a^3 + 3 * a^3 = 5 * a^6
def operationB (a : ℝ) : Prop := 7 * a^2 * a^3 = 7 * a^6
def operationC (a : ℝ) : Prop := (-2 * a^3)^2 = 4 * a^5
def operationD (a : ℝ) : Prop := a^8 / a^2 = a^6

theorem correct_optionD (a : ℝ) : ¬ operationA a ∧ ¬ operationB a ∧ ¬ operationC a ∧ operationD a :=
by
  unfold operationA operationB operationC operationD
  sorry

end NUMINAMATH_GPT_correct_optionD_l1669_166963


namespace NUMINAMATH_GPT_union_sets_l1669_166912

-- Definitions of sets A and B
def set_A : Set ℝ := {x | x / (x - 1) < 0}
def set_B : Set ℝ := {x | abs (1 - x) > 1 / 2}

-- The problem: prove that the union of sets A and B is (-∞, 1) ∪ (3/2, ∞)
theorem union_sets :
  set_A ∪ set_B = {x | x < 1} ∪ {x | x > 3 / 2} :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l1669_166912


namespace NUMINAMATH_GPT_find_f_of_3_l1669_166905

theorem find_f_of_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 1) = 3 * x - 5) : f 3 = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_3_l1669_166905


namespace NUMINAMATH_GPT_acai_berry_cost_correct_l1669_166976

def cost_superfruit_per_litre : ℝ := 1399.45
def cost_mixed_fruit_per_litre : ℝ := 262.85
def litres_mixed_fruit : ℝ := 36
def litres_acai_berry : ℝ := 24
def total_litres : ℝ := litres_mixed_fruit + litres_acai_berry
def expected_cost_acai_per_litre : ℝ := 3104.77

theorem acai_berry_cost_correct :
  cost_superfruit_per_litre * total_litres -
  cost_mixed_fruit_per_litre * litres_mixed_fruit = 
  expected_cost_acai_per_litre * litres_acai_berry :=
by sorry

end NUMINAMATH_GPT_acai_berry_cost_correct_l1669_166976


namespace NUMINAMATH_GPT_Jessica_cut_roses_l1669_166998

variable (initial_roses final_roses added_roses : Nat)

theorem Jessica_cut_roses
  (h_initial : initial_roses = 10)
  (h_final : final_roses = 18)
  (h_added : final_roses = initial_roses + added_roses) :
  added_roses = 8 := by
  sorry

end NUMINAMATH_GPT_Jessica_cut_roses_l1669_166998


namespace NUMINAMATH_GPT_min_value_xyz_l1669_166943

theorem min_value_xyz (x y z : ℝ) (h1 : xy + 2 * z = 1) (h2 : x^2 + y^2 + z^2 = 10 ) : xyz ≥ -28 :=
by
  sorry

end NUMINAMATH_GPT_min_value_xyz_l1669_166943


namespace NUMINAMATH_GPT_contradiction_of_distinct_roots_l1669_166936

theorem contradiction_of_distinct_roots
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (distinct_abc : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (H : ¬ (∃ x1 x2, x1 ≠ x2 ∧ (a * x1^2 + 2 * b * x1 + c = 0 ∨ b * x1^2 + 2 * c * x1 + a = 0 ∨ c * x1^2 + 2 * a * x1 + b = 0))) :
  False := 
sorry

end NUMINAMATH_GPT_contradiction_of_distinct_roots_l1669_166936


namespace NUMINAMATH_GPT_triangle_area_l1669_166979

variable (a b c : ℕ)
variable (s : ℕ := 21)
variable (area : ℕ := 84)

theorem triangle_area 
(h1 : c = a + b - 12) 
(h2 : (a + b + c) / 2 = s) 
(h3 : c - a = 2) 
: (21 * (21 - a) * (21 - b) * (21 - c)).sqrt = area := 
sorry

end NUMINAMATH_GPT_triangle_area_l1669_166979


namespace NUMINAMATH_GPT_quadratic_to_binomial_square_l1669_166965

theorem quadratic_to_binomial_square (m : ℝ) : 
  (∃ c : ℝ, (x : ℝ) → x^2 - 12 * x + m = (x + c)^2) ↔ m = 36 := 
sorry

end NUMINAMATH_GPT_quadratic_to_binomial_square_l1669_166965


namespace NUMINAMATH_GPT_value_of_trig_expression_l1669_166999

theorem value_of_trig_expression (α : Real) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = -3 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_trig_expression_l1669_166999


namespace NUMINAMATH_GPT_find_x_l1669_166953

theorem find_x (x : ℝ) (h1 : |x + 7| = 3) (h2 : x^2 + 2*x - 3 = 5) : x = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1669_166953


namespace NUMINAMATH_GPT_ratio_of_books_l1669_166974

theorem ratio_of_books (books_last_week : ℕ) (pages_per_book : ℕ) (pages_this_week : ℕ)
  (h_books_last_week : books_last_week = 5)
  (h_pages_per_book : pages_per_book = 300)
  (h_pages_this_week : pages_this_week = 4500) :
  (pages_this_week / pages_per_book) / books_last_week = 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_books_l1669_166974


namespace NUMINAMATH_GPT_find_angle_FYD_l1669_166925

noncomputable def angle_FYD (AB CD AXF FYG : ℝ) : ℝ := 180 - AXF

theorem find_angle_FYD (AB CD : ℝ) (AXF : ℝ) (FYG : ℝ) (h1 : AB = CD) (h2 : AXF = 125) (h3 : FYG = 40) :
  angle_FYD AB CD AXF FYG = 55 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_FYD_l1669_166925


namespace NUMINAMATH_GPT_max_value_proof_l1669_166980

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  x + y^3 + z^4

theorem max_value_proof
  (x y z : ℝ)
  (hx : 0 ≤ x)
  (hy : 0 ≤ y)
  (hz : 0 ≤ z)
  (h1 : x + y + z = 1)
  (h2 : x^2 + y^2 + z^2 = 1) :
  maximum_value x y z ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_proof_l1669_166980


namespace NUMINAMATH_GPT_simplify_fraction_l1669_166913

theorem simplify_fraction (x : ℝ) (h : x ≠ 2) : (x^2 / (x - 2) - 4 / (x - 2)) = x + 2 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1669_166913


namespace NUMINAMATH_GPT_f_eq_g_l1669_166958

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

variable (f_onto : ∀ m : ℕ, ∃ n : ℕ, f n = m)
variable (g_one_one : ∀ m n : ℕ, g m = g n → m = n)
variable (f_ge_g : ∀ n : ℕ, f n ≥ g n)

theorem f_eq_g : f = g :=
sorry

end NUMINAMATH_GPT_f_eq_g_l1669_166958


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_l1669_166921

-- Definitions and conditions
variable (a : ℝ) (b : ℝ) (ha : a > 0) (hb : b > 0) (hineq : a - 2 * Real.sqrt b > 0)

-- Problem 1: √(a - 2√b) = √m - √n
theorem problem1 (h₁ : a = 5) (h₂ : b = 6) : Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 := sorry

-- Problem 2: √(a + 2√b) = √m + √n
theorem problem2 (h₁ : a = 12) (h₂ : b = 35) : Real.sqrt (12 + 2 * Real.sqrt 35) = Real.sqrt 7 + Real.sqrt 5 := sorry

-- Problem 3: √(a + 6√b) = √m + √n
theorem problem3 (h₁ : a = 9) (h₂ : b = 6) : Real.sqrt (9 + 6 * Real.sqrt 2) = Real.sqrt 6 + Real.sqrt 3 := sorry

-- Problem 4: √(a - 4√b) = √m - √n
theorem problem4 (h₁ : a = 16) (h₂ : b = 60) : Real.sqrt (16 - 4 * Real.sqrt 15) = Real.sqrt 10 - Real.sqrt 6 := sorry

-- Problem 5: √(a - √b) + √(c + √d)
theorem problem5 (h₁ : a = 3) (h₂ : b = 5) (h₃ : c = 2) (h₄ : d = 3) 
  : Real.sqrt (3 - Real.sqrt 5) + Real.sqrt (2 + Real.sqrt 3) = (Real.sqrt 10 + Real.sqrt 6) / 2 := sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_l1669_166921


namespace NUMINAMATH_GPT_ellipse_focal_length_l1669_166952

theorem ellipse_focal_length :
  let a_squared := 20
    let b_squared := 11
    let c := Real.sqrt (a_squared - b_squared)
    let focal_length := 2 * c
  11 * x^2 + 20 * y^2 = 220 →
  focal_length = 6 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_focal_length_l1669_166952


namespace NUMINAMATH_GPT_socks_ratio_l1669_166962

-- Definitions based on the conditions
def initial_black_socks : ℕ := 6
def initial_white_socks (B : ℕ) : ℕ := 4 * B
def remaining_white_socks (B : ℕ) : ℕ := B + 6

-- The theorem to prove the ratio is 1/2
theorem socks_ratio (B : ℕ) (hB : B = initial_black_socks) :
  ((initial_white_socks B - remaining_white_socks B) : ℚ) / initial_white_socks B = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_socks_ratio_l1669_166962


namespace NUMINAMATH_GPT_solve_for_x_l1669_166941

variable (x y : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (h : 3 * x^2 + 9 * x * y = x^3 + 3 * x^2 * y)

theorem solve_for_x : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1669_166941


namespace NUMINAMATH_GPT_number_of_divisors_not_multiples_of_14_l1669_166997

theorem number_of_divisors_not_multiples_of_14 
  (n : ℕ)
  (h1: ∃ k : ℕ, n = 2 * k * k)
  (h2: ∃ k : ℕ, n = 3 * k * k * k)
  (h3: ∃ k : ℕ, n = 5 * k * k * k * k * k)
  (h4: ∃ k : ℕ, n = 7 * k * k * k * k * k * k * k)
  : 
  ∃ num_divisors : ℕ, num_divisors = 19005 ∧ (∀ d : ℕ, d ∣ n → ¬(14 ∣ d)) := sorry

end NUMINAMATH_GPT_number_of_divisors_not_multiples_of_14_l1669_166997


namespace NUMINAMATH_GPT_fraction_inequality_solution_l1669_166968

theorem fraction_inequality_solution (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3) :
  3 * x + 2 < 2 * (5 * x - 4) → (10 / 7) < x ∧ x ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_inequality_solution_l1669_166968


namespace NUMINAMATH_GPT_probability_even_sum_l1669_166964

-- Defining the probabilities for the first wheel
def P_even_1 : ℚ := 2/3
def P_odd_1 : ℚ := 1/3

-- Defining the probabilities for the second wheel
def P_even_2 : ℚ := 1/2
def P_odd_2 : ℚ := 1/2

-- Prove that the probability that the sum of the two selected numbers is even is 1/2
theorem probability_even_sum : 
  P_even_1 * P_even_2 + P_odd_1 * P_odd_2 = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_probability_even_sum_l1669_166964


namespace NUMINAMATH_GPT_right_triangle_iff_sum_excircles_a_right_triangle_iff_sum_excircles_b_l1669_166937

-- Definitions for part (a)
def is_right_angled_triangle_a (a b c r r_a r_b r_c : ℝ) :=
  r + r_a + r_b + r_c = a + b + c

def right_angled_triangle_a (a b c : ℝ) :=
  a^2 + b^2 = c^2 -- Assuming c is the hypotenuse

theorem right_triangle_iff_sum_excircles_a (a b c r r_a r_b r_c : ℝ) :
  right_angled_triangle_a a b c ↔ is_right_angled_triangle_a a b c r r_a r_b r_c := sorry

-- Definitions for part (b)
def is_right_angled_triangle_b (a b c r r_a r_b r_c : ℝ) :=
  r^2 + r_a^2 + r_b^2 + r_c^2 = a^2 + b^2 + c^2

def right_angled_triangle_b (a b c : ℝ) :=
  a^2 + b^2 = c^2 -- Assuming c is the hypotenuse

theorem right_triangle_iff_sum_excircles_b (a b c r r_a r_b r_c : ℝ) :
  right_angled_triangle_b a b c ↔ is_right_angled_triangle_b a b c r r_a r_b r_c := sorry

end NUMINAMATH_GPT_right_triangle_iff_sum_excircles_a_right_triangle_iff_sum_excircles_b_l1669_166937


namespace NUMINAMATH_GPT_angle_of_inclination_l1669_166994

/--
Given the direction vector of line l as (-sqrt(3), 3),
prove that the angle of inclination α of line l is 120 degrees.
-/
theorem angle_of_inclination (α : ℝ) :
  let direction_vector : Real × Real := (-Real.sqrt 3, 3)
  let slope := direction_vector.2 / direction_vector.1
  slope = -Real.sqrt 3 → α = 120 :=
by
  sorry

end NUMINAMATH_GPT_angle_of_inclination_l1669_166994


namespace NUMINAMATH_GPT_part1_part2_l1669_166906

def A : Set ℝ := {x | x^2 + x - 12 < 0}
def B : Set ℝ := {x | 4 / (x + 3) ≤ 1}
def C (m : ℝ) : Set ℝ := {x | x^2 - 2 * m * x + m^2 - 1 ≤ 0}

theorem part1 : A ∩ B = {x | -4 < x ∧ x < -3 ∨ 1 ≤ x ∧ x < 3} := sorry

theorem part2 (m : ℝ) : (-3 < m ∧ m < 2) ↔ ∀ x, (x ∈ A → x ∈ C m) ∧ ∃ x, x ∈ C m ∧ x ∉ A := sorry

end NUMINAMATH_GPT_part1_part2_l1669_166906


namespace NUMINAMATH_GPT_quadratic_eq_k_value_l1669_166934

theorem quadratic_eq_k_value (k : ℤ) : (∀ x : ℝ, (k - 1) * x ^ (|k| + 1) - x + 5 = 0 → (k - 1) ≠ 0 ∧ |k| + 1 = 2) -> k = -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_k_value_l1669_166934


namespace NUMINAMATH_GPT_probability_cello_viola_same_tree_l1669_166919

noncomputable section

def cellos : ℕ := 800
def violas : ℕ := 600
def cello_viola_pairs_same_tree : ℕ := 100

theorem probability_cello_viola_same_tree : 
  (cello_viola_pairs_same_tree: ℝ) / ((cellos * violas : ℕ) : ℝ) = 1 / 4800 := 
by
  sorry

end NUMINAMATH_GPT_probability_cello_viola_same_tree_l1669_166919


namespace NUMINAMATH_GPT_percent_deficit_in_width_l1669_166975

theorem percent_deficit_in_width (L W : ℝ) (h : 1.08 * (1 - (d : ℝ) / W) = 1.0044) : d = 0.07 * W :=
by sorry

end NUMINAMATH_GPT_percent_deficit_in_width_l1669_166975


namespace NUMINAMATH_GPT_exactly_one_true_l1669_166939

-- Given conditions
def p (x : ℝ) : Prop := (x^2 - 3 * x + 2 ≠ 0) → (x ≠ 2)

-- Define the contrapositive of p
def contrapositive_p (x : ℝ) : Prop := (x = 2) → (x^2 - 3 * x + 2 = 0)

-- Define the converse of p
def converse_p (x : ℝ) : Prop := (x ≠ 2) → (x^2 - 3 * x + 2 ≠ 0)

-- Define the inverse of p
def inverse_p (x : ℝ) : Prop := (x = 2 → x^2 - 3 * x + 2 = 0)

-- Formalize the problem: Prove that exactly one of the converse, inverse, and contrapositive of p is true.
theorem exactly_one_true :
  (∀ x : ℝ, p x) →
  ((∃ x : ℝ, contrapositive_p x) ∧ ¬(∀ x : ℝ, converse_p x) ∧ ¬(∀ x : ℝ, inverse_p x) ∨
   ¬(∃ x : ℝ, contrapositive_p x) ∧ (∀ x : ℝ, converse_p x) ∧ ¬(∀ x : ℝ, inverse_p x) ∨
   ¬(∃ x : ℝ, contrapositive_p x) ∧ ¬(∀ x : ℝ, converse_p x) ∧ (∀ x : ℝ, inverse_p x)) :=
sorry

end NUMINAMATH_GPT_exactly_one_true_l1669_166939


namespace NUMINAMATH_GPT_items_count_l1669_166984

variable (N : ℕ)

-- Conditions
def item_price : ℕ := 50
def discount_rate : ℕ := 80
def sell_percentage : ℕ := 90
def creditors_owed : ℕ := 15000
def money_left : ℕ := 3000

-- Definitions based on the conditions
def sale_price : ℕ := (item_price * (100 - discount_rate)) / 100
def money_before_paying_creditors : ℕ := money_left + creditors_owed
def total_revenue (N : ℕ) : ℕ := (sell_percentage * N * sale_price) / 100

-- Problem statement
theorem items_count : total_revenue N = money_before_paying_creditors → N = 2000 := by
  intros h
  sorry

end NUMINAMATH_GPT_items_count_l1669_166984


namespace NUMINAMATH_GPT_multiply_fractions_l1669_166945

theorem multiply_fractions :
  (1 / 3) * (4 / 7) * (9 / 13) * (2 / 5) = 72 / 1365 :=
by sorry

end NUMINAMATH_GPT_multiply_fractions_l1669_166945


namespace NUMINAMATH_GPT_intersection_complement_eq_l1669_166909

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | x^2 - 5 * x + 4 < 0}

theorem intersection_complement_eq :
  A ∩ {x | x ≤ 1 ∨ x ≥ 4} = {0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l1669_166909


namespace NUMINAMATH_GPT_pentagon_right_angles_l1669_166917

theorem pentagon_right_angles (angles : Finset ℕ) :
  angles = {0, 1, 2, 3} ↔ ∀ (k : ℕ), k ∈ angles ↔ ∃ (a b c d e : ℕ), 
  a + b + c + d + e = 540 ∧ (a = 90 ∨ b = 90 ∨ c = 90 ∨ d = 90 ∨ e = 90) 
  ∧ Finset.card (Finset.filter (λ x => x = 90) {a, b, c, d, e}) = k := 
sorry

end NUMINAMATH_GPT_pentagon_right_angles_l1669_166917


namespace NUMINAMATH_GPT_ratio_M_N_l1669_166986

theorem ratio_M_N (M Q P R N : ℝ) 
(h1 : M = 0.40 * Q) 
(h2 : Q = 0.25 * P) 
(h3 : R = 0.60 * P) 
(h4 : N = 0.75 * R) : 
  M / N = 2 / 9 := 
by
  sorry

end NUMINAMATH_GPT_ratio_M_N_l1669_166986


namespace NUMINAMATH_GPT_evaluate_expression_l1669_166931

theorem evaluate_expression (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxy : x > y) (hyz : y > z) :
  (x ^ (y + z) * z ^ (x + y)) / (y ^ (x + z) * z ^ (y + x)) = (x / y) ^ (y + z) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1669_166931


namespace NUMINAMATH_GPT_necessary_condition_for_inequality_l1669_166967

theorem necessary_condition_for_inequality (a b : ℝ) (h : a * b > 0) : 
  (a ≠ b) → (a ≠ 0) → (b ≠ 0) → ((b / a) + (a / b) > 2) :=
by
  sorry

end NUMINAMATH_GPT_necessary_condition_for_inequality_l1669_166967


namespace NUMINAMATH_GPT_sector_angle_l1669_166956

theorem sector_angle (r θ : ℝ) 
  (h1 : r * θ + 2 * r = 6) 
  (h2 : 1/2 * r^2 * θ = 2) : 
  θ = 1 ∨ θ = 4 :=
by 
  sorry

end NUMINAMATH_GPT_sector_angle_l1669_166956


namespace NUMINAMATH_GPT_initial_quantity_of_milk_l1669_166923

theorem initial_quantity_of_milk (A B C : ℝ) 
    (h1 : B = 0.375 * A)
    (h2 : C = 0.625 * A)
    (h3 : B + 148 = C - 148) : A = 1184 :=
by
  sorry

end NUMINAMATH_GPT_initial_quantity_of_milk_l1669_166923


namespace NUMINAMATH_GPT_intersection_P_Q_l1669_166971

def setP : Set ℝ := {1, 2, 3, 4}
def setQ : Set ℝ := {x | abs x ≤ 2}

theorem intersection_P_Q : (setP ∩ setQ) = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l1669_166971


namespace NUMINAMATH_GPT_solve_for_x_l1669_166960

theorem solve_for_x (x : ℚ) (h : 3 / x - 3 / x / (9 / x) = 0.5) : x = 6 / 5 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1669_166960


namespace NUMINAMATH_GPT_relationship_xy_l1669_166982

variable (x y : ℝ)

theorem relationship_xy (h₁ : x - y > x + 2) (h₂ : x + y + 3 < y - 1) : x < -4 ∧ y < -2 := 
by sorry

end NUMINAMATH_GPT_relationship_xy_l1669_166982


namespace NUMINAMATH_GPT_triangle_with_ratio_is_right_triangle_l1669_166983

/-- If the ratio of the interior angles of a triangle is 1:2:3, then the triangle is a right triangle. -/
theorem triangle_with_ratio_is_right_triangle (x : ℝ) (h : x + 2*x + 3*x = 180) : 
  3*x = 90 :=
sorry

end NUMINAMATH_GPT_triangle_with_ratio_is_right_triangle_l1669_166983
