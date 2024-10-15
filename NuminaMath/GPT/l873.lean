import Mathlib

namespace NUMINAMATH_GPT_range_of_b_l873_87340

theorem range_of_b (b : ℝ) :
  (∃ x : ℝ, x^2 - 2 * b * x + b^2 + b - 5 = 0) ∧
  (∀ x < 3.5, ∃ δ > 0, ∀ ε, x < ε → ε^2 - 2 * b * ε + b^2 + b - 5 < x^2 - 2 * b * x + b^2 + b - 5) →
  (3.5 ≤ b ∧ b ≤ 5) :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l873_87340


namespace NUMINAMATH_GPT_pure_imaginary_complex_l873_87392

theorem pure_imaginary_complex (m : ℝ) (i : ℂ) (h : i^2 = -1) :
    (∃ (y : ℂ), (2 - m * i) / (1 + i) = y * i) ↔ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_complex_l873_87392


namespace NUMINAMATH_GPT_division_remainder_unique_u_l873_87329

theorem division_remainder_unique_u :
  ∃! u : ℕ, ∃ q : ℕ, 15 = u * q + 4 ∧ u > 4 :=
sorry

end NUMINAMATH_GPT_division_remainder_unique_u_l873_87329


namespace NUMINAMATH_GPT_solve_floor_equation_l873_87374

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem solve_floor_equation :
  (∃ x : ℝ, (floor ((x - 1) / 2))^2 + 2 * x + 2 = 0) → x = -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_floor_equation_l873_87374


namespace NUMINAMATH_GPT_sum_of_sins_is_zero_l873_87357

variable {x y z : ℝ}

theorem sum_of_sins_is_zero
  (h1 : Real.sin x = Real.tan y)
  (h2 : Real.sin y = Real.tan z)
  (h3 : Real.sin z = Real.tan x) :
  Real.sin x + Real.sin y + Real.sin z = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_sins_is_zero_l873_87357


namespace NUMINAMATH_GPT_people_visited_both_l873_87301

theorem people_visited_both (total iceland norway neither both : ℕ) (h_total: total = 100) (h_iceland: iceland = 55) (h_norway: norway = 43) (h_neither: neither = 63)
  (h_both_def: both = iceland + norway - (total - neither)) :
  both = 61 :=
by 
  rw [h_total, h_iceland, h_norway, h_neither] at h_both_def
  simp at h_both_def
  exact h_both_def

end NUMINAMATH_GPT_people_visited_both_l873_87301


namespace NUMINAMATH_GPT_cost_price_of_article_l873_87352

theorem cost_price_of_article
  (C SP1 SP2 : ℝ)
  (h1 : SP1 = 0.8 * C)
  (h2 : SP2 = 1.05 * C)
  (h3 : SP2 = SP1 + 100) : 
  C = 400 := 
sorry

end NUMINAMATH_GPT_cost_price_of_article_l873_87352


namespace NUMINAMATH_GPT_integer_solutions_l873_87363

theorem integer_solutions (x : ℤ) : 
  (⌊(x : ℚ) / 2⌋ * ⌊(x : ℚ) / 3⌋ * ⌊(x : ℚ) / 4⌋ = x^2) ↔ (x = 0 ∨ x = 24) := 
sorry

end NUMINAMATH_GPT_integer_solutions_l873_87363


namespace NUMINAMATH_GPT_completing_square_to_simplify_eq_l873_87353

theorem completing_square_to_simplify_eq : 
  ∃ (c : ℝ), (∀ x : ℝ, x^2 - 6 * x + 4 = 0 ↔ (x - 3)^2 = c) :=
by
  use 5
  intro x
  constructor
  { intro h
    -- proof conversion process (skipped)
    sorry }
  { intro h
    -- reverse proof process (skipped)
    sorry }

end NUMINAMATH_GPT_completing_square_to_simplify_eq_l873_87353


namespace NUMINAMATH_GPT_david_and_maria_ages_l873_87355

theorem david_and_maria_ages 
  (D Y M : ℕ)
  (h1 : Y = D + 7)
  (h2 : Y = 2 * D)
  (h3 : M = D + 4)
  (h4 : M = Y / 2)
  : D = 7 ∧ M = 11 := by
  sorry

end NUMINAMATH_GPT_david_and_maria_ages_l873_87355


namespace NUMINAMATH_GPT_problem_a_problem_b_l873_87335

-- Define given points and lines
variables (A B P Q R L T K S : Type) 
variables (l : A) -- line through A
variables (a : A) -- line through A perpendicular to l
variables (b : B) -- line through B perpendicular to l
variables (PQ_intersects_a : Q) (PR_intersects_b : R)
variables (line_through_A_perp_BQ : L) (line_through_B_perp_AR : K)
variables (intersects_BQ_at_L : L) (intersects_BR_at_T : T)
variables (intersects_AR_at_K : K) (intersects_AQ_at_S : S)

-- Define collinearity properties
def collinear (X Y Z : Type) : Prop := sorry

-- Formalize the mathematical proofs as Lean theorems
theorem problem_a : collinear P T S :=
sorry

theorem problem_b : collinear P K L :=
sorry

end NUMINAMATH_GPT_problem_a_problem_b_l873_87335


namespace NUMINAMATH_GPT_greatest_divisor_of_620_and_180_l873_87387

/-- This theorem asserts that the greatest divisor of 620 that 
    is smaller than 100 and also a factor of 180 is 20. -/
theorem greatest_divisor_of_620_and_180 (d : ℕ) (h1 : d ∣ 620) (h2 : d ∣ 180) (h3 : d < 100) : d ≤ 20 :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_of_620_and_180_l873_87387


namespace NUMINAMATH_GPT_paityn_red_hats_l873_87356

theorem paityn_red_hats (R : ℕ) : 
  (R + 24 + (4 / 5) * ↑R + 48 = 108) → R = 20 :=
by
  intro h
  sorry


end NUMINAMATH_GPT_paityn_red_hats_l873_87356


namespace NUMINAMATH_GPT_grocery_store_price_l873_87313

-- Definitions based on the conditions
def bulk_price_per_case : ℝ := 12.00
def bulk_cans_per_case : ℝ := 48.0
def grocery_cans_per_pack : ℝ := 12.0
def additional_cost_per_can : ℝ := 0.25

-- The proof statement
theorem grocery_store_price : 
  (bulk_price_per_case / bulk_cans_per_case + additional_cost_per_can) * grocery_cans_per_pack = 6.00 :=
by
  sorry

end NUMINAMATH_GPT_grocery_store_price_l873_87313


namespace NUMINAMATH_GPT_root_of_quadratic_eq_l873_87346

open Complex

theorem root_of_quadratic_eq :
  ∃ z1 z2 : ℂ, (z1 = 3.5 - I) ∧ (z2 = -2.5 + I) ∧ (∀ z : ℂ, z^2 - z = 6 - 6 * I → (z = z1 ∨ z = z2)) := 
sorry

end NUMINAMATH_GPT_root_of_quadratic_eq_l873_87346


namespace NUMINAMATH_GPT_arithmetic_seq_ratio_l873_87324

theorem arithmetic_seq_ratio
  (a b : ℕ → ℝ)
  (S T : ℕ → ℝ)
  (H_seq_a : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (H_seq_b : ∀ n, T n = (n * (b 1 + b n)) / 2)
  (H_ratio : ∀ n, S n / T n = (2 * n - 3) / (4 * n - 3)) :
  (a 3 + a 15) / (2 * (b 3 + b 9)) + a 3 / (b 2 + b 10) = 19 / 41 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_ratio_l873_87324


namespace NUMINAMATH_GPT_ratio_of_m1_m2_l873_87362

open Real

theorem ratio_of_m1_m2 :
  ∀ (m : ℝ) (p q : ℝ), p ≠ 0 ∧ q ≠ 0 ∧ m ≠ 0 ∧
    (p + q = -((3 - 2 * m) / m)) ∧ 
    (p * q = 4 / m) ∧ 
    (p / q + q / p = 2) → 
   ∃ (m1 m2 : ℝ), 
    (4 * m1^2 - 28 * m1 + 9 = 0) ∧
    (4 * m2^2 - 28 * m2 + 9 = 0) ∧ 
    (m1 ≠ m2) ∧ 
    (m1 + m2 = 7) ∧ 
    (m1 * m2 = 9 / 4) ∧ 
    (m1 / m2 + m2 / m1 = 178 / 9) :=
by sorry

end NUMINAMATH_GPT_ratio_of_m1_m2_l873_87362


namespace NUMINAMATH_GPT_remainder_zero_l873_87319

theorem remainder_zero (x : ℕ) (h1 : x = 1680) :
  (x % 5 = 0) ∧ (x % 6 = 0) ∧ (x % 7 = 0) ∧ (x % 8 = 0) :=
by
  sorry

end NUMINAMATH_GPT_remainder_zero_l873_87319


namespace NUMINAMATH_GPT_art_group_students_count_l873_87348

theorem art_group_students_count (x : ℕ) (h1 : x * (1 / 60) + 2 * (x + 15) * (1 / 60) = 1) : x = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_art_group_students_count_l873_87348


namespace NUMINAMATH_GPT_condition_suff_not_necess_l873_87312

theorem condition_suff_not_necess (x : ℝ) (h : |x - (1 / 2)| < 1 / 2) : x^3 < 1 :=
by
  have h1 : 0 < x := sorry
  have h2 : x < 1 := sorry
  sorry

end NUMINAMATH_GPT_condition_suff_not_necess_l873_87312


namespace NUMINAMATH_GPT_frequency_calculation_l873_87345

-- Define the given conditions
def sample_capacity : ℕ := 20
def group_frequency : ℚ := 0.25

-- The main theorem statement
theorem frequency_calculation :
  sample_capacity * group_frequency = 5 :=
by sorry

end NUMINAMATH_GPT_frequency_calculation_l873_87345


namespace NUMINAMATH_GPT_isosceles_trapezoid_with_inscribed_circle_area_is_20_l873_87306

def isosceles_trapezoid_area (a b c1 c2 h : ℕ) : ℕ :=
  (a + b) * h / 2

theorem isosceles_trapezoid_with_inscribed_circle_area_is_20
  (a b c h : ℕ)
  (ha : a = 2)
  (hb : b = 8)
  (hc : a + b = 2 * c)
  (hh : h ^ 2 = c ^ 2 - ((b - a) / 2) ^ 2) :
  isosceles_trapezoid_area a b c c h = 20 := 
by {
  sorry
}

end NUMINAMATH_GPT_isosceles_trapezoid_with_inscribed_circle_area_is_20_l873_87306


namespace NUMINAMATH_GPT_paul_has_five_dogs_l873_87315

theorem paul_has_five_dogs
  (w1 w2 w3 w4 w5 : ℕ)
  (food_per_10_pounds : ℕ)
  (total_food_required : ℕ)
  (h1 : w1 = 20)
  (h2 : w2 = 40)
  (h3 : w3 = 10)
  (h4 : w4 = 30)
  (h5 : w5 = 50)
  (h6 : food_per_10_pounds = 1)
  (h7 : total_food_required = 15) :
  (w1 / 10 * food_per_10_pounds) +
  (w2 / 10 * food_per_10_pounds) +
  (w3 / 10 * food_per_10_pounds) +
  (w4 / 10 * food_per_10_pounds) +
  (w5 / 10 * food_per_10_pounds) = total_food_required → 
  5 = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_paul_has_five_dogs_l873_87315


namespace NUMINAMATH_GPT_horner_eval_at_neg2_l873_87303

noncomputable def f (x : ℝ) : ℝ := x^5 - 3 * x^3 - 6 * x^2 + x - 1

theorem horner_eval_at_neg2 : f (-2) = -35 :=
by
  sorry

end NUMINAMATH_GPT_horner_eval_at_neg2_l873_87303


namespace NUMINAMATH_GPT_subset_implication_l873_87380

noncomputable def M (x : ℝ) : Prop := -2 * x + 1 ≥ 0
noncomputable def N (a x : ℝ) : Prop := x < a

theorem subset_implication (a : ℝ) :
  (∀ x, M x → N a x) → a > 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_subset_implication_l873_87380


namespace NUMINAMATH_GPT_solve_a_solve_inequality_solution_set_l873_87371

theorem solve_a (a : ℝ) :
  (∀ x : ℝ, (1 / 2 < x ∧ x < 2) ↔ ax^2 + 5 * x - 2 > 0) →
  a = -2 :=
by
  sorry

theorem solve_inequality_solution_set (x : ℝ) :
  (a = -2) →
  (2 * x^2 + 5 * x - 3 < 0) ↔
  (-3 < x ∧ x < 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_a_solve_inequality_solution_set_l873_87371


namespace NUMINAMATH_GPT_exists_x_abs_ge_one_fourth_l873_87322

theorem exists_x_abs_ge_one_fourth :
  ∀ (a b c : ℝ), ∃ x : ℝ, |x| ≤ 1 ∧ |x^3 + a * x^2 + b * x + c| ≥ 1 / 4 :=
by sorry

end NUMINAMATH_GPT_exists_x_abs_ge_one_fourth_l873_87322


namespace NUMINAMATH_GPT_second_smallest_is_3_probability_l873_87368

noncomputable def probability_of_second_smallest_is_3 : ℚ := 
  let total_ways := Nat.choose 10 6
  let favorable_ways := 2 * Nat.choose 7 4
  favorable_ways / total_ways

theorem second_smallest_is_3_probability : probability_of_second_smallest_is_3 = 1 / 3 := sorry

end NUMINAMATH_GPT_second_smallest_is_3_probability_l873_87368


namespace NUMINAMATH_GPT_eval_power_expr_of_196_l873_87318

theorem eval_power_expr_of_196 (a b : ℕ) (ha : 2^a ∣ 196 ∧ ¬ 2^(a + 1) ∣ 196) (hb : 7^b ∣ 196 ∧ ¬ 7^(b + 1) ∣ 196) :
  (1 / 7 : ℝ)^(b - a) = 1 := by
  have ha_val : a = 2 := sorry
  have hb_val : b = 2 := sorry
  rw [ha_val, hb_val]
  simp

end NUMINAMATH_GPT_eval_power_expr_of_196_l873_87318


namespace NUMINAMATH_GPT_find_integer_solutions_l873_87389

theorem find_integer_solutions (k : ℕ) (hk : k > 1) : 
  ∃ x y : ℤ, y^k = x^2 + x ↔ (k = 2 ∧ (x = 0 ∨ x = -1)) ∨ (k > 2 ∧ y^k ≠ x^2 + x) :=
by
  sorry

end NUMINAMATH_GPT_find_integer_solutions_l873_87389


namespace NUMINAMATH_GPT_subtract_square_l873_87382

theorem subtract_square (n : ℝ) (h : n = 68.70953354520753) : (n^2 - 20^2) = 4321.000000000001 := by
  sorry

end NUMINAMATH_GPT_subtract_square_l873_87382


namespace NUMINAMATH_GPT_train_cross_bridge_time_l873_87365

open Nat

-- Defining conditions as per the problem
def train_length : ℕ := 200
def bridge_length : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := speed_kmph * 5 / 18
def total_distance : ℕ := train_length + bridge_length
def time_to_cross : ℕ := total_distance / speed_mps

-- Stating the theorem
theorem train_cross_bridge_time : time_to_cross = 35 := by
  sorry

end NUMINAMATH_GPT_train_cross_bridge_time_l873_87365


namespace NUMINAMATH_GPT_zeros_in_decimal_representation_l873_87369

def term_decimal_zeros (x : ℚ) : ℕ := sorry  -- Function to calculate the number of zeros in the terminating decimal representation.

theorem zeros_in_decimal_representation :
  term_decimal_zeros (1 / (2^7 * 5^9)) = 8 :=
sorry

end NUMINAMATH_GPT_zeros_in_decimal_representation_l873_87369


namespace NUMINAMATH_GPT_order_of_numbers_l873_87323

variable (a b c : ℝ)
variable (h₁ : a = (1 / 2) ^ (1 / 3))
variable (h₂ : b = (1 / 2) ^ (2 / 3))
variable (h₃ : c = (1 / 5) ^ (2 / 3))

theorem order_of_numbers (a b c : ℝ) (h₁ : a = (1 / 2) ^ (1 / 3)) (h₂ : b = (1 / 2) ^ (2 / 3)) (h₃ : c = (1 / 5) ^ (2 / 3)) :
  c < b ∧ b < a := 
by
  sorry

end NUMINAMATH_GPT_order_of_numbers_l873_87323


namespace NUMINAMATH_GPT_nat_games_volunteer_allocation_l873_87361

theorem nat_games_volunteer_allocation 
  (volunteers : Fin 6 → Type) 
  (venues : Fin 3 → Type)
  (A B : volunteers 0)
  (remaining : Fin 4 → Type) 
  (assigned_pairings : Π (v : Fin 3), Fin 2 → volunteers 0) :
  (∀ v, assigned_pairings v 0 = A ∨ assigned_pairings v 1 = B) →
  (3 * 6 = 18) := 
by
  sorry

end NUMINAMATH_GPT_nat_games_volunteer_allocation_l873_87361


namespace NUMINAMATH_GPT_range_of_a_analytical_expression_l873_87360

variables {f : ℝ → ℝ}

-- Problem 1
theorem range_of_a (h_odd : ∀ x, f (-x) = -f x)
  (h_mono : ∀ x y, x < y → f x ≥ f y)
  {a : ℝ} (h_ineq : f (1 - a) + f (1 - 2 * a) < 0) :
  0 < a ∧ a ≤ 2 / 3 :=
sorry

-- Problem 2
theorem analytical_expression 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, 0 < x ∧ x < 1 → f x = x^2 + x + 1)
  (h_zero : f 0 = 0) :
  ∀ x : ℝ, -1 < x ∧ x < 1 → f x = 
    if x > 0 then x^2 + x + 1
    else if x = 0 then 0
    else -x^2 + x - 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_analytical_expression_l873_87360


namespace NUMINAMATH_GPT_soccer_team_wins_l873_87367

theorem soccer_team_wins :
  ∃ W D : ℕ, 
    (W + 2 + D = 20) ∧  -- total games
    (3 * W + D = 46) ∧  -- total points
    (W = 14) :=         -- correct answer
by
  sorry

end NUMINAMATH_GPT_soccer_team_wins_l873_87367


namespace NUMINAMATH_GPT_sum_equals_1584_l873_87398

-- Let's define the function that computes the sum, according to the pattern
def sumPattern : ℕ → ℝ
  | 0 => 0
  | k + 1 => if (k + 1) % 3 = 0 then - (k + 1) + sumPattern k
             else (k + 1) + sumPattern k

-- This function defines the problem setting and the final expected result
theorem sum_equals_1584 : sumPattern 99 = 1584 := by
  sorry

end NUMINAMATH_GPT_sum_equals_1584_l873_87398


namespace NUMINAMATH_GPT_number_of_5card_hands_with_4_of_a_kind_l873_87336

-- Definitions based on the given conditions
def deck_size : Nat := 52
def num_values : Nat := 13
def suits_per_value : Nat := 4

-- The function to count the number of 5-card hands with exactly four cards of the same value
def count_hands_with_four_of_a_kind : Nat :=
  num_values * (deck_size - suits_per_value)

-- Proof statement
theorem number_of_5card_hands_with_4_of_a_kind : count_hands_with_four_of_a_kind = 624 :=
by
  -- Steps to show the computation results may be added here
  -- We use the formula: 13 * (52 - 4)
  sorry

end NUMINAMATH_GPT_number_of_5card_hands_with_4_of_a_kind_l873_87336


namespace NUMINAMATH_GPT_total_time_correct_l873_87316

variable (b n : ℕ)

def total_travel_time (b n : ℕ) : ℚ := (3*b + 4*n + 2*b) / 150

theorem total_time_correct :
  total_travel_time b n = (5 * b + 4 * n) / 150 :=
by sorry

end NUMINAMATH_GPT_total_time_correct_l873_87316


namespace NUMINAMATH_GPT_tan_ineq_solution_l873_87376

theorem tan_ineq_solution (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : ∀ x, x = a * Real.pi → ¬ (Real.tan x = a * Real.pi)) :
    {x : ℝ | ∃ k : ℤ, k * Real.pi + Real.pi / 4 ≤ x ∧ x < k * Real.pi + Real.pi / 2}
    = {x : ℝ | ∃ k : ℤ, k * Real.pi + Real.pi / 4 ≤ x ∧ x < k * Real.pi + Real.pi / 2} := sorry

end NUMINAMATH_GPT_tan_ineq_solution_l873_87376


namespace NUMINAMATH_GPT_sum_of_coefficients_l873_87349

def u (n : ℕ) : ℕ := 
  match n with
  | 0 => 6 -- Assume the sequence starts at u_0 for easier indexing
  | n + 1 => u n + 5 + 2 * n

theorem sum_of_coefficients (u : ℕ → ℕ) : 
  (∀ n, u (n + 1) = u n + 5 + 2 * n) ∧ u 1 = 6 → 
  (∃ a b c : ℕ, (∀ n, u n = a * n^2 + b * n + c) ∧ a + b + c = 6) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l873_87349


namespace NUMINAMATH_GPT_sum_of_cube_edges_l873_87379

theorem sum_of_cube_edges (edge_len : ℝ) (num_edges : ℕ) (lengths : ℝ) (h1 : edge_len = 15) (h2 : num_edges = 12) : lengths = num_edges * edge_len :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cube_edges_l873_87379


namespace NUMINAMATH_GPT_part_a_part_b_l873_87331

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (λ x y => x + y) 0

-- Part A: There exists a sequence of 158 consecutive integers where the sum of digits is not divisible by 17
theorem part_a : ∃ (n : ℕ), ∀ (k : ℕ), k < 158 → sum_of_digits (n + k) % 17 ≠ 0 := by
  sorry

-- Part B: Among any 159 consecutive integers, there exists at least one integer whose sum of digits is divisible by 17
theorem part_b : ∀ (n : ℕ), ∃ (k : ℕ), k < 159 ∧ sum_of_digits (n + k) % 17 = 0 := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l873_87331


namespace NUMINAMATH_GPT_value_of_M_l873_87344

-- Define M as given in the conditions
def M : ℤ :=
  (150^2 + 2) + (149^2 - 2) - (148^2 + 2) - (147^2 - 2) + (146^2 + 2) +
  (145^2 - 2) - (144^2 + 2) - (143^2 - 2) + (142^2 + 2) + (141^2 - 2) -
  (140^2 + 2) - (139^2 - 2) + (138^2 + 2) + (137^2 - 2) - (136^2 + 2) -
  (135^2 - 2) + (134^2 + 2) + (133^2 - 2) - (132^2 + 2) - (131^2 - 2) +
  (130^2 + 2) + (129^2 - 2) - (128^2 + 2) - (127^2 - 2) + (126^2 + 2) +
  (125^2 - 2) - (124^2 + 2) - (123^2 - 2) + (122^2 + 2) + (121^2 - 2) -
  (120^2 + 2) - (119^2 - 2) + (118^2 + 2) + (117^2 - 2) - (116^2 + 2) -
  (115^2 - 2) + (114^2 + 2) + (113^2 - 2) - (112^2 + 2) - (111^2 - 2) +
  (110^2 + 2) + (109^2 - 2) - (108^2 + 2) - (107^2 - 2) + (106^2 + 2) +
  (105^2 - 2) - (104^2 + 2) - (103^2 - 2) + (102^2 + 2) + (101^2 - 2) -
  (100^2 + 2) - (99^2 - 2) + (98^2 + 2) + (97^2 - 2) - (96^2 + 2) -
  (95^2 - 2) + (94^2 + 2) + (93^2 - 2) - (92^2 + 2) - (91^2 - 2) +
  (90^2 + 2) + (89^2 - 2) - (88^2 + 2) - (87^2 - 2) + (86^2 + 2) +
  (85^2 - 2) - (84^2 + 2) - (83^2 - 2) + (82^2 + 2) + (81^2 - 2) -
  (80^2 + 2) - (79^2 - 2) + (78^2 + 2) + (77^2 - 2) - (76^2 + 2) -
  (75^2 - 2) + (74^2 + 2) + (73^2 - 2) - (72^2 + 2) - (71^2 - 2) +
  (70^2 + 2) + (69^2 - 2) - (68^2 + 2) - (67^2 - 2) + (66^2 + 2) +
  (65^2 - 2) - (64^2 + 2) - (63^2 - 2) + (62^2 + 2) + (61^2 - 2) -
  (60^2 + 2) - (59^2 - 2) + (58^2 + 2) + (57^2 - 2) - (56^2 + 2) -
  (55^2 - 2) + (54^2 + 2) + (53^2 - 2) - (52^2 + 2) - (51^2 - 2) +
  (50^2 + 2) + (49^2 - 2) - (48^2 + 2) - (47^2 - 2) + (46^2 + 2) +
  (45^2 - 2) - (44^2 + 2) - (43^2 - 2) + (42^2 + 2) + (41^2 - 2) -
  (40^2 + 2) - (39^2 - 2) + (38^2 + 2) + (37^2 - 2) - (36^2 + 2) -
  (35^2 - 2) + (34^2 + 2) + (33^2 - 2) - (32^2 + 2) - (31^2 - 2) +
  (30^2 + 2) + (29^2 - 2) - (28^2 + 2) - (27^2 - 2) + (26^2 + 2) +
  (25^2 - 2) - (24^2 + 2) - (23^2 - 2) + (22^2 + 2) + (21^2 - 2) -
  (20^2 + 2) - (19^2 - 2) + (18^2 + 2) + (17^2 - 2) - (16^2 + 2) -
  (15^2 - 2) + (14^2 + 2) + (13^2 - 2) - (12^2 + 2) - (11^2 - 2) +
  (10^2 + 2) + (9^2 - 2) - (8^2 + 2) - (7^2 - 2) + (6^2 + 2) +
  (5^2 - 2) - (4^2 + 2) - (3^2 - 2) + (2^2 + 2) + (1^2 - 2)

-- Statement to prove that the value of M is 22700
theorem value_of_M : M = 22700 :=
  by sorry

end NUMINAMATH_GPT_value_of_M_l873_87344


namespace NUMINAMATH_GPT_largest_digit_not_in_odd_units_digits_l873_87302

-- Defining the sets of digits
def odd_units_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_units_digits : Set ℕ := {0, 2, 4, 6, 8}

-- Statement to prove
theorem largest_digit_not_in_odd_units_digits : 
  ∀ n ∈ even_units_digits, n ≤ 8 ∧ (∀ d ∈ odd_units_digits, d < n) → n = 8 :=
by
  sorry

end NUMINAMATH_GPT_largest_digit_not_in_odd_units_digits_l873_87302


namespace NUMINAMATH_GPT_original_length_equals_13_l873_87375

-- Definitions based on conditions
def original_width := 18
def increased_length (x : ℕ) := x + 2
def increased_width := 20

-- Total area condition
def total_area (x : ℕ) := 
  4 * ((increased_length x) * increased_width) + 2 * ((increased_length x) * increased_width)

theorem original_length_equals_13 (x : ℕ) (h : total_area x = 1800) : x = 13 := 
by
  sorry

end NUMINAMATH_GPT_original_length_equals_13_l873_87375


namespace NUMINAMATH_GPT_badminton_costs_l873_87327

variables (x : ℕ) (h : x > 16)

-- Define costs at Store A and Store B
def cost_A : ℕ := 1760 + 40 * x
def cost_B : ℕ := 1920 + 32 * x

-- Lean statement to prove the costs
theorem badminton_costs : 
  cost_A x = 1760 + 40 * x ∧ cost_B x = 1920 + 32 * x :=
by {
  -- This proof is expected but not required for the task
  sorry
}

end NUMINAMATH_GPT_badminton_costs_l873_87327


namespace NUMINAMATH_GPT_barbata_interest_rate_l873_87338

theorem barbata_interest_rate (r : ℝ) : 
  let initial_investment := 2800
  let additional_investment := 1400
  let total_investment := initial_investment + additional_investment
  let annual_income := 0.06 * total_investment
  let additional_interest_rate := 0.08
  let income_from_initial := initial_investment * r
  let income_from_additional := additional_investment * additional_interest_rate
  income_from_initial + income_from_additional = annual_income → 
  r = 0.05 :=
by
  intros
  sorry

end NUMINAMATH_GPT_barbata_interest_rate_l873_87338


namespace NUMINAMATH_GPT_range_of_a_exists_x_l873_87339

theorem range_of_a_exists_x (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 3, 2 * x - x ^ 2 ≥ a) ↔ a ≤ 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_exists_x_l873_87339


namespace NUMINAMATH_GPT_triangular_25_l873_87394

-- Defining the formula for the n-th triangular number.
def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Stating that the 25th triangular number is 325.
theorem triangular_25 : triangular 25 = 325 :=
  by
    -- We don't prove it here, so we simply state it requires a proof.
    sorry

end NUMINAMATH_GPT_triangular_25_l873_87394


namespace NUMINAMATH_GPT_one_third_of_1206_is_100_5_percent_of_400_l873_87390

theorem one_third_of_1206_is_100_5_percent_of_400 : (1206 / 3) / 400 * 100 = 100.5 := by
  sorry

end NUMINAMATH_GPT_one_third_of_1206_is_100_5_percent_of_400_l873_87390


namespace NUMINAMATH_GPT_pond_water_amount_l873_87381

theorem pond_water_amount : 
  let initial_water := 500 
  let evaporation_rate := 4
  let rain_amount := 2
  let days := 40
  initial_water - days * (evaporation_rate - rain_amount) = 420 :=
by
  sorry

end NUMINAMATH_GPT_pond_water_amount_l873_87381


namespace NUMINAMATH_GPT_ellipse_parameters_sum_l873_87326

theorem ellipse_parameters_sum 
  (h k a b : ℤ) 
  (h_def : h = 3) 
  (k_def : k = -5) 
  (a_def : a = 7) 
  (b_def : b = 2) : 
  h + k + a + b = 7 := 
by 
  -- definitions and sums will be handled by autogenerated proof
  sorry

end NUMINAMATH_GPT_ellipse_parameters_sum_l873_87326


namespace NUMINAMATH_GPT_number_of_marked_points_l873_87347

theorem number_of_marked_points
  (a1 a2 b1 b2 : ℕ)
  (hA : a1 * a2 = 50)
  (hB : b1 * b2 = 56)
  (h_sum : a1 + a2 = b1 + b2) :
  a1 + a2 + 1 = 16 :=
sorry

end NUMINAMATH_GPT_number_of_marked_points_l873_87347


namespace NUMINAMATH_GPT_slope_of_line_through_midpoints_l873_87304

theorem slope_of_line_through_midpoints (A B C D : ℝ × ℝ) (hA : A = (1, 1)) (hB : B = (3, 4)) (hC : C = (4, 1)) (hD : D = (7, 4)) :
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let N := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  (N.2 - M.2) / (N.1 - M.1) = 0 := by
  sorry

end NUMINAMATH_GPT_slope_of_line_through_midpoints_l873_87304


namespace NUMINAMATH_GPT_original_cookie_price_l873_87351

theorem original_cookie_price (C : ℝ) (h1 : 1.5 * 16 + (C / 2) * 8 = 32) : C = 2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_original_cookie_price_l873_87351


namespace NUMINAMATH_GPT_negation_of_proposition_l873_87395

theorem negation_of_proposition :
  ¬(∃ x₀ : ℝ, 0 < x₀ ∧ Real.log x₀ = x₀ - 1) ↔ ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l873_87395


namespace NUMINAMATH_GPT_total_estate_value_l873_87325

theorem total_estate_value :
  ∃ (E : ℝ), ∀ (x : ℝ),
    (5 * x + 4 * x = (2 / 3) * E) ∧
    (E = 13.5 * x) ∧
    (wife_share = 3 * 4 * x) ∧
    (gardener_share = 600) ∧
    (nephew_share = 1000) →
    E = 2880 := 
by 
  -- Declarations
  let E : ℝ := sorry
  let x : ℝ := sorry
  
  -- Set up conditions
  -- Daughter and son share
  have c1 : 5 * x + 4 * x = (2 / 3) * E := sorry
  
  -- E expressed through x
  have c2 : E = 13.5 * x := sorry
  
  -- Wife's share
  have c3 : wife_share = 3 * (4 * x) := sorry
  
  -- Gardener's share and Nephew's share
  have c4 : gardener_share = 600 := sorry
  have c5 : nephew_share = 1000 := sorry
  
  -- Equate expressions and solve
  have eq1 : E = 21 * x + 1600 := sorry
  have eq2 : E = 2880 := sorry
  use E
  intro x
  -- Prove the equalities under the given conditions
  sorry

end NUMINAMATH_GPT_total_estate_value_l873_87325


namespace NUMINAMATH_GPT_bakery_used_0_2_bags_of_wheat_flour_l873_87300

-- Define the conditions
def total_flour := 0.3
def white_flour := 0.1

-- Define the number of bags of wheat flour used
def wheat_flour := total_flour - white_flour

-- The proof statement
theorem bakery_used_0_2_bags_of_wheat_flour : wheat_flour = 0.2 := 
by
  sorry

end NUMINAMATH_GPT_bakery_used_0_2_bags_of_wheat_flour_l873_87300


namespace NUMINAMATH_GPT_fair_coin_three_flips_probability_l873_87364

theorem fair_coin_three_flips_probability :
  ∀ (prob : ℕ → ℚ) (independent : ∀ n, prob n = 1 / 2),
    prob 0 * prob 1 * prob 2 = 1 / 8 := 
by
  intros prob independent
  sorry

end NUMINAMATH_GPT_fair_coin_three_flips_probability_l873_87364


namespace NUMINAMATH_GPT_soda_cost_l873_87396

theorem soda_cost (b s : ℕ) 
  (h₁ : 3 * b + 2 * s = 450) 
  (h₂ : 2 * b + 3 * s = 480) : 
  s = 108 := 
by
  sorry

end NUMINAMATH_GPT_soda_cost_l873_87396


namespace NUMINAMATH_GPT_base_k_132_eq_30_l873_87399

theorem base_k_132_eq_30 (k : ℕ) (h : 1 * k^2 + 3 * k + 2 = 30) : k = 4 :=
sorry

end NUMINAMATH_GPT_base_k_132_eq_30_l873_87399


namespace NUMINAMATH_GPT_even_and_monotonically_increasing_f3_l873_87320

noncomputable def f1 (x : ℝ) : ℝ := x^3
noncomputable def f2 (x : ℝ) : ℝ := -x^2 + 1
noncomputable def f3 (x : ℝ) : ℝ := abs x + 1
noncomputable def f4 (x : ℝ) : ℝ := 2^(-abs x)

theorem even_and_monotonically_increasing_f3 :
  (∀ x, f3 x = f3 (-x)) ∧ (∀ x > 0, ∀ y > x, f3 y > f3 x) := 
sorry

end NUMINAMATH_GPT_even_and_monotonically_increasing_f3_l873_87320


namespace NUMINAMATH_GPT_perpendicular_bisector_of_circles_l873_87388

theorem perpendicular_bisector_of_circles
  (circle1 : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 0)
  (circle2 : ∀ x y : ℝ, x^2 + y^2 - 6 * x = 0) :
  ∃ x y : ℝ, (3 * x - y - 9 = 0) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_bisector_of_circles_l873_87388


namespace NUMINAMATH_GPT_polar_distance_l873_87317

noncomputable def distance_point (r1 θ1 r2 θ2 : ℝ) : ℝ :=
  Real.sqrt ((r1 ^ 2) + (r2 ^ 2) - 2 * r1 * r2 * Real.cos (θ1 - θ2))

theorem polar_distance :
  ∀ (θ1 θ2 : ℝ), (θ1 - θ2 = Real.pi / 2) → distance_point 5 θ1 12 θ2 = 13 :=
by
  intros θ1 θ2 hθ
  rw [distance_point, hθ, Real.cos_pi_div_two]
  norm_num
  sorry

end NUMINAMATH_GPT_polar_distance_l873_87317


namespace NUMINAMATH_GPT_andrea_sod_rectangles_l873_87343

def section_1_length : ℕ := 35
def section_1_width : ℕ := 42
def section_2_length : ℕ := 55
def section_2_width : ℕ := 86
def section_3_length : ℕ := 20
def section_3_width : ℕ := 50
def section_4_length : ℕ := 48
def section_4_width : ℕ := 66

def sod_length : ℕ := 3
def sod_width : ℕ := 4

def area (length width : ℕ) : ℕ := length * width
def sod_area : ℕ := area sod_length sod_width

def rectangles_needed (section_length section_width sod_area : ℕ) : ℕ :=
  (area section_length section_width + sod_area - 1) / sod_area

def total_rectangles_needed : ℕ :=
  rectangles_needed section_1_length section_1_width sod_area +
  rectangles_needed section_2_length section_2_width sod_area +
  rectangles_needed section_3_length section_3_width sod_area +
  rectangles_needed section_4_length section_4_width sod_area

theorem andrea_sod_rectangles : total_rectangles_needed = 866 := by
  sorry

end NUMINAMATH_GPT_andrea_sod_rectangles_l873_87343


namespace NUMINAMATH_GPT_simplify_expression_l873_87321

theorem simplify_expression (a : ℝ) :
  (1/2) * (8 * a^2 + 4 * a) - 3 * (a - (1/3) * a^2) = 5 * a^2 - a :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l873_87321


namespace NUMINAMATH_GPT_increasing_on_positive_reals_l873_87354

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem increasing_on_positive_reals : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y := by
  sorry

end NUMINAMATH_GPT_increasing_on_positive_reals_l873_87354


namespace NUMINAMATH_GPT_hyperbola_equation_l873_87378

theorem hyperbola_equation :
  ∃ (b : ℝ), (∀ (x y : ℝ), ((x = 2) ∧ (y = 2)) →
    ((x^2 / 5) - (y^2 / b^2) = 1)) ∧
    (∀ x y, (y = (2 / Real.sqrt 5) * x) ∨ (y = -(2 / Real.sqrt 5) * x) → 
    (∀ (a b : ℝ), (a = 2) → (b = 2) →
      (b^2 = 4) → ((5 * y^2 / 4) - x^2 = 1))) :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_l873_87378


namespace NUMINAMATH_GPT_cube_root_less_than_five_count_l873_87337

theorem cube_root_less_than_five_count :
  (∃ n : ℕ, n = 124 ∧ ∀ x : ℕ, 1 ≤ x → x < 5^3 → x < 125) := 
sorry

end NUMINAMATH_GPT_cube_root_less_than_five_count_l873_87337


namespace NUMINAMATH_GPT_sequence_general_term_l873_87370

open Nat

def sequence_a (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  a 2 = 3 ∧
  (∀ n : ℕ, 0 < n → a (n + 2) ≤ a n + 3 * 2^n) ∧
  (∀ n : ℕ, 0 < n → a (n + 1) ≥ 2 * a n + 1)

theorem sequence_general_term (a : ℕ → ℕ) (h : sequence_a a) :
  ∀ n : ℕ, 0 < n → a n = 2^n - 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l873_87370


namespace NUMINAMATH_GPT_find_a9_l873_87342

variable (a : ℕ → ℤ)

-- Condition 1: The sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ n, a (n + 1) = a n + d

-- Condition 2: Given a_4 = 5
def a4_value (a : ℕ → ℤ) : Prop :=
  a 4 = 5

-- Condition 3: Given a_5 = 4
def a5_value (a : ℕ → ℤ) : Prop :=
  a 5 = 4

-- Problem: Prove a_9 = 0
theorem find_a9 (h1 : arithmetic_sequence a) (h2 : a4_value a) (h3 : a5_value a) : a 9 = 0 := 
sorry

end NUMINAMATH_GPT_find_a9_l873_87342


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l873_87308

theorem arithmetic_geometric_sequence {a b c x y : ℝ} (h₁: a ≠ b) (h₂: b ≠ c) (h₃: a ≠ c)
  (h₄ : 2 * b = a + c) (h₅ : x^2 = a * b) (h₆ : y^2 = b * c) :
  (x^2 + y^2 = 2 * b^2) ∧ (x^2 * y^2 ≠ b^4) :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l873_87308


namespace NUMINAMATH_GPT_solve_fraction_equation_l873_87385

theorem solve_fraction_equation (x : ℝ) (hx1 : 0 < x) (hx2 : (x - 6) / 12 = 6 / (x - 12)) : x = 18 := 
sorry

end NUMINAMATH_GPT_solve_fraction_equation_l873_87385


namespace NUMINAMATH_GPT_find_point_B_l873_87384

structure Point where
  x : Int
  y : Int

def translation (p : Point) (dx dy : Int) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem find_point_B :
  let A := Point.mk (-2) 3
  let A' := Point.mk 3 2
  let B' := Point.mk 4 0
  let dx := 5
  let dy := -1
  (translation A dx dy = A') →
  ∃ B : Point, translation B dx dy = B' ∧ B = Point.mk (-1) (-1) :=
by
  intros
  use Point.mk (-1) (-1)
  constructor
  sorry
  rfl

end NUMINAMATH_GPT_find_point_B_l873_87384


namespace NUMINAMATH_GPT_lana_needs_to_sell_more_muffins_l873_87377

/--
Lana aims to sell 20 muffins at the bake sale.
She sells 12 muffins in the morning.
She sells another 4 in the afternoon.
How many more muffins does Lana need to sell to hit her goal?
-/
theorem lana_needs_to_sell_more_muffins (goal morningSales afternoonSales : ℕ)
  (h_goal : goal = 20) (h_morning : morningSales = 12) (h_afternoon : afternoonSales = 4) :
  goal - (morningSales + afternoonSales) = 4 :=
by
  sorry

end NUMINAMATH_GPT_lana_needs_to_sell_more_muffins_l873_87377


namespace NUMINAMATH_GPT_Shekar_science_marks_l873_87386

-- Define Shekar's known marks
def math_marks : ℕ := 76
def social_studies_marks : ℕ := 82
def english_marks : ℕ := 47
def biology_marks : ℕ := 85

-- Define the average mark and the number of subjects
def average_mark : ℕ := 71
def number_of_subjects : ℕ := 5

-- Define Shekar's unknown mark in Science
def science_marks : ℕ := sorry  -- We expect to prove science_marks = 65

-- State the theorem to be proved
theorem Shekar_science_marks :
  average_mark * number_of_subjects = math_marks + science_marks + social_studies_marks + english_marks + biology_marks →
  science_marks = 65 :=
by sorry

end NUMINAMATH_GPT_Shekar_science_marks_l873_87386


namespace NUMINAMATH_GPT_store_A_has_highest_capacity_l873_87359

noncomputable def total_capacity_A : ℕ := 5 * 6 * 9
noncomputable def total_capacity_B : ℕ := 8 * 4 * 7
noncomputable def total_capacity_C : ℕ := 10 * 3 * 8

theorem store_A_has_highest_capacity : total_capacity_A = 270 ∧ total_capacity_A > total_capacity_B ∧ total_capacity_A > total_capacity_C := 
by 
  -- Proof skipped with a placeholder
  sorry

end NUMINAMATH_GPT_store_A_has_highest_capacity_l873_87359


namespace NUMINAMATH_GPT_lorry_weight_l873_87397

theorem lorry_weight : 
  let empty_lorry_weight := 500
  let apples_weight := 10 * 55
  let oranges_weight := 5 * 45
  let watermelons_weight := 3 * 125
  let firewood_weight := 2 * 75
  let loaded_items_weight := apples_weight + oranges_weight + watermelons_weight + firewood_weight
  let total_weight := empty_lorry_weight + loaded_items_weight
  total_weight = 1800 :=
by 
  sorry

end NUMINAMATH_GPT_lorry_weight_l873_87397


namespace NUMINAMATH_GPT_smallest_M_l873_87350

def Q (M : ℕ) := (2 * M / 3 + 1) / (M + 1)

theorem smallest_M (M : ℕ) (h : M % 6 = 0) (h_pos : 0 < M) : 
  (∃ k, M = 6 * k ∧ Q M < 3 / 4) ↔ M = 6 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_M_l873_87350


namespace NUMINAMATH_GPT_DM_eq_r_plus_R_l873_87314

noncomputable def radius_incircle (A B D : ℝ) (s K : ℝ) : ℝ := K / s

noncomputable def radius_excircle (A C D : ℝ) (s' K' : ℝ) (AD : ℝ) : ℝ := K' / (s' - AD)

theorem DM_eq_r_plus_R 
  (A B C D M : ℝ)
  (h1 : A ≠ B)
  (h2 : B ≠ C)
  (h3 : A ≠ C)
  (h4 : D = (B + C) / 2)
  (h5 : M = (B + C) / 2)
  (r : ℝ)
  (h6 : r = radius_incircle A B D ((A + B + D) / 2) (abs ((A - B) * (A - D) / 2)))
  (R : ℝ)
  (h7 : R = radius_excircle A C D ((A + C + D) / 2) (abs ((A - C) * (A - D) / 2)) (abs (A - D))) :
  dist D M =r + R :=
by sorry

end NUMINAMATH_GPT_DM_eq_r_plus_R_l873_87314


namespace NUMINAMATH_GPT_dividend_expression_l873_87310

theorem dividend_expression 
  (D d q r P : ℕ)
  (hq_square : ∃ k, q = k^2)
  (hd_expr1 : d = 3 * r + 2)
  (hd_expr2 : d = 5 * q)
  (hr_val : r = 6)
  (hD_expr : D = d * q + r)
  (hP_prime : Prime P)
  (hP_div_D : P ∣ D)
  (hP_factor : P = 2 ∨ P = 43) :
  D = 86 := 
sorry

end NUMINAMATH_GPT_dividend_expression_l873_87310


namespace NUMINAMATH_GPT_part1_part2_l873_87366

-- Define the linear function
def linear_function (m x : ℝ) : ℝ := (m - 2) * x + 6

-- Prove part 1: If y increases as x increases, then m > 2
theorem part1 (m : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → linear_function m x1 < linear_function m x2) → m > 2 :=
sorry

-- Prove part 2: When -2 ≤ x ≤ 4, and y ≤ 10, the range of m is (2, 3] or [0, 2)
theorem part2 (m : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 4 → linear_function m x ≤ 10) →
  (2 < m ∧ m ≤ 3) ∨ (0 ≤ m ∧ m < 2) :=
sorry

end NUMINAMATH_GPT_part1_part2_l873_87366


namespace NUMINAMATH_GPT_art_piece_future_value_multiple_l873_87330

theorem art_piece_future_value_multiple (original_price increase_in_value future_value multiple : ℕ)
  (h1 : original_price = 4000)
  (h2 : increase_in_value = 8000)
  (h3 : future_value = original_price + increase_in_value)
  (h4 : multiple = future_value / original_price) :
  multiple = 3 := 
sorry

end NUMINAMATH_GPT_art_piece_future_value_multiple_l873_87330


namespace NUMINAMATH_GPT_p_distinct_roots_iff_l873_87309

variables {p : ℝ}

def quadratic_has_distinct_roots (a b c : ℝ) : Prop :=
  (b^2 - 4 * a * c) > 0

theorem p_distinct_roots_iff (hp: p > 0 ∨ p = -1) :
  (∀ x : ℝ, x^2 - 2 * |x| - p = 0 → 
    (quadratic_has_distinct_roots 1 (-2) (-p) ∨
      quadratic_has_distinct_roots 1 2 (-p))) :=
by sorry

end NUMINAMATH_GPT_p_distinct_roots_iff_l873_87309


namespace NUMINAMATH_GPT_initial_observations_l873_87341

theorem initial_observations {n : ℕ} (S : ℕ) (new_observation : ℕ) 
  (h1 : S = 15 * n) (h2 : new_observation = 14 - n)
  (h3 : (S + new_observation) / (n + 1) = 14) : n = 6 :=
sorry

end NUMINAMATH_GPT_initial_observations_l873_87341


namespace NUMINAMATH_GPT_sally_paid_peaches_l873_87393

def total_spent : ℝ := 23.86
def amount_spent_on_cherries : ℝ := 11.54
def amount_spent_on_peaches_after_coupon : ℝ := total_spent - amount_spent_on_cherries

theorem sally_paid_peaches : amount_spent_on_peaches_after_coupon = 12.32 :=
by 
  -- The actual proof will involve concrete calculation here.
  -- For now, we skip it with sorry.
  sorry

end NUMINAMATH_GPT_sally_paid_peaches_l873_87393


namespace NUMINAMATH_GPT_paint_cost_of_cube_l873_87307

theorem paint_cost_of_cube (side_length cost_per_kg coverage_per_kg : ℝ) (h₀ : side_length = 10) 
(h₁ : cost_per_kg = 60) (h₂ : coverage_per_kg = 20) : 
(cost_per_kg * (6 * (side_length^2) / coverage_per_kg) = 1800) :=
by
  sorry

end NUMINAMATH_GPT_paint_cost_of_cube_l873_87307


namespace NUMINAMATH_GPT_find_room_width_l873_87333

def room_height : ℕ := 12
def room_length : ℕ := 25
def door_height : ℕ := 6
def door_width : ℕ := 3
def window_height : ℕ := 4
def window_width : ℕ := 3
def number_of_windows : ℕ := 3
def cost_per_sqft : ℕ := 8
def total_cost : ℕ := 7248

theorem find_room_width (x : ℕ) (h : 8 * (room_height * (2 * room_length + 2 * x) - (door_height * door_width + window_height * window_width * number_of_windows)) = total_cost) : 
  x = 15 :=
sorry

end NUMINAMATH_GPT_find_room_width_l873_87333


namespace NUMINAMATH_GPT_geometric_sequence_sum_l873_87334

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Main statement to prove
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : is_geometric_sequence a q)
  (h2 : a 1 + a 2 = 40)
  (h3 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l873_87334


namespace NUMINAMATH_GPT_valid_triples_l873_87311

theorem valid_triples (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x ∣ (y + 1) ∧ y ∣ (z + 1) ∧ z ∣ (x + 1) ↔ (x, y, z) = (1, 1, 1) ∨ 
                                                      (x, y, z) = (1, 1, 2) ∨ 
                                                      (x, y, z) = (1, 3, 2) ∨ 
                                                      (x, y, z) = (3, 5, 4) :=
by
  sorry

end NUMINAMATH_GPT_valid_triples_l873_87311


namespace NUMINAMATH_GPT_find_cat_video_length_l873_87305

variables (C : ℕ)

def cat_video_length (C : ℕ) : Prop :=
  C + 2 * C + 6 * C = 36

theorem find_cat_video_length : cat_video_length 4 :=
by
  sorry

end NUMINAMATH_GPT_find_cat_video_length_l873_87305


namespace NUMINAMATH_GPT_sum_of_first_six_terms_l873_87358

theorem sum_of_first_six_terms (S : ℕ → ℝ) (a : ℕ → ℝ) (hS : ∀ n, S (n + 1) = S n + a (n + 1)) :
  S 2 = 2 → S 4 = 10 → S 6 = 24 := 
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sum_of_first_six_terms_l873_87358


namespace NUMINAMATH_GPT_spherical_coordinates_cone_l873_87372

open Real

-- Define spherical coordinates and the equation φ = c
def spherical_coordinates (ρ θ φ : ℝ) : Prop := 
  ∃ (c : ℝ), φ = c

-- Prove that φ = c describes a cone
theorem spherical_coordinates_cone (ρ θ : ℝ) (c : ℝ) :
  spherical_coordinates ρ θ c → ∃ ρ' θ', spherical_coordinates ρ' θ' c :=
by
  sorry

end NUMINAMATH_GPT_spherical_coordinates_cone_l873_87372


namespace NUMINAMATH_GPT_complement_union_l873_87383

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- State the theorem to prove the complement of the union of M and N in U is {5}
theorem complement_union {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l873_87383


namespace NUMINAMATH_GPT_jill_spent_more_l873_87391

def cost_per_ball_red : ℝ := 1.50
def cost_per_ball_yellow : ℝ := 1.25
def cost_per_ball_blue : ℝ := 1.00

def packs_red : ℕ := 5
def packs_yellow : ℕ := 4
def packs_blue : ℕ := 3

def balls_per_pack_red : ℕ := 18
def balls_per_pack_yellow : ℕ := 15
def balls_per_pack_blue : ℕ := 12

def balls_red : ℕ := packs_red * balls_per_pack_red
def balls_yellow : ℕ := packs_yellow * balls_per_pack_yellow
def balls_blue : ℕ := packs_blue * balls_per_pack_blue

def cost_red : ℝ := balls_red * cost_per_ball_red
def cost_yellow : ℝ := balls_yellow * cost_per_ball_yellow
def cost_blue : ℝ := balls_blue * cost_per_ball_blue

def combined_cost_yellow_blue : ℝ := cost_yellow + cost_blue

theorem jill_spent_more : cost_red = combined_cost_yellow_blue + 24 := by
  sorry

end NUMINAMATH_GPT_jill_spent_more_l873_87391


namespace NUMINAMATH_GPT_complete_the_square_l873_87373

theorem complete_the_square (x : ℝ) : x^2 - 8 * x + 1 = 0 → (x - 4)^2 = 15 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_complete_the_square_l873_87373


namespace NUMINAMATH_GPT_smallest_solution_proof_l873_87332

noncomputable def smallest_solution (x : ℝ) : Prop :=
  (1 / (x - 1) + 1 / (x - 5) = 4 / (x - 2)) ∧ 
  (∀ y : ℝ, 1 / (y - 1) + 1 / (y - 5) = 4 / (y - 2) → y ≥ x)

theorem smallest_solution_proof : smallest_solution ( (7 - Real.sqrt 33) / 2 ) :=
sorry

end NUMINAMATH_GPT_smallest_solution_proof_l873_87332


namespace NUMINAMATH_GPT_heather_aprons_l873_87328

variable {totalAprons : Nat} (apronsSewnBeforeToday apronsSewnToday apronsSewnTomorrow apronsSewnSoFar apronsRemaining : Nat)

theorem heather_aprons (h_total : totalAprons = 150)
                       (h_today : apronsSewnToday = 3 * apronsSewnBeforeToday)
                       (h_sewnSoFar : apronsSewnSoFar = apronsSewnBeforeToday + apronsSewnToday)
                       (h_tomorrow : apronsSewnTomorrow = 49)
                       (h_remaining : apronsRemaining = totalAprons - apronsSewnSoFar)
                       (h_halfRemaining : 2 * apronsSewnTomorrow = apronsRemaining) :
  apronsSewnBeforeToday = 13 :=
by
  sorry

end NUMINAMATH_GPT_heather_aprons_l873_87328
