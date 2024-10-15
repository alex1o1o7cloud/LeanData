import Mathlib

namespace NUMINAMATH_GPT_necessary_but_not_sufficient_not_sufficient_l174_17425

def P (x : ℝ) : Prop := x < 1
def Q (x : ℝ) : Prop := (x + 2) * (x - 1) < 0

theorem necessary_but_not_sufficient (x : ℝ) : P x → Q x := by
  intro hx
  sorry

theorem not_sufficient (x : ℝ) : ¬(Q x → P x) := by
  intro hq
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_not_sufficient_l174_17425


namespace NUMINAMATH_GPT_sum_and_count_evens_20_30_l174_17465

def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count_evens_20_30 :
  let x := sum_integers 20 30
  let y := count_even_integers 20 30
  x + y = 281 :=
by
  sorry

end NUMINAMATH_GPT_sum_and_count_evens_20_30_l174_17465


namespace NUMINAMATH_GPT_group_size_increase_by_4_l174_17449

theorem group_size_increase_by_4
    (N : ℕ)
    (weight_old : ℕ)
    (weight_new : ℕ)
    (average_increase : ℕ)
    (weight_increase_diff : ℕ)
    (h1 : weight_old = 55)
    (h2 : weight_new = 87)
    (h3 : average_increase = 4)
    (h4 : weight_increase_diff = weight_new - weight_old)
    (h5 : average_increase * N = weight_increase_diff) :
    N = 8 :=
by
  sorry

end NUMINAMATH_GPT_group_size_increase_by_4_l174_17449


namespace NUMINAMATH_GPT_pb_distance_l174_17473

theorem pb_distance (a b c d PA PD PC PB : ℝ)
  (hPA : PA = 5)
  (hPD : PD = 6)
  (hPC : PC = 7)
  (h1 : a^2 + b^2 = PA^2)
  (h2 : b^2 + c^2 = PC^2)
  (h3 : c^2 + d^2 = PD^2)
  (h4 : d^2 + a^2 = PB^2) :
  PB = Real.sqrt 38 := by
  sorry

end NUMINAMATH_GPT_pb_distance_l174_17473


namespace NUMINAMATH_GPT_max_ABC_ge_4_9_max_alpha_beta_gamma_ge_4_9_l174_17480

variable (p q : ℝ) (x y : ℝ)
variable (A B C α β γ : ℝ)

-- Conditions
axiom hp : 0 ≤ p ∧ p ≤ 1 
axiom hq : 0 ≤ q ∧ q ≤ 1 
axiom h1 : (p * x + (1 - p) * y)^2 = A * x^2 + B * x * y + C * y^2
axiom h2 : (p * x + (1 - p) * y) * (q * x + (1 - q) * y) = α * x^2 + β * x * y + γ * y^2

-- Problem
theorem max_ABC_ge_4_9 : max A (max B C) ≥ 4 / 9 := 
sorry

theorem max_alpha_beta_gamma_ge_4_9 : max α (max β γ) ≥ 4 / 9 := 
sorry

end NUMINAMATH_GPT_max_ABC_ge_4_9_max_alpha_beta_gamma_ge_4_9_l174_17480


namespace NUMINAMATH_GPT_triangle_DOE_area_l174_17414

theorem triangle_DOE_area
  (area_ABC : ℝ)
  (DO : ℝ) (OB : ℝ)
  (EO : ℝ) (OA : ℝ)
  (h_area_ABC : area_ABC = 1)
  (h_DO_OB : DO / OB = 1 / 3)
  (h_EO_OA : EO / OA = 4 / 5)
  : (1 / 4) * (4 / 9) * area_ABC = 11 / 135 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_DOE_area_l174_17414


namespace NUMINAMATH_GPT_water_removal_l174_17453

theorem water_removal (n : ℕ) : 
  (∀n, (2:ℚ) / (n + 2) = 1 / 8) ↔ (n = 14) := 
by 
  sorry

end NUMINAMATH_GPT_water_removal_l174_17453


namespace NUMINAMATH_GPT_units_digit_7_pow_103_l174_17422

theorem units_digit_7_pow_103 : Nat.mod (7 ^ 103) 10 = 3 := sorry

end NUMINAMATH_GPT_units_digit_7_pow_103_l174_17422


namespace NUMINAMATH_GPT_find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7_l174_17436

def isOdd (n : ℕ) : Prop := n % 2 = 1
def isInRange (n : ℕ) : Prop := 30 ≤ n ∧ n ≤ 50
def hasRemainderTwo (n : ℕ) : Prop := n % 7 = 2

theorem find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7 :
  ∃ n : ℕ, isInRange n ∧ isOdd n ∧ hasRemainderTwo n ∧ n = 37 :=
by
  sorry

end NUMINAMATH_GPT_find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7_l174_17436


namespace NUMINAMATH_GPT_boys_without_notebooks_l174_17482

/-
Given that:
1. There are 16 boys in Ms. Green's history class.
2. 20 students overall brought their notebooks to class.
3. 11 of the students who brought notebooks are girls.

Prove that the number of boys who did not bring their notebooks is 7.
-/

theorem boys_without_notebooks (total_boys : ℕ) (total_notebooks : ℕ) (girls_with_notebooks : ℕ)
  (hb : total_boys = 16) (hn : total_notebooks = 20) (hg : girls_with_notebooks = 11) : 
  (total_boys - (total_notebooks - girls_with_notebooks) = 7) :=
by
  sorry

end NUMINAMATH_GPT_boys_without_notebooks_l174_17482


namespace NUMINAMATH_GPT_smallest_c_for_inverse_l174_17411

def f (x : ℝ) : ℝ := (x - 3)^2 + 4

theorem smallest_c_for_inverse :
  ∃ c, (∀ x₁ x₂, (c ≤ x₁ ∧ c ≤ x₂ ∧ f x₁ = f x₂) → x₁ = x₂) ∧
       (∀ d, (∀ x₁ x₂, (d ≤ x₁ ∧ d ≤ x₂ ∧ f x₁ = f x₂) → x₁ = x₂) → c ≤ d) ∧
       c = 3 := sorry

end NUMINAMATH_GPT_smallest_c_for_inverse_l174_17411


namespace NUMINAMATH_GPT_total_inflation_over_two_years_real_interest_rate_over_two_years_l174_17403

section FinancialCalculations

-- Define the known conditions
def annual_inflation_rate : ℚ := 0.025
def nominal_interest_rate : ℚ := 0.06

-- Prove the total inflation rate over two years equals 5.0625%
theorem total_inflation_over_two_years :
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 5.0625 :=
sorry

-- Prove the real interest rate over two years equals 6.95%
theorem real_interest_rate_over_two_years :
  ((1 + nominal_interest_rate) * (1 + nominal_interest_rate) / (1 + (annual_inflation_rate * annual_inflation_rate)) - 1) * 100 = 6.95 :=
sorry

end FinancialCalculations

end NUMINAMATH_GPT_total_inflation_over_two_years_real_interest_rate_over_two_years_l174_17403


namespace NUMINAMATH_GPT_tangent_line_at_P_l174_17494

def tangent_line_eq (x y : ℝ) : ℝ := x - 2 * y + 1

theorem tangent_line_at_P (x y : ℝ) (h : x ^ 2 + y ^ 2 - 4 * x + 2 * y = 0 ∧ (x, y) = (1, 1)) :
    tangent_line_eq x y = 0 := 
sorry

end NUMINAMATH_GPT_tangent_line_at_P_l174_17494


namespace NUMINAMATH_GPT_larger_solution_of_quadratic_equation_l174_17490

open Nat

theorem larger_solution_of_quadratic_equation :
  ∃! x : ℝ, x * x - 13 * x + 36 = 0 ∧ ∀ y : ℝ, y * y - 13 * y + 36 = 0 → x ≥ y :=
by {
  sorry
}

end NUMINAMATH_GPT_larger_solution_of_quadratic_equation_l174_17490


namespace NUMINAMATH_GPT_inequality_proof_l174_17407

theorem inequality_proof (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) 
  (h : 1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1) :
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_proof_l174_17407


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l174_17477

theorem hyperbola_eccentricity : 
  (∃ (a b : ℝ), (a^2 = 1 ∧ b^2 = 2) ∧ ∀ e : ℝ, e = Real.sqrt (1 + b^2 / a^2) → e = Real.sqrt 3) :=
by 
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l174_17477


namespace NUMINAMATH_GPT_tree_growth_rate_l174_17458

-- Given conditions
def currentHeight : ℝ := 52
def futureHeightInches : ℝ := 1104
def oneFootInInches : ℝ := 12
def years : ℝ := 8

-- Prove the yearly growth rate in feet
theorem tree_growth_rate:
  (futureHeightInches / oneFootInInches - currentHeight) / years = 5 := 
by
  sorry

end NUMINAMATH_GPT_tree_growth_rate_l174_17458


namespace NUMINAMATH_GPT_score_comparison_l174_17495

theorem score_comparison :
  let sammy_score := 20
  let gab_score := 2 * sammy_score
  let cher_score := 2 * gab_score
  let alex_score := cher_score + cher_score / 10
  let combined_score := sammy_score + gab_score + cher_score + alex_score
  let opponent_score := 85
  combined_score - opponent_score = 143 :=
by
  let sammy_score := 20
  let gab_score := 2 * sammy_score
  let cher_score := 2 * gab_score
  let alex_score := cher_score + cher_score / 10
  let combined_score := sammy_score + gab_score + cher_score + alex_score
  let opponent_score := 85
  sorry

end NUMINAMATH_GPT_score_comparison_l174_17495


namespace NUMINAMATH_GPT_majka_numbers_product_l174_17421

/-- Majka created a three-digit funny and a three-digit cheerful number.
    - The funny number starts with an odd digit and alternates between odd and even.
    - The cheerful number starts with an even digit and alternates between even and odd.
    - All digits are distinct and nonzero.
    - The sum of these two numbers is 1617.
    - The product of these two numbers ends in 40.
    Prove that the product of these numbers is 635040.
-/
theorem majka_numbers_product :
  ∃ (a b c : ℕ) (D E F : ℕ),
    -- Define 3-digit funny number as (100 * a + 10 * b + c)
    -- with a and c odd, b even, and all distinct and nonzero
    (a % 2 = 1) ∧ (c % 2 = 1) ∧ (b % 2 = 0) ∧
    -- Define 3-digit cheerful number as (100 * D + 10 * E + F)
    -- with D and F even, E odd, and all distinct and nonzero
    (D % 2 = 0) ∧ (F % 2 = 0) ∧ (E % 2 = 1) ∧
    -- All digits are distinct and nonzero
    (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ D ≠ 0 ∧ E ≠ 0 ∧ F ≠ 0) ∧
    (a ≠ b ∧ a ≠ c ∧ a ≠ D ∧ a ≠ E ∧ a ≠ F ∧
     b ≠ c ∧ b ≠ D ∧ b ≠ E ∧ b ≠ F ∧
     c ≠ D ∧ c ≠ E ∧ c ≠ F ∧
     D ≠ E ∧ D ≠ F ∧ E ≠ F) ∧
    (100 * a + 10 * b + c + 100 * D + 10 * E + F = 1617) ∧
    ((100 * a + 10 * b + c) * (100 * D + 10 * E + F) = 635040) := sorry

end NUMINAMATH_GPT_majka_numbers_product_l174_17421


namespace NUMINAMATH_GPT_find_common_difference_find_possible_a1_l174_17412

structure ArithSeq :=
  (a : ℕ → ℤ) -- defining the sequence
  
noncomputable def S (n : ℕ) (a : ArithSeq) : ℤ :=
  (n * (2 * a.a 0 + (n - 1) * (a.a 1 - a.a 0))) / 2

axiom a4 (a : ArithSeq) : a.a 3 = 10

axiom S20 (a : ArithSeq) : S 20 a = 590

theorem find_common_difference (a : ArithSeq) (d : ℤ) : 
  (a.a 1 - a.a 0 = d) →
  d = 3 :=
sorry

theorem find_possible_a1 (a : ArithSeq) : 
  (∃a1: ℤ, a1 ∈ Set.range a.a) →
  (∀n : ℕ, S n a ≤ S 7 a) →
  Set.range a.a ∩ {n | 18 ≤ n ∧ n ≤ 20} = {18, 19, 20} :=
sorry

end NUMINAMATH_GPT_find_common_difference_find_possible_a1_l174_17412


namespace NUMINAMATH_GPT_convert_500_to_base5_l174_17498

def base10_to_base5 (n : ℕ) : ℕ :=
  -- A function to convert base 10 to base 5 would be defined here
  sorry

theorem convert_500_to_base5 : base10_to_base5 500 = 4000 := 
by 
  -- The actual proof would go here
  sorry

end NUMINAMATH_GPT_convert_500_to_base5_l174_17498


namespace NUMINAMATH_GPT_line_ellipse_common_points_l174_17438

theorem line_ellipse_common_points (m : ℝ) : (m ≥ 1 ∧ m ≠ 5) ↔ (∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ (x^2 / 5) + (y^2 / m) = 1) :=
by 
  sorry

end NUMINAMATH_GPT_line_ellipse_common_points_l174_17438


namespace NUMINAMATH_GPT_number_representation_correct_l174_17400

-- Conditions: 5 in both the tenths and hundredths places, 0 in remaining places.
def number : ℝ := 50.05

theorem number_representation_correct :
  number = 50.05 :=
by 
  -- The proof will show that the definition satisfies the condition.
  sorry

end NUMINAMATH_GPT_number_representation_correct_l174_17400


namespace NUMINAMATH_GPT_count_prime_digit_sums_less_than_10_l174_17426

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ := n / 10 + n % 10

def is_two_digit_number (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem count_prime_digit_sums_less_than_10 :
  ∃ count : ℕ, count = 17 ∧
  ∀ n : ℕ, is_two_digit_number n →
  (is_prime (sum_of_digits n) ∧ sum_of_digits n < 10) ↔
  n ∈ [11, 20, 12, 21, 30, 14, 23, 32, 41, 50, 16, 25, 34, 43, 52, 61, 70] :=
sorry

end NUMINAMATH_GPT_count_prime_digit_sums_less_than_10_l174_17426


namespace NUMINAMATH_GPT_ticket_cost_correct_l174_17476

theorem ticket_cost_correct : 
  ∀ (a : ℝ), 
  (3 * a + 5 * (a / 2) = 30) → 
  10 * a + 8 * (a / 2) ≥ 10 * a + 8 * (a / 2) * 0.9 →
  10 * a + 8 * (a / 2) * 0.9 = 68.733 :=
by
  intro a
  intro h1 h2
  sorry

end NUMINAMATH_GPT_ticket_cost_correct_l174_17476


namespace NUMINAMATH_GPT_fraction_of_180_l174_17435

theorem fraction_of_180 : (1 / 2) * (1 / 3) * (1 / 6) * 180 = 5 := by
  sorry

end NUMINAMATH_GPT_fraction_of_180_l174_17435


namespace NUMINAMATH_GPT_solve_fraction_eq_l174_17481

theorem solve_fraction_eq (x : ℚ) (h : (x^2 + 3 * x + 4) / (x + 3) = x + 6) : x = -7 / 3 :=
sorry

end NUMINAMATH_GPT_solve_fraction_eq_l174_17481


namespace NUMINAMATH_GPT_largest_multiple_of_7_l174_17442

def repeated_188 (k : Nat) : ℕ := (List.replicate k 188).foldr (λ x acc => x * 1000 + acc) 0

theorem largest_multiple_of_7 :
  ∃ n, n = repeated_188 100 ∧ ∃ m, m ≤ 303 ∧ m ≥ 0 ∧ m ≠ 300 ∧ (repeated_188 m % 7 = 0 → n ≥ repeated_188 m) :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_7_l174_17442


namespace NUMINAMATH_GPT_recurring_fraction_difference_l174_17444

theorem recurring_fraction_difference :
  let x := (36 / 99 : ℚ)
  let y := (36 / 100 : ℚ)
  x - y = (1 / 275 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_recurring_fraction_difference_l174_17444


namespace NUMINAMATH_GPT_problem_statement_l174_17437

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x * (x + 4) else x * (x - 4)

theorem problem_statement (a : ℝ) (h : f a > f (8 - a)) : 4 < a :=
by sorry

end NUMINAMATH_GPT_problem_statement_l174_17437


namespace NUMINAMATH_GPT_AndrewAge_l174_17432

noncomputable def AndrewAgeProof : Prop :=
  ∃ (a g : ℕ), g = 10 * a ∧ g - a = 45 ∧ a = 5

-- Proof is not required, so we use sorry to skip the proof.
theorem AndrewAge : AndrewAgeProof := by
  sorry

end NUMINAMATH_GPT_AndrewAge_l174_17432


namespace NUMINAMATH_GPT_min_max_expr_l174_17417

noncomputable def expr (a b c : ℝ) : ℝ :=
  (1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1)) *
  (a^2 / (a^2 + 1) + b^2 / (b^2 + 1) + c^2 / (c^2 + 1))

theorem min_max_expr (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h_cond : a * b + b * c + c * a = 1) :
  27 / 16 ≤ expr a b c ∧ expr a b c ≤ 2 :=
sorry

end NUMINAMATH_GPT_min_max_expr_l174_17417


namespace NUMINAMATH_GPT_relationship_of_y_values_l174_17456

theorem relationship_of_y_values (m : ℝ) (y1 y2 y3 : ℝ) :
  (∀ x y, (x = -2 ∧ y = y1 ∨ x = -1 ∧ y = y2 ∨ x = 1 ∧ y = y3) → (y = (m^2 + 1) / x)) →
  y2 < y1 ∧ y1 < y3 :=
by
  sorry

end NUMINAMATH_GPT_relationship_of_y_values_l174_17456


namespace NUMINAMATH_GPT_regression_equation_is_correct_l174_17406

theorem regression_equation_is_correct 
  (linear_corr : ∃ (f : ℝ → ℝ), ∀ (x : ℝ), ∃ (y : ℝ), y = f x)
  (mean_b : ℝ)
  (mean_x : ℝ)
  (mean_y : ℝ)
  (mean_b_eq : mean_b = 0.51)
  (mean_x_eq : mean_x = 61.75)
  (mean_y_eq : mean_y = 38.14) : 
  mean_y = mean_b * mean_x + 6.65 :=
sorry

end NUMINAMATH_GPT_regression_equation_is_correct_l174_17406


namespace NUMINAMATH_GPT_mary_age_l174_17413

theorem mary_age (x : ℤ) (n m : ℤ) : (x - 2 = n^2) ∧ (x + 2 = m^3) → x = 6 := by
  sorry

end NUMINAMATH_GPT_mary_age_l174_17413


namespace NUMINAMATH_GPT_device_records_720_instances_in_one_hour_l174_17416

-- Definitions
def seconds_per_hour : ℕ := 3600
def interval : ℕ := 5
def instances_per_hour := seconds_per_hour / interval

-- Theorem Statement
theorem device_records_720_instances_in_one_hour : instances_per_hour = 720 :=
by
  sorry

end NUMINAMATH_GPT_device_records_720_instances_in_one_hour_l174_17416


namespace NUMINAMATH_GPT_clarence_oranges_l174_17420

def initial_oranges := 5
def oranges_from_joyce := 3
def total_oranges := initial_oranges + oranges_from_joyce

theorem clarence_oranges : total_oranges = 8 :=
  by
  sorry

end NUMINAMATH_GPT_clarence_oranges_l174_17420


namespace NUMINAMATH_GPT_ambiguous_times_l174_17410

theorem ambiguous_times (h m : ℝ) : 
  (∃ k l : ℕ, 0 ≤ k ∧ k < 12 ∧ 0 ≤ l ∧ l < 12 ∧ 
              (12 * h = k * 360 + m) ∧ 
              (12 * m = l * 360 + h) ∧
              k ≠ l) → 
  (∃ n : ℕ, n = 132) := 
sorry

end NUMINAMATH_GPT_ambiguous_times_l174_17410


namespace NUMINAMATH_GPT_larger_cylinder_candies_l174_17484

theorem larger_cylinder_candies (v₁ v₂ : ℝ) (c₁ c₂ : ℕ) (h₁ : v₁ = 72) (h₂ : c₁ = 30) (h₃ : v₂ = 216) (h₄ : (c₁ : ℝ)/v₁ = (c₂ : ℝ)/v₂) : c₂ = 90 := by
  -- v1 h1 h2 v2 c2 h4 are directly appearing in the conditions
  -- ratio h4 states the condition for densities to be the same 
  sorry

end NUMINAMATH_GPT_larger_cylinder_candies_l174_17484


namespace NUMINAMATH_GPT_andy_questions_wrong_l174_17430

variable (a b c d : ℕ)

theorem andy_questions_wrong
  (h1 : a + b = c + d)
  (h2 : a + d = b + c + 6)
  (h3 : c = 7)
  (h4 : d = 9) :
  a = 10 :=
by {
  sorry  -- Proof would go here
}

end NUMINAMATH_GPT_andy_questions_wrong_l174_17430


namespace NUMINAMATH_GPT_range_of_sin_cos_expression_l174_17491

variable (a b c A B C : ℝ)

theorem range_of_sin_cos_expression
  (h1 : a = b)
  (h2 : c * Real.sin A = -a * Real.cos C) :
  1 < 2 * Real.sin (A + Real.pi / 6) :=
sorry

end NUMINAMATH_GPT_range_of_sin_cos_expression_l174_17491


namespace NUMINAMATH_GPT_eighteenth_prime_l174_17415

-- Define the necessary statements
def isPrime (n : ℕ) : Prop := sorry

def primeSeq (n : ℕ) : ℕ :=
  if n = 0 then
    2
  else if n = 1 then
    3
  else
    -- Function to generate the n-th prime number
    sorry

theorem eighteenth_prime :
  primeSeq 17 = 67 := by
  sorry

end NUMINAMATH_GPT_eighteenth_prime_l174_17415


namespace NUMINAMATH_GPT_volleyballTeam_starters_l174_17488

noncomputable def chooseStarters (totalPlayers : ℕ) (quadruplets : ℕ) (starters : ℕ) : ℕ :=
  let remainingPlayers := totalPlayers - quadruplets
  let chooseQuadruplet := quadruplets
  let chooseRemaining := Nat.choose remainingPlayers (starters - 1)
  chooseQuadruplet * chooseRemaining

theorem volleyballTeam_starters :
  chooseStarters 16 4 6 = 3168 :=
by
  sorry

end NUMINAMATH_GPT_volleyballTeam_starters_l174_17488


namespace NUMINAMATH_GPT_production_steps_description_l174_17455

-- Definition of the choices
inductive FlowchartType
| ProgramFlowchart
| ProcessFlowchart
| KnowledgeStructureDiagram
| OrganizationalStructureDiagram

-- Conditions
def describeProductionSteps (flowchart : FlowchartType) : Prop :=
flowchart = FlowchartType.ProcessFlowchart

-- The statement to be proved
theorem production_steps_description:
  describeProductionSteps FlowchartType.ProcessFlowchart := 
sorry -- proof to be provided

end NUMINAMATH_GPT_production_steps_description_l174_17455


namespace NUMINAMATH_GPT_hexagon_angles_l174_17428

theorem hexagon_angles
  (AB CD EF BC DE FA : ℝ)
  (F A B C D E : Type*)
  (FAB ABC EFA CDE : ℝ)
  (h1 : AB = CD)
  (h2 : AB = EF)
  (h3 : BC = DE)
  (h4 : BC = FA)
  (h5 : FAB + ABC = 240)
  (h6 : FAB + EFA = 240) :
  FAB + CDE = 240 :=
sorry

end NUMINAMATH_GPT_hexagon_angles_l174_17428


namespace NUMINAMATH_GPT_no_integer_pairs_satisfy_equation_l174_17431

theorem no_integer_pairs_satisfy_equation :
  ∀ (m n : ℤ), m^3 + 6 * m^2 + 5 * m ≠ 27 * n^3 + 27 * n^2 + 9 * n + 1 :=
by
  intros m n
  sorry

end NUMINAMATH_GPT_no_integer_pairs_satisfy_equation_l174_17431


namespace NUMINAMATH_GPT_range_of_a_l174_17429

theorem range_of_a (a : ℝ) (x : ℝ) :
  (¬(x > a) →¬(x^2 + 2*x - 3 > 0)) → (a ≥ 1 ) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l174_17429


namespace NUMINAMATH_GPT_machine_produces_one_item_in_40_seconds_l174_17418

theorem machine_produces_one_item_in_40_seconds :
  (60 * 1) / 90 * 60 = 40 :=
by
  sorry

end NUMINAMATH_GPT_machine_produces_one_item_in_40_seconds_l174_17418


namespace NUMINAMATH_GPT_hyperbola_condition_l174_17457

theorem hyperbola_condition (m : ℝ) : 
  (∀ x y : ℝ, (m-2) * (m+3) < 0 → (x^2) / (m-2) + (y^2) / (m+3) = 1) ↔ -3 < m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_condition_l174_17457


namespace NUMINAMATH_GPT_max_surface_area_l174_17499

theorem max_surface_area (l w h : ℕ) (h_conditions : l + w + h = 88) : 
  2 * (l * w + l * h + w * h) ≤ 224 :=
sorry

end NUMINAMATH_GPT_max_surface_area_l174_17499


namespace NUMINAMATH_GPT_complement_of_A_in_R_l174_17483

open Set

variable (R : Set ℝ) (A : Set ℝ)

def real_numbers : Set ℝ := {x | true}

def set_A : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2}

theorem complement_of_A_in_R : (real_numbers \ set_A) = {y | y < 0} := by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_R_l174_17483


namespace NUMINAMATH_GPT_ln_sqrt2_lt_sqrt2_div2_ln_sin_cos_sum_l174_17409

theorem ln_sqrt2_lt_sqrt2_div2 : Real.log (Real.sqrt 2) < Real.sqrt 2 / 2 :=
sorry

theorem ln_sin_cos_sum : 2 * Real.log (Real.sin (1/8) + Real.cos (1/8)) < 1 / 4 :=
sorry

end NUMINAMATH_GPT_ln_sqrt2_lt_sqrt2_div2_ln_sin_cos_sum_l174_17409


namespace NUMINAMATH_GPT_number_of_packs_l174_17452

-- Given conditions
def cost_per_pack : ℕ := 11
def total_money : ℕ := 110

-- Statement to prove
theorem number_of_packs :
  total_money / cost_per_pack = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_packs_l174_17452


namespace NUMINAMATH_GPT_smallest_possible_value_of_N_l174_17461

-- Define the dimensions of the block
variables (l m n : ℕ) 

-- Define the condition that the product of dimensions minus one is 143
def hidden_cubes_count (l m n : ℕ) : Prop := (l - 1) * (m - 1) * (n - 1) = 143

-- Define the total number of cubes in the outer block
def total_cubes (l m n : ℕ) : ℕ := l * m * n

-- The final proof statement
theorem smallest_possible_value_of_N : 
  ∃ (l m n : ℕ), hidden_cubes_count l m n → N = total_cubes l m n → N = 336 :=
sorry

end NUMINAMATH_GPT_smallest_possible_value_of_N_l174_17461


namespace NUMINAMATH_GPT_exists_rational_non_integer_a_not_exists_rational_non_integer_b_l174_17434

-- Define rational non-integer numbers
def is_rational_non_integer (x : ℚ) : Prop := ¬(∃ (z : ℤ), x = z)

-- (a) Proof for existance of rational non-integer numbers y and x such that 19x + 8y, 8x + 3y are integers
theorem exists_rational_non_integer_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ a b : ℤ, 19 * x + 8 * y = a ∧ 8 * x + 3 * y = b) :=
sorry

-- (b) Proof for non-existance of rational non-integer numbers y and x such that 19x² + 8y², 8x² + 3y² are integers
theorem not_exists_rational_non_integer_b :
  ¬ ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ m n : ℤ, 19 * x^2 + 8 * y^2 = m ∧ 8 * x^2 + 3 * y^2 = n) :=
sorry

end NUMINAMATH_GPT_exists_rational_non_integer_a_not_exists_rational_non_integer_b_l174_17434


namespace NUMINAMATH_GPT_find_certain_number_l174_17460

theorem find_certain_number (x y : ℕ) (h1 : x = 19) (h2 : x + y = 36) :
  8 * x + 3 * y = 203 := by
  sorry

end NUMINAMATH_GPT_find_certain_number_l174_17460


namespace NUMINAMATH_GPT_inequality_sine_cosine_l174_17472

theorem inequality_sine_cosine (t : ℝ) (ht : t > 0) : 3 * Real.sin t < 2 * t + t * Real.cos t := 
sorry

end NUMINAMATH_GPT_inequality_sine_cosine_l174_17472


namespace NUMINAMATH_GPT_return_trip_time_is_15_or_67_l174_17408

variable (d p w : ℝ)

-- Conditions
axiom h1 : (d / (p - w)) = 100
axiom h2 : ∃ t : ℝ, t = d / p ∧ (d / (p + w)) = t - 15

-- Correct answer to prove: time for the return trip is 15 minutes or 67 minutes
theorem return_trip_time_is_15_or_67 : (d / (p + w)) = 15 ∨ (d / (p + w)) = 67 := 
by 
  sorry

end NUMINAMATH_GPT_return_trip_time_is_15_or_67_l174_17408


namespace NUMINAMATH_GPT_surface_area_of_circumscribed_sphere_l174_17405

theorem surface_area_of_circumscribed_sphere :
  let a := 2
  let AD := Real.sqrt (a^2 - (a/2)^2)
  let r := Real.sqrt (1 + 1 + AD^2) / 2
  4 * Real.pi * r^2 = 5 * Real.pi := by
  sorry

end NUMINAMATH_GPT_surface_area_of_circumscribed_sphere_l174_17405


namespace NUMINAMATH_GPT_solution_of_fraction_l174_17497

theorem solution_of_fraction (x : ℝ) (h1 : x^2 - 9 = 0) (h2 : x + 3 ≠ 0) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solution_of_fraction_l174_17497


namespace NUMINAMATH_GPT_area_difference_of_squares_l174_17447

theorem area_difference_of_squares (d1 d2 : ℝ) (h1 : d1 = 19) (h2 : d2 = 17) : 
  let s1 := d1 / Real.sqrt 2
  let s2 := d2 / Real.sqrt 2
  let area1 := s1 * s1
  let area2 := s2 * s2
  (area1 - area2) = 36 :=
by
  sorry

end NUMINAMATH_GPT_area_difference_of_squares_l174_17447


namespace NUMINAMATH_GPT_sequence_formula_l174_17466

theorem sequence_formula (a : ℕ → ℕ) (c : ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = a n + c * n) 
(h₃ : a 1 ≠ a 2) (h₄ : a 2 * a 2 = a 1 * a 3) : c = 2 ∧ ∀ n, a n = n^2 - n + 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l174_17466


namespace NUMINAMATH_GPT_inverse_proportion_function_range_m_l174_17439

theorem inverse_proportion_function_range_m
  (x1 x2 y1 y2 m : ℝ)
  (h_func_A : y1 = (5 * m - 2) / x1)
  (h_func_B : y2 = (5 * m - 2) / x2)
  (h_x : x1 < x2)
  (h_x_neg : x2 < 0)
  (h_y : y1 < y2) :
  m < 2 / 5 :=
sorry

end NUMINAMATH_GPT_inverse_proportion_function_range_m_l174_17439


namespace NUMINAMATH_GPT_linear_function_point_l174_17454

theorem linear_function_point (a b : ℝ) (h : b = 2 * a - 1) : 2 * a - b + 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_point_l174_17454


namespace NUMINAMATH_GPT_max_value_of_x_plus_3y_l174_17463

theorem max_value_of_x_plus_3y (x y : ℝ) (h : x^2 / 9 + y^2 = 1) : 
    ∃ θ : ℝ, x = 3 * Real.cos θ ∧ y = Real.sin θ ∧ (x + 3 * y) ≤ 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_x_plus_3y_l174_17463


namespace NUMINAMATH_GPT_joao_speed_l174_17445

theorem joao_speed (d : ℝ) (v1 : ℝ) (t1 t2 : ℝ) (h1 : v1 = 10) (h2 : t1 = 6 / 60) (h3 : t2 = 8 / 60) : 
  d = v1 * t1 → d = 10 * (6 / 60) → (d / t2) = 7.5 := 
by
  sorry

end NUMINAMATH_GPT_joao_speed_l174_17445


namespace NUMINAMATH_GPT_train_crosses_lamp_post_in_30_seconds_l174_17401

open Real

/-- Prove that given a train that crosses a 2500 m long bridge in 120 s and has a length of
    833.33 m, it takes the train 30 seconds to cross a lamp post. -/
theorem train_crosses_lamp_post_in_30_seconds (L_train : ℝ) (L_bridge : ℝ) (T_bridge : ℝ) (T_lamp_post : ℝ)
  (hL_train : L_train = 833.33)
  (hL_bridge : L_bridge = 2500)
  (hT_bridge : T_bridge = 120)
  (ht : T_lamp_post = (833.33 / ((833.33 + 2500) / 120))) :
  T_lamp_post = 30 :=
by
  sorry

end NUMINAMATH_GPT_train_crosses_lamp_post_in_30_seconds_l174_17401


namespace NUMINAMATH_GPT_length_of_the_train_l174_17464

noncomputable def length_of_train (s1 s2 : ℝ) (t1 t2 : ℕ) : ℝ :=
  (s1 * t1 + s2 * t2) / 2

theorem length_of_the_train :
  ∀ (s1 s2 : ℝ) (t1 t2 : ℕ), s1 = 25 → t1 = 8 → s2 = 100 / 3 → t2 = 6 → length_of_train s1 s2 t1 t2 = 200 :=
by
  intros s1 s2 t1 t2 hs1 ht1 hs2 ht2
  rw [hs1, ht1, hs2, ht2]
  simp [length_of_train]
  norm_num

end NUMINAMATH_GPT_length_of_the_train_l174_17464


namespace NUMINAMATH_GPT_sin_cos_inequality_l174_17492

theorem sin_cos_inequality (x : ℝ) (n : ℕ) : 
  (Real.sin (2 * x))^n + (Real.sin x^n - Real.cos x^n)^2 ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_sin_cos_inequality_l174_17492


namespace NUMINAMATH_GPT_one_angle_greater_135_l174_17496

noncomputable def angles_sum_not_form_triangle (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : Prop :=
  ∀ (A B C : ℝ), 
   (A < a + b ∧ A < a + c ∧ A < b + c) →
  (B < a + b ∧ B < a + c ∧ B < b + c) →
  (C < a + b ∧ C < a + c ∧ C < b + c) →
  ∃ α β γ, α > 135 ∧ β < 60 ∧ γ < 60 ∧ α + β + γ = 180

theorem one_angle_greater_135 {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : angles_sum_not_form_triangle a b c ha hb hc) :
  ∃ α β γ, α > 135 ∧ α + β + γ = 180 :=
sorry

end NUMINAMATH_GPT_one_angle_greater_135_l174_17496


namespace NUMINAMATH_GPT_carla_smoothies_serving_l174_17451

theorem carla_smoothies_serving :
  ∀ (watermelon_puree : ℕ) (cream : ℕ) (serving_size : ℕ),
  watermelon_puree = 500 → cream = 100 → serving_size = 150 →
  (watermelon_puree + cream) / serving_size = 4 :=
by
  intros watermelon_puree cream serving_size
  intro h1 -- watermelon_puree = 500
  intro h2 -- cream = 100
  intro h3 -- serving_size = 150
  sorry

end NUMINAMATH_GPT_carla_smoothies_serving_l174_17451


namespace NUMINAMATH_GPT_even_function_value_at_2_l174_17493

theorem even_function_value_at_2 {a : ℝ} (h : ∀ x : ℝ, (x + 1) * (x - a) = (-x + 1) * (-x - a)) : 
  ((2 + 1) * (2 - a)) = 3 := by
  sorry

end NUMINAMATH_GPT_even_function_value_at_2_l174_17493


namespace NUMINAMATH_GPT_area_of_yard_proof_l174_17440

def area_of_yard (L W : ℕ) : ℕ :=
  L * W

theorem area_of_yard_proof (L W : ℕ) (hL : L = 40) (hFence : 2 * W + L = 52) : 
  area_of_yard L W = 240 := 
by 
  sorry

end NUMINAMATH_GPT_area_of_yard_proof_l174_17440


namespace NUMINAMATH_GPT_solve_for_x_l174_17487

theorem solve_for_x (x : ℝ) (h : x - 5.90 = 9.28) : x = 15.18 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l174_17487


namespace NUMINAMATH_GPT_farm_problem_l174_17474

theorem farm_problem (D C : ℕ) (h1 : D + C = 15) (h2 : 2 * D + 4 * C = 42) : C = 6 :=
sorry

end NUMINAMATH_GPT_farm_problem_l174_17474


namespace NUMINAMATH_GPT_remainder_of_n_when_divided_by_7_l174_17470

theorem remainder_of_n_when_divided_by_7 (n : ℕ) :
  (n^2 ≡ 2 [MOD 7]) ∧ (n^3 ≡ 6 [MOD 7]) → (n ≡ 3 [MOD 7]) :=
by sorry

end NUMINAMATH_GPT_remainder_of_n_when_divided_by_7_l174_17470


namespace NUMINAMATH_GPT_problem_statement_l174_17448

theorem problem_statement (a b c d n : Nat) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < n) (h_eq : 7 * 4^n = a^2 + b^2 + c^2 + d^2) : 
  a ≥ 2^(n-1) ∧ b ≥ 2^(n-1) ∧ c ≥ 2^(n-1) ∧ d ≥ 2^(n-1) :=
sorry

end NUMINAMATH_GPT_problem_statement_l174_17448


namespace NUMINAMATH_GPT_original_number_of_men_l174_17419

theorem original_number_of_men (x : ℕ) (h : 10 * x = 7 * (x + 10)) : x = 24 := 
by 
  -- Add your proof here 
  sorry

end NUMINAMATH_GPT_original_number_of_men_l174_17419


namespace NUMINAMATH_GPT_number_of_children_admitted_l174_17475

variable (children adults : ℕ)

def admission_fee_children : ℝ := 1.5
def admission_fee_adults  : ℝ := 4

def total_people : ℕ := 315
def total_fees   : ℝ := 810

theorem number_of_children_admitted :
  ∃ (C A : ℕ), C + A = total_people ∧ admission_fee_children * C + admission_fee_adults * A = total_fees ∧ C = 180 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_admitted_l174_17475


namespace NUMINAMATH_GPT_students_average_age_l174_17489

theorem students_average_age (A : ℝ) (students_count teacher_age total_average new_count : ℝ) 
  (h1 : students_count = 30)
  (h2 : teacher_age = 45)
  (h3 : new_count = students_count + 1)
  (h4 : total_average = 15) 
  (h5 : total_average = (A * students_count + teacher_age) / new_count) : 
  A = 14 :=
by
  sorry

end NUMINAMATH_GPT_students_average_age_l174_17489


namespace NUMINAMATH_GPT_difference_between_numbers_l174_17443

theorem difference_between_numbers : 
  ∃ (a : ℕ), a + 10 * a = 30000 → 9 * a = 24543 := 
by 
  sorry

end NUMINAMATH_GPT_difference_between_numbers_l174_17443


namespace NUMINAMATH_GPT_find_x_l174_17424

theorem find_x (x : ℝ) (h1 : x > 0) (h2 : 1/2 * (2 * x) * x = 72) : x = 6 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l174_17424


namespace NUMINAMATH_GPT_larry_expression_correct_l174_17471

theorem larry_expression_correct (a b c d : ℤ) (e : ℤ) :
  (a = 1) → (b = 2) → (c = 3) → (d = 4) →
  (a - b - c - d + e = -2 - e) → (e = 3) :=
by
  intros ha hb hc hd heq
  rw [ha, hb, hc, hd] at heq
  linarith

end NUMINAMATH_GPT_larry_expression_correct_l174_17471


namespace NUMINAMATH_GPT_jane_reading_days_l174_17485

theorem jane_reading_days
  (pages : ℕ)
  (half_pages : ℕ)
  (speed_first_half : ℕ)
  (speed_second_half : ℕ)
  (days_first_half : ℕ)
  (days_second_half : ℕ)
  (total_days : ℕ)
  (h1 : pages = 500)
  (h2 : half_pages = pages / 2)
  (h3 : speed_first_half = 10)
  (h4 : speed_second_half = 5)
  (h5 : days_first_half = half_pages / speed_first_half)
  (h6 : days_second_half = half_pages / speed_second_half)
  (h7 : total_days = days_first_half + days_second_half) :
  total_days = 75 :=
by
  sorry

end NUMINAMATH_GPT_jane_reading_days_l174_17485


namespace NUMINAMATH_GPT_smallest_m_plus_n_l174_17446

theorem smallest_m_plus_n (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_lt : m < n)
    (h_eq : 1978^m % 1000 = 1978^n % 1000) : m + n = 26 :=
sorry

end NUMINAMATH_GPT_smallest_m_plus_n_l174_17446


namespace NUMINAMATH_GPT_function_decreasing_interval_l174_17467

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

def decreasing_interval (a b : ℝ) : Prop :=
  ∀ x : ℝ, a < x ∧ x < b → 0 > (deriv f x)

theorem function_decreasing_interval : decreasing_interval (-1) 3 :=
by 
  sorry

end NUMINAMATH_GPT_function_decreasing_interval_l174_17467


namespace NUMINAMATH_GPT_find_distance_l174_17433

-- Definitions based on the given conditions
def speed_of_boat := 16 -- in kmph
def speed_of_stream := 2 -- in kmph
def total_time := 960 -- in hours
def downstream_speed := speed_of_boat + speed_of_stream
def upstream_speed := speed_of_boat - speed_of_stream

-- Prove that the distance D is 7590 km given the total time and speeds
theorem find_distance (D : ℝ) :
  (D / downstream_speed + D / upstream_speed = total_time) → D = 7590 :=
by
  sorry

end NUMINAMATH_GPT_find_distance_l174_17433


namespace NUMINAMATH_GPT_min_shift_value_l174_17468

theorem min_shift_value (φ : ℝ) (hφ : φ > 0) :
  (∃ k : ℤ, φ = -k * π / 3 + π / 6) →
  ∃ φ_min : ℝ, φ_min = π / 6 ∧ (∀ φ', φ' > 0 → ∃ k' : ℤ, φ' = -k' * π / 3 + π / 6 → φ_min ≤ φ') :=
by
  intro h
  use π / 6
  constructor
  . sorry
  . sorry

end NUMINAMATH_GPT_min_shift_value_l174_17468


namespace NUMINAMATH_GPT_judge_guilty_cases_l174_17459

theorem judge_guilty_cases :
  let total_cases := 27
  let dismissed_cases := 3
  let remaining_cases := total_cases - dismissed_cases
  let innocent_cases := 3 * remaining_cases / 4
  let delayed_rulings := 2
  remaining_cases - innocent_cases - delayed_rulings = 4 :=
by
  let total_cases := 27
  let dismissed_cases := 3
  let remaining_cases := total_cases - dismissed_cases
  let innocent_cases := 3 * remaining_cases / 4
  let delayed_rulings := 2
  show remaining_cases - innocent_cases - delayed_rulings = 4
  sorry

end NUMINAMATH_GPT_judge_guilty_cases_l174_17459


namespace NUMINAMATH_GPT_income_is_20000_l174_17404

-- Definitions from conditions
def income (x : ℕ) : ℕ := 4 * x
def expenditure (x : ℕ) : ℕ := 3 * x
def savings : ℕ := 5000

-- Theorem to prove the income
theorem income_is_20000 (x : ℕ) (h : income x - expenditure x = savings) : income x = 20000 :=
by
  sorry

end NUMINAMATH_GPT_income_is_20000_l174_17404


namespace NUMINAMATH_GPT_relationship_withdrawn_leftover_l174_17423

-- Definitions based on the problem conditions
def pie_cost : ℝ := 6
def sandwich_cost : ℝ := 3
def book_cost : ℝ := 10
def book_discount : ℝ := 0.2 * book_cost
def book_price_with_discount : ℝ := book_cost - book_discount
def total_spent_before_tax : ℝ := pie_cost + sandwich_cost + book_price_with_discount
def sales_tax_rate : ℝ := 0.05
def sales_tax : ℝ := sales_tax_rate * total_spent_before_tax
def total_spent_with_tax : ℝ := total_spent_before_tax + sales_tax

-- Given amount withdrawn and amount left after shopping
variables (X Y : ℝ)

-- Theorem statement
theorem relationship_withdrawn_leftover :
  Y = X - total_spent_with_tax :=
sorry

end NUMINAMATH_GPT_relationship_withdrawn_leftover_l174_17423


namespace NUMINAMATH_GPT_factor_expression_l174_17450

variable (x : ℝ)

theorem factor_expression : 75 * x^3 - 250 * x^7 = 25 * x^3 * (3 - 10 * x^4) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l174_17450


namespace NUMINAMATH_GPT_largest_possible_pencils_in_each_package_l174_17479

def ming_pencils : ℕ := 48
def catherine_pencils : ℕ := 36
def lucas_pencils : ℕ := 60

theorem largest_possible_pencils_in_each_package (d : ℕ) (h_ming: ming_pencils % d = 0) (h_catherine: catherine_pencils % d = 0) (h_lucas: lucas_pencils % d = 0) : d ≤ ming_pencils ∧ d ≤ catherine_pencils ∧ d ≤ lucas_pencils ∧ (∀ e, (ming_pencils % e = 0 ∧ catherine_pencils % e = 0 ∧ lucas_pencils % e = 0) → e ≤ d) → d = 12 :=
by 
  sorry

end NUMINAMATH_GPT_largest_possible_pencils_in_each_package_l174_17479


namespace NUMINAMATH_GPT_correct_average_of_10_numbers_l174_17441

theorem correct_average_of_10_numbers
  (incorrect_avg : ℕ)
  (n : ℕ)
  (incorrect_read : ℕ)
  (correct_read : ℕ)
  (incorrect_total_sum : ℕ) :
  incorrect_avg = 19 →
  n = 10 →
  incorrect_read = 26 →
  correct_read = 76 →
  incorrect_total_sum = incorrect_avg * n →
  (correct_total_sum : ℕ) = incorrect_total_sum - incorrect_read + correct_read →
  (correct_avg : ℕ) = correct_total_sum / n →
  correct_avg = 24 :=
by
  intros
  sorry

end NUMINAMATH_GPT_correct_average_of_10_numbers_l174_17441


namespace NUMINAMATH_GPT_total_cost_of_coat_l174_17402

def original_price : ℝ := 150
def sale_discount : ℝ := 0.25
def additional_discount : ℝ := 10
def sales_tax : ℝ := 0.10

theorem total_cost_of_coat :
  let sale_price := original_price * (1 - sale_discount)
  let price_after_discount := sale_price - additional_discount
  let final_price := price_after_discount * (1 + sales_tax)
  final_price = 112.75 :=
by
  -- sorry for the actual proof
  sorry

end NUMINAMATH_GPT_total_cost_of_coat_l174_17402


namespace NUMINAMATH_GPT_count_negative_terms_in_sequence_l174_17486

theorem count_negative_terms_in_sequence : 
  ∃ (s : List ℕ), (∀ n ∈ s, n^2 - 8*n + 12 < 0) ∧ s.length = 3 ∧ (∀ n ∈ s, 2 < n ∧ n < 6) :=
by
  sorry

end NUMINAMATH_GPT_count_negative_terms_in_sequence_l174_17486


namespace NUMINAMATH_GPT_find_y_l174_17469

theorem find_y (y : ℝ) (h : 9 * y^2 + 36 * y^2 + 9 * y^2 = 1300) : 
  y = Real.sqrt 1300 / Real.sqrt 54 :=
by 
  sorry

end NUMINAMATH_GPT_find_y_l174_17469


namespace NUMINAMATH_GPT_problem1_proof_problem2_proof_l174_17478

noncomputable def problem1 (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : Prop :=
  |a| + |b| ≤ Real.sqrt 2

noncomputable def problem2 (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : Prop :=
  |a^3 / b| + |b^3 / a| ≥ 1

theorem problem1_proof (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : problem1 a b h₁ h₂ h₃ :=
  sorry

theorem problem2_proof (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : problem2 a b h₁ h₂ h₃ :=
  sorry

end NUMINAMATH_GPT_problem1_proof_problem2_proof_l174_17478


namespace NUMINAMATH_GPT_arith_geo_seq_prop_l174_17427

theorem arith_geo_seq_prop (a1 a2 b1 b2 b3 : ℝ)
  (arith_seq_condition : 1 + 2 * (a1 - 1) = a2)
  (geo_seq_condition1 : b1 * b3 = 4)
  (geo_seq_condition2 : b1 > 0)
  (geo_seq_condition3 : 1 * b1 * b2 * b3 * 4 = (b1 * b3 * -4)) :
  (a2 - a1) / b2 = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_arith_geo_seq_prop_l174_17427


namespace NUMINAMATH_GPT_projection_vector_satisfies_conditions_l174_17462

variable (v1 v2 : ℚ)

def line_l (t : ℚ) : ℚ × ℚ :=
(2 + 3 * t, 5 - 2 * t)

def line_m (s : ℚ) : ℚ × ℚ :=
(-2 + 3 * s, 7 - 2 * s)

theorem projection_vector_satisfies_conditions :
  3 * v1 + 2 * v2 = 6 ∧ 
  ∃ k : ℚ, v1 = k * 3 ∧ v2 = k * (-2) → 
  (v1, v2) = (18 / 5, -12 / 5) :=
by
  sorry

end NUMINAMATH_GPT_projection_vector_satisfies_conditions_l174_17462
