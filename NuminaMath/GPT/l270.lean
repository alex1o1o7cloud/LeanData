import Mathlib

namespace NUMINAMATH_GPT_age_problem_l270_27065

-- Defining the conditions and the proof problem
variables (B A : ℕ) -- B and A are natural numbers

-- Given conditions
def B_age : ℕ := 38
def A_age (B : ℕ) : ℕ := B + 8
def age_in_10_years (A : ℕ) : ℕ := A + 10
def years_ago (B : ℕ) (X : ℕ) : ℕ := B - X

-- Lean statement of the problem
theorem age_problem (X : ℕ) (hB : B = B_age) (hA : A = A_age B):
  age_in_10_years A = 2 * (years_ago B X) → X = 10 :=
by
  sorry

end NUMINAMATH_GPT_age_problem_l270_27065


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l270_27007

theorem arithmetic_sequence_problem (a : Nat → Int) (d a1 : Int)
  (h1 : ∀ n, a n = a1 + (n - 1) * d) 
  (h2 : a 1 + 3 * a 8 = 1560) :
  2 * a 9 - a 10 = 507 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l270_27007


namespace NUMINAMATH_GPT_chord_segments_division_l270_27049

theorem chord_segments_division (O : Point) (r r0 : ℝ) (h : r0 < r) : 
  3 * r0 ≥ r :=
sorry

end NUMINAMATH_GPT_chord_segments_division_l270_27049


namespace NUMINAMATH_GPT_sum_of_three_integers_with_product_of_5_cubed_l270_27054

theorem sum_of_three_integers_with_product_of_5_cubed :
  ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  a * b * c = 5^3 ∧ 
  a + b + c = 31 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_integers_with_product_of_5_cubed_l270_27054


namespace NUMINAMATH_GPT_tall_students_proof_l270_27017

variables (T : ℕ) (Short Average Tall : ℕ)

-- Given in the problem:
def total_students := T = 400
def short_students := Short = 2 * T / 5
def average_height_students := Average = 150

-- Prove:
theorem tall_students_proof (hT : total_students T) (hShort : short_students T Short) (hAverage : average_height_students Average) :
  Tall = T - (Short + Average) :=
by
  sorry

end NUMINAMATH_GPT_tall_students_proof_l270_27017


namespace NUMINAMATH_GPT_intersection_points_count_l270_27097

def line1 (x y : ℝ) : Prop := 4 * y - 3 * x = 2
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y = 9
def line3 (x y : ℝ) : Prop := x - y = 1

theorem intersection_points_count :
  ∃ p1 p2 p3 : ℝ × ℝ,
  (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
  (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
  (line1 p3.1 p3.2 ∧ line3 p3.1 p3.2) ∧
  (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3) :=
  sorry

end NUMINAMATH_GPT_intersection_points_count_l270_27097


namespace NUMINAMATH_GPT_solve_for_a_l270_27095

theorem solve_for_a (x y a : ℤ) (h1 : x = 1) (h2 : y = 2) (h3 : x - a * y = 3) : a = -1 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_solve_for_a_l270_27095


namespace NUMINAMATH_GPT_trees_in_one_row_l270_27025

theorem trees_in_one_row (total_revenue : ℕ) (price_per_apple : ℕ) (apples_per_tree : ℕ) (trees_per_row : ℕ)
  (revenue_condition : total_revenue = 30)
  (price_condition : price_per_apple = 1 / 2)
  (apples_condition : apples_per_tree = 5)
  (trees_condition : trees_per_row = 4) :
  trees_per_row = 4 := by
  sorry

end NUMINAMATH_GPT_trees_in_one_row_l270_27025


namespace NUMINAMATH_GPT_product_third_side_approximation_l270_27014

def triangle_third_side (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

noncomputable def product_of_third_side_lengths : ℝ :=
  Real.sqrt 41 * 3

theorem product_third_side_approximation (a b : ℝ) (h₁ : a = 4) (h₂ : b = 5) :
  ∃ (c₁ c₂ : ℝ), triangle_third_side a b c₁ ∧ triangle_third_side a b c₂ ∧
  abs ((c₁ * c₂) - 19.2) < 0.1 :=
sorry

end NUMINAMATH_GPT_product_third_side_approximation_l270_27014


namespace NUMINAMATH_GPT_teacher_earnings_l270_27090

noncomputable def cost_per_half_hour : ℝ := 10
noncomputable def lesson_duration_in_hours : ℝ := 1
noncomputable def lessons_per_week : ℝ := 1
noncomputable def weeks : ℝ := 5

theorem teacher_earnings : 
  2 * cost_per_half_hour * lesson_duration_in_hours * lessons_per_week * weeks = 100 :=
by
  sorry

end NUMINAMATH_GPT_teacher_earnings_l270_27090


namespace NUMINAMATH_GPT_Mike_changed_2_sets_of_tires_l270_27000

theorem Mike_changed_2_sets_of_tires
  (wash_time_per_car : ℕ := 10)
  (oil_change_time_per_car : ℕ := 15)
  (tire_change_time_per_set : ℕ := 30)
  (num_washed_cars : ℕ := 9)
  (num_oil_changes : ℕ := 6)
  (total_work_time_minutes : ℕ := 4 * 60) :
  ((total_work_time_minutes - (num_washed_cars * wash_time_per_car + num_oil_changes * oil_change_time_per_car)) / tire_change_time_per_set) = 2 :=
by
  sorry

end NUMINAMATH_GPT_Mike_changed_2_sets_of_tires_l270_27000


namespace NUMINAMATH_GPT_abs_neg_2023_l270_27088

theorem abs_neg_2023 : |(-2023)| = 2023 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_2023_l270_27088


namespace NUMINAMATH_GPT_gallery_pieces_total_l270_27098

noncomputable def TotalArtGalleryPieces (A : ℕ) : Prop :=
  let D := (1 : ℚ) / 3 * A
  let N := A - D
  let notDisplayedSculptures := (2 : ℚ) / 3 * N
  let totalSculpturesNotDisplayed := 800
  (4 : ℚ) / 9 * A = 800

theorem gallery_pieces_total (A : ℕ) (h : (TotalArtGalleryPieces A)) : A = 1800 :=
by sorry

end NUMINAMATH_GPT_gallery_pieces_total_l270_27098


namespace NUMINAMATH_GPT_smallest_number_l270_27033

-- Define the conditions
def is_divisible_by (n d : ℕ) : Prop := d ∣ n

def conditions (n : ℕ) : Prop := 
  (n > 12) ∧ 
  is_divisible_by (n - 12) 12 ∧ 
  is_divisible_by (n - 12) 24 ∧
  is_divisible_by (n - 12) 36 ∧
  is_divisible_by (n - 12) 48 ∧
  is_divisible_by (n - 12) 56

-- State the theorem
theorem smallest_number : ∃ n : ℕ, conditions n ∧ n = 1020 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_l270_27033


namespace NUMINAMATH_GPT_find_abc_l270_27085

theorem find_abc (a b c : ℕ) (h1 : c = b^2) (h2 : (a + b + c)^3 - a^3 - b^3 - c^3 = 210) : a + b + c = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_abc_l270_27085


namespace NUMINAMATH_GPT_evaluate_F_2_f_3_l270_27057

def f (a : ℕ) : ℕ := a^2 - 2*a
def F (a b : ℕ) : ℕ := b^2 + a*b

theorem evaluate_F_2_f_3 : F 2 (f 3) = 15 := by
  sorry

end NUMINAMATH_GPT_evaluate_F_2_f_3_l270_27057


namespace NUMINAMATH_GPT_initial_eggs_proof_l270_27070

-- Definitions based on the conditions provided
def initial_eggs := 7
def added_eggs := 4
def total_eggs := 11

-- The statement to be proved
theorem initial_eggs_proof : initial_eggs + added_eggs = total_eggs :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_initial_eggs_proof_l270_27070


namespace NUMINAMATH_GPT_greatest_identical_snack_bags_l270_27056

-- Defining the quantities of each type of snack
def granola_bars : Nat := 24
def dried_fruit : Nat := 36
def nuts : Nat := 60

-- Statement of the problem: greatest number of identical snack bags Serena can make without any food left over.
theorem greatest_identical_snack_bags :
  Nat.gcd (Nat.gcd granola_bars dried_fruit) nuts = 12 :=
sorry

end NUMINAMATH_GPT_greatest_identical_snack_bags_l270_27056


namespace NUMINAMATH_GPT_largest_number_is_sqrt_7_l270_27082

noncomputable def largest_root (d e f : ℝ) : ℝ :=
if d ≥ e ∧ d ≥ f then d else if e ≥ d ∧ e ≥ f then e else f

theorem largest_number_is_sqrt_7 :
  ∃ (d e f : ℝ), (d + e + f = 3) ∧ (d * e + d * f + e * f = -14) ∧ (d * e * f = 21) ∧ (largest_root d e f = Real.sqrt 7) :=
sorry

end NUMINAMATH_GPT_largest_number_is_sqrt_7_l270_27082


namespace NUMINAMATH_GPT_ratio_lcm_gcf_280_476_l270_27071

theorem ratio_lcm_gcf_280_476 : 
  let a := 280
  let b := 476
  let lcm_ab := Nat.lcm a b
  let gcf_ab := Nat.gcd a b
  lcm_ab / gcf_ab = 170 := by
  sorry

end NUMINAMATH_GPT_ratio_lcm_gcf_280_476_l270_27071


namespace NUMINAMATH_GPT_symmetric_point_l270_27032

-- Define the given conditions
def pointP : (ℤ × ℤ) := (3, -2)
def symmetry_line (y : ℤ) := (y = 1)

-- Prove the assertion that point Q is (3, 4)
theorem symmetric_point (x y1 y2 : ℤ) (hx: x = 3) (hy1: y1 = -2) (hy : symmetry_line 1) :
  (x, 2 * 1 - y1) = (3, 4) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_l270_27032


namespace NUMINAMATH_GPT_ab_product_l270_27039

theorem ab_product (a b : ℝ) (h_sol : ∀ x, -1 < x ∧ x < 4 → x^2 + a * x + b < 0) 
  (h_roots : ∀ x, x^2 + a * x + b = 0 ↔ x = -1 ∨ x = 4) : 
  a * b = 12 :=
sorry

end NUMINAMATH_GPT_ab_product_l270_27039


namespace NUMINAMATH_GPT_find_dividend_l270_27078

-- Definitions based on conditions from the problem
def divisor : ℕ := 13
def quotient : ℕ := 17
def remainder : ℕ := 1

-- Statement of the proof problem
theorem find_dividend (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

-- Proof statement ensuring dividend is as expected
example : find_dividend divisor quotient remainder = 222 :=
by 
  sorry

end NUMINAMATH_GPT_find_dividend_l270_27078


namespace NUMINAMATH_GPT_pythagorean_inequality_l270_27001

variables (a b c : ℝ) (n : ℕ)

theorem pythagorean_inequality (h₀ : a > b) (h₁ : b > c) (h₂ : a^2 = b^2 + c^2) (h₃ : n > 2) : a^n > b^n + c^n :=
sorry

end NUMINAMATH_GPT_pythagorean_inequality_l270_27001


namespace NUMINAMATH_GPT_inequality_one_inequality_two_l270_27002

variable {a b r s : ℝ}

theorem inequality_one (h_a : 0 < a) (h_b : 0 < b) :
  a^2 * b ≤ 4 * ((a + b) / 3)^3 :=
sorry

theorem inequality_two (h_a : 0 < a) (h_b : 0 < b) (h_r : 0 < r) (h_s : 0 < s) 
  (h_eq : 1 / r + 1 / s = 1) : 
  (a^r / r) + (b^s / s) ≥ a * b :=
sorry

end NUMINAMATH_GPT_inequality_one_inequality_two_l270_27002


namespace NUMINAMATH_GPT_original_salary_l270_27094

theorem original_salary (S : ℝ) (h : 1.10 * S * 0.95 = 3135) : S = 3000 := 
by 
  sorry

end NUMINAMATH_GPT_original_salary_l270_27094


namespace NUMINAMATH_GPT_monthly_income_of_P_l270_27046

theorem monthly_income_of_P (P Q R : ℝ) 
    (h1 : (P + Q) / 2 = 2050) 
    (h2 : (Q + R) / 2 = 5250) 
    (h3 : (P + R) / 2 = 6200) : 
    P = 3000 :=
by
  sorry

end NUMINAMATH_GPT_monthly_income_of_P_l270_27046


namespace NUMINAMATH_GPT_inscribed_circle_circumference_l270_27009

theorem inscribed_circle_circumference (side_length : ℝ) (h : side_length = 10) : 
  ∃ C : ℝ, C = 2 * Real.pi * (side_length / 2) ∧ C = 10 * Real.pi := 
by 
  sorry

end NUMINAMATH_GPT_inscribed_circle_circumference_l270_27009


namespace NUMINAMATH_GPT_negation_equiv_l270_27037

-- Given problem conditions
def exists_real_x_lt_0 : Prop := ∃ x : ℝ, x^2 + 1 < 0

-- Mathematically equivalent proof problem statement
theorem negation_equiv :
  ¬exists_real_x_lt_0 ↔ ∀ x : ℝ, x^2 + 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_equiv_l270_27037


namespace NUMINAMATH_GPT_son_l270_27083

theorem son's_age (S M : ℕ) (h1 : M = S + 20) (h2 : M + 2 = 2 * (S + 2)) : S = 18 := by
  sorry

end NUMINAMATH_GPT_son_l270_27083


namespace NUMINAMATH_GPT_smallest_integer_to_perfect_cube_l270_27004

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem smallest_integer_to_perfect_cube :
  ∃ n : ℕ, 
    n > 0 ∧ 
    is_perfect_cube (45216 * n) ∧ 
    (∀ m : ℕ, m > 0 ∧ is_perfect_cube (45216 * m) → n ≤ m) ∧ 
    n = 7 := sorry

end NUMINAMATH_GPT_smallest_integer_to_perfect_cube_l270_27004


namespace NUMINAMATH_GPT_evaluate_expression_l270_27018

theorem evaluate_expression :
  (3^1003 + 7^1004)^2 - (3^1003 - 7^1004)^2 = 5.292 * 10^1003 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l270_27018


namespace NUMINAMATH_GPT_thirteen_members_divisible_by_13_l270_27011

theorem thirteen_members_divisible_by_13 (B : ℕ) (hB : B < 10) : 
  (∃ B, (2000 + B * 100 + 34) % 13 = 0) ↔ B = 6 :=
by
  sorry

end NUMINAMATH_GPT_thirteen_members_divisible_by_13_l270_27011


namespace NUMINAMATH_GPT_fraction_increase_by_50_percent_l270_27080

variable (x y : ℝ)
variable (h1 : 0 < y)

theorem fraction_increase_by_50_percent (h2 : 0.6 * x / 0.4 * y = 1.5 * x / y) : 
  1.5 * (x / y) = 1.5 * (x / y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_increase_by_50_percent_l270_27080


namespace NUMINAMATH_GPT_find_G16_l270_27086

variable (G : ℝ → ℝ)

def condition1 : Prop := G 8 = 28

def condition2 : Prop := ∀ x : ℝ, 
  (x^2 + 8*x + 16) ≠ 0 → 
  (G (4*x) / G (x + 4) = 16 - (64*x + 80) / (x^2 + 8*x + 16))

theorem find_G16 (h1 : condition1 G) (h2 : condition2 G) : G 16 = 120 :=
sorry

end NUMINAMATH_GPT_find_G16_l270_27086


namespace NUMINAMATH_GPT_friends_recycled_pounds_l270_27043

-- Definitions for the given conditions
def pounds_per_point : ℕ := 4
def paige_recycled : ℕ := 14
def total_points : ℕ := 4

-- The proof statement
theorem friends_recycled_pounds :
  ∃ p_friends : ℕ, 
  (paige_recycled / pounds_per_point) + (p_friends / pounds_per_point) = total_points 
  → p_friends = 4 := 
sorry

end NUMINAMATH_GPT_friends_recycled_pounds_l270_27043


namespace NUMINAMATH_GPT_isabella_hair_length_l270_27075

theorem isabella_hair_length (h : ℕ) (g : ℕ) (future_length : ℕ) (hg : g = 4) (future_length_eq : future_length = 22) :
  h = future_length - g :=
by
  rw [future_length_eq, hg]
  exact sorry

end NUMINAMATH_GPT_isabella_hair_length_l270_27075


namespace NUMINAMATH_GPT_smallest_prime_with_digit_sum_25_l270_27022

-- Definitions used in Lean statement:
-- 1. Prime predicate based on primality check.
-- 2. Digit sum function.

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Lean 4 statement to prove that the smallest prime whose digits sum to 25 is 1699.

theorem smallest_prime_with_digit_sum_25 : ∃ n : ℕ, is_prime n ∧ digit_sum n = 25 ∧ n = 1699 :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_with_digit_sum_25_l270_27022


namespace NUMINAMATH_GPT_pow_evaluation_l270_27036

theorem pow_evaluation (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end NUMINAMATH_GPT_pow_evaluation_l270_27036


namespace NUMINAMATH_GPT_ellipse_hyperbola_tangent_m_eq_l270_27073

variable (x y m : ℝ)

def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 2)^2 = 1
def curves_tangent (x m : ℝ) : Prop := ∃ y, ellipse x y ∧ hyperbola x y m

theorem ellipse_hyperbola_tangent_m_eq :
  (∃ x, curves_tangent x (12/13)) ↔ true := 
by
  sorry

end NUMINAMATH_GPT_ellipse_hyperbola_tangent_m_eq_l270_27073


namespace NUMINAMATH_GPT_business_value_l270_27028

theorem business_value (h₁ : (2/3 : ℝ) * (3/4 : ℝ) * V = 30000) : V = 60000 :=
by
  -- conditions and definitions go here
  sorry

end NUMINAMATH_GPT_business_value_l270_27028


namespace NUMINAMATH_GPT_remaining_episodes_l270_27010

theorem remaining_episodes (total_seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (H1 : total_seasons = 12) (H2 : episodes_per_season = 20) (H3 : fraction_watched = 1/3) : 
  (total_seasons * episodes_per_season) - (fraction_watched * (total_seasons * episodes_per_season)) = 160 :=
by
  sorry

end NUMINAMATH_GPT_remaining_episodes_l270_27010


namespace NUMINAMATH_GPT_largest_divisor_of_expression_l270_27050

theorem largest_divisor_of_expression :
  ∃ k : ℕ, (∀ m : ℕ, (m > k → m ∣ (1991 ^ k * 1990 ^ (1991 ^ 1992) + 1992 ^ (1991 ^ 1990)) = false))
  ∧ k = 1991 := by
sorry

end NUMINAMATH_GPT_largest_divisor_of_expression_l270_27050


namespace NUMINAMATH_GPT_ratio_of_b_l270_27020

theorem ratio_of_b (a b k a1 a2 b1 b2 : ℝ) (h_nonzero_a2 : a2 ≠ 0) (h_nonzero_b12: b1 ≠ 0 ∧ b2 ≠ 0) :
  (a * b = k) →
  (a1 * b1 = a2 * b2) →
  (a1 / a2 = 3 / 5) →
  (b1 / b2 = 5 / 3) := 
sorry

end NUMINAMATH_GPT_ratio_of_b_l270_27020


namespace NUMINAMATH_GPT_total_spider_legs_l270_27013

variable (numSpiders : ℕ)
variable (legsPerSpider : ℕ)
axiom h1 : numSpiders = 5
axiom h2 : legsPerSpider = 8

theorem total_spider_legs : numSpiders * legsPerSpider = 40 :=
by
  -- necessary for build without proof.
  sorry

end NUMINAMATH_GPT_total_spider_legs_l270_27013


namespace NUMINAMATH_GPT_income_exceeds_previous_l270_27077

noncomputable def a_n (a b : ℝ) (n : ℕ) : ℝ :=
if n = 1 then a
else a * (2 / 3)^(n - 1) + b * (3 / 2)^(n - 2)

theorem income_exceeds_previous (a b : ℝ) (h : b ≥ 3 * a / 8) (n : ℕ) (hn : n ≥ 2) : 
  a_n a b n ≥ a :=
sorry

end NUMINAMATH_GPT_income_exceeds_previous_l270_27077


namespace NUMINAMATH_GPT_fg_value_l270_27074

def g (x : ℤ) : ℤ := 4 * x - 3
def f (x : ℤ) : ℤ := 6 * x + 2

theorem fg_value : f (g 5) = 104 := by
  sorry

end NUMINAMATH_GPT_fg_value_l270_27074


namespace NUMINAMATH_GPT_find_g_function_l270_27041

noncomputable def g : ℝ → ℝ :=
  sorry

theorem find_g_function (x y : ℝ) (h1 : g 1 = 2) (h2 : ∀ (x y : ℝ), g (x + y) = 5^y * g x + 3^x * g y) :
  g x = 5^x - 3^x :=
by
  sorry

end NUMINAMATH_GPT_find_g_function_l270_27041


namespace NUMINAMATH_GPT_boys_attended_dance_l270_27062

variables (d b : ℕ)

theorem boys_attended_dance (h1 : b = 2 * d) (h2 : b = (d - 1) + 8) : b = 14 :=
by sorry

end NUMINAMATH_GPT_boys_attended_dance_l270_27062


namespace NUMINAMATH_GPT_choir_row_lengths_l270_27031

theorem choir_row_lengths (x : ℕ) : 
  ((x ∈ [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) ∧ (90 % x = 0)) → (x = 5 ∨ x = 6 ∨ x = 9 ∨ x = 10 ∨ x = 15) :=
by
  intro h
  cases h
  sorry

end NUMINAMATH_GPT_choir_row_lengths_l270_27031


namespace NUMINAMATH_GPT_ravish_maximum_marks_l270_27023

theorem ravish_maximum_marks (M : ℝ) (h_pass : 0.40 * M = 80) : M = 200 :=
sorry

end NUMINAMATH_GPT_ravish_maximum_marks_l270_27023


namespace NUMINAMATH_GPT_num_divisors_of_factorial_9_multiple_3_l270_27059

-- Define the prime factorization of 9!
def factorial_9 := 2^7 * 3^4 * 5 * 7

-- Define the conditions for the exponents a, b, c, d
def valid_exponents (a b c d : ℕ) : Prop :=
  (0 ≤ a ∧ a ≤ 7) ∧ (1 ≤ b ∧ b ≤ 4) ∧ (0 ≤ c ∧ c ≤ 1) ∧ (0 ≤ d ∧ d ≤ 1)

-- Define the number of valid exponent combinations
def num_valid_combinations : ℕ :=
  8 * 4 * 2 * 2

-- Theorem stating that the number of divisors of 9! that are multiples of 3 is 128
theorem num_divisors_of_factorial_9_multiple_3 : num_valid_combinations = 128 := by
  sorry

end NUMINAMATH_GPT_num_divisors_of_factorial_9_multiple_3_l270_27059


namespace NUMINAMATH_GPT_second_concert_attendance_correct_l270_27030

def first_concert_attendance : ℕ := 65899
def additional_people : ℕ := 119
def second_concert_attendance : ℕ := 66018

theorem second_concert_attendance_correct :
  first_concert_attendance + additional_people = second_concert_attendance :=
by sorry

end NUMINAMATH_GPT_second_concert_attendance_correct_l270_27030


namespace NUMINAMATH_GPT_profit_calculation_l270_27055

-- Definitions from conditions
def initial_shares := 20
def cost_per_share := 3
def sold_shares := 10
def sale_price_per_share := 4
def remaining_shares_value_multiplier := 2

-- Calculations based on conditions
def initial_cost := initial_shares * cost_per_share
def revenue_from_sold_shares := sold_shares * sale_price_per_share
def remaining_shares := initial_shares - sold_shares
def value_of_remaining_shares := remaining_shares * (cost_per_share * remaining_shares_value_multiplier)
def total_value := revenue_from_sold_shares + value_of_remaining_shares
def expected_profit := total_value - initial_cost

-- The problem statement to be proven
theorem profit_calculation : expected_profit = 40 := by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_profit_calculation_l270_27055


namespace NUMINAMATH_GPT_Lizzie_group_difference_l270_27084

theorem Lizzie_group_difference
  (lizzie_group_members : ℕ)
  (total_members : ℕ)
  (lizzie_more_than_other : lizzie_group_members > total_members - lizzie_group_members)
  (lizzie_members_eq : lizzie_group_members = 54)
  (total_members_eq : total_members = 91)
  : lizzie_group_members - (total_members - lizzie_group_members) = 17 := 
sorry

end NUMINAMATH_GPT_Lizzie_group_difference_l270_27084


namespace NUMINAMATH_GPT_calculate_product_l270_27093

theorem calculate_product : 6^6 * 3^6 = 34012224 := by
  sorry

end NUMINAMATH_GPT_calculate_product_l270_27093


namespace NUMINAMATH_GPT_cows_count_24_l270_27081

-- Declare the conditions as given in the problem.
variables (D C : Nat)

-- Define the total number of legs and heads and the given condition.
def total_legs := 2 * D + 4 * C
def total_heads := D + C
axiom condition : total_legs = 2 * total_heads + 48

-- The goal is to prove that the number of cows C is 24.
theorem cows_count_24 : C = 24 :=
by
  sorry

end NUMINAMATH_GPT_cows_count_24_l270_27081


namespace NUMINAMATH_GPT_pair_not_equal_to_64_l270_27045

theorem pair_not_equal_to_64 :
  ¬(4 * (9 / 2) = 64) := by
  sorry

end NUMINAMATH_GPT_pair_not_equal_to_64_l270_27045


namespace NUMINAMATH_GPT_bicycle_cost_l270_27051

theorem bicycle_cost (CP_A SP_B SP_C : ℝ) (h1 : SP_B = CP_A * 1.20) (h2 : SP_C = SP_B * 1.25) (h3 : SP_C = 225) : CP_A = 150 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_cost_l270_27051


namespace NUMINAMATH_GPT_blackboard_problem_l270_27042

theorem blackboard_problem (n : ℕ) (h_pos : 0 < n) :
  ∃ x, (∀ (t : ℕ), t < n - 1 → ∃ a b : ℕ, a + b + 2 * (t + 1) = n + 1 ∧ a > 0 ∧ b > 0) → 
  x ≥ 2 ^ ((4 * n ^ 2 - 4) / 3) :=
by
  sorry

end NUMINAMATH_GPT_blackboard_problem_l270_27042


namespace NUMINAMATH_GPT_collapsed_buildings_l270_27047

theorem collapsed_buildings (initial_collapse : ℕ) (collapse_one : initial_collapse = 4)
                            (collapse_double : ∀ n m, m = 2 * n) : (4 + 8 + 16 + 32 = 60) :=
by
  sorry

end NUMINAMATH_GPT_collapsed_buildings_l270_27047


namespace NUMINAMATH_GPT_find_radius_of_inscribed_sphere_l270_27067

variables (a b c s : ℝ)

theorem find_radius_of_inscribed_sphere
  (h1 : a + b + c = 18)
  (h2 : 2 * (a * b + b * c + c * a) = 216)
  (h3 : a^2 + b^2 + c^2 = 108) :
  s = 3 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_radius_of_inscribed_sphere_l270_27067


namespace NUMINAMATH_GPT_inequality_AM_GM_HM_l270_27021

theorem inequality_AM_GM_HM (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (hab : a ≠ b) : 
  (a + b) / 2 > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > 2 * (a * b) / (a + b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_AM_GM_HM_l270_27021


namespace NUMINAMATH_GPT_tile_chessboard_2n_l270_27063

theorem tile_chessboard_2n (n : ℕ) (board : Fin (2^n) → Fin (2^n) → Prop) (i j : Fin (2^n)) 
  (h : board i j = false) : ∃ tile : Fin (2^n) → Fin (2^n) → Bool, 
  (∀ i j, board i j = true ↔ tile i j = true) :=
sorry

end NUMINAMATH_GPT_tile_chessboard_2n_l270_27063


namespace NUMINAMATH_GPT_correct_option_e_l270_27048

theorem correct_option_e : 15618 = 1 + 5^6 - 1 * 8 :=
by sorry

end NUMINAMATH_GPT_correct_option_e_l270_27048


namespace NUMINAMATH_GPT_new_number_formed_l270_27016

variable (a b : ℕ)

theorem new_number_formed (ha : a < 10) (hb : b < 10) : 
  ((10 * a + b) * 10 + 2) = 100 * a + 10 * b + 2 := 
by
  sorry

end NUMINAMATH_GPT_new_number_formed_l270_27016


namespace NUMINAMATH_GPT_inequality_proof_l270_27024

theorem inequality_proof (a : ℝ) : 
  2 * a^4 + 2 * a^2 - 1 ≥ (3 / 2) * (a^2 + a - 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l270_27024


namespace NUMINAMATH_GPT_unit_digit_2_pow_15_l270_27035

theorem unit_digit_2_pow_15 : (2^15) % 10 = 8 := by
  sorry

end NUMINAMATH_GPT_unit_digit_2_pow_15_l270_27035


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_l270_27092

theorem arithmetic_sequence_general_formula :
  (∀ n:ℕ, ∃ (a_n : ℕ), ∀ k:ℕ, a_n = 2 * k → k = n)
  ∧ ( 2 * n + 2 * (n + 2) = 8 → 2 * n + 2 * (n + 3) = 12 → a_n = 2 * n )
  ∧ (S_n = (n * (n + 1)) / 2 → S_n = 420 → n = 20) :=
by { sorry }

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_l270_27092


namespace NUMINAMATH_GPT_correct_option_is_B_l270_27015

noncomputable def smallest_absolute_value := 0

theorem correct_option_is_B :
  (∀ x : ℝ, |x| ≥ 0) ∧ |(0 : ℝ)| = 0 :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_B_l270_27015


namespace NUMINAMATH_GPT_segment_length_aa_prime_l270_27027

/-- Given points A, B, and C, and their reflections, show that the length of AA' is 8 -/
theorem segment_length_aa_prime
  (A : ℝ × ℝ) (A_reflected : ℝ × ℝ)
  (x₁ y₁ y₁_neg : ℝ) :
  A = (x₁, y₁) →
  A_reflected = (x₁, y₁_neg) →
  y₁_neg = -y₁ →
  y₁ = 4 →
  x₁ = 2 →
  |y₁ - y₁_neg| = 8 :=
sorry

end NUMINAMATH_GPT_segment_length_aa_prime_l270_27027


namespace NUMINAMATH_GPT_count_lineups_not_last_l270_27052

theorem count_lineups_not_last (n : ℕ) (htallest_not_last : n = 5) :
  ∃ (k : ℕ), k = 96 :=
by { sorry }

end NUMINAMATH_GPT_count_lineups_not_last_l270_27052


namespace NUMINAMATH_GPT_solve_quadratic_eq_l270_27058

theorem solve_quadratic_eq (x y : ℝ) :
  (x = 3 ∧ y = 1) ∨ (x = -1 ∧ y = 1) ∨ (x = 3 ∧ y = -1) ∨ (x = -3 ∧ y = -1) ∨ (x = -1 ∧ y = -5) ↔
  x ^ 2 - x * y + y ^ 2 - x + 3 * y - 7 = 0 := sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l270_27058


namespace NUMINAMATH_GPT_number_of_band_students_l270_27005

noncomputable def total_students := 320
noncomputable def sports_students := 200
noncomputable def both_activities_students := 60
noncomputable def either_activity_students := 225

theorem number_of_band_students : 
  ∃ B : ℕ, either_activity_students = B + sports_students - both_activities_students ∧ B = 85 :=
by
  sorry

end NUMINAMATH_GPT_number_of_band_students_l270_27005


namespace NUMINAMATH_GPT_page_shoes_l270_27019

/-- Page's initial collection of shoes -/
def initial_collection : ℕ := 80

/-- Page donates 30% of her collection -/
def donation (n : ℕ) : ℕ := n * 30 / 100

/-- Page buys additional shoes -/
def additional_shoes : ℕ := 6

/-- Page's final collection after donation and purchase -/
def final_collection (n : ℕ) : ℕ := (n - donation n) + additional_shoes

/-- Proof that the final collection of shoes is 62 given the initial collection of 80 pairs -/
theorem page_shoes : (final_collection initial_collection) = 62 := 
by sorry

end NUMINAMATH_GPT_page_shoes_l270_27019


namespace NUMINAMATH_GPT_S6_values_l270_27040

noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def S (n : ℕ) : ℝ := sorry

axiom geo_seq (q : ℝ) :
  ∀ n : ℕ, a n = a 0 * q ^ n

variable (a3_eq_4 : a 2 = 4) 
variable (S3_eq_7 : S 3 = 7)

theorem S6_values : S 6 = 63 ∨ S 6 = 133 / 27 := sorry

end NUMINAMATH_GPT_S6_values_l270_27040


namespace NUMINAMATH_GPT_calculate_price_per_pound_of_meat_l270_27091

noncomputable def price_per_pound_of_meat : ℝ :=
  let total_hours := 50
  let w := 8
  let m_pounds := 20
  let fv_pounds := 15
  let fv_pp := 4
  let b_pounds := 60
  let b_pp := 1.5
  let j_wage := 10
  let j_hours := 10
  let j_rate := 1.5

  -- known costs
  let fv_cost := fv_pounds * fv_pp
  let b_cost := b_pounds * b_pp
  let j_cost := j_hours * j_wage * j_rate

  -- total costs
  let total_cost := total_hours * w
  let known_costs := fv_cost + b_cost + j_cost

  (total_cost - known_costs) / m_pounds

theorem calculate_price_per_pound_of_meat : price_per_pound_of_meat = 5 := by
  sorry

end NUMINAMATH_GPT_calculate_price_per_pound_of_meat_l270_27091


namespace NUMINAMATH_GPT_candies_on_second_day_l270_27008

noncomputable def total_candies := 45
noncomputable def days := 5
noncomputable def difference := 3

def arithmetic_sum (n : ℕ) (a₁ d : ℕ) :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

theorem candies_on_second_day (a : ℕ) (h : arithmetic_sum days a difference = total_candies) :
  a + difference = 6 := by
  sorry

end NUMINAMATH_GPT_candies_on_second_day_l270_27008


namespace NUMINAMATH_GPT_zero_function_l270_27053

noncomputable def f : ℝ → ℝ := sorry

theorem zero_function :
  (∀ x y : ℝ, f x + f y = f (f x * f y)) → (∀ x : ℝ, f x = 0) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_zero_function_l270_27053


namespace NUMINAMATH_GPT_ratio_jacob_edward_l270_27072

-- Definitions and conditions
def brian_shoes : ℕ := 22
def edward_shoes : ℕ := 3 * brian_shoes
def total_shoes : ℕ := 121
def jacob_shoes : ℕ := total_shoes - brian_shoes - edward_shoes

-- Statement of the problem
theorem ratio_jacob_edward (h_brian : brian_shoes = 22)
                          (h_edward : edward_shoes = 3 * brian_shoes)
                          (h_total : total_shoes = 121)
                          (h_jacob : jacob_shoes = total_shoes - brian_shoes - edward_shoes) :
                          jacob_shoes / edward_shoes = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_ratio_jacob_edward_l270_27072


namespace NUMINAMATH_GPT_total_weight_correct_weight_difference_correct_l270_27099

variables (baskets_of_apples baskets_of_pears : ℕ) (kg_per_basket_of_apples kg_per_basket_of_pears : ℕ)

def total_weight_apples_ppears (baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears : ℕ) : ℕ :=
  (baskets_of_apples * kg_per_basket_of_apples) + (baskets_of_pears * kg_per_basket_of_pears)

def weight_difference_pears_apples (baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears : ℕ) : ℕ :=
  (baskets_of_pears * kg_per_basket_of_pears) - (baskets_of_apples * kg_per_basket_of_apples)

theorem total_weight_correct (h_apples: baskets_of_apples = 120) (h_pears: baskets_of_pears = 130) (h_kg_apples: kg_per_basket_of_apples = 40) (h_kg_pears: kg_per_basket_of_pears = 50) : 
  total_weight_apples_ppears baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears = 11300 :=
by
  rw [h_apples, h_pears, h_kg_apples, h_kg_pears]
  sorry

theorem weight_difference_correct (h_apples: baskets_of_apples = 120) (h_pears: baskets_of_pears = 130) (h_kg_apples: kg_per_basket_of_apples = 40) (h_kg_pears: kg_per_basket_of_pears = 50) : 
  weight_difference_pears_apples baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears = 1700 :=
by
  rw [h_apples, h_pears, h_kg_apples, h_kg_pears]
  sorry

end NUMINAMATH_GPT_total_weight_correct_weight_difference_correct_l270_27099


namespace NUMINAMATH_GPT_students_with_one_talent_l270_27029

-- Define the given conditions
def total_students := 120
def cannot_sing := 30
def cannot_dance := 50
def both_skills := 10

-- Define the problem statement
theorem students_with_one_talent :
  (total_students - cannot_sing - both_skills) + (total_students - cannot_dance - both_skills) = 130 :=
by
  sorry

end NUMINAMATH_GPT_students_with_one_talent_l270_27029


namespace NUMINAMATH_GPT_paving_cost_correct_l270_27060

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_m : ℝ := 300
def area (length : ℝ) (width : ℝ) : ℝ := length * width
def cost (area : ℝ) (rate : ℝ) : ℝ := area * rate

theorem paving_cost_correct :
  cost (area length width) rate_per_sq_m = 6187.50 :=
by
  sorry

end NUMINAMATH_GPT_paving_cost_correct_l270_27060


namespace NUMINAMATH_GPT_trains_crossing_time_l270_27003

/-- Define the length of the first train in meters -/
def length_train1 : ℚ := 200

/-- Define the length of the second train in meters -/
def length_train2 : ℚ := 150

/-- Define the speed of the first train in kilometers per hour -/
def speed_train1_kmph : ℚ := 40

/-- Define the speed of the second train in kilometers per hour -/
def speed_train2_kmph : ℚ := 46

/-- Define conversion factor from kilometers per hour to meters per second -/
def kmph_to_mps : ℚ := 1000 / 3600

/-- Calculate the relative speed in meters per second assuming both trains are moving in the same direction -/
def relative_speed_mps : ℚ := (speed_train2_kmph - speed_train1_kmph) * kmph_to_mps

/-- Calculate the combined length of both trains in meters -/
def combined_length : ℚ := length_train1 + length_train2

/-- Prove the time in seconds for the two trains to cross each other when moving in the same direction is 210 seconds -/
theorem trains_crossing_time :
  (combined_length / relative_speed_mps) = 210 := by
  sorry

end NUMINAMATH_GPT_trains_crossing_time_l270_27003


namespace NUMINAMATH_GPT_school_club_profit_l270_27087

theorem school_club_profit : 
  let purchase_price_per_bar := 3 / 4
  let selling_price_per_bar := 2 / 3
  let total_bars := 1200
  let bars_with_discount := total_bars - 1000
  let discount_per_bar := 0.10
  let total_cost := total_bars * purchase_price_per_bar
  let total_revenue_without_discount := total_bars * selling_price_per_bar
  let total_discount := bars_with_discount * discount_per_bar
  let adjusted_revenue := total_revenue_without_discount - total_discount
  let profit := adjusted_revenue - total_cost
  profit = -116 :=
by sorry

end NUMINAMATH_GPT_school_club_profit_l270_27087


namespace NUMINAMATH_GPT_speed_including_stoppages_l270_27066

theorem speed_including_stoppages : 
  ∀ (speed_excluding_stoppages : ℝ) (stoppage_minutes_per_hour : ℝ), 
  speed_excluding_stoppages = 65 → 
  stoppage_minutes_per_hour = 15.69 → 
  (speed_excluding_stoppages * (1 - stoppage_minutes_per_hour / 60)) = 47.9025 := 
by intros speed_excluding_stoppages stoppage_minutes_per_hour h1 h2
   sorry

end NUMINAMATH_GPT_speed_including_stoppages_l270_27066


namespace NUMINAMATH_GPT_find_x_minus_y_l270_27061

variables (x y : ℚ)

theorem find_x_minus_y
  (h1 : 3 * x - 4 * y = 17)
  (h2 : x + 3 * y = 1) :
  x - y = 69 / 13 := 
sorry

end NUMINAMATH_GPT_find_x_minus_y_l270_27061


namespace NUMINAMATH_GPT_matching_polygons_pairs_l270_27096

noncomputable def are_matching_pairs (n m : ℕ) : Prop :=
  2 * ((n - 2) * 180 / n) = 3 * (360 / m)

theorem matching_polygons_pairs (n m : ℕ) :
  are_matching_pairs n m → (n, m) = (3, 9) ∨ (n, m) = (4, 6) ∨ (n, m) = (5, 5) ∨ (n, m) = (8, 4) :=
sorry

end NUMINAMATH_GPT_matching_polygons_pairs_l270_27096


namespace NUMINAMATH_GPT_solve_arctan_eq_pi_over_3_l270_27026

open Real

theorem solve_arctan_eq_pi_over_3 (x : ℝ) :
  arctan (1 / x) + arctan (1 / x^2) = π / 3 ↔ 
  x = (1 + sqrt (13 + 4 * sqrt 3)) / (2 * sqrt 3) ∨
  x = (1 - sqrt (13 + 4 * sqrt 3)) / (2 * sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_arctan_eq_pi_over_3_l270_27026


namespace NUMINAMATH_GPT_last_digit_of_power_of_two_l270_27089

theorem last_digit_of_power_of_two (n : ℕ) (h : n ≥ 2) : (2 ^ (2 ^ n) + 1) % 10 = 7 :=
sorry

end NUMINAMATH_GPT_last_digit_of_power_of_two_l270_27089


namespace NUMINAMATH_GPT_line_intersects_ellipse_slopes_l270_27076

theorem line_intersects_ellipse_slopes :
  {m : ℝ | ∃ x, 4 * x^2 + 25 * (m * x + 8)^2 = 100} = 
  {m : ℝ | m ≤ -Real.sqrt 2.4 ∨ Real.sqrt 2.4 ≤ m} := 
by
  sorry

end NUMINAMATH_GPT_line_intersects_ellipse_slopes_l270_27076


namespace NUMINAMATH_GPT_graph_of_equation_is_two_lines_l270_27006

-- define the condition
def equation_condition (x y : ℝ) : Prop :=
  (x - y) ^ 2 = x ^ 2 + y ^ 2

-- state the theorem
theorem graph_of_equation_is_two_lines :
  ∀ x y : ℝ, equation_condition x y → (x = 0) ∨ (y = 0) :=
by
  intros x y h
  -- proof here
  sorry

end NUMINAMATH_GPT_graph_of_equation_is_two_lines_l270_27006


namespace NUMINAMATH_GPT_simplify_expression_l270_27038

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : ( (3 * x + 6 - 5 * x) / 3 ) = ( (-2 * x) / 3 + 2 ) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l270_27038


namespace NUMINAMATH_GPT_min_value_inverse_sum_l270_27068

theorem min_value_inverse_sum {m n : ℝ} (h1 : -2 * m - 2 * n + 1 = 0) (h2 : m * n > 0) : 
  (1 / m + 1 / n) ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_value_inverse_sum_l270_27068


namespace NUMINAMATH_GPT_problem1_problem2_l270_27012

theorem problem1 (x y : ℝ) (h1 : x - y = 4) (h2 : x > 3) (h3 : y < 1) : 
  2 < x + y ∧ x + y < 6 :=
sorry

theorem problem2 (x y m : ℝ) (h1 : y > 1) (h2 : x < -1) (h3 : x - y = m) : 
  m + 2 < x + y ∧ x + y < -m - 2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l270_27012


namespace NUMINAMATH_GPT_total_cards_proof_l270_27069

-- Define the standard size of a deck of playing cards
def standard_deck_size : Nat := 52

-- Define the number of complete decks the shopkeeper has
def complete_decks : Nat := 6

-- Define the number of additional cards the shopkeeper has
def additional_cards : Nat := 7

-- Define the total number of cards from the complete decks
def total_deck_cards : Nat := complete_decks * standard_deck_size

-- Define the total number of all cards the shopkeeper has
def total_cards : Nat := total_deck_cards + additional_cards

-- The theorem statement that we need to prove
theorem total_cards_proof : total_cards = 319 := by
  sorry

end NUMINAMATH_GPT_total_cards_proof_l270_27069


namespace NUMINAMATH_GPT_complete_the_square_1_complete_the_square_2_complete_the_square_3_l270_27064

theorem complete_the_square_1 (x : ℝ) : 
  (x^2 - 2 * x + 3) = (x - 1)^2 + 2 :=
sorry

theorem complete_the_square_2 (x : ℝ) : 
  (3 * x^2 + 6 * x - 1) = 3 * (x + 1)^2 - 4 :=
sorry

theorem complete_the_square_3 (x : ℝ) : 
  (-2 * x^2 + 3 * x - 2) = -2 * (x - 3 / 4)^2 - 7 / 8 :=
sorry

end NUMINAMATH_GPT_complete_the_square_1_complete_the_square_2_complete_the_square_3_l270_27064


namespace NUMINAMATH_GPT_find_number_l270_27034

noncomputable def some_number : ℝ :=
  0.27712 / 9.237333333333334

theorem find_number :
  (69.28 * 0.004) / some_number = 9.237333333333334 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l270_27034


namespace NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_l270_27079

theorem sum_of_squares_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x^2 + (x + 1)^2 = 1625 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_l270_27079


namespace NUMINAMATH_GPT_range_of_m_l270_27044

def f (x : ℝ) := |x - 3|
def g (x : ℝ) (m : ℝ) := -|x - 7| + m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x ≥ g x m) → m < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l270_27044
