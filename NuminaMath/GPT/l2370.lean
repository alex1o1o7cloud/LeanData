import Mathlib

namespace NUMINAMATH_GPT_problem_1_l2370_237073

theorem problem_1
  (α : ℝ)
  (h : Real.tan α = -1/2) :
  1 / (Real.sin α ^ 2 - Real.sin α * Real.cos α - 2 * Real.cos α ^ 2) = -1 := 
sorry

end NUMINAMATH_GPT_problem_1_l2370_237073


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l2370_237083

theorem angle_in_third_quadrant (α : ℝ) (k : ℤ) :
  (2 * ↑k * Real.pi + Real.pi < α ∧ α < 2 * ↑k * Real.pi + 3 * Real.pi / 2) →
  (∃ (m : ℤ), (0 < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < Real.pi ∨
                π < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < 3 * Real.pi / 2 ∨ 
                -π < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < 0)) :=
by
  sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l2370_237083


namespace NUMINAMATH_GPT_ball_distribution_l2370_237010

theorem ball_distribution (n m : Nat) (h_n : n = 6) (h_m : m = 2) : 
  ∃ ways, 
    (ways = 2 ^ n - (1 + n)) ∧ ways = 57 :=
by
  sorry

end NUMINAMATH_GPT_ball_distribution_l2370_237010


namespace NUMINAMATH_GPT_range_of_a_l2370_237041

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ((1 - a) * x > 1 - a) → (x < 1)) → (1 < a) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l2370_237041


namespace NUMINAMATH_GPT_pythagorean_triangle_divisible_by_5_l2370_237094

theorem pythagorean_triangle_divisible_by_5 {a b c : ℕ} (h : a^2 + b^2 = c^2) : 
  5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c := 
by
  sorry

end NUMINAMATH_GPT_pythagorean_triangle_divisible_by_5_l2370_237094


namespace NUMINAMATH_GPT_probability_closer_to_6_l2370_237089

theorem probability_closer_to_6 :
  let interval : Set ℝ := Set.Icc 0 6
  let subinterval : Set ℝ := Set.Icc 3 6
  let length_interval := 6
  let length_subinterval := 3
  (length_subinterval / length_interval) = 0.5 := by
    sorry

end NUMINAMATH_GPT_probability_closer_to_6_l2370_237089


namespace NUMINAMATH_GPT_problem1_problem2_l2370_237063

variable (α : ℝ)

axiom tan_alpha_condition : Real.tan (Real.pi + α) = -1/2

-- Problem 1 Statement
theorem problem1 
  (tan_alpha_condition : Real.tan (Real.pi + α) = -1/2) : 
  (2 * Real.cos (Real.pi - α) - 3 * Real.sin (Real.pi + α)) / 
  (4 * Real.cos (α - 2 * Real.pi) + Real.cos (3 * Real.pi / 2 - α)) = -7/9 := 
sorry

-- Problem 2 Statement
theorem problem2
  (tan_alpha_condition : Real.tan (Real.pi + α) = -1/2) :
  Real.sin α ^ 2 - 2 * Real.sin α * Real.cos α + 4 * Real.cos α ^ 2 = 21/5 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l2370_237063


namespace NUMINAMATH_GPT_number_of_valid_arithmetic_sequences_l2370_237023

theorem number_of_valid_arithmetic_sequences : 
  ∃ S : Finset (Finset ℕ), 
  S.card = 16 ∧ 
  ∀ s ∈ S, s.card = 3 ∧ 
  (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ s = {a, b, c} ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 
  (b - a = c - b) ∧ (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0)) := 
sorry

end NUMINAMATH_GPT_number_of_valid_arithmetic_sequences_l2370_237023


namespace NUMINAMATH_GPT_expand_binomials_l2370_237013

variable (x y : ℝ)

theorem expand_binomials: (2 * x - 5) * (3 * y + 15) = 6 * x * y + 30 * x - 15 * y - 75 :=
by sorry

end NUMINAMATH_GPT_expand_binomials_l2370_237013


namespace NUMINAMATH_GPT_min_side_length_of_square_l2370_237091

theorem min_side_length_of_square (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ s : ℝ, s = 
    if a < (Real.sqrt 2 + 1) * b then 
      a 
    else 
      (Real.sqrt 2 / 2) * (a + b) := 
sorry

end NUMINAMATH_GPT_min_side_length_of_square_l2370_237091


namespace NUMINAMATH_GPT_sqrt_two_irrational_l2370_237036

def irrational (x : ℝ) := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem sqrt_two_irrational : irrational (Real.sqrt 2) := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_two_irrational_l2370_237036


namespace NUMINAMATH_GPT_least_integer_of_sum_in_ratio_l2370_237065

theorem least_integer_of_sum_in_ratio (a b c : ℕ) (h1 : a + b + c = 90) (h2 : a * 3 = b * 2) (h3 : a * 5 = c * 2) : a = 18 :=
by
  sorry

end NUMINAMATH_GPT_least_integer_of_sum_in_ratio_l2370_237065


namespace NUMINAMATH_GPT_sum_of_solutions_l2370_237074

theorem sum_of_solutions (x : ℝ) :
  (∀ x, x^2 - 17 * x + 54 = 0) → 
  (∃ r s : ℝ, r ≠ s ∧ r + s = 17) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l2370_237074


namespace NUMINAMATH_GPT_area_of_EFCD_l2370_237021

noncomputable def area_of_quadrilateral (AB CD altitude: ℝ) :=
  let sum_bases_half := (AB + CD) / 2
  let small_altitude := altitude / 2
  small_altitude * (sum_bases_half + CD) / 2

theorem area_of_EFCD
  (AB CD altitude : ℝ)
  (AB_len : AB = 10)
  (CD_len : CD = 24)
  (altitude_len : altitude = 15)
  : area_of_quadrilateral AB CD altitude = 153.75 :=
by
  rw [AB_len, CD_len, altitude_len]
  simp [area_of_quadrilateral]
  sorry

end NUMINAMATH_GPT_area_of_EFCD_l2370_237021


namespace NUMINAMATH_GPT_quadratic_inequality_range_l2370_237030

theorem quadratic_inequality_range (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + a - 1 < 0) → a ≤ 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_range_l2370_237030


namespace NUMINAMATH_GPT_max_xy_l2370_237099

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 3 * x + 2 * y = 12) : 
  xy ≤ 6 :=
sorry

end NUMINAMATH_GPT_max_xy_l2370_237099


namespace NUMINAMATH_GPT_equal_number_of_boys_and_girls_l2370_237014

theorem equal_number_of_boys_and_girls
  (m d M D : ℝ)
  (hm : m ≠ 0)
  (hd : d ≠ 0)
  (avg1 : M / m ≠ D / d)
  (avg2 : (M / m + D / d) / 2 = (M + D) / (m + d)) :
  m = d :=
by
  sorry

end NUMINAMATH_GPT_equal_number_of_boys_and_girls_l2370_237014


namespace NUMINAMATH_GPT_cost_of_three_pencils_and_two_pens_l2370_237001

theorem cost_of_three_pencils_and_two_pens
  (p q : ℝ)
  (h₁ : 8 * p + 3 * q = 5.20)
  (h₂ : 2 * p + 5 * q = 4.40) :
  3 * p + 2 * q = 2.5881 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_three_pencils_and_two_pens_l2370_237001


namespace NUMINAMATH_GPT_watch_hands_angle_120_l2370_237032

theorem watch_hands_angle_120 (n : ℝ) (h₁ : 0 ≤ n ∧ n ≤ 60) 
    (h₂ : abs ((210 + n / 2) - 6 * n) = 120) : n = 43.64 := sorry

end NUMINAMATH_GPT_watch_hands_angle_120_l2370_237032


namespace NUMINAMATH_GPT_Xiaogang_shooting_probability_l2370_237024

theorem Xiaogang_shooting_probability (total_shots : ℕ) (shots_made : ℕ) (h_total : total_shots = 50) (h_made : shots_made = 38) :
  (shots_made : ℝ) / total_shots = 0.76 :=
by
  sorry

end NUMINAMATH_GPT_Xiaogang_shooting_probability_l2370_237024


namespace NUMINAMATH_GPT_quadratic_nonneg_range_l2370_237098

theorem quadratic_nonneg_range (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_nonneg_range_l2370_237098


namespace NUMINAMATH_GPT_river_width_l2370_237027

theorem river_width (depth : ℝ) (flow_rate : ℝ) (volume_per_minute : ℝ) 
  (h1 : depth = 2) 
  (h2 : flow_rate = 4000 / 60)  -- Flow rate in meters per minute
  (h3 : volume_per_minute = 6000) :
  volume_per_minute / (flow_rate * depth) = 45 :=
by
  sorry

end NUMINAMATH_GPT_river_width_l2370_237027


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l2370_237051

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then x^2 - 2 * x else (abs x)^2 - 2 * abs x

-- Define the condition that f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Problem 1: Prove the minimum value of f(x) is -1.
theorem problem1 (h_even : even_function f) : ∃ x : ℝ, f x = -1 :=
by
  sorry

-- Problem 2: Prove the solution set of f(x) > 0 is (-∞, -2) ∪ (2, +∞).
theorem problem2 (h_even : even_function f) : 
  { x : ℝ | f x > 0 } = { x : ℝ | x < -2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

-- Problem 3: Prove there exists a real number x such that f(x+2) + f(-x) = 0.
theorem problem3 (h_even : even_function f) : ∃ x : ℝ, f (x + 2) + f (-x) = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l2370_237051


namespace NUMINAMATH_GPT_remainder_of_9_pow_333_div_50_l2370_237020

theorem remainder_of_9_pow_333_div_50 : (9 ^ 333) % 50 = 29 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_9_pow_333_div_50_l2370_237020


namespace NUMINAMATH_GPT_balcony_more_than_orchestra_l2370_237000

-- Conditions
def total_tickets (O B : ℕ) : Prop := O + B = 340
def total_cost (O B : ℕ) : Prop := 12 * O + 8 * B = 3320

-- The statement we need to prove based on the conditions
theorem balcony_more_than_orchestra (O B : ℕ) (h1 : total_tickets O B) (h2 : total_cost O B) :
  B - O = 40 :=
sorry

end NUMINAMATH_GPT_balcony_more_than_orchestra_l2370_237000


namespace NUMINAMATH_GPT_sphere_center_x_axis_eq_l2370_237096

theorem sphere_center_x_axis_eq (a : ℝ) (R : ℝ) (x y z : ℝ) :
  (x - a) ^ 2 + y ^ 2 + z ^ 2 = R ^ 2 → (0 - a) ^ 2 + (0 - 0) ^ 2 + (0 - 0) ^ 2 = R ^ 2 →
  a = R →
  (x ^ 2 - 2 * a * x + y ^ 2 + z ^ 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_sphere_center_x_axis_eq_l2370_237096


namespace NUMINAMATH_GPT_grade_representation_l2370_237040

theorem grade_representation :
  (8, 1) = (8, 1) :=
by
  sorry

end NUMINAMATH_GPT_grade_representation_l2370_237040


namespace NUMINAMATH_GPT_solve_triangle_l2370_237033

theorem solve_triangle (a b : ℝ) (A B : ℝ) : ((A + B < π ∧ A > 0 ∧ B > 0 ∧ a > 0) ∨ (a > 0 ∧ b > 0 ∧ (π > A) ∧ (A > 0))) → ∃ c C, c > 0 ∧ (π > C) ∧ C > 0 :=
sorry

end NUMINAMATH_GPT_solve_triangle_l2370_237033


namespace NUMINAMATH_GPT_hydrogen_atoms_in_compound_l2370_237042

theorem hydrogen_atoms_in_compound : 
  ∀ (C O H : ℕ) (molecular_weight : ℕ), 
  C = 1 → 
  O = 3 → 
  molecular_weight = 62 → 
  (12 * C + 16 * O + H = molecular_weight) → 
  H = 2 := 
by
  intros C O H molecular_weight hc ho hmw hcalc
  sorry

end NUMINAMATH_GPT_hydrogen_atoms_in_compound_l2370_237042


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_l2370_237058

-- First problem: \(\frac{1}{3} + \left(-\frac{1}{2}\right) = -\frac{1}{6}\)
theorem problem1 : (1 / 3 : ℚ) + (-1 / 2) = -1 / 6 := by sorry

-- Second problem: \(-2 - \left(-9\right) = 7\)
theorem problem2 : (-2 : ℚ) - (-9) = 7 := by sorry

-- Third problem: \(\frac{15}{16} - \left(-7\frac{1}{16}\right) = 8\)
theorem problem3 : (15 / 16 : ℚ) - (-(7 + 1 / 16)) = 8 := by sorry

-- Fourth problem: \(-\left|-4\frac{2}{7}\right| - \left|+1\frac{5}{7}\right| = -6\)
theorem problem4 : -|(-4 - 2 / 7 : ℚ)| - |(1 + 5 / 7)| = -6 := by sorry

-- Fifth problem: \(6 + \left(-12\right) + 8.3 + \left(-7.5\right) = -5.2\)
theorem problem5 : (6 : ℚ) + (-12) + (83 / 10) + (-75 / 10) = -52 / 10 := by sorry

-- Sixth problem: \(\left(-\frac{1}{8}\right) + 3.25 + 2\frac{3}{5} + \left(-5.875\right) + 1.15 = 1\)
theorem problem6 : (-1 / 8 : ℚ) + 3 + 1 / 4 + 2 + 3 / 5 + (-5 - 875 / 1000) + 1 + 15 / 100 = 1 := by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_l2370_237058


namespace NUMINAMATH_GPT_asymptote_equations_l2370_237076

open Real

noncomputable def hyperbola_asymptotes (a b : ℝ) (e : ℝ) (x y : ℝ) :=
  (a > 0) ∧ (b > 0) ∧ (e = sqrt 3) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

theorem asymptote_equations (a b : ℝ) (ha : a > 0) (hb : b > 0) (he : sqrt (a^2 + b^2) / a = sqrt 3) :
  ∀ (x : ℝ), ∃ (y : ℝ), y = sqrt 2 * x ∨ y = -sqrt 2 * x :=
sorry

end NUMINAMATH_GPT_asymptote_equations_l2370_237076


namespace NUMINAMATH_GPT_rectangle_area_l2370_237064

theorem rectangle_area (b l : ℕ) (P : ℕ) (h1 : l = 3 * b) (h2 : P = 64) (h3 : P = 2 * (l + b)) :
  l * b = 192 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2370_237064


namespace NUMINAMATH_GPT_max_value_of_expression_l2370_237053

def real_numbers (m n : ℝ) := m > 0 ∧ n < 0 ∧ (1 / m + 1 / n = 1)

theorem max_value_of_expression (m n : ℝ) (h : real_numbers m n) : 4 * m + n ≤ 1 :=
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l2370_237053


namespace NUMINAMATH_GPT_global_phone_company_customers_l2370_237090

theorem global_phone_company_customers :
  (total_customers = 25000) →
  (us_percentage = 0.20) →
  (canada_percentage = 0.12) →
  (australia_percentage = 0.15) →
  (uk_percentage = 0.08) →
  (india_percentage = 0.05) →
  (us_customers = total_customers * us_percentage) →
  (canada_customers = total_customers * canada_percentage) →
  (australia_customers = total_customers * australia_percentage) →
  (uk_customers = total_customers * uk_percentage) →
  (india_customers = total_customers * india_percentage) →
  (mentioned_countries_customers = us_customers + canada_customers + australia_customers + uk_customers + india_customers) →
  (other_countries_customers = total_customers - mentioned_countries_customers) →
  (other_countries_customers = 10000) ∧ (us_customers / other_countries_customers = 1 / 2) :=
by
  -- The further proof steps would go here if needed
  sorry

end NUMINAMATH_GPT_global_phone_company_customers_l2370_237090


namespace NUMINAMATH_GPT_pile_splitting_l2370_237075

theorem pile_splitting (single_stone_piles : ℕ) :
  ∃ (final_heap_size : ℕ), 
    (∀ heap_size ≤ single_stone_piles, heap_size > 0 → (heap_size * 2) ≥ heap_size) ∧ (final_heap_size = single_stone_piles) :=
by
  sorry

end NUMINAMATH_GPT_pile_splitting_l2370_237075


namespace NUMINAMATH_GPT_side_length_of_smaller_square_l2370_237082

theorem side_length_of_smaller_square (s : ℝ) (A1 A2 : ℝ) (h1 : 5 * 5 = A1 + A2) (h2 : 2 * A2 = A1 + 25)  : s = 5 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_side_length_of_smaller_square_l2370_237082


namespace NUMINAMATH_GPT_arithmetic_seq_a7_l2370_237072

structure arith_seq (a : ℕ → ℤ) : Prop :=
  (step : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d)

theorem arithmetic_seq_a7
  {a : ℕ → ℤ}
  (h_seq : arith_seq a)
  (h1 : a 1 = 2)
  (h2 : a 3 + a 5 = 10)
  : a 7 = 8 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a7_l2370_237072


namespace NUMINAMATH_GPT_white_pairs_coincide_l2370_237088

theorem white_pairs_coincide 
  (red_half : ℕ) (blue_half : ℕ) (white_half : ℕ)
  (red_pairs : ℕ) (blue_pairs : ℕ) (red_white_pairs : ℕ) :
  red_half = 2 → blue_half = 4 → white_half = 6 →
  red_pairs = 1 → blue_pairs = 2 → red_white_pairs = 2 →
  2 * (red_half - red_pairs + blue_half - 2 * blue_pairs + 
       white_half - 2 * red_white_pairs) = 4 :=
by
  intros 
    h_red_half h_blue_half h_white_half 
    h_red_pairs h_blue_pairs h_red_white_pairs
  rw [h_red_half, h_blue_half, h_white_half, 
      h_red_pairs, h_blue_pairs, h_red_white_pairs]
  sorry

end NUMINAMATH_GPT_white_pairs_coincide_l2370_237088


namespace NUMINAMATH_GPT_paint_needed_for_new_statues_l2370_237046

-- Conditions
def pint_for_original : ℕ := 1
def original_height : ℕ := 8
def num_statues : ℕ := 320
def new_height : ℕ := 2
def scale_ratio : ℚ := (new_height : ℚ) / (original_height : ℚ)
def area_ratio : ℚ := scale_ratio ^ 2

-- Correct Answer
def total_paint_needed : ℕ := 20

-- Theorem to be proved
theorem paint_needed_for_new_statues :
  pint_for_original * num_statues * area_ratio = total_paint_needed := 
by
  sorry

end NUMINAMATH_GPT_paint_needed_for_new_statues_l2370_237046


namespace NUMINAMATH_GPT_smallest_palindromic_primes_l2370_237034

def is_palindromic (n : ℕ) : Prop :=
  ∀ a b : ℕ, n = 1001 * a + 1010 * b → 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_palindromic_primes :
  ∃ n1 n2 : ℕ, 
  is_palindromic n1 ∧ is_palindromic n2 ∧ is_prime n1 ∧ is_prime n2 ∧ n1 < n2 ∧
  ∀ m : ℕ, (is_palindromic m ∧ is_prime m ∧ m < n2 → m = n1) ∧
           (is_palindromic m ∧ is_prime m ∧ m < n1 → m ≠ n2) ∧ n1 = 1221 ∧ n2 = 1441 := 
sorry

end NUMINAMATH_GPT_smallest_palindromic_primes_l2370_237034


namespace NUMINAMATH_GPT_isosceles_trapezoid_side_length_l2370_237078

theorem isosceles_trapezoid_side_length (A b1 b2 h half_diff s : ℝ) (h0 : A = 44) (h1 : b1 = 8) (h2 : b2 = 14) 
    (h3 : A = 0.5 * (b1 + b2) * h)
    (h4 : h = 4) 
    (h5 : half_diff = (b2 - b1) / 2) 
    (h6 : half_diff = 3)
    (h7 : s^2 = h^2 + half_diff^2)
    (h8 : s = 5) : 
    s = 5 :=
by 
    apply h8

end NUMINAMATH_GPT_isosceles_trapezoid_side_length_l2370_237078


namespace NUMINAMATH_GPT_cups_per_serving_l2370_237043

theorem cups_per_serving (total_cups servings : ℝ) (h1 : total_cups = 36) (h2 : servings = 18.0) :
  total_cups / servings = 2 :=
by 
  sorry

end NUMINAMATH_GPT_cups_per_serving_l2370_237043


namespace NUMINAMATH_GPT_unique_arrangements_of_MOON_l2370_237011

open Nat

theorem unique_arrangements_of_MOON : 
  let word := "MOON"
  let n := 4
  let numM := 1
  let numN := 1
  let numO := 2
  factorial n / (factorial numO * factorial numM * factorial numN) = 12 :=
by
  let word := "MOON"
  let n := 4
  let numM := 1
  let numN := 1
  let numO := 2
  sorry

end NUMINAMATH_GPT_unique_arrangements_of_MOON_l2370_237011


namespace NUMINAMATH_GPT_no_second_quadrant_l2370_237071

theorem no_second_quadrant (k : ℝ) :
  (∀ x : ℝ, (x < 0 → 3 * x + k - 2 ≤ 0)) → k ≤ 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_no_second_quadrant_l2370_237071


namespace NUMINAMATH_GPT_reversible_triangle_inequality_l2370_237006

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def reversible_triangle (a b c : ℝ) : Prop :=
  (is_triangle a b c) ∧ 
  (is_triangle (1 / a) (1 / b) (1 / c)) ∧
  (a ≤ b) ∧ (b ≤ c)

theorem reversible_triangle_inequality {a b c : ℝ} (h : reversible_triangle a b c) :
  a > (3 - Real.sqrt 5) / 2 * c :=
sorry

end NUMINAMATH_GPT_reversible_triangle_inequality_l2370_237006


namespace NUMINAMATH_GPT_rhombus_region_area_l2370_237045

noncomputable def region_area (s : ℝ) (angleB : ℝ) : ℝ :=
  let h := (s / 2) * (Real.sin (angleB / 2))
  let area_triangle := (1 / 2) * (s / 2) * h
  3 * area_triangle

theorem rhombus_region_area : region_area 3 150 = 0.87345 := by
    sorry

end NUMINAMATH_GPT_rhombus_region_area_l2370_237045


namespace NUMINAMATH_GPT_largest_number_l2370_237009

theorem largest_number (a b c : ℝ) (h1 : a + b + c = 67) (h2 : c - b = 7) (h3 : b - a = 5) : c = 86 / 3 := 
by sorry

end NUMINAMATH_GPT_largest_number_l2370_237009


namespace NUMINAMATH_GPT_find_y_l2370_237018

theorem find_y 
  (x y : ℕ) 
  (hx : x % y = 9) 
  (hxy : (x : ℝ) / y = 96.12) : y = 75 :=
sorry

end NUMINAMATH_GPT_find_y_l2370_237018


namespace NUMINAMATH_GPT_new_computer_price_l2370_237025

theorem new_computer_price (d : ℕ) (h : 2 * d = 560) : d + 3 * d / 10 = 364 :=
by
  sorry

end NUMINAMATH_GPT_new_computer_price_l2370_237025


namespace NUMINAMATH_GPT_find_b_l2370_237092

-- Given conditions
def p (x : ℝ) : ℝ := 2 * x - 7
def q (x : ℝ) (b : ℝ) : ℝ := 3 * x - b

-- Assertion we need to prove
theorem find_b (b : ℝ) (h : p (q 3 b) = 3) : b = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_b_l2370_237092


namespace NUMINAMATH_GPT_range_of_a_l2370_237002

variable (a x : ℝ)

def p (a x : ℝ) : Prop := a - 4 < x ∧ x < a + 4

def q (x : ℝ) : Prop := (x - 2) * (x - 3) > 0

theorem range_of_a (h : ∀ (x : ℝ), p a x → q x) : a <= -2 ∨ a >= 7 := 
by sorry

end NUMINAMATH_GPT_range_of_a_l2370_237002


namespace NUMINAMATH_GPT_friend_balloon_count_l2370_237066

theorem friend_balloon_count (you_balloons friend_balloons : ℕ) (h1 : you_balloons = 7) (h2 : you_balloons = friend_balloons + 2) : friend_balloons = 5 :=
by
  sorry

end NUMINAMATH_GPT_friend_balloon_count_l2370_237066


namespace NUMINAMATH_GPT_range_of_a_l2370_237056

noncomputable def odd_function_periodic_real (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ -- odd function condition
  (∀ x, f (x + 5) = f x) ∧ -- periodic function condition
  (f 1 < -1) ∧ -- given condition
  (f 4 = Real.log a / Real.log 2) -- condition using log base 2

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h : odd_function_periodic_real f a) : a > 2 :=
by sorry 

end NUMINAMATH_GPT_range_of_a_l2370_237056


namespace NUMINAMATH_GPT_stewart_farm_food_l2370_237080

variable (S H : ℕ) (HorseFoodPerHorsePerDay : Nat) (TotalSheep : Nat)

theorem stewart_farm_food (ratio_sheep_horses : 6 * H = 7 * S) 
  (total_sheep_count : S = 48) 
  (horse_food : HorseFoodPerHorsePerDay = 230) : 
  HorseFoodPerHorsePerDay * (7 * 48 / 6) = 12880 :=
by
  sorry

end NUMINAMATH_GPT_stewart_farm_food_l2370_237080


namespace NUMINAMATH_GPT_zahra_kimmie_money_ratio_l2370_237039

theorem zahra_kimmie_money_ratio (KimmieMoney ZahraMoney : ℕ) (hKimmie : KimmieMoney = 450)
  (totalSavings : ℕ) (hSaving : totalSavings = 375)
  (h : KimmieMoney / 2 + ZahraMoney / 2 = totalSavings) :
  ZahraMoney / KimmieMoney = 2 / 3 :=
by
  -- Conditions to be used in the proof, but skipped for now
  sorry

end NUMINAMATH_GPT_zahra_kimmie_money_ratio_l2370_237039


namespace NUMINAMATH_GPT_parabola_line_non_intersect_l2370_237067

def P (x : ℝ) : ℝ := x^2 + 3 * x + 1
def Q : ℝ × ℝ := (10, 50)

def line_through_Q_with_slope (m x : ℝ) : ℝ := m * (x - Q.1) + Q.2

theorem parabola_line_non_intersect (r s : ℝ) (h : ∀ m, (r < m ∧ m < s) ↔ (∀ x, 
  x^2 + (3 - m) * x + (10 * m - 49) ≠ 0)) : r + s = 46 := 
sorry

end NUMINAMATH_GPT_parabola_line_non_intersect_l2370_237067


namespace NUMINAMATH_GPT_arithmetic_seq_sum_ratio_l2370_237062

theorem arithmetic_seq_sum_ratio (a1 d : ℝ) (S : ℕ → ℝ) 
  (hSn : ∀ n, S n = n * a1 + d * (n * (n - 1) / 2))
  (h_ratio : S 3 / S 6 = 1 / 3) :
  S 9 / S 6 = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_ratio_l2370_237062


namespace NUMINAMATH_GPT_spinner_probability_l2370_237070

theorem spinner_probability (P_D P_E : ℝ) (hD : P_D = 2/5) (hE : P_E = 1/5) 
  (hTotal : P_D + P_E + P_F = 1) : P_F = 2/5 :=
by
  sorry

end NUMINAMATH_GPT_spinner_probability_l2370_237070


namespace NUMINAMATH_GPT_geom_seq_a_n_l2370_237037

theorem geom_seq_a_n (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, a (n + 1) = r * a n) 
  (h_a3 : a 3 = -1) 
  (h_a7 : a 7 = -9) :
  a 5 = -3 :=
sorry

end NUMINAMATH_GPT_geom_seq_a_n_l2370_237037


namespace NUMINAMATH_GPT_equation_of_line_passing_through_A_equation_of_circle_l2370_237044

variable {α β γ : ℝ}
variable {a b c u v w : ℝ}
variable (A : ℝ × ℝ × ℝ) -- Barycentric coordinates of point A

-- Statement for the equation of a line passing through point A in barycentric coordinates
theorem equation_of_line_passing_through_A (A : ℝ × ℝ × ℝ) : 
  ∃ (u v w : ℝ), u * α + v * β + w * γ = 0 := by
  sorry

-- Statement for the equation of a circle in barycentric coordinates
theorem equation_of_circle {u v w : ℝ} :
  -a^2 * β * γ - b^2 * γ * α - c^2 * α * β +
  (u * α + v * β + w * γ) * (α + β + γ) = 0 := by
  sorry

end NUMINAMATH_GPT_equation_of_line_passing_through_A_equation_of_circle_l2370_237044


namespace NUMINAMATH_GPT_buffy_whiskers_l2370_237026

/-- Definition of whisker counts for the cats --/
def whiskers_of_juniper : ℕ := 12
def whiskers_of_puffy : ℕ := 3 * whiskers_of_juniper
def whiskers_of_scruffy : ℕ := 2 * whiskers_of_puffy
def whiskers_of_buffy : ℕ := (whiskers_of_juniper + whiskers_of_puffy + whiskers_of_scruffy) / 3

/-- Proof statement for the number of whiskers of Buffy --/
theorem buffy_whiskers : whiskers_of_buffy = 40 := 
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_buffy_whiskers_l2370_237026


namespace NUMINAMATH_GPT_four_letters_three_mailboxes_l2370_237068

theorem four_letters_three_mailboxes : (3 ^ 4) = 81 :=
  by sorry

end NUMINAMATH_GPT_four_letters_three_mailboxes_l2370_237068


namespace NUMINAMATH_GPT_horner_method_v1_l2370_237079

def polynomial (x : ℝ) : ℝ := 4 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

theorem horner_method_v1 (x : ℝ) (h : x = 5) : 
  ((4 * x + 2) * x + 3.5) = 22 := by
  rw [h]
  norm_num
  sorry

end NUMINAMATH_GPT_horner_method_v1_l2370_237079


namespace NUMINAMATH_GPT_reflect_point_x_axis_correct_l2370_237097

-- Definition of the transformation reflecting a point across the x-axis
def reflect_x_axis (P : ℝ × ℝ) : ℝ × ℝ := (P.1, -P.2)

-- Define the original point coordinates
def P : ℝ × ℝ := (-2, 3)

-- The Lean proof statement
theorem reflect_point_x_axis_correct :
  reflect_x_axis P = (-2, -3) :=
sorry

end NUMINAMATH_GPT_reflect_point_x_axis_correct_l2370_237097


namespace NUMINAMATH_GPT_first_shaded_complete_cycle_seat_190_l2370_237047

theorem first_shaded_complete_cycle_seat_190 : 
  ∀ (n : ℕ), (n ≥ 1) → 
  ∃ m : ℕ, 
    ((m ≥ n) ∧ 
    (∀ i : ℕ, (1 ≤ i ∧ i ≤ 12) → 
    ∃ k : ℕ, (k ≤ m ∧ (k * (k + 1) / 2) % 12 = (i - 1) % 12))) ↔ 
  ∃ m : ℕ, (m = 19 ∧ 190 = (m * (m + 1)) / 2) :=
by
  sorry

end NUMINAMATH_GPT_first_shaded_complete_cycle_seat_190_l2370_237047


namespace NUMINAMATH_GPT_chord_length_of_circle_l2370_237028

theorem chord_length_of_circle (x y : ℝ) :
  (x^2 + y^2 - 4 * x - 4 * y - 1 = 0) ∧ (y = x + 2) → 
  2 * Real.sqrt 7 = 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_GPT_chord_length_of_circle_l2370_237028


namespace NUMINAMATH_GPT_total_slices_l2370_237060

def pizzas : ℕ := 2
def slices_per_pizza : ℕ := 8

theorem total_slices : pizzas * slices_per_pizza = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_slices_l2370_237060


namespace NUMINAMATH_GPT_roots_sum_of_squares_l2370_237035

noncomputable def proof_problem (p q r : ℝ) : Prop :=
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 598

theorem roots_sum_of_squares
  (p q r : ℝ)
  (h1 : p + q + r = 18)
  (h2 : p * q + q * r + r * p = 25)
  (h3 : p * q * r = 6) :
  proof_problem p q r :=
by {
  -- Solution steps here (omitted; not needed for the task)
  sorry
}

end NUMINAMATH_GPT_roots_sum_of_squares_l2370_237035


namespace NUMINAMATH_GPT_gain_percentage_l2370_237012

theorem gain_percentage (CP SP : ℕ) (h_sell : SP = 10 * CP) : 
  (10 * CP / 25 * CP) * 100 = 40 := by
  sorry

end NUMINAMATH_GPT_gain_percentage_l2370_237012


namespace NUMINAMATH_GPT_area_of_original_triangle_l2370_237085

theorem area_of_original_triangle (a : Real) (S_intuitive : Real) : 
  a = 2 -> S_intuitive = (Real.sqrt 3) -> (S_intuitive / (Real.sqrt 2 / 4)) = 2 * Real.sqrt 6 := 
by
  sorry

end NUMINAMATH_GPT_area_of_original_triangle_l2370_237085


namespace NUMINAMATH_GPT_find_natural_number_l2370_237015

theorem find_natural_number :
  ∃ x : ℕ, (∀ d1 d2 : ℕ, d1 ∣ x → d2 ∣ x → d1 < d2 → d2 - d1 = 4) ∧
           (∀ d1 d2 : ℕ, d1 ∣ x → d2 ∣ x → d1 < d2 → x - d2 = 308) ∧
           x = 385 :=
by
  sorry

end NUMINAMATH_GPT_find_natural_number_l2370_237015


namespace NUMINAMATH_GPT_singer_arrangements_l2370_237057

theorem singer_arrangements (s1 s2 : Type) [Fintype s1] [Fintype s2] 
  (h1 : Fintype.card s1 = 4) (h2 : Fintype.card s2 = 1) :
  ∃ n : ℕ, n = 18 :=
by
  sorry

end NUMINAMATH_GPT_singer_arrangements_l2370_237057


namespace NUMINAMATH_GPT_product_value_l2370_237016

theorem product_value : 
  (1 / 2) * 4 * (1 / 8) * 16 * (1 / 32) * 64 * (1 / 128) * 256 * (1 / 512) * 1024 = 32 := 
by
  sorry

end NUMINAMATH_GPT_product_value_l2370_237016


namespace NUMINAMATH_GPT_sum_gcd_lcm_is_244_l2370_237093

-- Definitions of the constants
def a : ℕ := 12
def b : ℕ := 80

-- Main theorem statement
theorem sum_gcd_lcm_is_244 : Nat.gcd a b + Nat.lcm a b = 244 := by
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_is_244_l2370_237093


namespace NUMINAMATH_GPT_swimming_speed_l2370_237007

theorem swimming_speed (s v : ℝ) (h_s : s = 4) (h_time : 1 / (v - s) = 2 * (1 / (v + s))) : v = 12 := 
by
  sorry

end NUMINAMATH_GPT_swimming_speed_l2370_237007


namespace NUMINAMATH_GPT_larger_number_of_two_l2370_237022

theorem larger_number_of_two
  (HCF : ℕ)
  (factor1 : ℕ)
  (factor2 : ℕ)
  (cond_HCF : HCF = 23)
  (cond_factor1 : factor1 = 15)
  (cond_factor2 : factor2 = 16) :
  ∃ (A : ℕ), A = 23 * 16 := by
  sorry

end NUMINAMATH_GPT_larger_number_of_two_l2370_237022


namespace NUMINAMATH_GPT_min_value_of_x_plus_y_l2370_237059

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy: 0 < y) (h: 9 * x + y = x * y) : x + y ≥ 16 := 
sorry

end NUMINAMATH_GPT_min_value_of_x_plus_y_l2370_237059


namespace NUMINAMATH_GPT_siblings_gmat_scores_l2370_237055

-- Define the problem conditions
variables (x y z : ℝ)

theorem siblings_gmat_scores (h1 : x - y = 1/3) (h2 : z = (x + y) / 2) : 
  y = x - 1/3 ∧ z = x - 1/6 :=
by
  sorry

end NUMINAMATH_GPT_siblings_gmat_scores_l2370_237055


namespace NUMINAMATH_GPT_total_cost_is_734_l2370_237029

-- Define the cost of each ice cream flavor
def cost_vanilla : ℕ := 99
def cost_chocolate : ℕ := 129
def cost_strawberry : ℕ := 149

-- Define the amount of each flavor Mrs. Hilt buys
def num_vanilla : ℕ := 2
def num_chocolate : ℕ := 3
def num_strawberry : ℕ := 1

-- Calculate the total cost in cents
def total_cost : ℕ :=
  (num_vanilla * cost_vanilla) +
  (num_chocolate * cost_chocolate) +
  (num_strawberry * cost_strawberry)

-- Statement of the proof problem
theorem total_cost_is_734 : total_cost = 734 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_is_734_l2370_237029


namespace NUMINAMATH_GPT_triangle_perimeter_l2370_237031

theorem triangle_perimeter (A r p : ℝ) (hA : A = 60) (hr : r = 2.5) (h_eq : A = r * p / 2) : p = 48 := 
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l2370_237031


namespace NUMINAMATH_GPT_find_m_value_l2370_237086

theorem find_m_value : ∃ m : ℤ, 81 - 6 = 25 + m ∧ m = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l2370_237086


namespace NUMINAMATH_GPT_number_of_valid_numbers_l2370_237052

-- Define a function that checks if a number is composed of digits from the set {1, 2, 3}
def composed_of_123 (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → d = 1 ∨ d = 2 ∨ d = 3

-- Define a predicate for a number being less than 200,000
def less_than_200000 (n : ℕ) : Prop := n < 200000

-- Define a predicate for a number being divisible by 3
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- The main theorem statement
theorem number_of_valid_numbers : ∃ (count : ℕ), count = 202 ∧ 
  (∀ (n : ℕ), less_than_200000 n → composed_of_123 n → divisible_by_3 n → n < count) :=
sorry

end NUMINAMATH_GPT_number_of_valid_numbers_l2370_237052


namespace NUMINAMATH_GPT_value_of_f_at_3_l2370_237069

def f (x : ℚ) : ℚ := (2 * x + 3) / (4 * x - 5)

theorem value_of_f_at_3 : f 3 = 9 / 7 := by
  sorry

end NUMINAMATH_GPT_value_of_f_at_3_l2370_237069


namespace NUMINAMATH_GPT_ab_inequality_l2370_237061

theorem ab_inequality
  {a b : ℝ}
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (a_b_sum : a + b = 2) :
  ∀ n : ℕ, 2 ≤ n → (a^n + 1) * (b^n + 1) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_ab_inequality_l2370_237061


namespace NUMINAMATH_GPT_polygon_sides_l2370_237005

theorem polygon_sides (n : ℕ) (h1 : n ≥ 3)
  (h2 : ∃ (theta theta' : ℝ), theta = (n - 2) * 180 / n ∧ theta' = (n + 7) * 180 / (n + 9) ∧ theta' = theta + 9) : n = 15 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l2370_237005


namespace NUMINAMATH_GPT_find_a_l2370_237038

noncomputable def f (x a : ℝ) : ℝ := x * (Real.exp x + a * Real.exp (-x))

theorem find_a (a : ℝ) : (∀ x : ℝ, f x a = -f (-x) a) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2370_237038


namespace NUMINAMATH_GPT_coins_from_brother_l2370_237049

-- Defining the conditions as variables
variables (piggy_bank_coins : ℕ) (father_coins : ℕ) (given_to_Laura : ℕ) (left_coins : ℕ)

-- Setting the conditions
def conditions : Prop :=
  piggy_bank_coins = 15 ∧
  father_coins = 8 ∧
  given_to_Laura = 21 ∧
  left_coins = 15

-- The main theorem statement
theorem coins_from_brother (B : ℕ) :
  conditions piggy_bank_coins father_coins given_to_Laura left_coins →
  piggy_bank_coins + B + father_coins - given_to_Laura = left_coins →
  B = 13 :=
by
  sorry

end NUMINAMATH_GPT_coins_from_brother_l2370_237049


namespace NUMINAMATH_GPT_trapezoid_area_l2370_237050

theorem trapezoid_area (x : ℝ) :
  let base1 := 4 * x
  let base2 := 6 * x
  let height := x
  (base1 + base2) / 2 * height = 5 * x^2 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l2370_237050


namespace NUMINAMATH_GPT_total_lunch_bill_l2370_237087

def cost_of_hotdog : ℝ := 5.36
def cost_of_salad : ℝ := 5.10

theorem total_lunch_bill : cost_of_hotdog + cost_of_salad = 10.46 := 
by
  sorry

end NUMINAMATH_GPT_total_lunch_bill_l2370_237087


namespace NUMINAMATH_GPT_energy_calculation_l2370_237017

noncomputable def stormy_day_energy_production 
  (energy_per_day : ℝ) (days : ℝ) (number_of_windmills : ℝ) (proportional_increase : ℝ) : ℝ :=
  proportional_increase * (energy_per_day * days * number_of_windmills)

theorem energy_calculation
  (energy_per_day : ℝ) (days : ℝ) (number_of_windmills : ℝ) (wind_speed_proportion : ℝ)
  (stormy_day_energy_per_windmill : ℝ) (s : ℝ)
  (H1 : energy_per_day = 400) 
  (H2 : days = 2) 
  (H3 : number_of_windmills = 3) 
  (H4 : stormy_day_energy_per_windmill = s * energy_per_day)
  : stormy_day_energy_production energy_per_day days number_of_windmills s = s * (400 * 3 * 2) :=
by
  sorry

end NUMINAMATH_GPT_energy_calculation_l2370_237017


namespace NUMINAMATH_GPT_trigonometric_identity_l2370_237019

theorem trigonometric_identity :
  (1 / Real.cos (40 * Real.pi / 180) - 2 * Real.sqrt 3 / Real.sin (40 * Real.pi / 180)) = -4 * Real.tan (20 * Real.pi / 180) := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l2370_237019


namespace NUMINAMATH_GPT_range_of_m_for_circle_l2370_237008

theorem range_of_m_for_circle (m : ℝ) :
  (∃ x y, x^2 + y^2 - 4 * x - 2 * y + m = 0) → m < 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_for_circle_l2370_237008


namespace NUMINAMATH_GPT_graph_single_point_c_eq_7_l2370_237054

theorem graph_single_point_c_eq_7 (x y : ℝ) (c : ℝ) :
  (∃ p : ℝ × ℝ, ∀ x y : ℝ, 3 * x^2 + 4 * y^2 + 6 * x - 8 * y + c = 0 ↔ (x, y) = p) →
  c = 7 :=
by
  sorry

end NUMINAMATH_GPT_graph_single_point_c_eq_7_l2370_237054


namespace NUMINAMATH_GPT_solve_for_y_l2370_237003

variable {b c y : Real}

theorem solve_for_y (h : b > c) (h_eq : y^2 + c^2 = (b - y)^2) : y = (b^2 - c^2) / (2 * b) := 
sorry

end NUMINAMATH_GPT_solve_for_y_l2370_237003


namespace NUMINAMATH_GPT_find_number_added_l2370_237004

theorem find_number_added (x : ℕ) : (1250 / 50) + x = 7525 ↔ x = 7500 := by
  sorry

end NUMINAMATH_GPT_find_number_added_l2370_237004


namespace NUMINAMATH_GPT_james_savings_l2370_237084

-- Define the conditions
def cost_vest : ℝ := 250
def weight_plates_pounds : ℕ := 200
def cost_per_pound : ℝ := 1.2
def original_weight_vest_cost : ℝ := 700
def discount : ℝ := 100

-- Define the derived quantities based on conditions
def cost_weight_plates : ℝ := weight_plates_pounds * cost_per_pound
def total_cost_setup : ℝ := cost_vest + cost_weight_plates
def discounted_weight_vest_cost : ℝ := original_weight_vest_cost - discount
def savings : ℝ := discounted_weight_vest_cost - total_cost_setup

-- The statement to prove the savings
theorem james_savings : savings = 110 := by
  sorry

end NUMINAMATH_GPT_james_savings_l2370_237084


namespace NUMINAMATH_GPT_handshake_problem_l2370_237095

theorem handshake_problem (x y : ℕ) 
  (H : (x * (x - 1)) / 2 + y = 159) : 
  x = 18 ∧ y = 6 := 
sorry

end NUMINAMATH_GPT_handshake_problem_l2370_237095


namespace NUMINAMATH_GPT_circumscribed_radius_of_triangle_ABC_l2370_237077

variable (A B C R : ℝ) (a b c : ℝ)

noncomputable def triangle_ABC (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ B = 2 * A ∧ C = 3 * A

noncomputable def side_length (A a : ℝ) : Prop :=
  a = 6

noncomputable def circumscribed_radius (A B C a R : ℝ) : Prop :=
  2 * R = a / (Real.sin (Real.pi * A / 180))

theorem circumscribed_radius_of_triangle_ABC:
  triangle_ABC A B C →
  side_length A a →
  circumscribed_radius A B C a R →
  R = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_circumscribed_radius_of_triangle_ABC_l2370_237077


namespace NUMINAMATH_GPT_calf_probability_l2370_237081

theorem calf_probability 
  (P_B1 : ℝ := 0.6)  -- Proportion of calves from the first farm
  (P_B2 : ℝ := 0.3)  -- Proportion of calves from the second farm
  (P_B3 : ℝ := 0.1)  -- Proportion of calves from the third farm
  (P_B1_A : ℝ := 0.15)  -- Conditional probability of a calf weighing more than 300 kg given it is from the first farm
  (P_B2_A : ℝ := 0.25)  -- Conditional probability of a calf weighing more than 300 kg given it is from the second farm
  (P_B3_A : ℝ := 0.35)  -- Conditional probability of a calf weighing more than 300 kg given it is from the third farm)
  (P_A : ℝ := P_B1 * P_B1_A + P_B2 * P_B2_A + P_B3 * P_B3_A) : 
  P_B3 * P_B3_A / P_A = 0.175 := 
by
  sorry

end NUMINAMATH_GPT_calf_probability_l2370_237081


namespace NUMINAMATH_GPT_line_shift_upwards_l2370_237048

theorem line_shift_upwards (x y : ℝ) (h : y = -2 * x) : y + 3 = -2 * x + 3 :=
by sorry

end NUMINAMATH_GPT_line_shift_upwards_l2370_237048
