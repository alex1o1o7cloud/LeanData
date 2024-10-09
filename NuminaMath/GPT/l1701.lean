import Mathlib

namespace infinite_solutions_xyz_l1701_170192

theorem infinite_solutions_xyz : ∀ k : ℕ, 
  (∃ n : ℕ, n > k ∧ ∃ x y z : ℕ, x^2 + y^2 + z^2 - x*y*z + 10 = 0 ∧ x > 2008 ∧ y > 2008 ∧ z > 2008) →
  ∃ x y z : ℕ, x^2 + y^2 + z^2 - x*y*z + 10 = 0 ∧ x > 2008 ∧ y > 2008 ∧ z > 2008 := 
sorry

end infinite_solutions_xyz_l1701_170192


namespace green_socks_count_l1701_170183

theorem green_socks_count: 
  ∀ (total_socks : ℕ) (white_socks : ℕ) (blue_socks : ℕ) (red_socks : ℕ) (green_socks : ℕ),
  total_socks = 900 →
  white_socks = total_socks / 3 →
  blue_socks = total_socks / 4 →
  red_socks = total_socks / 5 →
  green_socks = total_socks - (white_socks + blue_socks + red_socks) →
  green_socks = 195 :=
by
  intros total_socks white_socks blue_socks red_socks green_socks
  sorry

end green_socks_count_l1701_170183


namespace birds_never_gather_44_l1701_170115

theorem birds_never_gather_44 :
    ∀ (position : Fin 44 → Nat), 
    (∀ (i : Fin 44), position i ≤ 44) →
    (∀ (i j : Fin 44), position i ≠ position j) →
    ∃ (S : Nat), S % 4 = 2 →
    ∀ (moves : (Fin 44 → Fin 44) → (Fin 44 → Fin 44)),
    ¬(∃ (tree : Nat), ∀ (i : Fin 44), position i = tree) := 
sorry

end birds_never_gather_44_l1701_170115


namespace stan_average_speed_l1701_170116

/-- Given two trips with specified distances and times, prove that the overall average speed is 55 mph. -/
theorem stan_average_speed :
  let distance1 := 300
  let hours1 := 5
  let minutes1 := 20
  let distance2 := 360
  let hours2 := 6
  let minutes2 := 40
  let total_distance := distance1 + distance2
  let total_time := (hours1 + minutes1 / 60) + (hours2 + minutes2 / 60)
  total_distance / total_time = 55 := 
sorry

end stan_average_speed_l1701_170116


namespace prime_p_squared_plus_71_divisors_l1701_170154

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

def num_distinct_divisors (n : ℕ) : ℕ :=
  (factors n).toFinset.card

theorem prime_p_squared_plus_71_divisors (p : ℕ) (hp : is_prime p) 
  (hdiv : num_distinct_divisors (p ^ 2 + 71) ≤ 10) : p = 2 ∨ p = 3 :=
sorry

end prime_p_squared_plus_71_divisors_l1701_170154


namespace remainder_proof_l1701_170123

theorem remainder_proof (n : ℤ) (h : n % 6 = 1) : (3 * (n + 1812)) % 6 = 3 := 
by 
  sorry

end remainder_proof_l1701_170123


namespace solution_set_of_inequality_l1701_170113

theorem solution_set_of_inequality (x : ℝ) : (|2 * x - 1| < 1) ↔ (0 < x ∧ x < 1) :=
sorry

end solution_set_of_inequality_l1701_170113


namespace part1_part2_l1701_170103

-- We state the problem conditions and theorems to be proven accordingly
variable (A B C : Real) (a b c : Real)

-- Condition 1: In triangle ABC, opposite sides a, b, c with angles A, B, C such that a sin(B - C) = b sin(A - C)
axiom condition1 (A B C : Real) (a b c : Real) : a * Real.sin (B - C) = b * Real.sin (A - C)

-- Question 1: Prove that a = b under the given conditions
theorem part1 (A B C : Real) (a b c : Real) (h1 : a * Real.sin (B - C) = b * Real.sin (A - C)) : a = b := sorry

-- Condition 2: If c = 5 and cos C = 12/13
axiom condition2 (c : Real) : c = 5
axiom condition3 (C : Real) : Real.cos C = 12 / 13

-- Question 2: Prove that the area of triangle ABC is 125/4 under the given conditions
theorem part2 (A B C : Real) (a b c : Real) (h1 : a * Real.sin (B - C) = b * Real.sin (A - C)) 
               (h2 : c = 5) (h3 : Real.cos C = 12 / 13): (1 / 2) * a * b * (Real.sin C) = 125 / 4 := sorry

end part1_part2_l1701_170103


namespace star_evaluation_l1701_170117

def star (X Y : ℚ) := (X + Y) / 4

theorem star_evaluation : star (star 3 8) 6 = 35 / 16 := by
  sorry

end star_evaluation_l1701_170117


namespace initial_mixture_amount_l1701_170101

theorem initial_mixture_amount (x : ℝ) (h1 : 20 / 100 * x / (x + 3) = 6 / 35) : x = 18 :=
sorry

end initial_mixture_amount_l1701_170101


namespace percent_increase_decrease_condition_l1701_170125

theorem percent_increase_decrease_condition (p q M : ℝ) (hp : 0 < p) (hq : 0 < q) (hM : 0 < M) (hq50 : q < 50) :
  (M * (1 + p / 100) * (1 - q / 100) < M) ↔ (p < 100 * q / (100 - q)) := 
sorry

end percent_increase_decrease_condition_l1701_170125


namespace households_with_at_least_one_appliance_l1701_170172

theorem households_with_at_least_one_appliance (total: ℕ) (color_tvs: ℕ) (refrigerators: ℕ) (both: ℕ) :
  total = 100 → color_tvs = 65 → refrigerators = 84 → both = 53 →
  (color_tvs + refrigerators - both) = 96 :=
by
  intros
  sorry

end households_with_at_least_one_appliance_l1701_170172


namespace minimum_bottles_needed_l1701_170114

theorem minimum_bottles_needed (medium_volume jumbo_volume : ℕ) (h_medium : medium_volume = 120) (h_jumbo : jumbo_volume = 2000) : 
  let minimum_bottles := (jumbo_volume + medium_volume - 1) / medium_volume
  minimum_bottles = 17 :=
by
  sorry

end minimum_bottles_needed_l1701_170114


namespace one_over_x_plus_one_over_y_eq_two_l1701_170197

theorem one_over_x_plus_one_over_y_eq_two 
  (x y : ℝ)
  (h1 : 3^x = Real.sqrt 12)
  (h2 : 4^y = Real.sqrt 12) : 
  1 / x + 1 / y = 2 := 
by 
  sorry

end one_over_x_plus_one_over_y_eq_two_l1701_170197


namespace complement_of_M_in_U_is_1_4_l1701_170191

-- Define U
def U : Set ℕ := {x | x < 5 ∧ x ≠ 0}

-- Define M
def M : Set ℕ := {x | x^2 - 5 * x + 6 = 0}

-- The complement of M in U
def complement_U_M : Set ℕ := U \ M

-- The theorem to prove
theorem complement_of_M_in_U_is_1_4 : complement_U_M = {1, 4} := 
by sorry

end complement_of_M_in_U_is_1_4_l1701_170191


namespace multiply_scaled_values_l1701_170106

theorem multiply_scaled_values (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by 
  sorry

end multiply_scaled_values_l1701_170106


namespace rows_before_change_l1701_170135

-- Definitions and conditions
variables {r c : ℕ}

-- The total number of tiles before and after the change
def total_tiles_before (r c : ℕ) := r * c = 30
def total_tiles_after (r c : ℕ) := (r + 4) * (c - 2) = 30

-- Prove that the number of rows before the change is 3
theorem rows_before_change (h1 : total_tiles_before r c) (h2 : total_tiles_after r c) : r = 3 := 
sorry

end rows_before_change_l1701_170135


namespace parity_equivalence_l1701_170177

theorem parity_equivalence (p q : ℕ) :
  (Even (p^3 - q^3)) ↔ (Even (p + q)) :=
by
  sorry

end parity_equivalence_l1701_170177


namespace area_square_EFGH_equiv_144_l1701_170175

theorem area_square_EFGH_equiv_144 (a b : ℝ) (h : a = 6) (hb : b = 6)
  (side_length_EFGH : ℝ) (hs : side_length_EFGH = a + 3 + 3) : side_length_EFGH ^ 2 = 144 :=
by
  -- Given conditions
  sorry

end area_square_EFGH_equiv_144_l1701_170175


namespace buying_beams_l1701_170149

/-- Problem Statement:
Given:
1. The total money for beams is 6210 wen.
2. The transportation cost per beam is 3 wen.
3. Removing one beam means the remaining beams' total transportation cost equals the price of one beam.

Prove: 3 * (x - 1) = 6210 / x
-/
theorem buying_beams (x : ℕ) (h₁ : x > 0) (h₂ : 6210 % x = 0) :
  3 * (x - 1) = 6210 / x :=
sorry

end buying_beams_l1701_170149


namespace functions_increase_faster_l1701_170160

-- Define the functions
def y₁ (x : ℝ) : ℝ := 100 * x
def y₂ (x : ℝ) : ℝ := 1000 + 100 * x
def y₃ (x : ℝ) : ℝ := 10000 + 99 * x

-- Restate the problem in Lean
theorem functions_increase_faster :
  (∀ (x : ℝ), deriv y₁ x = 100) ∧
  (∀ (x : ℝ), deriv y₂ x = 100) ∧
  (∀ (x : ℝ), deriv y₃ x = 99) ∧
  (100 > 99) :=
by
  sorry

end functions_increase_faster_l1701_170160


namespace eval_expression_solve_inequalities_l1701_170189

-- Problem 1: Evaluation of the expression equals sqrt(2)
theorem eval_expression : (1 - 1^2023 + Real.sqrt 9 - (Real.pi - 3)^0 + |Real.sqrt 2 - 1|) = Real.sqrt 2 := 
by sorry

-- Problem 2: Solution set of the inequality system
theorem solve_inequalities (x : ℝ) : 
  ((3 * x + 1) / 2 ≥ (4 * x + 3) / 3 ∧ 2 * x + 7 ≥ 5 * x - 17) ↔ (3 ≤ x ∧ x ≤ 8) :=
by sorry

end eval_expression_solve_inequalities_l1701_170189


namespace good_numbers_10_70_l1701_170118

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def no_repeating_digits (n : ℕ) : Prop :=
  (n / 10 ≠ n % 10)

def is_good_number (n : ℕ) : Prop :=
  no_repeating_digits n ∧ (n % sum_of_digits n = 0)

theorem good_numbers_10_70 :
  is_good_number 10 ∧ is_good_number (10 + 11) ∧
  is_good_number 70 ∧ is_good_number (70 + 11) :=
by {
  -- Check that 10 is a good number
  -- Check that 21 is a good number
  -- Check that 70 is a good number
  -- Check that 81 is a good number
  sorry
}

end good_numbers_10_70_l1701_170118


namespace area_ratio_of_similar_polygons_l1701_170170

theorem area_ratio_of_similar_polygons (similarity_ratio: ℚ) (hratio: similarity_ratio = 1/5) : (similarity_ratio ^ 2 = 1/25) := 
by 
  sorry

end area_ratio_of_similar_polygons_l1701_170170


namespace geometric_sequence_ratio_l1701_170139

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (S3 : ℝ) 
  (h1 : a 1 = 1) (h2 : S3 = 3 / 4) 
  (h3 : S3 = a 1 + a 1 * q + a 1 * q^2) :
  q = -1 / 2 := 
by
  sorry

end geometric_sequence_ratio_l1701_170139


namespace distance_between_first_and_last_is_140_l1701_170143

-- Given conditions
def eightFlowers : ℕ := 8
def distanceFirstToFifth : ℕ := 80
def intervalsBetweenFirstAndFifth : ℕ := 4 -- 1 to 5 means 4 intervals
def intervalsBetweenFirstAndLast : ℕ := 7 -- 1 to 8 means 7 intervals
def distanceBetweenConsecutiveFlowers : ℕ := distanceFirstToFifth / intervalsBetweenFirstAndFifth
def totalDistanceFirstToLast : ℕ := distanceBetweenConsecutiveFlowers * intervalsBetweenFirstAndLast

-- Theorem to prove the question equals the correct answer
theorem distance_between_first_and_last_is_140 :
  totalDistanceFirstToLast = 140 := by
  sorry

end distance_between_first_and_last_is_140_l1701_170143


namespace sin_alpha_beta_gamma_values_l1701_170161

open Real

theorem sin_alpha_beta_gamma_values (α β γ : ℝ)
  (h1 : sin α = sin (α + β + γ) + 1)
  (h2 : sin β = 3 * sin (α + β + γ) + 2)
  (h3 : sin γ = 5 * sin (α + β + γ) + 3) :
  sin α * sin β * sin γ = (3/64) ∨ sin α * sin β * sin γ = (1/8) :=
sorry

end sin_alpha_beta_gamma_values_l1701_170161


namespace determine_constants_l1701_170163

theorem determine_constants
  (C D : ℝ)
  (h1 : 3 * C + D = 7)
  (h2 : 4 * C - 2 * D = -15) :
  C = -0.1 ∧ D = 7.3 :=
by
  sorry

end determine_constants_l1701_170163


namespace geometric_sum_is_correct_l1701_170147

theorem geometric_sum_is_correct : 
  let a := 1
  let r := 5
  let n := 6
  a * (r^n - 1) / (r - 1) = 3906 := by
  sorry

end geometric_sum_is_correct_l1701_170147


namespace intersection_of_lines_l1701_170188

theorem intersection_of_lines :
  ∃ (x y : ℚ), (8 * x - 3 * y = 24) ∧ (10 * x + 2 * y = 14) ∧ x = 45 / 23 ∧ y = -64 / 23 :=
by
  sorry

end intersection_of_lines_l1701_170188


namespace large_pizzas_sold_l1701_170173

variables (num_small_pizzas num_large_pizzas : ℕ) (price_small price_large total_revenue revenue_from_smalls revenue_from_larges : ℕ)

theorem large_pizzas_sold
  (price_small := 2)
  (price_large := 8)
  (total_revenue := 40)
  (num_small_pizzas := 8)
  (revenue_from_smalls := num_small_pizzas * price_small)
  (revenue_from_larges := total_revenue - revenue_from_smalls)
  (large_pizza_count := revenue_from_larges / price_large) :
  large_pizza_count = 3 :=
sorry

end large_pizzas_sold_l1701_170173


namespace algebraic_expression_zero_l1701_170174

theorem algebraic_expression_zero (a b : ℝ) (h : a^2 + 2 * a * b + b^2 = 0) : 
  a * (a + 4 * b) - (a + 2 * b) * (a - 2 * b) = 0 :=
by
  sorry

end algebraic_expression_zero_l1701_170174


namespace compound_interest_correct_amount_l1701_170171

-- Define constants and conditions
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

def compound_interest (P R T : ℕ) : ℕ := P * ((1 + R / 100) ^ T - 1)

-- Given values and conditions
def P₁ : ℕ := 1750
def R₁ : ℕ := 8
def T₁ : ℕ := 3
def R₂ : ℕ := 10
def T₂ : ℕ := 2

def SI : ℕ := simple_interest P₁ R₁ T₁
def CI : ℕ := 2 * SI

def P₂ : ℕ := 4000

-- The statement to be proven
theorem compound_interest_correct_amount : 
  compound_interest P₂ R₂ T₂ = CI := 
by 
  sorry

end compound_interest_correct_amount_l1701_170171


namespace cake_piece_volume_l1701_170133

theorem cake_piece_volume (h : ℝ) (d : ℝ) (n : ℕ) (V_piece : ℝ) : 
  h = 1/2 ∧ d = 16 ∧ n = 8 → V_piece = 4 * Real.pi :=
by
  sorry

end cake_piece_volume_l1701_170133


namespace algebraic_expression_value_l1701_170150

theorem algebraic_expression_value (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (  ( ((x + 2)^2 * (x^2 - 2 * x + 4)^2) / ( (x^3 + 8)^2 ))^2
   * ( ((x - 2)^2 * (x^2 + 2 * x + 4)^2) / ( (x^3 - 8)^2 ))^2 ) = 1 :=
by
  sorry

end algebraic_expression_value_l1701_170150


namespace pairs_satisfying_equation_l1701_170127

theorem pairs_satisfying_equation :
  ∀ x y : ℝ, (x ^ 4 + 1) * (y ^ 4 + 1) = 4 * x^2 * y^2 ↔ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by
  intros x y
  sorry

end pairs_satisfying_equation_l1701_170127


namespace consecutive_numbers_sum_l1701_170185

theorem consecutive_numbers_sum (n : ℤ) (h1 : (n - 1) * n * (n + 1) = 210) (h2 : ∀ m, (m - 1) * m * (m + 1) = 210 → (m - 1)^2 + m^2 + (m + 1)^2 ≥ (n - 1)^2 + n^2 + (n + 1)^2) :
  (n - 1) + n = 11 :=
by 
  sorry

end consecutive_numbers_sum_l1701_170185


namespace samantha_trip_l1701_170132

theorem samantha_trip (a b c d x : ℕ)
  (h1 : 1 ≤ a) (h2 : a + b + c + d ≤ 10) 
  (h3 : 1000 * d + 100 * c + 10 * b + a - (1000 * a + 100 * b + 10 * c + d) = 60 * x)
  : a^2 + b^2 + c^2 + d^2 = 83 :=
sorry

end samantha_trip_l1701_170132


namespace batsman_average_30_matches_l1701_170102

theorem batsman_average_30_matches (avg_20_matches : ℕ -> ℚ) (avg_10_matches : ℕ -> ℚ)
  (h1 : avg_20_matches 20 = 40)
  (h2 : avg_10_matches 10 = 20)
  : (20 * (avg_20_matches 20) + 10 * (avg_10_matches 10)) / 30 = 33.33 := by
  sorry

end batsman_average_30_matches_l1701_170102


namespace distance_second_day_l1701_170176

theorem distance_second_day 
  (total_distance : ℕ)
  (a1 : ℕ)
  (n : ℕ)
  (r : ℚ)
  (hn : n = 6)
  (htotal : total_distance = 378)
  (hr : r = 1 / 2)
  (geo_sum : a1 * (1 - r^n) / (1 - r) = total_distance) :
  a1 * r = 96 :=
by
  sorry

end distance_second_day_l1701_170176


namespace find_b_l1701_170195

-- Define the quadratic equation
def quadratic_eq (b : ℝ) (x : ℝ) : ℝ :=
  x^2 + b * x - 15

-- Prove that b = 49/8 given -8 is a solution to the quadratic equation
theorem find_b (b : ℝ) : quadratic_eq b (-8) = 0 -> b = 49 / 8 :=
by
  intro h
  sorry

end find_b_l1701_170195


namespace area_ratio_of_squares_l1701_170146

theorem area_ratio_of_squares (R x y : ℝ) (hx : x^2 = (4/5) * R^2) (hy : y = R * Real.sqrt 2) :
  x^2 / y^2 = 2 / 5 :=
by sorry

end area_ratio_of_squares_l1701_170146


namespace xyz_eq_7cubed_l1701_170119

theorem xyz_eq_7cubed (x y z : ℤ) (h1 : x^2 * y * z^3 = 7^4) (h2 : x * y^2 = 7^5) : x * y * z = 7^3 := 
by 
  sorry

end xyz_eq_7cubed_l1701_170119


namespace gcd_lcm_product_135_l1701_170140

theorem gcd_lcm_product_135 (a b : ℕ) (ha : a = 9) (hb : b = 15) :
  Nat.gcd a b * Nat.lcm a b = 135 :=
by
  sorry

end gcd_lcm_product_135_l1701_170140


namespace license_plate_count_l1701_170155

def num_license_plates : Nat :=
  let letters := 26 -- choices for each of the first two letters
  let primes := 4 -- choices for prime digits
  let composites := 4 -- choices for composite digits
  letters * letters * (primes * composites * 2)

theorem license_plate_count : num_license_plates = 21632 :=
  by
  sorry

end license_plate_count_l1701_170155


namespace min_value_arithmetic_sequence_l1701_170141

theorem min_value_arithmetic_sequence (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_arith_seq : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 1 + a 2014 = 2) :
  (∃ a2 a2013, a 2 = a2 ∧ a 2013 = a2013 ∧ a2 > 0 ∧ a2013 > 0 ∧ ∀ a2 a2013, a 2 = a2 ∧ a 2013 = a2013 → (1/a2 + 1/a2013) ≥ 2) :=
by
  sorry

end min_value_arithmetic_sequence_l1701_170141


namespace geometric_sequence_theorem_l1701_170100

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n+1) = a n * r

def holds_condition (a : ℕ → ℝ) : Prop := 
  a 1 * a 10 = -2

theorem geometric_sequence_theorem (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_cond : holds_condition a) : a 4 * a 7 = -2 :=
by
  sorry

end geometric_sequence_theorem_l1701_170100


namespace find_x_l1701_170120

theorem find_x (x : ℕ) (h1 : 8 = 2 ^ 3) (h2 : 32 = 2 ^ 5) :
  (2^(x+2) * 8^(x-1) = 32^3) ↔ (x = 4) :=
by
  sorry

end find_x_l1701_170120


namespace find_k_l1701_170126

-- Define the function f as described in the problem statement
def f (n : ℕ) : ℕ := 
  if n % 2 = 1 then 
    n + 3 
  else 
    n / 2

theorem find_k (k : ℕ) (h_odd : k % 2 = 1) : f (f (f k)) = k → k = 1 :=
by {
  sorry
}

end find_k_l1701_170126


namespace james_total_matches_l1701_170164

def boxes_count : ℕ := 5 * 12
def matches_per_box : ℕ := 20
def total_matches (boxes : ℕ) (matches_per_box : ℕ) : ℕ := boxes * matches_per_box

theorem james_total_matches : total_matches boxes_count matches_per_box = 1200 :=
by {
  sorry
}

end james_total_matches_l1701_170164


namespace brady_june_hours_l1701_170180

variable (x : ℕ) -- Number of hours worked every day in June

def hoursApril : ℕ := 6 * 30 -- Total hours in April
def hoursSeptember : ℕ := 8 * 30 -- Total hours in September
def hoursJune (x : ℕ) : ℕ := x * 30 -- Total hours in June
def totalHours (x : ℕ) : ℕ := hoursApril + hoursJune x + hoursSeptember -- Total hours over three months
def averageHours (x : ℕ) : ℕ := totalHours x / 3 -- Average hours per month

theorem brady_june_hours (h : averageHours x = 190) : x = 5 :=
by
  sorry

end brady_june_hours_l1701_170180


namespace restore_salary_l1701_170122

variable (W : ℝ) -- Define the initial wage as a real number
variable (newWage : ℝ := 0.7 * W) -- New wage after a 30% reduction

-- Define the hypothesis for the initial wage reduction
theorem restore_salary : (100 * (W / (0.7 * W) - 1)) = 42.86 :=
by
  sorry

end restore_salary_l1701_170122


namespace basic_astrophysics_degrees_l1701_170181

-- Define the percentages for various sectors
def microphotonics := 14
def home_electronics := 24
def food_additives := 15
def genetically_modified_microorganisms := 19
def industrial_lubricants := 8

-- The sum of the given percentages
def total_other_percentages := 
    microphotonics + home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants

-- The remaining percentage for basic astrophysics
def basic_astrophysics_percentage := 100 - total_other_percentages

-- Number of degrees in a full circle
def full_circle_degrees := 360

-- Calculate the degrees representing basic astrophysics
def degrees_for_basic_astrophysics := (basic_astrophysics_percentage * full_circle_degrees) / 100

-- Theorem statement
theorem basic_astrophysics_degrees : degrees_for_basic_astrophysics = 72 := 
by
  sorry

end basic_astrophysics_degrees_l1701_170181


namespace peony_total_count_l1701_170184

theorem peony_total_count (n : ℕ) (x : ℕ) (total_sample : ℕ) (single_sample : ℕ) (double_sample : ℕ) (thousand_sample : ℕ) (extra_thousand : ℕ)
    (h1 : thousand_sample > single_sample)
    (h2 : thousand_sample - single_sample = extra_thousand)
    (h3 : total_sample = single_sample + double_sample + thousand_sample)
    (h4 : total_sample = 12)
    (h5 : single_sample = 4)
    (h6 : double_sample = 2)
    (h7 : thousand_sample = 6)
    (h8 : extra_thousand = 30) :
    n = 180 :=
by 
  sorry

end peony_total_count_l1701_170184


namespace sum_of_bases_is_20_l1701_170129

theorem sum_of_bases_is_20
  (B1 B2 : ℕ)
  (G1 : ℚ)
  (G2 : ℚ)
  (hG1_B1 : G1 = (4 * B1 + 5) / (B1^2 - 1))
  (hG2_B1 : G2 = (5 * B1 + 4) / (B1^2 - 1))
  (hG1_B2 : G1 = (3 * B2) / (B2^2 - 1))
  (hG2_B2 : G2 = (6 * B2) / (B2^2 - 1)) :
  B1 + B2 = 20 :=
sorry

end sum_of_bases_is_20_l1701_170129


namespace generating_sets_Z2_l1701_170179

theorem generating_sets_Z2 (a b : ℤ × ℤ) (h : Submodule.span ℤ ({a, b} : Set (ℤ × ℤ)) = ⊤) :
  let a₁ := a.1
  let a₂ := a.2
  let b₁ := b.1
  let b₂ := b.2
  a₁ * b₂ - a₂ * b₁ = 1 ∨ a₁ * b₂ - a₂ * b₁ = -1 := 
by
  sorry

end generating_sets_Z2_l1701_170179


namespace second_quadrant_distance_l1701_170166

theorem second_quadrant_distance 
    (m : ℝ) 
    (P : ℝ × ℝ)
    (hP1 : P = (m - 3, m + 2))
    (hP2 : (m + 2) > 0)
    (hP3 : (m - 3) < 0)
    (hDist : |(m + 2)| = 4) : P = (-1, 4) := 
by
  have h1 : m + 2 = 4 := sorry
  have h2 : m = 2 := sorry
  have h3 : P = (2 - 3, 2 + 2) := sorry
  have h4 : P = (-1, 4) := sorry
  exact h4

end second_quadrant_distance_l1701_170166


namespace sum_of_k_values_l1701_170108

theorem sum_of_k_values (k : ℤ) :
  (∃ (r s : ℤ), (r ≠ s) ∧ (3 * r * s = 9) ∧ (r + s = k / 3)) → k = 0 :=
by sorry

end sum_of_k_values_l1701_170108


namespace function_equivalence_l1701_170152

theorem function_equivalence (f : ℝ → ℝ) (h : ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = 2020) :
  ∃ g : ℝ → ℝ, (∀ y : ℝ, g (-y) = -g y) ∧ (∀ x : ℝ, f x = g (1 - 2 * x^2) + 1010) :=
sorry

end function_equivalence_l1701_170152


namespace system_solution_and_range_l1701_170107

theorem system_solution_and_range (a x y : ℝ) (h1 : 2 * x + y = 5 * a) (h2 : x - 3 * y = -a + 7) :
  (x = 2 * a + 1 ∧ y = a - 2) ∧ (-1/2 ≤ a ∧ a < 2 → 2 * a + 1 ≥ 0 ∧ a - 2 < 0) :=
by
  sorry

end system_solution_and_range_l1701_170107


namespace commutativity_associativity_l1701_170178

variables {α : Type*} (op : α → α → α)

-- Define conditions as hypotheses
axiom cond1 : ∀ a b c : α, op a (op b c) = op b (op c a)
axiom cond2 : ∀ a b c : α, op a b = op a c → b = c
axiom cond3 : ∀ a b c : α, op a c = op b c → a = b

-- Commutativity statement
theorem commutativity (a b : α) : op a b = op b a := sorry

-- Associativity statement
theorem associativity (a b c : α) : op (op a b) c = op a (op b c) := sorry

end commutativity_associativity_l1701_170178


namespace ratio_alcohol_to_water_l1701_170110

theorem ratio_alcohol_to_water (vol_alcohol vol_water : ℚ) 
  (h_alcohol : vol_alcohol = 2/7) 
  (h_water : vol_water = 3/7) : 
  vol_alcohol / vol_water = 2 / 3 := 
by
  sorry

end ratio_alcohol_to_water_l1701_170110


namespace fraction_product_l1701_170169

theorem fraction_product : 
  (4 / 2) * (3 / 6) * (10 / 5) * (15 / 30) * (20 / 10) * (45 / 90) * (50 / 25) * (60 / 120) = 1 := 
by
  sorry

end fraction_product_l1701_170169


namespace cassidy_total_grounding_days_l1701_170145

-- Define the initial grounding days
def initial_grounding_days : ℕ := 14

-- Define the grounding days per grade below a B
def extra_days_per_grade : ℕ := 3

-- Define the number of grades below a B
def grades_below_B : ℕ := 4

-- Define the total grounding days calculation
def total_grounding_days : ℕ := initial_grounding_days + grades_below_B * extra_days_per_grade

-- The theorem statement
theorem cassidy_total_grounding_days :
  total_grounding_days = 26 := 
sorry

end cassidy_total_grounding_days_l1701_170145


namespace activity_popularity_order_l1701_170111

-- Definitions for the fractions representing activity popularity
def dodgeball_popularity : Rat := 9 / 24
def magic_show_popularity : Rat := 4 / 12
def singing_contest_popularity : Rat := 1 / 3

-- Theorem stating the order of activities based on popularity
theorem activity_popularity_order :
  dodgeball_popularity > magic_show_popularity ∧ magic_show_popularity = singing_contest_popularity :=
by 
  sorry

end activity_popularity_order_l1701_170111


namespace number_of_packages_sold_l1701_170137

noncomputable def supplier_charges (P : ℕ) : ℕ :=
  if P ≤ 10 then 25 * P
  else 250 + 20 * (P - 10)

theorem number_of_packages_sold
  (supplier_received : ℕ)
  (percent_to_X : ℕ)
  (percent_to_Y : ℕ)
  (percent_to_Z : ℕ)
  (per_package_price : ℕ)
  (discount_percent : ℕ)
  (discount_threshold : ℕ)
  (P : ℕ)
  (h_received : supplier_received = 1340)
  (h_to_X : percent_to_X = 15)
  (h_to_Y : percent_to_Y = 15)
  (h_to_Z : percent_to_Z = 70)
  (h_full_price : per_package_price = 25)
  (h_discount : discount_percent = 4 * per_package_price / 5)
  (h_threshold : discount_threshold = 10)
  (h_calculation : supplier_charges P = supplier_received) : P = 65 := 
sorry

end number_of_packages_sold_l1701_170137


namespace find_min_positive_n_l1701_170131

-- Assume the sequence {a_n} is given
variables {a : ℕ → ℤ}

-- Given conditions
-- a4 < 0 and a5 > |a4|
def condition1 (a : ℕ → ℤ) : Prop := a 4 < 0
def condition2 (a : ℕ → ℤ) : Prop := a 5 > abs (a 4)

-- Sum of the first n terms of the arithmetic sequence
def S (n : ℕ) (a : ℕ → ℤ) : ℤ := n * (a 1 + a n) / 2

-- The main theorem we need to prove
theorem find_min_positive_n (a : ℕ → ℤ) (h1 : condition1 a) (h2 : condition2 a) : ∃ n : ℕ, n = 8 ∧ S n a > 0 :=
by
  sorry

end find_min_positive_n_l1701_170131


namespace problem_statement_l1701_170130

-- Definitions of the sets P and Q
def P : Set ℝ := {x : ℝ | x > 1}
def Q : Set ℝ := {x : ℝ | abs x > 0}

-- Statement of the problem to prove that P is not a subset of Q
theorem problem_statement : ¬ (P ⊆ Q) :=
sorry

end problem_statement_l1701_170130


namespace find_b_l1701_170109

theorem find_b (b : ℤ) (h : ∃ x : ℝ, x^2 + b * x - 35 = 0 ∧ x = 5) : b = 2 :=
sorry

end find_b_l1701_170109


namespace ratio_of_length_to_perimeter_is_one_over_four_l1701_170182

-- We define the conditions as given in the problem.
def room_length_1 : ℕ := 23 -- length of the rectangle in feet
def room_width_1 : ℕ := 15  -- width of the rectangle in feet
def room_width_2 : ℕ := 8   -- side of the square in feet

-- Total dimensions after including the square
def total_length : ℕ := room_length_1  -- total length remains the same
def total_width : ℕ := room_width_1 + room_width_2  -- width is sum of widths

-- Defining the perimeter
def perimeter (length width : ℕ) : ℕ := 2 * length + 2 * width

-- Calculate the ratio
def length_to_perimeter_ratio (length perimeter : ℕ) : ℚ := length / perimeter

-- Theorem to prove the desired ratio is 1:4
theorem ratio_of_length_to_perimeter_is_one_over_four : 
  length_to_perimeter_ratio total_length (perimeter total_length total_width) = 1 / 4 :=
by
  -- Proof code would go here
  sorry

end ratio_of_length_to_perimeter_is_one_over_four_l1701_170182


namespace max_f_value_l1701_170159

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 13 :=
sorry

end max_f_value_l1701_170159


namespace sum_lent_l1701_170198

theorem sum_lent (P : ℝ) (r t : ℝ) (I : ℝ) (h1 : r = 6) (h2 : t = 6) (h3 : I = P - 672) (h4 : I = P * r * t / 100) :
  P = 1050 := by
  sorry

end sum_lent_l1701_170198


namespace red_marbles_initial_count_l1701_170144

theorem red_marbles_initial_count (r g : ℕ) 
  (h1 : 3 * r = 5 * g)
  (h2 : 4 * (r - 18) = g + 27) :
  r = 29 :=
sorry

end red_marbles_initial_count_l1701_170144


namespace original_paint_intensity_l1701_170162

theorem original_paint_intensity 
  (P : ℝ)
  (H1 : 0 ≤ P ∧ P ≤ 100)
  (H2 : ∀ (unit : ℝ), unit = 100)
  (H3 : ∀ (replaced_fraction : ℝ), replaced_fraction = 1.5)
  (H4 : ∀ (new_intensity : ℝ), new_intensity = 30)
  (H5 : ∀ (solution_intensity : ℝ), solution_intensity = 0.25) :
  P = 15 := 
by
  sorry

end original_paint_intensity_l1701_170162


namespace absolute_value_equation_solution_l1701_170157

theorem absolute_value_equation_solution (a b c : ℝ) :
  (∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + |c * x + a * y + b * z| = |x| + |y| + |z|) ↔
  ((a = 0 ∧ b = 0 ∧ (c = 1 ∨ c = -1)) ∨ 
   (a = 0 ∧ c = 0 ∧ (b = 1 ∨ b = -1)) ∨ 
   (b = 0 ∧ c = 0 ∧ (a = 1 ∨ a = -1))) :=
by
  sorry

end absolute_value_equation_solution_l1701_170157


namespace chess_tournament_participants_l1701_170121

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 136) : n = 17 :=
by {
  sorry -- Proof will be here.
}

end chess_tournament_participants_l1701_170121


namespace similar_rect_tiling_l1701_170190

-- Define the dimensions of rectangles A and B
variables {a1 a2 b1 b2 : ℝ}

-- Define the tiling condition
def similar_tiled (a1 a2 b1 b2 : ℝ) : Prop := 
  -- A placeholder for the actual definition of similar tiling
  sorry

-- The main theorem to prove
theorem similar_rect_tiling (h : similar_tiled a1 a2 b1 b2) : similar_tiled b1 b2 a1 a2 :=
sorry

end similar_rect_tiling_l1701_170190


namespace handshakes_at_event_l1701_170165

theorem handshakes_at_event 
  (num_couples : ℕ) 
  (num_people : ℕ) 
  (num_handshakes_men : ℕ) 
  (num_handshakes_men_women : ℕ) 
  (total_handshakes : ℕ) 
  (cond1 : num_couples = 15) 
  (cond2 : num_people = 2 * num_couples) 
  (cond3 : num_handshakes_men = (num_couples * (num_couples - 1)) / 2) 
  (cond4 : num_handshakes_men_women = num_couples * (num_couples - 1)) 
  (cond5 : total_handshakes = num_handshakes_men + num_handshakes_men_women) : 
  total_handshakes = 315 := 
by sorry

end handshakes_at_event_l1701_170165


namespace cubic_identity_l1701_170138

theorem cubic_identity (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / x^3) = 140 := 
  sorry

end cubic_identity_l1701_170138


namespace balls_total_correct_l1701_170187

-- Definitions based on the problem conditions
def red_balls_initial : ℕ := 16
def blue_balls : ℕ := 2 * red_balls_initial
def red_balls_lost : ℕ := 6
def red_balls_remaining : ℕ := red_balls_initial - red_balls_lost
def total_balls_after : ℕ := 74
def nonblue_red_balls_remaining : ℕ := red_balls_remaining + blue_balls

-- Goal: Find the number of yellow balls
def yellow_balls_bought : ℕ := total_balls_after - nonblue_red_balls_remaining

theorem balls_total_correct :
  yellow_balls_bought = 32 :=
by
  -- Proof would go here
  sorry

end balls_total_correct_l1701_170187


namespace b_bound_for_tangent_parallel_l1701_170124

theorem b_bound_for_tangent_parallel (b : ℝ) (c : ℝ) :
  (∃ x : ℝ, 3 * x^2 - x + b = 0) → b ≤ 1/12 :=
by
  intros h
  -- Placeholder proof
  sorry

end b_bound_for_tangent_parallel_l1701_170124


namespace trapezoid_area_calc_l1701_170156

noncomputable def isoscelesTrapezoidArea : ℝ :=
  let a := 1
  let b := 9
  let h := 2 * Real.sqrt 3
  0.5 * (a + b) * h

theorem trapezoid_area_calc : isoscelesTrapezoidArea = 20 * Real.sqrt 3 := by
  sorry

end trapezoid_area_calc_l1701_170156


namespace g_expression_f_expression_l1701_170134

-- Given functions f and g that satisfy the conditions
variable {f g : ℝ → ℝ}

-- Conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom sum_eq : ∀ x, f x + g x = 2^x + 2 * x

-- Theorem statements to prove
theorem g_expression : g = fun x => 2^x := by sorry
theorem f_expression : f = fun x => 2 * x := by sorry

end g_expression_f_expression_l1701_170134


namespace find_a_l1701_170168

theorem find_a (b c : ℤ) 
  (vertex_condition : ∀ (x : ℝ), x = -1 → (ax^2 + b*x + c) = -2)
  (point_condition : ∀ (x : ℝ), x = 0 → (a*x^2 + b*x + c) = -1) :
  ∃ (a : ℤ), a = 1 :=
by
  sorry

end find_a_l1701_170168


namespace log_property_l1701_170167

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem log_property (m n : ℝ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m + f n :=
by
  sorry

end log_property_l1701_170167


namespace satellite_modular_units_l1701_170158

variables (N S T U : ℕ)
variable (h1 : N = S / 3)
variable (h2 : S / T = 1 / 9)
variable (h3 : U * N = 8 * T / 9)

theorem satellite_modular_units :
  U = 24 :=
by sorry

end satellite_modular_units_l1701_170158


namespace find_two_digit_number_l1701_170199

-- Define the problem conditions and statement
theorem find_two_digit_number (a b n : ℕ) (h1 : a = 2 * b) (h2 : 10 * a + b + a^2 = n^2) : 
  10 * a + b = 21 :=
sorry

end find_two_digit_number_l1701_170199


namespace inequality_holds_l1701_170128

theorem inequality_holds (x : ℝ) (n : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 1) (h3 : n > 0) : 
  (1 + x) ^ n ≥ (1 - x) ^ n + 2 * n * x * (1 - x ^ 2) ^ ((n - 1) / 2) :=
sorry

end inequality_holds_l1701_170128


namespace mineral_sample_ages_l1701_170193

/--
We have a mineral sample with digits {2, 2, 3, 3, 5, 9}.
Given the condition that the age must start with an odd number,
we need to prove that the total number of possible ages is 120.
-/
theorem mineral_sample_ages : 
  ∀ (l : List ℕ), l = [2, 2, 3, 3, 5, 9] → 
  (l.filter odd).length > 0 →
  ∃ n : ℕ, n = 120 :=
by
  intros l h_digits h_odd
  sorry

end mineral_sample_ages_l1701_170193


namespace compute_value_condition_l1701_170148

theorem compute_value_condition (x : ℝ) (h : x + (1 / x) = 3) :
  (x - 2) ^ 2 + 25 / (x - 2) ^ 2 = -x + 5 := by
  sorry

end compute_value_condition_l1701_170148


namespace car_speed_l1701_170105

theorem car_speed (v : ℝ) (hv : 2 + (1 / v) * 3600 = (1 / 90) * 3600) :
  v = 600 / 7 :=
sorry

end car_speed_l1701_170105


namespace sum_of_exponents_l1701_170194

theorem sum_of_exponents (n : ℕ) (h : n = 896) : 
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 2^a + 2^b + 2^c = n ∧ a + b + c = 24 :=
by
  sorry

end sum_of_exponents_l1701_170194


namespace store_owner_marked_price_l1701_170196

theorem store_owner_marked_price (L M : ℝ) (h1 : M = (56 / 45) * L) : M / L = 124.44 / 100 :=
by
  sorry

end store_owner_marked_price_l1701_170196


namespace hypotenuse_is_2_sqrt_25_point_2_l1701_170142

open Real

noncomputable def hypotenuse_length_of_right_triangle (ma mb : ℝ) (a b c : ℝ) : ℝ :=
  if h1 : ma = 6 ∧ mb = sqrt 27 then
    c
  else
    0

theorem hypotenuse_is_2_sqrt_25_point_2 :
  hypotenuse_length_of_right_triangle 6 (sqrt 27) a b (2 * sqrt 25.2) = 2 * sqrt 25.2 :=
by
  sorry -- proof to be filled

end hypotenuse_is_2_sqrt_25_point_2_l1701_170142


namespace f_1_eq_zero_l1701_170104

-- Given a function f with the specified properties
variable {f : ℝ → ℝ}

-- Given 1) the domain of the function
axiom domain_f : ∀ x, (x < 0 ∨ x > 0) → true 

-- Given 2) the functional equation
axiom functional_eq_f : ∀ x₁ x₂, (x₁ < 0 ∨ x₁ > 0) ∧ (x₂ < 0 ∨ x₂ > 0) → f (x₁ * x₂) = f x₁ + f x₂

-- Prove that f(1) = 0
theorem f_1_eq_zero : f 1 = 0 := 
  sorry

end f_1_eq_zero_l1701_170104


namespace find_m_l1701_170112

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 5
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 9

theorem find_m (m : ℝ) : f 5 - g 5 m = 20 → m = -16.8 :=
by
  -- Given f(x) and g(x, m) definitions, we want to prove m = -16.8 given f 5 - g 5 m = 20.
  sorry

end find_m_l1701_170112


namespace squared_remainder_l1701_170136

theorem squared_remainder (N : ℤ) (k : ℤ) :
  (N % 9 = 2 ∨ N % 9 = 7) → 
  (N^2 % 9 = 4) :=
by
  sorry

end squared_remainder_l1701_170136


namespace sum_of_interior_edges_l1701_170151

noncomputable def interior_edge_sum (outer_length : ℝ) (wood_width : ℝ) (frame_area : ℝ) : ℝ := 
  let outer_width := (frame_area + 3 * (outer_length - 2 * wood_width) * 4) / outer_length
  let inner_length := outer_length - 2 * wood_width
  let inner_width := outer_width - 2 * wood_width
  2 * inner_length + 2 * inner_width

theorem sum_of_interior_edges :
  interior_edge_sum 7 2 34 = 9 := by
  sorry

end sum_of_interior_edges_l1701_170151


namespace speed_ratio_of_runners_l1701_170186

theorem speed_ratio_of_runners (v_A v_B : ℝ) (c : ℝ)
  (h1 : 0 < v_A ∧ 0 < v_B) -- They run at constant, but different speeds
  (h2 : (v_B / v_A) = (2 / 3)) -- Distance relationship from meeting points
  : v_B / v_A = 2 :=
by
  sorry

end speed_ratio_of_runners_l1701_170186


namespace math_problem_l1701_170153

theorem math_problem (x y : ℤ) (a b : ℤ) (h1 : x - 5 = 7 * a) (h2 : y + 7 = 7 * b) (h3 : (x ^ 2 + y ^ 3) % 11 = 0) : 
  ((y - x) / 13) = 13 :=
sorry

end math_problem_l1701_170153
