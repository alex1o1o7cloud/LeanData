import Mathlib

namespace NUMINAMATH_GPT_range_of_m_l1731_173105

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x1 : ℝ, 0 < x1 ∧ x1 < 3 / 2 → ∃ x2 : ℝ, 0 < x2 ∧ x2 < 3 / 2 ∧ f x1 > g x2) →
  (∀ x : ℝ, f x = -x + x * Real.log x + m) →
  (∀ x : ℝ, g x = -3 * Real.exp x / (3 + 4 * x ^ 2)) →
  m > 1 - 3 / 4 * Real.sqrt (Real.exp 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1731_173105


namespace NUMINAMATH_GPT_water_depth_is_60_l1731_173165

def Ron_height : ℕ := 12
def depth_of_water (h_R : ℕ) : ℕ := 5 * h_R

theorem water_depth_is_60 : depth_of_water Ron_height = 60 :=
by
  sorry

end NUMINAMATH_GPT_water_depth_is_60_l1731_173165


namespace NUMINAMATH_GPT_sum_of_first_12_terms_l1731_173112

noncomputable def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

def Sn (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_12_terms (a d : ℤ) (h1 : a + d * 4 = 3 * (a + d * 2))
                             (h2 : a + d * 9 = 14) : Sn a d 12 = 84 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_first_12_terms_l1731_173112


namespace NUMINAMATH_GPT_determine_d_and_vertex_l1731_173102

-- Definition of the quadratic equation
def g (x d : ℝ) : ℝ := 3 * x^2 + 12 * x + d

-- The proof problem
theorem determine_d_and_vertex (d : ℝ) :
  (∃ x : ℝ, g x d = 0 ∧ ∀ y : ℝ, g y d ≥ g x d) ↔ (d = 12 ∧ ∀ x : ℝ, 3 > 0 ∧ (g x d ≥ g 0 d)) := 
by 
  sorry

end NUMINAMATH_GPT_determine_d_and_vertex_l1731_173102


namespace NUMINAMATH_GPT_counterexample_disproves_statement_l1731_173111

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem counterexample_disproves_statement :
  ∃ n : ℕ, ¬ is_prime n ∧ is_prime (n + 3) :=
  by
    use 8
    -- Proof that 8 is not prime
    -- Proof that 11 (8 + 3) is prime
    sorry

end NUMINAMATH_GPT_counterexample_disproves_statement_l1731_173111


namespace NUMINAMATH_GPT_range_of_c_over_a_l1731_173163

theorem range_of_c_over_a (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + 2 * b + c = 0) :
    -3 < c / a ∧ c / a < -(1 / 3) := 
sorry

end NUMINAMATH_GPT_range_of_c_over_a_l1731_173163


namespace NUMINAMATH_GPT_factorize_2mn_cube_arithmetic_calculation_l1731_173136

-- Problem 1: Factorization problem
theorem factorize_2mn_cube (m n : ℝ) : 
  2 * m^3 * n - 8 * m * n^3 = 2 * m * n * (m + 2 * n) * (m - 2 * n) :=
by sorry

-- Problem 2: Arithmetic calculation problem
theorem arithmetic_calculation : 
  |1 - Real.sqrt 3| + 3 * Real.tan (Real.pi / 6) - ((Real.pi - 3)^0) + (-1/3)⁻¹ = 2 * Real.sqrt 3 - 5 :=
by sorry

end NUMINAMATH_GPT_factorize_2mn_cube_arithmetic_calculation_l1731_173136


namespace NUMINAMATH_GPT_water_addition_to_achieve_concentration_l1731_173162

theorem water_addition_to_achieve_concentration :
  ∀ (w1 w2 : ℝ), 
  (60 * 0.25 = 15) →              -- initial amount of acid
  (15 / (60 + w1) = 0.15) →       -- first dilution to 15%
  (15 / (100 + w2) = 0.10) →      -- second dilution to 10%
  w1 + w2 = 90 :=                 -- total water added to achieve final concentration
by
  intros w1 w2 h_initial h_first h_second
  sorry

end NUMINAMATH_GPT_water_addition_to_achieve_concentration_l1731_173162


namespace NUMINAMATH_GPT_find_smallest_number_l1731_173150

theorem find_smallest_number (x y z : ℝ) 
  (h1 : x + y + z = 150) 
  (h2 : y = 3 * x + 10) 
  (h3 : z = x^2 - 5) 
  : x = 10.21 :=
sorry

end NUMINAMATH_GPT_find_smallest_number_l1731_173150


namespace NUMINAMATH_GPT_total_legs_camden_dogs_l1731_173168

variable (c r j : ℕ) -- c: Camden's dogs, r: Rico's dogs, j: Justin's dogs

theorem total_legs_camden_dogs :
  (r = j + 10) ∧ (j = 14) ∧ (c = (3 * r) / 4) → 4 * c = 72 :=
by
  sorry

end NUMINAMATH_GPT_total_legs_camden_dogs_l1731_173168


namespace NUMINAMATH_GPT_complex_sum_real_imag_l1731_173183

theorem complex_sum_real_imag : 
  (Complex.re ((Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I))) + 
  Complex.im ((Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I)))) = 3/2 := 
by sorry

end NUMINAMATH_GPT_complex_sum_real_imag_l1731_173183


namespace NUMINAMATH_GPT_chess_tournament_third_place_wins_l1731_173138

theorem chess_tournament_third_place_wins :
  ∀ (points : Fin 8 → ℕ)
  (total_games : ℕ)
  (total_points : ℕ),
  (total_games = 28) →
  (∀ i j : Fin 8, i ≠ j → points i ≠ points j) →
  ((points 1) = (points 4 + points 5 + points 6 + points 7)) →
  (points 2 > points 4) →
  ∃ (games_won : Fin 8 → Fin 8 → Prop),
  (games_won 2 4) :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_third_place_wins_l1731_173138


namespace NUMINAMATH_GPT_randy_trip_distance_l1731_173122

theorem randy_trip_distance (x : ℝ) (h1 : x = x / 4 + 30 + x / 10 + (x - (x / 4 + 30 + x / 10))) :
  x = 60 :=
by {
  sorry -- Placeholder for the actual proof
}

end NUMINAMATH_GPT_randy_trip_distance_l1731_173122


namespace NUMINAMATH_GPT_daily_evaporation_l1731_173149

theorem daily_evaporation (initial_water: ℝ) (days: ℝ) (evap_percentage: ℝ) : 
  initial_water = 10 → days = 50 → evap_percentage = 2 →
  (initial_water * evap_percentage / 100) / days = 0.04 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_daily_evaporation_l1731_173149


namespace NUMINAMATH_GPT_prove_divisibility_l1731_173137

-- Definitions for natural numbers m, n, k
variables (m n k : ℕ)

-- Conditions stating divisibility
def div1 := m^n ∣ n^m
def div2 := n^k ∣ k^n

-- The final theorem to prove
theorem prove_divisibility (hmn : div1 m n) (hnk : div2 n k) : m^k ∣ k^m :=
sorry

end NUMINAMATH_GPT_prove_divisibility_l1731_173137


namespace NUMINAMATH_GPT_average_speed_round_trip_l1731_173175

theorem average_speed_round_trip
  (n : ℕ)
  (distance_km : ℝ := n / 1000)
  (pace_west_min_per_km : ℝ := 2)
  (speed_east_kmh : ℝ := 3)
  (wait_time_hr : ℝ := 30 / 60) :
  (2 * distance_km) / 
  ((pace_west_min_per_km * distance_km / 60) + wait_time_hr + (distance_km / speed_east_kmh)) = 
  60 * n / (11 * n + 150000) := by
  sorry

end NUMINAMATH_GPT_average_speed_round_trip_l1731_173175


namespace NUMINAMATH_GPT_total_coins_l1731_173110

-- Defining the conditions
def stack1 : Nat := 4
def stack2 : Nat := 8

-- Statement of the proof problem
theorem total_coins : stack1 + stack2 = 12 :=
by
  sorry

end NUMINAMATH_GPT_total_coins_l1731_173110


namespace NUMINAMATH_GPT_find_n_l1731_173119

theorem find_n (n : ℕ) (h1 : 0 ≤ n ∧ n ≤ 360) (h2 : Real.cos (n * Real.pi / 180) = Real.cos (340 * Real.pi / 180)) : 
  n = 20 ∨ n = 340 := 
by
  sorry

end NUMINAMATH_GPT_find_n_l1731_173119


namespace NUMINAMATH_GPT_jill_salary_l1731_173158

-- Defining the conditions
variables (S : ℝ) -- Jill's net monthly salary
variables (discretionary_income : ℝ) -- One fifth of her net monthly salary
variables (vacation_fund : ℝ) -- 30% of discretionary income into a vacation fund
variables (savings : ℝ) -- 20% of discretionary income into savings
variables (eating_out_socializing : ℝ) -- 35% of discretionary income on eating out and socializing
variables (leftover : ℝ) -- The remaining amount, which is $99

-- Given Conditions
-- One fifth of her net monthly salary left as discretionary income
def one_fifth_of_salary : Prop := discretionary_income = (1/5) * S

-- 30% into a vacation fund
def vacation_allocation : Prop := vacation_fund = 0.30 * discretionary_income

-- 20% into savings
def savings_allocation : Prop := savings = 0.20 * discretionary_income

-- 35% on eating out and socializing
def socializing_allocation : Prop := eating_out_socializing = 0.35 * discretionary_income

-- This leaves her with $99
def leftover_amount : Prop := leftover = 99

-- Eqution considering all conditions results her leftover being $99
def income_allocation : Prop := 
  vacation_fund + savings + eating_out_socializing + leftover = discretionary_income

-- The main proof goal: given all the conditions, Jill's net monthly salary is $3300
theorem jill_salary : 
  one_fifth_of_salary S discretionary_income → 
  vacation_allocation discretionary_income vacation_fund → 
  savings_allocation discretionary_income savings → 
  socializing_allocation discretionary_income eating_out_socializing → 
  leftover_amount leftover → 
  income_allocation discretionary_income vacation_fund savings eating_out_socializing leftover → 
  S = 3300 := by sorry

end NUMINAMATH_GPT_jill_salary_l1731_173158


namespace NUMINAMATH_GPT_area_of_rhombus_l1731_173132

-- Given values for the diagonals of a rhombus.
def d1 : ℝ := 14
def d2 : ℝ := 24

-- The target statement we want to prove.
theorem area_of_rhombus : (d1 * d2) / 2 = 168 := by
  sorry

end NUMINAMATH_GPT_area_of_rhombus_l1731_173132


namespace NUMINAMATH_GPT_units_digit_of_calculation_l1731_173197

-- Base definitions for units digits of given numbers
def units_digit (n : ℕ) : ℕ := n % 10

-- Main statement to prove
theorem units_digit_of_calculation : 
  units_digit ((25 ^ 3 + 17 ^ 3) * 12 ^ 2) = 2 :=
by
  -- This is where the proof would go, but it's omitted as requested
  sorry

end NUMINAMATH_GPT_units_digit_of_calculation_l1731_173197


namespace NUMINAMATH_GPT_original_card_count_l1731_173100

theorem original_card_count
  (r b : ℕ)
  (initial_prob_red : (r : ℚ) / (r + b) = 2 / 5)
  (prob_red_after_adding_black : (r : ℚ) / (r + (b + 6)) = 1 / 3) :
  r + b = 30 := sorry

end NUMINAMATH_GPT_original_card_count_l1731_173100


namespace NUMINAMATH_GPT_remainder_proof_l1731_173115

theorem remainder_proof (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v ∧ v < y) : 
  (x + y * u^2 + 3 * v) % y = 4 * v % y :=
by
  sorry

end NUMINAMATH_GPT_remainder_proof_l1731_173115


namespace NUMINAMATH_GPT_remainder_of_m_div_5_l1731_173178

theorem remainder_of_m_div_5 (m n : ℕ) (h1 : m = 15 * n - 1) (h2 : n > 0) : m % 5 = 4 :=
sorry

end NUMINAMATH_GPT_remainder_of_m_div_5_l1731_173178


namespace NUMINAMATH_GPT_positive_expression_with_b_l1731_173109

-- Defining the conditions and final statement
open Real

theorem positive_expression_with_b (a : ℝ) : (a + 2) * (a + 5) * (a + 8) * (a + 11) + 82 > 0 := 
sorry

end NUMINAMATH_GPT_positive_expression_with_b_l1731_173109


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1731_173190

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 - 1 = 0) ↔ (x = -1 ∨ x = 1) ∧ (x - 1 = 0) → (x^2 - 1 = 0) ∧ ¬((x^2 - 1 = 0) → (x - 1 = 0)) := 
by sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1731_173190


namespace NUMINAMATH_GPT_time_ratio_school_home_l1731_173169

open Real

noncomputable def time_ratio (y x : ℝ) : ℝ :=
  let time_school := (y / (3 * x)) + (2 * y / (2 * x)) + (y / (4 * x))
  let time_home := (y / (4 * x)) + (2 * y / (2 * x)) + (y / (3 * x))
  time_school / time_home

theorem time_ratio_school_home (y x : ℝ) (hy : y ≠ 0) (hx : x ≠ 0) : time_ratio y x = 19 / 16 :=
  sorry

end NUMINAMATH_GPT_time_ratio_school_home_l1731_173169


namespace NUMINAMATH_GPT_x_add_inv_ge_two_l1731_173177

theorem x_add_inv_ge_two (x : ℝ) (hx : x > 0) : x + (1 / x) ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_x_add_inv_ge_two_l1731_173177


namespace NUMINAMATH_GPT_james_income_ratio_l1731_173174

theorem james_income_ratio
  (January_earnings : ℕ := 4000)
  (Total_earnings : ℕ := 18000)
  (Earnings_difference : ℕ := 2000) :
  ∃ (February_earnings : ℕ), 
    (January_earnings + February_earnings + (February_earnings - Earnings_difference) = Total_earnings) ∧
    (February_earnings / January_earnings = 2) := by
  sorry

end NUMINAMATH_GPT_james_income_ratio_l1731_173174


namespace NUMINAMATH_GPT_total_boys_in_class_l1731_173198

/-- 
  Given 
    - n + 1 positions in a circle, where n is the number of boys and 1 position for the teacher.
    - The boy at the 6th position is exactly opposite to the boy at the 16th position.
  Prove that the total number of boys in the class is 20.
-/
theorem total_boys_in_class (n : ℕ) (h1 : n + 1 > 16) (h2 : (6 + 10) % (n + 1) = 16):
  n = 20 := 
by 
  sorry

end NUMINAMATH_GPT_total_boys_in_class_l1731_173198


namespace NUMINAMATH_GPT_least_add_to_divisible_by_17_l1731_173159

/-- Given that the remainder when 433124 is divided by 17 is 2,
    prove that the least number that must be added to 433124 to make 
    it divisible by 17 is 15. -/
theorem least_add_to_divisible_by_17: 
  (433124 % 17 = 2) → 
  (∃ n, n ≥ 0 ∧ (433124 + n) % 17 = 0 ∧ n = 15) := 
by
  sorry

end NUMINAMATH_GPT_least_add_to_divisible_by_17_l1731_173159


namespace NUMINAMATH_GPT_balance_squares_circles_l1731_173144

theorem balance_squares_circles (x y z : ℕ) (h1 : 5 * x + 2 * y = 21 * z) (h2 : 2 * x = y + 3 * z) : 
  3 * y = 9 * z :=
by 
  sorry

end NUMINAMATH_GPT_balance_squares_circles_l1731_173144


namespace NUMINAMATH_GPT_figure_Z_has_largest_shaded_area_l1731_173151

noncomputable def shaded_area_X :=
  let rectangle_area := 4 * 2
  let circle_area := Real.pi * (1)^2
  rectangle_area - circle_area

noncomputable def shaded_area_Y :=
  let rectangle_area := 4 * 2
  let semicircle_area := (1 / 2) * Real.pi * (1)^2
  rectangle_area - semicircle_area

noncomputable def shaded_area_Z :=
  let outer_square_area := 4^2
  let inner_square_area := 2^2
  outer_square_area - inner_square_area

theorem figure_Z_has_largest_shaded_area :
  shaded_area_Z > shaded_area_X ∧ shaded_area_Z > shaded_area_Y :=
by
  sorry

end NUMINAMATH_GPT_figure_Z_has_largest_shaded_area_l1731_173151


namespace NUMINAMATH_GPT_even_parts_impossible_odd_parts_possible_l1731_173166

theorem even_parts_impossible (n m : ℕ) (h₁ : n = 1) (h₂ : ∀ k, m = n + 2 * k) : n + 2 * m ≠ 100 := by
  -- Proof omitted
  sorry

theorem odd_parts_possible (n m : ℕ) (h₁ : n = 1) (h₂ : ∀ k, m = n + 2 * k) : ∃ k, n + 2 * k = 2017 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_even_parts_impossible_odd_parts_possible_l1731_173166


namespace NUMINAMATH_GPT_number_of_students_to_bring_donuts_l1731_173121

theorem number_of_students_to_bring_donuts (students_brownies students_cookies students_donuts : ℕ) :
  (students_brownies * 12 * 2) + (students_cookies * 24 * 2) + (students_donuts * 12 * 2) = 2040 →
  students_brownies = 30 →
  students_cookies = 20 →
  students_donuts = 15 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_number_of_students_to_bring_donuts_l1731_173121


namespace NUMINAMATH_GPT_find_tan_angle_F2_F1_B_l1731_173135

-- Definitions for the points and chord lengths
def F1 : Type := ℝ × ℝ
def F2 : Type := ℝ × ℝ
def A : Type := ℝ × ℝ
def B : Type := ℝ × ℝ

-- Given distances
def F1A : ℝ := 3
def AB : ℝ := 4
def BF1 : ℝ := 5

-- The angle we want to find the tangent of
def angle_F2_F1_B (F1 F2 A B : Type) : ℝ := sorry -- Placeholder for angle calculation

-- The main theorem to prove
theorem find_tan_angle_F2_F1_B (F1 F2 A B : Type) (F1A_dist : F1A = 3) (AB_dist : AB = 4) (BF1_dist : BF1 = 5) :
  angle_F2_F1_B F1 F2 A B = 1 / 7 :=
sorry

end NUMINAMATH_GPT_find_tan_angle_F2_F1_B_l1731_173135


namespace NUMINAMATH_GPT_tangent_line_ellipse_l1731_173185

theorem tangent_line_ellipse (a b x y x₀ y₀ : ℝ) (h : a > 0) (hb : b > 0) (ha_gt_hb : a > b) 
(h_on_ellipse : (x₀^2 / a^2) + (y₀^2 / b^2) = 1) :
    (x₀ * x / a^2) + (y₀ * y / b^2) = 1 := 
sorry

end NUMINAMATH_GPT_tangent_line_ellipse_l1731_173185


namespace NUMINAMATH_GPT_lily_milk_left_l1731_173172

theorem lily_milk_left : 
  let initial_milk := 5 
  let given_to_james := 18 / 7
  ∃ r : ℚ, r = 2 + 3/7 ∧ (initial_milk - given_to_james) = r :=
by
  sorry

end NUMINAMATH_GPT_lily_milk_left_l1731_173172


namespace NUMINAMATH_GPT_f_at_4_l1731_173154

-- Define the conditions on the function f
variable (f : ℝ → ℝ)
variable (h_domain : true) -- All ℝ → ℝ functions have ℝ as their domain.

-- f is an odd function
axiom h_odd : ∀ x : ℝ, f (-x) = -f x

-- Given functional equation
axiom h_eqn : ∀ x : ℝ, f (2 * x - 3) - 2 * f (3 * x - 10) + f (x - 3) = 28 - 6 * x 

-- The goal is to determine the value of f(4), which should be 8.
theorem f_at_4 : f 4 = 8 :=
sorry

end NUMINAMATH_GPT_f_at_4_l1731_173154


namespace NUMINAMATH_GPT_num_five_digit_integers_l1731_173194

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

theorem num_five_digit_integers : 
  let num_ways := factorial 5 / (factorial 2 * factorial 3)
  num_ways = 10 :=
by 
  sorry

end NUMINAMATH_GPT_num_five_digit_integers_l1731_173194


namespace NUMINAMATH_GPT_money_distribution_l1731_173161

theorem money_distribution (Maggie_share : ℝ) (fraction_Maggie : ℝ) (total_sum : ℝ) :
  Maggie_share = 7500 →
  fraction_Maggie = (1/8) →
  total_sum = Maggie_share / fraction_Maggie →
  total_sum = 60000 :=
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  linarith

end NUMINAMATH_GPT_money_distribution_l1731_173161


namespace NUMINAMATH_GPT_time_to_pass_bridge_l1731_173114

noncomputable def train_length : Real := 357
noncomputable def speed_km_per_hour : Real := 42
noncomputable def bridge_length : Real := 137

noncomputable def speed_m_per_s : Real := speed_km_per_hour * (1000 / 3600)

noncomputable def total_distance : Real := train_length + bridge_length

noncomputable def time_to_pass : Real := total_distance / speed_m_per_s

theorem time_to_pass_bridge : abs (time_to_pass - 42.33) < 0.01 :=
sorry

end NUMINAMATH_GPT_time_to_pass_bridge_l1731_173114


namespace NUMINAMATH_GPT_exists_directed_triangle_l1731_173176

structure Tournament (V : Type) :=
  (edges : V → V → Prop)
  (complete : ∀ x y, x ≠ y → edges x y ∨ edges y x)
  (outdegree_at_least_one : ∀ x, ∃ y, edges x y)

theorem exists_directed_triangle {V : Type} [Fintype V] (T : Tournament V) :
  ∃ (a b c : V), T.edges a b ∧ T.edges b c ∧ T.edges c a := by
sorry

end NUMINAMATH_GPT_exists_directed_triangle_l1731_173176


namespace NUMINAMATH_GPT_min_value_b_l1731_173184

noncomputable def f (x a : ℝ) := 3 * x^2 - 4 * a * x
noncomputable def g (x a b : ℝ) := 2 * a^2 * Real.log x - b
noncomputable def f' (x a : ℝ) := 6 * x - 4 * a
noncomputable def g' (x a : ℝ) := 2 * a^2 / x

theorem min_value_b (a : ℝ) (h_a : a > 0) :
  ∃ (b : ℝ), ∃ (x₀ : ℝ), 
  (f x₀ a = g x₀ a b ∧ f' x₀ a = g' x₀ a) ∧ 
  ∀ (b' : ℝ), (∀ (x' : ℝ), (f x' a = g x' a b' ∧ f' x' a = g' x' a) → b' ≥ -1 / Real.exp 2) := 
sorry

end NUMINAMATH_GPT_min_value_b_l1731_173184


namespace NUMINAMATH_GPT_verify_trees_in_other_row_l1731_173188

-- Definition of a normal lemon tree lemon production per year
def normalLemonTreeProduction : ℕ := 60

-- Definition of the percentage increase in lemon production for specially engineered lemon trees
def percentageIncrease : ℕ := 50

-- Definition of lemon production for specially engineered lemon trees
def specialLemonTreeProduction : ℕ := normalLemonTreeProduction * (1 + percentageIncrease / 100)

-- Number of trees in one row of the grove
def treesInOneRow : ℕ := 50

-- Total lemon production in 5 years
def totalLemonProduction : ℕ := 675000

-- Number of years
def years : ℕ := 5

-- Total number of trees in the grove
def totalNumberOfTrees : ℕ := totalLemonProduction / (specialLemonTreeProduction * years)

-- Number of trees in the other row
def treesInOtherRow : ℕ := totalNumberOfTrees - treesInOneRow

-- Theorem: Verification of the number of trees in the other row
theorem verify_trees_in_other_row : treesInOtherRow = 1450 :=
  by
  -- Proof logic is omitted, leaving as sorry
  sorry

end NUMINAMATH_GPT_verify_trees_in_other_row_l1731_173188


namespace NUMINAMATH_GPT_eval_expr_l1731_173140

theorem eval_expr (b c : ℕ) (hb : b = 2) (hc : c = 5) : b^3 * b^4 * c^2 = 3200 :=
by {
  -- the proof is omitted
  sorry
}

end NUMINAMATH_GPT_eval_expr_l1731_173140


namespace NUMINAMATH_GPT_inequality_false_implies_range_of_a_l1731_173129

theorem inequality_false_implies_range_of_a (a : ℝ) : 
  (∀ t : ℝ, t^2 - 2 * t - a ≥ 0) ↔ a ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_false_implies_range_of_a_l1731_173129


namespace NUMINAMATH_GPT_cube_path_count_l1731_173155

noncomputable def numberOfWaysToMoveOnCube : Nat :=
  20

theorem cube_path_count :
  ∀ (cube : Type) (top bottom side1 side2 side3 side4 : cube),
    (∀ (p : cube → cube → Prop), 
      (p top side1 ∨ p top side2 ∨ p top side3 ∨ p top side4) ∧ 
      (p side1 bottom ∨ p side2 bottom ∨ p side3 bottom ∨ p side4 bottom)) →
    numberOfWaysToMoveOnCube = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cube_path_count_l1731_173155


namespace NUMINAMATH_GPT_sqrt_range_l1731_173143

theorem sqrt_range (x : ℝ) : (1 - x ≥ 0) ↔ (x ≤ 1) := sorry

end NUMINAMATH_GPT_sqrt_range_l1731_173143


namespace NUMINAMATH_GPT_tile_in_center_l1731_173101

-- Define the coloring pattern of the grid
inductive Color
| A | B | C

-- Predicates for grid, tile placement, and colors
def Grid := Fin 5 × Fin 5

def is_1x3_tile (t : Grid × Grid × Grid) : Prop :=
  -- Ensure each tuple t represents three cells that form a $1 \times 3$ tile
  sorry

def is_tiling (g : Grid → Option Color) : Prop :=
  -- Ensure the entire grid is correctly tiled with the given tiles and within the coloring pattern
  sorry

def center : Grid := (Fin.mk 2 (by decide), Fin.mk 2 (by decide))

-- The theorem statement
theorem tile_in_center (g : Grid → Option Color) : is_tiling g → 
  (∃! tile : Grid, g tile = some Color.B) :=
sorry

end NUMINAMATH_GPT_tile_in_center_l1731_173101


namespace NUMINAMATH_GPT_discount_is_15_point_5_percent_l1731_173108

noncomputable def wholesale_cost (W : ℝ) := W
noncomputable def retail_price (W : ℝ) := 1.5384615384615385 * W
noncomputable def selling_price (W : ℝ) := 1.3 * W
noncomputable def discount_percentage (W : ℝ) := 
  let D := retail_price W - selling_price W
  (D / retail_price W) * 100

theorem discount_is_15_point_5_percent (W : ℝ) (hW : W > 0) : 
  discount_percentage W = 15.5 := 
by 
  sorry

end NUMINAMATH_GPT_discount_is_15_point_5_percent_l1731_173108


namespace NUMINAMATH_GPT_fraction_simplify_l1731_173124

theorem fraction_simplify (x : ℝ) (hx : x ≠ 1) (hx_ne_1 : x ≠ -1) :
  (x^2 - 1) / (x^2 - 2 * x + 1) = (x + 1) / (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplify_l1731_173124


namespace NUMINAMATH_GPT_product_of_three_consecutive_cubes_divisible_by_504_l1731_173123

theorem product_of_three_consecutive_cubes_divisible_by_504 (a : ℤ) : 
  ∃ k : ℤ, (a^3 - 1) * a^3 * (a^3 + 1) = 504 * k :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_product_of_three_consecutive_cubes_divisible_by_504_l1731_173123


namespace NUMINAMATH_GPT_probability_of_collinear_dots_l1731_173171

theorem probability_of_collinear_dots (dots : ℕ) (rows : ℕ) (columns : ℕ) (choose : ℕ → ℕ → ℕ) :
  dots = 20 ∧ rows = 5 ∧ columns = 4 ∧ choose 20 4 = 4845 → 
  (∃ sets_of_collinear_dots : ℕ, sets_of_collinear_dots = 20 ∧ 
   ∃ probability : ℚ,  probability = 4 / 969) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_collinear_dots_l1731_173171


namespace NUMINAMATH_GPT_football_points_difference_l1731_173113

theorem football_points_difference :
  let points_per_touchdown := 7
  let brayden_gavin_touchdowns := 7
  let cole_freddy_touchdowns := 9
  let brayden_gavin_points := brayden_gavin_touchdowns * points_per_touchdown
  let cole_freddy_points := cole_freddy_touchdowns * points_per_touchdown
  cole_freddy_points - brayden_gavin_points = 14 :=
by sorry

end NUMINAMATH_GPT_football_points_difference_l1731_173113


namespace NUMINAMATH_GPT_range_of_a_l1731_173141

theorem range_of_a {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h1 : x + y + 4 = 2 * x * y) (h2 : ∀ (x y : ℝ), x^2 + 2*x*y + y^2 - a*x - a*y + 1 ≥ 0) :
  a ≤ 17/4 := sorry

end NUMINAMATH_GPT_range_of_a_l1731_173141


namespace NUMINAMATH_GPT_largest_x_satisfies_condition_l1731_173127

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end NUMINAMATH_GPT_largest_x_satisfies_condition_l1731_173127


namespace NUMINAMATH_GPT_total_students_in_both_classrooms_l1731_173128

theorem total_students_in_both_classrooms
  (x y : ℕ)
  (hx1 : 80 * x - 250 = 90 * (x - 5))
  (hy1 : 85 * y - 480 = 95 * (y - 8)) :
  x + y = 48 := 
sorry

end NUMINAMATH_GPT_total_students_in_both_classrooms_l1731_173128


namespace NUMINAMATH_GPT_calculate_expression_l1731_173104

theorem calculate_expression : 2 * (-3)^3 - 4 * (-3) + 15 = -27 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1731_173104


namespace NUMINAMATH_GPT_range_of_m_hyperbola_l1731_173142

noncomputable def is_conic_hyperbola (expr : ℝ → ℝ → ℝ) : Prop :=
  ∃ f : ℝ, ∀ x y, expr x y = ((x - 2 * y + 3)^2 - f * (x^2 + y^2 + 2 * y + 1))

theorem range_of_m_hyperbola (m : ℝ) :
  is_conic_hyperbola (fun x y => m * (x^2 + y^2 + 2 * y + 1) - (x - 2 * y + 3)^2) → 5 < m :=
sorry

end NUMINAMATH_GPT_range_of_m_hyperbola_l1731_173142


namespace NUMINAMATH_GPT_cookie_ratio_l1731_173167

theorem cookie_ratio (K : ℕ) (h1 : K / 2 + K + 24 = 33) : 24 / K = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_cookie_ratio_l1731_173167


namespace NUMINAMATH_GPT_locus_of_centers_l1731_173133

theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (9 - r)^2) →
  12 * a^2 + 169 * b^2 - 36 * a - 1584 = 0 :=
by
  sorry

end NUMINAMATH_GPT_locus_of_centers_l1731_173133


namespace NUMINAMATH_GPT_total_people_in_line_l1731_173157

theorem total_people_in_line (n_front n_behind : ℕ) (hfront : n_front = 11) (hbehind : n_behind = 12) : n_front + n_behind + 1 = 24 := by
  sorry

end NUMINAMATH_GPT_total_people_in_line_l1731_173157


namespace NUMINAMATH_GPT_sum_ages_is_13_l1731_173156

-- Define the variables for the ages
variables (a b c : ℕ)

-- Define the conditions given in the problem
def conditions : Prop :=
  a * b * c = 72 ∧ a < b ∧ c < b

-- State the theorem to be proved
theorem sum_ages_is_13 (h : conditions a b c) : a + b + c = 13 :=
sorry

end NUMINAMATH_GPT_sum_ages_is_13_l1731_173156


namespace NUMINAMATH_GPT_isosceles_triangle_l1731_173131

theorem isosceles_triangle 
  {a b : ℝ} {α β : ℝ} 
  (h : a / (Real.cos α) = b / (Real.cos β)) : 
  a = b :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_l1731_173131


namespace NUMINAMATH_GPT_primitive_root_set_equality_l1731_173152

theorem primitive_root_set_equality 
  {p : ℕ} (hp : Nat.Prime p) (hodd: p % 2 = 1) (g : ℕ) (hg : g ^ (p - 1) % p = 1) :
  (∀ k, 1 ≤ k ∧ k ≤ (p - 1) / 2 → ∃ m, 1 ≤ m ∧ m ≤ (p - 1) / 2 ∧ (k^2 + 1) % p = g ^ m % p) ↔ p = 3 :=
by sorry

end NUMINAMATH_GPT_primitive_root_set_equality_l1731_173152


namespace NUMINAMATH_GPT_hospital_cost_minimization_l1731_173120

theorem hospital_cost_minimization :
  ∃ (x y : ℕ), (5 * x + 6 * y = 50) ∧ (10 * x + 20 * y = 140) ∧ (2 * x + 3 * y = 23) :=
by
  sorry

end NUMINAMATH_GPT_hospital_cost_minimization_l1731_173120


namespace NUMINAMATH_GPT_simplify_fraction_l1731_173107

theorem simplify_fraction
  (a b c : ℝ)
  (h : 2 * a - 3 * c - 4 - b ≠ 0)
  : (6 * a ^ 2 - 2 * b ^ 2 + 6 * c ^ 2 + a * b - 13 * a * c - 4 * b * c - 18 * a - 5 * b + 17 * c + 12) /
    (4 * a ^ 2 - b ^ 2 + 9 * c ^ 2 - 12 * a * c - 16 * a + 24 * c + 16) =
    (3 * a - 2 * c - 3 + 2 * b) / (2 * a - 3 * c - 4 + b) :=
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1731_173107


namespace NUMINAMATH_GPT_max_value_of_quadratic_l1731_173153

theorem max_value_of_quadratic (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) : 
  ∃ y, y = x * (1 - 2 * x) ∧ y ≤ 1 / 8 ∧ (y = 1 / 8 ↔ x = 1 / 4) :=
by sorry

end NUMINAMATH_GPT_max_value_of_quadratic_l1731_173153


namespace NUMINAMATH_GPT_max_min_sum_zero_l1731_173145

def cubic_function (x : ℝ) : ℝ :=
  x^3 - 3 * x

def first_derivative (x : ℝ) : ℝ :=
  3 * x^2 - 3

theorem max_min_sum_zero :
  let m := cubic_function (-1);
  let n := cubic_function 1;
  m + n = 0 :=
by
  sorry

end NUMINAMATH_GPT_max_min_sum_zero_l1731_173145


namespace NUMINAMATH_GPT_maximize_profit_l1731_173195

variable (k : ℚ) -- Proportional constant for deposits
variable (x : ℚ) -- Annual interest rate paid to depositors
variable (D : ℚ) -- Total amount of deposits

-- Define the condition for the total amount of deposits
def deposits (x : ℚ) : ℚ := k * x^2

-- Define the profit function
def profit (x : ℚ) : ℚ := 0.045 * k * x^2 - k * x^3

-- Define the derivative of the profit function
def profit_derivative (x : ℚ) : ℚ := 3 * k * x * (0.03 - x)

-- Statement that x = 0.03 maximizes the bank's profit
theorem maximize_profit : ∃ x, x = 0.03 ∧ (∀ y, profit_derivative y = 0 → x = y) :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l1731_173195


namespace NUMINAMATH_GPT_zero_of_f_l1731_173134

noncomputable def f (x : ℝ) : ℝ := 2^x - 4

theorem zero_of_f : f 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_of_f_l1731_173134


namespace NUMINAMATH_GPT_runners_meet_time_l1731_173130

theorem runners_meet_time :
  let time_runner_1 := 2
  let time_runner_2 := 4
  let time_runner_3 := 11 / 2
  Nat.lcm time_runner_1 (Nat.lcm time_runner_2 (Nat.lcm (11) 2)) = 44 := by
  sorry

end NUMINAMATH_GPT_runners_meet_time_l1731_173130


namespace NUMINAMATH_GPT_inequalities_quadrants_l1731_173117

theorem inequalities_quadrants :
  (∀ x y : ℝ, y > 2 * x → y > 4 - x → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)) := sorry

end NUMINAMATH_GPT_inequalities_quadrants_l1731_173117


namespace NUMINAMATH_GPT_M_is_set_of_positive_rationals_le_one_l1731_173139

def M : Set ℚ := {x | 0 < x ∧ x ≤ 1}

axiom contains_one (M : Set ℚ) : 1 ∈ M

axiom closed_under_operations (M : Set ℚ) :
  ∀ x ∈ M, (1 / (1 + x) ∈ M) ∧ (x / (1 + x) ∈ M)

theorem M_is_set_of_positive_rationals_le_one :
  M = {x | 0 < x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_GPT_M_is_set_of_positive_rationals_le_one_l1731_173139


namespace NUMINAMATH_GPT_half_of_one_point_zero_one_l1731_173164

theorem half_of_one_point_zero_one : (1.01 / 2) = 0.505 := 
by
  sorry

end NUMINAMATH_GPT_half_of_one_point_zero_one_l1731_173164


namespace NUMINAMATH_GPT_calc_area_of_quadrilateral_l1731_173160

-- Define the terms and conditions using Lean definitions
noncomputable def triangle_areas : ℕ × ℕ × ℕ := (6, 9, 15)

-- State the theorem
theorem calc_area_of_quadrilateral (a b c d : ℕ) (area1 area2 area3 : ℕ):
  area1 = 6 →
  area2 = 9 →
  area3 = 15 →
  a + b + c + d = area1 + area2 + area3 →
  d = 65 :=
  sorry

end NUMINAMATH_GPT_calc_area_of_quadrilateral_l1731_173160


namespace NUMINAMATH_GPT_min_value_frac_sum_l1731_173192

theorem min_value_frac_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) : 
  ∃ m, ∀ x y, 0 < x → 0 < y → 2 * x + y = 1 → m ≤ (1/x + 1/y) ∧ (1/x + 1/y) = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_frac_sum_l1731_173192


namespace NUMINAMATH_GPT_sin_pi_minus_alpha_cos_2pi_minus_alpha_sin_minus_cos_l1731_173146

-- Problem 1: Given that tan(α) = 3, prove that sin(π - α) * cos(2π - α) = 3 / 10.
theorem sin_pi_minus_alpha_cos_2pi_minus_alpha (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) = 3 / 10 :=
by
  sorry

-- Problem 2: Given that sin(α) * cos(α) = 1/4 and 0 < α < π/4, prove that sin(α) - cos(α) = - sqrt(2) / 2.
theorem sin_minus_cos (α : ℝ) (h₁ : Real.sin α * Real.cos α = 1 / 4) (h₂ : 0 < α) (h₃ : α < Real.pi / 4) :
  Real.sin α - Real.cos α = - (Real.sqrt 2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_pi_minus_alpha_cos_2pi_minus_alpha_sin_minus_cos_l1731_173146


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_x_plus_a_div_x_geq_2_l1731_173170

open Real

theorem sufficient_not_necessary_condition_x_plus_a_div_x_geq_2 (x a : ℝ)
  (h₁ : x > 0) :
  (∀ x > 0, x + a / x ≥ 2) → (a = 1) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_x_plus_a_div_x_geq_2_l1731_173170


namespace NUMINAMATH_GPT_parking_lot_perimeter_l1731_173196

theorem parking_lot_perimeter (a b : ℝ) 
  (h_diag : a^2 + b^2 = 784) 
  (h_area : a * b = 180) : 
  2 * (a + b) = 68 := 
by 
  sorry

end NUMINAMATH_GPT_parking_lot_perimeter_l1731_173196


namespace NUMINAMATH_GPT_renovation_project_total_l1731_173147

def sand : ℝ := 0.17
def dirt : ℝ := 0.33
def cement : ℝ := 0.17

theorem renovation_project_total : sand + dirt + cement = 0.67 := 
by
  sorry

end NUMINAMATH_GPT_renovation_project_total_l1731_173147


namespace NUMINAMATH_GPT_arithmetic_geometric_mean_l1731_173187

variable (x y : ℝ)

theorem arithmetic_geometric_mean (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 110) :
  x^2 + y^2 = 1380 := by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_mean_l1731_173187


namespace NUMINAMATH_GPT_inequality_proof_l1731_173118

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 3) : 
  (1 / (5 * a^2 - 4 * a + 11)) + (1 / (5 * b^2 - 4 * b + 11)) + (1 / (5 * c^2 - 4 * c + 11)) ≤ 1 / 4 := 
by
  -- proof steps will be here
  sorry

end NUMINAMATH_GPT_inequality_proof_l1731_173118


namespace NUMINAMATH_GPT_part1_part2_part3_l1731_173191

-- Part (1)
theorem part1 (m : ℝ) : (2 * m - 3) * (5 - 3 * m) = -6 * m^2 + 19 * m - 15 :=
  sorry

-- Part (2)
theorem part2 (a b : ℝ) : (3 * a^3) ^ 2 * (2 * b^2) ^ 3 / (6 * a * b) ^ 2 = 2 * a^4 * b^4 :=
  sorry

-- Part (3)
theorem part3 (a b : ℝ) : (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l1731_173191


namespace NUMINAMATH_GPT_discriminant_of_quadratic_l1731_173179

def a := 5
def b := 5 + 1/5
def c := 1/5
def discriminant (a b c : ℚ) := b^2 - 4 * a * c

theorem discriminant_of_quadratic :
  discriminant a b c = 576 / 25 :=
by
  sorry

end NUMINAMATH_GPT_discriminant_of_quadratic_l1731_173179


namespace NUMINAMATH_GPT_find_numbers_l1731_173103

theorem find_numbers (x y a : ℕ) (h1 : x = 6 * y - a) (h2 : x + y = 38) : 7 * x = 228 - a → y = 38 - x :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l1731_173103


namespace NUMINAMATH_GPT_time_taken_by_abc_l1731_173173

-- Define the work rates for a, b, and c
def work_rate_a_b : ℚ := 1 / 15
def work_rate_c : ℚ := 1 / 41.25

-- Define the combined work rate for a, b, and c
def combined_work_rate : ℚ := work_rate_a_b + work_rate_c

-- Define the reciprocal of the combined work rate, which is the time taken
def time_taken : ℚ := 1 / combined_work_rate

-- Prove that the time taken by a, b, and c together is 11 days
theorem time_taken_by_abc : time_taken = 11 := by
  -- Substitute the values to compute the result
  sorry

end NUMINAMATH_GPT_time_taken_by_abc_l1731_173173


namespace NUMINAMATH_GPT_quadratic_trinomial_with_integral_roots_l1731_173181

theorem quadratic_trinomial_with_integral_roots (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (∃ x : ℤ, a * x^2 + b * x + c = 0) ∧ 
  (∃ x : ℤ, (a + 1) * x^2 + (b + 1) * x + (c + 1) = 0) ∧ 
  (∃ x : ℤ, (a + 2) * x^2 + (b + 2) * x + (c + 2) = 0) :=
sorry

end NUMINAMATH_GPT_quadratic_trinomial_with_integral_roots_l1731_173181


namespace NUMINAMATH_GPT_principal_is_400_l1731_173193

-- Define the conditions
def rate_of_interest : ℚ := 12.5
def simple_interest : ℚ := 100
def time_in_years : ℚ := 2

-- Define the formula for principal amount based on the given conditions
def principal_amount (SI R T : ℚ) : ℚ := SI * 100 / (R * T)

-- Prove that the principal amount is 400
theorem principal_is_400 :
  principal_amount simple_interest rate_of_interest time_in_years = 400 := 
by
  simp [principal_amount, simple_interest, rate_of_interest, time_in_years]
  sorry

end NUMINAMATH_GPT_principal_is_400_l1731_173193


namespace NUMINAMATH_GPT_sqrt_fraction_value_l1731_173126

theorem sqrt_fraction_value (a b c d : Nat) (h : a = 2 ∧ b = 0 ∧ c = 2 ∧ d = 3) : 
  Real.sqrt (2023 / (a + b + c + d)) = 17 := by
  sorry

end NUMINAMATH_GPT_sqrt_fraction_value_l1731_173126


namespace NUMINAMATH_GPT_jenny_research_time_l1731_173106

noncomputable def time_spent_on_research (total_hours : ℕ) (proposal_hours : ℕ) (report_hours : ℕ) : ℕ :=
  total_hours - proposal_hours - report_hours

theorem jenny_research_time : time_spent_on_research 20 2 8 = 10 := by
  sorry

end NUMINAMATH_GPT_jenny_research_time_l1731_173106


namespace NUMINAMATH_GPT_charge_difference_percentage_l1731_173116

-- Given definitions
variables (G R P : ℝ)
def hotelR := 1.80 * G
def hotelP := 0.90 * G

-- Theorem statement
theorem charge_difference_percentage (G : ℝ) (hR : R = 1.80 * G) (hP : P = 0.90 * G) :
  (R - P) / R * 100 = 50 :=
by sorry

end NUMINAMATH_GPT_charge_difference_percentage_l1731_173116


namespace NUMINAMATH_GPT_shirt_cost_l1731_173189

theorem shirt_cost (S : ℝ) (hats_cost jeans_cost total_cost : ℝ)
  (h_hats : hats_cost = 4)
  (h_jeans : jeans_cost = 10)
  (h_total : total_cost = 51)
  (h_eq : 3 * S + 2 * jeans_cost + 4 * hats_cost = total_cost) :
  S = 5 :=
by
  -- The main proof will be provided here
  sorry

end NUMINAMATH_GPT_shirt_cost_l1731_173189


namespace NUMINAMATH_GPT_locus_of_circumcenter_l1731_173199

theorem locus_of_circumcenter (θ : ℝ) :
  let M := (3, 3 * Real.tan (θ - Real.pi / 3))
  let N := (3, 3 * Real.tan θ)
  let C := (3 / 2, 3 / 2 * Real.tan θ)
  ∃ (x y : ℝ), (x - 4) ^ 2 / 4 - y ^ 2 / 12 = 1 :=
by
  sorry

end NUMINAMATH_GPT_locus_of_circumcenter_l1731_173199


namespace NUMINAMATH_GPT_greatest_two_digit_number_l1731_173148

theorem greatest_two_digit_number (x y : ℕ) (h1 : x < y) (h2 : x * y = 12) : 10 * x + y = 34 :=
sorry

end NUMINAMATH_GPT_greatest_two_digit_number_l1731_173148


namespace NUMINAMATH_GPT_system_solutions_l1731_173180

theorem system_solutions : 
  ∃ (x y z t : ℝ), 
    (x * y - t^2 = 9) ∧ 
    (x^2 + y^2 + z^2 = 18) ∧ 
    ((x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ 
     (x = -3 ∧ y = -3 ∧ z = 0 ∧ t = 0)) :=
by {
  sorry
}

end NUMINAMATH_GPT_system_solutions_l1731_173180


namespace NUMINAMATH_GPT_sqrt6_eq_l1731_173182

theorem sqrt6_eq (r : Real) (h : r = Real.sqrt 2 + Real.sqrt 3) : Real.sqrt 6 = (r ^ 2 - 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt6_eq_l1731_173182


namespace NUMINAMATH_GPT_gcd_45_75_l1731_173186

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_GPT_gcd_45_75_l1731_173186


namespace NUMINAMATH_GPT_tony_bought_10_play_doughs_l1731_173125

noncomputable def num_play_doughs 
    (lego_cost : ℕ) 
    (sword_cost : ℕ) 
    (play_dough_cost : ℕ) 
    (bought_legos : ℕ) 
    (bought_swords : ℕ) 
    (total_paid : ℕ) : ℕ :=
  let lego_total := lego_cost * bought_legos
  let sword_total := sword_cost * bought_swords
  let total_play_dough_cost := total_paid - (lego_total + sword_total)
  total_play_dough_cost / play_dough_cost

theorem tony_bought_10_play_doughs : 
  num_play_doughs 250 120 35 3 7 1940 = 10 := 
sorry

end NUMINAMATH_GPT_tony_bought_10_play_doughs_l1731_173125
