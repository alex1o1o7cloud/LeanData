import Mathlib

namespace NUMINAMATH_GPT_inverse_proportional_k_value_l139_13937

theorem inverse_proportional_k_value (k : ℝ) :
  (∃ x y : ℝ, y = k / x ∧ x = - (Real.sqrt 2) / 2 ∧ y = Real.sqrt 2) → 
  k = -1 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportional_k_value_l139_13937


namespace NUMINAMATH_GPT_system_of_equations_a_solution_l139_13994

theorem system_of_equations_a_solution (x y a : ℝ) (h1 : 4 * x + y = a) (h2 : 3 * x + 4 * y^2 = 3 * a) (hx : x = 3) : a = 15 ∨ a = 9.75 :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_a_solution_l139_13994


namespace NUMINAMATH_GPT_arithmetic_sequence_l139_13911

variable (p q : ℕ) -- Assuming natural numbers for simplicity, but can be generalized.

def a (n : ℕ) : ℕ := p * n + q

theorem arithmetic_sequence:
  ∀ n : ℕ, n ≥ 1 → (a n - a (n-1) = p) := by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_l139_13911


namespace NUMINAMATH_GPT_find_smallest_k_l139_13943

theorem find_smallest_k : ∃ (k : ℕ), 64^k > 4^20 ∧ ∀ (m : ℕ), (64^m > 4^20) → m ≥ k := sorry

end NUMINAMATH_GPT_find_smallest_k_l139_13943


namespace NUMINAMATH_GPT_total_oranges_l139_13995

theorem total_oranges :
  let capacity_box1 := 80
  let capacity_box2 := 50
  let fullness_box1 := (3/4 : ℚ)
  let fullness_box2 := (3/5 : ℚ)
  let oranges_box1 := fullness_box1 * capacity_box1
  let oranges_box2 := fullness_box2 * capacity_box2
  oranges_box1 + oranges_box2 = 90 := 
by
  sorry

end NUMINAMATH_GPT_total_oranges_l139_13995


namespace NUMINAMATH_GPT_percentage_increase_l139_13924

theorem percentage_increase (initial final : ℝ)
  (h_initial: initial = 60) (h_final: final = 90) :
  (final - initial) / initial * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l139_13924


namespace NUMINAMATH_GPT_find_constant_C_l139_13971

def polynomial_remainder (C : ℝ) (x : ℝ) : ℝ :=
  C * x^3 - 3 * x^2 + x - 1

theorem find_constant_C :
  (polynomial_remainder 2 (-1) = -7) → 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_constant_C_l139_13971


namespace NUMINAMATH_GPT_initial_weight_of_solution_Y_is_8_l139_13982

theorem initial_weight_of_solution_Y_is_8
  (W : ℝ)
  (hw1 : 0.25 * W = 0.20 * W + 0.4)
  (hw2 : W ≠ 0) : W = 8 :=
by
  sorry

end NUMINAMATH_GPT_initial_weight_of_solution_Y_is_8_l139_13982


namespace NUMINAMATH_GPT_correct_operation_l139_13926

theorem correct_operation (x : ℝ) : (x^3 * x^2 = x^5) :=
by sorry

end NUMINAMATH_GPT_correct_operation_l139_13926


namespace NUMINAMATH_GPT_circle_constant_ratio_l139_13988

theorem circle_constant_ratio (b : ℝ) :
  (∀ (x y : ℝ), (x + 4)^2 + (y + b)^2 = 16 → 
    ∃ k : ℝ, 
      ∀ P : ℝ × ℝ, 
        P = (x, y) → 
        dist P (-2, 0) / dist P (4, 0) = k)
  → b = 0 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_circle_constant_ratio_l139_13988


namespace NUMINAMATH_GPT_find_x_range_l139_13903

noncomputable def f (x : ℝ) : ℝ := if h : x ≥ 0 then 3^(-x) else 3^(x)

theorem find_x_range (x : ℝ) (h1 : f 2 = -f (2*x - 1) ∧ f 2 < 0) : -1/2 < x ∧ x < 3/2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_x_range_l139_13903


namespace NUMINAMATH_GPT_maggie_kept_bouncy_balls_l139_13923

def packs_bought_yellow : ℝ := 8.0
def packs_given_away_green : ℝ := 4.0
def packs_bought_green : ℝ := 4.0
def balls_per_pack : ℝ := 10.0

theorem maggie_kept_bouncy_balls :
  packs_bought_yellow * balls_per_pack + (packs_bought_green - packs_given_away_green) * balls_per_pack = 80.0 :=
by sorry

end NUMINAMATH_GPT_maggie_kept_bouncy_balls_l139_13923


namespace NUMINAMATH_GPT_evaluate_expression_l139_13989

noncomputable def x : ℚ := 4 / 7
noncomputable def y : ℚ := 6 / 8

theorem evaluate_expression : (7 * x + 8 * y) / (56 * x * y) = 5 / 12 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l139_13989


namespace NUMINAMATH_GPT_initial_pens_l139_13967

-- Conditions as definitions
def initial_books := 108
def books_after_sale := 66
def books_sold := 42
def pens_after_sale := 59

-- Theorem statement proving the initial number of pens
theorem initial_pens:
  initial_books - books_after_sale = books_sold →
  ∃ (P : ℕ), P - pens_sold = pens_after_sale ∧ (P = 101) :=
by
  sorry

end NUMINAMATH_GPT_initial_pens_l139_13967


namespace NUMINAMATH_GPT_symmetric_line_equation_l139_13987

theorem symmetric_line_equation :
  (∃ line : ℝ → ℝ, ∀ x y, x + 2 * y - 3 = 0 → line 1 = 1 ∧ (∃ b, line 0 = b → x - 2 * y + 1 = 0)) :=
sorry

end NUMINAMATH_GPT_symmetric_line_equation_l139_13987


namespace NUMINAMATH_GPT_problem_l139_13981

noncomputable def a : ℝ := (Real.sqrt 5 + Real.sqrt 3) / (Real.sqrt 5 - Real.sqrt 3)
noncomputable def b : ℝ := (Real.sqrt 5 - Real.sqrt 3) / (Real.sqrt 5 + Real.sqrt 3)

theorem problem :
  a^4 + b^4 + (a + b)^4 = 7938 := by
  sorry

end NUMINAMATH_GPT_problem_l139_13981


namespace NUMINAMATH_GPT_problem1_problem2_l139_13908

-- Define condition p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2*x - 8 > 0)

-- Define the negation of p
def neg_p (x a : ℝ) : Prop := ¬ p x a
-- Define the negation of q
def neg_q (x : ℝ) : Prop := ¬ q x

-- Question 1: Prove that if a = 1 and p ∧ q is true, then 2 < x < 3
theorem problem1 (x : ℝ) (h1 : p x 1 ∧ q x) : 2 < x ∧ x < 3 := 
by sorry

-- Question 2: Prove that if ¬ p is a sufficient but not necessary condition for ¬ q, then 1 < a ≤ 2
theorem problem2 (a : ℝ) (h2 : ∀ x : ℝ, neg_p x a → neg_q x) : 1 < a ∧ a ≤ 2 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l139_13908


namespace NUMINAMATH_GPT_probability_of_rolling_number_less_than_5_is_correct_l139_13983

noncomputable def probability_of_rolling_number_less_than_5 : ℚ :=
  let total_outcomes := 8
  let favorable_outcomes := 4
  favorable_outcomes / total_outcomes

theorem probability_of_rolling_number_less_than_5_is_correct :
  probability_of_rolling_number_less_than_5 = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_of_rolling_number_less_than_5_is_correct_l139_13983


namespace NUMINAMATH_GPT_Lisa_income_percentage_J_M_combined_l139_13969

variables (T M J L : ℝ)

-- Conditions as definitions
def Mary_income_eq_1p6_T (M T : ℝ) : Prop := M = 1.60 * T
def Tim_income_eq_0p5_J (T J : ℝ) : Prop := T = 0.50 * J
def Lisa_income_eq_1p3_M (L M : ℝ) : Prop := L = 1.30 * M
def Lisa_income_eq_0p75_J (L J : ℝ) : Prop := L = 0.75 * J

-- Theorem statement
theorem Lisa_income_percentage_J_M_combined (M T J L : ℝ)
  (h1 : Mary_income_eq_1p6_T M T)
  (h2 : Tim_income_eq_0p5_J T J)
  (h3 : Lisa_income_eq_1p3_M L M)
  (h4 : Lisa_income_eq_0p75_J L J) :
  (L / (M + J)) * 100 = 41.67 := 
sorry

end NUMINAMATH_GPT_Lisa_income_percentage_J_M_combined_l139_13969


namespace NUMINAMATH_GPT_count_multiples_of_4_between_300_and_700_l139_13978

noncomputable def num_multiples_of_4_in_range (a b : ℕ) : ℕ :=
  (b - (b % 4) - (a - (a % 4) + 4)) / 4 + 1

theorem count_multiples_of_4_between_300_and_700 : 
  num_multiples_of_4_in_range 301 699 = 99 := by
  sorry

end NUMINAMATH_GPT_count_multiples_of_4_between_300_and_700_l139_13978


namespace NUMINAMATH_GPT_fraction_identity_l139_13913

theorem fraction_identity (a b : ℝ) (h₀ : a^2 + a = 4) (h₁ : b^2 + b = 4) (h₂ : a ≠ b) :
  (b / a) + (a / b) = - (9 / 4) :=
sorry

end NUMINAMATH_GPT_fraction_identity_l139_13913


namespace NUMINAMATH_GPT_gcd_g50_g52_l139_13977

def g (x : ℤ) := x^2 - 2*x + 2022

theorem gcd_g50_g52 : Int.gcd (g 50) (g 52) = 2 := by
  sorry

end NUMINAMATH_GPT_gcd_g50_g52_l139_13977


namespace NUMINAMATH_GPT_geometric_series_sum_l139_13954

noncomputable def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  geometric_sum (2/3) (2/3) 10 = 116050 / 59049 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l139_13954


namespace NUMINAMATH_GPT_find_fraction_l139_13939

def number : ℕ := 16

theorem find_fraction (f : ℚ) : f * number + 5 = 13 → f = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_l139_13939


namespace NUMINAMATH_GPT_amount_spent_on_shirt_l139_13910

-- Definitions and conditions
def total_spent_clothing : ℝ := 25.31
def spent_on_jacket : ℝ := 12.27

-- Goal: Prove the amount spent on the shirt is 13.04
theorem amount_spent_on_shirt : (total_spent_clothing - spent_on_jacket = 13.04) := by
  sorry

end NUMINAMATH_GPT_amount_spent_on_shirt_l139_13910


namespace NUMINAMATH_GPT_probability_5800_in_three_spins_l139_13953

def spinner_labels : List String := ["Bankrupt", "$600", "$1200", "$4000", "$800", "$2000", "$150"]

def total_outcomes (spins : Nat) : Nat :=
  let segments := spinner_labels.length
  segments ^ spins

theorem probability_5800_in_three_spins :
  (6 / total_outcomes 3 : ℚ) = 6 / 343 :=
by
  sorry

end NUMINAMATH_GPT_probability_5800_in_three_spins_l139_13953


namespace NUMINAMATH_GPT_total_number_of_fish_l139_13968

theorem total_number_of_fish :
  let goldfish := 8
  let angelfish := goldfish + 4
  let guppies := 2 * angelfish
  let tetras := goldfish - 3
  let bettas := tetras + 5
  goldfish + angelfish + guppies + tetras + bettas = 59 := by
  -- Provide the proof here.
  sorry

end NUMINAMATH_GPT_total_number_of_fish_l139_13968


namespace NUMINAMATH_GPT_range_of_x_given_p_and_q_range_of_m_given_neg_q_sufficient_for_neg_p_l139_13918

variable {x m : ℝ}

-- First statement: Given m = 4 and p ∧ q, prove the range of x is 4 < x < 5
theorem range_of_x_given_p_and_q (m : ℝ) (h : m = 4) :
  (x^2 - 7*x + 10 < 0) ∧ (x^2 - 4*m*x + 3*m^2 < 0) → (4 < x ∧ x < 5) :=
sorry

-- Second statement: Prove the range of m given ¬q is a sufficient but not necessary condition for ¬p
theorem range_of_m_given_neg_q_sufficient_for_neg_p :
  (m ≤ 2) ∧ (3*m ≥ 5) ∧ (m > 0) → (5/3 ≤ m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_GPT_range_of_x_given_p_and_q_range_of_m_given_neg_q_sufficient_for_neg_p_l139_13918


namespace NUMINAMATH_GPT_problem_statement_l139_13920

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 4) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 74.0625 :=
sorry

end NUMINAMATH_GPT_problem_statement_l139_13920


namespace NUMINAMATH_GPT_find_x_value_l139_13984

theorem find_x_value (a b x : ℤ) (h : a * b = (a - 1) * (b - 1)) (h2 : x * 9 = 160) :
  x = 21 :=
sorry

end NUMINAMATH_GPT_find_x_value_l139_13984


namespace NUMINAMATH_GPT_solution_l139_13961

-- Definitions for perpendicular and parallel relations
def perpendicular (a b : Type) : Prop := sorry -- Abstraction for perpendicularity
def parallel (a b : Type) : Prop := sorry -- Abstraction for parallelism

-- Here we define x, y, z as variables
variables {x y : Type} {z : Type}

-- Conditions for Case 2
def case2_lines_plane (x y : Type) (z : Type) := 
  (perpendicular x z) ∧ (perpendicular y z) → (parallel x y)

-- Conditions for Case 3
def case3_planes_line (x y : Type) (z : Type) := 
  (perpendicular x z) ∧ (perpendicular y z) → (parallel x y)

-- Theorem statement combining both cases
theorem solution : case2_lines_plane x y z ∧ case3_planes_line x y z := 
sorry

end NUMINAMATH_GPT_solution_l139_13961


namespace NUMINAMATH_GPT_find_a_if_y_is_even_l139_13973

noncomputable def y (x a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem find_a_if_y_is_even (a : ℝ) (h : ∀ x : ℝ, y x a = y (-x) a) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_if_y_is_even_l139_13973


namespace NUMINAMATH_GPT_unique_last_digit_divisible_by_7_l139_13948

theorem unique_last_digit_divisible_by_7 :
  ∃! d : ℕ, (∃ n : ℕ, n % 7 = 0 ∧ n % 10 = d) :=
sorry

end NUMINAMATH_GPT_unique_last_digit_divisible_by_7_l139_13948


namespace NUMINAMATH_GPT_fair_coin_toss_consecutive_heads_l139_13927

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem fair_coin_toss_consecutive_heads :
  let total_outcomes := 1024
  let favorable_outcomes := 
    1 + binom 10 1 + binom 9 2 + binom 8 3 + binom 7 4 + binom 6 5
  let prob := favorable_outcomes / total_outcomes
  let i := 9
  let j := 64
  Nat.gcd i j = 1 ∧ (prob = i / j) ∧ i + j = 73 :=
by
  sorry

end NUMINAMATH_GPT_fair_coin_toss_consecutive_heads_l139_13927


namespace NUMINAMATH_GPT_sum_of_geometric_progression_l139_13929

theorem sum_of_geometric_progression (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (a1 a3 : ℝ) (h1 : a1 + a3 = 5) (h2 : a1 * a3 = 4)
  (h3 : a 1 = a1) (h4 : a 3 = a3)
  (h5 : ∀ k, a (k + 1) > a k)  -- Sequence is increasing
  (h6 : S n = a 1 * ((1 - (2:ℝ) ^ n) / (1 - 2)))
  (h7 : n = 6) :
  S 6 = 63 :=
sorry

end NUMINAMATH_GPT_sum_of_geometric_progression_l139_13929


namespace NUMINAMATH_GPT_quadratic_roots_identity_l139_13942

theorem quadratic_roots_identity (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0) (hmn : m * n = -5) (hm_plus_n : m + n = -2) : m^2 + m * n + 2 * m = 0 :=
by {
    sorry
}

end NUMINAMATH_GPT_quadratic_roots_identity_l139_13942


namespace NUMINAMATH_GPT_total_distance_traveled_is_correct_l139_13999

-- Definitions of given conditions
def Vm : ℕ := 8
def Vr : ℕ := 2
def round_trip_time : ℝ := 1

-- Definitions needed for intermediate calculations (speed computations)
def upstream_speed (Vm Vr : ℕ) : ℕ := Vm - Vr
def downstream_speed (Vm Vr : ℕ) : ℕ := Vm + Vr

-- The equation representing the total time for the round trip
def time_equation (D : ℝ) (Vm Vr : ℕ) : Prop :=
  D / upstream_speed Vm Vr + D / downstream_speed Vm Vr = round_trip_time

-- Prove that the total distance traveled by the man is 7.5 km
theorem total_distance_traveled_is_correct : ∃ D : ℝ, D / upstream_speed Vm Vr + D / downstream_speed Vm Vr = round_trip_time ∧ 2 * D = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_is_correct_l139_13999


namespace NUMINAMATH_GPT_solution_set_of_inequality_l139_13941

theorem solution_set_of_inequality (x : ℝ) : 
  (x * |x - 1| > 0) ↔ ((0 < x ∧ x < 1) ∨ (x > 1)) := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l139_13941


namespace NUMINAMATH_GPT_solution_correct_l139_13922

noncomputable def solve_system (A1 A2 A3 A4 A5 : ℝ) (x1 x2 x3 x4 x5 : ℝ) :=
  (2 * x1 - 2 * x2 = A1) ∧
  (-x1 + 4 * x2 - 3 * x3 = A2) ∧
  (-2 * x2 + 6 * x3 - 4 * x4 = A3) ∧
  (-3 * x3 + 8 * x4 - 5 * x5 = A4) ∧
  (-4 * x4 + 10 * x5 = A5)

theorem solution_correct {A1 A2 A3 A4 A5 x1 x2 x3 x4 x5 : ℝ} :
  solve_system A1 A2 A3 A4 A5 x1 x2 x3 x4 x5 → 
  x1 = (5 * A1 + 4 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x2 = (2 * A1 + 4 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x3 = (A1 + 2 * A2 + 3 * A3 + 2 * A4 + A5) / 6 ∧
  x4 = (A1 + 2 * A2 + 3 * A3 + 4 * A4 + 2 * A5) / 12 ∧
  x5 = (A1 + 2 * A2 + 3 * A3 + 4 * A4 + 5 * A5) / 30 :=
sorry

end NUMINAMATH_GPT_solution_correct_l139_13922


namespace NUMINAMATH_GPT_digit_place_value_ratio_l139_13996

theorem digit_place_value_ratio :
  let number := 86304.2957
  let digit_6_value := 1000
  let digit_5_value := 0.1
  digit_6_value / digit_5_value = 10000 :=
by
  let number := 86304.2957
  let digit_6_value := 1000
  let digit_5_value := 0.1
  sorry

end NUMINAMATH_GPT_digit_place_value_ratio_l139_13996


namespace NUMINAMATH_GPT_remainder_division_l139_13938

/-- A number when divided by a certain divisor left a remainder, 
when twice the number was divided by the same divisor, the remainder was 112. 
The divisor is 398.
Prove that the remainder when the original number is divided by the divisor is 56. -/
theorem remainder_division (N R : ℤ) (D : ℕ) (Q Q' : ℤ)
  (hD : D = 398)
  (h1 : N = D * Q + R)
  (h2 : 2 * N = D * Q' + 112) :
  R = 56 :=
sorry

end NUMINAMATH_GPT_remainder_division_l139_13938


namespace NUMINAMATH_GPT_open_box_volume_l139_13962

theorem open_box_volume (l w s : ℝ) (hl : l = 48) (hw : w = 36) (hs : s = 8) :
  (l - 2 * s) * (w - 2 * s) * s = 5120 :=
by
  sorry

end NUMINAMATH_GPT_open_box_volume_l139_13962


namespace NUMINAMATH_GPT_housewife_spending_l139_13992

theorem housewife_spending (P R M : ℝ) (h1 : R = 65) (h2 : R = 0.75 * P) (h3 : M / R - M / P = 5) :
  M = 1300 :=
by
  -- Proof steps will be added here.
  sorry

end NUMINAMATH_GPT_housewife_spending_l139_13992


namespace NUMINAMATH_GPT_cost_of_bananas_and_cantaloupe_l139_13904

variable (a b c d : ℝ)

theorem cost_of_bananas_and_cantaloupe :
  (a + b + c + d = 30) →
  (d = 3 * a) →
  (c = a - b) →
  (b + c = 6) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_cost_of_bananas_and_cantaloupe_l139_13904


namespace NUMINAMATH_GPT_min_value_of_objective_function_l139_13979

theorem min_value_of_objective_function : 
  ∃ (x y : ℝ), 
    (2 * x + y - 2 ≥ 0) ∧ 
    (x - 2 * y + 4 ≥ 0) ∧ 
    (x - 1 ≤ 0) ∧ 
    (∀ (u v: ℝ), 
      (2 * u + v - 2 ≥ 0) → 
      (u - 2 * v + 4 ≥ 0) → 
      (u - 1 ≤ 0) → 
      (3 * u + 2 * v ≥ 3)) :=
  sorry

end NUMINAMATH_GPT_min_value_of_objective_function_l139_13979


namespace NUMINAMATH_GPT_divisor_of_7_l139_13990

theorem divisor_of_7 (a n : ℤ) (h1 : a ≥ 1) (h2 : a ∣ (n + 2)) (h3 : a ∣ (n^2 + n + 5)) : a = 1 ∨ a = 7 :=
by
  sorry

end NUMINAMATH_GPT_divisor_of_7_l139_13990


namespace NUMINAMATH_GPT_tracy_two_dogs_food_consumption_l139_13974

theorem tracy_two_dogs_food_consumption
  (cups_per_meal : ℝ)
  (meals_per_day : ℝ)
  (pounds_per_cup : ℝ)
  (num_dogs : ℝ) :
  cups_per_meal = 1.5 →
  meals_per_day = 3 →
  pounds_per_cup = 1 / 2.25 →
  num_dogs = 2 →
  num_dogs * (cups_per_meal * meals_per_day) * pounds_per_cup = 4 := by
  sorry

end NUMINAMATH_GPT_tracy_two_dogs_food_consumption_l139_13974


namespace NUMINAMATH_GPT_random_walk_expected_distance_l139_13933

noncomputable def expected_distance_after_random_walk (n : ℕ) : ℚ :=
(sorry : ℚ) -- We'll define this in the proof

-- Proof problem statement in Lean 4
theorem random_walk_expected_distance :
  expected_distance_after_random_walk 6 = 15 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_random_walk_expected_distance_l139_13933


namespace NUMINAMATH_GPT_anita_total_cartons_l139_13909

-- Defining the conditions
def cartons_of_strawberries : ℕ := 10
def cartons_of_blueberries : ℕ := 9
def additional_cartons_needed : ℕ := 7

-- Adding the core theorem to be proved
theorem anita_total_cartons :
  cartons_of_strawberries + cartons_of_blueberries + additional_cartons_needed = 26 := 
by
  sorry

end NUMINAMATH_GPT_anita_total_cartons_l139_13909


namespace NUMINAMATH_GPT_sequence_a_b_10_l139_13963

theorem sequence_a_b_10 (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7) (h5 : a^5 + b^5 = 11) : 
  a^10 + b^10 = 123 := 
sorry

end NUMINAMATH_GPT_sequence_a_b_10_l139_13963


namespace NUMINAMATH_GPT_fixed_cost_is_50000_l139_13975

-- Definition of conditions
def fixed_cost : ℕ := 50000
def books_sold : ℕ := 10000
def revenue_per_book : ℕ := 9 - 4

-- Theorem statement: Proving that the fixed cost of making books is $50,000
theorem fixed_cost_is_50000 (F : ℕ) (h : revenue_per_book * books_sold = F) : 
  F = fixed_cost :=
by sorry

end NUMINAMATH_GPT_fixed_cost_is_50000_l139_13975


namespace NUMINAMATH_GPT_eggs_in_each_basket_l139_13959

theorem eggs_in_each_basket (n : ℕ) (h1 : 30 % n = 0) (h2 : 42 % n = 0) (h3 : n ≥ 5) :
  n = 6 :=
by sorry

end NUMINAMATH_GPT_eggs_in_each_basket_l139_13959


namespace NUMINAMATH_GPT_gear_ratio_l139_13976

variable (a b c : ℕ) (ωG ωH ωI : ℚ)

theorem gear_ratio :
  (a * ωG = b * ωH) ∧ (b * ωH = c * ωI) ∧ (a * ωG = c * ωI) →
  ωG / ωH = bc / ac ∧ ωH / ωI = ac / ab ∧ ωG / ωI = bc / ab :=
by
  sorry

end NUMINAMATH_GPT_gear_ratio_l139_13976


namespace NUMINAMATH_GPT_fried_chicken_total_l139_13958

-- The Lean 4 statement encapsulates the problem conditions and the correct answer
theorem fried_chicken_total :
  let kobe_initial := 5
  let pau_initial := 2 * kobe_initial
  let another_set := 2
  pau_initial * another_set = 20 :=
by
  let kobe_initial := 5
  let pau_initial := 2 * kobe_initial
  let another_set := 2
  show pau_initial * another_set = 20
  sorry

end NUMINAMATH_GPT_fried_chicken_total_l139_13958


namespace NUMINAMATH_GPT_total_chickens_l139_13932

theorem total_chickens (coops chickens_per_coop : ℕ) (h1 : coops = 9) (h2 : chickens_per_coop = 60) :
  coops * chickens_per_coop = 540 := by
  sorry

end NUMINAMATH_GPT_total_chickens_l139_13932


namespace NUMINAMATH_GPT_part1_part2_axis_of_symmetry_part2_center_of_symmetry_l139_13972

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * (Real.cos x) ^ 2, Real.sin x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)

def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem part1 (x : ℝ) (h1 : 0 < x ∧ x < π) (h2 : perpendicular (m x) (n x)) :
  x = π / 2 ∨ x = 3 * π / 4 :=
sorry

theorem part2_axis_of_symmetry (k : ℤ) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = f (2 * c - x) ∧ 
    ((2 * x + π / 4) = k * π + π / 2 → x = k * π / 2 + π / 8) :=
sorry

theorem part2_center_of_symmetry (k : ℤ) :
  ∃ x c : ℝ, f x = 1 ∧ ((2 * x + π / 4) = k * π → x = k * π / 2 - π / 8) :=
sorry

end NUMINAMATH_GPT_part1_part2_axis_of_symmetry_part2_center_of_symmetry_l139_13972


namespace NUMINAMATH_GPT_number_of_sides_l139_13935

-- Define the given conditions as Lean definitions

def exterior_angle := 72
def sum_of_exterior_angles := 360

-- Now state the theorem based on these conditions

theorem number_of_sides (n : ℕ) (h1 : exterior_angle = 72) (h2 : sum_of_exterior_angles = 360) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sides_l139_13935


namespace NUMINAMATH_GPT_max_boxes_in_warehouse_l139_13940

def warehouse_length : ℕ := 50
def warehouse_width : ℕ := 30
def warehouse_height : ℕ := 5
def box_edge_length : ℕ := 2

theorem max_boxes_in_warehouse : (warehouse_length / box_edge_length) * (warehouse_width / box_edge_length) * (warehouse_height / box_edge_length) = 750 := 
by
  sorry

end NUMINAMATH_GPT_max_boxes_in_warehouse_l139_13940


namespace NUMINAMATH_GPT_each_child_gets_one_slice_l139_13916

-- Define the conditions
def couple_slices_per_person : ℕ := 3
def number_of_people : ℕ := 2
def number_of_children : ℕ := 6
def pizzas_ordered : ℕ := 3
def slices_per_pizza : ℕ := 4

-- Calculate slices required by the couple
def total_slices_for_couple : ℕ := couple_slices_per_person * number_of_people

-- Calculate total slices available
def total_slices : ℕ := pizzas_ordered * slices_per_pizza

-- Calculate slices for children
def slices_for_children : ℕ := total_slices - total_slices_for_couple

-- Calculate slices each child gets
def slices_per_child : ℕ := slices_for_children / number_of_children

-- The proof statement
theorem each_child_gets_one_slice : slices_per_child = 1 := by
  sorry

end NUMINAMATH_GPT_each_child_gets_one_slice_l139_13916


namespace NUMINAMATH_GPT_no_x0_leq_zero_implies_m_gt_1_l139_13928

theorem no_x0_leq_zero_implies_m_gt_1 (m : ℝ) :
  (¬ ∃ x0 : ℝ, x0^2 + 2 * x0 + m ≤ 0) ↔ m > 1 :=
sorry

end NUMINAMATH_GPT_no_x0_leq_zero_implies_m_gt_1_l139_13928


namespace NUMINAMATH_GPT_sum_geometric_sequence_terms_l139_13907

theorem sum_geometric_sequence_terms (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300) 
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 := 
by
  sorry

end NUMINAMATH_GPT_sum_geometric_sequence_terms_l139_13907


namespace NUMINAMATH_GPT_longest_side_of_similar_triangle_l139_13991

theorem longest_side_of_similar_triangle :
  ∀ (x : ℝ),
    let a := 8
    let b := 10
    let c := 12
    let s₁ := a * x
    let s₂ := b * x
    let s₃ := c * x
    a + b + c = 30 → 
    30 * x = 150 → 
    s₁ > 30 → 
    max s₁ (max s₂ s₃) = 60 :=
by
  intros x a b c s₁ s₂ s₃ h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_longest_side_of_similar_triangle_l139_13991


namespace NUMINAMATH_GPT_product_eval_at_3_l139_13970

theorem product_eval_at_3 : (3 - 2) * (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) = 720 := by
  sorry

end NUMINAMATH_GPT_product_eval_at_3_l139_13970


namespace NUMINAMATH_GPT_total_stoppage_time_l139_13952

theorem total_stoppage_time (stop1 stop2 stop3 : ℕ) (h1 : stop1 = 5)
  (h2 : stop2 = 8) (h3 : stop3 = 10) : stop1 + stop2 + stop3 = 23 :=
sorry

end NUMINAMATH_GPT_total_stoppage_time_l139_13952


namespace NUMINAMATH_GPT_sum_of_b_for_unique_solution_l139_13998

theorem sum_of_b_for_unique_solution :
  (∃ b1 b2, (3 * (0:ℝ)^2 + (b1 + 6) * 0 + 7 = 0 ∧ 3 * (0:ℝ)^2 + (b2 + 6) * 0 + 7 = 0) ∧ 
   ((b1 + 6)^2 - 4 * 3 * 7 = 0) ∧ ((b2 + 6)^2 - 4 * 3 * 7 = 0) ∧ 
   b1 + b2 = -12)  :=
by
  sorry

end NUMINAMATH_GPT_sum_of_b_for_unique_solution_l139_13998


namespace NUMINAMATH_GPT_problem_l139_13955

def f (x : ℝ) (a b : ℝ) := x^5 + a * x^3 + b * x - 2

-- We are given f(-2) = m
variables (a b m : ℝ)
theorem problem (h : f (-2) a b = m) : f 2 a b + f (-2) a b = -4 :=
by sorry

end NUMINAMATH_GPT_problem_l139_13955


namespace NUMINAMATH_GPT_students_not_in_biology_l139_13934

theorem students_not_in_biology (S : ℕ) (f : ℚ) (hS : S = 840) (hf : f = 0.35) :
  S - (f * S) = 546 :=
by
  sorry

end NUMINAMATH_GPT_students_not_in_biology_l139_13934


namespace NUMINAMATH_GPT_unique_line_intercept_l139_13956

noncomputable def is_positive_integer (n : ℕ) : Prop := n > 0
noncomputable def is_prime (n : ℕ) : Prop := n = 2 ∨ (n > 2 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem unique_line_intercept (a b : ℕ) :
  ((is_positive_integer a) ∧ (is_prime b) ∧ (6 * b + 5 * a = a * b)) ↔ (a = 11 ∧ b = 11) :=
by
  sorry

end NUMINAMATH_GPT_unique_line_intercept_l139_13956


namespace NUMINAMATH_GPT_radius_distance_relation_l139_13980

variables {A B C : Point} (Γ₁ Γ₂ ω₀ : Circle)
variables (ω : ℕ → Circle)
variables (r d : ℕ → ℝ)

def diam_circle (P Q : Point) : Circle := sorry  -- This is to define a circle with diameter PQ
def tangent (κ κ' κ'' : Circle) : Prop := sorry  -- This is to define that three circles are mutually tangent

-- Defining the properties as given in the conditions
axiom Γ₁_def : Γ₁ = diam_circle A B
axiom Γ₂_def : Γ₂ = diam_circle A C
axiom ω₀_def : ω₀ = diam_circle B C
axiom ω_def : ∀ n : ℕ, tangent (if n = 0 then ω₀ else ω (n - 1)) Γ₁ (ω n) ∧ tangent (if n = 0 then ω₀ else ω (n - 1)) Γ₂ (ω n) -- ωₙ is tangent to previous circle, Γ₁ and Γ₂

-- The main proof statement
theorem radius_distance_relation (n : ℕ) : r n = 2 * n * d n :=
sorry

end NUMINAMATH_GPT_radius_distance_relation_l139_13980


namespace NUMINAMATH_GPT_min_value_expression_l139_13985

theorem min_value_expression (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 1) : 
  ∃(x : ℝ), x ≤ (a - b) * (b - c) * (c - d) * (d - a) ∧ x = -1/8 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l139_13985


namespace NUMINAMATH_GPT_seventh_graders_more_than_sixth_graders_l139_13905

-- Definitions based on conditions
variables (S6 S7 : ℕ)
variable (h : 7 * S6 = 6 * S7)

-- Proposition based on the conclusion
theorem seventh_graders_more_than_sixth_graders (h : 7 * S6 = 6 * S7) : S7 > S6 :=
by {
  -- Skipping the proof with sorry
  sorry
}

end NUMINAMATH_GPT_seventh_graders_more_than_sixth_graders_l139_13905


namespace NUMINAMATH_GPT_solve_for_x_l139_13945

theorem solve_for_x (x : ℝ) :
    (1 / 3 * ((x + 8) + (7 * x + 3) + (3 * x + 9)) = 5 * x - 10) → x = 12.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l139_13945


namespace NUMINAMATH_GPT_slope_parallel_l139_13957

theorem slope_parallel (x y : ℝ) (m : ℝ) : (3:ℝ) * x - (6:ℝ) * y = (9:ℝ) → m = (1:ℝ) / (2:ℝ) :=
by
  sorry

end NUMINAMATH_GPT_slope_parallel_l139_13957


namespace NUMINAMATH_GPT_slope_of_tangent_at_A_l139_13915

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem slope_of_tangent_at_A :
  (deriv f 0) = 1 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_tangent_at_A_l139_13915


namespace NUMINAMATH_GPT_average_male_grade_l139_13901

theorem average_male_grade (avg_all avg_fem : ℝ) (N_male N_fem : ℕ) 
    (h1 : avg_all = 90) 
    (h2 : avg_fem = 92) 
    (h3 : N_male = 8) 
    (h4 : N_fem = 12) :
    let total_students := N_male + N_fem
    let total_sum_all := avg_all * total_students
    let total_sum_fem := avg_fem * N_fem
    let total_sum_male := total_sum_all - total_sum_fem
    let avg_male := total_sum_male / N_male
    avg_male = 87 :=
by 
  let total_students := N_male + N_fem
  let total_sum_all := avg_all * total_students
  let total_sum_fem := avg_fem * N_fem
  let total_sum_male := total_sum_all - total_sum_fem
  let avg_male := total_sum_male / N_male
  sorry

end NUMINAMATH_GPT_average_male_grade_l139_13901


namespace NUMINAMATH_GPT_find_percentage_l139_13900

theorem find_percentage (P : ℝ) :
  (P / 100) * 1280 = ((0.20 * 650) + 190) ↔ P = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_l139_13900


namespace NUMINAMATH_GPT_common_tangent_and_inequality_l139_13944

noncomputable def f (x : ℝ) := Real.log (1 + x)
noncomputable def g (x : ℝ) := x - (1 / 2) * x^2 + (1 / 3) * x^3

theorem common_tangent_and_inequality :
  -- Condition: common tangent at (0, 0)
  (∀ x, deriv f x = deriv g x) →
  -- Condition: values of a and b found to be 0 and 1 respectively
  (∀ x, f x ≤ g x) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_common_tangent_and_inequality_l139_13944


namespace NUMINAMATH_GPT_no_real_solution_for_eq_l139_13925

theorem no_real_solution_for_eq (y : ℝ) : ¬ ∃ y : ℝ, ((y - 4 * y + 10)^2 + 4 = -2 * |y|) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_for_eq_l139_13925


namespace NUMINAMATH_GPT_binary_sum_eq_669_l139_13986

def binary111111111 : ℕ := 511
def binary1111111 : ℕ := 127
def binary11111 : ℕ := 31

theorem binary_sum_eq_669 :
  binary111111111 + binary1111111 + binary11111 = 669 :=
by
  sorry

end NUMINAMATH_GPT_binary_sum_eq_669_l139_13986


namespace NUMINAMATH_GPT_ratio_of_colored_sheets_l139_13951

theorem ratio_of_colored_sheets
    (total_sheets : ℕ)
    (num_binders : ℕ)
    (sheets_colored_by_justine : ℕ)
    (sheets_per_binder : ℕ)
    (h1 : total_sheets = 2450)
    (h2 : num_binders = 5)
    (h3 : sheets_colored_by_justine = 245)
    (h4 : sheets_per_binder = total_sheets / num_binders) :
    (sheets_colored_by_justine / Nat.gcd sheets_colored_by_justine sheets_per_binder) /
    (sheets_per_binder / Nat.gcd sheets_colored_by_justine sheets_per_binder) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_colored_sheets_l139_13951


namespace NUMINAMATH_GPT_value_of_x_l139_13960

theorem value_of_x (x : ℝ) (h : ∃ k < 0, (x, 1) = k • (4, x)) : x = -2 :=
sorry

end NUMINAMATH_GPT_value_of_x_l139_13960


namespace NUMINAMATH_GPT_stormi_cars_washed_l139_13950

-- Definitions based on conditions
def cars_earning := 10
def lawns_number := 2
def lawn_earning := 13
def bicycle_cost := 80
def needed_amount := 24

-- Auxiliary calculations
def lawns_total_earning := lawns_number * lawn_earning
def already_earning := bicycle_cost - needed_amount
def cars_total_earning := already_earning - lawns_total_earning

-- Main problem statement
theorem stormi_cars_washed : (cars_total_earning / cars_earning) = 3 :=
  by sorry

end NUMINAMATH_GPT_stormi_cars_washed_l139_13950


namespace NUMINAMATH_GPT_pugs_cleaning_time_l139_13997

theorem pugs_cleaning_time : 
  (∀ (p t: ℕ), 15 * 12 = p * t ↔ 15 * 12 = 4 * 45) :=
by
  sorry

end NUMINAMATH_GPT_pugs_cleaning_time_l139_13997


namespace NUMINAMATH_GPT_computer_operations_correct_l139_13917

-- Define the rate of operations per second
def operations_per_second : ℝ := 4 * 10^8

-- Define the total number of seconds the computer operates
def total_seconds : ℝ := 6 * 10^5

-- Define the expected total number of operations
def expected_operations : ℝ := 2.4 * 10^14

-- Theorem stating the total number of operations is as expected
theorem computer_operations_correct :
  operations_per_second * total_seconds = expected_operations :=
by
  sorry

end NUMINAMATH_GPT_computer_operations_correct_l139_13917


namespace NUMINAMATH_GPT_eq_exponents_l139_13966

theorem eq_exponents (m n : ℤ) : ((5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n) → (m = 0 ∧ n = 0) :=
by
  sorry

end NUMINAMATH_GPT_eq_exponents_l139_13966


namespace NUMINAMATH_GPT_brian_expenses_l139_13936

def cost_apples_per_bag : ℕ := 14
def cost_kiwis : ℕ := 10
def cost_bananas : ℕ := cost_kiwis / 2
def subway_fare_one_way : ℕ := 350
def maximum_apples : ℕ := 24

theorem brian_expenses : 
  cost_kiwis + cost_bananas + (cost_apples_per_bag * (maximum_apples / 12)) + (subway_fare_one_way * 2) = 50 := by
sorry

end NUMINAMATH_GPT_brian_expenses_l139_13936


namespace NUMINAMATH_GPT_P_has_no_negative_roots_but_at_least_one_positive_root_l139_13921

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^6 - 4*x^5 - 9*x^3 + 2*x + 9

-- Statement of the problem
theorem P_has_no_negative_roots_but_at_least_one_positive_root :
  (∀ x : ℝ, x < 0 → P x ≠ 0 ∧ P x > 0) ∧ (∃ x : ℝ, x > 0 ∧ P x = 0) :=
by
  sorry

end NUMINAMATH_GPT_P_has_no_negative_roots_but_at_least_one_positive_root_l139_13921


namespace NUMINAMATH_GPT_geometric_sequence_sum_l139_13964

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_common_ratio : ∀ n, a (n + 1) = 2 * a n)
    (h_sum : a 1 + a 2 + a 3 = 21) : a 3 + a 4 + a 5 = 84 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l139_13964


namespace NUMINAMATH_GPT_gcd_lcm_product_l139_13914

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 24) (h₂ : b = 36) :
  Nat.gcd a b * Nat.lcm a b = 864 := 
by
  rw [h₁, h₂]
  -- You can include specific calculation just to express the idea
  -- rw [Nat.gcd_comm, Nat.gcd_rec]
  -- rw [Nat.lcm_def]
  -- rw [Nat.mul_subst]
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l139_13914


namespace NUMINAMATH_GPT_gcd_condition_l139_13930

theorem gcd_condition (a b c : ℕ) (h1 : Nat.gcd a b = 255) (h2 : Nat.gcd a c = 855) :
  Nat.gcd b c = 15 :=
sorry

end NUMINAMATH_GPT_gcd_condition_l139_13930


namespace NUMINAMATH_GPT_cakes_served_yesterday_l139_13993

theorem cakes_served_yesterday (cakes_today_lunch : ℕ) (cakes_today_dinner : ℕ) (total_cakes : ℕ)
  (h1 : cakes_today_lunch = 5) (h2 : cakes_today_dinner = 6) (h3 : total_cakes = 14) :
  total_cakes - (cakes_today_lunch + cakes_today_dinner) = 3 :=
by
  -- Import necessary libraries
  sorry

end NUMINAMATH_GPT_cakes_served_yesterday_l139_13993


namespace NUMINAMATH_GPT_isabella_hair_growth_l139_13947

def initial_hair_length : ℝ := 18
def final_hair_length : ℝ := 24
def hair_growth : ℝ := final_hair_length - initial_hair_length

theorem isabella_hair_growth : hair_growth = 6 := by
  sorry

end NUMINAMATH_GPT_isabella_hair_growth_l139_13947


namespace NUMINAMATH_GPT_part_I_l139_13965

variable (a b c n p q : ℝ)

theorem part_I (hne0 : a ≠ 0) (bne0 : b ≠ 0) (cne0 : c ≠ 0)
    (h1 : a^2 + b^2 + c^2 = 2) (h2 : n^2 + p^2 + q^2 = 2) :
    (n^4 / a^2 + p^4 / b^2 + q^4 / c^2) ≥ 2 := 
sorry

end NUMINAMATH_GPT_part_I_l139_13965


namespace NUMINAMATH_GPT_min_value_ineq_l139_13912

noncomputable def a_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ a 2018 = a 2017 + 2 * a 2016

theorem min_value_ineq (a : ℕ → ℝ) (m n : ℕ) 
  (h : a_sequence a) 
  (h2 : a m * a n = 16 * (a 1) ^ 2) :
  (4 / m) + (1 / n) ≥ 5 / 3 :=
sorry

end NUMINAMATH_GPT_min_value_ineq_l139_13912


namespace NUMINAMATH_GPT_opposite_seven_is_minus_seven_l139_13931

theorem opposite_seven_is_minus_seven :
  ∃ x : ℤ, 7 + x = 0 ∧ x = -7 := 
sorry

end NUMINAMATH_GPT_opposite_seven_is_minus_seven_l139_13931


namespace NUMINAMATH_GPT_find_k_l139_13902

variable {a_n : ℕ → ℤ}    -- Define the arithmetic sequence as a function from natural numbers to integers
variable {a1 d : ℤ}        -- a1 is the first term, d is the common difference

-- Conditions
axiom seq_def : ∀ n, a_n n = a1 + (n - 1) * d
axiom sum_condition : 9 * a1 + 36 * d = 4 * a1 + 6 * d
axiom ak_a4_zero (k : ℕ): a_n 4 + a_n k = 0

-- Problem Statement to prove
theorem find_k : ∃ k : ℕ, a_n 4 + a_n k = 0 → k = 10 :=
by
  use 10
  intro h
  -- proof omitted
  sorry

end NUMINAMATH_GPT_find_k_l139_13902


namespace NUMINAMATH_GPT_tan_A_of_triangle_conditions_l139_13906

open Real

def triangle_angles (A B C : ℝ) : Prop :=
  A + B + C = π ∧ 0 < A ∧ A < π / 2 ∧ B = π / 4

def form_arithmetic_sequence (a b c : ℝ) : Prop :=
  2 * b^2 = a^2 + c^2

theorem tan_A_of_triangle_conditions
  (A B C a b c : ℝ)
  (h_angles : triangle_angles A B C)
  (h_seq : form_arithmetic_sequence a b c) :
  tan A = sqrt 2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_A_of_triangle_conditions_l139_13906


namespace NUMINAMATH_GPT_min_sum_l139_13946

namespace MinimumSum

theorem min_sum (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (hc : 98 * m = n^3) : m + n = 42 :=
sorry

end MinimumSum

end NUMINAMATH_GPT_min_sum_l139_13946


namespace NUMINAMATH_GPT_length_of_wall_l139_13949

theorem length_of_wall (side_mirror length_wall width_wall : ℕ) 
  (mirror_area wall_area : ℕ) (H1 : side_mirror = 54) 
  (H2 : mirror_area = side_mirror * side_mirror) 
  (H3 : wall_area = 2 * mirror_area) 
  (H4 : width_wall = 68) 
  (H5 : wall_area = length_wall * width_wall) : 
  length_wall = 86 :=
by
  sorry

end NUMINAMATH_GPT_length_of_wall_l139_13949


namespace NUMINAMATH_GPT_range_of_fx₂_l139_13919

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * x + a * Real.log x

def is_extreme_point (a x : ℝ) : Prop := 
  (2 * x^2 - 2 * x + a) / x = 0

theorem range_of_fx₂ (a x₁ x₂ : ℝ) (h₀ : 0 < a) (h₁ : a < 1 / 2) 
  (h₂ : 0 < x₁ ∧ x₁ < x₂) (h₃ : is_extreme_point a x₁)
  (h₄ : is_extreme_point a x₂) : 
  (f a x₂) ∈ (Set.Ioo (-(3 + 2 * Real.log 2) / 4) (-1)) :=
sorry

end NUMINAMATH_GPT_range_of_fx₂_l139_13919
