import Mathlib

namespace power_of_i_l1510_151046

theorem power_of_i : (Complex.I ^ 2018) = -1 := by
  sorry

end power_of_i_l1510_151046


namespace solve_inequality_l1510_151083

theorem solve_inequality (x : ℝ) : 
  (-9 * x^2 + 6 * x + 15 > 0) ↔ (x > -1 ∧ x < 5/3) := 
sorry

end solve_inequality_l1510_151083


namespace avg_calculation_l1510_151020

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem avg_calculation :
  avg4 (avg2 1 2) (avg2 3 1) (avg2 2 0) (avg2 1 1) = 11 / 8 := by
  sorry

end avg_calculation_l1510_151020


namespace find_base_b_l1510_151089

theorem find_base_b :
  ∃ b : ℕ, (b > 7) ∧ (b > 10) ∧ (b > 8) ∧ (b > 12) ∧ 
    (4 + 3 = 7) ∧ ((2 + 7 + 1) % b = 3) ∧ ((3 + 4 + 1) % b = 5) ∧ 
    ((5 + 6 + 1) % b = 2) ∧ (1 + 1 = 2)
    ∧ b = 13 :=
by
  sorry

end find_base_b_l1510_151089


namespace remainder_698_div_D_l1510_151018

-- Defining the conditions
variables (D k1 k2 k3 R : ℤ)

-- Given conditions
axiom condition1 : 242 = k1 * D + 4
axiom condition2 : 940 = k3 * D + 7
axiom condition3 : 698 = k2 * D + R

-- The theorem to prove the remainder 
theorem remainder_698_div_D : R = 3 :=
by
  -- Here you would provide the logical deduction steps
  sorry

end remainder_698_div_D_l1510_151018


namespace fraction_ratio_l1510_151036

variable {α : Type*} [DivisionRing α] (a b : α)

theorem fraction_ratio (h1 : 2 * a = 3 * b) (h2 : b ≠ 0) : a / b = 3 / 2 := 
by sorry

end fraction_ratio_l1510_151036


namespace find_eccentricity_l1510_151065

def geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

def eccentricity_conic_section (m : ℝ) (e : ℝ) : Prop :=
  (m = 6 → e = (Real.sqrt 30) / 6) ∧
  (m = -6 → e = Real.sqrt 7)

theorem find_eccentricity (m : ℝ) :
  geometric_sequence 4 m 9 →
  eccentricity_conic_section m ((Real.sqrt 30) / 6) ∨
  eccentricity_conic_section m (Real.sqrt 7) :=
by
  sorry

end find_eccentricity_l1510_151065


namespace proof_x_plus_y_equals_30_l1510_151068

variable (x y : ℝ) (h_distinct : x ≠ y)
variable (h_det : Matrix.det ![
  ![2, 5, 10],
  ![4, x, y],
  ![4, y, x]
  ] = 0)

theorem proof_x_plus_y_equals_30 :
  x + y = 30 :=
sorry

end proof_x_plus_y_equals_30_l1510_151068


namespace chords_and_circle_l1510_151039

theorem chords_and_circle (R : ℝ) (A B C D : ℝ) 
  (hAB : 0 < A - B) (hCD : 0 < C - D) (hR : R > 0) 
  (h_perp : (A - B) * (C - D) = 0) 
  (h_radA : A ^ 2 + B ^ 2 = R ^ 2) 
  (h_radC : C ^ 2 + D ^ 2 = R ^ 2) :
  (A - C)^2 + (B - D)^2 = 4 * R^2 :=
by
  sorry

end chords_and_circle_l1510_151039


namespace probability_shaded_is_one_third_l1510_151060

-- Define the total number of regions as a constant
def total_regions : ℕ := 12

-- Define the number of shaded regions as a constant
def shaded_regions : ℕ := 4

-- The probability that the tip of a spinner stopping in a shaded region
def probability_shaded : ℚ := shaded_regions / total_regions

-- Main theorem stating the probability calculation is correct
theorem probability_shaded_is_one_third : probability_shaded = 1 / 3 :=
by
  sorry

end probability_shaded_is_one_third_l1510_151060


namespace smallest_a1_l1510_151029

theorem smallest_a1 (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_rec : ∀ n > 1, a n = 7 * a (n - 1) - n) :
  a 1 ≥ 13 / 36 :=
by
  sorry

end smallest_a1_l1510_151029


namespace find_g_2_l1510_151024

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_2
  (H : ∀ (x : ℝ), x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x ^ 2):
  g 2 = 67 / 14 :=
by
  sorry

end find_g_2_l1510_151024


namespace problem1_l1510_151061

theorem problem1 (x y : ℝ) (h1 : 2^(x + y) = x + 7) (h2 : x + y = 3) : (x = 1 ∧ y = 2) :=
by
  sorry

end problem1_l1510_151061


namespace question_1_question_2_l1510_151027

variable (a b c : ℝ × ℝ)
variable (k : ℝ)

def vect_a : ℝ × ℝ := (3, 2)
def vect_b : ℝ × ℝ := (-1, 2)
def vect_c : ℝ × ℝ := (4, 1)

theorem question_1 :
  3 • vect_a + vect_b - 2 • vect_c = (0, 6) := 
by
  sorry

theorem question_2 (k : ℝ) : 
  let lhs := (3 + 4 * k) * 2
  let rhs := -5 * (2 + k)
  (lhs = rhs) → k = -16 / 13 := 
by
  sorry

end question_1_question_2_l1510_151027


namespace volleyball_team_selection_l1510_151007

theorem volleyball_team_selection (total_players starting_players : ℕ) (libero : ℕ) : 
  total_players = 12 → 
  starting_players = 6 → 
  libero = 1 →
  (∃ (ways : ℕ), ways = 5544) :=
by
  intros h1 h2 h3
  sorry

end volleyball_team_selection_l1510_151007


namespace find_x_for_opposite_directions_l1510_151005

-- Define the vectors and the opposite direction condition
def vector_a (x : ℝ) : ℝ × ℝ := (1, -x)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -16)

-- Define the condition that vectors are in opposite directions
def opp_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = (-k) • b

-- The main theorem statement
theorem find_x_for_opposite_directions : ∃ x : ℝ, opp_directions (vector_a x) (vector_b x) ∧ x = -5 := 
sorry

end find_x_for_opposite_directions_l1510_151005


namespace number_of_strictly_increasing_sequences_l1510_151032

def strictly_increasing_sequences (n : ℕ) : ℕ :=
if n = 0 then 1 else if n = 1 then 1 else strictly_increasing_sequences (n - 1) + strictly_increasing_sequences (n - 2)

theorem number_of_strictly_increasing_sequences :
  strictly_increasing_sequences 12 = 144 :=
by
  sorry

end number_of_strictly_increasing_sequences_l1510_151032


namespace maximize_profit_l1510_151040

-- Define constants for purchase and selling prices
def priceA_purchase : ℝ := 16
def priceA_selling : ℝ := 20
def priceB_purchase : ℝ := 20
def priceB_selling : ℝ := 25

-- Define constant for total weight
def total_weight : ℝ := 200

-- Define profit function
def profit (weightA weightB : ℝ) : ℝ :=
  (priceA_selling - priceA_purchase) * weightA + (priceB_selling - priceB_purchase) * weightB

-- Define constraints
def constraint1 (weightA weightB : ℝ) : Prop :=
  weightA + weightB = total_weight

def constraint2 (weightA weightB : ℝ) : Prop :=
  weightA >= 3 * weightB

open Real

-- Define the maximum profit we aim to prove
def max_profit : ℝ := 850

-- The main theorem to prove
theorem maximize_profit : 
  ∃ weightA weightB : ℝ, constraint1 weightA weightB ∧ constraint2 weightA weightB ∧ profit weightA weightB = max_profit :=
by {
  sorry
}

end maximize_profit_l1510_151040


namespace remainder_T10_mod_5_l1510_151085

noncomputable def T : ℕ → ℕ
| 0     => 1
| 1     => 2
| (n+2) => T (n+1) + T n + T n

theorem remainder_T10_mod_5 :
  (T 10) % 5 = 4 :=
sorry

end remainder_T10_mod_5_l1510_151085


namespace compute_abc_l1510_151066

theorem compute_abc (a b c : ℤ) (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h₁ : a + b + c = 30) 
  (h₂ : (1 : ℚ)/a + (1 : ℚ)/b + (1 : ℚ)/c + 300/(a * b * c) = 1) : a * b * c = 768 := 
by 
  sorry

end compute_abc_l1510_151066


namespace ab_squared_non_positive_l1510_151086

theorem ab_squared_non_positive (a b : ℝ) (h : 7 * a + 9 * |b| = 0) : a * b^2 ≤ 0 :=
sorry

end ab_squared_non_positive_l1510_151086


namespace fraction_of_larger_part_l1510_151017

theorem fraction_of_larger_part (x y : ℝ) (f : ℝ) (h1 : x = 50) (h2 : x + y = 66) (h3 : f * x = 0.625 * y + 10) : f = 0.4 :=
by
  sorry

end fraction_of_larger_part_l1510_151017


namespace QED_mul_eq_neg_25I_l1510_151004

namespace ComplexMultiplication

open Complex

def Q : ℂ := 3 + 4 * Complex.I
def E : ℂ := -Complex.I
def D : ℂ := 3 - 4 * Complex.I

theorem QED_mul_eq_neg_25I : Q * E * D = -25 * Complex.I :=
by
  sorry

end ComplexMultiplication

end QED_mul_eq_neg_25I_l1510_151004


namespace quadratic_solution_l1510_151098

noncomputable def g (x : ℝ) : ℝ := x^2 + 2021 * x + 18

theorem quadratic_solution : ∀ x : ℝ, g (g x + x + 1) / g x = x^2 + 2023 * x + 2040 :=
by
  intros
  sorry

end quadratic_solution_l1510_151098


namespace owen_sleep_hours_l1510_151038

-- Define the time spent by Owen in various activities
def hours_work : ℕ := 6
def hours_chores : ℕ := 7
def total_hours_day : ℕ := 24

-- The proposition to be proven
theorem owen_sleep_hours : (total_hours_day - (hours_work + hours_chores) = 11) := by
  sorry

end owen_sleep_hours_l1510_151038


namespace montana_more_than_ohio_l1510_151064

-- Define the total number of combinations for Ohio and Montana
def ohio_combinations : ℕ := 26^4 * 10^3
def montana_combinations : ℕ := 26^5 * 10^2

-- The total number of combinations from both states
def ohio_total : ℕ := ohio_combinations
def montana_total : ℕ := montana_combinations

-- Prove the difference
theorem montana_more_than_ohio : montana_total - ohio_total = 731161600 := by
  sorry

end montana_more_than_ohio_l1510_151064


namespace minimize_g_function_l1510_151006

noncomputable def g (x : ℝ) : ℝ := (9 * x^2 + 18 * x + 29) / (8 * (2 + x))

theorem minimize_g_function : ∀ x : ℝ, x ≥ -1 → g x = 29 / 8 :=
sorry

end minimize_g_function_l1510_151006


namespace part_time_employees_l1510_151055

theorem part_time_employees (total_employees : ℕ) (full_time_employees : ℕ) (part_time_employees : ℕ) 
  (h1 : total_employees = 65134) 
  (h2 : full_time_employees = 63093) 
  (h3 : total_employees = full_time_employees + part_time_employees) : 
  part_time_employees = 2041 :=
by 
  sorry

end part_time_employees_l1510_151055


namespace range_of_m_l1510_151048

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.exp (2 * x)
noncomputable def g (m x : ℝ) : ℝ := m * x + 1

def exists_x0 (x1 : ℝ) (m : ℝ) : Prop :=
  ∃ (x0 : ℝ), -1 ≤ x0 ∧ x0 ≤ 1 ∧ g m x0 = f x1

theorem range_of_m (m : ℝ) (cond : ∀ (x1 : ℝ), -1 ≤ x1 → x1 ≤ 1 → exists_x0 x1 m) :
  m ∈ Set.Iic (1 - Real.exp 2) ∨ m ∈ Set.Ici (Real.exp 2 - 1) :=
sorry

end range_of_m_l1510_151048


namespace westward_measurement_l1510_151009

def east_mov (d : ℕ) : ℤ := - (d : ℤ)

def west_mov (d : ℕ) : ℤ := d

theorem westward_measurement :
  east_mov 50 = -50 →
  west_mov 60 = 60 :=
by
  intro h
  exact rfl

end westward_measurement_l1510_151009


namespace golden_section_AC_length_l1510_151019

namespace GoldenSection

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def AC_length (AB : ℝ) : ℝ :=
  let φ := golden_ratio
  AB / φ

theorem golden_section_AC_length (AB : ℝ) (C_gold : Prop) (hAB : AB = 2) (A_gt_B : AC_length AB > AB - AC_length AB) :
  AC_length AB = Real.sqrt 5 - 1 :=
  sorry

end GoldenSection

end golden_section_AC_length_l1510_151019


namespace average_value_eq_l1510_151016

variable (x : ℝ)

theorem average_value_eq :
  ( -4 * x + 0 + 4 * x + 12 * x + 20 * x ) / 5 = 6.4 * x :=
by
  sorry

end average_value_eq_l1510_151016


namespace triangle_area_l1510_151090

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : C = π / 3) : 
  (1/2 * a * b * Real.sin C) = (3 * Real.sqrt 3 / 2) :=
by
  sorry

end triangle_area_l1510_151090


namespace four_digit_numbers_starting_with_1_l1510_151050

theorem four_digit_numbers_starting_with_1 
: ∃ n : ℕ, (n = 234) ∧ 
  (∀ (x y z : ℕ), 
    (x ≠ y → x ≠ z → y ≠ z → -- ensuring these constraints
    x ≠ 1 → y ≠ 1 → z = 1 → -- exactly two identical digits which include 1
    (x * 1000 + y * 100 + z * 10 + 1) / 1000 = 1 ∨ (x * 1000 + z * 100 + y * 10 + 1) / 1000 = 1) ∨ 
    (∃ (x y : ℕ),  
    (x ≠ y → x ≠ 1 → y = 1 → 
    (x * 110 + y * 10 + 1) + (x * 11 + y * 10 + 1) + (x * 100 + y * 10 + 1) + (x * 110 + 1) = n))) := sorry

end four_digit_numbers_starting_with_1_l1510_151050


namespace find_number_l1510_151030

variable (x : ℝ)

theorem find_number 
  (h1 : 0.20 * x + 0.25 * 60 = 23) :
  x = 40 :=
sorry

end find_number_l1510_151030


namespace complex_fraction_simplification_l1510_151078

theorem complex_fraction_simplification :
  ((10^4 + 324) * (22^4 + 324) * (34^4 + 324) * (46^4 + 324) * (58^4 + 324)) /
  ((4^4 + 324) * (16^4 + 324) * (28^4 + 324) * (40^4 + 324) * (52^4 + 324)) = 373 :=
by
  sorry

end complex_fraction_simplification_l1510_151078


namespace find_factor_l1510_151094

-- Definitions based on the conditions
def n : ℤ := 155
def result : ℤ := 110
def constant : ℤ := 200

-- Statement to be proved
theorem find_factor (f : ℤ) (h : n * f - constant = result) : f = 2 := by
  sorry

end find_factor_l1510_151094


namespace compound_interest_rate_l1510_151071

theorem compound_interest_rate :
  ∀ (A P : ℝ) (t : ℕ),
  A = 4840.000000000001 ->
  P = 4000 ->
  t = 2 ->
  A = P * (1 + 0.1)^t :=
by
  intros A P t hA hP ht
  rw [hA, hP, ht]
  norm_num
  sorry

end compound_interest_rate_l1510_151071


namespace candy_store_spending_l1510_151062

variable (weekly_allowance : ℝ) (arcade_fraction : ℝ) (toy_store_fraction : ℝ)

def remaining_after_arcade (weekly_allowance arcade_fraction : ℝ) : ℝ :=
  weekly_allowance * (1 - arcade_fraction)

def remaining_after_toy_store (remaining_allowance toy_store_fraction : ℝ) : ℝ :=
  remaining_allowance * (1 - toy_store_fraction)

theorem candy_store_spending
  (h1 : weekly_allowance = 3.30)
  (h2 : arcade_fraction = 3 / 5)
  (h3 : toy_store_fraction = 1 / 3) :
  remaining_after_toy_store (remaining_after_arcade weekly_allowance arcade_fraction) toy_store_fraction = 0.88 := 
sorry

end candy_store_spending_l1510_151062


namespace borrowed_amount_correct_l1510_151002

noncomputable def principal_amount (I: ℚ) (r1 r2 r3 r4 t1 t2 t3 t4: ℚ): ℚ :=
  I / (r1 * t1 + r2 * t2 + r3 * t3 + r4 * t4)

def interest_rate_1 := (6.5 / 100 : ℚ)
def interest_rate_2 := (9.5 / 100 : ℚ)
def interest_rate_3 := (11 / 100 : ℚ)
def interest_rate_4 := (14.5 / 100 : ℚ)

def time_period_1 := (2.5 : ℚ)
def time_period_2 := (3.75 : ℚ)
def time_period_3 := (1.5 : ℚ)
def time_period_4 := (4.25 : ℚ)

def total_interest := (14500 : ℚ)

def expected_principal := (11153.846153846154 : ℚ)

theorem borrowed_amount_correct :
  principal_amount total_interest interest_rate_1 interest_rate_2 interest_rate_3 interest_rate_4 time_period_1 time_period_2 time_period_3 time_period_4 = expected_principal :=
by
  sorry

end borrowed_amount_correct_l1510_151002


namespace tenth_pirate_receives_exactly_1296_coins_l1510_151011

noncomputable def pirate_coins (n : ℕ) : ℕ :=
  if n = 0 then 0
  else Nat.factorial 9 / 11^9 * 11^(10 - n)

theorem tenth_pirate_receives_exactly_1296_coins :
  pirate_coins 10 = 1296 :=
sorry

end tenth_pirate_receives_exactly_1296_coins_l1510_151011


namespace smallest_n_mod_l1510_151000

theorem smallest_n_mod : ∃ n : ℕ, 5 * n ≡ 2024 [MOD 26] ∧ n > 0 ∧ ∀ m : ℕ, (5 * m ≡ 2024 [MOD 26] ∧ m > 0) → n ≤ m :=
  sorry

end smallest_n_mod_l1510_151000


namespace find_f_4_l1510_151031

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_4 : (∀ x : ℝ, f (x / 2 - 1) = 2 * x + 3) → f 4 = 23 :=
by
  sorry

end find_f_4_l1510_151031


namespace range_of_k_l1510_151088

open Set

variable {k : ℝ}

def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < k + 1}

theorem range_of_k (h : (compl A) ∩ B k ≠ ∅) : 0 < k ∧ k < 3 := sorry

end range_of_k_l1510_151088


namespace dog_biscuit_cost_l1510_151075

open Real

theorem dog_biscuit_cost :
  (∀ (x : ℝ),
    (4 * x + 2) * 7 = 21 →
    x = 1 / 4) :=
by
  intro x h
  sorry

end dog_biscuit_cost_l1510_151075


namespace kim_shoes_l1510_151096

variable (n : ℕ)

theorem kim_shoes : 
  (∀ n, 2 * n = 6 → (1 : ℚ) / (2 * n - 1) = (1 : ℚ) / 5 → n = 3) := 
sorry

end kim_shoes_l1510_151096


namespace cost_of_snake_toy_l1510_151025

-- Given conditions
def cost_of_cage : ℝ := 14.54
def dollar_bill_found : ℝ := 1.00
def total_cost : ℝ := 26.30

-- Theorem to find the cost of the snake toy
theorem cost_of_snake_toy : 
  (total_cost + dollar_bill_found - cost_of_cage) = 12.76 := 
  by sorry

end cost_of_snake_toy_l1510_151025


namespace divide_value_l1510_151087

def divide (a b c : ℝ) : ℝ := |b^2 - 5 * a * c|

theorem divide_value : divide 2 (-3) 1 = 1 :=
by
  sorry

end divide_value_l1510_151087


namespace value_of_c_l1510_151077

theorem value_of_c (a b c d w x y z : ℕ) (primes : ∀ p ∈ [w, x, y, z], Prime p)
  (h1 : w < x) (h2 : x < y) (h3 : y < z) 
  (h4 : (w^a) * (x^b) * (y^c) * (z^d) = 660) 
  (h5 : (a + b) - (c + d) = 1) : c = 1 :=
by {
  sorry
}

end value_of_c_l1510_151077


namespace simplify_expression_l1510_151082

theorem simplify_expression :
  ((9 * 10^8) * 2^2) / (3 * 2^3 * 10^3) = 150000 := by sorry

end simplify_expression_l1510_151082


namespace smallest_positive_period_centers_of_symmetry_maximum_value_minimum_value_l1510_151053

noncomputable def f (x : ℝ) : ℝ := -2 * (Real.sin x)^2 + 2 * (Real.sin x) * (Real.cos x) + 1

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem centers_of_symmetry :
  ∀ k : ℤ, ∃ x, x = -Real.pi / 4 + k * Real.pi ∧ f (-x) = f x := sorry

theorem maximum_value :
  ∀ x : ℝ, f x ≤ 2 := sorry

theorem minimum_value :
  ∀ x : ℝ, f x ≥ -1 := sorry

end smallest_positive_period_centers_of_symmetry_maximum_value_minimum_value_l1510_151053


namespace negation_of_universal_proposition_l1510_151013

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - x + 1 / 4 > 0)) = ∃ x : ℝ, x^2 - x + 1 / 4 ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l1510_151013


namespace log_expression_value_l1510_151033

theorem log_expression_value : 
  (Real.logb 10 (Real.sqrt 2) + Real.logb 10 (Real.sqrt 5) + 2 ^ 0 + (5 ^ (1 / 3)) ^ 2 * Real.sqrt 5 = 13 / 2) := 
by 
  -- The proof is omitted as per the instructions
  sorry

end log_expression_value_l1510_151033


namespace alex_loan_comparison_l1510_151015

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r * t)

theorem alex_loan_comparison :
  let P : ℝ := 15000
  let r1 : ℝ := 0.08
  let r2 : ℝ := 0.10
  let n : ℕ := 12
  let t1_10 : ℝ := 10
  let t1_5 : ℝ := 5
  let t2 : ℝ := 15
  let owed_after_10 := compound_interest P r1 n t1_10
  let payment_after_10 := owed_after_10 / 2
  let remaining_after_10 := owed_after_10 / 2
  let owed_after_15 := compound_interest remaining_after_10 r1 n t1_5
  let total_payment_option1 := payment_after_10 + owed_after_15
  let total_payment_option2 := simple_interest P r2 t2
  total_payment_option1 - total_payment_option2 = 4163 :=
by
  sorry

end alex_loan_comparison_l1510_151015


namespace crackers_per_person_l1510_151052

theorem crackers_per_person:
  ∀ (total_crackers friends : ℕ), total_crackers = 36 → friends = 18 → total_crackers / friends = 2 :=
by
  intros total_crackers friends h1 h2
  sorry

end crackers_per_person_l1510_151052


namespace number_of_males_who_listen_l1510_151042

theorem number_of_males_who_listen (females_listen : ℕ) (males_dont_listen : ℕ) (total_listen : ℕ) (total_dont_listen : ℕ) (total_females : ℕ) :
  females_listen = 72 →
  males_dont_listen = 88 →
  total_listen = 160 →
  total_dont_listen = 180 →
  (total_females = total_listen + total_dont_listen - (females_listen + males_dont_listen)) →
  (total_females + males_dont_listen + 92 = total_listen + total_dont_listen) →
  total_listen + total_dont_listen = females_listen + males_dont_listen + (total_females - females_listen) + 92 :=
sorry

end number_of_males_who_listen_l1510_151042


namespace tangent_line_y_intercept_l1510_151003

def circle1Center: ℝ × ℝ := (3, 0)
def circle1Radius: ℝ := 3
def circle2Center: ℝ × ℝ := (7, 0)
def circle2Radius: ℝ := 2

theorem tangent_line_y_intercept
    (tangent_line: ℝ × ℝ -> ℝ) 
    (P : tangent_line (circle1Center.1, circle1Center.2 + circle1Radius) = 0) -- Tangent condition for Circle 1
    (Q : tangent_line (circle2Center.1, circle2Center.2 + circle2Radius) = 0) -- Tangent condition for Circle 2
    :
    tangent_line (0, 4.5) = 0 := 
sorry

end tangent_line_y_intercept_l1510_151003


namespace leading_coefficient_of_f_l1510_151084

noncomputable def polynomial : Type := ℕ → ℝ

def satisfies_condition (f : polynomial) : Prop :=
  ∀ (x : ℕ), f (x + 1) - f x = 6 * x + 4

theorem leading_coefficient_of_f (f : polynomial) (h : satisfies_condition f) : 
  ∃ a b c : ℝ, (∀ (x : ℕ), f x = a * (x^2) + b * x + c) ∧ a = 3 := 
by
  sorry

end leading_coefficient_of_f_l1510_151084


namespace fraction_simplification_l1510_151043

theorem fraction_simplification :
  ∃ (p q : ℕ), p = 2021 ∧ q ≠ 0 ∧ gcd p q = 1 ∧ (1011 / 1010) - (1010 / 1011) = (p : ℚ) / q := 
sorry

end fraction_simplification_l1510_151043


namespace inequality_holds_for_m_l1510_151008

theorem inequality_holds_for_m (m : ℝ) :
  (-2 : ℝ) ≤ m ∧ m ≤ (3 : ℝ) ↔ ∀ x : ℝ, x < -1 →
    (m - m^2) * (4 : ℝ)^x + (2 : ℝ)^x + 1 > 0 :=
by sorry

end inequality_holds_for_m_l1510_151008


namespace distinct_ordered_pairs_proof_l1510_151010

def num_distinct_ordered_pairs_satisfying_reciprocal_sum : ℕ :=
  List.length [
    (7, 42), (8, 24), (9, 18), (10, 15), 
    (12, 12), (15, 10), (18, 9), (24, 8), 
    (42, 7)
  ]

theorem distinct_ordered_pairs_proof : num_distinct_ordered_pairs_satisfying_reciprocal_sum = 9 := by
  sorry

end distinct_ordered_pairs_proof_l1510_151010


namespace eight_and_five_l1510_151093

def my_and (a b : ℕ) : ℕ := (a + b) ^ 2 * (a - b)

theorem eight_and_five : my_and 8 5 = 507 := 
  by sorry

end eight_and_five_l1510_151093


namespace jeff_total_distance_l1510_151035

-- Define the conditions as constants
def speed1 : ℝ := 80
def time1 : ℝ := 3

def speed2 : ℝ := 50
def time2 : ℝ := 2

def speed3 : ℝ := 70
def time3 : ℝ := 1

def speed4 : ℝ := 60
def time4 : ℝ := 2

def speed5 : ℝ := 45
def time5 : ℝ := 3

def speed6 : ℝ := 40
def time6 : ℝ := 2

def speed7 : ℝ := 30
def time7 : ℝ := 2.5

-- Define the equation for the total distance traveled
def total_distance : ℝ :=
  speed1 * time1 + 
  speed2 * time2 + 
  speed3 * time3 + 
  speed4 * time4 + 
  speed5 * time5 + 
  speed6 * time6 + 
  speed7 * time7

-- Prove that the total distance is equal to 820 miles
theorem jeff_total_distance : total_distance = 820 := by
  sorry

end jeff_total_distance_l1510_151035


namespace coffee_break_l1510_151001

theorem coffee_break (n k : ℕ) (h1 : n = 14) (h2 : 0 < n - 2 * k) (h3 : n - 2 * k < n) :
  n - 2 * k = 6 ∨ n - 2 * k = 8 ∨ n - 2 * k = 10 ∨ n - 2 * k = 12 :=
by
  sorry

end coffee_break_l1510_151001


namespace function_satisfies_conditions_l1510_151045

theorem function_satisfies_conditions :
  (∃ f : ℤ × ℤ → ℝ,
    (∀ x y z : ℤ, f (x, y) * f (y, z) * f (z, x) = 1) ∧
    (∀ x : ℤ, f (x + 1, x) = 2) ∧
    (∀ x y : ℤ, f (x, y) = 2 ^ (x - y))) :=
by
  sorry

end function_satisfies_conditions_l1510_151045


namespace total_travel_time_in_minutes_l1510_151076

def riding_rate : ℝ := 10 -- 10 miles per hour
def initial_riding_time : ℝ := 30 -- 30 minutes
def another_riding_distance : ℝ := 15 -- 15 miles
def resting_time : ℝ := 30 -- 30 minutes
def remaining_distance : ℝ := 20 -- 20 miles

theorem total_travel_time_in_minutes :
  initial_riding_time +
  (another_riding_distance / riding_rate * 60) +
  resting_time +
  (remaining_distance / riding_rate * 60) = 270 :=
by
  sorry

end total_travel_time_in_minutes_l1510_151076


namespace solution_set_linear_inequalities_l1510_151051

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l1510_151051


namespace johns_total_spent_l1510_151012

def total_spent (num_tshirts: Nat) (price_per_tshirt: Nat) (price_pants: Nat): Nat :=
  (num_tshirts * price_per_tshirt) + price_pants

theorem johns_total_spent : total_spent 3 20 50 = 110 := by
  sorry

end johns_total_spent_l1510_151012


namespace problem1_problem2_l1510_151067

theorem problem1 : -3 + (-2) * 5 - (-3) = -10 :=
by
  sorry

theorem problem2 : -1^4 + ((-5)^2 - 3) / |(-2)| = 10 :=
by
  sorry

end problem1_problem2_l1510_151067


namespace must_be_true_l1510_151080

noncomputable def f (x : ℝ) := |Real.log x|

theorem must_be_true (a b c : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : b < c) 
                     (h3 : f b < f a) (h4 : f a < f c) :
                     (c > 1) ∧ (1 / c < a) ∧ (a < 1) ∧ (a < b) ∧ (b < 1 / a) :=
by
  sorry

end must_be_true_l1510_151080


namespace books_left_l1510_151058

namespace PaulBooksExample

-- Defining the initial conditions as given in the problem
def initial_books : ℕ := 134
def books_given : ℕ := 39
def books_sold : ℕ := 27

-- Proving that the final number of books Paul has is 68
theorem books_left : initial_books - (books_given + books_sold) = 68 := by
  sorry

end PaulBooksExample

end books_left_l1510_151058


namespace four_digits_sum_l1510_151074

theorem four_digits_sum (A B C D : ℕ) 
  (A_neq_B : A ≠ B) (A_neq_C : A ≠ C) (A_neq_D : A ≠ D) 
  (B_neq_C : B ≠ C) (B_neq_D : B ≠ D) 
  (C_neq_D : C ≠ D)
  (digits_A : A ≤ 9) (digits_B : B ≤ 9) (digits_C : C ≤ 9) (digits_D : D ≤ 9)
  (A_lt_B : A < B) 
  (minimize_fraction : ∃ k : ℕ, (A + B) = k ∧ k ≤ (A + B) ∧ (C + D) ≥ (C + D)) :
  C + D = 17 := 
by
  sorry

end four_digits_sum_l1510_151074


namespace oranges_savings_l1510_151073

-- Definitions for the conditions
def liam_oranges : Nat := 40
def liam_price_per_set : Real := 2.50
def oranges_per_set : Nat := 2

def claire_oranges : Nat := 30
def claire_price_per_orange : Real := 1.20

-- Statement of the problem to be proven
theorem oranges_savings : 
  liam_oranges / oranges_per_set * liam_price_per_set + 
  claire_oranges * claire_price_per_orange = 86 := 
by 
  sorry

end oranges_savings_l1510_151073


namespace find_linear_function_l1510_151095

theorem find_linear_function (a : ℝ) (a_pos : 0 < a) :
  ∃ (b : ℝ), ∀ (f : ℕ → ℝ),
  (∀ (k m : ℕ), (a * m ≤ k ∧ k < (a + 1) * m) → f (k + m) = f k + f m) →
  ∀ n : ℕ, f n = b * n :=
sorry

end find_linear_function_l1510_151095


namespace peter_remaining_money_l1510_151014

def initial_amount : Float := 500.0 
def sales_tax : Float := 0.05
def discount : Float := 0.10

def calculate_cost_with_tax (price_per_kilo: Float) (quantity: Float) (tax_rate: Float) : Float :=
  quantity * price_per_kilo * (1 + tax_rate)

def calculate_cost_with_discount (price_per_kilo: Float) (quantity: Float) (discount_rate: Float) : Float :=
  quantity * price_per_kilo * (1 - discount_rate)

def total_first_trip : Float :=
  calculate_cost_with_tax 2.0 6 sales_tax +
  calculate_cost_with_tax 3.0 9 sales_tax +
  calculate_cost_with_tax 4.0 5 sales_tax +
  calculate_cost_with_tax 5.0 3 sales_tax +
  calculate_cost_with_tax 3.50 2 sales_tax +
  calculate_cost_with_tax 4.25 7 sales_tax +
  calculate_cost_with_tax 6.0 4 sales_tax +
  calculate_cost_with_tax 5.50 8 sales_tax

def total_second_trip : Float :=
  calculate_cost_with_discount 1.50 2 discount +
  calculate_cost_with_discount 2.75 5 discount

def remaining_money (initial: Float) (first_trip: Float) (second_trip: Float) : Float :=
  initial - first_trip - second_trip

theorem peter_remaining_money : remaining_money initial_amount total_first_trip total_second_trip = 297.24 := 
  by
    -- Proof omitted
    sorry

end peter_remaining_money_l1510_151014


namespace unique_reconstruction_l1510_151099

-- Definition of the sums on the edges given the face values
variables (a b c d e f : ℤ)

-- The 12 edge sums
variables (e₁ e₂ e₃ e₄ e₅ e₆ e₇ e₈ e₉ e₁₀ e₁₁ e₁₂ : ℤ)
variables (h₁ : e₁ = a + b) (h₂ : e₂ = a + c) (h₃ : e₃ = a + d) 
          (h₄ : e₄ = a + e) (h₅ : e₅ = b + c) (h₆ : e₆ = b + f) 
          (h₇ : e₇ = c + f) (h₈ : e₈ = d + f) (h₉ : e₉ = d + e)
          (h₁₀ : e₁₀ = e + f) (h₁₁ : e₁₁ = b + d) (h₁₂ : e₁₂ = c + e)

-- Proving that the face values can be uniquely determined given the edge sums
theorem unique_reconstruction :
  ∃ a' b' c' d' e' f' : ℤ, 
    (e₁ = a' + b') ∧ (e₂ = a' + c') ∧ (e₃ = a' + d') ∧ (e₄ = a' + e') ∧ 
    (e₅ = b' + c') ∧ (e₆ = b' + f') ∧ (e₇ = c' + f') ∧ (e₈ = d' + f') ∧ 
    (e₉ = d' + e') ∧ (e₁₀ = e' + f') ∧ (e₁₁ = b' + d') ∧ (e₁₂ = c' + e') ∧ 
    (a = a') ∧ (b = b') ∧ (c = c') ∧ (d = d') ∧ (e = e') ∧ (f = f') := by
  sorry

end unique_reconstruction_l1510_151099


namespace negation_of_exists_l1510_151044

theorem negation_of_exists {x : ℝ} (h : ∃ x : ℝ, 3^x + x < 0) : ∀ x : ℝ, 3^x + x ≥ 0 :=
sorry

end negation_of_exists_l1510_151044


namespace ratio_of_speeds_l1510_151063

noncomputable def speed_ratios (d t_b t : ℚ) : ℚ × ℚ  :=
  let d_b := t_b * t
  let d_a := d - d_b
  let t_h := t / 60
  let s_a := d_a / t_h
  let s_b := t_b
  (s_a / 15, s_b / 15)

theorem ratio_of_speeds
  (d : ℚ) (s_b : ℚ) (t : ℚ)
  (h : d = 88) (h1 : s_b = 90) (h2 : t = 32) :
  speed_ratios d s_b t = (5, 6) :=
  by
  sorry

end ratio_of_speeds_l1510_151063


namespace fraction_division_correct_l1510_151059

theorem fraction_division_correct :
  (2 / 5) / 3 = 2 / 15 :=
by sorry

end fraction_division_correct_l1510_151059


namespace fraction_representing_repeating_decimal_l1510_151079

theorem fraction_representing_repeating_decimal (x a b : ℕ) (h : x = 35) (h1 : 100 * x - x = 35) 
(h2 : ∃ (a b : ℕ), x = a / b ∧ gcd a b = 1 ∧ a + b = 134) : a + b = 134 := 
sorry

end fraction_representing_repeating_decimal_l1510_151079


namespace rex_cards_remaining_l1510_151092

theorem rex_cards_remaining
  (nicole_cards : ℕ)
  (cindy_cards : ℕ)
  (rex_cards : ℕ)
  (cards_per_person : ℕ)
  (h1 : nicole_cards = 400)
  (h2 : cindy_cards = 2 * nicole_cards)
  (h3 : rex_cards = (nicole_cards + cindy_cards) / 2)
  (h4 : cards_per_person = rex_cards / 4) :
  cards_per_person = 150 :=
by
  sorry

end rex_cards_remaining_l1510_151092


namespace problem1_problem2_real_problem2_complex_problem3_l1510_151037

-- Problem 1: Prove that if 2 ∈ A, then {-1, 1/2} ⊆ A
theorem problem1 (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : 2 ∈ A) : -1 ∈ A ∧ (1/2) ∈ A := sorry

-- Problem 2: Prove that A cannot be a singleton set for real numbers, but can for complex numbers.
theorem problem2_real (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : ∃ a ∈ A, a ≠ 1) : ¬(∃ a, A = {a}) := sorry

theorem problem2_complex (A : Set ℂ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : ∃ a ∈ A, a ≠ 1) : (∃ a, A = {a}) := sorry

-- Problem 3: Prove that 1 - 1/a ∈ A given a ∈ A
theorem problem3 (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (a : ℝ) (ha : a ∈ A) : (1 - 1/a) ∈ A := sorry

end problem1_problem2_real_problem2_complex_problem3_l1510_151037


namespace algebraic_expression_value_l1510_151047

-- Given conditions as definitions and assumption
variables (a b : ℝ)
def expression1 (x : ℝ) := 2 * a * x^3 - 3 * b * x + 8
def expression2 := 9 * b - 6 * a + 2

theorem algebraic_expression_value
  (h1 : expression1 (-1) = 18) :
  expression2 = 32 :=
by
  sorry

end algebraic_expression_value_l1510_151047


namespace proof1_proof2_proof3_proof4_l1510_151070

-- Define variables.
variable (m n x y z : ℝ)

-- Prove the expressions equalities.
theorem proof1 : (m + 2 * n) - (m - 2 * n) = 4 * n := sorry
theorem proof2 : 2 * (x - 3) - (-x + 4) = 3 * x - 10 := sorry
theorem proof3 : 2 * x - 3 * (x - 2 * y + 3 * x) + 2 * (3 * x - 3 * y + 2 * z) = -4 * x + 4 * z := sorry
theorem proof4 : 8 * m^2 - (4 * m^2 - 2 * m - 4 * (2 * m^2 - 5 * m)) = 12 * m^2 - 18 * m := sorry

end proof1_proof2_proof3_proof4_l1510_151070


namespace monotonic_intervals_extremum_values_l1510_151056

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 8

theorem monotonic_intervals :
  (∀ x, x < -1 → deriv f x > 0) ∧
  (∀ x, x > 2 → deriv f x > 0) ∧
  (∀ x, -1 < x ∧ x < 2 → deriv f x < 0) := sorry

theorem extremum_values :
  ∃ a b : ℝ, (a = -12) ∧ (b = 15) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 3 → f x ≥ b → f x = b) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 3 → f x ≤ a → f x = a) := sorry

end monotonic_intervals_extremum_values_l1510_151056


namespace equal_piles_l1510_151054

theorem equal_piles (initial_rocks final_piles : ℕ) (moves : ℕ) (total_rocks : ℕ) (rocks_per_pile : ℕ) :
  initial_rocks = 36 →
  final_piles = 7 →
  moves = final_piles - 1 →
  total_rocks = initial_rocks + moves →
  rocks_per_pile = total_rocks / final_piles →
  rocks_per_pile = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end equal_piles_l1510_151054


namespace parallel_lines_l1510_151049

theorem parallel_lines (a : ℝ) (h : ∀ x y : ℝ, 2*x - a*y - 1 = 0 → a*x - y = 0) : a = Real.sqrt 2 ∨ a = -Real.sqrt 2 :=
sorry

end parallel_lines_l1510_151049


namespace increasing_function_of_a_l1510_151069

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (2 - (a / 2)) * x + 2

theorem increasing_function_of_a (a : ℝ) : (∀ x y, x < y → f a x ≤ f a y) ↔ 
  (8 / 3 ≤ a ∧ a < 4) :=
sorry

end increasing_function_of_a_l1510_151069


namespace find_ordered_pair_l1510_151097

theorem find_ordered_pair (x y : ℤ) 
  (h1 : x + y = (7 - x) + (7 - y))
  (h2 : x - y = (x - 2) + (y - 2))
  : (x, y) = (5, 2) := 
sorry

end find_ordered_pair_l1510_151097


namespace cheryl_used_material_l1510_151034

theorem cheryl_used_material
    (material1 : ℚ) (material2 : ℚ) (leftover : ℚ)
    (h1 : material1 = 5/9)
    (h2 : material2 = 1/3)
    (h_lf : leftover = 8/24) :
    material1 + material2 - leftover = 5/9 :=
by
  sorry

end cheryl_used_material_l1510_151034


namespace recurring_decimal_addition_l1510_151021

noncomputable def recurring_decimal_sum : ℚ :=
  (23 / 99) + (14 / 999) + (6 / 9999)

theorem recurring_decimal_addition :
  recurring_decimal_sum = 2469 / 9999 :=
sorry

end recurring_decimal_addition_l1510_151021


namespace complement_intersection_l1510_151028

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def M : Set ℕ := {1, 4}
noncomputable def N : Set ℕ := {2, 3}

theorem complement_intersection :
  ((U \ M) ∩ N) = {2, 3} :=
by
  sorry

end complement_intersection_l1510_151028


namespace intersection_eq_l1510_151041

def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6 * x + 8 < 0}

theorem intersection_eq : M ∩ N = {x | 2 < x ∧ x < 3} := 
by
  sorry

end intersection_eq_l1510_151041


namespace mixed_number_division_l1510_151081

theorem mixed_number_division :
  (5 + 1 / 2) / (2 / 11) = 121 / 4 :=
by sorry

end mixed_number_division_l1510_151081


namespace find_m_l1510_151057

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 36 m = 108) (h2 : Nat.lcm 45 m = 180) : m = 72 := 
by 
  sorry

end find_m_l1510_151057


namespace simplify_expression_l1510_151023

theorem simplify_expression :
  (2 + 1 / 2) / (1 - 3 / 4) = 10 :=
by
  sorry

end simplify_expression_l1510_151023


namespace count_angles_l1510_151091

open Real

noncomputable def isGeometricSequence (a b c : ℝ) : Prop :=
(a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (a / b = b / c ∨ b / a = a / c ∨ c / a = a / b)

theorem count_angles (h1 : ∀ θ : ℝ, 0 < θ ∧ θ < 2 * π → (sin θ * cos θ = tan θ) ∨ (sin θ ^ 3 = cos θ ^ 2)) :
  ∃ n : ℕ, 
    (∀ θ : ℝ, 0 < θ ∧ θ < 2 * π ∧ (θ % (π/2) ≠ 0) → isGeometricSequence (sin θ) (cos θ) (tan θ) ) → 
    n = 6 := 
sorry

end count_angles_l1510_151091


namespace gcd_expression_multiple_of_456_l1510_151026

theorem gcd_expression_multiple_of_456 (a : ℤ) (h : ∃ k : ℤ, a = 456 * k) : 
  Int.gcd (3 * a^3 + a^2 + 4 * a + 57) a = 57 := by
  sorry

end gcd_expression_multiple_of_456_l1510_151026


namespace brick_height_l1510_151072

theorem brick_height (h : ℝ) : 
    let wall_length := 900
    let wall_width := 600
    let wall_height := 22.5
    let num_bricks := 7200
    let brick_length := 25
    let brick_width := 11.25
    wall_length * wall_width * wall_height = num_bricks * (brick_length * brick_width * h) -> 
    h = 67.5 := 
by
  intros
  sorry

end brick_height_l1510_151072


namespace right_triangle_hypotenuse_product_square_l1510_151022

theorem right_triangle_hypotenuse_product_square (A₁ A₂ : ℝ) (a₁ b₁ a₂ b₂ : ℝ) 
(h₁ : a₁ * b₁ / 2 = A₁) (h₂ : a₂ * b₂ / 2 = A₂) 
(h₃ : A₁ = 2) (h₄ : A₂ = 3) 
(h₅ : a₁ = a₂) (h₆ : b₂ = 2 * b₁) : 
(a₁ ^ 2 + b₁ ^ 2) * (a₂ ^ 2 + b₂ ^ 2) = 325 := 
by sorry

end right_triangle_hypotenuse_product_square_l1510_151022
