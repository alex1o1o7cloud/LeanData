import Mathlib

namespace NUMINAMATH_GPT_least_n_probability_lt_1_over_10_l2250_225097

theorem least_n_probability_lt_1_over_10 : 
  ∃ (n : ℕ), (1 / 2 : ℝ) ^ n < 1 / 10 ∧ ∀ m < n, ¬ ((1 / 2 : ℝ) ^ m < 1 / 10) :=
by
  sorry

end NUMINAMATH_GPT_least_n_probability_lt_1_over_10_l2250_225097


namespace NUMINAMATH_GPT_car_trip_time_difference_l2250_225006

theorem car_trip_time_difference
  (average_speed : ℝ)
  (distance1 distance2 : ℝ)
  (speed_60_mph : average_speed = 60)
  (dist1_540 : distance1 = 540)
  (dist2_510 : distance2 = 510) :
  ((distance1 - distance2) / average_speed) * 60 = 30 := by
  sorry

end NUMINAMATH_GPT_car_trip_time_difference_l2250_225006


namespace NUMINAMATH_GPT_sum_inverses_of_roots_l2250_225036

open Polynomial

theorem sum_inverses_of_roots (a b c : ℝ) (h1 : a^3 - 2020 * a + 1010 = 0)
    (h2 : b^3 - 2020 * b + 1010 = 0) (h3 : c^3 - 2020 * c + 1010 = 0) :
    (1/a) + (1/b) + (1/c) = 2 := 
  sorry

end NUMINAMATH_GPT_sum_inverses_of_roots_l2250_225036


namespace NUMINAMATH_GPT_substance_volume_proportional_l2250_225005

theorem substance_volume_proportional (k : ℝ) (V₁ V₂ : ℝ) (W₁ W₂ : ℝ) 
  (h1 : V₁ = k * W₁) 
  (h2 : V₂ = k * W₂) 
  (h3 : V₁ = 48) 
  (h4 : W₁ = 112) 
  (h5 : W₂ = 84) 
  : V₂ = 36 := 
  sorry

end NUMINAMATH_GPT_substance_volume_proportional_l2250_225005


namespace NUMINAMATH_GPT_village_population_l2250_225050

theorem village_population (x : ℝ) (h : 0.96 * x = 23040) : x = 24000 := sorry

end NUMINAMATH_GPT_village_population_l2250_225050


namespace NUMINAMATH_GPT_contrapositive_abc_l2250_225013

theorem contrapositive_abc (a b c : ℝ) : (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → (abc ≠ 0) := 
sorry

end NUMINAMATH_GPT_contrapositive_abc_l2250_225013


namespace NUMINAMATH_GPT_coefficient_x_squared_l2250_225067

theorem coefficient_x_squared (a : ℝ) (x : ℝ) (h : x = 0.5) (eqn : a * x^2 + 9 * x - 5 = 0) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_x_squared_l2250_225067


namespace NUMINAMATH_GPT_gain_percentage_l2250_225083

theorem gain_percentage (C1 C2 SP1 SP2 : ℝ) (h1 : C1 + C2 = 540) (h2 : C1 = 315)
    (h3 : SP1 = C1 - (0.15 * C1)) (h4 : SP1 = SP2) :
    ((SP2 - C2) / C2) * 100 = 19 :=
by
  sorry

end NUMINAMATH_GPT_gain_percentage_l2250_225083


namespace NUMINAMATH_GPT_emerie_dimes_count_l2250_225063

variables (zain_coins emerie_coins num_quarters num_nickels : ℕ)
variable (emerie_dimes : ℕ)

-- Conditions as per part a)
axiom zain_has_more_coins : ∀ (e z : ℕ), z = e + 10
axiom total_zain_coins : zain_coins = 48
axiom emerie_coins_from_quarters_and_nickels : num_quarters = 6 ∧ num_nickels = 5
axiom emerie_known_coins : ∀ q n : ℕ, emerie_coins = q + n + emerie_dimes

-- The statement to prove
theorem emerie_dimes_count : emerie_coins = 38 → emerie_dimes = 27 := 
by 
  sorry

end NUMINAMATH_GPT_emerie_dimes_count_l2250_225063


namespace NUMINAMATH_GPT_acme_vs_beta_l2250_225066

theorem acme_vs_beta (x : ℕ) :
  (80 + 10 * x < 20 + 15 * x) → (13 ≤ x) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_acme_vs_beta_l2250_225066


namespace NUMINAMATH_GPT_number_of_truthful_monkeys_l2250_225075

-- Define the conditions of the problem
def num_tigers : ℕ := 100
def num_foxes : ℕ := 100
def num_monkeys : ℕ := 100
def total_groups : ℕ := 100
def animals_per_group : ℕ := 3
def yes_tiger : ℕ := 138
def yes_fox : ℕ := 188

-- Problem statement to be proved
theorem number_of_truthful_monkeys :
  ∃ m : ℕ, m = 76 ∧
  ∃ (x y z m n : ℕ),
    -- The number of monkeys mixed with tigers
    x + 2 * (74 - y) = num_monkeys ∧

    -- Given constraints
    m ∈ {n : ℕ | n ≤ x} ∧
    n ∈ {n : ℕ | n ≤ (num_foxes - x)} ∧

    -- Equation setup and derived equations
    (x - m) + (num_foxes - y) + n = yes_tiger ∧
    m + (num_tigers - x - n) + (num_tigers - z) = yes_fox ∧
    y + z = 74 ∧
    
    -- ensuring the groups are valid
    2 * (74 - y) = z :=

sorry

end NUMINAMATH_GPT_number_of_truthful_monkeys_l2250_225075


namespace NUMINAMATH_GPT_subset_relation_l2250_225059

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2*x + 2}
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.log (x - 4) / Real.log 2}

-- State the proof problem
theorem subset_relation : N ⊆ M := 
sorry

end NUMINAMATH_GPT_subset_relation_l2250_225059


namespace NUMINAMATH_GPT_evaluate_expression_l2250_225069

theorem evaluate_expression :
  ((Int.ceil ((21 : ℚ) / 5 - Int.ceil ((35 : ℚ) / 23))) : ℚ) /
  (Int.ceil ((35 : ℚ) / 5 + Int.ceil ((5 * 23 : ℚ) / 35))) = 3 / 11 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2250_225069


namespace NUMINAMATH_GPT_incenter_coordinates_l2250_225007

-- Define lengths of the sides of the triangle
def a : ℕ := 8
def b : ℕ := 10
def c : ℕ := 6

-- Define the incenter formula components
def sum_of_sides : ℕ := a + b + c
def x : ℚ := a / (sum_of_sides : ℚ)
def y : ℚ := b / (sum_of_sides : ℚ)
def z : ℚ := c / (sum_of_sides : ℚ)

-- Prove the result
theorem incenter_coordinates :
  (x, y, z) = (1 / 3, 5 / 12, 1 / 4) :=
by 
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_incenter_coordinates_l2250_225007


namespace NUMINAMATH_GPT_union_A_B_l2250_225076

def A : Set ℝ := {x | |x| < 3}
def B : Set ℝ := {x | 2 - x > 0}

theorem union_A_B (x : ℝ) : (x ∈ A ∨ x ∈ B) ↔ x < 3 := by
  sorry

end NUMINAMATH_GPT_union_A_B_l2250_225076


namespace NUMINAMATH_GPT_age_difference_l2250_225029

variable (A B C : ℕ)

def age_relationship (B C : ℕ) : Prop :=
  B = 2 * C

def total_ages (A B C : ℕ) : Prop :=
  A + B + C = 72

theorem age_difference (B : ℕ) (hB : B = 28) (h1 : age_relationship B C) (h2 : total_ages A B C) :
  A - B = 2 :=
sorry

end NUMINAMATH_GPT_age_difference_l2250_225029


namespace NUMINAMATH_GPT_no_real_solution_l2250_225014

-- Define the equation
def equation (a b : ℝ) : Prop := a^2 + 3 * b^2 + 2 = 3 * a * b

-- Prove that there do not exist real numbers a and b such that equation a b holds
theorem no_real_solution : ¬ ∃ a b : ℝ, equation a b :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_no_real_solution_l2250_225014


namespace NUMINAMATH_GPT_only_setA_forms_triangle_l2250_225056

-- Define the sets of line segments
def setA := [3, 5, 7]
def setB := [3, 6, 10]
def setC := [5, 5, 11]
def setD := [5, 6, 11]

-- Define a function to check the triangle inequality
def satisfies_triangle_inequality (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Formalize the question
theorem only_setA_forms_triangle :
  satisfies_triangle_inequality 3 5 7 ∧
  ¬(satisfies_triangle_inequality 3 6 10) ∧
  ¬(satisfies_triangle_inequality 5 5 11) ∧
  ¬(satisfies_triangle_inequality 5 6 11) :=
by
  sorry

end NUMINAMATH_GPT_only_setA_forms_triangle_l2250_225056


namespace NUMINAMATH_GPT_choose_positions_from_8_people_l2250_225030

theorem choose_positions_from_8_people : 
  ∃ (ways : ℕ), ways = 8 * 7 * 6 := 
sorry

end NUMINAMATH_GPT_choose_positions_from_8_people_l2250_225030


namespace NUMINAMATH_GPT_total_amount_paid_l2250_225098

-- Definitions based on conditions
def original_price : ℝ := 100
def discount_rate : ℝ := 0.20
def additional_discount : ℝ := 5
def sales_tax_rate : ℝ := 0.08

-- Theorem statement
theorem total_amount_paid :
  let discounted_price := original_price * (1 - discount_rate)
  let final_price := discounted_price - additional_discount
  let total_price_with_tax := final_price * (1 + sales_tax_rate)
  total_price_with_tax = 81 := sorry

end NUMINAMATH_GPT_total_amount_paid_l2250_225098


namespace NUMINAMATH_GPT_cost_per_person_is_correct_l2250_225012

-- Define the given conditions
def fee_per_30_minutes : ℕ := 4000
def bikes : ℕ := 4
def hours : ℕ := 3
def people : ℕ := 6

-- Calculate the correct answer based on the given conditions
noncomputable def cost_per_person : ℕ :=
  let fee_per_hour := 2 * fee_per_30_minutes
  let fee_per_3_hours := hours * fee_per_hour
  let total_cost := bikes * fee_per_3_hours
  total_cost / people

-- The theorem to be proved
theorem cost_per_person_is_correct : cost_per_person = 16000 := sorry

end NUMINAMATH_GPT_cost_per_person_is_correct_l2250_225012


namespace NUMINAMATH_GPT_opposite_of_neg_six_is_six_l2250_225088

theorem opposite_of_neg_six_is_six : 
  ∃ (x : ℝ), (-6 + x = 0) ∧ x = 6 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_six_is_six_l2250_225088


namespace NUMINAMATH_GPT_problem_solution_l2250_225019

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {1, 2}

-- Define set B
def B : Set ℕ := {2}

-- Define the complement function specific to our universal set U
def complement (U A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- Lean theorem to prove the given problem's correctness
theorem problem_solution : complement U (A ∪ B) = {3, 4} :=
by
  sorry -- Proof is omitted as per the instructions

end NUMINAMATH_GPT_problem_solution_l2250_225019


namespace NUMINAMATH_GPT_solve_system_l2250_225010

theorem solve_system (x y z : ℝ) (h1 : (x + 1) * y * z = 12) 
                               (h2 : (y + 1) * z * x = 4) 
                               (h3 : (z + 1) * x * y = 4) : 
  (x = 1 / 3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2) :=
sorry

end NUMINAMATH_GPT_solve_system_l2250_225010


namespace NUMINAMATH_GPT_sum_of_fractions_l2250_225025

theorem sum_of_fractions : (1 / 1) + (2 / 2) + (3 / 3) = 3 := 
by 
  norm_num

end NUMINAMATH_GPT_sum_of_fractions_l2250_225025


namespace NUMINAMATH_GPT_coin_combinations_l2250_225096

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end NUMINAMATH_GPT_coin_combinations_l2250_225096


namespace NUMINAMATH_GPT_problem_statement_l2250_225087

theorem problem_statement :
  ∃ (n : ℕ), n = 101 ∧
  (∀ (x : ℕ), x < 4032 → ((x^2 - 20) % 16 = 0) ∧ ((x^2 - 16) % 20 = 0) ↔ (∃ k1 k2 : ℕ, (x = 80 * k1 + 6 ∨ x = 80 * k2 + 74) ∧ k1 + k2 + 1 = n)) :=
by sorry

end NUMINAMATH_GPT_problem_statement_l2250_225087


namespace NUMINAMATH_GPT_cube_sum_equals_36_l2250_225082

variable {a b c k : ℝ}

theorem cube_sum_equals_36 (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
    (heq : (a^3 - 12) / a = (b^3 - 12) / b)
    (heq_another : (b^3 - 12) / b = (c^3 - 12) / c) :
    a^3 + b^3 + c^3 = 36 := by
  sorry

end NUMINAMATH_GPT_cube_sum_equals_36_l2250_225082


namespace NUMINAMATH_GPT_ratio_of_logs_eq_golden_ratio_l2250_225073

theorem ratio_of_logs_eq_golden_ratio
  (r s : ℝ) (hr : 0 < r) (hs : 0 < s)
  (h : Real.log r / Real.log 4 = Real.log s / Real.log 18 ∧ Real.log s / Real.log 18 = Real.log (r + s) / Real.log 24) :
  s / r = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_logs_eq_golden_ratio_l2250_225073


namespace NUMINAMATH_GPT_sin_theta_of_triangle_area_side_median_l2250_225089

-- Defining the problem statement and required conditions
theorem sin_theta_of_triangle_area_side_median (A : ℝ) (a m : ℝ) (θ : ℝ) 
  (hA : A = 30)
  (ha : a = 12)
  (hm : m = 8)
  (hTriangleArea : A = 1/2 * a * m * Real.sin θ) :
  Real.sin θ = 5 / 8 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sin_theta_of_triangle_area_side_median_l2250_225089


namespace NUMINAMATH_GPT_star_value_l2250_225017

-- Define the operation &
def and_operation (a b : ℕ) : ℕ := (a + b) * (a - b)

-- Define the operation star
def star_operation (c d : ℕ) : ℕ := and_operation c d + 2 * (c + d)

-- The proof problem
theorem star_value : star_operation 8 4 = 72 :=
by
  sorry

end NUMINAMATH_GPT_star_value_l2250_225017


namespace NUMINAMATH_GPT_frequency_of_group_samples_l2250_225058

-- Conditions
def sample_capacity : ℕ := 32
def group_frequency : ℝ := 0.125

-- Theorem statement
theorem frequency_of_group_samples : group_frequency * sample_capacity = 4 :=
by sorry

end NUMINAMATH_GPT_frequency_of_group_samples_l2250_225058


namespace NUMINAMATH_GPT_smallest_prime_divisor_and_cube_root_l2250_225091

theorem smallest_prime_divisor_and_cube_root (N : ℕ) (p : ℕ) (q : ℕ)
  (hN_composite : N > 1 ∧ ¬ (∃ p : ℕ, p > 1 ∧ p < N ∧ N = p))
  (h_divisor : N = p * q)
  (h_p_prime : Nat.Prime p)
  (h_min_prime : ∀ (d : ℕ), Nat.Prime d → d ∣ N → p ≤ d)
  (h_cube_root : p > Nat.sqrt (Nat.sqrt N)) :
  Nat.Prime q := 
sorry

end NUMINAMATH_GPT_smallest_prime_divisor_and_cube_root_l2250_225091


namespace NUMINAMATH_GPT_find_constants_l2250_225061

noncomputable def csc (x : ℝ) : ℝ := 1 / (Real.sin x)

theorem find_constants (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_min_value : ∃ x : ℝ, a * csc (b * x + c) = 3)
  (h_period : ∀ x, a * csc (b * (x + 4 * Real.pi) + c) = a * csc (b * x + c)) :
  a = 3 ∧ b = (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l2250_225061


namespace NUMINAMATH_GPT_marshmallow_challenge_l2250_225000

noncomputable def haley := 8
noncomputable def michael := 3 * haley
noncomputable def brandon := (1 / 2) * michael
noncomputable def sofia := 2 * (haley + brandon)
noncomputable def total := haley + michael + brandon + sofia

theorem marshmallow_challenge : total = 84 :=
by
  sorry

end NUMINAMATH_GPT_marshmallow_challenge_l2250_225000


namespace NUMINAMATH_GPT_smallest_x_with_18_factors_and_factors_18_21_exists_l2250_225028

def has_18_factors (x : ℕ) : Prop :=
(x.factors.length == 18)

def is_factor (a b : ℕ) : Prop :=
(b % a == 0)

theorem smallest_x_with_18_factors_and_factors_18_21_exists :
  ∃ x : ℕ, has_18_factors x ∧ is_factor 18 x ∧ is_factor 21 x ∧ ∀ y : ℕ, has_18_factors y ∧ is_factor 18 y ∧ is_factor 21 y → y ≥ x :=
sorry

end NUMINAMATH_GPT_smallest_x_with_18_factors_and_factors_18_21_exists_l2250_225028


namespace NUMINAMATH_GPT_probability_of_5_odd_in_6_rolls_l2250_225048

open Classical

noncomputable def prob_odd_in_six_rolls : ℚ :=
  let num_rolls := 6
  let prob_odd_single := 1 / 2
  let binom_coeff := Nat.choose num_rolls 5
  let total_outcomes := (2 : ℕ) ^ num_rolls
  binom_coeff * ((prob_odd_single ^ 5) * ((1 - prob_odd_single) ^ (num_rolls - 5))) / total_outcomes

theorem probability_of_5_odd_in_6_rolls :
  prob_odd_in_six_rolls = 3 / 32 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_5_odd_in_6_rolls_l2250_225048


namespace NUMINAMATH_GPT_expression_equality_l2250_225078

theorem expression_equality : 
  (∀ (x : ℝ) (a k n : ℝ), (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n → a = 6 ∧ k = -5 ∧ n = -6) → 
  ∀ (a k n : ℝ), a = 6 → k = -5 → n = -6 → a - n + k = 7 :=
by
  intro h
  intros a k n ha hk hn
  rw [ha, hk, hn]
  norm_num

end NUMINAMATH_GPT_expression_equality_l2250_225078


namespace NUMINAMATH_GPT_smartphone_cost_decrease_l2250_225094

theorem smartphone_cost_decrease :
  ∀ (cost2010 cost2020 : ℝ),
  cost2010 = 600 →
  cost2020 = 450 →
  ((cost2010 - cost2020) / cost2010) * 100 = 25 :=
by
  intros cost2010 cost2020 h1 h2
  sorry

end NUMINAMATH_GPT_smartphone_cost_decrease_l2250_225094


namespace NUMINAMATH_GPT_quadratic_maximum_or_minimum_l2250_225008

open Real

noncomputable def quadratic_function (a b x : ℝ) : ℝ := a * x^2 + b * x - b^2 / (3 * a)

theorem quadratic_maximum_or_minimum (a b : ℝ) (h : a ≠ 0) :
  (a > 0 → ∃ x₀, ∀ x, quadratic_function a b x₀ ≤ quadratic_function a b x) ∧
  (a < 0 → ∃ x₀, ∀ x, quadratic_function a b x₀ ≥ quadratic_function a b x) :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_quadratic_maximum_or_minimum_l2250_225008


namespace NUMINAMATH_GPT_AB_vector_eq_l2250_225027

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Given conditions
variables (A B C D : V)
variables (a b : V)
variable (ABCD_parallelogram : is_parallelogram A B C D)

-- Definition of the diagonals
def AC_vector : V := C - A
def BD_vector : V := D - B

-- The given condition that diagonals AC and BD are equal to a and b respectively
axiom AC_eq_a : AC_vector A C = a
axiom BD_eq_b : BD_vector B D = b

-- Proof problem
theorem AB_vector_eq : (B - A) = (1/2) • (a - b) :=
sorry

end NUMINAMATH_GPT_AB_vector_eq_l2250_225027


namespace NUMINAMATH_GPT_find_b_age_l2250_225080

variable (a b c : ℕ)
-- Condition 1: a is two years older than b
variable (h1 : a = b + 2)
-- Condition 2: b is twice as old as c
variable (h2 : b = 2 * c)
-- Condition 3: The total of the ages of a, b, and c is 17
variable (h3 : a + b + c = 17)

theorem find_b_age (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 17) : b = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_b_age_l2250_225080


namespace NUMINAMATH_GPT_pure_imaginary_x_l2250_225060

theorem pure_imaginary_x (x : ℝ) (h: (x - 2008) = 0) : x = 2008 :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_x_l2250_225060


namespace NUMINAMATH_GPT_minimum_value_of_expression_l2250_225018

theorem minimum_value_of_expression (x A B C : ℝ) (hx : x > 0) 
  (hA : A = x^2 + 1/x^2) (hB : B = x - 1/x) (hC : C = B * (A + 1)) : 
  ∃ m : ℝ, m = 6.4 ∧ m = A^3 / C :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_of_expression_l2250_225018


namespace NUMINAMATH_GPT_product_of_four_consecutive_integers_divisible_by_24_l2250_225093

theorem product_of_four_consecutive_integers_divisible_by_24 (n : ℤ) :
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end NUMINAMATH_GPT_product_of_four_consecutive_integers_divisible_by_24_l2250_225093


namespace NUMINAMATH_GPT_man_l2250_225026

theorem man's_age_twice_son's_age_in_2_years
  (S : ℕ) (M : ℕ) (Y : ℕ)
  (h1 : M = S + 24)
  (h2 : S = 22)
  (h3 : M + Y = 2 * (S + Y)) :
  Y = 2 := by
  sorry

end NUMINAMATH_GPT_man_l2250_225026


namespace NUMINAMATH_GPT_center_of_circle_l2250_225034

theorem center_of_circle (
  center : ℝ × ℝ
) :
  (∀ (p : ℝ × ℝ), (p.1 * 3 + p.2 * 4 = 24) ∨ (p.1 * 3 + p.2 * 4 = -6) → (dist center p = dist center p)) ∧
  (center.1 * 3 - center.2 = 0)
  → center = (3 / 5, 9 / 5) :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_l2250_225034


namespace NUMINAMATH_GPT_find_incorrect_statement_l2250_225024

def statement_A := ∀ (P Q : Prop), (P → Q) → (¬Q → ¬P)
def statement_B := ∀ (P : Prop), ((¬P) → false) → P
def statement_C := ∀ (shape : Type), (∃ s : shape, true) → false
def statement_D := ∀ (P : ℕ → Prop), P 0 → (∀ n, P n → P (n + 1)) → ∀ n, P n
def statement_E := ∀ {α : Type} (p : Prop), (¬p ∨ p)

theorem find_incorrect_statement : statement_C :=
sorry

end NUMINAMATH_GPT_find_incorrect_statement_l2250_225024


namespace NUMINAMATH_GPT_minimum_value_of_xy_l2250_225016

theorem minimum_value_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y + 6 = x * y) : 
  18 ≤ x * y :=
sorry

end NUMINAMATH_GPT_minimum_value_of_xy_l2250_225016


namespace NUMINAMATH_GPT_cubic_root_form_addition_l2250_225068

theorem cubic_root_form_addition (p q r : ℕ) 
(h_root_form : ∃ x : ℝ, 2 * x^3 + 3 * x^2 - 5 * x - 2 = 0 ∧ x = (p^(1/3) + q^(1/3) + 2) / r) : 
  p + q + r = 10 :=
sorry

end NUMINAMATH_GPT_cubic_root_form_addition_l2250_225068


namespace NUMINAMATH_GPT_solve_system_of_equations_l2250_225053

theorem solve_system_of_equations :
  {p : ℝ × ℝ | 
    (p.1^2 + p.2 + 1) * (p.2^2 + p.1 + 1) = 4 ∧
    (p.1^2 + p.2)^2 + (p.2^2 + p.1)^2 = 2} =
  {(0, 1), (1, 0), 
   ( (-1 + Real.sqrt 5) / 2, (-1 + Real.sqrt 5) / 2),
   ( (-1 - Real.sqrt 5) / 2, (-1 - Real.sqrt 5) / 2) } :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2250_225053


namespace NUMINAMATH_GPT_price_decrease_required_to_initial_l2250_225071

theorem price_decrease_required_to_initial :
  let P0 := 100.0
  let P1 := P0 * 1.15
  let P2 := P1 * 0.90
  let P3 := P2 * 1.20
  let P4 := P3 * 0.70
  let P5 := P4 * 1.10
  let P6 := P5 * (1.0 - d / 100.0)
  P6 = P0 -> d = 5.0 :=
by
  sorry

end NUMINAMATH_GPT_price_decrease_required_to_initial_l2250_225071


namespace NUMINAMATH_GPT_find_x_l2250_225003

theorem find_x (x : ℕ) :
  (3 * x > 91 ∧ x < 120 ∧ x < 27 ∧ ¬(4 * x > 37) ∧ ¬(2 * x ≥ 21) ∧ ¬(x > 7)) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ x < 27 ∧ 4 * x > 37 ∧ ¬(2 * x ≥ 21) ∧ ¬(x > 7)) ∨
  (¬(3 * x > 91) ∧ ¬(x < 120) ∧ x < 27 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ ¬(x < 27) ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ x < 27 ∧ ¬(4 * x > 37) ∧ 2 * x ≥ 21 ∧ x > 7) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ x < 27 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ ¬(x > 7)) →
  x = 9 :=
sorry

end NUMINAMATH_GPT_find_x_l2250_225003


namespace NUMINAMATH_GPT_unique_k_n_m_solution_l2250_225046

-- Problem statement
theorem unique_k_n_m_solution :
  ∃ (k : ℕ) (n : ℕ) (m : ℕ), k = 1 ∧ n = 2 ∧ m = 3 ∧ 3^k + 5^k = n^m ∧
  ∀ (k₀ : ℕ) (n₀ : ℕ) (m₀ : ℕ), (3^k₀ + 5^k₀ = n₀^m₀ ∧ m₀ ≥ 2) → (k₀ = 1 ∧ n₀ = 2 ∧ m₀ = 3) :=
by
  sorry

end NUMINAMATH_GPT_unique_k_n_m_solution_l2250_225046


namespace NUMINAMATH_GPT_lateral_surface_area_of_pyramid_inscribed_in_sphere_l2250_225002
-- Importing the entire Mathlib library to ensure all necessary definitions and theorems are available.

-- Formulate the problem as a Lean statement.

theorem lateral_surface_area_of_pyramid_inscribed_in_sphere :
  let R := (1 : ℝ)
  let theta := (45 : ℝ) * Real.pi / 180 -- Convert degrees to radians.
  -- Assuming the pyramid is regular and quadrilateral, inscribed in a sphere of radius 1
  ∃ S : ℝ, S = 4 :=
  sorry

end NUMINAMATH_GPT_lateral_surface_area_of_pyramid_inscribed_in_sphere_l2250_225002


namespace NUMINAMATH_GPT_five_less_than_sixty_percent_of_cats_l2250_225052

theorem five_less_than_sixty_percent_of_cats (hogs cats : ℕ) 
  (hogs_eq : hogs = 3 * cats)
  (hogs_value : hogs = 75) : 
  5 < 60 * cats / 100 :=
by {
  sorry
}

end NUMINAMATH_GPT_five_less_than_sixty_percent_of_cats_l2250_225052


namespace NUMINAMATH_GPT_binomial_square_l2250_225043

theorem binomial_square (p : ℝ) : (∃ b : ℝ, (3 * x + b)^2 = 9 * x^2 + 24 * x + p) → p = 16 := by
  sorry

end NUMINAMATH_GPT_binomial_square_l2250_225043


namespace NUMINAMATH_GPT_taxi_fare_l2250_225077

theorem taxi_fare (x : ℝ) (h : 3.00 + 0.25 * ((x - 0.75) / 0.1) = 12) : x = 4.35 :=
  sorry

end NUMINAMATH_GPT_taxi_fare_l2250_225077


namespace NUMINAMATH_GPT_second_train_length_l2250_225062

theorem second_train_length
  (train1_length : ℝ)
  (train1_speed_kmph : ℝ)
  (train2_speed_kmph : ℝ)
  (time_to_clear : ℝ)
  (h1 : train1_length = 135)
  (h2 : train1_speed_kmph = 80)
  (h3 : train2_speed_kmph = 65)
  (h4 : time_to_clear = 7.447680047665153) :
  ∃ l2 : ℝ, l2 = 165 :=
by
  let train1_speed_mps := train1_speed_kmph * 1000 / 3600
  let train2_speed_mps := train2_speed_kmph * 1000 / 3600
  let total_distance := (train1_speed_mps + train2_speed_mps) * time_to_clear
  have : total_distance = 300 := by sorry
  have l2 := total_distance - train1_length
  use l2
  have : l2 = 165 := by sorry
  assumption

end NUMINAMATH_GPT_second_train_length_l2250_225062


namespace NUMINAMATH_GPT_calculate_expression_l2250_225041

variable (x y : ℝ)

theorem calculate_expression : (-2 * x^2 * y) ^ 2 = 4 * x^4 * y^2 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2250_225041


namespace NUMINAMATH_GPT_mark_hourly_wage_before_raise_40_l2250_225084

-- Mark's hourly wage before the raise
def hourly_wage_before_raise (x : ℝ) : Prop :=
  let weekly_hours := 40
  let raise_percentage := 0.05
  let new_hourly_wage := x * (1 + raise_percentage)
  let new_weekly_earnings := weekly_hours * new_hourly_wage
  let old_bills := 600
  let personal_trainer := 100
  let new_expenses := old_bills + personal_trainer
  let leftover_income := 980
  new_weekly_earnings = new_expenses + leftover_income

-- Proving that Mark's hourly wage before the raise was 40 dollars
theorem mark_hourly_wage_before_raise_40 : hourly_wage_before_raise 40 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_mark_hourly_wage_before_raise_40_l2250_225084


namespace NUMINAMATH_GPT_simplify_polynomial_l2250_225033

theorem simplify_polynomial (x : ℝ) :
  3 + 5 * x - 7 * x^2 - 9 + 11 * x - 13 * x^2 + 15 - 17 * x + 19 * x^2 = 9 - x - x^2 := 
  by {
  -- placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_simplify_polynomial_l2250_225033


namespace NUMINAMATH_GPT_arithmetic_sequence_expression_l2250_225039

variable (a : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, (n < m) → (a (m + 1) - a m = a (n + 1) - a n)

theorem arithmetic_sequence_expression
  (h_arith_seq : is_arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = -3) :
  ∀ n : ℕ, a n = -2 * n + 3 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_expression_l2250_225039


namespace NUMINAMATH_GPT_range_a_l2250_225023

theorem range_a (x a : ℝ) (h1 : x^2 - 8 * x - 33 > 0) (h2 : |x - 1| > a) (h3 : a > 0) :
  0 < a ∧ a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_range_a_l2250_225023


namespace NUMINAMATH_GPT_tank_fill_fraction_l2250_225090

theorem tank_fill_fraction (a b c : ℝ) (h1 : a=9) (h2 : b=54) (h3 : c=3/4) : (c * b + a) / b = 23 / 25 := 
by 
  sorry

end NUMINAMATH_GPT_tank_fill_fraction_l2250_225090


namespace NUMINAMATH_GPT_cubic_polynomial_sum_l2250_225035

-- Define the roots and their properties according to Vieta's formulas
variables {p q r : ℝ}
axiom root_poly : p * q * r = -1
axiom pq_sum : p * q + p * r + q * r = -3
axiom roots_sum : p + q + r = 0

-- Define the target equality to prove
theorem cubic_polynomial_sum :
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_cubic_polynomial_sum_l2250_225035


namespace NUMINAMATH_GPT_squareable_numbers_l2250_225051

def is_squareable (n : ℕ) : Prop :=
  ∃ (perm : ℕ → ℕ), (∀ i, 1 ≤ perm i ∧ perm i ≤ n) ∧ (∀ i, ∃ k, perm i + i = k * k)

theorem squareable_numbers : is_squareable 9 ∧ is_squareable 15 ∧ ¬ is_squareable 7 ∧ ¬ is_squareable 11 :=
by sorry

end NUMINAMATH_GPT_squareable_numbers_l2250_225051


namespace NUMINAMATH_GPT_hyperbola_equation_l2250_225001

theorem hyperbola_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
  (eccentricity : Real.sqrt 2 = b / a)
  (line_through_FP_parallel_to_asymptote : ∃ c : ℝ, c = Real.sqrt 2 * a ∧ ∀ P : ℝ × ℝ, P = (0, 4) → (P.2 - 0) / (P.1 + c) = 1) :
  (∃ (a b : ℝ), a = b ∧ (a = 2 * Real.sqrt 2 ∧ b = 2 * Real.sqrt 2)) ∧
  (a = 2 * Real.sqrt 2 ∧ b = 2 * Real.sqrt 2) → 
  (∃ x y : ℝ, ((x^2 / 8) - (y^2 / 8) = 1)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l2250_225001


namespace NUMINAMATH_GPT_probability_zhang_watches_entire_news_l2250_225072

noncomputable def broadcast_time_start := 12 * 60 -- 12:00 in minutes
noncomputable def broadcast_time_end := 12 * 60 + 30 -- 12:30 in minutes
noncomputable def news_report_duration := 5 -- 5 minutes
noncomputable def zhang_on_tv_time := 12 * 60 + 20 -- 12:20 in minutes
noncomputable def favorable_time_start := zhang_on_tv_time
noncomputable def favorable_time_end := zhang_on_tv_time + news_report_duration -- 12:20 to 12:25

theorem probability_zhang_watches_entire_news : 
  let total_broadcast_time := broadcast_time_end - broadcast_time_start
  let favorable_time_span := favorable_time_end - favorable_time_start
  favorable_time_span / total_broadcast_time = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_probability_zhang_watches_entire_news_l2250_225072


namespace NUMINAMATH_GPT_average_nat_series_l2250_225015

theorem average_nat_series : 
  let a := 12  -- first term
  let l := 53  -- last term
  let n := (l - a) / 1 + 1  -- number of terms
  let sum := n / 2 * (a + l)  -- sum of the arithmetic series
  let average := sum / n  -- average of the series
  average = 32.5 :=
by
  let a := 12
  let l := 53
  let n := (l - a) / 1 + 1
  let sum := n / 2 * (a + l)
  let average := sum / n
  sorry

end NUMINAMATH_GPT_average_nat_series_l2250_225015


namespace NUMINAMATH_GPT_logarithm_base_l2250_225065

theorem logarithm_base (x : ℝ) (b : ℝ) : (9 : ℝ)^(x + 5) = (16 : ℝ)^x → b = 16 / 9 → x = Real.log 9^5 / Real.log b := by sorry

end NUMINAMATH_GPT_logarithm_base_l2250_225065


namespace NUMINAMATH_GPT_rug_area_is_180_l2250_225092

variables (w l : ℕ)

def length_eq_width_plus_eight (l w : ℕ) : Prop :=
  l = w + 8

def uniform_width_between_rug_and_room (d : ℕ) : Prop :=
  d = 8

def area_uncovered_by_rug (area : ℕ) : Prop :=
  area = 704

def area_of_rug (w l : ℕ) : ℕ :=
  l * w

theorem rug_area_is_180 (w l : ℕ) (hwld : length_eq_width_plus_eight l w)
  (huw : uniform_width_between_rug_and_room 8)
  (huar : area_uncovered_by_rug 704) :
  area_of_rug w l = 180 :=
sorry

end NUMINAMATH_GPT_rug_area_is_180_l2250_225092


namespace NUMINAMATH_GPT_total_seashells_l2250_225011

-- Definitions of the initial number of seashells and the number found
def initial_seashells : Nat := 19
def found_seashells : Nat := 6

-- Theorem stating the total number of seashells in the collection
theorem total_seashells : initial_seashells + found_seashells = 25 := by
  sorry

end NUMINAMATH_GPT_total_seashells_l2250_225011


namespace NUMINAMATH_GPT_duration_of_period_l2250_225074

variable (t : ℝ)

theorem duration_of_period:
  (2800 * 0.185 * t - 2800 * 0.15 * t = 294) ↔ (t = 3) :=
by
  sorry

end NUMINAMATH_GPT_duration_of_period_l2250_225074


namespace NUMINAMATH_GPT_probability_three_red_balls_l2250_225047

open scoped BigOperators

noncomputable def hypergeometric_prob (r : ℕ) (b : ℕ) (k : ℕ) (d : ℕ) : ℝ :=
  (Nat.choose r d * Nat.choose b (k - d) : ℝ) / Nat.choose (r + b) k

theorem probability_three_red_balls :
  hypergeometric_prob 10 5 5 3 = 1200 / 3003 :=
by sorry

end NUMINAMATH_GPT_probability_three_red_balls_l2250_225047


namespace NUMINAMATH_GPT_smallest_B_l2250_225004

-- Definitions and conditions
def known_digit_sum : Nat := 4 + 8 + 3 + 9 + 4 + 2
def divisible_by_3 (n : Nat) : Bool := n % 3 = 0

-- Statement to prove
theorem smallest_B (B : Nat) (h : B < 10) (hdiv : divisible_by_3 (B + known_digit_sum)) : B = 0 :=
sorry

end NUMINAMATH_GPT_smallest_B_l2250_225004


namespace NUMINAMATH_GPT_numberOfWaysToChooseLeadership_is_correct_l2250_225040

noncomputable def numberOfWaysToChooseLeadership (totalMembers : ℕ) : ℕ :=
  let choicesForGovernor := totalMembers
  let remainingAfterGovernor := totalMembers - 1

  let choicesForDeputies := Nat.choose remainingAfterGovernor 3
  let remainingAfterDeputies := remainingAfterGovernor - 3

  let choicesForLieutenants1 := Nat.choose remainingAfterDeputies 3
  let remainingAfterLieutenants1 := remainingAfterDeputies - 3

  let choicesForLieutenants2 := Nat.choose remainingAfterLieutenants1 3
  let remainingAfterLieutenants2 := remainingAfterLieutenants1 - 3

  let choicesForLieutenants3 := Nat.choose remainingAfterLieutenants2 3
  let remainingAfterLieutenants3 := remainingAfterLieutenants2 - 3

  let choicesForSubordinates : List ℕ := 
    (List.range 8).map (λ i => Nat.choose (remainingAfterLieutenants3 - 2*i) 2)

  choicesForGovernor 
  * choicesForDeputies 
  * choicesForLieutenants1 
  * choicesForLieutenants2 
  * choicesForLieutenants3 
  * List.prod choicesForSubordinates

theorem numberOfWaysToChooseLeadership_is_correct : 
  numberOfWaysToChooseLeadership 35 = 
    35 * Nat.choose 34 3 * Nat.choose 31 3 * Nat.choose 28 3 * Nat.choose 25 3 *
    Nat.choose 16 2 * Nat.choose 14 2 * Nat.choose 12 2 * Nat.choose 10 2 *
    Nat.choose 8 2 * Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2 :=
by
  sorry

end NUMINAMATH_GPT_numberOfWaysToChooseLeadership_is_correct_l2250_225040


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l2250_225021

theorem necessary_and_sufficient_condition (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  a + b > a * b ↔ (a = 1 ∨ b = 1) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l2250_225021


namespace NUMINAMATH_GPT_relationship_between_m_and_n_l2250_225086

variable (x : ℝ)

def m := x^2 + 2*x + 3
def n := 2

theorem relationship_between_m_and_n :
  m x ≥ n := by
  sorry

end NUMINAMATH_GPT_relationship_between_m_and_n_l2250_225086


namespace NUMINAMATH_GPT_ratio_of_inquisitive_tourist_l2250_225099

theorem ratio_of_inquisitive_tourist (questions_per_tourist : ℕ)
                                     (num_group1 : ℕ) (num_group2 : ℕ) (num_group3 : ℕ) (num_group4 : ℕ)
                                     (total_questions : ℕ) 
                                     (inquisitive_tourist_questions : ℕ) :
  questions_per_tourist = 2 ∧ 
  num_group1 = 6 ∧ 
  num_group2 = 11 ∧ 
  num_group3 = 8 ∧ 
  num_group4 = 7 ∧ 
  total_questions = 68 ∧ 
  inquisitive_tourist_questions = (total_questions - (num_group1 * questions_per_tourist + num_group2 * questions_per_tourist +
                                                        (num_group3 - 1) * questions_per_tourist + num_group4 * questions_per_tourist)) →
  (inquisitive_tourist_questions : ℕ) / questions_per_tourist = 3 :=
by sorry

end NUMINAMATH_GPT_ratio_of_inquisitive_tourist_l2250_225099


namespace NUMINAMATH_GPT_combination_20_6_l2250_225070

theorem combination_20_6 : Nat.choose 20 6 = 38760 :=
by
  sorry

end NUMINAMATH_GPT_combination_20_6_l2250_225070


namespace NUMINAMATH_GPT_prasanna_speed_l2250_225020

variable (L_speed P_speed time apart : ℝ)
variable (h1 : L_speed = 40)
variable (h2 : time = 1)
variable (h3 : apart = 78)

theorem prasanna_speed :
  P_speed = apart - (L_speed * time) / time := 
by
  rw [h1, h2, h3]
  simp
  sorry

end NUMINAMATH_GPT_prasanna_speed_l2250_225020


namespace NUMINAMATH_GPT_fifth_powers_sum_eq_l2250_225032

section PowerProof

variables (a b c d : ℝ)

-- Conditions:
def condition1 : a + b = c + d := sorry
def condition2 : a^3 + b^3 = c^3 + d^3 := sorry

-- Claim for fifth powers:
theorem fifth_powers_sum_eq : a + b = c + d → a^3 + b^3 = c^3 + d^3 → a^5 + b^5 = c^5 + d^5 := by
  intros h1 h2
  sorry

-- Clauses for disproving fourth powers under generality:
example : ¬ (∀ a b c d : ℝ, (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) :=
  by{
    sorry
  }

end PowerProof

end NUMINAMATH_GPT_fifth_powers_sum_eq_l2250_225032


namespace NUMINAMATH_GPT_find_beta_l2250_225044

open Real

theorem find_beta (α β : ℝ) (h1 : cos α = 1 / 7) (h2 : cos (α - β) = 13 / 14)
  (h3 : 0 < β) (h4 : β < α) (h5 : α < π / 2) : β = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_beta_l2250_225044


namespace NUMINAMATH_GPT_shop_width_correct_l2250_225031

-- Definition of the shop's monthly rent
def monthly_rent : ℝ := 2400

-- Definition of the shop's length in feet
def shop_length : ℝ := 10

-- Definition of the annual rent per square foot
def annual_rent_per_sq_ft : ℝ := 360

-- The mathematical assertion that the width of the shop is 8 feet
theorem shop_width_correct (width : ℝ) :
  (monthly_rent * 12) / annual_rent_per_sq_ft / shop_length = width :=
by
  sorry

end NUMINAMATH_GPT_shop_width_correct_l2250_225031


namespace NUMINAMATH_GPT_equivalent_single_percentage_increase_l2250_225085

noncomputable def calculate_final_price (p : ℝ) : ℝ :=
  let p1 := p * (1 + 0.15)
  let p2 := p1 * (1 + 0.20)
  let p_final := p2 * (1 - 0.10)
  p_final

theorem equivalent_single_percentage_increase (p : ℝ) : 
  calculate_final_price p = p * 1.242 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_single_percentage_increase_l2250_225085


namespace NUMINAMATH_GPT_part1_part2_l2250_225049

def proposition_p (m : ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → 2 * x - 4 ≥ m^2 - 5 * m

def proposition_q (m : ℝ) : Prop :=
  ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ x^2 - 2 * x + m - 1 ≤ 0

theorem part1 (m : ℝ) : proposition_p m → 1 ≤ m ∧ m ≤ 4 := 
sorry

theorem part2 (m : ℝ) : (proposition_p m ∨ proposition_q m) → m ≤ 4 := 
sorry

end NUMINAMATH_GPT_part1_part2_l2250_225049


namespace NUMINAMATH_GPT_arithmetic_sequence_n_sum_arithmetic_sequence_S17_arithmetic_sequence_S13_l2250_225045

-- Question 1
theorem arithmetic_sequence_n (a1 a4 a10 : ℤ) (d : ℤ) (n : ℤ) (Sn : ℤ) 
  (h1 : a1 + 3 * d = a4) 
  (h2 : a1 + 9 * d = a10)
  (h3 : Sn = n * (2 * a1 + (n - 1) * d) / 2)
  (h4 : a4 = 10)
  (h5 : a10 = -2)
  (h6 : Sn = 60)
  : n = 5 ∨ n = 6 := 
sorry

-- Question 2
theorem sum_arithmetic_sequence_S17 (a1 : ℤ) (d : ℤ) (a_n1 : ℤ → ℤ) (S17 : ℤ)
  (h1 : a1 = -7)
  (h2 : ∀ n, a_n1 (n + 1) = a_n1 n + d)
  (h3 : S17 = 17 * (2 * a1 + 16 * d) / 2)
  : S17 = 153 := 
sorry

-- Question 3
theorem arithmetic_sequence_S13 (a_2 a_7 a_12 : ℤ) (S13 : ℤ)
  (h1 : a_2 + a_7 + a_12 = 24)
  (h2 : S13 = a_7 * 13)
  : S13 = 104 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_sum_arithmetic_sequence_S17_arithmetic_sequence_S13_l2250_225045


namespace NUMINAMATH_GPT_part1_part2_part3_l2250_225042

universe u

def A : Set ℝ := {x | -3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12 * x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | x < a}
def CR_A : Set ℝ := {x | x < -3 ∨ x ≥ 7}

theorem part1 : A ∪ B = {x | -3 ≤ x ∧ x < 10} := by
  sorry

theorem part2 : CR_A ∩ B = {x | 7 ≤ x ∧ x < 10} := by
  sorry

theorem part3 (a : ℝ) (h : (A ∩ C a).Nonempty) : a > -3 := by
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l2250_225042


namespace NUMINAMATH_GPT_fraction_of_l2250_225079

theorem fraction_of (a b : ℝ) (h₁ : a = 1/5) (h₂ : b = 1/3) : (a / b) = 3/5 :=
by sorry

end NUMINAMATH_GPT_fraction_of_l2250_225079


namespace NUMINAMATH_GPT_power_increased_by_four_l2250_225081

-- Definitions from the conditions
variables (F k v : ℝ) (initial_force_eq_resistive : F = k * v)

-- Define the new conditions with double the force
variables (new_force : ℝ) (new_velocity : ℝ) (new_force_eq_resistive : new_force = k * new_velocity)
  (doubled_force : new_force = 2 * F)

-- The theorem statement
theorem power_increased_by_four (initial_force_eq_resistive : F = k * v) 
  (new_force_eq_resistive : new_force = k * new_velocity)
  (doubled_force : new_force = 2 * F) :
  new_velocity = 2 * v → 
  (new_force * new_velocity) = 4 * (F * v) :=
sorry

end NUMINAMATH_GPT_power_increased_by_four_l2250_225081


namespace NUMINAMATH_GPT_sprockets_produced_by_machines_l2250_225037

noncomputable def machine_sprockets (t : ℝ) : Prop :=
  let machineA_hours := t + 10
  let machineA_rate := 4
  let machineA_sprockets := machineA_hours * machineA_rate
  let machineB_hours := t
  let machineB_rate := 4.4
  let machineB_sprockets := machineB_hours * machineB_rate
  machineA_sprockets = 440 ∧ machineB_sprockets = 440

theorem sprockets_produced_by_machines (t : ℝ) (h : machine_sprockets t) : t = 100 :=
  sorry

end NUMINAMATH_GPT_sprockets_produced_by_machines_l2250_225037


namespace NUMINAMATH_GPT_tan_half_angle_product_l2250_225009

theorem tan_half_angle_product (a b : ℝ) (h : 3 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ (x : ℝ), x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) := 
sorry

end NUMINAMATH_GPT_tan_half_angle_product_l2250_225009


namespace NUMINAMATH_GPT_prime_count_of_first_10_sums_is_2_l2250_225038

open Nat

def consecutivePrimes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def consecutivePrimeSums (n : Nat) : List Nat :=
  (List.range n).scanl (λ sum i => sum + consecutivePrimes.getD i 0) 0

theorem prime_count_of_first_10_sums_is_2 :
  let sums := consecutivePrimeSums 10;
  (sums.count isPrime) = 2 :=
by
  sorry

end NUMINAMATH_GPT_prime_count_of_first_10_sums_is_2_l2250_225038


namespace NUMINAMATH_GPT_quadratic_has_negative_root_l2250_225095

def quadratic_function (m x : ℝ) : ℝ := (m - 2) * x^2 - 4 * m * x + 2 * m - 6

-- Define the discriminant of the quadratic function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the function that checks for a range of m such that the quadratic function intersects the negative x-axis
theorem quadratic_has_negative_root (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ quadratic_function m x = 0) ↔ (1 ≤ m ∧ m < 2 ∨ 2 < m ∧ m < 3) :=
sorry

end NUMINAMATH_GPT_quadratic_has_negative_root_l2250_225095


namespace NUMINAMATH_GPT_original_wage_before_increase_l2250_225057

theorem original_wage_before_increase (new_wage : ℝ) (increase_rate : ℝ) (original_wage : ℝ) (h : new_wage = original_wage + increase_rate * original_wage) : 
  new_wage = 42 → increase_rate = 0.50 → original_wage = 28 :=
by
  intros h_new_wage h_increase_rate
  have h1 : new_wage = 42 := h_new_wage
  have h2 : increase_rate = 0.50 := h_increase_rate
  have h3 : new_wage = original_wage + increase_rate * original_wage := h
  sorry

end NUMINAMATH_GPT_original_wage_before_increase_l2250_225057


namespace NUMINAMATH_GPT_geom_seq_sum_eqn_l2250_225054

theorem geom_seq_sum_eqn (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 2 + 2 * a 3 = a 1)
  (h2 : a 1 * a 4 = a 6)
  (h3 : ∀ n, a (n + 1) = a 1 * (1 / 2) ^ n)
  (h4 : ∀ n, S n = 2 * ((1 - (1 / 2) ^ n) / (1 - (1 / 2)))) :
  a n + S n = 4 :=
sorry

end NUMINAMATH_GPT_geom_seq_sum_eqn_l2250_225054


namespace NUMINAMATH_GPT_standard_deviation_calculation_l2250_225022

theorem standard_deviation_calculation : 
  let mean := 16.2 
  let stddev := 2.3 
  mean - 2 * stddev = 11.6 :=
by
  sorry

end NUMINAMATH_GPT_standard_deviation_calculation_l2250_225022


namespace NUMINAMATH_GPT_maximize_Sn_l2250_225055

theorem maximize_Sn (a1 : ℝ) (d : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (h1 : a1 > 0)
  (h2 : a1 + 9 * (a1 + 5 * d) = 0)
  (h_sn : ∀ n, S n = n / 2 * (2 * a1 + (n - 1) * d)) :
  ∃ n_max, ∀ n, S n ≤ S n_max ∧ n_max = 5 :=
by
  sorry

end NUMINAMATH_GPT_maximize_Sn_l2250_225055


namespace NUMINAMATH_GPT_triangle_side_length_l2250_225064

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (h₁ : a * Real.cos B = b * Real.sin A)
  (h₂ : C = Real.pi / 6) (h₃ : c = 2) : b = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l2250_225064
