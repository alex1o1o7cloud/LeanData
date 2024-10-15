import Mathlib

namespace NUMINAMATH_GPT_proof_problem_l1159_115918

def is_solution (x : ℝ) : Prop :=
  4 * Real.cos x * Real.cos (2 * x) * Real.cos (3 * x) = Real.cos (6 * x)

noncomputable def solution (l n : ℤ) : ℝ :=
  max (Real.pi / 3 * (3 * l + 1)) (Real.pi / 4 * (2 * n + 1))

theorem proof_problem (x : ℝ) (l n : ℤ) : is_solution x → x = solution l n :=
sorry

end NUMINAMATH_GPT_proof_problem_l1159_115918


namespace NUMINAMATH_GPT_problem_solution_includes_024_l1159_115964

theorem problem_solution_includes_024 (x : ℝ) :
  (2 * 88 * (abs (abs (abs (abs (x - 1) - 1) - 1) - 1)) = 0) →
  x = 0 ∨ x = 2 ∨ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_includes_024_l1159_115964


namespace NUMINAMATH_GPT_tan_arithmetic_sequence_l1159_115937

theorem tan_arithmetic_sequence {a : ℕ → ℝ}
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + n * d)
  (h_sum : a 1 + a 7 + a 13 = Real.pi) :
  Real.tan (a 2 + a 12) = - Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_arithmetic_sequence_l1159_115937


namespace NUMINAMATH_GPT_ratio_of_work_capacity_l1159_115945

theorem ratio_of_work_capacity (work_rate_A work_rate_B : ℝ)
  (hA : work_rate_A = 1 / 45)
  (hAB : work_rate_A + work_rate_B = 1 / 18) :
  work_rate_A⁻¹ / work_rate_B⁻¹ = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_work_capacity_l1159_115945


namespace NUMINAMATH_GPT_fan_airflow_in_one_week_l1159_115951

-- Define the conditions
def fan_airflow_per_second : ℕ := 10
def fan_working_minutes_per_day : ℕ := 10
def seconds_per_minute : ℕ := 60
def days_per_week : ℕ := 7

-- Define the proof problem
theorem fan_airflow_in_one_week : (fan_airflow_per_second * fan_working_minutes_per_day * seconds_per_minute * days_per_week = 42000) := 
by sorry

end NUMINAMATH_GPT_fan_airflow_in_one_week_l1159_115951


namespace NUMINAMATH_GPT_linear_condition_l1159_115999

theorem linear_condition (m : ℝ) : ¬ (m = 2) ↔ (∃ f : ℝ → ℝ, ∀ x, f x = (m - 2) * x + 2) :=
by
  sorry

end NUMINAMATH_GPT_linear_condition_l1159_115999


namespace NUMINAMATH_GPT_perfect_squares_of_nat_l1159_115962

theorem perfect_squares_of_nat (a b c : ℕ) (h : a^2 + b^2 + c^2 = (a - b)^2 + (b - c)^2 + (c - a)^2) :
  ∃ m n p q : ℕ, ab = m^2 ∧ bc = n^2 ∧ ca = p^2 ∧ ab + bc + ca = q^2 :=
by sorry

end NUMINAMATH_GPT_perfect_squares_of_nat_l1159_115962


namespace NUMINAMATH_GPT_fraction_zero_condition_l1159_115995

theorem fraction_zero_condition (x : ℝ) (h1 : (3 - |x|) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_zero_condition_l1159_115995


namespace NUMINAMATH_GPT_equal_elements_l1159_115968

theorem equal_elements (x : Fin 2011 → ℝ) (x' : Fin 2011 → ℝ)
  (h_perm : ∃ (σ : Equiv.Perm (Fin 2011)), ∀ i, x' i = x (σ i))
  (h_eq : ∀ i : Fin 2011, x i + x ((i + 1) % 2011) = 2 * x' i) :
  ∀ i j : Fin 2011, x i = x j :=
by
  sorry

end NUMINAMATH_GPT_equal_elements_l1159_115968


namespace NUMINAMATH_GPT_fred_dark_blue_marbles_count_l1159_115979

/-- Fred's Marble Problem -/
def freds_marbles (red green dark_blue : ℕ) : Prop :=
  red = 38 ∧ green = red / 2 ∧ red + green + dark_blue = 63

theorem fred_dark_blue_marbles_count (red green dark_blue : ℕ) (h : freds_marbles red green dark_blue) :
  dark_blue = 6 :=
by
  sorry

end NUMINAMATH_GPT_fred_dark_blue_marbles_count_l1159_115979


namespace NUMINAMATH_GPT_undefined_values_l1159_115983

-- Define the expression to check undefined values
noncomputable def is_undefined (x : ℝ) : Prop :=
  x^3 - 9 * x = 0

-- Statement: For which real values of x is the expression undefined?
theorem undefined_values (x : ℝ) : is_undefined x ↔ x = 0 ∨ x = -3 ∨ x = 3 :=
sorry

end NUMINAMATH_GPT_undefined_values_l1159_115983


namespace NUMINAMATH_GPT_find_other_integer_l1159_115976

theorem find_other_integer (x y : ℤ) (h_sum : 3 * x + 2 * y = 115) (h_one_is_25 : x = 25 ∨ y = 25) : (x = 25 → y = 20) ∧ (y = 25 → x = 20) :=
by
  sorry

end NUMINAMATH_GPT_find_other_integer_l1159_115976


namespace NUMINAMATH_GPT_trig_identity_l1159_115921

-- Given conditions
variables (α : ℝ) (h_tan : Real.tan (Real.pi - α) = -2)

-- The goal is to prove the desired equality.
theorem trig_identity :
  1 / (Real.cos (2 * α) + Real.cos α * Real.cos α) = -5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1159_115921


namespace NUMINAMATH_GPT_augmented_matrix_solution_l1159_115939

theorem augmented_matrix_solution (a b : ℝ) 
    (h1 : (∀ (x y : ℝ), (a * x = 2 ∧ y = b ↔ x = 2 ∧ y = 1))) : 
    a + b = 2 :=
by
  sorry

end NUMINAMATH_GPT_augmented_matrix_solution_l1159_115939


namespace NUMINAMATH_GPT_Lisa_weight_l1159_115906

theorem Lisa_weight : ∃ l a : ℝ, a + l = 240 ∧ l - a = l / 3 ∧ l = 144 :=
by
  sorry

end NUMINAMATH_GPT_Lisa_weight_l1159_115906


namespace NUMINAMATH_GPT_gardening_project_cost_l1159_115975

noncomputable def totalCost : Nat :=
  let roseBushes := 20
  let costPerRoseBush := 150
  let gardenerHourlyRate := 30
  let gardenerHoursPerDay := 5
  let gardenerDays := 4
  let soilCubicFeet := 100
  let soilCostPerCubicFoot := 5

  let costOfRoseBushes := costPerRoseBush * roseBushes
  let gardenerTotalHours := gardenerDays * gardenerHoursPerDay
  let costOfGardener := gardenerHourlyRate * gardenerTotalHours
  let costOfSoil := soilCostPerCubicFoot * soilCubicFeet

  costOfRoseBushes + costOfGardener + costOfSoil

theorem gardening_project_cost : totalCost = 4100 := by
  sorry

end NUMINAMATH_GPT_gardening_project_cost_l1159_115975


namespace NUMINAMATH_GPT_cost_per_day_additional_weeks_l1159_115946

theorem cost_per_day_additional_weeks :
  let first_week_days := 7
  let first_week_cost_per_day := 18.00
  let first_week_cost := first_week_days * first_week_cost_per_day
  let total_days := 23
  let total_cost := 302.00
  let additional_days := total_days - first_week_days
  let additional_cost := total_cost - first_week_cost
  let cost_per_day_additional := additional_cost / additional_days
  cost_per_day_additional = 11.00 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_day_additional_weeks_l1159_115946


namespace NUMINAMATH_GPT_sqrt_50_product_consecutive_integers_l1159_115935

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_50_product_consecutive_integers_l1159_115935


namespace NUMINAMATH_GPT_geometric_series_sum_infinity_l1159_115963

theorem geometric_series_sum_infinity (a₁ : ℝ) (q : ℝ) (S₆ S₃ : ℝ)
  (h₁ : a₁ = 3)
  (h₂ : S₆ / S₃ = 7 / 8)
  (h₃ : S₆ = a₁ * (1 - q ^ 6) / (1 - q))
  (h₄ : S₃ = a₁ * (1 - q ^ 3) / (1 - q)) :
  ∑' i : ℕ, a₁ * q ^ i = 2 := by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_infinity_l1159_115963


namespace NUMINAMATH_GPT_sequence_sum_difference_l1159_115920

def sum_odd (n : ℕ) : ℕ := n * n
def sum_even (n : ℕ) : ℕ := n * (n + 1)
def sum_triangular (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem sequence_sum_difference :
  sum_even 1500 - sum_odd 1500 + sum_triangular 1500 = 563628000 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_difference_l1159_115920


namespace NUMINAMATH_GPT_vertical_shift_d_l1159_115930

variable (a b c d : ℝ)

theorem vertical_shift_d (h1: d + a = 5) (h2: d - a = 1) : d = 3 := 
by
  sorry

end NUMINAMATH_GPT_vertical_shift_d_l1159_115930


namespace NUMINAMATH_GPT_value_of_40th_expression_l1159_115944

-- Define the sequence
def minuend (n : ℕ) : ℕ := 100 - (n - 1)
def subtrahend (n : ℕ) : ℕ := n
def expression_value (n : ℕ) : ℕ := minuend n - subtrahend n

-- Theorem: The value of the 40th expression in the sequence is 21
theorem value_of_40th_expression : expression_value 40 = 21 := by
  show 100 - (40 - 1) - 40 = 21
  sorry

end NUMINAMATH_GPT_value_of_40th_expression_l1159_115944


namespace NUMINAMATH_GPT_distinct_nonzero_real_product_l1159_115922

noncomputable section
open Real

theorem distinct_nonzero_real_product
  (a b c d : ℝ)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hcd : c ≠ d)
  (hda : d ≠ a)
  (ha_ne_0 : a ≠ 0)
  (hb_ne_0 : b ≠ 0)
  (hc_ne_0 : c ≠ 0)
  (hd_ne_0 : d ≠ 0)
  (h : a + 1/b = b + 1/c ∧ b + 1/c = c + 1/d ∧ c + 1/d = d + 1/a) :
  |a * b * c * d| = 1 :=
sorry

end NUMINAMATH_GPT_distinct_nonzero_real_product_l1159_115922


namespace NUMINAMATH_GPT_number_of_three_digit_multiples_of_7_l1159_115981

theorem number_of_three_digit_multiples_of_7 : 
  let smallest_multiple := 7 * Nat.ceil (100 / 7)
  let largest_multiple := 7 * Nat.floor (999 / 7)
  (largest_multiple - smallest_multiple) / 7 + 1 = 128 :=
by
  sorry

end NUMINAMATH_GPT_number_of_three_digit_multiples_of_7_l1159_115981


namespace NUMINAMATH_GPT_intersection_A_B_l1159_115967

open Set Real -- Opens necessary namespaces for sets and real numbers

-- Definitions for the sets A and B
def A : Set ℝ := {x | 1 / x < 1}
def B : Set ℝ := {x | x > -1}

-- The proof statement for the intersection of sets A and B
theorem intersection_A_B : A ∩ B = (Ioo (-1 : ℝ) 0) ∪ (Ioi 1) :=
by
  sorry -- Proof not included

end NUMINAMATH_GPT_intersection_A_B_l1159_115967


namespace NUMINAMATH_GPT_bryce_received_15_raisins_l1159_115923

theorem bryce_received_15_raisins (x : ℕ) (c : ℕ) (h1 : c = x - 10) (h2 : c = x / 3) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_bryce_received_15_raisins_l1159_115923


namespace NUMINAMATH_GPT_gcd_proof_l1159_115941

theorem gcd_proof :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b = 33 ∧ Nat.lcm a b = 90 ∧ Nat.gcd a b = 3 :=
sorry

end NUMINAMATH_GPT_gcd_proof_l1159_115941


namespace NUMINAMATH_GPT_prime_m_l1159_115932

theorem prime_m (m : ℕ) (hm : m ≥ 2) :
  (∀ n : ℕ, (m / 3 ≤ n) → (n ≤ m / 2) → (n ∣ Nat.choose n (m - 2 * n))) → Nat.Prime m :=
by
  intro h
  sorry

end NUMINAMATH_GPT_prime_m_l1159_115932


namespace NUMINAMATH_GPT_inequality_proof_l1159_115917

theorem inequality_proof (a b : ℝ) (x y : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_x : 0 < x) (h_y : 0 < y) : 
  (a^2 / x) + (b^2 / y) ≥ ((a + b)^2 / (x + y)) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1159_115917


namespace NUMINAMATH_GPT_range_of_a_for_quad_ineq_false_l1159_115902

variable (a : ℝ)

def quad_ineq_holds : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0

theorem range_of_a_for_quad_ineq_false :
  ¬ quad_ineq_holds a → 0 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_quad_ineq_false_l1159_115902


namespace NUMINAMATH_GPT_choose_two_items_proof_l1159_115954

   def number_of_ways_to_choose_two_items (n : ℕ) : ℕ :=
     n * (n - 1) / 2

   theorem choose_two_items_proof (n : ℕ) : number_of_ways_to_choose_two_items n = (n * (n - 1)) / 2 :=
   by
     sorry
   
end NUMINAMATH_GPT_choose_two_items_proof_l1159_115954


namespace NUMINAMATH_GPT_parabola_translation_l1159_115908

theorem parabola_translation :
  ∀ x y, (y = -2 * x^2) →
    ∃ x' y', y' = -2 * (x' - 2)^2 + 1 ∧ x' = x ∧ y' = y + 1 :=
sorry

end NUMINAMATH_GPT_parabola_translation_l1159_115908


namespace NUMINAMATH_GPT_math_problem_l1159_115934

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

def a_n (n : ℕ) : ℕ := 3 * n - 5

theorem math_problem (C5_4 : ℕ) (C6_4 : ℕ) (C7_4 : ℕ) :
  C5_4 = binomial 5 4 →
  C6_4 = binomial 6 4 →
  C7_4 = binomial 7 4 →
  C5_4 + C6_4 + C7_4 = 55 →
  ∃ n : ℕ, a_n n = 55 ∧ n = 20 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1159_115934


namespace NUMINAMATH_GPT_distance_between_A_and_B_l1159_115938

theorem distance_between_A_and_B (v_A v_B d d' : ℝ)
  (h1 : v_B = 50)
  (h2 : (v_A - v_B) * 30 = d')
  (h3 : (v_A + v_B) * 6 = d) :
  d = 750 :=
sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l1159_115938


namespace NUMINAMATH_GPT_part1_part2_l1159_115994

-- Definition of the function
def f (a x : ℝ) := |x - a|

-- Proof statement for question 1
theorem part1 (a : ℝ)
  (h : ∀ x : ℝ, f a x ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) :
  a = 3 := by
  sorry

-- Auxiliary function for question 2
def g (a x : ℝ) := f a (2 * x) + f a (x + 2)

-- Proof statement for question 2
theorem part2 (m : ℝ)
  (h : ∀ x : ℝ, g 3 x ≥ m) :
  m ≤ 1/2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1159_115994


namespace NUMINAMATH_GPT_cookies_per_batch_l1159_115903

theorem cookies_per_batch (students : ℕ) (cookies_per_student : ℕ) (chocolate_batches : ℕ) (oatmeal_batches : ℕ) (additional_batches : ℕ) (cookies_needed : ℕ) (dozens_per_batch : ℕ) :
  (students = 24) →
  (cookies_per_student = 10) →
  (chocolate_batches = 2) →
  (oatmeal_batches = 1) →
  (additional_batches = 2) →
  (cookies_needed = students * cookies_per_student) →
  dozens_per_batch * (12 * (chocolate_batches + oatmeal_batches + additional_batches)) = cookies_needed →
  dozens_per_batch = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cookies_per_batch_l1159_115903


namespace NUMINAMATH_GPT_find_a_value_l1159_115928

theorem find_a_value 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h : ∀ x : ℝ, f x = x^3 + a*x^2 + 3*x - 9)
  (extreme_at_minus_3 : ∀ f' : ℝ → ℝ, (∀ x, f' x = 3*x^2 + 2*a*x + 3) → f' (-3) = 0) :
  a = 5 := 
sorry

end NUMINAMATH_GPT_find_a_value_l1159_115928


namespace NUMINAMATH_GPT_guaranteed_winning_strategy_l1159_115996

variable (a b : ℝ)

theorem guaranteed_winning_strategy (h : a ≠ b) : (a^3 + b^3) > (a^2 * b + a * b^2) :=
by 
  sorry

end NUMINAMATH_GPT_guaranteed_winning_strategy_l1159_115996


namespace NUMINAMATH_GPT_perimeter_of_flowerbed_l1159_115993

def width : ℕ := 4
def length : ℕ := 2 * width - 1
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem perimeter_of_flowerbed : perimeter length width = 22 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_flowerbed_l1159_115993


namespace NUMINAMATH_GPT_simplify_fraction_l1159_115948

theorem simplify_fraction (a : ℝ) (h : a = 2) : (15 * a^4) / (75 * a^3) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1159_115948


namespace NUMINAMATH_GPT_math_problem_l1159_115977

theorem math_problem
    (p q s : ℕ)
    (prime_p : Nat.Prime p)
    (prime_q : Nat.Prime q)
    (prime_s : Nat.Prime s)
    (h1 : p * q = s + 6)
    (h2 : 3 < p)
    (h3 : p < q) :
    p = 5 :=
    sorry

end NUMINAMATH_GPT_math_problem_l1159_115977


namespace NUMINAMATH_GPT_train_length_l1159_115973

theorem train_length (T : ℕ) (S : ℕ) (conversion_factor : ℚ) (L : ℕ) 
  (hT : T = 16)
  (hS : S = 108)
  (hconv : conversion_factor = 5 / 18)
  (hL : L = 480) :
  L = ((S * conversion_factor : ℚ) * T : ℚ) :=
sorry

end NUMINAMATH_GPT_train_length_l1159_115973


namespace NUMINAMATH_GPT_woman_work_completion_woman_days_to_complete_l1159_115949

theorem woman_work_completion (M W B : ℝ) (h1 : M + W + B = 1/4) (h2 : M = 1/6) (h3 : B = 1/18) : W = 1/36 :=
by
  -- Substitute h2 and h3 into h1 and solve for W
  sorry

theorem woman_days_to_complete (W : ℝ) (h : W = 1/36) : 1 / W = 36 :=
by
  -- Calculate the reciprocal of h
  sorry

end NUMINAMATH_GPT_woman_work_completion_woman_days_to_complete_l1159_115949


namespace NUMINAMATH_GPT_ratio_problem_l1159_115980

theorem ratio_problem (a b c d : ℝ) (h1 : a / b = 5) (h2 : b / c = 1 / 2) (h3 : c / d = 6) : 
  d / a = 1 / 15 :=
by sorry

end NUMINAMATH_GPT_ratio_problem_l1159_115980


namespace NUMINAMATH_GPT_nelly_part_payment_is_875_l1159_115985

noncomputable def part_payment (total_cost remaining_amount : ℝ) :=
  0.25 * total_cost

theorem nelly_part_payment_is_875 (total_cost : ℝ) (remaining_amount : ℝ)
  (h1 : remaining_amount = 2625)
  (h2 : remaining_amount = 0.75 * total_cost) :
  part_payment total_cost remaining_amount = 875 :=
by
  sorry

end NUMINAMATH_GPT_nelly_part_payment_is_875_l1159_115985


namespace NUMINAMATH_GPT_divisibility_condition_l1159_115978

theorem divisibility_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ab ∣ (a^2 + b^2 - a - b + 1) → (a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_GPT_divisibility_condition_l1159_115978


namespace NUMINAMATH_GPT_inequality_always_holds_l1159_115998

theorem inequality_always_holds (a : ℝ) (h : a ≥ -2) : ∀ (x : ℝ), x^2 + a * |x| + 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_inequality_always_holds_l1159_115998


namespace NUMINAMATH_GPT_unique_x_floor_eq_20_7_l1159_115961

theorem unique_x_floor_eq_20_7 : ∀ x : ℝ, (⌊x⌋ + x + 1/2 = 20.7) → x = 10.2 :=
by
  sorry

end NUMINAMATH_GPT_unique_x_floor_eq_20_7_l1159_115961


namespace NUMINAMATH_GPT_solve_simultaneous_equations_l1159_115927

theorem solve_simultaneous_equations (a b : ℚ) : 
  (a + b) * (a^2 - b^2) = 4 ∧ (a - b) * (a^2 + b^2) = 5 / 2 → 
  (a = 3 / 2 ∧ b = 1 / 2) ∨ (a = -1 / 2 ∧ b = -3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_simultaneous_equations_l1159_115927


namespace NUMINAMATH_GPT_multiples_of_10_5_l1159_115959

theorem multiples_of_10_5 (n : ℤ) (h1 : ∀ k : ℤ, k % 10 = 0 → k % 5 = 0) (h2 : n % 10 = 0) : n % 5 = 0 := 
by
  sorry

end NUMINAMATH_GPT_multiples_of_10_5_l1159_115959


namespace NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l1159_115925

theorem isosceles_triangle_vertex_angle (A B C : ℝ) (hABC : A + B + C = 180) (h_iso : A = B ∨ B = C ∨ A = C) (h_angle : A = 50 ∨ B = 50 ∨ C = 50) : (A = 50 ∨ A = 80) ∨ (B = 50 ∨ B = 80) ∨ (C = 50 ∨ C = 80) :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l1159_115925


namespace NUMINAMATH_GPT_points_on_equation_correct_l1159_115987

theorem points_on_equation_correct (x y : ℝ) :
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) :=
by 
  sorry

end NUMINAMATH_GPT_points_on_equation_correct_l1159_115987


namespace NUMINAMATH_GPT_geometric_sequence_a2_l1159_115950

noncomputable def geometric_sequence_sum (n : ℕ) (a : ℝ) : ℝ :=
  a * (3^n) - 2

theorem geometric_sequence_a2 (a : ℝ) : (∃ a1 a2 a3 : ℝ, 
  a1 = geometric_sequence_sum 1 a ∧ 
  a1 + a2 = geometric_sequence_sum 2 a ∧ 
  a1 + a2 + a3 = geometric_sequence_sum 3 a ∧ 
  a2 = 6 * a ∧ 
  a3 = 18 * a ∧ 
  (6 * a)^2 = (a1) * (a3) ∧ 
  a = 2) →
  a2 = 12 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_geometric_sequence_a2_l1159_115950


namespace NUMINAMATH_GPT_find_m_value_l1159_115936

def magic_box (a b : ℝ) : ℝ := a^2 + 2 * b - 3

theorem find_m_value (m : ℝ) :
  magic_box m (-3 * m) = 4 ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l1159_115936


namespace NUMINAMATH_GPT_larger_factor_of_lcm_l1159_115965

theorem larger_factor_of_lcm (A B : ℕ) (hcf lcm X Y : ℕ) 
  (h_hcf: hcf = 63)
  (h_A: A = 1071)
  (h_lcm: lcm = hcf * X * Y)
  (h_X: X = 11)
  (h_factors: ∃ k: ℕ, A = hcf * k ∧ lcm = A * (B / k)):
  Y = 17 := 
by sorry

end NUMINAMATH_GPT_larger_factor_of_lcm_l1159_115965


namespace NUMINAMATH_GPT_find_num_cows_l1159_115914

variable (num_cows num_pigs : ℕ)

theorem find_num_cows (h1 : 4 * num_cows + 24 + 4 * num_pigs = 20 + 2 * (num_cows + 6 + num_pigs)) 
                      (h2 : 6 = 6) 
                      (h3 : ∀x, 2 * x = x + x) 
                      (h4 : ∀x, 4 * x = 2 * 2 * x) 
                      (h5 : ∀x, 4 * x = 4 * x) : 
                      num_cows = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_num_cows_l1159_115914


namespace NUMINAMATH_GPT_domain_of_f_2x_minus_1_l1159_115989

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) (dom : ∀ x, f x ≠ 0 → (0 < x ∧ x < 1)) :
  ∀ x, f (2*x - 1) ≠ 0 → (1/2 < x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_2x_minus_1_l1159_115989


namespace NUMINAMATH_GPT_problem_statement_l1159_115931

noncomputable def smallest_x : ℝ :=
  -8 - (Real.sqrt 292 / 2)

theorem problem_statement (x : ℝ) :
  (15 * x ^ 2 - 40 * x + 18) / (4 * x - 3) + 4 * x = 8 * x - 3 ↔ x = smallest_x :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1159_115931


namespace NUMINAMATH_GPT_hyperbola_eccentricity_correct_l1159_115991

noncomputable def hyperbola_eccentricity : Real :=
  let a := 5
  let b := 4
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  c / a

theorem hyperbola_eccentricity_correct :
  hyperbola_eccentricity = Real.sqrt 41 / 5 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_correct_l1159_115991


namespace NUMINAMATH_GPT_yolanda_three_point_avg_l1159_115905

-- Definitions based on conditions
def total_points_season := 345
def total_games := 15
def free_throws_per_game := 4
def two_point_baskets_per_game := 5

-- Definitions based on the derived quantities
def average_points_per_game := total_points_season / total_games
def points_from_two_point_baskets := two_point_baskets_per_game * 2
def points_from_free_throws := free_throws_per_game * 1
def points_from_non_three_point_baskets := points_from_two_point_baskets + points_from_free_throws
def points_from_three_point_baskets := average_points_per_game - points_from_non_three_point_baskets
def three_point_baskets_per_game := points_from_three_point_baskets / 3

-- The theorem to prove that Yolanda averaged 3 three-point baskets per game
theorem yolanda_three_point_avg:
  three_point_baskets_per_game = 3 := sorry

end NUMINAMATH_GPT_yolanda_three_point_avg_l1159_115905


namespace NUMINAMATH_GPT_birds_percentage_hawks_l1159_115997

-- Define the conditions and the main proof problem
theorem birds_percentage_hawks (H : ℝ) :
  (0.4 * (1 - H) + 0.25 * 0.4 * (1 - H) + H = 0.65) → (H = 0.3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_birds_percentage_hawks_l1159_115997


namespace NUMINAMATH_GPT_no_lonely_points_eventually_l1159_115956

structure Graph (α : Type) :=
(vertices : Finset α)
(edges : α → Finset α)

namespace Graph

def is_lonely {α : Type} (G : Graph α) (coloring : α → Bool) (v : α) : Prop :=
  let neighbors := G.edges v
  let different_color_neighbors := neighbors.filter (λ w => coloring w ≠ coloring v)
  2 * different_color_neighbors.card > neighbors.card

end Graph

theorem no_lonely_points_eventually
  {α : Type}
  (G : Graph α)
  (initial_coloring : α → Bool) :
  ∃ (steps : Nat),
  ∀ (coloring : α → Bool),
  (∃ (t : Nat), t ≤ steps ∧ 
    (∀ v, ¬ Graph.is_lonely G coloring v)) :=
sorry

end NUMINAMATH_GPT_no_lonely_points_eventually_l1159_115956


namespace NUMINAMATH_GPT_rhombus_area_correct_l1159_115943

def rhombus_area (d1 d2 : ℕ) : ℕ :=
  (d1 * d2) / 2

theorem rhombus_area_correct
  (d1 d2 : ℕ)
  (h1 : d1 = 70)
  (h2 : d2 = 160) :
  rhombus_area d1 d2 = 5600 := 
by
  sorry

end NUMINAMATH_GPT_rhombus_area_correct_l1159_115943


namespace NUMINAMATH_GPT_period_of_f_g_is_2_sin_x_g_is_odd_l1159_115916

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 3)

-- Theorem 1: Prove that f has period 2π.
theorem period_of_f : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry

-- Define g and prove the related properties.
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 3)

-- Theorem 2: Prove that g(x) = 2 * sin x.
theorem g_is_2_sin_x : ∀ x : ℝ, g x = 2 * Real.sin x := by
  sorry

-- Theorem 3: Prove that g is an odd function.
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end NUMINAMATH_GPT_period_of_f_g_is_2_sin_x_g_is_odd_l1159_115916


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l1159_115957

theorem solve_equation1 (x : ℝ) (h1 : 5 * x - 2 * (x - 1) = 3) : x = 1 / 3 := 
sorry

theorem solve_equation2 (x : ℝ) (h2 : (x + 3) / 2 - 1 = (2 * x - 1) / 3) : x = 5 :=
sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l1159_115957


namespace NUMINAMATH_GPT_min_period_and_max_value_l1159_115926

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - (Real.sin x)^2 + 2

theorem min_period_and_max_value :
  (∀ x, f (x + π) = f x) ∧ (∀ x, f x ≤ 4) ∧ (∃ x, f x = 4) :=
by
  sorry

end NUMINAMATH_GPT_min_period_and_max_value_l1159_115926


namespace NUMINAMATH_GPT_johns_profit_l1159_115907

noncomputable def profit_made 
  (trees_chopped : ℕ)
  (planks_per_tree : ℕ)
  (planks_per_table : ℕ)
  (price_per_table : ℕ)
  (labor_cost : ℕ) : ℕ :=
(trees_chopped * planks_per_tree / planks_per_table) * price_per_table - labor_cost

theorem johns_profit : profit_made 30 25 15 300 3000 = 12000 :=
by sorry

end NUMINAMATH_GPT_johns_profit_l1159_115907


namespace NUMINAMATH_GPT_find_length_AB_l1159_115942

theorem find_length_AB 
(distance_between_parallels : ℚ)
(radius_of_incircle : ℚ)
(is_isosceles : Prop)
(h_parallel : distance_between_parallels = 18 / 25)
(h_radius : radius_of_incircle = 8 / 3)
(h_isosceles : is_isosceles) :
  ∃ AB : ℚ, AB = 20 := 
sorry

end NUMINAMATH_GPT_find_length_AB_l1159_115942


namespace NUMINAMATH_GPT_max_sum_a_b_c_d_e_f_g_l1159_115971

theorem max_sum_a_b_c_d_e_f_g (a b c d e f g : ℕ)
  (h1 : a + b + c = 2)
  (h2 : b + c + d = 2)
  (h3 : c + d + e = 2)
  (h4 : d + e + f = 2)
  (h5 : e + f + g = 2) :
  a + b + c + d + e + f + g ≤ 6 := 
sorry

end NUMINAMATH_GPT_max_sum_a_b_c_d_e_f_g_l1159_115971


namespace NUMINAMATH_GPT_speed_of_stream_l1159_115988

theorem speed_of_stream
  (v_a v_s : ℝ)
  (h1 : v_a - v_s = 4)
  (h2 : v_a + v_s = 6) :
  v_s = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_speed_of_stream_l1159_115988


namespace NUMINAMATH_GPT_find_phi_l1159_115974

theorem find_phi (ϕ : ℝ) (h0 : 0 ≤ ϕ) (h1 : ϕ < π)
    (H : 2 * Real.cos (π / 3) = 2 * Real.sin (2 * (π / 3) + ϕ)) : ϕ = π / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_phi_l1159_115974


namespace NUMINAMATH_GPT_find_x_l1159_115901

theorem find_x (x : ℝ) (h : (2015 + x)^2 = x^2) : x = -2015 / 2 := by
  sorry

end NUMINAMATH_GPT_find_x_l1159_115901


namespace NUMINAMATH_GPT_least_z_minus_x_l1159_115969

theorem least_z_minus_x (x y z : ℤ) (h1 : x < y) (h2 : y < z) (h3 : y - x > 3) (h4 : Even x) (h5 : Odd y) (h6 : Odd z) : z - x = 7 :=
sorry

end NUMINAMATH_GPT_least_z_minus_x_l1159_115969


namespace NUMINAMATH_GPT_range_of_a_l1159_115972

theorem range_of_a (a : ℝ) :
  (∀ (x y z: ℝ), x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2 * y + 2 * z) ↔ (a ≤ -2 ∨ a ≥ 4) :=
by
sorry

end NUMINAMATH_GPT_range_of_a_l1159_115972


namespace NUMINAMATH_GPT_find_number_l1159_115900

theorem find_number (N : ℝ) (h : 6 + (1/2) * (1/3) * (1/5) * N = (1/15) * N) : N = 180 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l1159_115900


namespace NUMINAMATH_GPT_fred_has_9_dimes_l1159_115929

-- Fred has 90 cents in his bank.
def freds_cents : ℕ := 90

-- A dime is worth 10 cents.
def value_of_dime : ℕ := 10

-- Prove that the number of dimes Fred has is 9.
theorem fred_has_9_dimes : (freds_cents / value_of_dime) = 9 := by
  sorry

end NUMINAMATH_GPT_fred_has_9_dimes_l1159_115929


namespace NUMINAMATH_GPT_average_weight_of_boys_l1159_115955

theorem average_weight_of_boys 
  (n1 n2 : ℕ) 
  (w1 w2 : ℝ) 
  (h1 : n1 = 22) 
  (h2 : n2 = 8) 
  (h3 : w1 = 50.25) 
  (h4 : w2 = 45.15) : 
  (n1 * w1 + n2 * w2) / (n1 + n2) = 48.89 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_of_boys_l1159_115955


namespace NUMINAMATH_GPT_linear_function_decreases_l1159_115947

theorem linear_function_decreases (m b x : ℝ) (h : m < 0) : 
  ∃ y : ℝ, y = m * x + b ∧ ∀ x₁ x₂ : ℝ, x₁ < x₂ → (m * x₁ + b) > (m * x₂ + b) :=
by 
  sorry

end NUMINAMATH_GPT_linear_function_decreases_l1159_115947


namespace NUMINAMATH_GPT_maple_is_taller_l1159_115913

def pine_tree_height : ℚ := 13 + 1/4
def maple_tree_height : ℚ := 20 + 1/2
def height_difference : ℚ := maple_tree_height - pine_tree_height

theorem maple_is_taller : height_difference = 7 + 1/4 := by
  sorry

end NUMINAMATH_GPT_maple_is_taller_l1159_115913


namespace NUMINAMATH_GPT_math_problem_l1159_115919

theorem math_problem : (100 - (5050 - 450)) + (5050 - (450 - 100)) = 200 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1159_115919


namespace NUMINAMATH_GPT_problem_l1159_115952

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem problem (A_def : A = {-1, 0, 1}) : B = {0, 1} :=
by sorry

end NUMINAMATH_GPT_problem_l1159_115952


namespace NUMINAMATH_GPT_locus_of_point_T_l1159_115960

theorem locus_of_point_T (r : ℝ) (a b : ℝ) (x y x1 y1 x2 y2 : ℝ)
  (hM_inside : a^2 + b^2 < r^2)
  (hK_on_circle : x1^2 + y1^2 = r^2)
  (hP_on_circle : x2^2 + y2^2 = r^2)
  (h_midpoints_eq : (x + a) / 2 = (x1 + x2) / 2 ∧ (y + b) / 2 = (y1 + y2) / 2)
  (h_diagonal_eq : (x - a)^2 + (y - b)^2 = (x1 - x2)^2 + (y1 - y2)^2) :
  x^2 + y^2 = 2 * r^2 - (a^2 + b^2) :=
  sorry

end NUMINAMATH_GPT_locus_of_point_T_l1159_115960


namespace NUMINAMATH_GPT_chelsea_sugar_bags_l1159_115982

variable (n : ℕ)

-- Defining the conditions as hypotheses
def initial_sugar : ℕ := 24
def remaining_sugar : ℕ := 21
def sugar_lost : ℕ := initial_sugar - remaining_sugar
def torn_bag_sugar : ℕ := 2 * sugar_lost

-- Define the statement to prove
theorem chelsea_sugar_bags :
  n = initial_sugar / torn_bag_sugar → n = 4 :=
by
  sorry

end NUMINAMATH_GPT_chelsea_sugar_bags_l1159_115982


namespace NUMINAMATH_GPT_atomic_weight_Ca_l1159_115958

def molecular_weight_CaH2 : ℝ := 42
def atomic_weight_H : ℝ := 1.008

theorem atomic_weight_Ca : atomic_weight_H * 2 < molecular_weight_CaH2 :=
by sorry

end NUMINAMATH_GPT_atomic_weight_Ca_l1159_115958


namespace NUMINAMATH_GPT_find_bounds_l1159_115912

open Set

variable {U : Type} [TopologicalSpace U]

def A := {x : ℝ | 3 ≤ x ∧ x ≤ 4}
def C_UA := {x : ℝ | x > 4 ∨ x < 3}

theorem find_bounds (T : Type) [TopologicalSpace T] : 3 = 3 ∧ 4 = 4 := 
 by sorry

end NUMINAMATH_GPT_find_bounds_l1159_115912


namespace NUMINAMATH_GPT_miles_mike_ride_l1159_115909

theorem miles_mike_ride
  (cost_per_mile : ℝ) (start_fee : ℝ) (bridge_toll : ℝ)
  (annie_miles : ℝ) (annie_total_cost : ℝ)
  (mike_total_cost : ℝ) (M : ℝ)
  (h1 : cost_per_mile = 0.25)
  (h2 : start_fee = 2.50)
  (h3 : bridge_toll = 5.00)
  (h4 : annie_miles = 26)
  (h5 : annie_total_cost = start_fee + bridge_toll + cost_per_mile * annie_miles)
  (h6 : mike_total_cost = start_fee + cost_per_mile * M)
  (h7 : mike_total_cost = annie_total_cost) :
  M = 36 := 
sorry

end NUMINAMATH_GPT_miles_mike_ride_l1159_115909


namespace NUMINAMATH_GPT_chord_lengths_equal_l1159_115904

theorem chord_lengths_equal (D E F : ℝ) (hcond_1 : D^2 ≠ E^2) (hcond_2 : E^2 > 4 * F) :
  ∀ x y, (x^2 + y^2 + D * x + E * y + F = 0) → 
  (abs x = abs y) :=
by
  sorry

end NUMINAMATH_GPT_chord_lengths_equal_l1159_115904


namespace NUMINAMATH_GPT_solve_inequality_l1159_115915

theorem solve_inequality (x : ℝ) : 
  (x ≠ 1) → ( (x^3 - 3*x^2 + 2*x + 1) / (x^2 - 2*x + 1) ≤ 2 ) ↔ 
  (2 - Real.sqrt 3 < x ∧ x < 1) ∨ (1 < x ∧ x < 2 + Real.sqrt 3) := 
sorry

end NUMINAMATH_GPT_solve_inequality_l1159_115915


namespace NUMINAMATH_GPT_sample_size_l1159_115984

variable (total_employees : ℕ) (young_employees : ℕ) (middle_aged_employees : ℕ) (elderly_employees : ℕ) (young_in_sample : ℕ)

theorem sample_size (h1 : total_employees = 750) (h2 : young_employees = 350) (h3 : middle_aged_employees = 250) (h4 : elderly_employees = 150) (h5 : young_in_sample = 7) :
  ∃ sample_size, young_in_sample * total_employees / young_employees = sample_size ∧ sample_size = 15 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_l1159_115984


namespace NUMINAMATH_GPT_greatest_power_of_two_l1159_115970

theorem greatest_power_of_two (n : ℕ) (h1 : n = 1004) (h2 : 10^n - 4^(n / 2) = k) : ∃ m : ℕ, 2 ∣ k ∧ m = 1007 :=
by
  sorry

end NUMINAMATH_GPT_greatest_power_of_two_l1159_115970


namespace NUMINAMATH_GPT_jake_pure_alcohol_l1159_115966

theorem jake_pure_alcohol (total_shots : ℕ) (shots_per_split : ℕ) (ounces_per_shot : ℚ) (purity : ℚ) :
  total_shots = 8 →
  shots_per_split = 2 →
  ounces_per_shot = 1.5 →
  purity = 0.5 →
  (total_shots / shots_per_split) * ounces_per_shot * purity = 3 := 
by
  sorry

end NUMINAMATH_GPT_jake_pure_alcohol_l1159_115966


namespace NUMINAMATH_GPT_impossible_circular_arrangement_1_to_60_l1159_115924

theorem impossible_circular_arrangement_1_to_60 :
  (∀ (f : ℕ → ℕ), 
      (∀ n, 1 ≤ f n ∧ f n ≤ 60) ∧ 
      (∀ n, f (n + 2) + f n ≡ 0 [MOD 2]) ∧ 
      (∀ n, f (n + 3) + f n ≡ 0 [MOD 3]) ∧ 
      (∀ n, f (n + 7) + f n ≡ 0 [MOD 7]) 
      → false) := 
  sorry

end NUMINAMATH_GPT_impossible_circular_arrangement_1_to_60_l1159_115924


namespace NUMINAMATH_GPT_machine_x_produces_40_percent_l1159_115992

theorem machine_x_produces_40_percent (T X Y : ℝ) 
  (h1 : X + Y = T)
  (h2 : 0.009 * X + 0.004 * Y = 0.006 * T) :
  X = 0.4 * T :=
by
  sorry

end NUMINAMATH_GPT_machine_x_produces_40_percent_l1159_115992


namespace NUMINAMATH_GPT_ant_minimum_distance_l1159_115933

section
variables (x y z w u : ℝ)

-- Given conditions
axiom h1 : x + y + z = 22
axiom h2 : w + y + z = 29
axiom h3 : x + y + u = 30

-- Prove the ant crawls at least 47 cm to cover all paths
theorem ant_minimum_distance : x + y + z + w ≥ 47 :=
sorry
end

end NUMINAMATH_GPT_ant_minimum_distance_l1159_115933


namespace NUMINAMATH_GPT_circular_pond_area_l1159_115911

theorem circular_pond_area (AB CD : ℝ) (D_is_midpoint : Prop) (hAB : AB = 20) (hCD : CD = 12)
  (hD_midpoint : D_is_midpoint ∧ D_is_midpoint = (AB / 2 = 10)) :
  ∃ (A : ℝ), A = 244 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circular_pond_area_l1159_115911


namespace NUMINAMATH_GPT_expected_value_of_fair_8_sided_die_l1159_115986

-- Define the outcomes of the fair 8-sided die
def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Define the probability of each outcome for a fair die
def prob (n : ℕ) : ℚ := 1 / 8

-- Calculate the expected value of the outcomes
noncomputable def expected_value : ℚ :=
  (outcomes.map (λ x => prob x * x)).sum

-- State the theorem that the expected value is 4.5
theorem expected_value_of_fair_8_sided_die : expected_value = 4.5 :=
  sorry

end NUMINAMATH_GPT_expected_value_of_fair_8_sided_die_l1159_115986


namespace NUMINAMATH_GPT_margaret_time_l1159_115953

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

def total_permutations (n : Nat) : Nat :=
  factorial n

def total_time_in_minutes (total_permutations : Nat) (rate : Nat) : Nat :=
  total_permutations / rate

def time_in_hours_and_minutes (total_minutes : Nat) : Nat × Nat :=
  let hours := total_minutes / 60
  let minutes := total_minutes % 60
  (hours, minutes)

theorem margaret_time :
  let n := 8
  let r := 15
  let permutations := total_permutations n
  let total_minutes := total_time_in_minutes permutations r
  time_in_hours_and_minutes total_minutes = (44, 48) := by
  sorry

end NUMINAMATH_GPT_margaret_time_l1159_115953


namespace NUMINAMATH_GPT_jean_vs_pauline_cost_l1159_115910

-- Definitions based on the conditions given
def patty_cost (ida_cost : ℕ) : ℕ := ida_cost + 10
def ida_cost (jean_cost : ℕ) : ℕ := jean_cost + 30
def pauline_cost : ℕ := 30

noncomputable def total_cost (jean_cost : ℕ) : ℕ :=
jean_cost + ida_cost jean_cost + patty_cost (ida_cost jean_cost) + pauline_cost

-- Lean 4 statement to prove the required condition
theorem jean_vs_pauline_cost :
  ∃ (jean_cost : ℕ), total_cost jean_cost = 160 ∧ pauline_cost - jean_cost = 10 :=
by
  sorry

end NUMINAMATH_GPT_jean_vs_pauline_cost_l1159_115910


namespace NUMINAMATH_GPT_star_three_four_eq_zero_l1159_115990

def star (a b : ℕ) : ℕ := 4 * a + 3 * b - 2 * a * b

theorem star_three_four_eq_zero : star 3 4 = 0 := sorry

end NUMINAMATH_GPT_star_three_four_eq_zero_l1159_115990


namespace NUMINAMATH_GPT_lottery_probability_exactly_one_common_l1159_115940

open Nat

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_exactly_one_common :
  let total_combinations := binomial 45 6
  let successful_combinations := 6 * binomial 39 5
  let probability := (successful_combinations : ℚ) / total_combinations
  probability = (6 * binomial 39 5 : ℚ) / binomial 45 6 :=
by
  sorry

end NUMINAMATH_GPT_lottery_probability_exactly_one_common_l1159_115940
