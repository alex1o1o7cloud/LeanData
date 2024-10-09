import Mathlib

namespace sequence_solution_l2090_209053

theorem sequence_solution :
  ∀ (a : ℕ → ℝ), (∀ m n : ℕ, a (m^2 + n^2) = a m ^ 2 + a n ^ 2) →
  (0 ≤ a 0 ∧ a 0 ≤ a 1 ∧ a 1 ≤ a 2 ∧ ∀ n, a n ≤ a (n + 1)) →
  (∀ n, a n = 0) ∨ (∀ n, a n = n) ∨ (∀ n, a n = 1 / 2) :=
sorry

end sequence_solution_l2090_209053


namespace arithmetic_sequence_condition_l2090_209014

theorem arithmetic_sequence_condition (a : ℕ → ℕ) 
(h1 : a 4 = 4) 
(h2 : a 3 + a 8 = 5) : 
a 7 = 1 := 
sorry

end arithmetic_sequence_condition_l2090_209014


namespace carla_book_count_l2090_209072

theorem carla_book_count (tiles_count books_count : ℕ) 
  (tiles_monday : tiles_count = 38)
  (total_tuesday_count : 2 * tiles_count + 3 * books_count = 301) : 
  books_count = 75 :=
by
  sorry

end carla_book_count_l2090_209072


namespace sara_golf_balls_total_l2090_209038

-- Define the conditions
def dozens := 16
def dozen_to_balls := 12

-- The final proof statement
theorem sara_golf_balls_total : dozens * dozen_to_balls = 192 :=
by
  sorry

end sara_golf_balls_total_l2090_209038


namespace subset_relationship_l2090_209000

def S : Set ℕ := {x | ∃ n : ℕ, x = 3^n}
def T : Set ℕ := {x | ∃ n : ℕ, x = 3 * n}

theorem subset_relationship : S ⊆ T :=
by sorry

end subset_relationship_l2090_209000


namespace linear_equation_in_two_variables_l2090_209029

/--
Prove that Equation C (3x - 1 = 2 - 5y) is a linear equation in two variables 
given the equations in conditions.
-/
theorem linear_equation_in_two_variables :
  ∀ (x y : ℝ),
  (2 * x + 3 = x - 5) →
  (x * y + y = 2) →
  (3 * x - 1 = 2 - 5 * y) →
  (2 * x + (3 / y) = 7) →
  ∃ (A B C : ℝ), A * x + B * y = C :=
by 
  sorry

end linear_equation_in_two_variables_l2090_209029


namespace rhombus_diagonal_sum_l2090_209035

theorem rhombus_diagonal_sum (e f : ℝ) (h1: e^2 + f^2 = 16) (h2: 0 < e ∧ 0 < f):
  e + f = 5 :=
by
  sorry

end rhombus_diagonal_sum_l2090_209035


namespace subproblem1_l2090_209098

theorem subproblem1 (a b c q : ℝ) (h1 : c = b * q) (h2 : c = a * q^2) : 
  (Real.sqrt 5 - 1) / 2 < q ∧ q < (Real.sqrt 5 + 1) / 2 := 
sorry

end subproblem1_l2090_209098


namespace binomial_coefficient_plus_ten_l2090_209077

theorem binomial_coefficient_plus_ten :
  Nat.choose 9 5 + 10 = 136 := 
by
  sorry

end binomial_coefficient_plus_ten_l2090_209077


namespace max_value_amc_am_mc_ca_l2090_209049

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end max_value_amc_am_mc_ca_l2090_209049


namespace no_prime_divisible_by_45_l2090_209060

-- Definitions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

-- Theorem
theorem no_prime_divisible_by_45 : ∀ n : ℕ, is_prime n → ¬divisible_by_45 n :=
by 
  intros n h_prime h_div
  -- Proof steps are omitted
  sorry

end no_prime_divisible_by_45_l2090_209060


namespace number_of_students_l2090_209074

-- Define the conditions as hypotheses
def ordered_apples : ℕ := 6 + 15   -- 21 apples ordered
def extra_apples : ℕ := 16         -- 16 extra apples after distribution

-- Define the main theorem statement to prove S = 21
theorem number_of_students (S : ℕ) (H1 : ordered_apples = 21) (H2 : extra_apples = 16) : S = 21 := 
by
  sorry

end number_of_students_l2090_209074


namespace partitioning_staircase_l2090_209054

def number_of_ways_to_partition_staircase (n : ℕ) : ℕ :=
  2^(n-1)

theorem partitioning_staircase (n : ℕ) : 
  number_of_ways_to_partition_staircase n = 2^(n-1) :=
by 
  sorry

end partitioning_staircase_l2090_209054


namespace sqrt_four_eq_pm_two_l2090_209058

theorem sqrt_four_eq_pm_two : ∃ y : ℝ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  sorry

end sqrt_four_eq_pm_two_l2090_209058


namespace f_monotonically_decreasing_iff_l2090_209030

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 4 * a * x + 3 else (2 - 3 * a) * x + 1

theorem f_monotonically_decreasing_iff (a : ℝ) : 
  (∀ x₁ x₂, x₁ ≤ x₂ → f a x₁ ≥ f a x₂) ↔ (1/2 ≤ a ∧ a < 2/3) :=
by 
  sorry

end f_monotonically_decreasing_iff_l2090_209030


namespace alpha_in_second_quadrant_l2090_209081

theorem alpha_in_second_quadrant (α : Real) 
  (h1 : Real.sin (2 * α) < 0) 
  (h2 : Real.cos α - Real.sin α < 0) : 
  π / 2 < α ∧ α < π :=
sorry

end alpha_in_second_quadrant_l2090_209081


namespace negation_correct_l2090_209091

-- Define the initial statement
def initial_statement (s : Set ℝ) : Prop :=
  ∀ x ∈ s, |x| ≥ 3

-- Define the negated statement
def negated_statement (s : Set ℝ) : Prop :=
  ∃ x ∈ s, |x| < 3

-- The theorem to be proven
theorem negation_correct (s : Set ℝ) :
  ¬(initial_statement s) ↔ negated_statement s := by
  sorry

end negation_correct_l2090_209091


namespace sculpture_height_l2090_209082

def base_height: ℝ := 10  -- height of the base in inches
def combined_height_feet: ℝ := 3.6666666666666665  -- combined height in feet
def inches_per_foot: ℝ := 12  -- conversion factor from feet to inches

-- Convert combined height to inches
def combined_height_inches: ℝ := combined_height_feet * inches_per_foot

-- Math proof problem statement
theorem sculpture_height : combined_height_inches - base_height = 34 := by
  sorry

end sculpture_height_l2090_209082


namespace inspection_time_l2090_209052

theorem inspection_time 
  (num_digits : ℕ) (num_letters : ℕ) 
  (letter_opts : ℕ) (start_digits : ℕ) 
  (inspection_time_three_hours : ℕ) 
  (probability : ℝ) 
  (num_vehicles : ℕ) 
  (vehicles_inspected : ℕ)
  (cond1 : num_digits = 4)
  (cond2 : num_letters = 2)
  (cond3 : letter_opts = 3)
  (cond4 : start_digits = 2)
  (cond5 : inspection_time_three_hours = 180) 
  (cond6 : probability = 0.02)
  (cond7 : num_vehicles = 900)
  (cond8 : vehicles_inspected = num_vehicles * probability) :
  vehicles_inspected = (inspection_time_three_hours / 10) :=
  sorry

end inspection_time_l2090_209052


namespace group_size_l2090_209047

def total_people (I N B Ne : ℕ) : ℕ := I + N - B + B + Ne

theorem group_size :
  let I := 55
  let N := 43
  let B := 61
  let Ne := 63
  total_people I N B Ne = 161 :=
by
  sorry

end group_size_l2090_209047


namespace total_savings_l2090_209020

-- Definitions and Conditions
def thomas_monthly_savings : ℕ := 40
def joseph_saving_ratio : ℚ := 3 / 5
def saving_period_months : ℕ := 72

-- Problem Statement
theorem total_savings :
  let thomas_total := thomas_monthly_savings * saving_period_months
  let joseph_monthly_savings := thomas_monthly_savings * joseph_saving_ratio
  let joseph_total := joseph_monthly_savings * saving_period_months
  thomas_total + joseph_total = 4608 := 
by
  sorry

end total_savings_l2090_209020


namespace number_of_dogs_l2090_209055

theorem number_of_dogs (D C B x : ℕ) (h1 : D = 3 * x) (h2 : B = 9 * x) (h3 : D + B = 204) (h4 : 12 * x = 204) : D = 51 :=
by
  -- Proof skipped
  sorry

end number_of_dogs_l2090_209055


namespace good_quadruple_inequality_l2090_209059

theorem good_quadruple_inequality {p a b c : ℕ} (hp : Nat.Prime p) (hodd : p % 2 = 1) 
(habc_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
(hab : (a * b + 1) % p = 0) (hbc : (b * c + 1) % p = 0) (hca : (c * a + 1) % p = 0) :
  p + 2 ≤ (a + b + c) / 3 := 
by
  sorry

end good_quadruple_inequality_l2090_209059


namespace arithmetic_sequence_sum_l2090_209066

-- Define the arithmetic sequence {a_n}
noncomputable def a_n (n : ℕ) : ℝ := sorry

-- Given condition
axiom h1 : a_n 3 + a_n 7 = 37

-- Proof statement
theorem arithmetic_sequence_sum : a_n 2 + a_n 4 + a_n 6 + a_n 8 = 74 :=
by
  sorry

end arithmetic_sequence_sum_l2090_209066


namespace quadratic_distinct_real_roots_l2090_209064

theorem quadratic_distinct_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - a = 0 → x^2 - 2*x - a = 0 ∧ (∀ y : ℝ, y ≠ x → y^2 - 2*y - a = 0)) → 
  a > -1 :=
by
  sorry

end quadratic_distinct_real_roots_l2090_209064


namespace first_pipe_fills_cistern_in_10_hours_l2090_209031

noncomputable def time_to_fill (x : ℝ) : Prop :=
  let first_pipe_rate := 1 / x
  let second_pipe_rate := 1 / 12
  let third_pipe_rate := 1 / 15
  let combined_rate := first_pipe_rate + second_pipe_rate - third_pipe_rate
  combined_rate = 7 / 60

theorem first_pipe_fills_cistern_in_10_hours : time_to_fill 10 :=
by
  sorry

end first_pipe_fills_cistern_in_10_hours_l2090_209031


namespace find_fixed_point_c_l2090_209070

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - 2
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := 2 * x ^ 2 - c

theorem find_fixed_point_c (c : ℝ) : 
  (∃ a : ℝ, f a = a ∧ g a c = a) ↔ (c = 3 ∨ c = 6) := sorry

end find_fixed_point_c_l2090_209070


namespace min_value_of_function_l2090_209048

theorem min_value_of_function (x : ℝ) (hx : x > 0) :
  (x + 1/x + x^2 + 1/x^2 + 1 / (x + 1/x + x^2 + 1/x^2)) = 4.25 := by
  sorry

end min_value_of_function_l2090_209048


namespace original_number_of_men_l2090_209062

-- Define the conditions
def work_days_by_men (M : ℕ) (days : ℕ) : ℕ := M * days
def additional_men (M : ℕ) : ℕ := M + 10
def completed_days : ℕ := 9

-- The main theorem
theorem original_number_of_men : ∀ (M : ℕ), 
  work_days_by_men M 12 = work_days_by_men (additional_men M) completed_days → 
  M = 30 :=
by
  intros M h
  sorry

end original_number_of_men_l2090_209062


namespace sum_of_x_coords_l2090_209065

theorem sum_of_x_coords (x : ℝ) (y : ℝ) :
  y = abs (x^2 - 6*x + 8) ∧ y = 6 - x → (x = (5 + Real.sqrt 17) / 2 ∨ x = (5 - Real.sqrt 17) / 2 ∨ x = 2)
  →  ((5 + Real.sqrt 17) / 2 + (5 - Real.sqrt 17) / 2 + 2 = 7) :=
by
  intros h1 h2
  have H : ((5 + Real.sqrt 17) / 2 + (5 - Real.sqrt 17) / 2 + 2 = 7) := sorry
  exact H

end sum_of_x_coords_l2090_209065


namespace area_of_blackboard_l2090_209006

def side_length : ℝ := 6
def area (side : ℝ) : ℝ := side * side

theorem area_of_blackboard : area side_length = 36 := by
  -- proof
  sorry

end area_of_blackboard_l2090_209006


namespace length_of_floor_y_l2090_209092

theorem length_of_floor_y
  (A B : ℝ)
  (hx : A = 10)
  (hy : B = 18)
  (width_y : ℝ)
  (length_y : ℝ)
  (width_y_eq : width_y = 9)
  (area_eq : A * B = width_y * length_y) :
  length_y = 20 := 
sorry

end length_of_floor_y_l2090_209092


namespace find_a_find_n_l2090_209015

noncomputable def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d
noncomputable def sum_of_first_n_terms (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2
noncomputable def S (a d n : ℕ) : ℕ := if n = 1 then a else sum_of_first_n_terms a d n
noncomputable def arithmetic_sum_property (a d n : ℕ) : Prop :=
  ∀ n ≥ 2, (S a d n) ^ 2 = 3 * n ^ 2 * arithmetic_sequence a d n + (S a d (n - 1)) ^ 2

theorem find_a (a : ℕ) (h1 : ∀ n ≥ 2, S a 3 n ^ 2 = 3 * n ^ 2 * arithmetic_sequence a 3 n + S a 3 (n - 1) ^ 2) :
  a = 3 :=
sorry

noncomputable def c (n : ℕ) (a5 : ℕ) : ℕ := 3 ^ (n - 1) + a5
noncomputable def sum_of_first_n_terms_c (n a5 : ℕ) : ℕ := (3^n - 1) / 2 + 15 * n
noncomputable def T (n a5 : ℕ) : ℕ := sum_of_first_n_terms_c n a5

theorem find_n (a : ℕ) (a5 : ℕ) (h1 : ∀ n ≥ 2, S a 3 n ^ 2 = 3 * n ^ 2 * arithmetic_sequence a 3 n + S a 3 (n - 1) ^ 2)
  (h2 : a = 3) (h3 : a5 = 15) :
  ∃ n : ℕ, 4 * T n a5 > S a 3 10 ∧ n = 3 :=
sorry

end find_a_find_n_l2090_209015


namespace order_of_values_l2090_209097

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem order_of_values (h_even : is_even f) (h_incr : is_increasing_on_nonneg f) : f (-π) > f 3 ∧ f 3 > f (-2) :=
by
  -- Proof would go here
  sorry

end order_of_values_l2090_209097


namespace minimum_protein_content_is_at_least_1_8_l2090_209039

-- Define the net weight of the can and the minimum protein percentage
def netWeight : ℝ := 300
def minProteinPercentage : ℝ := 0.006

-- Prove that the minimum protein content is at least 1.8 grams
theorem minimum_protein_content_is_at_least_1_8 :
  netWeight * minProteinPercentage ≥ 1.8 := 
by
  sorry

end minimum_protein_content_is_at_least_1_8_l2090_209039


namespace ratio_dvds_to_cds_l2090_209028

def total_sold : ℕ := 273
def dvds_sold : ℕ := 168
def cds_sold : ℕ := total_sold - dvds_sold

theorem ratio_dvds_to_cds : (dvds_sold : ℚ) / cds_sold = 8 / 5 := by
  sorry

end ratio_dvds_to_cds_l2090_209028


namespace sqrt_meaningful_l2090_209017

theorem sqrt_meaningful (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 :=
sorry

end sqrt_meaningful_l2090_209017


namespace min_distance_from_point_to_line_l2090_209075

theorem min_distance_from_point_to_line : 
  ∀ (x₀ y₀ : Real), 3 * x₀ - 4 * y₀ - 10 = 0 → Real.sqrt (x₀^2 + y₀^2) = 2 :=
by sorry

end min_distance_from_point_to_line_l2090_209075


namespace total_leaves_l2090_209088

def fernTypeA_fronds := 15
def fernTypeA_leaves_per_frond := 45
def fernTypeB_fronds := 20
def fernTypeB_leaves_per_frond := 30
def fernTypeC_fronds := 25
def fernTypeC_leaves_per_frond := 40

def fernTypeA_count := 4
def fernTypeB_count := 5
def fernTypeC_count := 3

theorem total_leaves : 
  fernTypeA_count * (fernTypeA_fronds * fernTypeA_leaves_per_frond) + 
  fernTypeB_count * (fernTypeB_fronds * fernTypeB_leaves_per_frond) + 
  fernTypeC_count * (fernTypeC_fronds * fernTypeC_leaves_per_frond) = 
  8700 := 
sorry

end total_leaves_l2090_209088


namespace greatest_integer_b_not_in_range_l2090_209007

theorem greatest_integer_b_not_in_range :
  let f (x : ℝ) (b : ℝ) := x^2 + b*x + 20
  let g (x : ℝ) (b : ℝ) := x^2 + b*x + 24
  (¬ (∃ (x : ℝ), g x b = 0)) → (b = 9) :=
by
  sorry

end greatest_integer_b_not_in_range_l2090_209007


namespace piglet_gifted_balloons_l2090_209041

noncomputable def piglet_balloons_gifted (piglet_balloons : ℕ) : ℕ :=
  let winnie_balloons := 3 * piglet_balloons
  let owl_balloons := 4 * piglet_balloons
  let total_balloons := piglet_balloons + winnie_balloons + owl_balloons
  let burst_balloons := total_balloons - 60
  piglet_balloons - burst_balloons / 8

-- Prove that Piglet gifted 4 balloons given the conditions
theorem piglet_gifted_balloons :
  ∃ (piglet_balloons : ℕ), piglet_balloons = 8 ∧ piglet_balloons_gifted piglet_balloons = 4 := sorry

end piglet_gifted_balloons_l2090_209041


namespace relationship_between_fractions_l2090_209085

variable (a a' b b' : ℝ)
variable (h₁ : a > 0)
variable (h₂ : a' > 0)
variable (h₃ : (-(b / (2 * a)))^2 > (-(b' / (2 * a')))^2)

theorem relationship_between_fractions
  (a : ℝ) (a' : ℝ) (b : ℝ) (b' : ℝ)
  (h1 : a > 0) (h2 : a' > 0)
  (h3 : (-(b / (2 * a)))^2 > (-(b' / (2 * a')))^2) :
  (b^2) / (a^2) > (b'^2) / (a'^2) :=
by sorry

end relationship_between_fractions_l2090_209085


namespace total_chrome_parts_l2090_209042

theorem total_chrome_parts (a b : ℕ) 
  (h1 : a + b = 21) 
  (h2 : 3 * a + 2 * b = 50) : 2 * a + 4 * b = 68 := 
sorry

end total_chrome_parts_l2090_209042


namespace complement_intersection_l2090_209023

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {3, 4, 5}) (hN : N = {2, 3}) :
  (U \ N) ∩ M = {4, 5} := by
  sorry

end complement_intersection_l2090_209023


namespace proof_of_problem_l2090_209040

-- Define the problem conditions using a combination function
def problem_statement : Prop :=
  (Nat.choose 6 3 = 20)

theorem proof_of_problem : problem_statement :=
by
  sorry

end proof_of_problem_l2090_209040


namespace fish_estimation_l2090_209093

noncomputable def number_caught := 50
noncomputable def number_marked_caught := 2
noncomputable def number_released := 30

theorem fish_estimation (N : ℕ) (h1 : number_caught = 50) 
  (h2 : number_marked_caught = 2) 
  (h3 : number_released = 30) :
  (number_marked_caught : ℚ) / number_caught = number_released / N → 
  N = 750 :=
by
  sorry

end fish_estimation_l2090_209093


namespace original_volume_of_ice_l2090_209019

theorem original_volume_of_ice (V : ℝ) 
  (h1 : V * (1/4) * (1/4) = 0.4) : 
  V = 6.4 :=
sorry

end original_volume_of_ice_l2090_209019


namespace find_first_purchase_find_max_profit_purchase_plan_l2090_209083

-- Defining the parameters for the problem
structure KeychainParams where
  purchase_price_A : ℕ
  purchase_price_B : ℕ
  total_purchase_cost_first : ℕ
  total_keychains_first : ℕ
  total_purchase_cost_second : ℕ
  total_keychains_second : ℕ
  purchase_cap_second : ℕ
  selling_price_A : ℕ
  selling_price_B : ℕ

-- Define the initial setup
def params : KeychainParams := {
  purchase_price_A := 30,
  purchase_price_B := 25,
  total_purchase_cost_first := 850,
  total_keychains_first := 30,
  total_purchase_cost_second := 2200,
  total_keychains_second := 80,
  purchase_cap_second := 2200,
  selling_price_A := 45,
  selling_price_B := 37
}

-- Part 1: Prove the number of keychains purchased for each type
theorem find_first_purchase (x y : ℕ)
  (h₁ : x + y = params.total_keychains_first)
  (h₂ : params.purchase_price_A * x + params.purchase_price_B * y = params.total_purchase_cost_first) :
  x = 20 ∧ y = 10 :=
sorry

-- Part 2: Prove the purchase plan that maximizes the sales profit
theorem find_max_profit_purchase_plan (m : ℕ)
  (h₃ : m + (params.total_keychains_second - m) = params.total_keychains_second)
  (h₄ : params.purchase_price_A * m + params.purchase_price_B * (params.total_keychains_second - m) ≤ params.purchase_cap_second) :
  m = 40 ∧ (params.selling_price_A - params.purchase_price_A) * m + (params.selling_price_B - params.purchase_price_B) * (params.total_keychains_second - m) = 1080 :=
sorry

end find_first_purchase_find_max_profit_purchase_plan_l2090_209083


namespace find_q_l2090_209002

-- Defining the polynomial and conditions
def Q (x : ℝ) (p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

variable (p q r : ℝ)

-- Given conditions
def mean_of_zeros_eq_prod_of_zeros (p q r : ℝ) : Prop :=
  -p / 3 = r

def prod_of_zeros_eq_sum_of_coeffs (p q r : ℝ) : Prop :=
  r = 1 + p + q + r

def y_intercept_eq_three (r : ℝ) : Prop :=
  r = 3

-- Final proof statement asserting q = 5
theorem find_q (p q r : ℝ) (h1 : mean_of_zeros_eq_prod_of_zeros p q r)
  (h2 : prod_of_zeros_eq_sum_of_coeffs p q r)
  (h3 : y_intercept_eq_three r) :
  q = 5 :=
sorry

end find_q_l2090_209002


namespace hex_conversion_sum_l2090_209034

-- Convert hexadecimal E78 to decimal
def hex_to_decimal (h : String) : Nat :=
  match h with
  | "E78" => 3704
  | _ => 0

-- Convert decimal to radix 7
def decimal_to_radix7 (d : Nat) : String :=
  match d with
  | 3704 => "13541"
  | _ => ""

-- Convert radix 7 to decimal
def radix7_to_decimal (r : String) : Nat :=
  match r with
  | "13541" => 3704
  | _ => 0

-- Convert decimal to hexadecimal
def decimal_to_hex (d : Nat) : String :=
  match d with
  | 3704 => "E78"
  | 7408 => "1CF0"
  | _ => ""

theorem hex_conversion_sum :
  let initial_hex : String := "E78"
  let final_decimal := 3704 
  let final_hex := decimal_to_hex (final_decimal)
  let final_sum := hex_to_decimal initial_hex + final_decimal
  (decimal_to_hex final_sum) = "1CF0" :=
by
  sorry

end hex_conversion_sum_l2090_209034


namespace percentage_of_cash_is_20_l2090_209096

theorem percentage_of_cash_is_20
  (raw_materials : ℕ)
  (machinery : ℕ)
  (total_amount : ℕ)
  (h_raw_materials : raw_materials = 35000)
  (h_machinery : machinery = 40000)
  (h_total_amount : total_amount = 93750) :
  (total_amount - (raw_materials + machinery)) * 100 / total_amount = 20 :=
by
  sorry

end percentage_of_cash_is_20_l2090_209096


namespace paper_length_l2090_209037

theorem paper_length :
  ∃ (L : ℝ), (2 * (11 * L) = 2 * (8.5 * 11) + 100 ∧ L = 287 / 22) :=
sorry

end paper_length_l2090_209037


namespace correct_geometry_problems_l2090_209010

-- Let A_c be the number of correct algebra problems.
-- Let A_i be the number of incorrect algebra problems.
-- Let G_c be the number of correct geometry problems.
-- Let G_i be the number of incorrect geometry problems.

def algebra_correct_incorrect_ratio (A_c A_i : ℕ) : Prop :=
  A_c * 2 = A_i * 3

def geometry_correct_incorrect_ratio (G_c G_i : ℕ) : Prop :=
  G_c * 1 = G_i * 4

def total_algebra_problems (A_c A_i : ℕ) : Prop :=
  A_c + A_i = 25

def total_geometry_problems (G_c G_i : ℕ) : Prop :=
  G_c + G_i = 35

def total_problems (A_c A_i G_c G_i : ℕ) : Prop :=
  A_c + A_i + G_c + G_i = 60

theorem correct_geometry_problems (A_c A_i G_c G_i : ℕ) :
  algebra_correct_incorrect_ratio A_c A_i →
  geometry_correct_incorrect_ratio G_c G_i →
  total_algebra_problems A_c A_i →
  total_geometry_problems G_c G_i →
  total_problems A_c A_i G_c G_i →
  G_c = 28 :=
sorry

end correct_geometry_problems_l2090_209010


namespace ratio_of_perimeter_to_b_l2090_209089

theorem ratio_of_perimeter_to_b (b : ℝ) (hb : b ≠ 0) :
  let p1 := (-2*b, -2*b)
  let p2 := (2*b, -2*b)
  let p3 := (2*b, 2*b)
  let p4 := (-2*b, 2*b)
  let l := (y = b * x)
  let d1 := 4*b
  let d2 := 4*b
  let d3 := 4*b
  let d4 := 4*b*Real.sqrt 2
  let perimeter := d1 + d2 + d3 + d4
  let ratio := perimeter / b
  ratio = 12 + 4 * Real.sqrt 2 := by
  -- Placeholder for proof
  sorry

end ratio_of_perimeter_to_b_l2090_209089


namespace carrot_lettuce_ratio_l2090_209044

theorem carrot_lettuce_ratio :
  let lettuce_cal := 50
  let dressing_cal := 210
  let crust_cal := 600
  let pepperoni_cal := crust_cal / 3
  let cheese_cal := 400
  let total_pizza_cal := crust_cal + pepperoni_cal + cheese_cal
  let carrot_cal := C
  let total_salad_cal := lettuce_cal + carrot_cal + dressing_cal
  let jackson_salad_cal := (1 / 4) * total_salad_cal
  let jackson_pizza_cal := (1 / 5) * total_pizza_cal
  jackson_salad_cal + jackson_pizza_cal = 330 →
  carrot_cal / lettuce_cal = 2 :=
by
  intro lettuce_cal dressing_cal crust_cal pepperoni_cal cheese_cal total_pizza_cal carrot_cal total_salad_cal jackson_salad_cal jackson_pizza_cal h
  sorry

end carrot_lettuce_ratio_l2090_209044


namespace a_and_b_together_work_days_l2090_209061

-- Definitions for the conditions:
def a_work_rate : ℚ := 1 / 9
def b_work_rate : ℚ := 1 / 18

-- The theorem statement:
theorem a_and_b_together_work_days : (a_work_rate + b_work_rate)⁻¹ = 6 := by
  sorry

end a_and_b_together_work_days_l2090_209061


namespace MrWillamTaxPercentage_l2090_209050

-- Definitions
def TotalTaxCollected : ℝ := 3840
def MrWillamTax : ℝ := 480

-- Theorem Statement
theorem MrWillamTaxPercentage :
  (MrWillamTax / TotalTaxCollected) * 100 = 12.5 :=
by
  sorry

end MrWillamTaxPercentage_l2090_209050


namespace problem_sum_of_k_l2090_209067

theorem problem_sum_of_k {a b c k : ℂ} (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_ratio : a / (1 - b) = k ∧ b / (1 - c) = k ∧ c / (1 - a) = k) :
  (if (k^2 - k + 1 = 0) then -(-1)/1 else 0) = 1 :=
sorry

end problem_sum_of_k_l2090_209067


namespace card_draw_prob_l2090_209021

/-- Define the total number of cards in the deck -/
def total_cards : ℕ := 52

/-- Define the total number of diamonds or aces -/
def diamonds_and_aces : ℕ := 16

/-- Define the probability of drawing a card that is a diamond or an ace in one draw -/
def prob_diamond_or_ace : ℚ := diamonds_and_aces / total_cards

/-- Define the complementary probability of not drawing a diamond nor ace in one draw -/
def prob_not_diamond_or_ace : ℚ := (total_cards - diamonds_and_aces) / total_cards

/-- Define the probability of not drawing a diamond nor ace in three draws with replacement -/
def prob_not_diamond_or_ace_three_draws : ℚ := prob_not_diamond_or_ace ^ 3

/-- Define the probability of drawing at least one diamond or ace in three draws with replacement -/
def prob_at_least_one_diamond_or_ace_in_three_draws : ℚ := 1 - prob_not_diamond_or_ace_three_draws

/-- The final probability calculated -/
def final_prob : ℚ := 1468 / 2197

theorem card_draw_prob :
  prob_at_least_one_diamond_or_ace_in_three_draws = final_prob := by
  sorry

end card_draw_prob_l2090_209021


namespace molecular_weight_calculation_l2090_209087

theorem molecular_weight_calculation
    (moles_total_mw : ℕ → ℝ)
    (hw : moles_total_mw 9 = 900) :
    moles_total_mw 1 = 100 :=
by
  sorry

end molecular_weight_calculation_l2090_209087


namespace initial_cats_in_shelter_l2090_209069

theorem initial_cats_in_shelter
  (cats_found_monday : ℕ)
  (cats_found_tuesday : ℕ)
  (cats_adopted_wednesday : ℕ)
  (current_cats : ℕ)
  (total_adopted_cats : ℕ)
  (initial_cats : ℕ) :
  cats_found_monday = 2 →
  cats_found_tuesday = 1 →
  cats_adopted_wednesday = 3 →
  total_adopted_cats = cats_adopted_wednesday * 2 →
  current_cats = 17 →
  initial_cats = current_cats + total_adopted_cats - (cats_found_monday + cats_found_tuesday) →
  initial_cats = 20 :=
by
  intros
  sorry

end initial_cats_in_shelter_l2090_209069


namespace proof_1_proof_2_l2090_209043

noncomputable def problem_1 (x : ℝ) : Prop :=
  (3 * x - 2) / (x - 1) > 1 → x > 1

noncomputable def problem_2 (x a : ℝ) : Prop :=
  if a = 0 then False
  else if a > 0 then -a < x ∧ x < 2 * a
  else if a < 0 then 2 * a < x ∧ x < -a
  else False

-- Sorry to skip the proofs
theorem proof_1 (x : ℝ) (h : problem_1 x) : x > 1 :=
  sorry

theorem proof_2 (x a : ℝ) (h : x * x - a * x - 2 * a * a < 0) : problem_2 x a :=
  sorry

end proof_1_proof_2_l2090_209043


namespace unique_triple_solution_l2090_209012

theorem unique_triple_solution (a b c : ℝ) 
  (h1 : a * (b ^ 2 + c) = c * (c + a * b))
  (h2 : b * (c ^ 2 + a) = a * (a + b * c))
  (h3 : c * (a ^ 2 + b) = b * (b + c * a)) : 
  a = b ∧ b = c := 
sorry

end unique_triple_solution_l2090_209012


namespace find_x_l2090_209063

noncomputable def vector_a : ℝ × ℝ × ℝ := (-3, 2, 5)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -1)

theorem find_x (x : ℝ) (h : (-3) * 1 + 2 * x + 5 * (-1) = 2) : x = 5 :=
by 
  sorry

end find_x_l2090_209063


namespace correct_calculation_l2090_209024

theorem correct_calculation (a b : ℝ) :
  ¬(a^2 + 2 * a^2 = 3 * a^4) ∧
  ¬(a^6 / a^3 = a^2) ∧
  ¬((a^2)^3 = a^5) ∧
  (ab)^2 = a^2 * b^2 := by
  sorry

end correct_calculation_l2090_209024


namespace area_ratio_of_similar_isosceles_triangles_l2090_209011

theorem area_ratio_of_similar_isosceles_triangles
  (b1 b2 h1 h2 : ℝ)
  (h_ratio : h1 / h2 = 2 / 3)
  (similar_tri : b1 / b2 = 2 / 3) :
  (1 / 2 * b1 * h1) / (1 / 2 * b2 * h2) = 4 / 9 :=
by
  sorry

end area_ratio_of_similar_isosceles_triangles_l2090_209011


namespace coprime_squares_l2090_209068

theorem coprime_squares (a b : ℕ) (h1 : Nat.gcd a b = 1) (h2 : ∃ k : ℕ, ab = k^2) : 
  ∃ p q : ℕ, a = p^2 ∧ b = q^2 :=
by
  sorry

end coprime_squares_l2090_209068


namespace smallest_n_satisfying_congruence_l2090_209001

theorem smallest_n_satisfying_congruence :
  ∃ (n : ℕ), n > 0 ∧ (∀ m > 0, m < n → (7^m % 5) ≠ (m^7 % 5)) ∧ (7^n % 5) = (n^7 % 5) := 
by sorry

end smallest_n_satisfying_congruence_l2090_209001


namespace who_stole_the_broth_l2090_209095

-- Define the suspects
inductive Suspect
| MarchHare : Suspect
| MadHatter : Suspect
| Dormouse : Suspect

open Suspect

-- Define the statements
def stole_broth (s : Suspect) : Prop :=
  s = Dormouse

def told_truth (s : Suspect) : Prop :=
  s = Dormouse

-- The March Hare's testimony
def march_hare_testimony : Prop :=
  stole_broth MadHatter

-- Conditions
def condition1 : Prop := ∃! s, stole_broth s
def condition2 : Prop := ∀ s, told_truth s ↔ stole_broth s
def condition3 : Prop := told_truth MarchHare → stole_broth MadHatter

-- Combining conditions into a single proposition to prove
theorem who_stole_the_broth : 
  (condition1 ∧ condition2 ∧ condition3) → stole_broth Dormouse := sorry

end who_stole_the_broth_l2090_209095


namespace necessary_but_not_sufficient_l2090_209036

def M : Set ℝ := {x | -2 < x ∧ x < 3}
def P : Set ℝ := {x | x ≤ -1}

theorem necessary_but_not_sufficient :
  (∀ x, x ∈ M ∩ P → x ∈ M ∪ P) ∧ (∃ x, x ∈ M ∪ P ∧ x ∉ M ∩ P) :=
by
  sorry

end necessary_but_not_sufficient_l2090_209036


namespace first_year_after_2020_with_sum_4_l2090_209022

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000 / 100) + (n % 100 / 10) + (n % 10)

def is_year (y : ℕ) : Prop :=
  y > 2020 ∧ sum_of_digits y = 4

theorem first_year_after_2020_with_sum_4 : ∃ y, is_year y ∧ ∀ z, is_year z → z ≥ y :=
by sorry

end first_year_after_2020_with_sum_4_l2090_209022


namespace initial_milk_in_container_A_l2090_209025

theorem initial_milk_in_container_A (A B C D : ℝ) 
  (h1 : B = A - 0.625 * A) 
  (h2 : C - 158 = B) 
  (h3 : D = 0.45 * (C - 58)) 
  (h4 : D = 58) 
  : A = 231 := 
sorry

end initial_milk_in_container_A_l2090_209025


namespace star_property_l2090_209027

-- Define the operation a ⋆ b = (a - b) ^ 3
def star (a b : ℝ) : ℝ := (a - b) ^ 3

-- State the theorem
theorem star_property (x y : ℝ) : star ((x - y) ^ 3) ((y - x) ^ 3) = 8 * (x - y) ^ 9 := 
by 
  sorry

end star_property_l2090_209027


namespace no_solution_1221_l2090_209003

def equation_correctness (n : ℤ) : Prop :=
  -n^3 + 555^3 = n^2 - n * 555 + 555^2

-- Prove that the prescribed value 1221 does not satisfy the modified equation by contradiction
theorem no_solution_1221 : ¬ ∃ n : ℤ, equation_correctness n ∧ n = 1221 := by
  sorry

end no_solution_1221_l2090_209003


namespace calc_pow_l2090_209099

-- Definitions used in the conditions
def base := 2
def exp := 10
def power := 2 / 5

-- Given condition
def given_identity : Pow.pow base exp = 1024 := by sorry

-- Statement to be proved
theorem calc_pow : Pow.pow 1024 power = 16 := by
  -- Use the given identity and known exponentiation rules to derive the result
  sorry

end calc_pow_l2090_209099


namespace find_number_l2090_209084

theorem find_number (x : ℝ) (h : 0.6667 * x + 0.75 = 1.6667) : x = 1.375 :=
sorry

end find_number_l2090_209084


namespace hockey_team_ties_l2090_209079

theorem hockey_team_ties (W T : ℕ) (h1 : 2 * W + T = 60) (h2 : W = T + 12) : T = 12 :=
by
  sorry

end hockey_team_ties_l2090_209079


namespace solve_equation_l2090_209076

theorem solve_equation (x y z : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (hz : 0 ≤ z ∧ z ≤ 9) (h_eq : 1 / (x + y + z) = (x * 100 + y * 10 + z) / 1000) :
  x = 1 ∧ y = 2 ∧ z = 5 :=
by
  sorry

end solve_equation_l2090_209076


namespace simplify_to_x5_l2090_209080

theorem simplify_to_x5 (x : ℝ) :
  x^2 * x^3 = x^5 :=
by {
  -- proof goes here
  sorry
}

end simplify_to_x5_l2090_209080


namespace find_lcm_of_two_numbers_l2090_209094

theorem find_lcm_of_two_numbers (A B : ℕ) (hcf : ℕ) (prod : ℕ) 
  (h1 : hcf = 22) (h2 : prod = 62216) (h3 : A * B = prod) (h4 : Nat.gcd A B = hcf) :
  Nat.lcm A B = 2828 := 
by
  sorry

end find_lcm_of_two_numbers_l2090_209094


namespace money_left_correct_l2090_209005

def initial_amount : ℕ := 158
def cost_shoes : ℕ := 45
def cost_bag : ℕ := cost_shoes - 17
def cost_lunch : ℕ := cost_bag / 4
def total_spent : ℕ := cost_shoes + cost_bag + cost_lunch
def amount_left : ℕ := initial_amount - total_spent

theorem money_left_correct :
  amount_left = 78 := by
  sorry

end money_left_correct_l2090_209005


namespace jay_used_zero_fraction_of_gallon_of_paint_l2090_209056

theorem jay_used_zero_fraction_of_gallon_of_paint
    (dexter_used : ℝ := 3/8)
    (gallon_in_liters : ℝ := 4)
    (paint_left_liters : ℝ := 4) :
    dexter_used = 3/8 ∧ gallon_in_liters = 4 ∧ paint_left_liters = 4 →
    ∃ jay_used : ℝ, jay_used = 0 :=
by
  sorry

end jay_used_zero_fraction_of_gallon_of_paint_l2090_209056


namespace sector_angle_measure_l2090_209071

theorem sector_angle_measure
  (r l : ℝ)
  (h1 : 2 * r + l = 4)
  (h2 : (1 / 2) * l * r = 1) :
  l / r = 2 :=
sorry

end sector_angle_measure_l2090_209071


namespace arithmetic_mean_l2090_209051

theorem arithmetic_mean (x b : ℝ) (h : x ≠ 0) : 
  (1 / 2) * ((2 + (b / x)) + (2 - (b / x))) = 2 :=
by sorry

end arithmetic_mean_l2090_209051


namespace find_equation_of_line_l2090_209046

variable (x y : ℝ)

def line_parallel (x y : ℝ) (m : ℝ) :=
  x - 2*y + m = 0

def line_through_point (x y : ℝ) (px py : ℝ) (m : ℝ) :=
  (px - 2 * py + m = 0)
  
theorem find_equation_of_line :
  let px := -1
  let py := 3
  ∃ m, line_parallel x y m ∧ line_through_point x y px py m ∧ m = 7 :=
by
  sorry

end find_equation_of_line_l2090_209046


namespace first_digit_base_5_of_2197_l2090_209008

theorem first_digit_base_5_of_2197 : 
  ∃ k : ℕ, 2197 = k * 625 + r ∧ k = 3 ∧ r < 625 :=
by
  -- existence of k and r follows from the division algorithm
  -- sorry is used to indicate the part of the proof that needs to be filled in
  sorry

end first_digit_base_5_of_2197_l2090_209008


namespace interior_diagonals_sum_l2090_209013

theorem interior_diagonals_sum (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + c * a) = 112)
  (h2 : 4 * (a + b + c) = 60) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 113 := 
by 
  sorry

end interior_diagonals_sum_l2090_209013


namespace find_k_l2090_209032

theorem find_k 
  (x k : ℚ)
  (h1 : (x^2 - 3*k)*(x + 3*k) = x^3 + 3*k*(x^2 - x - 7))
  (h2 : k ≠ 0) : k = 7 / 3 := 
sorry

end find_k_l2090_209032


namespace circle_radius_l2090_209033

theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 - 2 * x + 6 * y + 1 = 0) → (∃ (r : ℝ), r = 3) :=
by
  sorry

end circle_radius_l2090_209033


namespace find_w_over_y_l2090_209045

theorem find_w_over_y 
  (w x y : ℝ) 
  (h1 : w / x = 2 / 3) 
  (h2 : (x + y) / y = 1.6) : 
  w / y = 0.4 := 
  sorry

end find_w_over_y_l2090_209045


namespace angle_A_is_60_degrees_value_of_b_plus_c_l2090_209090

noncomputable def triangleABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  let area := (3 * Real.sqrt 3) / 2
  c + 2 * a * Real.cos C = 2 * b ∧
  1/2 * b * c * Real.sin A = area 

theorem angle_A_is_60_degrees (A B C : ℝ) (a b c : ℝ) :
  triangleABC A B C a b c →
  Real.cos A = 1 / 2 → 
  A = 60 :=
by
  intros h1 h2 
  sorry

theorem value_of_b_plus_c (A B C : ℝ) (b c : ℝ) :
  triangleABC A B C (Real.sqrt 7) b c →
  b * c = 6 →
  (b + c) = 5 :=
by 
  intros h1 h2 
  sorry

end angle_A_is_60_degrees_value_of_b_plus_c_l2090_209090


namespace total_weight_lifted_l2090_209086

-- Definitions based on conditions
def original_lift : ℝ := 80
def after_training : ℝ := original_lift * 2
def specialization_increment : ℝ := after_training * 0.10
def specialized_lift : ℝ := after_training + specialization_increment

-- Statement of the theorem to prove total weight lifted
theorem total_weight_lifted : 
  (specialized_lift * 2) = 352 :=
sorry

end total_weight_lifted_l2090_209086


namespace geometric_sequence_condition_l2090_209078

theorem geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < a 1) 
  (h2 : ∀ n, a (n + 1) = a n * q) :
  (a 1 < a 3) ↔ (a 1 < a 3) ∧ (a 3 < a 6) :=
sorry

end geometric_sequence_condition_l2090_209078


namespace partial_fractions_sum_zero_l2090_209018

noncomputable def sum_of_coefficients (A B C D E : ℝ) : Prop :=
  (A + B + C + D + E = 0)

theorem partial_fractions_sum_zero :
  ∀ (A B C D E : ℝ),
    (∀ x : ℝ, 1 = A*(x+1)*(x+2)*(x+3)*(x+5) + B*x*(x+2)*(x+3)*(x+5) + 
              C*x*(x+1)*(x+3)*(x+5) + D*x*(x+1)*(x+2)*(x+5) + 
              E*x*(x+1)*(x+2)*(x+3)) →
    sum_of_coefficients A B C D E :=
by sorry

end partial_fractions_sum_zero_l2090_209018


namespace range_of_a_l2090_209026

open Function

def f (x : ℝ) : ℝ := -2 * x^5 - x^3 - 7 * x + 2

theorem range_of_a (a : ℝ) : f (a^2) + f (a - 2) > 4 → -2 < a ∧ a < 1 := 
by
  sorry

end range_of_a_l2090_209026


namespace find_number_l2090_209009

theorem find_number
  (x : ℝ)
  (h : (7.5 * 7.5) + 37.5 + (x * x) = 100) :
  x = 2.5 :=
sorry

end find_number_l2090_209009


namespace paired_products_not_equal_1000_paired_products_equal_10000_l2090_209057

open Nat

theorem paired_products_not_equal_1000 :
  ∀ (a : Fin 1000 → ℤ), (∃ p n : Nat, p + n = 1000 ∧
    p * (p - 1) / 2 + n * (n - 1) / 2 = 2 * p * n) → False :=
by 
  sorry

theorem paired_products_equal_10000 :
  ∀ (a : Fin 10000 → ℤ), (∃ p n : Nat, p + n = 10000 ∧
    p * (p - 1) / 2 + n * (n - 1) / 2 = 2 * p * n) ↔ p = 5050 ∨ p = 4950 :=
by 
  sorry

end paired_products_not_equal_1000_paired_products_equal_10000_l2090_209057


namespace janet_total_distance_l2090_209073

-- Define the distances covered in each week for each activity
def week1_running := 8 * 5
def week1_cycling := 7 * 3

def week2_running := 10 * 4
def week2_swimming := 2 * 2

def week3_running := 6 * 5
def week3_hiking := 3 * 2

-- Total distances for each activity
def total_running := week1_running + week2_running + week3_running
def total_cycling := week1_cycling
def total_swimming := week2_swimming
def total_hiking := week3_hiking

-- Total distance covered
def total_distance := total_running + total_cycling + total_swimming + total_hiking

-- Prove that the total distance is 141 miles
theorem janet_total_distance : total_distance = 141 := by
  sorry

end janet_total_distance_l2090_209073


namespace largest_integer_m_dividing_30_factorial_l2090_209016

theorem largest_integer_m_dividing_30_factorial :
  ∃ (m : ℕ), (∀ (k : ℕ), (18^k ∣ Nat.factorial 30) ↔ k ≤ m) ∧ m = 7 := by
  sorry

end largest_integer_m_dividing_30_factorial_l2090_209016


namespace find_number_l2090_209004

theorem find_number (x : ℕ) (h : x + 1015 = 3016) : x = 2001 :=
sorry

end find_number_l2090_209004
