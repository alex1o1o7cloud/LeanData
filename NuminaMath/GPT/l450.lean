import Mathlib

namespace value_of_y_l450_45007

theorem value_of_y (y: ℚ) (h: (2 / 5 - 1 / 7) = 14 / y): y = 490 / 9 :=
by
  sorry

end value_of_y_l450_45007


namespace distance_covered_downstream_l450_45070

-- Conditions
def boat_speed_still_water : ℝ := 16
def stream_rate : ℝ := 5
def time_downstream : ℝ := 6

-- Effective speed downstream
def effective_speed_downstream := boat_speed_still_water + stream_rate

-- Distance covered downstream
def distance_downstream := effective_speed_downstream * time_downstream

-- Theorem to prove
theorem distance_covered_downstream :
  (distance_downstream = 126) :=
by
  sorry

end distance_covered_downstream_l450_45070


namespace container_ratio_l450_45092

theorem container_ratio (A B : ℝ) (h : (4 / 5) * A = (2 / 3) * B) : (A / B) = (5 / 6) :=
by
  sorry

end container_ratio_l450_45092


namespace remainder_division_123456789012_by_112_l450_45046

-- Define the conditions
def M : ℕ := 123456789012
def m7 : ℕ := M % 7
def m16 : ℕ := M % 16

-- State the proof problem
theorem remainder_division_123456789012_by_112 : M % 112 = 76 :=
by
  -- Conditions
  have h1 : m7 = 3 := by sorry
  have h2 : m16 = 12 := by sorry
  -- Conclusion
  sorry

end remainder_division_123456789012_by_112_l450_45046


namespace inequality_holds_l450_45042

theorem inequality_holds (x : ℝ) (m : ℝ) :
  (∀ x : ℝ, (x^2 - m * x - 2) / (x^2 - 3 * x + 4) > -1) ↔ (-7 < m ∧ m < 1) :=
by
  sorry

end inequality_holds_l450_45042


namespace part1_part2_l450_45060

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.exp (2 * x) - a * Real.exp x - x * Real.exp x

theorem part1 :
  (∀ x : ℝ, f a x ≥ 0) → a = 1 := sorry

theorem part2 (h : a = 1) :
  ∃ x₀ : ℝ, (∀ x : ℝ, f a x ≤ f a x₀) ∧
    (Real.log 2 / (2 * Real.exp 1) + 1 / (4 * Real.exp (2 * 1)) ≤ f a x₀ ∧
    f a x₀ < 1 / 4) := sorry

end part1_part2_l450_45060


namespace negation_of_existence_l450_45083

theorem negation_of_existence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by
  sorry

end negation_of_existence_l450_45083


namespace distinct_integer_values_b_for_quadratic_l450_45089

theorem distinct_integer_values_b_for_quadratic :
  ∃ (S : Finset ℤ), ∀ b ∈ S, (∃ x y : ℤ, x^2 + b * x + 12 * b = 0 ∧ y^2 + b * y + 12 * b = 0) ∧ S.card = 16 :=
sorry

end distinct_integer_values_b_for_quadratic_l450_45089


namespace stanley_sold_4_cups_per_hour_l450_45078

theorem stanley_sold_4_cups_per_hour (S : ℕ) (Carl_Hour : ℕ) :
  (Carl_Hour = 7) →
  21 = (Carl_Hour * 3) →
  (21 - 9) = (S * 3) →
  S = 4 :=
by
  intros Carl_Hour_eq Carl_3hours Stanley_eq
  sorry

end stanley_sold_4_cups_per_hour_l450_45078


namespace initial_card_count_l450_45017

theorem initial_card_count (r b : ℕ) (h₁ : (r : ℝ)/(r + b) = 1/4)
    (h₂ : (r : ℝ)/(r + (b + 6)) = 1/6) : r + b = 12 :=
by
  sorry

end initial_card_count_l450_45017


namespace solve_x_l450_45054

theorem solve_x : ∃ x : ℝ, 65 + (5 * x) / (180 / 3) = 66 ∧ x = 12 := by
  sorry

end solve_x_l450_45054


namespace range_of_a_l450_45057

noncomputable def inequality_condition (x : ℝ) (a : ℝ) : Prop :=
  a - 2 * x - |Real.log x| ≤ 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → inequality_condition x a) ↔ a ≤ 1 + Real.log 2 :=
sorry

end range_of_a_l450_45057


namespace distance_between_stations_l450_45099

/-- Two trains start at the same time from two stations and proceed towards each other. 
    The first train travels at 20 km/hr and the second train travels at 25 km/hr. 
    When they meet, the second train has traveled 60 km more than the first train. -/
theorem distance_between_stations
    (t : ℝ) -- The time in hours when they meet
    (x : ℝ) -- The distance traveled by the slower train
    (d1 d2 : ℝ) -- Distances traveled by the two trains respectively
    (h1 : 20 * t = x)
    (h2 : 25 * t = x + 60) :
  d1 + d2 = 540 :=
by
  sorry

end distance_between_stations_l450_45099


namespace evaluate_x2_y2_l450_45087

theorem evaluate_x2_y2 (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 18) : x^2 - y^2 = -72 := 
sorry

end evaluate_x2_y2_l450_45087


namespace num_prime_divisors_50_fact_l450_45073
open Nat -- To simplify working with natural numbers

-- We define the prime numbers less than or equal to 50.
def primes_le_50 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

-- The problem statement: Prove that the number of prime divisors of 50! which are less than or equal to 50 is 15.
theorem num_prime_divisors_50_fact : (primes_le_50.length = 15) :=
by 
  -- Here we use sorry to skip the proof.
  sorry

end num_prime_divisors_50_fact_l450_45073


namespace find_d_l450_45010

-- Definitions of the functions f and g and condition on f(g(x))
def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d x : ℝ) (h : f (g x c) c = 15 * x + d) : d = 18 :=
sorry

end find_d_l450_45010


namespace molecular_weight_of_7_moles_of_NH4_2SO4_l450_45090

theorem molecular_weight_of_7_moles_of_NH4_2SO4 :
  let N_weight := 14.01
  let H_weight := 1.01
  let S_weight := 32.07
  let O_weight := 16.00
  let N_atoms := 2
  let H_atoms := 8
  let S_atoms := 1
  let O_atoms := 4
  let moles := 7
  let molecular_weight := (N_weight * N_atoms) + (H_weight * H_atoms) + (S_weight * S_atoms) + (O_weight * O_atoms)
  let total_weight := molecular_weight * moles
  total_weight = 924.19 :=
by
  sorry

end molecular_weight_of_7_moles_of_NH4_2SO4_l450_45090


namespace samantha_original_cans_l450_45008

theorem samantha_original_cans : 
  ∀ (cans_per_classroom : ℚ),
  (cans_per_classroom = (50 - 38) / 5) →
  (50 / cans_per_classroom) = 21 := 
by
  sorry

end samantha_original_cans_l450_45008


namespace smallest_product_l450_45068

theorem smallest_product (S : Set ℤ) (hS : S = { -8, -3, -2, 2, 4 }) :
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a * b = -32 ∧ ∀ (x y : ℤ), x ∈ S → y ∈ S → x * y ≥ -32 :=
by
  sorry

end smallest_product_l450_45068


namespace original_number_is_correct_l450_45069

noncomputable def original_number : ℝ :=
  let x := 11.26666666666667
  let y := 30.333333333333332
  x + y

theorem original_number_is_correct (x y : ℝ) (h₁ : 10 * x + 22 * y = 780) (h₂ : y = 30.333333333333332) : 
  original_number = 41.6 :=
by
  sorry

end original_number_is_correct_l450_45069


namespace frankie_candies_l450_45045

theorem frankie_candies (M D F : ℕ) (h1 : M = 92) (h2 : D = 18) (h3 : F = M - D) : F = 74 :=
by
  sorry

end frankie_candies_l450_45045


namespace zoo_camels_l450_45064

theorem zoo_camels (x y : ℕ) (h1 : x - y = 10) (h2 : x + 2 * y = 55) : x + y = 40 :=
by sorry

end zoo_camels_l450_45064


namespace extra_chairs_added_l450_45029

theorem extra_chairs_added (rows cols total_chairs extra_chairs : ℕ) 
  (h1 : rows = 7) 
  (h2 : cols = 12) 
  (h3 : total_chairs = 95) 
  (h4 : extra_chairs = total_chairs - rows * cols) : 
  extra_chairs = 11 := by 
  sorry

end extra_chairs_added_l450_45029


namespace equal_number_of_frogs_after_6_months_l450_45033

theorem equal_number_of_frogs_after_6_months :
  ∃ n : ℕ, 
    n = 6 ∧ 
    (∀ Dn Qn : ℕ, 
      (Dn = 5^(n + 1) ∧ Qn = 3^(n + 5)) → 
      Dn = Qn) :=
by
  sorry

end equal_number_of_frogs_after_6_months_l450_45033


namespace find_a_b_l450_45022

theorem find_a_b (a b x y : ℝ) (h1 : x = 2) (h2 : y = 4) (h3 : a * x + b * y = 16) (h4 : b * x - a * y = -12) : a = 4 ∧ b = 2 := by
  sorry

end find_a_b_l450_45022


namespace percent_value_in_quarters_l450_45048

def nickel_value : ℕ := 5
def quarter_value : ℕ := 25
def num_nickels : ℕ := 80
def num_quarters : ℕ := 40

def value_in_nickels : ℕ := num_nickels * nickel_value
def value_in_quarters : ℕ := num_quarters * quarter_value
def total_value : ℕ := value_in_nickels + value_in_quarters

theorem percent_value_in_quarters :
  (value_in_quarters : ℚ) / total_value = 5 / 7 :=
by
  sorry

end percent_value_in_quarters_l450_45048


namespace decryption_ease_comparison_l450_45043

def unique_letters_of_thermometer : Finset Char := {'т', 'е', 'р', 'м', 'о'}
def unique_letters_of_remont : Finset Char := {'р', 'е', 'м', 'о', 'н', 'т'}
def easier_to_decrypt : Prop :=
  unique_letters_of_remont.card > unique_letters_of_thermometer.card

theorem decryption_ease_comparison : easier_to_decrypt :=
by
  -- We need to prove that |unique_letters_of_remont| > |unique_letters_of_thermometer|
  sorry

end decryption_ease_comparison_l450_45043


namespace arithmetic_progression_K_l450_45020

theorem arithmetic_progression_K (K : ℕ) : 
  (∃ n : ℕ, K = 30 * n - 1) ↔ (K^K + 1) % 30 = 0 :=
sorry

end arithmetic_progression_K_l450_45020


namespace find_A_n_find_d1_d2_zero_l450_45016

-- Defining the arithmetic sequences {a_n} and {b_n} with common differences d1 and d2 respectively
variables (a b : ℕ → ℤ)
variables (d1 d2 : ℤ)

-- Conditions on the sequences
axiom a_n_arith : ∀ n, a (n + 1) = a n + d1
axiom b_n_arith : ∀ n, b (n + 1) = b n + d2

-- Definitions of A_n and B_n
def A_n (n : ℕ) : ℤ := a n + b n
def B_n (n : ℕ) : ℤ := a n * b n

-- Given initial conditions
axiom A_1 : A_n a b 1 = 1
axiom A_2 : A_n a b 2 = 3

-- Prove that A_n = 2n - 1
theorem find_A_n : ∀ n, A_n a b n = 2 * n - 1 :=
by sorry

-- Condition that B_n is an arithmetic sequence
axiom B_n_arith : ∀ n, B_n a b (n + 1) - B_n a b n = B_n a b 1 - B_n a b 0

-- Prove that d1 * d2 = 0
theorem find_d1_d2_zero : d1 * d2 = 0 :=
by sorry

end find_A_n_find_d1_d2_zero_l450_45016


namespace part1_part2_l450_45027

section
variables (x a m n : ℝ)
-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x - 3)

-- a) Prove the solution of the inequality f(x) >= 4 + |x-3| - |x-1| given a=3.
theorem part1 (h_a : a = 3) :
  {x | f x a ≥ 4 + abs (x - 3) - abs (x - 1)} = {x | x ≤ 0} ∪ {x | x ≥ 4} :=
sorry

-- b) Prove that m + 2n >= 2 given f(x) <= 1 + |x-3| with solution set [1, 3] and 1/m + 1/(2n) = a
theorem part2 (h_sol : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 1 + abs (x - 3)) 
  (h_a : 1 / m + 1 / (2 * n) = 2) (h_m_pos : m > 0) (h_n_pos : n > 0) :
  m + 2 * n ≥ 2 :=
sorry
end

end part1_part2_l450_45027


namespace average_salary_company_l450_45019

-- Define the conditions
def num_managers : Nat := 15
def num_associates : Nat := 75
def avg_salary_managers : ℤ := 90000
def avg_salary_associates : ℤ := 30000

-- Define the goal to prove
theorem average_salary_company : 
  (num_managers * avg_salary_managers + num_associates * avg_salary_associates) / (num_managers + num_associates) = 40000 := by
  sorry

end average_salary_company_l450_45019


namespace zoe_total_songs_l450_45013

def initial_songs : ℕ := 15
def deleted_songs : ℕ := 8
def added_songs : ℕ := 50

theorem zoe_total_songs : initial_songs - deleted_songs + added_songs = 57 := by
  sorry

end zoe_total_songs_l450_45013


namespace area_of_triangle_bounded_by_coordinate_axes_and_line_l450_45050

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

theorem area_of_triangle_bounded_by_coordinate_axes_and_line :
  area_of_triangle 4 6 = 12 :=
by
  sorry

end area_of_triangle_bounded_by_coordinate_axes_and_line_l450_45050


namespace weighted_average_fish_caught_l450_45003

-- Define the daily catches for each person
def AangCatches := [5, 7, 9]
def SokkaCatches := [8, 5, 6]
def TophCatches := [10, 12, 8]
def ZukoCatches := [6, 7, 10]

-- Define the group catches
def GroupCatches := AangCatches ++ SokkaCatches ++ TophCatches ++ ZukoCatches

-- Calculate the total number of fish caught by the group
def TotalFishCaught := List.sum GroupCatches

-- Calculate the total number of days fished by the group
def TotalDaysFished := 4 * 3

-- Calculate the weighted average
def WeightedAverage := TotalFishCaught.toFloat / TotalDaysFished.toFloat

-- Proof statement
theorem weighted_average_fish_caught :
  WeightedAverage = 7.75 := by
  sorry

end weighted_average_fish_caught_l450_45003


namespace initial_price_of_sugar_per_kg_l450_45065

theorem initial_price_of_sugar_per_kg
  (initial_price : ℝ)
  (final_price : ℝ)
  (required_reduction : ℝ)
  (initial_price_eq : initial_price = 6)
  (final_price_eq : final_price = 7.5)
  (required_reduction_eq : required_reduction = 0.19999999999999996) :
  initial_price = 6 :=
by
  sorry

end initial_price_of_sugar_per_kg_l450_45065


namespace carpenter_job_duration_l450_45000

theorem carpenter_job_duration
  (total_estimate : ℤ)
  (carpenter_hourly_rate : ℤ)
  (assistant_hourly_rate : ℤ)
  (material_cost : ℤ)
  (H1 : total_estimate = 1500)
  (H2 : carpenter_hourly_rate = 35)
  (H3 : assistant_hourly_rate = 25)
  (H4 : material_cost = 720) :
  (total_estimate - material_cost) / (carpenter_hourly_rate + assistant_hourly_rate) = 13 :=
by
  sorry

end carpenter_job_duration_l450_45000


namespace power_function_value_l450_45051

theorem power_function_value {α : ℝ} (h : 3^α = Real.sqrt 3) : (9 : ℝ)^α = 3 :=
by sorry

end power_function_value_l450_45051


namespace sum_remainder_zero_l450_45021

theorem sum_remainder_zero
  (a b c : ℕ)
  (h₁ : a % 53 = 31)
  (h₂ : b % 53 = 15)
  (h₃ : c % 53 = 7) :
  (a + b + c) % 53 = 0 :=
by
  sorry

end sum_remainder_zero_l450_45021


namespace r_s_t_u_bounds_l450_45038

theorem r_s_t_u_bounds (r s t u : ℝ) 
  (H1: 5 * r + 4 * s + 3 * t + 6 * u = 100)
  (H2: r ≥ s)
  (H3: s ≥ t)
  (H4: t ≥ u)
  (H5: u ≥ 0) :
  20 ≤ r + s + t + u ∧ r + s + t + u ≤ 25 := 
sorry

end r_s_t_u_bounds_l450_45038


namespace sum_of_solutions_of_fx_eq_0_l450_45009

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then 7 * x + 10 else 3 * x - 15

theorem sum_of_solutions_of_fx_eq_0 :
  let x1 := -10 / 7
  let x2 := 5
  f x1 = 0 ∧ f x2 = 0 ∧ x1 ≤ 1 ∧ x2 > 1 → x1 + x2 = 25 / 7 :=
by
  sorry

end sum_of_solutions_of_fx_eq_0_l450_45009


namespace not_possible_to_create_3_piles_l450_45032

theorem not_possible_to_create_3_piles (similar: ℝ → ℝ → Prop) (sqrt_2 : ℝ)
  (hsimilar : ∀ x y, similar x y ↔ x ≤ sqrt_2 * y ∧ y ≤ sqrt_2 * x) :
  ∀ x, ¬ ∃ x1 x2 x3, 
    x = x1 + x2 + x3 ∧ 
    similar x1 x2 ∧ 
    similar x2 x3 ∧ 
    similar x1 x3 :=
by { sorry }

end not_possible_to_create_3_piles_l450_45032


namespace root_equation_m_l450_45075

theorem root_equation_m (m : ℝ) : 
  (∃ (x : ℝ), x = -1 ∧ m*x^2 + x - m^2 + 1 = 0) → m = 1 :=
by 
  sorry

end root_equation_m_l450_45075


namespace fortieth_sequence_number_l450_45058

theorem fortieth_sequence_number :
  (∃ r n : ℕ, ((r * (r + 1)) - 40 = n) ∧ (40 ≤ r * (r + 1)) ∧ (40 > (r - 1) * r) ∧ n = 2 * r) :=
sorry

end fortieth_sequence_number_l450_45058


namespace sum_of_transformed_numbers_l450_45084

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) : 3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_transformed_numbers_l450_45084


namespace floor_ineq_l450_45062

theorem floor_ineq (α β : ℝ) : ⌊2 * α⌋ + ⌊2 * β⌋ ≥ ⌊α⌋ + ⌊β⌋ + ⌊α + β⌋ :=
sorry

end floor_ineq_l450_45062


namespace intersection_of_sets_l450_45047

def setA : Set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def setB : Set ℝ := { x : ℝ | 2 < x }

theorem intersection_of_sets : setA ∩ setB = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_of_sets_l450_45047


namespace sum_of_first_six_terms_of_geometric_series_l450_45036

-- Definitions for the conditions
def a : ℚ := 1 / 4
def r : ℚ := 1 / 4
def n : ℕ := 6

-- Define the formula for the sum of the first n terms of a geometric series
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- The equivalent Lean 4 statement
theorem sum_of_first_six_terms_of_geometric_series :
  geometric_series_sum a r n = 4095 / 12288 :=
by
  sorry

end sum_of_first_six_terms_of_geometric_series_l450_45036


namespace option_d_always_holds_l450_45031

theorem option_d_always_holds (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b := by
  sorry

end option_d_always_holds_l450_45031


namespace black_white_ratio_extended_pattern_l450_45088

theorem black_white_ratio_extended_pattern
  (original_black : ℕ) (original_white : ℕ) (added_black : ℕ)
  (h1 : original_black = 10)
  (h2 : original_white = 26)
  (h3 : added_black = 20) :
  (original_black + added_black) / original_white = 30 / 26 :=
by sorry

end black_white_ratio_extended_pattern_l450_45088


namespace janets_total_pockets_l450_45015

-- Define the total number of dresses
def totalDresses : ℕ := 36

-- Define the dresses with pockets
def dressesWithPockets : ℕ := totalDresses / 2

-- Define the dresses without pockets
def dressesWithoutPockets : ℕ := totalDresses - dressesWithPockets

-- Define the dresses with one hidden pocket
def dressesWithOneHiddenPocket : ℕ := (40 * dressesWithoutPockets) / 100

-- Define the dresses with 2 pockets
def dressesWithTwoPockets : ℕ := dressesWithPockets / 3

-- Define the dresses with 3 pockets
def dressesWithThreePockets : ℕ := dressesWithPockets / 4

-- Define the dresses with 4 pockets
def dressesWithFourPockets : ℕ := dressesWithPockets - dressesWithTwoPockets - dressesWithThreePockets

-- Calculate the total number of pockets
def totalPockets : ℕ := 
  2 * dressesWithTwoPockets + 
  3 * dressesWithThreePockets + 
  4 * dressesWithFourPockets + 
  dressesWithOneHiddenPocket

-- The theorem to prove the total number of pockets
theorem janets_total_pockets : totalPockets = 63 :=
  by
    -- Proof is omitted, use 'sorry'
    sorry

end janets_total_pockets_l450_45015


namespace octal_subtraction_l450_45098

theorem octal_subtraction : (53 - 27 : ℕ) = 24 :=
by sorry

end octal_subtraction_l450_45098


namespace expr1_correct_expr2_correct_expr3_correct_l450_45086

-- Define the expressions and corresponding correct answers
def expr1 : Int := 58 + 15 * 4
def expr2 : Int := 216 - 72 / 8
def expr3 : Int := (358 - 295) / 7

-- State the proof goals
theorem expr1_correct : expr1 = 118 := by
  sorry

theorem expr2_correct : expr2 = 207 := by
  sorry

theorem expr3_correct : expr3 = 9 := by
  sorry

end expr1_correct_expr2_correct_expr3_correct_l450_45086


namespace CandyGivenToJanetEmily_l450_45035

noncomputable def initial_candy : ℝ := 78.5
noncomputable def candy_left_after_janet : ℝ := 68.75
noncomputable def candy_given_to_emily : ℝ := 2.25

theorem CandyGivenToJanetEmily :
  initial_candy - candy_left_after_janet + candy_given_to_emily = 12 := 
by
  sorry

end CandyGivenToJanetEmily_l450_45035


namespace limit_a_n_l450_45059

open Nat Real

noncomputable def a_n (n : ℕ) : ℝ := (7 * n - 1) / (n + 1)

theorem limit_a_n : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - 7| < ε := 
by {
  -- The proof would go here.
  sorry
}

end limit_a_n_l450_45059


namespace side_length_of_square_l450_45026

theorem side_length_of_square (length_rect width_rect : ℝ) (h_length : length_rect = 7) (h_width : width_rect = 5) :
  (∃ side_length : ℝ, 4 * side_length = 2 * (length_rect + width_rect) ∧ side_length = 6) :=
by
  use 6
  simp [h_length, h_width]
  sorry

end side_length_of_square_l450_45026


namespace num_8tuples_satisfying_condition_l450_45094

theorem num_8tuples_satisfying_condition :
  (∃! (y : Fin 8 → ℝ),
    (2 - y 0)^2 + (y 0 - y 1)^2 + (y 1 - y 2)^2 + 
    (y 2 - y 3)^2 + (y 3 - y 4)^2 + (y 4 - y 5)^2 + 
    (y 5 - y 6)^2 + (y 6 - y 7)^2 + y 7^2 = 4 / 9) :=
sorry

end num_8tuples_satisfying_condition_l450_45094


namespace compare_xyz_l450_45093

noncomputable def x := (0.5 : ℝ)^(0.5 : ℝ)
noncomputable def y := (0.5 : ℝ)^(1.3 : ℝ)
noncomputable def z := (1.3 : ℝ)^(0.5 : ℝ)

theorem compare_xyz : z > x ∧ x > y := by
  sorry

end compare_xyz_l450_45093


namespace roots_of_quadratic_l450_45025

theorem roots_of_quadratic (a b : ℝ) (h : a ≠ 0) (h1 : a + b = 0) :
  ∀ x, (a * x^2 + b * x = 0) → (x = 0 ∨ x = 1) := 
by
  sorry

end roots_of_quadratic_l450_45025


namespace lunch_cost_total_l450_45006

theorem lunch_cost_total (x y : ℝ) (h1 : y = 45) (h2 : x = (2 / 3) * y) : 
  x + y + y = 120 := by
  sorry

end lunch_cost_total_l450_45006


namespace perpendicular_distance_H_to_plane_EFG_l450_45012

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def E : Point3D := ⟨5, 0, 0⟩
def F : Point3D := ⟨0, 3, 0⟩
def G : Point3D := ⟨0, 0, 4⟩
def H : Point3D := ⟨0, 0, 0⟩

def distancePointToPlane (H E F G : Point3D) : ℝ := sorry

theorem perpendicular_distance_H_to_plane_EFG :
  distancePointToPlane H E F G = 1.8 := sorry

end perpendicular_distance_H_to_plane_EFG_l450_45012


namespace mean_difference_is_882_l450_45049

variable (S : ℤ) (N : ℤ) (S_N_correct : N = 1000)

def actual_mean (S : ℤ) (N : ℤ) : ℚ :=
  (S + 98000) / N

def incorrect_mean (S : ℤ) (N : ℤ) : ℚ :=
  (S + 980000) / N

theorem mean_difference_is_882 
  (S : ℤ) 
  (N : ℤ) 
  (S_N_correct : N = 1000) 
  (S_in_range : 8200 ≤ S) 
  (S_actual : S + 98000 ≤ 980000) :
  incorrect_mean S N - actual_mean S N = 882 := 
by
  /- Proof steps would go here -/
  sorry

end mean_difference_is_882_l450_45049


namespace jason_cost_l450_45005

variable (full_page_cost_per_square_inch : ℝ := 6.50)
variable (half_page_cost_per_square_inch : ℝ := 8)
variable (quarter_page_cost_per_square_inch : ℝ := 10)

variable (full_page_area : ℝ := 9 * 12)
variable (half_page_area : ℝ := full_page_area / 2)
variable (quarter_page_area : ℝ := full_page_area / 4)

variable (half_page_ads : ℝ := 1)
variable (quarter_page_ads : ℝ := 4)

variable (total_ads : ℝ := half_page_ads + quarter_page_ads)
variable (bulk_discount : ℝ := if total_ads >= 4 then 0.10 else 0.0)

variable (half_page_cost : ℝ := half_page_area * half_page_cost_per_square_inch)
variable (quarter_page_cost : ℝ := quarter_page_ads * (quarter_page_area * quarter_page_cost_per_square_inch))

variable (total_cost_before_discount : ℝ := half_page_cost + quarter_page_cost)
variable (discount_amount : ℝ := total_cost_before_discount * bulk_discount)
variable (final_cost : ℝ := total_cost_before_discount - discount_amount)

theorem jason_cost :
  final_cost = 1360.80 := by
  sorry

end jason_cost_l450_45005


namespace triangle_sides_inequality_l450_45028

theorem triangle_sides_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) :
    (a/(b + c - a) + b/(c + a - b) + c/(a + b - c)) ≥ ((b + c - a)/a + (c + a - b)/b + (a + b - c)/c) ∧
    ((b + c - a)/a + (c + a - b)/b + (a + b - c)/c) ≥ 3 :=
by
  sorry

end triangle_sides_inequality_l450_45028


namespace total_ingredients_l450_45077

theorem total_ingredients (water : ℕ) (flour : ℕ) (salt : ℕ)
  (h_water : water = 10)
  (h_flour : flour = 16)
  (h_salt : salt = flour / 2) :
  water + flour + salt = 34 :=
by
  sorry

end total_ingredients_l450_45077


namespace pizza_remained_l450_45072

noncomputable def number_of_people := 15
noncomputable def fraction_eating_pizza := 3 / 5
noncomputable def total_pizza_pieces := 50
noncomputable def pieces_per_person := 4
noncomputable def pizza_remaining := total_pizza_pieces - (pieces_per_person * (fraction_eating_pizza * number_of_people))

theorem pizza_remained :
  pizza_remaining = 14 :=
by {
  sorry
}

end pizza_remained_l450_45072


namespace number_properties_l450_45061

-- Define what it means for a digit to be in a specific place
def digit_at_place (n place : ℕ) (d : ℕ) : Prop := 
  (n / 10 ^ place) % 10 = d

-- The given number
def specific_number : ℕ := 670154500

-- Conditions: specific number has specific digit in defined places
theorem number_properties : (digit_at_place specific_number 7 7) ∧ (digit_at_place specific_number 2 5) :=
by
  -- Proof of the theorem
  sorry

end number_properties_l450_45061


namespace men_build_fountain_l450_45066

theorem men_build_fountain (m1 m2 : ℕ) (l1 l2 d1 d2 : ℕ) (work_rate : ℚ)
  (h1 : m1 * d1 = l1 * work_rate)
  (h2 : work_rate = 56 / (20 * 7))
  (h3 : l1 = 56)
  (h4 : l2 = 42)
  (h5 : m1 = 20)
  (h6 : m2 = 35)
  (h7 : d1 = 7)
  : d2 = 3 :=
sorry

end men_build_fountain_l450_45066


namespace calculation_correct_l450_45095

theorem calculation_correct : 
  ((2 * (15^2 + 35^2 + 21^2) - (3^4 + 5^4 + 7^4)) / (3 + 5 + 7)) = 45 := by
  sorry

end calculation_correct_l450_45095


namespace strawb_eaten_by_friends_l450_45004

theorem strawb_eaten_by_friends (initial_strawberries remaining_strawberries eaten_strawberries : ℕ) : 
  initial_strawberries = 35 → 
  remaining_strawberries = 33 → 
  eaten_strawberries = initial_strawberries - remaining_strawberries → 
  eaten_strawberries = 2 := 
by 
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end strawb_eaten_by_friends_l450_45004


namespace tape_pieces_needed_l450_45096

-- Define the setup: cube edge length and tape width
def edge_length (n : ℕ) : ℕ := n
def tape_width : ℕ := 1

-- Define the statement we want to prove
theorem tape_pieces_needed (n : ℕ) (h₁ : edge_length n > 0) : 2 * n = 2 * (edge_length n) :=
  by
  sorry

end tape_pieces_needed_l450_45096


namespace remy_sold_110_bottles_l450_45080

theorem remy_sold_110_bottles 
    (price_per_bottle : ℝ)
    (total_evening_sales : ℝ)
    (evening_more_than_morning : ℝ)
    (nick_fewer_than_remy : ℝ)
    (R : ℝ) 
    (total_morning_sales_is : ℝ) :
    price_per_bottle = 0.5 →
    total_evening_sales = 55 →
    evening_more_than_morning = 3 →
    nick_fewer_than_remy = 6 →
    total_morning_sales_is = total_evening_sales - evening_more_than_morning →
    (R * price_per_bottle) + ((R - nick_fewer_than_remy) * price_per_bottle) = total_morning_sales_is →
    R = 110 :=
by
  intros
  sorry

end remy_sold_110_bottles_l450_45080


namespace gcd_lcm_of_consecutive_naturals_l450_45067

theorem gcd_lcm_of_consecutive_naturals (m : ℕ) (h : m > 0) (n : ℕ) (hn : n = m + 1) :
  gcd m n = 1 ∧ lcm m n = m * n :=
by
  sorry

end gcd_lcm_of_consecutive_naturals_l450_45067


namespace ratio_q_p_l450_45002

variable (p q : ℝ)
variable (hpq_pos : 0 < p ∧ 0 < q)
variable (hlog : Real.log p / Real.log 8 = Real.log q / Real.log 12 ∧ Real.log q / Real.log 12 = Real.log (p - q) / Real.log 18)

theorem ratio_q_p (p q : ℝ) (hpq_pos : 0 < p ∧ 0 < q) 
    (hlog : Real.log p / Real.log 8 = Real.log q / Real.log 12 ∧ Real.log q / Real.log 12 = Real.log (p - q) / Real.log 18) :
    q / p = (Real.sqrt 5 - 1) / 2 :=
  sorry

end ratio_q_p_l450_45002


namespace proposition_and_implication_l450_45014

theorem proposition_and_implication
  (m : ℝ)
  (h1 : 5/4 * (m^2 + m) > 0)
  (h2 : 1 + 9 - 4 * (5/4 * (m^2 + m)) > 0)
  (h3 : m + 3/2 ≥ 0)
  (h4 : m - 1/2 ≤ 0) :
  (-3/2 ≤ m ∧ m < -1) ∨ (0 < m ∧ m ≤ 1/2) :=
sorry

end proposition_and_implication_l450_45014


namespace find_B_share_l450_45063

-- Definitions for the conditions
def proportion (a b c d : ℕ) := 6 * a = 3 * b ∧ 3 * b = 5 * c ∧ 5 * c = 4 * d

def condition (c d : ℕ) := c = d + 1000

-- Statement of the problem
theorem find_B_share (A B C D : ℕ) (x : ℕ) 
  (h1 : proportion (6*x) (3*x) (5*x) (4*x)) 
  (h2 : condition (5*x) (4*x)) : 
  B = 3000 :=
by 
  sorry

end find_B_share_l450_45063


namespace oranges_weigh_4_ounces_each_l450_45030

def apple_weight : ℕ := 4
def max_bag_capacity : ℕ := 49
def num_bags : ℕ := 3
def total_weight : ℕ := num_bags * max_bag_capacity
def total_apple_weight : ℕ := 84
def num_apples : ℕ := total_apple_weight / apple_weight
def num_oranges : ℕ := num_apples
def total_orange_weight : ℕ := total_apple_weight
def weight_per_orange : ℕ := total_orange_weight / num_oranges

theorem oranges_weigh_4_ounces_each :
  weight_per_orange = 4 := by
  sorry

end oranges_weigh_4_ounces_each_l450_45030


namespace increasing_function_range_l450_45071

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (h₃ : ∀ x y : ℝ, x < y → f a x < f a y) : 
  3 / 2 ≤ a ∧ a < 2 := by
  sorry

end increasing_function_range_l450_45071


namespace parabola_directrix_l450_45091

theorem parabola_directrix (p : ℝ) :
  (∀ y x : ℝ, y^2 = 2 * p * x ↔ x = -1 → p = 2) :=
by
  sorry

end parabola_directrix_l450_45091


namespace gh_of_2_l450_45056

def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := 4 * x^3 + 1

theorem gh_of_2 :
  g (h 2) = 3269 :=
by
  sorry

end gh_of_2_l450_45056


namespace karen_locks_l450_45074

theorem karen_locks : 
  let L1 := 5
  let L2 := 3 * L1 - 3
  let Lboth := 5 * L2
  Lboth = 60 :=
by
  let L1 := 5
  let L2 := 3 * L1 - 3
  let Lboth := 5 * L2
  sorry

end karen_locks_l450_45074


namespace find_varphi_l450_45023

theorem find_varphi 
  (f g : ℝ → ℝ) 
  (x1 x2 varphi : ℝ) 
  (h_f : ∀ x, f x = 2 * Real.cos (2 * x)) 
  (h_g : ∀ x, g x = 2 * Real.cos (2 * x - 2 * varphi)) 
  (h_varphi_range : 0 < varphi ∧ varphi < π / 2) 
  (h_diff_cos : |f x1 - g x2| = 4) 
  (h_min_dist : |x1 - x2| = π / 6) 
: varphi = π / 3 := 
sorry

end find_varphi_l450_45023


namespace one_third_of_1206_is_100_5_percent_of_400_l450_45082

theorem one_third_of_1206_is_100_5_percent_of_400 (n m : ℕ) (f : ℝ) :
  n = 1206 → m = 400 → f = 1 / 3 → (n * f) / m * 100 = 100.5 :=
by
  intros h_n h_m h_f
  rw [h_n, h_m, h_f]
  sorry

end one_third_of_1206_is_100_5_percent_of_400_l450_45082


namespace sin_cos_alpha_frac_l450_45076

theorem sin_cos_alpha_frac (α : ℝ) (h : Real.tan (Real.pi - α) = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 3 := 
by
  sorry

end sin_cos_alpha_frac_l450_45076


namespace asymptote_of_hyperbola_l450_45055

theorem asymptote_of_hyperbola :
  ∀ x y : ℝ,
  (x^2 / 4 - y^2 = 1) → y = x / 2 ∨ y = - x / 2 :=
sorry

end asymptote_of_hyperbola_l450_45055


namespace find_a_m_range_c_l450_45053

noncomputable def f (x a : ℝ) := x^2 - 2*x + 2*a
def solution_set (f : ℝ → ℝ) (m : ℝ) := {x : ℝ | -2 ≤ x ∧ x ≤ m ∧ f x ≤ 0}

theorem find_a_m (a m : ℝ) : 
  (∀ x, f x a ≤ 0 ↔ -2 ≤ x ∧ x ≤ m) → a = -4 ∧ m = 4 := by
  sorry

theorem range_c (c : ℝ) : 
  (∀ x, (c - 4) * x^2 + 2 * (c - 4) * x - 1 < 0) → 13 / 4 < c ∧ c < 4 := by
  sorry

end find_a_m_range_c_l450_45053


namespace hyperbola_eccentricity_l450_45040

-- Definitions of conditions
def asymptotes_of_hyperbola (a b x y : ℝ) (h_a : a > 0) (h_b : b > 0) : Prop :=
  (b * x + a * y = 0) ∨ (b * x - a * y = 0)

def circle_tangent_to_asymptotes (x y a b : ℝ) : Prop :=
  ∀ x1 y1 : ℝ, 
  (x1, y1) = (0, 4) → 
  (Real.sqrt (b^2 + a^2) = 2 * a)

-- Main statement
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (h_asymptotes : ∀ (x y : ℝ), asymptotes_of_hyperbola a b x y h_a h_b) 
  (h_tangent : circle_tangent_to_asymptotes 0 4 a b) : 
  ∃ e : ℝ, e = 2 := 
sorry

end hyperbola_eccentricity_l450_45040


namespace garbage_collection_l450_45041

theorem garbage_collection (Daliah Dewei Zane : ℝ) 
(h1 : Daliah = 17.5)
(h2 : Dewei = Daliah - 2)
(h3 : Zane = 4 * Dewei) :
Zane = 62 :=
sorry

end garbage_collection_l450_45041


namespace alice_prank_combinations_l450_45081

theorem alice_prank_combinations : 
  let monday_choices := 1
  let tuesday_choices := 3
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 60 :=
by
  let monday_choices := 1
  let tuesday_choices := 3
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  exact (show 1 * 3 * 5 * 4 * 1 = 60 from sorry)

end alice_prank_combinations_l450_45081


namespace value_of_y_l450_45044

theorem value_of_y 
  (x y : ℤ) 
  (h1 : x - y = 10) 
  (h2 : x + y = 8) 
  : y = -1 := by
  sorry

end value_of_y_l450_45044


namespace sandy_correct_sums_l450_45097

theorem sandy_correct_sums (x y : ℕ) (h1 : x + y = 30) (h2 : 3 * x - 2 * y = 50) : x = 22 :=
  by
  sorry

end sandy_correct_sums_l450_45097


namespace age_problem_l450_45039

-- Definitions from conditions
variables (p q : ℕ) -- ages of p and q as natural numbers
variables (Y : ℕ) -- number of years ago p was half the age of q

-- Main statement
theorem age_problem :
  (p + q = 28) ∧ (p / q = 3 / 4) ∧ (p - Y = (q - Y) / 2) → Y = 8 :=
by
  sorry

end age_problem_l450_45039


namespace solve_triples_l450_45079

theorem solve_triples (a b c n : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : 0 < n) :
  a^2 + b^2 = n * Nat.lcm a b + n^2 ∧
  b^2 + c^2 = n * Nat.lcm b c + n^2 ∧
  c^2 + a^2 = n * Nat.lcm c a + n^2 →
  ∃ k : ℕ, 0 < k ∧ a = k ∧ b = k ∧ c = k :=
by
  intros h
  sorry

end solve_triples_l450_45079


namespace xyz_value_l450_45037

theorem xyz_value (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 20 / 3 :=
by
  sorry

end xyz_value_l450_45037


namespace mrs_jackson_boxes_l450_45034

theorem mrs_jackson_boxes (decorations_per_box used_decorations given_decorations : ℤ) 
(h1 : decorations_per_box = 15)
(h2 : used_decorations = 35)
(h3 : given_decorations = 25) :
  (used_decorations + given_decorations) / decorations_per_box = 4 := 
by sorry

end mrs_jackson_boxes_l450_45034


namespace minimum_minutes_for_planB_cheaper_l450_45024

-- Define the costs for Plan A and Plan B as functions of minutes
def planACost (x : Nat) : Nat := 1500 + 12 * x
def planBCost (x : Nat) : Nat := 3000 + 6 * x

-- Statement to prove
theorem minimum_minutes_for_planB_cheaper : 
  ∃ x : Nat, (planBCost x < planACost x) ∧ ∀ y : Nat, y < x → planBCost y ≥ planACost y :=
by
  sorry

end minimum_minutes_for_planB_cheaper_l450_45024


namespace reading_days_l450_45018

theorem reading_days (total_pages pages_per_day_1 pages_per_day_2 : ℕ ) :
  total_pages = 525 →
  pages_per_day_1 = 25 →
  pages_per_day_2 = 21 →
  (total_pages / pages_per_day_1 = 21) ∧ (total_pages / pages_per_day_2 = 25) :=
by
  sorry

end reading_days_l450_45018


namespace cuberoot_eight_is_512_l450_45085

-- Define the condition on x
def cuberoot_is_eight (x : ℕ) : Prop := 
  x^(1 / 3) = 8

-- The statement to be proved
theorem cuberoot_eight_is_512 : ∃ x : ℕ, cuberoot_is_eight x ∧ x = 512 := 
by 
  -- Proof is omitted
  sorry

end cuberoot_eight_is_512_l450_45085


namespace max_a3_b3_c3_d3_l450_45001

-- Define that a, b, c, d are real numbers that satisfy the given conditions.
theorem max_a3_b3_c3_d3 (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 16)
  (h2 : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d) :
  a^3 + b^3 + c^3 + d^3 ≤ 64 :=
sorry

end max_a3_b3_c3_d3_l450_45001


namespace num_first_graders_in_class_l450_45052

def numKindergartners := 14
def numSecondGraders := 4
def totalStudents := 42

def numFirstGraders : Nat := totalStudents - (numKindergartners + numSecondGraders)

theorem num_first_graders_in_class :
  numFirstGraders = 24 :=
by
  sorry

end num_first_graders_in_class_l450_45052


namespace min_value_fraction_subtraction_l450_45011

theorem min_value_fraction_subtraction
  (a b : ℝ)
  (ha : 0 < a ∧ a ≤ 3 / 4)
  (hb : 0 < b ∧ b ≤ 3 - a)
  (hineq : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → a * x + b - 3 ≤ 0) :
  ∃ a b, (0 < a ∧ a ≤ 3 / 4) ∧ (0 < b ∧ b ≤ 3 - a) ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → a * x + b - 3 ≤ 0) ∧ (1 / a - b = 1) :=
by 
  sorry

end min_value_fraction_subtraction_l450_45011
