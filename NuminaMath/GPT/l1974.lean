import Mathlib

namespace NUMINAMATH_GPT_min_deliveries_to_cover_cost_l1974_197485

theorem min_deliveries_to_cover_cost (cost_per_van earnings_per_delivery gasoline_cost_per_delivery : ℕ) (h1 : cost_per_van = 4500) (h2 : earnings_per_delivery = 15 ) (h3 : gasoline_cost_per_delivery = 5) : 
  ∃ d : ℕ, 10 * d ≥ cost_per_van ∧ ∀ x : ℕ, x < d → 10 * x < cost_per_van :=
by
  use 450
  sorry

end NUMINAMATH_GPT_min_deliveries_to_cover_cost_l1974_197485


namespace NUMINAMATH_GPT_law_firm_associates_l1974_197442

def percentage (total: ℕ) (part: ℕ): ℕ := part * 100 / total

theorem law_firm_associates (total: ℕ) (second_year: ℕ) (first_year: ℕ) (more_than_two_years: ℕ):
  percentage total more_than_two_years = 50 →
  percentage total second_year = 25 →
  first_year = more_than_two_years - second_year →
  percentage total first_year = 25 →
  percentage total (total - first_year) = 75 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_law_firm_associates_l1974_197442


namespace NUMINAMATH_GPT_arithmetic_sequence_a4_is_5_l1974_197451

variable (a : ℕ → ℕ)

-- Arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m k : ℕ, n < m ∧ m < k → 2 * a m = a n + a k

-- Given condition
axiom sum_third_and_fifth : a 3 + a 5 = 10

-- Prove that a_4 = 5
theorem arithmetic_sequence_a4_is_5
  (h : is_arithmetic_sequence a) : a 4 = 5 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a4_is_5_l1974_197451


namespace NUMINAMATH_GPT_mutually_exclusive_not_complementary_l1974_197409

def event_odd (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5
def event_greater_than_5 (n : ℕ) : Prop := n = 6

theorem mutually_exclusive_not_complementary :
  (∀ n : ℕ, event_odd n → ¬ event_greater_than_5 n) ∧
  (∃ n : ℕ, ¬ event_odd n ∧ ¬ event_greater_than_5 n) :=
by
  sorry

end NUMINAMATH_GPT_mutually_exclusive_not_complementary_l1974_197409


namespace NUMINAMATH_GPT_right_triangle_third_angle_l1974_197465

-- Define the problem
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- Define the given angles
def is_right_angle (a : ℝ) : Prop := a = 90
def given_angle (b : ℝ) : Prop := b = 25

-- Define the third angle
def third_angle (a b c : ℝ) : Prop := a + b + c = 180

-- The theorem to prove 
theorem right_triangle_third_angle : ∀ (a b c : ℝ), 
  is_right_angle a → given_angle b → third_angle a b c → c = 65 :=
by
  intros a b c ha hb h_triangle
  sorry

end NUMINAMATH_GPT_right_triangle_third_angle_l1974_197465


namespace NUMINAMATH_GPT_James_selling_percentage_l1974_197475

def James_selling_percentage_proof : Prop :=
  ∀ (total_cost original_price return_cost extra_item bought_price out_of_pocket sold_amount : ℝ),
    total_cost = 3000 →
    return_cost = 700 + 500 →
    extra_item = 500 * 1.2 →
    bought_price = 100 →
    out_of_pocket = 2020 →
    sold_amount = out_of_pocket - (total_cost - return_cost + bought_price) →
    sold_amount / extra_item * 100 = 20

theorem James_selling_percentage : James_selling_percentage_proof :=
by
  sorry

end NUMINAMATH_GPT_James_selling_percentage_l1974_197475


namespace NUMINAMATH_GPT_no_feasible_distribution_l1974_197413

-- Define the initial conditions
def initial_runs_player_A : ℕ := 320
def initial_runs_player_B : ℕ := 450
def initial_runs_player_C : ℕ := 550

def initial_innings : ℕ := 10

def required_increase_A : ℕ := 4
def required_increase_B : ℕ := 5
def required_increase_C : ℕ := 6

def total_run_limit : ℕ := 250

-- Define the total runs required after 11 innings
def total_required_runs_after_11_innings (initial_runs avg_increase : ℕ) : ℕ :=
  (initial_runs / initial_innings + avg_increase) * 11

-- Calculate the additional runs needed in the next innings
def additional_runs_needed (initial_runs avg_increase : ℕ) : ℕ :=
  total_required_runs_after_11_innings initial_runs avg_increase - initial_runs

-- Calculate the total additional runs needed for all players
def total_additional_runs_needed : ℕ :=
  additional_runs_needed initial_runs_player_A required_increase_A +
  additional_runs_needed initial_runs_player_B required_increase_B +
  additional_runs_needed initial_runs_player_C required_increase_C

-- The statement to verify if the total additional required runs exceed the limit
theorem no_feasible_distribution :
  total_additional_runs_needed > total_run_limit :=
by 
  -- Skipping proofs and just stating the condition is what we aim to show.
  sorry

end NUMINAMATH_GPT_no_feasible_distribution_l1974_197413


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1974_197472

-- Define the conditions
def a := 2
def b := -1

-- State the theorem
theorem simplify_and_evaluate_expression : 
  ((2 * a + 3 * b) * (2 * a - 3 * b) - (2 * a - b) ^ 2 - 3 * a * b) / (-b) = -12 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1974_197472


namespace NUMINAMATH_GPT_greatest_divisor_form_p_plus_1_l1974_197496

theorem greatest_divisor_form_p_plus_1 (n : ℕ) (hn : 0 < n):
  (∀ p : ℕ, Nat.Prime p → p % 3 = 2 → ¬ (p ∣ n) → 6 ∣ (p + 1)) ∧
  (∀ d : ℕ, (∀ p : ℕ, Nat.Prime p → p % 3 = 2 → ¬ (p ∣ n) → d ∣ (p + 1)) → d ≤ 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_divisor_form_p_plus_1_l1974_197496


namespace NUMINAMATH_GPT_quadratic_expression_negative_for_all_x_l1974_197410

theorem quadratic_expression_negative_for_all_x (k : ℝ) :
  (∀ x : ℝ, (5-k) * x^2 - 2 * (1-k) * x + 2 - 2 * k < 0) ↔ k > 9 :=
sorry

end NUMINAMATH_GPT_quadratic_expression_negative_for_all_x_l1974_197410


namespace NUMINAMATH_GPT_transaction_gain_per_year_l1974_197497

noncomputable def principal : ℝ := 9000
noncomputable def time : ℝ := 2
noncomputable def rate_lending : ℝ := 6
noncomputable def rate_borrowing : ℝ := 4

noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

noncomputable def total_interest_earned := simple_interest principal rate_lending time
noncomputable def total_interest_paid := simple_interest principal rate_borrowing time

noncomputable def total_gain := total_interest_earned - total_interest_paid
noncomputable def gain_per_year := total_gain / 2

theorem transaction_gain_per_year : gain_per_year = 180 :=
by
  sorry

end NUMINAMATH_GPT_transaction_gain_per_year_l1974_197497


namespace NUMINAMATH_GPT_sum_200_to_299_l1974_197406

variable (a : ℕ)

-- Condition: Sum of the first 100 natural numbers is equal to a
def sum_100 := (100 * 101) / 2

-- Main Theorem: Sum from 200 to 299 in terms of a
theorem sum_200_to_299 (h : sum_100 = a) : (299 * 300 / 2 - 199 * 200 / 2) = 19900 + a := by
  sorry

end NUMINAMATH_GPT_sum_200_to_299_l1974_197406


namespace NUMINAMATH_GPT_number_of_seedlings_l1974_197419

theorem number_of_seedlings (packets : ℕ) (seeds_per_packet : ℕ) (h1 : packets = 60) (h2 : seeds_per_packet = 7) : packets * seeds_per_packet = 420 :=
by
  sorry

end NUMINAMATH_GPT_number_of_seedlings_l1974_197419


namespace NUMINAMATH_GPT_units_digit_smallest_n_l1974_197491

theorem units_digit_smallest_n (n : ℕ) (h1 : 7 * n ≥ 10^2015) (h2 : 7 * (n - 1) < 10^2015) : (n % 10) = 6 :=
sorry

end NUMINAMATH_GPT_units_digit_smallest_n_l1974_197491


namespace NUMINAMATH_GPT_system_solutions_l1974_197499

theorem system_solutions (x a : ℝ) (h1 : a = -3*x^2 + 5*x - 2) (h2 : (x + 2) * a = 4 * (x^2 - 1)) (hx : x ≠ -2) :
  (x = 0 ∧ a = -2) ∨ (x = 1 ∧ a = 0) ∨ (x = -8/3 ∧ a = -110/3) :=
  sorry

end NUMINAMATH_GPT_system_solutions_l1974_197499


namespace NUMINAMATH_GPT_expression_constant_for_large_x_l1974_197466

theorem expression_constant_for_large_x (x : ℝ) (h : x ≥ 4 / 7) : 
  -4 * x + |4 - 7 * x| - |1 - 3 * x| + 4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_expression_constant_for_large_x_l1974_197466


namespace NUMINAMATH_GPT_factorial_fraction_integer_l1974_197450

open Nat

theorem factorial_fraction_integer (m n : ℕ) : 
  ∃ k : ℕ, k = (2 * m).factorial * (2 * n).factorial / (m.factorial * n.factorial * (m + n).factorial) := 
sorry

end NUMINAMATH_GPT_factorial_fraction_integer_l1974_197450


namespace NUMINAMATH_GPT_integral_x_squared_l1974_197483

theorem integral_x_squared:
  ∫ x in (0:ℝ)..(1:ℝ), x^2 = 1/3 :=
by
  sorry

end NUMINAMATH_GPT_integral_x_squared_l1974_197483


namespace NUMINAMATH_GPT_last_digit_2_pow_2023_l1974_197416

-- Definitions
def last_digit_cycle : List ℕ := [2, 4, 8, 6]

-- Theorem statement
theorem last_digit_2_pow_2023 : (2 ^ 2023) % 10 = 8 :=
by
  -- We will assume and use the properties mentioned in the solution steps.
  -- The proof process is skipped here with 'sorry'.
  sorry

end NUMINAMATH_GPT_last_digit_2_pow_2023_l1974_197416


namespace NUMINAMATH_GPT_max_m_plus_n_l1974_197459

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 + 3

theorem max_m_plus_n (m n : ℝ) (h : n = quadratic_function m) : m + n ≤ 13/4 :=
sorry

end NUMINAMATH_GPT_max_m_plus_n_l1974_197459


namespace NUMINAMATH_GPT_maximum_value_of_m_solve_inequality_l1974_197452

theorem maximum_value_of_m (a b : ℝ) (h : a ≠ 0) : 
  ∃ m : ℝ, (∀ a b : ℝ, a ≠ 0 → |a + b| + |a - b| ≥ m * |a|) ∧ (m = 2) :=
by
  use 2
  sorry

theorem solve_inequality (x : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| ≤ 2 → (1/2 ≤ x ∧ x ≤ 5/2)) :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_m_solve_inequality_l1974_197452


namespace NUMINAMATH_GPT_exists_n_prime_divides_exp_sum_l1974_197478

theorem exists_n_prime_divides_exp_sum (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_n_prime_divides_exp_sum_l1974_197478


namespace NUMINAMATH_GPT_money_distribution_l1974_197404

theorem money_distribution (p q r : ℝ) 
  (h1 : p + q + r = 9000) 
  (h2 : r = (2/3) * (p + q)) : 
  r = 3600 := 
by 
  sorry

end NUMINAMATH_GPT_money_distribution_l1974_197404


namespace NUMINAMATH_GPT_shirts_sold_l1974_197461

theorem shirts_sold (S : ℕ) (H_total : 69 = 7 * 7 + 5 * S) : S = 4 :=
by
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_shirts_sold_l1974_197461


namespace NUMINAMATH_GPT_price_of_70_cans_l1974_197455

noncomputable def discounted_price (regular_price : ℝ) (discount_percent : ℝ) : ℝ :=
  regular_price * (1 - discount_percent / 100)

noncomputable def total_price (regular_price : ℝ) (discount_percent : ℝ) (total_cans : ℕ) (cans_per_case : ℕ) : ℝ :=
  let price_per_can := discounted_price regular_price discount_percent
  let full_cases := total_cans / cans_per_case
  let remaining_cans := total_cans % cans_per_case
  full_cases * cans_per_case * price_per_can + remaining_cans * price_per_can

theorem price_of_70_cans :
  total_price 0.55 25 70 24 = 28.875 :=
by
  sorry

end NUMINAMATH_GPT_price_of_70_cans_l1974_197455


namespace NUMINAMATH_GPT_sequence_count_is_correct_l1974_197401

def has_integer_root (a_i a_i_plus_1 : ℕ) : Prop :=
  ∃ r : ℕ, r^2 - a_i * r + a_i_plus_1 = 0

def valid_sequence (seq : Fin 16 → ℕ) : Prop :=
  ∀ i : Fin 15, has_integer_root (seq i.val + 1) (seq (i + 1).val + 1) ∧ seq 15 = seq 0

-- This noncomputable definition is used because we are estimating a specific number without providing a concrete computable function.
noncomputable def sequence_count : ℕ :=
  1409

theorem sequence_count_is_correct :
  ∃ N, valid_sequence seq → N = 1409 :=
sorry 

end NUMINAMATH_GPT_sequence_count_is_correct_l1974_197401


namespace NUMINAMATH_GPT_time_needed_by_Alpha_and_Beta_l1974_197403

theorem time_needed_by_Alpha_and_Beta (A B C h : ℝ)
  (h₀ : 1 / (A - 4) = 1 / (B - 2))
  (h₁ : 1 / A + 1 / B + 1 / C = 3 / C)
  (h₂ : A = B + 2)
  (h₃ : 1 / 12 + 1 / 10 = 11 / 60)
  : h = 60 / 11 :=
sorry

end NUMINAMATH_GPT_time_needed_by_Alpha_and_Beta_l1974_197403


namespace NUMINAMATH_GPT_rectangle_perimeters_l1974_197434

theorem rectangle_perimeters (w h : ℝ) 
  (h1 : 2 * (w + h) = 20)
  (h2 : 2 * (4 * w + h) = 56) : 
  4 * (w + h) = 40 ∧ 2 * (w + 4 * h) = 44 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeters_l1974_197434


namespace NUMINAMATH_GPT_Heracles_age_l1974_197467

variable (A H : ℕ)

theorem Heracles_age :
  (A = H + 7) →
  (A + 3 = 2 * H) →
  H = 10 :=
by
  sorry

end NUMINAMATH_GPT_Heracles_age_l1974_197467


namespace NUMINAMATH_GPT_no_valid_n_values_l1974_197458

theorem no_valid_n_values :
  ¬ ∃ n : ℕ, (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
by
  sorry

end NUMINAMATH_GPT_no_valid_n_values_l1974_197458


namespace NUMINAMATH_GPT_part1_part2_l1974_197456

theorem part1 (x : ℝ) (m : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0) ↔ m ≤ 6 :=
sorry

theorem part2 (a b : ℝ) (n : ℝ) :
  n = 6 → (a > 0 ∧ b > 0 ∧ (4 / (a + 5 * b)) + (1 / (3 * a + 2 * b)) = 1) → (4 * a + 7 * b) ≥ 9 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1974_197456


namespace NUMINAMATH_GPT_largest_a_l1974_197433

open Real

theorem largest_a (a b c : ℝ) (h1 : a + b + c = 6) (h2 : ab + ac + bc = 11) : 
  a ≤ 2 + 2 * sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_largest_a_l1974_197433


namespace NUMINAMATH_GPT_number_of_valid_polynomials_l1974_197421

noncomputable def count_polynomials_meeting_conditions : ℕ := sorry

theorem number_of_valid_polynomials :
  count_polynomials_meeting_conditions = 7200 :=
sorry

end NUMINAMATH_GPT_number_of_valid_polynomials_l1974_197421


namespace NUMINAMATH_GPT_find_C_plus_D_l1974_197470

theorem find_C_plus_D (C D : ℝ) (h : ∀ x : ℝ, (Cx - 20) / (x^2 - 3 * x - 10) = D / (x + 2) + 4 / (x - 5)) :
  C + D = 4.7 :=
sorry

end NUMINAMATH_GPT_find_C_plus_D_l1974_197470


namespace NUMINAMATH_GPT_Sue_waited_in_NY_l1974_197480

-- Define the conditions as constants and assumptions
def T_NY_SF : ℕ := 24
def T_total : ℕ := 58
def T_NO_NY : ℕ := (3 * T_NY_SF) / 4

-- Define the waiting time
def T_wait : ℕ := T_total - T_NO_NY - T_NY_SF

-- Theorem stating the problem
theorem Sue_waited_in_NY :
  T_wait = 16 :=
by
  -- Implicitly using the given conditions
  sorry

end NUMINAMATH_GPT_Sue_waited_in_NY_l1974_197480


namespace NUMINAMATH_GPT_prob_snow_both_days_l1974_197462

-- Definitions for the conditions
def prob_snow_monday : ℚ := 40 / 100
def prob_snow_tuesday : ℚ := 30 / 100

def independent_events (A B : Prop) : Prop := true  -- A placeholder definition of independence

-- The proof problem: 
theorem prob_snow_both_days : 
  independent_events (prob_snow_monday = 0.40) (prob_snow_tuesday = 0.30) →
  prob_snow_monday * prob_snow_tuesday = 0.12 := 
by 
  sorry

end NUMINAMATH_GPT_prob_snow_both_days_l1974_197462


namespace NUMINAMATH_GPT_total_students_l1974_197420

theorem total_students (h1 : ∀ (n : ℕ), n = 5 → Jaya_ranks_nth_from_top)
                       (h2 : ∀ (m : ℕ), m = 49 → Jaya_ranks_mth_from_bottom) :
  ∃ (total : ℕ), total = 53 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l1974_197420


namespace NUMINAMATH_GPT_problem1_problem2_l1974_197449

section problems

variables (m n a b : ℕ)
variables (h1 : 4 ^ m = a) (h2 : 8 ^ n = b)

theorem problem1 : 2 ^ (2 * m + 3 * n) = a * b :=
sorry

theorem problem2 : 2 ^ (4 * m - 6 * n) = a ^ 2 / b ^ 2 :=
sorry

end problems

end NUMINAMATH_GPT_problem1_problem2_l1974_197449


namespace NUMINAMATH_GPT_total_cost_of_books_l1974_197487

theorem total_cost_of_books (C1 C2 : ℝ) 
  (hC1 : C1 = 268.33)
  (h_selling_prices_equal : 0.85 * C1 = 1.19 * C2) :
  C1 + C2 = 459.15 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_cost_of_books_l1974_197487


namespace NUMINAMATH_GPT_no_solution_fraction_eq_l1974_197429

theorem no_solution_fraction_eq {x m : ℝ} : 
  (∀ x, ¬ (1 - x = 0) → (2 - x) / (1 - x) = (m + x) / (1 - x) + 1) ↔ m = 0 := 
by
  sorry

end NUMINAMATH_GPT_no_solution_fraction_eq_l1974_197429


namespace NUMINAMATH_GPT_hyperbola_condition_l1974_197418

theorem hyperbola_condition (k : ℝ) (x y : ℝ) :
  (k ≠ 0 ∧ k ≠ 3 ∧ (x^2 / k + y^2 / (k - 3) = 1)) → 0 < k ∧ k < 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_condition_l1974_197418


namespace NUMINAMATH_GPT_bob_calories_l1974_197454

-- conditions
def slices : ℕ := 8
def half_slices (slices : ℕ) : ℕ := slices / 2
def calories_per_slice : ℕ := 300
def total_calories (half_slices : ℕ) (calories_per_slice : ℕ) : ℕ := half_slices * calories_per_slice

-- proof problem
theorem bob_calories : total_calories (half_slices slices) calories_per_slice = 1200 := by
  sorry

end NUMINAMATH_GPT_bob_calories_l1974_197454


namespace NUMINAMATH_GPT_find_f_at_1_l1974_197405

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem find_f_at_1 : f 1 = 2 := by
  sorry

end NUMINAMATH_GPT_find_f_at_1_l1974_197405


namespace NUMINAMATH_GPT_scientific_notation_of_56_point_5_million_l1974_197463

-- Definitions based on conditions
def million : ℝ := 10^6
def number_in_millions : ℝ := 56.5 * million

-- Statement to be proved
theorem scientific_notation_of_56_point_5_million : 
  number_in_millions = 5.65 * 10^7 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_56_point_5_million_l1974_197463


namespace NUMINAMATH_GPT_hours_per_toy_l1974_197493

-- Defining the conditions
def toys_produced (hours: ℕ) : ℕ := 40 
def hours_worked : ℕ := 80

-- Theorem: If a worker makes 40 toys in 80 hours, then it takes 2 hours to make one toy.
theorem hours_per_toy : (hours_worked / toys_produced hours_worked) = 2 :=
by
  sorry

end NUMINAMATH_GPT_hours_per_toy_l1974_197493


namespace NUMINAMATH_GPT_injective_function_equality_l1974_197425

def injective (f : ℕ → ℕ) : Prop :=
  ∀ ⦃a b : ℕ⦄, f a = f b → a = b

theorem injective_function_equality
  {f : ℕ → ℕ}
  (h_injective : injective f)
  (h_eq : ∀ n m : ℕ, (1 / f n) + (1 / f m) = 4 / (f n + f m)) :
  ∀ n m : ℕ, m = n :=
by
  sorry

end NUMINAMATH_GPT_injective_function_equality_l1974_197425


namespace NUMINAMATH_GPT_smallest_even_x_l1974_197423

theorem smallest_even_x (x : ℤ) (h1 : x < 3 * x - 10) (h2 : ∃ k : ℤ, x = 2 * k) : x = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_even_x_l1974_197423


namespace NUMINAMATH_GPT_sum_of_cubes_application_l1974_197446

theorem sum_of_cubes_application : 
  ¬ ((a+1) * (a^2 - a + 1) = a^3 + 1) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_application_l1974_197446


namespace NUMINAMATH_GPT_find_constants_l1974_197492

-- Define constants and the problem
variables (C D Q : Type) [AddCommGroup Q] [Module ℝ Q]
variables (CQ QD : ℝ) (h_ratio : CQ = 3 * QD / 5)

-- Define the conjecture we want to prove
theorem find_constants (t u : ℝ) (h_t : t = 5 / (3 + 5)) (h_u : u = 3 / (3 + 5)) :
  (CQ = 3 * QD / 5) → 
  (t * CQ + u * QD = (5 / 8) * CQ + (3 / 8) * QD) :=
sorry

end NUMINAMATH_GPT_find_constants_l1974_197492


namespace NUMINAMATH_GPT_three_monotonic_intervals_iff_a_lt_zero_l1974_197498

-- Definition of the function f
def f (a x : ℝ) : ℝ := a * x^3 + x

-- Definition of the first derivative of f
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 1

-- Main statement: Prove that f(x) has exactly three monotonic intervals if and only if a < 0.
theorem three_monotonic_intervals_iff_a_lt_zero (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' a x1 = 0 ∧ f' a x2 = 0) ↔ a < 0 :=
by
  sorry

end NUMINAMATH_GPT_three_monotonic_intervals_iff_a_lt_zero_l1974_197498


namespace NUMINAMATH_GPT_math_problem_l1974_197479

variable {a b c : ℝ}

theorem math_problem
  (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0)
  (h : a + b + c = -a * b * c) :
  (a^2 * b^2 / ((a^2 + b * c) * (b^2 + a * c)) +
  a^2 * c^2 / ((a^2 + b * c) * (c^2 + a * b)) +
  b^2 * c^2 / ((b^2 + a * c) * (c^2 + a * b))) = 1 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1974_197479


namespace NUMINAMATH_GPT_train_lengths_equal_l1974_197441

theorem train_lengths_equal (v_fast v_slow : ℝ) (t : ℝ) (L : ℝ)  
  (h1 : v_fast = 46) 
  (h2 : v_slow = 36) 
  (h3 : t = 36.00001) : 
  2 * L = (v_fast - v_slow) / 3600 * t → L = 1800.0005 := 
by
  sorry

end NUMINAMATH_GPT_train_lengths_equal_l1974_197441


namespace NUMINAMATH_GPT_loss_equals_cost_price_of_balls_l1974_197432

variable (selling_price : ℕ) (cost_price_ball : ℕ)
variable (number_of_balls : ℕ) (loss_incurred : ℕ) (x : ℕ)

-- Conditions
def condition1 : selling_price = 720 := sorry -- Selling price of 11 balls is Rs. 720
def condition2 : cost_price_ball = 120 := sorry -- Cost price of one ball is Rs. 120
def condition3 : number_of_balls = 11 := sorry -- Number of balls is 11

-- Cost price of 11 balls
def cost_price (n : ℕ) (cp_ball : ℕ): ℕ := n * cp_ball

-- Loss incurred on selling 11 balls
def loss (cp : ℕ) (sp : ℕ): ℕ := cp - sp

-- Equation for number of balls the loss equates to
def loss_equation (l : ℕ) (cp_ball : ℕ): ℕ := l / cp_ball

theorem loss_equals_cost_price_of_balls : 
  ∀ (n sp cp_ball cp l: ℕ), 
  sp = 720 ∧ cp_ball = 120 ∧ n = 11 ∧ 
  cp = cost_price n cp_ball ∧ 
  l = loss cp sp →
  loss_equation l cp_ball = 5 := sorry

end NUMINAMATH_GPT_loss_equals_cost_price_of_balls_l1974_197432


namespace NUMINAMATH_GPT_contradiction_proof_real_root_l1974_197411

theorem contradiction_proof_real_root (a b : ℝ) :
  (∀ x : ℝ, x^3 + a * x + b ≠ 0) → (∃ x : ℝ, x + a * x + b = 0) :=
sorry

end NUMINAMATH_GPT_contradiction_proof_real_root_l1974_197411


namespace NUMINAMATH_GPT_polynomial_inequality_l1974_197407

theorem polynomial_inequality (x : ℝ) : x * (x + 1) * (x + 2) * (x + 3) ≥ -1 :=
sorry

end NUMINAMATH_GPT_polynomial_inequality_l1974_197407


namespace NUMINAMATH_GPT_cosine_of_tangent_line_at_e_l1974_197437

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem cosine_of_tangent_line_at_e :
  let θ := Real.arctan 2
  Real.cos θ = Real.sqrt (1 / 5) := by
  sorry

end NUMINAMATH_GPT_cosine_of_tangent_line_at_e_l1974_197437


namespace NUMINAMATH_GPT_prob_le_45_l1974_197424

-- Define the probability conditions
def prob_between_1_and_45 : ℚ := 7 / 15
def prob_ge_1 : ℚ := 14 / 15

-- State the theorem to prove
theorem prob_le_45 : prob_between_1_and_45 = 7 / 15 := by
  sorry

end NUMINAMATH_GPT_prob_le_45_l1974_197424


namespace NUMINAMATH_GPT_find_value_of_fraction_l1974_197471

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_fraction_l1974_197471


namespace NUMINAMATH_GPT_line_through_midpoint_of_ellipse_l1974_197464

theorem line_through_midpoint_of_ellipse:
  (∀ x y : ℝ, (x - 4)^2 + (y - 2)^2 = (1/36) * ((9 * 4) + 36 * (1 / 4)) → (1 + 2 * (y - 2) / (x - 4) = 0)) →
  (x - 8) + 2 * (y - 4) = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_through_midpoint_of_ellipse_l1974_197464


namespace NUMINAMATH_GPT_number_of_girls_l1974_197417

theorem number_of_girls (total_students boys girls : ℕ)
  (h1 : boys = 300)
  (h2 : (girls : ℝ) = 0.6 * total_students)
  (h3 : (boys : ℝ) = 0.4 * total_students) : 
  girls = 450 := by
  sorry

end NUMINAMATH_GPT_number_of_girls_l1974_197417


namespace NUMINAMATH_GPT_bill_sunday_miles_l1974_197422

variable (B : ℕ)

-- Conditions
def miles_Bill_Saturday : ℕ := B
def miles_Bill_Sunday : ℕ := B + 4
def miles_Julia_Sunday : ℕ := 2 * (B + 4)
def total_miles : ℕ := miles_Bill_Saturday B + miles_Bill_Sunday B + miles_Julia_Sunday B

theorem bill_sunday_miles (h : total_miles B = 32) : miles_Bill_Sunday B = 9 := by
  sorry

end NUMINAMATH_GPT_bill_sunday_miles_l1974_197422


namespace NUMINAMATH_GPT_part1_part2_l1974_197495

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def b : ℝ × ℝ := (3, -Real.sqrt 3)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

noncomputable def f (x : ℝ) : ℝ := dot_product (a x) b

theorem part1 (hx : x ∈ Set.Icc 0 Real.pi) (h_perp : dot_product (a x) b = 0) : x = 5 * Real.pi / 6 :=
sorry

theorem part2 (hx : x ∈ Set.Icc 0 Real.pi) :
  (f x ≤ 2 * Real.sqrt 3) ∧ (f x = 2 * Real.sqrt 3 → x = 0) ∧
  (f x ≥ -2 * Real.sqrt 3) ∧ (f x = -2 * Real.sqrt 3 → x = 5 * Real.pi / 6) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1974_197495


namespace NUMINAMATH_GPT_intersecting_lines_l1974_197453

theorem intersecting_lines (x y : ℝ) : x ^ 2 - y ^ 2 = 0 ↔ (y = x ∨ y = -x) := by
  sorry

end NUMINAMATH_GPT_intersecting_lines_l1974_197453


namespace NUMINAMATH_GPT_find_initial_interest_rate_l1974_197408

-- Definitions of the initial conditions
def P1 : ℝ := 3000
def P2 : ℝ := 1499.9999999999998
def P_total : ℝ := 4500
def r2 : ℝ := 0.08
def total_annual_income : ℝ := P_total * 0.06

-- Defining the problem as a statement to prove
theorem find_initial_interest_rate (r1 : ℝ) :
  (P1 * r1) + (P2 * r2) = total_annual_income → r1 = 0.05 := by
  sorry

end NUMINAMATH_GPT_find_initial_interest_rate_l1974_197408


namespace NUMINAMATH_GPT_max_length_of_cuts_l1974_197435

-- Define the dimensions of the board and the number of parts
def board_size : ℕ := 30
def num_parts : ℕ := 225

-- Define the total possible length of the cuts
def max_possible_cuts_length : ℕ := 1065

-- Define the condition that the board is cut into parts of equal area
def equal_area_partition (board_size num_parts : ℕ) : Prop :=
  ∃ (area_per_part : ℕ), (board_size * board_size) / num_parts = area_per_part

-- Define the theorem to prove the maximum possible total length of the cuts
theorem max_length_of_cuts (h : equal_area_partition board_size num_parts) :
  max_possible_cuts_length = 1065 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_max_length_of_cuts_l1974_197435


namespace NUMINAMATH_GPT_tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m_l1974_197489

theorem tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m (m : ℝ) (α : ℝ)
  (h1 : Real.tan α = m / 3)
  (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
sorry

end NUMINAMATH_GPT_tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m_l1974_197489


namespace NUMINAMATH_GPT_polynomial_inequality_solution_l1974_197474

theorem polynomial_inequality_solution :
  { x : ℝ | x * (x - 5) * (x - 10)^2 > 0 } = { x : ℝ | 0 < x ∧ x < 5 ∨ 10 < x } :=
by
  sorry

end NUMINAMATH_GPT_polynomial_inequality_solution_l1974_197474


namespace NUMINAMATH_GPT_cube_surface_area_l1974_197460

noncomputable def total_surface_area_of_cube (Q : ℝ) : ℝ :=
  8 * Q * Real.sqrt 3 / 3

theorem cube_surface_area (Q : ℝ) (h : Q > 0) :
  total_surface_area_of_cube Q = 8 * Q * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_cube_surface_area_l1974_197460


namespace NUMINAMATH_GPT_factorial_expression_equals_l1974_197402

theorem factorial_expression_equals :
  7 * Nat.factorial 7 + 5 * Nat.factorial 5 - 3 * Nat.factorial 3 + 2 * Nat.factorial 2 = 35866 := by
  sorry

end NUMINAMATH_GPT_factorial_expression_equals_l1974_197402


namespace NUMINAMATH_GPT_average_last_three_l1974_197444

theorem average_last_three (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d + e + f + g) / 7 = 65) 
  (h2 : (a + b + c + d) / 4 = 60) : 
  (e + f + g) / 3 = 71.67 :=
by
  sorry

end NUMINAMATH_GPT_average_last_three_l1974_197444


namespace NUMINAMATH_GPT_symmetric_circle_eq_l1974_197445

theorem symmetric_circle_eq :
  (∃ f : ℝ → ℝ → Prop, (∀ x y, f x y ↔ (x - 2)^2 + (y + 1)^2 = 1)) →
  (∃ line : ℝ → ℝ → Prop, (∀ x y, line x y ↔ x - y + 3 = 0)) →
  (∃ eq : ℝ → ℝ → Prop, (∀ x y, eq x y ↔ (x - 4)^2 + (y - 5)^2 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_circle_eq_l1974_197445


namespace NUMINAMATH_GPT_find_p_q_sum_p_plus_q_l1974_197457

noncomputable def probability_third_six : ℚ :=
  have fair_die_prob_two_sixes := (1 / 6) * (1 / 6)
  have biased_die_prob_two_sixes := (2 / 3) * (2 / 3)
  have total_prob_two_sixes := (1 / 2) * fair_die_prob_two_sixes + (1 / 2) * biased_die_prob_two_sixes
  have prob_fair_given_two_sixes := fair_die_prob_two_sixes / total_prob_two_sixes
  have prob_biased_given_two_sixes := biased_die_prob_two_sixes / total_prob_two_sixes
  let prob_third_six :=
    prob_fair_given_two_sixes * (1 / 6) +
    prob_biased_given_two_sixes * (2 / 3)
  prob_third_six

theorem find_p_q_sum : 
  probability_third_six = 65 / 102 :=
by sorry

theorem p_plus_q : 
  65 + 102 = 167 :=
by sorry

end NUMINAMATH_GPT_find_p_q_sum_p_plus_q_l1974_197457


namespace NUMINAMATH_GPT_eval_expression_l1974_197439

theorem eval_expression : (825 * 825) - (824 * 826) = 1 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l1974_197439


namespace NUMINAMATH_GPT_rhombus_area_correct_l1974_197400

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area_correct :
  rhombus_area 80 120 = 4800 :=
by 
  -- the proof is skipped by including sorry
  sorry

end NUMINAMATH_GPT_rhombus_area_correct_l1974_197400


namespace NUMINAMATH_GPT_lineup_count_l1974_197438

theorem lineup_count (n k : ℕ) (h : n = 13) (k_eq : k = 4) : (n.choose k) = 715 := by
  sorry

end NUMINAMATH_GPT_lineup_count_l1974_197438


namespace NUMINAMATH_GPT_min_value_of_n_for_constant_term_l1974_197431

theorem min_value_of_n_for_constant_term :
  ∃ (n : ℕ) (r : ℕ) (h₁ : r > 0) (h₂ : n > 0), 
  (2 * n - 7 * r / 3 = 0) ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_n_for_constant_term_l1974_197431


namespace NUMINAMATH_GPT_number_of_pages_in_chunk_l1974_197476

-- Conditions
def first_page : Nat := 213
def last_page : Nat := 312

-- Define the property we need to prove
theorem number_of_pages_in_chunk : last_page - first_page + 1 = 100 := by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_number_of_pages_in_chunk_l1974_197476


namespace NUMINAMATH_GPT_jenny_problem_l1974_197473

def round_to_nearest_ten (n : ℤ) : ℤ :=
  if n % 10 < 5 then n - (n % 10) else n + (10 - n % 10)

theorem jenny_problem : round_to_nearest_ten (58 + 29) = 90 := 
by
  sorry

end NUMINAMATH_GPT_jenny_problem_l1974_197473


namespace NUMINAMATH_GPT_value_of_leftover_coins_l1974_197488

def quarters_per_roll : ℕ := 30
def dimes_per_roll : ℕ := 40

def ana_quarters : ℕ := 95
def ana_dimes : ℕ := 183

def ben_quarters : ℕ := 104
def ben_dimes : ℕ := 219

def leftover_quarters : ℕ := (ana_quarters + ben_quarters) % quarters_per_roll
def leftover_dimes : ℕ := (ana_dimes + ben_dimes) % dimes_per_roll

def dollar_value (quarters dimes : ℕ) : ℝ := quarters * 0.25 + dimes * 0.10

theorem value_of_leftover_coins : 
  dollar_value leftover_quarters leftover_dimes = 6.95 := 
  sorry

end NUMINAMATH_GPT_value_of_leftover_coins_l1974_197488


namespace NUMINAMATH_GPT_ratio_unit_price_brand_x_to_brand_y_l1974_197477

-- Definitions based on the conditions in the problem
def volume_brand_y (v : ℝ) := v
def price_brand_y (p : ℝ) := p
def volume_brand_x (v : ℝ) := 1.3 * v
def price_brand_x (p : ℝ) := 0.85 * p
noncomputable def unit_price (volume : ℝ) (price : ℝ) := price / volume

-- Theorems to prove the ratio of unit price of Brand X to Brand Y is 17/26
theorem ratio_unit_price_brand_x_to_brand_y (v p : ℝ) (hv : v ≠ 0) (hp : p ≠ 0) : 
  (unit_price (volume_brand_x v) (price_brand_x p)) / (unit_price (volume_brand_y v) (price_brand_y p)) = 17 / 26 := by
  sorry

end NUMINAMATH_GPT_ratio_unit_price_brand_x_to_brand_y_l1974_197477


namespace NUMINAMATH_GPT_prob_exactly_M_laws_in_concept_expected_laws_in_concept_l1974_197468

section Anchuria
variables (K N M : ℕ) (p : ℝ)

-- Part (a): Define P_M as the binomial probability distribution result
def probability_exactly_M_laws : ℝ :=
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M)

-- Part (a): Prove the result for the probability that exactly M laws are included in the Concept
theorem prob_exactly_M_laws_in_concept :
  probability_exactly_M_laws K N M p =
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) := 
sorry

-- Part (b): Define the expected number of laws included in the Concept
def expected_number_of_laws : ℝ :=
  K * (1 - (1 - p) ^ N)

-- Part (b): Prove the result for the expected number of laws included in the Concept
theorem expected_laws_in_concept :
  expected_number_of_laws K N p = K * (1 - (1 - p) ^ N) :=
sorry
end Anchuria

end NUMINAMATH_GPT_prob_exactly_M_laws_in_concept_expected_laws_in_concept_l1974_197468


namespace NUMINAMATH_GPT_complementary_implies_right_triangle_l1974_197484

theorem complementary_implies_right_triangle (A B C : ℝ) (h : A + B = 90 ∧ A + B + C = 180) :
  C = 90 :=
by
  sorry

end NUMINAMATH_GPT_complementary_implies_right_triangle_l1974_197484


namespace NUMINAMATH_GPT_max_reflections_max_reflections_example_l1974_197443

-- Definition of the conditions
def angle_cda := 10  -- angle in degrees
def max_angle := 90  -- practical limit for angle of reflections

-- Given that the angle of incidence after n reflections is 10n degrees,
-- prove that the largest possible n is 9 before exceeding practical limits.
theorem max_reflections (n : ℕ) (h₁ : angle_cda = 10) (h₂ : max_angle = 90) :
  10 * n ≤ 90 :=
by sorry

-- Specific case instantiating n = 9
theorem max_reflections_example : (10 : ℕ) * 9 ≤ 90 := max_reflections 9 rfl rfl

end NUMINAMATH_GPT_max_reflections_max_reflections_example_l1974_197443


namespace NUMINAMATH_GPT_total_hours_over_two_weeks_l1974_197428

-- Define the conditions of Bethany's riding schedule
def hours_per_week : ℕ :=
  1 * 3 + -- Monday, Wednesday, and Friday
  (30 / 60) * 2 + -- Tuesday and Thursday, converting minutes to hours
  2 -- Saturday

-- The theorem to prove the total hours over 2 weeks
theorem total_hours_over_two_weeks : hours_per_week * 2 = 12 := 
by
  -- Proof to be completed here
  sorry

end NUMINAMATH_GPT_total_hours_over_two_weeks_l1974_197428


namespace NUMINAMATH_GPT_percentage_more_l1974_197414

variables (J T M : ℝ)

-- Conditions
def Tim_income : Prop := T = 0.90 * J
def Mary_income : Prop := M = 1.44 * J

-- Theorem to be proved
theorem percentage_more (h1 : Tim_income J T) (h2 : Mary_income J M) :
  ((M - T) / T) * 100 = 60 :=
sorry

end NUMINAMATH_GPT_percentage_more_l1974_197414


namespace NUMINAMATH_GPT_area_of_vegetable_patch_l1974_197447

theorem area_of_vegetable_patch : ∃ (a b : ℕ), 
  (2 * (a + b) = 24 ∧ b = 3 * a + 2 ∧ (6 * (a + 1)) * (6 * (b + 1)) = 576) :=
sorry

end NUMINAMATH_GPT_area_of_vegetable_patch_l1974_197447


namespace NUMINAMATH_GPT_weighted_avg_M_B_eq_l1974_197415

-- Define the weightages and the given weighted total marks equation
def weight_physics : ℝ := 1.5
def weight_chemistry : ℝ := 2
def weight_mathematics : ℝ := 1.25
def weight_biology : ℝ := 1.75
def weighted_total_M_B : ℝ := 250
def weighted_sum_M_B : ℝ := weight_mathematics + weight_biology

-- Theorem statement: Prove that the weighted average mark for mathematics and biology is 83.33
theorem weighted_avg_M_B_eq :
  (weighted_total_M_B / weighted_sum_M_B) = 83.33 :=
by
  sorry

end NUMINAMATH_GPT_weighted_avg_M_B_eq_l1974_197415


namespace NUMINAMATH_GPT_brendan_fish_caught_afternoon_l1974_197469

theorem brendan_fish_caught_afternoon (morning_fish : ℕ) (thrown_fish : ℕ) (dads_fish : ℕ) (total_fish : ℕ) :
  morning_fish = 8 → thrown_fish = 3 → dads_fish = 13 → total_fish = 23 → 
  (morning_fish - thrown_fish) + dads_fish + brendan_afternoon_catch = total_fish → 
  brendan_afternoon_catch = 5 :=
by
  intros morning_fish_eq thrown_fish_eq dads_fish_eq total_fish_eq fish_sum_eq
  sorry

end NUMINAMATH_GPT_brendan_fish_caught_afternoon_l1974_197469


namespace NUMINAMATH_GPT_total_hours_l1974_197440

variable (K : ℕ) (P : ℕ) (M : ℕ)

-- Conditions:
axiom h1 : P = 2 * K
axiom h2 : P = (1 / 3 : ℝ) * M
axiom h3 : M = K + 105

-- Goal: Proving the total number of hours is 189
theorem total_hours : K + P + M = 189 := by
  sorry

end NUMINAMATH_GPT_total_hours_l1974_197440


namespace NUMINAMATH_GPT_pet_store_animals_left_l1974_197481

theorem pet_store_animals_left (initial_birds initial_puppies initial_cats initial_spiders initial_snakes : ℕ)
  (donation_fraction snakes_share_sold birds_sold puppies_adopted cats_transferred kittens_brought : ℕ)
  (spiders_loose spiders_captured : ℕ)
  (H_initial_birds : initial_birds = 12)
  (H_initial_puppies : initial_puppies = 9)
  (H_initial_cats : initial_cats = 5)
  (H_initial_spiders : initial_spiders = 15)
  (H_initial_snakes : initial_snakes = 8)
  (H_donation_fraction : donation_fraction = 25)
  (H_snakes_share_sold : snakes_share_sold = (donation_fraction * initial_snakes) / 100)
  (H_birds_sold : birds_sold = initial_birds / 2)
  (H_puppies_adopted : puppies_adopted = 3)
  (H_cats_transferred : cats_transferred = 4)
  (H_kittens_brought : kittens_brought = 2)
  (H_spiders_loose : spiders_loose = 7)
  (H_spiders_captured : spiders_captured = 5) :
  (initial_snakes - snakes_share_sold) + (initial_birds - birds_sold) + 
  (initial_puppies - puppies_adopted) + (initial_cats - cats_transferred + kittens_brought) + 
  (initial_spiders - (spiders_loose - spiders_captured)) = 34 := 
by 
  sorry

end NUMINAMATH_GPT_pet_store_animals_left_l1974_197481


namespace NUMINAMATH_GPT_questions_answered_second_half_l1974_197494

theorem questions_answered_second_half :
  ∀ (q1 q2 p s : ℕ), q1 = 3 → p = 3 → s = 15 → s = (q1 + q2) * p → q2 = 2 :=
by
  intros q1 q2 p s hq1 hp hs h_final_score
  -- proofs go here, but we skip them
  sorry

end NUMINAMATH_GPT_questions_answered_second_half_l1974_197494


namespace NUMINAMATH_GPT_bamboo_middle_node_capacity_l1974_197436

def capacities_form_arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

theorem bamboo_middle_node_capacity :
  ∃ (a : ℕ → ℚ) (d : ℚ), 
    capacities_form_arithmetic_sequence a d ∧ 
    (a 1 + a 2 + a 3 = 4) ∧
    (a 6 + a 7 + a 8 + a 9 = 3) ∧
    (a 5 = 67 / 66) :=
  sorry

end NUMINAMATH_GPT_bamboo_middle_node_capacity_l1974_197436


namespace NUMINAMATH_GPT_value_of_f_l1974_197448

def f (x z : ℕ) (y : ℕ) : ℕ := 2 * x^2 + y - z

theorem value_of_f (y : ℕ) (h1 : f 2 3 y = 100) : f 5 7 y = 138 := by
  sorry

end NUMINAMATH_GPT_value_of_f_l1974_197448


namespace NUMINAMATH_GPT_sabrina_fraction_books_second_month_l1974_197490

theorem sabrina_fraction_books_second_month (total_books : ℕ) (pages_per_book : ℕ) (books_first_month : ℕ) (pages_total_read : ℕ)
  (h_total_books : total_books = 14)
  (h_pages_per_book : pages_per_book = 200)
  (h_books_first_month : books_first_month = 4)
  (h_pages_total_read : pages_total_read = 1000) :
  let total_pages := total_books * pages_per_book
  let pages_first_month := books_first_month * pages_per_book
  let pages_remaining := total_pages - pages_first_month
  let books_remaining := total_books - books_first_month
  let pages_read_first_month := total_pages - pages_total_read
  let pages_read_second_month := pages_read_first_month - pages_first_month
  let books_second_month := pages_read_second_month / pages_per_book
  let fraction_books := books_second_month / books_remaining
  fraction_books = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sabrina_fraction_books_second_month_l1974_197490


namespace NUMINAMATH_GPT_sin_60_eq_sqrt3_div_2_l1974_197427

-- Problem statement translated to Lean
theorem sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_sin_60_eq_sqrt3_div_2_l1974_197427


namespace NUMINAMATH_GPT_inequality_proof_equality_condition_l1974_197426

variable {x1 x2 y1 y2 z1 z2 : ℝ}

-- Conditions
axiom x1_pos : x1 > 0
axiom x2_pos : x2 > 0
axiom x1y1_gz1sq : x1 * y1 > z1 ^ 2
axiom x2y2_gz2sq : x2 * y2 > z2 ^ 2

theorem inequality_proof : 
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) <= 
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) :=
sorry

theorem equality_condition : 
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) = 
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) ↔ 
  (x1 = x2 ∧ y1 = y2 ∧ z1 = z2) :=
sorry

end NUMINAMATH_GPT_inequality_proof_equality_condition_l1974_197426


namespace NUMINAMATH_GPT_log_simplification_l1974_197482

open Real

theorem log_simplification (a b d e z y : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (he : e ≠ 0)
  (ha : a ≠ 0) (hz : z ≠ 0) (hy : y ≠ 0) :
  log (a / b) + log (b / e) + log (e / d) - log (az / dy) = log (dy / z) :=
by
  sorry

end NUMINAMATH_GPT_log_simplification_l1974_197482


namespace NUMINAMATH_GPT_find_difference_l1974_197486

theorem find_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_difference_l1974_197486


namespace NUMINAMATH_GPT_range_of_m_l1974_197412

theorem range_of_m (A : Set ℝ) (m : ℝ) (h : ∃ x, x ∈ A ∩ {x | x ≠ 0}) :
  -4 < m ∧ m < 0 :=
by
  have A_def : A = {x | x^2 + (m+2)*x + 1 = 0} := sorry
  have h_non_empty : ∃ x, x ∈ A ∧ x ≠ 0 := sorry
  have discriminant : (m+2)^2 - 4 < 0 := sorry
  exact ⟨sorry, sorry⟩

end NUMINAMATH_GPT_range_of_m_l1974_197412


namespace NUMINAMATH_GPT_price_of_Microtron_stock_l1974_197430

theorem price_of_Microtron_stock
  (n d : ℕ) (p_d p p_m : ℝ) 
  (h1 : n = 300) 
  (h2 : d = 150) 
  (h3 : p_d = 44) 
  (h4 : p = 40) 
  (h5 : p_m = 36) : 
  (d * p_d + (n - d) * p_m) / n = p := 
sorry

end NUMINAMATH_GPT_price_of_Microtron_stock_l1974_197430
