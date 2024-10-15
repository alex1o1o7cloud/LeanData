import Mathlib

namespace NUMINAMATH_GPT_system_of_equations_solution_l583_58395

theorem system_of_equations_solution :
  ∃ x y z : ℝ, x + y = 1 ∧ y + z = 2 ∧ z + x = 3 ∧ x = 1 ∧ y = 0 ∧ z = 2 :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l583_58395


namespace NUMINAMATH_GPT_union_inter_distrib_inter_union_distrib_l583_58322

section
variables {α : Type*} (A B C : Set α)

-- Problem (a)
theorem union_inter_distrib : A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) :=
sorry

-- Problem (b)
theorem inter_union_distrib : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) :=
sorry
end

end NUMINAMATH_GPT_union_inter_distrib_inter_union_distrib_l583_58322


namespace NUMINAMATH_GPT_geometric_series_properties_l583_58372

theorem geometric_series_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  a 3 = 3 ∧ a 10 = 384 → 
  q = 2 ∧ 
  (∀ n, a n = (3 / 4) * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (3 / 4) * (2 ^ n - 1)) :=
by
  intro h
  -- Proofs will go here, if necessary.
  sorry

end NUMINAMATH_GPT_geometric_series_properties_l583_58372


namespace NUMINAMATH_GPT_unit_cost_calculation_l583_58323

theorem unit_cost_calculation : 
  ∀ (total_cost : ℕ) (ounces : ℕ), total_cost = 84 → ounces = 12 → (total_cost / ounces = 7) :=
by
  intros total_cost ounces h1 h2
  sorry

end NUMINAMATH_GPT_unit_cost_calculation_l583_58323


namespace NUMINAMATH_GPT_original_price_sarees_l583_58394

theorem original_price_sarees (P : ℝ) (h : 0.85 * 0.80 * P = 272) : P = 400 :=
by
  sorry

end NUMINAMATH_GPT_original_price_sarees_l583_58394


namespace NUMINAMATH_GPT_rectangle_y_value_l583_58386

theorem rectangle_y_value (y : ℝ) (h1 : -2 < 6) (h2 : y > 2) 
    (h3 : 8 * (y - 2) = 64) : y = 10 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_y_value_l583_58386


namespace NUMINAMATH_GPT_indeterminate_equation_solution_l583_58396

theorem indeterminate_equation_solution (x y : ℝ) (n : ℕ) :
  (x^2 + (x + 1)^2 = y^2) ↔ 
  (x = 1/4 * ((1 + Real.sqrt 2)^(2*n + 1) + (1 - Real.sqrt 2)^(2*n + 1) - 2) ∧ 
   y = 1/(2 * Real.sqrt 2) * ((1 + Real.sqrt 2)^(2*n + 1) - (1 - Real.sqrt 2)^(2*n + 1))) := 
sorry

end NUMINAMATH_GPT_indeterminate_equation_solution_l583_58396


namespace NUMINAMATH_GPT_average_speed_l583_58354

def dist1 : ℝ := 60
def dist2 : ℝ := 30
def time : ℝ := 2

theorem average_speed : (dist1 + dist2) / time = 45 := by
  sorry

end NUMINAMATH_GPT_average_speed_l583_58354


namespace NUMINAMATH_GPT_maplewood_total_population_l583_58361

-- Define the number of cities
def num_cities : ℕ := 25

-- Define the bounds for the average population
def lower_bound : ℕ := 5200
def upper_bound : ℕ := 5700

-- Define the average population, calculated as the midpoint of the bounds
def average_population : ℕ := (lower_bound + upper_bound) / 2

-- Define the total population as the product of the number of cities and the average population
def total_population : ℕ := num_cities * average_population

-- Theorem statement to prove the total population is 136,250
theorem maplewood_total_population : total_population = 136250 := by
  -- Insert formal proof here
  sorry

end NUMINAMATH_GPT_maplewood_total_population_l583_58361


namespace NUMINAMATH_GPT_hiker_total_distance_l583_58359

theorem hiker_total_distance :
  let day1_distance := 18
  let day1_speed := 3
  let day2_speed := day1_speed + 1
  let day1_time := day1_distance / day1_speed
  let day2_time := day1_time - 1
  let day2_distance := day2_speed * day2_time
  let day3_speed := 5
  let day3_time := 3
  let day3_distance := day3_speed * day3_time
  let total_distance := day1_distance + day2_distance + day3_distance
  total_distance = 53 :=
by
  sorry

end NUMINAMATH_GPT_hiker_total_distance_l583_58359


namespace NUMINAMATH_GPT_fourth_number_in_pascals_triangle_row_15_l583_58311

theorem fourth_number_in_pascals_triangle_row_15 : (Nat.choose 15 3) = 455 :=
by sorry

end NUMINAMATH_GPT_fourth_number_in_pascals_triangle_row_15_l583_58311


namespace NUMINAMATH_GPT_total_shopping_cost_l583_58334

theorem total_shopping_cost 
  (sandwiches : ℕ := 3)
  (sandwich_cost : ℕ := 3)
  (water_bottle : ℕ := 1)
  (water_cost : ℕ := 2)
  : sandwiches * sandwich_cost + water_bottle * water_cost = 11 :=
by
  sorry

end NUMINAMATH_GPT_total_shopping_cost_l583_58334


namespace NUMINAMATH_GPT_no_positive_rational_solution_l583_58324

theorem no_positive_rational_solution :
  ¬ ∃ q : ℚ, 0 < q ∧ q^3 - 10 * q^2 + q - 2021 = 0 :=
by sorry

end NUMINAMATH_GPT_no_positive_rational_solution_l583_58324


namespace NUMINAMATH_GPT_club_truncator_more_wins_than_losses_l583_58333

noncomputable def clubTruncatorWinsProbability : ℚ :=
  let total_matches := 8
  let prob := 1/3
  -- The combinatorial calculations for the balanced outcomes
  let balanced_outcomes := 70 + 560 + 420 + 28 + 1
  let total_outcomes := 3^total_matches
  let prob_balanced := balanced_outcomes / total_outcomes
  let prob_more_wins_or_more_losses := 1 - prob_balanced
  (prob_more_wins_or_more_losses / 2)

theorem club_truncator_more_wins_than_losses : 
  clubTruncatorWinsProbability = 2741 / 6561 := 
by 
  sorry

#check club_truncator_more_wins_than_losses

end NUMINAMATH_GPT_club_truncator_more_wins_than_losses_l583_58333


namespace NUMINAMATH_GPT_prime_product_correct_l583_58339

theorem prime_product_correct 
    (p1 : Nat := 1021031) (pr1 : Prime p1)
    (p2 : Nat := 237019) (pr2 : Prime p2) :
    p1 * p2 = 241940557349 :=
by
  sorry

end NUMINAMATH_GPT_prime_product_correct_l583_58339


namespace NUMINAMATH_GPT_clear_time_is_approximately_7_point_1_seconds_l583_58379

-- Constants for the lengths of the trains in meters
def length_train1 : ℕ := 121
def length_train2 : ℕ := 165

-- Constants for the speeds of the trains in km/h
def speed_train1 : ℕ := 80
def speed_train2 : ℕ := 65

-- Kilometer to meter conversion
def km_to_meter (km : ℕ) : ℕ := km * 1000

-- Hour to second conversion
def hour_to_second (h : ℕ) : ℕ := h * 3600

-- Relative speed of the trains in meters per second
noncomputable def relative_speed_m_per_s : ℕ := 
  (km_to_meter (speed_train1 + speed_train2)) / hour_to_second 1

-- Total distance to be covered in meters
def total_distance : ℕ := length_train1 + length_train2

-- Time to be completely clear of each other in seconds
noncomputable def clear_time : ℝ := total_distance / (relative_speed_m_per_s : ℝ)

theorem clear_time_is_approximately_7_point_1_seconds :
  abs (clear_time - 7.1) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_clear_time_is_approximately_7_point_1_seconds_l583_58379


namespace NUMINAMATH_GPT_geometric_sequence_sum_l583_58392

variable (a : ℕ → ℝ)

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_sum (h1 : geometric_sequence a)
  (h2 : a 1 > 0)
  (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) :
  a 3 + a 5 = 6 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l583_58392


namespace NUMINAMATH_GPT_calculate_length_X_l583_58375

theorem calculate_length_X 
  (X : ℝ)
  (h1 : 3 + X + 4 = 5 + 7 + X)
  : X = 5 :=
sorry

end NUMINAMATH_GPT_calculate_length_X_l583_58375


namespace NUMINAMATH_GPT_find_x_proportionally_l583_58306

theorem find_x_proportionally (k m x z : ℝ) (h1 : ∀ y, x = k * y^2) (h2 : ∀ z, y = m / (Real.sqrt z)) (h3 : x = 7 ∧ z = 16) :
  ∃ x, x = 7 / 9 := by
  sorry

end NUMINAMATH_GPT_find_x_proportionally_l583_58306


namespace NUMINAMATH_GPT_minimum_a_condition_l583_58346

-- Define the quadratic function
def f (a x : ℝ) := x^2 + a * x + 1

-- Define the condition that the function remains non-negative in the open interval (0, 1/2)
def f_non_negative_in_interval (a : ℝ) : Prop :=
  ∀ (x : ℝ), 0 < x ∧ x < 1 / 2 → f a x ≥ 0

-- State the theorem that the minimum value for a with the given condition is -5/2
theorem minimum_a_condition : ∀ (a : ℝ), f_non_negative_in_interval a → a ≥ -5 / 2 :=
by sorry

end NUMINAMATH_GPT_minimum_a_condition_l583_58346


namespace NUMINAMATH_GPT_odd_function_alpha_l583_58382
open Real

noncomputable def f (x : ℝ) : ℝ :=
  cos x * (sin x + sqrt 3 * cos x) - sqrt 3 / 2

noncomputable def g (x : ℝ) (α : ℝ) : ℝ :=
  f (x + α)

theorem odd_function_alpha (α : ℝ) (a : α > 0) :
  (∀ x : ℝ, g x α = - g (-x) α) ↔ 
  ∃ k : ℕ, α = (2 * k - 1) * π / 6 := sorry

end NUMINAMATH_GPT_odd_function_alpha_l583_58382


namespace NUMINAMATH_GPT_participated_in_both_l583_58366

-- Define the conditions
def total_students := 40
def math_competition := 31
def physics_competition := 20
def not_participating := 8

-- Define number of students participated in both competitions
def both_competitions := 59 - total_students

-- Theorem statement
theorem participated_in_both : both_competitions = 19 := 
sorry

end NUMINAMATH_GPT_participated_in_both_l583_58366


namespace NUMINAMATH_GPT_find_m_l583_58399

theorem find_m (x y m : ℤ) (h1 : 3 * x + 4 * y = 7) (h2 : 5 * x - 4 * y = m) (h3 : x + y = 0) : m = -63 := by
  sorry

end NUMINAMATH_GPT_find_m_l583_58399


namespace NUMINAMATH_GPT_minimize_quadratic_l583_58329

def f (x : ℝ) := 3 * x^2 - 18 * x + 7

theorem minimize_quadratic : ∃ x : ℝ, f x = -20 ∧ ∀ y : ℝ, f y ≥ -20 := by
  sorry

end NUMINAMATH_GPT_minimize_quadratic_l583_58329


namespace NUMINAMATH_GPT_find_k_l583_58367

theorem find_k (x k : ℤ) (h : 2 * k - x = 2) (hx : x = -4) : k = -1 :=
by
  rw [hx] at h
  -- Substituting x = -4 into the equation
  sorry  -- Skipping further proof steps

end NUMINAMATH_GPT_find_k_l583_58367


namespace NUMINAMATH_GPT_max_value_negative_one_l583_58349

theorem max_value_negative_one (f : ℝ → ℝ) (hx : ∀ x, x < 1 → f x ≤ -1) :
  ∀ x, x < 1 → ∃ M, (∀ y, y < 1 → f y ≤ M) ∧ f x = M :=
sorry

end NUMINAMATH_GPT_max_value_negative_one_l583_58349


namespace NUMINAMATH_GPT_james_used_5_containers_l583_58336

-- Conditions
def initial_balls : ℕ := 100
def balls_given_away : ℕ := initial_balls / 2
def remaining_balls : ℕ := initial_balls - balls_given_away
def balls_per_container : ℕ := 10

-- Question (statement of the theorem to prove)
theorem james_used_5_containers : (remaining_balls / balls_per_container) = 5 := by
  sorry

end NUMINAMATH_GPT_james_used_5_containers_l583_58336


namespace NUMINAMATH_GPT_person_B_days_l583_58398

theorem person_B_days (A_days : ℕ) (combined_work : ℚ) (x : ℕ) : 
  A_days = 30 → combined_work = (1 / 6) → 3 * (1 / 30 + 1 / x) = combined_work → x = 45 :=
by
  intros hA hCombined hEquation
  sorry

end NUMINAMATH_GPT_person_B_days_l583_58398


namespace NUMINAMATH_GPT_sum_of_integers_equals_75_l583_58331

theorem sum_of_integers_equals_75 
  (n m : ℤ) 
  (h1 : n * (n + 1) * (n + 2) = 924) 
  (h2 : m * (m + 1) * (m + 2) * (m + 3) = 924) 
  (sum_seven_integers : ℤ := n + (n + 1) + (n + 2) + m + (m + 1) + (m + 2) + (m + 3)) :
  sum_seven_integers = 75 := 
  sorry

end NUMINAMATH_GPT_sum_of_integers_equals_75_l583_58331


namespace NUMINAMATH_GPT_donny_spending_l583_58368

theorem donny_spending :
  (15 + 28 + 13) / 2 = 28 :=
by
  sorry

end NUMINAMATH_GPT_donny_spending_l583_58368


namespace NUMINAMATH_GPT_time_ratio_A_to_B_l583_58301

theorem time_ratio_A_to_B (T_A T_B : ℝ) (hB : T_B = 36) (hTogether : 1 / T_A + 1 / T_B = 1 / 6) : T_A / T_B = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_time_ratio_A_to_B_l583_58301


namespace NUMINAMATH_GPT_part1_part2_l583_58315

theorem part1 (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 2) : a^2 + b^2 = 21 :=
  sorry

theorem part2 (a b : ℝ) (h1 : a + b = 10) (h2 : a^2 + b^2 = 50^2) : a * b = -1200 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l583_58315


namespace NUMINAMATH_GPT_solve_for_y_l583_58353

theorem solve_for_y (x y : ℝ) (hx : x > 1) (hy : y > 1) (h1 : 1 / x + 1 / y = 1) (h2 : x * y = 9) :
  y = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l583_58353


namespace NUMINAMATH_GPT_sam_total_spent_l583_58316

-- Define the values of a penny and a dime in dollars
def penny_value : ℝ := 0.01
def dime_value : ℝ := 0.10

-- Define what Sam spent
def friday_spent : ℝ := 2 * penny_value
def saturday_spent : ℝ := 12 * dime_value

-- Define total spent
def total_spent : ℝ := friday_spent + saturday_spent

theorem sam_total_spent : total_spent = 1.22 := 
by
  -- The following is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_sam_total_spent_l583_58316


namespace NUMINAMATH_GPT_quadratic_min_value_l583_58385

theorem quadratic_min_value : ∀ x : ℝ, x^2 - 6 * x + 13 ≥ 4 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_min_value_l583_58385


namespace NUMINAMATH_GPT_smallest_k_remainder_1_l583_58317

theorem smallest_k_remainder_1
  (k : ℤ) : 
  (k > 1) ∧ (k % 13 = 1) ∧ (k % 8 = 1) ∧ (k % 4 = 1)
  ↔ k = 105 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_remainder_1_l583_58317


namespace NUMINAMATH_GPT_total_marks_l583_58338

variable (A M SS Mu : ℝ)

-- Conditions
def cond1 : Prop := M = A - 20
def cond2 : Prop := SS = Mu + 10
def cond3 : Prop := Mu = 70
def cond4 : Prop := M = (9 / 10) * A

-- Theorem statement
theorem total_marks (A M SS Mu : ℝ) (h1 : cond1 A M)
                                      (h2 : cond2 SS Mu)
                                      (h3 : cond3 Mu)
                                      (h4 : cond4 A M) :
    A + M + SS + Mu = 530 :=
by 
  sorry

end NUMINAMATH_GPT_total_marks_l583_58338


namespace NUMINAMATH_GPT_average_speed_before_increase_l583_58384

-- Definitions for the conditions
def t_before := 12   -- Travel time before the speed increase in hours
def t_after := 10    -- Travel time after the speed increase in hours
def speed_diff := 20 -- Speed difference between before and after in km/h

-- Variable for the speed before increase
variable (s_before : ℕ) -- Average speed before the speed increase in km/h

-- Definitions for the speeds
def s_after := s_before + speed_diff -- Average speed after the speed increase in km/h

-- Equations derived from the problem conditions
def dist_eqn_before := s_before * t_before
def dist_eqn_after := s_after * t_after

-- The proof problem stated in Lean
theorem average_speed_before_increase : dist_eqn_before = dist_eqn_after → s_before = 100 := by
  sorry

end NUMINAMATH_GPT_average_speed_before_increase_l583_58384


namespace NUMINAMATH_GPT_solve_for_x_l583_58312

theorem solve_for_x (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ -2) : 
  (x + 5) / (x - 3) = (x - 4) / (x + 2) → x = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l583_58312


namespace NUMINAMATH_GPT_cardinals_home_runs_second_l583_58355

-- Define the conditions
def cubs_home_runs_third : ℕ := 2
def cubs_home_runs_fifth : ℕ := 1
def cubs_home_runs_eighth : ℕ := 2
def cubs_total_home_runs := cubs_home_runs_third + cubs_home_runs_fifth + cubs_home_runs_eighth
def cubs_more_than_cardinals : ℕ := 3
def cardinals_home_runs_fifth : ℕ := 1

-- Define the proof problem
theorem cardinals_home_runs_second :
  (cubs_total_home_runs = cardinals_total_home_runs + cubs_more_than_cardinals) →
  (cardinals_total_home_runs - cardinals_home_runs_fifth = 1) :=
sorry

end NUMINAMATH_GPT_cardinals_home_runs_second_l583_58355


namespace NUMINAMATH_GPT_negation_of_p_l583_58380

theorem negation_of_p : 
  (¬(∀ x : ℝ, |x| < 0)) ↔ (∃ x : ℝ, |x| ≥ 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_of_p_l583_58380


namespace NUMINAMATH_GPT_tangent_line_parallel_to_x_axis_l583_58364

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

noncomputable def f_derivative (x : ℝ) : ℝ := (1 - Real.log x) / (x^2)

theorem tangent_line_parallel_to_x_axis :
  ∀ x₀ : ℝ, 
  f_derivative x₀ = 0 → 
  f x₀ = 1 / Real.exp 1 :=
by
  intro x₀ h_deriv_zero
  sorry

end NUMINAMATH_GPT_tangent_line_parallel_to_x_axis_l583_58364


namespace NUMINAMATH_GPT_carla_bought_marbles_l583_58313

def starting_marbles : ℕ := 2289
def total_marbles : ℝ := 2778.0

theorem carla_bought_marbles : (total_marbles - starting_marbles) = 489 := 
by
  sorry

end NUMINAMATH_GPT_carla_bought_marbles_l583_58313


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l583_58363

theorem arithmetic_sequence_problem
  (a : ℕ → ℤ)  -- the arithmetic sequence
  (S : ℕ → ℤ)  -- the sum of the first n terms
  (m : ℕ)      -- the m in question
  (h1 : a (m - 1) + a (m + 1) - a m ^ 2 = 0)
  (h2 : S (2 * m - 1) = 18) :
  m = 5 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l583_58363


namespace NUMINAMATH_GPT_number_of_teachers_l583_58327

theorem number_of_teachers (total_population sample_size teachers_within_sample students_within_sample : ℕ) 
    (h_total_population : total_population = 3000) 
    (h_sample_size : sample_size = 150) 
    (h_students_within_sample : students_within_sample = 140) 
    (h_teachers_within_sample : teachers_within_sample = sample_size - students_within_sample) 
    (h_ratio : (total_population - students_within_sample) * sample_size = total_population * teachers_within_sample) : 
    total_population - students_within_sample = 200 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_teachers_l583_58327


namespace NUMINAMATH_GPT_correct_fraction_l583_58350

theorem correct_fraction (x y : ℤ) (h : (5 / 6 : ℚ) * 384 = (x / y : ℚ) * 384 + 200) : x / y = 5 / 16 :=
by
  sorry

end NUMINAMATH_GPT_correct_fraction_l583_58350


namespace NUMINAMATH_GPT_bus_tour_total_sales_l583_58374

noncomputable def total_sales (total_tickets sold_senior_tickets : Nat) (cost_senior_ticket cost_regular_ticket : Nat) : Nat :=
  let sold_regular_tickets := total_tickets - sold_senior_tickets
  let sales_senior := sold_senior_tickets * cost_senior_ticket
  let sales_regular := sold_regular_tickets * cost_regular_ticket
  sales_senior + sales_regular

theorem bus_tour_total_sales :
  total_sales 65 24 10 15 = 855 := by
    sorry

end NUMINAMATH_GPT_bus_tour_total_sales_l583_58374


namespace NUMINAMATH_GPT_range_of_a_l583_58345

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ↔ 0 ≤ a ∧ a < 4 := sorry

end NUMINAMATH_GPT_range_of_a_l583_58345


namespace NUMINAMATH_GPT_current_population_is_15336_l583_58381

noncomputable def current_population : ℝ :=
  let growth_rate := 1.28
  let future_population : ℝ := 25460.736
  let years := 2
  future_population / (growth_rate ^ years)

theorem current_population_is_15336 :
  current_population = 15536 := sorry

end NUMINAMATH_GPT_current_population_is_15336_l583_58381


namespace NUMINAMATH_GPT_number_of_real_values_p_l583_58348

theorem number_of_real_values_p (p : ℝ) :
  (∀ p: ℝ, x^2 - (p + 1) * x + (p + 1)^2 = 0 -> (p + 1) ^ 2 = 0) ↔ p = -1 := by
  sorry

end NUMINAMATH_GPT_number_of_real_values_p_l583_58348


namespace NUMINAMATH_GPT_term_containing_x3_l583_58304

-- Define the problem statement in Lean 4
theorem term_containing_x3 (a : ℝ) (x : ℝ) (hx : x ≠ 0) 
(h_sum_coeff : (2 + a) ^ 5 = 0) :
  (2 * x + a / x) ^ 5 = -160 * x ^ 3 :=
sorry

end NUMINAMATH_GPT_term_containing_x3_l583_58304


namespace NUMINAMATH_GPT_compute_expression_l583_58383

theorem compute_expression : 1 + 6 * 2 - 3 + 5 * 4 / 2 = 20 :=
by sorry

end NUMINAMATH_GPT_compute_expression_l583_58383


namespace NUMINAMATH_GPT_difference_in_biking_distance_l583_58365

def biking_rate_alberto : ℕ := 18  -- miles per hour
def biking_rate_bjorn : ℕ := 20    -- miles per hour

def start_time_alberto : ℕ := 9    -- a.m.
def start_time_bjorn : ℕ := 10     -- a.m.

def end_time : ℕ := 15            -- 3 p.m. in 24-hour format

def biking_duration_alberto : ℕ := end_time - start_time_alberto
def biking_duration_bjorn : ℕ := end_time - start_time_bjorn

def distance_alberto : ℕ := biking_rate_alberto * biking_duration_alberto
def distance_bjorn : ℕ := biking_rate_bjorn * biking_duration_bjorn

theorem difference_in_biking_distance : 
  (distance_alberto - distance_bjorn) = 8 := by
  sorry

end NUMINAMATH_GPT_difference_in_biking_distance_l583_58365


namespace NUMINAMATH_GPT_area_of_trapezoid_MBCN_l583_58352

variables {AB BC MN : ℝ}
variables {Area_ABCD Area_MBCN : ℝ}
variables {Height : ℝ}

-- Given conditions
def cond1 : Area_ABCD = 40 := sorry
def cond2 : AB = 8 := sorry
def cond3 : BC = 5 := sorry
def cond4 : MN = 2 := sorry
def cond5 : Height = 5 := sorry

-- Define the theorem to be proven
theorem area_of_trapezoid_MBCN : 
  Area_ABCD = AB * BC → MN + BC = 6 → Height = 5 →
  Area_MBCN = (1/2) * (MN + BC) * Height → 
  Area_MBCN = 15 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_area_of_trapezoid_MBCN_l583_58352


namespace NUMINAMATH_GPT_binom_20_10_eq_184756_l583_58326

theorem binom_20_10_eq_184756 
  (h1 : Nat.choose 19 9 = 92378)
  (h2 : Nat.choose 19 10 = Nat.choose 19 9) : 
  Nat.choose 20 10 = 184756 := 
by
  sorry

end NUMINAMATH_GPT_binom_20_10_eq_184756_l583_58326


namespace NUMINAMATH_GPT_average_score_10_students_l583_58332

theorem average_score_10_students (x : ℝ)
  (h1 : 15 * 70 = 1050)
  (h2 : 25 * 78 = 1950)
  (h3 : 15 * 70 + 10 * x = 25 * 78) :
  x = 90 :=
sorry

end NUMINAMATH_GPT_average_score_10_students_l583_58332


namespace NUMINAMATH_GPT_problem_statement_l583_58307

def f (x : ℝ) : ℝ := 4 * x ^ 2 - 6

theorem problem_statement : f (f (-1)) = 10 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l583_58307


namespace NUMINAMATH_GPT_black_lambs_correct_l583_58373

-- Define the total number of lambs
def total_lambs : ℕ := 6048

-- Define the number of white lambs
def white_lambs : ℕ := 193

-- Define the number of black lambs
def black_lambs : ℕ := total_lambs - white_lambs

-- The goal is to prove that the number of black lambs is 5855
theorem black_lambs_correct : black_lambs = 5855 := by
  sorry

end NUMINAMATH_GPT_black_lambs_correct_l583_58373


namespace NUMINAMATH_GPT_angle_ADE_l583_58330

-- Definitions and conditions
variable (x : ℝ)

def angle_ABC := 60
def angle_CAD := x
def angle_BAD := x
def angle_BCA := 120 - 2 * x
def angle_DCE := 180 - (120 - 2 * x)

-- Theorem statement
theorem angle_ADE (x : ℝ) : angle_CAD x = x → angle_BAD x = x → angle_ABC = 60 → 
                            angle_DCE x = 180 - angle_BCA x → 
                            120 - 3 * x = 120 - 3 * x := 
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_angle_ADE_l583_58330


namespace NUMINAMATH_GPT_measure_of_C_l583_58340

-- Define angles and their magnitudes
variables (A B C X : Type) [LinearOrder C]
def angle_measure (angle : Type) : ℕ := sorry
def parallel (l1 l2 : Type) : Prop := sorry
def transversal (l1 l2 l3 : Type) : Prop := sorry
def alternate_interior (angle1 angle2 : Type) : Prop := sorry
def adjacent (angle1 angle2 : Type) : Prop := sorry
def complementary (angle1 angle2 : Type) : Prop := sorry

-- The given conditions
axiom h1 : parallel A X
axiom h2 : transversal A B X
axiom h3 : angle_measure A = 85
axiom h4 : angle_measure B = 35
axiom h5 : alternate_interior C A
axiom h6 : complementary B X
axiom h7 : adjacent C X

-- Define the proof problem
theorem measure_of_C : angle_measure C = 85 :=
by {
  -- The proof goes here, skipping with sorry
  sorry
}

end NUMINAMATH_GPT_measure_of_C_l583_58340


namespace NUMINAMATH_GPT_base_12_addition_l583_58302

theorem base_12_addition (A B: ℕ) (hA: A = 10) (hB: B = 11) : 
  8 * 12^2 + A * 12 + 2 + (3 * 12^2 + B * 12 + 7) = 1 * 12^3 + 0 * 12^2 + 9 * 12 + 9 := 
by
  sorry

end NUMINAMATH_GPT_base_12_addition_l583_58302


namespace NUMINAMATH_GPT_cassandra_makes_four_pies_l583_58318

-- Define the number of dozens and respective apples per dozen
def dozens : ℕ := 4
def apples_per_dozen : ℕ := 12

-- Define the total number of apples
def total_apples : ℕ := dozens * apples_per_dozen

-- Define apples per slice and slices per pie
def apples_per_slice : ℕ := 2
def slices_per_pie : ℕ := 6

-- Calculate the number of slices and number of pies based on conditions
def total_slices : ℕ := total_apples / apples_per_slice
def total_pies : ℕ := total_slices / slices_per_pie

-- Prove that the number of pies is 4
theorem cassandra_makes_four_pies : total_pies = 4 := by
  sorry

end NUMINAMATH_GPT_cassandra_makes_four_pies_l583_58318


namespace NUMINAMATH_GPT_beaker_volume_l583_58377

theorem beaker_volume {a b c d e f g h i j : ℝ} (h₁ : a = 7) (h₂ : b = 4) (h₃ : c = 5)
                      (h₄ : d = 4) (h₅ : e = 6) (h₆ : f = 8) (h₇ : g = 7)
                      (h₈ : h = 3) (h₉ : i = 9) (h₁₀ : j = 6) :
  (a + b + c + d + e + f + g + h + i + j) / 5 = 11.8 :=
by
  sorry

end NUMINAMATH_GPT_beaker_volume_l583_58377


namespace NUMINAMATH_GPT_knife_value_l583_58390

def sheep_sold (n : ℕ) : ℕ := n * n

def valid_units_digits (m : ℕ) : Bool :=
  (m ^ 2 = 16) ∨ (m ^ 2 = 36)

theorem knife_value (n : ℕ) (k : ℕ) (m : ℕ) (H1 : sheep_sold n = n * n) (H2 : n = 10 * k + m) (H3 : valid_units_digits m = true) :
  2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_knife_value_l583_58390


namespace NUMINAMATH_GPT_range_of_f_lt_0_l583_58351

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

variable (f : ℝ → ℝ)
variable (h_odd : is_odd f)
variable (h_decreasing : decreasing_on f (Set.Iic 0))
variable (h_at_2 : f 2 = 0)

theorem range_of_f_lt_0 : ∀ x, x ∈ (Set.Ioo (-2) 0 ∪ Set.Ioi 2) → f x < 0 := by
  sorry

end NUMINAMATH_GPT_range_of_f_lt_0_l583_58351


namespace NUMINAMATH_GPT_sector_area_max_angle_l583_58305

theorem sector_area_max_angle (r : ℝ) (θ : ℝ) (h : 0 < r ∧ r < 10) 
  (H : 2 * r + r * θ = 20) : θ = 2 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_max_angle_l583_58305


namespace NUMINAMATH_GPT_Jen_distance_from_start_l583_58310

-- Define the rate of Jen's walking (in miles per hour)
def walking_rate : ℝ := 4

-- Define the time Jen walks forward (in hours)
def forward_time : ℝ := 2

-- Define the time Jen walks back (in hours)
def back_time : ℝ := 1

-- Define the distance walked forward
def distance_forward : ℝ := walking_rate * forward_time

-- Define the distance walked back
def distance_back : ℝ := walking_rate * back_time

-- Define the net distance from the starting point
def net_distance : ℝ := distance_forward - distance_back

-- Theorem stating the net distance from the starting point is 4.0 miles
theorem Jen_distance_from_start : net_distance = 4.0 := by
  sorry

end NUMINAMATH_GPT_Jen_distance_from_start_l583_58310


namespace NUMINAMATH_GPT_jill_has_6_more_dolls_than_jane_l583_58328

theorem jill_has_6_more_dolls_than_jane
  (total_dolls : ℕ) 
  (jane_dolls : ℕ) 
  (more_dolls_than : ℕ → ℕ → Prop)
  (h1 : total_dolls = 32) 
  (h2 : jane_dolls = 13) 
  (jill_dolls : ℕ)
  (h3 : more_dolls_than jill_dolls jane_dolls) :
  (jill_dolls - jane_dolls) = 6 :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_jill_has_6_more_dolls_than_jane_l583_58328


namespace NUMINAMATH_GPT_number_of_shampoos_l583_58389

-- Define necessary variables in conditions
def h := 10 -- time spent hosing in minutes
def t := 55 -- total time spent cleaning in minutes
def p := 15 -- time per shampoo in minutes

-- State the theorem
theorem number_of_shampoos (h t p : Nat) (h_val : h = 10) (t_val : t = 55) (p_val : p = 15) :
    (t - h) / p = 3 := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_number_of_shampoos_l583_58389


namespace NUMINAMATH_GPT_betty_needs_more_flies_l583_58308

def flies_per_day := 2
def days_per_week := 7
def flies_needed_per_week := flies_per_day * days_per_week

def flies_caught_morning := 5
def flies_caught_afternoon := 6
def fly_escaped := 1

def flies_caught_total := flies_caught_morning + flies_caught_afternoon - fly_escaped

theorem betty_needs_more_flies : 
  flies_needed_per_week - flies_caught_total = 4 := by
  sorry

end NUMINAMATH_GPT_betty_needs_more_flies_l583_58308


namespace NUMINAMATH_GPT_value_of_expression_is_correct_l583_58320

-- Defining the sub-expressions as Lean terms
def three_squared : ℕ := 3^2
def intermediate_result : ℕ := three_squared - 3
def final_result : ℕ := intermediate_result^2

-- The statement we need to prove
theorem value_of_expression_is_correct : final_result = 36 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_is_correct_l583_58320


namespace NUMINAMATH_GPT_no_real_solutions_for_identical_lines_l583_58387

theorem no_real_solutions_for_identical_lines :
  ¬∃ (a d : ℝ), (∀ x y : ℝ, 5 * x + a * y + d = 0 ↔ 2 * d * x - 3 * y + 8 = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_for_identical_lines_l583_58387


namespace NUMINAMATH_GPT_graph_is_empty_l583_58370

theorem graph_is_empty : ∀ (x y : ℝ), 3 * x^2 + y^2 - 9 * x - 4 * y + 17 ≠ 0 :=
by
  intros x y
  sorry

end NUMINAMATH_GPT_graph_is_empty_l583_58370


namespace NUMINAMATH_GPT_explicit_formula_inequality_solution_l583_58319

noncomputable def f (x : ℝ) : ℝ := (x : ℝ) / (x^2 + 1)

-- Given conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a < x → y < b → x < y → f x < f y
def f_half_eq_two_fifths : Prop := f (1/2) = 2/5

-- Questions rewritten as goals
theorem explicit_formula :
  odd_function f ∧ increasing_on_interval f (-1) 1 ∧ f_half_eq_two_fifths →
  ∀ x, f x = x / (x^2 + 1) := by 
sorry

theorem inequality_solution :
  odd_function f ∧ increasing_on_interval f (-1) 1 →
  ∀ t, (f (t - 1) + f t < 0) ↔ (0 < t ∧ t < 1/2) := by 
sorry

end NUMINAMATH_GPT_explicit_formula_inequality_solution_l583_58319


namespace NUMINAMATH_GPT_brad_must_make_5_trips_l583_58342

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r ^ 2 * h

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ :=
  Real.pi * r ^ 2 * h

theorem brad_must_make_5_trips (r_barrel h_barrel r_bucket h_bucket : ℝ)
    (h1 : r_barrel = 10) (h2 : h_barrel = 15) (h3 : r_bucket = 10) (h4 : h_bucket = 10) :
    let trips := volume_of_cylinder r_barrel h_barrel / volume_of_cone r_bucket h_bucket
    let trips_needed := Int.ceil trips
    trips_needed = 5 := 
by
  sorry

end NUMINAMATH_GPT_brad_must_make_5_trips_l583_58342


namespace NUMINAMATH_GPT_number_of_minibuses_l583_58397

theorem number_of_minibuses (total_students : ℕ) (capacity : ℕ) (h : total_students = 48) (h_capacity : capacity = 8) : 
  ∃ minibuses, minibuses = (total_students + capacity - 1) / capacity ∧ minibuses = 7 :=
by
  have h1 : (48 + 8 - 1) = 55 := by simp [h, h_capacity]
  have h2 : 55 / 8 = 6 := by simp [h, h_capacity]
  use 7
  sorry

end NUMINAMATH_GPT_number_of_minibuses_l583_58397


namespace NUMINAMATH_GPT_election_votes_l583_58341

theorem election_votes (V : ℕ) (h1 : ∃ Vb, Vb = 2509 ∧ (0.8 * V : ℝ) = (Vb + 0.15 * (V : ℝ)) + Vb) : V = 7720 :=
sorry

end NUMINAMATH_GPT_election_votes_l583_58341


namespace NUMINAMATH_GPT_transformed_equation_sum_l583_58321

theorem transformed_equation_sum (a b : ℝ) (h_eqn : ∀ x : ℝ, x^2 - 6 * x - 5 = 0 ↔ (x + a)^2 = b) :
  a + b = 11 :=
sorry

end NUMINAMATH_GPT_transformed_equation_sum_l583_58321


namespace NUMINAMATH_GPT_math_problem_l583_58369

theorem math_problem (a b c d m : ℝ) (h1 : a = -b) (h2 : a ≠ 0) (h3 : c * d = 1)
  (h4 : m = -1 ∨ m = 3) : (a + b) * (c / d) + m * c * d + (b / a) = 2 ∨ (a + b) * (c / d) + m * c * d + (b / a) = -2 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l583_58369


namespace NUMINAMATH_GPT_ellipse_foci_l583_58337

noncomputable def focal_coordinates (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 - b^2)

-- Given the equation of the ellipse: x^2 / a^2 + y^2 / b^2 = 1
def ellipse_equation (x y a b : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1

-- Proposition stating that if the ellipse equation holds for a=√5 and b=2, then the foci are at (± c, 0)
theorem ellipse_foci (x y : ℝ) (h : ellipse_equation x y (Real.sqrt 5) 2) :
  y = 0 ∧ (x = 1 ∨ x = -1) :=
sorry

end NUMINAMATH_GPT_ellipse_foci_l583_58337


namespace NUMINAMATH_GPT_exists_infinitely_many_N_l583_58391

open Set

-- Conditions: Definition of the initial set S_0 and recursive sets S_n
variable {S_0 : Set ℕ} (h0 : Set.Finite S_0) -- S_0 is a finite set of positive integers
variable (S : ℕ → Set ℕ) 
(has_S : ∀ n, ∀ a, a ∈ S (n+1) ↔ (a-1 ∈ S n ∧ a ∉ S n ∨ a-1 ∉ S n ∧ a ∈ S n))

-- Main theorem: Proving the existence of infinitely many integers N such that 
-- S_N = S_0 ∪ {N + a : a ∈ S_0}
theorem exists_infinitely_many_N : 
  ∃ᶠ N in at_top, S N = S_0 ∪ {n | ∃ a ∈ S_0, n = N + a} := 
sorry

end NUMINAMATH_GPT_exists_infinitely_many_N_l583_58391


namespace NUMINAMATH_GPT_ratio_youngest_sister_to_yvonne_l583_58314

def laps_yvonne := 10
def laps_joel := 15
def joel_ratio := 3

theorem ratio_youngest_sister_to_yvonne
  (laps_yvonne : ℕ)
  (laps_joel : ℕ)
  (joel_ratio : ℕ)
  (H_joel : laps_joel = 3 * (laps_yvonne / joel_ratio))
  : (laps_joel / joel_ratio) = laps_yvonne / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_youngest_sister_to_yvonne_l583_58314


namespace NUMINAMATH_GPT_recruits_count_l583_58335

def x := 50
def y := 100
def z := 170

theorem recruits_count :
  ∃ n : ℕ, n = 211 ∧ (∀ a b c : ℕ, (b = 4 * a ∨ a = 4 * c ∨ c = 4 * b) → (b + 100 = a + 150) ∨ (a + 50 = c + 150) ∨ (c + 170 = b + 100)) :=
sorry

end NUMINAMATH_GPT_recruits_count_l583_58335


namespace NUMINAMATH_GPT_distance_city_A_to_C_l583_58303

variable (V_E V_F : ℝ) -- Define the average speeds of Eddy and Freddy
variable (time : ℝ) -- Define the time variable

-- Given conditions
def eddy_time : time = 3 := sorry
def freddy_time : time = 3 := sorry
def eddy_distance : ℝ := 600
def speed_ratio : V_E = 2 * V_F := sorry

-- Derived condition for Eddy's speed
def eddy_speed : V_E = eddy_distance / time := sorry

-- Derived conclusion for Freddy's distance
theorem distance_city_A_to_C (time : ℝ) (V_F : ℝ) : V_F * time = 300 := 
by 
  sorry

end NUMINAMATH_GPT_distance_city_A_to_C_l583_58303


namespace NUMINAMATH_GPT_circle_radius_l583_58344

theorem circle_radius (x y : ℝ) : x^2 + y^2 - 2*y = 0 → ∃ r : ℝ, r = 1 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l583_58344


namespace NUMINAMATH_GPT_union_M_N_eq_l583_58358

def M : Set ℝ := {x | x^2 - 4 * x < 0}
def N : Set ℝ := {0, 4}

theorem union_M_N_eq : M ∪ N = Set.Icc 0 4 := 
  by
    sorry

end NUMINAMATH_GPT_union_M_N_eq_l583_58358


namespace NUMINAMATH_GPT_min_value_fraction_l583_58376

theorem min_value_fraction (a b : ℝ) (h₀ : a > b) (h₁ : a * b = 1) :
  ∃ c, c = (2 * Real.sqrt 2) ∧ (a^2 + b^2) / (a - b) ≥ c :=
by sorry

end NUMINAMATH_GPT_min_value_fraction_l583_58376


namespace NUMINAMATH_GPT_curves_tangent_at_m_eq_two_l583_58378

-- Definitions of the ellipsoid and hyperbola equations.
def ellipse (x y : ℝ) : Prop := x^2 + 2 * y^2 = 2
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 1)^2 = 1

-- The proposition to be proved.
theorem curves_tangent_at_m_eq_two :
  ∃ m : ℝ, (∀ x y : ℝ, ellipse x y ∧ hyperbola x y m → m = 2) :=
sorry

end NUMINAMATH_GPT_curves_tangent_at_m_eq_two_l583_58378


namespace NUMINAMATH_GPT_polygon_with_15_diagonals_has_7_sides_l583_58325

-- Define the number of diagonals formula for a regular polygon
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The proof statement
theorem polygon_with_15_diagonals_has_7_sides :
  ∃ n : ℕ, number_of_diagonals n = 15 ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_polygon_with_15_diagonals_has_7_sides_l583_58325


namespace NUMINAMATH_GPT_blue_paint_needed_l583_58371

theorem blue_paint_needed (total_cans : ℕ) (blue_ratio : ℕ) (yellow_ratio : ℕ)
  (h_ratio: blue_ratio = 5) (h_yellow_ratio: yellow_ratio = 3) (h_total: total_cans = 45) : 
  ⌊total_cans * (blue_ratio : ℝ) / (blue_ratio + yellow_ratio)⌋ = 28 :=
by
  sorry

end NUMINAMATH_GPT_blue_paint_needed_l583_58371


namespace NUMINAMATH_GPT_geometric_sequence_collinear_vectors_l583_58393

theorem geometric_sequence_collinear_vectors (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : ∀ n, a (n + 1) = q * a n)
  (a2 a3 : ℝ)
  (h_a2 : a 2 = a2)
  (h_a3 : a 3 = a3)
  (h_parallel : 3 * a2 = 2 * a3) :
  (a2 + a 4) / (a3 + a 5) = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_collinear_vectors_l583_58393


namespace NUMINAMATH_GPT_find_natural_number_l583_58309

-- Definitions reflecting the conditions and result
def is_sum_of_two_squares (n : ℕ) := ∃ a b : ℕ, a * a + b * b = n

def has_exactly_one_not_sum_of_two_squares (n : ℕ) :=
  ∃! x : ℤ, ¬is_sum_of_two_squares (x.natAbs % n)

theorem find_natural_number (n : ℕ) (h : n ≥ 2) : 
  has_exactly_one_not_sum_of_two_squares n ↔ n = 4 :=
sorry

end NUMINAMATH_GPT_find_natural_number_l583_58309


namespace NUMINAMATH_GPT_kylie_coins_count_l583_58347

theorem kylie_coins_count 
  (P : ℕ) 
  (from_brother : ℕ) 
  (from_father : ℕ) 
  (given_to_Laura : ℕ) 
  (coins_left : ℕ) 
  (h1 : from_brother = 13) 
  (h2 : from_father = 8) 
  (h3 : given_to_Laura = 21) 
  (h4 : coins_left = 15) : (P + from_brother + from_father) - given_to_Laura = coins_left → P = 15 :=
by
  sorry

end NUMINAMATH_GPT_kylie_coins_count_l583_58347


namespace NUMINAMATH_GPT_negation_correct_l583_58343

namespace NegationProof

-- Define the original proposition 
def orig_prop : Prop := ∃ x : ℝ, x ≤ 0

-- Define the negation of the original proposition
def neg_prop : Prop := ∀ x : ℝ, x > 0

-- The theorem we need to prove
theorem negation_correct : ¬ orig_prop = neg_prop := by
  sorry

end NegationProof

end NUMINAMATH_GPT_negation_correct_l583_58343


namespace NUMINAMATH_GPT_probability_one_defective_l583_58356

def total_bulbs : ℕ := 20
def defective_bulbs : ℕ := 4
def non_defective_bulbs : ℕ := total_bulbs - defective_bulbs
def probability_non_defective_both : ℚ := (16 / 20) * (15 / 19)
def probability_at_least_one_defective : ℚ := 1 - probability_non_defective_both

theorem probability_one_defective :
  probability_at_least_one_defective = 7 / 19 :=
by
  sorry

end NUMINAMATH_GPT_probability_one_defective_l583_58356


namespace NUMINAMATH_GPT_number_of_teachers_under_40_in_sample_l583_58300

def proportion_teachers_under_40 (total_teachers teachers_under_40 : ℕ) : ℚ :=
  teachers_under_40 / total_teachers

def sample_teachers_under_40 (sample_size : ℕ) (proportion : ℚ) : ℚ :=
  sample_size * proportion

theorem number_of_teachers_under_40_in_sample
(total_teachers teachers_under_40 teachers_40_and_above sample_size : ℕ)
(h_total : total_teachers = 400)
(h_under_40 : teachers_under_40 = 250)
(h_40_and_above : teachers_40_and_above = 150)
(h_sample_size : sample_size = 80)
: sample_teachers_under_40 sample_size 
  (proportion_teachers_under_40 total_teachers teachers_under_40) = 50 := by
sorry

end NUMINAMATH_GPT_number_of_teachers_under_40_in_sample_l583_58300


namespace NUMINAMATH_GPT_inverse_proportion_quadrants_l583_58388

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  ∀ (x y : ℝ), y = k^2 / x → (x > 0 → y > 0) ∧ (x < 0 → y < 0) :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_quadrants_l583_58388


namespace NUMINAMATH_GPT_find_b_l583_58357

theorem find_b (b c : ℝ) : 
  (-11 : ℝ) = (-1)^2 + (-1) * b + c ∧ 
  17 = 3^2 + 3 * b + c ∧ 
  6 = 2^2 + 2 * b + c → 
  b = 14 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l583_58357


namespace NUMINAMATH_GPT_solve_for_x_l583_58360

noncomputable def solution_x : ℝ := -1011.5

theorem solve_for_x (x : ℝ) (h : (2023 + x)^2 = x^2) : x = solution_x :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l583_58360


namespace NUMINAMATH_GPT_squirrels_in_tree_l583_58362

-- Definitions based on the conditions
def nuts : Nat := 2
def squirrels : Nat := nuts + 2

-- Theorem stating the main proof problem
theorem squirrels_in_tree : squirrels = 4 := by
  -- Proof steps would go here, but we're adding sorry to skip them
  sorry

end NUMINAMATH_GPT_squirrels_in_tree_l583_58362
