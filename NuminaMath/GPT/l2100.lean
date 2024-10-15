import Mathlib

namespace NUMINAMATH_GPT_figure_surface_area_calculation_l2100_210013

-- Define the surface area of one bar
def bar_surface_area : ℕ := 18

-- Define the surface area lost at the junctions
def surface_area_lost : ℕ := 2

-- Define the effective surface area of one bar after accounting for overlaps
def effective_bar_surface_area : ℕ := bar_surface_area - surface_area_lost

-- Define the number of bars used in the figure
def number_of_bars : ℕ := 4

-- Define the total surface area of the figure
def total_surface_area : ℕ := number_of_bars * effective_bar_surface_area

-- The theorem stating the total surface area of the figure
theorem figure_surface_area_calculation : total_surface_area = 64 := by
  sorry

end NUMINAMATH_GPT_figure_surface_area_calculation_l2100_210013


namespace NUMINAMATH_GPT_domain_range_a_l2100_210018

theorem domain_range_a (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x + a > 0) ↔ 1 < a :=
by
  sorry

end NUMINAMATH_GPT_domain_range_a_l2100_210018


namespace NUMINAMATH_GPT_sum_of_radii_l2100_210035

noncomputable def tangency_equation (r : ℝ) : Prop :=
  (r - 5)^2 + r^2 = (r + 1.5)^2

theorem sum_of_radii : ∀ (r1 r2 : ℝ), tangency_equation r1 ∧ tangency_equation r2 →
  r1 + r2 = 13 :=
by
  intros r1 r2 h
  sorry

end NUMINAMATH_GPT_sum_of_radii_l2100_210035


namespace NUMINAMATH_GPT_parabola_directrix_eq_l2100_210024

theorem parabola_directrix_eq (x y : ℝ) : x^2 + 12 * y = 0 → y = 3 := 
by sorry

end NUMINAMATH_GPT_parabola_directrix_eq_l2100_210024


namespace NUMINAMATH_GPT_yogurt_price_is_5_l2100_210026

theorem yogurt_price_is_5
  (yogurt_pints : ℕ)
  (gum_packs : ℕ)
  (shrimp_trays : ℕ)
  (total_cost : ℝ)
  (shrimp_cost : ℝ)
  (gum_fraction : ℝ)
  (price_frozen_yogurt : ℝ) :
  yogurt_pints = 5 →
  gum_packs = 2 →
  shrimp_trays = 5 →
  total_cost = 55 →
  shrimp_cost = 5 →
  gum_fraction = 0.5 →
  5 * price_frozen_yogurt + 2 * (gum_fraction * price_frozen_yogurt) + 5 * shrimp_cost = total_cost →
  price_frozen_yogurt = 5 :=
by
  intro hp hg hs ht hc hf h_formula
  sorry

end NUMINAMATH_GPT_yogurt_price_is_5_l2100_210026


namespace NUMINAMATH_GPT_vova_last_grades_l2100_210084

theorem vova_last_grades (grades : Fin 19 → ℕ) 
  (first_four_2s : ∀ i : Fin 4, grades i = 2)
  (all_combinations_once : ∀ comb : Fin 4 → ℕ, 
    (∃ (start : Fin (19-3)), ∀ j : Fin 4, grades (start + j) = comb j) ∧
    (∀ i j : Fin (19-3), 
      (∀ k : Fin 4, grades (i + k) = grades (j + k)) → i = j)) :
  ∀ i : Fin 4, grades (15 + i) = if i = 0 then 3 else 2 :=
by
  sorry

end NUMINAMATH_GPT_vova_last_grades_l2100_210084


namespace NUMINAMATH_GPT_area_of_shaded_region_l2100_210072

noncomputable def r2 : ℝ := Real.sqrt 20
noncomputable def r1 : ℝ := 3 * r2

theorem area_of_shaded_region :
  let area := π * (r1 ^ 2) - π * (r2 ^ 2)
  area = 160 * π :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l2100_210072


namespace NUMINAMATH_GPT_number_of_ways_to_select_president_and_vice_president_l2100_210051

-- Define the given conditions
def num_candidates : Nat := 4

-- Define the problem to prove
theorem number_of_ways_to_select_president_and_vice_president : (num_candidates * (num_candidates - 1)) = 12 :=
by
  -- This is where the proof would go, but we are skipping it
  sorry

end NUMINAMATH_GPT_number_of_ways_to_select_president_and_vice_president_l2100_210051


namespace NUMINAMATH_GPT_percentage_of_tip_l2100_210087

-- Given conditions
def steak_cost : ℝ := 20
def drink_cost : ℝ := 5
def total_cost_before_tip : ℝ := 2 * (steak_cost + drink_cost)
def billy_tip_payment : ℝ := 8
def billy_tip_coverage : ℝ := 0.80

-- Required to prove
theorem percentage_of_tip : ∃ P : ℝ, (P = (billy_tip_payment / (billy_tip_coverage * total_cost_before_tip)) * 100) ∧ P = 20 := 
by {
  sorry
}

end NUMINAMATH_GPT_percentage_of_tip_l2100_210087


namespace NUMINAMATH_GPT_find_A_time_l2100_210098

noncomputable def work_rate_equations (W : ℝ) (A B C : ℝ) : Prop :=
  B + C = W / 2 ∧ A + B = W / 2 ∧ C = W / 3

theorem find_A_time {W A B C : ℝ} (h : work_rate_equations W A B C) :
  W / A = 3 :=
sorry

end NUMINAMATH_GPT_find_A_time_l2100_210098


namespace NUMINAMATH_GPT_estimate_students_height_at_least_165_l2100_210020

theorem estimate_students_height_at_least_165 
  (sample_size : ℕ)
  (total_school_size : ℕ)
  (students_165_170 : ℕ)
  (students_170_175 : ℕ)
  (h_sample : sample_size = 100)
  (h_total_school : total_school_size = 1000)
  (h_students_165_170 : students_165_170 = 20)
  (h_students_170_175 : students_170_175 = 30)
  : (students_165_170 + students_170_175) * (total_school_size / sample_size) = 500 := 
by
  sorry

end NUMINAMATH_GPT_estimate_students_height_at_least_165_l2100_210020


namespace NUMINAMATH_GPT_inverse_of_matrix_C_l2100_210027

-- Define the given matrix C
def C : Matrix (Fin 3) (Fin 3) ℚ := ![
  ![1, 2, 1],
  ![3, -5, 3],
  ![2, 7, -1]
]

-- Define the claimed inverse of the matrix C
def C_inv : Matrix (Fin 3) (Fin 3) ℚ := (1 / 33 : ℚ) • ![
  ![-16,  9,  11],
  ![  9, -3,   0],
  ![ 31, -3, -11]
]

-- Statement to prove that C_inv is the inverse of C
theorem inverse_of_matrix_C : C * C_inv = 1 ∧ C_inv * C = 1 := by
  sorry

end NUMINAMATH_GPT_inverse_of_matrix_C_l2100_210027


namespace NUMINAMATH_GPT_spiders_loose_l2100_210088

noncomputable def initial_birds : ℕ := 12
noncomputable def initial_puppies : ℕ := 9
noncomputable def initial_cats : ℕ := 5
noncomputable def initial_spiders : ℕ := 15
noncomputable def birds_sold : ℕ := initial_birds / 2
noncomputable def puppies_adopted : ℕ := 3
noncomputable def remaining_puppies : ℕ := initial_puppies - puppies_adopted
noncomputable def remaining_cats : ℕ := initial_cats
noncomputable def total_remaining_animals_except_spiders : ℕ := birds_sold + remaining_puppies + remaining_cats
noncomputable def total_animals_left : ℕ := 25
noncomputable def remaining_spiders : ℕ := total_animals_left - total_remaining_animals_except_spiders
noncomputable def spiders_went_loose : ℕ := initial_spiders - remaining_spiders

theorem spiders_loose : spiders_went_loose = 7 := by
  sorry

end NUMINAMATH_GPT_spiders_loose_l2100_210088


namespace NUMINAMATH_GPT_proof_problem_l2100_210059

theorem proof_problem
  (x y z : ℤ)
  (h1 : x = 11 * y + 4)
  (h2 : 2 * x = 3 * y * z + 3)
  (h3 : 13 * y - x = 1) :
  z = 8 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l2100_210059


namespace NUMINAMATH_GPT_sum_fractions_correct_l2100_210023

def sum_of_fractions (f1 f2 f3 f4 f5 f6 f7 : ℚ) : ℚ :=
  f1 + f2 + f3 + f4 + f5 + f6 + f7

theorem sum_fractions_correct : sum_of_fractions (1/3) (1/2) (-5/6) (1/5) (1/4) (-9/20) (-5/6) = -5/6 :=
by
  sorry

end NUMINAMATH_GPT_sum_fractions_correct_l2100_210023


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l2100_210082

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x > 4 → x^2 - 4 * x > 0) ∧ ¬ (x^2 - 4 * x > 0 → x > 4) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l2100_210082


namespace NUMINAMATH_GPT_meals_second_restaurant_l2100_210086

theorem meals_second_restaurant (r1 r2 r3 total_weekly_meals : ℕ) 
    (H1 : r1 = 20) 
    (H3 : r3 = 50) 
    (H_total : total_weekly_meals = 770) : 
    (7 * r2) = 280 := 
by 
    sorry

example (r2 : ℕ) : (40 = r2) :=
    by sorry

end NUMINAMATH_GPT_meals_second_restaurant_l2100_210086


namespace NUMINAMATH_GPT_parabola_properties_l2100_210022

theorem parabola_properties (a b c: ℝ) (ha : a ≠ 0) (hc : c > 1) (h1 : 4 * a + 2 * b + c = 0) (h2 : -b / (2 * a) = 1/2):
  a * b * c < 0 ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = a ∧ a * x2^2 + b * x2 + c = a) ∧ a < -1/2 :=
by {
    sorry
}

end NUMINAMATH_GPT_parabola_properties_l2100_210022


namespace NUMINAMATH_GPT_polygon_sides_l2100_210093

theorem polygon_sides (n : ℕ) : 
  (180 * (n - 2) / 360 = 5 / 2) → n = 7 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l2100_210093


namespace NUMINAMATH_GPT_number_of_squares_l2100_210064

theorem number_of_squares (total_streetlights squares_streetlights unused_streetlights : ℕ) 
  (h1 : total_streetlights = 200) 
  (h2 : squares_streetlights = 12) 
  (h3 : unused_streetlights = 20) : 
  (∃ S : ℕ, total_streetlights = squares_streetlights * S + unused_streetlights ∧ S = 15) :=
by
  sorry

end NUMINAMATH_GPT_number_of_squares_l2100_210064


namespace NUMINAMATH_GPT_inequality_holds_l2100_210090

variables (a b c : ℝ)

theorem inequality_holds 
  (h1 : a > b) : 
  a / (c^2 + 1) > b / (c^2 + 1) :=
sorry

end NUMINAMATH_GPT_inequality_holds_l2100_210090


namespace NUMINAMATH_GPT_min_value_function_l2100_210050

theorem min_value_function (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (∀ x y : ℝ, x > 1 ∧ y > 1 → (min ((x^2 + y) / (y^2 - 1) + (y^2 + x) / (x^2 - 1)) = 8 / 3)) := 
sorry

end NUMINAMATH_GPT_min_value_function_l2100_210050


namespace NUMINAMATH_GPT_intersection_M_N_l2100_210034

-- Definitions of sets M and N
def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Proof statement showing the intersection of M and N
theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2100_210034


namespace NUMINAMATH_GPT_repeating_decimal_eq_fraction_l2100_210075

noncomputable def repeating_decimal_to_fraction (x : ℝ) : ℝ :=
  let x : ℝ := 4.5656565656 -- * 0.5656... repeating
  (100*x - x) / (100 - 1)

-- Define the theorem we want to prove
theorem repeating_decimal_eq_fraction : 
  ∀ x : ℝ, x = 4.565656 -> x = (452 : ℝ) / (99 : ℝ) :=
by
  intro x h
  -- here we would provide the proof steps, but since it's omitted
  -- we'll use sorry to skip it.
  sorry

end NUMINAMATH_GPT_repeating_decimal_eq_fraction_l2100_210075


namespace NUMINAMATH_GPT_solve_inequality_l2100_210091

theorem solve_inequality : {x : ℝ | |x - 2| * (x - 1) < 2} = {x : ℝ | x < 3} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2100_210091


namespace NUMINAMATH_GPT_correct_sampling_methods_l2100_210021

-- Defining the conditions
def high_income_families : ℕ := 50
def middle_income_families : ℕ := 300
def low_income_families : ℕ := 150
def total_residents : ℕ := 500
def sample_size : ℕ := 100
def worker_group_size : ℕ := 10
def selected_workers : ℕ := 3

-- Definitions of sampling methods
inductive SamplingMethod
| random
| systematic
| stratified

open SamplingMethod

-- Problem statement in Lean 4
theorem correct_sampling_methods :
  (total_residents = high_income_families + middle_income_families + low_income_families) →
  (sample_size = 100) →
  (worker_group_size = 10) →
  (selected_workers = 3) →
  (chosen_method_for_task1 = SamplingMethod.stratified) →
  (chosen_method_for_task2 = SamplingMethod.random) →
  (chosen_method_for_task1, chosen_method_for_task2) = (SamplingMethod.stratified, SamplingMethod.random) :=
by
  intros
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_correct_sampling_methods_l2100_210021


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l2100_210044

theorem quadratic_no_real_roots 
  (p q a b c : ℝ) 
  (h1 : 0 < p) (h2 : 0 < q) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c)
  (h6 : p ≠ q)
  (h7 : a^2 = p * q)
  (h8 : b + c = p + q)
  (h9 : b = (2 * p + q) / 3)
  (h10 : c = (p + 2 * q) / 3) :
  (∀ x : ℝ, ¬ (b * x^2 - 2 * a * x + c = 0)) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l2100_210044


namespace NUMINAMATH_GPT_probability_neither_l2100_210048

variable (P : Set ℕ → ℝ) -- Use ℕ as a placeholder for the event space
variables (A B : Set ℕ)
variables (hA : P A = 0.25) (hB : P B = 0.35) (hAB : P (A ∩ B) = 0.15)

theorem probability_neither :
  P (Aᶜ ∩ Bᶜ) = 0.55 :=
by
  sorry

end NUMINAMATH_GPT_probability_neither_l2100_210048


namespace NUMINAMATH_GPT_third_consecutive_even_number_l2100_210076

theorem third_consecutive_even_number (n : ℕ) (h : n % 2 = 0) (sum_eq : n + (n + 2) + (n + 4) = 246) : (n + 4) = 84 :=
by
  -- This statement sets up the conditions and the goal of the proof.
  sorry

end NUMINAMATH_GPT_third_consecutive_even_number_l2100_210076


namespace NUMINAMATH_GPT_Sara_spent_on_hotdog_l2100_210079

-- Define the given constants
def totalCost : ℝ := 10.46
def costSalad : ℝ := 5.10

-- Define the value we need to prove
def costHotdog : ℝ := 5.36

-- Statement to prove
theorem Sara_spent_on_hotdog : totalCost - costSalad = costHotdog := by
  sorry

end NUMINAMATH_GPT_Sara_spent_on_hotdog_l2100_210079


namespace NUMINAMATH_GPT_appropriate_line_chart_for_temperature_l2100_210001

-- Define the assumption that line charts are effective in displaying changes in data over time
axiom effective_line_chart_display (changes_over_time : Prop) : Prop

-- Define the statement to be proved, using the assumption above
theorem appropriate_line_chart_for_temperature (changes_over_time : Prop) 
  (line_charts_effective : effective_line_chart_display changes_over_time) : Prop :=
  sorry

end NUMINAMATH_GPT_appropriate_line_chart_for_temperature_l2100_210001


namespace NUMINAMATH_GPT_calc_problem1_calc_problem2_calc_problem3_calc_problem4_l2100_210060

theorem calc_problem1 : (-3 + 8 - 15 - 6 = -16) :=
by
  sorry

theorem calc_problem2 : (-4/13 - (-4/17) + 4/13 + (-13/17) = -9/17) :=
by
  sorry

theorem calc_problem3 : (-25 - (5/4 * 4/5) - (-16) = -10) :=
by
  sorry

theorem calc_problem4 : (-2^4 - (1/2 * (5 - (-3)^2)) = -14) :=
by
  sorry

end NUMINAMATH_GPT_calc_problem1_calc_problem2_calc_problem3_calc_problem4_l2100_210060


namespace NUMINAMATH_GPT_rectangle_area_l2100_210096

theorem rectangle_area
  (x y : ℝ) -- sides of the rectangle
  (h1 : 2 * x + 2 * y = 12)  -- perimeter
  (h2 : x^2 + y^2 = 25)  -- diagonal
  : x * y = 5.5 :=
sorry

end NUMINAMATH_GPT_rectangle_area_l2100_210096


namespace NUMINAMATH_GPT_domain_intersection_l2100_210052

theorem domain_intersection (A B : Set ℝ) 
    (h1 : A = {x | x < 1})
    (h2 : B = {y | y ≥ 0}) : A ∩ B = {z | 0 ≤ z ∧ z < 1} := 
by
  sorry

end NUMINAMATH_GPT_domain_intersection_l2100_210052


namespace NUMINAMATH_GPT_Ivan_can_safely_make_the_journey_l2100_210058

def eruption_cycle_first_crater (t : ℕ) : Prop :=
  ∃ n : ℕ, t = 1 + 18 * n

def eruption_cycle_second_crater (t : ℕ) : Prop :=
  ∃ m : ℕ, t = 1 + 10 * m

def is_safe (start_time : ℕ) : Prop :=
  ∀ t, start_time ≤ t ∧ t < start_time + 16 → 
    ¬ eruption_cycle_first_crater t ∧ 
    ¬ (t ≥ start_time + 12 ∧ eruption_cycle_second_crater t)

theorem Ivan_can_safely_make_the_journey : ∃ t : ℕ, is_safe (38 + t) :=
sorry

end NUMINAMATH_GPT_Ivan_can_safely_make_the_journey_l2100_210058


namespace NUMINAMATH_GPT_train_problem_l2100_210054

variables (x : ℝ) (p q : ℝ)
variables (speed_p speed_q : ℝ) (dist_diff : ℝ)

theorem train_problem
  (speed_p : speed_p = 50)
  (speed_q : speed_q = 40)
  (dist_diff : ∀ x, x = 500 → p = 50 * x ∧ q = 40 * (500 - 100)) :
  p + q = 900 :=
by
sorry

end NUMINAMATH_GPT_train_problem_l2100_210054


namespace NUMINAMATH_GPT_mary_balloons_correct_l2100_210006

-- Define the number of black balloons Nancy has
def nancy_balloons : ℕ := 7

-- Define the multiplier that represents how many times more balloons Mary has compared to Nancy
def multiplier : ℕ := 4

-- Define the number of black balloons Mary has in terms of Nancy's balloons and the multiplier
def mary_balloons : ℕ := nancy_balloons * multiplier

-- The statement we want to prove
theorem mary_balloons_correct : mary_balloons = 28 :=
by
  sorry

end NUMINAMATH_GPT_mary_balloons_correct_l2100_210006


namespace NUMINAMATH_GPT_amy_seeds_l2100_210095

-- Define the conditions
def bigGardenSeeds : Nat := 47
def smallGardens : Nat := 9
def seedsPerSmallGarden : Nat := 6

-- Define the total seeds calculation
def totalSeeds := bigGardenSeeds + smallGardens * seedsPerSmallGarden

-- The theorem to be proved
theorem amy_seeds : totalSeeds = 101 := by
  sorry

end NUMINAMATH_GPT_amy_seeds_l2100_210095


namespace NUMINAMATH_GPT_trader_profit_percentage_l2100_210081

theorem trader_profit_percentage (P : ℝ) (hP : 0 < P) :
  let bought_price := 0.90 * P
  let sold_price := 1.80 * bought_price
  let profit := sold_price - P
  let profit_percentage := (profit / P) * 100
  profit_percentage = 62 := 
by
  let bought_price := 0.90 * P
  let sold_price := 1.80 * bought_price
  let profit := sold_price - P
  let profit_percentage := (profit / P) * 100
  sorry

end NUMINAMATH_GPT_trader_profit_percentage_l2100_210081


namespace NUMINAMATH_GPT_servant_position_for_28_purses_servant_position_for_27_purses_l2100_210037

-- Definitions based on problem conditions
def total_wealthy_men: ℕ := 7

def valid_purse_placement (n: ℕ): Prop := 
  (n ≤ total_wealthy_men * (total_wealthy_men + 1) / 2)

def get_servant_position (n: ℕ): ℕ := 
  if n = 28 then total_wealthy_men else if n = 27 then 6 else 0

-- Proof statements to equate conditions with the answers
theorem servant_position_for_28_purses : 
  get_servant_position 28 = 7 :=
sorry

theorem servant_position_for_27_purses : 
  get_servant_position 27 = 6 ∨ get_servant_position 27 = 7 :=
sorry

end NUMINAMATH_GPT_servant_position_for_28_purses_servant_position_for_27_purses_l2100_210037


namespace NUMINAMATH_GPT_calculate_ggg1_l2100_210036

def g (x : ℕ) : ℕ := 7 * x + 3

theorem calculate_ggg1 : g (g (g 1)) = 514 := 
by
  sorry

end NUMINAMATH_GPT_calculate_ggg1_l2100_210036


namespace NUMINAMATH_GPT_chinaman_change_possible_l2100_210014

def pence (x : ℕ) := x -- defining the value of pence as a natural number

def ching_chang_by_value (d : ℕ) := 
  (2 * pence d) + (4 * (2 * pence d) / 15)

def equivalent_value_of_half_crown (d : ℕ) := 30 * pence d

def coin_value_with_holes (holes_value : ℕ) (value_per_eleven : ℕ) := 
  (value_per_eleven * ching_chang_by_value 1) / 11

theorem chinaman_change_possible :
  ∃ (x y z : ℕ), 
  (7 * coin_value_with_holes 15 11) + (1 * coin_value_with_holes 16 11) + (0 * coin_value_with_holes 17 11) = 
  equivalent_value_of_half_crown 1 :=
sorry

end NUMINAMATH_GPT_chinaman_change_possible_l2100_210014


namespace NUMINAMATH_GPT_recurring_decimal_of_division_l2100_210080

theorem recurring_decimal_of_division (a b : ℤ) (h1 : a = 60) (h2 : b = 55) : (a : ℝ) / (b : ℝ) = 1.09090909090909090909090909090909 :=
by
  -- Import the necessary definitions and facts
  sorry

end NUMINAMATH_GPT_recurring_decimal_of_division_l2100_210080


namespace NUMINAMATH_GPT_math_problem_l2100_210085

theorem math_problem :
  (2^8 + 4^5) * (1^3 - (-1)^3)^2 = 5120 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2100_210085


namespace NUMINAMATH_GPT_man_profit_doubled_l2100_210032

noncomputable def percentage_profit (C SP1 SP2 : ℝ) : ℝ :=
  (SP2 - C) / C * 100

theorem man_profit_doubled (C SP1 SP2 : ℝ) (h1 : SP1 = 1.30 * C) (h2 : SP2 = 2 * SP1) :
  percentage_profit C SP1 SP2 = 160 := by
  sorry

end NUMINAMATH_GPT_man_profit_doubled_l2100_210032


namespace NUMINAMATH_GPT_person_a_work_days_l2100_210047

theorem person_a_work_days (x : ℝ) (h1 : 1 / 6 + 1 / x = 1 / 3.75) : x = 10 := 
sorry

end NUMINAMATH_GPT_person_a_work_days_l2100_210047


namespace NUMINAMATH_GPT_gcd_1729_867_l2100_210030

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_gcd_1729_867_l2100_210030


namespace NUMINAMATH_GPT_sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022_l2100_210010

theorem sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022:
  ( (Real.sqrt 10 + 3) ^ 2023 * (Real.sqrt 10 - 3) ^ 2022 = Real.sqrt 10 + 3 ) :=
by {
  sorry
}

end NUMINAMATH_GPT_sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022_l2100_210010


namespace NUMINAMATH_GPT_smallest_solution_l2100_210042

theorem smallest_solution (x : ℝ) :
  (∃ x, (3 * x) / (x - 3) + (3 * x^2 - 36) / (x + 3) = 15) →
  x = -1 := 
sorry

end NUMINAMATH_GPT_smallest_solution_l2100_210042


namespace NUMINAMATH_GPT_license_plate_count_l2100_210039

def num_license_plates : Nat :=
  26 * 10 * 36

theorem license_plate_count : num_license_plates = 9360 :=
by
  sorry

end NUMINAMATH_GPT_license_plate_count_l2100_210039


namespace NUMINAMATH_GPT_school_profit_calc_l2100_210062

-- Definitions based on the conditions provided
def pizza_slices : Nat := 8
def slices_per_pizza : ℕ := 8
def slice_price : ℝ := 1.0 -- Defining price per slice
def pizzas_bought : ℕ := 55
def cost_per_pizza : ℝ := 6.85
def total_revenue : ℝ := pizzas_bought * slices_per_pizza * slice_price
def total_cost : ℝ := pizzas_bought * cost_per_pizza

-- The lean mathematical statement we need to prove
theorem school_profit_calc :
  total_revenue - total_cost = 63.25 := by
  sorry

end NUMINAMATH_GPT_school_profit_calc_l2100_210062


namespace NUMINAMATH_GPT_quotient_of_division_l2100_210000

theorem quotient_of_division (L S Q : ℕ) (h1 : L - S = 2500) (h2 : L = 2982) (h3 : L = Q * S + 15) : Q = 6 := 
sorry

end NUMINAMATH_GPT_quotient_of_division_l2100_210000


namespace NUMINAMATH_GPT_last_two_non_zero_digits_of_75_factorial_l2100_210068

theorem last_two_non_zero_digits_of_75_factorial : 
  ∃ (d : ℕ), d = 32 := sorry

end NUMINAMATH_GPT_last_two_non_zero_digits_of_75_factorial_l2100_210068


namespace NUMINAMATH_GPT_inequality_satisfaction_l2100_210074

theorem inequality_satisfaction (a b : ℝ) (h : 0 < a ∧ a < b) : 
  a < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 ∧ (a + b) / 2 < b :=
by
  sorry

end NUMINAMATH_GPT_inequality_satisfaction_l2100_210074


namespace NUMINAMATH_GPT_final_number_proof_l2100_210056

/- Define the symbols and their corresponding values -/
def cat := 1
def chicken := 5
def crab := 2
def bear := 4
def goat := 3

/- Define the equations from the conditions -/
axiom row4_eq : 5 * crab = 10
axiom col5_eq : 4 * crab + goat = 11
axiom row2_eq : 2 * goat + crab + 2 * bear = 16
axiom col2_eq : cat + bear + 2 * goat + crab = 13
axiom col3_eq : 2 * crab + 2 * chicken + goat = 17

/- Final number is derived by concatenating digits -/
def final_number := cat * 10000 + chicken * 1000 + crab * 100 + bear * 10 + goat

/- Theorem to prove the final number is 15243 -/
theorem final_number_proof : final_number = 15243 := by
  -- Proof steps to be provided here.
  sorry

end NUMINAMATH_GPT_final_number_proof_l2100_210056


namespace NUMINAMATH_GPT_cost_of_item_is_200_l2100_210025

noncomputable def cost_of_each_item (x : ℕ) : ℕ :=
  let before_discount := 7 * x -- Total cost before discount
  let discount_part := before_discount - 1000 -- Part of the cost over $1000
  let discount := discount_part / 10 -- 10% of the part over $1000
  let after_discount := before_discount - discount -- Total cost after discount
  after_discount

theorem cost_of_item_is_200 :
  (∃ x : ℕ, cost_of_each_item x = 1360) ↔ x = 200 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_item_is_200_l2100_210025


namespace NUMINAMATH_GPT_problem_expression_eval_l2100_210089

theorem problem_expression_eval : (1 + 2 + 3) * (1 + 1/2 + 1/3) = 11 := by
  sorry

end NUMINAMATH_GPT_problem_expression_eval_l2100_210089


namespace NUMINAMATH_GPT_intersection_A_B_l2100_210007

def A := {x : ℝ | x^2 - x - 2 ≤ 0}
def B := {x : ℝ | ∃ y : ℝ, y = Real.log (1 - x)}

theorem intersection_A_B : (A ∩ B) = {x : ℝ | -1 ≤ x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2100_210007


namespace NUMINAMATH_GPT_origin_moves_3sqrt5_under_dilation_l2100_210061

/--
Given:
1. The original circle has radius 3 centered at point B(3, 3).
2. The dilated circle has radius 6 centered at point B'(9, 12).

Prove that the distance moved by the origin O(0, 0) under this dilation is 3 * sqrt(5).
-/
theorem origin_moves_3sqrt5_under_dilation:
  let B := (3, 3)
  let B' := (9, 12)
  let radius_B := 3
  let radius_B' := 6
  let dilation_center := (-3, -6)
  let origin := (0, 0)
  let k := radius_B' / radius_B
  let d_0 := Real.sqrt ((-3 : ℝ)^2 + (-6 : ℝ)^2)
  let d_1 := k * d_0
  d_1 - d_0 = 3 * Real.sqrt (5 : ℝ) := by sorry

end NUMINAMATH_GPT_origin_moves_3sqrt5_under_dilation_l2100_210061


namespace NUMINAMATH_GPT_Daniela_is_12_years_old_l2100_210077

noncomputable def auntClaraAge : Nat := 60

noncomputable def evelinaAge : Nat := auntClaraAge / 3

noncomputable def fidelAge : Nat := evelinaAge - 6

noncomputable def caitlinAge : Nat := fidelAge / 2

noncomputable def danielaAge : Nat := evelinaAge - 8

theorem Daniela_is_12_years_old (h_auntClaraAge : auntClaraAge = 60)
                                (h_evelinaAge : evelinaAge = 60 / 3)
                                (h_fidelAge : fidelAge = (60 / 3) - 6)
                                (h_caitlinAge : caitlinAge = ((60 / 3) - 6) / 2)
                                (h_danielaAge : danielaAge = (60 / 3) - 8) :
  danielaAge = 12 := 
  sorry

end NUMINAMATH_GPT_Daniela_is_12_years_old_l2100_210077


namespace NUMINAMATH_GPT_find_last_number_l2100_210046

theorem find_last_number (A B C D E F G : ℝ)
    (h1 : (A + B + C + D) / 4 = 13)
    (h2 : (D + E + F + G) / 4 = 15)
    (h3 : E + F + G = 55)
    (h4 : D^2 = G) :
  G = 25 := by 
  sorry

end NUMINAMATH_GPT_find_last_number_l2100_210046


namespace NUMINAMATH_GPT_exists_super_number_B_l2100_210045

-- Define a function is_super_number to identify super numbers.
def is_super_number (A : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 ≤ A n ∧ A n < 10

-- Define a function zero_super_number to represent the super number with all digits zero.
def zero_super_number (n : ℕ) := 0

-- Task: Prove the existence of B such that A + B = zero_super_number.
theorem exists_super_number_B (A : ℕ → ℕ) (hA : is_super_number A) :
  ∃ B : ℕ → ℕ, is_super_number B ∧ (∀ n : ℕ, (A n + B n) % 10 = zero_super_number n) :=
sorry

end NUMINAMATH_GPT_exists_super_number_B_l2100_210045


namespace NUMINAMATH_GPT_plane_coloring_l2100_210063

-- Define a type for colors to represent red and blue
inductive Color
| red
| blue

-- The main statement
theorem plane_coloring (x : ℝ) (h_pos : 0 < x) (coloring : ℝ × ℝ → Color) :
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ coloring p1 = coloring p2 ∧ dist p1 p2 = x :=
sorry

end NUMINAMATH_GPT_plane_coloring_l2100_210063


namespace NUMINAMATH_GPT_cos_seven_pi_over_six_l2100_210033

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  -- Place the proof here
  sorry

end NUMINAMATH_GPT_cos_seven_pi_over_six_l2100_210033


namespace NUMINAMATH_GPT_range_of_a_solution_set_of_inequality_l2100_210043

-- Lean statement for Part 1
theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x : ℝ, x^2 - 2 * a * x + a > 0 :=
by
  sorry

-- Lean statement for Part 2
theorem solution_set_of_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  { x : ℝ | a^(x^2 - 3) < a^(2 * x) ∧ a^(2 * x) < 1 } = { x : ℝ | x > 3 } :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_solution_set_of_inequality_l2100_210043


namespace NUMINAMATH_GPT_f_2017_eq_2018_l2100_210067

def f (n : ℕ) : ℕ := sorry

theorem f_2017_eq_2018 (f : ℕ → ℕ) (h1 : ∀ n, f (f n) + f n = 2 * n + 3) (h2 : f 0 = 1) : f 2017 = 2018 :=
sorry

end NUMINAMATH_GPT_f_2017_eq_2018_l2100_210067


namespace NUMINAMATH_GPT_sugar_per_bar_l2100_210057

theorem sugar_per_bar (bars_per_minute : ℕ) (sugar_per_2_minutes : ℕ)
  (h1 : bars_per_minute = 36)
  (h2 : sugar_per_2_minutes = 108) :
  (sugar_per_2_minutes / (bars_per_minute * 2) : ℚ) = 1.5 := 
by 
  sorry

end NUMINAMATH_GPT_sugar_per_bar_l2100_210057


namespace NUMINAMATH_GPT_cubic_polynomial_solution_l2100_210040

noncomputable def q (x : ℚ) : ℚ := (51/13) * x^3 + (-31/13) * x^2 + (16/13) * x + (3/13)

theorem cubic_polynomial_solution : 
  q 1 = 3 ∧ q 2 = 23 ∧ q 3 = 81 ∧ q 5 = 399 :=
by {
  sorry
}

end NUMINAMATH_GPT_cubic_polynomial_solution_l2100_210040


namespace NUMINAMATH_GPT_max_band_members_l2100_210071

theorem max_band_members (k n m : ℕ) : m = k^2 + 11 → m = n * (n + 9) → m ≤ 112 :=
by
  sorry

end NUMINAMATH_GPT_max_band_members_l2100_210071


namespace NUMINAMATH_GPT_black_to_white_area_ratio_l2100_210038

noncomputable def radius1 : ℝ := 2
noncomputable def radius2 : ℝ := 4
noncomputable def radius3 : ℝ := 6
noncomputable def radius4 : ℝ := 8
noncomputable def radius5 : ℝ := 10

noncomputable def area (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def black_area : ℝ :=
  area radius1 + (area radius3 - area radius2) + (area radius5 - area radius4)

noncomputable def white_area : ℝ :=
  (area radius2 - area radius1) + (area radius4 - area radius3)

theorem black_to_white_area_ratio :
  black_area / white_area = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_black_to_white_area_ratio_l2100_210038


namespace NUMINAMATH_GPT_ratio_of_m1_and_m2_l2100_210029

theorem ratio_of_m1_and_m2 (m a b m1 m2 : ℝ) (h1 : a^2 * m - 3 * a * m + 2 * a + 7 = 0) (h2 : b^2 * m - 3 * b * m + 2 * b + 7 = 0) 
  (h3 : (a / b) + (b / a) = 2) (h4 : m1^2 * 9 - m1 * 28 + 4 = 0) (h5 : m2^2 * 9 - m2 * 28 + 4 = 0) : 
  (m1 / m2) + (m2 / m1) = 194 / 9 := 
sorry

end NUMINAMATH_GPT_ratio_of_m1_and_m2_l2100_210029


namespace NUMINAMATH_GPT_geometric_sequence_root_product_l2100_210016

theorem geometric_sequence_root_product
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (a1_pos : 0 < a 1)
  (a19_root : a 1 * r^18 = (1 : ℝ))
  (h_poly : ∀ x, x^2 - 10 * x + 16 = 0) :
  a 8 * a 12 = 16  :=
sorry

end NUMINAMATH_GPT_geometric_sequence_root_product_l2100_210016


namespace NUMINAMATH_GPT_g_at_neg_two_is_fifteen_l2100_210015

def g (x : ℤ) : ℤ := 2 * x^2 - 3 * x + 1

theorem g_at_neg_two_is_fifteen : g (-2) = 15 :=
by 
  -- proof is skipped
  sorry

end NUMINAMATH_GPT_g_at_neg_two_is_fifteen_l2100_210015


namespace NUMINAMATH_GPT_problem_statement_l2100_210011

noncomputable def A : Set ℝ := {x | 2 < x ∧ x ≤ 6}
noncomputable def B : Set ℝ := {x | x^2 - 4 * x < 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2 * m - 1}

theorem problem_statement (m : ℝ) :
    (A ∩ B = {x | 2 < x ∧ x < 4}) ∧
    (¬(A ∪ B) = {x | x ≤ 0 ∨ x > 6}) ∧
    (C m ⊆ B → m ∈ Set.Iic (5/2)) := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2100_210011


namespace NUMINAMATH_GPT_work_done_in_one_day_by_A_and_B_l2100_210053

noncomputable def A_days : ℕ := 12
noncomputable def B_days : ℕ := A_days / 2

theorem work_done_in_one_day_by_A_and_B : 1 / (A_days : ℚ) + 1 / (B_days : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_work_done_in_one_day_by_A_and_B_l2100_210053


namespace NUMINAMATH_GPT_find_abc_triplet_l2100_210083

theorem find_abc_triplet (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_order : a < b ∧ b < c) 
  (h_eqn : (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) = (a + b + c) / 2) :
  ∃ d : ℕ, d > 0 ∧ ((a = d ∧ b = 2 * d ∧ c = 3 * d) ∨ (a = d ∧ b = 3 * d ∧ c = 6 * d)) :=
  sorry

end NUMINAMATH_GPT_find_abc_triplet_l2100_210083


namespace NUMINAMATH_GPT_album_count_l2100_210069

theorem album_count (A B S : ℕ) (hA : A = 23) (hB : B = 9) (hS : S = 15) : 
  (A - S) + B = 17 :=
by
  -- Variables and conditions
  have Andrew_unique : ℕ := A - S
  have Bella_unique : ℕ := B
  -- Proof starts here
  sorry

end NUMINAMATH_GPT_album_count_l2100_210069


namespace NUMINAMATH_GPT_polynomial_div_simplify_l2100_210065

theorem polynomial_div_simplify (x : ℝ) (hx : x ≠ 0) :
  (6 * x ^ 4 - 4 * x ^ 3 + 2 * x ^ 2) / (2 * x ^ 2) = 3 * x ^ 2 - 2 * x + 1 :=
by sorry

end NUMINAMATH_GPT_polynomial_div_simplify_l2100_210065


namespace NUMINAMATH_GPT_problem1_problem2_l2100_210008

-- Definitions for the conditions
variables {A B C : ℝ}
variables {a b c S : ℝ}

-- Problem 1: Proving the value of side "a" given certain conditions
theorem problem1 (h₁ : S = (1 / 2) * a * b * Real.sin C) (h₂ : a^2 = 4 * Real.sqrt 3 * S)
  (h₃ : C = Real.pi / 3) (h₄ : b = 1) : a = 3 := by
  sorry

-- Problem 2: Proving the measure of angle "A" given certain conditions
theorem problem2 (h₁ : S = (1 / 2) * a * b * Real.sin C) (h₂ : a^2 = 4 * Real.sqrt 3 * S)
  (h₃ : c / b = 2 + Real.sqrt 3) : A = Real.pi / 3 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2100_210008


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2100_210055

theorem arithmetic_sequence_common_difference (a_1 a_4 a_5 d : ℤ) 
  (h1 : a_1 + a_5 = 10) 
  (h2 : a_4 = 7) 
  (h3 : a_4 = a_1 + 3 * d) 
  (h4 : a_5 = a_1 + 4 * d) : 
  d = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2100_210055


namespace NUMINAMATH_GPT_convert_spherical_to_rectangular_l2100_210049

noncomputable def spherical_to_rectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin phi * Real.cos theta,
   rho * Real.sin phi * Real.sin theta,
   rho * Real.cos phi)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 4) = (2 * Real.sqrt 3, Real.sqrt 2, 2 * Real.sqrt 2) :=
by
  -- Define the spherical coordinates
  let rho := 4
  let theta := Real.pi / 6
  let phi := Real.pi / 4

  -- Calculate x, y, z using conversion formulas
  sorry

end NUMINAMATH_GPT_convert_spherical_to_rectangular_l2100_210049


namespace NUMINAMATH_GPT_union_A_B_intersection_A_complement_B_l2100_210017

def setA (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 2
def setB (x : ℝ) : Prop := x * (x - 4) ≤ 0

theorem union_A_B : {x : ℝ | setA x} ∪ {x : ℝ | setB x} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem intersection_A_complement_B : {x : ℝ | setA x} ∩ {x : ℝ | ¬ setB x} = {x : ℝ | -1 ≤ x ∧ x < 0} :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_intersection_A_complement_B_l2100_210017


namespace NUMINAMATH_GPT_find_line_equation_l2100_210005

theorem find_line_equation (k x y x₁ y₁ x₂ y₂ : ℝ) (h_parabola : y ^ 2 = 2 * x) 
  (h_line_ny_eq : y = k * x + 2) (h_intersect_1 : (y₁ - (k * x₁ + 2)) = 0)
  (h_intersect_2 : (y₂ - (k * x₂ + 2)) = 0) 
  (h_y_intercept : (0,2) = (x,y))-- the line has y-intercept 2 
  (h_origin : (0,0) = (x, y)) -- origin 
  (h_orthogonal : x₁ * x₂ + y₁ * y₂ = 0): 
  y = -x + 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_line_equation_l2100_210005


namespace NUMINAMATH_GPT_karl_savings_l2100_210019

noncomputable def cost_per_notebook : ℝ := 3.75
noncomputable def notebooks_bought : ℕ := 8
noncomputable def discount_rate : ℝ := 0.25
noncomputable def original_total_cost : ℝ := notebooks_bought * cost_per_notebook
noncomputable def discount_per_notebook : ℝ := cost_per_notebook * discount_rate
noncomputable def discounted_price_per_notebook : ℝ := cost_per_notebook - discount_per_notebook
noncomputable def discounted_total_cost : ℝ := notebooks_bought * discounted_price_per_notebook
noncomputable def total_savings : ℝ := original_total_cost - discounted_total_cost

theorem karl_savings : total_savings = 7.50 := by 
  sorry

end NUMINAMATH_GPT_karl_savings_l2100_210019


namespace NUMINAMATH_GPT_minimum_cost_to_buy_additional_sheets_l2100_210031

def total_sheets : ℕ := 98
def students : ℕ := 12
def cost_per_sheet : ℕ := 450

theorem minimum_cost_to_buy_additional_sheets : 
  (students * (1 + total_sheets / students) - total_sheets) * cost_per_sheet = 4500 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_cost_to_buy_additional_sheets_l2100_210031


namespace NUMINAMATH_GPT_line_properties_l2100_210094

theorem line_properties (m x_intercept : ℝ) (y_intercept point_on_line : ℝ × ℝ) :
  m = -4 → x_intercept = -3 → y_intercept = (0, -12) → point_on_line = (2, -20) → 
    (∀ x y, y = -4 * x - 12 → (y_intercept = (0, y) ∧ point_on_line = (x, y))) := 
by
  sorry

end NUMINAMATH_GPT_line_properties_l2100_210094


namespace NUMINAMATH_GPT_inequality_solution_l2100_210041

theorem inequality_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 6) : x^3 - 12 * x^2 + 36 * x > 0 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2100_210041


namespace NUMINAMATH_GPT_infinite_n_dividing_a_pow_n_plus_1_l2100_210004

theorem infinite_n_dividing_a_pow_n_plus_1 (a : ℕ) (h1 : 1 < a) (h2 : a % 2 = 0) :
  ∃ (S : Set ℕ), S.Infinite ∧ ∀ n ∈ S, n ∣ a^n + 1 := 
sorry

end NUMINAMATH_GPT_infinite_n_dividing_a_pow_n_plus_1_l2100_210004


namespace NUMINAMATH_GPT_find_min_difference_l2100_210066

theorem find_min_difference (p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h₁ : 3 * q < 5 * p)
  (h₂ : 8 * p < 5 * q)
  (h₃ : ∀ r s : ℤ, 0 < s → (3 * s < 5 * r ∧ 8 * r < 5 * s) → q ≤ s) :
  q - p = 5 :=
sorry

end NUMINAMATH_GPT_find_min_difference_l2100_210066


namespace NUMINAMATH_GPT_quadratic_cubic_inequalities_l2100_210073

noncomputable def f (x : ℝ) : ℝ := x ^ 2
noncomputable def g (x : ℝ) : ℝ := -x ^ 3 + 5 * x - 3

variable (x : ℝ)

theorem quadratic_cubic_inequalities (h : 0 < x) : 
  (f x ≥ 2 * x - 1) ∧ (g x ≤ 2 * x - 1) := 
sorry

end NUMINAMATH_GPT_quadratic_cubic_inequalities_l2100_210073


namespace NUMINAMATH_GPT_sum_of_center_coordinates_l2100_210028

theorem sum_of_center_coordinates : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 = 6*x - 10*y + 24) -> 
  (∃ (cx cy : ℝ), (x^2 - 6*x + y^2 + 10*y = (cx - 3)^2 + (cy + 5)^2 + 58) ∧ (cx + cy = -2)) :=
  sorry

end NUMINAMATH_GPT_sum_of_center_coordinates_l2100_210028


namespace NUMINAMATH_GPT_u_2023_is_4_l2100_210012

def f (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 2
  | 4 => 1
  | 5 => 4
  | _ => 0  -- f is only defined for x in {1, 2, 3, 4, 5}

def u : ℕ → ℕ
| 0 => 5
| (n + 1) => f (u n)

theorem u_2023_is_4 : u 2023 = 4 := by
  sorry

end NUMINAMATH_GPT_u_2023_is_4_l2100_210012


namespace NUMINAMATH_GPT_fraction_study_only_japanese_l2100_210092

variable (J : ℕ)

def seniors := 2 * J
def sophomores := (3 / 4) * J

def seniors_study_japanese := (3 / 8) * seniors J
def juniors_study_japanese := (1 / 4) * J
def sophomores_study_japanese := (2 / 5) * sophomores J

def seniors_study_both := (1 / 6) * seniors J
def juniors_study_both := (1 / 12) * J
def sophomores_study_both := (1 / 10) * sophomores J

def seniors_study_only_japanese := seniors_study_japanese J - seniors_study_both J
def juniors_study_only_japanese := juniors_study_japanese J - juniors_study_both J
def sophomores_study_only_japanese := sophomores_study_japanese J - sophomores_study_both J

def total_study_only_japanese := seniors_study_only_japanese J + juniors_study_only_japanese J + sophomores_study_only_japanese J
def total_students := J + seniors J + sophomores J

theorem fraction_study_only_japanese :
  (total_study_only_japanese J) / (total_students J) = 97 / 450 :=
by sorry

end NUMINAMATH_GPT_fraction_study_only_japanese_l2100_210092


namespace NUMINAMATH_GPT_Josh_lost_marbles_l2100_210009

theorem Josh_lost_marbles :
  let original_marbles := 9.5
  let current_marbles := 4.25
  original_marbles - current_marbles = 5.25 :=
by
  sorry

end NUMINAMATH_GPT_Josh_lost_marbles_l2100_210009


namespace NUMINAMATH_GPT_factor_quadratic_l2100_210070

theorem factor_quadratic (x : ℝ) (m n : ℝ) 
  (hm : m^2 = 16) (hn : n^2 = 25) (hmn : 2 * m * n = 40) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := 
by sorry

end NUMINAMATH_GPT_factor_quadratic_l2100_210070


namespace NUMINAMATH_GPT_value_of_f_3_and_f_neg_7_point_5_l2100_210002

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 1) = -f x
axiom definition_f : ∀ x : ℝ, -1 < x → x < 1 → f x = x

theorem value_of_f_3_and_f_neg_7_point_5 :
  f 3 + f (-7.5) = 0.5 :=
sorry

end NUMINAMATH_GPT_value_of_f_3_and_f_neg_7_point_5_l2100_210002


namespace NUMINAMATH_GPT_find_m_for_eccentric_ellipse_l2100_210097

theorem find_m_for_eccentric_ellipse (m : ℝ) : 
  (∀ x y : ℝ, (x^2)/5 + (y^2)/m = 1) ∧
  (∀ e : ℝ, e = (Real.sqrt 10)/5) → 
  (m = 25/3 ∨ m = 3) := sorry

end NUMINAMATH_GPT_find_m_for_eccentric_ellipse_l2100_210097


namespace NUMINAMATH_GPT_find_four_digit_number_l2100_210003

def is_four_digit_number (k : ℕ) : Prop :=
  1000 ≤ k ∧ k < 10000

def appended_number (k : ℕ) : ℕ :=
  4000000 + k

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem find_four_digit_number (k : ℕ) (hk : is_four_digit_number k) :
  is_perfect_square (appended_number k) ↔ k = 4001 ∨ k = 8004 :=
sorry

end NUMINAMATH_GPT_find_four_digit_number_l2100_210003


namespace NUMINAMATH_GPT_vertex_is_correct_l2100_210099

-- Define the equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 10 * y + 4 * x + 9 = 0

-- The vertex of the parabola
def vertex_of_parabola : ℝ × ℝ := (4, -5)

-- The theorem stating that the given vertex satisfies the parabola equation
theorem vertex_is_correct : 
  parabola_equation vertex_of_parabola.1 vertex_of_parabola.2 :=
sorry

end NUMINAMATH_GPT_vertex_is_correct_l2100_210099


namespace NUMINAMATH_GPT_find_k_l2100_210078

def line1 (x y : ℝ) : Prop := x + 3 * y - 7 = 0
def line2 (k x y : ℝ) : Prop := k * x + y - 2 = 0
def quadrilateral_has_circumscribed_circle (k : ℝ) : Prop :=
  ∀ x y : ℝ, line1 x y → line2 k x y →
  k = -3

theorem find_k (k : ℝ) (x y : ℝ) : 
  (line1 x y) ∧ (line2 k x y) → quadrilateral_has_circumscribed_circle k :=
by 
  sorry

end NUMINAMATH_GPT_find_k_l2100_210078
