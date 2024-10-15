import Mathlib

namespace NUMINAMATH_GPT_quadratic_roots_are_correct_l20_2019

theorem quadratic_roots_are_correct (x: ℝ) : 
    (x^2 + x - 1 = 0) ↔ (x = (-1 + Real.sqrt 5) / 2) ∨ (x = (-1 - Real.sqrt 5) / 2) := 
by sorry

end NUMINAMATH_GPT_quadratic_roots_are_correct_l20_2019


namespace NUMINAMATH_GPT_complement_of_angle_l20_2007

theorem complement_of_angle (A : ℝ) (hA : A = 35) : 180 - A = 145 := by
  sorry

end NUMINAMATH_GPT_complement_of_angle_l20_2007


namespace NUMINAMATH_GPT_sum_possible_values_of_y_l20_2041

theorem sum_possible_values_of_y (y : ℝ) (h : y^2 = 36) : y = 6 ∨ y = -6 → (6 + (-6) = 0) :=
by
  sorry

end NUMINAMATH_GPT_sum_possible_values_of_y_l20_2041


namespace NUMINAMATH_GPT_sugar_percentage_l20_2083

theorem sugar_percentage (S : ℝ) (P : ℝ) : 
  (3 / 4 * S * 0.10 + (1 / 4) * S * P / 100 = S * 0.20) → 
  P = 50 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_sugar_percentage_l20_2083


namespace NUMINAMATH_GPT_trees_died_in_typhoon_l20_2069

theorem trees_died_in_typhoon :
  ∀ (original_trees left_trees died_trees : ℕ), 
  original_trees = 20 → 
  left_trees = 4 → 
  died_trees = original_trees - left_trees → 
  died_trees = 16 :=
by
  intros original_trees left_trees died_trees h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_trees_died_in_typhoon_l20_2069


namespace NUMINAMATH_GPT_inverse_proportion_relation_l20_2023

theorem inverse_proportion_relation :
  let y1 := 3 / (-2)
  let y2 := 3 / (-1)
  let y3 := 3 / 1
  y2 < y1 ∧ y1 < y3 :=
by
  -- Variable definitions according to conditions
  let y1 := 3 / (-2)
  let y2 := 3 / (-1)
  let y3 := 3 / 1
  -- Proof steps go here (not required for the statement)
  -- Since proof steps are omitted, we use sorry to indicate it
  sorry

end NUMINAMATH_GPT_inverse_proportion_relation_l20_2023


namespace NUMINAMATH_GPT_cannot_achieve_90_cents_l20_2030

theorem cannot_achieve_90_cents :
  ∀ (p n d q : ℕ),        -- p: number of pennies, n: number of nickels, d: number of dimes, q: number of quarters
  (p + n + d + q = 6) →   -- exactly six coins chosen
  (p ≤ 4 ∧ n ≤ 4 ∧ d ≤ 4 ∧ q ≤ 4) →  -- no more than four of each kind of coin
  (p + 5 * n + 10 * d + 25 * q ≠ 90) -- total value should not equal 90 cents
:= by
  sorry

end NUMINAMATH_GPT_cannot_achieve_90_cents_l20_2030


namespace NUMINAMATH_GPT_louisa_second_day_distance_l20_2059

-- Definitions based on conditions
def time_on_first_day (distance : ℕ) (speed : ℕ) : ℕ := distance / speed
def time_on_second_day (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

def condition (distance_first_day : ℕ) (speed : ℕ) (time_difference : ℕ) (x : ℕ) : Prop := 
  time_on_first_day distance_first_day speed + time_difference = time_on_second_day x speed

-- The proof statement
theorem louisa_second_day_distance (distance_first_day : ℕ) (speed : ℕ) (time_difference : ℕ) (x : ℕ) :
  distance_first_day = 240 → 
  speed = 60 → 
  time_difference = 3 → 
  condition distance_first_day speed time_difference x → 
  x = 420 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_louisa_second_day_distance_l20_2059


namespace NUMINAMATH_GPT_expression_value_l20_2034

theorem expression_value (a b c : ℝ) (h : a * b + b * c + c * a = 3) : 
  (a * (b^2 + 3)) / (a + b) + (b * (c^2 + 3)) / (b + c) + (c * (a^2 + 3)) / (c + a) = 6 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_l20_2034


namespace NUMINAMATH_GPT_first_part_amount_l20_2012

-- Given Definitions
def total_amount : ℝ := 3200
def interest_rate_part1 : ℝ := 0.03
def interest_rate_part2 : ℝ := 0.05
def total_interest : ℝ := 144

-- The problem to be proven
theorem first_part_amount : 
  ∃ (x : ℝ), 0.03 * x + 0.05 * (3200 - x) = 144 ∧ x = 800 :=
by
  sorry

end NUMINAMATH_GPT_first_part_amount_l20_2012


namespace NUMINAMATH_GPT_weaving_additional_yards_l20_2087

theorem weaving_additional_yards {d : ℝ} :
  (∃ d : ℝ, (30 * 5 + (30 * 29) / 2 * d = 390) → d = 16 / 29) :=
sorry

end NUMINAMATH_GPT_weaving_additional_yards_l20_2087


namespace NUMINAMATH_GPT_total_profit_is_50_l20_2011

-- Define the initial conditions
def initial_milk : ℕ := 80
def initial_water : ℕ := 20
def milk_cost_per_liter : ℕ := 22
def first_mixture_milk : ℕ := 40
def first_mixture_water : ℕ := 5
def first_mixture_price : ℕ := 19
def second_mixture_milk : ℕ := 25
def second_mixture_water : ℕ := 10
def second_mixture_price : ℕ := 18
def third_mixture_milk : ℕ := initial_milk - (first_mixture_milk + second_mixture_milk)
def third_mixture_water : ℕ := 5
def third_mixture_price : ℕ := 21

-- Define variables for revenue calculations
def first_mixture_revenue : ℕ := (first_mixture_milk + first_mixture_water) * first_mixture_price
def second_mixture_revenue : ℕ := (second_mixture_milk + second_mixture_water) * second_mixture_price
def third_mixture_revenue : ℕ := (third_mixture_milk + third_mixture_water) * third_mixture_price
def total_revenue : ℕ := first_mixture_revenue + second_mixture_revenue + third_mixture_revenue

-- Define the total milk cost
def total_milk_used : ℕ := first_mixture_milk + second_mixture_milk + third_mixture_milk
def total_cost : ℕ := total_milk_used * milk_cost_per_liter

-- Define the profit as the difference between total revenue and total cost
def profit : ℕ := total_revenue - total_cost

-- Prove that the total profit is Rs. 50
theorem total_profit_is_50 : profit = 50 := by
  sorry

end NUMINAMATH_GPT_total_profit_is_50_l20_2011


namespace NUMINAMATH_GPT_eleven_twelve_divisible_by_133_l20_2040

theorem eleven_twelve_divisible_by_133 (n : ℕ) (h : n > 0) : 133 ∣ (11^(n+2) + 12^(2*n+1)) := 
by 
  sorry

end NUMINAMATH_GPT_eleven_twelve_divisible_by_133_l20_2040


namespace NUMINAMATH_GPT_monotonically_increasing_interval_l20_2070

noncomputable def f (x : ℝ) : ℝ := 4 * x - x^3

theorem monotonically_increasing_interval : ∀ x1 x2 : ℝ, -2 < x1 ∧ x1 < x2 ∧ x2 < 2 → f x1 < f x2 :=
by
  intros x1 x2 h
  sorry

end NUMINAMATH_GPT_monotonically_increasing_interval_l20_2070


namespace NUMINAMATH_GPT_train_speed_l20_2028

/-- 
Theorem: Given the length of the train L = 1200 meters and the time T = 30 seconds, the speed of the train S is 40 meters per second.
-/
theorem train_speed (L : ℕ) (T : ℕ) (hL : L = 1200) (hT : T = 30) : L / T = 40 := by
  sorry

end NUMINAMATH_GPT_train_speed_l20_2028


namespace NUMINAMATH_GPT_remainder_of_171_divided_by_21_l20_2001

theorem remainder_of_171_divided_by_21 : 
  ∃ r, 171 = (21 * 8) + r ∧ r = 3 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_171_divided_by_21_l20_2001


namespace NUMINAMATH_GPT_determine_a_l20_2003

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then x^3 + 1 else x^2 - a * x

theorem determine_a (a : ℝ) : 
  f (f 0 a) a = -2 → a = 3 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l20_2003


namespace NUMINAMATH_GPT_negate_at_most_two_l20_2074

def atMost (n : Nat) : Prop := ∃ k : Nat, k ≤ n
def atLeast (n : Nat) : Prop := ∃ k : Nat, k ≥ n

theorem negate_at_most_two : ¬ atMost 2 ↔ atLeast 3 := by
  sorry

end NUMINAMATH_GPT_negate_at_most_two_l20_2074


namespace NUMINAMATH_GPT_pauls_plumbing_hourly_charge_l20_2032

theorem pauls_plumbing_hourly_charge :
  ∀ P : ℕ,
  (55 + 4 * P = 75 + 4 * 30) → 
  P = 35 :=
by
  intros P h
  sorry

end NUMINAMATH_GPT_pauls_plumbing_hourly_charge_l20_2032


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_condition_l20_2080

theorem arithmetic_sequence_geometric_condition (a : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d = 3)
  (h3 : ∃ k, a (k+3) * a k = (a (k+1)) * (a (k+2))) :
  a 2 = -9 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_condition_l20_2080


namespace NUMINAMATH_GPT_ellipse_graph_equivalence_l20_2020

theorem ellipse_graph_equivalence :
  ∀ x y : ℝ, x^2 + 4 * y^2 - 6 * x + 8 * y + 9 = 0 ↔ (x - 3)^2 / 4 + (y + 1)^2 / 1 = 1 := by
  sorry

end NUMINAMATH_GPT_ellipse_graph_equivalence_l20_2020


namespace NUMINAMATH_GPT_intercept_sum_l20_2024

theorem intercept_sum (x y : ℤ) (h1 : 0 ≤ x) (h2 : x < 42) (h3 : 0 ≤ y) (h4 : y < 42)
  (h : 5 * x ≡ 3 * y - 2 [ZMOD 42]) : (x + y) = 36 :=
by
  sorry

end NUMINAMATH_GPT_intercept_sum_l20_2024


namespace NUMINAMATH_GPT_part_I_part_II_l20_2013

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - 1

theorem part_I {a : ℝ} (ha : a = 2) :
  { x : ℝ | f x a ≥ 4 - abs (x - 4)} = { x | x ≥ 11 / 2 ∨ x ≤ 1 / 2 } := 
by 
  sorry

theorem part_II {a : ℝ} (h : { x : ℝ | abs (f (2 * x + a) a - 2 * f x a) ≤ 1 } = 
      { x | 1 / 2 ≤ x ∧ x ≤ 1 }) : 
  a = 2 := 
by 
  sorry

end NUMINAMATH_GPT_part_I_part_II_l20_2013


namespace NUMINAMATH_GPT_find_k_values_l20_2098

noncomputable def parallel_vectors (k : ℝ) : Prop :=
  (k^2 / k = (k + 1) / 4)

theorem find_k_values (k : ℝ) : parallel_vectors k ↔ (k = 0 ∨ k = 1 / 3) :=
by sorry

end NUMINAMATH_GPT_find_k_values_l20_2098


namespace NUMINAMATH_GPT_original_number_l20_2078

theorem original_number (n : ℕ) (h1 : 2319 % 21 = 0) (h2 : 2319 = 21 * (n + 1) - 1) : n = 2318 := 
sorry

end NUMINAMATH_GPT_original_number_l20_2078


namespace NUMINAMATH_GPT_average_price_per_book_l20_2044

def books_from_shop1 := 42
def price_from_shop1 := 520
def books_from_shop2 := 22
def price_from_shop2 := 248

def total_books := books_from_shop1 + books_from_shop2
def total_price := price_from_shop1 + price_from_shop2
def average_price := total_price / total_books

theorem average_price_per_book : average_price = 12 := by
  sorry

end NUMINAMATH_GPT_average_price_per_book_l20_2044


namespace NUMINAMATH_GPT_B_days_solve_l20_2097

noncomputable def combined_work_rate (A_rate B_rate C_rate : ℝ) : ℝ := A_rate + B_rate + C_rate
noncomputable def A_rate : ℝ := 1 / 6
noncomputable def C_rate : ℝ := 1 / 7.5
noncomputable def combined_rate : ℝ := 1 / 2

theorem B_days_solve : ∃ (B_days : ℝ), combined_work_rate A_rate (1 / B_days) C_rate = combined_rate ∧ B_days = 5 :=
by
  use 5
  rw [←inv_div] -- simplifying the expression of 1/B_days
  have : ℝ := sorry -- steps to cancel and simplify, proving the equality
  sorry

end NUMINAMATH_GPT_B_days_solve_l20_2097


namespace NUMINAMATH_GPT_cricketer_boundaries_l20_2058

theorem cricketer_boundaries (total_runs : ℕ) (sixes : ℕ) (percent_runs_by_running : ℝ)
  (h1 : total_runs = 152)
  (h2 : sixes = 2)
  (h3 : percent_runs_by_running = 60.526315789473685) :
  let runs_by_running := round (total_runs * percent_runs_by_running / 100)
  let runs_from_sixes := sixes * 6
  let runs_from_boundaries := total_runs - runs_by_running - runs_from_sixes
  let boundaries := runs_from_boundaries / 4
  boundaries = 12 :=
by
  sorry

end NUMINAMATH_GPT_cricketer_boundaries_l20_2058


namespace NUMINAMATH_GPT_percentage_saved_l20_2055

theorem percentage_saved (saved spent : ℝ) (h_saved : saved = 3) (h_spent : spent = 27) : 
  (saved / (saved + spent)) * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_saved_l20_2055


namespace NUMINAMATH_GPT_siblings_ate_two_slices_l20_2051

-- Let slices_after_dinner be the number of slices left after eating one-fourth of 16 slices
def slices_after_dinner : ℕ := 16 - 16 / 4

-- Let slices_after_yves be the number of slices left after Yves ate one-fourth of the remaining pizza
def slices_after_yves : ℕ := slices_after_dinner - slices_after_dinner / 4

-- Let slices_left be the number of slices left after Yves's siblings ate some slices
def slices_left : ℕ := 5

-- Let slices_eaten_by_siblings be the number of slices eaten by Yves's siblings
def slices_eaten_by_siblings : ℕ := slices_after_yves - slices_left

-- Since there are two siblings, each ate half of the slices_eaten_by_siblings
def slices_per_sibling : ℕ := slices_eaten_by_siblings / 2

-- The theorem stating that each sibling ate 2 slices
theorem siblings_ate_two_slices : slices_per_sibling = 2 :=
by
  -- Definition of slices_after_dinner
  have h1 : slices_after_dinner = 12 := by sorry
  -- Definition of slices_after_yves
  have h2 : slices_after_yves = 9 := by sorry
  -- Definition of slices_eaten_by_siblings
  have h3 : slices_eaten_by_siblings = 4 := by sorry
  -- Final assertion of slices_per_sibling
  have h4 : slices_per_sibling = 2 := by sorry
  exact h4

end NUMINAMATH_GPT_siblings_ate_two_slices_l20_2051


namespace NUMINAMATH_GPT_least_value_of_x_for_divisibility_l20_2094

theorem least_value_of_x_for_divisibility (x : ℕ) (h : 1 + 8 + 9 + 4 = 22) :
  ∃ x : ℕ, (22 + x) % 3 = 0 ∧ x = 2 := by
sorry

end NUMINAMATH_GPT_least_value_of_x_for_divisibility_l20_2094


namespace NUMINAMATH_GPT_solution_set_of_inequality_l20_2042

open Set

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - x - 6 < 0} = Ioo (-2 : ℝ) 3 := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l20_2042


namespace NUMINAMATH_GPT_beautiful_point_coordinates_l20_2073

-- Define a "beautiful point"
def is_beautiful_point (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 = P.1 * P.2

theorem beautiful_point_coordinates (M : ℝ × ℝ) : 
  is_beautiful_point M ∧ abs M.1 = 2 → 
  (M = (2, 2) ∨ M = (-2, 2/3)) :=
by sorry

end NUMINAMATH_GPT_beautiful_point_coordinates_l20_2073


namespace NUMINAMATH_GPT_find_x_l20_2062

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l20_2062


namespace NUMINAMATH_GPT_integral_equals_result_l20_2076

noncomputable def integral_value : ℝ :=
  ∫ x in 1.0..2.0, (x^2 + 1) / x

theorem integral_equals_result :
  integral_value = (3 / 2) + Real.log 2 := 
by
  sorry

end NUMINAMATH_GPT_integral_equals_result_l20_2076


namespace NUMINAMATH_GPT_common_tangent_intersects_x_axis_at_point_A_l20_2036

-- Define the ellipses using their equations
def ellipse_C1 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def ellipse_C2 (x y : ℝ) : Prop := (x - 2)^2 + 4 * y^2 = 1

-- The theorem stating the coordinates of the point where the common tangent intersects the x-axis
theorem common_tangent_intersects_x_axis_at_point_A :
  (∃ x : ℝ, (ellipse_C1 x 0 ∧ ellipse_C2 x 0) ↔ x = 4) :=
sorry

end NUMINAMATH_GPT_common_tangent_intersects_x_axis_at_point_A_l20_2036


namespace NUMINAMATH_GPT_polynomial_term_count_l20_2010

open Nat

theorem polynomial_term_count (N : ℕ) (h : (N.choose 5) = 2002) : N = 17 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_term_count_l20_2010


namespace NUMINAMATH_GPT_smarties_modulo_l20_2049

theorem smarties_modulo (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 :=
by
  sorry

end NUMINAMATH_GPT_smarties_modulo_l20_2049


namespace NUMINAMATH_GPT_calculate_f_of_f_of_f_30_l20_2017

-- Define the function f (equivalent to $\#N = 0.5N + 2$)
def f (N : ℝ) : ℝ := 0.5 * N + 2

-- The proof statement
theorem calculate_f_of_f_of_f_30 : 
  f (f (f 30)) = 7.25 :=
by
  sorry

end NUMINAMATH_GPT_calculate_f_of_f_of_f_30_l20_2017


namespace NUMINAMATH_GPT_penguins_count_l20_2081

variable (P B : ℕ)

theorem penguins_count (h1 : B = 2 * P) (h2 : P + B = 63) : P = 21 :=
by
  sorry

end NUMINAMATH_GPT_penguins_count_l20_2081


namespace NUMINAMATH_GPT_arcsin_neg_sqrt3_div_2_l20_2046

theorem arcsin_neg_sqrt3_div_2 : 
  Real.arcsin (- (Real.sqrt 3 / 2)) = - (Real.pi / 3) := 
by sorry

end NUMINAMATH_GPT_arcsin_neg_sqrt3_div_2_l20_2046


namespace NUMINAMATH_GPT_fraction_values_l20_2060

theorem fraction_values (a b c : ℚ) (h1 : a / b = 2) (h2 : b / c = 4 / 3) : c / a = 3 / 8 := 
by
  sorry

end NUMINAMATH_GPT_fraction_values_l20_2060


namespace NUMINAMATH_GPT_cannot_transform_with_swap_rows_and_columns_l20_2022

def initialTable : Matrix (Fin 3) (Fin 3) ℕ :=
![![1, 2, 3], ![4, 5, 6], ![7, 8, 9]]

def goalTable : Matrix (Fin 3) (Fin 3) ℕ :=
![![1, 4, 7], ![2, 5, 8], ![3, 6, 9]]

theorem cannot_transform_with_swap_rows_and_columns :
  ¬ ∃ (is_transformed_by_swapping : Matrix (Fin 3) (Fin 3) ℕ → Matrix (Fin 3) (Fin 3) ℕ → Prop),
    is_transformed_by_swapping initialTable goalTable :=
by sorry

end NUMINAMATH_GPT_cannot_transform_with_swap_rows_and_columns_l20_2022


namespace NUMINAMATH_GPT_sum_first_53_odd_numbers_l20_2064

-- Definitions based on the given conditions
def first_odd_number := 1

def nth_odd_number (n : ℕ) : ℕ :=
  1 + (n - 1) * 2

def sum_n_odd_numbers (n : ℕ) : ℕ :=
  (n * n)

-- Theorem statement to prove
theorem sum_first_53_odd_numbers : sum_n_odd_numbers 53 = 2809 := 
by
  sorry

end NUMINAMATH_GPT_sum_first_53_odd_numbers_l20_2064


namespace NUMINAMATH_GPT_min_value_of_a_l20_2066

theorem min_value_of_a (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (x + y) * (1/x + a/y) ≥ 16) : a ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_l20_2066


namespace NUMINAMATH_GPT_sum_in_range_l20_2063

theorem sum_in_range : 
    let a := (2:ℝ) + 1/8
    let b := (3:ℝ) + 1/3
    let c := (5:ℝ) + 1/18
    10.5 < a + b + c ∧ a + b + c < 11 := 
by 
    sorry

end NUMINAMATH_GPT_sum_in_range_l20_2063


namespace NUMINAMATH_GPT_percentage_of_red_shirts_l20_2027

theorem percentage_of_red_shirts
  (Total : ℕ) 
  (P_blue P_green : ℝ) 
  (N_other : ℕ)
  (H_Total : Total = 600)
  (H_P_blue : P_blue = 0.45) 
  (H_P_green : P_green = 0.15) 
  (H_N_other : N_other = 102) :
  ( (Total - (P_blue * Total + P_green * Total + N_other)) / Total ) * 100 = 23 := by
  sorry

end NUMINAMATH_GPT_percentage_of_red_shirts_l20_2027


namespace NUMINAMATH_GPT_twenty_four_points_game_l20_2016

theorem twenty_four_points_game :
  let a := (-6 : ℚ)
  let b := (3 : ℚ)
  let c := (4 : ℚ)
  let d := (10 : ℚ)
  3 * (d - a + c) = 24 := 
by
  sorry

end NUMINAMATH_GPT_twenty_four_points_game_l20_2016


namespace NUMINAMATH_GPT_minimum_value_quadratic_function_l20_2088

noncomputable def quadratic_function (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

theorem minimum_value_quadratic_function : ∀ x, x ≥ 0 → quadratic_function x ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_quadratic_function_l20_2088


namespace NUMINAMATH_GPT_almonds_received_by_amanda_l20_2082

variable (totalAlmonds : ℚ)
variable (numberOfPiles : ℚ)
variable (pilesForAmanda : ℚ)

-- Conditions
def stephanie_has_almonds := totalAlmonds = 66 / 7
def distribute_equally_into_piles := numberOfPiles = 6
def amanda_receives_piles := pilesForAmanda = 3

-- Conclusion to prove
theorem almonds_received_by_amanda :
  stephanie_has_almonds totalAlmonds →
  distribute_equally_into_piles numberOfPiles →
  amanda_receives_piles pilesForAmanda →
  (totalAlmonds / numberOfPiles) * pilesForAmanda = 33 / 7 :=
by
  sorry

end NUMINAMATH_GPT_almonds_received_by_amanda_l20_2082


namespace NUMINAMATH_GPT_arithmetic_sequence_solution_l20_2035

-- Definitions of a, b, c, and d in terms of d and sequence difference
def is_in_arithmetic_sequence (a b c d : ℝ) (diff : ℝ) : Prop :=
  a + diff = b ∧ b + diff = c ∧ c + diff = d

-- Conditions
def pos_real_sequence (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

def product_condition (a b c d : ℝ) (prod : ℝ) : Prop :=
  a * b * c * d = prod

-- The resulting value of d
def d_value_as_fraction (d : ℝ) : Prop :=
  d = (3 + Real.sqrt 95) / (Real.sqrt 2)

-- Proof statement
theorem arithmetic_sequence_solution :
  ∃ a b c d : ℝ, pos_real_sequence a b c d ∧ 
                 is_in_arithmetic_sequence a b c d (Real.sqrt 2) ∧ 
                 product_condition a b c d 2021 ∧ 
                 d_value_as_fraction d :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_solution_l20_2035


namespace NUMINAMATH_GPT_maximize_garden_area_length_l20_2014

noncomputable def length_parallel_to_wall (cost_per_foot : ℝ) (fence_cost : ℝ) : ℝ :=
  let total_length := fence_cost / cost_per_foot 
  let y := total_length / 4 
  let length_parallel := total_length - 2 * y
  length_parallel

theorem maximize_garden_area_length :
  ∀ (cost_per_foot fence_cost : ℝ), cost_per_foot = 10 → fence_cost = 1500 → 
  length_parallel_to_wall cost_per_foot fence_cost = 75 :=
by
  intros
  simp [length_parallel_to_wall, *]
  sorry

end NUMINAMATH_GPT_maximize_garden_area_length_l20_2014


namespace NUMINAMATH_GPT_cost_price_of_article_l20_2037

theorem cost_price_of_article 
  (CP SP : ℝ)
  (H1 : SP = 1.13 * CP)
  (H2 : 1.10 * SP = 616) :
  CP = 495.58 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l20_2037


namespace NUMINAMATH_GPT_fraction_equality_l20_2092

theorem fraction_equality (a b : ℝ) (h : a / 4 = b / 3) : b / (a - b) = 3 :=
sorry

end NUMINAMATH_GPT_fraction_equality_l20_2092


namespace NUMINAMATH_GPT_long_fur_brown_dogs_l20_2000

-- Defining the basic parameters given in the problem
def total_dogs : ℕ := 45
def long_fur : ℕ := 26
def brown_dogs : ℕ := 30
def neither_long_fur_nor_brown : ℕ := 8

-- Statement of the theorem
theorem long_fur_brown_dogs : ∃ LB : ℕ, LB = 27 ∧ total_dogs = long_fur + brown_dogs - LB + neither_long_fur_nor_brown :=
by {
  -- skipping the proof
  sorry
}

end NUMINAMATH_GPT_long_fur_brown_dogs_l20_2000


namespace NUMINAMATH_GPT_find_n_l20_2026

theorem find_n (n : ℕ) (composite_n : n > 1 ∧ ¬Prime n) : 
  ((∃ m : ℕ, ∀ d : ℕ, d ∣ n ∧ 1 < d ∧ d < n → d + 1 ∣ m ∧ 1 < d + 1 ∧ d + 1 < m) ↔ 
    (n = 4 ∨ n = 8)) :=
by sorry

end NUMINAMATH_GPT_find_n_l20_2026


namespace NUMINAMATH_GPT_range_of_m_for_quadratic_sol_in_interval_l20_2043

theorem range_of_m_for_quadratic_sol_in_interval :
  {m : ℝ // ∀ x, (x^2 + (m-1)*x + 1 = 0) → (0 ≤ x ∧ x ≤ 2)} = {m : ℝ // m < -1} :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_for_quadratic_sol_in_interval_l20_2043


namespace NUMINAMATH_GPT_work_efficiency_ratio_l20_2068

variables (A_eff B_eff : ℚ) (a b : Type)

theorem work_efficiency_ratio (h1 : B_eff = 1 / 33)
  (h2 : A_eff + B_eff = 1 / 11) :
  A_eff / B_eff = 2 :=
by 
  sorry

end NUMINAMATH_GPT_work_efficiency_ratio_l20_2068


namespace NUMINAMATH_GPT_crayons_taken_out_l20_2031

-- Define the initial and remaining number of crayons
def initial_crayons : ℕ := 7
def remaining_crayons : ℕ := 4

-- Define the proposition to prove
theorem crayons_taken_out : initial_crayons - remaining_crayons = 3 := by
  sorry

end NUMINAMATH_GPT_crayons_taken_out_l20_2031


namespace NUMINAMATH_GPT_solve_for_x_l20_2045

theorem solve_for_x :
  ∃ x : ℝ, x ≠ 0 ∧ (9 * x) ^ 18 = (27 * x) ^ 9 + 81 * x ∧ x = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l20_2045


namespace NUMINAMATH_GPT_max_sum_of_prices_l20_2079

theorem max_sum_of_prices (R P : ℝ) 
  (h1 : 4 * R + 5 * P ≥ 27) 
  (h2 : 6 * R + 3 * P ≤ 27) : 
  3 * R + 4 * P ≤ 36 :=
by 
  sorry

end NUMINAMATH_GPT_max_sum_of_prices_l20_2079


namespace NUMINAMATH_GPT_find_number_l20_2008

theorem find_number :
  (∃ m : ℝ, 56 = (3 / 2) * m) ∧ (56 = 0.7 * 80) → m = 37 := by
  sorry

end NUMINAMATH_GPT_find_number_l20_2008


namespace NUMINAMATH_GPT_max_square_test_plots_l20_2084

theorem max_square_test_plots 
  (length : ℕ) (width : ℕ) (fence_available : ℕ) 
  (side_length : ℕ) (num_plots : ℕ) 
  (h_length : length = 30)
  (h_width : width = 60)
  (h_fencing : fence_available = 2500)
  (h_side_length : side_length = 10)
  (h_num_plots : num_plots = 18) :
  (length * width / side_length^2 = num_plots) ∧
  (30 * (60 / side_length - 1) + 60 * (30 / side_length - 1) ≤ fence_available) := 
sorry

end NUMINAMATH_GPT_max_square_test_plots_l20_2084


namespace NUMINAMATH_GPT_math_problem_l20_2086

-- Define the conditions
def a := -6
def b := 2
def c := 1 / 3
def d := 3 / 4
def e := 12
def f := -3

-- Statement of the problem
theorem math_problem : a / b + (c - d) * e + f^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l20_2086


namespace NUMINAMATH_GPT_radius_of_tangent_circle_l20_2052

theorem radius_of_tangent_circle (k r : ℝ) (hk : k > 8) (h1 : k - 8 = r) (h2 : r * Real.sqrt 2 = k) : 
  r = 8 * (Real.sqrt 2 + 1) := 
sorry

end NUMINAMATH_GPT_radius_of_tangent_circle_l20_2052


namespace NUMINAMATH_GPT_probability_of_B_l20_2025

theorem probability_of_B (P : Set ℕ → ℝ) (A B : Set ℕ) (hA : P A = 0.25) (hAB : P (A ∩ B) = 0.15) (hA_complement_B_complement : P (Aᶜ ∩ Bᶜ) = 0.5) : P B = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_B_l20_2025


namespace NUMINAMATH_GPT_cost_of_fencing_per_meter_l20_2091

def rectangular_farm_area : Real := 1200
def short_side_length : Real := 30
def total_cost : Real := 1440

theorem cost_of_fencing_per_meter : (total_cost / (short_side_length + (rectangular_farm_area / short_side_length) + Real.sqrt ((rectangular_farm_area / short_side_length)^2 + short_side_length^2))) = 12 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_per_meter_l20_2091


namespace NUMINAMATH_GPT_net_increase_in_bicycle_stock_l20_2015

-- Definitions for changes in stock over the three days
def net_change_friday : ℤ := 15 - 10
def net_change_saturday : ℤ := 8 - 12
def net_change_sunday : ℤ := 11 - 9

-- Total net increase in stock
def total_net_increase : ℤ := net_change_friday + net_change_saturday + net_change_sunday

-- Theorem statement
theorem net_increase_in_bicycle_stock : total_net_increase = 3 := by
  -- We would provide the detailed proof here.
  sorry

end NUMINAMATH_GPT_net_increase_in_bicycle_stock_l20_2015


namespace NUMINAMATH_GPT_lamps_remain_lit_after_toggling_l20_2033

theorem lamps_remain_lit_after_toggling :
  let n := 1997
  let lcm_2_3_5 := Nat.lcm (Nat.lcm 2 3) 5
  let multiples_30 := n / 30
  let multiples_6 := n / (2 * 3)
  let multiples_15 := n / (3 * 5)
  let multiples_10 := n / (2 * 5)
  let multiples_2 := n / 2
  let multiples_3 := n / 3
  let multiples_5 := n / 5
  let pulled_three_times := multiples_30
  let pulled_twice := multiples_6 + multiples_15 + multiples_10 - 3 * pulled_three_times
  let pulled_once := multiples_2 + multiples_3 + multiples_5 - 2 * pulled_twice - 3 * pulled_three_times
  1997 - pulled_three_times - pulled_once = 999 := by
  let n := 1997
  let lcm_2_3_5 := Nat.lcm (Nat.lcm 2 3) 5
  let multiples_30 := n / 30
  let multiples_6 := n / (2 * 3)
  let multiples_15 := n / (3 * 5)
  let multiples_10 := n / (2 * 5)
  let multiples_2 := n / 2
  let multiples_3 := n / 3
  let multiples_5 := n / 5
  let pulled_three_times := multiples_30
  let pulled_twice := multiples_6 + multiples_15 + multiples_10 - 3 * pulled_three_times
  let pulled_once := multiples_2 + multiples_3 + multiples_5 - 2 * pulled_twice - 3 * pulled_three_times
  have h : 1997 - pulled_three_times - (pulled_once) = 999 := sorry
  exact h

end NUMINAMATH_GPT_lamps_remain_lit_after_toggling_l20_2033


namespace NUMINAMATH_GPT_f_decreasing_in_interval_l20_2047

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

noncomputable def shifted_g (x : ℝ) : ℝ := g (x + Real.pi / 6)

noncomputable def f (x : ℝ) : ℝ := shifted_g (2 * x)

theorem f_decreasing_in_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 4 → f y < f x :=
by
  sorry

end NUMINAMATH_GPT_f_decreasing_in_interval_l20_2047


namespace NUMINAMATH_GPT_no_triangle_tangent_l20_2053

open Real

/-- Given conditions --/
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C2 (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0 ∧ (1 / a^2) + (1 / b^2) = 1

theorem no_triangle_tangent (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : (1 : ℝ) / a^2 + 1 / b^2 = 1) :
  ¬∃ (A B C : ℝ × ℝ), 
    (C1 A.1 A.2) ∧ (C1 B.1 B.2) ∧ (C1 C.1 C.2) ∧
    (∃ (l : ℝ) (m : ℝ) (n : ℝ), C2 l m a b ∧ C2 n l a b) :=
by
  sorry

end NUMINAMATH_GPT_no_triangle_tangent_l20_2053


namespace NUMINAMATH_GPT_sum_of_coefficients_l20_2067

theorem sum_of_coefficients (x : ℝ) : (∃ x : ℝ, 5 * x * (1 - x) = 3) → 5 + (-5) + 3 = 3 :=
by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l20_2067


namespace NUMINAMATH_GPT_square_side_length_difference_l20_2009

theorem square_side_length_difference : 
  let side_A := Real.sqrt 25
  let side_B := Real.sqrt 81
  side_B - side_A = 4 :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_difference_l20_2009


namespace NUMINAMATH_GPT_mother_l20_2048

theorem mother's_age (D M : ℕ) (h1 : 2 * D + M = 70) (h2 : D + 2 * M = 95) : M = 40 :=
sorry

end NUMINAMATH_GPT_mother_l20_2048


namespace NUMINAMATH_GPT_range_of_a_l20_2096

def A : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }

theorem range_of_a (a : ℝ) (h : a ∈ A) : -1 ≤ a ∧ a ≤ 3 := 
sorry

end NUMINAMATH_GPT_range_of_a_l20_2096


namespace NUMINAMATH_GPT_trigonometric_identities_l20_2095

theorem trigonometric_identities (α : Real) (h1 : 3 * π / 2 < α ∧ α < 2 * π) (h2 : Real.sin α = -3 / 5) :
  Real.tan α = 3 / 4 ∧ Real.tan (α - π / 4) = -1 / 7 ∧ Real.cos (2 * α) = 7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identities_l20_2095


namespace NUMINAMATH_GPT_tens_digit_3_pow_2016_eq_2_l20_2029

def tens_digit (n : Nat) : Nat := (n / 10) % 10

theorem tens_digit_3_pow_2016_eq_2 : tens_digit (3 ^ 2016) = 2 := by
  sorry

end NUMINAMATH_GPT_tens_digit_3_pow_2016_eq_2_l20_2029


namespace NUMINAMATH_GPT_wine_ages_l20_2099

-- Define the ages of the wines as variables
variable (C F T B Bo M : ℝ)

-- Define the six conditions
axiom h1 : F = 3 * C
axiom h2 : C = 4 * T
axiom h3 : B = (1 / 2) * T
axiom h4 : Bo = 2 * F
axiom h5 : M^2 = Bo
axiom h6 : C = 40

-- Prove the ages of the wines 
theorem wine_ages : 
  F = 120 ∧ 
  T = 10 ∧ 
  B = 5 ∧ 
  Bo = 240 ∧ 
  M = Real.sqrt 240 :=
by
  sorry

end NUMINAMATH_GPT_wine_ages_l20_2099


namespace NUMINAMATH_GPT_jim_total_cars_l20_2071

theorem jim_total_cars (B F C : ℕ) (h1 : B = 4 * F) (h2 : F = 2 * C + 3) (h3 : B = 220) :
  B + F + C = 301 :=
by
  sorry

end NUMINAMATH_GPT_jim_total_cars_l20_2071


namespace NUMINAMATH_GPT_find_n_on_angle_bisector_l20_2004

theorem find_n_on_angle_bisector (M : ℝ × ℝ) (hM : M = (3 * n - 2, 2 * n + 7) ∧ M.1 + M.2 = 0) : 
    n = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_n_on_angle_bisector_l20_2004


namespace NUMINAMATH_GPT_silk_dyeing_total_correct_l20_2021

open Real

theorem silk_dyeing_total_correct :
  let green := 61921
  let pink := 49500
  let blue := 75678
  let yellow := 34874.5
  let total_without_red := green + pink + blue + yellow
  let red := 0.10 * total_without_red
  let total_with_red := total_without_red + red
  total_with_red = 245270.85 :=
by
  sorry

end NUMINAMATH_GPT_silk_dyeing_total_correct_l20_2021


namespace NUMINAMATH_GPT_sum_divisible_by_15_l20_2089

theorem sum_divisible_by_15 (a : ℤ) : 15 ∣ (9 * a^5 - 5 * a^3 - 4 * a) :=
sorry

end NUMINAMATH_GPT_sum_divisible_by_15_l20_2089


namespace NUMINAMATH_GPT_tangent_line_curve_l20_2050

theorem tangent_line_curve (a b : ℚ) 
  (h1 : 3 * a + b = 1) 
  (h2 : a + b = 2) : 
  b - a = 3 := 
by 
  sorry

end NUMINAMATH_GPT_tangent_line_curve_l20_2050


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l20_2057

variable (a b c : ℝ)

-- Define the condition that the sequence forms a geometric sequence
def geometric_sequence (a1 a2 a3 a4 a5 : ℝ) :=
  ∃ q : ℝ, q ≠ 0 ∧ a1 * q = a2 ∧ a2 * q = a3 ∧ a3 * q = a4 ∧ a4 * q = a5

-- Lean statement proving the problem
theorem sufficient_not_necessary_condition :
  (geometric_sequence 1 a b c 16) → (b = 4) ∧ ¬ (b = 4 → geometric_sequence 1 a b c 16) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l20_2057


namespace NUMINAMATH_GPT_find_divisor_l20_2018

theorem find_divisor : 
  ∀ (dividend quotient remainder divisor : ℕ), 
    dividend = 140 →
    quotient = 9 →
    remainder = 5 →
    dividend = (divisor * quotient) + remainder →
    divisor = 15 :=
by
  intros dividend quotient remainder divisor hd hq hr hdiv
  sorry

end NUMINAMATH_GPT_find_divisor_l20_2018


namespace NUMINAMATH_GPT_max_OM_ON_value_l20_2061

noncomputable def maximum_OM_ON (a b : ℝ) : ℝ :=
  (1 + Real.sqrt 2) / 2 * (a + b)

-- Given the conditions in triangle ABC with sides BC and AC having fixed lengths a and b respectively,
-- and that AB can vary such that a square is constructed outward on side AB with center O,
-- and M and N are the midpoints of sides BC and AC respectively, prove the maximum value of OM + ON.
theorem max_OM_ON_value (a b : ℝ) : 
  ∃ OM ON : ℝ, OM + ON = maximum_OM_ON a b :=
sorry

end NUMINAMATH_GPT_max_OM_ON_value_l20_2061


namespace NUMINAMATH_GPT_det_A_eq_l20_2006

open Matrix

def A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, -3, 3],
    ![x, 5, -1],
    ![4, -2, 1]]

theorem det_A_eq (x : ℝ) : det (A x) = -3 * x - 45 :=
by sorry

end NUMINAMATH_GPT_det_A_eq_l20_2006


namespace NUMINAMATH_GPT_value_of_square_l20_2075

theorem value_of_square (z : ℝ) (h : 3 * z^2 + 2 * z = 5 * z + 11) : (6 * z - 5)^2 = 141 := by
  sorry

end NUMINAMATH_GPT_value_of_square_l20_2075


namespace NUMINAMATH_GPT_inequality_relationship_l20_2002

noncomputable def a := 1 / 2023
noncomputable def b := Real.exp (-2022 / 2023)
noncomputable def c := (Real.cos (1 / 2023)) / 2023

theorem inequality_relationship : b > a ∧ a > c :=
by
  -- Initializing and defining the variables
  let a := a
  let b := b
  let c := c
  -- Providing the required proof
  sorry

end NUMINAMATH_GPT_inequality_relationship_l20_2002


namespace NUMINAMATH_GPT_opposite_of_negative_seven_l20_2093

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_negative_seven : opposite (-7) = 7 := 
by 
  sorry

end NUMINAMATH_GPT_opposite_of_negative_seven_l20_2093


namespace NUMINAMATH_GPT_certain_number_is_32_l20_2038

theorem certain_number_is_32 (k t : ℚ) (certain_number : ℚ) 
  (h1 : t = 5/9 * (k - certain_number))
  (h2 : t = 75) (h3 : k = 167) :
  certain_number = 32 :=
sorry

end NUMINAMATH_GPT_certain_number_is_32_l20_2038


namespace NUMINAMATH_GPT_minimum_value_is_14_div_27_l20_2005

noncomputable def minimum_value_expression (x : ℝ) : ℝ :=
  (Real.sin x)^8 + (Real.cos x)^8 + 1 / (Real.sin x)^6 + (Real.cos x)^6 + 1

theorem minimum_value_is_14_div_27 :
  ∃ x : ℝ, minimum_value_expression x = (14 / 27) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_is_14_div_27_l20_2005


namespace NUMINAMATH_GPT_stacy_current_height_l20_2039

-- Conditions
def last_year_height_stacy : ℕ := 50
def brother_growth : ℕ := 1
def stacy_growth : ℕ := brother_growth + 6

-- Statement to prove
theorem stacy_current_height : last_year_height_stacy + stacy_growth = 57 :=
by
  sorry

end NUMINAMATH_GPT_stacy_current_height_l20_2039


namespace NUMINAMATH_GPT_vijay_work_alone_in_24_days_l20_2085

theorem vijay_work_alone_in_24_days (ajay_rate vijay_rate combined_rate : ℝ) 
  (h1 : ajay_rate = 1 / 8) 
  (h2 : combined_rate = 1 / 6) 
  (h3 : ajay_rate + vijay_rate = combined_rate) : 
  vijay_rate = 1 / 24 := 
sorry

end NUMINAMATH_GPT_vijay_work_alone_in_24_days_l20_2085


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l20_2065

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l20_2065


namespace NUMINAMATH_GPT_boys_and_girls_total_l20_2054

theorem boys_and_girls_total (c : ℕ) (h_lollipop_fraction : c = 90) 
  (h_one_third_lollipops : c / 3 = 30)
  (h_lollipops_shared : 30 / 3 = 10) 
  (h_candy_caness_shared : 60 / 2 = 30) : 
  10 + 30 = 40 :=
by
  simp [h_one_third_lollipops, h_lollipops_shared, h_candy_caness_shared]

end NUMINAMATH_GPT_boys_and_girls_total_l20_2054


namespace NUMINAMATH_GPT_length_of_fountain_built_by_20_men_in_6_days_l20_2077

noncomputable def work (workers : ℕ) (days : ℕ) : ℕ :=
  workers * days

theorem length_of_fountain_built_by_20_men_in_6_days :
  (work 35 3) / (work 20 6) * 49 = 56 :=
by
  sorry

end NUMINAMATH_GPT_length_of_fountain_built_by_20_men_in_6_days_l20_2077


namespace NUMINAMATH_GPT_plywood_cut_difference_l20_2090

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end NUMINAMATH_GPT_plywood_cut_difference_l20_2090


namespace NUMINAMATH_GPT_min_value_M_l20_2072

theorem min_value_M 
  (S_n : ℕ → ℝ)
  (T_n : ℕ → ℝ)
  (a : ℕ → ℝ)
  (h1 : ∀ n, S_n n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1)))
  (h2 : a 4 - a 2 = 8)
  (h3 : a 3 + a 5 = 26)
  (h4 : ∀ n, T_n n = S_n n / n^2) :
  ∃ M : ℝ, M = 2 ∧ (∀ n > 0, T_n n ≤ M) :=
by sorry

end NUMINAMATH_GPT_min_value_M_l20_2072


namespace NUMINAMATH_GPT_find_ABC_base10_l20_2056

theorem find_ABC_base10
  (A B C : ℕ)
  (h1 : 0 < A ∧ A < 6)
  (h2 : 0 < B ∧ B < 6)
  (h3 : 0 < C ∧ C < 6)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h5 : B + C = 6)
  (h6 : A + 1 = C)
  (h7 : A + B = C) :
  100 * A + 10 * B + C = 415 :=
by
  sorry

end NUMINAMATH_GPT_find_ABC_base10_l20_2056
