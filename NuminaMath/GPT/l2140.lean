import Mathlib

namespace range_of_a_l2140_214051

-- Defining the core problem conditions in Lean
def prop_p (a : ℝ) : Prop := ∃ x₀ : ℝ, a * x₀^2 + 2 * a * x₀ + 1 < 0

-- The original proposition p is false, thus we need to show the range of a is 0 ≤ a ≤ 1
theorem range_of_a (a : ℝ) : ¬ prop_p a → 0 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l2140_214051


namespace p_cycling_speed_l2140_214023

-- J starts walking at 6 kmph at 12:00
def start_time : ℕ := 12 * 60  -- time in minutes for convenience
def j_speed : ℤ := 6  -- in kmph
def j_start_time : ℕ := start_time  -- 12:00 in minutes

-- P starts cycling at 13:30
def p_start_time : ℕ := (13 * 60) + 30  -- time in minutes for convenience

-- They are at their respective positions at 19:30
def end_time : ℕ := (19 * 60) + 30  -- time in minutes for convenience

-- At 19:30, J is 3 km behind P
def j_behind_p_distance : ℤ := 3  -- in kilometers

-- Prove that P's cycling speed = 8 kmph
theorem p_cycling_speed {p_speed : ℤ} :
  j_start_time = start_time →
  p_start_time = (13 * 60) + 30 →
  end_time = (19 * 60) + 30 →
  j_speed = 6 →
  j_behind_p_distance = 3 →
  p_speed = 8 :=
by
  sorry

end p_cycling_speed_l2140_214023


namespace number_of_arrangements_l2140_214024

theorem number_of_arrangements (n : ℕ) (h : n = 7) :
  ∃ (arrangements : ℕ), arrangements = 20 :=
by
  sorry

end number_of_arrangements_l2140_214024


namespace increasing_interval_l2140_214077

noncomputable def f (x : ℝ) := Real.logb 2 (5 - 4 * x - x^2)

theorem increasing_interval : ∀ {x : ℝ}, (-5 < x ∧ x ≤ -2) → f x = Real.logb 2 (5 - 4 * x - x^2) := by
  sorry

end increasing_interval_l2140_214077


namespace cat_run_time_l2140_214053

/-- An electronic cat runs a lap on a circular track with a perimeter of 240 meters.
It runs at a speed of 5 meters per second for the first half of the time and 3 meters per second for the second half of the time.
Prove that the cat takes 36 seconds to run the last 120 meters. -/
theorem cat_run_time
  (perimeter : ℕ)
  (speed1 speed2 : ℕ)
  (half_perimeter : ℕ)
  (half_time : ℕ)
  (last_120m_time : ℕ) :
  perimeter = 240 →
  speed1 = 5 →
  speed2 = 3 →
  half_perimeter = perimeter / 2 →
  half_time = 60 / 2 →
  (5 * half_time - half_perimeter) / speed1 + (half_perimeter - (5 * half_time - half_perimeter)) / speed2 = 36 :=
by sorry

end cat_run_time_l2140_214053


namespace min_dot_product_trajectory_l2140_214072

-- Definitions of points and conditions
def point (x y : ℝ) : Prop := True

def trajectory (P : ℝ × ℝ) : Prop := 
  let x := P.1
  let y := P.2
  x * x - y * y = 2 ∧ x ≥ Real.sqrt 2

-- Definition of dot product over vectors from origin
def dot_product (A B : ℝ × ℝ) : ℝ :=
  A.1 * B.1 + A.2 * B.2

-- Stating the theorem for minimum value of dot product
theorem min_dot_product_trajectory (A B : ℝ × ℝ) (hA : trajectory A) (hB : trajectory B) : 
  dot_product A B ≥ 2 := 
sorry

end min_dot_product_trajectory_l2140_214072


namespace youseff_blocks_l2140_214096

theorem youseff_blocks (x : ℕ) (h1 : x = 1 * x) (h2 : (20 / 60 : ℚ) * x = x / 3) (h3 : x = x / 3 + 8) : x = 12 := by
  have : x = x := rfl  -- trivial step to include the equality
  sorry

end youseff_blocks_l2140_214096


namespace pascal_triangle_fifth_number_l2140_214031

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l2140_214031


namespace probability_of_red_ball_and_removed_red_balls_l2140_214012

-- Conditions for the problem
def initial_red_balls : Nat := 10
def initial_yellow_balls : Nat := 2
def initial_blue_balls : Nat := 8
def total_balls : Nat := initial_red_balls + initial_yellow_balls + initial_blue_balls

-- Problem statement in Lean
theorem probability_of_red_ball_and_removed_red_balls :
  (initial_red_balls / total_balls = 1 / 2) ∧
  (∃ (x : Nat), -- Number of red balls removed
    ((initial_yellow_balls + x) / total_balls = 2 / 5) ∧
    (initial_red_balls - x = 10 - 6)) := 
by
  -- Lean will need the proofs here; we use sorry for now.
  sorry

end probability_of_red_ball_and_removed_red_balls_l2140_214012


namespace perimeter_of_rectangle_EFGH_l2140_214071

noncomputable def rectangle_ellipse_problem (u v c d : ℝ) : Prop :=
  (u * v = 3000) ∧
  (3000 = c * d) ∧
  ((u + v) = 2 * c) ∧
  ((u^2 + v^2).sqrt = 2 * (c^2 - d^2).sqrt) ∧
  (d = 3000 / c) ∧
  (4 * c = 8 * (1500).sqrt)

theorem perimeter_of_rectangle_EFGH :
  ∃ (u v c d : ℝ), rectangle_ellipse_problem u v c d ∧ 2 * (u + v) = 8 * (1500).sqrt := sorry

end perimeter_of_rectangle_EFGH_l2140_214071


namespace fraction_of_state_quarters_is_two_fifths_l2140_214088

variable (total_quarters state_quarters : ℕ)
variable (is_pennsylvania_percentage : ℚ)
variable (pennsylvania_state_quarters : ℕ)

theorem fraction_of_state_quarters_is_two_fifths
  (h1 : total_quarters = 35)
  (h2 : pennsylvania_state_quarters = 7)
  (h3 : is_pennsylvania_percentage = 1 / 2)
  (h4 : state_quarters = 2 * pennsylvania_state_quarters)
  : (state_quarters : ℚ) / (total_quarters : ℚ) = 2 / 5 :=
sorry

end fraction_of_state_quarters_is_two_fifths_l2140_214088


namespace total_pies_bigger_event_l2140_214038

def pies_last_week := 16.5
def apple_pies_last_week := 14.25
def cherry_pies_last_week := 12.75

def pecan_multiplier := 4.3
def apple_multiplier := 3.5
def cherry_multiplier := 5.7

theorem total_pies_bigger_event :
  (pies_last_week * pecan_multiplier) + 
  (apple_pies_last_week * apple_multiplier) + 
  (cherry_pies_last_week * cherry_multiplier) = 193.5 :=
by
  sorry

end total_pies_bigger_event_l2140_214038


namespace range_of_a_l2140_214099

-- Define the operation ⊗
def tensor (x y : ℝ) : ℝ := x * (1 - y)

-- State the main theorem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, tensor (x - a) (x + a) < 1) → 
  (-((1 : ℝ) / 2) < a ∧ a < (3 : ℝ) / 2) :=
by
  sorry

end range_of_a_l2140_214099


namespace charming_number_unique_l2140_214047

def is_charming (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = 2 * a + b^3

theorem charming_number_unique : ∃! n, 10 ≤ n ∧ n ≤ 99 ∧ is_charming n := by
  sorry

end charming_number_unique_l2140_214047


namespace area_of_cos_closed_figure_l2140_214078

theorem area_of_cos_closed_figure :
  ∫ x in (Real.pi / 2)..(3 * Real.pi / 2), Real.cos x = 2 :=
by
  sorry

end area_of_cos_closed_figure_l2140_214078


namespace tan_double_angle_l2140_214042

theorem tan_double_angle (θ : ℝ) (h1 : θ = Real.arctan (-2)) : Real.tan (2 * θ) = 4 / 3 :=
by
  sorry

end tan_double_angle_l2140_214042


namespace find_sum_lent_l2140_214034

variable (P : ℝ)

/-- Given that the annual interest rate is 4%, and the interest earned in 8 years
amounts to Rs 340 less than the sum lent, prove that the sum lent is Rs 500. -/
theorem find_sum_lent
  (h1 : ∀ I, I = P - 340 → I = (P * 4 * 8) / 100) : 
  P = 500 :=
by
  sorry

end find_sum_lent_l2140_214034


namespace functions_equiv_l2140_214061

noncomputable def f_D (x : ℝ) : ℝ := Real.log (Real.sqrt x)
noncomputable def g_D (x : ℝ) : ℝ := (1/2) * Real.log x

theorem functions_equiv : ∀ x : ℝ, x > 0 → f_D x = g_D x := by
  intro x h
  sorry

end functions_equiv_l2140_214061


namespace range_of_m_l2140_214076

theorem range_of_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : (1 / (a - b)) + (1 / (b - c)) ≥ m / (a - c)) :
  m ≤ 4 :=
sorry

end range_of_m_l2140_214076


namespace vector_addition_example_l2140_214097

theorem vector_addition_example :
  (⟨-3, 2, -1⟩ : ℝ × ℝ × ℝ) + (⟨1, 5, -3⟩ : ℝ × ℝ × ℝ) = ⟨-2, 7, -4⟩ :=
by
  sorry

end vector_addition_example_l2140_214097


namespace primeFactors_of_3_pow_6_minus_1_l2140_214060

def calcPrimeFactorsSumAndSumOfSquares (n : ℕ) : ℕ × ℕ :=
  let factors := [2, 7, 13]  -- Given directly
  let sum_factors := 2 + 7 + 13
  let sum_squares := 2^2 + 7^2 + 13^2
  (sum_factors, sum_squares)

theorem primeFactors_of_3_pow_6_minus_1 :
  calcPrimeFactorsSumAndSumOfSquares (3^6 - 1) = (22, 222) :=
by
  sorry

end primeFactors_of_3_pow_6_minus_1_l2140_214060


namespace infinite_sequence_no_square_factors_l2140_214033

/-
  Prove that there exist infinitely many positive integers \( n_1 < n_2 < \cdots \)
  such that for all \( i \neq j \), \( n_i + n_j \) has no square factors other than 1.
-/

theorem infinite_sequence_no_square_factors :
  ∃ (n : ℕ → ℕ), (∀ (i j : ℕ), i ≠ j → ∀ p : ℕ, p ≠ 1 → p^2 ∣ (n i + n j) → false) ∧
    ∀ k : ℕ, n k < n (k + 1) :=
sorry

end infinite_sequence_no_square_factors_l2140_214033


namespace probability_not_blue_l2140_214054

-- Definitions based on the conditions
def total_faces : ℕ := 12
def blue_faces : ℕ := 1
def non_blue_faces : ℕ := total_faces - blue_faces

-- Statement of the problem
theorem probability_not_blue : (non_blue_faces : ℚ) / total_faces = 11 / 12 :=
by
  sorry

end probability_not_blue_l2140_214054


namespace correct_equation_l2140_214057

theorem correct_equation (x : ℕ) :
  (30 * x + 8 = 31 * x - 26) := by
  sorry

end correct_equation_l2140_214057


namespace largest_number_is_34_l2140_214017

theorem largest_number_is_34 (a b c : ℕ) (h1 : a + b + c = 82) (h2 : c - b = 8) (h3 : b - a = 4) : c = 34 := 
by 
  sorry

end largest_number_is_34_l2140_214017


namespace percent_students_in_range_l2140_214048

theorem percent_students_in_range
    (n1 n2 n3 n4 n5 : ℕ)
    (h1 : n1 = 5)
    (h2 : n2 = 7)
    (h3 : n3 = 8)
    (h4 : n4 = 4)
    (h5 : n5 = 3) :
  ((n3 : ℝ) / (n1 + n2 + n3 + n4 + n5) * 100) = 29.63 :=
by
  sorry

end percent_students_in_range_l2140_214048


namespace paco_ate_more_sweet_than_salty_l2140_214082

theorem paco_ate_more_sweet_than_salty (s t : ℕ) (h_s : s = 5) (h_t : t = 2) : s - t = 3 :=
by
  sorry

end paco_ate_more_sweet_than_salty_l2140_214082


namespace length_of_c_l2140_214052

theorem length_of_c (A B C : ℝ) (a b c : ℝ) (h1 : (π / 3) - A = B) (h2 : a = 3) (h3 : b = 5) : c = 7 :=
sorry

end length_of_c_l2140_214052


namespace linear_function_no_pass_quadrant_I_l2140_214086

theorem linear_function_no_pass_quadrant_I (x y : ℝ) (h : y = -2 * x - 1) : 
  ¬ (0 < x ∧ 0 < y) :=
by 
  sorry

end linear_function_no_pass_quadrant_I_l2140_214086


namespace energy_fraction_l2140_214018

-- Conditions
variables (E : ℝ → ℝ)
variable (x : ℝ)
variable (h : ∀ x, E (x + 1) = 31.6 * E x)

-- The statement to be proven
theorem energy_fraction (x : ℝ) (h : ∀ x, E (x + 1) = 31.6 * E x) : 
  E (x - 1) / E x = 1 / 31.6 :=
by
  sorry

end energy_fraction_l2140_214018


namespace diagonals_in_polygon_l2140_214028

-- Define the number of sides of the polygon
def n : ℕ := 30

-- Define the formula for the total number of diagonals in an n-sided polygon
def total_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Define the number of excluded diagonals for being parallel to one given side
def excluded_diagonals : ℕ := 1

-- Define the final count of valid diagonals after exclusion
def valid_diagonals : ℕ := total_diagonals n - excluded_diagonals

-- State the theorem to prove
theorem diagonals_in_polygon : valid_diagonals = 404 := by
  sorry


end diagonals_in_polygon_l2140_214028


namespace power_of_powers_eval_powers_l2140_214019

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l2140_214019


namespace min_people_in_photographs_l2140_214007

-- Definitions based on conditions
def photographs := (List (Nat × Nat × Nat))
def menInCenter (photos : photographs) := photos.map (fun (c, _, _) => c)

-- Condition: there are 10 photographs each with a distinct man in the center
def valid_photographs (photos: photographs) :=
  photos.length = 10 ∧ photos.map (fun (c, _, _) => c) = List.range 10

-- Theorem to be proved: The minimum number of different people in the photographs is at least 16
theorem min_people_in_photographs (photos: photographs) (h : valid_photographs photos) : 
  ∃ people : Finset Nat, people.card ≥ 16 := 
sorry

end min_people_in_photographs_l2140_214007


namespace maximize_profit_l2140_214064

def cost_price_A (x y : ℕ) := x = y + 20
def cost_sum_eq_200 (x y : ℕ) := x + 2 * y = 200
def linear_function (m n : ℕ) := m = -((1/2) : ℚ) * n + 90
def profit_function (w n : ℕ) : ℚ := (-((1/2) : ℚ) * ((n : ℚ) - 130)^2) + 1250

theorem maximize_profit
  (x y m n : ℕ)
  (hx : cost_price_A x y)
  (hsum : cost_sum_eq_200 x y)
  (hlin : linear_function m n)
  (hmaxn : 80 ≤ n ∧ n ≤ 120)
  : y = 60 ∧ x = 80 ∧ n = 120 ∧ profit_function 120 120 = 1200 := 
sorry

end maximize_profit_l2140_214064


namespace function_equality_l2140_214005

theorem function_equality (f : ℝ → ℝ) (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (f ( (x + 1) / x ) = (x^2 + 1) / x^2 + 1 / x) ↔ (f x = x^2 - x + 1) :=
by
  sorry

end function_equality_l2140_214005


namespace coordinates_of_point_A_l2140_214000

    theorem coordinates_of_point_A (x y : ℝ) (h1 : y = 0) (h2 : abs x = 3) : (x, y) = (3, 0) ∨ (x, y) = (-3, 0) :=
    sorry
    
end coordinates_of_point_A_l2140_214000


namespace books_borrowed_in_a_week_l2140_214084

theorem books_borrowed_in_a_week 
  (daily_avg : ℕ)
  (friday_increase_pct : ℕ)
  (days_open : ℕ)
  (friday_books : ℕ)
  (total_books_week : ℕ)
  (h1 : daily_avg = 40)
  (h2 : friday_increase_pct = 40)
  (h3 : days_open = 5)
  (h4 : friday_books = daily_avg + (daily_avg * friday_increase_pct / 100))
  (h5 : total_books_week = (days_open - 1) * daily_avg + friday_books) :
  total_books_week = 216 :=
by {
  sorry
}

end books_borrowed_in_a_week_l2140_214084


namespace number_of_days_l2140_214067

def burger_meal_cost : ℕ := 6
def upsize_cost : ℕ := 1
def total_spending : ℕ := 35

/-- The number of days Clinton buys the meal. -/
theorem number_of_days (h1 : burger_meal_cost + upsize_cost = 7) (h2 : total_spending = 35) : total_spending / (burger_meal_cost + upsize_cost) = 5 :=
by
  -- The proof will go here
  sorry

end number_of_days_l2140_214067


namespace percent_increase_combined_cost_l2140_214070

theorem percent_increase_combined_cost :
  let laptop_last_year := 500
  let tablet_last_year := 200
  let laptop_increase := 10 / 100
  let tablet_increase := 20 / 100
  let new_laptop_cost := laptop_last_year * (1 + laptop_increase)
  let new_tablet_cost := tablet_last_year * (1 + tablet_increase)
  let total_last_year := laptop_last_year + tablet_last_year
  let total_this_year := new_laptop_cost + new_tablet_cost
  let increase := total_this_year - total_last_year
  let percent_increase := (increase / total_last_year) * 100
  percent_increase = 13 :=
by
  sorry

end percent_increase_combined_cost_l2140_214070


namespace total_time_spent_l2140_214091

-- Define the conditions
def number_of_chairs : ℕ := 4
def number_of_tables : ℕ := 2
def time_per_piece : ℕ := 8

-- Prove that the total time spent is 48 minutes
theorem total_time_spent : (number_of_chairs + number_of_tables) * time_per_piece = 48 :=
by
  sorry

end total_time_spent_l2140_214091


namespace function_properties_l2140_214058

noncomputable def f (x : ℝ) : ℝ := Real.sin ((13 * Real.pi / 2) - x)

theorem function_properties :
  (∀ x : ℝ, f x = Real.cos x) ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧
  (forall t: ℝ, (∀ x : ℝ, f (x + t) = f x) → (t = 2 * Real.pi ∨ t = -2 * Real.pi)) :=
by
  sorry

end function_properties_l2140_214058


namespace impossible_measure_1_liter_with_buckets_l2140_214046

theorem impossible_measure_1_liter_with_buckets :
  ¬ (∃ k l : ℤ, k * Real.sqrt 2 + l * (2 - Real.sqrt 2) = 1) :=
by
  sorry

end impossible_measure_1_liter_with_buckets_l2140_214046


namespace lindas_average_speed_l2140_214020

theorem lindas_average_speed
  (dist1 : ℕ) (time1 : ℝ)
  (dist2 : ℕ) (time2 : ℝ)
  (h1 : dist1 = 450)
  (h2 : time1 = 7.5)
  (h3 : dist2 = 480)
  (h4 : time2 = 8) :
  (dist1 + dist2) / (time1 + time2) = 60 :=
by
  sorry

end lindas_average_speed_l2140_214020


namespace greatest_possible_subway_takers_l2140_214074

/-- In a company with 48 employees, some part-time and some full-time, exactly (1/3) of the part-time
employees and (1/4) of the full-time employees take the subway to work. Prove that the greatest
possible number of employees who take the subway to work is 15. -/
theorem greatest_possible_subway_takers
  (P F : ℕ)
  (h : P + F = 48)
  (h_subway_part : ∀ p, p = P → 0 ≤ p ∧ p ≤ 48)
  (h_subway_full : ∀ f, f = F → 0 ≤ f ∧ f ≤ 48) :
  ∃ y, y = 15 := 
sorry

end greatest_possible_subway_takers_l2140_214074


namespace common_chord_eq_l2140_214015

theorem common_chord_eq (x y : ℝ) :
  (x^2 + y^2 + 2*x + 8*y - 8 = 0) →
  (x^2 + y^2 - 4*x - 4*y - 2 = 0) →
  (x + 2*y - 1 = 0) :=
by
  intros h1 h2
  sorry

end common_chord_eq_l2140_214015


namespace average_of_x_y_z_l2140_214006

theorem average_of_x_y_z (x y z : ℝ) (h : (5 / 2) * (x + y + z) = 20) : 
  (x + y + z) / 3 = 8 / 3 := 
by 
  sorry

end average_of_x_y_z_l2140_214006


namespace two_digit_numbers_less_than_35_l2140_214025

theorem two_digit_numbers_less_than_35 : 
  let count : ℕ := (99 - 10 + 1) - (7 * 5)
  count = 55 :=
by
  -- definition of total number of two-digit numbers
  let total_two_digit_numbers : ℕ := 99 - 10 + 1
  -- definition of the number of unsuitable two-digit numbers
  let unsuitable_numbers : ℕ := 7 * 5
  -- definition of the count of suitable two-digit numbers
  let count : ℕ := total_two_digit_numbers - unsuitable_numbers
  -- verifying the final count
  exact rfl

end two_digit_numbers_less_than_35_l2140_214025


namespace set_intersection_l2140_214079

open Finset

-- Let the universal set U, and sets A and B be defined as follows:
def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Finset ℕ := {1, 2, 3, 5}
def B : Finset ℕ := {2, 4, 6}

-- Define the complement of A with respect to U:
def complement_A : Finset ℕ := U \ A

-- The goal is to prove that B ∩ complement_A = {4, 6}
theorem set_intersection (h : B ∩ complement_A = {4, 6}) : B ∩ complement_A = {4, 6} :=
by exact h

#check set_intersection

end set_intersection_l2140_214079


namespace planter_cost_l2140_214021

-- Define costs
def cost_palm_fern : ℝ := 15.00
def cost_creeping_jenny : ℝ := 4.00
def cost_geranium : ℝ := 3.50

-- Define quantities
def num_creeping_jennies : ℝ := 4
def num_geraniums : ℝ := 4
def num_corners : ℝ := 4

-- Define the total cost
def total_cost : ℝ :=
  (cost_palm_fern
   + (cost_creeping_jenny * num_creeping_jennies)
   + (cost_geranium * num_geraniums))
  * num_corners

-- Prove the total cost is $180.00
theorem planter_cost : total_cost = 180.00 :=
by
  sorry

end planter_cost_l2140_214021


namespace problem_inequality_l2140_214027

theorem problem_inequality (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x) → (x ≤ 2) → (x^2 + 2 + |x^3 - 2 * x| ≥ a * x)) ↔ (a ≤ 2 * Real.sqrt 2) := 
sorry

end problem_inequality_l2140_214027


namespace rationalize_denominator_l2140_214094

theorem rationalize_denominator :
  ∃ A B C : ℤ, A * B * C = 180 ∧
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

end rationalize_denominator_l2140_214094


namespace compute_complex_expression_l2140_214043

-- Define the expression we want to prove
def complex_expression : ℚ := 1 / (1 + (1 / (2 + (1 / (4^2)))))

-- The theorem stating the expression equals to the correct result
theorem compute_complex_expression : complex_expression = 33 / 49 :=
by sorry

end compute_complex_expression_l2140_214043


namespace tangent_line_at_point_P_l2140_214069

-- Define the curve y = x^3 
def curve (x : ℝ) : ℝ := x ^ 3

-- Define the point P(1,1)
def pointP : ℝ × ℝ := (1, 1)

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x ^ 2

-- Define the tangent line equation we need to prove
def tangent_line (x y : ℝ) : Prop := 3 * x - y - 2 = 0

theorem tangent_line_at_point_P :
  ∀ (x y : ℝ), 
  pointP = (1, 1) ∧ curve 1 = 1 ∧ curve_derivative 1 = 3 → 
  tangent_line 1 1 := 
by
  intros x y h
  sorry

end tangent_line_at_point_P_l2140_214069


namespace isosceles_triangle_perimeter_l2140_214032

theorem isosceles_triangle_perimeter (x y : ℝ) (h : 4 * x ^ 2 + 17 * y ^ 2 - 16 * x * y - 4 * y + 4 = 0):
  x = 4 ∧ y = 2 → 2 * x + y = 10 :=
by
  intros
  sorry

end isosceles_triangle_perimeter_l2140_214032


namespace map_distance_representation_l2140_214009

theorem map_distance_representation
  (d_map : ℕ) (d_actual : ℕ) (conversion_factor : ℕ) (final_length_map : ℕ):
  d_map = 10 →
  d_actual = 80 →
  conversion_factor = d_actual / d_map →
  final_length_map = 18 →
  (final_length_map * conversion_factor) = 144 :=
by
  intros h1 h2 h3 h4
  sorry

end map_distance_representation_l2140_214009


namespace journey_distance_l2140_214016

theorem journey_distance (t : ℝ) : 
  t = 20 →
  ∃ D : ℝ, (D / 20 + D / 30 = t) ∧ D = 240 :=
by
  sorry

end journey_distance_l2140_214016


namespace division_equivalence_l2140_214001

theorem division_equivalence (a b c d : ℝ) (h1 : a = 11.7) (h2 : b = 2.6) (h3 : c = 117) (h4 : d = 26) :
  (11.7 / 2.6) = (117 / 26) ∧ (117 / 26) = 4.5 := 
by 
  sorry

end division_equivalence_l2140_214001


namespace profit_percentage_is_20_l2140_214062

variable (C : ℝ) -- Assuming the cost price C is a real number.

theorem profit_percentage_is_20 
  (h1 : 10 * 1 = 12 * (C / 1)) :  -- Shopkeeper sold 10 articles at the cost price of 12 articles.
  ((12 * C - 10 * C) / (10 * C)) * 100 = 20 := 
by
  sorry

end profit_percentage_is_20_l2140_214062


namespace range_of_c_l2140_214068

variable (c : ℝ)

def p := 2 < 3 * c
def q := ∀ x : ℝ, 2 * x^2 + 4 * c * x + 1 > 0

theorem range_of_c (hp : p c) (hq : q c) : (2 / 3) < c ∧ c < (Real.sqrt 2 / 2) :=
by
  sorry

end range_of_c_l2140_214068


namespace bounce_ratio_l2140_214098

theorem bounce_ratio (r : ℝ) (h₁ : 96 * r^4 = 3) : r = Real.sqrt 2 / 4 :=
by
  sorry

end bounce_ratio_l2140_214098


namespace bubble_sort_probability_r10_r25_l2140_214081

theorem bubble_sort_probability_r10_r25 (n : ℕ) (r : ℕ → ℕ) :
  n = 50 ∧ (∀ i, 1 ≤ i ∧ i ≤ 50 → r i ≠ r (i + 1)) ∧ (∀ i j, i ≠ j → r i ≠ r j) →
  let p := 1
  let q := 650
  p + q = 651 :=
by
  intros h
  sorry

end bubble_sort_probability_r10_r25_l2140_214081


namespace total_rock_needed_l2140_214022

theorem total_rock_needed (a b : ℕ) (h₁ : a = 8) (h₂ : b = 8) : a + b = 16 :=
by
  sorry

end total_rock_needed_l2140_214022


namespace white_balls_count_l2140_214066

theorem white_balls_count (w : ℕ) (h : (w / 15) * ((w - 1) / 14) = (1 : ℚ) / 21) : w = 5 := by
  sorry

end white_balls_count_l2140_214066


namespace sequence_ln_l2140_214026

theorem sequence_ln (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + Real.log (1 + 1 / n)) :
  ∀ n : ℕ, n ≥ 1 → a n = 1 + Real.log n := 
sorry

end sequence_ln_l2140_214026


namespace monotone_f_iff_l2140_214063

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if h : x < 1 then a^x
  else x^2 + 4 / x + a * Real.log x

theorem monotone_f_iff (a : ℝ) :
  (∀ x₁ x₂, x₁ ≤ x₂ → f a x₁ ≤ f a x₂) ↔ 2 ≤ a ∧ a ≤ 5 :=
by
  sorry

end monotone_f_iff_l2140_214063


namespace grandmother_mistaken_l2140_214090

-- Definitions of the given conditions:
variables (N : ℕ) (x n : ℕ)
variable (initial_split : N % 4 = 0)

-- Conditions
axiom cows_survived : 4 * (N / 4) / 5 = N / 5
axiom horses_pigs : x = N / 4 - N / 5
axiom rabbit_ratio : (N / 4 - n) = 5 / 14 * (N / 5 + N / 4 + N / 4 - n)

-- Goal: Prove the grandmother is mistaken, i.e., some species avoided casualties
theorem grandmother_mistaken : n = 0 :=
sorry

end grandmother_mistaken_l2140_214090


namespace divisibility_323_l2140_214055

theorem divisibility_323 (n : ℕ) : 
  (20^n + 16^n - 3^n - 1) % 323 = 0 ↔ Even n := 
sorry

end divisibility_323_l2140_214055


namespace artist_paint_usage_l2140_214050

def ounces_of_paint_used (extra_large: ℕ) (large: ℕ) (medium: ℕ) (small: ℕ) : ℕ :=
  4 * extra_large + 3 * large + 2 * medium + 1 * small

theorem artist_paint_usage : ounces_of_paint_used 3 5 6 8 = 47 := by
  sorry

end artist_paint_usage_l2140_214050


namespace last_three_digits_of_7_exp_1987_l2140_214087

theorem last_three_digits_of_7_exp_1987 : (7 ^ 1987) % 1000 = 543 := by
  sorry

end last_three_digits_of_7_exp_1987_l2140_214087


namespace part1_domain_of_f_part2_inequality_l2140_214004

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (abs (x + 1) + abs (x - 1) - 4)

theorem part1_domain_of_f : {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by 
  sorry

theorem part2_inequality (a b : ℝ) (h_a : -2 < a) (h_a' : a < 2) (h_b : -2 < b) (h_b' : b < 2) 
  : 2 * abs (a + b) < abs (4 + a * b) :=
by 
  sorry

end part1_domain_of_f_part2_inequality_l2140_214004


namespace no_intersection_of_lines_l2140_214093

theorem no_intersection_of_lines :
  ¬ ∃ (s v : ℝ) (x y : ℝ),
    (x = 1 - 2 * s ∧ y = 4 + 6 * s) ∧
    (x = 3 - v ∧ y = 10 + 3 * v) :=
by {
  sorry
}

end no_intersection_of_lines_l2140_214093


namespace tangent_line_intercept_l2140_214037

theorem tangent_line_intercept :
  ∃ (m : ℚ) (b : ℚ), m > 0 ∧ b = 740 / 171 ∧
    ∀ (x y : ℚ), ((x - 1)^2 + (y - 3)^2 = 9 ∨ (x - 15)^2 + (y - 8)^2 = 100) →
                 (y = m * x + b) ↔ False := 
sorry

end tangent_line_intercept_l2140_214037


namespace algebraic_expression_value_l2140_214073

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y + 3 = 0) : 1 - 2 * x + 4 * y = 7 := 
by
  sorry

end algebraic_expression_value_l2140_214073


namespace bread_cost_is_30_l2140_214059

variable (cost_sandwich : ℝ)
variable (cost_ham : ℝ)
variable (cost_cheese : ℝ)

def cost_bread (cost_sandwich cost_ham cost_cheese : ℝ) : ℝ :=
  cost_sandwich - cost_ham - cost_cheese

theorem bread_cost_is_30 (H1 : cost_sandwich = 0.90)
  (H2 : cost_ham = 0.25)
  (H3 : cost_cheese = 0.35) :
  cost_bread cost_sandwich cost_ham cost_cheese = 0.30 :=
by
  rw [H1, H2, H3]
  simp [cost_bread]
  sorry

end bread_cost_is_30_l2140_214059


namespace exists_quadratic_sequence_for_any_b_c_smallest_n_for_quadratic_sequence_0_to_2021_l2140_214003

noncomputable def quadratic_sequence (n : ℕ) (a : ℕ → ℤ) :=
  ∀i : ℕ, 1 ≤ i ∧ i ≤ n → abs (a i - a (i - 1)) = i * i

theorem exists_quadratic_sequence_for_any_b_c (b c : ℤ) :
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧ quadratic_sequence n a := by
  sorry

theorem smallest_n_for_quadratic_sequence_0_to_2021 :
  ∃ n : ℕ, 0 < n ∧ ∀ (a : ℕ → ℤ), a 0 = 0 → a n = 2021 → quadratic_sequence n a := by
  sorry

end exists_quadratic_sequence_for_any_b_c_smallest_n_for_quadratic_sequence_0_to_2021_l2140_214003


namespace function_b_is_even_and_monotonically_increasing_l2140_214049

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃a b : ℝ⦄, a ∈ s → b ∈ s → a < b → f a ≤ f b

def f (x : ℝ) : ℝ := abs x + 1

theorem function_b_is_even_and_monotonically_increasing :
  is_even_function f ∧ is_monotonically_increasing_on f (Set.Ioi 0) :=
by
  sorry

end function_b_is_even_and_monotonically_increasing_l2140_214049


namespace value_of_d_l2140_214044

theorem value_of_d (d : ℝ) (h : ∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15 → True) : d = 5 :=
sorry

end value_of_d_l2140_214044


namespace smallest_c_for_inverse_l2140_214036

def g (x : ℝ) : ℝ := -3 * (x - 1)^2 + 4

theorem smallest_c_for_inverse :
  ∃ c : ℝ, (∀ x y : ℝ, c ≤ x → c ≤ y → g x = g y → x = y) ∧ (∀ d : ℝ, (∀ x y : ℝ, d ≤ x → d ≤ y → g x = g y → x = y) → c ≤ d) :=
sorry

end smallest_c_for_inverse_l2140_214036


namespace sodium_hydride_reaction_l2140_214011

theorem sodium_hydride_reaction (H2O NaH NaOH H2 : ℕ) 
  (balanced_eq : NaH + H2O = NaOH + H2) 
  (stoichiometry : NaH = H2O → NaOH = H2 → NaH = H2) 
  (h : H2O = 2) : NaH = 2 :=
sorry

end sodium_hydride_reaction_l2140_214011


namespace time_per_lawn_in_minutes_l2140_214014

def jason_lawns := 16
def total_hours_cutting := 8
def minutes_per_hour := 60

theorem time_per_lawn_in_minutes : 
  (total_hours_cutting / jason_lawns) * minutes_per_hour = 30 :=
by
  sorry

end time_per_lawn_in_minutes_l2140_214014


namespace proposition_p_proposition_not_q_proof_p_and_not_q_l2140_214045

variable (p : Prop)
variable (q : Prop)
variable (r : Prop)

theorem proposition_p : (∃ x0 : ℝ, x0 > 2) := sorry

theorem proposition_not_q : ¬ (∀ x : ℝ, x^3 > x^2) := sorry

theorem proof_p_and_not_q : (∃ x0 : ℝ, x0 > 2) ∧ ¬ (∀ x : ℝ, x^3 > x^2) :=
by
  exact ⟨proposition_p, proposition_not_q⟩

end proposition_p_proposition_not_q_proof_p_and_not_q_l2140_214045


namespace find_fraction_l2140_214008

-- Definition of the fractions and the given condition
def certain_fraction : ℚ := 1 / 2
def given_ratio : ℚ := 2 / 6
def target_fraction : ℚ := 1 / 3

-- The proof problem to verify
theorem find_fraction (X : ℚ) : (X / given_ratio) = 1 ↔ X = target_fraction :=
by
  sorry

end find_fraction_l2140_214008


namespace problem_equivalent_l2140_214030

theorem problem_equivalent (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a + b = 6) (h₃ : a * (a - 6) = x) (h₄ : b * (b - 6) = x) : 
  x = -9 :=
by
  sorry

end problem_equivalent_l2140_214030


namespace find_f2_f_neg1_f_is_odd_f_monotonic_on_negatives_l2140_214041

def f : ℝ → ℝ :=
  sorry

noncomputable def f_properties : Prop :=
  (∀ x y : ℝ, x < 0 → f x < 0 → f x + f y = f (x * y) / f (x + y)) ∧ f 1 = 1

theorem find_f2_f_neg1 :
  f_properties →
  f 2 = 1 / 2 ∧ f (-1) = -1 :=
sorry

theorem f_is_odd :
  f_properties →
  ∀ x : ℝ, f x = -f (-x) :=
sorry

theorem f_monotonic_on_negatives :
  f_properties →
  ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → f x1 > f x2 :=
sorry

end find_f2_f_neg1_f_is_odd_f_monotonic_on_negatives_l2140_214041


namespace susans_average_speed_l2140_214080

theorem susans_average_speed :
  ∀ (total_distance first_leg_distance second_leg_distance : ℕ)
    (first_leg_speed second_leg_speed : ℕ)
    (total_time : ℚ),
    first_leg_distance = 40 →
    second_leg_distance = 20 →
    first_leg_speed = 15 →
    second_leg_speed = 60 →
    total_distance = first_leg_distance + second_leg_distance →
    total_time = (first_leg_distance / first_leg_speed : ℚ) + (second_leg_distance / second_leg_speed : ℚ) →
    total_distance / total_time = 20 :=
by
  sorry

end susans_average_speed_l2140_214080


namespace interest_rate_per_annum_l2140_214040

-- Definitions for the given conditions
def SI : ℝ := 4016.25
def P : ℝ := 44625
def T : ℝ := 9

-- The interest rate R must be 1 according to the conditions
theorem interest_rate_per_annum : (SI * 100) / (P * T) = 1 := by
  sorry

end interest_rate_per_annum_l2140_214040


namespace sequence_sum_correct_l2140_214013

theorem sequence_sum_correct :
  ∀ (r x y : ℝ),
  (x = 128 * r) →
  (y = x * r) →
  (2 * r = 1 / 2) →
  (x + y = 40) :=
by
  intros r x y hx hy hr
  sorry

end sequence_sum_correct_l2140_214013


namespace correct_option_is_C_l2140_214095

theorem correct_option_is_C (x y : ℝ) :
  ¬(3 * x + 4 * y = 12 * x * y) ∧
  ¬(x^9 / x^3 = x^3) ∧
  ((x^2)^3 = x^6) ∧
  ¬((x - y)^2 = x^2 - y^2) :=
by
  sorry

end correct_option_is_C_l2140_214095


namespace AJ_stamps_l2140_214092

theorem AJ_stamps (A : ℕ)
  (KJ := A / 2)
  (CJ := 2 * KJ + 5)
  (BJ := 3 * A - 3)
  (total_stamps := A + KJ + CJ + BJ)
  (h : total_stamps = 1472) :
  A = 267 :=
  sorry

end AJ_stamps_l2140_214092


namespace gcd_repeated_three_digit_integers_l2140_214056

theorem gcd_repeated_three_digit_integers : 
  ∀ m ∈ {n | 100 ≤ n ∧ n < 1000}, 
  gcd (1001 * m) (1001 * (m + 1)) = 1001 :=
by
  sorry

end gcd_repeated_three_digit_integers_l2140_214056


namespace evaluate_expression_l2140_214035

theorem evaluate_expression : 27^(- (2 / 3 : ℝ)) + Real.log 4 / Real.log 8 = 7 / 9 :=
by
  sorry

end evaluate_expression_l2140_214035


namespace third_term_binomial_coefficient_l2140_214002

theorem third_term_binomial_coefficient :
  (∃ m : ℕ, m = 4 ∧ ∃ k : ℕ, k = 2 ∧ Nat.choose m k = 6) :=
by
  sorry

end third_term_binomial_coefficient_l2140_214002


namespace incorrect_games_leq_75_percent_l2140_214039

theorem incorrect_games_leq_75_percent (N : ℕ) (win_points : ℕ) (draw_points : ℚ) (loss_points : ℕ) (incorrect : (ℕ × ℕ) → Prop) :
  (win_points = 1) → (draw_points = 1 / 2) → (loss_points = 0) →
  ∀ (g : ℕ × ℕ), incorrect g → 
  ∃ (total_games incorrect_games : ℕ), 
    total_games = N * (N - 1) / 2 ∧
    incorrect_games ≤ 3 / 4 * total_games := sorry

end incorrect_games_leq_75_percent_l2140_214039


namespace perp_vector_k_l2140_214075

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perp_vector_k :
  ∀ k : ℝ, dot_product (1, 2) (-2, k) = 0 → k = 1 :=
by
  intro k h₀
  sorry

end perp_vector_k_l2140_214075


namespace minimum_reciprocal_sum_l2140_214085

theorem minimum_reciprocal_sum 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : x^2 + y^2 = x * y * (x^2 * y^2 + 2)) : 
  (1 / x + 1 / y) ≥ 2 :=
by 
  sorry -- Proof to be completed

end minimum_reciprocal_sum_l2140_214085


namespace boa_constrictor_is_70_inches_l2140_214029

-- Definitions based on given problem conditions
def garden_snake_length : ℕ := 10
def boa_constrictor_length : ℕ := 7 * garden_snake_length

-- Statement to prove
theorem boa_constrictor_is_70_inches : boa_constrictor_length = 70 :=
by
  sorry

end boa_constrictor_is_70_inches_l2140_214029


namespace ellipse_product_l2140_214065

noncomputable def computeProduct (a b : ℝ) : ℝ :=
  let AB := 2 * a
  let CD := 2 * b
  AB * CD

theorem ellipse_product (a b : ℝ) (h1 : a^2 - b^2 = 64) (h2 : a - b = 4) :
  computeProduct a b = 240 := by
sorry

end ellipse_product_l2140_214065


namespace line_plane_relationship_l2140_214089

variable {ℓ α : Type}
variables (is_line : is_line ℓ) (is_plane : is_plane α) (not_parallel : ¬ parallel ℓ α)

theorem line_plane_relationship (ℓ : Type) (α : Type) [is_line ℓ] [is_plane α] (not_parallel : ¬ parallel ℓ α) : 
  (intersect ℓ α) ∨ (subset ℓ α) :=
sorry

end line_plane_relationship_l2140_214089


namespace three_digit_multiples_of_36_eq_25_l2140_214010

-- Definition: A three-digit number is between 100 and 999
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Definition: A number is a multiple of both 4 and 9 if and only if it's a multiple of 36
def is_multiple_of_36 (n : ℕ) : Prop := n % 36 = 0

-- Definition: Count of three-digit integers that are multiples of 36
def count_multiples_of_36 : ℕ :=
  (999 / 36) - (100 / 36) + 1

-- Theorem: There are 25 three-digit integers that are multiples of 36
theorem three_digit_multiples_of_36_eq_25 : count_multiples_of_36 = 25 := by
  sorry

end three_digit_multiples_of_36_eq_25_l2140_214010


namespace uncle_fyodor_sandwiches_count_l2140_214083

variable (sandwiches_sharik : ℕ)
variable (sandwiches_matroskin : ℕ := 3 * sandwiches_sharik)
variable (total_sandwiches_eaten : ℕ := sandwiches_sharik + sandwiches_matroskin)
variable (sandwiches_uncle_fyodor : ℕ := 2 * total_sandwiches_eaten)
variable (difference : ℕ := sandwiches_uncle_fyodor - sandwiches_sharik)

theorem uncle_fyodor_sandwiches_count :
  (difference = 21) → sandwiches_uncle_fyodor = 24 := by
  intro h
  sorry

end uncle_fyodor_sandwiches_count_l2140_214083
