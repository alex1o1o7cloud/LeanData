import Mathlib

namespace NUMINAMATH_GPT_temperature_reaches_90_at_17_l810_81069

def temperature (t : ℝ) : ℝ := -t^2 + 14 * t + 40

theorem temperature_reaches_90_at_17 :
  ∃ t : ℝ, temperature t = 90 ∧ t = 17 :=
by
  exists 17
  dsimp [temperature]
  norm_num
  sorry

end NUMINAMATH_GPT_temperature_reaches_90_at_17_l810_81069


namespace NUMINAMATH_GPT_least_number_to_add_l810_81025

theorem least_number_to_add (a b : ℤ) (d : ℤ) (h : a = 1054) (hb : b = 47) (hd : d = 27) :
  ∃ n : ℤ, (a + d) % b = 0 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_add_l810_81025


namespace NUMINAMATH_GPT_units_digit_same_units_and_tens_digit_same_l810_81064

theorem units_digit_same (n : ℕ) : 
  (∃ a : ℕ, a ∈ [0, 1, 5, 6] ∧ n % 10 = a ∧ n^2 % 10 = a) := 
sorry

theorem units_and_tens_digit_same (n : ℕ) : 
  n ∈ [0, 1, 25, 76] ↔ (n % 100 = n^2 % 100) := 
sorry

end NUMINAMATH_GPT_units_digit_same_units_and_tens_digit_same_l810_81064


namespace NUMINAMATH_GPT_tracy_initial_candies_l810_81022

variable (x y : ℕ) (h1 : 2 ≤ y) (h2 : y ≤ 6)

theorem tracy_initial_candies :
  (x - (1/5 : ℚ) * x = (4/5 : ℚ) * x) ∧
  ((4/5 : ℚ) * x - (1/3 : ℚ) * (4/5 : ℚ) * x = (8/15 : ℚ) * x) ∧
  y - 10 * 2 + ((8/15 : ℚ) * x - 20) = 5 →
  x = 60 :=
by
  sorry

end NUMINAMATH_GPT_tracy_initial_candies_l810_81022


namespace NUMINAMATH_GPT_slope_range_l810_81033

theorem slope_range (α : Real) (hα : -1 ≤ Real.cos α ∧ Real.cos α ≤ 1) :
  ∃ k ∈ Set.Icc (- Real.sqrt 3 / 3) (Real.sqrt 3 / 3), ∀ x y : Real, x * Real.cos α - Real.sqrt 3 * y - 2 = 0 → y = k * x - (2 / Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_slope_range_l810_81033


namespace NUMINAMATH_GPT_find_values_of_p_l810_81016

def geometric_progression (p : ℝ) : Prop :=
  (2 * p)^2 = (4 * p + 5) * |p - 3|

theorem find_values_of_p :
  {p : ℝ | geometric_progression p} = {-1, 15 / 8} :=
by
  sorry

end NUMINAMATH_GPT_find_values_of_p_l810_81016


namespace NUMINAMATH_GPT_compute_fraction_power_l810_81026

theorem compute_fraction_power : 9 * (1 / 7)^4 = 9 / 2401 := by
  sorry

end NUMINAMATH_GPT_compute_fraction_power_l810_81026


namespace NUMINAMATH_GPT_value_of_x_when_y_is_six_l810_81071

theorem value_of_x_when_y_is_six 
  (k : ℝ) -- The constant of variation
  (h1 : ∀ y : ℝ, x = k / y^2) -- The inverse relationship
  (h2 : y = 2)
  (h3 : x = 1)
  : x = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_when_y_is_six_l810_81071


namespace NUMINAMATH_GPT_find_overhead_expenses_l810_81000

noncomputable def overhead_expenses : ℝ := 35.29411764705882 / (1 + 0.1764705882352942)

theorem find_overhead_expenses (cost_price selling_price profit_percent : ℝ) (h_cp : cost_price = 225) (h_sp : selling_price = 300) (h_pp : profit_percent = 0.1764705882352942) :
  overhead_expenses = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_overhead_expenses_l810_81000


namespace NUMINAMATH_GPT_factorization_correct_l810_81057

noncomputable def factorize_poly (m n : ℕ) : ℕ := 2 * m * n ^ 2 - 12 * m * n + 18 * m

theorem factorization_correct (m n : ℕ) :
  factorize_poly m n = 2 * m * (n - 3) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l810_81057


namespace NUMINAMATH_GPT_monkey_reaches_tree_top_in_hours_l810_81092

-- Definitions based on conditions
def height_of_tree : ℕ := 22
def hop_per_hour : ℕ := 3
def slip_per_hour : ℕ := 2
def effective_climb_per_hour : ℕ := hop_per_hour - slip_per_hour

-- The theorem we want to prove
theorem monkey_reaches_tree_top_in_hours
  (height_of_tree hop_per_hour slip_per_hour : ℕ)
  (h1 : height_of_tree = 22)
  (h2 : hop_per_hour = 3)
  (h3 : slip_per_hour = 2) :
  ∃ t : ℕ, t = 22 ∧ effective_climb_per_hour * (t - 1) + hop_per_hour = height_of_tree := by
  sorry

end NUMINAMATH_GPT_monkey_reaches_tree_top_in_hours_l810_81092


namespace NUMINAMATH_GPT_farmer_plough_rate_l810_81080

-- Define the problem statement and the required proof 

theorem farmer_plough_rate :
  ∀ (x y : ℕ),
  90 * x = 3780 ∧ y * (x + 2) = 3740 → y = 85 :=
by
  sorry

end NUMINAMATH_GPT_farmer_plough_rate_l810_81080


namespace NUMINAMATH_GPT_remainder_of_3_pow_800_mod_17_l810_81038

theorem remainder_of_3_pow_800_mod_17 :
    (3 ^ 800) % 17 = 1 :=
by
    sorry

end NUMINAMATH_GPT_remainder_of_3_pow_800_mod_17_l810_81038


namespace NUMINAMATH_GPT_triangle_perimeter_l810_81020

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def perimeter (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  distance p1 p2 + distance p1 p3 + distance p2 p3

theorem triangle_perimeter :
  let p1 := (1, 4)
  let p2 := (-7, 0)
  let p3 := (1, 0)
  perimeter p1 p2 p3 = 4 * Real.sqrt 5 + 12 :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l810_81020


namespace NUMINAMATH_GPT_tank_filling_time_l810_81059

noncomputable def fill_time (R1 R2 R3 : ℚ) : ℚ :=
  1 / (R1 + R2 + R3)

theorem tank_filling_time :
  let R1 := 1 / 18
  let R2 := 1 / 30
  let R3 := -1 / 45
  fill_time R1 R2 R3 = 15 :=
by
  intros
  unfold fill_time
  sorry

end NUMINAMATH_GPT_tank_filling_time_l810_81059


namespace NUMINAMATH_GPT_part1_decreasing_on_pos_part2_t_range_l810_81009

noncomputable def f (x : ℝ) : ℝ := -x + 2 / x

theorem part1_decreasing_on_pos (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < x2) : 
  f x1 > f x2 := by sorry

theorem part2_t_range (t : ℝ) (ht : ∀ x : ℝ, 1 ≤ x → f x ≤ (1 + t * x) / x) : 
  0 ≤ t := by sorry

end NUMINAMATH_GPT_part1_decreasing_on_pos_part2_t_range_l810_81009


namespace NUMINAMATH_GPT_sum_a_n_eq_2014_l810_81072

def f (n : ℕ) : ℤ :=
  if n % 2 = 1 then (n : ℤ)^2 else - (n : ℤ)^2

def a (n : ℕ) : ℤ :=
  f n + f (n + 1)

theorem sum_a_n_eq_2014 : (Finset.range 2014).sum a = 2014 :=
by
  sorry

end NUMINAMATH_GPT_sum_a_n_eq_2014_l810_81072


namespace NUMINAMATH_GPT_class_size_l810_81058

theorem class_size :
  ∃ (N : ℕ), (20 ≤ N) ∧ (N ≤ 30) ∧ (∃ (x : ℕ), N = 3 * x + 1) ∧ (∃ (y : ℕ), N = 4 * y + 1) ∧ (N = 25) :=
by { sorry }

end NUMINAMATH_GPT_class_size_l810_81058


namespace NUMINAMATH_GPT_credit_extended_by_automobile_finance_companies_l810_81077

def percentage_of_automobile_installment_credit : ℝ := 0.36
def total_consumer_installment_credit : ℝ := 416.66667
def fraction_extended_by_finance_companies : ℝ := 0.5

theorem credit_extended_by_automobile_finance_companies :
  fraction_extended_by_finance_companies * (percentage_of_automobile_installment_credit * total_consumer_installment_credit) = 75 :=
by
  sorry

end NUMINAMATH_GPT_credit_extended_by_automobile_finance_companies_l810_81077


namespace NUMINAMATH_GPT_ant_moves_probability_l810_81034

theorem ant_moves_probability :
  let m := 73
  let n := 48
  m + n = 121 := by
  sorry

end NUMINAMATH_GPT_ant_moves_probability_l810_81034


namespace NUMINAMATH_GPT_chord_line_equation_l810_81099

theorem chord_line_equation (x y : ℝ) 
  (ellipse : ∀ (x y : ℝ), x^2 / 36 + y^2 / 9 = 1)
  (bisect_point : x / 2 = 4 ∧ y / 2 = 2) : 
  x + 2 * y - 8 = 0 :=
sorry

end NUMINAMATH_GPT_chord_line_equation_l810_81099


namespace NUMINAMATH_GPT_problem_1_problem_2_l810_81045

open Real

noncomputable def f (x : ℝ) : ℝ := 3^x
noncomputable def g (x : ℝ) : ℝ := log x / log 3

theorem problem_1 : g 4 + g 8 - g (32 / 9) = 2 := 
by
  sorry

theorem problem_2 (x : ℝ) (h : 0 < x ∧ x < 1) : g (x / (1 - x)) < 1 ↔ 0 < x ∧ x < 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l810_81045


namespace NUMINAMATH_GPT_m_minus_n_eq_2_l810_81018

theorem m_minus_n_eq_2 (m n : ℕ) (h1 : ∃ x : ℕ, m = 101 * x) (h2 : ∃ y : ℕ, n = 63 * y) (h3 : m + n = 2018) : m - n = 2 :=
sorry

end NUMINAMATH_GPT_m_minus_n_eq_2_l810_81018


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l810_81013

theorem necessary_and_sufficient_condition (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  (m + n > m * n) ↔ (m = 1 ∨ n = 1) := by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l810_81013


namespace NUMINAMATH_GPT_length_of_rod_l810_81081

theorem length_of_rod (w1 w2 l1 l2 : ℝ) (h_uniform : ∀ m n, m * w1 = n * w2) (h1 : w1 = 42.75) (h2 : l1 = 11.25) : 
  l2 = 6 := 
  by
  have wpm := w1 / l1
  have h3 : 22.8 / wpm = l2 := by sorry
  rw [h1, h2] at *
  simp at *
  sorry

end NUMINAMATH_GPT_length_of_rod_l810_81081


namespace NUMINAMATH_GPT_smallest_area_is_10_l810_81010

noncomputable def smallest_square_area : ℝ :=
  let k₁ := 65
  let k₂ := -5
  10 * (9 + 4 * k₂)

theorem smallest_area_is_10 :
  smallest_square_area = 10 := by
  sorry

end NUMINAMATH_GPT_smallest_area_is_10_l810_81010


namespace NUMINAMATH_GPT_initial_pencils_correct_l810_81093

variable (initial_pencils : ℕ)
variable (pencils_added : ℕ := 45)
variable (total_pencils : ℕ := 72)

theorem initial_pencils_correct (h : total_pencils = initial_pencils + pencils_added) : initial_pencils = 27 := by
  sorry

end NUMINAMATH_GPT_initial_pencils_correct_l810_81093


namespace NUMINAMATH_GPT_swimmer_path_min_time_l810_81060

theorem swimmer_path_min_time (k : ℝ) :
  (k > Real.sqrt 2 → ∀ x y : ℝ, x = 0 ∧ y = 0 ∧ t = 2/k) ∧
  (k < Real.sqrt 2 → x = 1 ∧ y = 1 ∧ t = Real.sqrt 2) ∧
  (k = Real.sqrt 2 → ∀ x y : ℝ, x = y ∧ t = (1 / Real.sqrt 2) + Real.sqrt 2 + (1 / Real.sqrt 2)) :=
by sorry

end NUMINAMATH_GPT_swimmer_path_min_time_l810_81060


namespace NUMINAMATH_GPT_problem_statement_l810_81084

def atOp (a b : ℝ) := a * b ^ (1 / 2)

theorem problem_statement : atOp ((2 * 3) ^ 2) ((3 * 5) ^ 2 / 9) = 180 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l810_81084


namespace NUMINAMATH_GPT_ferris_wheel_seat_capacity_l810_81015

-- Define the given conditions
def people := 16
def seats := 4

-- Define the problem and the proof goal
theorem ferris_wheel_seat_capacity : people / seats = 4 := by
  sorry

end NUMINAMATH_GPT_ferris_wheel_seat_capacity_l810_81015


namespace NUMINAMATH_GPT_cooks_in_restaurant_l810_81079

theorem cooks_in_restaurant
  (C W : ℕ) 
  (h1 : C * 8 = 3 * W) 
  (h2 : C * 4 = (W + 12)) :
  C = 9 :=
by
  sorry

end NUMINAMATH_GPT_cooks_in_restaurant_l810_81079


namespace NUMINAMATH_GPT_polygon_sides_and_diagonals_l810_81094

theorem polygon_sides_and_diagonals (n : ℕ) :
  (180 * (n - 2) = 3 * 360 + 180) → n = 9 ∧ (n - 3 = 6) :=
by
  intro h_sum_angles
  -- This is where you would provide the proof.
  sorry

end NUMINAMATH_GPT_polygon_sides_and_diagonals_l810_81094


namespace NUMINAMATH_GPT_positive_solution_for_y_l810_81048

theorem positive_solution_for_y (x y z : ℝ) 
  (h1 : x * y = 4 - x - 2 * y)
  (h2 : y * z = 8 - 3 * y - 2 * z)
  (h3 : x * z = 40 - 5 * x - 2 * z) : y = 2 := 
sorry

end NUMINAMATH_GPT_positive_solution_for_y_l810_81048


namespace NUMINAMATH_GPT_negation_exists_l810_81087

theorem negation_exists :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ x) ↔ ∃ x : ℝ, x^2 + 1 < x :=
sorry

end NUMINAMATH_GPT_negation_exists_l810_81087


namespace NUMINAMATH_GPT_m_coins_can_collect_k_rubles_l810_81066

theorem m_coins_can_collect_k_rubles
  (a1 a2 a3 a4 a5 a6 a7 m k : ℕ)
  (h1 : a1 + 2 * a2 + 5 * a3 + 10 * a4 + 20 * a5 + 50 * a6 + 100 * a7 = m)
  (h2 : a1 + a2 + a3 + a4 + a5 + a6 + a7 = k) :
  ∃ (b1 b2 b3 b4 b5 b6 b7 : ℕ), 
    100 * (b1 + 2 * b2 + 5 * b3 + 10 * b4 + 20 * b5 + 50 * b6 + 100 * b7) = 100 * k ∧ 
    b1 + b2 + b3 + b4 + b5 + b6 + b7 = m := 
sorry

end NUMINAMATH_GPT_m_coins_can_collect_k_rubles_l810_81066


namespace NUMINAMATH_GPT_arithmetic_seq_geom_seq_l810_81037

theorem arithmetic_seq_geom_seq {a : ℕ → ℝ} 
  (h1 : ∀ n, 0 < a n)
  (h2 : a 2 + a 3 + a 4 = 15)
  (h3 : (a 1 + 2) * (a 6 + 16) = (a 3 + 4) ^ 2) :
  a 10 = 19 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_geom_seq_l810_81037


namespace NUMINAMATH_GPT_range_of_m_l810_81031

theorem range_of_m 
  (m : ℝ)
  (f : ℝ → ℝ)
  (f_def : ∀ x, f x = x^3 + (m / 2 + 2) * x^2 - 2 * x)
  (f_prime : ℝ → ℝ)
  (f_prime_def : ∀ x, f_prime x = 3 * x^2 + (m + 4) * x - 2)
  (f_prime_at_1 : f_prime 1 < 0)
  (f_prime_at_2 : f_prime 2 < 0)
  (f_prime_at_3 : f_prime 3 > 0) :
  -37 / 3 < m ∧ m < -9 := 
  sorry

end NUMINAMATH_GPT_range_of_m_l810_81031


namespace NUMINAMATH_GPT_value_of_M_l810_81055

theorem value_of_M :
  let row_seq := [25, 25 + (8 - 25) / 3, 25 + 2 * (8 - 25) / 3, 8, 8 + (8 - 25) / 3, 8 + 2 * (8 - 25) / 3, -9]
  let col_seq1 := [25, 25 - 4, 25 - 8]
  let col_seq2 := [16, 20, 20 + 4]
  let col_seq3 := [-9, -9 - 11/4, -9 - 2 * 11/4, -20]
  let M := -9 - (-11/4)
  M = -6.25 :=
by
  sorry

end NUMINAMATH_GPT_value_of_M_l810_81055


namespace NUMINAMATH_GPT_cos_alpha_value_l810_81039

open Real

theorem cos_alpha_value (α : ℝ) (h1 : sin (α + π / 3) = 3 / 5) (h2 : π / 6 < α ∧ α < 5 * π / 6) :
  cos α = (3 * sqrt 3 - 4) / 10 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_value_l810_81039


namespace NUMINAMATH_GPT_triangle_inequality_sides_l810_81004

theorem triangle_inequality_sides
  (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  (a + b) * Real.sqrt (a * b) + (a + c) * Real.sqrt (a * c) + (b + c) * Real.sqrt (b * c) ≥ (a + b + c)^2 / 2 := 
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_sides_l810_81004


namespace NUMINAMATH_GPT_car_travel_time_l810_81050

-- Definitions
def speed : ℝ := 50
def miles_per_gallon : ℝ := 30
def tank_capacity : ℝ := 15
def fraction_used : ℝ := 0.5555555555555556

-- Theorem statement
theorem car_travel_time : (fraction_used * tank_capacity * miles_per_gallon / speed) = 5 :=
sorry

end NUMINAMATH_GPT_car_travel_time_l810_81050


namespace NUMINAMATH_GPT_find_same_goldfish_number_l810_81014

noncomputable def B (n : ℕ) : ℕ := 3 * 4^n
noncomputable def G (n : ℕ) : ℕ := 243 * 3^n

theorem find_same_goldfish_number : ∃ n, B n = G n :=
by sorry

end NUMINAMATH_GPT_find_same_goldfish_number_l810_81014


namespace NUMINAMATH_GPT_find_sum_u_v_l810_81097

theorem find_sum_u_v (u v : ℤ) (huv : 0 < v ∧ v < u) (pentagon_area : u^2 + 3 * u * v = 451) : u + v = 21 :=
by 
  sorry

end NUMINAMATH_GPT_find_sum_u_v_l810_81097


namespace NUMINAMATH_GPT_non_neg_integer_solutions_for_inequality_l810_81008

theorem non_neg_integer_solutions_for_inequality :
  {x : ℤ | 5 * x - 1 < 3 * (x + 1) ∧ (1 - x) / 3 ≤ 1 ∧ 0 ≤ x } = {0, 1} := 
by {
  sorry
}

end NUMINAMATH_GPT_non_neg_integer_solutions_for_inequality_l810_81008


namespace NUMINAMATH_GPT_cost_per_minute_l810_81046

theorem cost_per_minute (monthly_fee cost total_bill : ℝ) (minutes : ℕ) :
  monthly_fee = 2 ∧ total_bill = 23.36 ∧ minutes = 178 → 
  cost = (total_bill - monthly_fee) / minutes → 
  cost = 0.12 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_cost_per_minute_l810_81046


namespace NUMINAMATH_GPT_jaden_toy_cars_l810_81095

theorem jaden_toy_cars :
  let initial : Nat := 14
  let bought : Nat := 28
  let birthday : Nat := 12
  let to_sister : Nat := 8
  let to_friend : Nat := 3
  initial + bought + birthday - to_sister - to_friend = 43 :=
by
  let initial : Nat := 14
  let bought : Nat := 28
  let birthday : Nat := 12
  let to_sister : Nat := 8
  let to_friend : Nat := 3
  show initial + bought + birthday - to_sister - to_friend = 43
  sorry

end NUMINAMATH_GPT_jaden_toy_cars_l810_81095


namespace NUMINAMATH_GPT_no_b_satisfies_143b_square_of_integer_l810_81053

theorem no_b_satisfies_143b_square_of_integer :
  ∀ b : ℤ, b > 4 → ¬ ∃ k : ℤ, b^2 + 4 * b + 3 = k^2 :=
by
  intro b hb
  by_contra h
  obtain ⟨k, hk⟩ := h
  have : b^2 + 4 * b + 3 = k ^ 2 := hk
  sorry

end NUMINAMATH_GPT_no_b_satisfies_143b_square_of_integer_l810_81053


namespace NUMINAMATH_GPT_number_of_subsets_l810_81040

-- Defining the type of the elements
variable {α : Type*}

-- Statement of the problem in Lean 4
theorem number_of_subsets (s : Finset α) (h : s.card = n) : (Finset.powerset s).card = 2^n := 
sorry

end NUMINAMATH_GPT_number_of_subsets_l810_81040


namespace NUMINAMATH_GPT_travel_time_l810_81083

-- Given conditions
def distance_per_hour : ℤ := 27
def distance_to_sfl : ℤ := 81

-- Theorem statement to prove
theorem travel_time (dph : ℤ) (dts : ℤ) (h1 : dph = distance_per_hour) (h2 : dts = distance_to_sfl) : 
  dts / dph = 3 := 
by
  -- immediately helps execute the Lean statement
  sorry

end NUMINAMATH_GPT_travel_time_l810_81083


namespace NUMINAMATH_GPT_range_of_x_l810_81043

-- Define the even and increasing properties of the function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

-- The main theorem to be proven
theorem range_of_x (f : ℝ → ℝ) (h_even : is_even f) (h_incr : is_increasing_on_nonneg f) 
  (h_cond : ∀ x : ℝ, f (x - 1) < f (2 - x)) :
  ∀ x : ℝ, x < 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l810_81043


namespace NUMINAMATH_GPT_chocolates_problem_l810_81070

theorem chocolates_problem 
  (N : ℕ)
  (h1 : ∃ R C : ℕ, N = R * C)
  (h2 : ∃ r1, r1 = 3 * C - 1)
  (h3 : ∃ c1, c1 = R * 5 - 1)
  (h4 : ∃ r2 c2 : ℕ, r2 = 3 ∧ c2 = 5 ∧ r2 * c2 = r1 - 1)
  (h5 : N = 3 * (3 * C - 1))
  (h6 : N / 3 = (12 * 5) / 3) : 
  N = 60 ∧ (N - (3 * C - 1)) = 25 :=
by 
  sorry

end NUMINAMATH_GPT_chocolates_problem_l810_81070


namespace NUMINAMATH_GPT_valid_k_l810_81065

theorem valid_k (k : ℕ) (h_pos : k ≥ 1) (h : 10^k - 1 = 9 * k^2) : k = 1 := by
  sorry

end NUMINAMATH_GPT_valid_k_l810_81065


namespace NUMINAMATH_GPT_find_four_digit_numbers_l810_81030

theorem find_four_digit_numbers (a b c d : ℕ) : 
  (1000 ≤ 1000 * a + 100 * b + 10 * c + d) ∧ 
  (1000 * a + 100 * b + 10 * c + d ≤ 9999) ∧ 
  (1000 ≤ 1000 * d + 100 * c + 10 * b + a) ∧ 
  (1000 * d + 100 * c + 10 * b + a ≤ 9999) ∧
  (a + d = 9) ∧ 
  (b + c = 13) ∧
  (1001 * (a + d) + 110 * (b + c) = 19448) → 
  (1000 * a + 100 * b + 10 * c + d = 9949 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9859 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9769 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9679 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9589 ∨ 
   1000 * a + 100 * b + 10 * c + d = 9499) :=
sorry

end NUMINAMATH_GPT_find_four_digit_numbers_l810_81030


namespace NUMINAMATH_GPT_total_seeds_gray_sections_combined_l810_81078

noncomputable def total_seeds_first_circle : ℕ := 87
noncomputable def seeds_white_first_circle : ℕ := 68
noncomputable def total_seeds_second_circle : ℕ := 110
noncomputable def seeds_white_second_circle : ℕ := 68

theorem total_seeds_gray_sections_combined :
  (total_seeds_first_circle - seeds_white_first_circle) +
  (total_seeds_second_circle - seeds_white_second_circle) = 61 :=
by
  sorry

end NUMINAMATH_GPT_total_seeds_gray_sections_combined_l810_81078


namespace NUMINAMATH_GPT_largest_of_seven_consecutive_integers_l810_81047

theorem largest_of_seven_consecutive_integers (a : ℕ) (h : a > 0) (sum_eq_77 : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6) = 77)) :
  a + 6 = 14 :=
by
  sorry

end NUMINAMATH_GPT_largest_of_seven_consecutive_integers_l810_81047


namespace NUMINAMATH_GPT_triangle_side_lengths_l810_81088

open Real

theorem triangle_side_lengths (a b c : ℕ) (R : ℝ)
    (h1 : a * a + 4 * d * d = 2500)
    (h2 : b * b + 4 * e * e = 2500)
    (h3 : R = 12.5)
    (h4 : (2:ℝ) * d ≤ a)
    (h5 : (2:ℝ) * e ≤ b)
    (h6 : a > b)
    (h7 : a ≠ b)
    (h8 : 2 * R = 25) :
    (a, b, c) = (15, 7, 20) := by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_l810_81088


namespace NUMINAMATH_GPT_integral_value_l810_81012

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the conditions of the problem
def a : ℝ := 2 -- This is derived from the problem condition

-- The main theorem statement
theorem integral_value :
  (∫ x in (0 : ℝ)..a, (Real.exp x + 2 * x)) = Real.exp 2 + 3 := by
  sorry

end NUMINAMATH_GPT_integral_value_l810_81012


namespace NUMINAMATH_GPT_four_digit_numbers_with_product_exceeds_10_l810_81052

theorem four_digit_numbers_with_product_exceeds_10 : 
  (∃ count : ℕ, 
    count = (6 * 58 * 10) ∧
    count = 3480) := 
by {
  sorry
}

end NUMINAMATH_GPT_four_digit_numbers_with_product_exceeds_10_l810_81052


namespace NUMINAMATH_GPT_correct_yeast_population_change_statement_l810_81098

def yeast_produces_CO2 (aerobic : Bool) : Bool := 
  True

def yeast_unicellular_fungus : Bool := 
  True

def boiling_glucose_solution_purpose : Bool := 
  True

def yeast_facultative_anaerobe : Bool := 
  True

theorem correct_yeast_population_change_statement : 
  (∀ (aerobic : Bool), yeast_produces_CO2 aerobic) →
  yeast_unicellular_fungus →
  boiling_glucose_solution_purpose →
  yeast_facultative_anaerobe →
  "D is correct" = "D is correct" :=
by
  intros
  exact rfl

end NUMINAMATH_GPT_correct_yeast_population_change_statement_l810_81098


namespace NUMINAMATH_GPT_average_speed_car_l810_81049

theorem average_speed_car (speed_first_hour ground_speed_headwind speed_second_hour : ℝ) (time_first_hour time_second_hour : ℝ) (h1 : speed_first_hour = 90) (h2 : ground_speed_headwind = 10) (h3 : speed_second_hour = 55) (h4 : time_first_hour = 1) (h5 : time_second_hour = 1) : 
(speed_first_hour + ground_speed_headwind) * time_first_hour + speed_second_hour * time_second_hour / (time_first_hour + time_second_hour) = 77.5 :=
sorry

end NUMINAMATH_GPT_average_speed_car_l810_81049


namespace NUMINAMATH_GPT_red_paint_cans_needed_l810_81007

-- Definitions for the problem
def ratio_red_white : ℚ := 3 / 2
def total_cans : ℕ := 30

-- Theorem statement to prove the number of cans of red paint
theorem red_paint_cans_needed : total_cans * (3 / 5) = 18 := by 
  sorry

end NUMINAMATH_GPT_red_paint_cans_needed_l810_81007


namespace NUMINAMATH_GPT_min_cookies_satisfy_conditions_l810_81096

theorem min_cookies_satisfy_conditions : ∃ (b : ℕ), b ≡ 5 [MOD 6] ∧ b ≡ 7 [MOD 8] ∧ b ≡ 8 [MOD 9] ∧ ∀ (b' : ℕ), (b' ≡ 5 [MOD 6] ∧ b' ≡ 7 [MOD 8] ∧ b' ≡ 8 [MOD 9]) → b ≤ b' := 
sorry

end NUMINAMATH_GPT_min_cookies_satisfy_conditions_l810_81096


namespace NUMINAMATH_GPT_competition_participants_l810_81023

theorem competition_participants (N : ℕ)
  (h1 : (1 / 12) * N = 18) :
  N = 216 := 
by
  sorry

end NUMINAMATH_GPT_competition_participants_l810_81023


namespace NUMINAMATH_GPT_chase_travel_time_l810_81056

-- Definitions of speeds
def chase_speed (C : ℝ) := C
def cameron_speed (C : ℝ) := 2 * C
def danielle_speed (C : ℝ) := 6 * (cameron_speed C)

-- Time taken by Danielle to cover distance
def time_taken_by_danielle (C : ℝ) := 30  
def distance_travelled (C : ℝ) := (time_taken_by_danielle C) * (danielle_speed C)  -- 180C

-- Speeds on specific stretches
def cameron_bike_speed (C : ℝ) := 0.75 * (cameron_speed C)
def chase_scooter_speed (C : ℝ) := 1.25 * (chase_speed C)

-- Prove the time Chase takes to travel the same distance D
theorem chase_travel_time (C : ℝ) : 
  (distance_travelled C) / (chase_speed C) = 180 := sorry

end NUMINAMATH_GPT_chase_travel_time_l810_81056


namespace NUMINAMATH_GPT_extreme_values_of_f_range_of_a_for_intersection_l810_81042

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 15 * x + a

theorem extreme_values_of_f :
  f (-1) = 5 ∧ f 3 = -27 :=
by {
  sorry
}

theorem range_of_a_for_intersection (a : ℝ) : 
  (-80 < a) ∧ (a < 28) ↔ ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = g x₁ a ∧ f x₂ = g x₂ a ∧ f x₃ = g x₃ a :=
by {
  sorry
}

end NUMINAMATH_GPT_extreme_values_of_f_range_of_a_for_intersection_l810_81042


namespace NUMINAMATH_GPT_largest_four_digit_number_divisible_by_six_l810_81086

theorem largest_four_digit_number_divisible_by_six : 
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ 
  (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 2 = 0) ∧ (m % 3 = 0) → m ≤ n) ∧ n = 9960 := 
by { sorry }

end NUMINAMATH_GPT_largest_four_digit_number_divisible_by_six_l810_81086


namespace NUMINAMATH_GPT_John_pays_more_than_Jane_l810_81029

theorem John_pays_more_than_Jane : 
  let original_price := 24.00000000000002
  let discount_rate := 0.10
  let tip_rate := 0.15
  let discount := discount_rate * original_price
  let discounted_price := original_price - discount
  let john_tip := tip_rate * original_price
  let jane_tip := tip_rate * discounted_price
  let john_total := discounted_price + john_tip
  let jane_total := discounted_price + jane_tip
  john_total - jane_total = 0.3600000000000003 :=
by
  sorry

end NUMINAMATH_GPT_John_pays_more_than_Jane_l810_81029


namespace NUMINAMATH_GPT_solution_set_x2_f_x_positive_l810_81090

noncomputable def f : ℝ → ℝ := sorry
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_at_2 : f 2 = 0
axiom derivative_condition : ∀ x, x > 0 → ((x * (deriv f x) - f x) / x^2) > 0

theorem solution_set_x2_f_x_positive :
  {x : ℝ | x^2 * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | x > 2} :=
sorry

end NUMINAMATH_GPT_solution_set_x2_f_x_positive_l810_81090


namespace NUMINAMATH_GPT_length_of_first_train_l810_81002

theorem length_of_first_train
    (speed_train1_kmph : ℝ) (speed_train2_kmph : ℝ) 
    (length_train2_m : ℝ) (cross_time_s : ℝ)
    (conv_factor : ℝ)         -- Conversion factor from kmph to m/s
    (relative_speed_ms : ℝ)   -- Relative speed in m/s 
    (distance_covered_m : ℝ)  -- Total distance covered in meters
    (length_train1_m : ℝ) : Prop :=
  speed_train1_kmph = 120 →
  speed_train2_kmph = 80 →
  length_train2_m = 210.04 →
  cross_time_s = 9 →
  conv_factor = 1000 / 3600 →
  relative_speed_ms = (200 * conv_factor) →
  distance_covered_m = (relative_speed_ms * cross_time_s) →
  length_train1_m = 290 →
  distance_covered_m = length_train1_m + length_train2_m

end NUMINAMATH_GPT_length_of_first_train_l810_81002


namespace NUMINAMATH_GPT_termite_ridden_not_collapsing_l810_81073

theorem termite_ridden_not_collapsing
  (total_homes : ℕ)
  (termite_ridden_fraction : ℚ)
  (collapsing_fraction_of_termite_ridden : ℚ)
  (h1 : termite_ridden_fraction = 1/3)
  (h2 : collapsing_fraction_of_termite_ridden = 1/4) :
  (termite_ridden_fraction - (termite_ridden_fraction * collapsing_fraction_of_termite_ridden)) = 1/4 := 
by {
  sorry
}

end NUMINAMATH_GPT_termite_ridden_not_collapsing_l810_81073


namespace NUMINAMATH_GPT_calc_expression_l810_81054

theorem calc_expression (a : ℝ) : 4 * a * a^3 - a^4 = 3 * a^4 := by
  sorry

end NUMINAMATH_GPT_calc_expression_l810_81054


namespace NUMINAMATH_GPT_fraction_Cal_to_Anthony_l810_81091

-- definitions for Mabel, Anthony, Cal, and Jade's transactions
def Mabel_transactions : ℕ := 90
def Anthony_transactions : ℕ := Mabel_transactions + (Mabel_transactions / 10)
def Jade_transactions : ℕ := 85
def Cal_transactions : ℕ := Jade_transactions - 19

-- goal: prove the fraction Cal handled compared to Anthony is 2/3
theorem fraction_Cal_to_Anthony : (Cal_transactions : ℚ) / (Anthony_transactions : ℚ) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_Cal_to_Anthony_l810_81091


namespace NUMINAMATH_GPT_mul_3_6_0_5_l810_81001

theorem mul_3_6_0_5 : 3.6 * 0.5 = 1.8 :=
by
  sorry

end NUMINAMATH_GPT_mul_3_6_0_5_l810_81001


namespace NUMINAMATH_GPT_polynomial_remainder_l810_81024

theorem polynomial_remainder (c a b : ℤ) 
  (h1 : (16 * c + 8 * a + 2 * b = -12)) 
  (h2 : (81 * c - 27 * a - 3 * b = -85)) : 
  (a, b, c) = (5, 7, 1) :=
sorry

end NUMINAMATH_GPT_polynomial_remainder_l810_81024


namespace NUMINAMATH_GPT_problem_solution_l810_81076

def p (x : ℝ) : ℝ := x^2 - 4*x + 3
def tilde_p (x : ℝ) : ℝ := p (p x)

-- Proof problem: Prove tilde_p 2 = -4 
theorem problem_solution : tilde_p 2 = -4 := sorry

end NUMINAMATH_GPT_problem_solution_l810_81076


namespace NUMINAMATH_GPT_tenth_battery_replacement_in_january_l810_81041

theorem tenth_battery_replacement_in_january : ∀ (months_to_replace: ℕ) (start_month: ℕ), 
  months_to_replace = 4 → start_month = 1 → (4 * (10 - 1)) % 12 = 0 → start_month = 1 :=
by
  intros months_to_replace start_month h_replace h_start h_calc
  sorry

end NUMINAMATH_GPT_tenth_battery_replacement_in_january_l810_81041


namespace NUMINAMATH_GPT_theater_loss_l810_81082

/-- 
A movie theater has a total capacity of 50 people and charges $8 per ticket.
On a Tuesday night, they only sold 24 tickets. 
Prove that the revenue lost by not selling out is $208.
-/
theorem theater_loss 
  (capacity : ℕ) 
  (price : ℕ) 
  (sold_tickets : ℕ) 
  (h_cap : capacity = 50) 
  (h_price : price = 8) 
  (h_sold : sold_tickets = 24) : 
  capacity * price - sold_tickets * price = 208 :=
by
  sorry

end NUMINAMATH_GPT_theater_loss_l810_81082


namespace NUMINAMATH_GPT_product_of_solutions_of_abs_equation_l810_81085

theorem product_of_solutions_of_abs_equation :
  (∃ x₁ x₂ : ℚ, |5 * x₁ - 2| + 7 = 52 ∧ |5 * x₂ - 2| + 7 = 52 ∧ x₁ ≠ x₂ ∧ (x₁ * x₂ = -2021 / 25)) :=
sorry

end NUMINAMATH_GPT_product_of_solutions_of_abs_equation_l810_81085


namespace NUMINAMATH_GPT_bananas_in_each_box_l810_81032

-- You might need to consider noncomputable if necessary here for Lean's real number support.
noncomputable def bananas_per_box (total_bananas : ℕ) (total_boxes : ℕ) : ℕ :=
  total_bananas / total_boxes

theorem bananas_in_each_box :
  bananas_per_box 40 8 = 5 := by
  sorry

end NUMINAMATH_GPT_bananas_in_each_box_l810_81032


namespace NUMINAMATH_GPT_cone_base_circumference_l810_81005

theorem cone_base_circumference (r : ℝ) (θ : ℝ) (circ_res : ℝ) :
  r = 4 → θ = 270 → circ_res = 6 * Real.pi :=
by 
  sorry

end NUMINAMATH_GPT_cone_base_circumference_l810_81005


namespace NUMINAMATH_GPT_exponent_zero_value_of_neg_3_raised_to_zero_l810_81036

theorem exponent_zero (x : ℤ) (hx : x ≠ 0) : x ^ 0 = 1 :=
by
  -- Proof goes here
  sorry

theorem value_of_neg_3_raised_to_zero : (-3 : ℤ) ^ 0 = 1 :=
by
  exact exponent_zero (-3) (by norm_num)

end NUMINAMATH_GPT_exponent_zero_value_of_neg_3_raised_to_zero_l810_81036


namespace NUMINAMATH_GPT_find_f_at_75_l810_81068

variables (f : ℝ → ℝ) (h₀ : ∀ x, f (x + 2) = -f x)
variables (h₁ : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x)
variables (h₂ : ∀ x, f (-x) = -f x)

theorem find_f_at_75 : f 7.5 = -0.5 := by
  sorry

end NUMINAMATH_GPT_find_f_at_75_l810_81068


namespace NUMINAMATH_GPT_Victor_can_carry_7_trays_at_a_time_l810_81075

-- Define the conditions
def trays_from_first_table : Nat := 23
def trays_from_second_table : Nat := 5
def number_of_trips : Nat := 4

-- Define the total number of trays
def total_trays : Nat := trays_from_first_table + trays_from_second_table

-- Prove that the number of trays Victor can carry at a time is 7
theorem Victor_can_carry_7_trays_at_a_time :
  total_trays / number_of_trips = 7 :=
by
  sorry

end NUMINAMATH_GPT_Victor_can_carry_7_trays_at_a_time_l810_81075


namespace NUMINAMATH_GPT_min_points_game_12_l810_81028

noncomputable def player_scores := (18, 22, 9, 29)

def avg_after_eleven_games (scores: ℕ × ℕ × ℕ × ℕ) := 
  let s₁ := 78 -- Sum of the points in 8th, 9th, 10th, 11th games
  (s₁: ℕ) / 4

def points_twelve_game_cond (n: ℕ) : Prop :=
  let total_points := 78 + n
  total_points > (20 * 12)

theorem min_points_game_12 (points_in_first_7_games: ℕ) (score_12th_game: ℕ) 
  (H1: avg_after_eleven_games player_scores > (points_in_first_7_games / 7)) 
  (H2: points_twelve_game_cond score_12th_game):
  score_12th_game = 30 := by
  sorry

end NUMINAMATH_GPT_min_points_game_12_l810_81028


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l810_81027

theorem sufficient_but_not_necessary_condition 
  (a : ℕ → ℤ) 
  (h : ∀ n, |a (n + 1)| < a n) : 
  (∀ n, a (n + 1) < a n) ∧ 
  ¬(∀ n, a (n + 1) < a n → |a (n + 1)| < a n) := 
by 
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l810_81027


namespace NUMINAMATH_GPT_functional_equation_solution_l810_81061

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = 2 - x) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l810_81061


namespace NUMINAMATH_GPT_sum_of_midpoints_l810_81006

theorem sum_of_midpoints (a b c : ℝ) (h : a + b + c = 12) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_midpoints_l810_81006


namespace NUMINAMATH_GPT_green_light_probability_l810_81089

def red_duration : ℕ := 30
def green_duration : ℕ := 25
def yellow_duration : ℕ := 5

def total_cycle : ℕ := red_duration + green_duration + yellow_duration
def green_probability : ℚ := green_duration / total_cycle

theorem green_light_probability :
  green_probability = 5 / 12 := by
  sorry

end NUMINAMATH_GPT_green_light_probability_l810_81089


namespace NUMINAMATH_GPT_integers_solution_l810_81017

theorem integers_solution (a b : ℤ) (S D : ℤ) 
  (h1 : S = a + b) (h2 : D = a - b) (h3 : S / D = 3) (h4 : S * D = 300) : 
  ((a = 20 ∧ b = 10) ∨ (a = -20 ∧ b = -10)) :=
by
  sorry

end NUMINAMATH_GPT_integers_solution_l810_81017


namespace NUMINAMATH_GPT_parallel_planes_x_plus_y_l810_81019

def planes_parallel (x y : ℝ) : Prop :=
  ∃ k : ℝ, (x = -k) ∧ (1 = k * y) ∧ (-2 = (1 / 2) * k)

theorem parallel_planes_x_plus_y (x y : ℝ) (h : planes_parallel x y) : x + y = 15 / 4 :=
sorry

end NUMINAMATH_GPT_parallel_planes_x_plus_y_l810_81019


namespace NUMINAMATH_GPT_fraction_of_draws_is_two_ninths_l810_81062

-- Define the fraction of games that Ben wins and Tom wins
def BenWins : ℚ := 4 / 9
def TomWins : ℚ := 1 / 3

-- Definition of the fraction of games ending in a draw
def fraction_of_draws (BenWins TomWins : ℚ) : ℚ :=
  1 - (BenWins + TomWins)

-- The theorem to be proved
theorem fraction_of_draws_is_two_ninths : fraction_of_draws BenWins TomWins = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_draws_is_two_ninths_l810_81062


namespace NUMINAMATH_GPT_complement_intersection_l810_81003

open Set

variable (A B U : Set ℕ) 

theorem complement_intersection (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4, 5}) (hU : U = A ∪ B) :
  (U \ A) ∩ B = {4, 5} :=
by sorry

end NUMINAMATH_GPT_complement_intersection_l810_81003


namespace NUMINAMATH_GPT_border_area_correct_l810_81074

-- Definition of the dimensions of the photograph
def photo_height := 8
def photo_width := 10
def frame_border := 3

-- Definition of the areas of the photograph and the framed area
def photo_area := photo_height * photo_width
def frame_height := photo_height + 2 * frame_border
def frame_width := photo_width + 2 * frame_border
def frame_area := frame_height * frame_width

-- Theorem stating that the area of the border is 144 square inches
theorem border_area_correct : (frame_area - photo_area) = 144 := 
by
  sorry

end NUMINAMATH_GPT_border_area_correct_l810_81074


namespace NUMINAMATH_GPT_jimmy_earnings_l810_81044

theorem jimmy_earnings : 
  let price15 := 15
  let price20 := 20
  let discount := 5
  let sale_price15 := price15 - discount
  let sale_price20 := price20 - discount
  let num_low_worth := 4
  let num_high_worth := 1
  num_low_worth * sale_price15 + num_high_worth * sale_price20 = 55 :=
by
  sorry

end NUMINAMATH_GPT_jimmy_earnings_l810_81044


namespace NUMINAMATH_GPT_angle_quadrant_l810_81011

theorem angle_quadrant (α : ℝ) (h : 0 < α ∧ α < 90) : 90 < α + 180 ∧ α + 180 < 270 :=
by
  sorry

end NUMINAMATH_GPT_angle_quadrant_l810_81011


namespace NUMINAMATH_GPT_gcd_of_8_and_12_l810_81067

theorem gcd_of_8_and_12 :
  let a := 8
  let b := 12
  let lcm_ab := 24
  Nat.lcm a b = lcm_ab → Nat.gcd a b = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_gcd_of_8_and_12_l810_81067


namespace NUMINAMATH_GPT_max_area_l810_81021

noncomputable def PA : ℝ := 3
noncomputable def PB : ℝ := 4
noncomputable def PC : ℝ := 5
noncomputable def BC : ℝ := 6

theorem max_area (PA PB PC BC : ℝ) (hPA : PA = 3) (hPB : PB = 4) (hPC : PC = 5) (hBC : BC = 6) : 
  ∃ (A B C : Type) (area_ABC : ℝ), area_ABC = 19 := 
by 
  sorry

end NUMINAMATH_GPT_max_area_l810_81021


namespace NUMINAMATH_GPT_strictly_decreasing_interval_l810_81051

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

theorem strictly_decreasing_interval :
  ∀ x y : ℝ, (0 < x ∧ x < 1) ∧ (0 < y ∧ y < 1) ∧ y < x → f y < f x :=
by
  sorry

end NUMINAMATH_GPT_strictly_decreasing_interval_l810_81051


namespace NUMINAMATH_GPT_find_b_l810_81035

-- Define the function f(x)
def f (x : ℝ) : ℝ := 5 * x - 7

-- State the theorem
theorem find_b (b : ℝ) : f b = 0 ↔ b = 7 / 5 := by
  sorry

end NUMINAMATH_GPT_find_b_l810_81035


namespace NUMINAMATH_GPT_p_necessary_for_q_l810_81063

variable (x : ℝ)

def p := (x - 3) * (|x| + 1) < 0
def q := |1 - x| < 2

theorem p_necessary_for_q : (∀ x, q x → p x) ∧ (∃ x, q x) ∧ (∃ x, ¬(p x ∧ q x)) := by
  sorry

end NUMINAMATH_GPT_p_necessary_for_q_l810_81063
