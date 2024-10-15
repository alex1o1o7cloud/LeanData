import Mathlib

namespace NUMINAMATH_GPT_div_1959_l33_3368

theorem div_1959 (n : ℕ) : ∃ k : ℤ, 5^(8 * n) - 2^(4 * n) * 7^(2 * n) = k * 1959 := 
by 
  sorry

end NUMINAMATH_GPT_div_1959_l33_3368


namespace NUMINAMATH_GPT_chord_length_of_circle_intersected_by_line_l33_3351

open Real

-- Definitions for the conditions given in the problem
def line_eqn (x y : ℝ) : Prop := x - y - 1 = 0
def circle_eqn (x y : ℝ) : Prop := x^2 - 4 * x + y^2 = 4

-- The proof statement (problem) in Lean 4
theorem chord_length_of_circle_intersected_by_line :
  ∀ (x y : ℝ), circle_eqn x y → line_eqn x y → ∃ L : ℝ, L = sqrt 17 := by
  sorry

end NUMINAMATH_GPT_chord_length_of_circle_intersected_by_line_l33_3351


namespace NUMINAMATH_GPT_eighteen_women_time_l33_3374

theorem eighteen_women_time (h : ∀ (n : ℕ), n = 6 → ∀ (t : ℕ), t = 60 → true) : ∀ (n : ℕ), n = 18 → ∀ (t : ℕ), t = 20 → true :=
by
  sorry

end NUMINAMATH_GPT_eighteen_women_time_l33_3374


namespace NUMINAMATH_GPT_problem_1_problem_2_l33_3312

def f (a : ℝ) (x : ℝ) : ℝ := abs (a * x + 1)

def g (a : ℝ) (x : ℝ) : ℝ := f a x - abs (x + 1)

theorem problem_1 (a : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 ↔ f a x ≤ 3) → a = 2 := by
  intro h
  sorry

theorem problem_2 (a : ℝ) : a = 2 → (∃ x : ℝ, ∀ y : ℝ, g a y ≥ g a x ∧ g a x = -1/2) := by
  intro ha2
  use -1/2
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l33_3312


namespace NUMINAMATH_GPT_max_fraction_value_l33_3309

theorem max_fraction_value :
  ∀ (x y : ℝ), (1/4 ≤ x ∧ x ≤ 3/5) ∧ (1/5 ≤ y ∧ y ≤ 1/2) → 
    xy / (x^2 + y^2) ≤ 2/5 :=
by
  sorry

end NUMINAMATH_GPT_max_fraction_value_l33_3309


namespace NUMINAMATH_GPT_num_possible_values_a_l33_3372

theorem num_possible_values_a (a : ℕ) :
  (9 ∣ a) ∧ (a ∣ 18) ∧ (0 < a) → ∃ n : ℕ, n = 2 :=
by
  sorry

end NUMINAMATH_GPT_num_possible_values_a_l33_3372


namespace NUMINAMATH_GPT_average_speed_last_segment_l33_3382

variable (total_distance : ℕ := 120)
variable (total_minutes : ℕ := 120)
variable (first_segment_minutes : ℕ := 40)
variable (first_segment_speed : ℕ := 50)
variable (second_segment_minutes : ℕ := 40)
variable (second_segment_speed : ℕ := 55)
variable (third_segment_speed : ℕ := 75)

theorem average_speed_last_segment :
  let total_hours := total_minutes / 60
  let average_speed := total_distance / total_hours
  let speed_first_segment := first_segment_speed * (first_segment_minutes / 60)
  let speed_second_segment := second_segment_speed * (second_segment_minutes / 60)
  let speed_third_segment := third_segment_speed * (third_segment_minutes / 60)
  average_speed = (speed_first_segment + speed_second_segment + speed_third_segment) / 3 →
  third_segment_speed = 75 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_last_segment_l33_3382


namespace NUMINAMATH_GPT_candy_bars_eaten_l33_3349

theorem candy_bars_eaten (calories_per_candy : ℕ) (total_calories : ℕ) (h1 : calories_per_candy = 31) (h2 : total_calories = 341) :
  total_calories / calories_per_candy = 11 :=
by
  sorry

end NUMINAMATH_GPT_candy_bars_eaten_l33_3349


namespace NUMINAMATH_GPT_no_arithmetic_seq_with_sum_n_cubed_l33_3381

theorem no_arithmetic_seq_with_sum_n_cubed (a1 d : ℕ) :
  ¬ (∀ (n : ℕ), (n > 0) → (n / 2) * (2 * a1 + (n - 1) * d) = n^3) :=
sorry

end NUMINAMATH_GPT_no_arithmetic_seq_with_sum_n_cubed_l33_3381


namespace NUMINAMATH_GPT_length_of_adjacent_side_l33_3343

variable (a b : ℝ)

theorem length_of_adjacent_side (area : ℝ) (side : ℝ) :
  area = 6 * a^3 + 9 * a^2 - 3 * a * b →
  side = 3 * a →
  (area / side = 2 * a^2 + 3 * a - b) :=
by
  intro h_area
  intro h_side
  sorry

end NUMINAMATH_GPT_length_of_adjacent_side_l33_3343


namespace NUMINAMATH_GPT_factorize_one_factorize_two_l33_3357

variable (x a b : ℝ)

-- Problem 1: Prove that 4x^2 - 64 = 4(x + 4)(x - 4)
theorem factorize_one : 4 * x^2 - 64 = 4 * (x + 4) * (x - 4) :=
sorry

-- Problem 2: Prove that 4ab^2 - 4a^2b - b^3 = -b(2a - b)^2
theorem factorize_two : 4 * a * b^2 - 4 * a^2 * b - b^3 = -b * (2 * a - b)^2 :=
sorry

end NUMINAMATH_GPT_factorize_one_factorize_two_l33_3357


namespace NUMINAMATH_GPT_train_speed_l33_3307

theorem train_speed (distance time : ℝ) (h1 : distance = 450) (h2 : time = 8) : distance / time = 56.25 := by
  sorry

end NUMINAMATH_GPT_train_speed_l33_3307


namespace NUMINAMATH_GPT_fraction_still_missing_l33_3387

theorem fraction_still_missing (x : ℕ) (hx : x > 0) :
  let lost := (1/3 : ℚ) * x
  let found := (2/3 : ℚ) * lost
  let remaining := x - lost + found
  (x - remaining) / x = (1/9 : ℚ) :=
by
  let lost := (1/3 : ℚ) * x
  let found := (2/3 : ℚ) * lost
  let remaining := x - lost + found
  have h_fraction_still_missing : (x - remaining) / x = (1/9 : ℚ) := sorry
  exact h_fraction_still_missing

end NUMINAMATH_GPT_fraction_still_missing_l33_3387


namespace NUMINAMATH_GPT_kaleb_tickets_l33_3356

variable (T : Nat)
variable (tickets_left : Nat) (ticket_cost : Nat) (total_spent : Nat)

theorem kaleb_tickets : tickets_left = 3 → ticket_cost = 9 → total_spent = 27 → T = 6 :=
by
  sorry

end NUMINAMATH_GPT_kaleb_tickets_l33_3356


namespace NUMINAMATH_GPT_emily_remainder_l33_3344

theorem emily_remainder (c d : ℤ) (h1 : c % 60 = 53) (h2 : d % 42 = 35) : (c + d) % 21 = 4 :=
by
  sorry

end NUMINAMATH_GPT_emily_remainder_l33_3344


namespace NUMINAMATH_GPT_flu_epidemic_infection_rate_l33_3365

theorem flu_epidemic_infection_rate : 
  ∃ x : ℝ, 1 + x + x * (1 + x) = 100 ∧ x = 9 := 
by
  sorry

end NUMINAMATH_GPT_flu_epidemic_infection_rate_l33_3365


namespace NUMINAMATH_GPT_find_M_coordinates_l33_3391

-- Definition of the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  y ^ 2 = 2 * p * x

-- Definition to check if point M lies according to given conditions
def matchesCondition
  (p : ℝ) (M P O F : ℝ × ℝ) : Prop :=
  let xO := O.1
  let yO := O.2
  let xP := P.1
  let yP := P.2
  let xM := M.1
  let yM := M.2
  let xF := F.1
  let yF := F.2
  (xP = 2) ∧ (yP = 2 * p) ∧
  (xO = 0) ∧ (yO = 0) ∧
  (xF = p / 2) ∧ (yF = 0) ∧
  (Real.sqrt ((xM - xP) ^ 2 + (yM - yP) ^ 2) =
  Real.sqrt ((xM - xO) ^ 2 + (yM - yO) ^ 2)) ∧
  (Real.sqrt ((xM - xP) ^ 2 + (yM - yP) ^ 2) =
  Real.sqrt ((xM - xF) ^ 2 + (yM - yF) ^ 2))

-- Prove the coordinates of M satisfy the conditions
theorem find_M_coordinates :
  ∀ p : ℝ, p > 0 →
  matchesCondition p (1/4, 7/4) (2, 2 * p) (0, 0) (p / 2, 0) :=
by
  intros p hp
  simp [parabola, matchesCondition]
  sorry

end NUMINAMATH_GPT_find_M_coordinates_l33_3391


namespace NUMINAMATH_GPT_train_speed_l33_3302

theorem train_speed
  (train_length : Real := 460)
  (bridge_length : Real := 140)
  (time_seconds : Real := 48) :
  ((train_length + bridge_length) / time_seconds) * 3.6 = 45 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l33_3302


namespace NUMINAMATH_GPT_polynomial_coeff_fraction_eq_neg_122_div_121_l33_3318

theorem polynomial_coeff_fraction_eq_neg_122_div_121
  (a0 a1 a2 a3 a4 a5 : ℤ)
  (h1 : (2 - 1) ^ 5 = a0 + a1 * 1 + a2 * 1^2 + a3 * 1^3 + a4 * 1^4 + a5 * 1^5)
  (h2 : (2 - (-1)) ^ 5 = a0 + a1 * (-1) + a2 * (-1)^2 + a3 * (-1)^3 + a4 * (-1)^4 + a5 * (-1)^5)
  (h_sum1 : a0 + a1 + a2 + a3 + a4 + a5 = 1)
  (h_sum2 : a0 - a1 + a2 - a3 + a4 - a5 = 243) :
  (a0 + a2 + a4) / (a1 + a3 + a5) = - 122 / 121 :=
sorry

end NUMINAMATH_GPT_polynomial_coeff_fraction_eq_neg_122_div_121_l33_3318


namespace NUMINAMATH_GPT_intersection_with_x_axis_l33_3336

theorem intersection_with_x_axis (a : ℝ) (h : 2 * a - 4 = 0) : a = 2 := by
  sorry

end NUMINAMATH_GPT_intersection_with_x_axis_l33_3336


namespace NUMINAMATH_GPT_certain_number_eq_neg_thirteen_over_two_l33_3330

noncomputable def CertainNumber (w : ℝ) : ℝ := 13 * w / (1 - w)

theorem certain_number_eq_neg_thirteen_over_two (w : ℝ) (h : w ^ 2 = 1) (hz : 1 - w ≠ 0) :
  CertainNumber w = -13 / 2 :=
sorry

end NUMINAMATH_GPT_certain_number_eq_neg_thirteen_over_two_l33_3330


namespace NUMINAMATH_GPT_min_value_of_ellipse_l33_3346

noncomputable def min_m_plus_n (a b : ℝ) (h_ab_nonzero : a * b ≠ 0) (h_abs_diff : |a| ≠ |b|) : ℝ :=
(a ^ (2/3) + b ^ (2/3)) ^ (3/2)

theorem min_value_of_ellipse (m n a b : ℝ) (h1 : m > n) (h2 : n > 0) (h_ellipse : (a^2 / m^2) + (b^2 / n^2) = 1) (h_ab_nonzero : a * b ≠ 0) (h_abs_diff : |a| ≠ |b|) :
  (m + n) = min_m_plus_n a b h_ab_nonzero h_abs_diff :=
sorry

end NUMINAMATH_GPT_min_value_of_ellipse_l33_3346


namespace NUMINAMATH_GPT_log_base_10_of_2_bounds_l33_3331

theorem log_base_10_of_2_bounds :
  (10^3 = 1000) ∧ (10^4 = 10000) ∧ (2^11 = 2048) ∧ (2^14 = 16384) →
  (3 / 11 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (2 / 7 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_log_base_10_of_2_bounds_l33_3331


namespace NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l33_3314

-- Problem 1
theorem solve_quadratic_1 (x : ℝ) : (x - 1) ^ 2 - 4 = 0 ↔ (x = -1 ∨ x = 3) :=
by
  sorry

-- Problem 2
theorem solve_quadratic_2 (x : ℝ) : (2 * x - 1) * (x + 3) = 4 ↔ (x = -7 / 2 ∨ x = 1) :=
by
  sorry

-- Problem 3
theorem solve_quadratic_3 (x : ℝ) : 2 * x ^ 2 - 5 * x + 2 = 0 ↔ (x = 2 ∨ x = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l33_3314


namespace NUMINAMATH_GPT_fran_travel_time_l33_3335

theorem fran_travel_time (joann_speed fran_speed : ℝ) (joann_time joann_distance : ℝ) :
  joann_speed = 15 → joann_time = 4 → joann_distance = joann_speed * joann_time →
  fran_speed = 20 → fran_time = joann_distance / fran_speed →
  fran_time = 3 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_fran_travel_time_l33_3335


namespace NUMINAMATH_GPT_triangle_ABC_problem_l33_3360

noncomputable def perimeter_of_triangle (a b c : ℝ) : ℝ := a + b + c

theorem triangle_ABC_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : a = 3) 
  (h2 : B = π / 3) 
  (area : ℝ)
  (h3 : (1/2) * a * c * Real.sin B = 6 * Real.sqrt 3) :

  perimeter_of_triangle a b c = 18 ∧ 
  Real.sin (2 * A) = 39 * Real.sqrt 3 / 98 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_ABC_problem_l33_3360


namespace NUMINAMATH_GPT_mike_office_visits_per_day_l33_3322

-- Define the constants from the conditions
def pull_ups_per_visit : ℕ := 2
def total_pull_ups_per_week : ℕ := 70
def days_per_week : ℕ := 7

-- Calculate total office visits per week
def office_visits_per_week : ℕ := total_pull_ups_per_week / pull_ups_per_visit

-- Lean statement that states Mike goes into his office 5 times a day
theorem mike_office_visits_per_day : office_visits_per_week / days_per_week = 5 := by
  sorry

end NUMINAMATH_GPT_mike_office_visits_per_day_l33_3322


namespace NUMINAMATH_GPT_value_of_polynomial_l33_3398

variable {R : Type} [CommRing R]

theorem value_of_polynomial 
  (m : R) 
  (h : 2 * m^2 - 3 * m - 1 = 0) : 
  6 * m^2 - 9 * m + 2019 = 2022 := by
  sorry

end NUMINAMATH_GPT_value_of_polynomial_l33_3398


namespace NUMINAMATH_GPT_fraction_day_crew_loaded_l33_3386

variable (D W : ℕ)  -- D: Number of boxes loaded by each worker on the day crew, W: Number of workers on the day crew

-- Condition 1: Each worker on the night crew loaded 3/4 as many boxes as each worker on the day crew
def boxes_loaded_night_worker : ℕ := 3 * D / 4
-- Condition 2: The night crew has 5/6 as many workers as the day crew
def workers_night : ℕ := 5 * W / 6

-- Question: Fraction of all the boxes loaded by the day crew
theorem fraction_day_crew_loaded :
  (D * W : ℚ) / ((D * W) + (3 * D / 4) * (5 * W / 6)) = (8 / 13) := by
  sorry

end NUMINAMATH_GPT_fraction_day_crew_loaded_l33_3386


namespace NUMINAMATH_GPT_price_per_pound_of_rocks_l33_3370

def number_of_rocks : ℕ := 10
def average_weight_per_rock : ℝ := 1.5
def total_amount_made : ℝ := 60

theorem price_per_pound_of_rocks:
  (total_amount_made / (number_of_rocks * average_weight_per_rock)) = 4 := 
by
  sorry

end NUMINAMATH_GPT_price_per_pound_of_rocks_l33_3370


namespace NUMINAMATH_GPT_cost_of_dozen_pens_l33_3350

theorem cost_of_dozen_pens 
  (x : ℝ)
  (hx_pos : 0 < x)
  (h1 : 3 * (5 * x) + 5 * x = 150)
  (h2 : 5 * x / x = 5): 
  12 * (5 * x) = 450 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_dozen_pens_l33_3350


namespace NUMINAMATH_GPT_ratio_of_15th_term_l33_3375

theorem ratio_of_15th_term (a d b e : ℤ) :
  (∀ n : ℕ, (n * (2 * a + (n - 1) * d)) / (n * (2 * b + (n - 1) * e)) = (7 * n^2 + 1) / (4 * n^2 + 27)) →
  (a + 14 * d) / (b + 14 * e) = 7 / 4 :=
by sorry

end NUMINAMATH_GPT_ratio_of_15th_term_l33_3375


namespace NUMINAMATH_GPT_non_square_solution_equiv_l33_3354

theorem non_square_solution_equiv 
  (a b : ℤ) (h1 : ¬∃ k : ℤ, a = k^2) (h2 : ¬∃ k : ℤ, b = k^2) :
  (∃ x y z w : ℤ, x^2 - a * y^2 - b * z^2 + a * b * w^2 = 0 ∧ (x, y, z, w) ≠ (0, 0, 0, 0)) ↔
  (∃ x y z : ℤ, x^2 - a * y^2 - b * z^2 = 0 ∧ (x, y, z) ≠ (0, 0, 0)) :=
by sorry

end NUMINAMATH_GPT_non_square_solution_equiv_l33_3354


namespace NUMINAMATH_GPT_quadrilateral_ABCD_pq_sum_l33_3361

noncomputable def AB_pq_sum : ℕ :=
  let p : ℕ := 9
  let q : ℕ := 141
  p + q

theorem quadrilateral_ABCD_pq_sum (BC CD AD : ℕ) (m_angle_A m_angle_B : ℕ) (hBC : BC = 8) (hCD : CD = 12) (hAD : AD = 10) (hAngleA : m_angle_A = 60) (hAngleB : m_angle_B = 60) : AB_pq_sum = 150 := by sorry

end NUMINAMATH_GPT_quadrilateral_ABCD_pq_sum_l33_3361


namespace NUMINAMATH_GPT_find_quotient_l33_3352

theorem find_quotient (dividend divisor remainder quotient : ℕ) 
  (h1 : dividend = 23) (h2 : divisor = 4) (h3 : remainder = 3)
  (h4 : dividend = (divisor * quotient) + remainder) : quotient = 5 :=
sorry

end NUMINAMATH_GPT_find_quotient_l33_3352


namespace NUMINAMATH_GPT_range_of_f_l33_3359

noncomputable def f (x : ℝ) : ℝ := (Real.arccos x) ^ 3 + (Real.arcsin x) ^ 3

theorem range_of_f : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 
           ∃ y : ℝ, y = f x ∧ (y ≥ (Real.pi ^ 3) / 32) ∧ (y ≤ (7 * (Real.pi ^ 3)) / 8) :=
sorry

end NUMINAMATH_GPT_range_of_f_l33_3359


namespace NUMINAMATH_GPT_smallest_model_length_l33_3345

theorem smallest_model_length 
  (full_size_length : ℕ)
  (mid_size_ratio : ℚ)
  (smallest_size_ratio : ℚ)
  (H1 : full_size_length = 240)
  (H2 : mid_size_ratio = 1/10)
  (H3 : smallest_size_ratio = 1/2) 
  : full_size_length * mid_size_ratio * smallest_size_ratio = 12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_model_length_l33_3345


namespace NUMINAMATH_GPT_difference_cubics_divisible_by_24_l33_3366

theorem difference_cubics_divisible_by_24 
    (a b : ℤ) (h : ∃ k : ℤ, a - b = 3 * k) : 
    ∃ k : ℤ, (2 * a + 1)^3 - (2 * b + 1)^3 = 24 * k :=
by
  sorry

end NUMINAMATH_GPT_difference_cubics_divisible_by_24_l33_3366


namespace NUMINAMATH_GPT_change_is_24_l33_3310

-- Define the prices and quantities
def price_basketball_card : ℕ := 3
def price_baseball_card : ℕ := 4
def num_basketball_cards : ℕ := 2
def num_baseball_cards : ℕ := 5
def money_paid : ℕ := 50

-- Define the total cost
def total_cost : ℕ := (num_basketball_cards * price_basketball_card) + (num_baseball_cards * price_baseball_card)

-- Define the change received
def change_received : ℕ := money_paid - total_cost

-- Prove that the change received is $24
theorem change_is_24 : change_received = 24 := by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_change_is_24_l33_3310


namespace NUMINAMATH_GPT_digit_sum_10_pow_93_minus_937_l33_3399

-- Define a function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem digit_sum_10_pow_93_minus_937 :
  sum_of_digits (10^93 - 937) = 819 :=
by
  sorry

end NUMINAMATH_GPT_digit_sum_10_pow_93_minus_937_l33_3399


namespace NUMINAMATH_GPT_sin_eq_sqrt3_div_2_range_l33_3376

theorem sin_eq_sqrt3_div_2_range :
  {x | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin x ≥ Real.sqrt 3 / 2} = 
  {x | Real.pi / 3 ≤ x ∧ x ≤ 2 * Real.pi / 3} :=
sorry

end NUMINAMATH_GPT_sin_eq_sqrt3_div_2_range_l33_3376


namespace NUMINAMATH_GPT_solve_for_x_l33_3389

theorem solve_for_x :
  ∃ x : ℝ, 5 ^ (Real.logb 5 15) = 7 * x + 2 ∧ x = 13 / 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l33_3389


namespace NUMINAMATH_GPT_geometric_series_sum_l33_3329

theorem geometric_series_sum 
  (a : ℝ) (r : ℝ) (s : ℝ)
  (h_a : a = 9)
  (h_r : r = -2/3)
  (h_abs_r : |r| < 1)
  (h_s : s = a / (1 - r)) : 
  s = 5.4 := by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l33_3329


namespace NUMINAMATH_GPT_ratio_of_pond_to_field_area_l33_3367

theorem ratio_of_pond_to_field_area
  (l w : ℕ)
  (field_area pond_area : ℕ)
  (h1 : l = 2 * w)
  (h2 : l = 36)
  (h3 : pond_area = 9 * 9)
  (field_area_def : field_area = l * w)
  (pond_area_def : pond_area = 81) :
  pond_area / field_area = 1 / 8 := 
sorry

end NUMINAMATH_GPT_ratio_of_pond_to_field_area_l33_3367


namespace NUMINAMATH_GPT_compound_interest_rate_l33_3324

theorem compound_interest_rate :
  ∀ (P A : ℝ) (t n : ℕ) (r : ℝ),
  P = 12000 →
  A = 21500 →
  t = 5 →
  n = 1 →
  A = P * (1 + r / n) ^ (n * t) →
  r = 0.121898 :=
by
  intros P A t n r hP hA ht hn hCompound
  sorry

end NUMINAMATH_GPT_compound_interest_rate_l33_3324


namespace NUMINAMATH_GPT_rectangle_vertices_complex_plane_l33_3390

theorem rectangle_vertices_complex_plane (b : ℝ) :
  (∀ (z : ℂ), z^4 - 10*z^3 + (16*b : ℂ)*z^2 - 2*(3*b^2 - 5*b + 4 : ℂ)*z + 6 = 0 →
    (∃ (w₁ w₂ : ℂ), z = w₁ ∨ z = w₂)) →
  (b = 5 / 3 ∨ b = 2) :=
sorry

end NUMINAMATH_GPT_rectangle_vertices_complex_plane_l33_3390


namespace NUMINAMATH_GPT_tree_height_at_2_years_l33_3377

theorem tree_height_at_2_years (h : ℕ → ℕ) 
  (h_growth : ∀ n, h (n + 1) = 3 * h n) 
  (h_5 : h 5 = 243) : 
  h 2 = 9 := 
sorry

end NUMINAMATH_GPT_tree_height_at_2_years_l33_3377


namespace NUMINAMATH_GPT_find_value_of_a_l33_3325

noncomputable def log_base_four (a : ℝ) : ℝ := Real.log a / Real.log 4

theorem find_value_of_a (a : ℝ) (h : log_base_four a = (1 : ℝ) / (2 : ℝ)) : a = 2 := by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l33_3325


namespace NUMINAMATH_GPT_problem_solution_l33_3308

-- Definitions for the digits and arithmetic conditions
def is_digit (n : ℕ) : Prop := n < 10

-- Problem conditions stated in Lean
variables (A B C D E : ℕ)

-- Define the conditions
axiom digits_A : is_digit A
axiom digits_B : is_digit B
axiom digits_C : is_digit C
axiom digits_D : is_digit D
axiom digits_E : is_digit E

-- Subtraction result for second equation
axiom sub_eq : A - C = A

-- Additional conditions derived from the problem
axiom add_eq : (E + E = D)

-- Now, state the problem in Lean
theorem problem_solution : D = 8 :=
sorry

end NUMINAMATH_GPT_problem_solution_l33_3308


namespace NUMINAMATH_GPT_equality_of_expressions_l33_3397

theorem equality_of_expressions :
  (2^3 ≠ 2 * 3) ∧
  (-(-2)^2 ≠ (-2)^2) ∧
  (-3^2 ≠ 3^2) ∧
  (-2^3 = (-2)^3) :=
by
  sorry

end NUMINAMATH_GPT_equality_of_expressions_l33_3397


namespace NUMINAMATH_GPT_sum_of_two_numbers_l33_3339

-- Define the two numbers and conditions
variables {x y : ℝ}
axiom prod_eq : x * y = 120
axiom sum_squares_eq : x^2 + y^2 = 289

-- The statement we want to prove
theorem sum_of_two_numbers (x y : ℝ) (prod_eq : x * y = 120) (sum_squares_eq : x^2 + y^2 = 289) : x + y = 23 :=
sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l33_3339


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_is_33_l33_3311

theorem sum_of_consecutive_integers_is_33 :
  ∃ (x : ℕ), x * (x + 1) = 272 ∧ x + (x + 1) = 33 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_is_33_l33_3311


namespace NUMINAMATH_GPT_ratio_of_intercepts_l33_3371

theorem ratio_of_intercepts (b s t : ℝ) (h1 : s = -2 * b / 5) (h2 : t = -3 * b / 7) :
  s / t = 14 / 15 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_intercepts_l33_3371


namespace NUMINAMATH_GPT_complement_set_A_in_U_l33_3300

-- Given conditions
def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {x | x ∈ U ∧ x^2 < 1}

-- Theorem to prove complement
theorem complement_set_A_in_U :
  U \ A = {-1, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_set_A_in_U_l33_3300


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l33_3373

-- Define the condition p: x^2 - x < 0
def p (x : ℝ) : Prop := x^2 - x < 0

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (x : ℝ) : Prop := -1 < x ∧ x < 1

-- State the theorem
theorem necessary_but_not_sufficient_condition :
  ∀ x : ℝ, p x → necessary_but_not_sufficient x :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l33_3373


namespace NUMINAMATH_GPT_additional_payment_is_65_l33_3380

def installments (n : ℕ) : ℤ := 65
def first_payment : ℕ := 20
def first_amount : ℤ := 410
def remaining_payment (x : ℤ) : ℕ := 45
def remaining_amount (x : ℤ) : ℤ := 410 + x
def average_amount : ℤ := 455

-- Define the total amount paid using both methods
def total_amount (x : ℤ) : ℤ := (20 * 410) + (45 * (410 + x))
def total_average : ℤ := 65 * 455

theorem additional_payment_is_65 :
  total_amount 65 = total_average :=
sorry

end NUMINAMATH_GPT_additional_payment_is_65_l33_3380


namespace NUMINAMATH_GPT_average_of_ABC_l33_3317

theorem average_of_ABC (A B C : ℤ)
  (h1 : 101 * C - 202 * A = 404)
  (h2 : 101 * B + 303 * A = 505)
  (h3 : 101 * A + 101 * B + 101 * C = 303) :
  (A + B + C) / 3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_average_of_ABC_l33_3317


namespace NUMINAMATH_GPT_min_value_expression_l33_3348

theorem min_value_expression (x : ℝ) (h : x > 10) : (x^2) / (x - 10) ≥ 40 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l33_3348


namespace NUMINAMATH_GPT_line_intersects_x_axis_at_point_l33_3332

theorem line_intersects_x_axis_at_point (x1 y1 x2 y2 : ℝ) 
  (h1 : (x1, y1) = (7, -3))
  (h2 : (x2, y2) = (3, 1)) : 
  ∃ x, (x, 0) = (4, 0) :=
by
  -- sorry serves as a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_line_intersects_x_axis_at_point_l33_3332


namespace NUMINAMATH_GPT_sum_of_reciprocal_squares_l33_3362

theorem sum_of_reciprocal_squares
  (p q r : ℝ)
  (h1 : p + q + r = 9)
  (h2 : p * q + q * r + r * p = 8)
  (h3 : p * q * r = -2) :
  (1 / p ^ 2 + 1 / q ^ 2 + 1 / r ^ 2) = 25 := by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocal_squares_l33_3362


namespace NUMINAMATH_GPT_points_on_opposite_sides_l33_3306

theorem points_on_opposite_sides (a : ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) 
    (hA : A = (a, 1)) 
    (hB : B = (2, a)) 
    (opposite_sides : A.1 < 0 ∧ B.1 > 0 ∨ A.1 > 0 ∧ B.1 < 0) 
    : a < 0 := 
  sorry

end NUMINAMATH_GPT_points_on_opposite_sides_l33_3306


namespace NUMINAMATH_GPT_special_sale_day_price_l33_3313

-- Define the original price
def original_price : ℝ := 250

-- Define the first discount rate
def first_discount_rate : ℝ := 0.40

-- Calculate the price after the first discount
def price_after_first_discount (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

-- Define the second discount rate (special sale day)
def second_discount_rate : ℝ := 0.10

-- Calculate the price after the second discount
def price_after_second_discount (discounted_price : ℝ) (discount_rate : ℝ) : ℝ :=
  discounted_price * (1 - discount_rate)

-- Theorem statement
theorem special_sale_day_price :
  price_after_second_discount (price_after_first_discount original_price first_discount_rate) second_discount_rate = 135 := by
  sorry

end NUMINAMATH_GPT_special_sale_day_price_l33_3313


namespace NUMINAMATH_GPT_winning_candidate_votes_l33_3393

theorem winning_candidate_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 336): 0.62 * V = 868 :=
by
  sorry

end NUMINAMATH_GPT_winning_candidate_votes_l33_3393


namespace NUMINAMATH_GPT_smallest_w_value_l33_3321

theorem smallest_w_value (w : ℕ) (hw : w > 0) :
  (∀ k : ℕ, (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (10^2 ∣ 936 * w)) ↔ w = 900 := 
sorry

end NUMINAMATH_GPT_smallest_w_value_l33_3321


namespace NUMINAMATH_GPT_equivalent_operation_l33_3384

theorem equivalent_operation (x : ℚ) : 
  (x * (2 / 3)) / (4 / 7) = x * (7 / 6) :=
by sorry

end NUMINAMATH_GPT_equivalent_operation_l33_3384


namespace NUMINAMATH_GPT_triangle_area_triangle_perimeter_l33_3342

noncomputable def area_of_triangle (A B C : ℝ) (a b c : ℝ) := 
  1/2 * b * c * (Real.sin A)

theorem triangle_area (A B C a b c : ℝ) 
  (h1 : b^2 + c^2 - a^2 = bc) 
  (h2 : A = Real.pi / 3) : 
  area_of_triangle A B C a b c = Real.sqrt 3 / 4 := 
  sorry

theorem triangle_perimeter (A B C a b c : ℝ) 
  (h1 : b^2 + c^2 - a^2 = bc) 
  (h2 : 4 * Real.cos B * Real.cos C - 1 = 0) 
  (h3 : b + c = 2)
  (h4 : a = 1) :
  a + b + c = 3 :=
  sorry

end NUMINAMATH_GPT_triangle_area_triangle_perimeter_l33_3342


namespace NUMINAMATH_GPT_cannot_be_right_angle_triangle_l33_3369

-- Definition of the converse of the Pythagorean theorem
def is_right_angle_triangle (a b c : ℕ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2

-- Definition to check if a given set of sides cannot form a right-angled triangle
def cannot_form_right_angle_triangle (a b c : ℕ) : Prop :=
  ¬ is_right_angle_triangle a b c

-- Given sides of the triangle option D
theorem cannot_be_right_angle_triangle : cannot_form_right_angle_triangle 3 4 6 :=
  by sorry

end NUMINAMATH_GPT_cannot_be_right_angle_triangle_l33_3369


namespace NUMINAMATH_GPT_correct_calculation_l33_3303

theorem correct_calculation (x : ℝ) (h : 63 + x = 69) : 36 / x = 6 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l33_3303


namespace NUMINAMATH_GPT_smallest_area_of_2020th_square_l33_3304

theorem smallest_area_of_2020th_square :
  ∃ (S : ℤ) (A : ℕ), 
    (S * S - 2019 = A) ∧ 
    (∃ k : ℕ, k * k = A) ∧ 
    (∀ (T : ℤ) (B : ℕ), ((T * T - 2019 = B) ∧ (∃ l : ℕ, l * l = B)) → (A ≤ B)) :=
sorry

end NUMINAMATH_GPT_smallest_area_of_2020th_square_l33_3304


namespace NUMINAMATH_GPT_ratio_of_u_to_v_l33_3320

theorem ratio_of_u_to_v (b u v : ℝ) (Hu : u = -b/12) (Hv : v = -b/8) : 
  u / v = 2 / 3 := 
sorry

end NUMINAMATH_GPT_ratio_of_u_to_v_l33_3320


namespace NUMINAMATH_GPT_find_a_l33_3316

def star (a b : ℝ) : ℝ := 2 * a - b^2

theorem find_a (a : ℝ) (h : star a 5 = 9) : a = 17 := by
  sorry

end NUMINAMATH_GPT_find_a_l33_3316


namespace NUMINAMATH_GPT_pineapples_sold_l33_3340

/-- 
There were initially 86 pineapples in the store. After selling some pineapples,
9 of the remaining pineapples were rotten and were discarded. Given that there 
are 29 fresh pineapples left, prove that the number of pineapples sold is 48.
-/
theorem pineapples_sold (initial_pineapples : ℕ) (rotten_pineapples : ℕ) (remaining_fresh_pineapples : ℕ)
  (h_init : initial_pineapples = 86)
  (h_rotten : rotten_pineapples = 9)
  (h_fresh : remaining_fresh_pineapples = 29) :
  initial_pineapples - (remaining_fresh_pineapples + rotten_pineapples) = 48 :=
sorry

end NUMINAMATH_GPT_pineapples_sold_l33_3340


namespace NUMINAMATH_GPT_range_of_a_l33_3301

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x < 2 → (a+1)*x > 2*a+2) → a < -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l33_3301


namespace NUMINAMATH_GPT_expand_and_simplify_l33_3334

theorem expand_and_simplify (x : ℝ) : (2*x + 6)*(x + 9) = 2*x^2 + 24*x + 54 :=
by
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l33_3334


namespace NUMINAMATH_GPT_seventy_fifth_elem_in_s_l33_3353

-- Define the set s
def s : Set ℕ := {x | ∃ n : ℕ, x = 8 * n + 5}

-- State the main theorem
theorem seventy_fifth_elem_in_s : (∃ n : ℕ, n = 74 ∧ (8 * n + 5) = 597) :=
by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_GPT_seventy_fifth_elem_in_s_l33_3353


namespace NUMINAMATH_GPT_find_N_l33_3364

-- Definitions and conditions directly appearing in the problem
variable (X Y Z N : ℝ)

axiom condition1 : 0.15 * X = 0.25 * N + Y
axiom condition2 : X + Y = Z

-- The theorem to prove
theorem find_N : N = 4.6 * X - 4 * Z := by
  sorry

end NUMINAMATH_GPT_find_N_l33_3364


namespace NUMINAMATH_GPT_min_value_of_a_plus_b_l33_3326

theorem min_value_of_a_plus_b (a b : ℤ) (h_ab : a * b = 72) (h_even : a % 2 = 0) : a + b ≥ -38 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_plus_b_l33_3326


namespace NUMINAMATH_GPT_unique_array_count_l33_3355

theorem unique_array_count (n m : ℕ) (h_conds : n * m = 49 ∧ n ≥ 2 ∧ m ≥ 2 ∧ n = m) :
  ∃! (n m : ℕ), (n * m = 49 ∧ n ≥ 2 ∧ m ≥ 2 ∧ n = m) :=
by
  sorry

end NUMINAMATH_GPT_unique_array_count_l33_3355


namespace NUMINAMATH_GPT_complement_A_inter_B_l33_3327

def A : Set ℝ := {x | abs (x - 2) ≤ 2}

def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

def A_inter_B : Set ℝ := A ∩ B

def C_R (s : Set ℝ) : Set ℝ := {x | x ∉ s}

theorem complement_A_inter_B :
  C_R A_inter_B = {x | x < 0} ∪ {x | x > 0} :=
by
  sorry

end NUMINAMATH_GPT_complement_A_inter_B_l33_3327


namespace NUMINAMATH_GPT_xy_product_eq_two_l33_3347

theorem xy_product_eq_two (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 2 / x = y + 2 / y) : x * y = 2 := 
sorry

end NUMINAMATH_GPT_xy_product_eq_two_l33_3347


namespace NUMINAMATH_GPT_chen_recording_l33_3333

variable (standard xia_steps chen_steps : ℕ)
variable (xia_record : ℤ)

-- Conditions: 
-- standard = 5000
-- Xia walked 6200 steps, recorded as +1200 steps
def met_standard (s : ℕ) : Prop :=
  s >= 5000

def xia_condition := (xia_steps = 6200) ∧ (xia_record = 1200) ∧ (xia_record = (xia_steps : ℤ) - 5000)

-- Question and solution combined into a statement: 
-- Chen walked 4800 steps, recorded as -200 steps
def chen_condition := (chen_steps = 4800) ∧ (met_standard chen_steps = false) → (((standard : ℤ) - chen_steps) * -1 = -200)

-- Proof goal:
theorem chen_recording (h₁ : standard = 5000) (h₂ : xia_condition xia_steps xia_record):
  chen_condition standard chen_steps :=
by
  sorry

end NUMINAMATH_GPT_chen_recording_l33_3333


namespace NUMINAMATH_GPT_find_valid_pairs_l33_3385

def satisfies_condition (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ (a ^ 2017 + b) % (a * b) = 0

theorem find_valid_pairs : 
  ∀ (a b : ℕ), satisfies_condition a b → (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 2 ^ 2017) := 
by
  sorry

end NUMINAMATH_GPT_find_valid_pairs_l33_3385


namespace NUMINAMATH_GPT_symmetric_points_add_l33_3341

theorem symmetric_points_add (a b : ℝ) : 
  (P : ℝ × ℝ) → (Q : ℝ × ℝ) →
  P = (a-1, 5) →
  Q = (2, b-1) →
  (P.fst = Q.fst) →
  P.snd = -Q.snd →
  a + b = -1 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_points_add_l33_3341


namespace NUMINAMATH_GPT_quadratic_min_value_l33_3388

theorem quadratic_min_value (p r : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = x^2 + 2 * p * x + r) (h₁ : ∃ x₀, f x₀ = 1 ∧ ∀ x, f x₀ ≤ f x) : r = p^2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_min_value_l33_3388


namespace NUMINAMATH_GPT_keith_missed_games_l33_3315

-- Define the total number of football games
def total_games : ℕ := 8

-- Define the number of games Keith attended
def attended_games : ℕ := 4

-- Define the number of games played at night (although it is not directly necessary for the proof)
def night_games : ℕ := 4

-- Define the number of games Keith missed
def missed_games : ℕ := total_games - attended_games

-- Prove that the number of games Keith missed is 4
theorem keith_missed_games : missed_games = 4 := by
  sorry

end NUMINAMATH_GPT_keith_missed_games_l33_3315


namespace NUMINAMATH_GPT_find_length_AE_l33_3338

theorem find_length_AE (AB BC CD DE AC CE AE : ℕ) 
  (h1 : AB = 2) 
  (h2 : BC = 2) 
  (h3 : CD = 5) 
  (h4 : DE = 7)
  (h5 : AC > 2) 
  (h6 : AC < 4) 
  (h7 : CE > 2) 
  (h8 : CE < 5)
  (h9 : AC ≠ CE)
  (h10 : AC ≠ AE)
  (h11 : CE ≠ AE)
  : AE = 5 :=
sorry

end NUMINAMATH_GPT_find_length_AE_l33_3338


namespace NUMINAMATH_GPT_pages_per_day_l33_3396

def notebooks : Nat := 5
def pages_per_notebook : Nat := 40
def total_days : Nat := 50

theorem pages_per_day (H1 : notebooks = 5) (H2 : pages_per_notebook = 40) (H3 : total_days = 50) : 
  (notebooks * pages_per_notebook / total_days) = 4 := by
  sorry

end NUMINAMATH_GPT_pages_per_day_l33_3396


namespace NUMINAMATH_GPT_circle_properties_intercept_length_l33_3395

theorem circle_properties (a r : ℝ) (h1 : a^2 + 16 = r^2) (h2 : (6 - a)^2 + 16 = r^2) (h3 : r > 0) :
  a = 3 ∧ r = 5 :=
by
  sorry

theorem intercept_length (m : ℝ) (h : |24 + m| / 5 = 3) :
  m = -4 ∨ m = -44 :=
by
  sorry

end NUMINAMATH_GPT_circle_properties_intercept_length_l33_3395


namespace NUMINAMATH_GPT_pencils_per_box_l33_3305

theorem pencils_per_box:
  ∀ (red_pencils blue_pencils yellow_pencils green_pencils total_pencils num_boxes : ℕ),
  red_pencils = 20 →
  blue_pencils = 2 * red_pencils →
  yellow_pencils = 40 →
  green_pencils = red_pencils + blue_pencils →
  total_pencils = red_pencils + blue_pencils + yellow_pencils + green_pencils →
  num_boxes = 8 →
  total_pencils / num_boxes = 20 :=
by
  intros red_pencils blue_pencils yellow_pencils green_pencils total_pencils num_boxes
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_pencils_per_box_l33_3305


namespace NUMINAMATH_GPT_solve_inequality_l33_3383

-- Define the conditions
def condition_inequality (x : ℝ) : Prop := abs x + abs (2 * x - 3) ≥ 6

-- Define the solution set form
def solution_set (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 3

-- State the theorem
theorem solve_inequality (x : ℝ) : condition_inequality x → solution_set x := 
by 
  sorry

end NUMINAMATH_GPT_solve_inequality_l33_3383


namespace NUMINAMATH_GPT_polynomial_divisible_by_square_l33_3328

def f (x : ℝ) (a1 a2 a3 a4 : ℝ) : ℝ := x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4
def f' (x : ℝ) (a1 a2 a3 : ℝ) : ℝ := 4 * x^3 + 3 * a1 * x^2 + 2 * a2 * x + a3

theorem polynomial_divisible_by_square (x0 a1 a2 a3 a4 : ℝ) 
  (h1 : f x0 a1 a2 a3 a4 = 0) 
  (h2 : f' x0 a1 a2 a3 = 0) : 
  ∃ g : ℝ → ℝ, ∀ x : ℝ, f x a1 a2 a3 a4 = (x - x0)^2 * (g x) :=
sorry

end NUMINAMATH_GPT_polynomial_divisible_by_square_l33_3328


namespace NUMINAMATH_GPT_max_value_of_a_l33_3358

theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a * x + a ≥ 0) → a ≤ 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_max_value_of_a_l33_3358


namespace NUMINAMATH_GPT_middle_aged_employees_participating_l33_3392

-- Define the total number of employees and the ratio
def total_employees : ℕ := 1200
def ratio_elderly : ℕ := 1
def ratio_middle_aged : ℕ := 5
def ratio_young : ℕ := 6

-- Define the number of employees chosen for the performance
def chosen_employees : ℕ := 36

-- Calculate the number of middle-aged employees participating in the performance
theorem middle_aged_employees_participating : (36 * ratio_middle_aged / (ratio_elderly + ratio_middle_aged + ratio_young)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_middle_aged_employees_participating_l33_3392


namespace NUMINAMATH_GPT_rectangle_placement_l33_3394

theorem rectangle_placement (a b c d : ℝ)
  (h1 : a < c)
  (h2 : c < d)
  (h3 : d < b)
  (h4 : a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b * d - a * c)^2 + (b * c - a * d)^2 :=
sorry

end NUMINAMATH_GPT_rectangle_placement_l33_3394


namespace NUMINAMATH_GPT_last_two_digits_x_pow_y_add_y_pow_x_l33_3323

noncomputable def proof_problem (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1/x + 1/y = 2/13) : ℕ :=
  (x^y + y^x) % 100

theorem last_two_digits_x_pow_y_add_y_pow_x {x y : ℕ} (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1/x + 1/y = 2/13) : 
  proof_problem x y h1 h2 h3 h4 = 74 :=
sorry

end NUMINAMATH_GPT_last_two_digits_x_pow_y_add_y_pow_x_l33_3323


namespace NUMINAMATH_GPT_cricket_team_players_l33_3379

theorem cricket_team_players (P N : ℕ) (h1 : 37 = 37) 
  (h2 : (57 - 37) = 20) 
  (h3 : ∀ N, (2 / 3 : ℚ) * N = 20 → N = 30) 
  (h4 : P = 37 + 30) : P = 67 := 
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_cricket_team_players_l33_3379


namespace NUMINAMATH_GPT_fraction_equality_l33_3337

theorem fraction_equality : (16 : ℝ) / (8 * 17) = (1.6 : ℝ) / (0.8 * 17) := 
sorry

end NUMINAMATH_GPT_fraction_equality_l33_3337


namespace NUMINAMATH_GPT_number_less_than_one_is_correct_l33_3319

theorem number_less_than_one_is_correct : (1 - 5 = -4) :=
by
  sorry

end NUMINAMATH_GPT_number_less_than_one_is_correct_l33_3319


namespace NUMINAMATH_GPT_greatest_number_of_quarters_l33_3363

def eva_has_us_coins : ℝ := 4.80
def quarters_and_dimes_have_same_count (q : ℕ) : Prop := (0.25 * q + 0.10 * q = eva_has_us_coins)

theorem greatest_number_of_quarters : ∃ (q : ℕ), quarters_and_dimes_have_same_count q ∧ q = 13 :=
sorry

end NUMINAMATH_GPT_greatest_number_of_quarters_l33_3363


namespace NUMINAMATH_GPT_certain_number_x_l33_3378

theorem certain_number_x :
  ∃ x : ℤ, (287 * 287 + 269 * 269 - x * (287 * 269) = 324) ∧ (x = 2) := 
by {
  use 2,
  sorry
}

end NUMINAMATH_GPT_certain_number_x_l33_3378
