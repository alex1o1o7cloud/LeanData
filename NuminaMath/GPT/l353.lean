import Mathlib

namespace NUMINAMATH_GPT_find_m_l353_35355

noncomputable def f (x m : ℝ) : ℝ := x ^ 2 + m
noncomputable def g (x : ℝ) : ℝ := 6 * Real.log x - 4 * x

theorem find_m (m : ℝ) : 
  ∃ a b : ℝ, (0 < a) ∧ (f a m = b) ∧ (g a = b) ∧ (2 * a = (6 / a) - 4) → m = -5 := 
by
  sorry

end NUMINAMATH_GPT_find_m_l353_35355


namespace NUMINAMATH_GPT_three_digit_multiples_of_3_and_11_l353_35390

theorem three_digit_multiples_of_3_and_11 : 
  ∃ n, n = 27 ∧ ∀ x, 100 ≤ x ∧ x ≤ 999 ∧ x % 33 = 0 ↔ ∃ k, x = 33 * k ∧ 4 ≤ k ∧ k ≤ 30 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_multiples_of_3_and_11_l353_35390


namespace NUMINAMATH_GPT_range_of_a_range_of_m_l353_35386

-- Definition of proposition p: Equation has real roots
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a * x + a + 3 = 0

-- Definition of proposition q: m - 1 <= a <= m + 1
def q (m a : ℝ) : Prop := m - 1 ≤ a ∧ a ≤ m + 1

-- Part (I): Range of a when ¬p is true
theorem range_of_a (a : ℝ) (hp : ¬ p a) : -2 < a ∧ a < 6 :=
sorry

-- Part (II): Range of m when p is a necessary but not sufficient condition for q
theorem range_of_m (m : ℝ) (hnp : ∀ a, q m a → p a) (hns : ∃ a, q m a ∧ ¬p a) : m ≤ -3 ∨ m ≥ 7 :=
sorry

end NUMINAMATH_GPT_range_of_a_range_of_m_l353_35386


namespace NUMINAMATH_GPT_exists_consecutive_numbers_divisible_by_3_5_7_l353_35327

theorem exists_consecutive_numbers_divisible_by_3_5_7 :
  ∃ (a : ℕ), 100 ≤ a ∧ a ≤ 200 ∧
    a % 3 = 0 ∧ (a + 1) % 5 = 0 ∧ (a + 2) % 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_exists_consecutive_numbers_divisible_by_3_5_7_l353_35327


namespace NUMINAMATH_GPT_polynomial_divisibility_l353_35352

theorem polynomial_divisibility (a : ℤ) : 
  (∀x : ℤ, x^2 - x + a ∣ x^13 + x + 94) → a = 2 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l353_35352


namespace NUMINAMATH_GPT_complex_purely_imaginary_a_eq_3_l353_35326

theorem complex_purely_imaginary_a_eq_3 (a : ℝ) :
  (∀ (a : ℝ), (a^2 - 2*a - 3) + (a + 1)*I = 0 + (a + 1)*I → a = 3) :=
by
  sorry

end NUMINAMATH_GPT_complex_purely_imaginary_a_eq_3_l353_35326


namespace NUMINAMATH_GPT_arithmetic_expression_l353_35334

theorem arithmetic_expression : (-9) + 18 + 2 + (-1) = 10 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_expression_l353_35334


namespace NUMINAMATH_GPT_value_of_fraction_l353_35322

variables {a b c : ℝ}

-- Conditions
def quadratic_has_no_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c < 0

def person_A_roots (a' b c : ℝ) : Prop :=
  b = -6 * a' ∧ c = 8 * a'

def person_B_roots (a b' c : ℝ) : Prop :=
  b' = -3 * a ∧ c = -4 * a

-- Proof Statement
theorem value_of_fraction (a b c a' b' : ℝ)
  (hnr : quadratic_has_no_real_roots a b c)
  (hA : person_A_roots a' b c)
  (hB : person_B_roots a b' c) :
  (2 * b + 3 * c) / a = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_fraction_l353_35322


namespace NUMINAMATH_GPT_ratio_four_l353_35305

variable {x y : ℝ}

theorem ratio_four : y = 0.25 * x → x / y = 4 := by
  sorry

end NUMINAMATH_GPT_ratio_four_l353_35305


namespace NUMINAMATH_GPT_magnitude_of_b_l353_35310

variable (a b : ℝ)

-- Defining the given conditions as hypotheses
def condition1 : Prop := (a - b) * (a - b) = 9
def condition2 : Prop := (a + 2 * b) * (a + 2 * b) = 36
def condition3 : Prop := a^2 + (a * b) - 2 * b^2 = -9

-- Defining the theorem to prove
theorem magnitude_of_b (ha : condition1 a b) (hb : condition2 a b) (hc : condition3 a b) : b^2 = 3 := 
sorry

end NUMINAMATH_GPT_magnitude_of_b_l353_35310


namespace NUMINAMATH_GPT_calculate_value_l353_35319

def a : ℤ := 3 * 4 * 5
def b : ℚ := 1/3 + 1/4 + 1/5

theorem calculate_value :
  (a : ℚ) * b = 47 := by
sorry

end NUMINAMATH_GPT_calculate_value_l353_35319


namespace NUMINAMATH_GPT_isosceles_triangle_angle_measure_l353_35377

theorem isosceles_triangle_angle_measure:
  ∀ (α β : ℝ), (α = 112.5) → (2 * β + α = 180) → β = 33.75 :=
by
  intros α β hα h_sum
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angle_measure_l353_35377


namespace NUMINAMATH_GPT_childSupportOwed_l353_35342

def annualIncomeBeforeRaise : ℕ := 30000
def yearsBeforeRaise : ℕ := 3
def raisePercentage : ℕ := 20
def annualIncomeAfterRaise (incomeBeforeRaise raisePercentage : ℕ) : ℕ :=
  incomeBeforeRaise + (incomeBeforeRaise * raisePercentage / 100)
def yearsAfterRaise : ℕ := 4
def childSupportPercentage : ℕ := 30
def amountPaid : ℕ := 1200

def calculateChildSupport (incomeYears : ℕ → ℕ → ℕ) (supportPercentage : ℕ) (years : ℕ) : ℕ :=
  (incomeYears years supportPercentage) * supportPercentage / 100 * years

def totalChildSupportOwed : ℕ :=
  (calculateChildSupport (λ _ _ => annualIncomeBeforeRaise) childSupportPercentage yearsBeforeRaise) +
  (calculateChildSupport (λ _ _ => annualIncomeAfterRaise annualIncomeBeforeRaise raisePercentage) childSupportPercentage yearsAfterRaise)

theorem childSupportOwed : totalChildSupportOwed - amountPaid = 69000 :=
by trivial

end NUMINAMATH_GPT_childSupportOwed_l353_35342


namespace NUMINAMATH_GPT_cube_vertex_plane_distance_l353_35364

theorem cube_vertex_plane_distance
  (d : ℝ)
  (h_dist : d = 9 - Real.sqrt 186)
  (h7 : ∀ (a b c  : ℝ), a^2 + b^2 + c^2 = 1 → 64 * (a^2 + b^2 + c^2) = 64)
  (h8 : ∀ (d : ℝ), 3 * d^2 - 54 * d + 181 = 0) :
  ∃ (p q r : ℕ), 
    p = 27 ∧ q = 186 ∧ r = 3 ∧ (p + q + r < 1000) ∧ (d = (p - Real.sqrt q) / r) := 
  by
    sorry

end NUMINAMATH_GPT_cube_vertex_plane_distance_l353_35364


namespace NUMINAMATH_GPT_combined_weight_of_boxes_l353_35343

-- Defining the weights of each box as constants
def weight1 : ℝ := 2.5
def weight2 : ℝ := 11.3
def weight3 : ℝ := 5.75
def weight4 : ℝ := 7.2
def weight5 : ℝ := 3.25

-- The main theorem statement
theorem combined_weight_of_boxes : weight1 + weight2 + weight3 + weight4 + weight5 = 30 := by
  sorry

end NUMINAMATH_GPT_combined_weight_of_boxes_l353_35343


namespace NUMINAMATH_GPT_pipe_individual_empty_time_l353_35328

variable (a b c : ℝ)

noncomputable def timeToEmptyFirstPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * c + b * c - a * b)

noncomputable def timeToEmptySecondPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * b + b * c - a * c)

noncomputable def timeToEmptyThirdPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * b + a * c - b * c)

theorem pipe_individual_empty_time
  (x y z : ℝ)
  (h1 : 1 / x + 1 / y = 1 / a)
  (h2 : 1 / x + 1 / z = 1 / b)
  (h3 : 1 / y + 1 / z = 1 / c) :
  x = timeToEmptyFirstPipe a b c ∧ y = timeToEmptySecondPipe a b c ∧ z = timeToEmptyThirdPipe a b c :=
sorry

end NUMINAMATH_GPT_pipe_individual_empty_time_l353_35328


namespace NUMINAMATH_GPT_smallest_sum_l353_35373

theorem smallest_sum (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x ≠ y) (eq : 1/x + 1/y = 1/10) : x + y = 45 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_l353_35373


namespace NUMINAMATH_GPT_bertha_descendants_without_daughters_l353_35392

-- Definitions based on conditions
def num_daughters : ℕ := 6
def total_daughters_and_granddaughters : ℕ := 30
def daughters_with_daughters := (total_daughters_and_granddaughters - num_daughters) / 6

-- The number of Bertha's daughters who have no daughters:
def daughters_without_daughters := num_daughters - daughters_with_daughters
-- The number of Bertha's granddaughters:
def num_granddaughters := total_daughters_and_granddaughters - num_daughters
-- All granddaughters have no daughters:
def granddaughters_without_daughters := num_granddaughters

-- The total number of daughters and granddaughters without daughters
def total_without_daughters := daughters_without_daughters + granddaughters_without_daughters

-- Main theorem statement
theorem bertha_descendants_without_daughters :
  total_without_daughters = 26 :=
by
  sorry

end NUMINAMATH_GPT_bertha_descendants_without_daughters_l353_35392


namespace NUMINAMATH_GPT_arithmetic_mean_of_fractions_l353_35311

theorem arithmetic_mean_of_fractions :
  (3 / 8 + 5 / 9 + 7 / 12) / 3 = 109 / 216 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_fractions_l353_35311


namespace NUMINAMATH_GPT_triangle_area_is_15_l353_35360

def Point := (ℝ × ℝ)

def A : Point := (2, 2)
def B : Point := (7, 2)
def C : Point := (4, 8)

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

theorem triangle_area_is_15 : area_of_triangle A B C = 15 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_triangle_area_is_15_l353_35360


namespace NUMINAMATH_GPT_range_of_sum_l353_35368

variable {x y t : ℝ}

theorem range_of_sum :
  (1 = x^2 + 4*y^2 - 2*x*y) ∧ (x < 0) ∧ (y < 0) →
  -2 <= x + 2*y ∧ x + 2*y < 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_sum_l353_35368


namespace NUMINAMATH_GPT_problem_statement_l353_35304

theorem problem_statement 
  (w x y z : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z = 945) :
  2 * w + 3 * x + 5 * y + 7 * z = 21 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l353_35304


namespace NUMINAMATH_GPT_place_signs_correct_l353_35382

theorem place_signs_correct :
  1 * 3 / 3^2 / 3^4 / 3^8 * 3^16 * 3^32 * 3^64 = 3^99 :=
by
  sorry

end NUMINAMATH_GPT_place_signs_correct_l353_35382


namespace NUMINAMATH_GPT_number_of_numbers_is_ten_l353_35306

open Nat

-- Define the conditions as given
variable (n : ℕ) -- Total number of numbers
variable (incorrect_average correct_average incorrect_value correct_value : ℤ)
variable (h1 : incorrect_average = 16)
variable (h2 : correct_average = 17)
variable (h3 : incorrect_value = 25)
variable (h4 : correct_value = 35)

-- Define the proof problem
theorem number_of_numbers_is_ten
  (h1 : incorrect_average = 16)
  (h2 : correct_average = 17)
  (h3 : incorrect_value = 25)
  (h4 : correct_value = 35)
  (h5 : ∀ (x : ℤ), x ≠ incorrect_value → incorrect_average * (n : ℤ) + x = correct_average * (n : ℤ) + correct_value - incorrect_value)
  : n = 10 := 
sorry

end NUMINAMATH_GPT_number_of_numbers_is_ten_l353_35306


namespace NUMINAMATH_GPT_bus_stops_for_28_minutes_per_hour_l353_35300

-- Definitions based on the conditions
def without_stoppages_speed : ℕ := 75
def with_stoppages_speed : ℕ := 40
def speed_difference : ℕ := without_stoppages_speed - with_stoppages_speed

-- Theorem statement
theorem bus_stops_for_28_minutes_per_hour : 
  ∀ (T : ℕ), (T = (speed_difference*60)/(without_stoppages_speed))  → 
  T = 28 := 
by
  sorry

end NUMINAMATH_GPT_bus_stops_for_28_minutes_per_hour_l353_35300


namespace NUMINAMATH_GPT_find_number_l353_35381

theorem find_number (x : ℤ) (h : x = 5 * (x - 4)) : x = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_number_l353_35381


namespace NUMINAMATH_GPT_towels_per_pack_l353_35336

open Nat

-- Define the given conditions
def packs : Nat := 9
def total_towels : Nat := 27

-- Define the property to prove
theorem towels_per_pack : total_towels / packs = 3 := by
  sorry

end NUMINAMATH_GPT_towels_per_pack_l353_35336


namespace NUMINAMATH_GPT_labor_day_to_national_day_l353_35385

theorem labor_day_to_national_day :
  let labor_day := 1 -- Monday is represented as 1
  let factor_31 := 31
  let factor_30 := 30
  let total_days := (factor_31 * 3 + factor_30 * 2)
  (labor_day + total_days % 7) % 7 = 0 := -- Since 0 corresponds to Sunday modulo 7
by
  let labor_day := 1
  let factor_31 := 31
  let factor_30 := 30
  let total_days := (factor_31 * 3 + factor_30 * 2)
  have h1 : (labor_day + total_days % 7) % 7 = ((1 + (31 * 3 + 30 * 2) % 7) % 7) := by rfl
  sorry

end NUMINAMATH_GPT_labor_day_to_national_day_l353_35385


namespace NUMINAMATH_GPT_slope_angle_of_vertical_line_l353_35339

theorem slope_angle_of_vertical_line :
  ∀ {θ : ℝ}, (∀ x, (x = 3) → x = 3) → θ = 90 := by
  sorry

end NUMINAMATH_GPT_slope_angle_of_vertical_line_l353_35339


namespace NUMINAMATH_GPT_lamplighter_monkey_distance_traveled_l353_35318

-- Define the parameters
def running_speed : ℕ := 15
def running_time : ℕ := 5
def swinging_speed : ℕ := 10
def swinging_time : ℕ := 10

-- Define the proof statement
theorem lamplighter_monkey_distance_traveled :
  (running_speed * running_time) + (swinging_speed * swinging_time) = 175 := by
  sorry

end NUMINAMATH_GPT_lamplighter_monkey_distance_traveled_l353_35318


namespace NUMINAMATH_GPT_obtuse_angle_probability_l353_35324

noncomputable def probability_obtuse_angle : ℝ :=
  let F : ℝ × ℝ := (0, 3)
  let G : ℝ × ℝ := (5, 0)
  let H : ℝ × ℝ := (2 * Real.pi + 2, 0)
  let I : ℝ × ℝ := (2 * Real.pi + 2, 3)
  let rectangle_area : ℝ := (2 * Real.pi + 2) * 3
  let semicircle_radius : ℝ := Real.sqrt (2.5^2 + 1.5^2)
  let semicircle_area : ℝ := (1 / 2) * Real.pi * semicircle_radius^2
  semicircle_area / rectangle_area

theorem obtuse_angle_probability :
  probability_obtuse_angle = 17 / (24 + 4 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_obtuse_angle_probability_l353_35324


namespace NUMINAMATH_GPT_inequality_proof_l353_35345

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b + b^2 / c + c^2 / a) + (a + b + c) ≥ (6 * (a^2 + b^2 + c^2) / (a + b + c)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l353_35345


namespace NUMINAMATH_GPT_will_initial_money_l353_35398

theorem will_initial_money (spent_game : ℕ) (number_of_toys : ℕ) (cost_per_toy : ℕ) (initial_money : ℕ) :
  spent_game = 27 →
  number_of_toys = 5 →
  cost_per_toy = 6 →
  initial_money = spent_game + number_of_toys * cost_per_toy →
  initial_money = 57 :=
by
  intros
  sorry

end NUMINAMATH_GPT_will_initial_money_l353_35398


namespace NUMINAMATH_GPT_impossible_coins_l353_35391

theorem impossible_coins (p_1 p_2 : ℝ) 
  (h1 : (1 - p_1) * (1 - p_2) = p_1 * p_2)
  (h2 : p_1 * (1 - p_2) + p_2 * (1 - p_1) = p_1 * p_2) : False := 
sorry

end NUMINAMATH_GPT_impossible_coins_l353_35391


namespace NUMINAMATH_GPT_number_of_real_roots_l353_35366

theorem number_of_real_roots (a : ℝ) :
  (|a| < (2 * Real.sqrt 3 / 9) → ∃ x y z : ℝ, x^3 - x - a = 0 ∧ y^3 - y - a = 0 ∧ z^3 - z - a = 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧
  (|a| = (2 * Real.sqrt 3 / 9) → ∃ x y : ℝ, x^3 - x - a = 0 ∧ y^3 - y - a = 0 ∧ x = y) ∧
  (|a| > (2 * Real.sqrt 3 / 9) → ∃ x : ℝ, x^3 - x - a = 0 ∧ ∀ y : ℝ, y^3 - y - a ≠ 0 ∨ y = x) :=
sorry

end NUMINAMATH_GPT_number_of_real_roots_l353_35366


namespace NUMINAMATH_GPT_number_of_ways_to_choose_one_book_is_correct_l353_35399

-- Definitions of the given problem conditions
def number_of_chinese_books : Nat := 10
def number_of_english_books : Nat := 7
def number_of_math_books : Nat := 5

-- Theorem stating the proof problem
theorem number_of_ways_to_choose_one_book_is_correct : 
  number_of_chinese_books + number_of_english_books + number_of_math_books = 22 := by
  -- This proof is left as an exercise.
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_one_book_is_correct_l353_35399


namespace NUMINAMATH_GPT_geom_seq_sum_l353_35333

theorem geom_seq_sum (a : ℕ → ℝ) (n : ℕ) (q : ℝ) (h1 : a 1 = 2) (h2 : a 1 * a 5 = 64) :
  (a 1 * (1 - q^n)) / (1 - q) = 2^(n+1) - 2 := 
sorry

end NUMINAMATH_GPT_geom_seq_sum_l353_35333


namespace NUMINAMATH_GPT_find_value_of_x_plus_5_l353_35379

-- Define a variable x
variable (x : ℕ)

-- Define the condition given in the problem
def condition := x - 10 = 15

-- The statement we need to prove
theorem find_value_of_x_plus_5 (h : x - 10 = 15) : x + 5 = 30 := 
by sorry

end NUMINAMATH_GPT_find_value_of_x_plus_5_l353_35379


namespace NUMINAMATH_GPT_kickball_students_l353_35357

theorem kickball_students (w t : ℕ) (hw : w = 37) (ht : t = w - 9) : w + t = 65 :=
by
  sorry

end NUMINAMATH_GPT_kickball_students_l353_35357


namespace NUMINAMATH_GPT_find_m_value_l353_35354

noncomputable def x0 : ℝ := sorry

noncomputable def m : ℝ := x0^3 + 2 * x0^2 + 2

theorem find_m_value :
  (x0^2 + x0 - 1 = 0) → (m = 3) :=
by
  intro h
  have hx : x0 = sorry := sorry
  have hm : m = x0 ^ 3 + 2 * x0^2 + 2 := rfl
  rw [hx] at hm
  sorry

end NUMINAMATH_GPT_find_m_value_l353_35354


namespace NUMINAMATH_GPT_paper_area_difference_l353_35370

def area (length width : ℕ) : ℕ := length * width

def combined_area (length width : ℕ) : ℕ := 2 * (area length width)

def sq_inch_to_sq_ft (sq_inch : ℕ) : ℕ := sq_inch / 144

theorem paper_area_difference :
  sq_inch_to_sq_ft (combined_area 15 24 - combined_area 12 18) = 2 :=
by
  sorry

end NUMINAMATH_GPT_paper_area_difference_l353_35370


namespace NUMINAMATH_GPT_inequality_proof_l353_35389

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a^3 + b^3 + c^3 = 3) :
  (1 / (a^4 + 3) + 1 / (b^4 + 3) + 1 / (c^4 + 3) >= 3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l353_35389


namespace NUMINAMATH_GPT_cinema_total_cost_l353_35338

theorem cinema_total_cost 
  (total_students : ℕ)
  (ticket_cost : ℕ)
  (half_price_interval : ℕ)
  (free_interval : ℕ)
  (half_price_cost : ℕ)
  (free_cost : ℕ)
  (total_cost : ℕ)
  (H_total_students : total_students = 84)
  (H_ticket_cost : ticket_cost = 50)
  (H_half_price_interval : half_price_interval = 12)
  (H_free_interval : free_interval = 35)
  (H_half_price_cost : half_price_cost = ticket_cost / 2)
  (H_free_cost : free_cost = 0)
  (H_total_cost : total_cost = 3925) :
  total_cost = ((total_students / half_price_interval) * half_price_cost +
                (total_students / free_interval) * free_cost +
                (total_students - (total_students / half_price_interval + total_students / free_interval)) * ticket_cost) :=
by 
  sorry

end NUMINAMATH_GPT_cinema_total_cost_l353_35338


namespace NUMINAMATH_GPT_ratio_of_men_to_women_l353_35309

theorem ratio_of_men_to_women
  (M W : ℕ)
  (h1 : W = M + 6)
  (h2 : M + W = 16) :
  M * 11 = 5 * W :=
by
    -- We can explicitly construct the necessary proof here, but according to instructions we add sorry to bypass for now
    sorry

end NUMINAMATH_GPT_ratio_of_men_to_women_l353_35309


namespace NUMINAMATH_GPT_striped_octopus_has_8_legs_l353_35367

-- Definitions for Octopus and Statements
structure Octopus :=
  (legs : ℕ)
  (tellsTruth : Prop)

-- Given conditions translations
def tellsTruthCondition (o : Octopus) : Prop :=
  if o.legs % 2 = 0 then o.tellsTruth else ¬o.tellsTruth

def green_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

def dark_blue_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

def violet_octopus : Octopus :=
  { legs := 9, tellsTruth := sorry }  -- Placeholder truth value

def striped_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

-- Octopus statements (simplified for output purposes)
def green_statement := (green_octopus.legs = 8) ∧ (dark_blue_octopus.legs = 6)
def dark_blue_statement := (dark_blue_octopus.legs = 8) ∧ (green_octopus.legs = 7)
def violet_statement := (dark_blue_octopus.legs = 8) ∧ (violet_octopus.legs = 9)
def striped_statement := ¬(green_octopus.legs = 8 ∨ dark_blue_octopus.legs = 8 ∨ violet_octopus.legs = 8) ∧ (striped_octopus.legs = 8)

-- The goal to prove that the striped octopus has exactly 8 legs
theorem striped_octopus_has_8_legs : striped_octopus.legs = 8 :=
sorry

end NUMINAMATH_GPT_striped_octopus_has_8_legs_l353_35367


namespace NUMINAMATH_GPT_identity_holds_l353_35394

theorem identity_holds (x : ℝ) : 
  (2 * x - 1) ^ 3 = 5 * x ^ 3 + (3 * x + 1) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x :=
by sorry

end NUMINAMATH_GPT_identity_holds_l353_35394


namespace NUMINAMATH_GPT_geometric_sequence_sum_l353_35363

variable {α : Type*} [NormedField α] [CompleteSpace α]

def geometric_sum (a r : α) (n : ℕ) : α :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (S : ℕ → α) (a r : α) (hS : ∀ n, S n = geometric_sum a r n) :
  S 2 = 6 → S 4 = 30 → S 6 = 126 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l353_35363


namespace NUMINAMATH_GPT_nine_wolves_nine_sheep_seven_days_l353_35312

theorem nine_wolves_nine_sheep_seven_days
    (wolves_sheep_seven_days : ∀ {n : ℕ}, 7 * n / 7 = n) :
    9 * 9 / 9 = 7 := by
  sorry

end NUMINAMATH_GPT_nine_wolves_nine_sheep_seven_days_l353_35312


namespace NUMINAMATH_GPT_christopher_avg_speed_l353_35331

-- Definition of a palindrome (not required for this proof, but helpful for context)
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Given conditions
def initial_reading : ℕ := 12321
def final_reading : ℕ := 12421
def duration : ℕ := 4

-- Definition of average speed calculation
def average_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- Main theorem to prove
theorem christopher_avg_speed : average_speed (final_reading - initial_reading) duration = 25 :=
by
  sorry

end NUMINAMATH_GPT_christopher_avg_speed_l353_35331


namespace NUMINAMATH_GPT_system_solution_l353_35307

theorem system_solution :
  (∀ x y : ℝ, (2 * x + 3 * y = 19) ∧ (3 * x + 4 * y = 26) → x = 2 ∧ y = 5) →
  (∃ x y : ℝ, (2 * (2 * x + 4) + 3 * (y + 3) = 19) ∧ (3 * (2 * x + 4) + 4 * (y + 3) = 26) ∧ x = -1 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_GPT_system_solution_l353_35307


namespace NUMINAMATH_GPT_floor_width_l353_35308

theorem floor_width (tile_length tile_width floor_length max_tiles : ℕ) (h1 : tile_length = 25) (h2 : tile_width = 65) (h3 : floor_length = 150) (h4 : max_tiles = 36) :
  ∃ floor_width : ℕ, floor_width = 450 :=
by
  sorry

end NUMINAMATH_GPT_floor_width_l353_35308


namespace NUMINAMATH_GPT_lcm_of_two_numbers_l353_35383

theorem lcm_of_two_numbers (A B : ℕ) 
  (h_prod : A * B = 987153000) 
  (h_hcf : Int.gcd A B = 440) : 
  Nat.lcm A B = 2243525 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_two_numbers_l353_35383


namespace NUMINAMATH_GPT_rope_total_in_inches_l353_35351

theorem rope_total_in_inches (feet_last_week feet_less_this_week feet_to_inch : ℕ) 
  (h1 : feet_last_week = 6)
  (h2 : feet_less_this_week = 4)
  (h3 : feet_to_inch = 12) :
  (feet_last_week + (feet_last_week - feet_less_this_week)) * feet_to_inch = 96 :=
by
  sorry

end NUMINAMATH_GPT_rope_total_in_inches_l353_35351


namespace NUMINAMATH_GPT_consecutive_integers_product_sum_l353_35337

theorem consecutive_integers_product_sum (a b c d : ℕ) :
  a * b * c * d = 3024 ∧ b = a + 1 ∧ c = b + 1 ∧ d = c + 1 → a + b + c + d = 30 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_integers_product_sum_l353_35337


namespace NUMINAMATH_GPT_find_painted_stencils_l353_35353

variable (hourly_wage racquet_wage grommet_wage stencil_wage total_earnings hours_worked racquets_restrung grommets_changed : ℕ)
variable (painted_stencils: ℕ)

axiom condition_hourly_wage : hourly_wage = 9
axiom condition_racquet_wage : racquet_wage = 15
axiom condition_grommet_wage : grommet_wage = 10
axiom condition_stencil_wage : stencil_wage = 1
axiom condition_total_earnings : total_earnings = 202
axiom condition_hours_worked : hours_worked = 8
axiom condition_racquets_restrung : racquets_restrung = 7
axiom condition_grommets_changed : grommets_changed = 2

theorem find_painted_stencils :
  painted_stencils = 5 :=
by
  -- Given:
  -- hourly_wage = 9
  -- racquet_wage = 15
  -- grommet_wage = 10
  -- stencil_wage = 1
  -- total_earnings = 202
  -- hours_worked = 8
  -- racquets_restrung = 7
  -- grommets_changed = 2

  -- We need to prove:
  -- painted_stencils = 5
  
  sorry

end NUMINAMATH_GPT_find_painted_stencils_l353_35353


namespace NUMINAMATH_GPT_vera_first_place_l353_35362

noncomputable def placement (anna vera katya natasha : ℕ) : Prop :=
  (anna ≠ 1 ∧ anna ≠ 4) ∧ (vera ≠ 4) ∧ (katya = 1) ∧ (natasha = 4)

theorem vera_first_place :
  ∃ (anna vera katya natasha : ℕ),
    (placement anna vera katya natasha) ∧ 
    (vera = 1) ∧ 
    (1 ≠ 4) → 
    ((anna ≠ 1 ∧ anna ≠ 4) ∧ (vera ≠ 4) ∧ (katya = 1) ∧ (natasha = 4)) ∧ 
    (1 = 1) ∧ 
    (∃ i j k l : ℕ, (i ≠ 1 ∧ i ≠ 4) ∧ (j = 1) ∧ (k ≠ 1) ∧ (l = 4)) ∧ 
    (vera = 1) :=
sorry

end NUMINAMATH_GPT_vera_first_place_l353_35362


namespace NUMINAMATH_GPT_keith_remaining_cards_l353_35371

-- Definitions and conditions
def initial_cards := 0
def new_cards := 8
def total_cards_after_purchase := initial_cards + new_cards
def remaining_cards := total_cards_after_purchase / 2

-- Proof statement (in Lean, the following would be a theorem)
theorem keith_remaining_cards : remaining_cards = 4 := sorry

end NUMINAMATH_GPT_keith_remaining_cards_l353_35371


namespace NUMINAMATH_GPT_ryan_lost_initially_l353_35380

-- Define the number of leaves initially collected
def initial_leaves : ℤ := 89

-- Define the number of leaves broken afterwards
def broken_leaves : ℤ := 43

-- Define the number of leaves left in the collection
def remaining_leaves : ℤ := 22

-- Define the lost leaves
def lost_leaves (L : ℤ) : Prop :=
  initial_leaves - L - broken_leaves = remaining_leaves

theorem ryan_lost_initially : ∃ L : ℤ, lost_leaves L ∧ L = 24 :=
by
  sorry

end NUMINAMATH_GPT_ryan_lost_initially_l353_35380


namespace NUMINAMATH_GPT_relationship_among_y_values_l353_35372

theorem relationship_among_y_values (c y1 y2 y3 : ℝ) :
  (-1)^2 - 2 * (-1) + c = y1 →
  (3)^2 - 2 * 3 + c = y2 →
  (5)^2 - 2 * 5 + c = y3 →
  y1 = y2 ∧ y2 > y3 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_relationship_among_y_values_l353_35372


namespace NUMINAMATH_GPT_daves_apps_count_l353_35384

theorem daves_apps_count (x : ℕ) : 
  let initial_apps : ℕ := 21
  let added_apps : ℕ := 89
  let total_apps : ℕ := initial_apps + added_apps
  let deleted_apps : ℕ := x
  let more_added_apps : ℕ := x + 3
  total_apps - deleted_apps + more_added_apps = 113 :=
by
  sorry

end NUMINAMATH_GPT_daves_apps_count_l353_35384


namespace NUMINAMATH_GPT_total_cost_of_shirt_and_coat_l353_35321

-- Definition of the conditions
def shirt_cost : ℕ := 150
def one_third_of_coat (coat_cost: ℕ) : Prop := shirt_cost = coat_cost / 3

-- Theorem stating the problem to prove
theorem total_cost_of_shirt_and_coat (coat_cost : ℕ) (h : one_third_of_coat coat_cost) : shirt_cost + coat_cost = 600 :=
by 
  -- Proof goes here, using sorry as placeholder
  sorry

end NUMINAMATH_GPT_total_cost_of_shirt_and_coat_l353_35321


namespace NUMINAMATH_GPT_sequence_nth_term_l353_35323

/-- The nth term of the sequence {a_n} defined by a_1 = 1 and
    the recurrence relation a_{n+1} = 2a_n + 2 for all n ∈ ℕ*,
    is given by the formula a_n = 3 * 2 ^ (n - 1) - 2. -/
theorem sequence_nth_term (n : ℕ) (h : n > 0) : 
  ∃ (a : ℕ → ℤ), a 1 = 1 ∧ (∀ n > 0, a (n + 1) = 2 * a n + 2) ∧ a n = 3 * 2 ^ (n - 1) - 2 :=
  sorry

end NUMINAMATH_GPT_sequence_nth_term_l353_35323


namespace NUMINAMATH_GPT_find_s_at_1_l353_35376

variable (t s : ℝ → ℝ)
variable (x : ℝ)

-- Define conditions
def t_def : t x = 4 * x - 9 := by sorry

def s_def : s (t x) = x^2 + 4 * x - 5 := by sorry

-- Prove the question
theorem find_s_at_1 : s 1 = 11.25 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_s_at_1_l353_35376


namespace NUMINAMATH_GPT_sum_of_natural_numbers_l353_35332

theorem sum_of_natural_numbers :
  36 + 17 + 32 + 54 + 28 + 3 = 170 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_natural_numbers_l353_35332


namespace NUMINAMATH_GPT_quadratic_root_value_m_l353_35320

theorem quadratic_root_value_m (m : ℝ) : ∃ x, x = 1 ∧ x^2 + x - m = 0 → m = 2 := by
  sorry

end NUMINAMATH_GPT_quadratic_root_value_m_l353_35320


namespace NUMINAMATH_GPT_pirate_coins_total_l353_35350

def total_coins (y : ℕ) := 6 * y

theorem pirate_coins_total : 
  (∃ y : ℕ, y ≠ 0 ∧ y * (y + 1) / 2 = 5 * y) →
  total_coins 9 = 54 :=
by
  sorry

end NUMINAMATH_GPT_pirate_coins_total_l353_35350


namespace NUMINAMATH_GPT_race_head_start_l353_35330

theorem race_head_start (v_A v_B : ℕ) (h : v_A = 4 * v_B) (d : ℕ) : 
  100 / v_A = (100 - d) / v_B → d = 75 :=
by
  sorry

end NUMINAMATH_GPT_race_head_start_l353_35330


namespace NUMINAMATH_GPT_Petya_receives_last_wrapper_l353_35387

variable (a b c : ℝ)
variable (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem Petya_receives_last_wrapper
  (h1 : discriminant a b c ≥ 0)
  (h2 : discriminant a c b ≥ 0)
  (h3 : discriminant b a c ≥ 0)
  (h4 : discriminant c a b < 0)
  (h5 : discriminant b c a < 0) :
  discriminant c b a ≥ 0 :=
sorry

end NUMINAMATH_GPT_Petya_receives_last_wrapper_l353_35387


namespace NUMINAMATH_GPT_approx_cube_of_331_l353_35340

noncomputable def cube (x : ℝ) : ℝ := x * x * x

theorem approx_cube_of_331 : 
  ∃ ε > 0, abs (cube 0.331 - 0.037) < ε :=
by
  sorry

end NUMINAMATH_GPT_approx_cube_of_331_l353_35340


namespace NUMINAMATH_GPT_circle_through_points_l353_35346

-- Definitions of the points
def O : (ℝ × ℝ) := (0, 0)
def M1 : (ℝ × ℝ) := (1, 1)
def M2 : (ℝ × ℝ) := (4, 2)

-- Definition of the center and radius of the circle
def center : (ℝ × ℝ) := (4, -3)
def radius : ℝ := 5

-- The circle equation function
def circle_eq (x y : ℝ) (c : ℝ × ℝ) (r : ℝ) : Prop :=
  (x - c.1)^2 + (y + c.2)^2 = r^2

theorem circle_through_points :
  circle_eq 0 0 center radius ∧ circle_eq 1 1 center radius ∧ circle_eq 4 2 center radius :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_circle_through_points_l353_35346


namespace NUMINAMATH_GPT_youngest_child_age_l353_35397

theorem youngest_child_age {x : ℝ} (h : x + (x + 1) + (x + 2) + (x + 3) = 12) : x = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_youngest_child_age_l353_35397


namespace NUMINAMATH_GPT_intersection_M_complement_N_l353_35313

noncomputable def M := {y : ℝ | 1 ≤ y ∧ y ≤ 2}
noncomputable def N_complement := {x : ℝ | 1 ≤ x}

theorem intersection_M_complement_N : M ∩ N_complement = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end NUMINAMATH_GPT_intersection_M_complement_N_l353_35313


namespace NUMINAMATH_GPT_minimum_value_of_f_minimum_value_achieved_sum_of_squares_ge_three_l353_35375

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 3 := by
  sorry

theorem minimum_value_achieved : ∃ x : ℝ, f x = 3 := by
  sorry

theorem sum_of_squares_ge_three (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (h : p + q + r = 3) : p^2 + q^2 + r^2 ≥ 3 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_minimum_value_achieved_sum_of_squares_ge_three_l353_35375


namespace NUMINAMATH_GPT_probability_of_successful_meeting_l353_35359

noncomputable def successful_meeting_probability : ℝ :=
  let volume_hypercube := 16.0
  let volume_pyramid := (1.0/3.0) * 2.0^3 * 2.0
  let volume_reduced_base := volume_pyramid / 4.0
  let successful_meeting_volume := volume_reduced_base
  successful_meeting_volume / volume_hypercube

theorem probability_of_successful_meeting : successful_meeting_probability = 1 / 12 :=
  sorry

end NUMINAMATH_GPT_probability_of_successful_meeting_l353_35359


namespace NUMINAMATH_GPT_cory_can_eat_fruits_in_105_ways_l353_35361

-- Define the number of apples, oranges, and bananas Cory has
def apples := 4
def oranges := 1
def bananas := 2

-- Define the total number of fruits Cory has
def total_fruits := apples + oranges + bananas

-- Calculate the number of distinct orders in which Cory can eat the fruits
theorem cory_can_eat_fruits_in_105_ways :
  (Nat.factorial total_fruits) / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas) = 105 :=
by
  -- Provide a sorry to skip the proof
  sorry

end NUMINAMATH_GPT_cory_can_eat_fruits_in_105_ways_l353_35361


namespace NUMINAMATH_GPT_lines_perpendicular_l353_35388

theorem lines_perpendicular
  (k₁ k₂ : ℝ)
  (h₁ : k₁^2 - 3*k₁ - 1 = 0)
  (h₂ : k₂^2 - 3*k₂ - 1 = 0) :
  k₁ * k₂ = -1 → 
  (∃ l₁ l₂: ℝ → ℝ, 
    ∀ x, l₁ x = k₁ * x ∧ l₂ x = k₂ * x → 
    ∃ m, m = -1) := 
sorry

end NUMINAMATH_GPT_lines_perpendicular_l353_35388


namespace NUMINAMATH_GPT_necessary_condition_l353_35358

theorem necessary_condition (m : ℝ) (h : ∀ x : ℝ, x^2 - x + m > 0) : m > 0 := 
sorry

end NUMINAMATH_GPT_necessary_condition_l353_35358


namespace NUMINAMATH_GPT_fractions_sum_correct_l353_35369

noncomputable def fractions_sum : ℝ := (3 / 20) + (5 / 200) + (7 / 2000) + 5

theorem fractions_sum_correct : fractions_sum = 5.1785 :=
by
  sorry

end NUMINAMATH_GPT_fractions_sum_correct_l353_35369


namespace NUMINAMATH_GPT_surface_area_of_glued_cubes_l353_35349

noncomputable def calculate_surface_area (large_cube_edge_length : ℕ) : ℕ :=
sorry

theorem surface_area_of_glued_cubes :
  calculate_surface_area 4 = 136 :=
sorry

end NUMINAMATH_GPT_surface_area_of_glued_cubes_l353_35349


namespace NUMINAMATH_GPT_inequality_proof_l353_35301

theorem inequality_proof (a b : ℝ) (h : a + b > 0) :
  (a / (b^2) + b / (a^2) ≥ 1 / a + 1 / b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l353_35301


namespace NUMINAMATH_GPT_tank_capacity_l353_35374

theorem tank_capacity :
  (∃ c: ℝ, (∃ w: ℝ, w / c = 1/6 ∧ (w + 5) / c = 1/3) → c = 30) :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l353_35374


namespace NUMINAMATH_GPT_positive_value_of_A_l353_35325

-- Define the relation
def hash (A B : ℝ) : ℝ := A^2 - B^2

-- State the main theorem
theorem positive_value_of_A (A : ℝ) : hash A 7 = 72 → A = 11 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_positive_value_of_A_l353_35325


namespace NUMINAMATH_GPT_expand_expression_l353_35347

variable (x y : ℝ)

theorem expand_expression :
  ((6 * x + 8 - 3 * y) * (4 * x - 5 * y)) = 
  (24 * x^2 - 42 * x * y + 32 * x - 40 * y + 15 * y^2) :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l353_35347


namespace NUMINAMATH_GPT_find_sin_θ_find_cos_2θ_find_cos_φ_l353_35393

noncomputable def θ : ℝ := sorry
noncomputable def φ : ℝ := sorry

-- Conditions
axiom cos_eq : Real.cos θ = Real.sqrt 5 / 5
axiom θ_in_quadrant_I : 0 < θ ∧ θ < Real.pi / 2
axiom sin_diff_eq : Real.sin (θ - φ) = Real.sqrt 10 / 10
axiom φ_in_quadrant_I : 0 < φ ∧ φ < Real.pi / 2

-- Goals
-- Part (I) Prove the value of sin θ
theorem find_sin_θ : Real.sin θ = 2 * Real.sqrt 5 / 5 :=
by
  sorry

-- Part (II) Prove the value of cos 2θ
theorem find_cos_2θ : Real.cos (2 * θ) = -3 / 5 :=
by
  sorry

-- Part (III) Prove the value of cos φ
theorem find_cos_φ : Real.cos φ = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_sin_θ_find_cos_2θ_find_cos_φ_l353_35393


namespace NUMINAMATH_GPT_booksJuly_l353_35314

-- Definitions of the conditions
def booksMay : ℕ := 2
def booksJune : ℕ := 6
def booksTotal : ℕ := 18

-- Theorem statement proving how many books Tom read in July
theorem booksJuly : (booksTotal - (booksMay + booksJune)) = 10 :=
by
  sorry

end NUMINAMATH_GPT_booksJuly_l353_35314


namespace NUMINAMATH_GPT_max_intersections_intersections_ge_n_special_case_l353_35302

variable {n m : ℕ}

-- Conditions: n points on a circumference, m and n are positive integers, relatively prime, 6 ≤ 2m < n
def valid_conditions (n m : ℕ) : Prop := Nat.gcd m n = 1 ∧ 6 ≤ 2 * m ∧ 2 * m < n

-- Maximum intersections I = (m-1)n
theorem max_intersections (h : valid_conditions n m) : ∃ I, I = (m - 1) * n :=
by
  sorry

-- Prove I ≥ n
theorem intersections_ge_n (h : valid_conditions n m) : ∃ I, I ≥ n :=
by
  sorry

-- Special case: m = 3 and n is even
theorem special_case (h : valid_conditions n 3) (hn : Even n) : ∃ I, I = n :=
by
  sorry

end NUMINAMATH_GPT_max_intersections_intersections_ge_n_special_case_l353_35302


namespace NUMINAMATH_GPT_number_added_is_59_l353_35316

theorem number_added_is_59 (x : ℤ) (h1 : -2 < 0) (h2 : -3 < 0) (h3 : -2 * -3 + x = 65) : x = 59 :=
by sorry

end NUMINAMATH_GPT_number_added_is_59_l353_35316


namespace NUMINAMATH_GPT_vector_x_solution_l353_35396

theorem vector_x_solution (x : ℝ) (a b c : ℝ × ℝ)
  (ha : a = (-2,0))
  (hb : b = (2,1))
  (hc : c = (x,1))
  (collinear : ∃ k : ℝ, 3 • a + b = k • c) :
  x = -4 :=
by
  sorry

end NUMINAMATH_GPT_vector_x_solution_l353_35396


namespace NUMINAMATH_GPT_unique_solution_m_n_eq_l353_35315

theorem unique_solution_m_n_eq (m n : ℕ) (h : m^2 = (10 * n + 1) * n + 2) : (m, n) = (11, 7) := by
  sorry

end NUMINAMATH_GPT_unique_solution_m_n_eq_l353_35315


namespace NUMINAMATH_GPT_smallest_x_satisfies_abs_eq_l353_35344

theorem smallest_x_satisfies_abs_eq (x : ℝ) :
  (|2 * x + 5| = 21) → (x = -13) :=
sorry

end NUMINAMATH_GPT_smallest_x_satisfies_abs_eq_l353_35344


namespace NUMINAMATH_GPT_correct_probability_statement_l353_35335

-- Define the conditions
def impossible_event_has_no_probability : Prop := ∀ (P : ℝ), P < 0 ∨ P > 0
def every_event_has_probability : Prop := ∀ (P : ℝ), 0 ≤ P ∧ P ≤ 1
def not_all_random_events_have_probability : Prop := ∃ (P : ℝ), P < 0 ∨ P > 1
def certain_events_do_not_have_probability : Prop := (∀ (P : ℝ), P ≠ 1)

-- The main theorem asserting that every event has a probability
theorem correct_probability_statement : every_event_has_probability :=
by sorry

end NUMINAMATH_GPT_correct_probability_statement_l353_35335


namespace NUMINAMATH_GPT_work_time_l353_35365

-- Definitions and conditions
variables (A B C D h : ℝ)
variable (h_def : ℝ := 1 / (1 / A + 1 / B + 1 / D))

-- Conditions
axiom cond1 : 1 / A + 1 / B + 1 / C + 1 / D = 1 / (A - 8)
axiom cond2 : 1 / A + 1 / B + 1 / C + 1 / D = 1 / (B - 2)
axiom cond3 : 1 / A + 1 / B + 1 / C + 1 / D = 3 / C
axiom cond4 : 1 / A + 1 / B + 1 / D = 2 / C

-- The statement to prove
theorem work_time : h_def = 16 / 11 := by
  sorry

end NUMINAMATH_GPT_work_time_l353_35365


namespace NUMINAMATH_GPT_theta_in_second_quadrant_l353_35341

open Real

-- Definitions for conditions
def cond1 (θ : ℝ) : Prop := sin θ > cos θ
def cond2 (θ : ℝ) : Prop := tan θ < 0

-- Main theorem statement
theorem theta_in_second_quadrant (θ : ℝ) 
  (h1 : cond1 θ) 
  (h2 : cond2 θ) : 
  θ > π/2 ∧ θ < π :=
sorry

end NUMINAMATH_GPT_theta_in_second_quadrant_l353_35341


namespace NUMINAMATH_GPT_gcd_of_product_diff_is_12_l353_35378

theorem gcd_of_product_diff_is_12
  (a b c d : ℤ) : ∃ (D : ℤ), D = 12 ∧
  ∀ (a b c d : ℤ), D ∣ (b - a) * (c - b) * (d - c) * (d - a) * (c - a) * (d - b) :=
by
  use 12
  sorry

end NUMINAMATH_GPT_gcd_of_product_diff_is_12_l353_35378


namespace NUMINAMATH_GPT_cosine_of_third_angle_l353_35348

theorem cosine_of_third_angle 
  (α β γ : ℝ) 
  (h1 : α < 40 * Real.pi / 180) 
  (h2 : β < 80 * Real.pi / 180) 
  (h3 : Real.sin γ = 5 / 8) :
  Real.cos γ = -Real.sqrt 39 / 8 := 
sorry

end NUMINAMATH_GPT_cosine_of_third_angle_l353_35348


namespace NUMINAMATH_GPT_certain_number_exists_l353_35395

theorem certain_number_exists (a b : ℝ) (C : ℝ) (h1 : a ≠ b) (h2 : a + b = 4) (h3 : a * (a - 4) = C) (h4 : b * (b - 4) = C) : 
  C = -3 := 
sorry

end NUMINAMATH_GPT_certain_number_exists_l353_35395


namespace NUMINAMATH_GPT_triangle_area_l353_35329

/-- Proof that the area of a triangle with side lengths 9 cm, 40 cm, and 41 cm is 180 square centimeters, 
    given that these lengths form a right triangle. -/
theorem triangle_area : ∀ (a b c : ℕ), a = 9 → b = 40 → c = 41 → a^2 + b^2 = c^2 → (a * b) / 2 = 180 := by
  intros a b c ha hb hc hpyth
  sorry

end NUMINAMATH_GPT_triangle_area_l353_35329


namespace NUMINAMATH_GPT_petes_original_number_l353_35317

theorem petes_original_number (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = y - 5) (h3 : 3 * z = 96) :
  x = 12.33 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_petes_original_number_l353_35317


namespace NUMINAMATH_GPT_probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l353_35356

noncomputable def probability_k_gnomes_fall (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  p * (1 - p) ^ (n - k)

noncomputable def expected_number_of_fallen_gnomes (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p)

theorem probability_k_gnomes_fall_correct (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  probability_k_gnomes_fall n k p h = p * (1 - p) ^ (n - k) :=
by sorry

theorem expected_number_of_fallen_gnomes_correct (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  expected_number_of_fallen_gnomes n p h = n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p) :=
by sorry

end NUMINAMATH_GPT_probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l353_35356


namespace NUMINAMATH_GPT_least_positive_three_digit_multiple_of_7_l353_35303

theorem least_positive_three_digit_multiple_of_7 : ∃ n : ℕ, n % 7 = 0 ∧ n ≥ 100 ∧ n < 1000 ∧ ∀ m : ℕ, (m % 7 = 0 ∧ m ≥ 100 ∧ m < 1000) → n ≤ m := 
by
  sorry

end NUMINAMATH_GPT_least_positive_three_digit_multiple_of_7_l353_35303
