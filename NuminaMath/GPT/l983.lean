import Mathlib

namespace NUMINAMATH_GPT_square_side_length_l983_98384

theorem square_side_length (x S : ℕ) (h1 : S > 0) (h2 : x = 4) (h3 : 4 * S = 6 * x) : S = 6 := by
  subst h2
  sorry

end NUMINAMATH_GPT_square_side_length_l983_98384


namespace NUMINAMATH_GPT_eighteen_gon_vertex_number_l983_98330

theorem eighteen_gon_vertex_number (a b : ℕ) (P : ℕ) (h₁ : a = 20) (h₂ : b = 18) (h₃ : P = a + b) : P = 38 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end NUMINAMATH_GPT_eighteen_gon_vertex_number_l983_98330


namespace NUMINAMATH_GPT_iggy_pace_l983_98398

theorem iggy_pace 
  (monday_miles : ℕ) (tuesday_miles : ℕ) (wednesday_miles : ℕ)
  (thursday_miles : ℕ) (friday_miles : ℕ) (total_hours : ℕ) 
  (h1 : monday_miles = 3) (h2 : tuesday_miles = 4) 
  (h3 : wednesday_miles = 6) (h4 : thursday_miles = 8) 
  (h5 : friday_miles = 3) (h6 : total_hours = 4) :
  (total_hours * 60) / (monday_miles + tuesday_miles + wednesday_miles + thursday_miles + friday_miles) = 10 :=
sorry

end NUMINAMATH_GPT_iggy_pace_l983_98398


namespace NUMINAMATH_GPT_sum_of_prime_factors_1729728_l983_98389

def prime_factors_sum (n : ℕ) : ℕ := 
  -- Suppose that a function defined to calculate the sum of distinct prime factors
  -- In a practical setting, you would define this function or use an existing library
  sorry 

theorem sum_of_prime_factors_1729728 : prime_factors_sum 1729728 = 36 :=
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_sum_of_prime_factors_1729728_l983_98389


namespace NUMINAMATH_GPT_village_current_population_l983_98313

def initial_population : ℕ := 4675
def died_by_bombardment : ℕ := (5*initial_population + 99) / 100 -- Equivalent to rounding (5/100) * 4675
def remaining_after_bombardment : ℕ := initial_population - died_by_bombardment
def left_due_to_fear : ℕ := (20*remaining_after_bombardment + 99) / 100 -- Equivalent to rounding (20/100) * remaining
def current_population : ℕ := remaining_after_bombardment - left_due_to_fear

theorem village_current_population : current_population = 3553 := by
  sorry

end NUMINAMATH_GPT_village_current_population_l983_98313


namespace NUMINAMATH_GPT_parabola_range_l983_98316

theorem parabola_range (x : ℝ) (h : 0 < x ∧ x < 3) : 
  1 ≤ (x^2 - 4*x + 5) ∧ (x^2 - 4*x + 5) < 5 :=
sorry

end NUMINAMATH_GPT_parabola_range_l983_98316


namespace NUMINAMATH_GPT_intersection_A_B_l983_98362

-- Define set A
def A : Set ℤ := {-1, 1, 2, 3, 4}

-- Define set B with the given condition
def B : Set ℤ := {x : ℤ | 1 ≤ x ∧ x < 3}

-- The main theorem statement showing the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 2} :=
    sorry -- Placeholder for the proof

end NUMINAMATH_GPT_intersection_A_B_l983_98362


namespace NUMINAMATH_GPT_find_cd_l983_98320

theorem find_cd : 
  (∀ x : ℝ, (4 * x - 3) / (x^2 - 3 * x - 18) = ((7 / 3) / (x - 6)) + ((5 / 3) / (x + 3))) :=
by
  intro x
  have h : x^2 - 3 * x - 18 = (x - 6) * (x + 3) := by
    sorry
  rw [h]
  sorry

end NUMINAMATH_GPT_find_cd_l983_98320


namespace NUMINAMATH_GPT_Jenny_reading_days_l983_98352

theorem Jenny_reading_days :
  let words_per_hour := 100
  let book1_words := 200
  let book2_words := 400
  let book3_words := 300
  let total_words := book1_words + book2_words + book3_words
  let total_hours := total_words / words_per_hour
  let minutes_per_day := 54
  let hours_per_day := minutes_per_day / 60
  total_hours / hours_per_day = 10 :=
by
  sorry

end NUMINAMATH_GPT_Jenny_reading_days_l983_98352


namespace NUMINAMATH_GPT_hilt_miles_traveled_l983_98393

theorem hilt_miles_traveled (initial_miles lunch_additional_miles : Real) (h_initial : initial_miles = 212.3) (h_lunch : lunch_additional_miles = 372.0) :
  initial_miles + lunch_additional_miles = 584.3 :=
by
  sorry

end NUMINAMATH_GPT_hilt_miles_traveled_l983_98393


namespace NUMINAMATH_GPT_parallel_lines_coefficient_l983_98325

theorem parallel_lines_coefficient (a : ℝ) :
  (x + 2*a*y - 1 = 0) → (3*a - 1)*x - a*y - 1 = 0 → (a = 0 ∨ a = 1/6) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_coefficient_l983_98325


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l983_98315

theorem hyperbola_asymptotes :
  ∀ x y : ℝ, x^2 - y^2 / 4 = 1 → (y = 2 * x ∨ y = -2 * x) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l983_98315


namespace NUMINAMATH_GPT_sin_alpha_in_second_quadrant_l983_98380

theorem sin_alpha_in_second_quadrant 
  (α : ℝ) 
  (h1 : π / 2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.tan α = -1 / 2)  -- tan α = -1/2
  : Real.sin α = Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_GPT_sin_alpha_in_second_quadrant_l983_98380


namespace NUMINAMATH_GPT_binary_addition_correct_l983_98312

-- define the binary numbers as natural numbers using their binary representations
def bin_1010 : ℕ := 0b1010
def bin_10 : ℕ := 0b10
def bin_sum : ℕ := 0b1100

-- state the theorem that needs to be proved
theorem binary_addition_correct : bin_1010 + bin_10 = bin_sum := by
  sorry

end NUMINAMATH_GPT_binary_addition_correct_l983_98312


namespace NUMINAMATH_GPT_find_value_of_complex_fraction_l983_98365

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem find_value_of_complex_fraction :
  (1 - 2 * i) / (1 + i) = -1 / 2 - 3 / 2 * i := 
sorry

end NUMINAMATH_GPT_find_value_of_complex_fraction_l983_98365


namespace NUMINAMATH_GPT_number_of_teachers_l983_98323

theorem number_of_teachers
  (students : ℕ) (lessons_per_student_per_day : ℕ) (lessons_per_teacher_per_day : ℕ) (students_per_class : ℕ)
  (h1 : students = 1200)
  (h2 : lessons_per_student_per_day = 5)
  (h3 : lessons_per_teacher_per_day = 4)
  (h4 : students_per_class = 30) :
  ∃ teachers : ℕ, teachers = 50 :=
by
  have total_lessons : ℕ := lessons_per_student_per_day * students
  have classes : ℕ := total_lessons / students_per_class
  have teachers : ℕ := classes / lessons_per_teacher_per_day
  use teachers
  sorry

end NUMINAMATH_GPT_number_of_teachers_l983_98323


namespace NUMINAMATH_GPT_solve_equation_l983_98332

theorem solve_equation : 
  ∀ x : ℝ, (x - 3 ≠ 0) → (x + 6) / (x - 3) = 4 → x = 6 :=
by
  intros x h1 h2
  sorry

end NUMINAMATH_GPT_solve_equation_l983_98332


namespace NUMINAMATH_GPT_trajectory_equation_l983_98396

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

theorem trajectory_equation (P : ℝ × ℝ) (h : |distance P (1, 0) - P.1| = 1) :
  (P.1 ≥ 0 → P.2 ^ 2 = 4 * P.1) ∧ (P.1 < 0 → P.2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_trajectory_equation_l983_98396


namespace NUMINAMATH_GPT_value_of_x_l983_98344

theorem value_of_x (x : ℕ) (M : Set ℕ) :
  M = {0, 1, 2} →
  M ∪ {x} = {0, 1, 2, 3} →
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l983_98344


namespace NUMINAMATH_GPT_sequence_x_value_l983_98350

theorem sequence_x_value
  (z y x : ℤ)
  (h1 : z + (-2) = -1)
  (h2 : y + 1 = -2)
  (h3 : x + (-3) = 1) :
  x = 4 := 
sorry

end NUMINAMATH_GPT_sequence_x_value_l983_98350


namespace NUMINAMATH_GPT_water_added_to_solution_l983_98306

theorem water_added_to_solution :
  let initial_volume := 340
  let initial_sugar := 0.20 * initial_volume
  let added_sugar := 3.2
  let added_kola := 6.8
  let final_sugar := initial_sugar + added_sugar
  let final_percentage_sugar := 19.66850828729282 / 100
  let final_volume := final_sugar / final_percentage_sugar
  let added_water := final_volume - initial_volume - added_sugar - added_kola
  added_water = 12 :=
by
  sorry

end NUMINAMATH_GPT_water_added_to_solution_l983_98306


namespace NUMINAMATH_GPT_percentage_excess_calculation_l983_98361

theorem percentage_excess_calculation (A B : ℝ) (x : ℝ) 
  (h1 : (A * (1 + x / 100)) * (B * 0.95) = A * B * 1.007) : 
  x = 6.05 :=
by
  sorry

end NUMINAMATH_GPT_percentage_excess_calculation_l983_98361


namespace NUMINAMATH_GPT_janet_total_pills_l983_98385

-- Define number of days per week
def days_per_week : ℕ := 7

-- Define pills per day for each week
def pills_first_2_weeks :=
  let multivitamins := 2 * days_per_week * 2
  let calcium := 3 * days_per_week * 2
  let magnesium := 5 * days_per_week * 2
  multivitamins + calcium + magnesium

def pills_third_week :=
  let multivitamins := 2 * days_per_week
  let calcium := 1 * days_per_week
  let magnesium := 0 * days_per_week
  multivitamins + calcium + magnesium

def pills_fourth_week :=
  let multivitamins := 3 * days_per_week
  let calcium := 2 * days_per_week
  let magnesium := 3 * days_per_week
  multivitamins + calcium + magnesium

def total_pills := pills_first_2_weeks + pills_third_week + pills_fourth_week

theorem janet_total_pills : total_pills = 245 := by
  -- Lean will generate a proof goal here with the left-hand side of the equation
  -- equal to an evaluated term, and we say that this equals 245 based on the problem's solution.
  sorry

end NUMINAMATH_GPT_janet_total_pills_l983_98385


namespace NUMINAMATH_GPT_find_intersection_l983_98353

noncomputable def intersection_of_lines : Prop :=
  ∃ (x y : ℚ), (5 * x - 3 * y = 15) ∧ (6 * x + 2 * y = 14) ∧ (x = 11 / 4) ∧ (y = -5 / 4)

theorem find_intersection : intersection_of_lines :=
  sorry

end NUMINAMATH_GPT_find_intersection_l983_98353


namespace NUMINAMATH_GPT_probability_scoring_less_than_8_l983_98399

theorem probability_scoring_less_than_8 
  (P10 P9 P8 : ℝ) 
  (hP10 : P10 = 0.3) 
  (hP9 : P9 = 0.3) 
  (hP8 : P8 = 0.2) : 
  1 - (P10 + P9 + P8) = 0.2 := 
by 
  sorry

end NUMINAMATH_GPT_probability_scoring_less_than_8_l983_98399


namespace NUMINAMATH_GPT_simplify_expression_l983_98387

variable (a : ℚ)
def expression := ((a + 3) / (a - 1) - 1 / (a - 1)) / ((a^2 + 4 * a + 4) / (a^2 - a))

theorem simplify_expression (h : a = 3) : expression a = 3 / 5 :=
by
  rw [h]
  -- additional simplifications would typically go here if the steps were spelled out
  sorry

end NUMINAMATH_GPT_simplify_expression_l983_98387


namespace NUMINAMATH_GPT_remainder_sum_first_150_l983_98317

-- Definitions based on the conditions
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Lean statement equivalent to the mathematical problem
theorem remainder_sum_first_150 :
  (sum_first_n 150) % 11250 = 75 :=
by 
sorry

end NUMINAMATH_GPT_remainder_sum_first_150_l983_98317


namespace NUMINAMATH_GPT_total_wheels_in_parking_lot_l983_98331

theorem total_wheels_in_parking_lot :
  let cars := 5
  let trucks := 3
  let bikes := 2
  let three_wheelers := 4
  let wheels_per_car := 4
  let wheels_per_truck := 6
  let wheels_per_bike := 2
  let wheels_per_three_wheeler := 3
  (cars * wheels_per_car + trucks * wheels_per_truck + bikes * wheels_per_bike + three_wheelers * wheels_per_three_wheeler) = 54 := by
  sorry

end NUMINAMATH_GPT_total_wheels_in_parking_lot_l983_98331


namespace NUMINAMATH_GPT_outlet_two_rate_l983_98348

/-- Definitions and conditions for the problem -/
def tank_volume_feet : ℝ := 20
def inlet_rate_cubic_inches_per_min : ℝ := 5
def outlet_one_rate_cubic_inches_per_min : ℝ := 9
def empty_time_minutes : ℝ := 2880
def cubic_feet_to_cubic_inches : ℝ := 1728
def tank_volume_cubic_inches := tank_volume_feet * cubic_feet_to_cubic_inches

/-- Statement to prove the rate of the other outlet pipe -/
theorem outlet_two_rate (x : ℝ) :
  tank_volume_cubic_inches / empty_time_minutes = outlet_one_rate_cubic_inches_per_min + x - inlet_rate_cubic_inches_per_min → 
  x = 8 :=
by
  sorry

end NUMINAMATH_GPT_outlet_two_rate_l983_98348


namespace NUMINAMATH_GPT_functional_equation_solution_l983_98300

theorem functional_equation_solution (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l983_98300


namespace NUMINAMATH_GPT_Alyssa_missed_games_l983_98334

theorem Alyssa_missed_games (total_games attended_games : ℕ) (h1 : total_games = 31) (h2 : attended_games = 13) : total_games - attended_games = 18 :=
by sorry

end NUMINAMATH_GPT_Alyssa_missed_games_l983_98334


namespace NUMINAMATH_GPT_proportion_correct_l983_98357

theorem proportion_correct {a b : ℝ} (h : 2 * a = 5 * b) : a / 5 = b / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_proportion_correct_l983_98357


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l983_98322

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (m : ℕ) 
  (h1 : S (m - 1) = -2) 
  (h2 : S m = 0) 
  (h3 : S (m + 1) = 3) : 
  m = 5 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l983_98322


namespace NUMINAMATH_GPT_total_distance_traveled_in_12_hours_l983_98346

variable (n a1 d : ℕ) (u : ℕ → ℕ)

def arithmetic_seq_sum (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ :=
  n * a1 + (n * (n - 1) * d) / 2

theorem total_distance_traveled_in_12_hours :
  arithmetic_seq_sum 12 55 2 = 792 := by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_in_12_hours_l983_98346


namespace NUMINAMATH_GPT_find_b_l983_98386

variable (a b c : ℕ)
variable (h1 : (a + b + c) / 3 = 45)
variable (h2 : (a + b) / 2 = 40)
variable (h3 : (b + c) / 2 = 43)

theorem find_b : b = 31 := sorry

end NUMINAMATH_GPT_find_b_l983_98386


namespace NUMINAMATH_GPT_question_1_question_2_question_3_l983_98373
-- Importing the Mathlib library for necessary functions

-- Definitions and assumptions based on the problem conditions
def z0 (m : ℝ) : ℂ := 1 - m * Complex.I
def z (x y : ℝ) : ℂ := x + y * Complex.I
def w (x' y' : ℝ) : ℂ := x' + y' * Complex.I

/-- The proof problem in Lean 4 to find necessary values and relationships -/
theorem question_1 (m : ℝ) (hm : m > 0) :
  (Complex.abs (z0 m) = 2 → m = Real.sqrt 3) ∧
  (∀ (x y : ℝ), ∃ (x' y' : ℝ), x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y) :=
by
  sorry

theorem question_2 (x y : ℝ) (hx : y = x + 1) :
  ∃ x' y', x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y ∧ 
  y' = (2 - Real.sqrt 3) * x' - 2 * Real.sqrt 3 + 2 :=
by
  sorry

theorem question_3 (x y : ℝ) :
  (∃ (k b : ℝ), y = k * x + b ∧ 
  (∀ (x y x' y' : ℝ), x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y ∧ y' = k * x' + b → 
  y = Real.sqrt 3 / 3 * x ∨ y = - Real.sqrt 3 * x)) :=
by
  sorry

end NUMINAMATH_GPT_question_1_question_2_question_3_l983_98373


namespace NUMINAMATH_GPT_Sam_memorized_more_digits_l983_98345

variable (MinaDigits SamDigits CarlosDigits : ℕ)
variable (h1 : MinaDigits = 6 * CarlosDigits)
variable (h2 : MinaDigits = 24)
variable (h3 : SamDigits = 10)
 
theorem Sam_memorized_more_digits :
  SamDigits - CarlosDigits = 6 :=
by
  -- Let's unfold the statements and perform basic arithmetic.
  sorry

end NUMINAMATH_GPT_Sam_memorized_more_digits_l983_98345


namespace NUMINAMATH_GPT_ratio_x_y_l983_98321

theorem ratio_x_y (x y : ℝ) (h1 : x * y = 9) (h2 : 0 < x) (h3 : 0 < y) (h4 : y = 0.5) : x / y = 36 :=
by
  sorry

end NUMINAMATH_GPT_ratio_x_y_l983_98321


namespace NUMINAMATH_GPT_unit_digit_of_six_consecutive_product_is_zero_l983_98364

theorem unit_digit_of_six_consecutive_product_is_zero (n : ℕ) (h : n > 0) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5)) % 10 = 0 := 
by sorry

end NUMINAMATH_GPT_unit_digit_of_six_consecutive_product_is_zero_l983_98364


namespace NUMINAMATH_GPT_investment_amount_l983_98360

noncomputable def annual_income (investment : ℝ) (percent_stock : ℝ) (market_price : ℝ) : ℝ :=
  (investment * percent_stock / 100) / market_price * market_price

theorem investment_amount (annual_income_value : ℝ) (percent_stock : ℝ) (market_price : ℝ) (investment : ℝ) :
  annual_income investment percent_stock market_price = annual_income_value →
  investment = 6800 :=
by
  intros
  sorry

end NUMINAMATH_GPT_investment_amount_l983_98360


namespace NUMINAMATH_GPT_ripe_oranges_count_l983_98308

/-- They harvest 52 sacks of unripe oranges per day. -/
def unripe_oranges_per_day : ℕ := 52

/-- After 26 days of harvest, they will have 2080 sacks of oranges. -/
def total_oranges_after_26_days : ℕ := 2080

/-- Define the number of sacks of ripe oranges harvested per day. -/
def ripe_oranges_per_day (R : ℕ) : Prop :=
  26 * (R + unripe_oranges_per_day) = total_oranges_after_26_days

/-- Prove that they harvest 28 sacks of ripe oranges per day. -/
theorem ripe_oranges_count : ripe_oranges_per_day 28 :=
by {
  -- This is where the proof would go
  sorry
}

end NUMINAMATH_GPT_ripe_oranges_count_l983_98308


namespace NUMINAMATH_GPT_remainder_when_x_squared_div_30_l983_98335

theorem remainder_when_x_squared_div_30 (x : ℤ) 
  (h1 : 5 * x ≡ 15 [ZMOD 30]) 
  (h2 : 7 * x ≡ 13 [ZMOD 30]) : 
  (x^2) % 30 = 21 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_when_x_squared_div_30_l983_98335


namespace NUMINAMATH_GPT_fraction_inequality_l983_98366

theorem fraction_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / b) + (b / c) + (c / a) ≤ (a^2 / b^2) + (b^2 / c^2) + (c^2 / a^2) := 
by
  sorry

end NUMINAMATH_GPT_fraction_inequality_l983_98366


namespace NUMINAMATH_GPT_jacob_younger_than_michael_l983_98395

-- Definitions based on the conditions.
def jacob_current_age : ℕ := 9
def michael_current_age : ℕ := 2 * (jacob_current_age + 3) - 3

-- Theorem to prove that Jacob is 12 years younger than Michael.
theorem jacob_younger_than_michael : michael_current_age - jacob_current_age = 12 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_jacob_younger_than_michael_l983_98395


namespace NUMINAMATH_GPT_hyperbola_standard_equation_correct_l983_98375

-- Define the initial values given in conditions
def a : ℝ := 12
def b : ℝ := 5
def c : ℝ := 4

-- Define the hyperbola equation form based on conditions and focal properties
noncomputable def hyperbola_standard_equation : Prop :=
  let a2 := (8 / 5)
  let b2 := (72 / 5)
  (∀ x y : ℝ, y^2 / a2 - x^2 / b2 = 1)

-- State the final problem as a theorem
theorem hyperbola_standard_equation_correct :
  ∀ x y : ℝ, y^2 / (8 / 5) - x^2 / (72 / 5) = 1 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_standard_equation_correct_l983_98375


namespace NUMINAMATH_GPT_triangle_perimeter_l983_98356

theorem triangle_perimeter (a b : ℕ) (h1 : a = 2) (h2 : b = 3) (x : ℕ) 
  (x_odd : x % 2 = 1) (triangle_ineq : 1 < x ∧ x < 5) : a + b + x = 8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l983_98356


namespace NUMINAMATH_GPT_convert_to_base_8_l983_98314

theorem convert_to_base_8 (n : ℕ) (hn : n = 3050) : 
  ∃ d1 d2 d3 d4 : ℕ, d1 = 5 ∧ d2 = 7 ∧ d3 = 5 ∧ d4 = 2 ∧ n = d1 * 8^3 + d2 * 8^2 + d3 * 8^1 + d4 * 8^0 :=
by 
  use 5, 7, 5, 2
  sorry

end NUMINAMATH_GPT_convert_to_base_8_l983_98314


namespace NUMINAMATH_GPT_inequality_proof_l983_98363

noncomputable def inequality (x y z : ℝ) : Prop :=
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7

theorem inequality_proof (x y z : ℝ) (hx : x ≥ y + z) (hx_pos: 0 < x) (hy_pos: 0 < y) (hz_pos: 0 < z) :
  inequality x y z :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l983_98363


namespace NUMINAMATH_GPT_initial_stock_decaf_percentage_l983_98355

-- Definitions as conditions of the problem
def initial_coffee_stock : ℕ := 400
def purchased_coffee_stock : ℕ := 100
def percentage_decaf_purchased : ℕ := 60
def total_percentage_decaf : ℕ := 32

/-- The proof problem statement -/
theorem initial_stock_decaf_percentage : 
  ∃ x : ℕ, x * initial_coffee_stock / 100 + percentage_decaf_purchased * purchased_coffee_stock / 100 = total_percentage_decaf * (initial_coffee_stock + purchased_coffee_stock) / 100 ∧ x = 25 :=
sorry

end NUMINAMATH_GPT_initial_stock_decaf_percentage_l983_98355


namespace NUMINAMATH_GPT_prime_p_p_plus_15_l983_98397

theorem prime_p_p_plus_15 (p : ℕ) (hp : Nat.Prime p) (hp15 : Nat.Prime (p + 15)) : p = 2 :=
sorry

end NUMINAMATH_GPT_prime_p_p_plus_15_l983_98397


namespace NUMINAMATH_GPT_fraction_product_l983_98339

theorem fraction_product :
  (2 / 3) * (5 / 7) * (9 / 11) * (4 / 13) = 360 / 3003 := by
  sorry

end NUMINAMATH_GPT_fraction_product_l983_98339


namespace NUMINAMATH_GPT_anie_days_to_complete_l983_98354

def normal_work_hours : ℕ := 10
def extra_hours : ℕ := 5
def total_project_hours : ℕ := 1500

theorem anie_days_to_complete :
  (total_project_hours / (normal_work_hours + extra_hours)) = 100 :=
by
  sorry

end NUMINAMATH_GPT_anie_days_to_complete_l983_98354


namespace NUMINAMATH_GPT_quadratic_has_at_most_two_solutions_l983_98359

theorem quadratic_has_at_most_two_solutions (a b c : ℝ) (h : a ≠ 0) :
  ¬(∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧
    a * x1^2 + b * x1 + c = 0 ∧ 
    a * x2^2 + b * x2 + c = 0 ∧ 
    a * x3^2 + b * x3 + c = 0) := 
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_has_at_most_two_solutions_l983_98359


namespace NUMINAMATH_GPT_julies_birthday_day_of_week_l983_98311

theorem julies_birthday_day_of_week
    (fred_birthday_monday : Nat)
    (pat_birthday_before_fred : Nat)
    (julie_birthday_before_pat : Nat)
    (fred_birthday_after_pat : fred_birthday_monday - pat_birthday_before_fred = 37)
    (julie_birthday_before_pat_eq : pat_birthday_before_fred - julie_birthday_before_pat = 67)
    : (julie_birthday_before_pat - julie_birthday_before_pat % 7 + ((julie_birthday_before_pat % 7) - fred_birthday_monday % 7)) % 7 = 2 :=
by
  sorry

end NUMINAMATH_GPT_julies_birthday_day_of_week_l983_98311


namespace NUMINAMATH_GPT_card_probability_multiple_l983_98379

def is_multiple_of (n k : ℕ) : Prop := k > 0 ∧ n % k = 0

def count_multiples (n k : ℕ) : ℕ :=
  if k = 0 then 0 else n / k

def inclusion_exclusion (a b c : ℕ) (n : ℕ) : ℕ :=
  (count_multiples n a) + (count_multiples n b) + (count_multiples n c) - 
  (count_multiples n (Nat.lcm a b)) - (count_multiples n (Nat.lcm a c)) - 
  (count_multiples n (Nat.lcm b c)) + 
  count_multiples n (Nat.lcm a (Nat.lcm b c))

theorem card_probability_multiple (n : ℕ) 
  (a b c : ℕ) (hne : n ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (inclusion_exclusion a b c n) / n = 47 / 100 := by
  sorry

end NUMINAMATH_GPT_card_probability_multiple_l983_98379


namespace NUMINAMATH_GPT_square_side_length_l983_98303

theorem square_side_length (A : ℝ) (h : A = 100) : ∃ s : ℝ, s * s = A ∧ s = 10 := by
  sorry

end NUMINAMATH_GPT_square_side_length_l983_98303


namespace NUMINAMATH_GPT_proof_problem_l983_98329

noncomputable def problem (a b c d : ℝ) : Prop :=
(a + b + c = 3) ∧ 
(a + b + d = -1) ∧ 
(a + c + d = 8) ∧ 
(b + c + d = 0) ∧ 
(a * b + c * d = -127 / 9)

theorem proof_problem (a b c d : ℝ) : 
  (a + b + c = 3) → 
  (a + b + d = -1) →
  (a + c + d = 8) → 
  (b + c + d = 0) → 
  (a * b + c * d = -127 / 9) :=
by 
  intro h1 h2 h3 h4
  -- Proof is omitted, "sorry" indicates it is to be filled in
  admit

end NUMINAMATH_GPT_proof_problem_l983_98329


namespace NUMINAMATH_GPT_rectangle_area_l983_98318

theorem rectangle_area (P l w : ℝ) (h1 : P = 60) (h2 : l / w = 3 / 2) (h3 : P = 2 * l + 2 * w) : l * w = 216 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l983_98318


namespace NUMINAMATH_GPT_richard_older_than_david_by_l983_98372

-- Definitions based on given conditions

def richard : ℕ := sorry
def david : ℕ := 14 -- David is 14 years old.
def scott : ℕ := david - 8 -- Scott is 8 years younger than David.

-- In 8 years, Richard will be twice as old as Scott
axiom richard_in_8_years : richard + 8 = 2 * (scott + 8)

-- To prove: How many years older is Richard than David?
theorem richard_older_than_david_by : richard - david = 6 := sorry

end NUMINAMATH_GPT_richard_older_than_david_by_l983_98372


namespace NUMINAMATH_GPT_determine_phi_l983_98374

theorem determine_phi
  (A ω : ℝ) (φ : ℝ) (x : ℝ)
  (hA : 0 < A)
  (hω : 0 < ω)
  (hφ : abs φ < Real.pi / 2)
  (h_symm : ∃ f : ℝ → ℝ, f (-Real.pi / 4) = A ∨ f (-Real.pi / 4) = -A)
  (h_zero : ∃ x₀ : ℝ, A * Real.sin (ω * x₀ + φ) = 0 ∧ abs (x₀ + Real.pi / 4) = Real.pi / 2) :
  φ = -Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_determine_phi_l983_98374


namespace NUMINAMATH_GPT_inlet_pipe_filling_rate_l983_98340

def leak_rate (volume : ℕ) (time_hours : ℕ) : ℕ :=
  volume / time_hours

def net_emptying_rate (volume : ℕ) (time_hours : ℕ) : ℕ :=
  volume / time_hours

def inlet_rate_per_hour (net_rate : ℕ) (leak_rate : ℕ) : ℕ :=
  leak_rate - net_rate

def convert_to_minutes (rate_per_hour : ℕ) : ℕ :=
  rate_per_hour / 60

theorem inlet_pipe_filling_rate :
  let volume := 4320
  let time_to_empty_with_leak := 6
  let net_time_to_empty := 12
  let leak_rate := leak_rate volume time_to_empty_with_leak
  let net_rate := net_emptying_rate volume net_time_to_empty
  let fill_rate_per_hour := inlet_rate_per_hour net_rate leak_rate
  convert_to_minutes fill_rate_per_hour = 6 := by
    -- Proof ends with a placeholder 'sorry'
    sorry

end NUMINAMATH_GPT_inlet_pipe_filling_rate_l983_98340


namespace NUMINAMATH_GPT_root_of_quadratic_eq_is_two_l983_98309

theorem root_of_quadratic_eq_is_two (k : ℝ) : (2^2 - 3 * 2 + k = 0) → k = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_root_of_quadratic_eq_is_two_l983_98309


namespace NUMINAMATH_GPT_smallest_integer_value_l983_98310

theorem smallest_integer_value (x : ℤ) (h : 7 - 3 * x < 22) : x ≥ -4 := 
sorry

end NUMINAMATH_GPT_smallest_integer_value_l983_98310


namespace NUMINAMATH_GPT_average_age_of_boys_l983_98324

theorem average_age_of_boys
  (N : ℕ) (G : ℕ) (A_G : ℕ) (A_S : ℚ) (B : ℕ)
  (hN : N = 652)
  (hG : G = 163)
  (hA_G : A_G = 11)
  (hA_S : A_S = 11.75)
  (hB : B = N - G) :
  (163 * 11 + 489 * x = 11.75 * 652) → x = 12 := by
  sorry

end NUMINAMATH_GPT_average_age_of_boys_l983_98324


namespace NUMINAMATH_GPT_pencils_count_l983_98391

theorem pencils_count (pens pencils : ℕ) 
  (h_ratio : 6 * pens = 5 * pencils) 
  (h_difference : pencils = pens + 6) : 
  pencils = 36 := 
by 
  sorry

end NUMINAMATH_GPT_pencils_count_l983_98391


namespace NUMINAMATH_GPT_find_m_values_l983_98381

theorem find_m_values (α : Real) (m : Real) (h1 : α ∈ Set.Ioo π (3 * π / 2)) 
  (h2 : Real.sin α = (3 * m - 2) / (m + 3)) 
  (h3 : Real.cos α = (m - 5) / (m + 3)) : m = (10 / 9) ∨ m = 2 := by 
  sorry

end NUMINAMATH_GPT_find_m_values_l983_98381


namespace NUMINAMATH_GPT_ellipse_foci_distance_l983_98392

noncomputable def distance_between_foci : ℝ := 2 * Real.sqrt 29

theorem ellipse_foci_distance : 
  ∀ (x y : ℝ), 
  (Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25) → 
  distance_between_foci = 2 * Real.sqrt 29 := 
by
  intros x y h
  -- proof goes here (skipped)
  sorry

end NUMINAMATH_GPT_ellipse_foci_distance_l983_98392


namespace NUMINAMATH_GPT_factorize_expression_l983_98301

theorem factorize_expression (x y : ℝ) : 
  (x + y)^2 - 14 * (x + y) + 49 = (x + y - 7)^2 := 
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l983_98301


namespace NUMINAMATH_GPT_greatest_common_length_cords_l983_98388

theorem greatest_common_length_cords (l1 l2 l3 l4 : ℝ) (h1 : l1 = Real.sqrt 20) (h2 : l2 = Real.pi) (h3 : l3 = Real.exp 1) (h4 : l4 = Real.sqrt 98) : 
  ∃ d : ℝ, d = 1 ∧ (∀ k1 k2 k3 k4 : ℝ, k1 * d = l1 → k2 * d = l2 → k3 * d = l3 → k4 * d = l4 → ∀i : ℝ, i = d) :=
by
  sorry

end NUMINAMATH_GPT_greatest_common_length_cords_l983_98388


namespace NUMINAMATH_GPT_same_color_combination_probability_l983_98337

-- Defining the number of each color candy 
def num_red : Nat := 12
def num_blue : Nat := 12
def num_green : Nat := 6

-- Terry and Mary each pick 3 candies at random
def total_pick : Nat := 3

-- The total number of candies in the jar
def total_candies : Nat := num_red + num_blue + num_green

-- Probability of Terry and Mary picking the same color combination
def probability_same_combination : ℚ := 2783 / 847525

-- The theorem statement
theorem same_color_combination_probability :
  let terry_picks_red := (num_red * (num_red - 1) * (num_red - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_red := num_red - total_pick
  let mary_picks_red := (remaining_red * (remaining_red - 1) * (remaining_red - 2)) / (27 * 26 * 25)
  let combined_red := terry_picks_red * mary_picks_red

  let terry_picks_blue := (num_blue * (num_blue - 1) * (num_blue - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_blue := num_blue - total_pick
  let mary_picks_blue := (remaining_blue * (remaining_blue - 1) * (remaining_blue - 2)) / (27 * 26 * 25)
  let combined_blue := terry_picks_blue * mary_picks_blue

  let terry_picks_green := (num_green * (num_green - 1) * (num_green - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_green := num_green - total_pick
  let mary_picks_green := (remaining_green * (remaining_green - 1) * (remaining_green - 2)) / (27 * 26 * 25)
  let combined_green := terry_picks_green * mary_picks_green

  let total_probability := 2 * combined_red + 2 * combined_blue + combined_green
  total_probability = probability_same_combination := sorry

end NUMINAMATH_GPT_same_color_combination_probability_l983_98337


namespace NUMINAMATH_GPT_trebled_resultant_is_correct_l983_98383

-- Definitions based on the conditions provided in step a)
def initial_number : ℕ := 5
def doubled_result : ℕ := initial_number * 2
def added_15_result : ℕ := doubled_result + 15
def trebled_resultant : ℕ := added_15_result * 3

-- We need to prove that the trebled resultant is equal to 75
theorem trebled_resultant_is_correct : trebled_resultant = 75 :=
by
  sorry

end NUMINAMATH_GPT_trebled_resultant_is_correct_l983_98383


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_S12_l983_98302

variable {a : ℕ → ℝ} -- Arithmetic sequence a_n
variable {S : ℕ → ℝ} -- Sum of the first n terms S_n

-- Conditions given in the problem
axiom condition1 (n : ℕ) : S n = (n / 2) * (a 1 + a n)
axiom condition2 : a 4 + a 9 = 10

-- Proving that S 12 = 60 given the conditions
theorem sum_arithmetic_sequence_S12 : S 12 = 60 := by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_S12_l983_98302


namespace NUMINAMATH_GPT_range_of_a_l983_98368

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x ∈ (Set.Iio (-1) ∪ Set.Ioi 3)) → ((x + a) * (x + 1) > 0)) ∧ 
  (∃ x : ℝ, ¬(x ∈ (Set.Iio (-1) ∪ Set.Ioi 3)) ∧ ((x + a) * (x + 1) > 0)) → 
  a ∈ Set.Iio (-3) := 
  sorry

end NUMINAMATH_GPT_range_of_a_l983_98368


namespace NUMINAMATH_GPT_min_value_2a_plus_b_value_of_t_l983_98371

theorem min_value_2a_plus_b (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 2/b = 2) :
  2 * a + b = 4 :=
sorry

theorem value_of_t (a b t : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 2/b = 2) (h₄ : 4^a = t) (h₅ : 3^b = t) :
  t = 6 :=
sorry

end NUMINAMATH_GPT_min_value_2a_plus_b_value_of_t_l983_98371


namespace NUMINAMATH_GPT_positive_difference_l983_98369

theorem positive_difference (x y : ℝ) (h1 : x + y = 50) (h2 : 3 * y - 3 * x = 27) : y - x = 9 :=
sorry

end NUMINAMATH_GPT_positive_difference_l983_98369


namespace NUMINAMATH_GPT_total_cost_correct_l983_98327

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59
def total_cost : ℝ := 12.30

theorem total_cost_correct : football_cost + marbles_cost = total_cost := 
by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l983_98327


namespace NUMINAMATH_GPT_obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l983_98326

-- The main theorem states that 100 can be obtained using fewer than ten 3's.

theorem obtain_100_using_fewer_than_ten_threes_example1 :
  100 = (333 / 3) - (33 / 3) :=
by
  sorry

theorem obtain_100_using_fewer_than_ten_threes_example2 :
  100 = (33 * 3) + (3 / 3) :=
by
  sorry

end NUMINAMATH_GPT_obtain_100_using_fewer_than_ten_threes_example1_obtain_100_using_fewer_than_ten_threes_example2_l983_98326


namespace NUMINAMATH_GPT_reciprocal_of_sum_l983_98328

-- Define the fractions
def a := (1: ℚ) / 2
def b := (1: ℚ) / 3

-- Define their sum
def c := a + b

-- Define the expected reciprocal
def reciprocal := (6: ℚ) / 5

-- The theorem we want to prove:
theorem reciprocal_of_sum : (c⁻¹ = reciprocal) :=
by 
  sorry

end NUMINAMATH_GPT_reciprocal_of_sum_l983_98328


namespace NUMINAMATH_GPT_converse_proposition_converse_proposition_true_l983_98394

theorem converse_proposition (x : ℝ) (h : x > 0) : x^2 - 1 > 0 :=
by sorry

theorem converse_proposition_true (x : ℝ) (h : x^2 - 1 > 0) : x > 0 :=
by sorry

end NUMINAMATH_GPT_converse_proposition_converse_proposition_true_l983_98394


namespace NUMINAMATH_GPT_min_max_value_sum_l983_98376

variable (a b c d e : ℝ)

theorem min_max_value_sum :
  a + b + c + d + e = 10 ∧ a^2 + b^2 + c^2 + d^2 + e^2 = 30 →
  let expr := 5 * (a^3 + b^3 + c^3 + d^3 + e^3) - (a^4 + b^4 + c^4 + d^4 + e^4)
  let m := 42
  let M := 52
  m + M = 94 := sorry

end NUMINAMATH_GPT_min_max_value_sum_l983_98376


namespace NUMINAMATH_GPT_number_of_molecules_correct_l983_98358

-- Define Avogadro's number
def avogadros_number : ℝ := 6.022 * 10^23

-- Define the given number of molecules
def given_number_of_molecules : ℝ := 3 * 10^26

-- State the problem
theorem number_of_molecules_correct :
  (number_of_molecules = given_number_of_molecules) :=
by
  sorry

end NUMINAMATH_GPT_number_of_molecules_correct_l983_98358


namespace NUMINAMATH_GPT_max_value_is_5_l983_98343

noncomputable def max_value (θ φ : ℝ) : ℝ :=
  3 * Real.sin θ * Real.cos φ + 2 * Real.sin φ ^ 2

theorem max_value_is_5 (θ φ : ℝ) (h1 : 0 ≤ θ) (h2 : θ ≤ Real.pi / 2) (h3 : 0 ≤ φ) (h4 : φ ≤ Real.pi / 2) :
  max_value θ φ ≤ 5 :=
sorry

end NUMINAMATH_GPT_max_value_is_5_l983_98343


namespace NUMINAMATH_GPT_max_sum_arithmetic_sequence_terms_l983_98333

theorem max_sum_arithmetic_sequence_terms (d : ℝ) (a : ℕ → ℝ) (n : ℕ) 
  (h0 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h1 : d < 0)
  (h2 : a 1 ^ 2 = a 11 ^ 2) : 
  (n = 5) ∨ (n = 6) :=
sorry

end NUMINAMATH_GPT_max_sum_arithmetic_sequence_terms_l983_98333


namespace NUMINAMATH_GPT_john_tran_probability_2_9_l983_98382

def johnArrivalProbability (train_start train_end john_min john_max: ℕ) : ℚ := 
  let overlap_area := ((train_end - train_start - 15) * 15) / 2 
  let total_area := (john_max - john_min) * (train_end - train_start)
  overlap_area / total_area

theorem john_tran_probability_2_9 :
  johnArrivalProbability 30 90 0 90 = 2 / 9 := by
  sorry

end NUMINAMATH_GPT_john_tran_probability_2_9_l983_98382


namespace NUMINAMATH_GPT_perimeter_of_square_C_l983_98377

theorem perimeter_of_square_C (s_A s_B s_C : ℝ)
  (h1 : 4 * s_A = 16)
  (h2 : 4 * s_B = 32)
  (h3 : s_C = s_B - s_A) :
  4 * s_C = 16 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_C_l983_98377


namespace NUMINAMATH_GPT_find_a_l983_98342

-- Condition: Define a * b as 2a - b^2
def star (a b : ℝ) := 2 * a - b^2

-- Proof problem: Prove the value of a given the condition and that a * 7 = 16.
theorem find_a : ∃ a : ℝ, star a 7 = 16 ∧ a = 32.5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l983_98342


namespace NUMINAMATH_GPT_zoo_people_l983_98351

def number_of_people (cars : ℝ) (people_per_car : ℝ) : ℝ :=
  cars * people_per_car

theorem zoo_people (h₁ : cars = 3.0) (h₂ : people_per_car = 63.0) :
  number_of_people cars people_per_car = 189.0 :=
by
  rw [h₁, h₂]
  -- multiply the numbers directly after substitution
  norm_num
  -- left this as a placeholder for now, can use calc or norm_num for final steps
  exact sorry

end NUMINAMATH_GPT_zoo_people_l983_98351


namespace NUMINAMATH_GPT_james_ate_eight_slices_l983_98319

-- Define the conditions
def num_pizzas := 2
def slices_per_pizza := 6
def fraction_james_ate := 2 / 3
def total_slices := num_pizzas * slices_per_pizza

-- Define the statement to prove
theorem james_ate_eight_slices : fraction_james_ate * total_slices = 8 :=
by
  sorry

end NUMINAMATH_GPT_james_ate_eight_slices_l983_98319


namespace NUMINAMATH_GPT_marble_problem_solution_l983_98347

noncomputable def probability_two_marbles (red_marble_initial white_marble_initial total_drawn : ℕ) : ℚ :=
  let total_initial := red_marble_initial + white_marble_initial
  let probability_first_white := (white_marble_initial : ℚ) / total_initial
  let red_marble_after_first_draw := red_marble_initial
  let total_after_first_draw := total_initial - 1
  let probability_second_red := (red_marble_after_first_draw : ℚ) / total_after_first_draw
  probability_first_white * probability_second_red

theorem marble_problem_solution :
  probability_two_marbles 4 6 2 = 4 / 15 := by
  sorry

end NUMINAMATH_GPT_marble_problem_solution_l983_98347


namespace NUMINAMATH_GPT_vector_dot_product_l983_98307

open Matrix

section VectorDotProduct

variables (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)
variables (E : ℝ × ℝ) (F : ℝ × ℝ)

def vector_sub (P Q : ℝ × ℝ) : ℝ × ℝ := (P.1 - Q.1, P.2 - Q.2)
def vector_add (P Q : ℝ × ℝ) : ℝ × ℝ := (P.1 + Q.1, P.2 + Q.2)
def scalar_mul (k : ℝ) (P : ℝ × ℝ) : ℝ × ℝ := (k * P.1, k * P.2)
def dot_product (P Q : ℝ × ℝ) : ℝ := P.1 * Q.1 + P.2 * Q.2

axiom A_coord : A = (1, 2)
axiom B_coord : B = (2, -1)
axiom C_coord : C = (2, 2)
axiom E_is_trisection : vector_add (vector_sub B A) (scalar_mul (1/3) (vector_sub C B)) = E
axiom F_is_trisection : vector_add (vector_sub B A) (scalar_mul (2/3) (vector_sub C B)) = F

theorem vector_dot_product : dot_product (vector_sub E A) (vector_sub F A) = 3 := by
  sorry

end VectorDotProduct

end NUMINAMATH_GPT_vector_dot_product_l983_98307


namespace NUMINAMATH_GPT_maximize_sequence_l983_98304

theorem maximize_sequence (n : ℕ) (an : ℕ → ℝ) (h : ∀ n, an n = (10/11)^n * (3 * n + 13)) : 
  (∃ n_max, (∀ m, an m ≤ an n_max) ∧ n_max = 6) :=
by
  sorry

end NUMINAMATH_GPT_maximize_sequence_l983_98304


namespace NUMINAMATH_GPT_area_of_side_face_l983_98367

theorem area_of_side_face (L W H : ℝ) 
  (h1 : W * H = (1/2) * (L * W))
  (h2 : L * W = 1.5 * (H * L))
  (h3 : L * W * H = 648) :
  H * L = 72 := 
by
  sorry

end NUMINAMATH_GPT_area_of_side_face_l983_98367


namespace NUMINAMATH_GPT_inequality_conditions_l983_98305

variable (a b : ℝ)

theorem inequality_conditions (ha : 1 / a < 1 / b) (hb : 1 / b < 0) : 
  (1 / (a + b) < 1 / (a * b)) ∧ ¬(a * - (1 / a) > b * - (1 / b)) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_conditions_l983_98305


namespace NUMINAMATH_GPT_area_within_fence_l983_98378

theorem area_within_fence : 
  let rectangle_area := 20 * 18
  let cutout_area := 4 * 4
  rectangle_area - cutout_area = 344 := by
    -- Definitions
    let rectangle_area := 20 * 18
    let cutout_area := 4 * 4
    
    -- Computation of areas
    show rectangle_area - cutout_area = 344
    sorry

end NUMINAMATH_GPT_area_within_fence_l983_98378


namespace NUMINAMATH_GPT_distance_to_destination_l983_98390

theorem distance_to_destination :
  ∀ (D : ℝ) (T : ℝ),
    (15:ℝ) = T →
    (30:ℝ) = T / 2 →
    T - (T / 2) = 3 →
    D = 15 * T → D = 90 :=
by
  intros D T Theon_speed Yara_speed time_difference distance_calc
  sorry

end NUMINAMATH_GPT_distance_to_destination_l983_98390


namespace NUMINAMATH_GPT_find_number_l983_98341

theorem find_number (x : ℝ) : (1.12 * x) / 4.98 = 528.0642570281125 → x = 2350 :=
  by 
  sorry

end NUMINAMATH_GPT_find_number_l983_98341


namespace NUMINAMATH_GPT_necessary_sufficient_condition_geometric_sequence_l983_98338

noncomputable def an_geometric (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem necessary_sufficient_condition_geometric_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ) (p q : ℝ) (h_sum : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
  (h_eq : ∀ n : ℕ, a (n + 1) = p * S n + q) :
  (a 1 = q) ↔ (∃ r : ℝ, an_geometric a r) :=
sorry

end NUMINAMATH_GPT_necessary_sufficient_condition_geometric_sequence_l983_98338


namespace NUMINAMATH_GPT_find_a_for_quadratic_l983_98336

theorem find_a_for_quadratic (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20 * y) + 20 = 0) ↔ a = 20 := 
sorry

end NUMINAMATH_GPT_find_a_for_quadratic_l983_98336


namespace NUMINAMATH_GPT_james_jump_height_is_16_l983_98370

-- Define given conditions
def mark_jump_height : ℕ := 6
def lisa_jump_height : ℕ := 2 * mark_jump_height
def jacob_jump_height : ℕ := 2 * lisa_jump_height
def james_jump_height : ℕ := (2 * jacob_jump_height) / 3

-- Problem Statement to prove
theorem james_jump_height_is_16 : james_jump_height = 16 :=
by
  sorry

end NUMINAMATH_GPT_james_jump_height_is_16_l983_98370


namespace NUMINAMATH_GPT_jack_last_10_shots_made_l983_98349

theorem jack_last_10_shots_made (initial_shots : ℕ) (initial_percentage : ℚ)
  (additional_shots : ℕ) (new_percentage : ℚ)
  (initial_successful_shots : initial_shots * initial_percentage = 18)
  (total_shots : initial_shots + additional_shots = 40)
  (total_successful_shots : (initial_shots + additional_shots) * new_percentage = 25) :
  ∃ x : ℕ, x = 7 := by
sorry

end NUMINAMATH_GPT_jack_last_10_shots_made_l983_98349
