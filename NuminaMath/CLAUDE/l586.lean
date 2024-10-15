import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l586_58656

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1/2) * a 8 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l586_58656


namespace NUMINAMATH_CALUDE_largest_integer_b_for_all_real_domain_l586_58679

theorem largest_integer_b_for_all_real_domain : 
  ∀ b : ℤ, (∀ x : ℝ, x^2 + b*x + 12 ≠ 0) → b ≤ 6 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_b_for_all_real_domain_l586_58679


namespace NUMINAMATH_CALUDE_book_purchases_l586_58614

/-- The number of people who purchased only book A -/
def v : ℕ := sorry

/-- The number of people who purchased only book B -/
def x : ℕ := sorry

/-- The number of people who purchased book B (both only and with book A) -/
def y : ℕ := sorry

/-- The number of people who purchased both books A and B -/
def both : ℕ := 500

theorem book_purchases : 
  (y = x + both) ∧ 
  (v = 2 * y) ∧ 
  (both = 2 * x) →
  v = 1500 := by sorry

end NUMINAMATH_CALUDE_book_purchases_l586_58614


namespace NUMINAMATH_CALUDE_right_triangle_area_l586_58610

/-- The area of a right triangle with hypotenuse 10√2 cm and one angle 45° is 50 cm² -/
theorem right_triangle_area (h : ℝ) (α : ℝ) (A : ℝ) : 
  h = 10 * Real.sqrt 2 →  -- hypotenuse is 10√2 cm
  α = 45 * π / 180 →      -- one angle is 45°
  A = h^2 / 4 →           -- area formula for 45-45-90 triangle
  A = 50 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_area_l586_58610


namespace NUMINAMATH_CALUDE_officer_selection_theorem_l586_58654

def total_candidates : ℕ := 20
def officer_positions : ℕ := 6
def past_officers : ℕ := 5

theorem officer_selection_theorem :
  (Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (officer_positions - 1)) +
  (Nat.choose past_officers 2 * Nat.choose (total_candidates - past_officers) (officer_positions - 2)) +
  (Nat.choose past_officers 3 * Nat.choose (total_candidates - past_officers) (officer_positions - 3)) = 33215 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_theorem_l586_58654


namespace NUMINAMATH_CALUDE_power_function_through_point_l586_58681

/-- A power function passing through (2, 1/8) has exponent -3 -/
theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, x > 0 → f x = x^α) →  -- f is a power function for positive x
  f 2 = 1/8 →                 -- f passes through (2, 1/8)
  α = -3 := by
sorry


end NUMINAMATH_CALUDE_power_function_through_point_l586_58681


namespace NUMINAMATH_CALUDE_min_value_theorem_l586_58646

theorem min_value_theorem (x y : ℝ) (h1 : x + y = 4) (h2 : x > y) (h3 : y > 0) :
  (∀ a b : ℝ, a + b = 4 → a > b → b > 0 → (2 / (a - b) + 1 / b) ≥ 2) ∧ 
  (∃ a b : ℝ, a + b = 4 ∧ a > b ∧ b > 0 ∧ 2 / (a - b) + 1 / b = 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l586_58646


namespace NUMINAMATH_CALUDE_exam_students_count_l586_58630

theorem exam_students_count (total_average : ℝ) (excluded_average : ℝ) (new_average : ℝ) 
  (excluded_count : ℕ) (h1 : total_average = 80) (h2 : excluded_average = 40) 
  (h3 : new_average = 90) (h4 : excluded_count = 5) : 
  ∃ (N : ℕ), N = 25 ∧ 
  (N : ℝ) * total_average = (N - excluded_count : ℝ) * new_average + 
    (excluded_count : ℝ) * excluded_average :=
by
  sorry

#check exam_students_count

end NUMINAMATH_CALUDE_exam_students_count_l586_58630


namespace NUMINAMATH_CALUDE_sum_integer_part_l586_58670

theorem sum_integer_part : ⌊(2010 : ℝ) / 1000 + (1219 : ℝ) / 100 + (27 : ℝ) / 10⌋ = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_integer_part_l586_58670


namespace NUMINAMATH_CALUDE_sine_fraction_equals_three_l586_58689

theorem sine_fraction_equals_three (d : ℝ) (h : d = π / 7) :
  (3 * Real.sin (2 * d) * Real.sin (4 * d) * Real.sin (6 * d) * Real.sin (8 * d) * Real.sin (10 * d)) /
  (Real.sin d * Real.sin (2 * d) * Real.sin (3 * d) * Real.sin (4 * d) * Real.sin (5 * d)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sine_fraction_equals_three_l586_58689


namespace NUMINAMATH_CALUDE_problem_statement_l586_58645

theorem problem_statement (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = -1) :
  a^3 / (b - c)^2 + b^3 / (c - a)^2 + c^3 / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l586_58645


namespace NUMINAMATH_CALUDE_recurrence_relations_hold_l586_58642

def circle_radius : ℝ := 1

def perimeter_circumscribed (n : ℕ) : ℝ := sorry

def perimeter_inscribed (n : ℕ) : ℝ := sorry

theorem recurrence_relations_hold (n : ℕ) (h : n ≥ 3) :
  perimeter_circumscribed (2 * n) = (2 * perimeter_circumscribed n * perimeter_inscribed n) / (perimeter_circumscribed n + perimeter_inscribed n) ∧
  perimeter_inscribed (2 * n) = Real.sqrt (perimeter_inscribed n * perimeter_circumscribed (2 * n)) :=
sorry

end NUMINAMATH_CALUDE_recurrence_relations_hold_l586_58642


namespace NUMINAMATH_CALUDE_bouncing_ball_original_height_l586_58634

/-- Represents the behavior of a bouncing ball -/
def BouncingBall (originalHeight : ℝ) : Prop :=
  let reboundFactor := (1/2 : ℝ)
  let totalTravel := originalHeight +
                     2 * (reboundFactor * originalHeight) +
                     2 * (reboundFactor^2 * originalHeight)
  totalTravel = 250

/-- Theorem stating the original height of the ball -/
theorem bouncing_ball_original_height :
  ∃ (h : ℝ), BouncingBall h ∧ h = 100 := by
  sorry

end NUMINAMATH_CALUDE_bouncing_ball_original_height_l586_58634


namespace NUMINAMATH_CALUDE_hilt_water_fountain_trips_l586_58653

/-- The number of times Mrs. Hilt will go to the water fountain -/
def water_fountain_trips (distance_to_fountain : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance / (2 * distance_to_fountain)

/-- Theorem: Mrs. Hilt will go to the water fountain 2 times -/
theorem hilt_water_fountain_trips :
  water_fountain_trips 30 120 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hilt_water_fountain_trips_l586_58653


namespace NUMINAMATH_CALUDE_inverse_of_twelve_point_five_l586_58615

theorem inverse_of_twelve_point_five (x : ℝ) : 1 / x = 12.5 → x = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_twelve_point_five_l586_58615


namespace NUMINAMATH_CALUDE_sheela_monthly_income_l586_58655

theorem sheela_monthly_income (deposit : ℝ) (percentage : ℝ) (monthly_income : ℝ) :
  deposit = 4500 →
  percentage = 28 →
  deposit = percentage / 100 * monthly_income →
  monthly_income = 16071.43 := by
  sorry

end NUMINAMATH_CALUDE_sheela_monthly_income_l586_58655


namespace NUMINAMATH_CALUDE_parallelogram_area_l586_58669

/-- The area of a parallelogram with base 30 cm and height 12 cm is 360 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 30 → 
  height = 12 → 
  area = base * height →
  area = 360 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l586_58669


namespace NUMINAMATH_CALUDE_nh4cl_molecular_weight_l586_58638

/-- The molecular weight of NH4Cl in grams per mole -/
def molecular_weight_NH4Cl : ℝ := 53

/-- The number of moles given in the problem -/
def moles : ℝ := 8

/-- The total weight of the given moles of NH4Cl in grams -/
def total_weight : ℝ := 424

/-- Theorem: The molecular weight of NH4Cl is 53 grams/mole -/
theorem nh4cl_molecular_weight :
  molecular_weight_NH4Cl = total_weight / moles :=
by sorry

end NUMINAMATH_CALUDE_nh4cl_molecular_weight_l586_58638


namespace NUMINAMATH_CALUDE_quadratic_roots_l586_58665

theorem quadratic_roots : ∀ x : ℝ, x^2 - 49 = 0 ↔ x = 7 ∨ x = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l586_58665


namespace NUMINAMATH_CALUDE_sin_30_deg_value_l586_58652

theorem sin_30_deg_value (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (3 * x)) :
  f (Real.sin (π / 6)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_deg_value_l586_58652


namespace NUMINAMATH_CALUDE_difference_second_third_bus_l586_58694

/-- The number of buses hired for the school trip -/
def num_buses : ℕ := 4

/-- The number of people on the first bus -/
def first_bus : ℕ := 12

/-- The number of people on the second bus -/
def second_bus : ℕ := 2 * first_bus

/-- The number of people on the fourth bus -/
def fourth_bus : ℕ := first_bus + 9

/-- The total number of people going to the museum -/
def total_people : ℕ := 75

/-- The number of people on the third bus -/
def third_bus : ℕ := total_people - (first_bus + second_bus + fourth_bus)

theorem difference_second_third_bus : second_bus - third_bus = 6 := by
  sorry

end NUMINAMATH_CALUDE_difference_second_third_bus_l586_58694


namespace NUMINAMATH_CALUDE_work_completion_time_l586_58691

theorem work_completion_time (a_time b_time b_remaining : ℚ) 
  (ha : a_time = 45)
  (hb : b_time = 40)
  (hc : b_remaining = 23) : 
  let x := (b_time * b_remaining * a_time - a_time * b_time) / (a_time * b_time + a_time * b_remaining)
  x = 9 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l586_58691


namespace NUMINAMATH_CALUDE_gcd_problem_l586_58682

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = 2345 * k) :
  Int.gcd (a^2 + 10*a + 25) (a + 5) = a + 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l586_58682


namespace NUMINAMATH_CALUDE_largest_expression_l586_58693

theorem largest_expression : 
  let a := 1 - 2 + 3 + 4
  let b := 1 + 2 - 3 + 4
  let c := 1 + 2 + 3 - 4
  let d := 1 + 2 - 3 - 4
  let e := 1 - 2 - 3 + 4
  (a ≥ b) ∧ (a ≥ c) ∧ (a ≥ d) ∧ (a ≥ e) := by
  sorry

#eval (1 - 2 + 3 + 4)
#eval (1 + 2 - 3 + 4)
#eval (1 + 2 + 3 - 4)
#eval (1 + 2 - 3 - 4)
#eval (1 - 2 - 3 + 4)

end NUMINAMATH_CALUDE_largest_expression_l586_58693


namespace NUMINAMATH_CALUDE_negation_of_existential_real_exp_l586_58671

theorem negation_of_existential_real_exp (p : Prop) : 
  (p ↔ ∃ x : ℝ, Real.exp x < 0) → 
  (¬p ↔ ∀ x : ℝ, Real.exp x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_real_exp_l586_58671


namespace NUMINAMATH_CALUDE_calculation_proof_l586_58664

theorem calculation_proof : (35 / (8 + 3 - 5) - 2) * 4 = 46 / 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l586_58664


namespace NUMINAMATH_CALUDE_grid_black_probability_l586_58640

/-- Represents a 4x4 grid where each cell can be either black or white -/
def Grid := Fin 4 → Fin 4 → Bool

/-- The probability of a single cell being black initially -/
def initial_black_prob : ℚ := 1/2

/-- Rotates the grid 90 degrees clockwise -/
def rotate (g : Grid) : Grid := sorry

/-- Applies the repainting rule after rotation -/
def repaint (g : Grid) : Grid := sorry

/-- The probability that the entire grid becomes black after rotation and repainting -/
def prob_all_black_after_process : ℚ := sorry

/-- Theorem stating the probability of the grid becoming entirely black -/
theorem grid_black_probability : 
  prob_all_black_after_process = 1 / 65536 := by sorry

end NUMINAMATH_CALUDE_grid_black_probability_l586_58640


namespace NUMINAMATH_CALUDE_person_age_puzzle_l586_58673

theorem person_age_puzzle : ∃ (x : ℝ), x > 0 ∧ x = 4 * (x + 4) - 4 * (x - 4) + (1/2) * (x - 6) ∧ x = 58 := by
  sorry

end NUMINAMATH_CALUDE_person_age_puzzle_l586_58673


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l586_58629

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and eccentricity √3,
    its asymptotes are given by y = ±√2 x -/
theorem hyperbola_asymptotes (a b : ℝ) (h : a > 0) (k : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  ((a^2 + b^2) / a^2 = 3) →
  (∃ c : ℝ, ∀ x : ℝ, (y = c * x ∨ y = -c * x) ↔ (x / a = y / b ∨ x / a = -y / b)) ∧
  c = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l586_58629


namespace NUMINAMATH_CALUDE_water_evaporation_rate_l586_58687

/-- Given a glass with 10 ounces of water, with 6% evaporating over 20 days,
    the amount of water evaporating each day is 0.03 ounces. -/
theorem water_evaporation_rate (initial_water : ℝ) (days : ℕ) (evaporation_percentage : ℝ) :
  initial_water = 10 →
  days = 20 →
  evaporation_percentage = 0.06 →
  (initial_water * evaporation_percentage) / days = 0.03 :=
by sorry

end NUMINAMATH_CALUDE_water_evaporation_rate_l586_58687


namespace NUMINAMATH_CALUDE_age_problem_l586_58621

theorem age_problem (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  ∃ k : ℤ, (b - 1) / (a - 1) = k ∧ (b + 1) / (a + 1) = k + 1 →
  ∃ m : ℤ, (c - 1) / (b - 1) = m ∧ (c + 1) / (b + 1) = m + 1 →
  a + b + c ≤ 150 →
  a = 2 ∧ b = 7 ∧ c = 49 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l586_58621


namespace NUMINAMATH_CALUDE_conference_hall_tables_l586_58618

theorem conference_hall_tables (chairs_per_table : ℕ) (chair_legs : ℕ) (table_legs : ℕ) (total_legs : ℕ) :
  chairs_per_table = 8 →
  chair_legs = 3 →
  table_legs = 5 →
  total_legs = 580 →
  ∃ (num_tables : ℕ), num_tables = 20 ∧ 
    chairs_per_table * num_tables * chair_legs + num_tables * table_legs = total_legs :=
by sorry

end NUMINAMATH_CALUDE_conference_hall_tables_l586_58618


namespace NUMINAMATH_CALUDE_value_of_expression_l586_58632

theorem value_of_expression (a : ℝ) (h : a^2 + 2*a + 1 = 0) : 2*a^2 + 4*a - 3 = -5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l586_58632


namespace NUMINAMATH_CALUDE_log_base_1024_integer_count_l586_58605

theorem log_base_1024_integer_count : 
  ∃! (S : Finset ℕ), 
    (∀ b ∈ S, b > 0 ∧ ∃ n : ℕ, n > 0 ∧ b ^ n = 1024) ∧ 
    (∀ b : ℕ, b > 0 → (∃ n : ℕ, n > 0 ∧ b ^ n = 1024) → b ∈ S) ∧
    S.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_log_base_1024_integer_count_l586_58605


namespace NUMINAMATH_CALUDE_sqrt_x_squared_plus_two_is_quadratic_radical_l586_58635

-- Define what it means for an expression to be a quadratic radical
def is_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ y : ℝ, f x = y ∧ y ≥ 0

-- Theorem statement
theorem sqrt_x_squared_plus_two_is_quadratic_radical :
  is_quadratic_radical (λ x : ℝ => Real.sqrt (x^2 + 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_squared_plus_two_is_quadratic_radical_l586_58635


namespace NUMINAMATH_CALUDE_divisibility_check_l586_58663

theorem divisibility_check : 
  (5641713 % 29 ≠ 0) ∧ (1379235 % 11 = 0) := by sorry

end NUMINAMATH_CALUDE_divisibility_check_l586_58663


namespace NUMINAMATH_CALUDE_impossible_transformation_l586_58607

/-- Represents the color and position of a token -/
inductive Token
  | Red : Token
  | BlueEven : Token
  | BlueOdd : Token

/-- Converts a token to its numeric representation -/
def tokenValue : Token → Int
  | Token.Red => 0
  | Token.BlueEven => 1
  | Token.BlueOdd => -1

/-- Represents the state of the line as a list of tokens -/
def Line := List Token

/-- Calculates the sum of the numeric representations of tokens in a line -/
def lineSum (l : Line) : Int :=
  l.map tokenValue |>.sum

/-- Represents a valid operation on the line -/
inductive Operation
  | Insert : Token → Token → Operation
  | Remove : Token → Token → Operation

/-- Applies an operation to a line -/
def applyOperation (l : Line) (op : Operation) : Line :=
  match op with
  | Operation.Insert t1 t2 => sorry
  | Operation.Remove t1 t2 => sorry

/-- Theorem: It's impossible to transform the initial state to the desired final state -/
theorem impossible_transformation : ∀ (ops : List Operation),
  let initial : Line := [Token.Red, Token.BlueEven]
  let final : Line := [Token.BlueOdd, Token.Red]
  (lineSum initial = lineSum (ops.foldl applyOperation initial)) ∧
  (ops.foldl applyOperation initial ≠ final) := by
  sorry

end NUMINAMATH_CALUDE_impossible_transformation_l586_58607


namespace NUMINAMATH_CALUDE_correct_sums_count_l586_58699

theorem correct_sums_count (total : ℕ) (correct : ℕ) (incorrect : ℕ) : 
  total = 75 → 
  incorrect = 2 * correct → 
  total = correct + incorrect →
  correct = 25 := by
sorry

end NUMINAMATH_CALUDE_correct_sums_count_l586_58699


namespace NUMINAMATH_CALUDE_diseased_corn_plants_l586_58611

theorem diseased_corn_plants (grid_size : Nat) (h : grid_size = 2015) :
  let center := grid_size / 2 + 1
  let days_to_corner := center - 1
  days_to_corner * 2 = 2014 :=
sorry

end NUMINAMATH_CALUDE_diseased_corn_plants_l586_58611


namespace NUMINAMATH_CALUDE_line_segments_proportion_l586_58678

theorem line_segments_proportion : 
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := 2
  let d : ℝ := 4
  (a / b = c / d) := by sorry

end NUMINAMATH_CALUDE_line_segments_proportion_l586_58678


namespace NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l586_58619

theorem quadratic_inequality_no_solution (m : ℝ) (h : m ≤ 1) :
  ¬∃ x : ℝ, x^2 + 2*x + 2 - m < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l586_58619


namespace NUMINAMATH_CALUDE_billion_yuan_scientific_notation_l586_58690

/-- Represents the value of 209.6 billion yuan in standard form -/
def billion_yuan : ℝ := 209.6 * (10^9)

/-- Represents the scientific notation of 209.6 billion yuan -/
def scientific_notation : ℝ := 2.096 * (10^10)

/-- Theorem stating that the standard form equals the scientific notation -/
theorem billion_yuan_scientific_notation : billion_yuan = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_billion_yuan_scientific_notation_l586_58690


namespace NUMINAMATH_CALUDE_f_expression_f_range_l586_58674

/-- A quadratic function f satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The property that f(x+1) - f(x) = 2x -/
axiom f_diff (x : ℝ) : f (x + 1) - f x = 2 * x

/-- The property that f(0) = 1 -/
axiom f_zero : f 0 = 1

/-- Theorem: The analytical expression of f(x) -/
theorem f_expression (x : ℝ) : f x = x^2 - x + 1 := sorry

/-- Theorem: The range of f(x) when x ∈ [-1, 1] -/
theorem f_range : Set.Icc (3/4 : ℝ) 3 = { y | ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = y } := sorry

end NUMINAMATH_CALUDE_f_expression_f_range_l586_58674


namespace NUMINAMATH_CALUDE_harkamal_payment_l586_58685

/-- The amount Harkamal paid to the shopkeeper -/
def total_amount (grape_quantity grape_rate mango_quantity mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem: Given the conditions, Harkamal paid 1010 to the shopkeeper -/
theorem harkamal_payment : total_amount 8 70 9 50 = 1010 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_payment_l586_58685


namespace NUMINAMATH_CALUDE_n_squared_divisible_by_144_l586_58603

theorem n_squared_divisible_by_144 (n : ℕ+) (h : ∃ t : ℕ+, t = 12 ∧ ∀ k : ℕ+, k ∣ n → k ≤ t) :
  144 ∣ n^2 := by
  sorry

end NUMINAMATH_CALUDE_n_squared_divisible_by_144_l586_58603


namespace NUMINAMATH_CALUDE_circle_radius_is_three_l586_58602

/-- Given a circle where the product of three inches and its circumference
    is twice its area, prove that its radius is 3 inches. -/
theorem circle_radius_is_three (r : ℝ) (h : 3 * (2 * π * r) = 2 * (π * r^2)) : r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_three_l586_58602


namespace NUMINAMATH_CALUDE_vasya_lives_on_fifth_floor_l586_58612

/-- The number of steps Petya walks from the first to the third floor -/
def petya_steps : ℕ := 36

/-- The number of steps Vasya walks from the first floor to his floor -/
def vasya_steps : ℕ := 72

/-- The floor on which Vasya lives -/
def vasya_floor : ℕ := 5

/-- Theorem stating that Vasya lives on the 5th floor given the conditions -/
theorem vasya_lives_on_fifth_floor :
  (petya_steps / 2 = vasya_steps / vasya_floor) →
  vasya_floor = 5 := by
  sorry

end NUMINAMATH_CALUDE_vasya_lives_on_fifth_floor_l586_58612


namespace NUMINAMATH_CALUDE_roots_star_zero_l586_58623

-- Define the new operation ※
def star (a b : ℝ) : ℝ := a * b - a - b

-- Define the theorem
theorem roots_star_zero {x₁ x₂ : ℝ} (h : x₁^2 + x₁ - 1 = 0 ∧ x₂^2 + x₂ - 1 = 0) : 
  star x₁ x₂ = 0 := by
  sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_roots_star_zero_l586_58623


namespace NUMINAMATH_CALUDE_pure_imaginary_quotient_l586_58643

theorem pure_imaginary_quotient (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (∃ (b : ℝ), z₁ / z₂ = b*I) → a = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_quotient_l586_58643


namespace NUMINAMATH_CALUDE_quadratic_properties_l586_58606

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- State the theorem
theorem quadratic_properties :
  (∀ x, f x ≥ -4) ∧  -- Minimum value is -4
  (f (-1) = -4) ∧    -- Minimum occurs at x = -1
  (f 0 = -3) ∧       -- Passes through (0, -3)
  (f 1 = 0) ∧        -- Intersects x-axis at (1, 0)
  (f (-3) = 0) ∧     -- Intersects x-axis at (-3, 0)
  (∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≤ 5) ∧  -- Maximum value in [-2, 2] is 5
  (f 2 = 5)  -- Maximum value occurs at x = 2
  := by sorry


end NUMINAMATH_CALUDE_quadratic_properties_l586_58606


namespace NUMINAMATH_CALUDE_decimal_point_problem_l586_58683

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 / x) : x = Real.sqrt 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l586_58683


namespace NUMINAMATH_CALUDE_total_third_grade_students_l586_58622

theorem total_third_grade_students : 
  let class_a : ℕ := 48
  let class_b : ℕ := 65
  let class_c : ℕ := 57
  let class_d : ℕ := 72
  class_a + class_b + class_c + class_d = 242 := by
sorry

end NUMINAMATH_CALUDE_total_third_grade_students_l586_58622


namespace NUMINAMATH_CALUDE_sector_max_area_l586_58631

/-- Given a sector with perimeter 40, its maximum area is 100 -/
theorem sector_max_area (r l : ℝ) (h_perimeter : 2 * r + l = 40) :
  (1 / 2) * l * r ≤ 100 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_l586_58631


namespace NUMINAMATH_CALUDE_four_digit_int_problem_l586_58677

/-- Represents a four-digit positive integer -/
structure FourDigitInt where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_pos : a > 0
  a_lt_10 : a < 10
  b_lt_10 : b < 10
  c_lt_10 : c < 10
  d_lt_10 : d < 10

/-- Converts a FourDigitInt to a natural number -/
def FourDigitInt.toNat (n : FourDigitInt) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

theorem four_digit_int_problem (n : FourDigitInt) 
  (h1 : n.a + n.b + n.c + n.d = 18)
  (h2 : n.b + n.c = 11)
  (h3 : n.a - n.d = 3)
  (h4 : n.toNat % 9 = 0) :
  n.toNat = 5472 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_int_problem_l586_58677


namespace NUMINAMATH_CALUDE_max_candies_l586_58604

theorem max_candies (vitya maria sasha : ℕ) : 
  vitya = 35 →
  maria < vitya →
  sasha = vitya + maria →
  Even sasha →
  vitya + maria + sasha ≤ 136 :=
by sorry

end NUMINAMATH_CALUDE_max_candies_l586_58604


namespace NUMINAMATH_CALUDE_square_region_perimeter_l586_58658

theorem square_region_perimeter (area : ℝ) (num_squares : ℕ) (rows : ℕ) (cols : ℕ) :
  area = 392 →
  num_squares = 8 →
  rows = 2 →
  cols = 4 →
  let side_length := Real.sqrt (area / num_squares)
  let perimeter := 2 * (rows * side_length + cols * side_length)
  perimeter = 126 := by
  sorry

end NUMINAMATH_CALUDE_square_region_perimeter_l586_58658


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l586_58659

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 + 5*x - 24 = 4*x + 38) → 
  (∃ a b : ℝ, (a + b = -1) ∧ (x = a ∨ x = b)) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l586_58659


namespace NUMINAMATH_CALUDE_tangent_line_curve1_tangent_lines_curve2_l586_58686

-- Define the curves
def curve1 (x : ℝ) : ℝ := x^3 + x^2 + 1
def curve2 (x : ℝ) : ℝ := x^2

-- Define the points
def P1 : ℝ × ℝ := (-1, 1)
def P2 : ℝ × ℝ := (3, 5)

-- Theorem for the first curve
theorem tangent_line_curve1 :
  ∃ (k m : ℝ), k * P1.1 + m * P1.2 + 2 = 0 ∧
  ∀ x y, y = curve1 x → k * x + m * y + 2 = 0 → x = P1.1 ∧ y = P1.2 :=
sorry

-- Theorem for the second curve
theorem tangent_lines_curve2 :
  ∃ (k1 m1 k2 m2 : ℝ),
  (k1 * P2.1 + m1 * P2.2 + 1 = 0 ∧ k2 * P2.1 + m2 * P2.2 + 25 = 0) ∧
  (∀ x y, y = curve2 x → (k1 * x + m1 * y + 1 = 0 ∨ k2 * x + m2 * y + 25 = 0) → x = P2.1 ∧ y = P2.2) ∧
  (k1 = 2 ∧ m1 = -1) ∧ (k2 = 10 ∧ m2 = -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_curve1_tangent_lines_curve2_l586_58686


namespace NUMINAMATH_CALUDE_calculate_expression_l586_58650

theorem calculate_expression : 500 * 997 * 0.0997 * (10^2) = 5 * 997^2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l586_58650


namespace NUMINAMATH_CALUDE_circle_equation_l586_58609

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (h : ℝ), (p.1 - h)^2 + p.2^2 = (h - 1)^2 + 1^2}

-- Define points A and B
def point_A : ℝ × ℝ := (5, 2)
def point_B : ℝ × ℝ := (-1, 4)

-- Theorem statement
theorem circle_equation :
  (∀ p : ℝ × ℝ, p ∈ circle_C ↔ (p.1 - 1)^2 + p.2^2 = 20) ∧
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  (∃ h : ℝ, ∀ p : ℝ × ℝ, p ∈ circle_C → p.2 = 0 → p.1 = h) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l586_58609


namespace NUMINAMATH_CALUDE_six_digit_numbers_count_l586_58639

/-- The number of ways to choose 2 items from 4 items -/
def choose_4_2 : ℕ := 6

/-- The number of ways to choose 1 item from 2 items -/
def choose_2_1 : ℕ := 2

/-- The number of ways to arrange 3 items -/
def arrange_3_3 : ℕ := 6

/-- The number of ways to choose 2 positions from 4 positions -/
def insert_2_in_4 : ℕ := 6

/-- The total number of valid six-digit numbers -/
def total_numbers : ℕ := choose_4_2 * choose_2_1 * arrange_3_3 * insert_2_in_4

theorem six_digit_numbers_count : total_numbers = 432 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_count_l586_58639


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l586_58641

/-- Given a line 2ax + by - 2 = 0 where a > 0 and b > 0, and the line passes through the point (1, 2),
    the minimum value of 1/a + 1/b is 4. -/
theorem min_value_sum_reciprocals (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + b*2 = 2 → (∀ x y : ℝ, x > 0 → y > 0 → 2*x + y*2 = 2 → 1/a + 1/b ≤ 1/x + 1/y) → 
  1/a + 1/b = 4 := by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l586_58641


namespace NUMINAMATH_CALUDE_regression_lines_common_point_l586_58696

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The average point of a dataset -/
structure AveragePoint where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a regression line -/
def pointOnLine (l : RegressionLine) (p : AveragePoint) : Prop :=
  p.y = l.slope * p.x + l.intercept

theorem regression_lines_common_point 
  (l₁ l₂ : RegressionLine) (avg : AveragePoint) : 
  pointOnLine l₁ avg ∧ pointOnLine l₂ avg := by
  sorry

#check regression_lines_common_point

end NUMINAMATH_CALUDE_regression_lines_common_point_l586_58696


namespace NUMINAMATH_CALUDE_apple_consumption_l586_58647

theorem apple_consumption (x : ℝ) : 
  x > 0 ∧ x + 2*x + x/2 = 14 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_apple_consumption_l586_58647


namespace NUMINAMATH_CALUDE_stock_price_increase_l586_58637

theorem stock_price_increase (initial_price : ℝ) (h : initial_price > 0) : 
  let price_after_year1 := initial_price * 1.2
  let price_after_year2 := price_after_year1 * 0.75
  let price_after_year3 := initial_price * 1.26
  (price_after_year3 / price_after_year2 - 1) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_stock_price_increase_l586_58637


namespace NUMINAMATH_CALUDE_quadratic_root_difference_condition_l586_58695

/-- For a quadratic equation x^2 + px + q = 0, 
    the condition for the difference of its roots to be 'a' is a^2 - p^2 = -4q -/
theorem quadratic_root_difference_condition 
  (p q a : ℝ) 
  (hq : ∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ x₁ ≠ x₂) :
  (∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ x₁ - x₂ = a) ↔ 
  a^2 - p^2 = -4*q :=
by sorry


end NUMINAMATH_CALUDE_quadratic_root_difference_condition_l586_58695


namespace NUMINAMATH_CALUDE_residue_neg_1234_mod_31_l586_58601

theorem residue_neg_1234_mod_31 : Int.mod (-1234) 31 = 6 := by
  sorry

end NUMINAMATH_CALUDE_residue_neg_1234_mod_31_l586_58601


namespace NUMINAMATH_CALUDE_parabola_and_line_equations_l586_58688

/-- Parabola with focus F and point (3,m) on it -/
structure Parabola where
  p : ℝ
  m : ℝ
  h_p_pos : p > 0
  h_on_parabola : m^2 = 2 * p * 3
  h_distance_to_focus : Real.sqrt ((3 - p/2)^2 + m^2) = 4

/-- Line passing through focus F and intersecting parabola at A and B -/
structure IntersectingLine (E : Parabola) where
  k : ℝ  -- slope of the line
  h_midpoint : ∃ (y_A y_B : ℝ), y_A^2 = 4 * (k * y_A + 1) ∧
                                 y_B^2 = 4 * (k * y_B + 1) ∧
                                 (y_A + y_B) / 2 = -1

/-- Main theorem -/
theorem parabola_and_line_equations (E : Parabola) (l : IntersectingLine E) :
  (E.p = 2 ∧ ∀ x y, y^2 = 2 * E.p * x ↔ y^2 = 4 * x) ∧
  (l.k = -1/2 ∧ ∀ x y, y = l.k * (x - 1) ↔ 2 * x + y - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_and_line_equations_l586_58688


namespace NUMINAMATH_CALUDE_mushroom_ratio_l586_58627

theorem mushroom_ratio (total : ℕ) (safe : ℕ) (uncertain : ℕ) 
  (h1 : total = 32) 
  (h2 : safe = 9) 
  (h3 : uncertain = 5) : 
  (total - safe - uncertain) / safe = 2 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_ratio_l586_58627


namespace NUMINAMATH_CALUDE_a_8_equals_16_l586_58684

def sequence_property (a : ℕ+ → ℕ) : Prop :=
  ∀ p q : ℕ+, a (p + q) = a p * a q

theorem a_8_equals_16 (a : ℕ+ → ℕ) (h1 : sequence_property a) (h2 : a 2 = 2) :
  a 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_a_8_equals_16_l586_58684


namespace NUMINAMATH_CALUDE_distance_to_grandmas_house_l586_58649

-- Define the car's efficiency in miles per gallon
def car_efficiency : ℝ := 20

-- Define the amount of gas needed to reach Grandma's house in gallons
def gas_needed : ℝ := 5

-- Theorem to prove the distance to Grandma's house
theorem distance_to_grandmas_house : car_efficiency * gas_needed = 100 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_grandmas_house_l586_58649


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l586_58661

/-- The equation 9x^2 + nx + 1 = 0 has exactly one solution in x if and only if n = 6 -/
theorem unique_solution_quadratic (n : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + n * x + 1 = 0) ↔ n = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l586_58661


namespace NUMINAMATH_CALUDE_regular_polygon_2022_probability_l586_58625

/-- A regular polygon with 2022 sides -/
structure RegularPolygon2022 where
  area : ℝ
  sides : Nat
  is_regular : sides = 2022

/-- A point on the perimeter of a polygon -/
structure PerimeterPoint (P : RegularPolygon2022) where
  x : ℝ
  y : ℝ
  on_perimeter : True  -- This is a placeholder for the actual condition

/-- The distance between two points -/
def distance (A B : PerimeterPoint P) : ℝ := sorry

/-- The probability of an event -/
def probability (event : Prop) : ℝ := sorry

theorem regular_polygon_2022_probability 
  (P : RegularPolygon2022) 
  (h : P.area = 1) :
  probability (
    ∀ (A B : PerimeterPoint P), 
    distance A B ≥ Real.sqrt (2 / Real.pi)
  ) = 1/2 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_2022_probability_l586_58625


namespace NUMINAMATH_CALUDE_negation_of_all_seated_l586_58672

universe u

-- Define the predicates
variable (in_room : α → Prop)
variable (seated : α → Prop)

-- State the theorem
theorem negation_of_all_seated :
  ¬(∀ (x : α), in_room x → seated x) ↔ ∃ (x : α), in_room x ∧ ¬(seated x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_seated_l586_58672


namespace NUMINAMATH_CALUDE_sum_parts_is_24_l586_58660

/-- A rectangular prism with two opposite corners colored red -/
structure ColoredRectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ
  red_corners : ℕ
  h_red_corners : red_corners = 2

/-- The sum of edges, non-red corners, and faces of a colored rectangular prism -/
def sum_parts (prism : ColoredRectangularPrism) : ℕ :=
  12 + (8 - prism.red_corners) + 6

theorem sum_parts_is_24 (prism : ColoredRectangularPrism) :
  sum_parts prism = 24 :=
sorry

end NUMINAMATH_CALUDE_sum_parts_is_24_l586_58660


namespace NUMINAMATH_CALUDE_elise_comic_book_cost_l586_58668

/-- Calculates the amount spent on a comic book given initial money, saved money, puzzle cost, and final money --/
def comic_book_cost (initial_money saved_money puzzle_cost final_money : ℕ) : ℕ :=
  initial_money + saved_money - puzzle_cost - final_money

/-- Proves that Elise spent $2 on the comic book --/
theorem elise_comic_book_cost :
  comic_book_cost 8 13 18 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_elise_comic_book_cost_l586_58668


namespace NUMINAMATH_CALUDE_probability_at_least_two_defective_l586_58633

/-- The probability of selecting at least 2 defective items from a batch of products -/
theorem probability_at_least_two_defective (total : Nat) (good : Nat) (defective : Nat) 
  (selected : Nat) (h1 : total = good + defective) (h2 : total = 10) (h3 : good = 6) 
  (h4 : defective = 4) (h5 : selected = 3) : 
  (Nat.choose defective 2 * Nat.choose good 1 + Nat.choose defective 3) / 
  Nat.choose total selected = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_defective_l586_58633


namespace NUMINAMATH_CALUDE_lake_shore_distance_l586_58667

/-- Given two points A and B on the shore of a lake, and a point C chosen such that
    CA = 50 meters, CB = 30 meters, and ∠ACB = 120°, prove that the distance AB is 70 meters. -/
theorem lake_shore_distance (A B C : ℝ × ℝ) : 
  let CA := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let CB := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let cos_ACB := ((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2)) / (CA * CB)
  CA = 50 ∧ CB = 30 ∧ cos_ACB = -1/2 → AB = 70 := by
  sorry


end NUMINAMATH_CALUDE_lake_shore_distance_l586_58667


namespace NUMINAMATH_CALUDE_parallel_vectors_angle_l586_58616

theorem parallel_vectors_angle (x : Real) : 
  let a : ℝ × ℝ := (Real.sin x, 3/4)
  let b : ℝ × ℝ := (1/3, (1/2) * Real.cos x)
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → 
  0 < x ∧ x < π/2 → 
  x = π/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_angle_l586_58616


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_42_l586_58651

theorem no_primes_divisible_by_42 : 
  ∀ p : ℕ, Prime p → ¬(42 ∣ p) :=
by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_42_l586_58651


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_l586_58675

theorem least_positive_linear_combination : 
  ∃ (n : ℕ), n > 0 ∧ (∀ (m : ℤ), (∃ (x y : ℤ), m = 24*x + 16*y) → m = 0 ∨ m.natAbs ≥ n) ∧ 
  (∃ (x y : ℤ), n = 24*x + 16*y) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_l586_58675


namespace NUMINAMATH_CALUDE_negation_equivalence_l586_58600

theorem negation_equivalence :
  (¬ ∃ a : ℝ, a < 0 ∧ a + 4 / a ≤ -4) ↔ (∀ a : ℝ, a < 0 → a + 4 / a > -4) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l586_58600


namespace NUMINAMATH_CALUDE_no_simultaneous_integer_fractions_l586_58666

theorem no_simultaneous_integer_fractions :
  ¬ ∃ (n : ℤ), (∃ (k m : ℤ), (n - 6 : ℚ) / 15 = k ∧ (n - 5 : ℚ) / 24 = m) :=
by sorry

end NUMINAMATH_CALUDE_no_simultaneous_integer_fractions_l586_58666


namespace NUMINAMATH_CALUDE_parking_lot_problem_l586_58620

theorem parking_lot_problem :
  let total_cars : ℝ := 300
  let valid_ticket_ratio : ℝ := 0.75
  let permanent_pass_ratio : ℝ := 0.2
  let unpaid_cars : ℝ := 30
  valid_ticket_ratio * total_cars +
  permanent_pass_ratio * (valid_ticket_ratio * total_cars) +
  unpaid_cars = total_cars :=
by sorry

end NUMINAMATH_CALUDE_parking_lot_problem_l586_58620


namespace NUMINAMATH_CALUDE_max_blocks_in_box_l586_58657

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of blocks that can fit in a box -/
def maxBlocks (box : BoxDimensions) (block : BlockDimensions) : ℕ :=
  sorry

/-- The theorem stating the maximum number of blocks that can fit in the given box -/
theorem max_blocks_in_box :
  let box := BoxDimensions.mk 4 3 2
  let block := BlockDimensions.mk 3 1 1
  maxBlocks box block = 6 :=
sorry

end NUMINAMATH_CALUDE_max_blocks_in_box_l586_58657


namespace NUMINAMATH_CALUDE_hamburger_meat_price_per_pound_l586_58608

/-- Given the following grocery items and their prices:
    - 2 pounds of hamburger meat (price unknown)
    - 1 pack of hamburger buns for $1.50
    - A head of lettuce for $1.00
    - A 1.5-pound tomato priced at $2.00 per pound
    - A jar of pickles that cost $2.50 with a $1.00 off coupon
    And given that Lauren paid with a $20 bill and got $6 change back,
    prove that the price per pound of hamburger meat is $3.50. -/
theorem hamburger_meat_price_per_pound
  (hamburger_meat_weight : ℝ)
  (buns_price : ℝ)
  (lettuce_price : ℝ)
  (tomato_weight : ℝ)
  (tomato_price_per_pound : ℝ)
  (pickles_price : ℝ)
  (pickles_discount : ℝ)
  (paid_amount : ℝ)
  (change_amount : ℝ)
  (h1 : hamburger_meat_weight = 2)
  (h2 : buns_price = 1.5)
  (h3 : lettuce_price = 1)
  (h4 : tomato_weight = 1.5)
  (h5 : tomato_price_per_pound = 2)
  (h6 : pickles_price = 2.5)
  (h7 : pickles_discount = 1)
  (h8 : paid_amount = 20)
  (h9 : change_amount = 6) :
  (paid_amount - change_amount - (buns_price + lettuce_price + tomato_weight * tomato_price_per_pound + pickles_price - pickles_discount)) / hamburger_meat_weight = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_meat_price_per_pound_l586_58608


namespace NUMINAMATH_CALUDE_equation_represents_line_and_hyperbola_l586_58626

-- Define the equation
def equation (x y : ℝ) : Prop := y^6 - 6*x^6 = 3*y^2 - 8

-- Define what it means for the equation to represent a line
def represents_line (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ a b : ℝ, ∀ x y : ℝ, eq x y → y = a*x + b

-- Define what it means for the equation to represent a hyperbola
def represents_hyperbola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ a b c d e f : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a*b < 0 ∧
    ∀ x y : ℝ, eq x y → a*x^2 + b*y^2 + c*x*y + d*x + e*y + f = 0

-- Theorem statement
theorem equation_represents_line_and_hyperbola :
  represents_line equation ∧ represents_hyperbola equation :=
sorry

end NUMINAMATH_CALUDE_equation_represents_line_and_hyperbola_l586_58626


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l586_58648

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_a3 : a 3 = 1)
  (h_a5 : a 5 = 4) :
  ∃ q : ℝ, (q = 2 ∨ q = -2) ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l586_58648


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l586_58697

-- Define a parallelogram
def Parallelogram : Type := sorry

-- Define the property of having equal diagonals
def has_equal_diagonals (p : Parallelogram) : Prop := sorry

-- Define the property of diagonals bisecting each other
def diagonals_bisect_each_other (p : Parallelogram) : Prop := sorry

-- State the theorem
theorem negation_of_universal_proposition :
  (¬ ∀ p : Parallelogram, has_equal_diagonals p ∧ diagonals_bisect_each_other p) ↔
  (∃ p : Parallelogram, ¬(has_equal_diagonals p ∧ diagonals_bisect_each_other p)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l586_58697


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l586_58628

theorem circle_diameter_ratio (R S : ℝ) (hR : R > 0) (hS : S > 0)
  (h_area : π * R^2 = 0.25 * (π * S^2)) :
  2 * R = 0.5 * (2 * S) := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l586_58628


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l586_58644

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x, x ≠ -(1/a) → a*x^2 + 2*x + a > 0) ∧ 
  (∃ x, a*x^2 + 2*x + a ≤ 0) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l586_58644


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l586_58613

theorem geometric_progression_problem (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive real numbers
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- Geometric progression
  a * b * c = 64 →  -- Product is 64
  (a + b + c) / 3 = 14 / 3 →  -- Arithmetic mean is 14/3
  ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 8 ∧ b = 4 ∧ c = 2)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l586_58613


namespace NUMINAMATH_CALUDE_cone_base_radius_l586_58680

/-- Given a cone whose lateral surface is formed by a sector with radius 6cm and central angle 120°,
    the radius of the base of the cone is 2cm. -/
theorem cone_base_radius (r : ℝ) : r > 0 → 2 * π * r = 120 * π * 6 / 180 → r = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l586_58680


namespace NUMINAMATH_CALUDE_policeman_can_catch_gangster_l586_58698

/-- Represents a point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square --/
structure Square where
  sideLength : ℝ
  center : Point
  
/-- Represents the policeman --/
structure Policeman where
  position : Point
  speed : ℝ

/-- Represents the gangster --/
structure Gangster where
  position : Point
  speed : ℝ

/-- A function to check if a point is on the edge of a square --/
def isOnEdge (p : Point) (s : Square) : Prop :=
  (p.x = s.center.x - s.sideLength / 2 ∨ p.x = s.center.x + s.sideLength / 2) ∨
  (p.y = s.center.y - s.sideLength / 2 ∨ p.y = s.center.y + s.sideLength / 2)

/-- The main theorem --/
theorem policeman_can_catch_gangster 
  (s : Square) 
  (p : Policeman) 
  (g : Gangster) 
  (h1 : s.sideLength > 0)
  (h2 : p.position = s.center)
  (h3 : isOnEdge g.position s)
  (h4 : p.speed = g.speed / 2)
  (h5 : g.speed > 0) :
  ∃ (t : ℝ) (pFinal gFinal : Point), 
    t ≥ 0 ∧
    isOnEdge pFinal s ∧
    isOnEdge gFinal s ∧
    ∃ (edge : Set Point), 
      edge.Subset {p | isOnEdge p s} ∧
      pFinal ∈ edge ∧
      gFinal ∈ edge :=
by sorry

end NUMINAMATH_CALUDE_policeman_can_catch_gangster_l586_58698


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l586_58692

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l586_58692


namespace NUMINAMATH_CALUDE_log_function_k_range_l586_58636

theorem log_function_k_range (a : ℝ) (h_a : a > 0) :
  {k : ℝ | ∀ x > a, x > max a (k * a)} = {k : ℝ | -1 ≤ k ∧ k ≤ 1} := by
sorry

end NUMINAMATH_CALUDE_log_function_k_range_l586_58636


namespace NUMINAMATH_CALUDE_geometric_series_sum_l586_58676

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h : r ≠ 1) :
  let series_sum := a * (1 - r^n) / (1 - r)
  let a := (1 : ℚ) / 4
  let r := -(1 : ℚ) / 4
  let n := 5
  series_sum = 205 / 1024 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l586_58676


namespace NUMINAMATH_CALUDE_max_value_2sin_l586_58624

theorem max_value_2sin (x : ℝ) : ∃ (M : ℝ), M = 2 ∧ ∀ y : ℝ, 2 * Real.sin y ≤ M := by
  sorry

end NUMINAMATH_CALUDE_max_value_2sin_l586_58624


namespace NUMINAMATH_CALUDE_george_candy_count_l586_58617

/-- The number of bags of candy -/
def num_bags : ℕ := 8

/-- The number of candy pieces in each bag -/
def pieces_per_bag : ℕ := 81

/-- The total number of candy pieces -/
def total_pieces : ℕ := num_bags * pieces_per_bag

theorem george_candy_count : total_pieces = 648 := by
  sorry

end NUMINAMATH_CALUDE_george_candy_count_l586_58617


namespace NUMINAMATH_CALUDE_right_angled_parallelopiped_l586_58662

structure Parallelopiped where
  AB : ℝ
  AA' : ℝ

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def is_right_angled (M N P : Point) : Prop :=
  (M.x - N.x) * (P.x - N.x) + (M.y - N.y) * (P.y - N.y) + (M.z - N.z) * (P.z - N.z) = 0

theorem right_angled_parallelopiped (p : Parallelopiped) (N : Point) :
  p.AB = 12 * Real.sqrt 3 →
  p.AA' = 18 →
  N.x = 9 * Real.sqrt 3 ∧ N.y = 0 ∧ N.z = 0 →
  ∃ P : Point, P.x = 0 ∧ P.y = 0 ∧ P.z = 27 / 2 ∧
    ∀ M : Point, M.x = 12 * Real.sqrt 3 → M.z = 18 →
      is_right_angled M N P := by
  sorry

#check right_angled_parallelopiped

end NUMINAMATH_CALUDE_right_angled_parallelopiped_l586_58662
