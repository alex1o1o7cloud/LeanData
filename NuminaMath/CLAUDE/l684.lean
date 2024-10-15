import Mathlib

namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l684_68435

theorem fractional_equation_solution_range (x a : ℝ) : 
  (1 / (x + 3) - 1 = a / (x + 3)) → -- Given equation
  (x < 0) → -- Solution for x is negative
  (a > -2 ∧ a ≠ 1) -- Range of a
  :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l684_68435


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l684_68448

theorem company_picnic_attendance
  (total_employees : ℕ)
  (men_percentage : ℝ)
  (men_attendance_rate : ℝ)
  (women_attendance_rate : ℝ)
  (h1 : men_percentage = 0.55)
  (h2 : men_attendance_rate = 0.2)
  (h3 : women_attendance_rate = 0.4) :
  let women_percentage : ℝ := 1 - men_percentage
  let men_count : ℝ := men_percentage * total_employees
  let women_count : ℝ := women_percentage * total_employees
  let men_attended : ℝ := men_attendance_rate * men_count
  let women_attended : ℝ := women_attendance_rate * women_count
  let total_attended : ℝ := men_attended + women_attended
  total_attended / total_employees = 0.29 := by
sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l684_68448


namespace NUMINAMATH_CALUDE_boys_without_notebooks_l684_68471

def history_class (total_boys : ℕ) (students_with_notebooks : ℕ) (girls_with_notebooks : ℕ) : ℕ :=
  total_boys - (students_with_notebooks - girls_with_notebooks)

theorem boys_without_notebooks :
  history_class 16 20 11 = 7 :=
by sorry

end NUMINAMATH_CALUDE_boys_without_notebooks_l684_68471


namespace NUMINAMATH_CALUDE_f_strictly_decreasing_on_interval_l684_68424

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- Theorem statement
theorem f_strictly_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 3, StrictMonoOn f (Set.Ioo (-1 : ℝ) 3) := by
  sorry


end NUMINAMATH_CALUDE_f_strictly_decreasing_on_interval_l684_68424


namespace NUMINAMATH_CALUDE_triangular_gcd_bound_l684_68497

/-- The nth triangular number -/
def T (n : ℕ+) : ℕ := (n * (n + 1)) / 2

/-- Theorem: The GCD of 6T_n and n-1 is at most 3, and this bound is achievable -/
theorem triangular_gcd_bound (n : ℕ+) : 
  ∃ (m : ℕ+), Nat.gcd (6 * T m) (m - 1) = 3 ∧ 
  ∀ (k : ℕ+), Nat.gcd (6 * T k) (k - 1) ≤ 3 := by
  sorry

#check triangular_gcd_bound

end NUMINAMATH_CALUDE_triangular_gcd_bound_l684_68497


namespace NUMINAMATH_CALUDE_sin_x_plus_y_eq_one_sixth_l684_68449

theorem sin_x_plus_y_eq_one_sixth (x y : ℝ) 
  (h1 : 3 * Real.sin x + 4 * Real.cos y = 5) 
  (h2 : 4 * Real.sin y + 3 * Real.cos x = 2) : 
  Real.sin (x + y) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_sin_x_plus_y_eq_one_sixth_l684_68449


namespace NUMINAMATH_CALUDE_race_distance_l684_68466

-- Define the race distance
variable (d : ℝ)

-- Define the speeds of A, B, and C
variable (a b c : ℝ)

-- Define the conditions of the race
variable (h1 : d / a = (d - 30) / b)
variable (h2 : d / b = (d - 15) / c)
variable (h3 : d / a = (d - 40) / c)

-- The theorem to prove
theorem race_distance : d = 90 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l684_68466


namespace NUMINAMATH_CALUDE_loan_period_calculation_l684_68428

/-- The time period (in years) for which A lent money to C -/
def loan_period_C : ℚ := 2/3

theorem loan_period_calculation (principal_B principal_C total_interest : ℚ) 
  (loan_period_B interest_rate : ℚ) :
  principal_B = 5000 →
  principal_C = 3000 →
  loan_period_B = 2 →
  interest_rate = 1/10 →
  total_interest = 2200 →
  principal_B * interest_rate * loan_period_B + 
  principal_C * interest_rate * loan_period_C = total_interest :=
by sorry

end NUMINAMATH_CALUDE_loan_period_calculation_l684_68428


namespace NUMINAMATH_CALUDE_proportion_solution_l684_68452

theorem proportion_solution (x : ℝ) : (0.60 : ℝ) / x = (6 : ℝ) / 2 → x = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l684_68452


namespace NUMINAMATH_CALUDE_constant_term_expansion_l684_68408

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expansion function
def expansion_term (x : ℚ) (r k : ℕ) : ℚ :=
  (-1)^k * binomial r k * x^(r - 2*k)

-- Define the constant term of the expansion
def constant_term : ℚ :=
  1 - binomial 2 1 * binomial 4 2 + binomial 4 2 * binomial 0 0

-- Theorem statement
theorem constant_term_expansion :
  constant_term = -5 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l684_68408


namespace NUMINAMATH_CALUDE_at_most_one_solution_l684_68421

/-- The floor function, mapping a real number to its integer part -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Theorem stating that the equation ax + b⌊x⌋ - c = 0 has at most one solution -/
theorem at_most_one_solution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃! x, a * x + b * (floor x : ℝ) - c = 0 :=
sorry

end NUMINAMATH_CALUDE_at_most_one_solution_l684_68421


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_special_case_l684_68455

/-- An ellipse with focal length equal to minor axis length has eccentricity √2/2 -/
theorem ellipse_eccentricity_special_case (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a > b) (h5 : c = b) (h6 : a^2 = b^2 + c^2) : 
  (c / a) = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_special_case_l684_68455


namespace NUMINAMATH_CALUDE_multiply_divide_multiply_l684_68489

theorem multiply_divide_multiply : 8 * 7 / 8 * 7 = 49 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_multiply_l684_68489


namespace NUMINAMATH_CALUDE_initial_number_solution_l684_68469

theorem initial_number_solution : 
  ∃ x : ℤ, x - 12 * 3 * 2 = 1234490 ∧ x = 1234562 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_solution_l684_68469


namespace NUMINAMATH_CALUDE_sugar_sold_is_two_kilograms_l684_68406

/-- The number of sugar packets sold per week -/
def packets_per_week : ℕ := 20

/-- The amount of sugar in grams per packet -/
def grams_per_packet : ℕ := 100

/-- Conversion factor from grams to kilograms -/
def grams_per_kilogram : ℕ := 1000

/-- The amount of sugar sold per week in kilograms -/
def sugar_sold_per_week : ℚ :=
  (packets_per_week * grams_per_packet : ℚ) / grams_per_kilogram

theorem sugar_sold_is_two_kilograms :
  sugar_sold_per_week = 2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_sold_is_two_kilograms_l684_68406


namespace NUMINAMATH_CALUDE_truncated_cone_radii_relation_l684_68437

/-- Represents a truncated cone with given dimensions and properties -/
structure TruncatedCone where
  top_radius : ℝ
  bottom_radius : ℝ
  section_ratio : ℝ

/-- Theorem stating the relationship between the radii of a truncated cone
    given specific conditions on its section -/
theorem truncated_cone_radii_relation (cone : TruncatedCone)
  (h1 : cone.top_radius = 5)
  (h2 : cone.section_ratio = 1/2) :
  cone.bottom_radius = 25 := by
  sorry

#check truncated_cone_radii_relation

end NUMINAMATH_CALUDE_truncated_cone_radii_relation_l684_68437


namespace NUMINAMATH_CALUDE_smallest_muffin_boxes_l684_68480

theorem smallest_muffin_boxes : ∃ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), 0 < k ∧ k < n → ¬(11 ∣ (17 * k - 1))) ∧ (11 ∣ (17 * n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_muffin_boxes_l684_68480


namespace NUMINAMATH_CALUDE_ice_cream_scoop_arrangements_l684_68410

theorem ice_cream_scoop_arrangements : (Finset.range 5).card.factorial = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoop_arrangements_l684_68410


namespace NUMINAMATH_CALUDE_twentieth_term_of_arithmetic_sequence_l684_68441

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 20th term of the specified arithmetic sequence is -49. -/
theorem twentieth_term_of_arithmetic_sequence :
  ∀ a : ℕ → ℤ,
  isArithmeticSequence a →
  a 1 = 8 →
  a 2 = 5 →
  a 3 = 2 →
  a 20 = -49 := by
sorry

end NUMINAMATH_CALUDE_twentieth_term_of_arithmetic_sequence_l684_68441


namespace NUMINAMATH_CALUDE_final_bill_amount_l684_68418

def original_bill : ℝ := 500
def late_charge_rate : ℝ := 0.02

theorem final_bill_amount :
  let first_increase := original_bill * (1 + late_charge_rate)
  let final_bill := first_increase * (1 + late_charge_rate)
  final_bill = 520.20 := by sorry

end NUMINAMATH_CALUDE_final_bill_amount_l684_68418


namespace NUMINAMATH_CALUDE_sum_of_sides_equals_two_point_five_l684_68484

/-- Represents a polygon ABCDEFGH with given properties -/
structure Polygon where
  area : ℝ
  AB : ℝ
  BC : ℝ
  HA : ℝ

/-- The sum of lengths DE, EF, FG, and GH in the polygon -/
def sum_of_sides (p : Polygon) : ℝ := sorry

/-- Theorem stating that for a polygon with given properties, the sum of certain sides equals 2.5 -/
theorem sum_of_sides_equals_two_point_five (p : Polygon) 
  (h1 : p.area = 85)
  (h2 : p.AB = 7)
  (h3 : p.BC = 10)
  (h4 : p.HA = 6) :
  sum_of_sides p = 2.5 := by sorry

end NUMINAMATH_CALUDE_sum_of_sides_equals_two_point_five_l684_68484


namespace NUMINAMATH_CALUDE_inequality_implication_l684_68426

theorem inequality_implication (a b : ℝ) (h : a < b) : a - b < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l684_68426


namespace NUMINAMATH_CALUDE_power_function_through_point_and_value_l684_68493

/-- A power function that passes through the point (2,8) -/
def f (x : ℝ) : ℝ := x^3

theorem power_function_through_point_and_value : 
  f 2 = 8 ∧ ∃ x : ℝ, f x = 27 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_and_value_l684_68493


namespace NUMINAMATH_CALUDE_cars_meeting_time_l684_68403

/-- Two cars meeting on a highway -/
theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) (meeting_time : ℝ) : 
  highway_length = 60 →
  speed1 = 13 →
  speed2 = 17 →
  meeting_time * (speed1 + speed2) = highway_length →
  meeting_time = 2 := by
sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l684_68403


namespace NUMINAMATH_CALUDE_divisibility_problem_l684_68407

theorem divisibility_problem (n : ℕ) (h1 : n > 0) (h2 : (n + 1) % 6 = 4) :
  n % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l684_68407


namespace NUMINAMATH_CALUDE_least_number_with_divisibility_property_l684_68419

theorem least_number_with_divisibility_property : ∃ m : ℕ, 
  (m > 0) ∧ 
  (∀ n : ℕ, n > 0 → n < m → ¬(∃ q r : ℕ, n = 5 * q ∧ n = 34 * (q - 8) + r ∧ r < 34)) ∧
  (∃ q r : ℕ, m = 5 * q ∧ m = 34 * (q - 8) + r ∧ r < 34) ∧
  m = 162 :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_divisibility_property_l684_68419


namespace NUMINAMATH_CALUDE_child_height_calculation_l684_68456

/-- Given a child's previous height and growth, calculate the current height -/
def current_height (previous_height growth : ℝ) : ℝ :=
  previous_height + growth

/-- Theorem: The child's current height is 41.5 inches -/
theorem child_height_calculation : 
  current_height 38.5 3 = 41.5 := by
  sorry

end NUMINAMATH_CALUDE_child_height_calculation_l684_68456


namespace NUMINAMATH_CALUDE_graph_sequence_periodic_l684_68477

/-- A graph on n vertices -/
def Graph (n : ℕ) := Fin n → Fin n → Prop

/-- The rule for constructing G_(n+1) from G_n -/
def nextGraph (G : Graph n) : Graph n :=
  λ i j => ∃ k, k ≠ i ∧ k ≠ j ∧ G k i ∧ G k j

/-- The sequence of graphs -/
def graphSequence (G₀ : Graph n) : ℕ → Graph n
  | 0 => G₀
  | m + 1 => nextGraph (graphSequence G₀ m)

/-- Two graphs are equal if they have the same edges -/
def graphEqual (G H : Graph n) : Prop :=
  ∀ i j, G i j ↔ H i j

theorem graph_sequence_periodic (n : ℕ) (G₀ : Graph n) :
  ∃ (m₀ T : ℕ), T ≤ 2^n ∧
    ∀ m ≥ m₀, graphEqual (graphSequence G₀ (m + T)) (graphSequence G₀ m) :=
sorry

end NUMINAMATH_CALUDE_graph_sequence_periodic_l684_68477


namespace NUMINAMATH_CALUDE_total_jellybeans_needed_l684_68401

/-- The number of jellybeans needed to fill a large glass -/
def large_glass_jellybeans : ℕ := 50

/-- The number of large glasses to be filled -/
def num_large_glasses : ℕ := 5

/-- The number of small glasses to be filled -/
def num_small_glasses : ℕ := 3

/-- The number of jellybeans needed to fill a small glass -/
def small_glass_jellybeans : ℕ := large_glass_jellybeans / 2

/-- Theorem: The total number of jellybeans needed to fill all glasses is 325 -/
theorem total_jellybeans_needed : 
  num_large_glasses * large_glass_jellybeans + num_small_glasses * small_glass_jellybeans = 325 := by
  sorry

end NUMINAMATH_CALUDE_total_jellybeans_needed_l684_68401


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l684_68461

theorem right_triangle_leg_square (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 2*a + 1 →      -- Hypotenuse condition
  b^2 = 3*a^2 + 4*a + 1 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l684_68461


namespace NUMINAMATH_CALUDE_solutions_for_twenty_l684_68465

-- Define a function that counts the number of distinct integer solutions
def count_solutions (n : ℕ+) : ℕ := 4 * n

-- State the theorem
theorem solutions_for_twenty : count_solutions 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_solutions_for_twenty_l684_68465


namespace NUMINAMATH_CALUDE_infinitely_many_an_power_an_mod_8_l684_68476

theorem infinitely_many_an_power_an_mod_8 :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ (8 * n + 3)^(8 * n + 3) ≡ 8 * n + 3 [ZMOD 8] := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_an_power_an_mod_8_l684_68476


namespace NUMINAMATH_CALUDE_andrew_vacation_days_in_march_l684_68446

/-- Calculates the number of vacation days taken in March given the conditions of Andrew's work and vacation schedule. -/
def vacation_days_in_march (days_worked : ℕ) (days_per_vacation : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_vacation_days := days_worked / days_per_vacation
  let used_vacation_days := total_vacation_days - remaining_days
  used_vacation_days / 3

theorem andrew_vacation_days_in_march :
  vacation_days_in_march 300 10 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_andrew_vacation_days_in_march_l684_68446


namespace NUMINAMATH_CALUDE_polynomial_coefficient_problem_l684_68443

theorem polynomial_coefficient_problem (x a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) 
  (h1 : (x + a)^9 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9)
  (h2 : a₅ = 126) :
  a = 0 ∨ a = 2 := by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_problem_l684_68443


namespace NUMINAMATH_CALUDE_simplify_expression_l684_68472

theorem simplify_expression (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) : 
  ((3 / (a + 1) - 1) / ((a - 2) / (a^2 + 2*a + 1))) = -a - 1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l684_68472


namespace NUMINAMATH_CALUDE_odd_integers_square_divisibility_l684_68463

theorem odd_integers_square_divisibility (m n : ℤ) :
  Odd m → Odd n → (m^2 - n^2 + 1) ∣ (n^2 - 1) → ∃ k : ℤ, m^2 - n^2 + 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_integers_square_divisibility_l684_68463


namespace NUMINAMATH_CALUDE_stationery_problem_solution_l684_68436

/-- Represents a box of stationery --/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Defines the conditions of the stationery problem --/
def stationeryProblem (box : StationeryBox) : Prop :=
  -- Tom's condition: all envelopes used, 100 sheets left
  box.sheets - box.envelopes = 100 ∧
  -- Jerry's condition: all sheets used, 25 envelopes left
  box.envelopes + 25 = box.sheets / 3

/-- The theorem stating the solution to the stationery problem --/
theorem stationery_problem_solution :
  ∃ (box : StationeryBox), stationeryProblem box ∧ box.sheets = 120 :=
sorry

end NUMINAMATH_CALUDE_stationery_problem_solution_l684_68436


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l684_68458

/-- Given that the solution set of ax^2 - bx + c > 0 is (-1, 2), prove the following statements -/
theorem quadratic_inequality_properties (a b c : ℝ) 
  (h : Set.Ioo (-1 : ℝ) 2 = {x : ℝ | a * x^2 - b * x + c > 0}) :
  (b < 0 ∧ c > 0) ∧ 
  (a - b + c > 0) ∧ 
  ({x : ℝ | a * x^2 + b * x + c > 0} = Set.Ioo (-2 : ℝ) 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l684_68458


namespace NUMINAMATH_CALUDE_det_2x2_matrix_l684_68496

def matrix_det (x y : ℝ) : ℝ := x * y - 6

theorem det_2x2_matrix (x y : ℝ) :
  Matrix.det ![![x, 2], ![3, y]] = matrix_det x y := by
  sorry

end NUMINAMATH_CALUDE_det_2x2_matrix_l684_68496


namespace NUMINAMATH_CALUDE_quadratic_max_value_l684_68422

/-- The quadratic function f(x) = -x^2 - 2x - 3 -/
def f (x : ℝ) : ℝ := -x^2 - 2*x - 3

theorem quadratic_max_value :
  (∀ x : ℝ, f x ≤ -2) ∧ (∃ x : ℝ, f x = -2) := by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l684_68422


namespace NUMINAMATH_CALUDE_sixth_graders_and_parents_average_age_l684_68499

/-- The average age of a group of sixth-graders and their parents -/
def average_age (num_children : ℕ) (num_parents : ℕ) (avg_age_children : ℚ) (avg_age_parents : ℚ) : ℚ :=
  ((num_children : ℚ) * avg_age_children + (num_parents : ℚ) * avg_age_parents) / ((num_children + num_parents) : ℚ)

/-- Theorem stating the average age of sixth-graders and their parents -/
theorem sixth_graders_and_parents_average_age :
  average_age 45 60 12 35 = 25142857142857142 / 1000000000000000 :=
by sorry

end NUMINAMATH_CALUDE_sixth_graders_and_parents_average_age_l684_68499


namespace NUMINAMATH_CALUDE_aluminum_weight_l684_68464

-- Define the weights of the metal pieces
def iron_weight : ℝ := 11.17
def weight_difference : ℝ := 10.33

-- Theorem to prove
theorem aluminum_weight :
  iron_weight - weight_difference = 0.84 := by
  sorry

end NUMINAMATH_CALUDE_aluminum_weight_l684_68464


namespace NUMINAMATH_CALUDE_natural_number_decomposition_l684_68425

theorem natural_number_decomposition (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), (x : ℤ) = a^2 + b^2 ∧ (y : ℤ) = c^2 + d^2 ∧ (z : ℤ) = a * c + b * d := by
  sorry

end NUMINAMATH_CALUDE_natural_number_decomposition_l684_68425


namespace NUMINAMATH_CALUDE_integer_pairs_sum_reciprocals_l684_68440

theorem integer_pairs_sum_reciprocals (x y : ℤ) : 
  x ≤ y ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 4 ↔ 
  (x = -4 ∧ y = 2) ∨ 
  (x = -12 ∧ y = 3) ∨ 
  (x = 5 ∧ y = 20) ∨ 
  (x = 6 ∧ y = 12) ∨ 
  (x = 8 ∧ y = 8) := by
sorry

end NUMINAMATH_CALUDE_integer_pairs_sum_reciprocals_l684_68440


namespace NUMINAMATH_CALUDE_equations_solvability_l684_68451

theorem equations_solvability :
  (∃ (x y z : ℕ), 
    (x % 2 = 1) ∧ (y % 2 = 1) ∧ (z % 2 = 1) ∧
    (y = x + 2) ∧ (z = y + 2) ∧
    (x + y + z = 51)) ∧
  (∃ (x y z w : ℕ),
    (x % 6 = 0) ∧ (y % 6 = 0) ∧ (z % 6 = 0) ∧ (w % 6 = 0) ∧
    (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (w > 0) ∧
    (x + y + z + w = 60)) :=
by sorry

end NUMINAMATH_CALUDE_equations_solvability_l684_68451


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l684_68427

/-- Proves that the speed of a boat in still water is 20 km/hr given the specified conditions -/
theorem boat_speed_in_still_water :
  ∀ (boat_speed : ℝ),
    (boat_speed + 5) * 0.4 = 10 →
    boat_speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l684_68427


namespace NUMINAMATH_CALUDE_larger_number_proof_l684_68432

theorem larger_number_proof (A B : ℕ+) : 
  (Nat.gcd A B = 20) → 
  (∃ (k : ℕ+), Nat.lcm A B = 20 * 21 * 23 * k) → 
  (max A B = 460) :=
by sorry

end NUMINAMATH_CALUDE_larger_number_proof_l684_68432


namespace NUMINAMATH_CALUDE_repunit_divisibility_l684_68490

theorem repunit_divisibility (p : Nat) (h_prime : Prime p) (h_not_two : p ≠ 2) (h_not_five : p ≠ 5) :
  ∃ n : Nat, ∃ k : Nat, k > 0 ∧ p ∣ (10^n - 1) / 9 :=
sorry

end NUMINAMATH_CALUDE_repunit_divisibility_l684_68490


namespace NUMINAMATH_CALUDE_cookie_consumption_l684_68478

theorem cookie_consumption (total cookies_left father_ate : ℕ) 
  (h1 : total = 30)
  (h2 : cookies_left = 8)
  (h3 : father_ate = 10) :
  let mother_ate := father_ate / 2
  let total_eaten := total - cookies_left
  let brother_ate := total_eaten - (father_ate + mother_ate)
  brother_ate - mother_ate = 2 := by sorry

end NUMINAMATH_CALUDE_cookie_consumption_l684_68478


namespace NUMINAMATH_CALUDE_bond_interest_rate_proof_l684_68479

/-- Proves that the interest rate of a bond is 5.75% given specific investment conditions -/
theorem bond_interest_rate_proof (total_investment : ℝ) (unknown_bond_investment : ℝ) 
  (known_bond_investment : ℝ) (known_interest_rate : ℝ) (desired_interest_income : ℝ) :
  total_investment = 32000 →
  unknown_bond_investment = 20000 →
  known_bond_investment = 12000 →
  known_interest_rate = 0.0625 →
  desired_interest_income = 1900 →
  ∃ unknown_interest_rate : ℝ,
    unknown_interest_rate = 0.0575 ∧
    desired_interest_income = unknown_bond_investment * unknown_interest_rate + 
                              known_bond_investment * known_interest_rate :=
by sorry

end NUMINAMATH_CALUDE_bond_interest_rate_proof_l684_68479


namespace NUMINAMATH_CALUDE_boyden_family_ticket_cost_l684_68459

/-- The cost of tickets for a family visit to a leisure park -/
def ticket_cost (adult_price : ℕ) (child_price : ℕ) (num_adults : ℕ) (num_children : ℕ) : ℕ :=
  adult_price * num_adults + child_price * num_children

theorem boyden_family_ticket_cost :
  let adult_price : ℕ := 19
  let child_price : ℕ := adult_price - 6
  let num_adults : ℕ := 2
  let num_children : ℕ := 3
  ticket_cost adult_price child_price num_adults num_children = 77 := by
  sorry

end NUMINAMATH_CALUDE_boyden_family_ticket_cost_l684_68459


namespace NUMINAMATH_CALUDE_beetle_journey_l684_68405

/-- Represents the beetle's movements in centimeters -/
def beetle_movements : List ℝ := [10, -9, 8, -6, 7.5, -6, 8, -7]

/-- Time taken per centimeter in seconds -/
def time_per_cm : ℝ := 2

/-- Calculates the final position of the beetle relative to the starting point -/
def final_position (movements : List ℝ) : ℝ :=
  movements.sum

/-- Calculates the total distance traveled by the beetle -/
def total_distance (movements : List ℝ) : ℝ :=
  movements.map abs |>.sum

/-- Calculates the total time taken for the journey -/
def total_time (movements : List ℝ) (time_per_cm : ℝ) : ℝ :=
  (total_distance movements) * time_per_cm

theorem beetle_journey :
  final_position beetle_movements = 5.5 ∧
  total_time beetle_movements time_per_cm = 123 := by
  sorry

#eval final_position beetle_movements
#eval total_time beetle_movements time_per_cm

end NUMINAMATH_CALUDE_beetle_journey_l684_68405


namespace NUMINAMATH_CALUDE_modular_inverse_31_mod_35_l684_68413

theorem modular_inverse_31_mod_35 : ∃ x : ℕ, x ≤ 34 ∧ (31 * x) % 35 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_31_mod_35_l684_68413


namespace NUMINAMATH_CALUDE_smallest_egg_solution_l684_68460

/-- Represents the egg selling scenario over 5 days -/
def egg_selling (initial_eggs : ℕ) (sold_per_day : ℕ) : Prop :=
  ∃ (remaining : ℕ → ℕ),
    remaining 0 = initial_eggs ∧
    remaining 1 = initial_eggs - sold_per_day ∧
    remaining 2 = 2 * (remaining 1) - sold_per_day ∧
    remaining 3 = 3 * (remaining 2) - sold_per_day ∧
    remaining 4 = 4 * (remaining 3) - sold_per_day ∧
    5 * (remaining 4) - sold_per_day = 0

/-- The smallest initial number of eggs and the number sold per day -/
theorem smallest_egg_solution :
  egg_selling 103 60 ∧
  ∀ n s, egg_selling n s → n ≥ 103 := by
  sorry

end NUMINAMATH_CALUDE_smallest_egg_solution_l684_68460


namespace NUMINAMATH_CALUDE_logistics_service_assignments_logistics_service_assignments_proof_l684_68445

theorem logistics_service_assignments : ℕ :=
  let total_students : ℕ := 5
  let total_athletes : ℕ := 3
  let athlete_A_in_own_team : Bool := true

  50

theorem logistics_service_assignments_proof :
  logistics_service_assignments = 50 := by
  sorry

end NUMINAMATH_CALUDE_logistics_service_assignments_logistics_service_assignments_proof_l684_68445


namespace NUMINAMATH_CALUDE_shaded_area_of_intersecting_rectangles_l684_68416

/-- The area of the shaded region formed by two intersecting perpendicular rectangles -/
theorem shaded_area_of_intersecting_rectangles (rect1_width rect1_height rect2_width rect2_height : ℝ) 
  (h1 : rect1_width = 2 ∧ rect1_height = 10)
  (h2 : rect2_width = 3 ∧ rect2_height = 8)
  (h3 : rect1_width ≤ rect2_height ∧ rect2_width ≤ rect1_height) : 
  rect1_width * rect1_height + rect2_width * rect2_height - rect1_width * rect2_width = 38 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_of_intersecting_rectangles_l684_68416


namespace NUMINAMATH_CALUDE_students_without_A_l684_68485

theorem students_without_A (total : ℕ) (chemistry_A : ℕ) (physics_A : ℕ) (both_A : ℕ) 
  (h1 : total = 35)
  (h2 : chemistry_A = 9)
  (h3 : physics_A = 15)
  (h4 : both_A = 5) :
  total - (chemistry_A + physics_A - both_A) = 16 := by
  sorry

end NUMINAMATH_CALUDE_students_without_A_l684_68485


namespace NUMINAMATH_CALUDE_ages_sum_l684_68488

theorem ages_sum (a b c : ℕ) : 
  a = b + c + 20 → 
  a^2 = (b + c)^2 + 1800 → 
  a + b + c = 90 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l684_68488


namespace NUMINAMATH_CALUDE_time_taken_by_A_l684_68434

/-- The time taken by A to reach the destination given the specified conditions -/
theorem time_taken_by_A (distance : ℝ) (speed_A speed_B : ℝ) (time_B : ℝ) : 
  speed_A / speed_B = 3 / 4 →
  time_B * 60 + 30 = speed_B * distance / speed_A →
  speed_A * (time_B * 60 + 30) / 60 = distance →
  speed_A * 2 = distance :=
by sorry

end NUMINAMATH_CALUDE_time_taken_by_A_l684_68434


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l684_68473

-- System 1
theorem system_one_solution (x y : ℝ) :
  (3 * x + 2 * y = 10 ∧ x / 2 - (y + 1) / 3 = 1) →
  (x = 3 ∧ y = 1/2) :=
by sorry

-- System 2
theorem system_two_solution (x y : ℝ) :
  (4 * x - 5 * y = 3 ∧ (x - 2 * y) / 0.4 = 0.6) →
  (x = 1.6 ∧ y = 0.68) :=
by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l684_68473


namespace NUMINAMATH_CALUDE_water_sulfuric_oxygen_equivalence_l684_68468

/-- Represents the number of oxygen atoms in a molecule --/
def oxygenAtoms (molecule : String) : ℕ :=
  match molecule with
  | "H2SO4" => 4
  | "H2O" => 1
  | _ => 0

/-- Theorem stating that 4n water molecules have the same number of oxygen atoms as n sulfuric acid molecules --/
theorem water_sulfuric_oxygen_equivalence (n : ℕ) :
  n * oxygenAtoms "H2SO4" = 4 * n * oxygenAtoms "H2O" :=
by sorry


end NUMINAMATH_CALUDE_water_sulfuric_oxygen_equivalence_l684_68468


namespace NUMINAMATH_CALUDE_circle_area_difference_l684_68409

/-- Given two circles where the smaller circle has radius 4 and the center of the larger circle
    is on the circumference of the smaller circle, the difference in areas between the larger
    and smaller circles is 48π. -/
theorem circle_area_difference (r : ℝ) (h : r = 4) : 
  π * (2 * r)^2 - π * r^2 = 48 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l684_68409


namespace NUMINAMATH_CALUDE_bottle_cap_count_l684_68423

theorem bottle_cap_count (caps_per_box : ℝ) (num_boxes : ℝ) 
  (h1 : caps_per_box = 35.0) 
  (h2 : num_boxes = 7.0) : 
  caps_per_box * num_boxes = 245.0 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_count_l684_68423


namespace NUMINAMATH_CALUDE_usual_time_calculation_l684_68486

/-- Represents the scenario of a person catching a bus -/
structure BusScenario where
  usual_speed : ℝ
  usual_time : ℝ
  faster_speed : ℝ
  missed_time : ℝ

/-- The theorem stating the relationship between usual time and missed time -/
theorem usual_time_calculation (scenario : BusScenario) 
  (h1 : scenario.faster_speed = (5/4) * scenario.usual_speed)
  (h2 : scenario.missed_time = scenario.usual_time + 5)
  (h3 : scenario.usual_speed * scenario.usual_time = scenario.faster_speed * scenario.missed_time) :
  scenario.usual_time = 25 := by
  sorry

#check usual_time_calculation

end NUMINAMATH_CALUDE_usual_time_calculation_l684_68486


namespace NUMINAMATH_CALUDE_lee_class_b_students_l684_68417

theorem lee_class_b_students (kipling_total : ℕ) (kipling_b : ℕ) (lee_total : ℕ) 
  (h1 : kipling_total = 12)
  (h2 : kipling_b = 8)
  (h3 : lee_total = 30) :
  ∃ (lee_b : ℕ), (lee_b : ℚ) / lee_total = (kipling_b : ℚ) / kipling_total ∧ lee_b = 20 := by
  sorry


end NUMINAMATH_CALUDE_lee_class_b_students_l684_68417


namespace NUMINAMATH_CALUDE_set_operations_l684_68494

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

theorem set_operations :
  (A ∩ B = {x | 1 ≤ x ∧ x ≤ 4}) ∧
  ((Set.univ : Set ℝ) \ (A ∪ B) = {x | x < -1 ∨ x > 5}) ∧
  (((Set.univ : Set ℝ) \ A) ∪ ((Set.univ : Set ℝ) \ B) = {x | x < 1 ∨ x > 4}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l684_68494


namespace NUMINAMATH_CALUDE_min_value_expression_l684_68483

theorem min_value_expression (a b c k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_eq : a = k ∧ b = k ∧ c = k) : 
  (a + b + c) * ((a + b)⁻¹ + (a + c)⁻¹ + (b + c)⁻¹) = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l684_68483


namespace NUMINAMATH_CALUDE_simplify_expression_l684_68402

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  12 * x^5 * y / (6 * x * y) = 2 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l684_68402


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l684_68457

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) : 
  (∀ x : ℤ, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l684_68457


namespace NUMINAMATH_CALUDE_roots_sum_inverse_squares_l684_68412

theorem roots_sum_inverse_squares (a b c : ℝ) (r s : ℂ) (h₁ : a ≠ 0) (h₂ : c ≠ 0) 
  (h₃ : a * r^2 + b * r - c = 0) (h₄ : a * s^2 + b * s - c = 0) : 
  1 / r^2 + 1 / s^2 = (b^2 + 2*a*c) / c^2 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_inverse_squares_l684_68412


namespace NUMINAMATH_CALUDE_exists_tangent_circle_l684_68453

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles intersecting
def intersects (c1 c2 : Circle) : Prop :=
  ∃ (p : ℝ × ℝ), (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
                 (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2

-- Define the property of a point being on a circle
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the property of a circle being tangent to another circle
def isTangent (c1 c2 : Circle) : Prop :=
  ∃ (p : ℝ × ℝ), onCircle p c1 ∧ onCircle p c2 ∧
    ∀ (q : ℝ × ℝ), q ≠ p → ¬(onCircle q c1 ∧ onCircle q c2)

-- Theorem statement
theorem exists_tangent_circle (S₁ S₂ S₃ : Circle) (O : ℝ × ℝ) :
  intersects S₁ S₂ ∧ intersects S₂ S₃ ∧ intersects S₃ S₁ ∧
  onCircle O S₁ ∧ onCircle O S₂ ∧ onCircle O S₃ →
  ∃ (S : Circle), isTangent S S₁ ∧ isTangent S S₂ ∧ isTangent S S₃ :=
sorry

end NUMINAMATH_CALUDE_exists_tangent_circle_l684_68453


namespace NUMINAMATH_CALUDE_negation_of_positive_square_plus_x_positive_l684_68492

theorem negation_of_positive_square_plus_x_positive :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_positive_square_plus_x_positive_l684_68492


namespace NUMINAMATH_CALUDE_binary_op_three_seven_l684_68487

def binary_op (c d : ℤ) : ℤ := 4 * c + 3 * d - c * d

theorem binary_op_three_seven : binary_op 3 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binary_op_three_seven_l684_68487


namespace NUMINAMATH_CALUDE_infinitely_many_prime_divisors_l684_68491

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluation of a quadratic polynomial at an integer -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℤ) : ℤ :=
  p.a * x^2 + p.b * x + p.c

/-- The set of prime divisors of a polynomial's values -/
def primeDivisors (p : QuadraticPolynomial) : Set ℕ :=
  {q : ℕ | Nat.Prime q ∧ ∃ n : ℤ, (q : ℤ) ∣ p.eval n}

/-- The main theorem: there are infinitely many prime divisors for any quadratic polynomial -/
theorem infinitely_many_prime_divisors (p : QuadraticPolynomial) :
  Set.Infinite (primeDivisors p) := by
  sorry


end NUMINAMATH_CALUDE_infinitely_many_prime_divisors_l684_68491


namespace NUMINAMATH_CALUDE_density_of_cube_root_differences_l684_68444

theorem density_of_cube_root_differences :
  ∀ ε > 0, ∀ x : ℝ, ∃ n m : ℕ, |x - (n^(1/3) - m^(1/3))| < ε :=
sorry

end NUMINAMATH_CALUDE_density_of_cube_root_differences_l684_68444


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l684_68454

theorem polynomial_root_implies_coefficients :
  ∀ (a b : ℝ),
  (Complex.I : ℂ) ^ 4 + a * (Complex.I : ℂ) ^ 3 - (Complex.I : ℂ) ^ 2 + b * (Complex.I : ℂ) - 6 = 0 →
  (2 - Complex.I : ℂ) ^ 4 + a * (2 - Complex.I : ℂ) ^ 3 - (2 - Complex.I : ℂ) ^ 2 + b * (2 - Complex.I : ℂ) - 6 = 0 →
  a = -4 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l684_68454


namespace NUMINAMATH_CALUDE_problem_solution_l684_68495

theorem problem_solution (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2003 + b^2004 = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l684_68495


namespace NUMINAMATH_CALUDE_root_condition_implies_a_range_l684_68462

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + (a^2 - 1)*x + (a - 2)

-- State the theorem
theorem root_condition_implies_a_range :
  ∀ a : ℝ,
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ x > 1 ∧ y < 1) →
  -2 < a ∧ a < 1 :=
by
  sorry


end NUMINAMATH_CALUDE_root_condition_implies_a_range_l684_68462


namespace NUMINAMATH_CALUDE_light_bulbs_problem_l684_68414

theorem light_bulbs_problem (initial : ℕ) : 
  (initial - 16) / 2 = 12 → initial = 40 := by
  sorry

end NUMINAMATH_CALUDE_light_bulbs_problem_l684_68414


namespace NUMINAMATH_CALUDE_expression_equality_l684_68420

theorem expression_equality : (-1)^2 - |(-3)| + (-5) / (-5/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l684_68420


namespace NUMINAMATH_CALUDE_expression_values_l684_68467

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (x : ℝ), x ∈ ({-4, 0, 4} : Set ℝ) ∧
  x = a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l684_68467


namespace NUMINAMATH_CALUDE_cricket_average_l684_68474

theorem cricket_average (current_innings : ℕ) (next_innings_runs : ℕ) (average_increase : ℕ) :
  current_innings = 10 →
  next_innings_runs = 80 →
  average_increase = 4 →
  (current_innings * x + next_innings_runs) / (current_innings + 1) = x + average_increase →
  x = 36 :=
by sorry

end NUMINAMATH_CALUDE_cricket_average_l684_68474


namespace NUMINAMATH_CALUDE_baseball_cards_count_l684_68415

theorem baseball_cards_count (num_friends : ℕ) (cards_per_friend : ℕ) : 
  num_friends = 5 → cards_per_friend = 91 → num_friends * cards_per_friend = 455 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_count_l684_68415


namespace NUMINAMATH_CALUDE_center_of_given_hyperbola_l684_68431

/-- The equation of a hyperbola in the form (ay + b)^2/c^2 - (dx + e)^2/f^2 = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- The center of a hyperbola --/
def center (h : Hyperbola) : ℝ × ℝ := sorry

/-- The given hyperbola --/
def given_hyperbola : Hyperbola :=
  { a := 4
    b := 8
    c := 7
    d := 5
    e := -5
    f := 3 }

/-- Theorem: The center of the given hyperbola is (1, -2) --/
theorem center_of_given_hyperbola :
  center given_hyperbola = (1, -2) := by sorry

end NUMINAMATH_CALUDE_center_of_given_hyperbola_l684_68431


namespace NUMINAMATH_CALUDE_max_cute_pairs_is_43_l684_68447

/-- A pair of ages (a, b) is cute if each person is at least seven years older than half the age of the other person. -/
def is_cute_pair (a b : ℕ) : Prop :=
  a ≥ b / 2 + 7 ∧ b ≥ a / 2 + 7

/-- The set of ages from 1 to 100. -/
def age_set : Finset ℕ :=
  Finset.range 100

/-- A function that returns the maximum number of pairwise disjoint cute pairs that can be formed from a set of ages. -/
def max_disjoint_cute_pairs (ages : Finset ℕ) : ℕ :=
  sorry

theorem max_cute_pairs_is_43 :
  max_disjoint_cute_pairs age_set = 43 :=
sorry

end NUMINAMATH_CALUDE_max_cute_pairs_is_43_l684_68447


namespace NUMINAMATH_CALUDE_staircase_expansion_l684_68481

/-- Calculates the number of toothpicks needed for a staircase of given steps -/
def toothpicks_for_steps (n : ℕ) : ℕ :=
  if n ≤ 1 then 4
  else if n = 2 then 10
  else 10 + 8 * (n - 2)

/-- The problem statement -/
theorem staircase_expansion :
  let initial_steps := 4
  let initial_toothpicks := 26
  let main_final_steps := 6
  let adjacent_steps := 3
  let additional_toothpicks := 
    (toothpicks_for_steps main_final_steps + toothpicks_for_steps adjacent_steps) - initial_toothpicks
  additional_toothpicks = 34 := by sorry

end NUMINAMATH_CALUDE_staircase_expansion_l684_68481


namespace NUMINAMATH_CALUDE_exam_time_allocation_l684_68442

theorem exam_time_allocation :
  ∀ (total_time total_questions type_a_questions : ℕ) 
    (type_a_time_ratio : ℚ),
  total_time = 180 →
  total_questions = 200 →
  type_a_questions = 20 →
  type_a_time_ratio = 2 →
  ∃ (type_a_time : ℕ),
    type_a_time = 36 ∧
    type_a_time * (total_questions - type_a_questions) = 
      (total_time - type_a_time) * type_a_questions * type_a_time_ratio :=
by sorry

end NUMINAMATH_CALUDE_exam_time_allocation_l684_68442


namespace NUMINAMATH_CALUDE_field_length_is_180_l684_68439

/-- Represents a rectangular field with a surrounding path -/
structure FieldWithPath where
  fieldLength : ℝ
  fieldWidth : ℝ
  pathWidth : ℝ

/-- Calculates the area of the path around the field -/
def pathArea (f : FieldWithPath) : ℝ :=
  (f.fieldLength + 2 * f.pathWidth) * (f.fieldWidth + 2 * f.pathWidth) - f.fieldLength * f.fieldWidth

/-- Theorem: If a rectangular field has width 55m, a surrounding path of 2.5m width, 
    and the path area is 1200 sq m, then the field length is 180m -/
theorem field_length_is_180 (f : FieldWithPath) 
    (h1 : f.fieldWidth = 55)
    (h2 : f.pathWidth = 2.5)
    (h3 : pathArea f = 1200) : 
  f.fieldLength = 180 := by
  sorry


end NUMINAMATH_CALUDE_field_length_is_180_l684_68439


namespace NUMINAMATH_CALUDE_handshake_theorem_l684_68433

theorem handshake_theorem (n : ℕ) (h : n > 0) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧
  (∃ (f : Fin n → ℕ), (∀ x, f x ≤ n - 1) ∧ f i = k ∧ f j = k) :=
sorry

end NUMINAMATH_CALUDE_handshake_theorem_l684_68433


namespace NUMINAMATH_CALUDE_meeting_arrangements_count_l684_68470

/-- Represents the number of schools in the community -/
def num_schools : ℕ := 4

/-- Represents the number of members in each school -/
def members_per_school : ℕ := 6

/-- Represents the number of representatives each school sends -/
def reps_per_school : ℕ := 2

/-- The number of ways to arrange the leadership meeting -/
def meeting_arrangements : ℕ := num_schools * (members_per_school.choose reps_per_school) * (members_per_school.choose reps_per_school)^(num_schools - 1)

/-- Theorem stating that the number of meeting arrangements is 202500 -/
theorem meeting_arrangements_count : meeting_arrangements = 202500 := by
  sorry

end NUMINAMATH_CALUDE_meeting_arrangements_count_l684_68470


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_decrease_l684_68475

/-- Prove that for an equilateral triangle with an area of 100√3 cm², 
    if each side is decreased by 6 cm, the decrease in area is 51√3 cm². -/
theorem equilateral_triangle_area_decrease 
  (original_area : ℝ) 
  (side_decrease : ℝ) :
  original_area = 100 * Real.sqrt 3 →
  side_decrease = 6 →
  let original_side := Real.sqrt ((4 * original_area) / Real.sqrt 3)
  let new_side := original_side - side_decrease
  let new_area := (new_side^2 * Real.sqrt 3) / 4
  original_area - new_area = 51 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_equilateral_triangle_area_decrease_l684_68475


namespace NUMINAMATH_CALUDE_x_eq_one_iff_z_purely_imaginary_l684_68411

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z as a function of x. -/
def z (x : ℝ) : ℂ :=
  ⟨x^2 - 1, x + 1⟩

/-- Theorem stating that x = 1 is necessary and sufficient for z(x) to be purely imaginary. -/
theorem x_eq_one_iff_z_purely_imaginary :
  ∀ x : ℝ, x = 1 ↔ IsPurelyImaginary (z x) :=
sorry

end NUMINAMATH_CALUDE_x_eq_one_iff_z_purely_imaginary_l684_68411


namespace NUMINAMATH_CALUDE_cream_fraction_after_pouring_l684_68482

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℚ
  cream : ℚ

/-- Represents the state of both cups --/
structure CupState where
  cup1 : CupContents
  cup2 : CupContents

/-- Performs one round of pouring between cups --/
def pour (state : CupState) : CupState := sorry

/-- Calculates the fraction of cream in cup1 after the pouring process --/
def creamFractionInCup1 (initial : CupState) : ℚ := sorry

theorem cream_fraction_after_pouring :
  let initial := CupState.mk
    (CupContents.mk 5 0)  -- 5 oz coffee, 0 oz cream in cup1
    (CupContents.mk 0 3)  -- 0 oz coffee, 3 oz cream in cup2
  let final := pour (pour initial)
  creamFractionInCup1 final = (11 : ℚ) / 21 := by sorry

#check cream_fraction_after_pouring

end NUMINAMATH_CALUDE_cream_fraction_after_pouring_l684_68482


namespace NUMINAMATH_CALUDE_no_double_application_function_l684_68400

theorem no_double_application_function : ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f (f x) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l684_68400


namespace NUMINAMATH_CALUDE_g_value_at_negative_1001_l684_68430

/-- A function g satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x

theorem g_value_at_negative_1001 (g : ℝ → ℝ) 
    (h1 : FunctionalEquation g) (h2 : g 1 = 3) : g (-1001) = 1005 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_negative_1001_l684_68430


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_problem_l684_68438

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric sequence -/
def a : ℚ := 1/3

/-- The common ratio of the geometric sequence -/
def r : ℚ := 1/3

/-- The sum we're looking for -/
def target_sum : ℚ := 80/243

theorem geometric_sequence_sum_problem :
  ∃ n : ℕ, geometric_sum a r n = target_sum ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_problem_l684_68438


namespace NUMINAMATH_CALUDE_balloon_unique_arrangements_l684_68498

def balloon_arrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 3)

theorem balloon_unique_arrangements :
  balloon_arrangements = 420 := by
  sorry

end NUMINAMATH_CALUDE_balloon_unique_arrangements_l684_68498


namespace NUMINAMATH_CALUDE_greatest_b_for_nonrange_l684_68450

theorem greatest_b_for_nonrange (b : ℤ) : (∀ x : ℝ, x^2 + b*x + 20 ≠ 5) ↔ b ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_greatest_b_for_nonrange_l684_68450


namespace NUMINAMATH_CALUDE_valentino_farm_birds_l684_68429

/-- The number of birds on Mr. Valentino's farm -/
def total_birds (chickens ducks turkeys : ℕ) : ℕ :=
  chickens + ducks + turkeys

/-- Theorem stating the total number of birds on Mr. Valentino's farm -/
theorem valentino_farm_birds :
  ∀ (chickens ducks turkeys : ℕ),
    chickens = 200 →
    ducks = 2 * chickens →
    turkeys = 3 * ducks →
    total_birds chickens ducks turkeys = 1800 := by
  sorry

end NUMINAMATH_CALUDE_valentino_farm_birds_l684_68429


namespace NUMINAMATH_CALUDE_largest_four_digit_multiple_of_three_l684_68404

theorem largest_four_digit_multiple_of_three : ∃ n : ℕ, 
  n = 9999 ∧ 
  n % 3 = 0 ∧ 
  ∀ m : ℕ, m < 10000 ∧ m % 3 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_multiple_of_three_l684_68404
