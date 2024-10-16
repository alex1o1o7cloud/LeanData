import Mathlib

namespace NUMINAMATH_CALUDE_painted_cells_theorem_l1488_148865

theorem painted_cells_theorem (k l : ℕ) : 
  k * l = 74 → 
  (((2 * k + 1) * (2 * l + 1) - 74 = 373) ∨ 
   ((2 * k + 1) * (2 * l + 1) - 74 = 301)) := by
  sorry

end NUMINAMATH_CALUDE_painted_cells_theorem_l1488_148865


namespace NUMINAMATH_CALUDE_lacy_correct_percentage_l1488_148883

theorem lacy_correct_percentage (x : ℕ) (x_pos : x > 0) :
  let total_problems := 6 * x
  let missed_problems := 2 * x
  let correct_problems := total_problems - missed_problems
  (correct_problems : ℚ) / (total_problems : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_lacy_correct_percentage_l1488_148883


namespace NUMINAMATH_CALUDE_emily_coloring_books_l1488_148846

theorem emily_coloring_books (x : ℕ) : 
  x - 2 + 14 = 19 → x = 7 := by
sorry

end NUMINAMATH_CALUDE_emily_coloring_books_l1488_148846


namespace NUMINAMATH_CALUDE_bens_gross_income_l1488_148848

theorem bens_gross_income (car_payment insurance maintenance fuel : ℝ)
  (h1 : car_payment = 400)
  (h2 : insurance = 150)
  (h3 : maintenance = 75)
  (h4 : fuel = 50)
  (h5 : ∀ after_tax_income : ℝ, 
    0.2 * after_tax_income = car_payment + insurance + maintenance + fuel)
  (h6 : ∀ gross_income : ℝ, 
    (2/3) * gross_income = after_tax_income) :
  ∃ gross_income : ℝ, gross_income = 5062.50 := by
sorry

end NUMINAMATH_CALUDE_bens_gross_income_l1488_148848


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1488_148872

theorem complex_expression_simplification :
  (7 + 4 * Real.sqrt 3) * (2 - Real.sqrt 3)^2 + (2 + Real.sqrt 3) * (2 - Real.sqrt 3) - Real.sqrt 3 = 2 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1488_148872


namespace NUMINAMATH_CALUDE_village_foods_customers_l1488_148885

/-- The number of customers per month for Village Foods --/
def customers_per_month (lettuce_cost tomato_cost total_cost_per_customer total_sales : ℚ) : ℚ :=
  total_sales / total_cost_per_customer

/-- Theorem: Village Foods gets 500 customers per month --/
theorem village_foods_customers :
  customers_per_month 2 2 4 2000 = 500 := by
  sorry

end NUMINAMATH_CALUDE_village_foods_customers_l1488_148885


namespace NUMINAMATH_CALUDE_triangle_properties_l1488_148864

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : (2 * abc.a - abc.c) * Real.cos abc.B = abc.b * Real.cos abc.C)
  (h2 : abc.A = Real.pi / 4)
  (h3 : abc.a = 2) :
  abc.B = Real.pi / 3 ∧ 
  (abc.a * abc.b * Real.sin abc.C) / 2 = (3 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1488_148864


namespace NUMINAMATH_CALUDE_seating_arrangements_l1488_148816

/-- The number of boys -/
def num_boys : Nat := 5

/-- The number of girls -/
def num_girls : Nat := 4

/-- The total number of chairs -/
def total_chairs : Nat := 9

/-- The number of odd-numbered chairs -/
def odd_chairs : Nat := (total_chairs + 1) / 2

/-- The number of even-numbered chairs -/
def even_chairs : Nat := total_chairs / 2

/-- Factorial function -/
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem seating_arrangements :
  factorial num_boys * factorial num_girls = 2880 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1488_148816


namespace NUMINAMATH_CALUDE_min_value_expression_l1488_148861

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 4) :
  (a + 3 * b) * (2 * b + 3 * c) * (a * c + 2) ≥ 192 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 4 ∧
    (a₀ + 3 * b₀) * (2 * b₀ + 3 * c₀) * (a₀ * c₀ + 2) = 192 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1488_148861


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_l1488_148842

/-- The parabola function -/
def f (d : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + d

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 3

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y (d : ℝ) : ℝ := f d vertex_x

/-- The theorem stating that the vertex lies on the x-axis iff d = 9 -/
theorem vertex_on_x_axis (d : ℝ) : vertex_y d = 0 ↔ d = 9 := by sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_l1488_148842


namespace NUMINAMATH_CALUDE_min_triangle_area_l1488_148871

/-- The minimum non-zero area of a triangle with vertices (0,0), (50,20), and (p,q),
    where p and q are integers. -/
theorem min_triangle_area :
  ∀ p q : ℤ,
  let area := (1/2 : ℝ) * |20 * p - 50 * q|
  ∃ p' q' : ℤ,
    (area > 0 → area ≥ 15) ∧
    (∃ a : ℝ, a > 0 ∧ a < 15 → ¬∃ p'' q'' : ℤ, (1/2 : ℝ) * |20 * p'' - 50 * q''| = a) :=
by sorry

end NUMINAMATH_CALUDE_min_triangle_area_l1488_148871


namespace NUMINAMATH_CALUDE_cookies_in_box_proof_l1488_148839

/-- The number of cookies in each bag -/
def cookies_per_bag : ℕ := 7

/-- The number of boxes -/
def num_boxes : ℕ := 8

/-- The number of bags -/
def num_bags : ℕ := 9

/-- The additional number of cookies in boxes compared to bags -/
def additional_cookies : ℕ := 33

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 12

theorem cookies_in_box_proof :
  num_boxes * cookies_per_box = num_bags * cookies_per_bag + additional_cookies :=
sorry

end NUMINAMATH_CALUDE_cookies_in_box_proof_l1488_148839


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_squared_minus_one_l1488_148892

theorem imaginary_part_of_z_squared_minus_one (z : ℂ) :
  z = 1 + Complex.I →
  Complex.im ((z + 1) * (z - 1)) = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_squared_minus_one_l1488_148892


namespace NUMINAMATH_CALUDE_expression_evaluation_l1488_148808

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  2 * x^2 + y^2 - z^2 + 3 * x * y = -6 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1488_148808


namespace NUMINAMATH_CALUDE_magnification_factor_l1488_148867

theorem magnification_factor (magnified_diameter actual_diameter : ℝ) 
  (h1 : magnified_diameter = 0.2)
  (h2 : actual_diameter = 0.0002) :
  magnified_diameter / actual_diameter = 1000 := by
sorry

end NUMINAMATH_CALUDE_magnification_factor_l1488_148867


namespace NUMINAMATH_CALUDE_watson_class_second_graders_l1488_148831

/-- The number of second graders in Ms. Watson's class -/
def second_graders (kindergartners first_graders total_students : ℕ) : ℕ :=
  total_students - (kindergartners + first_graders)

/-- Theorem stating the number of second graders in Ms. Watson's class -/
theorem watson_class_second_graders :
  second_graders 14 24 42 = 4 := by
  sorry

end NUMINAMATH_CALUDE_watson_class_second_graders_l1488_148831


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l1488_148804

/-- Theorem: Given a car's speed of 145 km/h in the first hour and an average speed of 102.5 km/h over two hours, the speed in the second hour is 60 km/h. -/
theorem car_speed_second_hour (speed_first_hour : ℝ) (average_speed : ℝ) (speed_second_hour : ℝ) :
  speed_first_hour = 145 →
  average_speed = 102.5 →
  (speed_first_hour + speed_second_hour) / 2 = average_speed →
  speed_second_hour = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l1488_148804


namespace NUMINAMATH_CALUDE_remainder_problem_l1488_148812

theorem remainder_problem (f y : ℤ) : 
  y % 5 = 4 → (f + y) % 5 = 2 → f % 5 = 3 :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l1488_148812


namespace NUMINAMATH_CALUDE_sqrt_meaningful_value_l1488_148873

theorem sqrt_meaningful_value (x : ℝ) : 
  (x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 3) → 
  (x - 2 ≥ 0 ↔ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_value_l1488_148873


namespace NUMINAMATH_CALUDE_midpoint_arrival_time_l1488_148836

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Represents a hiking event -/
structure HikingEvent where
  planned_start : Time
  planned_end : Time
  actual_start_delay : Nat
  actual_end_early : Nat

def midpoint_time (event : HikingEvent) : Time :=
  sorry

theorem midpoint_arrival_time (event : HikingEvent) : 
  event.planned_start = { hours := 10, minutes := 10 } →
  event.planned_end = { hours := 13, minutes := 10 } →
  event.actual_start_delay = 5 →
  event.actual_end_early = 4 →
  midpoint_time event = { hours := 11, minutes := 50 } :=
sorry

end NUMINAMATH_CALUDE_midpoint_arrival_time_l1488_148836


namespace NUMINAMATH_CALUDE_mother_ate_five_cookies_l1488_148886

def total_cookies : ℕ := 30
def charlie_cookies : ℕ := 15
def father_cookies : ℕ := 10

def mother_cookies : ℕ := total_cookies - (charlie_cookies + father_cookies)

theorem mother_ate_five_cookies : mother_cookies = 5 := by
  sorry

end NUMINAMATH_CALUDE_mother_ate_five_cookies_l1488_148886


namespace NUMINAMATH_CALUDE_chessboard_inner_square_probability_l1488_148805

/-- Represents a square chessboard -/
structure Chessboard where
  size : ℕ

/-- Calculates the number of squares on the perimeter of the chessboard -/
def perimeterSquares (board : Chessboard) : ℕ :=
  4 * board.size - 4

/-- Calculates the number of squares not on the perimeter of the chessboard -/
def innerSquares (board : Chessboard) : ℕ :=
  board.size * board.size - perimeterSquares board

/-- The probability of choosing an inner square on the chessboard -/
def innerSquareProbability (board : Chessboard) : ℚ :=
  innerSquares board / (board.size * board.size)

theorem chessboard_inner_square_probability :
  let board := Chessboard.mk 10
  innerSquareProbability board = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_inner_square_probability_l1488_148805


namespace NUMINAMATH_CALUDE_appliance_cost_l1488_148823

theorem appliance_cost (a b : ℝ) 
  (eq1 : a + 2 * b = 2300)
  (eq2 : 2 * a + b = 2050) :
  a = 600 ∧ b = 850 := by
  sorry

end NUMINAMATH_CALUDE_appliance_cost_l1488_148823


namespace NUMINAMATH_CALUDE_box_surface_area_is_1600_l1488_148802

/-- Calculates the surface area of the interior of an open box formed by removing square corners from a rectangular sheet and folding the sides. -/
def boxSurfaceArea (length width cornerSize : ℕ) : ℕ :=
  length * width - 4 * (cornerSize * cornerSize)

/-- Theorem stating that the surface area of the interior of the box is 1600 square units. -/
theorem box_surface_area_is_1600 :
  boxSurfaceArea 40 50 10 = 1600 := by
  sorry

#eval boxSurfaceArea 40 50 10

end NUMINAMATH_CALUDE_box_surface_area_is_1600_l1488_148802


namespace NUMINAMATH_CALUDE_valid_lineup_count_is_14_l1488_148821

/-- Represents the four athletes in the relay race -/
inductive Athlete : Type
| A : Athlete
| B : Athlete
| C : Athlete
| D : Athlete

/-- Represents the four positions in the relay race -/
inductive Position : Type
| first : Position
| second : Position
| third : Position
| fourth : Position

/-- A valid lineup for the relay race -/
def Lineup := Position → Athlete

/-- Predicate to check if a lineup is valid according to the given conditions -/
def isValidLineup (l : Lineup) : Prop :=
  l Position.first ≠ Athlete.A ∧ l Position.fourth ≠ Athlete.B

/-- The number of valid lineups -/
def validLineupCount : ℕ := sorry

/-- Theorem stating that the number of valid lineups is 14 -/
theorem valid_lineup_count_is_14 : validLineupCount = 14 := by sorry

end NUMINAMATH_CALUDE_valid_lineup_count_is_14_l1488_148821


namespace NUMINAMATH_CALUDE_sum_extrema_l1488_148813

theorem sum_extrema (x y z w : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ w ≥ 0) 
  (h_eq : x^2 + y^2 + z^2 + w^2 + x + 2*y + 3*z + 4*w = 17/2) : 
  (∀ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 + a + 2*b + 3*c + 4*d = 17/2 → 
    a + b + c + d ≤ 3) ∧ 
  (x + y + z + w ≥ -2 + 5/2 * Real.sqrt 2) ∧
  (∃ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
    a^2 + b^2 + c^2 + d^2 + a + 2*b + 3*c + 4*d = 17/2 ∧ 
    a + b + c + d = 3) ∧
  (∃ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
    a^2 + b^2 + c^2 + d^2 + a + 2*b + 3*c + 4*d = 17/2 ∧ 
    a + b + c + d = -2 + 5/2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_extrema_l1488_148813


namespace NUMINAMATH_CALUDE_solution_of_equation_l1488_148841

theorem solution_of_equation (x : ℚ) : -2 * x + 11 = 0 ↔ x = 11 / 2 := by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l1488_148841


namespace NUMINAMATH_CALUDE_spool_length_problem_l1488_148820

/-- Calculates the length of each spool of wire -/
def spool_length (total_spools : ℕ) (wire_per_necklace : ℕ) (total_necklaces : ℕ) : ℕ :=
  (wire_per_necklace * total_necklaces) / total_spools

theorem spool_length_problem :
  let total_spools : ℕ := 3
  let wire_per_necklace : ℕ := 4
  let total_necklaces : ℕ := 15
  spool_length total_spools wire_per_necklace total_necklaces = 20 := by
  sorry

end NUMINAMATH_CALUDE_spool_length_problem_l1488_148820


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_five_l1488_148838

theorem smallest_four_digit_mod_five : ∃ n : ℕ,
  (n ≥ 1000) ∧                 -- four-digit number
  (n < 10000) ∧                -- four-digit number
  (n % 5 = 4) ∧                -- equivalent to 4 mod 5
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 5 = 4 → m ≥ n) ∧  -- smallest such number
  (n = 1004) := by             -- the answer is 1004
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_five_l1488_148838


namespace NUMINAMATH_CALUDE_workers_total_earnings_l1488_148803

/-- Calculates the total earnings of three workers given their daily wages and days worked. -/
def total_earnings (daily_wage_a daily_wage_b daily_wage_c : ℕ) (days_a days_b days_c : ℕ) : ℕ :=
  daily_wage_a * days_a + daily_wage_b * days_b + daily_wage_c * days_c

/-- Theorem stating that under the given conditions, the total earnings of three workers is 1480. -/
theorem workers_total_earnings :
  ∀ (daily_wage_a daily_wage_b daily_wage_c : ℕ),
    daily_wage_a * 3 = daily_wage_b * 3 * 3/4 →
    daily_wage_b * 4 = daily_wage_c * 4 * 4/5 →
    daily_wage_c = 100 →
    total_earnings daily_wage_a daily_wage_b daily_wage_c 6 9 4 = 1480 :=
by
  sorry

#eval total_earnings 60 80 100 6 9 4

end NUMINAMATH_CALUDE_workers_total_earnings_l1488_148803


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l1488_148884

theorem largest_integer_negative_quadratic : 
  (∀ m : ℤ, m > 7 → m^2 - 11*m + 24 ≥ 0) ∧ 
  (7^2 - 11*7 + 24 < 0) := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l1488_148884


namespace NUMINAMATH_CALUDE_average_allowance_proof_l1488_148870

theorem average_allowance_proof (total_students : ℕ) (total_amount : ℚ) 
  (h1 : total_students = 60)
  (h2 : total_amount = 320)
  (h3 : (2 : ℚ) / 3 * total_students + (1 : ℚ) / 3 * total_students = total_students)
  (h4 : (1 : ℚ) / 3 * total_students * 4 + (2 : ℚ) / 3 * total_students * x = total_amount) :
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_average_allowance_proof_l1488_148870


namespace NUMINAMATH_CALUDE_line_circle_intersection_l1488_148895

/-- The intersection of a line and a circle with specific properties implies a unique value for the parameter a. -/
theorem line_circle_intersection (a : ℝ) (h_a_pos : a > 0) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = A.1 + 2*a ∧ B.2 = B.1 + 2*a) ∧ 
    (A.1^2 + A.2^2 - 2*a*A.2 - 2 = 0 ∧ B.1^2 + B.2^2 - 2*a*B.2 - 2 = 0) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 12)) →
  a = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l1488_148895


namespace NUMINAMATH_CALUDE_divisor_problem_l1488_148856

theorem divisor_problem (x : ℝ) (h : x = 1280) : ∃ d : ℝ, (x + 720) / d = 7392 / 462 ∧ d = 125 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1488_148856


namespace NUMINAMATH_CALUDE_function_root_implies_parameter_range_l1488_148851

theorem function_root_implies_parameter_range :
  ∀ a : ℝ,
  (∃ c ∈ Set.Icc (-2 : ℝ) 1, 2 * c - a + 1 = 0) →
  a ∈ Set.Icc (-3 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_function_root_implies_parameter_range_l1488_148851


namespace NUMINAMATH_CALUDE_append_12_to_three_digit_number_l1488_148890

theorem append_12_to_three_digit_number (h t u : ℕ) :
  let original := 100 * h + 10 * t + u
  let new_number := original * 100 + 12
  new_number = 10000 * h + 1000 * t + 100 * u + 12 :=
by sorry

end NUMINAMATH_CALUDE_append_12_to_three_digit_number_l1488_148890


namespace NUMINAMATH_CALUDE_parabola_focus_l1488_148899

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y = 9 * x^2 + 6 * x - 2

/-- The focus of a parabola -/
def is_focus (f : ℝ × ℝ) (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a h k : ℝ), 
    (∀ x y, eq x y ↔ y = a * (x - h)^2 + k) ∧
    f = (h, k + 1 / (4 * a))

/-- Theorem: The focus of the parabola y = 9x^2 + 6x - 2 is (-1/3, -107/36) -/
theorem parabola_focus :
  is_focus (-1/3, -107/36) parabola_equation :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1488_148899


namespace NUMINAMATH_CALUDE_max_segments_on_unit_disc_l1488_148825

/-- The maximum number of segments with lengths greater than 1 determined by n points on a unit disc -/
def maxSegments (n : ℕ) : ℚ :=
  2 * n^2 / 5

/-- Theorem stating the maximum number of segments with lengths greater than 1 -/
theorem max_segments_on_unit_disc (n : ℕ) (h : n ≥ 2) :
  maxSegments n = (2 * n^2 : ℚ) / 5 :=
by sorry

end NUMINAMATH_CALUDE_max_segments_on_unit_disc_l1488_148825


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l1488_148819

-- Define the angle α
def α : Real := sorry

-- Define the point P through which the terminal side of α passes
def P : (Real × Real) := (1, -2)

-- Theorem statement
theorem tan_double_angle_special_case :
  (∃ (k : Real), k > 0 ∧ k * (Real.cos α) = 1 ∧ k * (Real.sin α) = -2) →
  Real.tan (2 * α) = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l1488_148819


namespace NUMINAMATH_CALUDE_min_value_inequality_l1488_148855

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- State the theorem
theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) ∧ a + b + c = m) →
  a^2 + b^2 + c^2 ≥ 16/3 := by
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1488_148855


namespace NUMINAMATH_CALUDE_salary_increase_l1488_148874

theorem salary_increase (last_year_salary : ℝ) (last_year_savings_rate : ℝ) 
  (this_year_savings_rate : ℝ) (salary_increase_rate : ℝ) :
  last_year_savings_rate = 0.06 →
  this_year_savings_rate = 0.05 →
  this_year_savings_rate * (last_year_salary * (1 + salary_increase_rate)) = 
    last_year_savings_rate * last_year_salary →
  salary_increase_rate = 0.2 := by
  sorry

#check salary_increase

end NUMINAMATH_CALUDE_salary_increase_l1488_148874


namespace NUMINAMATH_CALUDE_function_equality_l1488_148854

theorem function_equality (f : ℤ → ℤ) :
  (∀ a b : ℤ, f (a^2 + b^2) + f (a * b) = f a ^ 2 + f b + 1) →
  (∀ a : ℤ, f a = 1) := by
sorry

end NUMINAMATH_CALUDE_function_equality_l1488_148854


namespace NUMINAMATH_CALUDE_inequality_proof_l1488_148877

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : 1/x + 1/y + 1/z = 1) : 
  Real.sqrt (x + y*z) + Real.sqrt (y + z*x) + Real.sqrt (z + x*y) ≥ 
  Real.sqrt (x*y*z) + Real.sqrt x + Real.sqrt y + Real.sqrt z := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1488_148877


namespace NUMINAMATH_CALUDE_x_fifth_minus_five_x_l1488_148826

theorem x_fifth_minus_five_x (x : ℝ) : x = 4 → x^5 - 5*x = 1004 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_minus_five_x_l1488_148826


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1488_148817

theorem fraction_equivalence : 
  ∃ (n : ℚ), (4 + n) / (7 + n) = 6 / 7 ∧ n = 14 := by
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1488_148817


namespace NUMINAMATH_CALUDE_circle_equation_l1488_148876

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := 3*x + 4*y + 2 = 0

-- Define the general circle
def general_circle (x y b r : ℝ) : Prop := (x - 1)^2 + (y - b)^2 = r^2

-- Define the specific circle we want to prove
def specific_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- State the theorem
theorem circle_equation :
  ∀ (b r : ℝ),
  (∀ x y, parabola x y → x = 1 ∧ y = 0) →  -- Focus of parabola is (1, 0)
  (∀ x y, line x y → general_circle x y b r) →  -- Line is tangent to circle
  (∀ x y, specific_circle x y) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1488_148876


namespace NUMINAMATH_CALUDE_taller_tree_height_l1488_148894

theorem taller_tree_height (h_taller h_shorter : ℝ) : 
  h_taller - h_shorter = 18 →
  h_shorter / h_taller = 5 / 6 →
  h_taller = 108 := by
sorry

end NUMINAMATH_CALUDE_taller_tree_height_l1488_148894


namespace NUMINAMATH_CALUDE_same_solution_implies_value_l1488_148869

theorem same_solution_implies_value (a b : ℝ) :
  (∃ x y : ℝ, 5 * x + y = 3 ∧ a * x + 5 * y = 4 ∧ x - 2 * y = 5 ∧ 5 * x + b * y = 1) →
  1/2 * a - b = 5 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_value_l1488_148869


namespace NUMINAMATH_CALUDE_larry_stickers_l1488_148888

/-- The number of stickers Larry starts with -/
def initial_stickers : ℕ := 93

/-- The number of stickers Larry loses -/
def lost_stickers : ℕ := 6

/-- The number of stickers Larry ends with -/
def final_stickers : ℕ := initial_stickers - lost_stickers

theorem larry_stickers : final_stickers = 87 := by
  sorry

end NUMINAMATH_CALUDE_larry_stickers_l1488_148888


namespace NUMINAMATH_CALUDE_x_value_l1488_148862

theorem x_value (x y : ℝ) (h : x / (x - 2) = (y^2 + 3*y + 1) / (y^2 + 3*y - 1)) : 
  x = 2*y^2 + 6*y + 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1488_148862


namespace NUMINAMATH_CALUDE_kimberly_store_visits_l1488_148827

/-- Represents the number of peanuts Kimberly buys each time she goes to the store. -/
def peanuts_per_visit : ℕ := 7

/-- Represents the total number of peanuts Kimberly bought last month. -/
def total_peanuts : ℕ := 21

/-- Represents the number of times Kimberly went to the store last month. -/
def store_visits : ℕ := total_peanuts / peanuts_per_visit

/-- Proves that Kimberly went to the store 3 times last month. -/
theorem kimberly_store_visits : store_visits = 3 := by
  sorry

end NUMINAMATH_CALUDE_kimberly_store_visits_l1488_148827


namespace NUMINAMATH_CALUDE_red_face_probability_l1488_148879

def large_cube_edge : ℕ := 6
def small_cube_edge : ℕ := 1

def total_small_cubes : ℕ := large_cube_edge ^ 3

def corner_cubes : ℕ := 8
def edge_cubes : ℕ := 4 * 12
def face_cubes : ℕ := 4 * 6

def red_faced_cubes : ℕ := corner_cubes + edge_cubes + face_cubes

theorem red_face_probability :
  (red_faced_cubes : ℚ) / total_small_cubes = 10 / 27 := by sorry

end NUMINAMATH_CALUDE_red_face_probability_l1488_148879


namespace NUMINAMATH_CALUDE_barium_chloride_molecular_weight_l1488_148853

/-- The molecular weight of one mole of Barium chloride, given the molecular weight of 4 moles. -/
theorem barium_chloride_molecular_weight :
  let moles : ℝ := 4
  let total_weight : ℝ := 828
  let one_mole_weight : ℝ := total_weight / moles
  one_mole_weight = 207 := by sorry

end NUMINAMATH_CALUDE_barium_chloride_molecular_weight_l1488_148853


namespace NUMINAMATH_CALUDE_expected_value_of_coins_l1488_148840

/-- The expected value of coins coming up heads when flipping four coins simultaneously -/
theorem expected_value_of_coins (nickel quarter half_dollar dollar : ℕ) 
  (h_nickel : nickel = 5)
  (h_quarter : quarter = 25)
  (h_half_dollar : half_dollar = 50)
  (h_dollar : dollar = 100)
  (p_heads : ℚ)
  (h_p_heads : p_heads = 1 / 2) : 
  p_heads * (nickel + quarter + half_dollar + dollar : ℚ) = 90 := by
sorry

end NUMINAMATH_CALUDE_expected_value_of_coins_l1488_148840


namespace NUMINAMATH_CALUDE_seven_divides_special_integer_l1488_148844

/-- Represents a 7-digit positive integer with the specified structure -/
structure SevenDigitInteger where
  value : ℕ
  is_seven_digit : 1000000 ≤ value ∧ value < 10000000
  first_three_equals_middle_three : ∃ (a b c : ℕ), value = a * 1000000 + b * 100000 + c * 10000 + a * 1000 + b * 100 + c * 10 + (value % 10)
  last_digit_multiple_of_first : ∃ (k : ℕ), value % 10 = k * ((value / 1000000) % 10)

/-- Theorem stating that 7 is a factor of any SevenDigitInteger -/
theorem seven_divides_special_integer (W : SevenDigitInteger) : 7 ∣ W.value := by
  sorry

end NUMINAMATH_CALUDE_seven_divides_special_integer_l1488_148844


namespace NUMINAMATH_CALUDE_sarah_boxes_count_l1488_148896

def total_apples : ℕ := 49
def apples_per_box : ℕ := 7

theorem sarah_boxes_count :
  total_apples / apples_per_box = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_sarah_boxes_count_l1488_148896


namespace NUMINAMATH_CALUDE_area_equality_l1488_148878

/-- Given a function g defined on {a, b, c}, prove that the area of the triangle
    formed by y = 3g(3x) is equal to the area of the triangle formed by y = g(x) -/
theorem area_equality (g : ℝ → ℝ) (a b c : ℝ) (area : ℝ) 
    (h1 : Set.range g = {g a, g b, g c})
    (h2 : area = 50)
    (h3 : area = abs ((b - a) * (g c - g a) - (c - a) * (g b - g a)) / 2) :
  abs ((b/3 - a/3) * (3 * g c - 3 * g a) - (c/3 - a/3) * (3 * g b - 3 * g a)) / 2 = area := by
  sorry

end NUMINAMATH_CALUDE_area_equality_l1488_148878


namespace NUMINAMATH_CALUDE_uncle_payment_ratio_l1488_148875

/-- Represents the cost structure and payment for James' singing lessons -/
structure LessonPayment where
  total_lessons : ℕ
  free_lessons : ℕ
  full_price_lessons : ℕ
  half_price_lessons : ℕ
  lesson_cost : ℕ
  james_payment : ℕ

/-- Calculates the total cost of lessons -/
def total_cost (l : LessonPayment) : ℕ :=
  l.lesson_cost * (l.full_price_lessons + l.half_price_lessons)

/-- Calculates the amount paid by James' uncle -/
def uncle_payment (l : LessonPayment) : ℕ :=
  total_cost l - l.james_payment

/-- Theorem stating the ratio of uncle's payment to total cost is 1:2 -/
theorem uncle_payment_ratio (l : LessonPayment) 
  (h1 : l.total_lessons = 20)
  (h2 : l.free_lessons = 1)
  (h3 : l.full_price_lessons = 10)
  (h4 : l.half_price_lessons = 4)
  (h5 : l.lesson_cost = 5)
  (h6 : l.james_payment = 35) :
  2 * uncle_payment l = total_cost l := by
  sorry

#check uncle_payment_ratio

end NUMINAMATH_CALUDE_uncle_payment_ratio_l1488_148875


namespace NUMINAMATH_CALUDE_quadratic_translation_l1488_148834

/-- Given a quadratic function f(x) = 2x^2, translating its graph upwards by 2 units
    results in the function g(x) = 2x^2 + 2. -/
theorem quadratic_translation (x : ℝ) :
  let f : ℝ → ℝ := λ x => 2 * x^2
  let g : ℝ → ℝ := λ x => 2 * x^2 + 2
  g x = f x + 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_translation_l1488_148834


namespace NUMINAMATH_CALUDE_vector_sum_norm_l1488_148857

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_sum_norm (a b : E) 
  (ha : ‖a‖ = 1) 
  (hb : ‖b‖ = 1) 
  (hab : ‖a - b‖ = 1) : 
  ‖a + b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_norm_l1488_148857


namespace NUMINAMATH_CALUDE_sunny_gave_away_two_cakes_l1488_148868

/-- The number of cakes Sunny initially baked -/
def initial_cakes : ℕ := 8

/-- The number of candles Sunny puts on each remaining cake -/
def candles_per_cake : ℕ := 6

/-- The total number of candles Sunny uses -/
def total_candles : ℕ := 36

/-- The number of cakes Sunny gave away -/
def cakes_given_away : ℕ := initial_cakes - (total_candles / candles_per_cake)

theorem sunny_gave_away_two_cakes : cakes_given_away = 2 := by
  sorry

end NUMINAMATH_CALUDE_sunny_gave_away_two_cakes_l1488_148868


namespace NUMINAMATH_CALUDE_painting_job_completion_time_l1488_148880

/-- Represents the time in hours it takes to paint a wall -/
structure PaintingTime where
  hours : ℚ
  is_positive : 0 < hours

/-- Represents a painter's rate in terms of wall painted per hour -/
def painting_rate (time : PaintingTime) : ℚ :=
  1 / time.hours

theorem painting_job_completion_time 
  (gina_time : PaintingTime)
  (tom_time : PaintingTime)
  (joint_work_time : ℚ)
  (h_gina : gina_time.hours = 3)
  (h_tom : tom_time.hours = 5)
  (h_joint : joint_work_time = 2)
  : ∃ (t : ℚ), t = 20/3 ∧ 
    (painting_rate gina_time + painting_rate tom_time) * joint_work_time + 
    painting_rate tom_time * (t - joint_work_time) = 1 :=
sorry

end NUMINAMATH_CALUDE_painting_job_completion_time_l1488_148880


namespace NUMINAMATH_CALUDE_cos_2x_quadratic_equation_l1488_148824

theorem cos_2x_quadratic_equation (a b c : ℝ) :
  ∃ (f : ℝ → ℝ), 
    (∀ x, a * (Real.cos x)^2 + b * Real.cos x + c = 0) →
    (∀ x, f (Real.cos (2 * x)) = 0) ∧
    (∃ p q r : ℝ, ∀ y, f y = p * y^2 + q * y + r ∧
      p = a^2 ∧
      q = 2 * (a^2 + 2 * a * c - b^2) ∧
      r = (a^2 + 2 * c)^2 - 2 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_cos_2x_quadratic_equation_l1488_148824


namespace NUMINAMATH_CALUDE_sphere_radius_is_six_l1488_148860

/-- The shadow length of the sphere -/
def sphere_shadow : ℝ := 12

/-- The height of the meter stick -/
def stick_height : ℝ := 1.5

/-- The shadow length of the meter stick -/
def stick_shadow : ℝ := 3

/-- The radius of the sphere -/
def sphere_radius : ℝ := 6

/-- Theorem stating that the radius of the sphere is 6 meters given the conditions -/
theorem sphere_radius_is_six :
  stick_height / stick_shadow = sphere_radius / sphere_shadow :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_is_six_l1488_148860


namespace NUMINAMATH_CALUDE_planted_fraction_for_specific_field_l1488_148815

/-- Represents a right triangle field with an unplanted square at the right angle -/
structure TriangleField where
  /-- Length of the first leg of the triangle -/
  leg1 : ℝ
  /-- Length of the second leg of the triangle -/
  leg2 : ℝ
  /-- Side length of the unplanted square -/
  square_side : ℝ
  /-- Shortest distance from the square to the hypotenuse -/
  distance_to_hypotenuse : ℝ

/-- The fraction of the field that is planted -/
def planted_fraction (field : TriangleField) : ℝ :=
  sorry

/-- Theorem stating the planted fraction for the specific field described in the problem -/
theorem planted_fraction_for_specific_field :
  let field : TriangleField := {
    leg1 := 5,
    leg2 := 12,
    square_side := 60 / 49,
    distance_to_hypotenuse := 3
  }
  planted_fraction field = 11405 / 12005 := by
  sorry

end NUMINAMATH_CALUDE_planted_fraction_for_specific_field_l1488_148815


namespace NUMINAMATH_CALUDE_power_product_equals_2025_l1488_148859

theorem power_product_equals_2025 (a b : ℕ) (h1 : 5^a = 3125) (h2 : 3^b = 81) :
  5^(a - 3) * 3^(2*b - 4) = 2025 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_2025_l1488_148859


namespace NUMINAMATH_CALUDE_five_sixteenths_decimal_l1488_148807

theorem five_sixteenths_decimal : (5 : ℚ) / 16 = 0.3125 := by
  sorry

end NUMINAMATH_CALUDE_five_sixteenths_decimal_l1488_148807


namespace NUMINAMATH_CALUDE_basketball_win_calculation_l1488_148837

/-- Proves the number of games a basketball team needs to win to achieve a specific win percentage -/
theorem basketball_win_calculation (total_games : ℕ) (first_games : ℕ) (first_wins : ℕ) (remaining_games : ℕ) 
  (target_percentage : ℚ) (h1 : total_games = first_games + remaining_games) 
  (h2 : total_games = 100) (h3 : first_games = 45) (h4 : first_wins = 30) 
  (h5 : remaining_games = 55) (h6 : target_percentage = 65 / 100) : 
  ∃ (x : ℕ), (first_wins + x : ℚ) / total_games = target_percentage ∧ x = 35 := by
sorry

end NUMINAMATH_CALUDE_basketball_win_calculation_l1488_148837


namespace NUMINAMATH_CALUDE_ice_cream_melt_time_l1488_148833

/-- The time it takes for an ice cream cone to melt, given the distance to the beach and Jack's jogging speed -/
theorem ice_cream_melt_time 
  (blocks_to_beach : ℕ)
  (miles_per_block : ℚ)
  (jogging_speed : ℚ)
  (h1 : blocks_to_beach = 16)
  (h2 : miles_per_block = 1 / 8)
  (h3 : jogging_speed = 12) :
  (blocks_to_beach : ℚ) * miles_per_block / jogging_speed * 60 = 10 := by
  sorry

#check ice_cream_melt_time

end NUMINAMATH_CALUDE_ice_cream_melt_time_l1488_148833


namespace NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_35_seconds_l1488_148881

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger 
  (jogger_speed : Real) 
  (train_speed : Real) 
  (train_length : Real) 
  (initial_lead : Real) : Real :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_lead + train_length
  total_distance / relative_speed

/-- Proof that the train passes the jogger in 35 seconds -/
theorem train_passes_jogger_in_35_seconds : 
  train_passing_jogger 9 45 110 240 = 35 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_35_seconds_l1488_148881


namespace NUMINAMATH_CALUDE_distance_inequality_l1488_148889

theorem distance_inequality (x b : ℝ) (h1 : b > 0) (h2 : |x - 3| + |x - 5| < b) : b > 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_inequality_l1488_148889


namespace NUMINAMATH_CALUDE_grid_arithmetic_sequences_l1488_148858

/-- Given a 7x1 grid of numbers with two additional columns of length 3 and 5,
    prove that the value M satisfies the arithmetic sequence properties. -/
theorem grid_arithmetic_sequences (a : ℤ) (b c : ℚ) (M : ℚ) : 
  a = 25 ∧ 
  b = 16 ∧ 
  c = 20 ∧ 
  (∀ i : Fin 7, ∃ d : ℚ, a + i.val * d = a + 6 * d) ∧  -- row is arithmetic
  (∀ j : Fin 3, ∃ e : ℚ, a + j.val * e = b) ∧  -- first column is arithmetic
  (∀ k : Fin 5, ∃ f : ℚ, M + k.val * f = -20) ∧  -- second column is arithmetic
  (a + 3 * (b - a) / 3 = b) ∧  -- 4th element in row equals top of middle column
  (a + 6 * (M - a) / 6 = M) →  -- last element in row equals top of right column
  M = -6.25 := by
sorry

end NUMINAMATH_CALUDE_grid_arithmetic_sequences_l1488_148858


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l1488_148822

theorem smallest_lcm_with_gcd_5 :
  ∀ m n : ℕ,
  1000 ≤ m ∧ m < 10000 ∧
  1000 ≤ n ∧ n < 10000 ∧
  Nat.gcd m n = 5 →
  203010 ≤ Nat.lcm m n ∧
  ∃ m₀ n₀ : ℕ,
    1000 ≤ m₀ ∧ m₀ < 10000 ∧
    1000 ≤ n₀ ∧ n₀ < 10000 ∧
    Nat.gcd m₀ n₀ = 5 ∧
    Nat.lcm m₀ n₀ = 203010 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l1488_148822


namespace NUMINAMATH_CALUDE_function_value_solution_l1488_148847

theorem function_value_solution (x : ℝ) :
  (x^2 + x - 1 = 5) ↔ (x = 2 ∨ x = -3) := by sorry

end NUMINAMATH_CALUDE_function_value_solution_l1488_148847


namespace NUMINAMATH_CALUDE_chad_earnings_problem_l1488_148835

/-- Chad's earnings and savings problem -/
theorem chad_earnings_problem (mowing_earnings : ℝ) : 
  (mowing_earnings + 250 + 150 + 150) * 0.4 = 460 → mowing_earnings = 600 := by
  sorry

end NUMINAMATH_CALUDE_chad_earnings_problem_l1488_148835


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l1488_148811

theorem min_value_complex_expression (w : ℂ) (h : Complex.abs (w - (3 - I)) = 3) :
  Complex.abs (w + (1 - I))^2 + Complex.abs (w - (7 - 2*I))^2 = 38 := by
  sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l1488_148811


namespace NUMINAMATH_CALUDE_two_equidistant_points_l1488_148882

/-- A line in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Three distinct lines in a plane -/
structure ThreeLines :=
  (l₁ : Line)
  (l₂ : Line)
  (l₃ : Line)
  (distinct : l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃)

/-- l₂ intersects l₁ -/
def intersects (l₁ l₂ : Line) : Prop :=
  l₁.slope ≠ l₂.slope

/-- l₃ is parallel to l₁ -/
def parallel (l₁ l₃ : Line) : Prop :=
  l₁.slope = l₃.slope

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A point is equidistant from three lines -/
def equidistant (p : Point) (lines : ThreeLines) : Prop := sorry

/-- The main theorem -/
theorem two_equidistant_points (lines : ThreeLines) 
  (h₁ : intersects lines.l₁ lines.l₂)
  (h₂ : parallel lines.l₁ lines.l₃) :
  ∃! (p₁ p₂ : Point), p₁ ≠ p₂ ∧ 
    equidistant p₁ lines ∧ 
    equidistant p₂ lines ∧
    ∀ (p : Point), equidistant p lines → p = p₁ ∨ p = p₂ :=
sorry

end NUMINAMATH_CALUDE_two_equidistant_points_l1488_148882


namespace NUMINAMATH_CALUDE_tangent_intersections_symmetric_l1488_148832

/-- A line intersecting two hyperbolas -/
structure IntersectingLine where
  m : ℝ  -- slope of the line
  q : ℝ  -- y-intercept of the line

/-- The intersection points of tangents to a hyperbola -/
structure TangentIntersection where
  x : ℝ
  y : ℝ

/-- Calculate the intersection point of tangents for y = 1/x hyperbola -/
noncomputable def tangentIntersection1 (line : IntersectingLine) : TangentIntersection :=
  { x := 2 * line.m / line.q
  , y := -2 / line.q }

/-- Calculate the intersection point of tangents for y = -1/x hyperbola -/
noncomputable def tangentIntersection2 (line : IntersectingLine) : TangentIntersection :=
  { x := -2 * line.m / line.q
  , y := 2 / line.q }

/-- Two points are symmetric about the origin -/
def symmetricAboutOrigin (p1 p2 : TangentIntersection) : Prop :=
  p1.x = -p2.x ∧ p1.y = -p2.y

/-- Main theorem: The intersection points of tangents are symmetric about the origin -/
theorem tangent_intersections_symmetric (line : IntersectingLine) :
  symmetricAboutOrigin (tangentIntersection1 line) (tangentIntersection2 line) := by
  sorry

end NUMINAMATH_CALUDE_tangent_intersections_symmetric_l1488_148832


namespace NUMINAMATH_CALUDE_trigonometric_expressions_l1488_148830

theorem trigonometric_expressions (θ : ℝ) 
  (h : (4 * Real.sin θ - 2 * Real.cos θ) / (3 * Real.sin θ + 5 * Real.cos θ) = 6 / 11) : 
  (5 * (Real.cos θ)^2) / ((Real.sin θ)^2 + 2 * Real.sin θ * Real.cos θ - 3 * (Real.cos θ)^2) = 1 ∧ 
  1 - 4 * Real.sin θ * Real.cos θ + 2 * (Real.cos θ)^2 = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expressions_l1488_148830


namespace NUMINAMATH_CALUDE_equivalent_operations_l1488_148806

theorem equivalent_operations (x : ℝ) : 
  (x * (4/5)) / (2/7) = x * (7/4) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operations_l1488_148806


namespace NUMINAMATH_CALUDE_average_weight_decrease_l1488_148810

theorem average_weight_decrease (n : ℕ) (initial_avg : ℝ) (new_weight : ℝ) :
  n = 30 →
  initial_avg = 102 →
  new_weight = 40 →
  let total_weight := n * initial_avg
  let new_total_weight := total_weight + new_weight
  let new_avg := new_total_weight / (n + 1)
  initial_avg - new_avg = 2 := by
sorry

end NUMINAMATH_CALUDE_average_weight_decrease_l1488_148810


namespace NUMINAMATH_CALUDE_garden_bug_problem_l1488_148898

theorem garden_bug_problem (initial_plants : ℕ) (day1_eaten : ℕ) (day3_eaten : ℕ) : 
  initial_plants = 30 →
  day1_eaten = 20 →
  day3_eaten = 1 →
  initial_plants - day1_eaten - (initial_plants - day1_eaten) / 2 - day3_eaten = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_bug_problem_l1488_148898


namespace NUMINAMATH_CALUDE_mass_of_six_moles_l1488_148897

/-- Given a compound with a molecular weight of 444 g/mol, 
    the mass of 6 moles of this compound is 2664 g. -/
theorem mass_of_six_moles (molecular_weight : ℝ) (h : molecular_weight = 444) : 
  6 * molecular_weight = 2664 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_six_moles_l1488_148897


namespace NUMINAMATH_CALUDE_largest_multiple_of_nine_below_negative_seventy_l1488_148814

theorem largest_multiple_of_nine_below_negative_seventy :
  ∀ n : ℤ, n % 9 = 0 ∧ n < -70 → n ≤ -72 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_nine_below_negative_seventy_l1488_148814


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l1488_148849

theorem complex_magnitude_product : Complex.abs (3 - 5 * Complex.I) * Complex.abs (3 + 5 * Complex.I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l1488_148849


namespace NUMINAMATH_CALUDE_sqrt_five_squared_times_seven_fourth_l1488_148829

theorem sqrt_five_squared_times_seven_fourth (x : ℝ) : 
  x = Real.sqrt (5^2 * 7^4) → x = 245 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_squared_times_seven_fourth_l1488_148829


namespace NUMINAMATH_CALUDE_f_geq_one_range_f_lt_a_plus_two_l1488_148893

/-- The quadratic function f(x) = ax² + (2-a)x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 - a) * x + a

/-- Theorem stating the range of a for which f(x) ≥ 1 holds for all real x -/
theorem f_geq_one_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 1) ↔ a ≥ 2 * Real.sqrt 3 / 3 := by sorry

/-- Helper function to describe the solution set of f(x) < a+2 -/
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x < 1}
  else if a > 0 then {x | -2/a < x ∧ x < 1}
  else if -2 < a ∧ a < 0 then {x | x < 1 ∨ x > -2/a}
  else if a = -2 then Set.univ
  else {x | x < -2/a ∨ x > 1}

/-- Theorem stating the solution set of f(x) < a+2 -/
theorem f_lt_a_plus_two (a : ℝ) (x : ℝ) :
  f a x < a + 2 ↔ x ∈ solution_set a := by sorry

end NUMINAMATH_CALUDE_f_geq_one_range_f_lt_a_plus_two_l1488_148893


namespace NUMINAMATH_CALUDE_volume_of_five_adjacent_cubes_l1488_148828

/-- The volume of a solid formed by placing n equal cubes with side length s adjacent to each other -/
def volume_of_adjacent_cubes (n : ℕ) (s : ℝ) : ℝ := n * s^3

/-- Theorem: The volume of a solid formed by placing five equal cubes with side length 5 cm adjacent to each other is 625 cm³ -/
theorem volume_of_five_adjacent_cubes :
  volume_of_adjacent_cubes 5 5 = 625 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_five_adjacent_cubes_l1488_148828


namespace NUMINAMATH_CALUDE_halfway_fraction_l1488_148887

theorem halfway_fraction : 
  let a := (1 : ℚ) / 2
  let b := (3 : ℚ) / 4
  (a + b) / 2 = (5 : ℚ) / 8 := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_l1488_148887


namespace NUMINAMATH_CALUDE_constant_r_is_circle_l1488_148800

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- A circle centered at the origin -/
def Circle (radius : ℝ) := {p : PolarPoint | p.r = radius}

/-- The set of points satisfying r = 5 in polar coordinates -/
def ConstantR : Set PolarPoint := {p : PolarPoint | p.r = 5}

/-- Theorem stating that ConstantR is a circle with radius 5 -/
theorem constant_r_is_circle : ConstantR = Circle 5 := by sorry

end NUMINAMATH_CALUDE_constant_r_is_circle_l1488_148800


namespace NUMINAMATH_CALUDE_function_inequality_l1488_148845

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x, (x - 3) * deriv f x ≤ 0) : 
  f 0 + f 6 ≤ 2 * f 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1488_148845


namespace NUMINAMATH_CALUDE_polynomial_coefficient_a1_l1488_148852

theorem polynomial_coefficient_a1 (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^10 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
            a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + a₉*(x-1)^9 + a₁₀*(x-1)^10) →
  a₁ = 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_a1_l1488_148852


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l1488_148866

theorem arithmetic_simplification : 2 - (-3) - 4 - (-5) * 2 - 6 - (-7) = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l1488_148866


namespace NUMINAMATH_CALUDE_open_box_volume_l1488_148843

/-- The volume of an open box formed by cutting squares from a rectangular sheet -/
theorem open_box_volume (sheet_length sheet_width cut_size : ℝ)
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_size = 7)
  (h4 : sheet_length > 2 * cut_size)
  (h5 : sheet_width > 2 * cut_size) :
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 5244 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l1488_148843


namespace NUMINAMATH_CALUDE_sin_cos_relation_l1488_148801

theorem sin_cos_relation (α : Real) (h : Real.sin (π / 3 + α) = 1 / 3) :
  Real.cos (α - 7 * π / 6) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_relation_l1488_148801


namespace NUMINAMATH_CALUDE_no_real_numbers_with_integer_roots_l1488_148891

theorem no_real_numbers_with_integer_roots : 
  ¬ ∃ (a b c : ℝ), 
    (∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) ∧
    (∃ (y₁ y₂ : ℤ), y₁ ≠ y₂ ∧ (a+1) * y₁^2 + (b+1) * y₁ + (c+1) = 0 ∧ (a+1) * y₂^2 + (b+1) * y₂ + (c+1) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_no_real_numbers_with_integer_roots_l1488_148891


namespace NUMINAMATH_CALUDE_evaluate_expression_l1488_148850

theorem evaluate_expression : 5 - 7 * (8 - 3^2) * 4 = 33 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1488_148850


namespace NUMINAMATH_CALUDE_intersection_sum_l1488_148818

theorem intersection_sum (a b : ℚ) : 
  (∀ x y : ℚ, x = (1/3) * y + a ↔ y = (1/3) * x + b) →
  (2 : ℚ) = (1/3) * 3 + a →
  (3 : ℚ) = (1/3) * 2 + b →
  a + b = 10/3 := by sorry

end NUMINAMATH_CALUDE_intersection_sum_l1488_148818


namespace NUMINAMATH_CALUDE_additional_amount_needed_l1488_148809

def fundraiser_goal : ℕ := 750
def bronze_donation : ℕ := 25
def silver_donation : ℕ := 50
def gold_donation : ℕ := 100
def bronze_count : ℕ := 10
def silver_count : ℕ := 7
def gold_count : ℕ := 1

theorem additional_amount_needed : 
  fundraiser_goal - (bronze_donation * bronze_count + silver_donation * silver_count + gold_donation * gold_count) = 50 := by
  sorry

end NUMINAMATH_CALUDE_additional_amount_needed_l1488_148809


namespace NUMINAMATH_CALUDE_wage_cut_and_raise_l1488_148863

theorem wage_cut_and_raise (original_wage : ℝ) (h : original_wage > 0) :
  let cut_wage := 0.7 * original_wage
  let required_raise := (original_wage / cut_wage) - 1
  ∃ ε > 0, abs (required_raise - 0.4286) < ε :=
by sorry

end NUMINAMATH_CALUDE_wage_cut_and_raise_l1488_148863
