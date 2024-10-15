import Mathlib

namespace NUMINAMATH_CALUDE_range_of_f_on_interval_existence_of_a_l56_5643

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a

-- Part 1
theorem range_of_f_on_interval :
  let f1 := f 1
  ∃ (y : ℝ), y ∈ Set.range (fun x => f1 x) ∩ Set.Icc 0 4 ∧
  ∀ (z : ℝ), z ∈ Set.range (fun x => f1 x) ∩ Set.Icc 0 3 → z ∈ Set.Icc 0 4 :=
sorry

-- Part 2
theorem existence_of_a :
  ∃ (a : ℝ), 
    (∀ x, x ∈ Set.Icc (-1) 1 → f a x ∈ Set.Icc (-2) 2) ∧
    (∀ y, y ∈ Set.Icc (-2) 2 → ∃ x ∈ Set.Icc (-1) 1, f a x = y) ∧
    a = -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_existence_of_a_l56_5643


namespace NUMINAMATH_CALUDE_fraction_subtraction_equality_l56_5674

theorem fraction_subtraction_equality : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equality_l56_5674


namespace NUMINAMATH_CALUDE_cyclist_time_is_pi_over_five_l56_5673

/-- Represents the problem of a cyclist riding on a highway strip -/
def CyclistProblem (width : ℝ) (length : ℝ) (large_semicircle_distance : ℝ) (speed : ℝ) : Prop :=
  width = 40 ∧ 
  length = 5280 ∧ 
  large_semicircle_distance = 528 ∧ 
  speed = 5

/-- Calculates the time taken for the cyclist to cover the entire strip -/
noncomputable def cycleTime (width : ℝ) (length : ℝ) (large_semicircle_distance : ℝ) (speed : ℝ) : ℝ :=
  (Real.pi * length) / (speed * width)

/-- Theorem stating that the time taken is π/5 hours -/
theorem cyclist_time_is_pi_over_five 
  (width : ℝ) (length : ℝ) (large_semicircle_distance : ℝ) (speed : ℝ) 
  (h : CyclistProblem width length large_semicircle_distance speed) : 
  cycleTime width length large_semicircle_distance speed = Real.pi / 5 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_time_is_pi_over_five_l56_5673


namespace NUMINAMATH_CALUDE_mark_leftover_money_l56_5640

-- Define the given conditions
def old_hourly_wage : ℝ := 40
def raise_percentage : ℝ := 0.05
def hours_per_day : ℝ := 8
def days_per_week : ℝ := 5
def old_weekly_bills : ℝ := 600
def personal_trainer_cost : ℝ := 100

-- Define the calculation steps
def new_hourly_wage : ℝ := old_hourly_wage * (1 + raise_percentage)
def weekly_hours : ℝ := hours_per_day * days_per_week
def weekly_earnings : ℝ := new_hourly_wage * weekly_hours
def new_weekly_expenses : ℝ := old_weekly_bills + personal_trainer_cost

-- Theorem to prove
theorem mark_leftover_money :
  weekly_earnings - new_weekly_expenses = 980 := by
  sorry


end NUMINAMATH_CALUDE_mark_leftover_money_l56_5640


namespace NUMINAMATH_CALUDE_min_value_tangent_l56_5613

/-- Given a function f(x) = 2cos(x) - 3sin(x) that reaches its minimum value when x = θ,
    prove that tan(θ) = -3/2 --/
theorem min_value_tangent (θ : ℝ) (h : ∀ x, 2 * Real.cos x - 3 * Real.sin x ≥ 2 * Real.cos θ - 3 * Real.sin θ) :
  Real.tan θ = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_tangent_l56_5613


namespace NUMINAMATH_CALUDE_regular_polygon_with_160_degree_angles_has_18_sides_l56_5677

/-- A regular polygon with interior angles measuring 160° has 18 sides. -/
theorem regular_polygon_with_160_degree_angles_has_18_sides :
  ∀ n : ℕ, n ≥ 3 →
  (∀ θ : ℝ, θ = 160 → (n : ℝ) * θ = (n - 2 : ℝ) * 180) →
  n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_160_degree_angles_has_18_sides_l56_5677


namespace NUMINAMATH_CALUDE_hexagonal_lattice_triangles_l56_5688

/-- Represents a point in the hexagonal lattice -/
structure LatticePoint where
  x : ℝ
  y : ℝ

/-- The hexagonal lattice with two concentric hexagons -/
structure HexagonalLattice where
  center : LatticePoint
  inner_hexagon : List LatticePoint
  outer_hexagon : List LatticePoint

/-- Checks if three points form an equilateral triangle -/
def is_equilateral_triangle (p1 p2 p3 : LatticePoint) : Prop :=
  sorry

/-- Counts the number of equilateral triangles in the lattice -/
def count_equilateral_triangles (lattice : HexagonalLattice) : ℕ :=
  sorry

/-- Main theorem: The number of equilateral triangles in the described hexagonal lattice is 18 -/
theorem hexagonal_lattice_triangles 
  (lattice : HexagonalLattice)
  (h1 : lattice.inner_hexagon.length = 6)
  (h2 : lattice.outer_hexagon.length = 6)
  (h3 : ∀ p ∈ lattice.inner_hexagon, 
    ∃ q ∈ lattice.inner_hexagon, 
    (p.x - q.x)^2 + (p.y - q.y)^2 = 1)
  (h4 : ∀ p ∈ lattice.outer_hexagon, 
    ∃ q ∈ lattice.inner_hexagon, 
    (p.x - q.x)^2 + (p.y - q.y)^2 = 4) :
  count_equilateral_triangles lattice = 18 :=
sorry

end NUMINAMATH_CALUDE_hexagonal_lattice_triangles_l56_5688


namespace NUMINAMATH_CALUDE_least_whole_number_for_ratio_l56_5617

theorem least_whole_number_for_ratio (x : ℕ) : x = 3 ↔ 
  (x > 0 ∧ 
   ∀ y : ℕ, y > 0 → y < x → (6 - y : ℚ) / (7 - y) ≥ 16 / 21) ∧
  (6 - x : ℚ) / (7 - x) < 16 / 21 :=
by sorry

end NUMINAMATH_CALUDE_least_whole_number_for_ratio_l56_5617


namespace NUMINAMATH_CALUDE_four_students_three_communities_l56_5604

/-- The number of ways to assign students to communities -/
def assignStudents (num_students : ℕ) (num_communities : ℕ) : ℕ :=
  num_communities ^ num_students

/-- Theorem stating that assigning 4 students to 3 communities results in 3^4 arrangements -/
theorem four_students_three_communities :
  assignStudents 4 3 = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_four_students_three_communities_l56_5604


namespace NUMINAMATH_CALUDE_quadratic_coefficients_4x2_eq_3_l56_5698

/-- Given a quadratic equation ax^2 + bx + c = 0, returns the tuple (a, b, c) -/
def quadratic_coefficients (f : ℝ → ℝ) : ℝ × ℝ × ℝ := sorry

theorem quadratic_coefficients_4x2_eq_3 :
  quadratic_coefficients (fun x => 4 * x^2 - 3) = (4, 0, -3) := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_4x2_eq_3_l56_5698


namespace NUMINAMATH_CALUDE_morning_campers_count_l56_5619

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 39

/-- The additional number of campers who went rowing in the morning compared to the afternoon -/
def additional_morning_campers : ℕ := 5

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := afternoon_campers + additional_morning_campers

theorem morning_campers_count : morning_campers = 44 := by
  sorry

end NUMINAMATH_CALUDE_morning_campers_count_l56_5619


namespace NUMINAMATH_CALUDE_reciprocal_of_three_halves_l56_5666

theorem reciprocal_of_three_halves (x : ℚ) : x = 3 / 2 → 1 / x = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_three_halves_l56_5666


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_solution_l56_5670

theorem complex_magnitude_equation_solution :
  ∃ x : ℝ, x > 0 ∧ 
  Complex.abs (x + Complex.I * Real.sqrt 7) * Complex.abs (3 - 2 * Complex.I * Real.sqrt 5) = 45 ∧
  x = Real.sqrt (1822 / 29) :=
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_solution_l56_5670


namespace NUMINAMATH_CALUDE_divisor_and_expression_l56_5638

theorem divisor_and_expression (k : ℕ) : 
  (30^k : ℕ) ∣ 929260 → 3^k - k^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisor_and_expression_l56_5638


namespace NUMINAMATH_CALUDE_pin_purchase_cost_l56_5693

theorem pin_purchase_cost (num_pins : ℕ) (original_price : ℚ) (discount_percent : ℚ) :
  num_pins = 10 →
  original_price = 20 →
  discount_percent = 15 / 100 →
  (num_pins : ℚ) * (original_price * (1 - discount_percent)) = 170 :=
by sorry

end NUMINAMATH_CALUDE_pin_purchase_cost_l56_5693


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l56_5691

/-- Given an arithmetic sequence {a_n}, prove that a₃ + a₆ + a₉ = 33,
    when a₁ + a₄ + a₇ = 45 and a₂ + a₅ + a₈ = 39 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 + a 4 + a 7 = 45 →
  a 2 + a 5 + a 8 = 39 →
  a 3 + a 6 + a 9 = 33 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l56_5691


namespace NUMINAMATH_CALUDE_painting_supplies_theorem_l56_5614

/-- Represents the cost and quantity of painting supplies -/
structure PaintingSupplies where
  brush_cost : ℝ
  board_cost : ℝ
  total_items : ℕ
  max_cost : ℝ

/-- Theorem stating the properties of the painting supplies purchase -/
theorem painting_supplies_theorem (ps : PaintingSupplies) 
  (h1 : 340 / ps.brush_cost = 300 / ps.board_cost)
  (h2 : ps.brush_cost = ps.board_cost + 2)
  (h3 : ps.total_items = 30)
  (h4 : ∀ a : ℕ, a ≤ ps.total_items → 
    ps.brush_cost * (ps.total_items - a) + ps.board_cost * a ≤ ps.max_cost) :
  ps.brush_cost = 17 ∧ ps.board_cost = 15 ∧ 
  (∃ min_boards : ℕ, min_boards = 18 ∧ 
    ∀ a : ℕ, a < min_boards → 
      ps.brush_cost * (ps.total_items - a) + ps.board_cost * a > ps.max_cost) := by
  sorry

#check painting_supplies_theorem

end NUMINAMATH_CALUDE_painting_supplies_theorem_l56_5614


namespace NUMINAMATH_CALUDE_remainder_2519_div_8_l56_5675

theorem remainder_2519_div_8 : 2519 % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2519_div_8_l56_5675


namespace NUMINAMATH_CALUDE_remaining_time_indeterminate_l56_5684

/-- Represents the state of a math test -/
structure MathTest where
  totalProblems : ℕ
  firstInterval : ℕ
  secondInterval : ℕ
  problemsCompletedFirst : ℕ
  problemsCompletedSecond : ℕ
  problemsLeft : ℕ

/-- Theorem stating that the remaining time cannot be determined -/
theorem remaining_time_indeterminate (test : MathTest) 
  (h1 : test.totalProblems = 75)
  (h2 : test.firstInterval = 20)
  (h3 : test.secondInterval = 20)
  (h4 : test.problemsCompletedFirst = 10)
  (h5 : test.problemsCompletedSecond = 2 * test.problemsCompletedFirst)
  (h6 : test.problemsLeft = 45)
  (h7 : test.totalProblems = test.problemsCompletedFirst + test.problemsCompletedSecond + test.problemsLeft) :
  ¬∃ (remainingTime : ℕ), True := by
  sorry

#check remaining_time_indeterminate

end NUMINAMATH_CALUDE_remaining_time_indeterminate_l56_5684


namespace NUMINAMATH_CALUDE_inequality_proof_l56_5699

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  a^4 * b + b^4 * c + c^4 * a > a * b^4 + b * c^4 + c * a^4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l56_5699


namespace NUMINAMATH_CALUDE_impossible_arrangement_l56_5600

/-- Represents a person at the table -/
structure Person :=
  (id : Nat)

/-- Represents the circular table with 40 people -/
def Table := Fin 40 → Person

/-- Returns the number of people between two positions on the table -/
def distanceBetween (table : Table) (p1 p2 : Fin 40) : Nat :=
  sorry

/-- Checks if two people have a common acquaintance -/
def haveCommonAcquaintance (table : Table) (p1 p2 : Fin 40) : Prop :=
  sorry

/-- The main theorem stating the impossibility of the arrangement -/
theorem impossible_arrangement (table : Table) : 
  ¬(∀ (p1 p2 : Fin 40), 
    (distanceBetween table p1 p2 % 2 = 0 → haveCommonAcquaintance table p1 p2) ∧
    (distanceBetween table p1 p2 % 2 = 1 → ¬haveCommonAcquaintance table p1 p2)) :=
  sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l56_5600


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l56_5657

theorem complex_magnitude_theorem (ω : ℂ) (h : ω = 8 + 3*I) : 
  Complex.abs (ω^2 + 6*ω + 73) = Real.sqrt 32740 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l56_5657


namespace NUMINAMATH_CALUDE_sum_of_xyz_equals_six_l56_5671

theorem sum_of_xyz_equals_six (a b : ℝ) (x y z : ℤ) : 
  a^2 = 9/36 → 
  b^2 = (1 + Real.sqrt 3)^2 / 8 → 
  a < 0 → 
  b > 0 → 
  (a - b)^2 = (x : ℝ) * Real.sqrt y / z → 
  x + y + z = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_xyz_equals_six_l56_5671


namespace NUMINAMATH_CALUDE_average_equation_l56_5661

theorem average_equation (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 74 → a = 28 := by sorry

end NUMINAMATH_CALUDE_average_equation_l56_5661


namespace NUMINAMATH_CALUDE_expression_simplification_l56_5615

theorem expression_simplification (x : ℝ) (h : x = -2) :
  (1 - 2 / (x + 1)) / ((x^2 - x) / (x^2 - 1)) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l56_5615


namespace NUMINAMATH_CALUDE_initial_girls_percentage_l56_5694

theorem initial_girls_percentage 
  (initial_total : ℕ)
  (new_boys : ℕ)
  (new_girls_percentage : ℚ)
  (h1 : initial_total = 20)
  (h2 : new_boys = 5)
  (h3 : new_girls_percentage = 32 / 100) :
  let initial_girls := (new_girls_percentage * (initial_total + new_boys)).floor
  let initial_girls_percentage := initial_girls / initial_total
  initial_girls_percentage = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_initial_girls_percentage_l56_5694


namespace NUMINAMATH_CALUDE_minimum_value_of_function_minimum_value_achieved_l56_5618

theorem minimum_value_of_function (x : ℝ) (h : x > 1) :
  2 * x + 2 / (x - 1) ≥ 6 :=
sorry

theorem minimum_value_achieved (x : ℝ) (h : x > 1) :
  2 * x + 2 / (x - 1) = 6 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_function_minimum_value_achieved_l56_5618


namespace NUMINAMATH_CALUDE_range_of_m_l56_5631

-- Define propositions P and Q as functions of m
def P (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the theorem
theorem range_of_m : 
  (∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m)) → 
  (∀ m : ℝ, (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) ↔ (P m ∨ Q m) ∧ ¬(P m ∧ Q m)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l56_5631


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l56_5690

theorem min_value_sum_reciprocals (n : ℕ) (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) :
  (1 / (1 + a^n)) + (1 / (1 + b^n)) ≥ 1 ∧ 
  ((1 / (1 + 1^n)) + (1 / (1 + 1^n)) = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l56_5690


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_l56_5662

/-- The line equation in polar coordinates -/
def line_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.sqrt 3 * Real.cos θ - Real.sin θ) = 2

/-- The circle equation in polar coordinates -/
def circle_equation (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.sin θ

/-- The theorem stating that the point (2, π/6) satisfies both equations -/
theorem intersection_point_satisfies_equations :
  line_equation 2 (Real.pi / 6) ∧ circle_equation 2 (Real.pi / 6) := by
  sorry


end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_l56_5662


namespace NUMINAMATH_CALUDE_scientific_notation_equality_coefficient_range_l56_5612

-- Define the number we want to express in scientific notation
def number : ℕ := 18480000

-- Define the components of the scientific notation
def coefficient : ℝ := 1.848
def exponent : ℕ := 7

-- Theorem to prove
theorem scientific_notation_equality :
  (coefficient * (10 : ℝ) ^ exponent : ℝ) = number := by
  sorry

-- Verify that the coefficient is between 1 and 10
theorem coefficient_range :
  1 < coefficient ∧ coefficient < 10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_coefficient_range_l56_5612


namespace NUMINAMATH_CALUDE_carol_peanuts_count_l56_5626

/-- The number of peanuts Carol initially collects -/
def initial_peanuts : ℕ := 2

/-- The number of peanuts Carol's father gives her -/
def given_peanuts : ℕ := 5

/-- The total number of peanuts Carol has -/
def total_peanuts : ℕ := initial_peanuts + given_peanuts

theorem carol_peanuts_count : total_peanuts = 7 := by
  sorry

end NUMINAMATH_CALUDE_carol_peanuts_count_l56_5626


namespace NUMINAMATH_CALUDE_linear_equation_solutions_l56_5601

theorem linear_equation_solutions (x y : ℝ) : 
  (x = 1 ∧ y = 2 → 2*x + y = 4) ∧
  (x = 2 ∧ y = 0 → 2*x + y = 4) ∧
  (x = 0.5 ∧ y = 3 → 2*x + y = 4) ∧
  (x = -2 ∧ y = 4 → 2*x + y ≠ 4) := by
  sorry

#check linear_equation_solutions

end NUMINAMATH_CALUDE_linear_equation_solutions_l56_5601


namespace NUMINAMATH_CALUDE_sugar_cups_correct_l56_5645

/-- Represents the number of cups of sugar in the lemonade mixture -/
def sugar : ℕ := 28

/-- Represents the number of cups of water in the lemonade mixture -/
def water : ℕ := 56

/-- The total number of cups used in the mixture -/
def total_cups : ℕ := 84

/-- Theorem stating that the number of cups of sugar is correct given the conditions -/
theorem sugar_cups_correct :
  (sugar + water = total_cups) ∧ (2 * sugar = water) ∧ (sugar = 28) := by
  sorry

end NUMINAMATH_CALUDE_sugar_cups_correct_l56_5645


namespace NUMINAMATH_CALUDE_goat_price_calculation_l56_5689

theorem goat_price_calculation (total_cost total_hens total_goats hen_price : ℕ) 
  (h1 : total_cost = 10000)
  (h2 : total_hens = 35)
  (h3 : total_goats = 15)
  (h4 : hen_price = 125) :
  (total_cost - total_hens * hen_price) / total_goats = 375 := by
  sorry

end NUMINAMATH_CALUDE_goat_price_calculation_l56_5689


namespace NUMINAMATH_CALUDE_function_value_proof_l56_5682

/-- Given a function f(x) = ax^5 + bx^3 + cx + 8, prove that if f(-2) = 10, then f(2) = 6 -/
theorem function_value_proof (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^5 + b * x^3 + c * x + 8
  f (-2) = 10 → f 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_function_value_proof_l56_5682


namespace NUMINAMATH_CALUDE_cube_sum_one_l56_5686

theorem cube_sum_one (a b c : ℝ) 
  (sum_one : a + b + c = 1)
  (sum_products : a * b + a * c + b * c = -5)
  (product : a * b * c = 5) :
  a^3 + b^3 + c^3 = 1 := by sorry

end NUMINAMATH_CALUDE_cube_sum_one_l56_5686


namespace NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l56_5602

/-- The length of a diagonal in a quadrilateral with given area and offsets -/
theorem diagonal_length_of_quadrilateral (area : ℝ) (offset1 offset2 : ℝ) :
  area = 210 →
  offset1 = 9 →
  offset2 = 6 →
  (∃ d : ℝ, area = 0.5 * d * (offset1 + offset2) ∧ d = 28) :=
by sorry

end NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l56_5602


namespace NUMINAMATH_CALUDE_painters_work_days_l56_5633

/-- Represents the time taken to complete a job given a number of painters -/
def time_to_complete (num_painters : ℕ) (work_days : ℚ) : ℚ := num_painters * work_days

/-- Proves that if 6 painters can finish a job in 2 work-days, 
    then 4 painters will take 3 work-days to finish the same job -/
theorem painters_work_days (initial_painters : ℕ) (initial_days : ℚ) 
  (new_painters : ℕ) : 
  initial_painters = 6 → initial_days = 2 → new_painters = 4 →
  time_to_complete new_painters (3 : ℚ) = time_to_complete initial_painters initial_days :=
by
  sorry

end NUMINAMATH_CALUDE_painters_work_days_l56_5633


namespace NUMINAMATH_CALUDE_running_race_participants_l56_5665

theorem running_race_participants (first_grade : ℕ) (second_grade : ℕ) : 
  first_grade = 8 →
  second_grade = 5 * first_grade →
  first_grade + second_grade = 48 := by
  sorry

end NUMINAMATH_CALUDE_running_race_participants_l56_5665


namespace NUMINAMATH_CALUDE_kevins_phone_repair_l56_5678

/-- Given the initial conditions of Kevin's phone repair scenario, 
    prove that the number of phones each person needs to repair is 9. -/
theorem kevins_phone_repair 
  (initial_phones : ℕ) 
  (repaired_phones : ℕ) 
  (new_phones : ℕ) 
  (h1 : initial_phones = 15)
  (h2 : repaired_phones = 3)
  (h3 : new_phones = 6) :
  (initial_phones - repaired_phones + new_phones) / 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_kevins_phone_repair_l56_5678


namespace NUMINAMATH_CALUDE_ab_length_not_unique_l56_5622

/-- Given two line segments AC and BC with lengths 1 and 3 respectively,
    the length of AB cannot be uniquely determined. -/
theorem ab_length_not_unique (AC BC : ℝ) (hAC : AC = 1) (hBC : BC = 3) :
  ¬ ∃! AB : ℝ, (0 < AB ∧ AB < AC + BC) ∨ (AB = AC + BC ∨ AB = |BC - AC|) :=
sorry

end NUMINAMATH_CALUDE_ab_length_not_unique_l56_5622


namespace NUMINAMATH_CALUDE_time_after_2051_hours_l56_5628

/-- Calculates the time on a 12-hour clock after a given number of hours have passed -/
def timeAfter (startTime : Nat) (hoursPassed : Nat) : Nat :=
  (startTime + hoursPassed) % 12

/-- Proves that 2051 hours after 9 o'clock, it will be 8 o'clock on a 12-hour clock -/
theorem time_after_2051_hours :
  timeAfter 9 2051 = 8 := by
  sorry

#eval timeAfter 9 2051  -- This should output 8

end NUMINAMATH_CALUDE_time_after_2051_hours_l56_5628


namespace NUMINAMATH_CALUDE_dads_contribution_undetermined_l56_5649

/-- Represents the number of toy cars in Olaf's collection --/
structure ToyCarCollection where
  initial : ℕ
  fromUncle : ℕ
  fromGrandpa : ℕ
  fromAuntie : ℕ
  fromMum : ℕ
  fromDad : ℕ
  final : ℕ

/-- The conditions of Olaf's toy car collection --/
def olafCollection : ToyCarCollection where
  initial := 150
  fromUncle := 5
  fromGrandpa := 10
  fromAuntie := 6
  fromMum := 0  -- Unknown value
  fromDad := 0  -- Unknown value
  final := 196

/-- Theorem stating that Dad's contribution is undetermined --/
theorem dads_contribution_undetermined (c : ToyCarCollection) 
  (h1 : c.initial = 150)
  (h2 : c.fromGrandpa = 2 * c.fromUncle)
  (h3 : c.fromAuntie = c.fromUncle + 1)
  (h4 : c.final = 196)
  (h5 : c.final = c.initial + c.fromUncle + c.fromGrandpa + c.fromAuntie + c.fromMum + c.fromDad) :
  ∃ (x y : ℕ), x ≠ y ∧ 
    (c.fromMum = x ∧ c.fromDad = 25 - x) ∧
    (c.fromMum = y ∧ c.fromDad = 25 - y) :=
sorry

#check dads_contribution_undetermined

end NUMINAMATH_CALUDE_dads_contribution_undetermined_l56_5649


namespace NUMINAMATH_CALUDE_kiera_envelopes_l56_5641

theorem kiera_envelopes (blue : ℕ) (yellow : ℕ) (green : ℕ) 
  (h1 : blue = 14)
  (h2 : yellow = blue - 6)
  (h3 : green = 3 * yellow) :
  blue + yellow + green = 46 := by
  sorry

end NUMINAMATH_CALUDE_kiera_envelopes_l56_5641


namespace NUMINAMATH_CALUDE_josie_safari_count_l56_5629

/-- The total number of animals Josie counted on safari -/
def total_animals (antelopes rabbits hyenas wild_dogs leopards giraffes lions elephants : ℕ) : ℕ :=
  antelopes + rabbits + hyenas + wild_dogs + leopards + giraffes + lions + elephants

/-- Theorem stating the total number of animals Josie counted -/
theorem josie_safari_count : ∃ (antelopes rabbits hyenas wild_dogs leopards giraffes lions elephants : ℕ),
  antelopes = 80 ∧
  rabbits = antelopes + 34 ∧
  hyenas = antelopes + rabbits - 42 ∧
  wild_dogs = hyenas + 50 ∧
  leopards = rabbits / 2 ∧
  giraffes = antelopes + 15 ∧
  lions = leopards + giraffes ∧
  elephants = 3 * lions ∧
  total_animals antelopes rabbits hyenas wild_dogs leopards giraffes lions elephants = 1308 :=
by
  sorry

end NUMINAMATH_CALUDE_josie_safari_count_l56_5629


namespace NUMINAMATH_CALUDE_smallest_n_proof_l56_5623

/-- The capacity of adults on a single bench section -/
def adult_capacity : ℕ := 8

/-- The capacity of children on a single bench section -/
def child_capacity : ℕ := 12

/-- Predicate to check if a number of bench sections can seat an equal number of adults and children -/
def can_seat_equally (n : ℕ) : Prop :=
  ∃ (x : ℕ), x > 0 ∧ adult_capacity * n = x ∧ child_capacity * n = x

/-- The smallest positive integer number of bench sections that can seat an equal number of adults and children -/
def smallest_n : ℕ := 3

theorem smallest_n_proof :
  (can_seat_equally smallest_n) ∧
  (∀ m : ℕ, m > 0 ∧ m < smallest_n → ¬(can_seat_equally m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_proof_l56_5623


namespace NUMINAMATH_CALUDE_probability_two_pairs_is_5_21_l56_5685

-- Define the total number of socks and colors
def total_socks : ℕ := 10
def num_colors : ℕ := 5
def socks_per_color : ℕ := 2
def socks_drawn : ℕ := 5

-- Define the probability function
def probability_two_pairs : ℚ :=
  let total_combinations := Nat.choose total_socks socks_drawn
  let favorable_combinations := Nat.choose num_colors 2 * Nat.choose (num_colors - 2) 1 * socks_per_color
  (favorable_combinations : ℚ) / total_combinations

-- Theorem statement
theorem probability_two_pairs_is_5_21 : 
  probability_two_pairs = 5 / 21 := by sorry

end NUMINAMATH_CALUDE_probability_two_pairs_is_5_21_l56_5685


namespace NUMINAMATH_CALUDE_cabbage_sales_proof_l56_5607

def price_per_kg : ℝ := 2
def earnings_wednesday : ℝ := 30
def earnings_friday : ℝ := 24
def earnings_today : ℝ := 42

theorem cabbage_sales_proof :
  (earnings_wednesday + earnings_friday + earnings_today) / price_per_kg = 48 := by
  sorry

end NUMINAMATH_CALUDE_cabbage_sales_proof_l56_5607


namespace NUMINAMATH_CALUDE_investment_distribution_l56_5646

def total_investment : ℝ := 1500
def final_amount : ℝ := 1800
def years : ℕ := 3

def interest_rate_trusty : ℝ := 0.04
def interest_rate_solid : ℝ := 0.06
def interest_rate_quick : ℝ := 0.07

def compound_factor (rate : ℝ) (years : ℕ) : ℝ :=
  (1 + rate) ^ years

theorem investment_distribution (x y : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ total_investment →
  x * compound_factor interest_rate_trusty years +
  y * compound_factor interest_rate_solid years +
  (total_investment - x - y) * compound_factor interest_rate_quick years = final_amount →
  x = 375 := by sorry

end NUMINAMATH_CALUDE_investment_distribution_l56_5646


namespace NUMINAMATH_CALUDE_adam_has_more_apples_l56_5621

/-- The number of apples Adam has -/
def adam_apples : ℕ := 10

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 2

/-- The number of apples Michael has -/
def michael_apples : ℕ := 5

/-- Theorem: Adam has 3 more apples than the combined total of Jackie's and Michael's apples -/
theorem adam_has_more_apples : adam_apples - (jackie_apples + michael_apples) = 3 := by
  sorry


end NUMINAMATH_CALUDE_adam_has_more_apples_l56_5621


namespace NUMINAMATH_CALUDE_extremum_condition_l56_5608

/-- A function f: ℝ → ℝ has an extremum at x₀ if f(x₀) is either a maximum or minimum value of f. -/
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ ∀ x, f x ≥ f x₀

/-- Theorem: For a differentiable function f: ℝ → ℝ, f'(x₀) = 0 is a necessary but not sufficient
    condition for f(x₀) to be an extremum of f(x). -/
theorem extremum_condition (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x₀ : ℝ, HasExtremumAt f x₀ → deriv f x₀ = 0) ∧
  ¬(∀ x₀ : ℝ, deriv f x₀ = 0 → HasExtremumAt f x₀) :=
sorry

end NUMINAMATH_CALUDE_extremum_condition_l56_5608


namespace NUMINAMATH_CALUDE_min_value_x_minus_3y_l56_5630

theorem min_value_x_minus_3y (x y : ℝ) (hx : x > 1) (hy : y < 0) (h : 3 * y * (1 - x) = x + 8) :
  ∀ z, x - 3 * y ≥ z → z ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_minus_3y_l56_5630


namespace NUMINAMATH_CALUDE_first_year_selection_probability_l56_5687

/-- Represents the number of students in each year --/
structure StudentCounts where
  firstYear : ℕ
  secondYear : ℕ
  thirdYear : ℕ

/-- Represents the sampling information --/
structure SamplingInfo where
  thirdYearSample : ℕ

/-- Calculates the probability of a student being selected in stratified sampling --/
def stratifiedSamplingProbability (counts : StudentCounts) (info : SamplingInfo) : ℚ :=
  info.thirdYearSample / counts.thirdYear

/-- Theorem stating the probability of a first-year student being selected --/
theorem first_year_selection_probability 
  (counts : StudentCounts) 
  (info : SamplingInfo) 
  (h1 : counts.firstYear = 800)
  (h2 : counts.thirdYear = 500)
  (h3 : info.thirdYearSample = 25) :
  stratifiedSamplingProbability counts info = 1 / 20 := by
  sorry


end NUMINAMATH_CALUDE_first_year_selection_probability_l56_5687


namespace NUMINAMATH_CALUDE_correct_ratio_maintenance_l56_5697

-- Define the original recipe ratios
def flour_original : ℚ := 4
def sugar_original : ℚ := 7
def salt_original : ℚ := 2

-- Define Mary's mistake
def flour_mistake : ℚ := 2

-- Define the function to calculate additional flour needed
def additional_flour (f_orig f_mistake s_orig : ℚ) : ℚ :=
  f_orig - f_mistake

-- Define the function to calculate the difference between additional flour and salt
def flour_salt_difference (f_orig f_mistake s_orig : ℚ) : ℚ :=
  additional_flour f_orig f_mistake s_orig - 0

-- Theorem statement
theorem correct_ratio_maintenance :
  flour_salt_difference flour_original flour_mistake salt_original = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_ratio_maintenance_l56_5697


namespace NUMINAMATH_CALUDE_square_of_sum_of_squares_is_sum_of_squares_l56_5632

def is_sum_of_two_squares (x : ℕ) : Prop :=
  ∃ (a b : ℕ), x = a^2 + b^2 ∧ a > 0 ∧ b > 0

theorem square_of_sum_of_squares_is_sum_of_squares (n : ℕ) :
  (is_sum_of_two_squares (n - 1) ∧ 
   is_sum_of_two_squares n ∧ 
   is_sum_of_two_squares (n + 1)) →
  (is_sum_of_two_squares (n^2 - 1) ∧ 
   is_sum_of_two_squares (n^2) ∧ 
   is_sum_of_two_squares (n^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_square_of_sum_of_squares_is_sum_of_squares_l56_5632


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l56_5647

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- The function f(x) = x^2 - |x + a| -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 - |x + a|

theorem even_function_implies_a_zero :
  ∀ a : ℝ, IsEven (f a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l56_5647


namespace NUMINAMATH_CALUDE_base_prime_630_l56_5660

/-- Base prime representation of a natural number -/
def BasePrime : ℕ → List ℕ := sorry

/-- Check if a list represents a valid base prime representation -/
def IsValidBasePrime : List ℕ → Prop := sorry

theorem base_prime_630 : 
  let bp := BasePrime 630
  IsValidBasePrime bp ∧ bp = [2, 1, 1, 0] := by sorry

end NUMINAMATH_CALUDE_base_prime_630_l56_5660


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l56_5635

theorem division_multiplication_equality : (-150) / (-50) * (1/3 : ℚ) = 1 := by sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l56_5635


namespace NUMINAMATH_CALUDE_quartic_root_sum_l56_5696

theorem quartic_root_sum (a b c d : ℂ) : 
  (a^4 - 34*a^3 + 15*a^2 - 42*a - 8 = 0) →
  (b^4 - 34*b^3 + 15*b^2 - 42*b - 8 = 0) →
  (c^4 - 34*c^3 + 15*c^2 - 42*c - 8 = 0) →
  (d^4 - 34*d^3 + 15*d^2 - 42*d - 8 = 0) →
  (a / ((1/a) + b*c*d) + b / ((1/b) + a*c*d) + c / ((1/c) + a*b*d) + d / ((1/d) + a*b*c) = -161) :=
by sorry

end NUMINAMATH_CALUDE_quartic_root_sum_l56_5696


namespace NUMINAMATH_CALUDE_heartsuit_zero_heartsuit_self_heartsuit_positive_l56_5656

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem 1: x ♡ 0 = x^2 for all real x
theorem heartsuit_zero (x : ℝ) : heartsuit x 0 = x^2 := by sorry

-- Theorem 2: x ♡ x = 0 for all real x
theorem heartsuit_self (x : ℝ) : heartsuit x x = 0 := by sorry

-- Theorem 3: If x > y, then x ♡ y > 0 for all real x and y
theorem heartsuit_positive {x y : ℝ} (h : x > y) : heartsuit x y > 0 := by sorry

end NUMINAMATH_CALUDE_heartsuit_zero_heartsuit_self_heartsuit_positive_l56_5656


namespace NUMINAMATH_CALUDE_equation_has_two_solutions_l56_5625

-- Define the equation
def equation (x : ℝ) : Prop := Real.sqrt (9 - x) = x * Real.sqrt (9 - x)

-- Theorem statement
theorem equation_has_two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ equation a ∧ equation b ∧ 
  ∀ (x : ℝ), equation x → (x = a ∨ x = b) :=
sorry

end NUMINAMATH_CALUDE_equation_has_two_solutions_l56_5625


namespace NUMINAMATH_CALUDE_complex_equation_result_l56_5664

theorem complex_equation_result (m n : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : m / (1 + i) = 1 - n * i) : 
  m - n = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_result_l56_5664


namespace NUMINAMATH_CALUDE_min_value_theorem_l56_5663

open Real

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 1/x + 4/y ≤ 1/a + 4/b) ∧ (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 1/a + 4/b = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l56_5663


namespace NUMINAMATH_CALUDE_square_area_error_l56_5606

theorem square_area_error (s : ℝ) (s' : ℝ) (h : s' = 1.04 * s) :
  (s' ^ 2 - s ^ 2) / s ^ 2 = 0.0816 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l56_5606


namespace NUMINAMATH_CALUDE_no_positive_integers_satisfying_conditions_l56_5611

theorem no_positive_integers_satisfying_conditions : ¬∃ (a b c d : ℕ+) (p : ℕ),
  (a.val * b.val = c.val * d.val) ∧ 
  (a.val + b.val + c.val + d.val = p) ∧
  Nat.Prime p :=
sorry

end NUMINAMATH_CALUDE_no_positive_integers_satisfying_conditions_l56_5611


namespace NUMINAMATH_CALUDE_slowest_pump_time_l56_5683

/-- Three pumps with rates in ratio 2:3:4 fill a pool in 6 hours. The slowest pump fills it in 27 hours. -/
theorem slowest_pump_time (pool_volume : ℝ) (h : pool_volume > 0) : 
  ∃ (r₁ r₂ r₃ : ℝ), 
    r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧  -- Pump rates are positive
    r₂ = (3/2) * r₁ ∧           -- Ratio of rates
    r₃ = 2 * r₁ ∧               -- Ratio of rates
    (r₁ + r₂ + r₃) * 6 = pool_volume ∧  -- All pumps fill the pool in 6 hours
    r₁ * 27 = pool_volume       -- Slowest pump fills the pool in 27 hours
  := by sorry

end NUMINAMATH_CALUDE_slowest_pump_time_l56_5683


namespace NUMINAMATH_CALUDE_roots_sum_of_reciprocal_cubes_l56_5676

theorem roots_sum_of_reciprocal_cubes (a b c : ℝ) (r s : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : c ≠ 0) 
  (h3 : a * r^2 + b * r + c = 0) 
  (h4 : a * s^2 + b * s + c = 0) 
  (h5 : r ≠ 0) 
  (h6 : s ≠ 0) : 
  1 / r^3 + 1 / s^3 = (-b^3 + 3*a*b*c) / c^3 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_reciprocal_cubes_l56_5676


namespace NUMINAMATH_CALUDE_stream_speed_l56_5642

/-- Given a boat's travel times and distances, prove the speed of the stream --/
theorem stream_speed (downstream_distance : ℝ) (downstream_time : ℝ) 
  (upstream_distance : ℝ) (upstream_time : ℝ) 
  (h1 : downstream_distance = 100) 
  (h2 : downstream_time = 4)
  (h3 : upstream_distance = 75)
  (h4 : upstream_time = 15) :
  ∃ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = downstream_distance / downstream_time ∧
    boat_speed - stream_speed = upstream_distance / upstream_time ∧
    stream_speed = 10 := by
  sorry

#check stream_speed

end NUMINAMATH_CALUDE_stream_speed_l56_5642


namespace NUMINAMATH_CALUDE_perfect_square_product_l56_5605

theorem perfect_square_product (a b c d : ℤ) (h : a + b + c + d = 0) :
  ∃ k : ℤ, (a * b - c * d) * (b * c - a * d) * (c * a - b * d) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_product_l56_5605


namespace NUMINAMATH_CALUDE_arithmetic_equality_l56_5616

theorem arithmetic_equality : 5 * 7 + 9 * 4 - 36 / 3 = 59 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l56_5616


namespace NUMINAMATH_CALUDE_complex_square_simplification_l56_5650

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 + 3 * i)^2 = 7 + 24 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l56_5650


namespace NUMINAMATH_CALUDE_heat_engine_efficiencies_l56_5672

/-- Heat engine efficiencies problem -/
theorem heat_engine_efficiencies
  (η₀ η₁ η₂ Q₁₂ Q₁₃ Q₃₄ α : ℝ)
  (h₀ : η₀ = 1 - Q₃₄ / Q₁₂)
  (h₁ : η₁ = 1 - Q₁₃ / Q₁₂)
  (h₂ : η₂ = 1 - Q₃₄ / Q₁₃)
  (h₃ : η₂ = (η₀ - η₁) / (1 - η₁))
  (h₄ : η₁ < η₀)
  (h₅ : η₂ < η₀)
  (h₆ : η₀ < 1)
  (h₇ : η₁ < 1)
  (h₈ : η₁ = (1 - 0.01 * α) * η₀) :
  η₂ = α / (100 - (100 - α) * η₀) := by
  sorry

end NUMINAMATH_CALUDE_heat_engine_efficiencies_l56_5672


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l56_5636

theorem smallest_right_triangle_area :
  let side1 : ℝ := 6
  let side2 : ℝ := 8
  let area1 : ℝ := (1/2) * side1 * side2
  let area2 : ℝ := (1/2) * side1 * Real.sqrt (side2^2 - side1^2)
  min area1 area2 = 6 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l56_5636


namespace NUMINAMATH_CALUDE_polynomial_value_at_two_l56_5624

def f (x : ℝ) : ℝ := 4*x^5 + 2*x^4 + 3*x^3 - 2*x^2 - 2500*x + 434

theorem polynomial_value_at_two :
  f 2 = -3390 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_value_at_two_l56_5624


namespace NUMINAMATH_CALUDE_village_population_equality_l56_5655

/-- The number of years it takes for the populations of two villages to be equal -/
def years_to_equal_population (x_initial : ℕ) (x_decrease : ℕ) (y_initial : ℕ) (y_increase : ℕ) : ℕ :=
  (x_initial - y_initial) / (x_decrease + y_increase)

theorem village_population_equality :
  years_to_equal_population 68000 1200 42000 800 = 13 := by
  sorry

end NUMINAMATH_CALUDE_village_population_equality_l56_5655


namespace NUMINAMATH_CALUDE_simplify_expression_one_simplify_expression_two_l56_5644

-- Part 1
theorem simplify_expression_one : 2 * Real.sqrt 3 * 31.5 * 612 = 6 := by sorry

-- Part 2
theorem simplify_expression_two : 
  (Real.log 3 / Real.log 4 - Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 1/4 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_one_simplify_expression_two_l56_5644


namespace NUMINAMATH_CALUDE_eggs_problem_l56_5658

theorem eggs_problem (initial_eggs : ℕ) : 
  (initial_eggs / 2 : ℕ) - 15 = 21 → initial_eggs = 72 := by
  sorry

end NUMINAMATH_CALUDE_eggs_problem_l56_5658


namespace NUMINAMATH_CALUDE_find_b_value_l56_5652

theorem find_b_value (circle_sum : ℕ) (total_sum : ℕ) (d : ℕ) :
  circle_sum = 21 * 5 ∧
  total_sum = 69 ∧
  d + 5 + 9 = 21 →
  ∃ b : ℕ, b = 10 ∧ circle_sum - (2 + 8 + 9 + b + d) = total_sum :=
by sorry

end NUMINAMATH_CALUDE_find_b_value_l56_5652


namespace NUMINAMATH_CALUDE_total_selling_price_calculation_l56_5695

def calculate_total_selling_price (item1_cost item2_cost item3_cost : ℚ)
  (loss1 loss2 loss3 tax_rate : ℚ) (overhead : ℚ) : ℚ :=
  let total_purchase := item1_cost + item2_cost + item3_cost
  let tax := tax_rate * total_purchase
  let selling_price1 := item1_cost * (1 - loss1)
  let selling_price2 := item2_cost * (1 - loss2)
  let selling_price3 := item3_cost * (1 - loss3)
  let total_selling := selling_price1 + selling_price2 + selling_price3
  total_selling + overhead + tax

theorem total_selling_price_calculation :
  calculate_total_selling_price 750 1200 500 0.1 0.15 0.05 0.05 300 = 2592.5 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_calculation_l56_5695


namespace NUMINAMATH_CALUDE_equiangular_polygons_unique_angle_l56_5603

theorem equiangular_polygons_unique_angle : ∃! x : ℝ,
  0 < x ∧ x < 180 ∧
  ∃ n₁ : ℕ, n₁ ≥ 3 ∧ x = 180 - 360 / n₁ ∧
  ∃ n₃ : ℕ, n₃ ≥ 3 ∧ 3/2 * x = 180 - 360 / n₃ ∧
  n₁ ≠ n₃ := by sorry

end NUMINAMATH_CALUDE_equiangular_polygons_unique_angle_l56_5603


namespace NUMINAMATH_CALUDE_motorboat_travel_time_l56_5648

/-- Represents the scenario of a motorboat and kayak traveling on a river --/
structure RiverTrip where
  r : ℝ  -- River current speed (also kayak speed)
  p : ℝ  -- Motorboat speed relative to the river
  t : ℝ  -- Time for motorboat to travel from X to Y

/-- The conditions of the river trip --/
def trip_conditions (trip : RiverTrip) : Prop :=
  trip.p > 0 ∧ 
  trip.r > 0 ∧ 
  trip.t > 0 ∧ 
  (trip.p + trip.r) * trip.t + (trip.p - trip.r) * (11 - trip.t) = 12 * trip.r

/-- The theorem stating that under the given conditions, 
    the motorboat's initial travel time from X to Y is 4 hours --/
theorem motorboat_travel_time (trip : RiverTrip) : 
  trip_conditions trip → trip.t = 4 := by
  sorry

end NUMINAMATH_CALUDE_motorboat_travel_time_l56_5648


namespace NUMINAMATH_CALUDE_packet_weight_l56_5654

-- Define constants
def pounds_per_ton : ℚ := 2200
def ounces_per_pound : ℚ := 16
def bag_capacity_tons : ℚ := 13
def num_packets : ℚ := 1760

-- Define the theorem
theorem packet_weight :
  let total_weight := bag_capacity_tons * pounds_per_ton
  let weight_per_packet := total_weight / num_packets
  weight_per_packet = 16.25 := by
sorry

end NUMINAMATH_CALUDE_packet_weight_l56_5654


namespace NUMINAMATH_CALUDE_eve_walking_distance_l56_5668

theorem eve_walking_distance (ran_distance : Real) (extra_distance : Real) :
  ran_distance = 0.7 ∧ extra_distance = 0.1 →
  ∃ walked_distance : Real, walked_distance = ran_distance - extra_distance ∧ walked_distance = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_eve_walking_distance_l56_5668


namespace NUMINAMATH_CALUDE_complex_number_equal_parts_l56_5653

theorem complex_number_equal_parts (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (2 - Complex.I)
  Complex.re z = Complex.im z → a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equal_parts_l56_5653


namespace NUMINAMATH_CALUDE_children_at_play_l56_5680

/-- Represents the number of children attending a play given specific conditions --/
def children_attending (adult_price child_price total_people total_revenue : ℕ) 
  (senior_citizens group_size : ℕ) : ℕ :=
  total_people - (total_revenue - child_price * (total_people - senior_citizens - group_size)) / 
    (adult_price - child_price)

/-- Theorem stating that under the given conditions, 20 children attended the play --/
theorem children_at_play : children_attending 12 6 80 840 3 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_children_at_play_l56_5680


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l56_5659

theorem angle_sum_is_pi_over_two (α β : Real) : 
  (0 < α ∧ α < π / 2) →  -- α is acute
  (0 < β ∧ β < π / 2) →  -- β is acute
  3 * (Real.sin α) ^ 2 + 2 * (Real.sin β) ^ 2 = 1 →
  3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0 →
  α + 2 * β = π / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l56_5659


namespace NUMINAMATH_CALUDE_exists_divisible_figure_l56_5669

/-- A non-rectangular grid figure composed of cells -/
structure GridFigure where
  cells : ℕ
  nonRectangular : Bool

/-- Represents the ability to divide a figure into equal parts -/
def isDivisible (f : GridFigure) (n : ℕ) : Prop :=
  ∃ (k : ℕ), f.cells = n * k

/-- The main theorem stating the existence of a figure divisible by 2 to 7 -/
theorem exists_divisible_figure :
  ∃ (f : GridFigure), f.nonRectangular ∧
    (∀ n : ℕ, 2 ≤ n ∧ n ≤ 7 → isDivisible f n) :=
  sorry

end NUMINAMATH_CALUDE_exists_divisible_figure_l56_5669


namespace NUMINAMATH_CALUDE_relay_race_selection_methods_l56_5609

/-- The number of students good at sprinting -/
def total_students : ℕ := 6

/-- The number of students to be selected for the relay race -/
def selected_students : ℕ := 4

/-- The number of possible positions for A and B (they must be consecutive with A before B) -/
def positions_for_AB : ℕ := 3

/-- The number of remaining students to be selected -/
def remaining_students : ℕ := total_students - 2

/-- The number of positions to be filled by the remaining students -/
def positions_to_fill : ℕ := selected_students - 2

theorem relay_race_selection_methods :
  (positions_for_AB * (remaining_students.factorial / (remaining_students - positions_to_fill).factorial)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_selection_methods_l56_5609


namespace NUMINAMATH_CALUDE_prob_six_queen_ace_l56_5679

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of cards of each rank (e.g., 6, Queen, Ace) in a standard deck -/
def CardsPerRank : ℕ := 4

/-- Probability of drawing a specific sequence of three cards from a standard deck -/
def prob_specific_sequence (deck_size : ℕ) (cards_per_rank : ℕ) : ℚ :=
  (cards_per_rank : ℚ) / deck_size *
  (cards_per_rank : ℚ) / (deck_size - 1) *
  (cards_per_rank : ℚ) / (deck_size - 2)

theorem prob_six_queen_ace :
  prob_specific_sequence StandardDeck CardsPerRank = 16 / 33150 := by
  sorry

end NUMINAMATH_CALUDE_prob_six_queen_ace_l56_5679


namespace NUMINAMATH_CALUDE_solution_set_f_leq_5_max_m_for_f_geq_quadratic_l56_5681

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem for part (I)
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for part (II)
theorem max_m_for_f_geq_quadratic :
  ∃ (m : ℝ), m = 2 ∧
  (∀ x ∈ Set.Icc 0 2, f x ≥ -x^2 + 2*x + m) ∧
  (∀ m' > m, ∃ x ∈ Set.Icc 0 2, f x < -x^2 + 2*x + m') := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_5_max_m_for_f_geq_quadratic_l56_5681


namespace NUMINAMATH_CALUDE_triangle_abc_area_l56_5637

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is 3 under the given conditions. -/
theorem triangle_abc_area (a b c : ℝ) (A B C : ℝ) : 
  a = Real.sqrt 5 →
  b = 3 →
  Real.sin C = 2 * Real.sin A →
  (1/2) * a * c * Real.sin B = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_area_l56_5637


namespace NUMINAMATH_CALUDE_leaves_broke_after_initial_loss_l56_5667

/-- 
Given that Ryan initially collected 89 leaves, lost 24 leaves, and now has 22 leaves left,
this theorem proves that 43 leaves broke after the initial loss of 24 leaves.
-/
theorem leaves_broke_after_initial_loss 
  (initial_leaves : ℕ) 
  (initial_loss : ℕ) 
  (final_leaves : ℕ) 
  (h1 : initial_leaves = 89)
  (h2 : initial_loss = 24)
  (h3 : final_leaves = 22) :
  initial_leaves - initial_loss - final_leaves = 43 := by
  sorry

end NUMINAMATH_CALUDE_leaves_broke_after_initial_loss_l56_5667


namespace NUMINAMATH_CALUDE_trigonometric_problem_l56_5634

open Real

theorem trigonometric_problem (α β : Real)
  (h1 : α ∈ Set.Ioo 0 (π/2))
  (h2 : β ∈ Set.Ioo (π/2) π)
  (h3 : cos β = -1/3)
  (h4 : sin (α + β) = (4 - Real.sqrt 2) / 6) :
  tan (2 * β) = (4 * Real.sqrt 2) / 7 ∧ α = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l56_5634


namespace NUMINAMATH_CALUDE_bullying_instances_count_l56_5651

-- Define the given constants
def suspension_days_per_instance : ℕ := 3
def typical_person_digits : ℕ := 20
def suspension_multiplier : ℕ := 3

-- Define the total suspension days
def total_suspension_days : ℕ := suspension_multiplier * typical_person_digits

-- Define the number of bullying instances
def bullying_instances : ℕ := total_suspension_days / suspension_days_per_instance

-- Theorem statement
theorem bullying_instances_count : bullying_instances = 20 := by
  sorry

end NUMINAMATH_CALUDE_bullying_instances_count_l56_5651


namespace NUMINAMATH_CALUDE_unique_student_count_l56_5692

theorem unique_student_count : ∃! n : ℕ, 
  100 < n ∧ n < 200 ∧ 
  ∃ k : ℕ, n = 4 * k + 1 ∧
  ∃ m : ℕ, n = 3 * m + 2 ∧
  ∃ l : ℕ, n = 7 * l + 3 ∧
  n = 101 :=
sorry

end NUMINAMATH_CALUDE_unique_student_count_l56_5692


namespace NUMINAMATH_CALUDE_carl_typing_speed_l56_5639

theorem carl_typing_speed :
  ∀ (hours_per_day : ℕ) (total_words : ℕ) (total_days : ℕ),
    hours_per_day = 4 →
    total_words = 84000 →
    total_days = 7 →
    (total_words / total_days) / (hours_per_day * 60) = 50 := by
  sorry

end NUMINAMATH_CALUDE_carl_typing_speed_l56_5639


namespace NUMINAMATH_CALUDE_circle_tangent_perpendicular_l56_5620

-- Define a structure for a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate the angle between three points
def angle (p1 p2 p3 : Point) : ℝ := sorry

-- Define a predicate to check if three points are collinear
def collinear (p1 p2 p3 : Point) : Prop := sorry

theorem circle_tangent_perpendicular (A B C : Point) :
  ¬collinear A B C →
  ∃ (α β γ : ℝ),
    (β + γ + angle B A C = π / 2 ∨ β + γ + angle B A C = -π / 2) ∧
    (γ + α + angle A B C = π / 2 ∨ γ + α + angle A B C = -π / 2) ∧
    (α + β + angle A C B = π / 2 ∨ α + β + angle A C B = -π / 2) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_perpendicular_l56_5620


namespace NUMINAMATH_CALUDE_inequality_system_solution_l56_5610

theorem inequality_system_solution (k : ℝ) : 
  (∀ x : ℝ, (2 * x + 9 > 6 * x + 1 ∧ x - k < 1) ↔ x < 2) →
  k ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l56_5610


namespace NUMINAMATH_CALUDE_juan_distance_l56_5627

/-- Given a speed and time, calculate the distance traveled. -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Juan's distance traveled is 80 miles. -/
theorem juan_distance : distance 10 8 = 80 := by
  sorry

end NUMINAMATH_CALUDE_juan_distance_l56_5627
