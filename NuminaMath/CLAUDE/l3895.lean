import Mathlib

namespace NUMINAMATH_CALUDE_mark_cans_proof_l3895_389533

/-- The number of cans Mark bought -/
def mark_cans : ℕ := 27

/-- The number of cans Jennifer initially bought -/
def jennifer_initial : ℕ := 40

/-- The total number of cans Jennifer brought home -/
def jennifer_total : ℕ := 100

/-- For every 5 cans Mark bought, Jennifer bought 11 cans -/
def jennifer_to_mark_ratio : ℚ := 11 / 5

theorem mark_cans_proof :
  (jennifer_total - jennifer_initial : ℚ) / jennifer_to_mark_ratio = mark_cans := by
  sorry

end NUMINAMATH_CALUDE_mark_cans_proof_l3895_389533


namespace NUMINAMATH_CALUDE_fraction_division_problem_expression_evaluation_problem_l3895_389569

-- Problem 1
theorem fraction_division_problem :
  (3/4 - 7/8) / (-7/8) = 1 + 1/7 := by sorry

-- Problem 2
theorem expression_evaluation_problem :
  2^1 - |0 - 4| + (1/3) * (-3^2) = -5 := by sorry

end NUMINAMATH_CALUDE_fraction_division_problem_expression_evaluation_problem_l3895_389569


namespace NUMINAMATH_CALUDE_decimal_operation_order_l3895_389585

theorem decimal_operation_order : ¬ ∀ (a b c : ℚ), a + b - c = a + (b - c) := by
  sorry

end NUMINAMATH_CALUDE_decimal_operation_order_l3895_389585


namespace NUMINAMATH_CALUDE_children_clothing_production_l3895_389589

-- Define the constants
def total_sets : ℕ := 50
def type_a_fabric : ℝ := 38
def type_b_fabric : ℝ := 26

-- Define the fabric requirements and profits for each size
def size_l_type_a : ℝ := 0.5
def size_l_type_b : ℝ := 1
def size_l_profit : ℝ := 45

def size_m_type_a : ℝ := 0.9
def size_m_type_b : ℝ := 0.2
def size_m_profit : ℝ := 30

-- Define the profit function
def profit_function (x : ℝ) : ℝ := 15 * x + 1500

-- Theorem statement
theorem children_clothing_production (x : ℝ) :
  (17.5 ≤ x ∧ x ≤ 20) →
  (∀ y : ℝ, y = profit_function x) ∧
  (x * size_l_type_a + (total_sets - x) * size_m_type_a ≤ type_a_fabric) ∧
  (x * size_l_type_b + (total_sets - x) * size_m_type_b ≤ type_b_fabric) :=
by sorry

end NUMINAMATH_CALUDE_children_clothing_production_l3895_389589


namespace NUMINAMATH_CALUDE_lineup_count_l3895_389546

/-- The number of ways to choose a starting lineup for a football team -/
def choose_lineup (total_members : ℕ) (offensive_linemen : ℕ) (quarterbacks : ℕ) : ℕ :=
  let remaining := total_members - offensive_linemen - quarterbacks
  offensive_linemen * quarterbacks * remaining * (remaining - 1) * (remaining - 2)

/-- Theorem stating that the number of ways to choose the starting lineup is 5760 -/
theorem lineup_count :
  choose_lineup 12 4 2 = 5760 :=
by sorry

end NUMINAMATH_CALUDE_lineup_count_l3895_389546


namespace NUMINAMATH_CALUDE_page_lines_increase_l3895_389584

theorem page_lines_increase (original : ℕ) (new : ℕ) (increase_percent : ℚ) : 
  new = 240 ∧ 
  increase_percent = 50 ∧ 
  new = original + (increase_percent / 100 : ℚ) * original →
  new - original = 80 := by
  sorry

end NUMINAMATH_CALUDE_page_lines_increase_l3895_389584


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l3895_389523

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_b_value :
  ∀ b : ℝ,
  let A : Point := ⟨3, 1⟩
  let B : Point := ⟨-2, b⟩
  let C : Point := ⟨8, 11⟩
  collinear A B C → b = -9 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l3895_389523


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l3895_389577

theorem ratio_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (hsum : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l3895_389577


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3895_389536

theorem complex_number_quadrant (z : ℂ) (h : z * (1 + Complex.I) = 1 + 2 * Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3895_389536


namespace NUMINAMATH_CALUDE_line_and_circle_properties_l3895_389598

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - 7*y + 8 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := y = -7/2*x + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x-3)^2 + (y-2)^2 = 13

-- Define points A and B
def point_A : ℝ × ℝ := (6, 0)
def point_B : ℝ × ℝ := (1, 5)

-- Theorem statement
theorem line_and_circle_properties :
  -- Line l passes through (3,2)
  line_l 3 2 ∧
  -- Line l is perpendicular to y = -7/2x + 1
  (∀ x y : ℝ, line_l x y → perp_line x y → x = y) ∧
  -- The center of circle C lies on line l
  (∃ x y : ℝ, line_l x y ∧ circle_C x y) ∧
  -- Circle C passes through points A and B
  circle_C point_A.1 point_A.2 ∧ circle_C point_B.1 point_B.2 →
  -- Conclusion 1: The equation of line l is 2x - 7y + 8 = 0
  (∀ x y : ℝ, line_l x y ↔ 2*x - 7*y + 8 = 0) ∧
  -- Conclusion 2: The standard equation of circle C is (x-3)^2 + (y-2)^2 = 13
  (∀ x y : ℝ, circle_C x y ↔ (x-3)^2 + (y-2)^2 = 13) :=
by
  sorry

end NUMINAMATH_CALUDE_line_and_circle_properties_l3895_389598


namespace NUMINAMATH_CALUDE_m_range_l3895_389590

-- Define the quadratic equations
def eq1 (m : ℝ) (x : ℝ) : Prop := x^2 + m*x + 1 = 0
def eq2 (m : ℝ) (x : ℝ) : Prop := 4*x^2 + 4*(m+2)*x + 1 = 0

-- Define the conditions
def has_two_distinct_roots (m : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ eq1 m x ∧ eq1 m y

def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x, ¬(eq2 m x)

-- State the theorem
theorem m_range (m : ℝ) :
  has_two_distinct_roots m ∧ has_no_real_roots m ↔ -3 < m ∧ m < -2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l3895_389590


namespace NUMINAMATH_CALUDE_correct_regression_sequence_l3895_389540

-- Define the steps of linear regression analysis
inductive RegressionStep
  | collectData
  | drawScatterPlot
  | calculateEquation
  | interpretEquation

-- Define a type for sequences of regression steps
def RegressionSequence := List RegressionStep

-- Define the correct sequence
def correctSequence : RegressionSequence :=
  [RegressionStep.collectData, RegressionStep.drawScatterPlot, 
   RegressionStep.calculateEquation, RegressionStep.interpretEquation]

-- Define a property for variables being linearly related
def linearlyRelated (x y : ℝ → ℝ) : Prop := sorry

-- Theorem stating that given linearly related variables, 
-- the correct sequence is as defined above
theorem correct_regression_sequence 
  (x y : ℝ → ℝ) 
  (h : linearlyRelated x y) : 
  correctSequence = 
    [RegressionStep.collectData, RegressionStep.drawScatterPlot, 
     RegressionStep.calculateEquation, RegressionStep.interpretEquation] :=
by
  sorry

end NUMINAMATH_CALUDE_correct_regression_sequence_l3895_389540


namespace NUMINAMATH_CALUDE_problem_1_l3895_389509

theorem problem_1 (x y : ℝ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3895_389509


namespace NUMINAMATH_CALUDE_lacson_sweet_potato_sales_l3895_389554

/-- The problem of Mrs. Lacson's sweet potato sales -/
theorem lacson_sweet_potato_sales 
  (total : ℕ)
  (sold_to_adams : ℕ)
  (unsold : ℕ)
  (h1 : total = 80)
  (h2 : sold_to_adams = 20)
  (h3 : unsold = 45) :
  total - sold_to_adams - unsold = 15 := by
  sorry

end NUMINAMATH_CALUDE_lacson_sweet_potato_sales_l3895_389554


namespace NUMINAMATH_CALUDE_linear_equation_condition_l3895_389558

/-- If mx^(m+2) + m - 2 = 0 is a linear equation with respect to x, then m = -1 -/
theorem linear_equation_condition (m : ℝ) : 
  (∃ a b, ∀ x, m * x^(m + 2) + m - 2 = a * x + b) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l3895_389558


namespace NUMINAMATH_CALUDE_intersection_A_B_zero_union_A_B_equals_A_l3895_389520

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 ≤ x ∧ x < a + 5}

-- Theorem 1: Intersection of A and B when a = 0
theorem intersection_A_B_zero : A ∩ B 0 = {x | -1 < x ∧ x < 5} := by sorry

-- Theorem 2: Range of a for which A ∪ B = A
theorem union_A_B_equals_A (a : ℝ) : A ∪ B a = A ↔ a ∈ Set.Ioo 0 1 ∪ Set.Ici 6 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_zero_union_A_B_equals_A_l3895_389520


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_base_ten_is_perfect_square_ten_is_smallest_base_l3895_389594

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 4 → (2 * b + 5 = n ^ 2 → n ≥ 5) → b ≥ 10 :=
by
  sorry

theorem base_ten_is_perfect_square : 
  ∃ n : ℕ, 2 * 10 + 5 = n ^ 2 :=
by
  sorry

theorem ten_is_smallest_base :
  (∀ b : ℕ, b > 4 ∧ b < 10 → ¬∃ n : ℕ, 2 * b + 5 = n ^ 2) ∧
  (∃ n : ℕ, 2 * 10 + 5 = n ^ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_base_ten_is_perfect_square_ten_is_smallest_base_l3895_389594


namespace NUMINAMATH_CALUDE_relationship_abc_l3895_389568

theorem relationship_abc : 
  let a : ℝ := (1/2)^2
  let b : ℝ := 2^(1/2)
  let c : ℝ := Real.log 2 / Real.log (1/2)
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3895_389568


namespace NUMINAMATH_CALUDE_square_side_length_equals_rectangle_root_area_l3895_389591

theorem square_side_length_equals_rectangle_root_area 
  (rectangle_length : ℝ) 
  (rectangle_breadth : ℝ) 
  (square_side : ℝ) 
  (h1 : rectangle_length = 250) 
  (h2 : rectangle_breadth = 160) 
  (h3 : square_side * square_side = rectangle_length * rectangle_breadth) : 
  square_side = 200 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_equals_rectangle_root_area_l3895_389591


namespace NUMINAMATH_CALUDE_not_right_triangle_l3895_389563

/-- A predicate to check if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a)

/-- Theorem stating that (11, 40, 41) cannot form a right triangle --/
theorem not_right_triangle : ¬ is_right_triangle 11 40 41 := by
  sorry

#check not_right_triangle

end NUMINAMATH_CALUDE_not_right_triangle_l3895_389563


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l3895_389538

theorem complex_sum_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1) 
  (h2 : Complex.abs z₂ = 1) 
  (h3 : Complex.abs (z₁ - z₂) = Real.sqrt 3) : 
  Complex.abs (z₁ + z₂) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l3895_389538


namespace NUMINAMATH_CALUDE_rectangle_count_in_grid_l3895_389528

/-- The number of dots in each row and column of the square array -/
def gridSize : Nat := 5

/-- The number of different rectangles that can be formed in the grid -/
def numRectangles : Nat := (gridSize.choose 2) * (gridSize.choose 2)

theorem rectangle_count_in_grid : numRectangles = 100 := by sorry

end NUMINAMATH_CALUDE_rectangle_count_in_grid_l3895_389528


namespace NUMINAMATH_CALUDE_xy_squared_l3895_389524

theorem xy_squared (x y : ℝ) (h1 : 2 * x * (x + y) = 36) (h2 : 3 * y * (x + y) = 81) :
  (x + y)^2 = 117 / 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_l3895_389524


namespace NUMINAMATH_CALUDE_set_intersection_problem_l3895_389542

theorem set_intersection_problem :
  let A : Set ℤ := {0, 1, 2}
  let B : Set ℤ := {-2, -1, 0, 1}
  A ∩ B = {0, 1} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l3895_389542


namespace NUMINAMATH_CALUDE_dog_pickup_duration_l3895_389503

-- Define the time in minutes for each activity
def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dry_cleaning_time : ℕ := 10
def cooking_time : ℕ := 90

-- Define the total time from work end to dinner
def total_time : ℕ := 180

-- Define the time to pick up the dog (unknown)
def dog_pickup_time : ℕ := total_time - (commute_time + grocery_time + dry_cleaning_time + cooking_time)

-- Theorem to prove
theorem dog_pickup_duration : dog_pickup_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_dog_pickup_duration_l3895_389503


namespace NUMINAMATH_CALUDE_profit_percentage_l3895_389531

theorem profit_percentage (cost_price selling_price : ℝ) 
  (h : 58 * cost_price = 50 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 16 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l3895_389531


namespace NUMINAMATH_CALUDE_confidence_level_error_probability_l3895_389559

/-- Represents the confidence level as a real number between 0 and 1 -/
def ConfidenceLevel : Type := { r : ℝ // 0 < r ∧ r < 1 }

/-- Represents the probability of making an incorrect inference -/
def ErrorProbability : Type := { r : ℝ // 0 ≤ r ∧ r ≤ 1 }

/-- Given a confidence level, calculates the probability of making an incorrect inference -/
def calculateErrorProbability (cl : ConfidenceLevel) : ErrorProbability :=
  sorry

theorem confidence_level_error_probability 
  (cl : ConfidenceLevel) 
  (hp : cl.val = 0.95) :
  (calculateErrorProbability cl).val = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_confidence_level_error_probability_l3895_389559


namespace NUMINAMATH_CALUDE_carrots_picked_first_day_l3895_389550

theorem carrots_picked_first_day (carrots_thrown_out carrots_second_day total_carrots : ℕ) 
  (h1 : carrots_thrown_out = 4)
  (h2 : carrots_second_day = 46)
  (h3 : total_carrots = 61) :
  ∃ carrots_first_day : ℕ, 
    carrots_first_day + carrots_second_day - carrots_thrown_out = total_carrots ∧ 
    carrots_first_day = 19 := by
  sorry

end NUMINAMATH_CALUDE_carrots_picked_first_day_l3895_389550


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_numbers_l3895_389593

theorem infinitely_many_divisible_numbers :
  ∃ (a : ℕ → ℕ), (∀ n : ℕ, a n ∣ 2^(a n) + 3^(a n)) ∧
                 (∀ n : ℕ, a n < a (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_numbers_l3895_389593


namespace NUMINAMATH_CALUDE_expand_and_simplify_product_l3895_389575

theorem expand_and_simplify_product (x : ℝ) :
  (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_product_l3895_389575


namespace NUMINAMATH_CALUDE_partnership_investment_timing_l3895_389566

/-- A partnership problem where three partners invest at different times --/
theorem partnership_investment_timing 
  (x : ℝ) -- A's investment
  (annual_gain : ℝ) -- Total annual gain
  (a_share : ℝ) -- A's share of the gain
  (h1 : annual_gain = 12000) -- Given annual gain
  (h2 : a_share = 4000) -- Given A's share
  (h3 : a_share / annual_gain = 1/3) -- A's share ratio
  : ∃ (m : ℝ), -- The number of months after which C invests
    (x * 12) / (x * 12 + 2*x * 6 + 3*x * (12 - m)) = 1/3 ∧ 
    m = 8 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_timing_l3895_389566


namespace NUMINAMATH_CALUDE_expression_evaluation_l3895_389527

theorem expression_evaluation (a b c : ℚ) : 
  a = 5 → 
  b = a + 4 → 
  c = b - 12 → 
  a + 2 ≠ 0 → 
  b - 3 ≠ 0 → 
  c + 7 ≠ 0 → 
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 7 / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3895_389527


namespace NUMINAMATH_CALUDE_share_of_a_l3895_389543

def total : ℕ := 366

def shares (a b c : ℕ) : Prop :=
  a + b + c = total ∧
  a = (b + c) / 2 ∧
  b = (a + c) * 2 / 3

theorem share_of_a : ∃ a b c : ℕ, shares a b c ∧ a = 122 := by sorry

end NUMINAMATH_CALUDE_share_of_a_l3895_389543


namespace NUMINAMATH_CALUDE_line_parameterization_l3895_389514

/-- Given a line y = 2x - 40 parameterized by (x, y) = (g(t), 20t - 14),
    prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t x y : ℝ, y = 2*x - 40 ∧ x = g t ∧ y = 20*t - 14) →
  (∀ t : ℝ, g t = 10*t + 13) :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_l3895_389514


namespace NUMINAMATH_CALUDE_rabbit_log_sawing_l3895_389599

theorem rabbit_log_sawing (cuts pieces : ℕ) (h1 : cuts = 10) (h2 : pieces = 16) :
  pieces - cuts = 6 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_log_sawing_l3895_389599


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3895_389573

/-- Given an arithmetic sequence {aₙ} with a₃ = 0 and a₁ = 4, 
    the common difference d is -2. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h1 : a 3 = 0) 
  (h2 : a 1 = 4) 
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) :
  a 2 - a 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3895_389573


namespace NUMINAMATH_CALUDE_probability_two_boys_three_girls_l3895_389537

def probability_boy_or_girl : ℝ := 0.5

def number_of_children : ℕ := 5

def number_of_boys : ℕ := 2

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem probability_two_boys_three_girls :
  (binomial_coefficient number_of_children number_of_boys : ℝ) *
  probability_boy_or_girl ^ number_of_boys *
  probability_boy_or_girl ^ (number_of_children - number_of_boys) =
  0.3125 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_boys_three_girls_l3895_389537


namespace NUMINAMATH_CALUDE_problem_solution_l3895_389571

theorem problem_solution (a b : ℝ) : 
  (∀ x : ℝ, (x + a) * (x + 8) = x^2 + b*x + 24) → 
  a + b = 14 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3895_389571


namespace NUMINAMATH_CALUDE_sin_2phi_value_l3895_389530

theorem sin_2phi_value (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := by
sorry

end NUMINAMATH_CALUDE_sin_2phi_value_l3895_389530


namespace NUMINAMATH_CALUDE_min_value_expression_l3895_389560

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y)
  (h_sum : x + y + 1/x + 1/y = 2022) :
  (x + 1/y) * (x + 1/y - 2016) + (y + 1/x) * (y + 1/x - 2016) ≥ -2032188 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3895_389560


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l3895_389506

/-- The volume difference between a sphere and an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (r_sphere : ℝ) (r_cylinder : ℝ) 
  (h_sphere : r_sphere = 7)
  (h_cylinder : r_cylinder = 4) :
  (4 / 3 * π * r_sphere^3) - (π * r_cylinder^2 * Real.sqrt (4 * r_sphere^2 - 4 * r_cylinder^2)) = 
  1372 * π / 3 - 32 * π * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l3895_389506


namespace NUMINAMATH_CALUDE_point_placement_l3895_389504

theorem point_placement (x : ℕ) : 
  (9 * x - 8 = 82) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_point_placement_l3895_389504


namespace NUMINAMATH_CALUDE_sandwiches_problem_l3895_389507

def sandwiches_left (initial : ℕ) (ruth_ate : ℕ) (brother_given : ℕ) (first_cousin_ate : ℕ) (other_cousins_ate : ℕ) : ℕ :=
  initial - ruth_ate - brother_given - first_cousin_ate - other_cousins_ate

theorem sandwiches_problem :
  sandwiches_left 10 1 2 2 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_problem_l3895_389507


namespace NUMINAMATH_CALUDE_correct_horses_b_l3895_389532

/-- Represents the number of horses put in the pasture by party b -/
def horses_b : ℕ := 6

/-- Represents the total cost of the pasture -/
def total_cost : ℕ := 870

/-- Represents the amount b should pay -/
def b_payment : ℕ := 360

/-- Represents the number of horses put in by party a -/
def horses_a : ℕ := 12

/-- Represents the number of months horses from party a stayed -/
def months_a : ℕ := 8

/-- Represents the number of months horses from party b stayed -/
def months_b : ℕ := 9

/-- Represents the number of horses put in by party c -/
def horses_c : ℕ := 18

/-- Represents the number of months horses from party c stayed -/
def months_c : ℕ := 6

theorem correct_horses_b :
  (horses_b * months_b : ℚ) / (horses_a * months_a + horses_b * months_b + horses_c * months_c) * total_cost = b_payment :=
by sorry

end NUMINAMATH_CALUDE_correct_horses_b_l3895_389532


namespace NUMINAMATH_CALUDE_bertha_descendants_without_daughters_l3895_389519

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The given conditions of Bertha's family -/
def bertha_family : BerthaFamily :=
  { daughters := 8,
    granddaughters := 20,
    total_descendants := 28,
    daughters_with_children := 5 }

/-- Theorem stating the number of Bertha's descendants without daughters -/
theorem bertha_descendants_without_daughters :
  bertha_family.total_descendants - bertha_family.daughters_with_children = 23 := by
  sorry

#check bertha_descendants_without_daughters

end NUMINAMATH_CALUDE_bertha_descendants_without_daughters_l3895_389519


namespace NUMINAMATH_CALUDE_lending_period_is_one_year_l3895_389581

/-- 
Given a person who:
- Borrows an amount at a certain interest rate
- Lends the same amount at a higher interest rate
- Makes a fixed gain per year

This theorem proves that the lending period is 1 year under specific conditions.
-/
theorem lending_period_is_one_year 
  (borrowed_amount : ℝ) 
  (borrowing_rate : ℝ) 
  (lending_rate : ℝ) 
  (gain_per_year : ℝ) 
  (h1 : borrowed_amount = 5000)
  (h2 : borrowing_rate = 0.04)
  (h3 : lending_rate = 0.06)
  (h4 : gain_per_year = 100)
  : ∃ t : ℝ, t = 1 ∧ borrowed_amount * lending_rate * t - borrowed_amount * borrowing_rate * t = gain_per_year :=
sorry

end NUMINAMATH_CALUDE_lending_period_is_one_year_l3895_389581


namespace NUMINAMATH_CALUDE_opposite_points_probability_l3895_389545

theorem opposite_points_probability (n : ℕ) (h : n = 12) : 
  (n / 2) / (n.choose 2) = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_opposite_points_probability_l3895_389545


namespace NUMINAMATH_CALUDE_four_valid_a_values_l3895_389583

theorem four_valid_a_values : 
  let equation_solution (a : ℝ) := (a - 2 : ℝ)
  let inequality_system (a y : ℝ) := y + 9 ≤ 2 * (y + 2) ∧ (2 * y - a) / 3 ≥ 1
  let valid_a (a : ℤ) := 
    equation_solution a > 0 ∧ 
    equation_solution a ≠ 3 ∧ 
    (∀ y : ℝ, inequality_system a y ↔ y ≥ 5)
  ∃! (s : Finset ℤ), (∀ a ∈ s, valid_a a) ∧ s.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_four_valid_a_values_l3895_389583


namespace NUMINAMATH_CALUDE_triangle_formation_l3895_389582

theorem triangle_formation (a b c : ℝ) : 
  a = 4 ∧ b = 9 ∧ c = 9 → 
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l3895_389582


namespace NUMINAMATH_CALUDE_carbon_emissions_solution_l3895_389544

theorem carbon_emissions_solution :
  ∃! (x y : ℝ), x + y = 70 ∧ x = 5 * y - 8 ∧ x = 57 ∧ y = 13 := by
  sorry

end NUMINAMATH_CALUDE_carbon_emissions_solution_l3895_389544


namespace NUMINAMATH_CALUDE_initial_books_on_shelf_l3895_389567

theorem initial_books_on_shelf (books_taken : ℕ) (books_left : ℕ) : 
  books_taken = 10 → books_left = 28 → books_taken + books_left = 38 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_on_shelf_l3895_389567


namespace NUMINAMATH_CALUDE_faster_increase_l3895_389535

-- Define the functions
def y₁ (x : ℝ) : ℝ := 100 * x
def y₂ (x : ℝ) : ℝ := 1000 + 100 * x
def y₃ (x : ℝ) : ℝ := 10000 + 99 * x

-- State the theorem
theorem faster_increase : 
  (∀ x : ℝ, (deriv y₁ x) = (deriv y₂ x)) ∧ 
  (∀ x : ℝ, (deriv y₁ x) > (deriv y₃ x)) := by
  sorry

end NUMINAMATH_CALUDE_faster_increase_l3895_389535


namespace NUMINAMATH_CALUDE_marsh_birds_count_l3895_389502

theorem marsh_birds_count (geese ducks : ℕ) (h1 : geese = 58) (h2 : ducks = 37) :
  geese + ducks = 95 := by
  sorry

end NUMINAMATH_CALUDE_marsh_birds_count_l3895_389502


namespace NUMINAMATH_CALUDE_square_side_length_average_l3895_389572

theorem square_side_length_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l3895_389572


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l3895_389518

/-- Given two parallel lines, one passing through P(3,2m) and Q(m,2),
    and another passing through M(2,-1) and N(-3,4), prove that m = -1 -/
theorem parallel_lines_m_value :
  ∀ m : ℝ,
  let P : ℝ × ℝ := (3, 2*m)
  let Q : ℝ × ℝ := (m, 2)
  let M : ℝ × ℝ := (2, -1)
  let N : ℝ × ℝ := (-3, 4)
  let slope_PQ := (Q.2 - P.2) / (Q.1 - P.1)
  let slope_MN := (N.2 - M.2) / (N.1 - M.1)
  slope_PQ = slope_MN →
  m = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l3895_389518


namespace NUMINAMATH_CALUDE_reverse_two_digit_number_l3895_389552

/-- For a two-digit number with tens digit x and units digit y,
    the number formed by reversing its digits is 10y + x. -/
theorem reverse_two_digit_number (x y : ℕ) 
  (h1 : x ≥ 1 ∧ x ≤ 9) (h2 : y ≥ 0 ∧ y ≤ 9) : 
  (10 * y + x) = (10 * y + x) := by
  sorry

#check reverse_two_digit_number

end NUMINAMATH_CALUDE_reverse_two_digit_number_l3895_389552


namespace NUMINAMATH_CALUDE_complex_product_real_imag_parts_l3895_389548

theorem complex_product_real_imag_parts 
  (c d : ℂ) (x : ℝ) 
  (h1 : Complex.abs c = 3) 
  (h2 : Complex.abs d = 5) 
  (h3 : c * d = x + 6 * Complex.I) :
  x = 3 * Real.sqrt 21 :=
sorry

end NUMINAMATH_CALUDE_complex_product_real_imag_parts_l3895_389548


namespace NUMINAMATH_CALUDE_correct_distribution_probability_l3895_389586

/-- Represents the number of guests -/
def num_guests : ℕ := 4

/-- Represents the total number of rolls -/
def total_rolls : ℕ := 8

/-- Represents the number of cheese rolls -/
def cheese_rolls : ℕ := 4

/-- Represents the number of fruit rolls -/
def fruit_rolls : ℕ := 4

/-- Represents the number of rolls per guest -/
def rolls_per_guest : ℕ := 2

/-- The probability of each guest getting one cheese roll and one fruit roll -/
def probability_correct_distribution : ℚ := 1 / 35

theorem correct_distribution_probability :
  probability_correct_distribution = 
    (cheese_rolls.choose 1 * fruit_rolls.choose 1 / (total_rolls.choose 2)) *
    ((cheese_rolls - 1).choose 1 * (fruit_rolls - 1).choose 1 / ((total_rolls - 2).choose 2)) *
    ((cheese_rolls - 2).choose 1 * (fruit_rolls - 2).choose 1 / ((total_rolls - 4).choose 2)) *
    1 := by sorry

#check correct_distribution_probability

end NUMINAMATH_CALUDE_correct_distribution_probability_l3895_389586


namespace NUMINAMATH_CALUDE_sqrt_problem_1_sqrt_problem_2_sqrt_problem_3_sqrt_problem_4_l3895_389500

-- 1. Prove that √18 - √32 + √2 = 0
theorem sqrt_problem_1 : Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0 := by sorry

-- 2. Prove that (√27 - √12) / √3 = 1
theorem sqrt_problem_2 : (Real.sqrt 27 - Real.sqrt 12) / Real.sqrt 3 = 1 := by sorry

-- 3. Prove that √(1/6) + √24 - √600 = -43/6 * √6
theorem sqrt_problem_3 : Real.sqrt (1/6) + Real.sqrt 24 - Real.sqrt 600 = -43/6 * Real.sqrt 6 := by sorry

-- 4. Prove that (√3 + 1)(√3 - 1) = 2
theorem sqrt_problem_4 : (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_problem_1_sqrt_problem_2_sqrt_problem_3_sqrt_problem_4_l3895_389500


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3895_389534

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 = 4 → a 2 = 6 →
  (a 1 + a 2 + a 3 + a 4 = 28) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3895_389534


namespace NUMINAMATH_CALUDE_functions_properties_l3895_389512

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (ω * x + φ)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 2 * Real.cos (ω * x)

theorem functions_properties (ω φ : ℝ) :
  ω > 0 ∧
  0 ≤ φ ∧ φ < π ∧
  (∀ x : ℝ, f ω φ (x + π / ω) = f ω φ x) ∧
  (∀ x : ℝ, g ω (x + π / ω) = g ω x) ∧
  f ω φ (-π/6) + g ω (-π/6) = 0 →
  ω = 2 ∧
  φ = π/6 ∧
  ∀ x : ℝ, f ω φ x + g ω x = Real.sqrt 6 * Real.sin (2 * x + π/3) := by
sorry

end NUMINAMATH_CALUDE_functions_properties_l3895_389512


namespace NUMINAMATH_CALUDE_exam_score_problem_l3895_389511

theorem exam_score_problem (total_questions : ℕ) 
  (correct_score wrong_score total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 120 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_answers * correct_score + (total_questions - correct_answers) * wrong_score = total_score ∧
    correct_answers = 36 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l3895_389511


namespace NUMINAMATH_CALUDE_charlie_missing_coins_l3895_389517

/-- Represents the fraction of coins Charlie has at different stages -/
structure CoinFraction where
  total : ℚ
  dropped : ℚ
  found : ℚ

/-- Calculates the fraction of coins still missing -/
def missing_fraction (cf : CoinFraction) : ℚ :=
  cf.total - (cf.total - cf.dropped + cf.found * cf.dropped)

/-- Theorem stating that the fraction of missing coins is 1/9 -/
theorem charlie_missing_coins :
  let cf : CoinFraction := { total := 1, dropped := 1/3, found := 2/3 }
  missing_fraction cf = 1/9 := by
  sorry

#check charlie_missing_coins

end NUMINAMATH_CALUDE_charlie_missing_coins_l3895_389517


namespace NUMINAMATH_CALUDE_writer_birthday_theorem_l3895_389501

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the number of leap years in a given range -/
def leapYearsInRange (startYear endYear : Nat) : Nat :=
  sorry

/-- Calculates the day of the week given a number of days before Friday -/
def dayBeforeFriday (days : Nat) : DayOfWeek :=
  sorry

theorem writer_birthday_theorem :
  let startYear := 1780
  let endYear := 2020
  let yearsDiff := endYear - startYear
  let leapYears := leapYearsInRange startYear endYear
  let regularYears := yearsDiff - leapYears
  let totalDaysBackward := regularYears + 2 * leapYears
  dayBeforeFriday (totalDaysBackward % 7) = DayOfWeek.Sunday :=
by sorry

end NUMINAMATH_CALUDE_writer_birthday_theorem_l3895_389501


namespace NUMINAMATH_CALUDE_biased_coin_probability_l3895_389539

def probability_of_heads (p : ℝ) (k : ℕ) (n : ℕ) : ℝ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem biased_coin_probability : 
  ∀ p : ℝ, 
  0 < p → p < 1 →
  probability_of_heads p 1 7 = probability_of_heads p 2 7 →
  probability_of_heads p 1 7 ≠ 0 →
  probability_of_heads p 4 7 = 945 / 16384 := by
sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l3895_389539


namespace NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l3895_389549

theorem cos_pi_half_plus_alpha (α : Real) : 
  (∃ P : Real × Real, P.1 = -4/5 ∧ P.2 = 3/5 ∧ P.1^2 + P.2^2 = 1 ∧ 
   P.1 = Real.cos α ∧ P.2 = Real.sin α) → 
  Real.cos (π/2 + α) = -3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l3895_389549


namespace NUMINAMATH_CALUDE_cupboard_cost_price_l3895_389529

theorem cupboard_cost_price (selling_price selling_price_increased : ℝ) 
  (h1 : selling_price = 0.84 * 3750)
  (h2 : selling_price_increased = 1.16 * 3750)
  (h3 : selling_price_increased = selling_price + 1200) : 
  ∃ (cost_price : ℝ), cost_price = 3750 := by
sorry

end NUMINAMATH_CALUDE_cupboard_cost_price_l3895_389529


namespace NUMINAMATH_CALUDE_orange_distribution_l3895_389547

-- Define the total number of oranges
def total_oranges : ℕ := 30

-- Define the number of people
def num_people : ℕ := 3

-- Define the minimum number of oranges each person must receive
def min_oranges : ℕ := 3

-- Define the function to calculate the number of ways to distribute oranges
def ways_to_distribute (total : ℕ) (people : ℕ) (min : ℕ) : ℕ :=
  Nat.choose (total - people * min + people - 1) (people - 1)

-- Theorem statement
theorem orange_distribution :
  ways_to_distribute total_oranges num_people min_oranges = 253 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l3895_389547


namespace NUMINAMATH_CALUDE_trajectory_classification_l3895_389508

/-- The trajectory of a point P(x,y) satisfying |PF₁| + |PF₂| = 2a, where F₁(-5,0) and F₂(5,0) are fixed points -/
def trajectory (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p (-5, 0) + dist p (5, 0) = 2 * a}

/-- The distance between F₁ and F₂ -/
def f₁f₂_distance : ℝ := 10

theorem trajectory_classification (a : ℝ) (h : a > 0) :
  (a = f₁f₂_distance → trajectory a = {p : ℝ × ℝ | p.1 ∈ Set.Icc (-5 : ℝ) 5 ∧ p.2 = 0}) ∧
  (a > f₁f₂_distance → ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ trajectory a = {p : ℝ × ℝ | (p.1 / c)^2 + (p.2 / d)^2 = 1}) ∧
  (a < f₁f₂_distance → trajectory a = ∅) :=
sorry

end NUMINAMATH_CALUDE_trajectory_classification_l3895_389508


namespace NUMINAMATH_CALUDE_machine_ok_l3895_389505

/-- The nominal portion weight -/
def nominal_weight : ℝ := 390

/-- The greatest deviation from the mean among preserved measurements -/
def max_deviation : ℝ := 39

/-- Condition: The greatest deviation doesn't exceed 10% of the nominal weight -/
axiom deviation_within_limit : max_deviation ≤ 0.1 * nominal_weight

/-- Condition: Deviations of unreadable measurements are less than the max deviation -/
axiom unreadable_deviations_less : ∀ x : ℝ, x < max_deviation → x < nominal_weight - 380

/-- Definition: A machine requires repair if the standard deviation exceeds max_deviation -/
def requires_repair (std_dev : ℝ) : Prop := std_dev > max_deviation

/-- Theorem: The machine does not require repair -/
theorem machine_ok : ∃ std_dev : ℝ, std_dev ≤ max_deviation ∧ ¬(requires_repair std_dev) := by
  sorry


end NUMINAMATH_CALUDE_machine_ok_l3895_389505


namespace NUMINAMATH_CALUDE_factorial_bounds_l3895_389592

theorem factorial_bounds (n : ℕ) (h : n ≥ 1) : 2^(n-1) ≤ n! ∧ n! ≤ n^n := by
  sorry

end NUMINAMATH_CALUDE_factorial_bounds_l3895_389592


namespace NUMINAMATH_CALUDE_project_completion_proof_l3895_389515

/-- The number of days Person A takes to complete the project alone -/
def person_a_days : ℕ := 20

/-- The number of days Person B takes to complete the project alone -/
def person_b_days : ℕ := 10

/-- The total number of days taken to complete the project -/
def total_days : ℕ := 12

/-- The number of days Person B worked alone -/
def person_b_worked_days : ℕ := 8

theorem project_completion_proof :
  (1 : ℚ) = (total_days - person_b_worked_days : ℚ) / person_a_days + 
            (person_b_worked_days : ℚ) / person_b_days :=
by sorry

end NUMINAMATH_CALUDE_project_completion_proof_l3895_389515


namespace NUMINAMATH_CALUDE_range_of_m_l3895_389588

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (m : ℝ) : 
  (p m ∨ q m) → m < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3895_389588


namespace NUMINAMATH_CALUDE_max_b_value_l3895_389579

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = a * 1000000 + 2 * 100000 + b * 10000 + 3 * 1000 + 4 * 100 + c

def is_divisible_by_55 (n : ℕ) : Prop :=
  n % 55 = 0

theorem max_b_value (n : ℕ) 
  (h1 : is_valid_number n) 
  (h2 : is_divisible_by_55 n) : 
  ∃ (a c : ℕ), ∃ (b : ℕ), b ≤ 7 ∧ 
    n = a * 1000000 + 2 * 100000 + b * 10000 + 3 * 1000 + 4 * 100 + c :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l3895_389579


namespace NUMINAMATH_CALUDE_inequality_proof_l3895_389522

theorem inequality_proof (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hy₁ : x₁ * y₁ > z₁^2) (hy₂ : x₂ * y₂ > z₂^2) : 
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3895_389522


namespace NUMINAMATH_CALUDE_sum_20_terms_l3895_389564

/-- An arithmetic progression with the sum of its 4th and 12th terms equal to 20 -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_4_12 : a + 3*d + a + 11*d = 20  -- Sum of 4th and 12th terms is 20

/-- Theorem about the sum of first 20 terms of the arithmetic progression -/
theorem sum_20_terms (ap : ArithmeticProgression) :
  ∃ k : ℝ, k = 200 + 120 * ap.d ∧ 
  (∀ n : ℕ, n ≤ 20 → (n : ℝ) / 2 * (2 * ap.a + (n - 1) * ap.d) ≤ k) ∧
  (∀ ε > 0, ∃ n : ℕ, n ≤ 20 ∧ k - (n : ℝ) / 2 * (2 * ap.a + (n - 1) * ap.d) < ε) :=
by sorry


end NUMINAMATH_CALUDE_sum_20_terms_l3895_389564


namespace NUMINAMATH_CALUDE_janes_age_l3895_389556

/-- Jane's babysitting problem -/
theorem janes_age (start_age : ℕ) (years_since_stop : ℕ) (oldest_babysat_now : ℕ)
  (h1 : start_age = 18)
  (h2 : years_since_stop = 12)
  (h3 : oldest_babysat_now = 23) :
  ∃ (current_age : ℕ),
    current_age = 34 ∧
    current_age ≥ start_age + years_since_stop ∧
    2 * (oldest_babysat_now - years_since_stop) ≤ current_age - years_since_stop :=
by sorry

end NUMINAMATH_CALUDE_janes_age_l3895_389556


namespace NUMINAMATH_CALUDE_shoe_discount_percentage_l3895_389597

def shoe_price : ℝ := 200
def shirts_price : ℝ := 160
def final_discount : ℝ := 0.05
def final_amount : ℝ := 285

theorem shoe_discount_percentage :
  ∃ (x : ℝ), 
    (shoe_price * (1 - x / 100) + shirts_price) * (1 - final_discount) = final_amount ∧
    x = 30 :=
  sorry

end NUMINAMATH_CALUDE_shoe_discount_percentage_l3895_389597


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3895_389587

theorem contrapositive_equivalence :
  (∀ (a b : ℝ), ab > 0 → a > 0) ↔ (∀ (a b : ℝ), a ≤ 0 → ab ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3895_389587


namespace NUMINAMATH_CALUDE_division_problem_l3895_389551

theorem division_problem (dividend : Nat) (divisor : Nat) (remainder : Nat) (quotient : Nat) :
  dividend = divisor * quotient + remainder →
  dividend = 34 →
  divisor = 7 →
  remainder = 6 →
  quotient = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3895_389551


namespace NUMINAMATH_CALUDE_qin_jiushao_v3_l3895_389562

def qin_jiushao_algorithm (coeffs : List ℤ) (x : ℤ) : List ℤ :=
  let f := λ acc coeff => acc * x + coeff
  List.scanl f 0 coeffs.reverse

def polynomial : List ℤ := [64, -192, 240, -160, 60, -12, 1]

theorem qin_jiushao_v3 :
  (qin_jiushao_algorithm polynomial 2).get! 3 = -80 := by sorry

end NUMINAMATH_CALUDE_qin_jiushao_v3_l3895_389562


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l3895_389553

def polar_equation (ρ θ : ℝ) : Prop :=
  ρ = Real.sqrt 2 * (Real.cos θ + Real.sin θ)

theorem circle_center_coordinates :
  ∃ (r θ : ℝ), polar_equation r θ ∧ r = 1 ∧ θ = π / 4 :=
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l3895_389553


namespace NUMINAMATH_CALUDE_orange_fraction_l3895_389578

theorem orange_fraction (total_fruit : ℕ) (oranges peaches apples : ℕ) :
  total_fruit = 56 →
  peaches = oranges / 2 →
  apples = 5 * peaches →
  apples = 35 →
  oranges = total_fruit / 4 := by
  sorry

end NUMINAMATH_CALUDE_orange_fraction_l3895_389578


namespace NUMINAMATH_CALUDE_class_duration_theorem_l3895_389555

/-- Calculates the total duration of classes given the number of periods, period length, number of breaks, and break length. -/
def classDuration (numPeriods : ℕ) (periodLength : ℕ) (numBreaks : ℕ) (breakLength : ℕ) : ℕ :=
  numPeriods * periodLength + numBreaks * breakLength

/-- Proves that the total duration of classes with 5 periods of 40 minutes each and 4 breaks of 5 minutes each is 220 minutes. -/
theorem class_duration_theorem :
  classDuration 5 40 4 5 = 220 := by
  sorry

#eval classDuration 5 40 4 5

end NUMINAMATH_CALUDE_class_duration_theorem_l3895_389555


namespace NUMINAMATH_CALUDE_female_officers_count_l3895_389576

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) 
  (female_ratio : ℚ) (female_on_duty_percent : ℚ) :
  total_on_duty = 160 →
  female_ratio = 1/2 →
  female_on_duty_percent = 16/100 →
  (female_on_duty_ratio * ↑total_on_duty : ℚ) = (female_ratio * ↑total_on_duty : ℚ) →
  (female_on_duty_percent * (female_on_duty_ratio * ↑total_on_duty / female_on_duty_percent : ℚ) : ℚ) = 500 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l3895_389576


namespace NUMINAMATH_CALUDE_power_equation_l3895_389557

/-- Given a real number a and integers m and n such that a^m = 2 and a^n = 5,
    prove that a^(3m+2n) = 200 -/
theorem power_equation (a : ℝ) (m n : ℤ) (h1 : a^m = 2) (h2 : a^n = 5) :
  a^(3*m + 2*n) = 200 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l3895_389557


namespace NUMINAMATH_CALUDE_stating_exists_k_no_carries_l3895_389596

/-- 
Given two positive integers a and b, returns true if adding a to b
results in no carries during the whole calculation in base 10.
-/
def no_carries (a b : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → (a / 10^d % 10 + b / 10^d % 10 < 10)

/-- 
Theorem stating that there exists a positive integer k such that
adding 1996k to 1997k results in no carries during the whole calculation.
-/
theorem exists_k_no_carries : ∃ k : ℕ, k > 0 ∧ no_carries (1996 * k) (1997 * k) := by
  sorry

end NUMINAMATH_CALUDE_stating_exists_k_no_carries_l3895_389596


namespace NUMINAMATH_CALUDE_math_class_students_count_l3895_389580

theorem math_class_students_count :
  ∃! n : ℕ, n < 50 ∧ n % 8 = 5 ∧ n % 6 = 3 ∧ n = 45 :=
by sorry

end NUMINAMATH_CALUDE_math_class_students_count_l3895_389580


namespace NUMINAMATH_CALUDE_factorization_proof_l3895_389510

theorem factorization_proof (x y m n : ℝ) : 
  x^2 * (m - n) + y^2 * (n - m) = (m - n) * (x + y) * (x - y) := by
sorry

end NUMINAMATH_CALUDE_factorization_proof_l3895_389510


namespace NUMINAMATH_CALUDE_max_value_expression_l3895_389525

theorem max_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 12) :
  a * b + b * c + a * c + a * b * c ≤ 112 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 12 ∧ 
    a' * b' + b' * c' + a' * c' + a' * b' * c' = 112 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3895_389525


namespace NUMINAMATH_CALUDE_prob_sum_gt_five_l3895_389526

/-- The probability of rolling two dice and getting a sum greater than five -/
def prob_sum_greater_than_five : ℚ := 2/3

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum is less than or equal to five -/
def outcomes_sum_le_five : ℕ := 12

theorem prob_sum_gt_five :
  prob_sum_greater_than_five = 1 - (outcomes_sum_le_five : ℚ) / total_outcomes :=
sorry

end NUMINAMATH_CALUDE_prob_sum_gt_five_l3895_389526


namespace NUMINAMATH_CALUDE_power_of_difference_squared_l3895_389521

theorem power_of_difference_squared : (3^2 - 3)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_power_of_difference_squared_l3895_389521


namespace NUMINAMATH_CALUDE_intersection_when_m_is_two_subset_condition_l3895_389513

-- Define set A
def A : Set ℝ := {y | ∃ x, -13/2 ≤ x ∧ x ≤ 3/2 ∧ y = Real.sqrt (3 - 2*x)}

-- Define set B
def B (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ m + 1}

-- Theorem 1: When m = 2, A ∩ B = [0, 3]
theorem intersection_when_m_is_two : 
  A ∩ B 2 = Set.Icc 0 3 := by sorry

-- Theorem 2: B ⊆ A if and only if m ≤ 1
theorem subset_condition : 
  ∀ m : ℝ, B m ⊆ A ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_two_subset_condition_l3895_389513


namespace NUMINAMATH_CALUDE_stretch_cosine_curve_l3895_389516

/-- Given a curve y = cos x and a stretch transformation x' = 2x and y' = 3y,
    prove that the new equation of the curve is y' = 3 cos (x' / 2) -/
theorem stretch_cosine_curve (x x' y y' : ℝ) :
  y = Real.cos x →
  x' = 2 * x →
  y' = 3 * y →
  y' = 3 * Real.cos (x' / 2) := by
  sorry

end NUMINAMATH_CALUDE_stretch_cosine_curve_l3895_389516


namespace NUMINAMATH_CALUDE_divisors_of_6440_l3895_389561

theorem divisors_of_6440 : 
  let n : ℕ := 6440
  let prime_factorization : List (ℕ × ℕ) := [(2, 3), (5, 1), (7, 1), (23, 1)]
  ∀ (is_valid_factorization : n = (List.foldl (λ acc (p, e) => acc * p^e) 1 prime_factorization)),
  (List.foldl (λ acc (_, e) => acc * (e + 1)) 1 prime_factorization) = 32 := by
sorry

end NUMINAMATH_CALUDE_divisors_of_6440_l3895_389561


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3895_389595

/-- Given a rectangle where the sum of its length and width is 24 centimeters,
    prove that its perimeter is 48 centimeters. -/
theorem rectangle_perimeter (length width : ℝ) (h : length + width = 24) :
  2 * (length + width) = 48 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3895_389595


namespace NUMINAMATH_CALUDE_max_money_earned_is_zero_l3895_389570

/-- Represents the state of the three piles of stones -/
structure PileState where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ

/-- Represents a single move of a stone from one pile to another -/
inductive Move
  | oneToTwo
  | oneToThree
  | twoToOne
  | twoToThree
  | threeToOne
  | threeToTwo

/-- Applies a move to a pile state, returning the new state and the money earned -/
def applyMove (state : PileState) (move : Move) : PileState × ℤ := sorry

/-- A sequence of moves -/
def MoveSequence := List Move

/-- Applies a sequence of moves to an initial state, returning the final state and total money earned -/
def applyMoves (initial : PileState) (moves : MoveSequence) : PileState × ℤ := sorry

/-- The main theorem: the maximum money that can be earned is 0 -/
theorem max_money_earned_is_zero (initial : PileState) :
  (∀ moves : MoveSequence, applyMoves initial moves = (initial, 0)) → 
  (∀ moves : MoveSequence, (applyMoves initial moves).2 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_max_money_earned_is_zero_l3895_389570


namespace NUMINAMATH_CALUDE_investment_proof_l3895_389574

/-- Represents the monthly interest rate as a decimal -/
def monthly_interest_rate : ℝ := 0.10

/-- Calculates the total amount after n months given an initial investment -/
def total_after_n_months (initial_investment : ℝ) (n : ℕ) : ℝ :=
  initial_investment * (1 + monthly_interest_rate) ^ n

/-- Theorem stating that an initial investment of $300 results in $363 after 2 months -/
theorem investment_proof :
  ∃ (initial_investment : ℝ),
    total_after_n_months initial_investment 2 = 363 ∧
    initial_investment = 300 :=
by
  sorry


end NUMINAMATH_CALUDE_investment_proof_l3895_389574


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3895_389541

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 8| + 2*y = 12 :=
by
  -- The unique solution is y = 4
  use 4
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3895_389541


namespace NUMINAMATH_CALUDE_track_width_l3895_389565

theorem track_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi) : 
  r₁ - r₂ = 10 := by
sorry

end NUMINAMATH_CALUDE_track_width_l3895_389565
