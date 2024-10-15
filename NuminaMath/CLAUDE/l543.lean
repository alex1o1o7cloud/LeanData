import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l543_54364

/-- The quadratic equation qx^2 - 8x + 2 = 0 has only one solution when q = 8 -/
theorem unique_solution_quadratic :
  ∃! (q : ℝ), q ≠ 0 ∧ (∃! x : ℝ, q * x^2 - 8 * x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l543_54364


namespace NUMINAMATH_CALUDE_grayson_collection_l543_54338

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 48

/-- The number of boxes Abigail collected -/
def abigail_boxes : ℕ := 2

/-- The number of boxes Olivia collected -/
def olivia_boxes : ℕ := 3

/-- The total number of cookies collected -/
def total_cookies : ℕ := 276

/-- The fraction of a box Grayson collected -/
def grayson_fraction : ℚ := 3/4

theorem grayson_collection :
  grayson_fraction * cookies_per_box = 
    total_cookies - (abigail_boxes + olivia_boxes) * cookies_per_box :=
by sorry

end NUMINAMATH_CALUDE_grayson_collection_l543_54338


namespace NUMINAMATH_CALUDE_complex_expression_calculation_l543_54371

theorem complex_expression_calculation (a b : ℂ) :
  a = 3 + 2*I ∧ b = 1 - 3*I → 4*a + 5*b + a*b = 26 - 14*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_calculation_l543_54371


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l543_54373

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (new_mean : ℝ) :
  n = 40 ∧ 
  wrong_value = 75 ∧ 
  correct_value = 50 ∧ 
  new_mean = 99.075 →
  ∃ initial_mean : ℝ, 
    initial_mean = 98.45 ∧ 
    n * new_mean = n * initial_mean + (wrong_value - correct_value) :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l543_54373


namespace NUMINAMATH_CALUDE_sunway_performance_equivalence_l543_54361

/-- The peak performance of the Sunway TaihuLight supercomputer in calculations per second -/
def peak_performance : ℝ := 12.5 * 1e12

/-- The scientific notation representation of the peak performance -/
def scientific_notation : ℝ := 1.25 * 1e13

theorem sunway_performance_equivalence :
  peak_performance = scientific_notation := by sorry

end NUMINAMATH_CALUDE_sunway_performance_equivalence_l543_54361


namespace NUMINAMATH_CALUDE_circular_field_diameter_specific_field_diameter_l543_54378

/-- The diameter of a circular field, given the cost per meter of fencing and the total cost. -/
theorem circular_field_diameter (cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := total_cost / cost_per_meter
  circumference / Real.pi

/-- The diameter of the specific circular field is approximately 16 meters. -/
theorem specific_field_diameter :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |circular_field_diameter 3 150.79644737231007 - 16| < ε :=
sorry

end NUMINAMATH_CALUDE_circular_field_diameter_specific_field_diameter_l543_54378


namespace NUMINAMATH_CALUDE_problem_statement_l543_54320

theorem problem_statement (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 108) :
  a^2 * b + a * b^2 = 108 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l543_54320


namespace NUMINAMATH_CALUDE_problem_1_l543_54330

theorem problem_1 : (1 + 1/4 - 5/6 + 1/2) * (-12) = -11 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l543_54330


namespace NUMINAMATH_CALUDE_coat_price_calculations_l543_54396

def original_price : ℝ := 500
def initial_reduction : ℝ := 300
def discount1 : ℝ := 0.1
def discount2 : ℝ := 0.15

theorem coat_price_calculations :
  let percent_reduction := (initial_reduction / original_price) * 100
  let reduced_price := original_price - initial_reduction
  let percent_increase := ((original_price - reduced_price) / reduced_price) * 100
  let price_after_initial_reduction := reduced_price
  let price_after_discount1 := price_after_initial_reduction * (1 - discount1)
  let final_price := price_after_discount1 * (1 - discount2)
  (percent_reduction = 60 ∧
   percent_increase = 150 ∧
   final_price = 153) := by sorry

end NUMINAMATH_CALUDE_coat_price_calculations_l543_54396


namespace NUMINAMATH_CALUDE_parabola_properties_l543_54331

/-- A parabola with vertex at the origin, focus on the x-axis, and passing through (2, 2) -/
def parabola (x y : ℝ) : Prop := y^2 = 2*x

theorem parabola_properties :
  (parabola 0 0) ∧ 
  (∃ p : ℝ, p > 0 ∧ parabola p 0) ∧ 
  (parabola 2 2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l543_54331


namespace NUMINAMATH_CALUDE_student_rank_from_right_l543_54333

theorem student_rank_from_right 
  (total_students : ℕ) 
  (rank_from_left : ℕ) 
  (h1 : total_students = 21) 
  (h2 : rank_from_left = 5) : 
  total_students - rank_from_left + 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_from_right_l543_54333


namespace NUMINAMATH_CALUDE_system_solution_l543_54303

theorem system_solution (x y z : ℚ) : 
  (1/x + 1/y = 6) ∧ (1/y + 1/z = 4) ∧ (1/z + 1/x = 5) → 
  (x = 2/7) ∧ (y = 2/5) ∧ (z = 2/3) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l543_54303


namespace NUMINAMATH_CALUDE_rectangle_has_equal_diagonals_l543_54332

-- Define a rectangle
def isRectangle (ABCD : Quadrilateral) : Prop := sorry

-- Define equal diagonals
def hasEqualDiagonals (ABCD : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem rectangle_has_equal_diagonals (ABCD : Quadrilateral) :
  isRectangle ABCD → hasEqualDiagonals ABCD := by sorry

end NUMINAMATH_CALUDE_rectangle_has_equal_diagonals_l543_54332


namespace NUMINAMATH_CALUDE_fraction_simplification_l543_54374

theorem fraction_simplification : (240 / 18) * (6 / 135) * (9 / 4) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l543_54374


namespace NUMINAMATH_CALUDE_tan_beta_value_l543_54363

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 2) 
  (h2 : Real.tan (α + β) = -1) : 
  Real.tan β = 3 := by sorry

end NUMINAMATH_CALUDE_tan_beta_value_l543_54363


namespace NUMINAMATH_CALUDE_flag_design_count_l543_54324

/-- The number of possible colors for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The number of possible flag designs -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

theorem flag_design_count :
  num_flag_designs = 27 :=
sorry

end NUMINAMATH_CALUDE_flag_design_count_l543_54324


namespace NUMINAMATH_CALUDE_smallest_cover_l543_54316

/-- The side length of the rectangle. -/
def rectangle_width : ℕ := 3

/-- The height of the rectangle. -/
def rectangle_height : ℕ := 4

/-- The area of a single rectangle. -/
def rectangle_area : ℕ := rectangle_width * rectangle_height

/-- The side length of the square that can be covered exactly by the rectangles. -/
def square_side : ℕ := 12

/-- The area of the square. -/
def square_area : ℕ := square_side * square_side

/-- The number of rectangles needed to cover the square. -/
def num_rectangles : ℕ := square_area / rectangle_area

theorem smallest_cover :
  (∀ n : ℕ, n < square_side → n * n % rectangle_area ≠ 0) ∧
  square_area % rectangle_area = 0 ∧
  num_rectangles = 12 := by
  sorry

#check smallest_cover

end NUMINAMATH_CALUDE_smallest_cover_l543_54316


namespace NUMINAMATH_CALUDE_linda_age_l543_54310

theorem linda_age (carlos_age maya_age linda_age : ℕ) : 
  carlos_age = 12 →
  maya_age = carlos_age + 4 →
  linda_age = maya_age - 5 →
  linda_age = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_linda_age_l543_54310


namespace NUMINAMATH_CALUDE_bookstore_discount_theorem_l543_54399

variable (Book : Type)
variable (bookstore : Set Book)
variable (discounted_by_20_percent : Book → Prop)

theorem bookstore_discount_theorem 
  (h : ¬ ∀ b ∈ bookstore, discounted_by_20_percent b) : 
  (∃ b ∈ bookstore, ¬ discounted_by_20_percent b) ∧ 
  (¬ ∀ b ∈ bookstore, discounted_by_20_percent b) := by
  sorry

end NUMINAMATH_CALUDE_bookstore_discount_theorem_l543_54399


namespace NUMINAMATH_CALUDE_degree_of_polynomial_l543_54386

def p (x : ℝ) : ℝ := (2*x^5 - 3*x^3 + x^2 - 14) * (3*x^11 - 9*x^8 + 9*x^5 + 30) - (x^3 + 5)^7

theorem degree_of_polynomial : 
  ∃ (a : ℝ) (q : ℝ → ℝ), a ≠ 0 ∧ 
  (∀ (x : ℝ), p x = a * x^21 + q x) ∧ 
  (∃ (N : ℝ), ∀ (x : ℝ), |x| > N → |q x| < |a| * |x|^21) :=
sorry

end NUMINAMATH_CALUDE_degree_of_polynomial_l543_54386


namespace NUMINAMATH_CALUDE_temperature_average_bounds_l543_54343

theorem temperature_average_bounds (temps : List ℝ) 
  (h_count : temps.length = 5)
  (h_min : temps.minimum? = some 42)
  (h_max : ∀ t ∈ temps, t ≤ 57) : 
  let avg := temps.sum / temps.length
  42 ≤ avg ∧ avg ≤ 57 := by sorry

end NUMINAMATH_CALUDE_temperature_average_bounds_l543_54343


namespace NUMINAMATH_CALUDE_smallest_product_is_690_l543_54306

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def smallest_three_digit_product (m : ℕ) : Prop :=
  ∃ a b : ℕ,
    m = a * b * (10*a + b) * (a + b) ∧
    a < 10 ∧ b < 10 ∧
    is_prime a ∧ is_prime b ∧
    is_prime (10*a + b) ∧ is_prime (a + b) ∧
    (a + b) % 5 = 1 ∧
    m ≥ 100 ∧ m < 1000 ∧
    ∀ n : ℕ, n ≥ 100 ∧ n < m → ¬(smallest_three_digit_product n)

theorem smallest_product_is_690 :
  smallest_three_digit_product 690 :=
sorry

end NUMINAMATH_CALUDE_smallest_product_is_690_l543_54306


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l543_54315

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 2 + a 6 = 8 ∧ 
  a 3 + a 4 = 3

/-- The common difference of the arithmetic sequence is 5 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l543_54315


namespace NUMINAMATH_CALUDE_gcd_15378_21333_48906_l543_54325

theorem gcd_15378_21333_48906 : Nat.gcd 15378 (Nat.gcd 21333 48906) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_15378_21333_48906_l543_54325


namespace NUMINAMATH_CALUDE_geometric_series_sum_proof_l543_54368

/-- The sum of the infinite geometric series 1/4 + 1/12 + 1/36 + 1/108 + ... -/
def geometric_series_sum : ℚ := 3/8

/-- The first term of the geometric series -/
def a : ℚ := 1/4

/-- The common ratio of the geometric series -/
def r : ℚ := 1/3

/-- Theorem stating that the sum of the infinite geometric series
    1/4 + 1/12 + 1/36 + 1/108 + ... is equal to 3/8 -/
theorem geometric_series_sum_proof :
  geometric_series_sum = a / (1 - r) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_proof_l543_54368


namespace NUMINAMATH_CALUDE_camel_cost_l543_54322

/-- The cost of animals in an economy where the relative prices are fixed. -/
structure AnimalCosts where
  camel : ℚ
  horse : ℚ
  goat : ℚ
  ox : ℚ
  elephant : ℚ

/-- The conditions of the animal costs problem. -/
def animal_costs_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  26 * costs.horse = 50 * costs.goat ∧
  20 * costs.goat = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 170000

/-- The theorem stating that under the given conditions, a camel costs 27200. -/
theorem camel_cost (costs : AnimalCosts) :
  animal_costs_conditions costs → costs.camel = 27200 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l543_54322


namespace NUMINAMATH_CALUDE_max_discount_rate_l543_54350

/-- Proves the maximum discount rate for a given cost price, selling price, and minimum profit margin. -/
theorem max_discount_rate 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (min_profit_margin : ℝ) 
  (h1 : cost_price = 4)
  (h2 : selling_price = 5)
  (h3 : min_profit_margin = 0.1)
  : ∃ (max_discount : ℝ), 
    max_discount = 12 ∧ 
    ∀ (discount : ℝ), 
      discount ≤ max_discount ↔ 
      selling_price * (1 - discount / 100) ≥ cost_price * (1 + min_profit_margin) :=
by sorry

end NUMINAMATH_CALUDE_max_discount_rate_l543_54350


namespace NUMINAMATH_CALUDE_cafeteria_duty_assignments_l543_54360

def class_size : ℕ := 28
def duty_size : ℕ := 4

theorem cafeteria_duty_assignments :
  (Nat.choose class_size duty_size = 20475) ∧
  (Nat.choose (class_size - 1) (duty_size - 1) = 2925) := by
  sorry

#check cafeteria_duty_assignments

end NUMINAMATH_CALUDE_cafeteria_duty_assignments_l543_54360


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l543_54329

/-- Given two people moving in opposite directions for 1 hour, 
    where one moves at 35 km/h and they end up 60 km apart,
    prove that the speed of the other person is 25 km/h. -/
theorem opposite_direction_speed 
  (speed_person1 : ℝ) 
  (speed_person2 : ℝ) 
  (time : ℝ) 
  (total_distance : ℝ) 
  (h1 : speed_person2 = 35) 
  (h2 : time = 1) 
  (h3 : total_distance = 60) 
  (h4 : speed_person1 * time + speed_person2 * time = total_distance) : 
  speed_person1 = 25 := by
  sorry

#check opposite_direction_speed

end NUMINAMATH_CALUDE_opposite_direction_speed_l543_54329


namespace NUMINAMATH_CALUDE_collinear_points_triangle_inequality_l543_54348

/-- Given five distinct collinear points A, B, C, D, E in order, with segment lengths AB = p, AC = q, AD = r, BE = s, DE = t,
    if AB and DE can be rotated about B and D respectively to form a triangle with positive area,
    then p < r/2 and s < t + p/2 must be true. -/
theorem collinear_points_triangle_inequality (p q r s t : ℝ) 
  (h_distinct : p > 0 ∧ q > p ∧ r > q ∧ s > 0 ∧ t > 0) 
  (h_triangle : p + s > r + t - s ∧ s + (r + t - s) > p ∧ p + (r + t - s) > s) :
  p < r / 2 ∧ s < t + p / 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_triangle_inequality_l543_54348


namespace NUMINAMATH_CALUDE_sum_of_cubes_l543_54319

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -2) : 
  a^3 + b^3 = 45 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l543_54319


namespace NUMINAMATH_CALUDE_percentage_problem_l543_54379

/-- Given a number N and a percentage P, this theorem proves that
    if P% of N is 24 less than 50% of N, and N = 160, then P = 35. -/
theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 160 → 
  (P / 100) * N = (50 / 100) * N - 24 → 
  P = 35 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l543_54379


namespace NUMINAMATH_CALUDE_min_value_of_a_l543_54392

theorem min_value_of_a (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 9) ↔ a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l543_54392


namespace NUMINAMATH_CALUDE_equation_system_properties_l543_54318

/-- Represents a system of equations mx + ny² = 0 and mx² + ny² = 1 -/
structure EquationSystem where
  m : ℝ
  n : ℝ
  h_m_neg : m < 0
  h_n_pos : n > 0

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point satisfies both equations in the system -/
def satisfies_equations (sys : EquationSystem) (p : Point) : Prop :=
  sys.m * p.x + sys.n * p.y^2 = 0 ∧ sys.m * p.x^2 + sys.n * p.y^2 = 1

/-- States that the equation system represents a parabola -/
def is_parabola (sys : EquationSystem) : Prop :=
  ∃ (a b c : ℝ), ∀ (x y : ℝ), sys.m * x + sys.n * y^2 = 0 ↔ y = a * x^2 + b * x + c

/-- States that the equation system represents a hyperbola -/
def is_hyperbola (sys : EquationSystem) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), sys.m * x^2 + sys.n * y^2 = 1 ↔ (x^2 / a^2) - (y^2 / b^2) = 1

theorem equation_system_properties (sys : EquationSystem) :
  is_parabola sys ∧ 
  is_hyperbola sys ∧ 
  satisfies_equations sys ⟨0, 0⟩ ∧ 
  satisfies_equations sys ⟨1, 0⟩ :=
sorry

end NUMINAMATH_CALUDE_equation_system_properties_l543_54318


namespace NUMINAMATH_CALUDE_eraser_cost_mary_eraser_cost_l543_54351

/-- The cost of each eraser given Mary's school supplies purchase --/
theorem eraser_cost (classes : ℕ) (folders_per_class : ℕ) (pencils_per_class : ℕ) 
  (pencils_per_eraser : ℕ) (folder_cost : ℚ) (pencil_cost : ℚ) (total_spent : ℚ) 
  (paint_cost : ℚ) : ℚ :=
  let folders := classes * folders_per_class
  let pencils := classes * pencils_per_class
  let erasers := pencils / pencils_per_eraser
  let folder_total := folders * folder_cost
  let pencil_total := pencils * pencil_cost
  let eraser_total := total_spent - folder_total - pencil_total - paint_cost
  eraser_total / erasers

/-- The cost of each eraser in Mary's specific purchase is $1 --/
theorem mary_eraser_cost : 
  eraser_cost 6 1 3 6 6 2 80 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_eraser_cost_mary_eraser_cost_l543_54351


namespace NUMINAMATH_CALUDE_slope_angle_vertical_line_l543_54342

-- Define a vertical line
def vertical_line (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = a}

-- Define the slope angle of a line
def slope_angle (L : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem slope_angle_vertical_line :
  slope_angle (vertical_line 2) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_vertical_line_l543_54342


namespace NUMINAMATH_CALUDE_intersection_of_sets_l543_54355

theorem intersection_of_sets : 
  let A : Set ℝ := {x | x + 2 = 0}
  let B : Set ℝ := {x | x^2 - 4 = 0}
  A ∩ B = {-2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l543_54355


namespace NUMINAMATH_CALUDE_incorrect_quotient_calculation_l543_54317

theorem incorrect_quotient_calculation (dividend : ℕ) (correct_divisor incorrect_divisor correct_quotient : ℕ) 
  (h1 : dividend = correct_divisor * correct_quotient)
  (h2 : correct_divisor = 21)
  (h3 : incorrect_divisor = 12)
  (h4 : correct_quotient = 28) :
  dividend / incorrect_divisor = 49 := by
sorry

end NUMINAMATH_CALUDE_incorrect_quotient_calculation_l543_54317


namespace NUMINAMATH_CALUDE_inequality_proof_l543_54394

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ+, (a + b)^n.val - a^n.val - b^n.val ≥ 2^(2*n.val) - 2^(n.val+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l543_54394


namespace NUMINAMATH_CALUDE_triangle_side_length_l543_54305

/-- Given two triangles ABC and DEF with specified side lengths and angles,
    prove that the length of EF is 3.75 units when the area of DEF is half that of ABC. -/
theorem triangle_side_length (AB DE AC DF : ℝ) (angleBAC angleEDF : ℝ) :
  AB = 5 →
  DE = 2 →
  AC = 6 →
  DF = 3 →
  angleBAC = 30 * π / 180 →
  angleEDF = 45 * π / 180 →
  (1 / 2 * DE * DF * Real.sin angleEDF) = (1 / 4 * AB * AC * Real.sin angleBAC) →
  ∃ (EF : ℝ), EF = 3.75 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l543_54305


namespace NUMINAMATH_CALUDE_triangle_side_altitude_inequality_l543_54347

/-- Given a triangle with sides a, b, c where a > b > c and corresponding altitudes h_a, h_b, h_c,
    prove that a + h_a > b + h_b > c + h_c. -/
theorem triangle_side_altitude_inequality 
  (a b c h_a h_b h_c : ℝ) 
  (h_positive : 0 < c ∧ 0 < b ∧ 0 < a)
  (h_order : c < b ∧ b < a) 
  (h_triangle : h_a * a = h_b * b ∧ h_b * b = h_c * c) : 
  a + h_a > b + h_b ∧ b + h_b > c + h_c :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_altitude_inequality_l543_54347


namespace NUMINAMATH_CALUDE_mass_of_man_is_60kg_l543_54376

/-- The mass of a man causing a boat to sink in water -/
def mass_of_man (boat_length boat_breadth boat_sinking water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinking * water_density

/-- Theorem stating the mass of the man is 60 kg -/
theorem mass_of_man_is_60kg : 
  mass_of_man 3 2 0.01 1000 = 60 := by sorry

end NUMINAMATH_CALUDE_mass_of_man_is_60kg_l543_54376


namespace NUMINAMATH_CALUDE_race_time_difference_l543_54340

/-- Proves that in a 1000-meter race where runner A finishes in 90 seconds and is 100 meters ahead of runner B at the finish line, A beats B by 9 seconds. -/
theorem race_time_difference (race_length : ℝ) (a_time : ℝ) (distance_difference : ℝ) : 
  race_length = 1000 →
  a_time = 90 →
  distance_difference = 100 →
  (race_length / a_time) * (distance_difference / race_length) * a_time = 9 :=
by
  sorry

#check race_time_difference

end NUMINAMATH_CALUDE_race_time_difference_l543_54340


namespace NUMINAMATH_CALUDE_angle_properties_l543_54377

theorem angle_properties (α : ℝ) (h : Real.tan α = -4/3) :
  (2 * Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 2 * Real.cos α ^ 2 = 2) ∧
  ((2 * Real.sin (π - α) + Real.sin (π/2 - α) + Real.sin (4*π)) / 
   (Real.cos (3*π/2 - α) + Real.cos (-α)) = -5/7) :=
by sorry

end NUMINAMATH_CALUDE_angle_properties_l543_54377


namespace NUMINAMATH_CALUDE_smallest_label_same_as_1993_solution_l543_54358

/-- The number of points on the circle -/
def num_points : ℕ := 2000

/-- The highest label used in the problem -/
def max_label : ℕ := 1993

/-- Function to calculate the position of a label -/
def label_position (n : ℕ) : ℕ :=
  (n * (n + 1) / 2 - 1) % num_points

/-- Theorem stating that 118 is the smallest positive integer that labels the same point as 1993 -/
theorem smallest_label_same_as_1993 :
  ∀ k : ℕ, 0 < k → k < 118 → label_position k ≠ label_position max_label ∧
  label_position 118 = label_position max_label := by
  sorry

/-- Main theorem proving the solution -/
theorem solution : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, 0 < k → k < n → label_position k ≠ label_position max_label) ∧
  label_position n = label_position max_label ∧ n = 118 := by
  sorry

end NUMINAMATH_CALUDE_smallest_label_same_as_1993_solution_l543_54358


namespace NUMINAMATH_CALUDE_prob_second_odd_given_first_even_l543_54383

/-- A card is represented by a natural number between 1 and 5 -/
def Card : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 5 }

/-- The set of all cards -/
def allCards : Finset Card := sorry

/-- A card is even if its number is even -/
def isEven (c : Card) : Prop := c.val % 2 = 0

/-- A card is odd if its number is odd -/
def isOdd (c : Card) : Prop := c.val % 2 = 1

/-- The set of even cards -/
def evenCards : Finset Card := sorry

/-- The set of odd cards -/
def oddCards : Finset Card := sorry

theorem prob_second_odd_given_first_even :
  (Finset.card oddCards : ℚ) / (Finset.card allCards - 1 : ℚ) = 3/4 := by sorry

end NUMINAMATH_CALUDE_prob_second_odd_given_first_even_l543_54383


namespace NUMINAMATH_CALUDE_correct_algorithm_structures_l543_54341

-- Define the possible algorithm structures
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop
  | Flow
  | Nested

-- Define a function that checks if a list of structures is correct
def isCorrectStructureList (list : List AlgorithmStructure) : Prop :=
  list = [AlgorithmStructure.Sequential, AlgorithmStructure.Conditional, AlgorithmStructure.Loop]

-- State the theorem
theorem correct_algorithm_structures :
  isCorrectStructureList [AlgorithmStructure.Sequential, AlgorithmStructure.Conditional, AlgorithmStructure.Loop] :=
by sorry


end NUMINAMATH_CALUDE_correct_algorithm_structures_l543_54341


namespace NUMINAMATH_CALUDE_consecutive_five_digit_numbers_l543_54352

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def abccc (a b c : ℕ) : ℕ := 10000 * a + 1000 * b + 100 * c + 10 * c + c

def abbbb (a b : ℕ) : ℕ := 10000 * a + 1000 * b + 100 * b + 10 * b + b

theorem consecutive_five_digit_numbers :
  ∀ a b c : ℕ,
    a < 10 → b < 10 → c < 10 →
    is_five_digit (abccc a b c) →
    is_five_digit (abbbb a b) →
    (abccc a b c).succ = abbbb a b ∨ (abbbb a b).succ = abccc a b c →
    ((a = 1 ∧ b = 0 ∧ c = 9) ∨ (a = 8 ∧ b = 9 ∧ c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_five_digit_numbers_l543_54352


namespace NUMINAMATH_CALUDE_talia_father_age_l543_54395

/-- Represents the ages of Talia, her mother, and her father -/
structure FamilyAges where
  talia : ℕ
  mother : ℕ
  father : ℕ

/-- Conditions for the family ages problem -/
def FamilyAgeProblem (ages : FamilyAges) : Prop :=
  (ages.talia + 7 = 20) ∧
  (ages.mother = 3 * ages.talia) ∧
  (ages.father + 3 = ages.mother)

/-- Theorem stating that given the conditions, Talia's father is 36 years old -/
theorem talia_father_age (ages : FamilyAges) :
  FamilyAgeProblem ages → ages.father = 36 := by
  sorry

end NUMINAMATH_CALUDE_talia_father_age_l543_54395


namespace NUMINAMATH_CALUDE_problem_solution_l543_54334

theorem problem_solution : 25 * (216 / 3 + 36 / 6 + 16 / 25 + 2) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l543_54334


namespace NUMINAMATH_CALUDE_no_99_cents_combination_l543_54384

/-- Represents the types of coins available -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a combination of five coins -/
def CoinCombination := Vector Coin 5

/-- Calculates the total value of a coin combination in cents -/
def totalValue (combo : CoinCombination) : Nat :=
  combo.toList.map coinValue |>.sum

/-- Theorem: It's impossible to make 99 cents with exactly five coins -/
theorem no_99_cents_combination :
  ¬∃ (combo : CoinCombination), totalValue combo = 99 := by
  sorry


end NUMINAMATH_CALUDE_no_99_cents_combination_l543_54384


namespace NUMINAMATH_CALUDE_roadwork_truckloads_per_mile_l543_54367

theorem roadwork_truckloads_per_mile :
  let road_length : ℝ := 16
  let gravel_bags_per_truck : ℕ := 2
  let gravel_to_pitch_ratio : ℕ := 5
  let day1_miles : ℝ := 4
  let day2_miles : ℝ := 7
  let day3_pitch_barrels : ℕ := 6
  
  let total_paved_miles : ℝ := day1_miles + day2_miles
  let remaining_miles : ℝ := road_length - total_paved_miles
  let truckloads_per_mile : ℝ := day3_pitch_barrels / remaining_miles
  
  truckloads_per_mile = 1.2 := by sorry

end NUMINAMATH_CALUDE_roadwork_truckloads_per_mile_l543_54367


namespace NUMINAMATH_CALUDE_greatest_c_not_in_range_l543_54345

def f (c : ℝ) (x : ℝ) : ℝ := x^2 + c*x + 15

theorem greatest_c_not_in_range : 
  ∀ c : ℤ, (∀ x : ℝ, f c x ≠ -9) ↔ c ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_c_not_in_range_l543_54345


namespace NUMINAMATH_CALUDE_square_difference_cubed_l543_54335

theorem square_difference_cubed : (7^2 - 5^2)^3 = 13824 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_cubed_l543_54335


namespace NUMINAMATH_CALUDE_power_equation_solution_l543_54339

theorem power_equation_solution (p : ℕ) : (81 ^ 10 : ℕ) = 3 ^ p → p = 40 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l543_54339


namespace NUMINAMATH_CALUDE_sqrt_inequality_and_sum_of_squares_l543_54346

theorem sqrt_inequality_and_sum_of_squares (a b c : ℝ) : 
  (Real.sqrt 6 + Real.sqrt 10 > 2 * Real.sqrt 3 + 2) ∧ 
  (a^2 + b^2 + c^2 ≥ a*b + b*c + a*c) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_and_sum_of_squares_l543_54346


namespace NUMINAMATH_CALUDE_light_travel_100_years_l543_54354

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- Theorem stating the distance light travels in 100 years -/
theorem light_travel_100_years :
  100 * light_year_distance = 587 * (10 ^ 12 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_light_travel_100_years_l543_54354


namespace NUMINAMATH_CALUDE_two_numbers_problem_l543_54344

theorem two_numbers_problem : ∃ (A B : ℕ+), 
  A + B = 581 ∧ 
  Nat.lcm A B / Nat.gcd A B = 240 ∧ 
  ((A = 560 ∧ B = 21) ∨ (A = 21 ∧ B = 560)) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l543_54344


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l543_54362

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 1 → (1 / a) < 1) ∧ ¬((1 / a) < 1 → a > 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l543_54362


namespace NUMINAMATH_CALUDE_ohara_triple_81_49_l543_54314

/-- O'Hara triple definition -/
def is_ohara_triple (a b x : ℕ) : Prop := Real.sqrt a + Real.sqrt b = x

/-- The main theorem -/
theorem ohara_triple_81_49 (x : ℕ) :
  is_ohara_triple 81 49 x → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_81_49_l543_54314


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l543_54327

theorem semicircle_area_with_inscribed_rectangle (r : ℝ) : 
  r > 0 → 
  (∃ (w h : ℝ), w > 0 ∧ h > 0 ∧ w = 1 ∧ h = 3 ∧ h = r) → 
  (π * r^2) / 2 = 9 * π / 2 := by
sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l543_54327


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l543_54349

/-- Given the cost of 3 pens and 5 pencils is Rs. 100, and the cost ratio of one pen to one pencil
    is 5:1, prove that the cost of one dozen pens is Rs. 300. -/
theorem cost_of_dozen_pens (pen_cost pencil_cost : ℕ) : 
  (3 * pen_cost + 5 * pencil_cost = 100) →
  (pen_cost = 5 * pencil_cost) →
  (12 * pen_cost = 300) :=
by sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l543_54349


namespace NUMINAMATH_CALUDE_base_conversion_problem_l543_54370

theorem base_conversion_problem (n d : ℕ) (hn : n > 0) (hd : d ≤ 9) :
  3 * n^2 + 2 * n + d = 263 ∧ 3 * n^2 + 2 * n + 4 = 253 + 6 * d → n + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l543_54370


namespace NUMINAMATH_CALUDE_cos_double_angle_unit_circle_l543_54301

theorem cos_double_angle_unit_circle (α : Real) :
  (Real.cos α = -Real.sqrt 3 / 2 ∧ Real.sin α = 1 / 2) →
  Real.cos (2 * α) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_unit_circle_l543_54301


namespace NUMINAMATH_CALUDE_M_sum_l543_54312

def M : ℕ → ℕ
| 0 => 3^2
| 1 => 6^2
| n+2 => (3*n + 6)^2 - (3*n + 3)^2 + M n

theorem M_sum : M 49 = 11475 := by
  sorry

end NUMINAMATH_CALUDE_M_sum_l543_54312


namespace NUMINAMATH_CALUDE_range_of_m_for_linear_system_l543_54393

/-- Given a system of linear equations and an inequality condition, 
    prove that m must be less than 1. -/
theorem range_of_m_for_linear_system (x y m : ℝ) : 
  3 * x + y = 3 * m + 1 →
  x + 2 * y = 3 →
  2 * x - y < 1 →
  m < 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_linear_system_l543_54393


namespace NUMINAMATH_CALUDE_two_digit_number_proof_l543_54308

theorem two_digit_number_proof : ∃! n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (n / 10 = 2 * (n % 10)) ∧ 
  (∃ m : ℕ, n + (n / 10)^2 = m^2) ∧
  n = 21 := by sorry

end NUMINAMATH_CALUDE_two_digit_number_proof_l543_54308


namespace NUMINAMATH_CALUDE_plane_equation_l543_54387

/-- A plane in 3D space --/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_pos : a > 0
  coprime : Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Int.natAbs d))) = 1

/-- A point in 3D space --/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

def is_parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.a = k * p2.a ∧ p1.b = k * p2.b ∧ p1.c = k * p2.c

def point_on_plane (pt : Point3D) (p : Plane) : Prop :=
  p.a * pt.x + p.b * pt.y + p.c * pt.z + p.d = 0

theorem plane_equation :
  ∃ (p : Plane),
    is_parallel p { a := 3, b := -2, c := 4, d := -6, a_pos := by simp, coprime := by sorry } ∧
    point_on_plane { x := 2, y := 3, z := -1 } p ∧
    p.a = 3 ∧ p.b = -2 ∧ p.c = 4 ∧ p.d = 4 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_l543_54387


namespace NUMINAMATH_CALUDE_smallest_positive_integer_3001m_24567n_l543_54372

theorem smallest_positive_integer_3001m_24567n : 
  ∃ (m n : ℤ), 3001 * m + 24567 * n = (Nat.gcd 3001 24567 : ℤ) ∧
  ∀ (k : ℤ), (∃ (a b : ℤ), k = 3001 * a + 24567 * b) → k = 0 ∨ abs k ≥ (Nat.gcd 3001 24567 : ℤ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_3001m_24567n_l543_54372


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l543_54311

/-- Represents the financial data of a person -/
structure FinancialData where
  income : ℕ
  savings : ℕ

/-- Calculates the expenditure given income and savings -/
def expenditure (data : FinancialData) : ℕ :=
  data.income - data.savings

/-- Simplifies a ratio represented by two natural numbers -/
def simplifyRatio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

/-- The main theorem stating the ratio of income to expenditure -/
theorem income_expenditure_ratio (data : FinancialData) 
  (h1 : data.income = 40000) 
  (h2 : data.savings = 5000) : 
  simplifyRatio data.income (expenditure data) = (8, 7) := by
  sorry

#eval simplifyRatio 40000 35000

end NUMINAMATH_CALUDE_income_expenditure_ratio_l543_54311


namespace NUMINAMATH_CALUDE_winter_mows_calculation_winter_mows_value_l543_54389

/-- The number of times Kale mowed his lawn in winter -/
def winter_mows : ℕ := sorry

/-- The number of times Kale mowed his lawn in spring -/
def spring_mows : ℕ := 8

/-- The number of times Kale mowed his lawn in summer -/
def summer_mows : ℕ := 5

/-- The number of times Kale mowed his lawn in fall -/
def fall_mows : ℕ := 12

/-- The average number of times Kale mowed his lawn per season -/
def average_mows_per_season : ℕ := 7

/-- The number of seasons in a year -/
def seasons_per_year : ℕ := 4

theorem winter_mows_calculation :
  winter_mows = average_mows_per_season * seasons_per_year - (spring_mows + summer_mows + fall_mows) :=
by sorry

theorem winter_mows_value : winter_mows = 3 :=
by sorry

end NUMINAMATH_CALUDE_winter_mows_calculation_winter_mows_value_l543_54389


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l543_54359

theorem expression_equals_negative_one (b y : ℝ) (hb : b ≠ 0) (hy1 : y ≠ b) (hy2 : y ≠ -b) :
  (((b / (b + y)) + (y / (b - y))) / ((y / (b + y)) - (b / (b - y)))) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l543_54359


namespace NUMINAMATH_CALUDE_correct_number_of_pitbulls_l543_54300

/-- Represents the number of pitbulls James has -/
def num_pitbulls : ℕ := 2

/-- Represents the number of huskies James has -/
def num_huskies : ℕ := 5

/-- Represents the number of golden retrievers James has -/
def num_golden_retrievers : ℕ := 4

/-- Represents the number of pups each husky and pitbull has -/
def pups_per_husky_pitbull : ℕ := 3

/-- Represents the additional number of pups each golden retriever has compared to huskies -/
def additional_pups_golden : ℕ := 2

/-- Represents the difference between total pups and adult dogs -/
def pup_adult_difference : ℕ := 30

theorem correct_number_of_pitbulls :
  (num_huskies * pups_per_husky_pitbull) +
  (num_golden_retrievers * (pups_per_husky_pitbull + additional_pups_golden)) +
  (num_pitbulls * pups_per_husky_pitbull) =
  (num_huskies + num_golden_retrievers + num_pitbulls) + pup_adult_difference := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_pitbulls_l543_54300


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l543_54390

theorem sum_of_four_numbers : 5678 + 6785 + 7856 + 8567 = 28886 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l543_54390


namespace NUMINAMATH_CALUDE_sum_of_solutions_equals_six_l543_54337

theorem sum_of_solutions_equals_six :
  ∃ (x₁ x₂ : ℝ), 
    (3 : ℝ) ^ (x₁^2 - 4*x₁ - 3) = 9 ^ (x₁ - 5) ∧
    (3 : ℝ) ^ (x₂^2 - 4*x₂ - 3) = 9 ^ (x₂ - 5) ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 6 ∧
    ∀ (x : ℝ), (3 : ℝ) ^ (x^2 - 4*x - 3) = 9 ^ (x - 5) → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_equals_six_l543_54337


namespace NUMINAMATH_CALUDE_marble_selection_ways_l543_54309

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of marbles -/
def total_marbles : ℕ := 15

/-- The number of marbles to be chosen -/
def marbles_to_choose : ℕ := 5

/-- The number of specific colored marbles (red + green + blue) -/
def specific_colored_marbles : ℕ := 6

/-- The number of ways to choose 2 marbles from the specific colored ones -/
def ways_to_choose_specific : ℕ := 9

/-- The number of remaining marbles after removing the specific colored ones -/
def remaining_marbles : ℕ := total_marbles - specific_colored_marbles

theorem marble_selection_ways :
  ways_to_choose_specific * choose remaining_marbles (marbles_to_choose - 2) = 1980 :=
sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l543_54309


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_inequality_l543_54385

/-- An arithmetic sequence of 8 terms with positive values and non-zero common difference -/
structure ArithmeticSequence8 where
  a : Fin 8 → ℝ
  positive : ∀ i, a i > 0
  common_diff : ℝ
  common_diff_neq_zero : common_diff ≠ 0
  is_arithmetic : ∀ i j, i < j → a j - a i = common_diff * (j - i)

/-- For an arithmetic sequence of 8 terms with positive values and non-zero common difference,
    the product of the first and last terms is less than the product of the fourth and fifth terms -/
theorem arithmetic_sequence_product_inequality (seq : ArithmeticSequence8) :
  seq.a 0 * seq.a 7 < seq.a 3 * seq.a 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_inequality_l543_54385


namespace NUMINAMATH_CALUDE_remaining_tickets_l543_54353

/-- Represents the number of tickets Tom won and spent at the arcade -/
def arcade_tickets (x y : ℕ) : Prop :=
  let whack_a_mole := 32
  let skee_ball := 25
  let space_invaders := x
  let hat := 7
  let keychain := 10
  let small_toy := 15
  y = (whack_a_mole + skee_ball + space_invaders) - (hat + keychain + small_toy)

/-- Theorem stating that the number of tickets Tom has left is 25 plus the number of tickets he won from 'space invaders' -/
theorem remaining_tickets (x y : ℕ) :
  arcade_tickets x y → y = 25 + x := by
  sorry

end NUMINAMATH_CALUDE_remaining_tickets_l543_54353


namespace NUMINAMATH_CALUDE_sunflower_seed_tins_l543_54336

theorem sunflower_seed_tins (candy_bags : ℕ) (candies_per_bag : ℕ) (seeds_per_tin : ℕ) (total_items : ℕ) : 
  candy_bags = 19 →
  candies_per_bag = 46 →
  seeds_per_tin = 170 →
  total_items = 1894 →
  (total_items - candy_bags * candies_per_bag) / seeds_per_tin = 6 :=
by sorry

end NUMINAMATH_CALUDE_sunflower_seed_tins_l543_54336


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l543_54357

-- Define the function f implicitly
def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) > 0
def solution_set_f (x : ℝ) : Prop := x < -1 ∨ x > 1/2

-- Define the solution set of f(10^x) > 0
def solution_set_f_exp (x : ℝ) : Prop := x > -Real.log 2 / Real.log 10

-- Theorem statement
theorem solution_set_equivalence :
  (∀ x, f x > 0 ↔ solution_set_f x) →
  (∀ x, f (10^x) > 0 ↔ solution_set_f_exp x) :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l543_54357


namespace NUMINAMATH_CALUDE_ln_x_plus_one_negative_condition_l543_54323

theorem ln_x_plus_one_negative_condition (x : ℝ) :
  (∀ x, (Real.log (x + 1) < 0) → (x < 0)) ∧
  (∃ x, x < 0 ∧ ¬(Real.log (x + 1) < 0)) :=
sorry

end NUMINAMATH_CALUDE_ln_x_plus_one_negative_condition_l543_54323


namespace NUMINAMATH_CALUDE_fraction_subtraction_decreases_l543_54366

theorem fraction_subtraction_decreases (a b n : ℕ) 
  (h1 : a < b) (h2 : 0 < n) (h3 : n < a) : 
  (a - n : ℚ) / (b - n) < (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_decreases_l543_54366


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l543_54304

def M : Set ℝ := {x | x^2 - 1 ≤ 0}
def N : Set ℝ := {x | x^2 - 3*x > 0}

theorem intersection_of_M_and_N : ∀ x : ℝ, x ∈ M ∩ N ↔ -1 ≤ x ∧ x < 0 := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l543_54304


namespace NUMINAMATH_CALUDE_two_pedestrians_problem_l543_54356

/-- Two pedestrians problem -/
theorem two_pedestrians_problem (meet_time : ℝ) (time_difference : ℝ) :
  meet_time = 2 ∧ time_difference = 5/3 →
  ∃ (distance_AB : ℝ) (speed_A : ℝ) (speed_B : ℝ),
    distance_AB = 18 ∧
    speed_A = 5 ∧
    speed_B = 4 ∧
    distance_AB = speed_A * meet_time + speed_B * meet_time ∧
    distance_AB / speed_A = meet_time + time_difference ∧
    distance_AB / speed_B = meet_time + (meet_time + time_difference) :=
by sorry

end NUMINAMATH_CALUDE_two_pedestrians_problem_l543_54356


namespace NUMINAMATH_CALUDE_f_difference_l543_54397

/-- The function f(x) = 3x^3 + 2x^2 - 4x - 1 -/
def f (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 - 4 * x - 1

/-- Theorem stating that f(x + h) - f(x) = h(9x^2 + 9xh + 3h^2 + 4x + 2h - 4) for all real x and h -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (9 * x^2 + 9 * x * h + 3 * h^2 + 4 * x + 2 * h - 4) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l543_54397


namespace NUMINAMATH_CALUDE_only_zero_is_purely_imaginary_l543_54328

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number parameterized by m. -/
def complexNumber (m : ℝ) : ℂ :=
  ⟨m^2 - 3*m, m^2 - 5*m + 6⟩

theorem only_zero_is_purely_imaginary :
  ∃! m : ℝ, isPurelyImaginary (complexNumber m) ∧ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_only_zero_is_purely_imaginary_l543_54328


namespace NUMINAMATH_CALUDE_binomial_square_example_l543_54369

theorem binomial_square_example : 15^2 + 2*(15*3) + 3^2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_example_l543_54369


namespace NUMINAMATH_CALUDE_largest_number_l543_54321

theorem largest_number : 
  let a := 0.989
  let b := 0.9879
  let c := 0.98809
  let d := 0.9807
  let e := 0.9819
  (a ≥ b) ∧ (a ≥ c) ∧ (a ≥ d) ∧ (a ≥ e) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l543_54321


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l543_54302

theorem absolute_value_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 5*x*y) :
  |((x+y)/(x-y))| = Real.sqrt (7/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l543_54302


namespace NUMINAMATH_CALUDE_total_luggage_is_142_l543_54382

/-- Calculates the total number of luggage pieces allowed on an international flight --/
def totalLuggageAllowed (economyPassengers businessPassengers firstClassPassengers : ℕ) : ℕ :=
  let economyAllowance := 5
  let businessAllowance := 8
  let firstClassAllowance := 12
  economyPassengers * economyAllowance + businessPassengers * businessAllowance + firstClassPassengers * firstClassAllowance

/-- Theorem stating that the total luggage allowed for the given passenger numbers is 142 --/
theorem total_luggage_is_142 :
  totalLuggageAllowed 10 7 3 = 142 := by
  sorry

#eval totalLuggageAllowed 10 7 3

end NUMINAMATH_CALUDE_total_luggage_is_142_l543_54382


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l543_54380

theorem rectangle_area_increase (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  1.11 * L * (B * (1 + 22/100)) = 1.3542 * (L * B) := by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l543_54380


namespace NUMINAMATH_CALUDE_average_decrease_l543_54307

theorem average_decrease (n : ℕ) (old_avg new_obs : ℚ) : 
  n = 6 →
  old_avg = 12 →
  new_obs = 5 →
  (n * old_avg + new_obs) / (n + 1) = old_avg - 1 := by
  sorry

end NUMINAMATH_CALUDE_average_decrease_l543_54307


namespace NUMINAMATH_CALUDE_number_operation_l543_54388

theorem number_operation (x : ℝ) : x - 10 = 15 → x + 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_l543_54388


namespace NUMINAMATH_CALUDE_circles_intersect_at_right_angle_l543_54398

/-- Two circles intersect at right angles if and only if the sum of the squares of their radii equals the square of the distance between their centers. -/
theorem circles_intersect_at_right_angle (a b c : ℝ) :
  ∃ (x y : ℝ), (x^2 + y^2 - 2*a*x + b^2 = 0 ∧ x^2 + y^2 - 2*c*y - b^2 = 0) →
  (a^2 - b^2) + (b^2 + c^2) = a^2 + c^2 :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_at_right_angle_l543_54398


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l543_54326

/-- Given a circle with area 49π, prove its radius is 7 -/
theorem circle_radius_from_area (A : ℝ) (r : ℝ) : 
  A = 49 * Real.pi → A = Real.pi * r^2 → r = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l543_54326


namespace NUMINAMATH_CALUDE_twentieth_15gonal_number_l543_54391

/-- The n-th k-gonal number -/
def N (n k : ℕ) : ℕ := ((k - 2) * n^2 + (4 - k) * n) / 2

/-- Theorem: The 20th 15-gonal number is 2490 -/
theorem twentieth_15gonal_number : N 20 15 = 2490 := by sorry

end NUMINAMATH_CALUDE_twentieth_15gonal_number_l543_54391


namespace NUMINAMATH_CALUDE_f_behavior_l543_54381

def f (x : ℝ) := 2 * x^3 - 7

theorem f_behavior :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → f x > M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < N → f x < M) :=
sorry

end NUMINAMATH_CALUDE_f_behavior_l543_54381


namespace NUMINAMATH_CALUDE_opposite_of_fraction_l543_54365

theorem opposite_of_fraction : 
  -(11 : ℚ) / 2022 = -(11 / 2022) := by sorry

end NUMINAMATH_CALUDE_opposite_of_fraction_l543_54365


namespace NUMINAMATH_CALUDE_acid_solution_volume_l543_54313

theorem acid_solution_volume (V : ℝ) : 
  (V > 0) →                              -- Initial volume is positive
  (0.2 * V - 4 + 20 = 0.4 * V) →         -- Equation representing the acid balance
  (V = 80) :=                            -- Conclusion: initial volume is 80 ml
by
  sorry

end NUMINAMATH_CALUDE_acid_solution_volume_l543_54313


namespace NUMINAMATH_CALUDE_translator_assignment_count_l543_54375

def total_translators : ℕ := 9
def english_only_translators : ℕ := 6
def korean_only_translators : ℕ := 2
def bilingual_translators : ℕ := 1
def groups_needing_korean : ℕ := 2
def groups_needing_english : ℕ := 3

def assignment_ways : ℕ := sorry

theorem translator_assignment_count : 
  assignment_ways = 900 := by sorry

end NUMINAMATH_CALUDE_translator_assignment_count_l543_54375
