import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l1889_188978

theorem equation_solution : 
  ∃ x : ℚ, (x^2 + x + 1) / (x + 1) = x + 3 ∧ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1889_188978


namespace NUMINAMATH_CALUDE_high_school_total_students_l1889_188946

/-- Represents a high school with three grades and stratified sampling -/
structure HighSchool where
  total_students : ℕ
  freshmen : ℕ
  sample_size : ℕ
  sampled_sophomores : ℕ
  sampled_seniors : ℕ

/-- The conditions of the problem -/
def problem_conditions (hs : HighSchool) : Prop :=
  hs.freshmen = 600 ∧
  hs.sample_size = 45 ∧
  hs.sampled_sophomores = 20 ∧
  hs.sampled_seniors = 10

/-- The theorem to prove -/
theorem high_school_total_students (hs : HighSchool) 
  (h : problem_conditions hs) : 
  hs.total_students = 1800 :=
sorry

end NUMINAMATH_CALUDE_high_school_total_students_l1889_188946


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1889_188927

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + a 5 = 14)
  (h_prod : a 2 * a 6 = 33) :
  a 1 * a 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1889_188927


namespace NUMINAMATH_CALUDE_parabola_through_points_point_not_on_parabola_l1889_188998

/-- A parabola of the form y = ax² + bx passing through (1, 3) and (-1, -1) -/
def Parabola (a b : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x

theorem parabola_through_points (a b : ℝ) :
  Parabola a b 1 = 3 ∧ Parabola a b (-1) = -1 → Parabola a b = λ x => x^2 + 2*x :=
sorry

theorem point_not_on_parabola :
  Parabola 1 2 2 ≠ 6 :=
sorry

end NUMINAMATH_CALUDE_parabola_through_points_point_not_on_parabola_l1889_188998


namespace NUMINAMATH_CALUDE_circular_cross_section_shapes_l1889_188936

-- Define the shapes
inductive Shape
  | Cone
  | Cylinder
  | Sphere
  | PentagonalPrism

-- Define a function to check if a shape can have a circular cross-section
def canHaveCircularCrossSection (s : Shape) : Prop :=
  match s with
  | Shape.Cone => true
  | Shape.Cylinder => true
  | Shape.Sphere => true
  | Shape.PentagonalPrism => false

-- Theorem statement
theorem circular_cross_section_shapes :
  ∀ s : Shape, canHaveCircularCrossSection s ↔ (s = Shape.Cone ∨ s = Shape.Cylinder ∨ s = Shape.Sphere) :=
by sorry

end NUMINAMATH_CALUDE_circular_cross_section_shapes_l1889_188936


namespace NUMINAMATH_CALUDE_one_minus_repeating_eight_eq_one_ninth_l1889_188915

/-- The value of 0.888... (repeating decimal) -/
def repeating_eight : ℚ := 8/9

/-- Proof that 1 - 0.888... = 1/9 -/
theorem one_minus_repeating_eight_eq_one_ninth :
  1 - repeating_eight = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_eight_eq_one_ninth_l1889_188915


namespace NUMINAMATH_CALUDE_euler_property_division_l1889_188969

theorem euler_property_division (x : ℝ) : 
  (x > 0) →
  (1/2 * x - 3000 + 1/3 * x - 1000 + 1/4 * x + 1/5 * x + 600 = x) →
  (x = 12000 ∧ 
   1/2 * x - 3000 = 3000 ∧
   1/3 * x - 1000 = 3000 ∧
   1/4 * x = 3000 ∧
   1/5 * x + 600 = 3000) :=
by sorry

end NUMINAMATH_CALUDE_euler_property_division_l1889_188969


namespace NUMINAMATH_CALUDE_hexagon_area_is_52_l1889_188991

-- Define the hexagon vertices
def hexagon_vertices : List (ℝ × ℝ) := [
  (0, 0), (2, 4), (5, 4), (7, 0), (5, -4), (2, -4)
]

-- Function to calculate the area of a trapezoid given its four vertices
def trapezoid_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ := sorry

-- Function to calculate the area of the hexagon
def hexagon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem stating that the area of the hexagon is 52 square units
theorem hexagon_area_is_52 : hexagon_area hexagon_vertices = 52 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_is_52_l1889_188991


namespace NUMINAMATH_CALUDE_angle_ADB_is_right_angle_l1889_188901

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Checks if a triangle is isosceles with two sides equal -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2

/-- Checks if a point lies on a circle -/
def Circle.contains (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Calculates the angle between three points -/
def angle (A B C : Point) : ℝ := sorry

/-- Main theorem -/
theorem angle_ADB_is_right_angle 
  (t : Triangle) 
  (c : Circle) 
  (D : Point) 
  (h1 : t.isIsosceles)
  (h2 : c.center = t.C)
  (h3 : c.radius = 15)
  (h4 : c.contains t.B)
  (h5 : ∃ k : ℝ, D.x = t.C.x + k * (t.C.x - t.A.x) ∧ D.y = t.C.y + k * (t.C.y - t.A.y))
  (h6 : c.contains D) :
  angle t.A D t.B = 90 := by sorry

end NUMINAMATH_CALUDE_angle_ADB_is_right_angle_l1889_188901


namespace NUMINAMATH_CALUDE_inequality_proof_l1889_188940

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (2 * x^2) / (y + z) + (2 * y^2) / (z + x) + (2 * z^2) / (x + y) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1889_188940


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1889_188988

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set of the quadratic inequality
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x < 0}

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = {x : ℝ | x < -4 ∨ x > 3}) :
  (a + b + c > 0) ∧
  ({x : ℝ | (a * x - b) / (a * x - c) ≤ 0} = {x : ℝ | -12 < x ∧ x ≤ 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1889_188988


namespace NUMINAMATH_CALUDE_ratio_problem_l1889_188967

theorem ratio_problem (a b c : ℚ) : 
  b / a = 4 → 
  b = 18 - 7 * a → 
  c = 2 * a - 6 → 
  a = 18 / 11 ∧ c = -30 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1889_188967


namespace NUMINAMATH_CALUDE_trig_expression_value_l1889_188917

/-- The value of the trigonometric expression is approximately 1.481 -/
theorem trig_expression_value : 
  let expr := (2 * Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
               Real.cos (160 * π / 180) * Real.cos (110 * π / 180)) /
              (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) + 
               Real.cos (156 * π / 180) * Real.cos (94 * π / 180))
  ∃ ε > 0, |expr - 1.481| < ε :=
by sorry

end NUMINAMATH_CALUDE_trig_expression_value_l1889_188917


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1889_188975

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^3 - 3*x^2 - 9*x + 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1889_188975


namespace NUMINAMATH_CALUDE_inverse_expression_equals_one_sixth_l1889_188935

theorem inverse_expression_equals_one_sixth :
  (2 + 4 * (4 - 3)⁻¹)⁻¹ = (1 : ℚ) / 6 := by sorry

end NUMINAMATH_CALUDE_inverse_expression_equals_one_sixth_l1889_188935


namespace NUMINAMATH_CALUDE_solve_equation_l1889_188904

theorem solve_equation : ∀ x : ℝ, 3 * x + 4 = x + 2 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1889_188904


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1889_188916

/-- Represents the simple interest calculation problem --/
theorem simple_interest_problem (P T : ℝ) (h1 : P = 2500) (h2 : T = 5) : 
  let SI := P - 2000
  let R := (SI * 100) / (P * T)
  R = 4 := by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1889_188916


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_equals_four_l1889_188948

theorem intersection_nonempty_implies_a_equals_four :
  ∀ (a : ℝ), 
  let A : Set ℝ := {3, 4, 2*a - 3}
  let B : Set ℝ := {a}
  (A ∩ B).Nonempty → a = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_equals_four_l1889_188948


namespace NUMINAMATH_CALUDE_sin_plus_cos_shift_l1889_188910

theorem sin_plus_cos_shift (x : ℝ) :
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.sin (3 * x + π / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_shift_l1889_188910


namespace NUMINAMATH_CALUDE_age_ratio_after_two_years_l1889_188929

/-- Proves that the ratio of a man's age to his student's age after two years is 2:1,
    given that the man is 26 years older than his 24-year-old student. -/
theorem age_ratio_after_two_years (student_age : ℕ) (man_age : ℕ) : 
  student_age = 24 →
  man_age = student_age + 26 →
  (man_age + 2) / (student_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_after_two_years_l1889_188929


namespace NUMINAMATH_CALUDE_train_length_l1889_188981

/-- The length of a train given its speed and time to cross a bridge -/
theorem train_length (speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  speed = 10 →
  bridge_length = 250 →
  crossing_time = 34.997200223982084 →
  ∃ train_length : ℝ, 
    train_length + bridge_length = speed * crossing_time ∧ 
    abs (train_length - 99.97) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1889_188981


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1889_188989

theorem triangle_angle_measure (D E F : ℝ) :
  D + E + F = 180 →  -- Sum of angles in a triangle
  F = D + 40 →       -- Angle F is 40 degrees more than angle D
  E = 2 * D →        -- Angle E is twice the measure of angle D
  F = 75 :=          -- Measure of angle F is 75 degrees
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1889_188989


namespace NUMINAMATH_CALUDE_output_after_five_years_l1889_188955

/-- The output value after n years of growth at a given rate -/
def output_after_n_years (initial_value : ℝ) (growth_rate : ℝ) (n : ℕ) : ℝ :=
  initial_value * (1 + growth_rate) ^ n

/-- Theorem: The output value after 5 years with 10% annual growth -/
theorem output_after_five_years (a : ℝ) :
  output_after_n_years a 0.1 5 = 1.1^5 * a := by
  sorry

end NUMINAMATH_CALUDE_output_after_five_years_l1889_188955


namespace NUMINAMATH_CALUDE_negation_of_universal_quadratic_inequality_l1889_188911

theorem negation_of_universal_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quadratic_inequality_l1889_188911


namespace NUMINAMATH_CALUDE_hash_2_3_2_1_l1889_188931

def hash (a b c d : ℝ) : ℝ := b^2 - 4*a*c*d

theorem hash_2_3_2_1 : hash 2 3 2 1 = -7 := by
  sorry

end NUMINAMATH_CALUDE_hash_2_3_2_1_l1889_188931


namespace NUMINAMATH_CALUDE_stratified_sampling_middle_aged_l1889_188963

theorem stratified_sampling_middle_aged (total_teachers : ℕ) (middle_aged : ℕ) (sample_size : ℕ)
  (h1 : total_teachers = 480)
  (h2 : middle_aged = 160)
  (h3 : sample_size = 60) :
  (middle_aged : ℚ) / total_teachers * sample_size = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_middle_aged_l1889_188963


namespace NUMINAMATH_CALUDE_farmer_chicken_sales_l1889_188944

def duck_price : ℕ := 10
def chicken_price : ℕ := 8
def ducks_sold : ℕ := 2

theorem farmer_chicken_sales : 
  ∃ (chickens_sold : ℕ),
    (duck_price * ducks_sold + chicken_price * chickens_sold) / 2 = 30 ∧
    chickens_sold = 5 := by
  sorry

end NUMINAMATH_CALUDE_farmer_chicken_sales_l1889_188944


namespace NUMINAMATH_CALUDE_cafe_chairs_distribution_l1889_188902

theorem cafe_chairs_distribution (indoor_tables outdoor_tables : ℕ) 
  (chairs_per_indoor_table : ℕ) (total_chairs : ℕ) : 
  indoor_tables = 9 → 
  outdoor_tables = 11 → 
  chairs_per_indoor_table = 10 → 
  total_chairs = 123 → 
  (total_chairs - indoor_tables * chairs_per_indoor_table) / outdoor_tables = 3 := by
sorry

end NUMINAMATH_CALUDE_cafe_chairs_distribution_l1889_188902


namespace NUMINAMATH_CALUDE_product_difference_of_squares_l1889_188965

theorem product_difference_of_squares (m n : ℝ) : 
  m * n = ((m + n)/2)^2 - ((m - n)/2)^2 := by sorry

end NUMINAMATH_CALUDE_product_difference_of_squares_l1889_188965


namespace NUMINAMATH_CALUDE_shaded_quadrilateral_area_l1889_188997

theorem shaded_quadrilateral_area (s : ℝ) (a b : ℝ) : 
  s = 20 → a = 15 → b = 20 →
  s^2 - (1/2 * a * b) - (1/2 * (s * b / (a^2 + b^2).sqrt) * (s * a / (a^2 + b^2).sqrt)) = 154 :=
by sorry

end NUMINAMATH_CALUDE_shaded_quadrilateral_area_l1889_188997


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1889_188912

/-- Parabola C: y² = 4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Line: y = 2x - 4 -/
def line (x y : ℝ) : Prop := y = 2*x - 4

/-- Point A is on both the parabola and the line -/
def point_A (x y : ℝ) : Prop := parabola_C x y ∧ line x y

/-- Point B is on both the parabola and the line, and is distinct from A -/
def point_B (x y : ℝ) : Prop := parabola_C x y ∧ line x y ∧ (x, y) ≠ (x_A, y_A)
  where
  x_A : ℝ := sorry
  y_A : ℝ := sorry

/-- Point P is on the parabola C -/
def point_P (x y : ℝ) : Prop := parabola_C x y

/-- The area of triangle ABP is 12 -/
def triangle_area (x_A y_A x_B y_B x_P y_P : ℝ) : Prop :=
  abs ((x_A - x_P) * (y_B - y_P) - (x_B - x_P) * (y_A - y_P)) / 2 = 12

/-- The main theorem -/
theorem parabola_line_intersection
  (x_A y_A x_B y_B x_P y_P : ℝ)
  (hA : point_A x_A y_A)
  (hB : point_B x_B y_B)
  (hP : point_P x_P y_P)
  (hArea : triangle_area x_A y_A x_B y_B x_P y_P) :
  (((x_B - x_A)^2 + (y_B - y_A)^2)^(1/2 : ℝ) = 3 * 5^(1/2 : ℝ)) ∧
  ((x_P = 9 ∧ y_P = 6) ∨ (x_P = 4 ∧ y_P = -4)) :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1889_188912


namespace NUMINAMATH_CALUDE_farm_problem_l1889_188943

theorem farm_problem :
  ∃ (l g : ℕ), l > 0 ∧ g > 0 ∧ 30 * l + 32 * g = 1200 ∧ l > g :=
by sorry

end NUMINAMATH_CALUDE_farm_problem_l1889_188943


namespace NUMINAMATH_CALUDE_special_function_value_l1889_188907

/-- A function satisfying the given property -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y

theorem special_function_value (f : ℝ → ℝ) 
  (h : SpecialFunction f) (h250 : f 250 = 4) : f 300 = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l1889_188907


namespace NUMINAMATH_CALUDE_stock_transaction_profit_l1889_188923

/-- Represents a stock transaction and calculates the profit -/
def stock_transaction (initial_shares : ℕ) (initial_price : ℚ) (sold_shares : ℕ) (selling_price : ℚ) : ℚ :=
  let initial_cost := initial_shares * initial_price
  let sale_revenue := sold_shares * selling_price
  let remaining_shares := initial_shares - sold_shares
  let final_value := sale_revenue + (remaining_shares * (2 * initial_price))
  final_value - initial_cost

/-- Proves that the profit from the given stock transaction is $40 -/
theorem stock_transaction_profit :
  stock_transaction 20 3 10 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_stock_transaction_profit_l1889_188923


namespace NUMINAMATH_CALUDE_red_to_blue_ratio_l1889_188914

/-- Represents the number of marbles of each color in Cara's bag. -/
structure MarbleCounts where
  total : ℕ
  yellow : ℕ
  green : ℕ
  red : ℕ
  blue : ℕ

/-- Theorem stating the ratio of red to blue marbles given the conditions -/
theorem red_to_blue_ratio (m : MarbleCounts) : 
  m.total = 60 ∧ 
  m.yellow = 20 ∧ 
  m.green = m.yellow / 2 ∧ 
  m.total = m.yellow + m.green + m.red + m.blue ∧ 
  m.blue = m.total / 4 →
  m.red / m.blue = 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_red_to_blue_ratio_l1889_188914


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l1889_188968

theorem fahrenheit_to_celsius (C F : ℝ) : C = (5 / 9) * (F - 32) → C = 40 → F = 104 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l1889_188968


namespace NUMINAMATH_CALUDE_percent_composition_l1889_188951

theorem percent_composition (y : ℝ) : (18 / 100) * y = (30 / 100) * ((60 / 100) * y) := by
  sorry

end NUMINAMATH_CALUDE_percent_composition_l1889_188951


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1889_188987

theorem sum_of_x_and_y (x y : ℚ) 
  (h1 : 2 / x + 3 / y = 4) 
  (h2 : 2 / x - 3 / y = -2) : 
  x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1889_188987


namespace NUMINAMATH_CALUDE_toothpicks_for_1002_base_l1889_188945

/-- Calculates the number of toothpicks required for a large equilateral triangle
    constructed with rows of small equilateral triangles. -/
def toothpicks_count (base_triangles : ℕ) : ℕ :=
  let total_triangles := base_triangles * (base_triangles + 1) / 2
  let total_sides := 3 * total_triangles
  let boundary_sides := 3 * base_triangles
  (total_sides - boundary_sides) / 2 + boundary_sides

/-- Theorem stating that for a large equilateral triangle with 1002 small triangles
    in its base, the total number of toothpicks required is 752253. -/
theorem toothpicks_for_1002_base : toothpicks_count 1002 = 752253 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_for_1002_base_l1889_188945


namespace NUMINAMATH_CALUDE_total_money_raised_is_correct_l1889_188979

/-- Represents the total amount of money raised in a two-month period of telethons --/
def total_money_raised : ℝ :=
  let friday_rate_first_12h := 4000
  let friday_rate_last_14h := friday_rate_first_12h * 1.1
  let saturday_rate_first_12h := 5000
  let saturday_rate_last_14h := saturday_rate_first_12h * 1.2
  let sunday_initial_rate := saturday_rate_first_12h * 0.85
  let sunday_rate_5_percent_increase := sunday_initial_rate * 1.05
  let sunday_rate_30_percent_increase := sunday_initial_rate * 1.3
  let sunday_rate_10_percent_decrease := sunday_initial_rate * 0.9
  let sunday_rate_20_percent_increase := sunday_initial_rate * 1.2
  let sunday_rate_25_percent_decrease := sunday_initial_rate * 0.75

  let friday_total := friday_rate_first_12h * 12 + friday_rate_last_14h * 14
  let saturday_total := saturday_rate_first_12h * 12 + saturday_rate_last_14h * 14
  let sunday_total := sunday_initial_rate * 10 + sunday_rate_5_percent_increase * 2 +
                      sunday_rate_30_percent_increase * 4 + sunday_rate_10_percent_decrease * 2 +
                      sunday_rate_20_percent_increase * 1 + sunday_rate_25_percent_decrease * 7

  let weekend_total := friday_total + saturday_total + sunday_total
  weekend_total * 8

/-- The theorem states that the total money raised in the two-month period is $2,849,500 --/
theorem total_money_raised_is_correct : total_money_raised = 2849500 := by
  sorry

end NUMINAMATH_CALUDE_total_money_raised_is_correct_l1889_188979


namespace NUMINAMATH_CALUDE_smallest_binary_divisible_by_225_proof_l1889_188995

/-- A function that checks if a natural number only contains digits 0 and 1 in base 10 -/
def only_zero_one_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The smallest natural number with only 0 and 1 digits divisible by 225 -/
def smallest_binary_divisible_by_225 : ℕ := 11111111100

theorem smallest_binary_divisible_by_225_proof :
  (smallest_binary_divisible_by_225 % 225 = 0) ∧
  only_zero_one_digits smallest_binary_divisible_by_225 ∧
  ∀ n : ℕ, n < smallest_binary_divisible_by_225 →
    ¬(n % 225 = 0 ∧ only_zero_one_digits n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_binary_divisible_by_225_proof_l1889_188995


namespace NUMINAMATH_CALUDE_value_of_m_l1889_188986

theorem value_of_m (m : ℕ) : 
  (((1 : ℚ) ^ m) / (5 ^ m)) * (((1 : ℚ) ^ 16) / (4 ^ 16)) = 1 / (2 * (10 ^ 31)) → 
  m = 31 := by
sorry

end NUMINAMATH_CALUDE_value_of_m_l1889_188986


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l1889_188971

/-- The number of small boxes in the large box -/
def small_boxes : ℕ := 15

/-- The number of chocolate bars in each small box -/
def bars_per_box : ℕ := 25

/-- The total number of chocolate bars in the large box -/
def total_bars : ℕ := small_boxes * bars_per_box

theorem chocolate_bar_count : total_bars = 375 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_count_l1889_188971


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_l1889_188918

def numbers : List Nat := [18, 24, 36]

theorem gcf_lcm_sum (A B : Nat) 
  (h1 : A = Nat.gcd 18 (Nat.gcd 24 36))
  (h2 : B = Nat.lcm 18 (Nat.lcm 24 36)) : 
  A + B = 78 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_l1889_188918


namespace NUMINAMATH_CALUDE_parallelogram_point_D_l1889_188905

/-- A parallelogram in the complex plane -/
structure ComplexParallelogram where
  A : ℂ
  B : ℂ
  C : ℂ
  D : ℂ
  is_parallelogram : (B - A) = (C - D)

/-- Theorem: Given a parallelogram ABCD in the complex plane with A = 1+3i, B = 2-i, and C = -3+i, then D = -4+5i -/
theorem parallelogram_point_D (ABCD : ComplexParallelogram) 
  (hA : ABCD.A = 1 + 3*I)
  (hB : ABCD.B = 2 - I)
  (hC : ABCD.C = -3 + I) :
  ABCD.D = -4 + 5*I :=
sorry

end NUMINAMATH_CALUDE_parallelogram_point_D_l1889_188905


namespace NUMINAMATH_CALUDE_triangle_transformation_exists_l1889_188984

-- Define a point in the 2D plane
structure Point :=
  (x : Int) (y : Int)

-- Define a triangle as a set of three points
structure Triangle :=
  (a : Point) (b : Point) (c : Point)

-- Define the 90° counterclockwise rotation transformation
def rotate90 (center : Point) (p : Point) : Point :=
  let dx := p.x - center.x
  let dy := p.y - center.y
  Point.mk (center.x - dy) (center.y + dx)

-- Define the initial and target triangles
def initialTriangle : Triangle :=
  Triangle.mk (Point.mk 0 0) (Point.mk 1 0) (Point.mk 0 1)

def targetTriangle : Triangle :=
  Triangle.mk (Point.mk 0 0) (Point.mk 1 0) (Point.mk 1 1)

-- Theorem statement
theorem triangle_transformation_exists :
  ∃ (rotationCenter : Point),
    rotate90 rotationCenter initialTriangle.a = targetTriangle.a ∧
    rotate90 rotationCenter initialTriangle.b = targetTriangle.b ∧
    rotate90 rotationCenter initialTriangle.c = targetTriangle.c :=
by sorry

end NUMINAMATH_CALUDE_triangle_transformation_exists_l1889_188984


namespace NUMINAMATH_CALUDE_matrix_product_is_zero_l1889_188964

def A (k a c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, k * d, -k * c],
    ![-k * d, 0, k * a],
    ![k * c, -k * a, 0]]

def B (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![d^2, d * e, d * f],
    ![d * e, e^2, e * f],
    ![d * f, e * f, f^2]]

theorem matrix_product_is_zero (k a c d e f : ℝ) :
  A k a c d * B d e f = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_is_zero_l1889_188964


namespace NUMINAMATH_CALUDE_cow_herd_division_l1889_188903

theorem cow_herd_division (total : ℕ) : 
  (2 : ℚ) / 3 * total + (1 : ℚ) / 6 * total + (1 : ℚ) / 9 * total + 6 = total → 
  total = 108 := by
sorry

end NUMINAMATH_CALUDE_cow_herd_division_l1889_188903


namespace NUMINAMATH_CALUDE_second_car_distance_l1889_188961

-- Define the initial distance between the cars
def initial_distance : ℝ := 150

-- Define the final distance between the cars
def final_distance : ℝ := 65

-- Theorem to prove
theorem second_car_distance :
  ∃ (x : ℝ), x ≥ 0 ∧ initial_distance - x = final_distance ∧ x = 85 :=
by sorry

end NUMINAMATH_CALUDE_second_car_distance_l1889_188961


namespace NUMINAMATH_CALUDE_andy_tomato_plants_l1889_188928

theorem andy_tomato_plants :
  ∀ (P : ℕ),
  (∃ (total_tomatoes dried_tomatoes sauce_tomatoes remaining_tomatoes : ℕ),
    total_tomatoes = 7 * P ∧
    dried_tomatoes = total_tomatoes / 2 ∧
    sauce_tomatoes = (total_tomatoes - dried_tomatoes) / 3 ∧
    remaining_tomatoes = total_tomatoes - dried_tomatoes - sauce_tomatoes ∧
    remaining_tomatoes = 42) →
  P = 18 :=
by sorry

end NUMINAMATH_CALUDE_andy_tomato_plants_l1889_188928


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1889_188953

/-- Ellipse C with equation (x^2 / 4) + (y^2 / m) = 1 -/
def ellipse_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / m = 1

/-- Point P on the x-axis -/
def point_P : ℝ × ℝ := (-1, 0)

/-- Line l passing through point P -/
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + 1)

/-- Condition for circle with AB as diameter passing through origin -/
def circle_condition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

/-- Main theorem: If there exists a line l intersecting ellipse C at points A and B
    such that the circle with AB as diameter passes through the origin,
    then m is in the range (0, 4/3] -/
theorem ellipse_intersection_theorem (m : ℝ) :
  (m > 0) →
  (∃ (k : ℝ) (A B : ℝ × ℝ),
    ellipse_C m A.1 A.2 ∧
    ellipse_C m B.1 B.2 ∧
    line_l k A.1 A.2 ∧
    line_l k B.1 B.2 ∧
    circle_condition A B) →
  0 < m ∧ m ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1889_188953


namespace NUMINAMATH_CALUDE_theater_seats_l1889_188906

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increment : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increment + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- Theorem stating that a theater with given conditions has 570 seats -/
theorem theater_seats :
  ∃ (t : Theater), t.first_row_seats = 12 ∧ t.seat_increment = 2 ∧ t.last_row_seats = 48 ∧ total_seats t = 570 :=
by
  sorry


end NUMINAMATH_CALUDE_theater_seats_l1889_188906


namespace NUMINAMATH_CALUDE_g_composition_equals_49_l1889_188920

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2*n + 3

theorem g_composition_equals_49 : g (g (g 3)) = 49 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_equals_49_l1889_188920


namespace NUMINAMATH_CALUDE_average_of_xyz_is_one_l1889_188985

theorem average_of_xyz_is_one (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_prod : x * y * z = 1)
  (h_sum : x + y + z = 1/x + 1/y + 1/z) :
  (x + y + z) / 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_average_of_xyz_is_one_l1889_188985


namespace NUMINAMATH_CALUDE_inequality_proof_l1889_188950

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a * b + b * c + c * a = 3) :
  (b * c / (1 + a^4)) + (c * a / (1 + b^4)) + (a * b / (1 + c^4)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1889_188950


namespace NUMINAMATH_CALUDE_equation_solution_l1889_188909

theorem equation_solution (x : ℝ) (h1 : 0 < x) (h2 : x < 12) (h3 : x ≠ 1) :
  (1 + 2 * Real.log 2 / Real.log 9) / (Real.log x / Real.log 9) - 1 = 
  2 * (Real.log 3 / Real.log x) * (Real.log (12 - x) / Real.log 9) → x = 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1889_188909


namespace NUMINAMATH_CALUDE_pig_farm_fence_length_l1889_188972

theorem pig_farm_fence_length 
  (area : ℝ) 
  (short_side : ℝ) 
  (long_side : ℝ) :
  area = 1250 ∧ 
  long_side = 2 * short_side ∧ 
  area = long_side * short_side →
  short_side + short_side + long_side = 100 := by
sorry

end NUMINAMATH_CALUDE_pig_farm_fence_length_l1889_188972


namespace NUMINAMATH_CALUDE_consecutive_integers_product_divisibility_l1889_188994

theorem consecutive_integers_product_divisibility
  (m n : ℕ) (h : m < n) :
  ∀ (a : ℕ), ∃ (i j : ℕ), i ≠ j ∧ i < n ∧ j < n ∧ (mn ∣ (a + i) * (a + j)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_divisibility_l1889_188994


namespace NUMINAMATH_CALUDE_min_value_expression_l1889_188952

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  let A := (Real.sqrt (a^6 + b^4 * c^6)) / b + 
           (Real.sqrt (b^6 + c^4 * a^6)) / c + 
           (Real.sqrt (c^6 + a^4 * b^6)) / a
  A ≥ 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1889_188952


namespace NUMINAMATH_CALUDE_divisible_by_five_not_ending_in_five_l1889_188996

theorem divisible_by_five_not_ending_in_five : ∃ n : ℕ, 5 ∣ n ∧ n % 10 ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_not_ending_in_five_l1889_188996


namespace NUMINAMATH_CALUDE_accuracy_of_rounded_number_l1889_188938

def is_accurate_to_hundreds_place (n : ℕ) : Prop :=
  n % 1000 ≠ 0 ∧ n % 100 = 0

theorem accuracy_of_rounded_number :
  ∀ (n : ℕ), 
    (31500 ≤ n ∧ n < 32500) →
    is_accurate_to_hundreds_place n :=
by
  sorry

end NUMINAMATH_CALUDE_accuracy_of_rounded_number_l1889_188938


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_when_a_is_2_complement_A_union_B_equals_reals_iff_l1889_188925

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1) + Real.sqrt (2 - x)}
def B (a : ℝ) : Set ℝ := {y | ∃ x ≥ a, y = 2^x}

-- Define the complement of A
def complementA : Set ℝ := {x | x ∉ A}

-- Statement I
theorem complement_A_intersect_B_when_a_is_2 :
  (complementA ∩ B 2) = {x | x ≥ 4} := by sorry

-- Statement II
theorem complement_A_union_B_equals_reals_iff (a : ℝ) :
  (complementA ∪ B a) = Set.univ ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_when_a_is_2_complement_A_union_B_equals_reals_iff_l1889_188925


namespace NUMINAMATH_CALUDE_function_inequality_l1889_188992

-- Define the interval (3,7)
def openInterval : Set ℝ := {x : ℝ | 3 < x ∧ x < 7}

-- Define the theorem
theorem function_inequality
  (f g : ℝ → ℝ)
  (h_diff_f : DifferentiableOn ℝ f openInterval)
  (h_diff_g : DifferentiableOn ℝ g openInterval)
  (h_deriv : ∀ x ∈ openInterval, deriv f x < deriv g x) :
  ∀ x ∈ openInterval, f x + g 3 < g x + f 3 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l1889_188992


namespace NUMINAMATH_CALUDE_dog_bones_problem_l1889_188957

theorem dog_bones_problem (initial_bones : ℕ) : 
  initial_bones + 8 = 23 → initial_bones = 15 := by
  sorry

end NUMINAMATH_CALUDE_dog_bones_problem_l1889_188957


namespace NUMINAMATH_CALUDE_noahs_age_ratio_l1889_188913

theorem noahs_age_ratio (joe_age : ℕ) (noah_future_age : ℕ) (years_to_future : ℕ) :
  joe_age = 6 →
  noah_future_age = 22 →
  years_to_future = 10 →
  ∃ k : ℕ, k * joe_age = noah_future_age - years_to_future →
  (noah_future_age - years_to_future) / joe_age = 2 := by
sorry

end NUMINAMATH_CALUDE_noahs_age_ratio_l1889_188913


namespace NUMINAMATH_CALUDE_path_construction_cost_l1889_188973

/-- Given a rectangular grass field with surrounding path, calculate the total cost of constructing the path -/
theorem path_construction_cost 
  (field_length : ℝ) 
  (field_width : ℝ) 
  (long_side_path_width : ℝ) 
  (short_side1_path_width : ℝ) 
  (short_side2_path_width : ℝ) 
  (long_side_cost_per_sqm : ℝ) 
  (short_side1_cost_per_sqm : ℝ) 
  (short_side2_cost_per_sqm : ℝ) 
  (h1 : field_length = 75) 
  (h2 : field_width = 55) 
  (h3 : long_side_path_width = 2.5) 
  (h4 : short_side1_path_width = 3) 
  (h5 : short_side2_path_width = 4) 
  (h6 : long_side_cost_per_sqm = 7) 
  (h7 : short_side1_cost_per_sqm = 9) 
  (h8 : short_side2_cost_per_sqm = 12) :
  let long_sides_area := 2 * field_length * long_side_path_width
  let short_side1_area := field_width * short_side1_path_width
  let short_side2_area := field_width * short_side2_path_width
  let long_sides_cost := long_sides_area * long_side_cost_per_sqm
  let short_side1_cost := short_side1_area * short_side1_cost_per_sqm
  let short_side2_cost := short_side2_area * short_side2_cost_per_sqm
  let total_cost := long_sides_cost + short_side1_cost + short_side2_cost
  total_cost = 6750 := by sorry


end NUMINAMATH_CALUDE_path_construction_cost_l1889_188973


namespace NUMINAMATH_CALUDE_jackson_souvenirs_count_l1889_188933

theorem jackson_souvenirs_count :
  let hermit_crabs : ℕ := 120
  let shells_per_crab : ℕ := 8
  let starfish_per_shell : ℕ := 5
  let sand_dollars_per_starfish : ℕ := 3
  let sand_dollars_per_coral : ℕ := 4

  let total_shells : ℕ := hermit_crabs * shells_per_crab
  let total_starfish : ℕ := total_shells * starfish_per_shell
  let total_sand_dollars : ℕ := total_starfish * sand_dollars_per_starfish
  let total_coral : ℕ := total_sand_dollars / sand_dollars_per_coral

  let total_souvenirs : ℕ := hermit_crabs + total_shells + total_starfish + total_sand_dollars + total_coral

  total_souvenirs = 22880 :=
by
  sorry

end NUMINAMATH_CALUDE_jackson_souvenirs_count_l1889_188933


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1889_188921

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, x^2 + (1 - a) * x + 1 < 0) → a ∈ Set.Ioi 3 ∪ Set.Iio (-1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1889_188921


namespace NUMINAMATH_CALUDE_line_parabola_intersection_count_l1889_188982

theorem line_parabola_intersection_count : 
  ∃! (s : Finset ℝ), 
    (∀ a ∈ s, ∃ x y : ℝ, 
      y = 2*x + a + 1 ∧ 
      y = x^2 + (a+1)^2 ∧ 
      ∀ x' : ℝ, x'^2 + (a+1)^2 ≥ x^2 + (a+1)^2) ∧
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_count_l1889_188982


namespace NUMINAMATH_CALUDE_num_quadrilaterals_is_495_l1889_188960

/-- The number of ways to choose 4 points from 12 distinct points on a circle's circumference to form convex quadrilaterals -/
def num_quadrilaterals : ℕ := Nat.choose 12 4

/-- Theorem stating that the number of quadrilaterals is 495 -/
theorem num_quadrilaterals_is_495 : num_quadrilaterals = 495 := by
  sorry

end NUMINAMATH_CALUDE_num_quadrilaterals_is_495_l1889_188960


namespace NUMINAMATH_CALUDE_impossibility_proof_l1889_188956

def Square := Fin 4 → ℕ

def initial_state : Square := fun i => if i = 0 then 1 else 0

def S (state : Square) : ℤ :=
  state 0 - state 1 + state 2 - state 3

def is_valid_move (before after : Square) : Prop :=
  ∃ (i : Fin 4) (k : ℕ), 
    after i + k = before i ∧
    after ((i + 1) % 4) = before ((i + 1) % 4) + k ∧
    after ((i + 3) % 4) = before ((i + 3) % 4) + k ∧
    (∀ j, j ≠ i ∧ j ≠ (i + 1) % 4 ∧ j ≠ (i + 3) % 4 → after j = before j)

def reachable (start goal : Square) : Prop :=
  ∃ (n : ℕ) (path : Fin (n + 1) → Square),
    path 0 = start ∧
    path n = goal ∧
    ∀ i : Fin n, is_valid_move (path i) (path (i + 1))

def target_state : Square := fun i => 
  if i = 0 then 1
  else if i = 1 then 9
  else if i = 2 then 8
  else 9

theorem impossibility_proof :
  ¬(reachable initial_state target_state) :=
sorry

end NUMINAMATH_CALUDE_impossibility_proof_l1889_188956


namespace NUMINAMATH_CALUDE_expression_equals_25_l1889_188999

theorem expression_equals_25 (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 1) :
  x + y = 25 := by sorry

end NUMINAMATH_CALUDE_expression_equals_25_l1889_188999


namespace NUMINAMATH_CALUDE_power_product_equality_l1889_188990

theorem power_product_equality (a b : ℝ) : (-a * b)^3 * (-3 * b)^2 = -9 * a^3 * b^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1889_188990


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_meaningful_l1889_188941

theorem sqrt_x_plus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_meaningful_l1889_188941


namespace NUMINAMATH_CALUDE_parallel_segments_k_value_l1889_188974

/-- Given four points A(-3, 0), B(0, -3), X(0, 9), and Y(18, k) on a Cartesian plane,
    if segment AB is parallel to segment XY, then k = -9. -/
theorem parallel_segments_k_value (k : ℝ) : 
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (0, -3)
  let X : ℝ × ℝ := (0, 9)
  let Y : ℝ × ℝ := (18, k)
  (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1) →
  k = -9 := by
sorry

end NUMINAMATH_CALUDE_parallel_segments_k_value_l1889_188974


namespace NUMINAMATH_CALUDE_min_value_of_f_l1889_188993

theorem min_value_of_f (a₁ a₂ a₃ a₄ : ℝ) 
  (pos₁ : 0 < a₁) (pos₂ : 0 < a₂) (pos₃ : 0 < a₃) (pos₄ : 0 < a₄)
  (sum_cond : a₁ + 2*a₂ + 3*a₃ + 4*a₄ ≤ 10)
  (lower_bound₁ : a₁ ≥ 1/8) (lower_bound₂ : a₂ ≥ 1/4)
  (lower_bound₃ : a₃ ≥ 1/2) (lower_bound₄ : a₄ ≥ 1) : 
  1/(1 + a₁) + 1/(1 + a₂^2) + 1/(1 + a₃^3) + 1/(1 + a₄^4) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1889_188993


namespace NUMINAMATH_CALUDE_common_divisors_of_36_and_60_l1889_188970

/-- The number of positive integers that are divisors of both 36 and 60 -/
def common_divisors_count : ℕ := 
  (Finset.filter (fun d => 36 % d = 0 ∧ 60 % d = 0) (Finset.range 61)).card

/-- Theorem stating that the number of common divisors of 36 and 60 is 6 -/
theorem common_divisors_of_36_and_60 : common_divisors_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_of_36_and_60_l1889_188970


namespace NUMINAMATH_CALUDE_factor_quadratic_l1889_188942

theorem factor_quadratic (a : ℝ) : a^2 - 2*a - 15 = (a + 3)*(a - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_quadratic_l1889_188942


namespace NUMINAMATH_CALUDE_renovation_sand_required_l1889_188959

/-- The amount of sand required for a renovation project -/
theorem renovation_sand_required (total_material dirt cement : ℝ) 
  (h_total : total_material = 0.67)
  (h_dirt : dirt = 0.33)
  (h_cement : cement = 0.17) :
  total_material - dirt - cement = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_renovation_sand_required_l1889_188959


namespace NUMINAMATH_CALUDE_blanket_collection_proof_l1889_188924

/-- Calculates the total number of blankets collected over three days -/
def totalBlankets (teamSize : ℕ) (firstDayPerPerson : ℕ) (secondDayMultiplier : ℕ) (thirdDayTotal : ℕ) : ℕ :=
  let firstDay := teamSize * firstDayPerPerson
  let secondDay := firstDay * secondDayMultiplier
  firstDay + secondDay + thirdDayTotal

/-- Proves that the total number of blankets collected is 142 given the specific conditions -/
theorem blanket_collection_proof :
  totalBlankets 15 2 3 22 = 142 := by
  sorry

end NUMINAMATH_CALUDE_blanket_collection_proof_l1889_188924


namespace NUMINAMATH_CALUDE_jack_morning_emails_l1889_188939

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 10

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 7

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 17

/-- Theorem stating that Jack received 10 emails in the morning -/
theorem jack_morning_emails : 
  morning_emails = afternoon_emails + 3 := by sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l1889_188939


namespace NUMINAMATH_CALUDE_permutations_minus_combinations_l1889_188980

/-- The number of r-permutations from n elements -/
def permutations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of r-combinations from n elements -/
def combinations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem permutations_minus_combinations : permutations 7 3 - combinations 6 4 = 195 := by
  sorry

end NUMINAMATH_CALUDE_permutations_minus_combinations_l1889_188980


namespace NUMINAMATH_CALUDE_oil_measurement_l1889_188983

/-- The total amount of oil in a measuring cup after adding more -/
theorem oil_measurement (initial : ℚ) (additional : ℚ) : 
  initial = 0.16666666666666666 →
  additional = 0.6666666666666666 →
  initial + additional = 0.8333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_oil_measurement_l1889_188983


namespace NUMINAMATH_CALUDE_dividend_calculation_l1889_188937

theorem dividend_calculation (divisor quotient remainder dividend : ℕ) : 
  divisor = 3 →
  quotient = 7 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  dividend = 23 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1889_188937


namespace NUMINAMATH_CALUDE_car_braking_distance_l1889_188932

def braking_sequence (n : ℕ) : ℕ :=
  max (50 - 10 * n) 0

def total_distance (n : ℕ) : ℕ :=
  (List.range n).map braking_sequence |>.sum

theorem car_braking_distance :
  ∃ n : ℕ, total_distance n = 150 ∧ braking_sequence n = 0 :=
sorry

end NUMINAMATH_CALUDE_car_braking_distance_l1889_188932


namespace NUMINAMATH_CALUDE_f_min_value_l1889_188930

/-- The function f(x) = x^2 + 26x + 7 -/
def f (x : ℝ) : ℝ := x^2 + 26*x + 7

/-- The minimum value of f(x) is -162 -/
theorem f_min_value : ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = -162 := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l1889_188930


namespace NUMINAMATH_CALUDE_train_passes_jogger_l1889_188966

/-- The time it takes for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed train_speed : ℝ) (train_length initial_distance : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  initial_distance = 230 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 35 :=
by sorry

end NUMINAMATH_CALUDE_train_passes_jogger_l1889_188966


namespace NUMINAMATH_CALUDE_solve_candy_problem_l1889_188922

def candy_problem (initial_candies : ℕ) (friend_multiplier : ℕ) (friend_eaten : ℕ) : Prop :=
  let friend_brought := initial_candies * friend_multiplier
  let total_candies := initial_candies + friend_brought
  let each_share := total_candies / 2
  let friend_final := each_share - friend_eaten
  friend_final = 65

theorem solve_candy_problem :
  candy_problem 50 2 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l1889_188922


namespace NUMINAMATH_CALUDE_integer_count_inequality_l1889_188958

theorem integer_count_inequality (x : ℤ) : 
  (Finset.filter (fun i => (i - 1)^2 ≤ 9) (Finset.range 7)).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_integer_count_inequality_l1889_188958


namespace NUMINAMATH_CALUDE_people_counting_l1889_188926

theorem people_counting (first_day second_day : ℕ) : 
  first_day = 2 * second_day →
  first_day + second_day = 1500 →
  second_day = 500 := by
sorry

end NUMINAMATH_CALUDE_people_counting_l1889_188926


namespace NUMINAMATH_CALUDE_polynomial_factor_theorem_l1889_188934

theorem polynomial_factor_theorem (a : ℝ) : 
  (∃ b : ℝ, ∀ y : ℝ, y^2 + 3*y - a = (y - 3) * (y + b)) → a = 18 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_theorem_l1889_188934


namespace NUMINAMATH_CALUDE_roof_metal_bars_l1889_188954

/-- The number of sets of metal bars needed for the roof -/
def num_sets : ℕ := 2

/-- The number of metal bars in each set -/
def bars_per_set : ℕ := 7

/-- The total number of metal bars needed for the roof -/
def total_bars : ℕ := num_sets * bars_per_set

theorem roof_metal_bars : total_bars = 14 := by
  sorry

end NUMINAMATH_CALUDE_roof_metal_bars_l1889_188954


namespace NUMINAMATH_CALUDE_triangle_with_ratio_1_2_3_is_right_triangle_l1889_188949

-- Define a triangle type
structure Triangle :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)

-- Define the properties of a valid triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.angle1 > 0 ∧ t.angle2 > 0 ∧ t.angle3 > 0 ∧ t.angle1 + t.angle2 + t.angle3 = 180

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

-- Define a triangle with angles in the ratio 1:2:3
def triangle_with_ratio_1_2_3 (t : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t.angle1 = k ∧ t.angle2 = 2*k ∧ t.angle3 = 3*k

-- Theorem statement
theorem triangle_with_ratio_1_2_3_is_right_triangle (t : Triangle) :
  is_valid_triangle t → triangle_with_ratio_1_2_3 t → is_right_triangle t :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_ratio_1_2_3_is_right_triangle_l1889_188949


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l1889_188976

theorem divisibility_of_sum_of_squares (p x y z : ℕ) : 
  Prime p → 
  0 < x → x < y → y < z → z < p → 
  x^3 % p = y^3 % p → y^3 % p = z^3 % p → 
  (x + y + z) ∣ (x^2 + y^2 + z^2) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l1889_188976


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1889_188977

theorem inequality_solution_range (m : ℝ) :
  (∀ x : ℝ, (m + 1) * x^2 - 2 * (m - 1) * x + 3 * (m - 1) < 0) ↔ m < -1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1889_188977


namespace NUMINAMATH_CALUDE_modulus_of_3_minus_2i_l1889_188908

theorem modulus_of_3_minus_2i : Complex.abs (3 - 2*Complex.I) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_3_minus_2i_l1889_188908


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l1889_188919

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 15*x + 36 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 7*x - 60 = (x + b)*(x - c)) →
  a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l1889_188919


namespace NUMINAMATH_CALUDE_ratio_calculation_l1889_188900

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l1889_188900


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1889_188947

def U : Set ℕ := {1,2,3,4,5,6}
def A : Set ℕ := {2,4,6}

theorem complement_of_A_in_U :
  (U \ A) = {1,3,5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1889_188947


namespace NUMINAMATH_CALUDE_prism_height_to_base_ratio_l1889_188962

/-- 
For a regular quadrangular prism where a plane passes through the diagonal 
of the lower base and the opposite vertex of the upper base, forming a 
cross-section with angle α between its equal sides, the ratio of the prism's 
height to the side length of its base is (√(2 cos α)) / (2 sin(α/2)).
-/
theorem prism_height_to_base_ratio (α : Real) : 
  let h := Real.sqrt (2 * Real.cos α) / (2 * Real.sin (α / 2))
  let a := 1  -- Assuming unit side length for simplicity
  (h : Real) = (Real.sqrt (2 * Real.cos α)) / (2 * Real.sin (α / 2)) := by
  sorry

end NUMINAMATH_CALUDE_prism_height_to_base_ratio_l1889_188962
