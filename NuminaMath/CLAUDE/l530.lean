import Mathlib

namespace NUMINAMATH_CALUDE_A_finish_work_l530_53055

/-- The number of days it takes A to finish the work -/
def days_A : ℝ := 12

/-- The number of days it takes B to finish the work -/
def days_B : ℝ := 15

/-- The number of days B worked before leaving -/
def days_B_worked : ℝ := 10

/-- The number of days it takes A to finish the remaining work after B left -/
def days_A_remaining : ℝ := 4

/-- Theorem stating that A can finish the work in 12 days -/
theorem A_finish_work : 
  days_A = 12 :=
by sorry

end NUMINAMATH_CALUDE_A_finish_work_l530_53055


namespace NUMINAMATH_CALUDE_max_page_number_with_25_threes_l530_53028

/-- Counts the occurrences of a specific digit in a number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- Counts the total occurrences of a specific digit in numbers from 1 to n -/
def countDigitUpTo (n : ℕ) (d : ℕ) : ℕ := sorry

/-- The maximum page number that can be reached with a given number of '3's -/
def maxPageNumber (threes : ℕ) : ℕ := sorry

theorem max_page_number_with_25_threes :
  maxPageNumber 25 = 139 := by sorry

end NUMINAMATH_CALUDE_max_page_number_with_25_threes_l530_53028


namespace NUMINAMATH_CALUDE_parabola_midpoint_trajectory_l530_53030

theorem parabola_midpoint_trajectory (x y : ℝ) : 
  let parabola := {(x, y) : ℝ × ℝ | x^2 = 4*y}
  let focus := (0, 1)
  ∀ (p : ℝ × ℝ), p ∈ parabola → 
    let midpoint := ((p.1 + focus.1)/2, (p.2 + focus.2)/2)
    midpoint.1^2 = 2*midpoint.2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_trajectory_l530_53030


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_l530_53002

theorem arithmetic_mean_reciprocals : 
  let numbers := [2, 3, 7, 11]
  let reciprocals := numbers.map (λ x => 1 / x)
  let sum := reciprocals.sum
  let mean := sum / 4
  mean = 493 / 1848 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_l530_53002


namespace NUMINAMATH_CALUDE_students_who_got_off_l530_53060

/-- Given a school bus scenario where some students get off at a stop, 
    this theorem proves the number of students who got off. -/
theorem students_who_got_off (initial : ℕ) (remaining : ℕ) 
  (h1 : initial = 10) (h2 : remaining = 7) : initial - remaining = 3 := by
  sorry

#check students_who_got_off

end NUMINAMATH_CALUDE_students_who_got_off_l530_53060


namespace NUMINAMATH_CALUDE_units_digit_17_pow_2023_l530_53090

theorem units_digit_17_pow_2023 :
  ∃ (n : ℕ), n < 10 ∧ 17^2023 ≡ n [ZMOD 10] ∧ n = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_17_pow_2023_l530_53090


namespace NUMINAMATH_CALUDE_mean_home_runs_l530_53022

def player_count : ℕ := 13
def total_home_runs : ℕ := 80

def home_run_distribution : List (ℕ × ℕ) :=
  [(5, 5), (5, 6), (1, 7), (1, 8), (1, 10)]

theorem mean_home_runs :
  (total_home_runs : ℚ) / player_count = 80 / 13 := by
  sorry

end NUMINAMATH_CALUDE_mean_home_runs_l530_53022


namespace NUMINAMATH_CALUDE_bowl_water_percentage_l530_53093

theorem bowl_water_percentage (x : ℝ) (h1 : x > 0) (h2 : x / 2 + 4 = 14) : 
  (14 / x) * 100 = 70 :=
sorry

end NUMINAMATH_CALUDE_bowl_water_percentage_l530_53093


namespace NUMINAMATH_CALUDE_no_solution_3a_squared_equals_b_squared_plus_1_l530_53051

theorem no_solution_3a_squared_equals_b_squared_plus_1 :
  ¬ ∃ (a b : ℕ), 3 * a^2 = b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_3a_squared_equals_b_squared_plus_1_l530_53051


namespace NUMINAMATH_CALUDE_unique_triple_sum_l530_53023

theorem unique_triple_sum (x y z : ℕ) : 
  x ≤ y ∧ y ≤ z ∧ x^x + y^y + z^z = 3382 ↔ (x, y, z) = (1, 4, 5) :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_sum_l530_53023


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_property_l530_53009

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  is_isosceles : (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

/-- Distance squared between two points -/
def dist_squared (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

/-- The theorem to be proved -/
theorem isosceles_right_triangle_property (triangle : IsoscelesRightTriangle) :
  ∀ P : ℝ × ℝ, (P.2 = triangle.A.2 ∧ P.2 = triangle.B.2) →
    dist_squared P triangle.A + dist_squared P triangle.B = 2 * dist_squared P triangle.C :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_property_l530_53009


namespace NUMINAMATH_CALUDE_sodium_sulfate_decahydrate_weight_sodium_sulfate_decahydrate_weight_is_966_75_l530_53063

/-- The molecular weight of 3 moles of Na2SO4·10H2O -/
theorem sodium_sulfate_decahydrate_weight : ℝ → ℝ → ℝ → ℝ → ℝ := 
  fun (na_weight : ℝ) (s_weight : ℝ) (o_weight : ℝ) (h_weight : ℝ) =>
  let mw := 2 * na_weight + s_weight + 14 * o_weight + 20 * h_weight
  3 * mw

/-- The molecular weight of 3 moles of Na2SO4·10H2O is 966.75 grams -/
theorem sodium_sulfate_decahydrate_weight_is_966_75 :
  sodium_sulfate_decahydrate_weight 22.99 32.07 16.00 1.01 = 966.75 := by
  sorry

end NUMINAMATH_CALUDE_sodium_sulfate_decahydrate_weight_sodium_sulfate_decahydrate_weight_is_966_75_l530_53063


namespace NUMINAMATH_CALUDE_not_divisible_by_seven_l530_53053

theorem not_divisible_by_seven (a b : ℕ) : 
  ¬(7 ∣ (a * b)) → ¬(7 ∣ a ∨ 7 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_seven_l530_53053


namespace NUMINAMATH_CALUDE_complex_magnitude_l530_53025

theorem complex_magnitude (x y : ℝ) (h : x * (1 + Complex.I) = 1 + y * Complex.I) : 
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l530_53025


namespace NUMINAMATH_CALUDE_distance_to_karasuk_is_210_l530_53052

/-- The distance from Novosibirsk to Karasuk --/
def distance_to_karasuk : ℝ := 210

/-- The initial distance between the bus and the car --/
def initial_distance : ℝ := 70

/-- The distance the car travels after catching up with the bus --/
def car_distance_after_catchup : ℝ := 40

/-- The distance the bus travels after the car catches up --/
def bus_distance_after_catchup : ℝ := 20

/-- The speed of the bus --/
def bus_speed : ℝ := sorry

/-- The speed of the car --/
def car_speed : ℝ := sorry

/-- The time taken for the car to catch up with the bus --/
def catchup_time : ℝ := sorry

theorem distance_to_karasuk_is_210 :
  distance_to_karasuk = initial_distance + car_speed * catchup_time :=
by sorry

end NUMINAMATH_CALUDE_distance_to_karasuk_is_210_l530_53052


namespace NUMINAMATH_CALUDE_initial_milk_amount_l530_53014

/-- Proves that the initial amount of milk is 10 liters given the conditions of the problem -/
theorem initial_milk_amount (initial_water_content : Real) 
                             (target_water_content : Real)
                             (pure_milk_added : Real) :
  initial_water_content = 0.05 →
  target_water_content = 0.02 →
  pure_milk_added = 15 →
  ∃ (initial_milk : Real),
    initial_milk * initial_water_content = 
      (initial_milk + pure_milk_added) * target_water_content ∧
    initial_milk = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_milk_amount_l530_53014


namespace NUMINAMATH_CALUDE_book_purchase_ratio_l530_53020

/-- Represents the number of people who purchased only book A -/
def C : ℕ := 1000

/-- Represents the number of people who purchased both books A and B -/
def AB : ℕ := 500

/-- Represents the total number of people who purchased book A -/
def A : ℕ := C + AB

/-- Represents the total number of people who purchased book B -/
def B : ℕ := AB + (A / 2 - AB)

theorem book_purchase_ratio : (AB : ℚ) / (B - AB : ℚ) = 2 := by sorry

end NUMINAMATH_CALUDE_book_purchase_ratio_l530_53020


namespace NUMINAMATH_CALUDE_line_circle_intersection_l530_53059

theorem line_circle_intersection (k : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = k * A.1 + 2) ∧ 
    (B.2 = k * B.1 + 2) ∧ 
    ((A.1 - 3)^2 + (A.2 - 1)^2 = 9) ∧ 
    ((B.1 - 3)^2 + (B.2 - 1)^2 = 9) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 32)) →
  (k = 0 ∨ k = -3/4) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l530_53059


namespace NUMINAMATH_CALUDE_problem_statement_l530_53008

theorem problem_statement (a b : ℝ) (h1 : a * b = 3) (h2 : a - 2 * b = 5) :
  a^2 * b - 2 * a * b^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l530_53008


namespace NUMINAMATH_CALUDE_cakes_baked_yesterday_prove_cakes_baked_yesterday_l530_53016

def cakes_baked_today : ℕ := 5
def cakes_sold_dinner : ℕ := 6
def cakes_left : ℕ := 2

theorem cakes_baked_yesterday : ℕ :=
  cakes_sold_dinner - cakes_baked_today + cakes_left

theorem prove_cakes_baked_yesterday :
  cakes_baked_yesterday = 3 := by
  sorry

end NUMINAMATH_CALUDE_cakes_baked_yesterday_prove_cakes_baked_yesterday_l530_53016


namespace NUMINAMATH_CALUDE_no_x_with_both_rational_l530_53095

theorem no_x_with_both_rational : ¬∃ x : ℝ, ∃ p q : ℚ, 
  (Real.sin x + Real.sqrt 2 = ↑p) ∧ (Real.cos x - Real.sqrt 2 = ↑q) := by
  sorry

end NUMINAMATH_CALUDE_no_x_with_both_rational_l530_53095


namespace NUMINAMATH_CALUDE_inequality_system_solution_l530_53027

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x, (x - b < 0 ∧ x + a > 0) ↔ (2 < x ∧ x < 3)) → 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l530_53027


namespace NUMINAMATH_CALUDE_symmetry_of_regular_polygons_l530_53058

-- Define the types of polygons we're considering
inductive RegularPolygon
  | EquilateralTriangle
  | Square
  | RegularPentagon
  | RegularHexagon

-- Define the properties of symmetry
def isAxiSymmetric (p : RegularPolygon) : Prop :=
  match p with
  | RegularPolygon.EquilateralTriangle => true
  | RegularPolygon.Square => true
  | RegularPolygon.RegularPentagon => true
  | RegularPolygon.RegularHexagon => true

def isCentrallySymmetric (p : RegularPolygon) : Prop :=
  match p with
  | RegularPolygon.EquilateralTriangle => false
  | RegularPolygon.Square => true
  | RegularPolygon.RegularPentagon => false
  | RegularPolygon.RegularHexagon => true

-- Theorem statement
theorem symmetry_of_regular_polygons :
  ∀ p : RegularPolygon, 
    (isAxiSymmetric p ∧ isCentrallySymmetric p) ↔ 
    (p = RegularPolygon.Square ∨ p = RegularPolygon.RegularHexagon) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_regular_polygons_l530_53058


namespace NUMINAMATH_CALUDE_ninth_grade_students_l530_53085

/-- Proves that given a total of 50 students from three grades, with the seventh grade having 2x - 1 students and the eighth grade having x students, the number of students in the ninth grade is 51 - 3x. -/
theorem ninth_grade_students (x : ℕ) : 
  (50 : ℕ) = (2 * x - 1) + x + (51 - 3 * x) := by
  sorry

end NUMINAMATH_CALUDE_ninth_grade_students_l530_53085


namespace NUMINAMATH_CALUDE_trapezoid_longer_base_l530_53036

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  shorter_base : ℝ
  altitude : ℝ
  longer_base : ℝ
  area : ℝ

/-- The trapezoid satisfies the given conditions -/
def satisfies_conditions (t : Trapezoid) : Prop :=
  t.shorter_base = 5 ∧
  t.altitude = 7 ∧
  t.area = 63 ∧
  ∃ (d : ℝ), t.shorter_base = t.altitude - d ∧ t.longer_base = t.altitude + d

theorem trapezoid_longer_base (t : Trapezoid) 
  (h : satisfies_conditions t) : t.longer_base = 13 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_longer_base_l530_53036


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l530_53094

/-- Given a rectangle with length 4π cm and width 2 cm that is rolled into a cylinder
    using the longer side as the circumference of the base, prove that the total
    surface area of the resulting cylinder is 16π cm². -/
theorem cylinder_surface_area (π : ℝ) (h : π > 0) :
  let rectangle_length : ℝ := 4 * π
  let rectangle_width : ℝ := 2
  let base_circumference : ℝ := rectangle_length
  let base_radius : ℝ := base_circumference / (2 * π)
  let cylinder_height : ℝ := rectangle_width
  let total_surface_area : ℝ := 2 * π * base_radius^2 + 2 * π * base_radius * cylinder_height
  total_surface_area = 16 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l530_53094


namespace NUMINAMATH_CALUDE_f_power_of_two_divides_l530_53012

/-- f(d) is the smallest possible integer that has exactly d positive divisors -/
def f (d : ℕ) : ℕ := sorry

/-- Theorem: For every non-negative integer k, f(2^k) divides f(2^(k+1)) -/
theorem f_power_of_two_divides (k : ℕ) : 
  (f (2^k)) ∣ (f (2^(k+1))) := by sorry

end NUMINAMATH_CALUDE_f_power_of_two_divides_l530_53012


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l530_53044

-- Problem 1
theorem problem_one : (9/4)^(3/2) - (-9.6)^0 - (27/8)^(2/3) + (3/2)^(-2) = 1/2 := by
  sorry

-- Problem 2
theorem problem_two (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l530_53044


namespace NUMINAMATH_CALUDE_solve_for_y_l530_53092

theorem solve_for_y (x y : ℝ) (h1 : 3 * (x - y) = 18) (h2 : x + y = 20) : y = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l530_53092


namespace NUMINAMATH_CALUDE_rectangle_area_l530_53019

theorem rectangle_area (width : ℝ) (length : ℝ) (perimeter : ℝ) :
  length = 4 * width →
  perimeter = 2 * (length + width) →
  perimeter = 200 →
  width * length = 1600 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l530_53019


namespace NUMINAMATH_CALUDE_complex_modulus_theorem_l530_53086

theorem complex_modulus_theorem (r : ℝ) (z : ℂ) 
  (h1 : |r| < 3) 
  (h2 : r ≠ 2) 
  (h3 : z + r * z⁻¹ = 2) : 
  Complex.abs z = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_theorem_l530_53086


namespace NUMINAMATH_CALUDE_function_property_l530_53039

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 2) + f x = 3) 
  (h2 : f 1 = 0) : 
  f 2023 = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l530_53039


namespace NUMINAMATH_CALUDE_collinear_necessary_not_sufficient_l530_53073

/-- Four points in 3D space -/
structure FourPoints where
  p1 : ℝ × ℝ × ℝ
  p2 : ℝ × ℝ × ℝ
  p3 : ℝ × ℝ × ℝ
  p4 : ℝ × ℝ × ℝ

/-- Predicate: three of the four points lie on the same straight line -/
def threePointsCollinear (points : FourPoints) : Prop :=
  sorry

/-- Predicate: all four points lie on the same plane -/
def fourPointsCoplanar (points : FourPoints) : Prop :=
  sorry

/-- Theorem: Three points collinear is necessary but not sufficient for four points coplanar -/
theorem collinear_necessary_not_sufficient :
  (∀ points : FourPoints, fourPointsCoplanar points → threePointsCollinear points) ∧
  (∃ points : FourPoints, threePointsCollinear points ∧ ¬fourPointsCoplanar points) :=
sorry

end NUMINAMATH_CALUDE_collinear_necessary_not_sufficient_l530_53073


namespace NUMINAMATH_CALUDE_x_minus_y_values_l530_53037

theorem x_minus_y_values (x y : ℤ) (hx : x = -3) (hy : |y| = 2) : 
  x - y = -5 ∨ x - y = -1 := by sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l530_53037


namespace NUMINAMATH_CALUDE_cat_food_sale_calculation_l530_53070

/-- Theorem: Cat Food Sale Calculation
Given:
- 20 people bought cat food
- First 8 customers bought 3 cases each
- Next 4 customers bought 2 cases each
- Last 8 customers bought 1 case each

Prove: The total number of cases of cat food sold is 40.
-/
theorem cat_food_sale_calculation (total_customers : Nat) 
  (first_group_size : Nat) (first_group_cases : Nat)
  (second_group_size : Nat) (second_group_cases : Nat)
  (third_group_size : Nat) (third_group_cases : Nat)
  (h1 : total_customers = 20)
  (h2 : first_group_size = 8)
  (h3 : first_group_cases = 3)
  (h4 : second_group_size = 4)
  (h5 : second_group_cases = 2)
  (h6 : third_group_size = 8)
  (h7 : third_group_cases = 1)
  (h8 : total_customers = first_group_size + second_group_size + third_group_size) :
  first_group_size * first_group_cases + 
  second_group_size * second_group_cases + 
  third_group_size * third_group_cases = 40 := by
  sorry

end NUMINAMATH_CALUDE_cat_food_sale_calculation_l530_53070


namespace NUMINAMATH_CALUDE_rectangle_square_ratio_l530_53078

theorem rectangle_square_ratio (s a b : ℝ) (h1 : a * b = 2 * s ^ 2) (h2 : a = 2 * b) :
  a / s = 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_square_ratio_l530_53078


namespace NUMINAMATH_CALUDE_sin_300_degrees_l530_53024

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l530_53024


namespace NUMINAMATH_CALUDE_exam_average_l530_53096

theorem exam_average (x : ℝ) : 
  (15 * x + 10 * 90) / 25 = 81 → x = 75 := by sorry

end NUMINAMATH_CALUDE_exam_average_l530_53096


namespace NUMINAMATH_CALUDE_more_stable_performance_l530_53075

/-- Represents a person's shooting performance -/
structure ShootingPerformance where
  average : ℝ
  variance : ℝ

/-- Determines if the first performance is more stable than the second -/
def isMoreStable (p1 p2 : ShootingPerformance) : Prop :=
  p1.variance < p2.variance

/-- Theorem: Given two shooting performances with the same average,
    the one with smaller variance is more stable -/
theorem more_stable_performance 
  (personA personB : ShootingPerformance)
  (h_same_average : personA.average = personB.average)
  (h_variance_A : personA.variance = 1.4)
  (h_variance_B : personB.variance = 0.6) :
  isMoreStable personB personA :=
sorry

end NUMINAMATH_CALUDE_more_stable_performance_l530_53075


namespace NUMINAMATH_CALUDE_sum_after_2015_iterations_l530_53033

/-- The process of adding digits and appending the sum -/
def process (n : ℕ) : ℕ := sorry

/-- The result of applying the process n times to the initial number -/
def iterate_process (initial : ℕ) (n : ℕ) : ℕ := sorry

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_after_2015_iterations :
  sum_of_digits (iterate_process 2015 2015) = 8065 := by sorry

end NUMINAMATH_CALUDE_sum_after_2015_iterations_l530_53033


namespace NUMINAMATH_CALUDE_gcd_3869_6497_l530_53072

theorem gcd_3869_6497 : Nat.gcd 3869 6497 = 73 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3869_6497_l530_53072


namespace NUMINAMATH_CALUDE_lego_airplane_model_l530_53038

theorem lego_airplane_model (total_legos : ℕ) (additional_legos : ℕ) (num_models : ℕ) :
  total_legos = 400 →
  additional_legos = 80 →
  num_models = 2 →
  (total_legos + additional_legos) / num_models = 240 :=
by sorry

end NUMINAMATH_CALUDE_lego_airplane_model_l530_53038


namespace NUMINAMATH_CALUDE_quadratic_function_form_l530_53017

/-- A quadratic function with two equal real roots and derivative 2x + 2 -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
  equal_roots : ∃ (r : ℝ), (∀ x, f x = 0 ↔ x = r)
  derivative : ∀ x, deriv f x = 2 * x + 2

/-- The quadratic function with the given properties is x^2 + 2x + 1 -/
theorem quadratic_function_form (qf : QuadraticFunction) : 
  ∀ x, qf.f x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_form_l530_53017


namespace NUMINAMATH_CALUDE_area_of_trapezoid_l530_53011

structure Triangle where
  area : ℝ

structure Trapezoid where
  area : ℝ

def isosceles_triangle (t : Triangle) : Prop := sorry

theorem area_of_trapezoid (PQR : Triangle) (smallest : Triangle) (QSTM : Trapezoid) :
  isosceles_triangle PQR →
  PQR.area = 100 →
  smallest.area = 2 →
  QSTM.area = 90 := by
  sorry

end NUMINAMATH_CALUDE_area_of_trapezoid_l530_53011


namespace NUMINAMATH_CALUDE_uncovered_side_length_l530_53062

/-- Represents a rectangular field with three sides fenced -/
structure FencedField where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing : ℝ

/-- The uncovered side of a fenced field is 20 feet given the conditions -/
theorem uncovered_side_length (field : FencedField)
  (h_area : field.area = 80)
  (h_fencing : field.fencing = 28)
  (h_rect_area : field.area = field.length * field.width)
  (h_fencing_sum : field.fencing = 2 * field.width + field.length) :
  field.length = 20 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_side_length_l530_53062


namespace NUMINAMATH_CALUDE_equal_even_odd_probability_l530_53032

/-- The number of dice being rolled -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def sides_per_die : ℕ := 6

/-- The probability of rolling an even number on a single die -/
def prob_even : ℚ := 1/2

/-- The probability of rolling an odd number on a single die -/
def prob_odd : ℚ := 1/2

/-- The number of ways to choose half the dice to show even numbers -/
def ways_to_choose_half : ℕ := Nat.choose num_dice (num_dice / 2)

/-- Theorem: The probability of rolling 8 six-sided dice and getting an equal number of even and odd results is 35/128 -/
theorem equal_even_odd_probability : 
  (ways_to_choose_half : ℚ) * prob_even^num_dice = 35/128 := by sorry

end NUMINAMATH_CALUDE_equal_even_odd_probability_l530_53032


namespace NUMINAMATH_CALUDE_inequality_system_solution_l530_53041

theorem inequality_system_solution (x : ℝ) :
  (4 * x - 2 ≥ 3 * (x - 1)) ∧ ((x - 5) / 2 > x - 4) → -1 ≤ x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l530_53041


namespace NUMINAMATH_CALUDE_students_passed_test_l530_53087

/-- Represents the result of a proficiency test -/
structure TestResult where
  total_students : ℕ
  passing_score : ℕ
  passed_students : ℕ

/-- The proficiency test result for the university -/
def university_test : TestResult :=
  { total_students := 1000
  , passing_score := 70
  , passed_students := 600 }

/-- Theorem stating the number of students who passed the test -/
theorem students_passed_test : university_test.passed_students = 600 := by
  sorry

#check students_passed_test

end NUMINAMATH_CALUDE_students_passed_test_l530_53087


namespace NUMINAMATH_CALUDE_complex_product_real_l530_53021

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 1 + a * Complex.I
  let z₂ : ℂ := 3 + 2 * Complex.I
  (z₁ * z₂).im = 0 → a = -2/3 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l530_53021


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l530_53034

-- Define the hyperbola C
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the line l
def line (t : ℝ) (x y : ℝ) : Prop := x = t * y + 2

-- Define the condition for the circle with diameter MN passing through A(2,-2)
def circle_condition (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 - 2) * (x2 - 2) + (y1 + 2) * (y2 + 2) = 0

theorem hyperbola_line_intersection :
  (hyperbola 3 4) →
  (hyperbola (Real.sqrt 2) (Real.sqrt 2)) →
  ∀ t : ℝ,
    (∃ x1 y1 x2 y2 : ℝ,
      x1 ≠ x2 ∧
      hyperbola x1 y1 ∧
      hyperbola x2 y2 ∧
      line t x1 y1 ∧
      line t x2 y2 ∧
      circle_condition x1 y1 x2 y2) →
    (t = 1 ∨ t = 1/7) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l530_53034


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l530_53045

/-- Represents the loan with varying interest rates over time -/
structure Loan where
  principal : ℝ
  rate1 : ℝ
  rate2 : ℝ
  rate3 : ℝ
  period1 : ℝ
  period2 : ℝ
  period3 : ℝ

/-- Calculates the total interest for a given loan -/
def totalInterest (loan : Loan) : ℝ :=
  loan.principal * (loan.rate1 * loan.period1 + loan.rate2 * loan.period2 + loan.rate3 * loan.period3)

/-- Theorem stating that given the specific interest rates and periods, 
    if the total interest is 11400, then the principal is 12000 -/
theorem loan_principal_calculation (loan : Loan) 
  (h1 : loan.rate1 = 0.06)
  (h2 : loan.rate2 = 0.09)
  (h3 : loan.rate3 = 0.14)
  (h4 : loan.period1 = 2)
  (h5 : loan.period2 = 3)
  (h6 : loan.period3 = 4)
  (h7 : totalInterest loan = 11400) :
  loan.principal = 12000 := by
  sorry

#check loan_principal_calculation

end NUMINAMATH_CALUDE_loan_principal_calculation_l530_53045


namespace NUMINAMATH_CALUDE_line_through_parabola_vertex_l530_53015

/-- The number of real values of b for which the line y = 2x + b passes through the vertex of the parabola y = x^2 + b^2 + 1 is zero. -/
theorem line_through_parabola_vertex (b : ℝ) : ¬∃ b, 2 * 0 + b = 0^2 + b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_parabola_vertex_l530_53015


namespace NUMINAMATH_CALUDE_min_value_on_line_l530_53084

/-- Given a point A(m,n) on the line x + 2y = 1 where m > 0 and n > 0,
    the minimum value of 2/m + 1/n is 8 -/
theorem min_value_on_line (m n : ℝ) (h1 : m + 2*n = 1) (h2 : m > 0) (h3 : n > 0) :
  ∀ (x y : ℝ), x + 2*y = 1 → x > 0 → y > 0 → 2/m + 1/n ≤ 2/x + 1/y :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_line_l530_53084


namespace NUMINAMATH_CALUDE_popsicle_sticks_remaining_l530_53066

theorem popsicle_sticks_remaining (initial : Real) (given_away : Real) :
  initial = 63.0 →
  given_away = 50.0 →
  initial - given_away = 13.0 := by sorry

end NUMINAMATH_CALUDE_popsicle_sticks_remaining_l530_53066


namespace NUMINAMATH_CALUDE_min_sum_and_inequality_l530_53003

theorem min_sum_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3 * a * b) :
  (∃ (min : ℝ), min = 4/3 ∧ ∀ x y, x > 0 → y > 0 → x + y = 3 * x * y → x + y ≥ min) ∧
  (a / b + b / a ≥ 8 / (9 * a * b)) := by
sorry

end NUMINAMATH_CALUDE_min_sum_and_inequality_l530_53003


namespace NUMINAMATH_CALUDE_students_in_all_classes_l530_53097

theorem students_in_all_classes (total_students : ℕ) (drama_students : ℕ) (music_students : ℕ) (dance_students : ℕ) (students_in_two_plus : ℕ) :
  total_students = 25 →
  drama_students = 15 →
  music_students = 17 →
  dance_students = 11 →
  students_in_two_plus = 13 →
  ∃ (students_all_three : ℕ), students_all_three = 4 ∧
    students_all_three ≤ students_in_two_plus ∧
    students_all_three ≤ drama_students ∧
    students_all_three ≤ music_students ∧
    students_all_three ≤ dance_students :=
by
  sorry

end NUMINAMATH_CALUDE_students_in_all_classes_l530_53097


namespace NUMINAMATH_CALUDE_exactly_one_divisible_by_3_5_7_l530_53065

theorem exactly_one_divisible_by_3_5_7 :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 200 ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_divisible_by_3_5_7_l530_53065


namespace NUMINAMATH_CALUDE_exam_question_distribution_l530_53005

theorem exam_question_distribution :
  ∃ (P M E : ℕ),
    P + M + E = 50 ∧
    P ≥ 39 ∧ P ≤ 41 ∧
    M ≥ 7 ∧ M ≤ 8 ∧
    E ≥ 2 ∧ E ≤ 3 ∧
    P = 40 ∧ M = 7 ∧ E = 3 :=
by sorry

end NUMINAMATH_CALUDE_exam_question_distribution_l530_53005


namespace NUMINAMATH_CALUDE_euro_puzzle_l530_53006

theorem euro_puzzle (E M n : ℕ) : 
  (M + 3 = n * (E - 3)) →
  (E + n = 3 * (M - n)) →
  n > 0 →
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7) :=
by sorry

end NUMINAMATH_CALUDE_euro_puzzle_l530_53006


namespace NUMINAMATH_CALUDE_line_through_points_with_45_degree_inclination_l530_53071

/-- Given a line passing through points P(-2, m) and Q(m, 4) with an inclination angle of 45°, prove that m = 1. -/
theorem line_through_points_with_45_degree_inclination (m : ℝ) : 
  (∃ (line : Set (ℝ × ℝ)), 
    ((-2, m) ∈ line) ∧ 
    ((m, 4) ∈ line) ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → (y - m) = (x + 2))) → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_line_through_points_with_45_degree_inclination_l530_53071


namespace NUMINAMATH_CALUDE_draw_three_one_probability_l530_53088

/-- The probability of drawing exactly 3 balls of one color and 1 of the other color
    from a bin containing 10 black balls and 8 white balls, when 4 balls are drawn at random -/
theorem draw_three_one_probability (black_balls : ℕ) (white_balls : ℕ) (total_draw : ℕ) :
  black_balls = 10 →
  white_balls = 8 →
  total_draw = 4 →
  (Nat.choose black_balls 3 * Nat.choose white_balls 1 +
   Nat.choose black_balls 1 * Nat.choose white_balls 3) /
  Nat.choose (black_balls + white_balls) total_draw = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_draw_three_one_probability_l530_53088


namespace NUMINAMATH_CALUDE_units_digit_of_k_cubed_plus_five_to_k_l530_53018

theorem units_digit_of_k_cubed_plus_five_to_k (k : ℕ) : 
  k = 2024^2 + 3^2024 → (k^3 + 5^k) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_cubed_plus_five_to_k_l530_53018


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l530_53043

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 - 4*m + 3)
  (z.re = 0 ∧ z.im ≠ 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l530_53043


namespace NUMINAMATH_CALUDE_angle_measure_l530_53064

theorem angle_measure (x : ℝ) : 
  (180 - x = 3 * x - 10) → x = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l530_53064


namespace NUMINAMATH_CALUDE_range_of_a_l530_53067

-- Define the quadratic function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a^2 - 1)*x + (a - 2)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < 1 ∧ 1 < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ -2 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l530_53067


namespace NUMINAMATH_CALUDE_expression_evaluation_l530_53046

theorem expression_evaluation :
  let a : ℚ := 2
  let b : ℚ := -1/2
  let c : ℚ := -1
  a * b * c - (2 * a * b - (3 * a * b * c - b * c) + 4 * a * b * c) = 3/2 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l530_53046


namespace NUMINAMATH_CALUDE_tangent_line_equation_l530_53061

/-- The function f(x) = x³ - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The point A through which the tangent line passes --/
def A : ℝ × ℝ := (0, 16)

/-- The point of tangency M --/
def M : ℝ × ℝ := (-2, f (-2))

theorem tangent_line_equation :
  ∀ x y : ℝ, (9:ℝ)*x - y + 16 = 0 ↔ 
  (y - M.2 = f' M.1 * (x - M.1) ∧ f M.1 = M.2 ∧ A.2 - M.2 = f' M.1 * (A.1 - M.1)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l530_53061


namespace NUMINAMATH_CALUDE_division_in_base4_l530_53007

/-- Converts a base 4 number to base 10 -/
def base4ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 -/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Represents division in base 4 -/
def divBase4 (a b : List Nat) : List Nat :=
  base10ToBase4 ((base4ToBase10 a) / (base4ToBase10 b))

theorem division_in_base4 :
  divBase4 [3, 1, 2, 2] [3, 1] = [3, 5] := by sorry

end NUMINAMATH_CALUDE_division_in_base4_l530_53007


namespace NUMINAMATH_CALUDE_james_chores_time_l530_53069

/-- Given James spends 3 hours vacuuming and 3 times as long on other chores,
    prove that he spends 12 hours in total on his chores. -/
theorem james_chores_time :
  let vacuuming_time : ℝ := 3
  let other_chores_factor : ℝ := 3
  let other_chores_time : ℝ := vacuuming_time * other_chores_factor
  let total_time : ℝ := vacuuming_time + other_chores_time
  total_time = 12 := by sorry

end NUMINAMATH_CALUDE_james_chores_time_l530_53069


namespace NUMINAMATH_CALUDE_quadratic_equation_nonnegative_solutions_l530_53080

theorem quadratic_equation_nonnegative_solutions :
  ∃! (n : ℕ), n^2 + 3*n - 18 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_nonnegative_solutions_l530_53080


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l530_53048

theorem quadratic_inequality_solution_set (a b : ℝ) :
  (∀ x, x^2 + a*x + b > 0 ↔ x ∈ Set.Iio (-3) ∪ Set.Ioi 1) →
  (∀ x, a*x^2 + b*x - 2 < 0 ↔ x ∈ Set.Ioo (-1/2) 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l530_53048


namespace NUMINAMATH_CALUDE_x_condition_l530_53057

theorem x_condition (x : ℝ) : |x - 1| + |x - 5| = 4 → 1 ≤ x ∧ x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_x_condition_l530_53057


namespace NUMINAMATH_CALUDE_specific_bulb_probability_l530_53081

/-- The number of light bulbs -/
def num_bulbs : ℕ := 4

/-- The number of bulbs to be installed -/
def num_installed : ℕ := 3

/-- The number of ways to arrange n items taken k at a time -/
def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The probability of installing a specific bulb at a specific vertex -/
def probability : ℚ := (permutations (num_bulbs - 1) (num_installed - 1)) / (permutations num_bulbs num_installed)

theorem specific_bulb_probability : probability = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_specific_bulb_probability_l530_53081


namespace NUMINAMATH_CALUDE_stating_min_additional_games_is_ten_l530_53029

/-- Represents the number of games initially played -/
def initial_games : ℕ := 5

/-- Represents the number of games initially won by the Wolves -/
def initial_wolves_wins : ℕ := 2

/-- Represents the minimum winning percentage required for the Wolves -/
def min_winning_percentage : ℚ := 4/5

/-- 
Determines if a given number of additional games results in the Wolves
winning at least the minimum required percentage of all games
-/
def meets_winning_percentage (additional_games : ℕ) : Prop :=
  (initial_wolves_wins + additional_games : ℚ) / (initial_games + additional_games) ≥ min_winning_percentage

/-- 
Theorem stating that 10 is the minimum number of additional games
needed for the Wolves to meet the minimum winning percentage
-/
theorem min_additional_games_is_ten :
  (∀ n < 10, ¬(meets_winning_percentage n)) ∧ meets_winning_percentage 10 :=
sorry

end NUMINAMATH_CALUDE_stating_min_additional_games_is_ten_l530_53029


namespace NUMINAMATH_CALUDE_expression_simplification_l530_53042

theorem expression_simplification (y : ℝ) :
  3 * y - 2 * y^2 + 4 - (5 - 3 * y + 2 * y^2 - y^3) = y^3 + 6 * y - 4 * y^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l530_53042


namespace NUMINAMATH_CALUDE_equation_implication_l530_53074

theorem equation_implication (x y : ℝ) :
  x^2 - 3*x*y + 2*y^2 + x - y = 0 →
  x^2 - 2*x*y + y^2 - 5*x + 7*y = 0 →
  x*y - 12*x + 15*y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_implication_l530_53074


namespace NUMINAMATH_CALUDE_last_digit_of_fraction_l530_53089

/-- The last digit of the decimal expansion of 1 / (3^15 * 2^5) is 5 -/
theorem last_digit_of_fraction : ∃ (n : ℕ), (1 : ℚ) / (3^15 * 2^5) = n / 10 + 5 / 10^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_fraction_l530_53089


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l530_53031

theorem unique_positive_integer_solution : 
  ∃! (z : ℕ), z > 0 ∧ (4 * z)^2 - z = 2345 :=
by
  use 7
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l530_53031


namespace NUMINAMATH_CALUDE_perfect_square_completion_l530_53001

theorem perfect_square_completion (ε : ℝ) (hε : ε > 0) : 
  ∃ x : ℝ, ∃ y : ℝ, 
    (12.86 * 12.86 + 12.86 * x + 0.14 * 0.14 = y * y) ∧ 
    (|x - 0.28| < ε) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_completion_l530_53001


namespace NUMINAMATH_CALUDE_cell_growth_l530_53056

/-- The number of hours in 3 days and nights -/
def total_hours : ℕ := 72

/-- The number of hours required for one cell division -/
def division_time : ℕ := 12

/-- The initial number of cells -/
def initial_cells : ℕ := 2^10

/-- The number of cell divisions that occur in the given time period -/
def num_divisions : ℕ := total_hours / division_time

/-- The final number of cells after the given time period -/
def final_cells : ℕ := initial_cells * 2^num_divisions

theorem cell_growth :
  final_cells = 2^16 := by sorry

end NUMINAMATH_CALUDE_cell_growth_l530_53056


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l530_53040

theorem quadratic_equation_roots (x : ℝ) :
  (x^2 - 2*x - 1 = 0) ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l530_53040


namespace NUMINAMATH_CALUDE_eagles_winning_percentage_min_additional_games_is_minimum_l530_53013

/-- The minimum number of additional games needed for the Eagles to win at least 90% of all games -/
def min_additional_games : ℕ := 26

/-- The initial number of games played -/
def initial_games : ℕ := 4

/-- The initial number of games won by the Eagles -/
def initial_eagles_wins : ℕ := 1

/-- The minimum winning percentage required for the Eagles -/
def min_winning_percentage : ℚ := 9/10

theorem eagles_winning_percentage (M : ℕ) :
  (initial_eagles_wins + M : ℚ) / (initial_games + M) ≥ min_winning_percentage ↔ M ≥ min_additional_games :=
sorry

theorem min_additional_games_is_minimum :
  ∀ M : ℕ, M < min_additional_games →
    (initial_eagles_wins + M : ℚ) / (initial_games + M) < min_winning_percentage :=
sorry

end NUMINAMATH_CALUDE_eagles_winning_percentage_min_additional_games_is_minimum_l530_53013


namespace NUMINAMATH_CALUDE_books_per_continent_l530_53098

theorem books_per_continent 
  (total_books : ℕ) 
  (num_continents : ℕ) 
  (h1 : total_books = 488) 
  (h2 : num_continents = 4) 
  (h3 : total_books % num_continents = 0) : 
  total_books / num_continents = 122 := by
sorry

end NUMINAMATH_CALUDE_books_per_continent_l530_53098


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l530_53026

theorem least_positive_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 2 ∧ 
  n % 6 = 3 ∧
  ∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 6 = 3 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l530_53026


namespace NUMINAMATH_CALUDE_part_one_part_two_l530_53068

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 2 ∧ Real.cos t.B = 3/5

-- Part 1
theorem part_one (t : Triangle) (h : triangle_conditions t) (h_b : t.b = 4) :
  Real.sin t.A = 2/5 := by sorry

-- Part 2
theorem part_two (t : Triangle) (h : triangle_conditions t) 
  (h_area : (1/2) * t.a * t.c * Real.sin t.B = 4) :
  t.b = Real.sqrt 17 ∧ t.c = 5 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l530_53068


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l530_53077

theorem matrix_inverse_proof :
  let A : Matrix (Fin 4) (Fin 4) ℝ := !![2, -3, 0, 0;
                                       -4, 6, 0, 0;
                                        0, 0, 3, -5;
                                        0, 0, 1, -2]
  let M : Matrix (Fin 4) (Fin 4) ℝ := !![0, 0, 0.5, -0.5;
                                        0, 0, 0.5, -0.5;
                                        0, 0, 0.5, -0.5;
                                        0, 0, 0.5, -0.5]
  M * A = (1 : Matrix (Fin 4) (Fin 4) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l530_53077


namespace NUMINAMATH_CALUDE_train_speed_l530_53076

/-- Proves that a train of given length crossing a bridge of given length in a given time has a specific speed in km/hr -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 110)
  (h2 : bridge_length = 265)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l530_53076


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l530_53004

theorem least_positive_integer_with_remainders : ∃ b : ℕ, 
  b > 0 ∧ 
  b % 2 = 1 ∧ 
  b % 5 = 2 ∧ 
  b % 7 = 3 ∧ 
  ∀ c : ℕ, c > 0 ∧ c % 2 = 1 ∧ c % 5 = 2 ∧ c % 7 = 3 → b ≤ c :=
by
  use 17
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l530_53004


namespace NUMINAMATH_CALUDE_composite_number_probability_l530_53054

/-- Represents a standard 6-sided die -/
def StandardDie : Type := Fin 6

/-- Represents the special die with only prime numbers -/
def SpecialDie : Type := Fin 3

/-- The total number of possible outcomes when rolling 6 dice -/
def TotalOutcomes : ℕ := 6^5 * 3

/-- The number of non-composite outcomes -/
def NonCompositeOutcomes : ℕ := 4

/-- The probability of getting a composite number -/
def CompositeNumberProbability : ℚ := 5831 / 5832

/-- Theorem stating the probability of getting a composite number when rolling 6 dice
    (5 standard 6-sided dice and 1 special die with prime numbers 2, 3, 5) and
    multiplying their face values -/
theorem composite_number_probability :
  (TotalOutcomes - NonCompositeOutcomes : ℚ) / TotalOutcomes = CompositeNumberProbability :=
sorry

end NUMINAMATH_CALUDE_composite_number_probability_l530_53054


namespace NUMINAMATH_CALUDE_inequality_sum_l530_53099

theorem inequality_sum (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_sum_l530_53099


namespace NUMINAMATH_CALUDE_prob_sum_less_than_9_l530_53035

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The maximum sum we're considering -/
def maxSum : ℕ := 9

/-- The set of possible outcomes when rolling two dice -/
def outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range sides) (Finset.range sides)

/-- The favorable outcomes (sum less than maxSum) -/
def favorableOutcomes : Finset (ℕ × ℕ) :=
  outcomes.filter (fun p => p.1 + p.2 < maxSum)

/-- Probability of rolling a sum less than maxSum with two fair dice -/
theorem prob_sum_less_than_9 :
  (favorableOutcomes.card : ℚ) / (outcomes.card : ℚ) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_less_than_9_l530_53035


namespace NUMINAMATH_CALUDE_profit_percentage_is_30_percent_l530_53079

/-- Calculate the percentage of profit given the cost price and selling price --/
def percentage_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating that the percentage profit is 30% for the given prices --/
theorem profit_percentage_is_30_percent :
  percentage_profit 350 455 = 30 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_30_percent_l530_53079


namespace NUMINAMATH_CALUDE_picture_position_l530_53091

theorem picture_position (wall_width picture_width shift : ℝ) 
  (hw : wall_width = 25)
  (hp : picture_width = 4)
  (hs : shift = 1) :
  let center := wall_width / 2
  let picture_center := center + shift
  let left_edge := picture_center - picture_width / 2
  left_edge = 11.5 := by sorry

end NUMINAMATH_CALUDE_picture_position_l530_53091


namespace NUMINAMATH_CALUDE_last_digit_is_two_l530_53049

/-- Represents a 2000-digit integer as a list of natural numbers -/
def LongInteger := List Nat

/-- Checks if two consecutive digits are divisible by 17 or 23 -/
def validPair (a b : Nat) : Prop := (a * 10 + b) % 17 = 0 ∨ (a * 10 + b) % 23 = 0

/-- Defines the properties of our specific 2000-digit integer -/
def SpecialInteger (n : LongInteger) : Prop :=
  n.length = 2000 ∧
  n.head? = some 3 ∧
  ∀ i, i < 1999 → validPair (n.get! i) (n.get! (i + 1))

theorem last_digit_is_two (n : LongInteger) (h : SpecialInteger n) : 
  n.getLast? = some 2 := by
  sorry

#check last_digit_is_two

end NUMINAMATH_CALUDE_last_digit_is_two_l530_53049


namespace NUMINAMATH_CALUDE_fifteen_percent_of_a_minus_70_l530_53047

theorem fifteen_percent_of_a_minus_70 (a : ℝ) : (0.15 * a) - 70 = 0.15 * a - 70 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_a_minus_70_l530_53047


namespace NUMINAMATH_CALUDE_ellipse_parameter_sum_l530_53010

-- Define the ellipse parameters
def F₁ : ℝ × ℝ := (0, 0)
def F₂ : ℝ × ℝ := (8, 0)
def distance_sum : ℝ := 10

-- Define the ellipse equation
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  ∃ (h k a b : ℝ), 
    (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ∧
    (x - F₁.1)^2 + (y - F₁.2)^2 + (x - F₂.1)^2 + (y - F₂.2)^2 = distance_sum^2

-- Theorem statement
theorem ellipse_parameter_sum :
  ∃ (h k a b : ℝ),
    (∀ P, is_on_ellipse P → 
      (P.1 - h)^2 / a^2 + (P.2 - k)^2 / b^2 = 1) ∧
    h + k + a + b = 12 :=
sorry

end NUMINAMATH_CALUDE_ellipse_parameter_sum_l530_53010


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l530_53083

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : a + b + c = 45) 
  (h2 : a^2 + b^2 + c^2 = 625) : 
  2 * (a * b + b * c + c * a) = 1400 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l530_53083


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l530_53082

/-- An isosceles triangle with congruent sides of 8 cm and perimeter of 25 cm has a base of 9 cm -/
theorem isosceles_triangle_base_length : 
  ∀ (base congruent_side : ℝ),
  congruent_side = 8 →
  base + 2 * congruent_side = 25 →
  base = 9 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l530_53082


namespace NUMINAMATH_CALUDE_residual_analysis_characteristics_l530_53050

/-- Represents a residual in a statistical model. -/
structure Residual where
  value : ℝ

/-- Represents a statistical analysis method. -/
structure AnalysisMethod where
  name : String
  uses_residuals : Bool
  judges_model_fitting : Bool
  identifies_suspicious_data : Bool

/-- Definition of residual analysis based on its characteristics. -/
def residual_analysis : AnalysisMethod :=
  { name := "residual analysis",
    uses_residuals := true,
    judges_model_fitting := true,
    identifies_suspicious_data := true }

/-- Theorem stating that the analysis method using residuals to judge model fitting
    and identify suspicious data is residual analysis. -/
theorem residual_analysis_characteristics :
  ∀ (method : AnalysisMethod),
    method.uses_residuals ∧
    method.judges_model_fitting ∧
    method.identifies_suspicious_data →
    method = residual_analysis :=
by sorry

end NUMINAMATH_CALUDE_residual_analysis_characteristics_l530_53050


namespace NUMINAMATH_CALUDE_divisible_by_seven_last_digit_l530_53000

theorem divisible_by_seven_last_digit :
  ∃! d : ℕ, d < 10 ∧ ∀ n : ℕ, n % 10 = d → (7 ∣ n ↔ 7 ∣ d) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_divisible_by_seven_last_digit_l530_53000
