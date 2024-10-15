import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1178_117803

-- Define the side lengths of the isosceles triangle
def side_a : ℝ := 9
def side_b : ℝ := sorry  -- This will be either 3 or 5

-- Define the equation for side_b
axiom side_b_equation : side_b^2 - 8*side_b + 15 = 0

-- Define the perimeter of the triangle
def perimeter : ℝ := 2*side_a + side_b

-- Theorem statement
theorem isosceles_triangle_perimeter :
  perimeter = 19 ∨ perimeter = 21 ∨ perimeter = 23 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1178_117803


namespace NUMINAMATH_CALUDE_five_times_seven_and_two_fifths_l1178_117874

theorem five_times_seven_and_two_fifths (x : ℚ) : x = 5 * (7 + 2/5) → x = 37 := by
  sorry

end NUMINAMATH_CALUDE_five_times_seven_and_two_fifths_l1178_117874


namespace NUMINAMATH_CALUDE_train_overtake_l1178_117822

-- Define the speeds of the trains
def speed_A : ℝ := 30
def speed_B : ℝ := 45

-- Define the overtake distance
def overtake_distance : ℝ := 180

-- Define the time difference between train departures
def time_difference : ℝ := 2

-- Theorem statement
theorem train_overtake :
  speed_A * (time_difference + (overtake_distance / speed_B)) = overtake_distance ∧
  speed_B * (overtake_distance / speed_B) = overtake_distance := by
  sorry

end NUMINAMATH_CALUDE_train_overtake_l1178_117822


namespace NUMINAMATH_CALUDE_arithmetic_expressions_evaluation_l1178_117824

theorem arithmetic_expressions_evaluation :
  (2 * (-1)^3 - (-2)^2 / 4 + 10 = 7) ∧
  (abs (-3) - (-6 + 4) / (-1/2)^3 + (-1)^2013 = -14) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expressions_evaluation_l1178_117824


namespace NUMINAMATH_CALUDE_yogurt_combinations_l1178_117883

def num_flavors : ℕ := 5
def num_toppings : ℕ := 8

def combinations_with_no_topping : ℕ := 1
def combinations_with_one_topping (n : ℕ) : ℕ := n
def combinations_with_two_toppings (n : ℕ) : ℕ := n * (n - 1) / 2

def total_topping_combinations (n : ℕ) : ℕ :=
  combinations_with_no_topping + 
  combinations_with_one_topping n + 
  combinations_with_two_toppings n

theorem yogurt_combinations : 
  num_flavors * total_topping_combinations num_toppings = 185 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l1178_117883


namespace NUMINAMATH_CALUDE_stock_quoted_value_l1178_117844

/-- Proves that given an investment of 1620 in an 8% stock that earns 135, the stock is quoted at 96 --/
theorem stock_quoted_value (investment : ℝ) (dividend_rate : ℝ) (dividend_earned : ℝ) 
  (h1 : investment = 1620)
  (h2 : dividend_rate = 8 / 100)
  (h3 : dividend_earned = 135) :
  (investment / ((dividend_earned * 100) / dividend_rate)) * 100 = 96 := by
  sorry

end NUMINAMATH_CALUDE_stock_quoted_value_l1178_117844


namespace NUMINAMATH_CALUDE_inequality_proof_l1178_117899

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a + b + c = 1) : 
  (2*a*b + b*c + c*a + c^2/2 ≤ 1/2) ∧ 
  ((a^2 + c^2)/b + (b^2 + a^2)/c + (c^2 + b^2)/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1178_117899


namespace NUMINAMATH_CALUDE_sequence_general_term_l1178_117857

theorem sequence_general_term (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - 2 * a n = 2^n) →
  ∀ n : ℕ, n ≥ 1 → a n = n * 2^(n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1178_117857


namespace NUMINAMATH_CALUDE_simplify_fourth_roots_l1178_117898

theorem simplify_fourth_roots : 64^(1/4) - 144^(1/4) = 2 * Real.sqrt 2 - 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fourth_roots_l1178_117898


namespace NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l1178_117850

theorem product_zero_implies_factor_zero (a b c : ℝ) : a * b * c = 0 → (a = 0 ∨ b = 0 ∨ c = 0) := by
  sorry

end NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l1178_117850


namespace NUMINAMATH_CALUDE_divisor_problem_l1178_117845

theorem divisor_problem (n : ℕ+) : 
  (∃ k : ℕ, n = 2019 * k) →
  (∃ d : Fin 38 → ℕ+, 
    (∀ i j, i < j → d i < d j) ∧
    (∀ i, d i ∣ n) ∧
    (d 0 = 1) ∧
    (d 37 = n) ∧
    (n = d 18 * d 19)) →
  (n = 3^18 * 673 ∨ n = 673^18 * 3) := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l1178_117845


namespace NUMINAMATH_CALUDE_no_real_roots_l1178_117827

theorem no_real_roots (m : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 4 ∧ ∀ x ∈ s, (x : ℝ) - m < 0 ∧ 7 - 2*(x : ℝ) ≤ 1) →
  ∀ x : ℝ, 8*x^2 - 8*x + m ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_l1178_117827


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1178_117863

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1178_117863


namespace NUMINAMATH_CALUDE_hex_numeric_count_and_sum_l1178_117889

/-- Represents a hexadecimal digit --/
inductive HexDigit
| Numeric (n : Fin 10)
| Alpha (a : Fin 6)

/-- Converts a natural number to its hexadecimal representation --/
def toHex (n : ℕ) : List HexDigit := sorry

/-- Checks if a hexadecimal representation contains only numeric digits --/
def onlyNumeric (hex : List HexDigit) : Bool := sorry

/-- Counts the number of positive integers up to n whose hexadecimal 
    representation contains only numeric digits --/
def countNumericHex (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a natural number --/
def sumOfDigits (n : ℕ) : ℕ := sorry

theorem hex_numeric_count_and_sum : 
  countNumericHex 2000 = 1999 ∧ sumOfDigits 1999 = 28 := by sorry

end NUMINAMATH_CALUDE_hex_numeric_count_and_sum_l1178_117889


namespace NUMINAMATH_CALUDE_find_p_value_l1178_117848

theorem find_p_value (p q r : ℂ) (h_p_real : p.im = 0) 
  (h_sum : p + q + r = 5)
  (h_sum_prod : p * q + q * r + r * p = 5)
  (h_prod : p * q * r = 5) : 
  p = 4 := by sorry

end NUMINAMATH_CALUDE_find_p_value_l1178_117848


namespace NUMINAMATH_CALUDE_derivative_tan_and_exp_minus_sqrt_l1178_117823

open Real

theorem derivative_tan_and_exp_minus_sqrt (x : ℝ) : 
  (deriv tan x = 1 / (cos x)^2) ∧ 
  (deriv (fun x => exp x - sqrt x) x = exp x - 1 / (2 * sqrt x)) :=
by sorry

end NUMINAMATH_CALUDE_derivative_tan_and_exp_minus_sqrt_l1178_117823


namespace NUMINAMATH_CALUDE_unique_zero_implies_a_range_l1178_117897

/-- The cubic function f(x) = ax^3 - 3x^2 + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

theorem unique_zero_implies_a_range 
  (a : ℝ) 
  (h_unique : ∃! x₀ : ℝ, f a x₀ = 0) 
  (h_neg : ∃ x₀ : ℝ, f a x₀ = 0 ∧ x₀ < 0) :
  a > 2 :=
sorry

end NUMINAMATH_CALUDE_unique_zero_implies_a_range_l1178_117897


namespace NUMINAMATH_CALUDE_range_of_a_l1178_117882

open Real

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, |2^x - a| < |5 - 2^x|) →
  3 < a ∧ a < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1178_117882


namespace NUMINAMATH_CALUDE_teachers_liking_beverages_l1178_117801

theorem teachers_liking_beverages 
  (total : ℕ) 
  (tea : ℕ) 
  (coffee : ℕ) 
  (h1 : total = 90)
  (h2 : tea = 66)
  (h3 : coffee = 42)
  (h4 : ∃ (both neither : ℕ), both = 3 * neither ∧ tea + coffee - both + neither = total) :
  ∃ (at_least_one : ℕ), at_least_one = 81 ∧ at_least_one = tea + coffee - (tea + coffee - total + (total - tea - coffee) / 2) :=
by sorry

end NUMINAMATH_CALUDE_teachers_liking_beverages_l1178_117801


namespace NUMINAMATH_CALUDE_isosceles_triangle_areas_sum_l1178_117896

/-- Represents the areas of right isosceles triangles constructed on the sides of a right triangle -/
structure TriangleAreas where
  A : ℝ  -- Area of the isosceles triangle on side 5
  B : ℝ  -- Area of the isosceles triangle on side 12
  C : ℝ  -- Area of the isosceles triangle on side 13

/-- Theorem: For a right triangle with sides 5, 12, and 13, 
    if right isosceles triangles are constructed on each side, 
    then the sum of the areas of the triangles on the two shorter sides 
    equals the area of the triangle on the hypotenuse -/
theorem isosceles_triangle_areas_sum (areas : TriangleAreas) 
  (h1 : areas.A = (5 * 5) / 2)
  (h2 : areas.B = (12 * 12) / 2)
  (h3 : areas.C = (13 * 13) / 2) : 
  areas.A + areas.B = areas.C := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_areas_sum_l1178_117896


namespace NUMINAMATH_CALUDE_incenter_coordinates_specific_triangle_l1178_117891

/-- Given a triangle PQR with side lengths p, q, r, this function returns the coordinates of the incenter I -/
def incenter_coordinates (p q r : ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating that for a triangle with side lengths 8, 10, and 6, the incenter coordinates are (1/3, 5/12, 1/4) -/
theorem incenter_coordinates_specific_triangle :
  let (x, y, z) := incenter_coordinates 8 10 6
  x = 1/3 ∧ y = 5/12 ∧ z = 1/4 ∧ x + y + z = 1 := by sorry

end NUMINAMATH_CALUDE_incenter_coordinates_specific_triangle_l1178_117891


namespace NUMINAMATH_CALUDE_textbook_delivery_problem_l1178_117853

theorem textbook_delivery_problem (x y : ℝ) : 
  (0.5 * x + 0.2 * y = 390) ∧ 
  (0.5 * x = 3 * 0.8 * y) →
  (x = 720 ∧ y = 150) := by
sorry

end NUMINAMATH_CALUDE_textbook_delivery_problem_l1178_117853


namespace NUMINAMATH_CALUDE_new_rectangle_area_comparison_l1178_117842

theorem new_rectangle_area_comparison (a b : ℝ) (h : 0 < a ∧ a < b) :
  let new_base := 2 * a * b
  let new_height := (a * Real.sqrt (a^2 + b^2)) / 2
  let new_area := new_base * new_height
  let circle_area := Real.pi * b^2
  new_area = a^2 * b * Real.sqrt (a^2 + b^2) ∧ 
  ∃ (a b : ℝ), new_area ≠ circle_area :=
by sorry

end NUMINAMATH_CALUDE_new_rectangle_area_comparison_l1178_117842


namespace NUMINAMATH_CALUDE_min_inequality_solution_set_l1178_117816

open Set Real

theorem min_inequality_solution_set (x : ℝ) (hx : x ≠ 0) :
  min 4 (x + 4 / x) ≥ 8 * min x (1 / x) ↔ x ∈ Iic 0 ∪ Ioo 0 (1 / 2) ∪ Ici 2 :=
sorry

end NUMINAMATH_CALUDE_min_inequality_solution_set_l1178_117816


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l1178_117878

/-- Given an integer n and a digit d, returns the number composed of n repetitions of d -/
def repeat_digit (n : ℕ) (d : ℕ) : ℕ := 
  d * (10^n - 1) / 9

/-- Returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem sum_of_digits_9ab (a b : ℕ) : 
  a = repeat_digit 1985 8 → 
  b = repeat_digit 1985 5 → 
  sum_of_digits (9 * a * b) = 17865 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l1178_117878


namespace NUMINAMATH_CALUDE_garden_dimensions_l1178_117867

/-- Represents a rectangular garden with given perimeter and length-width relationship --/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_width_relation : length = width + 3
  perimeter_formula : perimeter = 2 * (length + width)

/-- Theorem stating the dimensions of the garden given the conditions --/
theorem garden_dimensions (g : RectangularGarden) 
  (h : g.perimeter = 26) : g.width = 5 ∧ g.length = 8 := by
  sorry

#check garden_dimensions

end NUMINAMATH_CALUDE_garden_dimensions_l1178_117867


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1178_117835

theorem quadratic_inequality_solution_set 
  (a b c x₁ x₂ : ℝ) 
  (h₁ : x₁ < x₂) 
  (h₂ : a < 0) 
  (h₃ : ∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) :
  ∀ x, a * x^2 + b * x + c > 0 ↔ x₁ < x ∧ x < x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1178_117835


namespace NUMINAMATH_CALUDE_paco_cookies_l1178_117830

theorem paco_cookies (initial_cookies : ℕ) (eaten_cookies : ℕ) (given_cookies : ℕ) 
  (h1 : initial_cookies = 17)
  (h2 : eaten_cookies = 14)
  (h3 : eaten_cookies + given_cookies ≤ initial_cookies) :
  given_cookies = 3 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l1178_117830


namespace NUMINAMATH_CALUDE_max_obtuse_triangles_four_points_l1178_117852

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle formed by three points -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Predicate to check if a triangle is obtuse -/
def isObtuse (t : Triangle) : Prop :=
  sorry

/-- The set of all possible triangles formed by 4 points -/
def allTriangles (p1 p2 p3 p4 : Point) : Set Triangle :=
  sorry

/-- The number of obtuse triangles in a set of triangles -/
def numObtuseTriangles (ts : Set Triangle) : ℕ :=
  sorry

/-- Theorem: The maximum number of obtuse triangles formed by 4 points is 4 -/
theorem max_obtuse_triangles_four_points (p1 p2 p3 p4 : Point) :
  ∃ (arrangement : Point → Point),
    numObtuseTriangles (allTriangles (arrangement p1) (arrangement p2) (arrangement p3) (arrangement p4)) ≤ 4 ∧
    ∃ (q1 q2 q3 q4 : Point),
      numObtuseTriangles (allTriangles q1 q2 q3 q4) = 4 :=
sorry

end NUMINAMATH_CALUDE_max_obtuse_triangles_four_points_l1178_117852


namespace NUMINAMATH_CALUDE_inequality_proof_l1178_117892

theorem inequality_proof : -2 < (-1)^3 ∧ (-1)^3 < (-0.6)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1178_117892


namespace NUMINAMATH_CALUDE_average_visitors_theorem_l1178_117879

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let numSundays := 4
  let numOtherDays := 30 - numSundays
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / 30

/-- Theorem stating that the average number of visitors per day is 292 -/
theorem average_visitors_theorem (sundayVisitors : ℕ) (otherDayVisitors : ℕ) 
  (h1 : sundayVisitors = 630) (h2 : otherDayVisitors = 240) : 
  averageVisitorsPerDay sundayVisitors otherDayVisitors = 292 := by
  sorry

#eval averageVisitorsPerDay 630 240

end NUMINAMATH_CALUDE_average_visitors_theorem_l1178_117879


namespace NUMINAMATH_CALUDE_larger_number_proof_l1178_117881

/-- Given two positive integers with HCF 23 and LCM factors 13 and 19, prove the larger is 437 -/
theorem larger_number_proof (a b : ℕ+) 
  (hcf : Nat.gcd a b = 23)
  (lcm : Nat.lcm a b = 23 * 13 * 19) :
  max a b = 437 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1178_117881


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_eq_one_l1178_117886

theorem min_sum_of_reciprocal_sum_eq_one (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 1/b = 1) : a + b ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 1/b₀ = 1 ∧ a₀ + b₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_eq_one_l1178_117886


namespace NUMINAMATH_CALUDE_power_of_power_l1178_117861

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1178_117861


namespace NUMINAMATH_CALUDE_other_solution_quadratic_equation_l1178_117854

theorem other_solution_quadratic_equation :
  let f : ℚ → ℚ := λ x ↦ 45 * x^2 - 56 * x + 31
  ∃ x : ℚ, x ≠ 2/5 ∧ f x = 0 ∧ x = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_equation_l1178_117854


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l1178_117875

/-- The equation of a reflected light ray given specific conditions -/
theorem reflected_ray_equation :
  let origin : ℝ × ℝ := (0, 0)
  let incident_line : ℝ → ℝ → Prop := λ x y => 2 * x - y + 5 = 0
  let reflection_point : ℝ × ℝ := (1, 3)
  let reflected_line : ℝ → ℝ → Prop := λ x y => x - 5 * y + 14 = 0
  ∀ (x y : ℝ), reflected_line x y ↔
    ∃ (p : ℝ × ℝ),
      incident_line p.1 p.2 ∧
      (p.1 - origin.1) * (y - p.2) = (x - p.1) * (p.2 - origin.2) ∧
      (p.1 - reflection_point.1) * (y - p.2) = (x - p.1) * (p.2 - reflection_point.2) :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l1178_117875


namespace NUMINAMATH_CALUDE_compare_expressions_l1178_117872

theorem compare_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a^2 / b + b^2 / a) > (a + b) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l1178_117872


namespace NUMINAMATH_CALUDE_sum_of_products_equals_25079720_l1178_117802

def T : Finset ℕ := Finset.image (fun i => 3^i) (Finset.range 8)

def M : ℕ := (Finset.sum T fun x => 
  (Finset.sum (T.erase x) fun y => x * y))

theorem sum_of_products_equals_25079720 : M = 25079720 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_equals_25079720_l1178_117802


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_84_l1178_117868

/-- Represents a quadrilateral ABCD -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- The length of a side of a quadrilateral -/
def side_length (q : Quadrilateral) (side : Fin 4) : ℝ := sorry

/-- The measure of an angle in a quadrilateral -/
def angle_measure (q : Quadrilateral) (vertex : Fin 4) : ℝ := sorry

/-- Whether a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

theorem quadrilateral_area_is_84 (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_ab : side_length q 0 = 5)
  (h_bc : side_length q 1 = 12)
  (h_cd : side_length q 2 = 13)
  (h_ad : side_length q 3 = 15)
  (h_angle_abc : angle_measure q 1 = 90) :
  area q = 84 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_84_l1178_117868


namespace NUMINAMATH_CALUDE_quadratic_minimum_unique_minimum_l1178_117871

/-- The quadratic function f(x) = x^2 - 14x + 45 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 45

theorem quadratic_minimum (x : ℝ) : f x ≥ f 7 := by
  sorry

theorem unique_minimum : ∀ x : ℝ, x ≠ 7 → f x > f 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_unique_minimum_l1178_117871


namespace NUMINAMATH_CALUDE_thompson_class_median_l1178_117865

/-- Represents the number of families with a specific number of children -/
structure FamilyCount where
  childCount : ℕ
  familyCount : ℕ

/-- Calculates the median of a list of natural numbers -/
def median (l : List ℕ) : ℚ :=
  sorry

/-- Expands a list of FamilyCount into a list of individual family sizes -/
def expandCounts (counts : List FamilyCount) : List ℕ :=
  sorry

theorem thompson_class_median :
  let familyCounts : List FamilyCount := [
    ⟨1, 4⟩, ⟨2, 3⟩, ⟨3, 5⟩, ⟨4, 2⟩, ⟨5, 1⟩
  ]
  let expandedList := expandCounts familyCounts
  median expandedList = 3 := by
  sorry

end NUMINAMATH_CALUDE_thompson_class_median_l1178_117865


namespace NUMINAMATH_CALUDE_arccos_zero_l1178_117828

theorem arccos_zero : Real.arccos 0 = π / 2 := by sorry

end NUMINAMATH_CALUDE_arccos_zero_l1178_117828


namespace NUMINAMATH_CALUDE_train_speed_l1178_117807

/-- Prove that given the conditions, the train's speed is 20 miles per hour. -/
theorem train_speed (distance_to_work : ℝ) (walking_speed : ℝ) (additional_train_time : ℝ) 
  (walking_vs_train_time_diff : ℝ) :
  distance_to_work = 1.5 →
  walking_speed = 3 →
  additional_train_time = 10.5 / 60 →
  walking_vs_train_time_diff = 15 / 60 →
  ∃ (train_speed : ℝ), 
    train_speed = 20 ∧
    distance_to_work / walking_speed = 
      distance_to_work / train_speed + additional_train_time + walking_vs_train_time_diff :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l1178_117807


namespace NUMINAMATH_CALUDE_system_solution_l1178_117869

theorem system_solution :
  let eq1 := (fun (x y : ℝ) ↦ x^2 + y^2 + 6*x*y = 68)
  let eq2 := (fun (x y : ℝ) ↦ 2*x^2 + 2*y^2 - 3*x*y = 16)
  (∀ x y, eq1 x y ∧ eq2 x y ↔ 
    ((x = 4 ∧ y = 2) ∨ (x = 2 ∧ y = 4) ∨ 
     (x = -4 ∧ y = -2) ∨ (x = -2 ∧ y = -4))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1178_117869


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l1178_117839

theorem quadratic_roots_product (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + m*x + 2*m = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ + x₂ = 1 →
  x₁ * x₂ = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l1178_117839


namespace NUMINAMATH_CALUDE_ahmed_min_grade_l1178_117895

/-- The number of assignments excluding the final one -/
def num_assignments : ℕ := 9

/-- Ahmed's current average score -/
def ahmed_average : ℕ := 91

/-- Emily's current average score -/
def emily_average : ℕ := 92

/-- Sarah's current average score -/
def sarah_average : ℕ := 94

/-- The minimum passing score -/
def min_score : ℕ := 70

/-- The maximum possible score -/
def max_score : ℕ := 100

/-- Emily's score on the final assignment -/
def emily_final : ℕ := 90

/-- Function to calculate the total score -/
def total_score (average : ℕ) (final : ℕ) : ℕ :=
  average * num_assignments + final

/-- Theorem stating the minimum grade Ahmed needs -/
theorem ahmed_min_grade :
  ∀ x : ℕ, 
    (x ≤ max_score) →
    (total_score ahmed_average x > total_score emily_average emily_final) →
    (total_score ahmed_average x > total_score sarah_average min_score) →
    (∀ y : ℕ, y < x → (total_score ahmed_average y ≤ total_score emily_average emily_final ∨
                       total_score ahmed_average y ≤ total_score sarah_average min_score)) →
    x = 98 := by
  sorry

end NUMINAMATH_CALUDE_ahmed_min_grade_l1178_117895


namespace NUMINAMATH_CALUDE_lives_per_player_l1178_117825

theorem lives_per_player (initial_players : ℕ) (players_quit : ℕ) (total_lives : ℕ) : 
  initial_players = 8 → players_quit = 3 → total_lives = 15 → 
  (total_lives / (initial_players - players_quit) = 3) := by
  sorry

end NUMINAMATH_CALUDE_lives_per_player_l1178_117825


namespace NUMINAMATH_CALUDE_impossible_wire_arrangement_l1178_117832

-- Define a regular heptagon with columns
structure RegularHeptagonWithColumns where
  vertices : Fin 7 → ℝ
  is_regular : True  -- Placeholder for regularity condition

-- Define the connection between vertices
def second_nearest_neighbors (i : Fin 7) : Fin 7 × Fin 7 :=
  ((i + 2) % 7, (i + 5) % 7)

-- Define the intersection of wires
def wire_intersections (h : RegularHeptagonWithColumns) (i j : Fin 7) : Prop :=
  let (a, b) := second_nearest_neighbors i
  let (c, d) := second_nearest_neighbors j
  (a = c ∧ b ≠ d) ∨ (a = d ∧ b ≠ c) ∨ (b = c ∧ a ≠ d) ∨ (b = d ∧ a ≠ c)

-- Define the condition for wire arrangement
def valid_wire_arrangement (h : RegularHeptagonWithColumns) : Prop :=
  ∀ i j k : Fin 7, wire_intersections h i j → wire_intersections h i k →
    (h.vertices i < h.vertices j ∧ h.vertices i > h.vertices k) ∨
    (h.vertices i > h.vertices j ∧ h.vertices i < h.vertices k)

-- Theorem statement
theorem impossible_wire_arrangement :
  ¬∃ (h : RegularHeptagonWithColumns), valid_wire_arrangement h :=
sorry

end NUMINAMATH_CALUDE_impossible_wire_arrangement_l1178_117832


namespace NUMINAMATH_CALUDE_y_coordinate_range_l1178_117812

/-- The parabola equation y^2 = x + 4 -/
def parabola (x y : ℝ) : Prop := y^2 = x + 4

/-- Point A is at (0,2) -/
def point_A : ℝ × ℝ := (0, 2)

/-- B is on the parabola -/
def B_on_parabola (B : ℝ × ℝ) : Prop := parabola B.1 B.2

/-- C is on the parabola -/
def C_on_parabola (C : ℝ × ℝ) : Prop := parabola C.1 C.2

/-- AB is perpendicular to BC -/
def AB_perp_BC (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.2 - B.2) = -(B.1 - A.1) * (C.1 - B.1)

/-- The main theorem -/
theorem y_coordinate_range (B C : ℝ × ℝ) :
  B_on_parabola B → C_on_parabola C → AB_perp_BC point_A B C →
  C.2 ≤ 0 ∨ C.2 ≥ 4 := by sorry

end NUMINAMATH_CALUDE_y_coordinate_range_l1178_117812


namespace NUMINAMATH_CALUDE_inequality_proof_l1178_117800

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1178_117800


namespace NUMINAMATH_CALUDE_g_evaluation_and_derivative_l1178_117890

def g (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 2 * x^3 - 28 * x^2 + 15 * x - 90

theorem g_evaluation_and_derivative :
  g 6 = 17568 ∧ (deriv g) 6 = 15879 := by sorry

end NUMINAMATH_CALUDE_g_evaluation_and_derivative_l1178_117890


namespace NUMINAMATH_CALUDE_fifth_day_distance_l1178_117809

def running_sequence (n : ℕ) : ℕ := 2 + n - 1

theorem fifth_day_distance : running_sequence 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fifth_day_distance_l1178_117809


namespace NUMINAMATH_CALUDE_cucumbers_for_apples_l1178_117810

-- Define the cost relationships
def apple_banana_ratio : ℚ := 10 / 5
def banana_cucumber_ratio : ℚ := 3 / 4

-- Define the number of apples we're interested in
def apples_of_interest : ℚ := 20

-- Theorem to prove
theorem cucumbers_for_apples :
  let bananas_for_apples : ℚ := apples_of_interest / apple_banana_ratio
  let cucumbers_for_bananas : ℚ := bananas_for_apples * (1 / banana_cucumber_ratio)
  cucumbers_for_bananas = 40 / 3 :=
by sorry

end NUMINAMATH_CALUDE_cucumbers_for_apples_l1178_117810


namespace NUMINAMATH_CALUDE_plot_width_l1178_117837

/-- 
Given a rectangular plot with length 90 meters, if 60 poles placed 5 meters apart 
are needed to enclose the plot, then the width of the plot is 60 meters.
-/
theorem plot_width (poles : ℕ) (pole_distance : ℝ) (length width : ℝ) : 
  poles = 60 → 
  pole_distance = 5 → 
  length = 90 → 
  poles * pole_distance = 2 * (length + width) → 
  width = 60 := by sorry

end NUMINAMATH_CALUDE_plot_width_l1178_117837


namespace NUMINAMATH_CALUDE_two_cars_meeting_time_l1178_117826

/-- Two cars traveling between cities problem -/
theorem two_cars_meeting_time 
  (distance : ℝ) 
  (speed1 speed2 : ℝ) 
  (h1 : distance = 450) 
  (h2 : speed1 = 45) 
  (h3 : speed2 = 30) :
  (2 * distance) / (speed1 + speed2) = 12 := by
sorry

end NUMINAMATH_CALUDE_two_cars_meeting_time_l1178_117826


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1178_117851

theorem functional_equation_solution (f : ℤ → ℤ) :
  (∀ m n : ℤ, f (f m + n) + f m = f n + f (3 * m) + 2014) →
  ∃ c : ℤ, ∀ m : ℤ, f m = 2 * m + c :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1178_117851


namespace NUMINAMATH_CALUDE_factorization_count_mod_1000_l1178_117885

/-- A polynomial x^2 + ax + b can be factored into linear factors with integer coefficients -/
def HasIntegerFactors (a b : ℤ) : Prop :=
  ∃ c d : ℤ, a = c + d ∧ b = c * d

/-- The count of pairs (a,b) satisfying the conditions -/
def S : ℕ :=
  (Finset.range 100).sum (fun a => 
    (Finset.range (a + 1)).card)

/-- The main theorem -/
theorem factorization_count_mod_1000 : S % 1000 = 50 := by
  sorry

end NUMINAMATH_CALUDE_factorization_count_mod_1000_l1178_117885


namespace NUMINAMATH_CALUDE_two_numbers_with_given_means_l1178_117821

theorem two_numbers_with_given_means : ∃ a b : ℝ, 
  a > 0 ∧ b > 0 ∧ 
  Real.sqrt (a * b) = Real.sqrt 5 ∧
  2 / (1/a + 1/b) = 5/3 ∧
  a = (15 + Real.sqrt 145) / 4 ∧
  b = (15 - Real.sqrt 145) / 4 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_given_means_l1178_117821


namespace NUMINAMATH_CALUDE_max_d_is_15_l1178_117840

/-- Represents a 6-digit number of the form x5d,33e -/
structure SixDigitNumber where
  x : Nat
  d : Nat
  e : Nat
  h_x : x < 10
  h_d : d < 10
  h_e : e < 10

/-- Checks if a SixDigitNumber is divisible by 33 -/
def isDivisibleBy33 (n : SixDigitNumber) : Prop :=
  (n.x + n.d + n.e + 11) % 3 = 0 ∧ (n.x + n.d - n.e - 5) % 11 = 0

/-- The maximum value of d in a SixDigitNumber divisible by 33 is 15 -/
theorem max_d_is_15 : 
  ∀ n : SixDigitNumber, isDivisibleBy33 n → n.d ≤ 15 ∧ 
  ∃ m : SixDigitNumber, isDivisibleBy33 m ∧ m.d = 15 := by sorry

end NUMINAMATH_CALUDE_max_d_is_15_l1178_117840


namespace NUMINAMATH_CALUDE_all_three_sports_count_l1178_117817

/-- Represents a sports club with members playing various sports -/
structure SportsClub where
  total_members : ℕ
  badminton_players : ℕ
  tennis_players : ℕ
  basketball_players : ℕ
  no_sport_players : ℕ
  badminton_tennis_players : ℕ
  badminton_basketball_players : ℕ
  tennis_basketball_players : ℕ

/-- Calculates the number of members playing all three sports -/
def all_three_sports (club : SportsClub) : ℕ :=
  club.total_members - club.no_sport_players -
    (club.badminton_players + club.tennis_players + club.basketball_players -
     club.badminton_tennis_players - club.badminton_basketball_players - club.tennis_basketball_players)

/-- Theorem stating the number of members playing all three sports -/
theorem all_three_sports_count (club : SportsClub)
    (h1 : club.total_members = 60)
    (h2 : club.badminton_players = 25)
    (h3 : club.tennis_players = 30)
    (h4 : club.basketball_players = 15)
    (h5 : club.no_sport_players = 10)
    (h6 : club.badminton_tennis_players = 15)
    (h7 : club.badminton_basketball_players = 10)
    (h8 : club.tennis_basketball_players = 5) :
    all_three_sports club = 10 := by
  sorry


end NUMINAMATH_CALUDE_all_three_sports_count_l1178_117817


namespace NUMINAMATH_CALUDE_fruit_supply_theorem_l1178_117806

/-- Represents the weekly fruit requirements for a bakery -/
structure BakeryRequirement where
  strawberries : ℕ
  blueberries : ℕ
  raspberries : ℕ

/-- Calculates the total number of sacks needed for a given fruit over 10 weeks -/
def totalSacksFor10Weeks (weeklyRequirements : List BakeryRequirement) (getFruit : BakeryRequirement → ℕ) : ℕ :=
  10 * (weeklyRequirements.map getFruit).sum

/-- The list of weekly requirements for all bakeries -/
def allBakeries : List BakeryRequirement := [
  ⟨2, 3, 5⟩,
  ⟨4, 2, 8⟩,
  ⟨12, 10, 7⟩,
  ⟨8, 4, 3⟩,
  ⟨15, 6, 12⟩,
  ⟨5, 9, 11⟩
]

theorem fruit_supply_theorem :
  totalSacksFor10Weeks allBakeries (·.strawberries) = 460 ∧
  totalSacksFor10Weeks allBakeries (·.blueberries) = 340 ∧
  totalSacksFor10Weeks allBakeries (·.raspberries) = 460 := by
  sorry

end NUMINAMATH_CALUDE_fruit_supply_theorem_l1178_117806


namespace NUMINAMATH_CALUDE_triangle_reciprocal_side_angle_bisector_equality_l1178_117893

/-- For any triangle, the sum of reciprocals of side lengths equals the sum of cosines of half angles divided by their respective angle bisector lengths. -/
theorem triangle_reciprocal_side_angle_bisector_equality
  (a b c : ℝ) (α β γ : ℝ) (f_α f_β f_γ : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_angle_sum : α + β + γ = π)
  (h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_f_α : f_α = (2 * b * c * Real.cos (α / 2)) / (b + c))
  (h_f_β : f_β = (2 * a * c * Real.cos (β / 2)) / (a + c))
  (h_f_γ : f_γ = (2 * a * b * Real.cos (γ / 2)) / (a + b)) :
  1 / a + 1 / b + 1 / c = Real.cos (α / 2) / f_α + Real.cos (β / 2) / f_β + Real.cos (γ / 2) / f_γ :=
by sorry

end NUMINAMATH_CALUDE_triangle_reciprocal_side_angle_bisector_equality_l1178_117893


namespace NUMINAMATH_CALUDE_lunch_ratio_proof_l1178_117847

theorem lunch_ratio_proof (total_students : Nat) (cafeteria_students : Nat) (no_lunch_students : Nat) :
  total_students = 60 →
  cafeteria_students = 10 →
  no_lunch_students = 20 →
  ∃ k : Nat, total_students - cafeteria_students - no_lunch_students = k * cafeteria_students →
  (total_students - cafeteria_students - no_lunch_students) / cafeteria_students = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_lunch_ratio_proof_l1178_117847


namespace NUMINAMATH_CALUDE_function_value_at_four_l1178_117811

/-- Given a function f: ℝ → ℝ satisfying f(x) + 2f(1 - x) = 3x^2 for all x,
    prove that f(4) = 2 -/
theorem function_value_at_four (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 3 * x^2) : 
    f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_four_l1178_117811


namespace NUMINAMATH_CALUDE_solve_salary_problem_l1178_117819

def salary_problem (S : ℝ) : Prop :=
  let rent := (2/5) * S
  let food := (3/10) * S
  let conveyance := (1/8) * S
  (food + conveyance = 3400) →
  (S - (rent + food + conveyance) = 1400)

theorem solve_salary_problem :
  ∃ S : ℝ, salary_problem S :=
sorry

end NUMINAMATH_CALUDE_solve_salary_problem_l1178_117819


namespace NUMINAMATH_CALUDE_nth_power_divisibility_l1178_117818

theorem nth_power_divisibility (b n : ℕ) (h1 : b > 1) (h2 : n > 1)
  (h3 : ∀ k : ℕ, k > 1 → ∃ a_k : ℕ, k ∣ (b - a_k^n)) :
  ∃ A : ℕ, b = A^n := by sorry

end NUMINAMATH_CALUDE_nth_power_divisibility_l1178_117818


namespace NUMINAMATH_CALUDE_window_purchase_savings_l1178_117813

def window_price : ℕ := 150
def alice_windows : ℕ := 9
def bob_windows : ℕ := 10

def discount (n : ℕ) : ℕ :=
  (n / 6) * window_price

def cost (n : ℕ) : ℕ :=
  n * window_price - discount n

def total_separate_cost : ℕ :=
  cost alice_windows + cost bob_windows

def joint_windows : ℕ :=
  alice_windows + bob_windows

def joint_cost : ℕ :=
  cost joint_windows

def savings : ℕ :=
  total_separate_cost - joint_cost

theorem window_purchase_savings :
  savings = 150 := by sorry

end NUMINAMATH_CALUDE_window_purchase_savings_l1178_117813


namespace NUMINAMATH_CALUDE_combination_equality_implies_x_values_l1178_117873

theorem combination_equality_implies_x_values (x : ℕ) : 
  (Nat.choose 25 (2 * x) = Nat.choose 25 (x + 4)) → (x = 4 ∨ x = 7) := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_implies_x_values_l1178_117873


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1178_117814

/-- Given a right triangle with legs a and b, if rotating about leg a produces a cone
    of volume 500π cm³ and rotating about leg b produces a cone of volume 1800π cm³,
    then the length of the hypotenuse is approximately 24.46 cm. -/
theorem right_triangle_hypotenuse (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1/3 * π * a * b^2 = 500 * π) →
  (1/3 * π * b * a^2 = 1800 * π) →
  abs ((a^2 + b^2).sqrt - 24.46) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1178_117814


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1178_117858

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 1) :
  1/x + 1/y + 1/z ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1178_117858


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1178_117815

theorem algebraic_expression_value (x y : ℝ) (h : x - y - 7 = 0) :
  x^2 - y^2 - 14*y = 49 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1178_117815


namespace NUMINAMATH_CALUDE_unique_solution_two_power_minus_three_power_l1178_117887

theorem unique_solution_two_power_minus_three_power : 
  ∀ m n : ℕ+, 2^(m:ℕ) - 3^(n:ℕ) = 7 → m = 4 ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_two_power_minus_three_power_l1178_117887


namespace NUMINAMATH_CALUDE_squared_gt_iff_abs_gt_l1178_117866

theorem squared_gt_iff_abs_gt (a b : ℝ) : a^2 > b^2 ↔ |a| > |b| := by sorry

end NUMINAMATH_CALUDE_squared_gt_iff_abs_gt_l1178_117866


namespace NUMINAMATH_CALUDE_f_is_even_and_has_zero_point_l1178_117829

-- Define the function f(x) = x^2 - 1
def f (x : ℝ) : ℝ := x^2 - 1

-- Theorem stating that f is an even function and has a zero point
theorem f_is_even_and_has_zero_point :
  (∀ x : ℝ, f (-x) = f x) ∧ (∃ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_and_has_zero_point_l1178_117829


namespace NUMINAMATH_CALUDE_batsman_average_l1178_117870

/-- Calculates the new average of a batsman after the 17th inning -/
def newAverage (prevAverage : ℚ) (inningScore : ℕ) (numInnings : ℕ) : ℚ :=
  (prevAverage * (numInnings - 1) + inningScore) / numInnings

/-- Proves that the batsman's new average is 39 runs -/
theorem batsman_average : 
  ∀ (prevAverage : ℚ),
  newAverage prevAverage 87 17 = prevAverage + 3 →
  newAverage prevAverage 87 17 = 39 := by
    sorry

end NUMINAMATH_CALUDE_batsman_average_l1178_117870


namespace NUMINAMATH_CALUDE_parallel_intersections_l1178_117860

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection relation for planes and lines
variable (intersect : Plane → Plane → Line → Prop)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_intersections
  (α β γ : Plane) (m n : Line)
  (h1 : parallel_planes α β)
  (h2 : intersect α γ m)
  (h3 : intersect β γ n) :
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_parallel_intersections_l1178_117860


namespace NUMINAMATH_CALUDE_square_of_1009_l1178_117876

theorem square_of_1009 : 1009 * 1009 = 1018081 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1009_l1178_117876


namespace NUMINAMATH_CALUDE_inscribed_rectangle_coefficient_l1178_117862

/-- Triangle ABC with inscribed rectangle PQRS --/
structure TriangleWithRectangle where
  /-- Side length AB --/
  ab : ℝ
  /-- Side length BC --/
  bc : ℝ
  /-- Side length CA --/
  ca : ℝ
  /-- Width of the inscribed rectangle (PQ) --/
  ω : ℝ
  /-- Coefficient α in the area formula --/
  α : ℝ
  /-- Coefficient β in the area formula --/
  β : ℝ
  /-- P is on AB, Q on AC, R and S on BC --/
  rectangle_inscribed : Bool
  /-- Area formula for rectangle PQRS --/
  area_formula : ℝ → ℝ := fun ω => α * ω - β * ω^2

/-- The main theorem --/
theorem inscribed_rectangle_coefficient
  (t : TriangleWithRectangle)
  (h1 : t.ab = 15)
  (h2 : t.bc = 26)
  (h3 : t.ca = 25)
  (h4 : t.rectangle_inscribed = true) :
  t.β = 33 / 28 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_coefficient_l1178_117862


namespace NUMINAMATH_CALUDE_complex_magnitude_l1178_117864

theorem complex_magnitude (z : ℂ) : z = -2 + I → Complex.abs (z + 1) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1178_117864


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l1178_117846

theorem quadratic_polynomial_satisfies_conditions :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = 2.5 * x^2 - 5.5 * x + 13) ∧
    q (-1) = 10 ∧
    q 2 = 1 ∧
    q 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l1178_117846


namespace NUMINAMATH_CALUDE_rectangle_side_length_l1178_117859

/-- Given two rectangles A and B, where A has sides of length 3 and 6,
    and the ratio of corresponding sides of A to B is 3/4,
    prove that the length of side c in Rectangle B is 4. -/
theorem rectangle_side_length (a b c d : ℝ) : 
  a = 3 → b = 6 → a / c = 3 / 4 → b / d = 3 / 4 → c = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l1178_117859


namespace NUMINAMATH_CALUDE_triangle_median_theorem_l1178_117880

-- Define the triangle and its medians
structure Triangle :=
  (D E F : ℝ × ℝ)
  (DP EQ : ℝ × ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  -- DP and EQ are medians
  ∃ P Q : ℝ × ℝ,
    t.DP = P - t.D ∧
    t.EQ = Q - t.E ∧
    P = (t.E + t.F) / 2 ∧
    Q = (t.D + t.F) / 2 ∧
  -- DP and EQ are perpendicular
  t.DP.1 * t.EQ.1 + t.DP.2 * t.EQ.2 = 0 ∧
  -- Lengths of DP and EQ
  Real.sqrt (t.DP.1^2 + t.DP.2^2) = 18 ∧
  Real.sqrt (t.EQ.1^2 + t.EQ.2^2) = 24

-- Theorem statement
theorem triangle_median_theorem (t : Triangle) (h : is_valid_triangle t) :
  Real.sqrt ((t.F.1 - t.D.1)^2 + (t.F.2 - t.D.2)^2) = 8 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_triangle_median_theorem_l1178_117880


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1178_117808

/-- Two 2D vectors are parallel if the cross product of their coordinates is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (m, m + 1)
  parallel a b → m = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1178_117808


namespace NUMINAMATH_CALUDE_wine_equation_correct_l1178_117849

/-- Represents the value of clear wine in terms of grain -/
def clear_wine_value : ℝ := 10

/-- Represents the value of turbid wine in terms of grain -/
def turbid_wine_value : ℝ := 3

/-- Represents the total amount of grain available -/
def total_grain : ℝ := 30

/-- Represents the total amount of wine obtained -/
def total_wine : ℝ := 5

/-- Theorem stating that the equation 10x + 3(5-x) = 30 correctly represents
    the relationship between clear wine, turbid wine, and total grain value -/
theorem wine_equation_correct (x : ℝ) :
  0 ≤ x ∧ x ≤ total_wine →
  clear_wine_value * x + turbid_wine_value * (total_wine - x) = total_grain :=
by sorry

end NUMINAMATH_CALUDE_wine_equation_correct_l1178_117849


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_l1178_117834

/-- The intersection points of a hyperbola and a circle -/
theorem hyperbola_circle_intersection :
  ∀ x y : ℝ, x^2 - 9*y^2 = 36 ∧ x^2 + y^2 = 36 → (x = 6 ∧ y = 0) ∨ (x = -6 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_circle_intersection_l1178_117834


namespace NUMINAMATH_CALUDE_locus_of_P_l1178_117894

def M : ℝ × ℝ := (0, 5)
def N : ℝ × ℝ := (0, -5)

def perimeter : ℝ := 36

def is_on_locus (P : ℝ × ℝ) : Prop :=
  P.1 ≠ 0 ∧ (P.1^2 / 144 + P.2^2 / 169 = 1)

theorem locus_of_P (P : ℝ × ℝ) : 
  (dist M P + dist N P + dist M N = perimeter) → is_on_locus P :=
by sorry


end NUMINAMATH_CALUDE_locus_of_P_l1178_117894


namespace NUMINAMATH_CALUDE_cannot_determine_best_method_l1178_117877

/-- Represents an investment method --/
inductive InvestmentMethod
  | OneYear
  | ThreeYearThenOneYear
  | FiveOneYearThenFiveYear

/-- Calculates the final amount for a given investment method --/
def calculateFinalAmount (method : InvestmentMethod) (initialAmount : ℝ) : ℝ :=
  match method with
  | .OneYear => initialAmount * (1 + 0.0156) ^ 10
  | .ThreeYearThenOneYear => initialAmount * (1 + 0.0206 * 3) ^ 3 * (1 + 0.0156)
  | .FiveOneYearThenFiveYear => initialAmount * (1 + 0.0156) ^ 5 * (1 + 0.0282 * 5)

/-- Theorem stating that the best investment method cannot be determined without calculation --/
theorem cannot_determine_best_method (initialAmount : ℝ) :
  ∀ (m1 m2 : InvestmentMethod), m1 ≠ m2 →
  ∃ (result1 result2 : ℝ),
    calculateFinalAmount m1 initialAmount = result1 ∧
    calculateFinalAmount m2 initialAmount = result2 ∧
    (result1 > result2 ∨ result1 < result2) :=
by
  sorry

#check cannot_determine_best_method

end NUMINAMATH_CALUDE_cannot_determine_best_method_l1178_117877


namespace NUMINAMATH_CALUDE_farm_chickens_count_l1178_117888

/-- Proves that the total number of chickens on a farm is 69, given the number of ducks, geese, and their relationships to hens and roosters. -/
theorem farm_chickens_count (ducks geese : ℕ) 
  (h1 : ducks = 45)
  (h2 : geese = 28)
  (h3 : ∃ hens : ℕ, hens = ducks - 13)
  (h4 : ∃ roosters : ℕ, roosters = geese + 9) :
  ∃ total_chickens : ℕ, total_chickens = 69 ∧ 
    ∃ (hens roosters : ℕ), 
      hens = ducks - 13 ∧ 
      roosters = geese + 9 ∧ 
      total_chickens = hens + roosters := by
sorry

end NUMINAMATH_CALUDE_farm_chickens_count_l1178_117888


namespace NUMINAMATH_CALUDE_min_fence_length_for_given_garden_l1178_117831

/-- Calculates the minimum fence length for a rectangular garden with one side against a wall -/
def min_fence_length (length width : ℝ) : ℝ :=
  2 * width + length

theorem min_fence_length_for_given_garden :
  min_fence_length 32 14 = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_fence_length_for_given_garden_l1178_117831


namespace NUMINAMATH_CALUDE_binomial_expansion_and_specific_case_l1178_117820

theorem binomial_expansion_and_specific_case :
  ∀ (a b : ℝ),
    (a + b)^4 = a^4 + 4*a^3*b + 6*a^2*b^2 + 4*a*b^3 + b^4 ∧
    (2 - 1/3)^4 = 625/81 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_and_specific_case_l1178_117820


namespace NUMINAMATH_CALUDE_gcd_eight_factorial_seven_factorial_l1178_117838

theorem gcd_eight_factorial_seven_factorial :
  Nat.gcd (Nat.factorial 8) (Nat.factorial 7) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_eight_factorial_seven_factorial_l1178_117838


namespace NUMINAMATH_CALUDE_jane_dolls_l1178_117841

theorem jane_dolls (total : ℕ) (difference : ℕ) : total = 32 → difference = 6 → ∃ jane : ℕ, jane = 13 ∧ jane + (jane + difference) = total := by
  sorry

end NUMINAMATH_CALUDE_jane_dolls_l1178_117841


namespace NUMINAMATH_CALUDE_shoes_sold_l1178_117856

theorem shoes_sold (shoes sandals : ℕ) 
  (ratio : shoes / sandals = 9 / 5)
  (sandals_count : sandals = 40) : 
  shoes = 72 := by
  sorry

end NUMINAMATH_CALUDE_shoes_sold_l1178_117856


namespace NUMINAMATH_CALUDE_remainder_after_addition_l1178_117843

theorem remainder_after_addition (m : ℤ) (h : m % 5 = 2) : (m + 2535) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_addition_l1178_117843


namespace NUMINAMATH_CALUDE_kendra_suvs_count_l1178_117855

/-- The number of SUVs Kendra saw in the afternoon -/
def afternoon_suvs : ℕ := 10

/-- The number of SUVs Kendra saw in the evening -/
def evening_suvs : ℕ := 5

/-- The total number of SUVs Kendra saw during her road trip -/
def total_suvs : ℕ := afternoon_suvs + evening_suvs

theorem kendra_suvs_count : total_suvs = 15 := by
  sorry

end NUMINAMATH_CALUDE_kendra_suvs_count_l1178_117855


namespace NUMINAMATH_CALUDE_buffy_breath_holding_time_l1178_117805

/-- Represents the breath-holding times of Kelly, Brittany, and Buffy in seconds -/
structure BreathHoldingTimes where
  kelly : ℕ
  brittany : ℕ
  buffy : ℕ

/-- The breath-holding contest results -/
def contest : BreathHoldingTimes :=
  { kelly := 3 * 60,  -- Kelly's time in seconds
    brittany := 3 * 60 - 20,  -- Brittany's time is 20 seconds less than Kelly's
    buffy := (3 * 60 - 20) - 40  -- Buffy's time is 40 seconds less than Brittany's
  }

/-- Theorem stating that Buffy held her breath for 120 seconds -/
theorem buffy_breath_holding_time :
  contest.buffy = 120 := by
  sorry

end NUMINAMATH_CALUDE_buffy_breath_holding_time_l1178_117805


namespace NUMINAMATH_CALUDE_expression_evaluation_l1178_117804

theorem expression_evaluation : (3 * 5 * 6) * (1/3 + 1/5 + 1/6) = 63 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1178_117804


namespace NUMINAMATH_CALUDE_question_types_sum_steve_answerable_relation_l1178_117833

/-- Represents a math test with different types of questions -/
structure MathTest where
  total : ℕ
  word : ℕ
  addition_subtraction : ℕ
  geometry : ℕ

/-- Defines the properties of a valid math test -/
def is_valid_test (test : MathTest) : Prop :=
  test.word = test.total / 2 ∧
  test.addition_subtraction = test.total / 3 ∧
  test.geometry = test.total - test.word - test.addition_subtraction

/-- Theorem stating the relationship between question types and total questions -/
theorem question_types_sum (test : MathTest) (h : is_valid_test test) :
  test.word + test.addition_subtraction + test.geometry = test.total := by
  sorry

/-- Function representing the number of questions Steve can answer -/
def steve_answerable (total : ℕ) : ℕ :=
  total / 2 - 4

/-- Theorem stating the relationship between Steve's answerable questions and total questions -/
theorem steve_answerable_relation (test : MathTest) (h : is_valid_test test) :
  steve_answerable test.total = test.total / 2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_question_types_sum_steve_answerable_relation_l1178_117833


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1178_117836

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |x - 3| = 5 - 2*x ↔ x = 8/3 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1178_117836


namespace NUMINAMATH_CALUDE_cos_equality_proof_l1178_117884

theorem cos_equality_proof (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_proof_l1178_117884
