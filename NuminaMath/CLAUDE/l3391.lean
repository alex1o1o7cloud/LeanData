import Mathlib

namespace NUMINAMATH_CALUDE_smallest_solution_biquadratic_l3391_339139

theorem smallest_solution_biquadratic (x : ℝ) :
  x^4 - 26*x^2 + 169 = 0 → x ≥ -Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_biquadratic_l3391_339139


namespace NUMINAMATH_CALUDE_bookstore_purchase_equation_l3391_339114

theorem bookstore_purchase_equation (x : ℝ) : 
  (500 : ℝ) > 0 ∧ (700 : ℝ) > 0 ∧ x > 0 →
  (500 / x = 700 / (x + 4)) ↔ 
  (∃ (price_per_set : ℝ), 
    price_per_set > 0 ∧
    500 = price_per_set * x ∧
    700 = price_per_set * (x + 4)) :=
by sorry

end NUMINAMATH_CALUDE_bookstore_purchase_equation_l3391_339114


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l3391_339183

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_distinct (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def matches_pattern (a b c d e : ℕ) : Prop :=
  let abba := a * 1000 + b * 100 + b * 10 + a
  let cdea := c * 1000 + d * 100 + e * 10 + a
  let product := abba * cdea
  ∃ (x y z : ℕ),
    product = z * 100000 + b * 1000 + b * 100 + e * 10 + e ∧
    z = x * 10000 + y * 1000 + c * 100 + e * 10 + e

theorem multiplication_puzzle :
  ∀ (a b c d e : ℕ),
    is_valid_digit a → is_valid_digit b → is_valid_digit c → is_valid_digit d → is_valid_digit e →
    are_distinct a b c d e →
    matches_pattern a b c d e →
    a = 3 ∧ b = 0 ∧ c = 7 ∧ d = 2 ∧ e = 9 :=
sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l3391_339183


namespace NUMINAMATH_CALUDE_alla_boris_meeting_l3391_339199

/-- The number of streetlights -/
def total_streetlights : ℕ := 400

/-- Alla's position when the snapshot is taken -/
def alla_snapshot : ℕ := 55

/-- Boris's position when the snapshot is taken -/
def boris_snapshot : ℕ := 321

/-- The meeting point of Alla and Boris -/
def meeting_point : ℕ := 163

theorem alla_boris_meeting :
  ∀ (v_a v_b : ℝ), v_a > 0 → v_b > 0 →
  (alla_snapshot - 1 : ℝ) / v_a = (total_streetlights - boris_snapshot : ℝ) / v_b →
  (meeting_point - 1 : ℝ) / v_a = (total_streetlights - meeting_point : ℝ) / v_b :=
by sorry

end NUMINAMATH_CALUDE_alla_boris_meeting_l3391_339199


namespace NUMINAMATH_CALUDE_triangle_point_coordinates_l3391_339159

/-- Given a triangle ABC with median CM and angle bisector BL, prove that the coordinates of C are (14, 2) -/
theorem triangle_point_coordinates (A M L : ℝ × ℝ) : 
  A = (2, 8) → M = (4, 11) → L = (6, 6) → 
  ∃ (B C : ℝ × ℝ), 
    (M.1 = (A.1 + C.1) / 2 ∧ M.2 = (A.2 + C.2) / 2) ∧  -- M is midpoint of AC
    (∃ (t : ℝ), L = B + t • (A - C)) ∧                 -- L is on angle bisector BL
    C = (14, 2) := by
sorry

end NUMINAMATH_CALUDE_triangle_point_coordinates_l3391_339159


namespace NUMINAMATH_CALUDE_square_fraction_count_l3391_339190

theorem square_fraction_count : 
  ∃! n : ℤ, (∃ k : ℤ, 30 - 2*n ≠ 0 ∧ n/(30 - 2*n) = k^2 ∧ n/(30 - 2*n) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_square_fraction_count_l3391_339190


namespace NUMINAMATH_CALUDE_twentieth_number_in_base8_l3391_339104

/-- Converts a decimal number to its base 8 representation -/
def toBase8 (n : ℕ) : ℕ := sorry

/-- Represents the sequence of numbers in base 8 -/
def base8Sequence : ℕ → ℕ := sorry

theorem twentieth_number_in_base8 :
  base8Sequence 20 = toBase8 24 := by sorry

end NUMINAMATH_CALUDE_twentieth_number_in_base8_l3391_339104


namespace NUMINAMATH_CALUDE_quadratic_functions_property_l3391_339101

/-- Two quadratic functions with specific properties -/
theorem quadratic_functions_property (h j k : ℝ) : 
  (∃ (a b c d : ℕ), a ≠ b ∧ c ≠ d ∧ 
    3 * (a - h)^2 + j = 0 ∧ 
    3 * (b - h)^2 + j = 0 ∧
    2 * (c - h)^2 + k = 0 ∧ 
    2 * (d - h)^2 + k = 0) →
  (3 * h^2 + j = 2013 ∧ 2 * h^2 + k = 2014) →
  h = 36 := by
sorry

end NUMINAMATH_CALUDE_quadratic_functions_property_l3391_339101


namespace NUMINAMATH_CALUDE_positive_integer_solutions_count_l3391_339152

theorem positive_integer_solutions_count : 
  (Finset.filter (fun (xyz : ℕ × ℕ × ℕ) => 
    xyz.1 + xyz.2.1 + xyz.2.2 = 12 ∧ 
    xyz.1 > 0 ∧ xyz.2.1 > 0 ∧ xyz.2.2 > 0) 
    (Finset.product (Finset.range 12) (Finset.product (Finset.range 12) (Finset.range 12)))).card = 55 :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_count_l3391_339152


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3391_339123

/-- The area of the unfolded lateral surface of a cylinder with base radius 2 and height 2 is 8π. -/
theorem cylinder_lateral_surface_area : 
  ∀ (r h : ℝ), r = 2 → h = 2 → 2 * π * r * h = 8 * π :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3391_339123


namespace NUMINAMATH_CALUDE_complex_multiplication_l3391_339118

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (2 + 2*i) * (1 - 2*i) = 6 - 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3391_339118


namespace NUMINAMATH_CALUDE_period_of_sin_plus_cos_l3391_339144

/-- The period of the function y = 3sin(x) + 3cos(x) is 2π -/
theorem period_of_sin_plus_cos : 
  let f : ℝ → ℝ := λ x => 3 * Real.sin x + 3 * Real.cos x
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ ∀ q : ℝ, 0 < q ∧ q < p → ∃ x : ℝ, f (x + q) ≠ f x :=
by
  sorry

end NUMINAMATH_CALUDE_period_of_sin_plus_cos_l3391_339144


namespace NUMINAMATH_CALUDE_integral_exp_abs_plus_sqrt_l3391_339157

theorem integral_exp_abs_plus_sqrt : ∫ (x : ℝ) in (-1)..(1), (Real.exp (|x|) + Real.sqrt (1 - x^2)) = 2 * (Real.exp 1 - 1) + π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_exp_abs_plus_sqrt_l3391_339157


namespace NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l3391_339169

def late_flights : ℕ := 1
def initial_on_time_flights : ℕ := 3

def on_time_rate (additional_on_time : ℕ) : ℚ :=
  (initial_on_time_flights + additional_on_time) / (late_flights + initial_on_time_flights + additional_on_time)

theorem phoenix_airport_on_time_rate :
  ∃ n : ℕ, n > 0 ∧ on_time_rate n > (2 : ℚ) / 5 :=
sorry

end NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l3391_339169


namespace NUMINAMATH_CALUDE_proper_subsets_count_l3391_339119

def S : Finset Nat := {2, 4, 6, 8}

theorem proper_subsets_count : (Finset.powerset S).card - 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_proper_subsets_count_l3391_339119


namespace NUMINAMATH_CALUDE_line_circle_intersection_a_eq_one_l3391_339138

/-- A line intersecting a circle forming a right triangle -/
structure LineCircleIntersection where
  a : ℝ
  -- Line equation: ax - y + 6 = 0
  line : ℝ → ℝ → Prop := fun x y ↦ a * x - y + 6 = 0
  -- Circle equation: (x + 1)^2 + (y - a)^2 = 16
  circle : ℝ → ℝ → Prop := fun x y ↦ (x + 1)^2 + (y - a)^2 = 16
  -- Circle center
  center : ℝ × ℝ := (-1, a)
  -- Existence of intersection points A and B
  A : ℝ × ℝ
  B : ℝ × ℝ
  hA : line A.1 A.2 ∧ circle A.1 A.2
  hB : line B.1 B.2 ∧ circle B.1 B.2
  -- Triangle ABC is a right triangle
  hRight : (A.1 - B.1) * (center.1 - B.1) + (A.2 - B.2) * (center.2 - B.2) = 0

/-- The positive value of a in the LineCircleIntersection is 1 -/
theorem line_circle_intersection_a_eq_one (lci : LineCircleIntersection) : 
  lci.a > 0 → lci.a = 1 := by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_a_eq_one_l3391_339138


namespace NUMINAMATH_CALUDE_fewer_bees_than_flowers_l3391_339156

theorem fewer_bees_than_flowers (flowers : ℕ) (bees : ℕ) 
  (h1 : flowers = 5) (h2 : bees = 3) : flowers - bees = 2 := by
  sorry

end NUMINAMATH_CALUDE_fewer_bees_than_flowers_l3391_339156


namespace NUMINAMATH_CALUDE_abcd_sum_l3391_339197

theorem abcd_sum (a b c d : ℝ) 
  (eq1 : a + b + c = 3)
  (eq2 : a + b + d = -2)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = -1) :
  a * b + c * d = -190 / 9 := by
  sorry

end NUMINAMATH_CALUDE_abcd_sum_l3391_339197


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l3391_339121

theorem exponential_equation_solution (x : ℝ) :
  3^(3*x + 2) = (1 : ℝ) / 27 → x = -(5 : ℝ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l3391_339121


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt14_l3391_339188

theorem sqrt_sum_equals_2sqrt14 : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt14_l3391_339188


namespace NUMINAMATH_CALUDE_youtube_dislikes_difference_l3391_339168

theorem youtube_dislikes_difference (D : ℕ) : 
  D + 1000 = 2600 → D - D / 2 = 800 := by
  sorry

end NUMINAMATH_CALUDE_youtube_dislikes_difference_l3391_339168


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l3391_339181

theorem smallest_lcm_with_gcd_five (a b : ℕ) : 
  1000 ≤ a ∧ a < 10000 ∧ 
  1000 ≤ b ∧ b < 10000 ∧ 
  Nat.gcd a b = 5 →
  201000 ≤ Nat.lcm a b ∧ 
  ∃ (x y : ℕ), 1000 ≤ x ∧ x < 10000 ∧ 
               1000 ≤ y ∧ y < 10000 ∧ 
               Nat.gcd x y = 5 ∧ 
               Nat.lcm x y = 201000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l3391_339181


namespace NUMINAMATH_CALUDE_smallest_number_l3391_339172

theorem smallest_number (a b c : ℝ) (ha : a = 0.8) (hb : b = 1/2) (hc : c = 0.5) :
  min (min a b) c > 0.1 ∧ min (min a b) c = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3391_339172


namespace NUMINAMATH_CALUDE_author_average_earnings_l3391_339147

theorem author_average_earnings 
  (months_per_book : ℕ) 
  (years_writing : ℕ) 
  (total_earnings : ℕ) : 
  months_per_book = 2 → 
  years_writing = 20 → 
  total_earnings = 3600000 → 
  (total_earnings : ℚ) / ((12 / months_per_book) * years_writing) = 30000 :=
by sorry

end NUMINAMATH_CALUDE_author_average_earnings_l3391_339147


namespace NUMINAMATH_CALUDE_consecutive_numbers_equation_l3391_339187

theorem consecutive_numbers_equation (x y z : ℤ) : 
  (x = y + 1) → 
  (z = y - 1) → 
  (x > y) → 
  (y > z) → 
  (z = 2) → 
  (2 * x + 3 * y + 3 * z = 8 * y - 1) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_equation_l3391_339187


namespace NUMINAMATH_CALUDE_odd_divisors_of_power_minus_one_smallest_odd_divisors_second_smallest_odd_divisor_three_divides_nine_divides_infinitely_many_divisors_l3391_339160

theorem odd_divisors_of_power_minus_one (n : ℕ) :
  Odd n → n ∣ 2023^n - 1 → n ≥ 3 :=
sorry

theorem smallest_odd_divisors :
  (∃ (n : ℕ), Odd n ∧ n ∣ 2023^n - 1 ∧ n < 3) → False :=
sorry

theorem second_smallest_odd_divisor :
  (∃ (n : ℕ), Odd n ∧ n ∣ 2023^n - 1 ∧ 3 < n ∧ n < 9) → False :=
sorry

theorem three_divides : 3 ∣ 2023^3 - 1 :=
sorry

theorem nine_divides : 9 ∣ 2023^9 - 1 :=
sorry

theorem infinitely_many_divisors (k : ℕ) :
  k ≥ 1 → 3^k ∣ 2023^(3^k) - 1 :=
sorry

end NUMINAMATH_CALUDE_odd_divisors_of_power_minus_one_smallest_odd_divisors_second_smallest_odd_divisor_three_divides_nine_divides_infinitely_many_divisors_l3391_339160


namespace NUMINAMATH_CALUDE_student_age_proof_l3391_339195

theorem student_age_proof (n : ℕ) (initial_avg : ℚ) (new_avg : ℚ) (teacher_age : ℕ) 
  (h1 : n = 30)
  (h2 : initial_avg = 10)
  (h3 : new_avg = 11)
  (h4 : teacher_age = 41) :
  ∃ (student_age : ℕ), 
    (n : ℚ) * initial_avg - student_age + teacher_age = (n : ℚ) * new_avg ∧ 
    student_age = 11 := by
  sorry

end NUMINAMATH_CALUDE_student_age_proof_l3391_339195


namespace NUMINAMATH_CALUDE_new_students_count_l3391_339130

theorem new_students_count (initial_students : ℕ) (left_students : ℕ) (final_students : ℕ) 
  (h1 : initial_students = 31)
  (h2 : left_students = 5)
  (h3 : final_students = 37) :
  final_students - (initial_students - left_students) = 11 :=
by sorry

end NUMINAMATH_CALUDE_new_students_count_l3391_339130


namespace NUMINAMATH_CALUDE_equation_solution_set_l3391_339111

theorem equation_solution_set (x : ℝ) : 
  (((9 : ℝ)^x + 32^x) / (15^x + 24^x) = 4/3) ↔ 
  (x = (Real.log (3/4)) / (Real.log (3/2)) ∨ x = (Real.log 4) / (Real.log 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_set_l3391_339111


namespace NUMINAMATH_CALUDE_iesha_school_books_l3391_339116

/-- The number of books Iesha has about school -/
def books_about_school (total_books sports_books : ℕ) : ℕ :=
  total_books - sports_books

/-- Theorem stating that Iesha has 136 books about school -/
theorem iesha_school_books :
  books_about_school 344 208 = 136 := by
  sorry

end NUMINAMATH_CALUDE_iesha_school_books_l3391_339116


namespace NUMINAMATH_CALUDE_female_employees_count_l3391_339135

/-- Proves that the total number of female employees in a company is 500 under given conditions -/
theorem female_employees_count (total_employees : ℕ) (male_employees : ℕ) (female_managers : ℕ) :
  female_managers = 200 →
  (2 : ℚ) / 5 * total_employees = female_managers + (2 : ℚ) / 5 * male_employees →
  total_employees = male_employees + 500 :=
by sorry

end NUMINAMATH_CALUDE_female_employees_count_l3391_339135


namespace NUMINAMATH_CALUDE_unit_digit_of_23_power_100000_l3391_339107

theorem unit_digit_of_23_power_100000 : 23^100000 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_23_power_100000_l3391_339107


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3391_339191

-- Problem 1
theorem problem_1 : 
  (Real.sqrt 24 - Real.sqrt (1/2)) - (Real.sqrt (1/8) + Real.sqrt 6) = Real.sqrt 6 - (3 * Real.sqrt 2) / 4 := by
  sorry

-- Problem 2
theorem problem_2 : 
  ∀ x : ℝ, (x - 2)^2 = 3*(x - 2) ↔ x = 2 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3391_339191


namespace NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l3391_339185

theorem sqrt_six_div_sqrt_two_eq_sqrt_three : 
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l3391_339185


namespace NUMINAMATH_CALUDE_find_p_l3391_339198

theorem find_p (a b p : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : a^2 - 5*p*a + 2*p^3 = 0)
  (h2 : b^2 - 5*p*b + 2*p^3 = 0)
  (h3 : ∃! x, x^2 - a*x + b = 0) :
  p = 3 := by sorry

end NUMINAMATH_CALUDE_find_p_l3391_339198


namespace NUMINAMATH_CALUDE_beacon_population_l3391_339171

/-- Given the populations of three cities with specific relationships, prove the population of Beacon. -/
theorem beacon_population
  (richmond victoria beacon : ℕ)
  (h1 : richmond = victoria + 1000)
  (h2 : victoria = 4 * beacon)
  (h3 : richmond = 3000) :
  beacon = 500 := by
  sorry

end NUMINAMATH_CALUDE_beacon_population_l3391_339171


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l3391_339186

/-- Calculates the area of a rectangular plot given its breadth and the fact that its length is thrice its breadth -/
def rectangle_area (breadth : ℝ) : ℝ :=
  3 * breadth * breadth

/-- Proves that the area of a rectangular plot with breadth 26 meters and length thrice its breadth is 2028 square meters -/
theorem rectangle_area_proof : rectangle_area 26 = 2028 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l3391_339186


namespace NUMINAMATH_CALUDE_y_value_l3391_339165

theorem y_value (x y z : ℤ) 
  (eq1 : x + y + z = 25) 
  (eq2 : x + y = 19) 
  (eq3 : y + z = 18) : 
  y = 12 := by
sorry

end NUMINAMATH_CALUDE_y_value_l3391_339165


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3391_339117

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 + a 2 + a 3 + a 4 = 30) →
  (a 2 + a 3 = 15) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3391_339117


namespace NUMINAMATH_CALUDE_green_paint_calculation_l3391_339176

/-- Given a paint mixture ratio and the amount of white paint, 
    calculate the amount of green paint needed. -/
theorem green_paint_calculation 
  (blue green white : ℚ) 
  (ratio : blue / green = 5 / 3 ∧ green / white = 3 / 7) 
  (white_amount : white = 21) : 
  green = 9 := by
sorry

end NUMINAMATH_CALUDE_green_paint_calculation_l3391_339176


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3391_339167

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
def point_on_parabola (x₀ : ℝ) : Prop := parabola x₀ 4

-- Define a line passing through a point and intersecting the parabola at two other points
def intersecting_line (x₀ m t : ℝ) : Prop :=
  point_on_parabola x₀ ∧ ∃ y₁ y₂ : ℝ, 
    y₁ ≠ y₂ ∧ 
    parabola (m*y₁ + t) y₁ ∧ 
    parabola (m*y₂ + t) y₂

-- Define perpendicularity condition
def perpendicular_condition (x₀ m t : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, 
    (m*y₁ + t - x₀) * (m*y₂ + t - x₀) + (y₁ - 4) * (y₂ - 4) = 0

-- Theorem statement
theorem line_passes_through_fixed_point (x₀ m t : ℝ) :
  intersecting_line x₀ m t ∧ perpendicular_condition x₀ m t →
  t = 4*m + 8 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3391_339167


namespace NUMINAMATH_CALUDE_problem_solution_l3391_339102

theorem problem_solution (a b : ℚ) 
  (eq1 : 3020 * a + 3026 * b = 3030)
  (eq2 : 3024 * a + 3028 * b = 3034) :
  a - 2 * b = -1509 / 1516 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3391_339102


namespace NUMINAMATH_CALUDE_snyder_cookies_l3391_339184

/-- Mrs. Snyder's cookie problem -/
theorem snyder_cookies (red_cookies pink_cookies : ℕ) 
  (h1 : red_cookies = 36)
  (h2 : pink_cookies = 50) :
  red_cookies + pink_cookies = 86 := by
  sorry

end NUMINAMATH_CALUDE_snyder_cookies_l3391_339184


namespace NUMINAMATH_CALUDE_gambler_final_amount_l3391_339143

def bet_sequence := [true, false, true, false, false, true, false, true]

def apply_bet (current_amount : ℚ) (is_win : Bool) : ℚ :=
  if is_win then
    current_amount + (current_amount / 2)
  else
    current_amount / 2

def final_amount (initial_amount : ℚ) (bets : List Bool) : ℚ :=
  bets.foldl apply_bet initial_amount

theorem gambler_final_amount :
  final_amount 128 bet_sequence = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_gambler_final_amount_l3391_339143


namespace NUMINAMATH_CALUDE_smallest_integer_with_two_cube_sum_representations_l3391_339155

def is_sum_of_three_cubes (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ n = a^3 + b^3 + c^3

def has_two_representations (n : ℕ) : Prop :=
  ∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℕ,
    a₁ > 0 ∧ b₁ > 0 ∧ c₁ > 0 ∧
    a₂ > 0 ∧ b₂ > 0 ∧ c₂ > 0 ∧
    n = a₁^3 + b₁^3 + c₁^3 ∧
    n = a₂^3 + b₂^3 + c₂^3 ∧
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂)

theorem smallest_integer_with_two_cube_sum_representations :
  (has_two_representations 251) ∧
  (∀ m : ℕ, m < 251 → ¬(has_two_representations m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_two_cube_sum_representations_l3391_339155


namespace NUMINAMATH_CALUDE_dentist_bill_ratio_l3391_339174

def cleaning_cost : ℕ := 70
def filling_cost : ℕ := 120
def extraction_cost : ℕ := 290

def total_bill : ℕ := cleaning_cost + 2 * filling_cost + extraction_cost

theorem dentist_bill_ratio :
  (total_bill : ℚ) / filling_cost = 5 := by sorry

end NUMINAMATH_CALUDE_dentist_bill_ratio_l3391_339174


namespace NUMINAMATH_CALUDE_unique_x_for_all_y_l3391_339113

theorem unique_x_for_all_y : ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 15 * y + 2 * x - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_x_for_all_y_l3391_339113


namespace NUMINAMATH_CALUDE_stating_snail_reaches_top_l3391_339166

/-- Represents the height of the tree in meters -/
def tree_height : ℕ := 10

/-- Represents the distance the snail climbs during the day in meters -/
def day_climb : ℕ := 4

/-- Represents the distance the snail slips at night in meters -/
def night_slip : ℕ := 3

/-- Calculates the net distance the snail moves in one day -/
def net_daily_progress : ℤ := day_climb - night_slip

/-- Represents the number of days it takes for the snail to reach the top -/
def days_to_reach_top : ℕ := 7

/-- 
Theorem stating that the snail reaches the top of the tree in 7 days
given the defined tree height, day climb, and night slip distances.
-/
theorem snail_reaches_top : 
  (days_to_reach_top - 1) * net_daily_progress + day_climb ≥ tree_height :=
sorry

end NUMINAMATH_CALUDE_stating_snail_reaches_top_l3391_339166


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l3391_339142

theorem circus_ticket_cost (total_cost : ℕ) (num_tickets : ℕ) (cost_per_ticket : ℕ) :
  total_cost = 308 →
  num_tickets = 7 →
  cost_per_ticket * num_tickets = total_cost →
  cost_per_ticket = 44 := by
sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l3391_339142


namespace NUMINAMATH_CALUDE_corner_sum_is_168_l3391_339106

def checkerboard_size : Nat := 9

def min_number : Nat := 2
def max_number : Nat := 82

def top_left : Nat := min_number
def top_right : Nat := min_number + checkerboard_size - 1
def bottom_left : Nat := max_number - checkerboard_size + 1
def bottom_right : Nat := max_number

theorem corner_sum_is_168 :
  top_left + top_right + bottom_left + bottom_right = 168 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_168_l3391_339106


namespace NUMINAMATH_CALUDE_vehicle_speeds_theorem_l3391_339146

/-- Represents the speeds of two vehicles traveling in opposite directions -/
structure VehicleSpeeds where
  slow : ℝ
  fast : ℝ
  speed_diff : fast = slow + 8

/-- Proves that given the conditions, the speeds of the vehicles are 44 and 52 mph -/
theorem vehicle_speeds_theorem (v : VehicleSpeeds) 
  (h : 4 * (v.slow + v.fast) = 384) : 
  v.slow = 44 ∧ v.fast = 52 := by
  sorry

#check vehicle_speeds_theorem

end NUMINAMATH_CALUDE_vehicle_speeds_theorem_l3391_339146


namespace NUMINAMATH_CALUDE_truck_distance_l3391_339127

theorem truck_distance (distance : ℝ) (time_minutes : ℝ) (travel_time_hours : ℝ) : 
  distance = 2 ∧ time_minutes = 2.5 ∧ travel_time_hours = 3 →
  (distance / time_minutes) * (travel_time_hours * 60) = 144 := by
sorry

end NUMINAMATH_CALUDE_truck_distance_l3391_339127


namespace NUMINAMATH_CALUDE_each_girl_receives_two_dollars_l3391_339140

def debt : ℕ := 40

def lulu_savings : ℕ := 6

def nora_savings : ℕ := 5 * lulu_savings

def tamara_savings : ℕ := nora_savings / 3

def total_savings : ℕ := tamara_savings + nora_savings + lulu_savings

def remaining_money : ℕ := total_savings - debt

theorem each_girl_receives_two_dollars : 
  remaining_money / 3 = 2 := by sorry

end NUMINAMATH_CALUDE_each_girl_receives_two_dollars_l3391_339140


namespace NUMINAMATH_CALUDE_yasmin_has_two_children_l3391_339109

/-- The number of children Yasmin has -/
def yasmin_children : ℕ := 2

/-- The number of children John has -/
def john_children : ℕ := 2 * yasmin_children

/-- The total number of grandchildren -/
def total_grandchildren : ℕ := 6

theorem yasmin_has_two_children :
  yasmin_children = 2 ∧
  john_children = 2 * yasmin_children ∧
  yasmin_children + john_children = total_grandchildren :=
sorry

end NUMINAMATH_CALUDE_yasmin_has_two_children_l3391_339109


namespace NUMINAMATH_CALUDE_box_content_theorem_l3391_339108

theorem box_content_theorem (total : ℕ) (pencil : ℕ) (pen : ℕ) (both : ℕ) :
  total = 12 →
  pencil = 7 →
  pen = 4 →
  both = 3 →
  total - (pencil + pen - both) = 4 := by
  sorry

end NUMINAMATH_CALUDE_box_content_theorem_l3391_339108


namespace NUMINAMATH_CALUDE_min_value_inequality_l3391_339110

theorem min_value_inequality (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq : x + y + z = 9)
  (prod_sum_eq : x*y + y*z + z*x = 14) :
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3391_339110


namespace NUMINAMATH_CALUDE_circle_equation_l3391_339161

/-- The equation of a circle with center (-1, 2) and radius 4 -/
theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (-1, 2)
  let radius : ℝ := 4
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ (x + 1)^2 + (y - 2)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3391_339161


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3391_339179

/-- The diagonal of a rectangle with length 6 and width 8 is 10. -/
theorem rectangle_diagonal : ∀ (l w d : ℝ), 
  l = 6 → w = 8 → d^2 = l^2 + w^2 → d = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3391_339179


namespace NUMINAMATH_CALUDE_largest_marble_count_l3391_339145

theorem largest_marble_count : ∃ n : ℕ, n < 400 ∧ 
  n % 3 = 1 ∧ n % 7 = 2 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, m < 400 → m % 3 = 1 → m % 7 = 2 → m % 5 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_marble_count_l3391_339145


namespace NUMINAMATH_CALUDE_greatest_constant_right_triangle_l3391_339158

theorem greatest_constant_right_triangle (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_positive : c > 0) :
  ∀ N : ℝ, (∀ a b c : ℝ, c > 0 → a^2 + b^2 = c^2 → (a^2 + b^2 - c^2) / c^2 > N) → N ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_greatest_constant_right_triangle_l3391_339158


namespace NUMINAMATH_CALUDE_largest_circle_radius_l3391_339153

/-- Represents a chessboard square --/
structure Square where
  x : Nat
  y :Nat
  isWhite : Bool

/-- Represents a chessboard --/
def Chessboard := List Square

/-- Creates an 8x8 chessboard with alternating white and black squares --/
def createChessboard : Chessboard :=
  sorry

/-- Checks if a given point (x, y) is on a white square or corner --/
def isOnWhiteSquareOrCorner (board : Chessboard) (x : ℝ) (y : ℝ) : Prop :=
  sorry

/-- Represents a circle on the chessboard --/
structure Circle where
  centerX : ℝ
  centerY : ℝ
  radius : ℝ

/-- Checks if a circle's circumference is entirely on white squares or corners --/
def isValidCircle (board : Chessboard) (circle : Circle) : Prop :=
  sorry

/-- The theorem to be proved --/
theorem largest_circle_radius (board : Chessboard := createChessboard) :
  ∃ (c : Circle), isValidCircle board c ∧
    ∀ (c' : Circle), isValidCircle board c' → c'.radius ≤ c.radius ∧
    c.radius = Real.sqrt 10 / 2 :=
  sorry

end NUMINAMATH_CALUDE_largest_circle_radius_l3391_339153


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_k_l3391_339182

/-- Given vectors a, b, and c in R², prove that if (a - 2b) is perpendicular to c, then k = -3 -/
theorem perpendicular_vectors_imply_k (a b c : ℝ × ℝ) (h1 : a = (Real.sqrt 3, 1))
    (h2 : b = (0, -1)) (h3 : c = (k, Real.sqrt 3)) 
    (h4 : (a - 2 • b) • c = 0) : k = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_k_l3391_339182


namespace NUMINAMATH_CALUDE_unique_occurrence_l3391_339164

-- Define the sequence type
def IntegerSequence := ℕ → ℤ

-- Define the property of having infinitely many positive and negative elements
def HasInfinitelyManyPositiveAndNegative (a : IntegerSequence) : Prop :=
  (∀ N : ℕ, ∃ n > N, a n > 0) ∧ (∀ N : ℕ, ∃ n > N, a n < 0)

-- Define the property of distinct remainders
def HasDistinctRemainders (a : IntegerSequence) : Prop :=
  ∀ n : ℕ, ∀ i j : ℕ, i < n → j < n → i ≠ j → a i % n ≠ a j % n

-- The main theorem
theorem unique_occurrence (a : IntegerSequence) 
  (h1 : HasInfinitelyManyPositiveAndNegative a)
  (h2 : HasDistinctRemainders a)
  (k : ℤ) : 
  ∃! n : ℕ, a n = k :=
sorry

end NUMINAMATH_CALUDE_unique_occurrence_l3391_339164


namespace NUMINAMATH_CALUDE_locus_of_centers_l3391_339120

-- Define the circles C1 and C3
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C3 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the property of being externally tangent to C1 and internally tangent to C3
def is_tangent_to_C1_C3 (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 2)^2) ∧ ((a - 3)^2 + b^2 = (3 - r)^2)

-- State the theorem
theorem locus_of_centers (a b : ℝ) :
  (∃ r, is_tangent_to_C1_C3 a b r) → a^2 - 12*a + 4*b^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_centers_l3391_339120


namespace NUMINAMATH_CALUDE_greatest_area_difference_l3391_339150

/-- A rectangle with integer dimensions and perimeter 160 cm -/
structure Rectangle where
  length : ℕ
  width : ℕ
  perimeter_condition : length * 2 + width * 2 = 160

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The theorem stating the greatest possible difference between areas of two such rectangles -/
theorem greatest_area_difference :
  ∃ (r1 r2 : Rectangle), ∀ (s1 s2 : Rectangle),
    (area r1 - area r2 : ℤ) ≥ (area s1 - area s2 : ℤ) ∧
    (area r1 - area r2 : ℕ) = 1521 := by
  sorry

end NUMINAMATH_CALUDE_greatest_area_difference_l3391_339150


namespace NUMINAMATH_CALUDE_height_difference_in_inches_l3391_339115

-- Define conversion factors
def meters_to_feet : ℝ := 3.28084
def inches_per_foot : ℕ := 12

-- Define heights in meters
def mark_height : ℝ := 1.60
def mike_height : ℝ := 1.85

-- Function to convert meters to inches
def meters_to_inches (m : ℝ) : ℝ := m * meters_to_feet * inches_per_foot

-- Theorem statement
theorem height_difference_in_inches :
  ⌊meters_to_inches mike_height - meters_to_inches mark_height⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_in_inches_l3391_339115


namespace NUMINAMATH_CALUDE_union_equality_implies_a_equals_three_l3391_339196

-- Define the sets A and B
def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}

-- State the theorem
theorem union_equality_implies_a_equals_three (a : ℝ) : A ∪ B a = A → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_equals_three_l3391_339196


namespace NUMINAMATH_CALUDE_min_x_plus_y_l3391_339180

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  ∀ z w : ℝ, z > 0 → w > 0 → 2*z + 8*w - z*w = 0 → x + y ≤ z + w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a + 8*b - a*b = 0 ∧ a + b = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_x_plus_y_l3391_339180


namespace NUMINAMATH_CALUDE_inequality_proof_l3391_339129

theorem inequality_proof (a b c d e : ℝ) 
  (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ e)
  (h6 : a + b + c + d + e = 1) : 
  a * d + d * c + c * b + b * e + e * a ≤ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3391_339129


namespace NUMINAMATH_CALUDE_quadratic_inequality_relation_l3391_339162

theorem quadratic_inequality_relation :
  (∀ x : ℝ, x > 2 → x^2 + 5*x - 6 > 0) ∧
  (∃ x : ℝ, x^2 + 5*x - 6 > 0 ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_relation_l3391_339162


namespace NUMINAMATH_CALUDE_perpendicular_lines_l3391_339189

theorem perpendicular_lines (a : ℝ) : 
  (∃ (x y : ℝ), y = a * x - 2) ∧ 
  (∃ (x y : ℝ), y = (a + 2) * x + 1) ∧ 
  (a * (a + 2) = -1) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l3391_339189


namespace NUMINAMATH_CALUDE_parking_cost_savings_l3391_339132

-- Define the cost per week
def cost_per_week : ℕ := 10

-- Define the cost per month
def cost_per_month : ℕ := 35

-- Define the number of weeks in a year
def weeks_per_year : ℕ := 52

-- Define the number of months in a year
def months_per_year : ℕ := 12

-- Theorem statement
theorem parking_cost_savings : 
  (weeks_per_year * cost_per_week) - (months_per_year * cost_per_month) = 100 := by
  sorry


end NUMINAMATH_CALUDE_parking_cost_savings_l3391_339132


namespace NUMINAMATH_CALUDE_neighbor_field_size_l3391_339193

/-- Represents the yield of a cornfield over a period of time -/
structure CornYield where
  amount : ℕ
  months : ℕ

/-- Represents a cornfield -/
structure Cornfield where
  hectares : ℕ
  yield_per_period : CornYield

def total_yield (field : Cornfield) (months : ℕ) : ℕ :=
  field.hectares * field.yield_per_period.amount * (months / field.yield_per_period.months)

def johnson_field : Cornfield :=
  { hectares := 1
  , yield_per_period := { amount := 80, months := 2 }
  }

def neighbor_field (hectares : ℕ) : Cornfield :=
  { hectares := hectares
  , yield_per_period := { amount := 160, months := 2 }
  }

theorem neighbor_field_size :
  ∃ (x : ℕ), total_yield johnson_field 6 + total_yield (neighbor_field x) 6 = 1200 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_neighbor_field_size_l3391_339193


namespace NUMINAMATH_CALUDE_tangent_circle_center_l3391_339175

/-- A circle passes through (0,3) and is tangent to y = x^2 at (1,1) -/
structure TangentCircle where
  center : ℝ × ℝ
  passes_through : center.1^2 + (center.2 - 3)^2 = (center.1 - 0)^2 + (center.2 - 3)^2
  tangent_at : center.1^2 + (center.2 - 1)^2 = (center.1 - 1)^2 + (center.2 - 1)^2
  on_parabola : 1 = 1^2

/-- The center of the circle is (0, 3/2) -/
theorem tangent_circle_center : ∀ c : TangentCircle, c.center = (0, 3/2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_center_l3391_339175


namespace NUMINAMATH_CALUDE_class_average_l3391_339141

theorem class_average (total_students : Nat) (perfect_score_students : Nat) (zero_score_students : Nat) (class_average : ℚ) : 
  total_students = 20 →
  perfect_score_students = 2 →
  zero_score_students = 3 →
  class_average = 40 →
  let remaining_students := total_students - perfect_score_students - zero_score_students
  let total_score := total_students * class_average
  let perfect_score_total := perfect_score_students * 100
  let remaining_score := total_score - perfect_score_total
  remaining_score / remaining_students = 40 := by
sorry

end NUMINAMATH_CALUDE_class_average_l3391_339141


namespace NUMINAMATH_CALUDE_sector_central_angle_l3391_339103

/-- Given a sector with radius 1 and arc length 2, its central angle is 2 radians -/
theorem sector_central_angle (radius : ℝ) (arc_length : ℝ) (h1 : radius = 1) (h2 : arc_length = 2) :
  arc_length / radius = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3391_339103


namespace NUMINAMATH_CALUDE_night_games_count_l3391_339148

theorem night_games_count (total_games : ℕ) (h1 : total_games = 864) 
  (h2 : ∃ (night_games day_games : ℕ), night_games + day_games = total_games ∧ night_games = day_games) : 
  ∃ (night_games : ℕ), night_games = 432 := by
sorry

end NUMINAMATH_CALUDE_night_games_count_l3391_339148


namespace NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l3391_339178

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l3391_339178


namespace NUMINAMATH_CALUDE_trigonometric_problem_l3391_339137

theorem trigonometric_problem (α β : Real) 
  (h1 : 3 * Real.sin α - Real.sin β = Real.sqrt 10)
  (h2 : α + β = Real.pi / 2) :
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l3391_339137


namespace NUMINAMATH_CALUDE_second_plant_production_l3391_339149

/-- Represents the production of tomatoes from three plants -/
structure TomatoProduction where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the tomato production problem -/
def TomatoProblem (p : TomatoProduction) : Prop :=
  p.first = 24 ∧
  p.third = p.second + 2 ∧
  p.first + p.second + p.third = 60

theorem second_plant_production (p : TomatoProduction) 
  (h : TomatoProblem p) : p.first - p.second = 7 :=
by
  sorry

#check second_plant_production

end NUMINAMATH_CALUDE_second_plant_production_l3391_339149


namespace NUMINAMATH_CALUDE_tutors_next_meeting_l3391_339105

def chris_schedule : ℕ := 5
def alex_schedule : ℕ := 6
def jordan_schedule : ℕ := 8
def taylor_schedule : ℕ := 9

theorem tutors_next_meeting :
  lcm (lcm (lcm chris_schedule alex_schedule) jordan_schedule) taylor_schedule = 360 := by
  sorry

end NUMINAMATH_CALUDE_tutors_next_meeting_l3391_339105


namespace NUMINAMATH_CALUDE_sum_of_ages_l3391_339170

/-- Given the ages and relationships of Beckett, Olaf, Shannen, and Jack, prove that the sum of their ages is 71. -/
theorem sum_of_ages (beckett_age olaf_age shannen_age jack_age : ℕ) : 
  beckett_age = 12 →
  olaf_age = beckett_age + 3 →
  shannen_age = olaf_age - 2 →
  jack_age = 2 * shannen_age + 5 →
  beckett_age + olaf_age + shannen_age + jack_age = 71 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_ages_l3391_339170


namespace NUMINAMATH_CALUDE_constant_b_equals_negative_two_l3391_339126

/-- Given a polynomial equation, prove that the constant b must equal -2. -/
theorem constant_b_equals_negative_two :
  ∀ (a c : ℝ) (b : ℝ),
  (fun x : ℝ => (4 * x^3 - 2 * x + 5/2) * (a * x^3 + b * x^2 + c)) =
  (fun x : ℝ => 20 * x^6 - 8 * x^4 + 15 * x^3 - 5 * x^2 + 5) →
  b = -2 := by
sorry

end NUMINAMATH_CALUDE_constant_b_equals_negative_two_l3391_339126


namespace NUMINAMATH_CALUDE_cookie_recipe_total_cups_l3391_339173

theorem cookie_recipe_total_cups (butter flour sugar : ℕ) (total : ℕ) : 
  (butter : ℚ) / flour = 2 / 5 →
  (sugar : ℚ) / flour = 3 / 5 →
  flour = 15 →
  total = butter + flour + sugar →
  total = 30 := by
sorry

end NUMINAMATH_CALUDE_cookie_recipe_total_cups_l3391_339173


namespace NUMINAMATH_CALUDE_greatest_y_value_l3391_339151

theorem greatest_y_value (y : ℝ) : 
  (3 * y^2 + 5 * y + 2 = 6) → 
  y ≤ (-5 + Real.sqrt 73) / 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_y_value_l3391_339151


namespace NUMINAMATH_CALUDE_solve_yellow_balloons_problem_l3391_339122

def yellow_balloons_problem (sam_initial : Real) (sam_gives : Real) (total : Real) : Prop :=
  let sam_remaining : Real := sam_initial - sam_gives
  let mary_balloons : Real := total - sam_remaining
  mary_balloons = 7.0

theorem solve_yellow_balloons_problem :
  yellow_balloons_problem 6.0 5.0 8.0 := by
  sorry

end NUMINAMATH_CALUDE_solve_yellow_balloons_problem_l3391_339122


namespace NUMINAMATH_CALUDE_max_value_x_minus_x_squared_l3391_339100

theorem max_value_x_minus_x_squared (f : ℝ → ℝ) (h : ∀ x, 0 < x → x < 1 → f x = x * (1 - x)) :
  ∃ m : ℝ, m = 1/4 ∧ ∀ x, 0 < x → x < 1 → f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_x_minus_x_squared_l3391_339100


namespace NUMINAMATH_CALUDE_count_integers_with_three_digits_under_50000_l3391_339154

/-- A function that counts the number of positive integers less than n with at most k different digits. -/
def count_integers_with_limited_digits (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the count of positive integers less than 50,000 with at most three different digits is 7862. -/
theorem count_integers_with_three_digits_under_50000 :
  count_integers_with_limited_digits 50000 3 = 7862 :=
sorry

end NUMINAMATH_CALUDE_count_integers_with_three_digits_under_50000_l3391_339154


namespace NUMINAMATH_CALUDE_discount_difference_l3391_339128

theorem discount_difference (bill : ℝ) (single_discount : ℝ) (first_discount : ℝ) (second_discount : ℝ) :
  bill = 15000 ∧ 
  single_discount = 0.3 ∧ 
  first_discount = 0.25 ∧ 
  second_discount = 0.05 →
  bill * (1 - first_discount) * (1 - second_discount) - bill * (1 - single_discount) = 187.5 := by
sorry

end NUMINAMATH_CALUDE_discount_difference_l3391_339128


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l3391_339131

/-- Calculates the sampling interval for systematic sampling -/
def sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

theorem systematic_sampling_interval :
  sampling_interval 630 45 = 14 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l3391_339131


namespace NUMINAMATH_CALUDE_probability_adjacent_points_l3391_339112

/-- Represents a point on the 3x3 square --/
inductive SquarePoint
| Corner
| MidSide
| Center

/-- The set of all points on the 3x3 square --/
def squarePoints : Finset SquarePoint := sorry

/-- Two points are considered adjacent if they are one unit apart --/
def adjacent : SquarePoint → SquarePoint → Prop := sorry

/-- The number of pairs of adjacent points --/
def adjacentPairsCount : ℕ := sorry

theorem probability_adjacent_points :
  (adjacentPairsCount : ℚ) / (Finset.card (squarePoints.powerset.filter (λ s => s.card = 2)) : ℚ) = 16/45 := by
  sorry

end NUMINAMATH_CALUDE_probability_adjacent_points_l3391_339112


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_not_roots_l3391_339192

theorem quadratic_inequality_implies_not_roots (a b x : ℝ) :
  x^2 - (a + b)*x + a*b ≠ 0 → ¬(x = a ∨ x = b) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_not_roots_l3391_339192


namespace NUMINAMATH_CALUDE_total_books_calculation_l3391_339136

/-- The total number of books assigned to Mcgregor and Floyd -/
def total_books : ℕ := 89

/-- The number of books Mcgregor finished -/
def mcgregor_books : ℕ := 34

/-- The number of books Floyd finished -/
def floyd_books : ℕ := 32

/-- The number of books remaining to be read -/
def remaining_books : ℕ := 23

/-- Theorem stating that the total number of books is the sum of the books finished by Mcgregor and Floyd, plus the remaining books -/
theorem total_books_calculation : 
  total_books = mcgregor_books + floyd_books + remaining_books :=
by sorry

end NUMINAMATH_CALUDE_total_books_calculation_l3391_339136


namespace NUMINAMATH_CALUDE_sum_of_cubes_inequality_l3391_339177

theorem sum_of_cubes_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1)^3 / b + (b + 1)^3 / c + (c + 1)^3 / a ≥ 81 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_inequality_l3391_339177


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3391_339124

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes y = ±2x is √5 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 2) : 
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3391_339124


namespace NUMINAMATH_CALUDE_power_function_properties_l3391_339133

def f (m n : ℕ+) (x : ℝ) : ℝ := x ^ (m.val / n.val)

theorem power_function_properties (m n : ℕ+) (h_coprime : Nat.Coprime m.val n.val) :
  (∀ x, m.val % 2 = 1 ∧ n.val % 2 = 1 → f m n (-x) = -f m n x) ∧
  (∀ x, m.val % 2 = 0 ∧ n.val % 2 = 1 → f m n (-x) = f m n x) :=
sorry

end NUMINAMATH_CALUDE_power_function_properties_l3391_339133


namespace NUMINAMATH_CALUDE_intersection_and_complement_l3391_339125

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (1 - x)}

-- State the theorem
theorem intersection_and_complement :
  (M ∩ N = {x : ℝ | -2 ≤ x ∧ x ≤ 1}) ∧
  (Nᶜ = {x : ℝ | x > 1}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_complement_l3391_339125


namespace NUMINAMATH_CALUDE_zack_andrew_same_team_probability_l3391_339163

-- Define the total number of players
def total_players : ℕ := 27

-- Define the number of teams
def num_teams : ℕ := 3

-- Define the number of players per team
def players_per_team : ℕ := 9

-- Define the set of players
def Player : Type := Fin total_players

-- Define the function that assigns players to teams
def team_assignment : Player → Fin num_teams := sorry

-- Define Zack, Mihir, and Andrew as specific players
def Zack : Player := sorry
def Mihir : Player := sorry
def Andrew : Player := sorry

-- State that Zack and Mihir are on different teams
axiom zack_mihir_different : team_assignment Zack ≠ team_assignment Mihir

-- State that Mihir and Andrew are on different teams
axiom mihir_andrew_different : team_assignment Mihir ≠ team_assignment Andrew

-- Define the probability function
def probability_same_team (p1 p2 : Player) : ℚ := sorry

-- State the theorem to be proved
theorem zack_andrew_same_team_probability :
  probability_same_team Zack Andrew = 8 / 17 := sorry

end NUMINAMATH_CALUDE_zack_andrew_same_team_probability_l3391_339163


namespace NUMINAMATH_CALUDE_highest_winner_number_l3391_339194

/-- Represents a single-elimination tournament with wrestlers having qualification numbers. -/
structure WrestlingTournament where
  num_wrestlers : ℕ
  can_win : ℕ → ℕ → Prop

/-- The conditions of our specific tournament. -/
def our_tournament : WrestlingTournament where
  num_wrestlers := 512
  can_win := fun a b => b ≤ a + 2

/-- The number of rounds in a single-elimination tournament. -/
def num_rounds (t : WrestlingTournament) : ℕ :=
  Nat.log 2 t.num_wrestlers

/-- The highest possible qualification number for the winner. -/
def max_winner_number (t : WrestlingTournament) : ℕ :=
  1 + 2 * num_rounds t

theorem highest_winner_number (t : WrestlingTournament) :
  t = our_tournament →
  max_winner_number t = 18 :=
by sorry

end NUMINAMATH_CALUDE_highest_winner_number_l3391_339194


namespace NUMINAMATH_CALUDE_complex_number_properties_l3391_339134

/-- For a real number m and a complex number z = (m^2 - 5m + 6) + (m^2 - 3m)i, we define the following properties --/

def is_real (m : ℝ) : Prop := m^2 - 3*m = 0

def is_complex (m : ℝ) : Prop := m^2 - 3*m ≠ 0

def is_purely_imaginary (m : ℝ) : Prop := m^2 - 5*m + 6 = 0 ∧ m^2 - 3*m ≠ 0

def is_in_third_quadrant (m : ℝ) : Prop := m^2 - 5*m + 6 < 0 ∧ m^2 - 3*m < 0

/-- Main theorem stating the conditions for each case --/
theorem complex_number_properties (m : ℝ) :
  (is_real m ↔ (m = 0 ∨ m = 3)) ∧
  (is_complex m ↔ (m ≠ 0 ∧ m ≠ 3)) ∧
  (is_purely_imaginary m ↔ m = 2) ∧
  (is_in_third_quadrant m ↔ (2 < m ∧ m < 3)) :=
sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3391_339134
